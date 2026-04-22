//===-- AMDGPUResourceTrackerTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUResourceTracker.h"
#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnitTests.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

static std::unique_ptr<Module> parseMIR(LLVMContext &Context,
                                        const TargetMachine &TM,
                                        StringRef MIRCode, const char *FnName,
                                        MachineModuleInfo &MMI) {
  SMDiagnostic Diagnostic;
  std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRCode);
  auto MIR = createMIRParser(std::move(MBuffer), Context);
  if (!MIR)
    return nullptr;

  std::unique_ptr<Module> Mod = MIR->parseIRModule();
  if (!Mod)
    return nullptr;

  Mod->setDataLayout(TM.createDataLayout());

  if (MIR->parseMachineFunctions(*Mod, MMI))
    return nullptr;

  return Mod;
}

// Test fixture with additional members for MIR parsing.
class AMDGPUResourceTrackerTest : public AMDGPUTestBase {
protected:
  LLVMContext Context;
  std::unique_ptr<GCNTargetMachine> TM;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<Module> M;
  /// Shared global sequence number for ResourceTracker construction in tests.
  uint64_t TestSeqNum = 0;

  /// Parses the MIR string and returns the specified basic block.
  /// Sets up TM, MMI, and M as side effects. Returns nullptr on failure.
  MachineBasicBlock *parseMIRGetBB(StringRef CPU, StringRef MIRString,
                                   unsigned BBNum = 0) {
    TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), CPU, "");
    if (!TM)
      return nullptr;
    MMI = std::make_unique<MachineModuleInfo>(TM.get());
    M = parseMIR(Context, *TM, MIRString, "test", *MMI);
    if (!M)
      return nullptr;
    auto *MF = MMI->getMachineFunction(*M->getFunction("test"));
    if (!MF)
      return nullptr;
    return MF->getBlockNumbered(BBNum);
  }

  /// Returns the basic block with the given name. Asserts if not found.
  MachineBasicBlock &getBlockByNumber(MachineFunction *MF, unsigned Number) {
    MachineBasicBlock *MBB = MF->getBlockNumbered(Number);
    assert(MBB && "Block not found");
    return *MBB;
  }
};

TEST_F(AMDGPUTestBase, ResourceTracker_GetCountersForInstr) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  ; BB0: SMEM instructions - KM_CNT
  bb.0:
    liveins: $sgpr0_sgpr1, $sgpr0_sgpr1_sgpr2_sgpr3
    ; s_load_* variants
    $sgpr12 = S_LOAD_DWORD_IMM $sgpr0_sgpr1, 0, 0
    $sgpr12_sgpr13 = S_LOAD_DWORDX2_IMM $sgpr0_sgpr1, 0, 0
    $sgpr12_sgpr13_sgpr14 = S_LOAD_DWORDX3_IMM $sgpr0_sgpr1, 0, 0
    $sgpr12_sgpr13_sgpr14_sgpr15 = S_LOAD_DWORDX4_IMM $sgpr0_sgpr1, 0, 0
    $sgpr12_sgpr13_sgpr14_sgpr15_sgpr16_sgpr17_sgpr18_sgpr19 = S_LOAD_DWORDX8_IMM $sgpr0_sgpr1, 0, 0
    ; s_buffer_load_* variants
    $sgpr12 = S_BUFFER_LOAD_DWORD_IMM $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0
    $sgpr12_sgpr13 = S_BUFFER_LOAD_DWORDX2_IMM $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0
    $sgpr12_sgpr13_sgpr14 = S_BUFFER_LOAD_DWORDX3_IMM $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0
    $sgpr12_sgpr13_sgpr14_sgpr15 = S_BUFFER_LOAD_DWORDX4_IMM $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0
    ; s_memtime, s_memrealtime
    $sgpr14_sgpr15 = S_MEMTIME
    $sgpr16_sgpr17 = S_MEMREALTIME
    ; s_sendmsg variants
    S_SENDMSG 0, implicit $m0, implicit $exec
    S_SENDMSGHALT 0, implicit $m0, implicit $exec
    $sgpr12 = S_SENDMSG_RTN_B32 128
    $sgpr12_sgpr13 = S_SENDMSG_RTN_B64 128
    S_BRANCH %bb.1

  ; BB1: DS (LDS) instructions - DS_CNT
  bb.1:
    liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3, $vgpr0_vgpr1, $vgpr2_vgpr3
    ; ds_load_* variants
    $vgpr5 = DS_READ_B32_gfx9 $vgpr0, 0, 0, implicit $exec, implicit $m0
    $vgpr5_vgpr6 = DS_READ_B64_gfx9 $vgpr0, 0, 0, implicit $exec, implicit $m0
    $vgpr5_vgpr6_vgpr7 = DS_READ_B96_gfx9 $vgpr0, 0, 0, implicit $exec, implicit $m0
    $vgpr5_vgpr6_vgpr7_vgpr8 = DS_READ_B128_gfx9 $vgpr0, 0, 0, implicit $exec, implicit $m0
    ; ds_store_* variants
    DS_WRITE_B32_gfx9 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    DS_WRITE_B64_gfx9 $vgpr0, $vgpr0_vgpr1, 0, 0, implicit $m0, implicit $exec
    DS_WRITE_B128_gfx9 $vgpr0, $vgpr0_vgpr1_vgpr2_vgpr3, 0, 0, implicit $m0, implicit $exec
    ; ds_atomic_* variants (no return)
    DS_ADD_U32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    DS_SUB_U32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    DS_AND_B32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    DS_OR_B32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    DS_XOR_B32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    DS_MIN_I32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    DS_MAX_I32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    DS_MIN_U32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    DS_MAX_U32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    DS_INC_U32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    DS_DEC_U32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    DS_ADD_F32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    ; ds_atomic_* variants (with return)
    $vgpr5 = DS_ADD_RTN_U32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    $vgpr5 = DS_SUB_RTN_U32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    $vgpr5 = DS_AND_RTN_B32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    $vgpr5 = DS_OR_RTN_B32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    $vgpr5 = DS_XOR_RTN_B32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    $vgpr5 = DS_MIN_RTN_I32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    $vgpr5 = DS_MAX_RTN_I32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    $vgpr5 = DS_WRXCHG_RTN_B32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    ; ds_cmpstore, ds_permute, ds_bpermute
    DS_CMPSTORE_B32 $vgpr0, $vgpr1, $vgpr2, 0, 0, implicit $m0, implicit $exec
    $vgpr5 = DS_CMPSTORE_RTN_B32 $vgpr0, $vgpr1, $vgpr2, 0, 0, implicit $m0, implicit $exec
    $vgpr5 = DS_SWIZZLE_B32 $vgpr0, 0, 0, implicit $exec
    S_BRANCH %bb.2

  ; BB2: VMEM load instructions - LOAD_CNT
  bb.2:
    liveins: $sgpr0, $sgpr0_sgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $vgpr0, $vgpr0_vgpr1
    ; global_load_* variants
    $vgpr5 = GLOBAL_LOAD_UBYTE $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr5 = GLOBAL_LOAD_SBYTE $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr5 = GLOBAL_LOAD_USHORT $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr5 = GLOBAL_LOAD_SSHORT $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr5 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr5_vgpr6 = GLOBAL_LOAD_DWORDX2 $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr5_vgpr6_vgpr7 = GLOBAL_LOAD_DWORDX3 $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr5_vgpr6_vgpr7_vgpr8 = GLOBAL_LOAD_DWORDX4 $vgpr0_vgpr1, 0, 0, implicit $exec
    ; global_load_* with SADDR
    $vgpr5 = GLOBAL_LOAD_DWORD_SADDR $sgpr0_sgpr1, $vgpr0, 0, 0, implicit $exec
    $vgpr5_vgpr6 = GLOBAL_LOAD_DWORDX2_SADDR $sgpr0_sgpr1, $vgpr0, 0, 0, implicit $exec
    ; buffer_load_* variants
    $vgpr5 = BUFFER_LOAD_UBYTE_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_LOAD_SBYTE_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_LOAD_USHORT_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_LOAD_SSHORT_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_LOAD_DWORD_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    $vgpr5_vgpr6 = BUFFER_LOAD_DWORDX2_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    $vgpr5_vgpr6_vgpr7 = BUFFER_LOAD_DWORDX3_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    $vgpr5_vgpr6_vgpr7_vgpr8 = BUFFER_LOAD_DWORDX4_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    ; buffer_load_format_* variants
    $vgpr5 = BUFFER_LOAD_FORMAT_X_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    $vgpr5_vgpr6 = BUFFER_LOAD_FORMAT_XY_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    $vgpr5_vgpr6_vgpr7 = BUFFER_LOAD_FORMAT_XYZ_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    $vgpr5_vgpr6_vgpr7_vgpr8 = BUFFER_LOAD_FORMAT_XYZW_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    ; scratch_load_* variants
    $vgpr5 = SCRATCH_LOAD_UBYTE $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = SCRATCH_LOAD_SBYTE $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = SCRATCH_LOAD_USHORT $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = SCRATCH_LOAD_SSHORT $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = SCRATCH_LOAD_DWORD $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5_vgpr6 = SCRATCH_LOAD_DWORDX2 $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5_vgpr6_vgpr7 = SCRATCH_LOAD_DWORDX3 $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5_vgpr6_vgpr7_vgpr8 = SCRATCH_LOAD_DWORDX4 $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = SCRATCH_LOAD_DWORD_SADDR $sgpr0, 0, 0, implicit $exec, implicit $flat_scr
    S_BRANCH %bb.3

  ; BB3: VMEM store instructions - STORE_CNT
  bb.3:
    liveins: $sgpr0, $sgpr0_sgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $vgpr0, $vgpr1, $vgpr2, $vgpr0_vgpr1, $vgpr2_vgpr3, $vgpr4_vgpr5_vgpr6, $vgpr4_vgpr5_vgpr6_vgpr7
    ; global_store_* variants
    GLOBAL_STORE_BYTE $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_STORE_SHORT $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_STORE_DWORDX2 $vgpr0_vgpr1, $vgpr2_vgpr3, 0, 0, implicit $exec
    GLOBAL_STORE_DWORDX3 $vgpr0_vgpr1, $vgpr4_vgpr5_vgpr6, 0, 0, implicit $exec
    GLOBAL_STORE_DWORDX4 $vgpr0_vgpr1, $vgpr4_vgpr5_vgpr6_vgpr7, 0, 0, implicit $exec
    ; global_store_* with SADDR
    GLOBAL_STORE_DWORD_SADDR $vgpr0, $vgpr1, $sgpr0_sgpr1, 0, 0, implicit $exec
    GLOBAL_STORE_DWORDX2_SADDR $vgpr0, $vgpr2_vgpr3, $sgpr0_sgpr1, 0, 0, implicit $exec
    ; buffer_store_* variants
    BUFFER_STORE_BYTE_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    BUFFER_STORE_SHORT_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    BUFFER_STORE_DWORD_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    BUFFER_STORE_DWORD_OFFSET $vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    BUFFER_STORE_DWORDX2_OFFEN $vgpr2_vgpr3, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    BUFFER_STORE_DWORDX3_OFFEN $vgpr4_vgpr5_vgpr6, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    BUFFER_STORE_DWORDX4_OFFEN $vgpr4_vgpr5_vgpr6_vgpr7, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    ; buffer_store_format_* variants
    BUFFER_STORE_FORMAT_X_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    BUFFER_STORE_FORMAT_XY_OFFEN $vgpr2_vgpr3, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    BUFFER_STORE_FORMAT_XYZ_OFFEN $vgpr4_vgpr5_vgpr6, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    BUFFER_STORE_FORMAT_XYZW_OFFEN $vgpr4_vgpr5_vgpr6_vgpr7, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    ; scratch_store_* variants
    SCRATCH_STORE_BYTE $vgpr1, $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    SCRATCH_STORE_SHORT $vgpr1, $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    SCRATCH_STORE_DWORD $vgpr1, $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    SCRATCH_STORE_DWORDX2 $vgpr2_vgpr3, $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    SCRATCH_STORE_DWORDX3 $vgpr4_vgpr5_vgpr6, $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    SCRATCH_STORE_DWORDX4 $vgpr4_vgpr5_vgpr6_vgpr7, $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    SCRATCH_STORE_DWORD_SADDR $vgpr1, $sgpr0, 0, 0, implicit $exec, implicit $flat_scr
    S_BRANCH %bb.4

  ; BB4: VMEM atomic instructions
  bb.4:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3, $vgpr0, $vgpr1, $vgpr2, $vgpr5, $vgpr0_vgpr1, $vgpr2_vgpr3
    ; Global atomic no-return - STORE_CNT
    GLOBAL_ATOMIC_ADD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_ATOMIC_SUB $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_ATOMIC_AND $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_ATOMIC_OR $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_ATOMIC_XOR $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_ATOMIC_SMIN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_ATOMIC_SMAX $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_ATOMIC_UMIN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_ATOMIC_UMAX $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_ATOMIC_INC $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_ATOMIC_DEC $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_ATOMIC_SWAP $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_ATOMIC_CMPSWAP $vgpr0_vgpr1, $vgpr2_vgpr3, 0, 0, implicit $exec
    GLOBAL_ATOMIC_ADD_F32 $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    ; Buffer atomic no-return - STORE_CNT
    BUFFER_ATOMIC_ADD_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    BUFFER_ATOMIC_SUB_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    BUFFER_ATOMIC_AND_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    BUFFER_ATOMIC_OR_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    BUFFER_ATOMIC_XOR_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    BUFFER_ATOMIC_SMIN_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    BUFFER_ATOMIC_SMAX_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    BUFFER_ATOMIC_UMIN_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    BUFFER_ATOMIC_UMAX_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    BUFFER_ATOMIC_INC_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    BUFFER_ATOMIC_DEC_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    BUFFER_ATOMIC_SWAP_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    BUFFER_ATOMIC_CMPSWAP_OFFEN $vgpr2_vgpr3, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    BUFFER_ATOMIC_ADD_F32_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    ; Global atomic with return - LOAD_CNT
    $vgpr5 = GLOBAL_ATOMIC_ADD_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    $vgpr5 = GLOBAL_ATOMIC_SUB_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    $vgpr5 = GLOBAL_ATOMIC_AND_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    $vgpr5 = GLOBAL_ATOMIC_OR_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    $vgpr5 = GLOBAL_ATOMIC_XOR_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    $vgpr5 = GLOBAL_ATOMIC_SMIN_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    $vgpr5 = GLOBAL_ATOMIC_SMAX_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    $vgpr5 = GLOBAL_ATOMIC_UMIN_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    $vgpr5 = GLOBAL_ATOMIC_UMAX_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    $vgpr5 = GLOBAL_ATOMIC_INC_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    $vgpr5 = GLOBAL_ATOMIC_DEC_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    $vgpr5 = GLOBAL_ATOMIC_SWAP_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    ; Buffer atomic with return - LOAD_CNT
    $vgpr5 = BUFFER_ATOMIC_ADD_OFFEN_RTN $vgpr5, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_ATOMIC_SUB_OFFEN_RTN $vgpr5, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_ATOMIC_AND_OFFEN_RTN $vgpr5, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_ATOMIC_OR_OFFEN_RTN $vgpr5, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_ATOMIC_XOR_OFFEN_RTN $vgpr5, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_ATOMIC_SMIN_OFFEN_RTN $vgpr5, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_ATOMIC_SMAX_OFFEN_RTN $vgpr5, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_ATOMIC_UMIN_OFFEN_RTN $vgpr5, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_ATOMIC_UMAX_OFFEN_RTN $vgpr5, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_ATOMIC_INC_OFFEN_RTN $vgpr5, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_ATOMIC_DEC_OFFEN_RTN $vgpr5, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    $vgpr5 = BUFFER_ATOMIC_SWAP_OFFEN_RTN $vgpr5, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    S_BRANCH %bb.5

  ; BB5: FLAT instructions - LOAD_CNT/STORE_CNT + DS_CNT
  bb.5:
    liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr5, $vgpr0_vgpr1, $vgpr2_vgpr3, $vgpr4_vgpr5_vgpr6, $vgpr4_vgpr5_vgpr6_vgpr7
    ; FLAT load variants - LOAD_CNT and DS_CNT
    $vgpr5 = FLAT_LOAD_UBYTE $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_LOAD_SBYTE $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_LOAD_USHORT $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_LOAD_SSHORT $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5_vgpr6 = FLAT_LOAD_DWORDX2 $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5_vgpr6_vgpr7 = FLAT_LOAD_DWORDX3 $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5_vgpr6_vgpr7_vgpr8 = FLAT_LOAD_DWORDX4 $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    ; FLAT store variants - STORE_CNT and DS_CNT
    FLAT_STORE_BYTE $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_STORE_SHORT $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_STORE_DWORD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_STORE_DWORDX2 $vgpr0_vgpr1, $vgpr2_vgpr3, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_STORE_DWORDX3 $vgpr0_vgpr1, $vgpr4_vgpr5_vgpr6, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_STORE_DWORDX4 $vgpr0_vgpr1, $vgpr4_vgpr5_vgpr6_vgpr7, 0, 0, implicit $exec, implicit $flat_scr
    ; FLAT atomic no-return - STORE_CNT and DS_CNT
    FLAT_ATOMIC_ADD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_ATOMIC_SUB $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_ATOMIC_AND $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_ATOMIC_OR $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_ATOMIC_XOR $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_ATOMIC_SMIN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_ATOMIC_SMAX $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_ATOMIC_UMIN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_ATOMIC_UMAX $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_ATOMIC_INC $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_ATOMIC_DEC $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_ATOMIC_SWAP $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_ATOMIC_CMPSWAP $vgpr0_vgpr1, $vgpr2_vgpr3, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_ATOMIC_ADD_F32 $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    ; FLAT atomic with return - LOAD_CNT and DS_CNT
    $vgpr5 = FLAT_ATOMIC_ADD_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_ATOMIC_SUB_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_ATOMIC_AND_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_ATOMIC_OR_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_ATOMIC_XOR_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_ATOMIC_SMIN_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_ATOMIC_SMAX_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_ATOMIC_UMIN_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_ATOMIC_UMAX_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_ATOMIC_INC_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_ATOMIC_DEC_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_ATOMIC_SWAP_RTN $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec, implicit $flat_scr
    S_BRANCH %bb.6

  ; BB6: IMAGE instructions
  bb.6:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, $vgpr0, $vgpr3, $vgpr4, $vgpr0_vgpr1_vgpr2_vgpr3, $vgpr5_vgpr6
    ; IMAGE_SAMPLE variants - SAMPLE_CNT
    $vgpr10_vgpr11_vgpr12 = IMAGE_SAMPLE_LZ_V3_V2_gfx12 $vgpr3, $vgpr4, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, 1, 1, -1, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s128))
    ; IMAGE_LOAD variants - LOAD_CNT
    $vgpr10_vgpr11_vgpr12_vgpr13 = IMAGE_LOAD_V4_V1_gfx12 $vgpr3, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, 15, 0, 0, 0, 0, 0, 0, implicit $exec :: (dereferenceable load (s128), addrspace 8)
    ; IMAGE_STORE variants - STORE_CNT
    IMAGE_STORE_V4_V1_gfx12 $vgpr0_vgpr1_vgpr2_vgpr3, $vgpr4, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, 15, 0, 0, 0, 0, 0, 0, implicit $exec :: (dereferenceable store (s128), addrspace 8)
    S_BRANCH %bb.7

  ; BB7: EXP and LDSDIR instructions - EXP_CNT
  bb.7:
    liveins: $vgpr0, $vgpr1, $vgpr2, $vgpr3
    ; EXP instruction
    EXP 0, $vgpr0, $vgpr0, $vgpr0, $vgpr0, -1, -1, 15, implicit $exec
    EXP 1, $vgpr0, $vgpr1, $vgpr2, $vgpr3, -1, -1, 15, implicit $exec
    ; LDSDIR instructions
    $vgpr5 = LDS_PARAM_LOAD 0, 0, 0, implicit $m0, implicit $exec
    $vgpr5 = LDS_PARAM_LOAD 1, 0, 0, implicit $m0, implicit $exec
    $vgpr5 = LDS_DIRECT_LOAD 0, implicit $m0, implicit $exec
    S_BRANCH %bb.8

  ; BB8: Instructions that don't use any counter
  bb.8:
    liveins: $vgpr0, $vgpr1, $vgpr2, $sgpr0, $sgpr1, $sgpr2, $vcc
    ; VOP1 instructions (integer ops don't need implicit $mode)
    $vgpr5 = V_MOV_B32_e32 0, implicit $exec
    $vgpr5 = V_NOT_B32_e32 $vgpr0, implicit $exec
    $vgpr5 = V_BFREV_B32_e32 $vgpr0, implicit $exec
    ; VOP1 FP instructions need implicit $mode
    $vgpr5 = V_CVT_F32_I32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr5 = V_CVT_F32_U32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr5 = V_CVT_I32_F32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr5 = V_CVT_U32_F32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr5 = V_FRACT_F32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr5 = V_TRUNC_F32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr5 = V_CEIL_F32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr5 = V_FLOOR_F32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr5 = V_RNDNE_F32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr5 = V_EXP_F32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr5 = V_LOG_F32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr5 = V_RCP_F32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr5 = V_RSQ_F32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr5 = V_SQRT_F32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr5 = V_SIN_F32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr5 = V_COS_F32_e32 $vgpr0, implicit $mode, implicit $exec
    ; VOP2 integer instructions
    $vgpr5 = V_ADD_U32_e32 $vgpr0, $vgpr1, implicit $exec
    $vgpr5 = V_SUB_U32_e32 $vgpr0, $vgpr1, implicit $exec
    $vgpr5 = V_AND_B32_e32 $vgpr0, $vgpr1, implicit $exec
    $vgpr5 = V_OR_B32_e32 $vgpr0, $vgpr1, implicit $exec
    $vgpr5 = V_XOR_B32_e32 $vgpr0, $vgpr1, implicit $exec
    $vgpr5 = V_LSHLREV_B32_e32 $vgpr0, $vgpr1, implicit $exec
    $vgpr5 = V_LSHRREV_B32_e32 $vgpr0, $vgpr1, implicit $exec
    $vgpr5 = V_ASHRREV_I32_e32 $vgpr0, $vgpr1, implicit $exec
    $vgpr5 = V_MAX_I32_e32 $vgpr0, $vgpr1, implicit $exec
    $vgpr5 = V_MIN_I32_e32 $vgpr0, $vgpr1, implicit $exec
    $vgpr5 = V_MAX_U32_e32 $vgpr0, $vgpr1, implicit $exec
    $vgpr5 = V_MIN_U32_e32 $vgpr0, $vgpr1, implicit $exec
    ; VOP2 FP instructions need implicit $mode
    $vgpr5 = V_ADD_F32_e32 $vgpr0, $vgpr1, implicit $mode, implicit $exec
    $vgpr5 = V_SUB_F32_e32 $vgpr0, $vgpr1, implicit $mode, implicit $exec
    $vgpr5 = V_MUL_F32_e32 $vgpr0, $vgpr1, implicit $mode, implicit $exec
    $vgpr5 = V_MAX_F32_e32 $vgpr0, $vgpr1, implicit $mode, implicit $exec
    $vgpr5 = V_MIN_F32_e32 $vgpr0, $vgpr1, implicit $mode, implicit $exec
    $vgpr5 = V_FMAC_F32_e32 $vgpr0, $vgpr1, $vgpr5, implicit $mode, implicit $exec
    ; VOP3 integer instructions
    $vgpr5 = V_MAD_I32_I24_e64 $vgpr0, $vgpr1, $vgpr2, 0, implicit $exec
    $vgpr5 = V_MAD_U32_U24_e64 $vgpr0, $vgpr1, $vgpr2, 0, implicit $exec
    $vgpr5 = V_BFE_U32_e64 $vgpr0, $vgpr1, $vgpr2, implicit $exec
    $vgpr5 = V_BFE_I32_e64 $vgpr0, $vgpr1, $vgpr2, implicit $exec
    $vgpr5 = V_BFI_B32_e64 $vgpr0, $vgpr1, $vgpr2, implicit $exec
    $vgpr5 = V_ALIGNBIT_B32_e64 $vgpr0, $vgpr1, $vgpr2, implicit $exec
    $vgpr5 = V_ALIGNBYTE_B32_e64 $vgpr0, $vgpr1, $vgpr2, implicit $exec
    $vgpr5 = V_PERM_B32_e64 $vgpr0, $vgpr1, $vgpr2, implicit $exec
    $vgpr5 = V_LERP_U8_e64 $vgpr0, $vgpr1, $vgpr2, implicit $exec
    $vgpr5 = V_SAD_U8_e64 $vgpr0, $vgpr1, $vgpr2, 0, implicit $exec
    $vgpr5 = V_SAD_U16_e64 $vgpr0, $vgpr1, $vgpr2, 0, implicit $exec
    $vgpr5 = V_SAD_U32_e64 $vgpr0, $vgpr1, $vgpr2, 0, implicit $exec
    $vgpr5 = V_MBCNT_LO_U32_B32_e64 $vgpr0, $vgpr1, implicit $exec
    $vgpr5 = V_MBCNT_HI_U32_B32_e64 $vgpr0, $vgpr1, implicit $exec
    ; VOP3 FP instructions need implicit $mode
    $vgpr5 = V_FMA_F32_e64 0, $vgpr0, 0, $vgpr1, 0, $vgpr2, 0, 0, implicit $mode, implicit $exec
    $vgpr5 = V_MED3_F32_e64 0, $vgpr0, 0, $vgpr1, 0, $vgpr2, 0, 0, implicit $mode, implicit $exec
    $vgpr5 = V_MIN3_F32_e64 0, $vgpr0, 0, $vgpr1, 0, $vgpr2, 0, 0, implicit $mode, implicit $exec
    $vgpr5 = V_MAX3_F32_e64 0, $vgpr0, 0, $vgpr1, 0, $vgpr2, 0, 0, implicit $mode, implicit $exec
    $vgpr5 = V_LDEXP_F32_e64 0, $vgpr0, 0, $vgpr1, 0, 0, implicit $mode, implicit $exec
    ; Lane/register operations
    $sgpr5 = V_READLANE_B32 $vgpr0, $sgpr0
    $sgpr5 = V_READFIRSTLANE_B32 $vgpr0, implicit $exec
    $vgpr5 = V_WRITELANE_B32 $sgpr0, $sgpr1, $vgpr5, implicit $exec
    ; VOPC integer instructions
    $vcc_lo = V_CMP_EQ_I32_e64 $vgpr0, $vgpr1, implicit $exec
    $vcc_lo = V_CMP_LT_I32_e64 $vgpr0, $vgpr1, implicit $exec
    $vcc_lo = V_CMP_LE_I32_e64 $vgpr0, $vgpr1, implicit $exec
    $vcc_lo = V_CMP_GT_I32_e64 $vgpr0, $vgpr1, implicit $exec
    $vcc_lo = V_CMP_GE_I32_e64 $vgpr0, $vgpr1, implicit $exec
    $vcc_lo = V_CMP_EQ_U32_e64 $vgpr0, $vgpr1, implicit $exec
    $vcc_lo = V_CMP_LT_U32_e64 $vgpr0, $vgpr1, implicit $exec
    $vcc_lo = V_CMP_LE_U32_e64 $vgpr0, $vgpr1, implicit $exec
    $vcc_lo = V_CMP_GT_U32_e64 $vgpr0, $vgpr1, implicit $exec
    $vcc_lo = V_CMP_GE_U32_e64 $vgpr0, $vgpr1, implicit $exec
    ; VOPC FP instructions
    $vcc_lo = V_CMP_EQ_F32_e64 0, $vgpr0, 0, $vgpr1, 0, implicit $mode, implicit $exec
    $vcc_lo = V_CMP_LT_F32_e64 0, $vgpr0, 0, $vgpr1, 0, implicit $mode, implicit $exec
    $vcc_lo = V_CMP_LE_F32_e64 0, $vgpr0, 0, $vgpr1, 0, implicit $mode, implicit $exec
    $vcc_lo = V_CMP_GT_F32_e64 0, $vgpr0, 0, $vgpr1, 0, implicit $mode, implicit $exec
    $vcc_lo = V_CMP_GE_F32_e64 0, $vgpr0, 0, $vgpr1, 0, implicit $mode, implicit $exec
    $vgpr5 = V_CNDMASK_B32_e64 0, $vgpr0, 0, $vgpr1, $vcc_lo, implicit $exec
    ; SOP1 instructions
    $sgpr12 = S_MOV_B32 0
    $sgpr12_sgpr13 = S_MOV_B64 0
    $sgpr12 = S_NOT_B32 $sgpr0, implicit-def $scc
    $sgpr12 = S_BREV_B32 $sgpr0
    $sgpr12 = S_ABS_I32 $sgpr0, implicit-def $scc
    $sgpr12 = S_BCNT1_I32_B32 $sgpr0, implicit-def $scc
    $sgpr0 = S_BITSET0_B32 $sgpr1, $sgpr0
    $sgpr0 = S_BITSET1_B32 $sgpr1, $sgpr0
    ; SOP2 instructions
    $sgpr12 = S_ADD_U32 $sgpr0, 1, implicit-def $scc
    $sgpr12 = S_SUB_U32 $sgpr0, 1, implicit-def $scc
    $sgpr12 = S_MUL_I32 $sgpr0, $sgpr1
    $sgpr12 = S_AND_B32 $sgpr0, $sgpr1, implicit-def $scc
    $sgpr12 = S_OR_B32 $sgpr0, $sgpr1, implicit-def $scc
    $sgpr12 = S_XOR_B32 $sgpr0, $sgpr1, implicit-def $scc
    $sgpr12 = S_NAND_B32 $sgpr0, $sgpr1, implicit-def $scc
    $sgpr12 = S_NOR_B32 $sgpr0, $sgpr1, implicit-def $scc
    $sgpr12 = S_XNOR_B32 $sgpr0, $sgpr1, implicit-def $scc
    $sgpr12 = S_LSHL_B32 $sgpr0, $sgpr1, implicit-def $scc
    $sgpr12 = S_LSHR_B32 $sgpr0, $sgpr1, implicit-def $scc
    $sgpr12 = S_ASHR_I32 $sgpr0, $sgpr1, implicit-def $scc
    $sgpr12 = S_MIN_I32 $sgpr0, $sgpr1, implicit-def $scc
    $sgpr12 = S_MAX_I32 $sgpr0, $sgpr1, implicit-def $scc
    $sgpr12 = S_MIN_U32 $sgpr0, $sgpr1, implicit-def $scc
    $sgpr12 = S_MAX_U32 $sgpr0, $sgpr1, implicit-def $scc
    $sgpr12 = S_BFE_U32 $sgpr0, 0, implicit-def $scc
    $sgpr12 = S_BFE_I32 $sgpr0, 0, implicit-def $scc
    $sgpr12 = S_BFM_B32 $sgpr0, $sgpr1
    $sgpr12 = S_CSELECT_B32 $sgpr0, $sgpr1, implicit $scc
    $sgpr12 = S_ABSDIFF_I32 $sgpr0, $sgpr1, implicit-def $scc
    ; SOP2 FP instructions need implicit $mode
    $sgpr12 = S_ADD_F32 $sgpr0, $sgpr1, implicit-def $scc, implicit $mode
    $sgpr12 = S_SUB_F32 $sgpr0, $sgpr1, implicit-def $scc, implicit $mode
    $sgpr12 = S_MUL_F32 $sgpr0, $sgpr1, implicit-def $scc, implicit $mode
    ; SOPC instructions
    S_CMP_EQ_I32 $sgpr0, $sgpr1, implicit-def $scc
    S_CMP_LT_I32 $sgpr0, $sgpr1, implicit-def $scc
    S_CMP_LE_I32 $sgpr0, $sgpr1, implicit-def $scc
    S_CMP_GT_I32 $sgpr0, $sgpr1, implicit-def $scc
    S_CMP_GE_I32 $sgpr0, $sgpr1, implicit-def $scc
    S_CMP_EQ_U32 $sgpr0, $sgpr1, implicit-def $scc
    S_CMP_LT_U32 $sgpr0, $sgpr1, implicit-def $scc
    S_CMP_LE_U32 $sgpr0, $sgpr1, implicit-def $scc
    S_CMP_GT_U32 $sgpr0, $sgpr1, implicit-def $scc
    S_CMP_GE_U32 $sgpr0, $sgpr1, implicit-def $scc
    S_BITCMP0_B32 $sgpr0, $sgpr1, implicit-def $scc
    S_BITCMP1_B32 $sgpr0, $sgpr1, implicit-def $scc
    ; Wait instructions
    S_WAIT_LOADCNT 0
    S_WAIT_STORECNT 0
    S_WAIT_SAMPLECNT 0
    S_WAIT_BVHCNT 0
    S_WAIT_EXPCNT 0
    S_WAIT_DSCNT 0
    S_WAIT_KMCNT 0
    S_WAITCNT 0
    ; Control flow and misc
    S_BARRIER
    S_NOP 0
    S_SLEEP 0
    S_SETHALT 0
    S_SETPRIO 0
    ; Buffer invalidation instructions don't use any counter (GFX12+)
    BUFFER_GL0_INV implicit $exec
    BUFFER_GL1_INV implicit $exec
    S_BRANCH %bb.9

  ; BB9: GLOBAL_INV - LOAD_CNT (GFX12+ FLAT cache invalidate)
  bb.9:
    GLOBAL_INV 16, implicit $exec
    S_BRANCH %bb.10

  ; BB10: GLOBAL_WB/WBINV - STORE_CNT (GFX12+ FLAT cache writeback)
  bb.10:
    GLOBAL_WB 16, implicit $exec
    GLOBAL_WBINV 16, implicit $exec
    S_BRANCH %bb.11

  ; BB11: ASYNCMARK instruction - no hardware counter (meta instruction)
  bb.11:
    ASYNCMARK implicit $exec
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

  // BB0: SMEM instructions - KM_CNT
  {
    auto *MBB = MF->getBlockNumbered(0);
    ASSERT_TRUE(MBB) << "Failed to get BB0";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::KmCnt()))
          << MI;
    }
  }

  // BB1: DS (LDS) instructions - DS_CNT
  {
    auto *MBB = MF->getBlockNumbered(1);
    ASSERT_TRUE(MBB) << "Failed to get BB1";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::DsCnt()))
          << MI;
    }
  }

  // BB2: VMEM load instructions - LOAD_CNT
  {
    auto *MBB = MF->getBlockNumbered(2);
    ASSERT_TRUE(MBB) << "Failed to get BB2";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::LoadCnt()))
          << MI;
    }
  }

  // BB3: VMEM store instructions - STORE_CNT
  {
    auto *MBB = MF->getBlockNumbered(3);
    ASSERT_TRUE(MBB) << "Failed to get BB3";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::StoreCnt()))
          << MI;
    }
  }

  // BB4: VMEM atomic instructions
  {
    auto *MBB = MF->getBlockNumbered(4);
    ASSERT_TRUE(MBB) << "Failed to get BB4";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      // Atomic no-return uses STORE_CNT, atomic with return uses LOAD_CNT
      if (SIInstrInfo::isAtomicNoRet(MI))
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::StoreCnt()))
            << MI;
      else
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::LoadCnt()))
            << MI;
    }
  }

  // BB5: FLAT instructions - LOAD_CNT/STORE_CNT + DS_CNT
  {
    auto *MBB = MF->getBlockNumbered(5);
    ASSERT_TRUE(MBB) << "Failed to get BB5";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      // FLAT load/store/atomic all access both VMEM and LDS
      if (MI.mayLoad() && !SIInstrInfo::isAtomicNoRet(MI))
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST),
                    ElementsAre(AMDGPU::LoadCnt(), AMDGPU::DsCnt()))
            << MI;
      else
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST),
                    ElementsAre(AMDGPU::StoreCnt(), AMDGPU::DsCnt()))
            << MI;
    }
  }

  // BB6: IMAGE instructions
  {
    auto *MBB = MF->getBlockNumbered(6);
    ASSERT_TRUE(MBB) << "Failed to get BB6";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      // IMAGE_SAMPLE uses SAMPLE_CNT, IMAGE_LOAD uses LOAD_CNT,
      // IMAGE_STORE uses STORE_CNT
      if (MI.mayStore())
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::StoreCnt()))
            << MI;
      else if (SIInstrInfo::isMIMG(MI)) {
        const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(MI.getOpcode());
        const AMDGPU::MIMGBaseOpcodeInfo *BaseInfo =
            AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);
        if (BaseInfo->Sampler)
          EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST),
                      ElementsAre(AMDGPU::SampleCnt()))
              << MI;
        else
          EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST),
                      ElementsAre(AMDGPU::LoadCnt()))
              << MI;
      }
    }
  }

  // BB7: EXP and LDSDIR instructions - EXP_CNT
  {
    auto *MBB = MF->getBlockNumbered(7);
    ASSERT_TRUE(MBB) << "Failed to get BB7";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::ExpCnt()))
          << MI;
    }
  }

  // BB8: Instructions that don't use any counter
  {
    auto *MBB = MF->getBlockNumbered(8);
    ASSERT_TRUE(MBB) << "Failed to get BB8";
    for (MachineInstr &MI : *MBB) {
      if (MI.isTerminator())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), IsEmpty()) << MI;
    }
  }

  // BB9: GLOBAL_INV - LOAD_CNT (GFX12+ FLAT cache invalidate)
  {
    auto *MBB = MF->getBlockNumbered(9);
    ASSERT_TRUE(MBB) << "Failed to get BB9";
    for (MachineInstr &MI : *MBB) {
      if (MI.isTerminator())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::LoadCnt()))
          << MI;
    }
  }

  // BB10: GLOBAL_WB/WBINV - STORE_CNT (GFX12+ FLAT cache writeback)
  {
    auto *MBB = MF->getBlockNumbered(10);
    ASSERT_TRUE(MBB) << "Failed to get BB10";
    for (MachineInstr &MI : *MBB) {
      if (MI.isTerminator())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::StoreCnt()))
          << MI;
    }
  }

  // BB11: ASYNCMARK instruction - no hardware counter (meta instruction only)
  {
    auto *MBB = MF->getBlockNumbered(11);
    ASSERT_TRUE(MBB) << "Failed to get BB11";
    for (MachineInstr &MI : *MBB) {
      if (MI.isTerminator())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), IsEmpty())
          << MI;
    }
  }
}

TEST_F(AMDGPUTestBase, ResourceTracker_GetCountersForInstr_PreGFX12) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1030", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  ; BB0: SMEM instructions - LGKM_CNT (pre-gfx12)
  bb.0:
    liveins: $sgpr0_sgpr1, $sgpr0_sgpr1_sgpr2_sgpr3
    $sgpr12 = S_LOAD_DWORD_IMM $sgpr0_sgpr1, 0, 0
    $sgpr12 = S_BUFFER_LOAD_DWORD_IMM $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0
    S_SENDMSG 0, implicit $m0, implicit $exec
    S_BRANCH %bb.1

  ; BB1: DS (LDS) instructions - LGKM_CNT (pre-gfx12)
  bb.1:
    liveins: $vgpr0, $vgpr1
    $vgpr5 = DS_READ_B32_gfx9 $vgpr0, 0, 0, implicit $exec, implicit $m0
    DS_WRITE_B32_gfx9 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    DS_ADD_U32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    $vgpr5 = DS_ADD_RTN_U32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    S_BRANCH %bb.2

  ; BB2: VMEM load instructions - VM_CNT (pre-gfx12)
  bb.2:
    liveins: $sgpr0_sgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $vgpr0, $vgpr0_vgpr1
    $vgpr5 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr5 = BUFFER_LOAD_DWORD_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    $vgpr5 = SCRATCH_LOAD_DWORD $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    S_BRANCH %bb.3

  ; BB3: VMEM store instructions - VS_CNT (pre-gfx12)
  bb.3:
    liveins: $sgpr0_sgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $vgpr0, $vgpr1, $vgpr0_vgpr1
    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr1, 0, 0, implicit $exec
    BUFFER_STORE_DWORD_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    SCRATCH_STORE_DWORD $vgpr1, $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    S_BRANCH %bb.4

  ; BB4: VMEM atomic instructions
  bb.4:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3, $vgpr0, $vgpr1, $vgpr5, $vgpr0_vgpr1
    ; Atomic no-return uses VS_CNT
    GLOBAL_ATOMIC_ADD $vgpr0_vgpr1, $vgpr1, 0, 0, implicit $exec
    BUFFER_ATOMIC_ADD_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    ; Atomic with return uses VM_CNT
    $vgpr5 = GLOBAL_ATOMIC_ADD_RTN $vgpr0_vgpr1, $vgpr1, 0, 0, implicit $exec
    $vgpr5 = BUFFER_ATOMIC_ADD_OFFEN_RTN $vgpr5, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    S_BRANCH %bb.5

  ; BB5: FLAT instructions - VM_CNT/VS_CNT + LGKM_CNT
  bb.5:
    liveins: $vgpr0, $vgpr1, $vgpr0_vgpr1
    $vgpr5 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_STORE_DWORD $vgpr0_vgpr1, $vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_ATOMIC_ADD $vgpr0_vgpr1, $vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_ATOMIC_ADD_RTN $vgpr0_vgpr1, $vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    S_BRANCH %bb.6

  ; BB6: IMAGE instructions - VM_CNT for all (pre-gfx12 doesn't have SAMPLE_CNT/BVH_CNT)
  bb.6:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, $vgpr3_vgpr4, $vgpr0_vgpr1_vgpr2_vgpr3
    ; IMAGE_SAMPLE - uses VM_CNT (not SAMPLE_CNT) on pre-gfx12
    $vgpr10_vgpr11_vgpr12_vgpr13 = IMAGE_SAMPLE_V4_V2_gfx10 $vgpr3_vgpr4, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, 15, 1, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (dereferenceable load (s128), addrspace 8)
    ; IMAGE_LOAD - uses VM_CNT on pre-gfx12
    $vgpr10_vgpr11_vgpr12_vgpr13 = IMAGE_LOAD_V4_V2_gfx10 $vgpr3_vgpr4, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, 15, 1, -1, 0, 0, 0, 0, 0, 0, implicit $exec :: (dereferenceable load (s128), addrspace 8)
    S_BRANCH %bb.7

  ; BB7: IMAGE_STORE - VS_CNT on pre-gfx12
  bb.7:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $vgpr0_vgpr1_vgpr2_vgpr3, $vgpr3_vgpr4
    IMAGE_STORE_V4_V2_gfx10 $vgpr0_vgpr1_vgpr2_vgpr3, $vgpr3_vgpr4, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, 15, 1, -1, 0, 0, 0, 0, 0, 0, implicit $exec :: (dereferenceable store (s128), addrspace 8)
    S_BRANCH %bb.8

  ; BB8: EXP instructions - EXP_CNT
  bb.8:
    liveins: $vgpr0
    EXP 0, $vgpr0, $vgpr0, $vgpr0, $vgpr0, -1, -1, 15, implicit $exec
    S_BRANCH %bb.9

  ; BB9: Buffer invalidation instructions don't use any counter
  bb.9:
    BUFFER_WBINVL1 implicit $exec
    BUFFER_WBINVL1_VOL implicit $exec
    BUFFER_GL0_INV implicit $exec
    BUFFER_GL1_INV implicit $exec
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

  // BB0: SMEM instructions - LGKM_CNT (pre-gfx12)
  {
    auto *MBB = MF->getBlockNumbered(0);
    ASSERT_TRUE(MBB) << "Failed to get BB0";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::LgkmCnt()))
          << MI;
    }
  }

  // BB1: DS (LDS) instructions - LGKM_CNT (pre-gfx12)
  {
    auto *MBB = MF->getBlockNumbered(1);
    ASSERT_TRUE(MBB) << "Failed to get BB1";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::LgkmCnt()))
          << MI;
    }
  }

  // BB2: VMEM load instructions - VM_CNT (pre-gfx12)
  {
    auto *MBB = MF->getBlockNumbered(2);
    ASSERT_TRUE(MBB) << "Failed to get BB2";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::VmCnt()))
          << MI;
    }
  }

  // BB3: VMEM store instructions - VS_CNT (pre-gfx12)
  {
    auto *MBB = MF->getBlockNumbered(3);
    ASSERT_TRUE(MBB) << "Failed to get BB3";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::VsCnt()))
          << MI;
    }
  }

  // BB4: VMEM atomic instructions
  {
    auto *MBB = MF->getBlockNumbered(4);
    ASSERT_TRUE(MBB) << "Failed to get BB4";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      // Atomic no-return uses VS_CNT, atomic with return uses VM_CNT
      if (SIInstrInfo::isAtomicNoRet(MI))
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::VsCnt()))
            << MI;
      else
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::VmCnt()))
            << MI;
    }
  }

  // BB5: FLAT instructions - VM_CNT/VS_CNT + LGKM_CNT
  {
    auto *MBB = MF->getBlockNumbered(5);
    ASSERT_TRUE(MBB) << "Failed to get BB5";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      // FLAT load/store/atomic all access both VMEM and LDS
      if (MI.mayLoad() && !SIInstrInfo::isAtomicNoRet(MI))
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST),
                    ElementsAre(AMDGPU::VmCnt(), AMDGPU::LgkmCnt()))
            << MI;
      else
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST),
                    ElementsAre(AMDGPU::VsCnt(), AMDGPU::LgkmCnt()))
            << MI;
    }
  }

  // BB6: IMAGE load instructions - VM_CNT (not SAMPLE_CNT on pre-gfx12)
  {
    auto *MBB = MF->getBlockNumbered(6);
    ASSERT_TRUE(MBB) << "Failed to get BB6";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::VmCnt()))
          << MI;
    }
  }

  // BB7: IMAGE_STORE - VS_CNT
  {
    auto *MBB = MF->getBlockNumbered(7);
    ASSERT_TRUE(MBB) << "Failed to get BB7";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::VsCnt()))
          << MI;
    }
  }

  // BB8: EXP instructions - EXP_CNT
  {
    auto *MBB = MF->getBlockNumbered(8);
    ASSERT_TRUE(MBB) << "Failed to get BB8";
    for (MachineInstr &MI : *MBB) {
      if (MI.isTerminator())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::ExpCnt()))
          << MI;
    }
  }

  // BB9: Buffer invalidation instructions don't use any counter
  {
    auto *MBB = MF->getBlockNumbered(9);
    ASSERT_TRUE(MBB) << "Failed to get BB9";
    for (MachineInstr &MI : *MBB) {
      if (MI.isTerminator())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), IsEmpty()) << MI;
    }
  }
}

TEST_F(AMDGPUTestBase, ResourceTracker_GetCountersForInstr_GFX1250) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1250", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  ; BB0: SMEM instructions - KM_CNT (gfx12+)
  bb.0:
    liveins: $sgpr0_sgpr1, $sgpr0_sgpr1_sgpr2_sgpr3
    $sgpr12 = S_LOAD_DWORD_IMM $sgpr0_sgpr1, 0, 0
    $sgpr12 = S_BUFFER_LOAD_DWORD_IMM $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0
    S_SENDMSG 0, implicit $m0, implicit $exec
    S_BRANCH %bb.1

  ; BB1: DS (LDS) instructions - DS_CNT (gfx12+)
  bb.1:
    liveins: $vgpr0, $vgpr1
    $vgpr5 = DS_READ_B32_gfx9 $vgpr0, 0, 0, implicit $exec, implicit $m0
    DS_WRITE_B32_gfx9 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    DS_ADD_U32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    $vgpr5 = DS_ADD_RTN_U32 $vgpr0, $vgpr1, 0, 0, implicit $m0, implicit $exec
    S_BRANCH %bb.2

  ; BB2: VMEM load instructions - LOAD_CNT (gfx12+)
  bb.2:
    liveins: $sgpr0_sgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $vgpr0, $vgpr0_vgpr1
    $vgpr5 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr5 = BUFFER_LOAD_DWORD_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    $vgpr5 = SCRATCH_LOAD_DWORD $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    S_BRANCH %bb.3

  ; BB3: VMEM store instructions - STORE_CNT (gfx12+)
  bb.3:
    liveins: $sgpr0_sgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $vgpr0, $vgpr1, $vgpr0_vgpr1
    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr1, 0, 0, implicit $exec
    BUFFER_STORE_DWORD_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    SCRATCH_STORE_DWORD $vgpr1, $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    S_BRANCH %bb.4

  ; BB4: VMEM atomic instructions
  bb.4:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3, $vgpr0, $vgpr1, $vgpr5, $vgpr0_vgpr1
    ; Atomic no-return uses STORE_CNT
    GLOBAL_ATOMIC_ADD $vgpr0_vgpr1, $vgpr1, 0, 0, implicit $exec
    BUFFER_ATOMIC_ADD_OFFEN $vgpr1, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    ; Atomic with return uses LOAD_CNT
    $vgpr5 = GLOBAL_ATOMIC_ADD_RTN $vgpr0_vgpr1, $vgpr1, 0, 0, implicit $exec
    $vgpr5 = BUFFER_ATOMIC_ADD_OFFEN_RTN $vgpr5, $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, implicit $exec
    S_BRANCH %bb.5

  ; BB5: FLAT instructions - LOAD_CNT/STORE_CNT + DS_CNT
  bb.5:
    liveins: $vgpr0, $vgpr1, $vgpr0_vgpr1
    $vgpr5 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_STORE_DWORD $vgpr0_vgpr1, $vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    FLAT_ATOMIC_ADD $vgpr0_vgpr1, $vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr5 = FLAT_ATOMIC_ADD_RTN $vgpr0_vgpr1, $vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    S_BRANCH %bb.6

  ; BB6: EXP instructions - EXP_CNT
  bb.6:
    liveins: $vgpr0
    EXP 0, $vgpr0, $vgpr0, $vgpr0, $vgpr0, -1, -1, 15, implicit $exec
    S_BRANCH %bb.7

  ; BB7: Async LDS DMA instructions - ASYNC_CNT (gfx1250)
  bb.7:
    liveins: $vgpr0, $vgpr0_vgpr1
    GLOBAL_LOAD_ASYNC_TO_LDS_B32 $vgpr0, $vgpr0_vgpr1, 0, 0, implicit-def $asynccnt, implicit $exec, implicit $asynccnt
    S_BRANCH %bb.8

  ; BB8: FLAT cache control ops. These perform no address translation, so on
  ; gfx1250 they must be tracked on a single counter only - never XCnt (and
  ; never DsCnt). GLOBAL_INV -> LOAD_CNT, GLOBAL_WB/WBINV -> STORE_CNT.
  bb.8:
    GLOBAL_INV 24, implicit $exec
    GLOBAL_WB 24, implicit $exec
    GLOBAL_WBINV 24, implicit $exec
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

  // BB0: SMEM instructions - KM_CNT, plus X_CNT for SMRD (gfx1250 has XCnt)
  {
    auto *MBB = MF->getBlockNumbered(0);
    ASSERT_TRUE(MBB) << "Failed to get BB0";
    const SIInstrInfo *TII = ST->getInstrInfo();
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      // SMRD instructions get both KmCnt and XCnt, but S_SENDMSG only gets KmCnt
      if (TII->isSMRD(MI))
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST),
                    ElementsAre(AMDGPU::KmCnt(), AMDGPU::XCnt()))
            << MI;
      else
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::KmCnt()))
            << MI;
    }
  }

  // BB1: DS (LDS) instructions - DS_CNT (gfx12+)
  {
    auto *MBB = MF->getBlockNumbered(1);
    ASSERT_TRUE(MBB) << "Failed to get BB1";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::DsCnt()))
          << MI;
    }
  }

  // BB2: VMEM load instructions - LOAD_CNT + X_CNT (gfx1250 has XCnt)
  {
    auto *MBB = MF->getBlockNumbered(2);
    ASSERT_TRUE(MBB) << "Failed to get BB2";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST),
                  ElementsAre(AMDGPU::LoadCnt(), AMDGPU::XCnt()))
          << MI;
    }
  }

  // BB3: VMEM store instructions - STORE_CNT + X_CNT (gfx1250 has XCnt)
  {
    auto *MBB = MF->getBlockNumbered(3);
    ASSERT_TRUE(MBB) << "Failed to get BB3";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST),
                  ElementsAre(AMDGPU::StoreCnt(), AMDGPU::XCnt()))
          << MI;
    }
  }

  // BB4: VMEM atomic instructions - STORE_CNT/LOAD_CNT + X_CNT
  {
    auto *MBB = MF->getBlockNumbered(4);
    ASSERT_TRUE(MBB) << "Failed to get BB4";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      // Atomic no-return uses STORE_CNT, atomic with return uses LOAD_CNT
      // Both also use XCnt on gfx1250
      if (SIInstrInfo::isAtomicNoRet(MI))
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST),
                    ElementsAre(AMDGPU::StoreCnt(), AMDGPU::XCnt()))
            << MI;
      else
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST),
                    ElementsAre(AMDGPU::LoadCnt(), AMDGPU::XCnt()))
            << MI;
    }
  }

  // BB5: FLAT instructions - LOAD_CNT/STORE_CNT + DS_CNT + X_CNT
  {
    auto *MBB = MF->getBlockNumbered(5);
    ASSERT_TRUE(MBB) << "Failed to get BB5";
    for (MachineInstr &MI : *MBB) {
      if (MI.isBranch())
        continue;
      // FLAT load/store/atomic all access both VMEM and LDS.
      // On gfx1250, XCnt is also tracked for VMEM access.
      if (MI.mayLoad() && !SIInstrInfo::isAtomicNoRet(MI))
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST),
                    ElementsAre(AMDGPU::LoadCnt(), AMDGPU::DsCnt(),
                                AMDGPU::XCnt()))
            << MI;
      else
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST),
                    ElementsAre(AMDGPU::StoreCnt(), AMDGPU::DsCnt(),
                                AMDGPU::XCnt()))
            << MI;
    }
  }

  // BB6: EXP instructions - EXP_CNT
  {
    auto *MBB = MF->getBlockNumbered(6);
    ASSERT_TRUE(MBB) << "Failed to get BB6";
    for (MachineInstr &MI : *MBB) {
      if (MI.isTerminator())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::ExpCnt()))
          << MI;
    }
  }

  // BB7: Async LDS DMA instructions - ASYNC_CNT (gfx1250)
  {
    auto *MBB = MF->getBlockNumbered(7);
    ASSERT_TRUE(MBB) << "Failed to get BB7";
    for (MachineInstr &MI : *MBB) {
      if (MI.isTerminator())
        continue;
      EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST), ElementsAre(AMDGPU::AsyncCnt()))
          << MI;
    }
  }

  // BB8: FLAT cache control ops on gfx1250. These do no address translation, so
  // they must map to exactly one counter and must not pick up XCnt or DsCnt.
  {
    auto *MBB = MF->getBlockNumbered(8);
    ASSERT_TRUE(MBB) << "Failed to get BB8";
    for (MachineInstr &MI : *MBB) {
      if (MI.isTerminator())
        continue;
      if (MI.getOpcode() == AMDGPU::GLOBAL_INV)
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST),
                    ElementsAre(AMDGPU::LoadCnt()))
            << MI;
      else
        EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(MI, *ST),
                    ElementsAre(AMDGPU::StoreCnt()))
            << MI;
    }
  }
}

// Test that VALU instructions return VaVdst in expert mode (GFX12+).
TEST_F(AMDGPUTestBase, ResourceTracker_GetCountersForInstr_ExpertMode) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0, $vgpr1, $vgpr2, $sgpr0_sgpr1, $vgpr0_vgpr1
    ; VALU instructions - should return VaVdst in expert mode
    $vgpr3 = V_ADD_F32_e32 $vgpr0, $vgpr1, implicit $mode, implicit $exec
    $vgpr4 = V_MUL_F32_e32 $vgpr0, $vgpr1, implicit $mode, implicit $exec
    $vgpr5 = V_SUB_F32_e32 $vgpr0, $vgpr1, implicit $mode, implicit $exec
    $vgpr6 = V_MAX_F32_e32 $vgpr0, $vgpr1, implicit $mode, implicit $exec
    ; Transcendental VALU
    $vgpr7 = V_EXP_F32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr8 = V_LOG_F32_e32 $vgpr0, implicit $mode, implicit $exec
    $vgpr9 = V_RCP_F32_e32 $vgpr0, implicit $mode, implicit $exec
    ; VMEM load - should return LoadCnt (no VaVdst since not VALU)
    $vgpr12 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    ; DS load - should return DsCnt (no VaVdst since not VALU)
    $vgpr13 = DS_READ_B32_gfx9 $vgpr0, 0, 0, implicit $exec, implicit $m0
    ; SMEM load - should return KmCnt (no VaVdst since not VALU)
    $sgpr12 = S_LOAD_DWORD_IMM $sgpr0_sgpr1, 0, 0
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();

  // Test with expert mode enabled
  AMDGPU::ResourceTracker RTExpert(ST, /*AA=*/nullptr,
                                   AMDGPU::SchedulingMode::ExpertMode2);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  auto It = MBB->begin();

  // VALU instructions should return VaVdst in expert mode
  // V_ADD_F32
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST, AMDGPU::SchedulingMode::ExpertMode2), ElementsAre(AMDGPU::VaVdst()))
      << *It;
  ++It;
  // V_MUL_F32
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST, AMDGPU::SchedulingMode::ExpertMode2), ElementsAre(AMDGPU::VaVdst()))
      << *It;
  ++It;
  // V_SUB_F32
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST, AMDGPU::SchedulingMode::ExpertMode2), ElementsAre(AMDGPU::VaVdst()))
      << *It;
  ++It;
  // V_MAX_F32
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST, AMDGPU::SchedulingMode::ExpertMode2), ElementsAre(AMDGPU::VaVdst()))
      << *It;
  ++It;
  // V_EXP_F32 (transcendental)
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST, AMDGPU::SchedulingMode::ExpertMode2), ElementsAre(AMDGPU::VaVdst()))
      << *It;
  ++It;
  // V_LOG_F32 (transcendental)
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST, AMDGPU::SchedulingMode::ExpertMode2), ElementsAre(AMDGPU::VaVdst()))
      << *It;
  ++It;
  // V_RCP_F32 (transcendental)
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST, AMDGPU::SchedulingMode::ExpertMode2), ElementsAre(AMDGPU::VaVdst()))
      << *It;
  ++It;

  // GLOBAL_LOAD_DWORD - should return LoadCnt and VmVsrc (not VaVdst)
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST, AMDGPU::SchedulingMode::ExpertMode2), ElementsAre(AMDGPU::LoadCnt(), AMDGPU::VmVsrc()))
      << *It;
  ++It;
  // DS_READ_B32 - should return DsCnt and VmVsrc (not VaVdst)
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST, AMDGPU::SchedulingMode::ExpertMode2), ElementsAre(AMDGPU::DsCnt(), AMDGPU::VmVsrc()))
      << *It;
  ++It;
  // S_LOAD_DWORD - should return KmCnt only (not VALU)
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST, AMDGPU::SchedulingMode::ExpertMode2), ElementsAre(AMDGPU::KmCnt()))
      << *It;
  ++It;

  // Now test with NoExpert mode - VALU instructions should NOT return VaVdst
  AMDGPU::ResourceTracker RTNoExpert(ST, /*AA=*/nullptr,
                                     AMDGPU::SchedulingMode::NoExpert);

  It = MBB->begin();

  // V_ADD_F32 - no counters in NoExpert mode
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST), IsEmpty()) << *It;
  ++It;
  // V_MUL_F32
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST), IsEmpty()) << *It;
  ++It;
  // V_SUB_F32
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST), IsEmpty()) << *It;
  ++It;
  // V_MAX_F32
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST), IsEmpty()) << *It;
  ++It;
  // V_EXP_F32
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST), IsEmpty()) << *It;
  ++It;
  // V_LOG_F32
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST), IsEmpty()) << *It;
  ++It;
  // V_RCP_F32
  EXPECT_THAT(AMDGPU::Counter::getCountersForInstr(*It, *ST), IsEmpty()) << *It;
}

// Test ResourceTracker::track() and getWaitFor() for register dependencies.
TEST_F(AMDGPUTestBase, ResourceTracker_TrackRegisterDependencies) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $sgpr0_sgpr1, $vgpr0_vgpr1
    ; Load into vgpr5, then use vgpr5 - should require wait
    $vgpr5 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    ; Load into sgpr12
    $sgpr12 = S_LOAD_DWORD_IMM $sgpr0_sgpr1, 0, 0
    ; DS load into vgpr6
    $vgpr6 = DS_READ_B32_gfx9 $vgpr0, 0, 0, implicit $exec, implicit $m0
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr,
                             AMDGPU::SchedulingMode::NoExpert);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 4> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 3u);

  MachineInstr *GlobalLoad = Instrs[0];  // GLOBAL_LOAD_DWORD -> $vgpr5
  MachineInstr *SLoad = Instrs[1];       // S_LOAD_DWORD_IMM -> $sgpr12
  MachineInstr *DSRead = Instrs[2];      // DS_READ_B32 -> $vgpr6

  // Before tracking, no waits should be needed
  EXPECT_TRUE(RT.getWaitForReg(AMDGPU::VGPR5).empty());
  EXPECT_TRUE(RT.getWaitForReg(AMDGPU::SGPR12).empty());
  EXPECT_TRUE(RT.getWaitForReg(AMDGPU::VGPR6).empty());

  // Track the first instruction (GLOBAL_LOAD_DWORD)
  RT.track(*GlobalLoad);

  // Now vgpr5 should require a wait on LoadCnt
  auto WaitsVgpr5 = RT.getWaitForReg(AMDGPU::VGPR5);
  ASSERT_EQ(WaitsVgpr5.size(), 1u);
  EXPECT_EQ(WaitsVgpr5.begin()->Cntr, AMDGPU::LoadCnt());
  EXPECT_EQ(WaitsVgpr5.begin()->Wait, 0u);  // First instruction, wait for 0

  // Track the second instruction (S_LOAD_DWORD_IMM)
  RT.track(*SLoad);

  // sgpr12 should require a wait on KmCnt
  auto WaitsSgpr12 = RT.getWaitForReg(AMDGPU::SGPR12);
  ASSERT_EQ(WaitsSgpr12.size(), 1u);
  EXPECT_EQ(WaitsSgpr12.begin()->Cntr, AMDGPU::KmCnt());
  EXPECT_EQ(WaitsSgpr12.begin()->Wait, 0u);

  // Track the third instruction (DS_READ_B32)
  RT.track(*DSRead);

  // vgpr6 should require a wait on DsCnt
  auto WaitsVgpr6 = RT.getWaitForReg(AMDGPU::VGPR6);
  ASSERT_EQ(WaitsVgpr6.size(), 1u);
  EXPECT_EQ(WaitsVgpr6.begin()->Cntr, AMDGPU::DsCnt());
  EXPECT_EQ(WaitsVgpr6.begin()->Wait, 0u);
}

// A store's register DEF must not create a StoreCnt/VsCnt RAW hazard. Stores
// deliver no register result (the store counter only orders memory); a register
// def on a store is a liveness artifact, e.g. the implicit-def of the whole
// vgpr tuple emitted by a GISel tuple spill. A later instruction that reads such
// a register must not get a spurious store wait.
TEST_F(AMDGPUTestBase, ResourceTracker_StoreRegDefNoRawHazard) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1030", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
tracksRegLiveness: true
stack:
  - { id: 0, type: spill-slot, offset: 0, size: 32, alignment: 4 }
body:             |
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr32, $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7
    ; First spill store carries the GISel tuple-spill implicit-def of the whole
    ; vgpr tuple. This must NOT register a StoreCnt result on those vgprs.
    BUFFER_STORE_DWORD_OFFSET killed $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr32, 12, 0, 0, implicit $exec, implicit-def $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7, implicit $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7 :: (store (s32) into %stack.0, addrspace 5)
    BUFFER_STORE_DWORD_OFFSET killed $vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr32, 16, 0, 0, implicit $exec :: (store (s32) into %stack.0 + 4, addrspace 5)
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr,
                             AMDGPU::SchedulingMode::NoExpert);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  MachineInstr *FirstStore = &*MBB->begin();
  RT.track(*FirstStore);

  // The store registers its memory ordering on the store counter, so a wait
  // FOR the store itself is legitimate...
  EXPECT_FALSE(RT.getCounter(AMDGPU::VsCnt()).empty());
  // ...but no register (including $vgpr1, part of the implicit-def tuple) may
  // carry a StoreCnt RAW dependency back to the store.
  EXPECT_TRUE(RT.getWaitForReg(AMDGPU::VGPR1).empty());
  EXPECT_TRUE(RT.getWaitForReg(AMDGPU::VGPR0).empty());
  EXPECT_TRUE(RT.getWaitForReg(AMDGPU::VGPR7).empty());
}

// A load's real result is its explicit def; the implicit-def of the whole tuple
// that GISel attaches to a multi-register reload is a liveness artifact. Only
// the explicit result may carry a load-counter dependency - the other tuple
// registers must not, or a later write/read of one of them would get a spurious
// wait for the load.
TEST_F(AMDGPUTestBase, ResourceTracker_LoadImplicitTupleDefNoHazard) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1030", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
tracksRegLiveness: true
stack:
  - { id: 0, type: spill-slot, offset: 0, size: 16, alignment: 4 }
body:             |
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr32
    ; Reload's real result is the explicit $vgpr0; the implicit-def of the tuple
    ; is bookkeeping and must NOT register a load result on $vgpr1..$vgpr3.
    $vgpr0 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr32, 0, 0, 0, implicit $exec, implicit-def $vgpr0_vgpr1_vgpr2_vgpr3 :: (load (s32) from %stack.0, addrspace 5)
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr,
                             AMDGPU::SchedulingMode::NoExpert);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  RT.track(*MBB->begin());

  // The explicit result $vgpr0 carries a VmCnt load dependency.
  EXPECT_FALSE(RT.getWaitForReg(AMDGPU::VGPR0).empty());
  // The other tuple registers (implicit-def only) must not.
  EXPECT_TRUE(RT.getWaitForReg(AMDGPU::VGPR1).empty());
  EXPECT_TRUE(RT.getWaitForReg(AMDGPU::VGPR2).empty());
  EXPECT_TRUE(RT.getWaitForReg(AMDGPU::VGPR3).empty());
}

// Test that multiple loads to the same counter increment the wait value.
TEST_F(AMDGPUTestBase, ResourceTracker_TrackMultipleLoadsToSameCounter) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1
    ; Multiple VMEM loads - all use LoadCnt
    $vgpr5 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr6 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec
    $vgpr7 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 8, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 4> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 3u);

  // Track all three loads
  for (MachineInstr *MI : Instrs)
    RT.track(*MI);

  // Instruction order: [vgpr5, vgpr6, vgpr7] (oldest to newest)
  // Wait values: oldest=2, middle=1, newest=0 (distance from end)

  // vgpr5 was the first (oldest) load - wait=2
  auto WaitsVgpr5 = RT.getWaitForReg(AMDGPU::VGPR5);
  ASSERT_EQ(WaitsVgpr5.size(), 1u);
  EXPECT_EQ(WaitsVgpr5.begin()->Wait, 2u);

  // vgpr6 was the second (middle) load - wait=1
  auto WaitsVgpr6 = RT.getWaitForReg(AMDGPU::VGPR6);
  ASSERT_EQ(WaitsVgpr6.size(), 1u);
  EXPECT_EQ(WaitsVgpr6.begin()->Wait, 1u);

  // vgpr7 was the third (newest) load - wait=0
  auto WaitsVgpr7 = RT.getWaitForReg(AMDGPU::VGPR7);
  ASSERT_EQ(WaitsVgpr7.size(), 1u);
  EXPECT_EQ(WaitsVgpr7.begin()->Wait, 0u);
}

// Test ResourceTracker::isNonZeroWaitLegal() for VMEM WAW optimization.
// Same VmemType instructions complete in order (no wait needed for WAW).
TEST_F(AMDGPUTestBase, ResourceTracker_IsNonZeroWaitLegal) {
  // Use gfx1010 - GFX10+ has hasFlatLgkmVMemCountInOrder which is needed for
  // FLAT operations to safely use position-based waits.
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1010", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, $sgpr8_sgpr9_sgpr10_sgpr11, $vgpr0_vgpr1, $vgpr2_vgpr3_vgpr4_vgpr5
    ; Two BUFFER loads (same VmemType: VMEM_NOSAMPLER)
    $vgpr10 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    $vgpr10 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 4, 0, 0, implicit $exec
    ; Two GLOBAL loads (same VmemType: VMEM_NOSAMPLER)
    $vgpr11 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr11 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec
    ; IMAGE_LOAD (VMEM_NOSAMPLER) vs IMAGE_SAMPLE (VMEM_SAMPLER) - different types
    $vgpr12 = IMAGE_LOAD_V1_V4 $vgpr2_vgpr3_vgpr4_vgpr5, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, 1, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s32))
    $vgpr12 = IMAGE_SAMPLE_V1_V2 $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, 1, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s32))
    ; IMAGE_SAMPLE (VMEM_SAMPLER) consumed by a BUFFER store (VMEM_NOSAMPLER).
    ; The store reads the sampled value (RAW), but produces no VMEM result, so
    ; its VmemType is irrelevant - a position-based wait is legal.
    BUFFER_STORE_DWORD_OFFSET $vgpr12, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec :: (store (s32))
    ; FLAT + BUFFER
    $vgpr13 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr13 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    ; BUFFER + FLAT
    $vgpr20 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    $vgpr20 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    ; FLAT + FLAT
    $vgpr15 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr15 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec, implicit $flat_scr
    ; FLAT + GLOBAL
    $vgpr16 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr16 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    ; GLOBAL + FLAT
    $vgpr21 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr21 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    ; FLAT + SCRATCH
    $vgpr22 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr22 = SCRATCH_LOAD_DWORD $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    ; SCRATCH + FLAT
    $vgpr23 = SCRATCH_LOAD_DWORD $vgpr0, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr23 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    ; FLAT + IMAGE_LOAD (NOSAMPLER)
    $vgpr24 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr24 = IMAGE_LOAD_V1_V4 $vgpr2_vgpr3_vgpr4_vgpr5, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, 1, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s32))
    ; IMAGE_LOAD + FLAT
    $vgpr25 = IMAGE_LOAD_V1_V4 $vgpr2_vgpr3_vgpr4_vgpr5, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, 1, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s32))
    $vgpr25 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    ; FLAT + SAMPLER (different VmemType)
    $vgpr17 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr17 = IMAGE_SAMPLE_V1_V2 $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, 1, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s32))
    ; SAMPLER + FLAT
    $vgpr26 = IMAGE_SAMPLE_V1_V2 $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, 1, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s32))
    $vgpr26 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    ; DS (LDS) instructions are not VMEM
    $vgpr14 = DS_READ_B32_gfx9 $vgpr0, 0, 0, implicit $exec, implicit $m0
    $vgpr14 = DS_READ_B32_gfx9 $vgpr0, 4, 0, implicit $exec, implicit $m0
    ; FLAT + VALU (non-memory consumer)
    $vgpr27 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr27 = V_MOV_B32_e32 $vgpr0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  ASSERT_TRUE(ST->hasFlatLgkmVMemCountInOrder())
      << "gfx1010 should have hasFlatLgkmVMemCountInOrder";

  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 32> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }

  MachineInstr *BufferLoad1 = Instrs[0];
  MachineInstr *BufferLoad2 = Instrs[1];
  MachineInstr *GlobalLoad1 = Instrs[2];
  MachineInstr *GlobalLoad2 = Instrs[3];
  MachineInstr *ImageLoad1 = Instrs[4];
  MachineInstr *ImageSample1 = Instrs[5];
  // IMAGE_SAMPLE -> store consumer
  MachineInstr *StoreAfterSample = Instrs[6];
  // FLAT + BUFFER
  MachineInstr *FlatBeforeBuffer = Instrs[7];
  MachineInstr *BufferAfterFlat = Instrs[8];
  // BUFFER + FLAT
  MachineInstr *BufferBeforeFlat = Instrs[9];
  MachineInstr *FlatAfterBuffer = Instrs[10];
  // FLAT + FLAT
  MachineInstr *FlatLoad1 = Instrs[11];
  MachineInstr *FlatLoad2 = Instrs[12];
  // FLAT + GLOBAL
  MachineInstr *FlatBeforeGlobal = Instrs[13];
  MachineInstr *GlobalAfterFlat = Instrs[14];
  // GLOBAL + FLAT
  MachineInstr *GlobalBeforeFlat = Instrs[15];
  MachineInstr *FlatAfterGlobal = Instrs[16];
  // FLAT + SCRATCH
  MachineInstr *FlatBeforeScratch = Instrs[17];
  MachineInstr *ScratchAfterFlat = Instrs[18];
  // SCRATCH + FLAT
  MachineInstr *ScratchBeforeFlat = Instrs[19];
  MachineInstr *FlatAfterScratch = Instrs[20];
  // FLAT + IMAGE_LOAD
  MachineInstr *FlatBeforeImageLoad = Instrs[21];
  MachineInstr *ImageLoadAfterFlat = Instrs[22];
  // IMAGE_LOAD + FLAT
  MachineInstr *ImageLoadBeforeFlat = Instrs[23];
  MachineInstr *FlatAfterImageLoad = Instrs[24];
  // FLAT + SAMPLER
  MachineInstr *FlatBeforeSampler = Instrs[25];
  MachineInstr *SamplerAfterFlat = Instrs[26];
  // SAMPLER + FLAT
  MachineInstr *SamplerBeforeFlat = Instrs[27];
  MachineInstr *FlatAfterSampler = Instrs[28];
  // DS
  MachineInstr *DSRead1 = Instrs[29];
  MachineInstr *DSRead2 = Instrs[30];
  // FLAT + VALU
  MachineInstr *FlatBeforeValu = Instrs[31];
  MachineInstr *ValuAfterFlat = Instrs[32];

  const AMDGPU::Counter &VmCntr = RT.getCounter(AMDGPU::VmCnt());
  const AMDGPU::Counter &LgkmCntr = RT.getCounter(AMDGPU::LgkmCnt());
  AMDGPU::SchedulingMode SchedMode = AMDGPU::SchedulingMode::NoExpert;

  // Same VmemType (VMEM_NOSAMPLER): should complete in order for vmcnt
  EXPECT_TRUE(
      VmCntr.isNonZeroWaitLegal(*BufferLoad1, *BufferLoad2, *ST, SchedMode))
      << "BUFFER + BUFFER should complete in order for vmcnt";
  EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*GlobalLoad1, *GlobalLoad2, *ST, SchedMode))
      << "GLOBAL + GLOBAL should complete in order for vmcnt";
  EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*BufferLoad1, *GlobalLoad1, *ST, SchedMode))
      << "BUFFER + GLOBAL (both VMEM_NOSAMPLER) should complete in order";

  // Different VmemType: should NOT complete in order
  EXPECT_FALSE(VmCntr.isNonZeroWaitLegal(*ImageLoad1, *ImageSample1, *ST, SchedMode))
      << "IMAGE_LOAD (NOSAMPLER) + IMAGE_SAMPLE (SAMPLER) should NOT complete "
         "in order";

  // Different VmemType but the consumer is a store: a store produces no VMEM
  // result, so its VmemType is irrelevant and a position-based wait is legal.
  // This differs from the WAW case above (IMAGE_LOAD + IMAGE_SAMPLE) where the
  // consumer is itself a result-producing load.
  EXPECT_TRUE(
      VmCntr.isNonZeroWaitLegal(*ImageSample1, *StoreAfterSample, *ST, SchedMode))
      << "IMAGE_SAMPLE (SAMPLER) -> BUFFER store should complete in order for "
         "vmcnt (store consumer's VmemType is irrelevant)";

  // FLAT + BUFFER for register WAW :
  // FLAT's global memory path is in-order with BUFFER for vmcnt, so no vmcnt
  // wait is needed. The lgkmcnt wait handles FLAT's potential LDS access.
  // On gfx9+ with hasFlatLgkmVMemCountInOrder, pure FLAT also completes in-order on
  // lgkmcnt (LDS path).
  EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*FlatBeforeBuffer, *BufferAfterFlat, *ST, SchedMode))
      << "FLAT + BUFFER should complete in order for vmcnt";
  EXPECT_TRUE(LgkmCntr.isNonZeroWaitLegal(*FlatBeforeBuffer, *BufferAfterFlat, *ST, SchedMode))
      << "FLAT + BUFFER should complete in order for lgkmcnt (gfx10+ hasFlatLgkmVMemCountInOrder)";

  // BUFFER + FLAT: when no FLAT is pending on the counter, all pending ops are
  // VMEM-only (same event type), so position-based waits are safe even though the
  // consumer is a pure FLAT. Matches the old pass's hasPendingFlat() check.
  EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*BufferBeforeFlat, *FlatAfterBuffer, *ST, SchedMode))
      << "BUFFER + FLAT with no pending FLAT should be legal for vmcnt";
  EXPECT_FALSE(LgkmCntr.isNonZeroWaitLegal(*BufferBeforeFlat, *FlatAfterBuffer, *ST, SchedMode))
      << "BUFFER + FLAT should NOT complete in order for lgkmcnt";

  // FLAT + FLAT for register WAW :
  // On gfx9+ with hasFlatLgkmVMemCountInOrder, pure FLAT completes in-order on both
  // vmcnt (global path) and lgkmcnt (LDS path). The old pass emits position-
  // based waits (vmcnt 1, lgkmcnt 1) on gfx10+ for FLAT->FLAT.
  EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*FlatLoad1, *FlatLoad2, *ST, SchedMode))
      << "FLAT + FLAT should complete in order for vmcnt (gfx10+ hasFlatLgkmVMemCountInOrder)";
  EXPECT_TRUE(LgkmCntr.isNonZeroWaitLegal(*FlatLoad1, *FlatLoad2, *ST, SchedMode))
      << "FLAT + FLAT should complete in order for lgkmcnt (gfx10+ hasFlatLgkmVMemCountInOrder)";

  // FLAT + GLOBAL for register WAW :
  // FLAT's global memory path is in-order with GLOBAL for vmcnt.
  // On gfx9+ with hasFlatLgkmVMemCountInOrder, FLAT also completes in-order on lgkmcnt.
  EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*FlatBeforeGlobal, *GlobalAfterFlat, *ST, SchedMode))
      << "FLAT + GLOBAL should complete in order for vmcnt";
  EXPECT_TRUE(LgkmCntr.isNonZeroWaitLegal(*FlatBeforeGlobal, *GlobalAfterFlat, *ST, SchedMode))
      << "FLAT + GLOBAL should complete in order for lgkmcnt (gfx10+ hasFlatLgkmVMemCountInOrder)";

  // GLOBAL + FLAT: when no FLAT is pending on the counter, position-based waits
  // are safe (all pending ops are VMEM-only). Matches the old pass.
  EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*GlobalBeforeFlat, *FlatAfterGlobal, *ST, SchedMode))
      << "GLOBAL + FLAT with no pending FLAT should be legal for vmcnt";
  EXPECT_FALSE(LgkmCntr.isNonZeroWaitLegal(*GlobalBeforeFlat, *FlatAfterGlobal, *ST, SchedMode))
      << "GLOBAL + FLAT should NOT complete in order for lgkmcnt";

  // FLAT + SCRATCH for register WAW :
  // FLAT's global memory path is in-order with SCRATCH for vmcnt.
  // With hasFlatLgkmVMemCountInOrder, FLAT also completes in-order on lgkmcnt.
  EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*FlatBeforeScratch, *ScratchAfterFlat, *ST, SchedMode))
      << "FLAT + SCRATCH should complete in order for vmcnt";
  EXPECT_TRUE(LgkmCntr.isNonZeroWaitLegal(*FlatBeforeScratch, *ScratchAfterFlat, *ST, SchedMode))
      << "FLAT + SCRATCH should complete in order for lgkmcnt (hasFlatLgkmVMemCountInOrder)";

  // SCRATCH + FLAT: when no FLAT is pending on the counter, position-based waits
  // are safe. Matches the old pass's hasPendingFlat() check.
  EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*ScratchBeforeFlat, *FlatAfterScratch, *ST, SchedMode))
      << "SCRATCH + FLAT with no pending FLAT should be legal for vmcnt";
  EXPECT_FALSE(
      LgkmCntr.isNonZeroWaitLegal(*ScratchBeforeFlat, *FlatAfterScratch, *ST, SchedMode))
      << "SCRATCH + FLAT should NOT complete in order for lgkmcnt";

  // FLAT + IMAGE_LOAD for register WAW :
  // FLAT's global memory path is in-order with IMAGE_LOAD (NOSAMPLER) for vmcnt.
  // With hasFlatLgkmVMemCountInOrder, FLAT also completes in-order on lgkmcnt.
  EXPECT_TRUE(
      VmCntr.isNonZeroWaitLegal(*FlatBeforeImageLoad, *ImageLoadAfterFlat, *ST, SchedMode))
      << "FLAT + IMAGE_LOAD should complete in order for vmcnt";
  EXPECT_TRUE(
      LgkmCntr.isNonZeroWaitLegal(*FlatBeforeImageLoad, *ImageLoadAfterFlat, *ST, SchedMode))
      << "FLAT + IMAGE_LOAD should complete in order for lgkmcnt (hasFlatLgkmVMemCountInOrder)";

  // IMAGE_LOAD + FLAT: when no FLAT is pending, position-based waits are safe.
  EXPECT_TRUE(
      VmCntr.isNonZeroWaitLegal(*ImageLoadBeforeFlat, *FlatAfterImageLoad, *ST, SchedMode))
      << "IMAGE_LOAD + FLAT with no pending FLAT should be legal for vmcnt";
  EXPECT_FALSE(
      LgkmCntr.isNonZeroWaitLegal(*ImageLoadBeforeFlat, *FlatAfterImageLoad, *ST, SchedMode))
      << "IMAGE_LOAD + FLAT should NOT complete in order for lgkmcnt";

  // FLAT + SAMPLER for register WAW :
  // FLAT's global memory path is in-order with SAMPLER for vmcnt.
  // With hasFlatLgkmVMemCountInOrder, FLAT also completes in-order on lgkmcnt.
  EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*FlatBeforeSampler, *SamplerAfterFlat, *ST, SchedMode))
      << "FLAT + SAMPLER should complete in order for vmcnt";
  EXPECT_TRUE(LgkmCntr.isNonZeroWaitLegal(*FlatBeforeSampler, *SamplerAfterFlat, *ST, SchedMode))
      << "FLAT + SAMPLER should complete in order for lgkmcnt (hasFlatLgkmVMemCountInOrder)";

  // SAMPLER + FLAT: when no FLAT is pending, position-based waits are safe.
  EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*SamplerBeforeFlat, *FlatAfterSampler, *ST, SchedMode))
      << "SAMPLER + FLAT with no pending FLAT should be legal for vmcnt";
  EXPECT_FALSE(
      LgkmCntr.isNonZeroWaitLegal(*SamplerBeforeFlat, *FlatAfterSampler, *ST, SchedMode))
      << "SAMPLER + FLAT should NOT complete in order for lgkmcnt";

  // BUFFER + FLAT with a pending FLAT on the counter: on gfx1010+
  // (hasFlatLgkmVMemCountInOrder), FLAT completes in order so the early
  // completion concern doesn't apply and position-based waits are legal.
  // On pre-gfx10, the pending FLAT could cause early completion, forcing
  // vmcnt(0).
  {
    AMDGPU::ResourceTracker RTWithFlat(ST, /*AA=*/nullptr,
                                       AMDGPU::SchedulingMode::NoExpert);
    // Track a pure FLAT load into the counter, then check BUFFER → FLAT.
    RTWithFlat.track(*FlatLoad1);
    const AMDGPU::Counter &VmWithFlat =
        RTWithFlat.getCounter(AMDGPU::VmCnt());
    // gfx1010 has hasFlatLgkmVMemCountInOrder, so non-zero waits are legal.
    EXPECT_TRUE(VmWithFlat.isNonZeroWaitLegal(*BufferBeforeFlat,
                                               *FlatAfterBuffer, *ST, SchedMode))
        << "BUFFER + FLAT with pending FLAT on gfx1010+ should be legal";
  }

  // Same scenario on pre-gfx10 (gfx900): FLAT can report early completion,
  // so a pending FLAT on VmCnt forces vmcnt(0).
  {
    auto TM_GFX9 = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx900", "");
    ASSERT_TRUE(TM_GFX9) << "No target machine for gfx900";
    StringRef MIRStringGFX9 = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, $vgpr0_vgpr1
    $vgpr10 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    $vgpr11 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr12 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec, implicit $flat_scr
    S_ENDPGM 0
...
)MIR";
    LLVMContext Ctx9;
    MachineModuleInfo MMI9(TM_GFX9.get());
    auto M9 = parseMIR(Ctx9, *TM_GFX9, MIRStringGFX9, "test", MMI9);
    ASSERT_TRUE(M9) << "Failed to parse MIR for gfx900";
    auto *MF9 = MMI9.getMachineFunction(*M9->getFunction("test"));
    ASSERT_TRUE(MF9);
    auto *MBB9 = MF9->getBlockNumbered(0);
    ASSERT_TRUE(MBB9);
    const GCNSubtarget *ST9 = &MF9->getSubtarget<GCNSubtarget>();
    ASSERT_FALSE(ST9->hasFlatLgkmVMemCountInOrder())
        << "gfx900 should NOT have hasFlatLgkmVMemCountInOrder";

    AMDGPU::ResourceTracker RT9(ST9, /*AA=*/nullptr,
                                AMDGPU::SchedulingMode::NoExpert);
    auto It9 = MBB9->begin();
    MachineInstr &Buffer9 = *It9++;
    MachineInstr &Flat9_1 = *It9++;
    MachineInstr &Flat9_2 = *It9++;

    RT9.track(Buffer9);
    RT9.track(Flat9_1);
    const AMDGPU::Counter &Vm9 = RT9.getCounter(AMDGPU::VmCnt());
    // Pre-gfx10: pending FLAT can cause early completion, forcing vmcnt(0).
    EXPECT_FALSE(Vm9.isNonZeroWaitLegal(Buffer9, Flat9_2, *ST9,
                                         AMDGPU::SchedulingMode::NoExpert))
        << "BUFFER + FLAT with pending FLAT on pre-gfx10 should force vmcnt(0)";
  }

  // DS operations on pre-gfx12 use lgkmcnt. When only DS reads (or only writes)
  // are pending (no mixed types), they complete in order and position-based
  // waits are safe. This matches the old pass behavior (lgkmcnt 1 for DS→DS).
  EXPECT_TRUE(LgkmCntr.isNonZeroWaitLegal(*DSRead1, *DSRead2, *ST, SchedMode))
      << "DS + DS should complete in order for lgkmcnt (pre-gfx12, same type)";
  // A DS consumer is on a different counter (lgkmcnt), but that's irrelevant to
  // the VmCnt ordering of the BUFFER loads — position-based vmcnt waits are valid.
  EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*BufferLoad1, *DSRead1, *ST, SchedMode))
      << "BUFFER + DS: DS consumer doesn't affect VmCnt load ordering";

  // FLAT + VALU for register WAW :
  // VALU doesn't use any memory counter, so FLAT always completes before VALU
  // needs the result. Position-based waits are safe.
  EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*FlatBeforeValu, *ValuAfterFlat, *ST, SchedMode))
      << "FLAT + VALU should complete in order for vmcnt";
  EXPECT_TRUE(LgkmCntr.isNonZeroWaitLegal(*FlatBeforeValu, *ValuAfterFlat, *ST, SchedMode))
      << "FLAT + VALU should complete in order for lgkmcnt";

  // --- GFX12-specific behavior ---
  // On GFX12+, FLAT→GLOBAL optimization is disabled by default (per-counter
  // FLAT optimizations are conservative). However, FLAT→VALU and FLAT→FLAT
  // still complete in order.
  {
    auto TM12 = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
    ASSERT_TRUE(TM12) << "No target machine for gfx1200";

    LLVMContext Context12;
    MachineModuleInfo MMI12(TM12.get());
    auto M12 = parseMIR(Context12, *TM12, MIRString, "test", MMI12);
    ASSERT_TRUE(M12) << "Failed to parse MIR for gfx1200";

    auto *MF12 = MMI12.getMachineFunction(*M12->getFunction("test"));
    ASSERT_TRUE(MF12) << "Failed to get MachineFunction for gfx1200";

    const GCNSubtarget *ST12 = &MF12->getSubtarget<GCNSubtarget>();
    AMDGPU::ResourceTracker RT12(ST12, /*AA=*/nullptr,
                                 AMDGPU::SchedulingMode::NoExpert);

    auto *MBB12 = MF12->getBlockNumbered(0);
    ASSERT_TRUE(MBB12) << "Failed to get BB0 for gfx1200";

    SmallVector<MachineInstr *, 32> Instrs12;
    for (MachineInstr &MI : *MBB12) {
      if (!MI.isTerminator())
        Instrs12.push_back(&MI);
    }

    // Use GFX12 counters
    const AMDGPU::Counter &LoadCntr = RT12.getCounter(AMDGPU::LoadCnt());
    const AMDGPU::Counter &DsCntr = RT12.getCounter(AMDGPU::DsCnt());

    // On GFX12+, FLAT→GLOBAL does NOT complete in order for loadcnt
    // (optimization disabled by default). But for dscnt, GLOBAL doesn't use
    // dscnt so FLAT always completes before GLOBAL needs dscnt.
    MachineInstr *FlatBeforeGlobal12 = Instrs12[13];
    MachineInstr *GlobalAfterFlat12 = Instrs12[14];
    EXPECT_FALSE(
        LoadCntr.isNonZeroWaitLegal(*FlatBeforeGlobal12, *GlobalAfterFlat12, *ST12, SchedMode))
        << "GFX12: FLAT + GLOBAL should NOT complete in order for loadcnt";
    EXPECT_TRUE(
        DsCntr.isNonZeroWaitLegal(*FlatBeforeGlobal12, *GlobalAfterFlat12, *ST12, SchedMode))
        << "GFX12: FLAT + GLOBAL should complete in order for dscnt (GLOBAL doesn't use dscnt)";

    // On GFX12+, FLAT→FLAT does NOT complete in order for loadcnt (conservative)
    // but does for dscnt (both are pure FLAT, same LDS path).
    MachineInstr *FlatLoad1_12 = Instrs12[11];
    MachineInstr *FlatLoad2_12 = Instrs12[12];
    EXPECT_FALSE(LoadCntr.isNonZeroWaitLegal(*FlatLoad1_12, *FlatLoad2_12, *ST12, SchedMode))
        << "GFX12: FLAT + FLAT should NOT complete in order for loadcnt (conservative)";
    EXPECT_TRUE(DsCntr.isNonZeroWaitLegal(*FlatLoad1_12, *FlatLoad2_12, *ST12, SchedMode))
        << "GFX12: FLAT + FLAT should complete in order for dscnt";

    // FLAT→VALU still completes in order (VALU doesn't use memory counter)
    MachineInstr *FlatBeforeValu12 = Instrs12[31];
    MachineInstr *ValuAfterFlat12 = Instrs12[32];
    EXPECT_TRUE(
        LoadCntr.isNonZeroWaitLegal(*FlatBeforeValu12, *ValuAfterFlat12, *ST12, SchedMode))
        << "GFX12: FLAT + VALU should complete in order for loadcnt";
    EXPECT_TRUE(DsCntr.isNonZeroWaitLegal(*FlatBeforeValu12, *ValuAfterFlat12, *ST12, SchedMode))
        << "GFX12: FLAT + VALU should complete in order for dscnt";
  }
}

// Test DS operations on gfx12+ with DsCnt. LDS operations (reads and writes)
// complete in order because they're the same event type (LDS_ACCESS).
// Only when both LDS and GDS operations are pending do they complete out of
// order relative to each other.
TEST_F(AMDGPUTestBase, ResourceTracker_IsNonZeroWaitLegal_DsCnt) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0, $vgpr1, $vgpr2, $m0
    ; LDS reads
    $vgpr10 = DS_READ_B32_gfx9 $vgpr0, 0, 0, implicit $exec, implicit $m0
    $vgpr11 = DS_READ_B32_gfx9 $vgpr0, 4, 0, implicit $exec, implicit $m0
    ; LDS writes
    DS_WRITE_B32_gfx9 $vgpr0, $vgpr1, 0, 0, implicit $exec, implicit $m0
    DS_WRITE_B32_gfx9 $vgpr0, $vgpr2, 4, 0, implicit $exec, implicit $m0
    ; GDS operation (DS_GWS_INIT is always GDS)
    DS_GWS_INIT $vgpr0, 0, implicit $m0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 8> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }

  MachineInstr *DSRead1 = Instrs[0];
  MachineInstr *DSRead2 = Instrs[1];
  MachineInstr *DSWrite1 = Instrs[2];
  MachineInstr *DSWrite2 = Instrs[3];
  MachineInstr *GWSInit = Instrs[4];

  const AMDGPU::Counter &DsCntr = RT.getCounter(AMDGPU::DsCnt());
  AMDGPU::SchedulingMode SchedMode = AMDGPU::SchedulingMode::NoExpert;

  // With no instructions tracked, no mixed types, so DS ops complete in order.
  EXPECT_TRUE(DsCntr.isNonZeroWaitLegal(*DSRead1, *DSRead2, *ST, SchedMode))
      << "DS reads should complete in order when no mixed types pending";
  EXPECT_TRUE(DsCntr.isNonZeroWaitLegal(*DSWrite1, *DSWrite2, *ST, SchedMode))
      << "DS writes should complete in order when no mixed types pending";
  EXPECT_TRUE(DsCntr.isNonZeroWaitLegal(*DSRead1, *DSWrite1, *ST, SchedMode))
      << "DS read + write should complete in order when no mixed types pending";

  // Track a DS read - still no mixed types (only LDS).
  RT.track(*DSRead1);
  EXPECT_TRUE(DsCntr.isNonZeroWaitLegal(*DSRead1, *DSRead2, *ST, SchedMode))
      << "DS reads should complete in order with only LDS pending";

  // Track a DS write - still no mixed types (both are LDS operations).
  RT.track(*DSWrite1);
  EXPECT_TRUE(DsCntr.isNonZeroWaitLegal(*DSRead1, *DSRead2, *ST, SchedMode))
      << "LDS read + LDS write are same event type, should complete in order";
  EXPECT_TRUE(DsCntr.isNonZeroWaitLegal(*DSWrite1, *DSWrite2, *ST, SchedMode))
      << "LDS read + LDS write are same event type, should complete in order";

  // Track a GDS operation - now we have mixed types (LDS + GDS).
  RT.track(*GWSInit);
  EXPECT_FALSE(DsCntr.isNonZeroWaitLegal(*DSRead1, *DSRead2, *ST, SchedMode))
      << "LDS + GDS mixed types, should NOT complete in order";
  EXPECT_FALSE(DsCntr.isNonZeroWaitLegal(*DSWrite1, *DSWrite2, *ST, SchedMode))
      << "LDS + GDS mixed types, should NOT complete in order";
}

// Test XCnt isNonZeroWaitLegal on gfx1250+. VMEM address translation completes
// in order, but SMEM can complete out of order. When SMEM is pending on XCnt,
// position-based waits are not safe.
TEST_F(AMDGPUTestBase, ResourceTracker_IsNonZeroWaitLegal_XCnt) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1250", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $sgpr0_sgpr1
    ; VMEM loads (in-order for XCnt)
    $vgpr10 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr11 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec
    ; SMEM load (out-of-order for XCnt)
    $sgpr10 = S_LOAD_DWORD_IMM $sgpr0_sgpr1, 0, 0 :: (load (s32))
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 8> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 3u);

  MachineInstr *GlobalLoad1 = Instrs[0];
  MachineInstr *GlobalLoad2 = Instrs[1];
  MachineInstr *SmemLoad = Instrs[2];

  const AMDGPU::Counter &XCntr = RT.getCounter(AMDGPU::XCnt());
  AMDGPU::SchedulingMode SchedMode = AMDGPU::SchedulingMode::NoExpert;

  // With no instructions tracked, VMEM completes in order for XCnt.
  EXPECT_TRUE(XCntr.isNonZeroWaitLegal(*GlobalLoad1, *GlobalLoad2, *ST, SchedMode))
      << "VMEM-only XCnt should complete in order";

  // Track a VMEM load - still no mixed types (VMEM only).
  RT.track(*GlobalLoad1);
  EXPECT_TRUE(XCntr.isNonZeroWaitLegal(*GlobalLoad1, *GlobalLoad2, *ST, SchedMode))
      << "VMEM-only XCnt should complete in order with VMEM pending";

  // Track an SMEM load - now we have mixed types (VMEM + SMEM).
  // SMEM can complete out of order, so XCnt is no longer safe for non-zero waits.
  RT.track(*SmemLoad);
  EXPECT_FALSE(XCntr.isNonZeroWaitLegal(*GlobalLoad1, *GlobalLoad2, *ST, SchedMode))
      << "XCnt with SMEM pending should NOT complete in order";

  // When the source instruction we are waiting for is itself SMEM, position-
  // based waits are never legal: SMEM address translation completes out of
  // order regardless of what else is pending.
  EXPECT_FALSE(XCntr.isNonZeroWaitLegal(*SmemLoad, *GlobalLoad2, *ST, SchedMode))
      << "XCnt with SMEM source should NOT complete in order";
}

// Test the FLAT early completion workaround. On pre-GFX10 targets, FLAT
// operations can report early completion (counter decrements before the memory
// operation actually completes). When there's a pending FLAT, position-based
// wait values are unsafe and we must wait for 0.
TEST_F(AMDGPUTestBase, ResourceTracker_NeedsFlatEarlyCompletionWorkaround) {
  // gfx90a is pre-GFX10 (GFX9), so hasFlatLgkmVMemCountInOrder() returns false.
  auto TM_GFX9 = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx90a", "");
  ASSERT_TRUE(TM_GFX9) << "No target machine for gfx90a";

  // gfx1030 is GFX10+, so hasFlatLgkmVMemCountInOrder() returns true.
  auto TM_GFX10 = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1030", "");
  ASSERT_TRUE(TM_GFX10) << "No target machine for gfx1030";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $vgpr2, $vgpr3
    ; Pure FLAT load (not GLOBAL_* or SCRATCH_*)
    $vgpr10 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    ; GLOBAL load (VMEM-only, not pure FLAT)
    $vgpr11 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  // Test with GFX9 (pre-GFX10)
  {
    LLVMContext Context;
    MachineModuleInfo MMI(TM_GFX9.get());
    auto M = parseMIR(Context, *TM_GFX9, MIRString, "test", MMI);
    ASSERT_TRUE(M) << "Failed to parse MIR for gfx90a";

    auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
    ASSERT_TRUE(MF) << "Failed to get MachineFunction";

    const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
    ASSERT_FALSE(ST->hasFlatLgkmVMemCountInOrder())
        << "gfx90a should NOT have hasFlatLgkmVMemCountInOrder";

    AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

    auto *MBB = MF->getBlockNumbered(0);
    SmallVector<MachineInstr *, 4> Instrs;
    for (MachineInstr &MI : *MBB) {
      if (!MI.isTerminator())
        Instrs.push_back(&MI);
    }
    ASSERT_EQ(Instrs.size(), 2u);

    MachineInstr *FlatLoad = Instrs[0];
    MachineInstr *GlobalLoad = Instrs[1];

    // Track both loads
    RT.track(*FlatLoad);
    RT.track(*GlobalLoad);

    // Get the VmCnt counter
    const AMDGPU::Counter &VmCntr = RT.getCounter(AMDGPU::VmCnt());

    // On gfx90a with pending FLAT, workaround should be needed
    EXPECT_TRUE(VmCntr.needsFlatEarlyCompletionWorkaround(*ST))
        << "Workaround needed on gfx90a with pending FLAT";
  }

  // Test with GFX10+ (has hasFlatLgkmVMemCountInOrder)
  {
    LLVMContext Context;
    MachineModuleInfo MMI(TM_GFX10.get());
    auto M = parseMIR(Context, *TM_GFX10, MIRString, "test", MMI);
    ASSERT_TRUE(M) << "Failed to parse MIR for gfx1030";

    auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
    ASSERT_TRUE(MF) << "Failed to get MachineFunction";

    const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
    ASSERT_TRUE(ST->hasFlatLgkmVMemCountInOrder())
        << "gfx1030 should have hasFlatLgkmVMemCountInOrder";

    AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

    auto *MBB = MF->getBlockNumbered(0);
    SmallVector<MachineInstr *, 4> Instrs;
    for (MachineInstr &MI : *MBB) {
      if (!MI.isTerminator())
        Instrs.push_back(&MI);
    }
    ASSERT_EQ(Instrs.size(), 2u);

    MachineInstr *FlatLoad = Instrs[0];
    MachineInstr *GlobalLoad = Instrs[1];

    // Track both loads
    RT.track(*FlatLoad);
    RT.track(*GlobalLoad);

    // Get the VmCnt counter (gfx1030 is pre-GFX12, so uses VmCnt not LoadCnt)
    const AMDGPU::Counter &VmCntr = RT.getCounter(AMDGPU::VmCnt());

    // On gfx1030, workaround should NOT be needed (hardware fixed)
    EXPECT_FALSE(VmCntr.needsFlatEarlyCompletionWorkaround(*ST))
        << "Workaround NOT needed on gfx1030 (hardware fixed)";
  }

  // Test with GFX9 but only GLOBAL (no pure FLAT)
  {
    StringRef MIRStringGlobalOnly = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1
    ; GLOBAL load only (not pure FLAT)
    $vgpr10 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

    LLVMContext Context;
    MachineModuleInfo MMI(TM_GFX9.get());
    auto M = parseMIR(Context, *TM_GFX9, MIRStringGlobalOnly, "test", MMI);
    ASSERT_TRUE(M) << "Failed to parse MIR for gfx90a (GLOBAL only)";

    auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
    ASSERT_TRUE(MF) << "Failed to get MachineFunction";

    const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
    AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

    auto *MBB = MF->getBlockNumbered(0);
    SmallVector<MachineInstr *, 4> Instrs;
    for (MachineInstr &MI : *MBB) {
      if (!MI.isTerminator())
        Instrs.push_back(&MI);
    }
    ASSERT_EQ(Instrs.size(), 1u);

    MachineInstr *GlobalLoad = Instrs[0];

    // Track only GLOBAL (no pure FLAT)
    RT.track(*GlobalLoad);

    // Get the VmCnt counter
    const AMDGPU::Counter &VmCntr = RT.getCounter(AMDGPU::VmCnt());

    // On gfx90a but with no pending FLAT, workaround should NOT be needed
    EXPECT_FALSE(VmCntr.needsFlatEarlyCompletionWorkaround(*ST))
        << "Workaround NOT needed on gfx90a with only GLOBAL (no FLAT)";
  }
}

// Test Counter::getWaitFor() returns correct wait values.
// The wait value should be: size - 1 - position
// where position 0 is the oldest instruction.
TEST_F(AMDGPUTestBase, Counter_GetWaitFor) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, $vgpr0_vgpr1
    $vgpr10 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    $vgpr11 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 4, 0, 0, implicit $exec
    $vgpr12 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 8, 0, 0, implicit $exec
    $vgpr13 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 12, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 4> Loads;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Loads.push_back(&MI);
  }
  ASSERT_EQ(Loads.size(), 4u);

  // Create a counter and insert instructions in order (oldest first).
  AMDGPU::Counter Ctr{AMDGPU::LoadCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
  Ctr.insert(Loads[0]); // position 0 (oldest)
  Ctr.insert(Loads[1]); // position 1
  Ctr.insert(Loads[2]); // position 2
  Ctr.insert(Loads[3]); // position 3 (newest)

  ASSERT_EQ(Ctr.size(), 4u);

  // Hardware counter starts at 4 and decrements as instructions complete
  // (oldest first):
  //   counter=4: nothing complete
  //   counter=3: Loads[0] complete
  //   counter=2: Loads[0,1] complete
  //   counter=1: Loads[0,1,2] complete
  //   counter=0: all complete
  //
  // To wait for Loads[0] (oldest), need counter <= 3 → wait=3
  // To wait for Loads[1], need counter <= 2 → wait=2
  // To wait for Loads[2], need counter <= 1 → wait=1
  // To wait for Loads[3] (newest), need counter <= 0 → wait=0
  //
  // Formula: wait = size - 1 - position

  auto Wait0 = Ctr.getWaitFor(*Loads[0]);
  ASSERT_TRUE(Wait0.has_value());
  EXPECT_EQ(*Wait0, 3u) << "Oldest instruction (position 0) should have wait=3";

  auto Wait1 = Ctr.getWaitFor(*Loads[1]);
  ASSERT_TRUE(Wait1.has_value());
  EXPECT_EQ(*Wait1, 2u) << "Position 1 should have wait=2";

  auto Wait2 = Ctr.getWaitFor(*Loads[2]);
  ASSERT_TRUE(Wait2.has_value());
  EXPECT_EQ(*Wait2, 1u) << "Position 2 should have wait=1";

  auto Wait3 = Ctr.getWaitFor(*Loads[3]);
  ASSERT_TRUE(Wait3.has_value());
  EXPECT_EQ(*Wait3, 0u) << "Newest instruction (position 3) should have wait=0";

  // Test with instruction not in the counter.
  MachineInstr *Terminator = &MBB->back();
  auto WaitNotFound = Ctr.getWaitFor(*Terminator);
  EXPECT_FALSE(WaitNotFound.has_value())
      << "Instruction not in counter should return nullopt";
}

// Test Counter::hasMixedEventTypes() for DS_CNT counter.
// Returns true when both LDS and GDS operations are pending.
// LDS reads and writes are the same event type (LDS_ACCESS), so they don't
// cause mixed types. Only LDS + GDS causes mixed types.
TEST_F(AMDGPUTestBase, Counter_HasMixedEventTypes) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0, $vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, $m0
    ; LDS reads
    $vgpr10 = DS_READ_B32_gfx9 $vgpr0, 0, 0, implicit $exec, implicit $m0
    $vgpr11 = DS_READ_B32_gfx9 $vgpr0, 4, 0, implicit $exec, implicit $m0
    ; LDS writes
    DS_WRITE_B32_gfx9 $vgpr0, $vgpr1, 8, 0, implicit $exec, implicit $m0
    DS_WRITE_B32_gfx9 $vgpr0, $vgpr1, 12, 0, implicit $exec, implicit $m0
    ; GDS operation (DS_GWS_INIT is always GDS)
    DS_GWS_INIT $vgpr0, 0, implicit $m0, implicit $exec
    ; VMEM load (for testing non-DS counter)
    $vgpr12 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST->getInstrInfo();

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 8> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 6u);

  MachineInstr *DSRead1 = Instrs[0];
  MachineInstr *DSRead2 = Instrs[1];
  MachineInstr *DSWrite1 = Instrs[2];
  MachineInstr *DSWrite2 = Instrs[3];
  MachineInstr *GWSInit = Instrs[4];
  MachineInstr *BufferLoad = Instrs[5];

  // Test DsCnt counter with LDS operations only: no mixed types
  {
    AMDGPU::Counter DsCtr{AMDGPU::DsCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};

    // Empty counter: no mixed types
    EXPECT_FALSE(DsCtr.hasMixedEventTypes(*TII))
        << "Empty counter should not have mixed types";

    // Single LDS read: no mixed types
    DsCtr.insert(DSRead1);
    EXPECT_FALSE(DsCtr.hasMixedEventTypes(*TII))
        << "Single LDS read should not have mixed types";

    // Two LDS reads: no mixed types
    DsCtr.insert(DSRead2);
    EXPECT_FALSE(DsCtr.hasMixedEventTypes(*TII))
        << "Only LDS reads should not have mixed types";

    // Add LDS write: still no mixed types (both are LDS_ACCESS)
    DsCtr.insert(DSWrite1);
    EXPECT_FALSE(DsCtr.hasMixedEventTypes(*TII))
        << "LDS reads + LDS write are same event type, no mixed types";

    // Add another LDS write: still no mixed types
    DsCtr.insert(DSWrite2);
    EXPECT_FALSE(DsCtr.hasMixedEventTypes(*TII))
        << "LDS reads + LDS writes are same event type, no mixed types";
  }

  // Test with only LDS writes: no mixed types
  {
    AMDGPU::Counter DsCtr{AMDGPU::DsCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    DsCtr.insert(DSWrite1);
    EXPECT_FALSE(DsCtr.hasMixedEventTypes(*TII))
        << "Single LDS write should not have mixed types";

    DsCtr.insert(DSWrite2);
    EXPECT_FALSE(DsCtr.hasMixedEventTypes(*TII))
        << "Only LDS writes should not have mixed types";
  }

  // Test LDS + GDS: mixed types
  {
    AMDGPU::Counter DsCtr{AMDGPU::DsCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    DsCtr.insert(DSRead1);
    EXPECT_FALSE(DsCtr.hasMixedEventTypes(*TII))
        << "Single LDS op should not have mixed types";

    // Add GDS operation: now mixed types (LDS + GDS)
    DsCtr.insert(GWSInit);
    EXPECT_TRUE(DsCtr.hasMixedEventTypes(*TII))
        << "LDS + GDS should have mixed types";
  }

  // Test LoadCnt counter with same VmemType: no mixed types
  {
    AMDGPU::Counter LoadCtr{AMDGPU::LoadCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    LoadCtr.insert(BufferLoad);
    EXPECT_FALSE(LoadCtr.hasMixedEventTypes(*TII))
        << "Single VMEM load should not have mixed types";

    // DS instructions are skipped (not VMEM-only), so they don't affect
    // the mixed type check for LoadCnt
    LoadCtr.insert(DSRead1);
    LoadCtr.insert(DSWrite1);
    EXPECT_FALSE(LoadCtr.hasMixedEventTypes(*TII))
        << "DS instructions should be ignored for LoadCnt mixed type check";
  }
}

// Test Counter::hasMixedEventTypes() for VMEM counters (LoadCnt/VmCnt).
// Returns true when multiple VmemTypes (NOSAMPLER, SAMPLER, BVH) are pending.
TEST_F(AMDGPUTestBase, Counter_HasMixedEventTypes_VMEM) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $vgpr2_vgpr3_vgpr4_vgpr5, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, $sgpr8_sgpr9_sgpr10_sgpr11
    ; BUFFER loads (VMEM_NOSAMPLER)
    $vgpr10 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    $vgpr11 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 4, 0, 0, implicit $exec
    ; GLOBAL load (VMEM_NOSAMPLER)
    $vgpr12 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    ; IMAGE_SAMPLE (VMEM_SAMPLER)
    $vgpr13 = IMAGE_SAMPLE_V1_V2 $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, 1, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s32))
    ; IMAGE_LOAD (VMEM_NOSAMPLER)
    $vgpr14 = IMAGE_LOAD_V1_V4 $vgpr2_vgpr3_vgpr4_vgpr5, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, 1, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s32))
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST->getInstrInfo();

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 8> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 5u);

  MachineInstr *BufferLoad1 = Instrs[0];
  MachineInstr *BufferLoad2 = Instrs[1];
  MachineInstr *GlobalLoad = Instrs[2];
  MachineInstr *ImageSample = Instrs[3];
  MachineInstr *ImageLoad = Instrs[4];

  // Test LoadCnt with same VmemType (VMEM_NOSAMPLER): no mixed types
  {
    AMDGPU::Counter LoadCtr{AMDGPU::LoadCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};

    // Empty counter
    EXPECT_FALSE(LoadCtr.hasMixedEventTypes(*TII))
        << "Empty counter should not have mixed types";

    // Single BUFFER load
    LoadCtr.insert(BufferLoad1);
    EXPECT_FALSE(LoadCtr.hasMixedEventTypes(*TII))
        << "Single VMEM load should not have mixed types";

    // Two BUFFER loads (same VmemType: NOSAMPLER)
    LoadCtr.insert(BufferLoad2);
    EXPECT_FALSE(LoadCtr.hasMixedEventTypes(*TII))
        << "Same VmemType should not have mixed types";

    // Add GLOBAL load (also NOSAMPLER)
    LoadCtr.insert(GlobalLoad);
    EXPECT_FALSE(LoadCtr.hasMixedEventTypes(*TII))
        << "BUFFER + GLOBAL (both NOSAMPLER) should not have mixed types";

    // Add IMAGE_LOAD (also NOSAMPLER)
    LoadCtr.insert(ImageLoad);
    EXPECT_FALSE(LoadCtr.hasMixedEventTypes(*TII))
        << "All NOSAMPLER types should not have mixed types";
  }

  // Test LoadCnt with different VmemTypes: mixed types
  {
    AMDGPU::Counter LoadCtr{AMDGPU::LoadCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};

    // BUFFER load (NOSAMPLER)
    LoadCtr.insert(BufferLoad1);
    EXPECT_FALSE(LoadCtr.hasMixedEventTypes(*TII))
        << "Single VMEM load should not have mixed types";

    // Add IMAGE_SAMPLE (SAMPLER) - now mixed types
    LoadCtr.insert(ImageSample);
    EXPECT_TRUE(LoadCtr.hasMixedEventTypes(*TII))
        << "NOSAMPLER + SAMPLER should have mixed types";
  }

  // Test VmCnt (pre-gfx12) behaves the same way
  {
    AMDGPU::Counter VmCtr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};

    VmCtr.insert(BufferLoad1);
    EXPECT_FALSE(VmCtr.hasMixedEventTypes(*TII))
        << "Single VMEM load should not have mixed types";

    VmCtr.insert(ImageSample);
    EXPECT_TRUE(VmCtr.hasMixedEventTypes(*TII))
        << "NOSAMPLER + SAMPLER should have mixed types for VmCnt";
  }
}

// Test that getWaitFor returns wait=0 for RAW dependencies when mixed event
// types (DS reads and writes) are pending on DsCnt.
// Test that getWaitFor returns wait=0 for RAW dependencies when mixed event
// types (LDS + GDS) are pending on DsCnt.
// Note: LDS reads and writes are the same event type (LDS_ACCESS), so they
// don't cause mixed types. Only LDS + GDS causes mixed types.
TEST_F(AMDGPUTestBase, ResourceTracker_GetWaitFor_MixedDsTypes) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0, $vgpr1, $vgpr2, $m0
    ; LDS read to vgpr10
    $vgpr10 = DS_READ_B32_gfx9 $vgpr0, 0, 0, implicit $exec, implicit $m0
    ; LDS write (same event type as LDS read, NOT mixed types)
    DS_WRITE_B32_gfx9 $vgpr0, $vgpr1, 4, 0, implicit $exec, implicit $m0
    ; GDS operation (creates mixed types: LDS + GDS)
    DS_GWS_INIT $vgpr0, 0, implicit $m0, implicit $exec
    ; Use vgpr10 - should require wait=0 when mixed types pending
    $vgpr11 = V_ADD_U32_e32 $vgpr10, $vgpr2, implicit $exec
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 8> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }

  MachineInstr *DSRead = Instrs[0];
  MachineInstr *DSWrite = Instrs[1];
  MachineInstr *GWSInit = Instrs[2];
  MachineInstr *UseVgpr10 = Instrs[3];

  // Track LDS read only - no mixed types yet.
  RT.track(*DSRead);

  // getWaitFor for vgpr10 should return position-based wait (0 since only 1
  // instruction pending).
  auto WaitsNoMixed = RT.getWaitFor(AMDGPU::VGPR10, *UseVgpr10,
                                    AMDGPU::ResourceTracker::RegAccessType::Use);
  ASSERT_EQ(WaitsNoMixed.size(), 1u);
  EXPECT_EQ(WaitsNoMixed.begin()->Wait, 0u)
      << "With only LDS read pending, wait should be position-based (0)";

  // Track LDS write - still NOT mixed types (both are LDS_ACCESS).
  RT.track(*DSWrite);

  // getWaitFor for vgpr10 should return wait=1 (position-based, 2 LDS ops
  // pending but LDS read is at position 1).
  auto WaitsLdsOnly = RT.getWaitFor(AMDGPU::VGPR10, *UseVgpr10,
                                    AMDGPU::ResourceTracker::RegAccessType::Use);
  ASSERT_EQ(WaitsLdsOnly.size(), 1u);
  EXPECT_EQ(WaitsLdsOnly.begin()->Wait, 1u)
      << "With only LDS ops pending (no mixed types), wait should be position-based (1)";

  // Now track GDS operation - creates mixed types (LDS + GDS).
  RT.track(*GWSInit);

  // getWaitFor for vgpr10 should return wait=0 because mixed types
  // forces wait=0 regardless of position.
  auto WaitsMixed = RT.getWaitFor(AMDGPU::VGPR10, *UseVgpr10,
                                  AMDGPU::ResourceTracker::RegAccessType::Use);
  ASSERT_EQ(WaitsMixed.size(), 1u);
  EXPECT_EQ(WaitsMixed.begin()->Wait, 0u)
      << "With mixed DS types (LDS + GDS) pending, wait must be 0";
}

// Test that WAW skip takes precedence over FLAT early completion workaround.
// On GFX9 (hasVmemWriteVgprInOrder but no hasFlatLgkmVMemCountInOrder), FLAT
// is pending which would normally force vmcnt=0. But for WAW where DstMI is
// VMEM-only, the skip should apply and no vmcnt wait should be generated.
TEST_F(AMDGPUTestBase, ResourceTracker_GetWaitFor_WawSkipWithFlat) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx900", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1
    ; FLAT load to vgpr2 - affects both vmcnt and lgkmcnt
    $vgpr2 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    ; GLOBAL load to vgpr2 - WAW with FLAT, VMEM-only
    $vgpr2 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  ASSERT_TRUE(ST->hasVmemWriteVgprInOrder())
      << "gfx900 should have VmemWriteVgprInOrder";
  ASSERT_FALSE(ST->hasFlatLgkmVMemCountInOrder())
      << "gfx900 should NOT have hasFlatLgkmVMemCountInOrder";

  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 4> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 2u);

  MachineInstr *FlatLoad = Instrs[0];
  MachineInstr *GlobalLoad = Instrs[1];

  // Track FLAT load
  RT.track(*FlatLoad);

  // getWaitFor for vgpr2 with GLOBAL_LOAD as DstMI (WAW, Def access type)
  // should return empty for vmcnt because WAW skip applies.
  auto Waits = RT.getWaitFor(AMDGPU::VGPR2, *GlobalLoad,
                             AMDGPU::ResourceTracker::RegAccessType::Def);

  // Check that vmcnt is NOT in the result (WAW skip applied)
  bool HasVmcnt = false;
  for (const auto &W : Waits) {
    if (W.Cntr == AMDGPU::VmCnt())
      HasVmcnt = true;
  }
  EXPECT_FALSE(HasVmcnt)
      << "WAW skip should prevent vmcnt wait even with FLAT pending";
}

// Test that WAW skip does NOT apply for different VmemTypes.
// NOSAMPLER→SAMPLER WAW should still require vmcnt=0 because different
// VmemTypes can complete out of order.
TEST_F(AMDGPUTestBase, ResourceTracker_GetWaitFor_WawDifferentVmemTypes) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx900", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, $vgpr0_vgpr1_vgpr2_vgpr3
    ; IMAGE_LOAD is NOSAMPLER type
    $vgpr4 = IMAGE_LOAD_V1_V4 $vgpr0_vgpr1_vgpr2_vgpr3, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, 2, -1, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s128))
    ; IMAGE_SAMPLE is SAMPLER type - different from NOSAMPLER
    $vgpr4 = IMAGE_SAMPLE_L_V1_V4 $vgpr0_vgpr1_vgpr2_vgpr3, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, 8, 0, 0, 0, 0, 0, -1, 0, implicit $exec :: (load (s128))
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  ASSERT_TRUE(ST->hasVmemWriteVgprInOrder())
      << "gfx900 should have VmemWriteVgprInOrder";

  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 4> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 2u);

  MachineInstr *ImageLoad = Instrs[0];   // NOSAMPLER
  MachineInstr *ImageSample = Instrs[1]; // SAMPLER

  // Track IMAGE_LOAD (NOSAMPLER)
  RT.track(*ImageLoad);

  // getWaitFor for vgpr4 with IMAGE_SAMPLE as DstMI (WAW, Def access type)
  // should return vmcnt=0 because different VmemTypes can complete out of order.
  auto Waits = RT.getWaitFor(AMDGPU::VGPR4, *ImageSample,
                             AMDGPU::ResourceTracker::RegAccessType::Def);

  // Check that vmcnt IS in the result with wait=0
  bool HasVmcnt = false;
  unsigned VmcntWait = 0;
  for (const auto &W : Waits) {
    if (W.Cntr == AMDGPU::VmCnt()) {
      HasVmcnt = true;
      VmcntWait = W.Wait;
    }
  }
  EXPECT_TRUE(HasVmcnt)
      << "Different VmemTypes (NOSAMPLER→SAMPLER) should require vmcnt wait";
  EXPECT_EQ(VmcntWait, 0u)
      << "Different VmemTypes should force vmcnt=0";
}

// Test that WAW skip does NOT apply when Point Sample Acceleration is involved.
// On GFX1150 (hasPointSampleAccel), IMAGE_SAMPLE instructions can be accelerated
// to behave like NOSAMPLER, so they may complete out of order with other SAMPLER
// instructions like IMAGE_GATHER4.
TEST_F(AMDGPUTestBase, ResourceTracker_GetWaitFor_WawPointSampleAccel) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1150", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, $vgpr0_vgpr1, $vgpr2_vgpr3_vgpr4
    ; IMAGE_SAMPLE has PointSampleAccel=1 (can be accelerated to NOSAMPLER)
    $vgpr10_vgpr11_vgpr12_vgpr13 = IMAGE_SAMPLE_V4_V2 $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, 15, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s128))
    ; IMAGE_GATHER4 has PointSampleAccel=0 (not accelerated)
    $vgpr10_vgpr11_vgpr12_vgpr13 = IMAGE_GATHER4_LZ_O_V4_V3 $vgpr2_vgpr3_vgpr4, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, 1, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s128))
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  ASSERT_TRUE(ST->hasPointSampleAccel())
      << "gfx1150 should have PointSampleAccel";
  ASSERT_TRUE(ST->hasVmemWriteVgprInOrder())
      << "gfx1150 should have VmemWriteVgprInOrder";

  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 4> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 2u);

  MachineInstr *ImageSample = Instrs[0];  // Has PointSampleAccel
  MachineInstr *ImageGather4 = Instrs[1]; // No PointSampleAccel

  // Track IMAGE_SAMPLE (has PSA)
  RT.track(*ImageSample);

  // getWaitFor for vgpr10 with IMAGE_GATHER4 as DstMI (WAW, Def access type)
  // should return vmcnt=0 because IMAGE_SAMPLE has PSA and can complete as
  // NOSAMPLER, causing out-of-order completion with GATHER4 (SAMPLER).
  auto Waits = RT.getWaitFor(AMDGPU::VGPR10, *ImageGather4,
                             AMDGPU::ResourceTracker::RegAccessType::Def);

  // Check that vmcnt IS in the result with wait=0
  bool HasVmcnt = false;
  unsigned VmcntWait = 0;
  for (const auto &W : Waits) {
    if (W.Cntr == AMDGPU::VmCnt()) {
      HasVmcnt = true;
      VmcntWait = W.Wait;
    }
  }
  EXPECT_TRUE(HasVmcnt)
      << "PSA (SAMPLE→GATHER4) should require vmcnt wait";
  EXPECT_EQ(VmcntWait, 0u)
      << "PSA should force vmcnt=0";
}

// Test that WAW skip DOES apply for GATHER4→GATHER4 (neither has PSA, same type).
TEST_F(AMDGPUTestBase, ResourceTracker_GetWaitFor_WawGather4NoSkip) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1150", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, $vgpr2_vgpr3_vgpr4
    ; Two IMAGE_GATHER4 instructions - neither has PointSampleAccel
    $vgpr10_vgpr11_vgpr12_vgpr13 = IMAGE_GATHER4_LZ_O_V4_V3 $vgpr2_vgpr3_vgpr4, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, 1, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s128))
    $vgpr10_vgpr11_vgpr12_vgpr13 = IMAGE_GATHER4_LZ_O_V4_V3 $vgpr2_vgpr3_vgpr4, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, 1, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s128))
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  ASSERT_TRUE(ST->hasPointSampleAccel())
      << "gfx1150 should have PointSampleAccel";
  ASSERT_TRUE(ST->hasVmemWriteVgprInOrder())
      << "gfx1150 should have VmemWriteVgprInOrder";

  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 4> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 2u);

  MachineInstr *Gather4_1 = Instrs[0];
  MachineInstr *Gather4_2 = Instrs[1];

  // Track first GATHER4
  RT.track(*Gather4_1);

  // getWaitFor for vgpr10 with second GATHER4 as DstMI (WAW, Def access type)
  // should return empty for vmcnt because WAW skip applies (same type, no PSA).
  auto Waits = RT.getWaitFor(AMDGPU::VGPR10, *Gather4_2,
                             AMDGPU::ResourceTracker::RegAccessType::Def);

  // Check that vmcnt is NOT in the result (WAW skip applied)
  bool HasVmcnt = false;
  for (const auto &W : Waits) {
    if (W.Cntr == AMDGPU::VmCnt())
      HasVmcnt = true;
  }
  EXPECT_FALSE(HasVmcnt)
      << "GATHER4→GATHER4 WAW skip should apply (no PSA, same type)";
}

TEST_F(AMDGPUResourceTrackerTest, Counter_Merge) {
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4
    $vgpr2 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    $vgpr3 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 4, 0, 0, implicit $exec
    $vgpr4 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 8, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);

  // Get the three load instructions
  auto It = MBB->begin();
  MachineInstr *Load1 = &*It++;
  MachineInstr *Load2 = &*It++;
  MachineInstr *Load3 = &*It++;

  // Test 1: Merge empty counter with non-empty counter
  {
    AMDGPU::Counter C1{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    AMDGPU::Counter C2{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    C2.insert(Load1);
    C2.insert(Load2);

    C1.merge(C2);
    EXPECT_EQ(C1.size(), 2u);
  }

  // Test 2: Merge non-empty counter with empty counter
  {
    AMDGPU::Counter C1{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    AMDGPU::Counter C2{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    C1.insert(Load1);
    C1.insert(Load2);

    C1.merge(C2);
    EXPECT_EQ(C1.size(), 2u);
  }

  // Test 3: Merge two counters with overlapping instructions
  {
    AMDGPU::Counter C1{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    AMDGPU::Counter C2{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    C1.insert(Load1);
    C1.insert(Load2);
    C2.insert(Load2);
    C2.insert(Load3);

    C1.merge(C2);
    // Should have Load1, Load2, Load3 (no duplicates)
    EXPECT_EQ(C1.size(), 3u);
  }

  // Test 4: Merge two counters with no overlap
  {
    AMDGPU::Counter C1{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    AMDGPU::Counter C2{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    C1.insert(Load1);
    C2.insert(Load2);
    C2.insert(Load3);

    C1.merge(C2);
    EXPECT_EQ(C1.size(), 3u);
  }

  // Test 5: Merge identical counters
  {
    AMDGPU::Counter C1{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    AMDGPU::Counter C2{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    C1.insert(Load1);
    C1.insert(Load2);
    C2.insert(Load1);
    C2.insert(Load2);

    C1.merge(C2);
    EXPECT_EQ(C1.size(), 2u);
  }
}

// Test that getWaitFor clamps the wait value to MaxSize - 1 when the counter
// has overflowed. This matches the old pass behavior where we wait for the
// oldest trackable instruction.
TEST_F(AMDGPUResourceTrackerTest, Counter_GetWaitFor_Overflow) {
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4
    $vgpr0 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    $vgpr1 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 4, 0, 0, implicit $exec
    $vgpr2 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 8, 0, 0, implicit $exec
    $vgpr3 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 12, 0, 0, implicit $exec
    $vgpr4 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 16, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);

  SmallVector<MachineInstr *, 5> Loads;
  for (MachineInstr &MI : *MBB) {
    if (MI.getOpcode() == AMDGPU::BUFFER_LOAD_DWORD_OFFSET)
      Loads.push_back(&MI);
  }
  ASSERT_EQ(Loads.size(), 5u);

  // Counter with MaxSize=3 and 5 loads.
  // Without clamping, wait values would be 4, 3, 2, 1, 0.
  // With clamping to MaxSize-1=2, wait values should be 2, 2, 2, 1, 0.
  AMDGPU::Counter C{AMDGPU::LoadCnt{}, /*MaxSize=*/3, /*DropOnOverflow=*/false};
  C.insert(Loads[0]);
  C.insert(Loads[1]);
  C.insert(Loads[2]);
  C.insert(Loads[3]);
  C.insert(Loads[4]);
  EXPECT_EQ(C.size(), 5u);

  // Loads[0] at index 0: unclamped wait = 5-1-0 = 4, clamped to 2
  EXPECT_EQ(C.getWaitFor(*Loads[0]), 2u);
  // Loads[1] at index 1: unclamped wait = 5-1-1 = 3, clamped to 2
  EXPECT_EQ(C.getWaitFor(*Loads[1]), 2u);
  // Loads[2] at index 2: unclamped wait = 5-1-2 = 2, no clamping needed
  EXPECT_EQ(C.getWaitFor(*Loads[2]), 2u);
  // Loads[3] at index 3: unclamped wait = 5-1-3 = 1, no clamping needed
  EXPECT_EQ(C.getWaitFor(*Loads[3]), 1u);
  // Loads[4] at index 4: unclamped wait = 5-1-4 = 0, no clamping needed
  EXPECT_EQ(C.getWaitFor(*Loads[4]), 0u);
}

// With DropOnOverflow, a counter that overflows past MaxSize drops its oldest
// entries instead of keeping them and clamping (the ExpCnt behavior). An entry
// that has fallen out of the window is gone, so it produces no wait.
TEST_F(AMDGPUResourceTrackerTest, Counter_DropOnOverflow) {
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4
    $vgpr0 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    $vgpr1 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 4, 0, 0, implicit $exec
    $vgpr2 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 8, 0, 0, implicit $exec
    $vgpr3 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 12, 0, 0, implicit $exec
    $vgpr4 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 16, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);

  SmallVector<MachineInstr *, 5> Loads;
  for (MachineInstr &MI : *MBB) {
    if (MI.getOpcode() == AMDGPU::BUFFER_LOAD_DWORD_OFFSET)
      Loads.push_back(&MI);
  }
  ASSERT_EQ(Loads.size(), 5u);

  // Counter with MaxSize=3, DropOnOverflow=true, inserting 5 entries. Each insert
  // past MaxSize drops the oldest, so only the 3 most recent remain.
  AMDGPU::Counter C{AMDGPU::ExpCnt{}, /*MaxSize=*/3, /*DropOnOverflow=*/true};
  C.insert(Loads[0]);
  C.insert(Loads[1]);
  C.insert(Loads[2]);
  EXPECT_EQ(C.size(), 3u);
  C.insert(Loads[3]);
  C.insert(Loads[4]);
  // Overflowed: only the last 3 (Loads[2], Loads[3], Loads[4]) are tracked.
  EXPECT_EQ(C.size(), 3u);

  // The two oldest were dropped: no wait for them.
  EXPECT_FALSE(C.getWaitFor(*Loads[0]));
  EXPECT_FALSE(C.getWaitFor(*Loads[1]));
  // The surviving three keep their position-based wait values (2, 1, 0).
  EXPECT_EQ(C.getWaitFor(*Loads[2]), 2u);
  EXPECT_EQ(C.getWaitFor(*Loads[3]), 1u);
  EXPECT_EQ(C.getWaitFor(*Loads[4]), 0u);
}

// getEffectiveDepReg widens a 16-bit VGPR to its enclosing 32-bit register only
// for a VALU consumer on targets where a D16 VALU instruction may write the
// whole VGPR (gfx11), and only when the other half has a pending op. A memory
// consumer keeps the halves independent. Full-width VGPRs and SGPRs are
// unchanged.
TEST_F(AMDGPUResourceTrackerTest, GetEffectiveDepReg) {
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $vgpr2
    renamable $vgpr0_lo16 = GLOBAL_LOAD_SHORT_D16_t16 renamable $vgpr0_vgpr1, 0, 0, implicit $exec
    renamable $vgpr0_hi16 = V_SUB_NC_U16_t16_e64 0, 0, 0, $vgpr2_lo16, 0, 0, implicit $exec
    GLOBAL_STORE_SHORT_t16 killed renamable $vgpr0_vgpr1, killed renamable $vgpr0_hi16, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1100", MIRString);
  ASSERT_TRUE(MBB);
  const GCNSubtarget *ST = &MBB->getParent()->getSubtarget<GCNSubtarget>();
  ASSERT_TRUE(ST->hasD16Writes32BitVgpr())
      << "gfx1100 should have D16Writes32BitVgpr";

  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr,
                             AMDGPU::SchedulingMode::NoExpert);

  auto It = MBB->begin();
  MachineInstr *Load = &*It++;     // writes $vgpr0_lo16
  MachineInstr *ValuOp = &*It++;   // VALU, accesses $vgpr0_hi16
  MachineInstr *StoreOp = &*It++;  // memory, accesses $vgpr0_hi16

  // With a pending load into $vgpr0_lo16, a VALU access to $vgpr0_hi16 widens to
  // the enclosing $vgpr0 (the D16 VALU op may clobber the whole register).
  RT.track(*Load);
  EXPECT_EQ(RT.getEffectiveDepReg(AMDGPU::VGPR0_HI16, *ValuOp), AMDGPU::VGPR0);
  // A memory access to $vgpr0_hi16 stays narrow even with the pending lo16 load:
  // a D16 memory op writes only its own half.
  EXPECT_EQ(RT.getEffectiveDepReg(AMDGPU::VGPR0_HI16, *StoreOp),
            MCRegister(AMDGPU::VGPR0_HI16));

  // With nothing pending on the other half, even a VALU access stays narrow.
  AMDGPU::ResourceTracker Empty(ST, /*AA=*/nullptr,
                                AMDGPU::SchedulingMode::NoExpert);
  EXPECT_EQ(Empty.getEffectiveDepReg(AMDGPU::VGPR0_HI16, *ValuOp),
            MCRegister(AMDGPU::VGPR0_HI16));
  // A full 32-bit VGPR is unchanged.
  EXPECT_EQ(Empty.getEffectiveDepReg(AMDGPU::VGPR0, *ValuOp), AMDGPU::VGPR0);
  // A 32-bit SGPR is unchanged (the widening only applies to VGPRs).
  EXPECT_EQ(Empty.getEffectiveDepReg(AMDGPU::SGPR0, *ValuOp), AMDGPU::SGPR0);
  // Special registers without a base class (e.g. EXEC) must be handled without
  // crashing and returned unchanged.
  EXPECT_EQ(Empty.getEffectiveDepReg(AMDGPU::EXEC, *ValuOp),
            MCRegister(AMDGPU::EXEC));
}

// End-to-end: on a hasD16Writes32BitVgpr target, writing one 16-bit half of a
// VGPR must wait for a pending load into the *other* half, because the D16 load
// may write the whole 32-bit register. getWaitFor must report a vmcnt dependency
// for $vgpr0_hi16 even though only $vgpr0_lo16 was loaded.
TEST_F(AMDGPUResourceTrackerTest, GetWaitFor_D16CrossHalf) {
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $vgpr2
    renamable $vgpr0_lo16 = GLOBAL_LOAD_SHORT_D16_t16 killed renamable $vgpr0_vgpr1, 0, 0, implicit $exec
    renamable $vgpr0_hi16 = V_SUB_NC_U16_t16_e64 0, 0, 0, $vgpr2_lo16, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1100", MIRString);
  ASSERT_TRUE(MBB);
  const GCNSubtarget *ST = &MBB->getParent()->getSubtarget<GCNSubtarget>();
  ASSERT_TRUE(ST->hasD16Writes32BitVgpr())
      << "gfx1100 should have D16Writes32BitVgpr";

  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr,
                             AMDGPU::SchedulingMode::NoExpert);

  auto It = MBB->begin();
  MachineInstr *Load = &*It++;
  MachineInstr *SubHi = &*It++;
  ASSERT_TRUE(Load->getOpcode() == AMDGPU::GLOBAL_LOAD_SHORT_D16_t16);
  RT.track(*Load);

  // The V_SUB writes $vgpr0_hi16 (a DEF) - a hazard against the pending load into
  // $vgpr0_lo16: a vmcnt wait is required because the d16 load may write the
  // whole $vgpr0.
  auto Waits = RT.getWaitFor(AMDGPU::VGPR0_HI16, *SubHi,
                             AMDGPU::ResourceTracker::RegAccessType::Def);
  bool HasVmcnt = false;
  for (const auto &W : Waits)
    if (W.Cntr == AMDGPU::VmCnt())
      HasVmcnt = true;
  EXPECT_TRUE(HasVmcnt)
      << "Writing one 16-bit half must wait for a load into the other half on "
         "hasD16Writes32BitVgpr targets";
}

TEST_F(AMDGPUResourceTrackerTest, Counter_Equality) {
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4
    $vgpr2 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    $vgpr3 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 4, 0, 0, implicit $exec
    $vgpr4 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 8, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);

  auto It = MBB->begin();
  MachineInstr *Load1 = &*It++;
  MachineInstr *Load2 = &*It++;
  MachineInstr *Load3 = &*It++;

  // Test 1: Two empty counters are equal
  {
    AMDGPU::Counter C1{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    AMDGPU::Counter C2{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    EXPECT_TRUE(C1 == C2);
    EXPECT_FALSE(C1 != C2);
  }

  // Test 2: Two counters with same instructions are equal
  {
    AMDGPU::Counter C1{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    AMDGPU::Counter C2{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    C1.insert(Load1);
    C1.insert(Load2);
    C2.insert(Load1);
    C2.insert(Load2);
    EXPECT_TRUE(C1 == C2);
  }

  // Test 3: Two counters with same instructions in different order are NOT equal
  // because the instruction indices differ, which affects wait values.
  {
    AMDGPU::Counter C1{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    AMDGPU::Counter C2{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    C1.insert(Load1);
    C1.insert(Load2);
    C2.insert(Load2);
    C2.insert(Load1);
    EXPECT_FALSE(C1 == C2);
  }

  // Test 4: Counters with different sizes are not equal
  {
    AMDGPU::Counter C1{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    AMDGPU::Counter C2{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    C1.insert(Load1);
    C2.insert(Load1);
    C2.insert(Load2);
    EXPECT_FALSE(C1 == C2);
    EXPECT_TRUE(C1 != C2);
  }

  // Test 5: Counters with different instructions are not equal
  {
    AMDGPU::Counter C1{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    AMDGPU::Counter C2{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    C1.insert(Load1);
    C1.insert(Load2);
    C2.insert(Load1);
    C2.insert(Load3);
    EXPECT_FALSE(C1 == C2);
  }
}

TEST_F(AMDGPUResourceTrackerTest, Counter_Clear) {
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4
    $vgpr2 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    $vgpr3 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 4, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);

  auto It = MBB->begin();
  MachineInstr *Load1 = &*It++;
  MachineInstr *Load2 = &*It++;

  AMDGPU::Counter C{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
  C.insert(Load1);
  C.insert(Load2);
  EXPECT_EQ(C.size(), 2u);
  EXPECT_FALSE(C.empty());

  C.clear();
  EXPECT_EQ(C.size(), 0u);
  EXPECT_TRUE(C.empty());
}

TEST_F(AMDGPUResourceTrackerTest, Counter_IncomingUnknown) {
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4
    $vgpr2 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    $vgpr3 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 4, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);

  auto It = MBB->begin();
  MachineInstr *Load1 = &*It++;
  MachineInstr *Load2 = &*It++;

  // setIncomingUnknown makes an otherwise-empty counter non-empty. It behaves
  // like a dummy entry at the oldest position, so it also counts toward size(),
  // keeping the invariant empty() == (size() == 0).
  {
    AMDGPU::Counter C{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    EXPECT_TRUE(C.empty());
    EXPECT_EQ(C.size(), 0u);
    C.setIncomingUnknown();
    EXPECT_FALSE(C.empty());
    EXPECT_EQ(C.size(), 1u);
  }

  // The incoming-unknown dummy adds one to the count of tracked instructions.
  {
    AMDGPU::Counter C{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    C.insert(Load1);
    C.insert(Load2);
    EXPECT_EQ(C.size(), 2u);
    C.setIncomingUnknown();
    EXPECT_EQ(C.size(), 3u);
    // A full drain clears both the tracked instructions and the unknown.
    C.applyWait(0);
    EXPECT_EQ(C.size(), 0u);
    EXPECT_TRUE(C.empty());
  }

  // A wait for zero (full drain) resolves the unknown incoming state.
  {
    AMDGPU::Counter C{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    C.setIncomingUnknown();
    C.applyWait(0);
    EXPECT_TRUE(C.empty());
  }

  // A partial wait does not drain the counter to empty, so the unknown state
  // (modeled as a dummy at the oldest position) survives.
  {
    AMDGPU::Counter C{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    C.setIncomingUnknown();
    C.insert(Load1);
    C.insert(Load2);
    C.applyWait(1); // leaves one tracked instruction pending
    EXPECT_FALSE(C.empty());
    // Draining fully now clears both the tracked instruction and the unknown.
    C.applyWait(0);
    EXPECT_TRUE(C.empty());
  }

  // merge() ORs the unknown state: it is set if either counter has it.
  {
    AMDGPU::Counter C1{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    AMDGPU::Counter C2{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    C2.setIncomingUnknown();
    C1.merge(C2);
    EXPECT_FALSE(C1.empty());
  }

  // operator== distinguishes counters that differ only in unknown state.
  {
    AMDGPU::Counter C1{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    AMDGPU::Counter C2{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    EXPECT_TRUE(C1 == C2);
    C1.setIncomingUnknown();
    EXPECT_FALSE(C1 == C2);
    C2.setIncomingUnknown();
    EXPECT_TRUE(C1 == C2);
  }

  // clear() resets the unknown state.
  {
    AMDGPU::Counter C{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    C.setIncomingUnknown();
    C.clear();
    EXPECT_TRUE(C.empty());
  }
}

TEST_F(AMDGPUResourceTrackerTest, CounterArray_Merge) {
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4
    $vgpr2 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    $vgpr3 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 4, 0, 0, implicit $exec
    $vgpr4 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 8, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);
  const GCNSubtarget &ST = MBB->getParent()->getSubtarget<GCNSubtarget>();

  auto It = MBB->begin();
  MachineInstr *Load1 = &*It++;
  MachineInstr *Load2 = &*It++;
  MachineInstr *Load3 = &*It++;

  AMDGPU::InstCounters ICounters;

  // Test 1: Merging with empty AllCounters is a no-op.
  {
    AMDGPU::AllCounters CA1(ST, AMDGPU::SchedulingMode::NoExpert, &ICounters);
    AMDGPU::AllCounters CA2(ST, AMDGPU::SchedulingMode::NoExpert, &ICounters);
    CA1[AMDGPU::LoadCnt()].insert(Load1);
    CA1.merge(CA2);
    EXPECT_EQ(CA1[AMDGPU::LoadCnt()].size(), 1u);
    EXPECT_EQ(CA1[AMDGPU::LoadCnt()].getWaitFor(*Load1), 0u);
  }

  // Test 2: Merging adds instructions from the other AllCounters.
  // CA1: [Load1], CA2: [Load2]
  // After merge: [{Load1,Load2}]
  {
    AMDGPU::AllCounters CA1(ST, AMDGPU::SchedulingMode::NoExpert, &ICounters);
    AMDGPU::AllCounters CA2(ST, AMDGPU::SchedulingMode::NoExpert, &ICounters);
    CA1[AMDGPU::LoadCnt()].insert(Load1);
    CA2[AMDGPU::LoadCnt()].insert(Load2);
    CA1.merge(CA2);
    EXPECT_EQ(CA1[AMDGPU::LoadCnt()].size(), 2u);
    // Both instructions are at the same index after merge.
    EXPECT_EQ(CA1[AMDGPU::LoadCnt()].getWaitFor(*Load1), 0u);
    EXPECT_EQ(CA1[AMDGPU::LoadCnt()].getWaitFor(*Load2), 0u);
  }

  // Test 3: Merging with overlapping instructions (AsyncCounter allows duplicates).
  // CA1: [Load1, Load2]
  //   Load1: wait = 1, Load2: wait = 0
  // CA2: [Load2, Load3]
  //   Load2: wait = 1, Load3: wait = 0
  // AsyncCounter (no-dedup) keeps both occurrences of Load2 at different slots.
  // size=4 (Load1, Load2@slot1, Load2@slot2, Load3). getWaitFor returns the
  // most recent (lowest wait) occurrence of Load2 = 0.
  {
    AMDGPU::AllCounters CA1(ST, AMDGPU::SchedulingMode::NoExpert, &ICounters);
    AMDGPU::AllCounters CA2(ST, AMDGPU::SchedulingMode::NoExpert, &ICounters);
    CA1[AMDGPU::LoadCnt()].insert(Load1);
    CA1[AMDGPU::LoadCnt()].insert(Load2);
    CA2[AMDGPU::LoadCnt()].insert(Load2);
    CA2[AMDGPU::LoadCnt()].insert(Load3);
    CA1.merge(CA2);
    EXPECT_EQ(CA1[AMDGPU::LoadCnt()].size(), 4u);
    EXPECT_EQ(CA1[AMDGPU::LoadCnt()].getWaitFor(*Load1), 1u);
    EXPECT_EQ(CA1[AMDGPU::LoadCnt()].getWaitFor(*Load2), 0u); // most recent occurrence
    EXPECT_EQ(CA1[AMDGPU::LoadCnt()].getWaitFor(*Load3), 0u);
  }

  // Test 4: Merging affects multiple counters independently.
  {
    AMDGPU::AllCounters CA1(ST, AMDGPU::SchedulingMode::NoExpert, &ICounters);
    AMDGPU::AllCounters CA2(ST, AMDGPU::SchedulingMode::NoExpert, &ICounters);
    CA1[AMDGPU::LoadCnt()].insert(Load1);
    CA2[AMDGPU::DsCnt()].insert(Load2);
    CA1.merge(CA2);
    EXPECT_EQ(CA1[AMDGPU::LoadCnt()].size(), 1u);
    EXPECT_EQ(CA1[AMDGPU::DsCnt()].size(), 1u);
    EXPECT_EQ(CA1[AMDGPU::LoadCnt()].getWaitFor(*Load1), 0u);
    EXPECT_EQ(CA1[AMDGPU::DsCnt()].getWaitFor(*Load2), 0u);
  }
}

TEST_F(AMDGPUResourceTrackerTest, CounterArray_Clear) {
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4
    $vgpr2 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    $vgpr3 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 4, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);
  const GCNSubtarget &ST = MBB->getParent()->getSubtarget<GCNSubtarget>();

  auto It = MBB->begin();
  MachineInstr *Load1 = &*It++;
  MachineInstr *Load2 = &*It++;

  AMDGPU::InstCounters ICounters;
  AMDGPU::AllCounters CA(ST, AMDGPU::SchedulingMode::NoExpert, &ICounters);

  CA[AMDGPU::LoadCnt()].insert(Load1);
  CA[AMDGPU::DsCnt()].insert(Load2);
  EXPECT_EQ(CA[AMDGPU::LoadCnt()].size(), 1u);
  EXPECT_EQ(CA[AMDGPU::DsCnt()].size(), 1u);

  CA.clear();
  EXPECT_EQ(CA[AMDGPU::LoadCnt()].size(), 0u);
  EXPECT_EQ(CA[AMDGPU::DsCnt()].size(), 0u);
}

TEST_F(AMDGPUResourceTrackerTest, ResourceTracker_Merge) {
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4
    $vgpr2 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    $vgpr3 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 4, 0, 0, implicit $exec
    $vgpr4 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 8, 0, 0, implicit $exec
    BUFFER_STORE_DWORD_OFFSET $vgpr2, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);
  const GCNSubtarget &ST = MBB->getParent()->getSubtarget<GCNSubtarget>();

  auto It = MBB->begin();
  MachineInstr *Load1 = &*It++;
  MachineInstr *Load2 = &*It++;
  MachineInstr *Load3 = &*It++;
  MachineInstr *Store = &*It++;

  // Test: Merging combines counters, register mappings, and pending stores.
  {
    AMDGPU::ResourceTracker RT1(&ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);
    AMDGPU::ResourceTracker RT2(&ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

    RT1.track(*Load1);
    RT1.track(*Store);
    RT2.track(*Load2);
    RT2.track(*Load3);

    EXPECT_EQ(RT1.getCounter(AMDGPU::LoadCnt()).size(), 1u);
    EXPECT_EQ(RT1.getCounter(AMDGPU::StoreCnt()).size(), 1u);
    EXPECT_EQ(RT2.getCounter(AMDGPU::LoadCnt()).size(), 2u);

    RT1.merge(RT2);

    // After merge (always max semantics, 2 slots):
    // RT1: [Load1] (1 instr), RT2: [Load2, Load3] (2 instrs).
    // OtherTopIdx(2) > ThisTopIdx(1): Load1 shifts by delta=1 to InternalIdx=1,
    // preserving its wait=0 (UserIdx=1, wait=2-1-1=0). Then RT2's entries are
    // added: Load2 at wait=1 (slot 0), Load3 at wait=0 (slot 1, same as Load1).
    // Result: Load2 at wait=1, {Load1,Load3} at wait=0. All 3 instrs present.
    EXPECT_EQ(RT1.getCounter(AMDGPU::LoadCnt()).size(), 3u);
    EXPECT_EQ(RT1.getCounter(AMDGPU::LoadCnt()).getWaitFor(*Load1), 0u);
    EXPECT_EQ(RT1.getCounter(AMDGPU::LoadCnt()).getWaitFor(*Load2), 1u);
    EXPECT_EQ(RT1.getCounter(AMDGPU::LoadCnt()).getWaitFor(*Load3), 0u);
    // StoreCnt should still have the store from RT1.
    EXPECT_EQ(RT1.getCounter(AMDGPU::StoreCnt()).size(), 1u);
    EXPECT_EQ(RT1.getCounter(AMDGPU::StoreCnt()).getWaitFor(*Store), 0u);
  }
}

// Test that getInstrsFor() returns multiple writers from different blocks after
// merge.
TEST_F(AMDGPUResourceTrackerTest, ResourceTracker_GetInstrsFor_AfterMerge) {
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4
    $vgpr2 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    S_ENDPGM 0
  bb.1:
    liveins: $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4
    $vgpr2 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 4, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB0 = parseMIRGetBB("gfx1200", MIRString, 0);
  ASSERT_TRUE(MBB0);
  MachineFunction *MF = MBB0->getParent();
  MachineBasicBlock &MBB1 = getBlockByNumber(MF, 1);
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();

  MachineInstr *Load1 = &*MBB0->begin();
  MachineInstr *Load2 = &*MBB1.begin();

  AMDGPU::ResourceTracker RT1(&ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);
  AMDGPU::ResourceTracker RT2(&ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

  RT1.track(*Load1);
  RT2.track(*Load2);

  // Before merge, each tracker has one accessor for vgpr2.
  MCRegUnit RU = *TRI->regunits(AMDGPU::VGPR2).begin();
  EXPECT_EQ(RT1.getInstrsFor(RU).size(), 1u);
  EXPECT_EQ(RT1.getInstrsFor(RU)[0].MI, Load1);
  EXPECT_EQ(RT2.getInstrsFor(RU).size(), 1u);
  EXPECT_EQ(RT2.getInstrsFor(RU)[0].MI, Load2);

  // After merge, RT1 should have both accessors for vgpr2.
  RT1.merge(RT2);
  auto Accessors = RT1.getInstrsFor(RU);
  EXPECT_EQ(Accessors.size(), 2u);
  // Both Load1 and Load2 should be in the accessors list.
  SmallVector<MachineInstr *, 2> MIs;
  for (const auto &Info : Accessors)
    MIs.push_back(Info.MI);
  EXPECT_TRUE(llvm::is_contained(MIs, Load1));
  EXPECT_TRUE(llvm::is_contained(MIs, Load2));
}

TEST_F(AMDGPUResourceTrackerTest, ResourceTracker_Clear) {
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4
    $vgpr2 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    $vgpr3 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 4, 0, 0, implicit $exec
    BUFFER_STORE_DWORD_OFFSET $vgpr2, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);
  const GCNSubtarget &ST = MBB->getParent()->getSubtarget<GCNSubtarget>();

  auto It = MBB->begin();
  MachineInstr *Load1 = &*It++;
  MachineInstr *Load2 = &*It++;
  MachineInstr *Store = &*It++;

  AMDGPU::ResourceTracker RT(&ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);
  RT.track(*Load1);
  RT.track(*Load2);
  RT.track(*Store);

  EXPECT_EQ(RT.getCounter(AMDGPU::LoadCnt()).size(), 2u);
  EXPECT_EQ(RT.getCounter(AMDGPU::StoreCnt()).size(), 1u);

  RT.clear();

  EXPECT_EQ(RT.getCounter(AMDGPU::LoadCnt()).size(), 0u);
  EXPECT_EQ(RT.getCounter(AMDGPU::StoreCnt()).size(), 0u);
}

// Test that GLOBAL_INV increments LoadCnt and affects wait values.
// GLOBAL_INV doesn't write to VGPRs but does increment the LoadCnt counter.
TEST_F(AMDGPUResourceTrackerTest, GetCountersForInstr_GLOBAL_INV) {
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, $vgpr0_vgpr1
    $vgpr2 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    GLOBAL_INV 16, implicit $exec
    $vgpr3 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);
  const GCNSubtarget &ST = MBB->getParent()->getSubtarget<GCNSubtarget>();

  auto It = MBB->begin();
  MachineInstr *Load1 = &*It++;
  MachineInstr *GlobalInv = &*It++;
  MachineInstr *Load2 = &*It++;

  AMDGPU::ResourceTracker RT(&ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);
  RT.track(*Load1);
  RT.track(*GlobalInv);
  RT.track(*Load2);

  // After tracking Load1, GLOBAL_INV, and Load2, the LoadCnt counter should
  // have 3 pending instructions (Load1, GLOBAL_INV, Load2).
  // GLOBAL_INV increments the counter even though it doesn't write to VGPRs.
  EXPECT_EQ(RT.getCounter(AMDGPU::LoadCnt()).size(), 3u);

  // When waiting for Load1, we need to wait for count 2 (Load1 is oldest).
  // Load1 is at position 0, size=3, so wait = 3-1-0 = 2
  auto Wait1 = RT.getCounter(AMDGPU::LoadCnt()).getWaitFor(*Load1);
  ASSERT_TRUE(Wait1.has_value());
  EXPECT_EQ(*Wait1, 2u);

  // GLOBAL_INV is at position 1, so wait = 3-1-1 = 1
  auto WaitInv = RT.getCounter(AMDGPU::LoadCnt()).getWaitFor(*GlobalInv);
  ASSERT_TRUE(WaitInv.has_value());
  EXPECT_EQ(*WaitInv, 1u);

  // Load2 is at position 2 (newest), so wait = 3-1-2 = 0
  auto Wait2 = RT.getCounter(AMDGPU::LoadCnt()).getWaitFor(*Load2);
  ASSERT_TRUE(Wait2.has_value());
  EXPECT_EQ(*Wait2, 0u);
}

// Test ResourceTracker::getWaitForMemory() for cross-unit LDS dependencies.
// VMEM-to-LDS instructions (BUFFER_LOAD_*_LDS, GLOBAL_LOAD_LDS_*) write to LDS
// via vmcnt, while DS instructions access LDS via lgkmcnt/dscnt. These are
// different hardware units that don't automatically synchronize, so we need
// vmcnt waits before DS operations when VMEM-to-LDS operations are pending.
TEST_F(AMDGPUTestBase, ResourceTracker_GetWaitForMemory) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx942", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0, $vgpr1, $vgpr2, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4
    $m0 = S_MOV_B32 0
    ; VMEM-to-LDS: loads from global memory to LDS
    BUFFER_LOAD_DWORD_LDS_IDXEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, 0, 0, 0, 0, implicit $exec, implicit $m0 :: (load (s32) from `ptr addrspace(1) poison`), (store (s32) into `ptr addrspace(3) poison`)
    ; DS operation that accesses LDS - needs wait for VMEM-to-LDS
    $vgpr1 = DS_READ_B32_gfx9 $vgpr2, 0, 0, implicit $m0, implicit $exec :: (load (s32) from `ptr addrspace(3) poison`)
    ; Regular VMEM load - doesn't access LDS
    $vgpr3 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr, AMDGPU::SchedulingMode::NoExpert);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 5> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 4u);

  MachineInstr *MovM0 = Instrs[0];
  MachineInstr *BufferLoadLds = Instrs[1];
  MachineInstr *DsRead = Instrs[2];
  MachineInstr *GlobalLoad = Instrs[3];

  // Before tracking VMEM-to-LDS, no memory waits needed
  EXPECT_TRUE(RT.getWaitForMemory(*DsRead).empty());

  // Track S_MOV_B32 (doesn't affect counters)
  RT.track(*MovM0);
  EXPECT_TRUE(RT.getWaitForMemory(*DsRead).empty());

  // Track BUFFER_LOAD_LDS - this is a VMEM-to-LDS operation
  RT.track(*BufferLoadLds);

  // Now DS operation needs to wait for vmcnt(0) due to cross-unit LDS dependency.
  // Since AA is nullptr, we conservatively assume aliasing.
  auto MemWaits = RT.getWaitForMemory(*DsRead);
  ASSERT_EQ(MemWaits.size(), 1u);
  EXPECT_EQ(MemWaits.begin()->Cntr, AMDGPU::VmCnt());
  EXPECT_EQ(MemWaits.begin()->Wait, 0u);
  EXPECT_EQ(MemWaits.begin()->MI, BufferLoadLds);

  // Regular VMEM load (not to LDS) doesn't need memory wait
  EXPECT_TRUE(RT.getWaitForMemory(*GlobalLoad).empty());
}


// Test Counter::getNthFromEnd()
TEST_F(AMDGPUResourceTrackerTest, Counter_GetNthFromEnd) {
  StringRef MIRString = R"MIR(
---
name: test
body: |
  bb.0:
    $vgpr0 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr1 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec
    $vgpr2 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 8, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);
  MachineFunction *MF = MBB->getParent();

  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  AMDGPU::ResourceTracker RT(&ST, /*AA=*/nullptr,
                             AMDGPU::SchedulingMode::NoExpert);

  // Collect the load instructions
  SmallVector<MachineInstr *, 3> Loads;
  for (MachineInstr &MI : *MBB) {
    if (MI.getOpcode() == AMDGPU::GLOBAL_LOAD_DWORD)
      Loads.push_back(&MI);
  }
  ASSERT_EQ(Loads.size(), 3u);

  const AMDGPU::Counter &LoadCntr = RT.getCounter(AMDGPU::LoadCnt());

  // Initially empty
  EXPECT_THAT(LoadCntr.getNthFromEnd(0), IsEmpty());
  EXPECT_THAT(LoadCntr.getNthFromEnd(1), IsEmpty());

  // Track first load
  RT.track(*Loads[0]);
  EXPECT_THAT(LoadCntr.getNthFromEnd(0), UnorderedElementsAre(Loads[0]));
  EXPECT_THAT(LoadCntr.getNthFromEnd(1), IsEmpty());  // Out of bounds

  // Track second load
  RT.track(*Loads[1]);
  EXPECT_THAT(LoadCntr.getNthFromEnd(0), UnorderedElementsAre(Loads[1]));  // Most recent
  EXPECT_THAT(LoadCntr.getNthFromEnd(1), UnorderedElementsAre(Loads[0]));  // Second most recent
  EXPECT_THAT(LoadCntr.getNthFromEnd(2), IsEmpty());   // Out of bounds

  // Track third load
  RT.track(*Loads[2]);
  EXPECT_THAT(LoadCntr.getNthFromEnd(0), UnorderedElementsAre(Loads[2]));  // Most recent
  EXPECT_THAT(LoadCntr.getNthFromEnd(1), UnorderedElementsAre(Loads[1]));  // Second most recent
  EXPECT_THAT(LoadCntr.getNthFromEnd(2), UnorderedElementsAre(Loads[0]));  // Third most recent
  EXPECT_THAT(LoadCntr.getNthFromEnd(3), IsEmpty());   // Out of bounds

  // Test getNthFromEnd() when multiple instructions exist at the same index.
  // This happens after merging two counters where different instructions
  // occupy the same slot (e.g., from two CFG predecessors).
  // C1: [Load0], C2: [Load1] -> after merge: [{Load0, Load1}]
  {
    AMDGPU::Counter C1{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    AMDGPU::Counter C2{AMDGPU::LoadCnt{}, /*MaxSize=*/0, /*DropOnOverflow=*/false};
    C1.insert(Loads[0]);
    C2.insert(Loads[1]);
    C1.merge(C2);
    EXPECT_EQ(C1.size(), 2u);
    // Both Load0 and Load1 are at the same index after merge.
    EXPECT_THAT(C1.getNthFromEnd(0), UnorderedElementsAre(Loads[0], Loads[1]));
    EXPECT_THAT(C1.getNthFromEnd(1), IsEmpty());  // Out of bounds
  }
}

// Test Counter::getNthFromEnd's Saturate option.
// When the counter is at MaxSize capacity, Saturate=true clamps an out-of-range
// request to the oldest position; when not full it still returns empty.
TEST_F(AMDGPUResourceTrackerTest, Counter_GetNthFromEndSaturate) {
  StringRef MIRString = R"MIR(
---
name: test
body: |
  bb.0:
    $vgpr0 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr1 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec
    $vgpr2 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 8, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);

  SmallVector<MachineInstr *, 3> Marks;
  for (MachineInstr &MI : *MBB)
    if (MI.getOpcode() == AMDGPU::GLOBAL_LOAD_DWORD)
      Marks.push_back(&MI);
  ASSERT_EQ(Marks.size(), 3u);

  // Not full: Saturate=true still returns empty for an out-of-range request
  // (no entries were dropped, so the pass emits no wait).
  {
    AMDGPU::Counter C{AMDGPU::LoadCnt(), /*MaxSize=*/4,
                      /*DropOnOverflow=*/false};
    C.insert(Marks[0]);
    C.insert(Marks[1]); // newest
    EXPECT_THAT(C.getNthFromEnd(0, /*Saturate=*/true),
                UnorderedElementsAre(Marks[1]));
    EXPECT_THAT(C.getNthFromEnd(1, /*Saturate=*/true),
                UnorderedElementsAre(Marks[0]));
    // size (2) < MaxSize (4): not full, so out-of-range yields empty even with
    // Saturate.
    EXPECT_THAT(C.getNthFromEnd(2, /*Saturate=*/true), IsEmpty());
    EXPECT_THAT(C.getNthFromEnd(5, /*Saturate=*/true), IsEmpty());
  }
}

TEST_F(AMDGPUResourceTrackerTest, Counter_InsertAtSameIndex) {
  StringRef MIRString = R"MIR(
---
name: test
body: |
  bb.0:
    $vgpr0 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr1 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec
    $vgpr2 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 8, 0, implicit $exec
    $vgpr3 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 12, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);

  SmallVector<MachineInstr *, 4> Loads;
  for (MachineInstr &MI : *MBB)
    if (MI.getOpcode() == AMDGPU::GLOBAL_LOAD_DWORD)
      Loads.push_back(&MI);
  ASSERT_EQ(Loads.size(), 4u);
  // insert() with a single new instruction.
  {
    AMDGPU::Counter C{AMDGPU::LoadCnt(), 0, false};
    C.insert(Loads[0]);
    C.insert(Loads[1]);
    EXPECT_EQ(C.size(), 2u);
    EXPECT_THAT(C.getNthFromEnd(0), UnorderedElementsAre(Loads[1]));
    EXPECT_THAT(C.getNthFromEnd(1), UnorderedElementsAre(Loads[0]));
  }

  // insert() with two instructions places them at the same index.
  // Both are returned by getNthFromEnd(0).
  {
    AMDGPU::Counter C{AMDGPU::LoadCnt(), 0, false};
    C.insert({Loads[0], Loads[1]});
    EXPECT_EQ(C.size(), 2u);
    EXPECT_THAT(C.getNthFromEnd(0),
                UnorderedElementsAre(Loads[0], Loads[1]));
    EXPECT_THAT(C.getNthFromEnd(1), IsEmpty());
  }

  // Two successive insert() calls produce two distinct positions.
  {
    AMDGPU::Counter C{AMDGPU::LoadCnt(), 0, false};
    C.insert({Loads[0], Loads[1]});  // position 1 (older)
    C.insert({Loads[2], Loads[3]});  // position 0 (newest)
    EXPECT_EQ(C.size(), 4u);
    EXPECT_THAT(C.getNthFromEnd(0),
                UnorderedElementsAre(Loads[2], Loads[3]));
    EXPECT_THAT(C.getNthFromEnd(1),
                UnorderedElementsAre(Loads[0], Loads[1]));
    EXPECT_THAT(C.getNthFromEnd(2), IsEmpty());
  }

  // Mixing insert() and insert() works correctly.
  // insert(L0), insert({L1, L2}):
  //   position 0: {L1, L2} (newest), position 1: {L0}
  {
    AMDGPU::Counter C{AMDGPU::LoadCnt(), 0, false};
    C.insert(Loads[0]);
    C.insert({Loads[1], Loads[2]});
    EXPECT_EQ(C.size(), 3u);
    EXPECT_THAT(C.getNthFromEnd(0),
                UnorderedElementsAre(Loads[1], Loads[2]));
    EXPECT_THAT(C.getNthFromEnd(1), UnorderedElementsAre(Loads[0]));
  }
}

TEST_F(AMDGPUResourceTrackerTest, InstrBuffer) {
  StringRef MIRString = R"MIR(
---
name: test
body: |
  bb.0:
    $vgpr0 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr1 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec
    $vgpr2 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 8, 0, implicit $exec
    $vgpr3 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 12, 0, implicit $exec
    $vgpr4 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 16, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);

  SmallVector<MachineInstr *, 5> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (MI.getOpcode() == AMDGPU::GLOBAL_LOAD_DWORD)
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 5u);

  // Test pushBack and getTopIndex
  AMDGPU::InstrBuffer Buf;
  EXPECT_EQ(Buf.getTopIndex(), 0u);
  Buf.pushBack(Instrs[0]);
  EXPECT_EQ(Buf.getTopIndex(), 1u);
  Buf.pushBack(Instrs[1]);
  Buf.pushBack(Instrs[2]);
  EXPECT_EQ(Buf.getTopIndex(), 3u);

  // Test getIndex (indices are 0-based relative to bottom)
  EXPECT_EQ(Buf.getIndex(Instrs[0]), 0u);
  EXPECT_EQ(Buf.getIndex(Instrs[1]), 1u);
  EXPECT_EQ(Buf.getIndex(Instrs[2]), 2u);

  // Test popFront - indices shift down
  Buf.popFront(1);
  EXPECT_EQ(Buf.getTopIndex(), 2u);
  EXPECT_EQ(Buf.getIndex(Instrs[1]), 0u);
  EXPECT_EQ(Buf.getIndex(Instrs[2]), 1u);

  // Test getIndex after popFront and pushBack
  Buf.pushBack(Instrs[0]);
  EXPECT_EQ(Buf.getTopIndex(), 3u);
  EXPECT_EQ(Buf.getIndex(Instrs[1]), 0u);
  EXPECT_EQ(Buf.getIndex(Instrs[2]), 1u);
  EXPECT_EQ(Buf.getIndex(Instrs[0]), 2u);

  // Test equality: two empty buffers
  AMDGPU::InstrBuffer Buf1, Buf2;
  EXPECT_TRUE(Buf1 == Buf2);
  EXPECT_FALSE(Buf1 != Buf2);

  // Test equality: same instructions same order
  Buf1.pushBack(Instrs[0]);
  Buf1.pushBack(Instrs[1]);
  Buf2.pushBack(Instrs[0]);
  Buf2.pushBack(Instrs[1]);
  EXPECT_TRUE(Buf1 == Buf2);

  // Test inequality: different sizes
  Buf2.pushBack(Instrs[2]);
  EXPECT_FALSE(Buf1 == Buf2);
  EXPECT_TRUE(Buf1 != Buf2);

  // Test inequality: same size, different instructions
  AMDGPU::InstrBuffer Buf3, Buf4;
  Buf3.pushBack(Instrs[0]);
  Buf4.pushBack(Instrs[1]);
  EXPECT_FALSE(Buf3 == Buf4);

  // Test equality after same push/pop sequence
  AMDGPU::InstrBuffer Buf5, Buf6;
  Buf5.pushBack(Instrs[0]);
  Buf5.pushBack(Instrs[1]);
  Buf5.popFront(1);
  Buf6.pushBack(Instrs[0]);
  Buf6.pushBack(Instrs[1]);
  Buf6.popFront(1);
  EXPECT_TRUE(Buf5 == Buf6);

  // Test equality of buffers created via different paths.
  // This tests dataflow convergence: a buffer created via duplicate pushBack
  // should equal one created via direct pushBack in the final order.
  // BufA: push I0, push I1, push I0 (dup) -> [I1, I0]
  // BufB: push I1, push I0 -> [I1, I0]
  AMDGPU::InstrBuffer BufA, BufB;
  BufA.pushBack(Instrs[0]);
  BufA.pushBack(Instrs[1]);
  BufA.pushBack(Instrs[0]);  // Duplicate, moves I0 to end
  BufB.pushBack(Instrs[1]);
  BufB.pushBack(Instrs[0]);
  EXPECT_TRUE(BufA == BufB);

  // Test pushBack with duplicate at last position (no-op).
  // [I0, I1] -> pushBack(I1) -> [I0, I1]
  AMDGPU::InstrBuffer BufDupLast;
  BufDupLast.pushBack(Instrs[0]);
  BufDupLast.pushBack(Instrs[1]);
  EXPECT_EQ(BufDupLast.getTopIndex(), 2u);
  EXPECT_EQ(BufDupLast.getIndex(Instrs[0]), 0u);
  EXPECT_EQ(BufDupLast.getIndex(Instrs[1]), 1u);
  BufDupLast.pushBack(Instrs[1]);
  EXPECT_EQ(BufDupLast.getTopIndex(), 2u);
  EXPECT_EQ(BufDupLast.getIndex(Instrs[0]), 0u);
  EXPECT_EQ(BufDupLast.getIndex(Instrs[1]), 1u);

  // Test pushBack with duplicate at first position (swap).
  // [I0, I1] -> pushBack(I0) -> [I1, I0]
  AMDGPU::InstrBuffer BufDupFirst;
  BufDupFirst.pushBack(Instrs[0]);
  BufDupFirst.pushBack(Instrs[1]);
  EXPECT_EQ(BufDupFirst.getTopIndex(), 2u);
  BufDupFirst.pushBack(Instrs[0]);
  EXPECT_EQ(BufDupFirst.getTopIndex(), 2u);
  EXPECT_EQ(BufDupFirst.getIndex(Instrs[1]), 0u);
  EXPECT_EQ(BufDupFirst.getIndex(Instrs[0]), 1u);
  EXPECT_EQ(BufDupFirst.numInstrs(), 2u);

  // Test pushBack with duplicate in middle position (swap).
  // [I0, I1, I2] -> pushBack(I1) -> [I0, I2, I1]
  AMDGPU::InstrBuffer BufDupMid;
  BufDupMid.pushBack(Instrs[0]);
  BufDupMid.pushBack(Instrs[1]);
  BufDupMid.pushBack(Instrs[2]);
  EXPECT_EQ(BufDupMid.getTopIndex(), 3u);
  BufDupMid.pushBack(Instrs[1]);
  EXPECT_EQ(BufDupMid.getTopIndex(), 3u);
  EXPECT_EQ(BufDupMid.getIndex(Instrs[0]), 0u);
  EXPECT_EQ(BufDupMid.getIndex(Instrs[2]), 1u);
  EXPECT_EQ(BufDupMid.getIndex(Instrs[1]), 2u);
  EXPECT_EQ(BufDupMid.numInstrs(), 3u);

  // Test pushBack with duplicate at first position in 3-element buffer (swap).
  // [I0, I1, I2] -> pushBack(I0) -> [I2, I1, I0]
  AMDGPU::InstrBuffer BufDupFirst3;
  BufDupFirst3.pushBack(Instrs[0]);
  BufDupFirst3.pushBack(Instrs[1]);
  BufDupFirst3.pushBack(Instrs[2]);
  EXPECT_EQ(BufDupFirst3.getTopIndex(), 3u);
  BufDupFirst3.pushBack(Instrs[0]);
  EXPECT_EQ(BufDupFirst3.getTopIndex(), 3u);
  EXPECT_EQ(BufDupFirst3.getIndex(Instrs[2]), 0u);
  EXPECT_EQ(BufDupFirst3.getIndex(Instrs[1]), 1u);
  EXPECT_EQ(BufDupFirst3.getIndex(Instrs[0]), 2u);
  EXPECT_EQ(BufDupFirst3.numInstrs(), 3u);

  // Test empty()
  AMDGPU::InstrBuffer BufEmpty;
  EXPECT_TRUE(BufEmpty.empty());
  BufEmpty.pushBack(Instrs[0]);
  EXPECT_FALSE(BufEmpty.empty());
  BufEmpty.popFront(1);
  EXPECT_TRUE(BufEmpty.empty());

  // Test numInstrs()
  AMDGPU::InstrBuffer BufNum;
  EXPECT_EQ(BufNum.numInstrs(), 0u);
  BufNum.pushBack(Instrs[0]);
  EXPECT_EQ(BufNum.numInstrs(), 1u);
  BufNum.pushBack(Instrs[1]);
  EXPECT_EQ(BufNum.numInstrs(), 2u);
  BufNum.popFront(1);
  EXPECT_EQ(BufNum.numInstrs(), 1u);
  // After merge, numInstrs counts all unique instructions even at same index.
  // BufNum: [I1], BufNum2: [I2] -> after merge: [{I1, I2}]
  AMDGPU::InstrBuffer BufNum2;
  BufNum2.pushBack(Instrs[2]);  // BufNum2: [I2]
  BufNum.merge(BufNum2);
  EXPECT_EQ(BufNum.numInstrs(), 2u);
  EXPECT_EQ(BufNum.getTopIndex(), 1u);  // Only one index

  // Test contains()
  AMDGPU::InstrBuffer BufContains;
  EXPECT_FALSE(BufContains.contains(Instrs[0]));
  BufContains.pushBack(Instrs[0]);
  EXPECT_TRUE(BufContains.contains(Instrs[0]));
  EXPECT_FALSE(BufContains.contains(Instrs[1]));
  BufContains.popFront(1);
  EXPECT_FALSE(BufContains.contains(Instrs[0]));

  // Test back()
  AMDGPU::InstrBuffer BufBack;
  EXPECT_THAT(BufBack.back(), IsEmpty());
  BufBack.pushBack(Instrs[0]);
  EXPECT_THAT(BufBack.back(), UnorderedElementsAre(Instrs[0]));
  BufBack.pushBack(Instrs[1]);
  EXPECT_THAT(BufBack.back(), UnorderedElementsAre(Instrs[1]));
  BufBack.pushBack(Instrs[2]);
  EXPECT_THAT(BufBack.back(), UnorderedElementsAre(Instrs[2]));

  // Test getNthFromEnd()
  AMDGPU::InstrBuffer BufNth;
  BufNth.pushBack(Instrs[0]);
  BufNth.pushBack(Instrs[1]);
  BufNth.pushBack(Instrs[2]);
  EXPECT_THAT(BufNth.getNthFromEnd(0), UnorderedElementsAre(Instrs[2]));
  EXPECT_THAT(BufNth.getNthFromEnd(1), UnorderedElementsAre(Instrs[1]));
  EXPECT_THAT(BufNth.getNthFromEnd(2), UnorderedElementsAre(Instrs[0]));
  EXPECT_THAT(BufNth.getNthFromEnd(3), IsEmpty());

  // Test clear()
  AMDGPU::InstrBuffer BufClear;
  BufClear.pushBack(Instrs[0]);
  BufClear.pushBack(Instrs[1]);
  EXPECT_FALSE(BufClear.empty());
  BufClear.clear();
  EXPECT_TRUE(BufClear.empty());
  EXPECT_EQ(BufClear.getTopIndex(), 0u);
  EXPECT_FALSE(BufClear.contains(Instrs[0]));

  // Test iterator
  AMDGPU::InstrBuffer BufIter;
  BufIter.pushBack(Instrs[0]);
  BufIter.pushBack(Instrs[1]);
  BufIter.pushBack(Instrs[2]);
  SmallVector<MachineInstr *, 3> Collected;
  for (const auto &Set : BufIter)
    for (AMDGPU::TrackedInstr TI : Set)
      Collected.push_back(TI.getMI());
  EXPECT_THAT(Collected, ElementsAre(Instrs[0], Instrs[1], Instrs[2]));

  // Test iterator after popFront
  BufIter.popFront(1);
  Collected.clear();
  for (const auto &Set : BufIter)
    for (AMDGPU::TrackedInstr TI : Set)
      Collected.push_back(TI.getMI());
  EXPECT_THAT(Collected, ElementsAre(Instrs[1], Instrs[2]));

  // Test pushBack with duplicate at front (no gap created).
  // pushBack(I0): [I0]
  // pushBack(I1): [I0, I1]
  // pushBack(I0): I0 removed from idx 0, idx 0 becomes empty at front,
  //               BottomIdxInternal advances, I0 added at new top.
  //               Result: [I1, I0]
  AMDGPU::InstrBuffer BufPopGap;
  BufPopGap.pushBack(Instrs[0]);
  BufPopGap.pushBack(Instrs[1]);
  BufPopGap.pushBack(Instrs[0]);
  EXPECT_EQ(BufPopGap.getTopIndex(), 2u);
  EXPECT_EQ(BufPopGap.numInstrs(), 2u);
  EXPECT_EQ(BufPopGap.getIndex(Instrs[1]), 0u);
  EXPECT_EQ(BufPopGap.getIndex(Instrs[0]), 1u);
  BufPopGap.popFront(1);  // Pops I1
  EXPECT_EQ(BufPopGap.getTopIndex(), 1u);
  EXPECT_EQ(BufPopGap.numInstrs(), 1u);
  EXPECT_TRUE(BufPopGap.contains(Instrs[0]));
  EXPECT_FALSE(BufPopGap.contains(Instrs[1]));

  // Test merge() with disjoint buffers at same index
  // BufMerge1: [I0], BufMerge2: [I1] -> [{I0,I1}]
  AMDGPU::InstrBuffer BufMerge1, BufMerge2;
  BufMerge1.pushBack(Instrs[0]);
  BufMerge2.pushBack(Instrs[1]);
  BufMerge1.merge(BufMerge2);
  EXPECT_EQ(BufMerge1.getTopIndex(), 1u);
  EXPECT_THAT(BufMerge1.getNthFromEnd(0),
              UnorderedElementsAre(Instrs[0], Instrs[1]));

  // Test merge() with overlapping instructions (keep lowest wait = most recent).
  // BufMerge3: [I0, I1] (TopIndex=2)
  //   I0: wait = 1 (older)
  //   I1: wait = 0 (newer)
  // BufMerge4: [I1, I2] (TopIndex=2)
  //   I1: wait = 1 (older)
  //   I2: wait = 0 (newer)
  // I1 exists in both: keep position with lowest wait (BufMerge3's wait=0 wins).
  // I2's wait=0 also maps to newest slot → {I1,I2} at slot0, I0 at slot1.
  // Result: [I0, {I1,I2}]
  AMDGPU::InstrBuffer BufMerge3, BufMerge4;
  BufMerge3.pushBack(Instrs[0]);
  BufMerge3.pushBack(Instrs[1]);
  BufMerge4.pushBack(Instrs[1]);
  BufMerge4.pushBack(Instrs[2]);
  BufMerge3.merge(BufMerge4);
  EXPECT_EQ(BufMerge3.getTopIndex(), 2u);
  EXPECT_THAT(BufMerge3.getNthFromEnd(0), UnorderedElementsAre(Instrs[1], Instrs[2]));
  EXPECT_THAT(BufMerge3.getNthFromEnd(1),
              UnorderedElementsAre(Instrs[0]));

  // Test merge() with empty buffer
  // BufMerge5: [I0], BufMergeEmpty: [] -> [I0]
  AMDGPU::InstrBuffer BufMerge5, BufMergeEmpty;
  BufMerge5.pushBack(Instrs[0]);
  BufMerge5.merge(BufMergeEmpty);
  EXPECT_EQ(BufMerge5.getTopIndex(), 1u);
  EXPECT_THAT(BufMerge5.getNthFromEnd(0), UnorderedElementsAre(Instrs[0]));

  // Test merge() into empty buffer
  // BufMerge6: [], BufMerge7: [I0, I1] -> [I0, I1]
  AMDGPU::InstrBuffer BufMerge6, BufMerge7;
  BufMerge7.pushBack(Instrs[0]);
  BufMerge7.pushBack(Instrs[1]);
  BufMerge6.merge(BufMerge7);
  EXPECT_EQ(BufMerge6.getTopIndex(), 2u);
  EXPECT_THAT(BufMerge6.getNthFromEnd(0), UnorderedElementsAre(Instrs[1]));
  EXPECT_THAT(BufMerge6.getNthFromEnd(1), UnorderedElementsAre(Instrs[0]));

  // Test merge() preserves wait values when Other has larger index.
  // BufMerge8: [I0] (TopIndex=1), I0 wait=0
  // BufMerge9: [I1, I2] (TopIndex=2), I1 wait=1, I2 wait=0
  // After merge, I0 shifts to preserve wait=0, I1/I2 map to same waits:
  //   I0 (wait=0) -> pos 1
  //   I1 (wait=1) -> pos 0
  //   I2 (wait=0) -> pos 1
  // Result: [I1, {I0,I2}]
  AMDGPU::InstrBuffer BufMerge8, BufMerge9;
  BufMerge8.pushBack(Instrs[0]);
  BufMerge9.pushBack(Instrs[1]);
  BufMerge9.pushBack(Instrs[2]);
  BufMerge8.merge(BufMerge9);
  EXPECT_EQ(BufMerge8.getTopIndex(), 2u);
  EXPECT_THAT(BufMerge8.getNthFromEnd(0),
              UnorderedElementsAre(Instrs[0], Instrs[2]));
  EXPECT_THAT(BufMerge8.getNthFromEnd(1), UnorderedElementsAre(Instrs[1]));

  // Test merge() with different internal indices (this has popFront)
  // BufMerge10: push I0,I1 then pop -> [I1] (BottomIdx=1, TopIdx=2)
  // BufMerge11: [I2] (BottomIdx=0, TopIdx=1)
  // User-visible: both have index 0, so merge -> [{I1,I2}]
  AMDGPU::InstrBuffer BufMerge10, BufMerge11;
  BufMerge10.pushBack(Instrs[0]);
  BufMerge10.pushBack(Instrs[1]);
  BufMerge10.popFront(1);
  BufMerge11.pushBack(Instrs[2]);
  BufMerge10.merge(BufMerge11);
  EXPECT_EQ(BufMerge10.getTopIndex(), 1u);
  EXPECT_THAT(BufMerge10.getNthFromEnd(0),
              UnorderedElementsAre(Instrs[1], Instrs[2]));

  // Test merge() where Other has popFront
  // BufMerge12: [I0] (BottomIdx=0, TopIdx=1)
  // BufMerge13: push I1,I2 then pop -> [I2] (BottomIdx=1, TopIdx=2)
  // User-visible: both have index 0, so merge -> [{I0,I2}]
  AMDGPU::InstrBuffer BufMerge12, BufMerge13;
  BufMerge12.pushBack(Instrs[0]);
  BufMerge13.pushBack(Instrs[1]);
  BufMerge13.pushBack(Instrs[2]);
  BufMerge13.popFront(1);
  BufMerge12.merge(BufMerge13);
  EXPECT_EQ(BufMerge12.getTopIndex(), 1u);
  EXPECT_THAT(BufMerge12.getNthFromEnd(0),
              UnorderedElementsAre(Instrs[0], Instrs[2]));

  // Test merge() where both have different internal indices
  // BufMerge14: push I0,I1,I2 then pop 2 -> [I2] (BottomIdx=2, TopIdx=3)
  // BufMerge15: push I0,I1 then pop 1 -> [I1] (BottomIdx=1, TopIdx=2)
  // User-visible: both have index 0, so merge -> [{I1,I2}]
  AMDGPU::InstrBuffer BufMerge14, BufMerge15;
  BufMerge14.pushBack(Instrs[0]);
  BufMerge14.pushBack(Instrs[1]);
  BufMerge14.pushBack(Instrs[2]);
  BufMerge14.popFront(2);
  BufMerge15.pushBack(Instrs[0]);
  BufMerge15.pushBack(Instrs[1]);
  BufMerge15.popFront(1);
  BufMerge14.merge(BufMerge15);
  EXPECT_EQ(BufMerge14.getTopIndex(), 1u);
  EXPECT_THAT(BufMerge14.getNthFromEnd(0),
              UnorderedElementsAre(Instrs[1], Instrs[2]));

  // Test equality with different internal indices but same user-visible state.
  // BufEq1: push I0, push I1, pop -> [I1] (BottomIdx=1, TopIdx=2, user idx 0)
  // BufEq2: push I1 -> [I1] (BottomIdx=0, TopIdx=1, user idx 0)
  AMDGPU::InstrBuffer BufEq1, BufEq2;
  BufEq1.pushBack(Instrs[0]);
  BufEq1.pushBack(Instrs[1]);
  BufEq1.popFront(1);
  BufEq2.pushBack(Instrs[1]);
  EXPECT_TRUE(BufEq1 == BufEq2);

  // Test equality after merge (multiple instructions per index).
  // Both buffers: [{I0, I1}] at user-visible index 0.
  AMDGPU::InstrBuffer BufEq3, BufEq4, BufEq5, BufEq6;
  BufEq3.pushBack(Instrs[0]);
  BufEq4.pushBack(Instrs[1]);
  BufEq3.merge(BufEq4);
  BufEq5.pushBack(Instrs[0]);
  BufEq6.pushBack(Instrs[1]);
  BufEq5.merge(BufEq6);
  EXPECT_TRUE(BufEq3 == BufEq5);

  // Test inequality: same instructions but at different user-visible indices.
  // BufNeq1: [I0, I1]
  // BufNeq2: push I2, push I0, push I1, pop -> [I0, I1]
  // These should be equal.
  AMDGPU::InstrBuffer BufNeq1, BufNeq2;
  BufNeq1.pushBack(Instrs[0]);
  BufNeq1.pushBack(Instrs[1]);
  BufNeq2.pushBack(Instrs[2]);
  BufNeq2.pushBack(Instrs[0]);
  BufNeq2.pushBack(Instrs[1]);
  BufNeq2.popFront(1);
  EXPECT_TRUE(BufNeq1 == BufNeq2);

  // Test pushBack() with duplicate instruction.
  // When the same instruction is pushed twice, pushBack should remove it from
  // its old index before adding it to the new index. This ensures each
  // instruction appears at exactly one index.
  //
  // Sequence: push I0, push I1, push I0
  // After push I0:       [I0]
  // After push I1:       [I0, I1]
  // After push I0 again: I0 removed from idx 0, gap at front compacted,
  //                      I0 added at new top -> [I1, I0]
  //
  // Wait values (TopIndex - 1 - getIndex):
  //   I0: 2 - 1 - 1 = 0  (newest, no wait needed)
  //   I1: 2 - 1 - 0 = 1  (older, wait for 1)
  AMDGPU::InstrBuffer BufPushDup;
  BufPushDup.pushBack(Instrs[0]);
  BufPushDup.pushBack(Instrs[1]);
  BufPushDup.pushBack(Instrs[0]);
  EXPECT_EQ(BufPushDup.getTopIndex(), 2u);
  EXPECT_EQ(BufPushDup.getIndex(Instrs[0]), 1u);
  EXPECT_EQ(BufPushDup.getIndex(Instrs[1]), 0u);
  EXPECT_THAT(BufPushDup.getNthFromEnd(0), UnorderedElementsAre(Instrs[0]));
  EXPECT_THAT(BufPushDup.getNthFromEnd(1), UnorderedElementsAre(Instrs[1]));

  // Test merge() with buffer created by duplicate pushBack.
  // BufPushDup: [I1, I0] (from above)
  // Merging into empty BufMergeDup preserves this structure.
  AMDGPU::InstrBuffer BufMergeDup;
  BufMergeDup.merge(BufPushDup);
  EXPECT_EQ(BufMergeDup.getTopIndex(), 2u);
  EXPECT_EQ(BufMergeDup.getIndex(Instrs[0]), 1u);
  EXPECT_EQ(BufMergeDup.getIndex(Instrs[1]), 0u);
  EXPECT_THAT(BufMergeDup.getNthFromEnd(0), UnorderedElementsAre(Instrs[0]));
  EXPECT_THAT(BufMergeDup.getNthFromEnd(1), UnorderedElementsAre(Instrs[1]));

  // Test merge() preserves wait values when buffers have different sizes.
  // This simulates merging counter states from different CFG paths where
  // one path issued more instructions than the other.
  //
  // Case 1: Large.merge(Small) - merge smaller buffer into larger
  //
  // BufLarge: [I0, I1, I2] (TopIndex=3)
  //   I0: wait = 3-1-0 = 2
  //   I1: wait = 3-1-1 = 1
  //   I2: wait = 3-1-2 = 0
  //
  // BufSmall: [I3, I4] (TopIndex=2)
  //   I3: wait = 2-1-0 = 1
  //   I4: wait = 2-1-1 = 0
  //
  // After merge, positions are remapped to preserve wait values:
  //   I3 (wait=1) -> pos = 3-1-1 = 1
  //   I4 (wait=0) -> pos = 3-1-0 = 2
  //
  // Result: [I0, {I1,I3}, {I2,I4}]
  AMDGPU::InstrBuffer BufLarge, BufSmall;
  BufLarge.pushBack(Instrs[0]);
  BufLarge.pushBack(Instrs[1]);
  BufLarge.pushBack(Instrs[2]);
  BufSmall.pushBack(Instrs[3]);
  BufSmall.pushBack(Instrs[4]);
  BufLarge.merge(BufSmall);
  EXPECT_EQ(BufLarge.getTopIndex(), 3u);
  EXPECT_THAT(BufLarge.getNthFromEnd(0),
              UnorderedElementsAre(Instrs[2], Instrs[4]));
  EXPECT_THAT(BufLarge.getNthFromEnd(1),
              UnorderedElementsAre(Instrs[1], Instrs[3]));
  EXPECT_THAT(BufLarge.getNthFromEnd(2), UnorderedElementsAre(Instrs[0]));

  // Case 2: Small.merge(Large) - merge larger buffer into smaller
  //
  // BufSmall2: [I0, I1] (TopIndex=2)
  //   I0: wait = 2-1-0 = 1
  //   I1: wait = 2-1-1 = 0
  //
  // BufLarge2: [I2, I3, I4] (TopIndex=3)
  //   I2: wait = 3-1-0 = 2
  //   I3: wait = 3-1-1 = 1
  //   I4: wait = 3-1-2 = 0
  //
  // After merge, existing instructions are shifted to preserve their waits:
  //   I0 (wait=1) -> pos = 3-1-1 = 1
  //   I1 (wait=0) -> pos = 3-1-0 = 2
  // Incoming instructions are remapped to preserve their waits:
  //   I2 (wait=2) -> pos = 3-1-2 = 0
  //   I3 (wait=1) -> pos = 3-1-1 = 1
  //   I4 (wait=0) -> pos = 3-1-0 = 2
  //
  // Result: [I2, {I0,I3}, {I1,I4}]
  AMDGPU::InstrBuffer BufSmall2, BufLarge2;
  BufSmall2.pushBack(Instrs[0]);
  BufSmall2.pushBack(Instrs[1]);
  BufLarge2.pushBack(Instrs[2]);
  BufLarge2.pushBack(Instrs[3]);
  BufLarge2.pushBack(Instrs[4]);
  BufSmall2.merge(BufLarge2);
  EXPECT_EQ(BufSmall2.getTopIndex(), 3u);
  EXPECT_THAT(BufSmall2.getNthFromEnd(0),
              UnorderedElementsAre(Instrs[1], Instrs[4]));
  EXPECT_THAT(BufSmall2.getNthFromEnd(1),
              UnorderedElementsAre(Instrs[0], Instrs[3]));
  EXPECT_THAT(BufSmall2.getNthFromEnd(2), UnorderedElementsAre(Instrs[2]));

  // Test removeIf() removing a middle instruction leaves a gap but preserves
  // the positions (and therefore wait values) of the surviving instructions.
  // BufRm: [I0, I1, I2] (TopIndex=3). Remove I1 -> [I0, <gap>, I2].
  //   I0 stays at pos 0 (wait = 3-1-0 = 2)
  //   I2 stays at pos 2 (wait = 3-1-2 = 0)
  AMDGPU::InstrBuffer BufRm;
  BufRm.pushBack(Instrs[0]);
  BufRm.pushBack(Instrs[1]);
  BufRm.pushBack(Instrs[2]);
  BufRm.removeIf([&](AMDGPU::TrackedInstr TI) { return TI.getMI() == Instrs[1]; });
  EXPECT_EQ(BufRm.getTopIndex(), 3u);
  EXPECT_EQ(BufRm.numInstrs(), 2u);
  EXPECT_FALSE(BufRm.contains(Instrs[1]));
  EXPECT_EQ(BufRm.getIndex(Instrs[0]), 0u);
  EXPECT_EQ(BufRm.getIndex(Instrs[2]), 2u);
  EXPECT_THAT(BufRm.getNthFromEnd(1), IsEmpty()); // gap where I1 was

  // Test removeIf() trims leading gaps: removing the oldest instruction should
  // advance the bottom so positions of survivors decrease accordingly.
  // BufRmFront: [I0, I1, I2]. Remove I0 -> bottom advances -> [I1, I2].
  //   I1 now at pos 0, I2 at pos 1, TopIndex = 2.
  AMDGPU::InstrBuffer BufRmFront;
  BufRmFront.pushBack(Instrs[0]);
  BufRmFront.pushBack(Instrs[1]);
  BufRmFront.pushBack(Instrs[2]);
  BufRmFront.removeIf([&](AMDGPU::TrackedInstr TI) { return TI.getMI() == Instrs[0]; });
  EXPECT_EQ(BufRmFront.getTopIndex(), 2u);
  EXPECT_EQ(BufRmFront.numInstrs(), 2u);
  EXPECT_EQ(BufRmFront.getIndex(Instrs[1]), 0u);
  EXPECT_EQ(BufRmFront.getIndex(Instrs[2]), 1u);

  // Test removeIf() trims trailing gaps: removing the newest instruction should
  // lower the top so the buffer reports the correct size.
  // BufRmBack: [I0, I1, I2]. Remove I2 -> top lowers -> [I0, I1].
  AMDGPU::InstrBuffer BufRmBack;
  BufRmBack.pushBack(Instrs[0]);
  BufRmBack.pushBack(Instrs[1]);
  BufRmBack.pushBack(Instrs[2]);
  BufRmBack.removeIf([&](AMDGPU::TrackedInstr TI) { return TI.getMI() == Instrs[2]; });
  EXPECT_EQ(BufRmBack.getTopIndex(), 2u);
  EXPECT_EQ(BufRmBack.numInstrs(), 2u);
  EXPECT_EQ(BufRmBack.getIndex(Instrs[0]), 0u);
  EXPECT_EQ(BufRmBack.getIndex(Instrs[1]), 1u);

  // Test removeIf() that removes everything leaves an empty buffer.
  AMDGPU::InstrBuffer BufRmAll;
  BufRmAll.pushBack(Instrs[0]);
  BufRmAll.pushBack(Instrs[1]);
  BufRmAll.removeIf([](AMDGPU::TrackedInstr) { return true; });
  EXPECT_TRUE(BufRmAll.empty());
  EXPECT_EQ(BufRmAll.getTopIndex(), 0u);
  EXPECT_EQ(BufRmAll.numInstrs(), 0u);

  // Test removeIf() with a predicate matching nothing is a no-op.
  AMDGPU::InstrBuffer BufRmNone;
  BufRmNone.pushBack(Instrs[0]);
  BufRmNone.pushBack(Instrs[1]);
  BufRmNone.removeIf([](AMDGPU::TrackedInstr) { return false; });
  EXPECT_EQ(BufRmNone.getTopIndex(), 2u);
  EXPECT_EQ(BufRmNone.numInstrs(), 2u);
  EXPECT_EQ(BufRmNone.getIndex(Instrs[0]), 0u);
  EXPECT_EQ(BufRmNone.getIndex(Instrs[1]), 1u);

  // Test pushBack() with a single new instruction.
  // Behaves like pushBack: [I0, I1] -> pushBack({I2}) -> [I0, I1, I2]
  {
    AMDGPU::InstrBuffer Buf;
    Buf.pushBack(Instrs[0]);
    Buf.pushBack(Instrs[1]);
    Buf.pushBack({Instrs[2]});
    EXPECT_EQ(Buf.getTopIndex(), 3u);
    EXPECT_EQ(Buf.getIndex(Instrs[0]), 0u);
    EXPECT_EQ(Buf.getIndex(Instrs[1]), 1u);
    EXPECT_EQ(Buf.getIndex(Instrs[2]), 2u);
    EXPECT_THAT(Buf.getNthFromEnd(0), UnorderedElementsAre(Instrs[2]));
  }

  // Test pushBack() with two new instructions at the same index.
  // [I0, I1] -> pushBack({I2, I3}) -> [I0, I1, {I2, I3}]
  // I2 and I3 share the same index (TopIndex becomes 3).
  {
    AMDGPU::InstrBuffer Buf;
    Buf.pushBack(Instrs[0]);
    Buf.pushBack(Instrs[1]);
    Buf.pushBack({Instrs[2], Instrs[3]});
    EXPECT_EQ(Buf.getTopIndex(), 3u);
    EXPECT_EQ(Buf.getIndex(Instrs[0]), 0u);
    EXPECT_EQ(Buf.getIndex(Instrs[1]), 1u);
    EXPECT_EQ(Buf.getIndex(Instrs[2]), 2u);
    EXPECT_EQ(Buf.getIndex(Instrs[3]), 2u);
    EXPECT_EQ(Buf.numInstrs(), 4u);
    EXPECT_THAT(Buf.getNthFromEnd(0),
                UnorderedElementsAre(Instrs[2], Instrs[3]));
  }

  // Test pushBack() on empty buffer inserts at index 0.
  {
    AMDGPU::InstrBuffer Buf;
    Buf.pushBack({Instrs[0], Instrs[1]});
    EXPECT_EQ(Buf.getTopIndex(), 1u);
    EXPECT_EQ(Buf.getIndex(Instrs[0]), 0u);
    EXPECT_EQ(Buf.getIndex(Instrs[1]), 0u);
    EXPECT_EQ(Buf.numInstrs(), 2u);
  }

  // Test pushBack() with an already-present instruction at top — no-op.
  // [I0, I1] -> pushBack({I1}) -> [I0, I1] (I1 already at top)
  {
    AMDGPU::InstrBuffer Buf;
    Buf.pushBack(Instrs[0]);
    Buf.pushBack(Instrs[1]);
    Buf.pushBack({Instrs[1]});
    EXPECT_EQ(Buf.getTopIndex(), 2u);
    EXPECT_EQ(Buf.getIndex(Instrs[0]), 0u);
    EXPECT_EQ(Buf.getIndex(Instrs[1]), 1u);
    EXPECT_EQ(Buf.numInstrs(), 2u);
  }

  // Test pushBack() with an already-present instruction not at top.
  // [I0, I1, I2] -> pushBack({I0}) -> [I2, I1, I0]
  // I0 is moved from index 0 to the top (index 2), swapping with I2.
  {
    AMDGPU::InstrBuffer Buf;
    Buf.pushBack(Instrs[0]);
    Buf.pushBack(Instrs[1]);
    Buf.pushBack(Instrs[2]);
    Buf.pushBack({Instrs[0]});
    EXPECT_EQ(Buf.getTopIndex(), 3u);
    EXPECT_EQ(Buf.getIndex(Instrs[2]), 0u);
    EXPECT_EQ(Buf.getIndex(Instrs[1]), 1u);
    EXPECT_EQ(Buf.getIndex(Instrs[0]), 2u);
    EXPECT_EQ(Buf.numInstrs(), 3u);
  }

  // Test pushBack() with one new and one existing instruction.
  // [I0, I1] -> pushBack({I0, I2})
  // I0 (existing, at idx 0) is swapped to LastIdx (idx 1), I1 moves to idx 0.
  // I2 (new) is inserted at new top (idx 2).
  // Note: existing instruction I0 ends up at the old LastIdx (1), not the new
  // top — it does NOT share the same index as the new instruction I2.
  // Result: [I1(0), I0(1), I2(2)]
  {
    AMDGPU::InstrBuffer Buf;
    Buf.pushBack(Instrs[0]);
    Buf.pushBack(Instrs[1]);
    Buf.pushBack({Instrs[0], Instrs[2]});
    EXPECT_EQ(Buf.getTopIndex(), 3u);
    EXPECT_EQ(Buf.getIndex(Instrs[1]), 0u);
    EXPECT_EQ(Buf.getIndex(Instrs[0]), 1u);
    EXPECT_EQ(Buf.getIndex(Instrs[2]), 2u);
    EXPECT_EQ(Buf.numInstrs(), 3u);
    EXPECT_THAT(Buf.getNthFromEnd(0), UnorderedElementsAre(Instrs[2]));
    EXPECT_THAT(Buf.getNthFromEnd(1), UnorderedElementsAre(Instrs[0]));
    EXPECT_THAT(Buf.getNthFromEnd(2), UnorderedElementsAre(Instrs[1]));
  }

  // Test getNthFromEnd() after pushBack() with multiple instructions.
  // Verify WAIT_ASYNCMARK(N) semantics: getNthFromEnd(0) returns the set at the
  // most recent mark, getNthFromEnd(1) returns the previous mark's set.
  {
    AMDGPU::InstrBuffer Buf;
    // First mark: {I0, I1} at same index.
    Buf.pushBack({Instrs[0], Instrs[1]});
    // Second mark: {I2, I3} at same index (new top).
    Buf.pushBack({Instrs[2], Instrs[3]});
    EXPECT_EQ(Buf.getTopIndex(), 2u);
    EXPECT_THAT(Buf.getNthFromEnd(0),
                UnorderedElementsAre(Instrs[2], Instrs[3]));
    EXPECT_THAT(Buf.getNthFromEnd(1),
                UnorderedElementsAre(Instrs[0], Instrs[1]));
    EXPECT_THAT(Buf.getNthFromEnd(2), IsEmpty());
  }

#ifndef NDEBUG
  // Test overflow/TombstoneKey assertion in pushBack.
  // Use max() - 1: the assertion checks TopIdxInternal < max() - 1, so
  // max() - 1 fails immediately. This value also happens to be DenseMap's
  // TombstoneKey, so the assertion must fire before the DenseMap insert.
  AMDGPU::InstrBuffer BufWrap;
  BufWrap.TopIdxInternal = std::numeric_limits<unsigned>::max() - 1;
  BufWrap.BottomIdxInternal = std::numeric_limits<unsigned>::max() - 1;
  EXPECT_DEATH(BufWrap.pushBack(Instrs[0]),
               "TopIdxInternal overflow or TombstoneKey collision!");
#endif
}

// Test Counter::hasMixedEventTypes() for XCnt counter (gfx1250+).
// XCnt tracks address translation for VMEM and SMEM instructions.
// SMEM operations can complete out of order (unlike VMEM), so XCnt for SMEM
// is also out of order. Returns true when any SMEM instruction is pending.
TEST_F(AMDGPUTestBase, Counter_HasMixedEventTypes_XCnt) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1250", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $sgpr0_sgpr1, $vgpr0_vgpr1
    ; SMEM loads
    $sgpr2 = S_LOAD_DWORD_IMM $sgpr0_sgpr1, 0, 0
    $sgpr3 = S_LOAD_DWORD_IMM $sgpr0_sgpr1, 4, 0
    ; VMEM loads
    $vgpr2 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr3 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec
    ; FLAT loads (count as VMEM for XCnt purposes)
    $vgpr4 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST->getInstrInfo();

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 8> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 5u);

  MachineInstr *SmemLoad1 = Instrs[0];
  MachineInstr *SmemLoad2 = Instrs[1];
  MachineInstr *VmemLoad1 = Instrs[2];
  MachineInstr *VmemLoad2 = Instrs[3];
  MachineInstr *FlatLoad = Instrs[4];

  // Test XCnt counter with only SMEM: out of order (SMEM is always out of order)
  {
    AMDGPU::Counter XCtr{AMDGPU::XCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};

    // Empty counter: no mixed types
    EXPECT_FALSE(XCtr.hasMixedEventTypes(*TII))
        << "Empty counter should not have mixed types";

    // Single SMEM: out of order (SMEM can complete out of order)
    XCtr.insert(SmemLoad1);
    EXPECT_TRUE(XCtr.hasMixedEventTypes(*TII))
        << "SMEM should cause out of order (SMEM completes out of order)";

    // Two SMEMs: still out of order
    XCtr.insert(SmemLoad2);
    EXPECT_TRUE(XCtr.hasMixedEventTypes(*TII))
        << "Multiple SMEM loads should be out of order";
  }

  // Test XCnt counter with only VMEM: in order (no mixed types)
  {
    AMDGPU::Counter XCtr{AMDGPU::XCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};

    // Single VMEM: no mixed types (VMEM completes in order)
    XCtr.insert(VmemLoad1);
    EXPECT_FALSE(XCtr.hasMixedEventTypes(*TII))
        << "Single VMEM should not have mixed types";

    // Two VMEMs: no mixed types
    XCtr.insert(VmemLoad2);
    EXPECT_FALSE(XCtr.hasMixedEventTypes(*TII))
        << "Only VMEM loads should not have mixed types";
  }

  // Test XCnt counter with VMEM + FLAT: no mixed types (both are VMEM group)
  {
    AMDGPU::Counter XCtr{AMDGPU::XCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};

    XCtr.insert(VmemLoad1);
    XCtr.insert(FlatLoad);
    EXPECT_FALSE(XCtr.hasMixedEventTypes(*TII))
        << "VMEM + FLAT should not have mixed types (both are VMEM group)";
  }

  // Test XCnt counter with SMEM + VMEM: out of order (has SMEM)
  {
    AMDGPU::Counter XCtr{AMDGPU::XCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};

    XCtr.insert(SmemLoad1);
    XCtr.insert(VmemLoad1);
    EXPECT_TRUE(XCtr.hasMixedEventTypes(*TII))
        << "SMEM + VMEM should be out of order (SMEM is out of order)";
  }

  // Test XCnt counter with SMEM + FLAT: out of order (has SMEM)
  {
    AMDGPU::Counter XCtr{AMDGPU::XCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};

    XCtr.insert(SmemLoad1);
    XCtr.insert(FlatLoad);
    EXPECT_TRUE(XCtr.hasMixedEventTypes(*TII))
        << "SMEM + FLAT should be out of order (SMEM is out of order)";
  }

  // Test XCnt counter with all types: out of order (has SMEM)
  {
    AMDGPU::Counter XCtr{AMDGPU::XCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};

    XCtr.insert(SmemLoad1);
    XCtr.insert(SmemLoad2);
    XCtr.insert(VmemLoad1);
    XCtr.insert(VmemLoad2);
    XCtr.insert(FlatLoad);
    EXPECT_TRUE(XCtr.hasMixedEventTypes(*TII))
        << "SMEM + VMEM + FLAT should be out of order (SMEM is out of order)";
  }
}

// Test isNonZeroWaitLegal with pending instructions - covers all combinations
// from the test matrix in newwaitcnt-is-non-zero-wait-legal.mir.
//
// VMCNT combinations (4x4: NOSAMPLER, SAMPLER, BVH, FLAT):
//   - Same type pairs complete in order (vmcnt N)
//   - FLAT with VMEM-only: GFX9 conservative (vmcnt 0), GFX10+ in order (vmcnt N)
//
// LGKMCNT combinations (3x3: DS, SMEM, FLAT):
//   - DS only completes in order (lgkmcnt N)
//   - SMEM always out of order (lgkmcnt 0)
//   - FLAT with DS: GFX9 conservative (lgkmcnt 0), GFX10+ in order (lgkmcnt N)
TEST_F(AMDGPUTestBase, ResourceTracker_IsNonZeroWaitLegal_WithPending) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1010", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $vgpr2_vgpr3_vgpr4_vgpr5, $vgpr10, $sgpr0_sgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, $sgpr8_sgpr9_sgpr10_sgpr11, $m0, $vgpr20_vgpr21_vgpr22_vgpr23_vgpr24_vgpr25_vgpr26_vgpr27_vgpr28_vgpr29_vgpr30
    ; VMEM NOSAMPLER
    $vgpr40 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr41 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec
    ; VMEM SAMPLER
    $vgpr42 = IMAGE_SAMPLE_V1_V2 $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, 1, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s32))
    $vgpr43 = IMAGE_SAMPLE_V1_V2 $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, 1, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s32))
    ; VMEM BVH
    $vgpr44_vgpr45_vgpr46_vgpr47 = IMAGE_BVH_INTERSECT_RAY_sa_gfx10 $vgpr20_vgpr21_vgpr22_vgpr23_vgpr24_vgpr25_vgpr26_vgpr27_vgpr28_vgpr29_vgpr30, $sgpr0_sgpr1_sgpr2_sgpr3, 0, implicit $exec :: (load (s128))
    $vgpr48_vgpr49_vgpr50_vgpr51 = IMAGE_BVH_INTERSECT_RAY_sa_gfx10 $vgpr20_vgpr21_vgpr22_vgpr23_vgpr24_vgpr25_vgpr26_vgpr27_vgpr28_vgpr29_vgpr30, $sgpr0_sgpr1_sgpr2_sgpr3, 0, implicit $exec :: (load (s128))
    ; FLAT
    $vgpr52 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    $vgpr53 = FLAT_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec, implicit $flat_scr
    ; DS (LDS)
    $vgpr54 = DS_READ_B32_gfx9 $vgpr10, 0, 0, implicit $exec, implicit $m0
    $vgpr55 = DS_READ_B32_gfx9 $vgpr10, 4, 0, implicit $exec, implicit $m0
    ; SMEM
    $sgpr10 = S_LOAD_DWORD_IMM $sgpr0_sgpr1, 0, 0 :: (load (s32))
    $sgpr11 = S_LOAD_DWORD_IMM $sgpr0_sgpr1, 4, 0 :: (load (s32))
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  ASSERT_TRUE(ST->hasFlatLgkmVMemCountInOrder())
      << "gfx1010 should have hasFlatLgkmVMemCountInOrder";

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 16> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 12u);

  MachineInstr *Global1 = Instrs[0];
  MachineInstr *Global2 = Instrs[1];
  MachineInstr *Sample1 = Instrs[2];
  MachineInstr *Sample2 = Instrs[3];
  MachineInstr *Bvh1 = Instrs[4];
  MachineInstr *Bvh2 = Instrs[5];
  MachineInstr *Flat1 = Instrs[6];
  MachineInstr *Flat2 = Instrs[7];
  MachineInstr *Ds1 = Instrs[8];
  MachineInstr *Ds2 = Instrs[9];
  MachineInstr *Smem1 = Instrs[10];
  MachineInstr *Smem2 = Instrs[11];

  AMDGPU::SchedulingMode SchedMode = AMDGPU::SchedulingMode::NoExpert;

  const SIInstrInfo *TII = ST->getInstrInfo();

  // =========================================================================
  // VMCNT tests with pending instructions
  // Uses Counter::insert() to track pending instructions directly in the
  // counter, then checks isNonZeroWaitLegal which internally calls
  // hasMixedEventTypes() on the counter's pending set.
  // =========================================================================

  // Test: GLOBAL src with SAMPLER pending -> still in order (vmcnt N)
  // Different VmemTypes (NOSAMPLER, SAMPLER) are in-order for pre-GFX12
  {
    AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    VmCntr.insert(Sample1);  // SAMPLER pending
    VmCntr.insert(Global1);  // NOSAMPLER src
    VmCntr.insert(Global2);  // NOSAMPLER dst

    EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*Global1, *Global2, *ST, SchedMode))
        << "GLOBAL + GLOBAL with SAMPLER pending should be in order";
  }

  // Test: SAMPLER src with NOSAMPLER pending -> still in order (vmcnt N)
  {
    AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    VmCntr.insert(Global1);  // NOSAMPLER pending
    VmCntr.insert(Sample1);  // SAMPLER src
    VmCntr.insert(Sample2);  // SAMPLER dst

    EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*Sample1, *Sample2, *ST, SchedMode))
        << "SAMPLER + SAMPLER with NOSAMPLER pending should be in order";
  }

  // Test: BVH src with NOSAMPLER pending -> still in order (vmcnt N)
  {
    AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    VmCntr.insert(Global1);  // NOSAMPLER pending
    VmCntr.insert(Bvh1);     // BVH src
    VmCntr.insert(Bvh2);     // BVH dst

    EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*Bvh1, *Bvh2, *ST, SchedMode))
        << "BVH + BVH with NOSAMPLER pending should be in order";
  }

  // Test: NOSAMPLER src with BVH pending -> still in order (vmcnt N)
  {
    AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    VmCntr.insert(Bvh1);     // BVH pending
    VmCntr.insert(Global1);  // NOSAMPLER src
    VmCntr.insert(Global2);  // NOSAMPLER dst

    EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*Global1, *Global2, *ST, SchedMode))
        << "GLOBAL + GLOBAL with BVH pending should be in order";
  }

  // Test: BVH src with SAMPLER pending -> still in order (vmcnt N)
  {
    AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    VmCntr.insert(Sample1);  // SAMPLER pending
    VmCntr.insert(Bvh1);     // BVH src
    VmCntr.insert(Bvh2);     // BVH dst

    EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*Bvh1, *Bvh2, *ST, SchedMode))
        << "BVH + BVH with SAMPLER pending should be in order";
  }

  // Test: SAMPLER src with BVH pending -> still in order (vmcnt N)
  {
    AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    VmCntr.insert(Bvh1);     // BVH pending
    VmCntr.insert(Sample1);  // SAMPLER src
    VmCntr.insert(Sample2);  // SAMPLER dst

    EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*Sample1, *Sample2, *ST, SchedMode))
        << "SAMPLER + SAMPLER with BVH pending should be in order";
  }

  // Test: NOSAMPLER src with FLAT pending -> GFX10+ in order
  {
    AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    VmCntr.insert(Flat1);    // FLAT pending
    VmCntr.insert(Global1);  // NOSAMPLER src
    VmCntr.insert(Global2);  // NOSAMPLER dst

    EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*Global1, *Global2, *ST, SchedMode))
        << "GLOBAL + GLOBAL with FLAT pending should be in order on GFX10+";
    // Verify hasMixedEventTypes returns false (FLAT mixed is OK on GFX10+)
    EXPECT_FALSE(VmCntr.hasMixedEventTypes(*TII))
        << "NOSAMPLER + FLAT should NOT have mixed event types on GFX10+";
  }

  // Test: FLAT src with NOSAMPLER pending -> GFX10+ in order
  {
    AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    VmCntr.insert(Global1);  // NOSAMPLER pending
    VmCntr.insert(Flat1);    // FLAT src
    VmCntr.insert(Flat2);    // FLAT dst

    EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*Flat1, *Flat2, *ST, SchedMode))
        << "FLAT + FLAT with NOSAMPLER pending should be in order on GFX10+";
  }

  // Test: BVH src with FLAT pending -> GFX10+ in order
  {
    AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    VmCntr.insert(Flat1);  // FLAT pending
    VmCntr.insert(Bvh1);   // BVH src
    VmCntr.insert(Bvh2);   // BVH dst

    EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*Bvh1, *Bvh2, *ST, SchedMode))
        << "BVH + BVH with FLAT pending should be in order on GFX10+";
  }

  // Test: FLAT src with BVH pending -> GFX10+ in order
  {
    AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    VmCntr.insert(Bvh1);   // BVH pending
    VmCntr.insert(Flat1);  // FLAT src
    VmCntr.insert(Flat2);  // FLAT dst

    EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*Flat1, *Flat2, *ST, SchedMode))
        << "FLAT + FLAT with BVH pending should be in order on GFX10+";
  }

  // Test: SAMPLER src with FLAT pending -> GFX10+ in order
  {
    AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    VmCntr.insert(Flat1);    // FLAT pending
    VmCntr.insert(Sample1);  // SAMPLER src
    VmCntr.insert(Sample2);  // SAMPLER dst

    EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*Sample1, *Sample2, *ST, SchedMode))
        << "SAMPLER + SAMPLER with FLAT pending should be in order on GFX10+";
  }

  // Test: FLAT src with SAMPLER pending -> GFX10+ in order
  {
    AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    VmCntr.insert(Sample1);  // SAMPLER pending
    VmCntr.insert(Flat1);    // FLAT src
    VmCntr.insert(Flat2);    // FLAT dst

    EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*Flat1, *Flat2, *ST, SchedMode))
        << "FLAT + FLAT with SAMPLER pending should be in order on GFX10+";
  }

  // =========================================================================
  // LGKMCNT tests with pending instructions
  // =========================================================================
  // Test: DS src with DS pending only -> in order (lgkmcnt N)
  {
    AMDGPU::Counter LgkmCntr{AMDGPU::LgkmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    LgkmCntr.insert(Ds1);  // DS src
    LgkmCntr.insert(Ds2);  // DS dst

    EXPECT_TRUE(LgkmCntr.isNonZeroWaitLegal(*Ds1, *Ds2, *ST, SchedMode))
        << "DS + DS should be in order";
    EXPECT_FALSE(LgkmCntr.hasMixedEventTypes(*TII))
        << "DS only should NOT have mixed event types";
  }

  // Test: DS src with SMEM pending -> NOT in order (SMEM always OoO)
  {
    AMDGPU::Counter LgkmCntr{AMDGPU::LgkmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    LgkmCntr.insert(Smem1);  // SMEM pending (always OoO)
    LgkmCntr.insert(Ds1);    // DS src
    LgkmCntr.insert(Ds2);    // DS dst

    EXPECT_FALSE(LgkmCntr.isNonZeroWaitLegal(*Ds1, *Ds2, *ST, SchedMode))
        << "DS + DS with SMEM pending should NOT be in order";
    EXPECT_TRUE(LgkmCntr.hasMixedEventTypes(*TII))
        << "DS + SMEM should have mixed event types";
  }

  // Test: SMEM src with DS pending -> NOT in order (SMEM always OoO)
  {
    AMDGPU::Counter LgkmCntr{AMDGPU::LgkmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    LgkmCntr.insert(Ds1);    // DS pending
    LgkmCntr.insert(Smem1);  // SMEM src (always OoO)
    LgkmCntr.insert(Smem2);  // SMEM dst

    EXPECT_FALSE(LgkmCntr.isNonZeroWaitLegal(*Smem1, *Smem2, *ST, SchedMode))
        << "SMEM + SMEM with DS pending should NOT be in order";
  }

  // Test: SMEM src with SMEM pending only -> NOT in order (SMEM always OoO)
  {
    AMDGPU::Counter LgkmCntr{AMDGPU::LgkmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    LgkmCntr.insert(Smem1);  // SMEM src
    LgkmCntr.insert(Smem2);  // SMEM dst

    EXPECT_FALSE(LgkmCntr.isNonZeroWaitLegal(*Smem1, *Smem2, *ST, SchedMode))
        << "SMEM + SMEM should NOT be in order (SMEM always OoO)";
    // SMEM only is still mixed due to its inherent OoO nature
    EXPECT_TRUE(LgkmCntr.hasMixedEventTypes(*TII))
        << "SMEM only should have mixed event types (SMEM is OoO)";
  }

  // Test: SMEM src with FLAT pending -> NOT in order (SMEM always OoO)
  {
    AMDGPU::Counter LgkmCntr{AMDGPU::LgkmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    LgkmCntr.insert(Flat1);  // FLAT pending
    LgkmCntr.insert(Smem1);  // SMEM src (always OoO)
    LgkmCntr.insert(Smem2);  // SMEM dst

    EXPECT_FALSE(LgkmCntr.isNonZeroWaitLegal(*Smem1, *Smem2, *ST, SchedMode))
        << "SMEM + SMEM with FLAT pending should NOT be in order";
  }

  // Test: DS src with FLAT pending -> GFX10+ in order
  {
    AMDGPU::Counter LgkmCntr{AMDGPU::LgkmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    LgkmCntr.insert(Flat1);  // FLAT pending
    LgkmCntr.insert(Ds1);    // DS src
    LgkmCntr.insert(Ds2);    // DS dst

    EXPECT_TRUE(LgkmCntr.isNonZeroWaitLegal(*Ds1, *Ds2, *ST, SchedMode))
        << "DS + DS with FLAT pending should be in order on GFX10+";
    EXPECT_FALSE(LgkmCntr.hasMixedEventTypes(*TII))
        << "DS + FLAT should NOT have mixed event types on GFX10+";
  }

  // Test: FLAT src with DS pending -> GFX10+ in order
  {
    AMDGPU::Counter LgkmCntr{AMDGPU::LgkmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    LgkmCntr.insert(Ds1);    // DS pending
    LgkmCntr.insert(Flat1);  // FLAT src
    LgkmCntr.insert(Flat2);  // FLAT dst

    EXPECT_TRUE(LgkmCntr.isNonZeroWaitLegal(*Flat1, *Flat2, *ST, SchedMode))
        << "FLAT + FLAT with DS pending should be in order on GFX10+";
  }

  // Test: FLAT src with SMEM pending -> NOT in order (SMEM always OoO)
  {
    AMDGPU::Counter LgkmCntr{AMDGPU::LgkmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    LgkmCntr.insert(Smem1);  // SMEM pending (always OoO)
    LgkmCntr.insert(Flat1);  // FLAT src
    LgkmCntr.insert(Flat2);  // FLAT dst

    EXPECT_FALSE(LgkmCntr.isNonZeroWaitLegal(*Flat1, *Flat2, *ST, SchedMode))
        << "FLAT + FLAT with SMEM pending should NOT be in order";
  }

  // =========================================================================
  // GFX9 behavior (no hasFlatLgkmVMemCountInOrder)
  // On GFX9, FLAT can report early completion, so isNonZeroWaitLegal returns
  // false when FLAT is pending. This is checked via needsFlatEarlyCompletionWorkaround
  // at the start of isNonZeroWaitLegal.
  // =========================================================================
  {
    auto TM_GFX9 = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx900", "");
    ASSERT_TRUE(TM_GFX9) << "No target machine for gfx900";

    LLVMContext Context9;
    MachineModuleInfo MMI9(TM_GFX9.get());
    auto M9 = parseMIR(Context9, *TM_GFX9, MIRString, "test", MMI9);
    ASSERT_TRUE(M9) << "Failed to parse MIR for gfx900";

    auto *MF9 = MMI9.getMachineFunction(*M9->getFunction("test"));
    ASSERT_TRUE(MF9) << "Failed to get MachineFunction";

    const GCNSubtarget *ST9 = &MF9->getSubtarget<GCNSubtarget>();
    ASSERT_FALSE(ST9->hasFlatLgkmVMemCountInOrder())
        << "gfx900 should NOT have hasFlatLgkmVMemCountInOrder";

    const SIInstrInfo *TII9 = ST9->getInstrInfo();

    auto *MBB9 = MF9->getBlockNumbered(0);
    SmallVector<MachineInstr *, 16> Instrs9;
    for (MachineInstr &MI : *MBB9) {
      if (!MI.isTerminator())
        Instrs9.push_back(&MI);
    }

    MachineInstr *Global1_9 = Instrs9[0];
    MachineInstr *Global2_9 = Instrs9[1];
    MachineInstr *Flat1_9 = Instrs9[6];
    MachineInstr *Flat2_9 = Instrs9[7];
    MachineInstr *Ds1_9 = Instrs9[8];
    MachineInstr *Ds2_9 = Instrs9[9];

    // Test: NOSAMPLER src with FLAT pending on vmcnt
    // isNonZeroWaitLegal returns false because needsFlatEarlyCompletionWorkaround
    // is true on GFX9 when FLAT is pending.
    {
      AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
      VmCntr.insert(Flat1_9);    // FLAT pending
      VmCntr.insert(Global1_9);  // NOSAMPLER src
      VmCntr.insert(Global2_9);  // NOSAMPLER dst

      EXPECT_TRUE(VmCntr.needsFlatEarlyCompletionWorkaround(*ST9))
          << "GFX9: FLAT pending triggers workaround";
      EXPECT_FALSE(VmCntr.isNonZeroWaitLegal(*Global1_9, *Global2_9, *ST9, SchedMode))
          << "GFX9: GLOBAL + GLOBAL with FLAT pending - isNonZeroWaitLegal "
             "returns false due to FLAT early completion workaround";
      EXPECT_FALSE(VmCntr.hasMixedEventTypes(*TII9))
          << "hasMixedEventTypes doesn't count FLAT without strict-mixed-flat-check";
    }

    // Test: DS src with FLAT pending on lgkmcnt
    // isNonZeroWaitLegal returns false because needsFlatEarlyCompletionWorkaround
    // is true on GFX9 when FLAT is pending.
    {
      AMDGPU::Counter LgkmCntr{AMDGPU::LgkmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
      LgkmCntr.insert(Flat1_9);  // FLAT pending
      LgkmCntr.insert(Ds1_9);    // DS src
      LgkmCntr.insert(Ds2_9);    // DS dst

      EXPECT_TRUE(LgkmCntr.needsFlatEarlyCompletionWorkaround(*ST9))
          << "GFX9: FLAT pending on lgkmcnt triggers workaround";
      EXPECT_FALSE(LgkmCntr.isNonZeroWaitLegal(*Ds1_9, *Ds2_9, *ST9, SchedMode))
          << "GFX9: DS + DS with FLAT pending - isNonZeroWaitLegal returns "
             "false due to FLAT early completion workaround";
      EXPECT_FALSE(LgkmCntr.hasMixedEventTypes(*TII9))
          << "hasMixedEventTypes doesn't count FLAT without strict-mixed-flat-check";
    }

    // Test: FLAT src with NOSAMPLER pending on vmcnt
    // isNonZeroWaitLegal returns false because needsFlatEarlyCompletionWorkaround
    // is true on GFX9 when FLAT is pending.
    {
      AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
      VmCntr.insert(Global1_9);  // NOSAMPLER pending
      VmCntr.insert(Flat1_9);    // FLAT src
      VmCntr.insert(Flat2_9);    // FLAT dst

      EXPECT_TRUE(VmCntr.needsFlatEarlyCompletionWorkaround(*ST9))
          << "GFX9: FLAT pending triggers workaround";
      EXPECT_FALSE(VmCntr.isNonZeroWaitLegal(*Flat1_9, *Flat2_9, *ST9, SchedMode))
          << "GFX9: FLAT + FLAT with NOSAMPLER pending - isNonZeroWaitLegal "
             "returns false due to FLAT early completion workaround";
    }
  }
}

// Test isNonZeroWaitLegal with pending instructions at different positions:
// - Pending BEFORE the definition (src)
// - Pending AFTER the use (dst)
//
// These tests verify that isNonZeroWaitLegal correctly considers all pending
// instructions in the counter, regardless of their position relative to the
// src/dst pair.
TEST_F(AMDGPUTestBase, ResourceTracker_IsNonZeroWaitLegal_PendingPosition) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1010", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $vgpr10, $sgpr0_sgpr1, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr4, $sgpr8_sgpr9_sgpr10_sgpr11, $m0
    ; VMEM NOSAMPLER
    $vgpr40 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr41 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec
    $vgpr42 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 8, 0, implicit $exec
    ; VMEM SAMPLER
    $vgpr43 = IMAGE_SAMPLE_V1_V2 $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3_sgpr4_sgpr5_sgpr6_sgpr7, $sgpr8_sgpr9_sgpr10_sgpr11, 1, 0, 0, 0, 0, 0, 0, 0, implicit $exec :: (load (s32))
    ; DS (LDS)
    $vgpr50 = DS_READ_B32_gfx9 $vgpr10, 0, 0, implicit $exec, implicit $m0
    $vgpr51 = DS_READ_B32_gfx9 $vgpr10, 4, 0, implicit $exec, implicit $m0
    $vgpr52 = DS_READ_B32_gfx9 $vgpr10, 8, 0, implicit $exec, implicit $m0
    ; SMEM
    $sgpr10 = S_LOAD_DWORD_IMM $sgpr0_sgpr1, 0, 0 :: (load (s32))
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 16> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 8u);

  MachineInstr *Global1 = Instrs[0];
  MachineInstr *Global2 = Instrs[1];
  MachineInstr *Global3 = Instrs[2];
  MachineInstr *Sample1 = Instrs[3];
  MachineInstr *Ds1 = Instrs[4];
  MachineInstr *Ds2 = Instrs[5];
  MachineInstr *Ds3 = Instrs[6];
  MachineInstr *Smem1 = Instrs[7];

  AMDGPU::SchedulingMode SchedMode = AMDGPU::SchedulingMode::NoExpert;

  // =========================================================================
  // Tests for pending BEFORE the definition (src)
  // Pattern: Pending, Def (src), Filler, Use (dst)
  // =========================================================================

  // VMCNT: SAMPLER pending before GLOBAL def -> in order (pre-GFX12)
  {
    AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    VmCntr.insert(Sample1);  // Pending BEFORE def
    VmCntr.insert(Global1);  // Def (src)
    VmCntr.insert(Global2);  // Filler (same counter)
    // Global1 is src, Global2 is where wait is emitted (dst)

    EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*Global1, *Global2, *ST, SchedMode))
        << "SAMPLER pending before GLOBAL def should be in order (pre-GFX12)";
  }

  // LGKMCNT: SMEM pending before DS def -> NOT in order (SMEM always OoO)
  {
    AMDGPU::Counter LgkmCntr{AMDGPU::LgkmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    LgkmCntr.insert(Smem1);  // Pending BEFORE def (SMEM is OoO)
    LgkmCntr.insert(Ds1);    // Def (src)
    LgkmCntr.insert(Ds2);    // Filler (same counter)
    // Ds1 is src, Ds2 is dst

    EXPECT_FALSE(LgkmCntr.isNonZeroWaitLegal(*Ds1, *Ds2, *ST, SchedMode))
        << "SMEM pending before DS def should NOT be in order (SMEM OoO)";
  }

  // LGKMCNT: DS pending before DS def (all DS) -> in order
  {
    AMDGPU::Counter LgkmCntr{AMDGPU::LgkmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    LgkmCntr.insert(Ds1);  // Pending BEFORE def
    LgkmCntr.insert(Ds2);  // Def (src)
    LgkmCntr.insert(Ds3);  // Filler (same counter)
    // Ds2 is src, Ds3 is dst

    EXPECT_TRUE(LgkmCntr.isNonZeroWaitLegal(*Ds2, *Ds3, *ST, SchedMode))
        << "DS pending before DS def should be in order";
  }

  // =========================================================================
  // Tests for pending AFTER the use (dst)
  // Pattern: Def (src), Filler, Use (dst), Pending
  // The wait is inserted before Use, so Pending has not been seen yet.
  // However, isNonZeroWaitLegal considers all instructions in the counter.
  // =========================================================================

  // VMCNT: GLOBAL def, then SAMPLER pending after use
  // When the wait is inserted, Pending hasn't been issued yet, but
  // isNonZeroWaitLegal checks hasMixedEventTypes on the counter, which
  // includes all inserted instructions.
  {
    AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    VmCntr.insert(Global1);  // Def (src)
    VmCntr.insert(Global2);  // Filler (same counter)
    VmCntr.insert(Sample1);  // Pending AFTER use

    // Even with SAMPLER after use, GLOBAL+GLOBAL pair is in order (pre-GFX12)
    EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*Global1, *Global2, *ST, SchedMode))
        << "SAMPLER pending after GLOBAL use should be in order (pre-GFX12)";
  }

  // LGKMCNT: DS def, then SMEM pending after use -> NOT in order
  {
    AMDGPU::Counter LgkmCntr{AMDGPU::LgkmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    LgkmCntr.insert(Ds1);    // Def (src)
    LgkmCntr.insert(Ds2);    // Filler (same counter)
    LgkmCntr.insert(Smem1);  // Pending AFTER use

    EXPECT_FALSE(LgkmCntr.isNonZeroWaitLegal(*Ds1, *Ds2, *ST, SchedMode))
        << "SMEM pending after DS use should NOT be in order (SMEM OoO)";
  }

  // LGKMCNT: DS def, then DS pending after use (all DS) -> in order
  {
    AMDGPU::Counter LgkmCntr{AMDGPU::LgkmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    LgkmCntr.insert(Ds1);  // Def (src)
    LgkmCntr.insert(Ds2);  // Filler (same counter)
    LgkmCntr.insert(Ds3);  // Pending AFTER use

    EXPECT_TRUE(LgkmCntr.isNonZeroWaitLegal(*Ds1, *Ds2, *ST, SchedMode))
        << "DS pending after DS use should be in order";
  }

  // =========================================================================
  // Combined: pending BEFORE def AND AFTER use
  // Pattern: PendingBefore, Def (src), Filler, Use (dst), PendingAfter
  // =========================================================================

  // VMCNT: GLOBAL with SAMPLER before AND after
  {
    AMDGPU::Counter VmCntr{AMDGPU::VmCnt(), /*MaxSize=*/0, /*DropOnOverflow=*/false};
    VmCntr.insert(Sample1);  // Pending BEFORE def
    VmCntr.insert(Global1);  // Def (src)
    VmCntr.insert(Global2);  // Filler
    // Note: In practice Global3 would be "after use" but since we're testing
    // the counter state, the order of insert() matters for position tracking
    VmCntr.insert(Global3);  // Another pending (could be before or after)

    EXPECT_TRUE(VmCntr.isNonZeroWaitLegal(*Global1, *Global2, *ST, SchedMode))
        << "SAMPLER + GLOBAL pending should be in order (pre-GFX12)";
  }
}

// Expert-mode counters (VaVdst, VmVsrc) always complete in order, so
// position-based (non-zero) waits are legal.
TEST_F(AMDGPUTestBase, ResourceTracker_IsNonZeroWaitLegal_ExpertCounters) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $sgpr0, $sgpr1, $vgpr0_vgpr1
    $vgpr2 = V_MOV_B32_e32 $sgpr0, implicit $exec
    $vgpr3 = V_MOV_B32_e32 $sgpr1, implicit $exec
    renamable $vgpr4 = FLAT_LOAD_DWORD renamable $vgpr0_vgpr1, 0, 0, implicit $exec, implicit $flat_scr
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";
  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";
  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr,
                              AMDGPU::SchedulingMode::ExpertMode2);

  auto It = MBB->begin();
  MachineInstr &VMov0 = *It++;   // V_MOV vgpr2
  MachineInstr &VMov1 = *It++;   // V_MOV vgpr3
  MachineInstr &FlatLoad = *It++; // FLAT_LOAD vgpr4

  RT.track(VMov0);
  RT.track(VMov1);

  // VaVdst should allow non-zero waits (in-order VALU pipeline).
  const AMDGPU::Counter &VaVdstCntr = RT.getCounter(AMDGPU::VaVdst());
  EXPECT_TRUE(VaVdstCntr.isNonZeroWaitLegal(VMov0, FlatLoad, *ST,
                                              AMDGPU::SchedulingMode::ExpertMode2));

  RT.track(FlatLoad);

  // VmVsrc should allow non-zero waits (in-order VMEM source read).
  const AMDGPU::Counter &VmVsrcCntr = RT.getCounter(AMDGPU::VmVsrc());
  EXPECT_TRUE(VmVsrcCntr.isNonZeroWaitLegal(FlatLoad, VMov0, *ST,
                                              AMDGPU::SchedulingMode::ExpertMode2));
}

// Test that getWaitFor() returns an ExpCnt wait for a write-after-read hazard on
// a store's data source register on gfx6 (pre-SEA_ISLANDS). There a VMEM store
// reads its data register some time after issue and protects it via ExpCnt, so
// overwriting that register needs s_waitcnt expcnt(0).
TEST_F(AMDGPUTestBase, ResourceTracker_GetWaitFor_StoreDataExpCntWar) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "verde", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3
    ; Store reads $vgpr0 as data.
    BUFFER_STORE_DWORD_OFFSET $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    ; Overwrite $vgpr0 - WAR hazard on the store's data source.
    $vgpr0 = V_MOV_B32_e32 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  ASSERT_TRUE(ST->vmemWriteNeedsExpWaitcnt())
      << "verde (gfx6) should need an ExpCnt wait for store data sources";

  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr,
                             AMDGPU::SchedulingMode::NoExpert);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 4> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 2u);

  MachineInstr *Store = Instrs[0];
  MachineInstr *MovOverwrite = Instrs[1];

  // Track the store so its data-source USE operand is recorded.
  RT.track(*Store);

  // getWaitFor for $vgpr0 with the overwriting V_MOV as DstMI (WAR, Def access)
  // should return an ExpCnt wait of 0.
  auto Waits = RT.getWaitFor(AMDGPU::VGPR0, *MovOverwrite,
                             AMDGPU::ResourceTracker::RegAccessType::Def);

  bool HasExpCnt = false;
  unsigned ExpCntWait = ~0u;
  for (const auto &W : Waits) {
    if (W.Cntr == AMDGPU::ExpCnt()) {
      HasExpCnt = true;
      ExpCntWait = W.Wait;
    }
  }
  EXPECT_TRUE(HasExpCnt)
      << "WAR on store data source should require an ExpCnt wait on gfx6";
  EXPECT_EQ(ExpCntWait, 0u) << "Store data WAR should wait expcnt(0)";
}

// Test InstrBuffer::merge() — always uses max semantics (backedge merge).
TEST_F(AMDGPUResourceTrackerTest, InstrBuffer_MergeUseMax) {
  StringRef MIRString = R"MIR(
---
name: test
body: |
  bb.0:
    $vgpr0 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr1 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec
    $vgpr2 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 8, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);

  SmallVector<MachineInstr *, 3> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (MI.getOpcode() == AMDGPU::GLOBAL_LOAD_DWORD)
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 3u);

  // Backedge merge scenario: models two sequential loops.
  // Entry (bb.0): {I0 at pos 0}, TopIdx=1, I0 wait=0
  // Backedge (bb.1 exit): {I0 at pos 0, I1 at pos 1}, TopIdx=2,
  //   I0 wait=1, I1 wait=0
  //
  // merge() keeps lowest wait (most recent position) for shared instructions.
  // Entry=[I0@wait=0], Backedge=[I0@wait=1, I1@wait=0].
  // I0 in Entry is at merged idx=1 (offset by ThisOffset=1). I0 in Backedge
  // is at merged idx=0. Keep higher index (lower wait) → I0 at idx=1, I1 also
  // at idx=1. Both collapse to the same slot.
  {
    AMDGPU::InstrBuffer Entry, Backedge;
    Entry.pushBack(Instrs[0]);    // I0 at pos 0, wait=0
    Backedge.pushBack(Instrs[0]); // I0 at pos 0, wait=1
    Backedge.pushBack(Instrs[1]); // I1 at pos 1, wait=0

    AMDGPU::InstrBuffer Merged = Entry;
    Merged.merge(Backedge);
    // After merge: I0 kept at merged idx=1 (ThisOffset=1), I1 also at idx=1.
    // Both share the same slot, TopIndex=1.
    EXPECT_EQ(Merged.getTopIndex(), 1u);
    EXPECT_THAT(Merged.getNthFromEnd(0), UnorderedElementsAre(Instrs[0], Instrs[1]));
  }

  // Backedge merge with reordering: entry has I1 newer, backedge has I0 newer.
  // Entry: {I0@idx=0, I1@idx=1}, Backedge: {I1@idx=0, I0@idx=1}.
  // Both offsets=0. I1: Entry has idx=1, Backedge has idx=0 → keep idx=1 (lower wait).
  // I0: Entry has idx=0, Backedge has idx=1 → replace with idx=1 (lower wait).
  // Both I0 and I1 end up at idx=1. Same slot, TopIndex=1.
  {
    AMDGPU::InstrBuffer Entry, Backedge;
    Entry.pushBack(Instrs[0]);
    Entry.pushBack(Instrs[1]);
    Backedge.pushBack(Instrs[1]);
    Backedge.pushBack(Instrs[0]);

    AMDGPU::InstrBuffer MaxMerge = Entry;
    MaxMerge.merge(Backedge);
    EXPECT_EQ(MaxMerge.getTopIndex(), 1u);
    EXPECT_THAT(MaxMerge.getNthFromEnd(0),
                UnorderedElementsAre(Instrs[0], Instrs[1]));
  }
}

// Test InstrBuffer::merge() where one buffer contains {I0} and the other
// contains {I0, I1, I2}. The result must contain all three instructions.
TEST_F(AMDGPUResourceTrackerTest, InstrBuffer_MergeAllRetained) {
  StringRef MIRString = R"MIR(
---
name: test
body: |
  bb.0:
    $vgpr0 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr1 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec
    $vgpr2 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 8, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);

  SmallVector<MachineInstr *, 3> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (MI.getOpcode() == AMDGPU::GLOBAL_LOAD_DWORD)
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 3u);

  // One predecessor has only I0 pending (1 entry).
  // The other predecessor has I0, I1, I2 pending (3 entries, I0 oldest).
  // After merge, all three instructions must be retained so that register
  // dependencies on I0 and I1 are not lost at the CFG join.
  AMDGPU::InstrBuffer BufSmall, BufLarge;
  BufSmall.pushBack(Instrs[0]); // {I0}

  BufLarge.pushBack(Instrs[0]); // oldest
  BufLarge.pushBack(Instrs[1]);
  BufLarge.pushBack(Instrs[2]); // newest

  BufSmall.merge(BufLarge);

  // All three instructions must be present in the merged buffer.
  EXPECT_TRUE(BufSmall.contains(Instrs[0]));
  EXPECT_TRUE(BufSmall.contains(Instrs[1]));
  EXPECT_TRUE(BufSmall.contains(Instrs[2]));

  // Entry had I0 at offset+2=merged idx=2 (lowest wait). BufLarge I0 at idx=0
  // is skipped (keep lower wait). I1@idx=1 is inserted. I2@idx=2 shares slot
  // with I0. Result: TopIndex=2, slot0(wait=1)={I1}, slot1(wait=0)={I0,I2}.
  EXPECT_EQ(BufSmall.getTopIndex(), 2u);
  EXPECT_THAT(BufSmall.getNthFromEnd(0), UnorderedElementsAre(Instrs[0], Instrs[2]));
  EXPECT_THAT(BufSmall.getNthFromEnd(1), UnorderedElementsAre(Instrs[1]));
}


TEST_F(AMDGPUResourceTrackerTest, InstrBuffer_Merge) {
  StringRef MIRString = R"MIR(
---
name: test
body: |
  bb.0:
    BUFFER_STORE_DWORD_ADDR64 $vgpr4, $vgpr1_vgpr2, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, implicit $exec
    S_CBRANCH_SCC1 %bb.2, implicit $scc

  bb.1:
    BUFFER_STORE_DWORD_ADDR64 $vgpr5, $vgpr1_vgpr2, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 4, 0, 0, implicit $exec
    BUFFER_STORE_DWORD_ADDR64 $vgpr6, $vgpr1_vgpr2, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 8, 0, 0, implicit $exec

  bb.2:
    S_ENDPGM 0
...
)MIR";

  auto *MF = parseMIRGetBB("gfx1200", MIRString)->getParent();
  MachineBasicBlock *MBB0 = MF->getBlockNumbered(0);
  MachineBasicBlock *MBB1 = MF->getBlockNumbered(1);

  auto It = MBB0->begin();
  MachineInstr *MI0_0 = &*It++;

  It = MBB1->begin();
  MachineInstr *MI1_0 = &*It++;
  MachineInstr *MI1_1 = &*It++;

  {
    // Merge [MI1_0, MI1_1] into [MI0_0].
    AMDGPU::InstrBuffer This, Other;
    This.pushBack(MI0_0);

    Other.pushBack(MI1_0);
    Other.pushBack(MI1_1);

    This.merge(Other);

    EXPECT_TRUE(This.contains(MI0_0));
    EXPECT_TRUE(This.contains(MI1_0));
    EXPECT_TRUE(This.contains(MI1_1));

    EXPECT_EQ(This.getIndex(MI0_0), 1u);
    EXPECT_EQ(This.getIndex(MI1_0), 0u);
    EXPECT_EQ(This.getIndex(MI1_1), 1u);

    EXPECT_EQ(This.getTopIndex(), 2u);
  }
  {
    // Same but in reverse: Merge [MI0_0] into [MI1_0, MI1_1].
    AMDGPU::InstrBuffer This, Other;
    This.pushBack(MI1_0);
    This.pushBack(MI1_1);

    Other.pushBack(MI0_0);

    This.merge(Other);

    EXPECT_TRUE(This.contains(MI0_0));
    EXPECT_TRUE(This.contains(MI1_0));
    EXPECT_TRUE(This.contains(MI1_1));

    EXPECT_EQ(This.getIndex(MI0_0), 1u);
    EXPECT_EQ(This.getIndex(MI1_0), 0u);
    EXPECT_EQ(This.getIndex(MI1_1), 1u);

    EXPECT_EQ(This.getTopIndex(), 2u);
  }
  {
    // Handle duplicates.
    // Merging [MI1_0, MI1_1] into [MI1_0] should result in [{MI1_0, MI1_1}].
    AMDGPU::InstrBuffer This, Other;
    This.pushBack(MI1_0);

    Other.pushBack(MI1_0);
    Other.pushBack(MI1_1);

    This.merge(Other);

    EXPECT_TRUE(This.contains(MI1_0));
    EXPECT_TRUE(This.contains(MI1_1));

    // The internal index is 1u but the external is 0 since the bottom internal
    // index has moved to 1.
    EXPECT_EQ(This.getIndex(MI1_0), 0u);
    EXPECT_EQ(This.getIndex(MI1_1), 0u);

    EXPECT_EQ(This.getTopIndex(), 1u);
    // Make sure we have updated the BottomIndex.
    EXPECT_EQ(This.BottomIdxInternal, 1u);
  }
  {
    // Handle duplicates, same as above but in reverse.
    // Merging [MI1_0] into [MI1_0, MI1_1] should result in [{MI1_0, MI1_1}].
    AMDGPU::InstrBuffer This, Other;
    This.pushBack(MI1_0);
    This.pushBack(MI1_1);

    Other.pushBack(MI1_0);

    This.merge(Other);

    EXPECT_TRUE(This.contains(MI1_0));
    EXPECT_TRUE(This.contains(MI1_1));

    // The internal index is 1u but the external is 0 since the bottom internal
    // index has moved to 1.
    EXPECT_EQ(This.getIndex(MI1_0), 0u);
    EXPECT_EQ(This.getIndex(MI1_1), 0u);

    EXPECT_EQ(This.getTopIndex(), 1u);
    // Make sure we have updated the BottomIndex.
    EXPECT_EQ(This.BottomIdxInternal, 1u);
  }
}

TEST_F(AMDGPUResourceTrackerTest, AsyncBuffer_PushBack) {
  StringRef MIRString = R"MIR(
---
name: test
body: |
  bb.0:
    BUFFER_STORE_DWORD_ADDR64 $vgpr5, $vgpr1_vgpr2, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 4, 0, 0, implicit $exec
    BUFFER_STORE_DWORD_ADDR64 $vgpr6, $vgpr1_vgpr2, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 8, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  auto *MF = parseMIRGetBB("gfx1200", MIRString)->getParent();
  MachineBasicBlock *MBB0 = MF->getBlockNumbered(0);
  auto It = MBB0->begin();
  MachineInstr *MI0 = &*It++;
  MachineInstr *MI1 = &*It++;

  {
    // Duplicate should get maximum index push_back MI1_0 into [MI1_0, MI1_1].
    // MI1_0's index should be 2.
    uint64_t SeqNum = 0;
    AMDGPU::AsyncBuffer Buff(SeqNum);
    Buff.pushBack(MI0);
    EXPECT_EQ(Buff.getIndex(MI0), 0u);
    Buff.pushBack(MI1);
    EXPECT_EQ(Buff.getIndex(MI1), 1u);
    Buff.pushBack(MI0);
    EXPECT_EQ(Buff.getIndex(MI0), 2u);
  }
}

TEST_F(AMDGPUResourceTrackerTest, AsyncBuffer_Merge) {
  // bb.0: one async DMA + ASYNCMARK (MI0_0 is marked).
  // bb.1: one unmarked store (MI1_0) then one async DMA + ASYNCMARK (MI1_1 is marked).
  StringRef MIRString = R"MIR(
---
name: test
body: |
  bb.0:
    BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, 1, implicit $exec, implicit $m0
    ASYNCMARK implicit $exec
    S_CBRANCH_SCC1 %bb.2, implicit $scc

  bb.1:
    BUFFER_STORE_DWORD_ADDR64 $vgpr5, $vgpr1_vgpr2, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 4, 0, 0, implicit $exec
    BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, 1, implicit $exec, implicit $m0
    ASYNCMARK implicit $exec

  bb.2:
    S_ENDPGM 0
...
)MIR";

  auto *MF = parseMIRGetBB("gfx900", MIRString)->getParent();
  MachineBasicBlock *MBB0 = MF->getBlockNumbered(0);
  MachineBasicBlock *MBB1 = MF->getBlockNumbered(1);

  // MI0_0: async DMA in bb.0 (marked, followed by ASYNCMARK).
  MachineInstr *MI0_0 = &*MBB0->begin();

  auto It = MBB1->begin();
  MachineInstr *MI1_0 = &*It++; // unmarked store
  MachineInstr *MI1_1 = &*It++; // async DMA (marked, followed by ASYNCMARK)

  {
    // Merge [MI1_0, MI1_1] into [MI0_0].
    uint64_t SeqNum = 0;
    AMDGPU::AsyncBuffer This(SeqNum), Other(SeqNum);
    This.pushBack(MI0_0);

    Other.pushBack(MI1_0);
    Other.pushBack(MI1_1);

    This.merge(Other);

    EXPECT_TRUE(This.contains(MI0_0));
    EXPECT_TRUE(This.contains(MI1_0));
    EXPECT_TRUE(This.contains(MI1_1));

    EXPECT_EQ(This.getIndex(MI0_0), 1u);
    EXPECT_EQ(This.getIndex(MI1_0), 0u);
    EXPECT_EQ(This.getIndex(MI1_1), 1u);

    EXPECT_EQ(This.getTopIndex(), 2u);
  }
  {
    // Check duplicates.
    // Merge [MI1_0] into [MI1_0, MI1_1],
    // Expected result: [MI1_0, {MI1_0,MI1_1}]
    uint64_t SeqNum = 0;
    AMDGPU::AsyncBuffer This(SeqNum), Other(SeqNum);
    This.pushBack(MI1_0);

    Other.pushBack(MI1_0);
    Other.pushBack(MI1_1);

    This.merge(Other);

    EXPECT_TRUE(This.contains(MI1_0));
    EXPECT_TRUE(This.contains(MI1_1));

    EXPECT_EQ(This.getIndex(MI1_0), 1u);
    EXPECT_EQ(This.getIndex(MI1_1), 1u);

    EXPECT_EQ(This.getTopIndex(), 2u);
  }
  {
    // Check that marks are merged.
    // This=[MI0_0(marked, seqnum=0)], Other=[MI1_0(unmarked), MI1_1(marked, seqnum=1)].
    // After merge: [MI1_0, {MI0_0(M,seq=0), MI1_1(M,seq=1)}].
    // Slot 0 (MI1_0) is unmarked; slot 1 ({MI0_0,MI1_1}) is marked.
    // getSeqNumsForIndex(0) is empty, getSeqNumsForIndex(1) is non-empty.
    uint64_t SeqNum = 0;
    AMDGPU::AsyncBuffer This(SeqNum), Other(SeqNum);
    // MI0_0 is followed by ASYNCMARK in bb.0, so isAsyncMarked fires in pushBack.
    This.pushBack(MI0_0); // marked: seqnum=0

    Other.pushBack(MI1_0); // unmarked (MI1_0 not followed by ASYNCMARK)
    Other.pushBack(MI1_1); // marked: seqnum=1

    This.merge(Other);

    EXPECT_TRUE(This.contains(MI0_0));
    EXPECT_TRUE(This.contains(MI1_0));
    EXPECT_TRUE(This.contains(MI1_1));

    EXPECT_EQ(This.getIndex(MI1_0), 0u);
    EXPECT_EQ(This.getIndex(MI0_0), 1u);
    EXPECT_EQ(This.getIndex(MI1_1), 1u);

    // Slot 0 (MI1_0) is unmarked; slot 1 ({MI0_0,MI1_1}) is marked.
    EXPECT_TRUE(This.getSeqNumsForIndex(0).empty());
    EXPECT_FALSE(This.getSeqNumsForIndex(1).empty());

    EXPECT_EQ(This.getTopIndex(), 2u);
  }
  {
    // Check that SeqNums are remapped after merge.
    // This=[MI0_0(marked, seqnum=0)], Other=[MI1_0(marked, seqnum=1), MI1_1(marked, seqnum=2)].
    // After merge (Other is larger), indices shift: This's entry moves by
    // ThisOffset=1. Other's entries shift by OtherOffset=0.
    // Merged: slot0=MI1_0(seqnum=1), slot1={MI0_0(seqnum=0), MI1_1(seqnum=2)}.
    // Both slots are marked. slot0 has seqnum {1}, slot1 has seqnums {0, 2}
    // (two seqnums map to the same slot after merge).
    uint64_t SeqNum = 0;
    AMDGPU::AsyncBuffer This(SeqNum), Other(SeqNum);
    This.pushBack(MI0_0); // marked: seqnum=0
    Other.pushBack(MI1_0); // marked: seqnum=1
    Other.pushBack(MI1_1); // marked: seqnum=2

    This.merge(Other);

    EXPECT_EQ(This.getTopIndex(), 2u);
    // slot0 (MI1_0) is the unmarked store — no seqnum.
    EXPECT_TRUE(This.getSeqNumsForIndex(0).empty());
    // slot1 ({MI0_0(seq=0), MI1_1(seq=1)}) is marked — seqnums present.
    EXPECT_FALSE(This.getSeqNumsForIndex(1).empty());
  }
}

// Test WaitDescriptors sorted insertion and pruning (min-wait dedup).
TEST_F(AMDGPUTestBase, CounterAndWaitVec_Pruning) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $sgpr0_sgpr1, $vgpr0_vgpr1, $vgpr0
    $vgpr5 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $sgpr12 = S_LOAD_DWORD_IMM $sgpr0_sgpr1, 0, 0
    $vgpr6 = DS_READ_B32_gfx9 $vgpr0, 0, 0, implicit $exec, implicit $m0
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";

  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  AMDGPU::ResourceTracker RT(ST, /*AA=*/nullptr,
                             AMDGPU::SchedulingMode::NoExpert);

  auto *MBB = MF->getBlockNumbered(0);
  ASSERT_TRUE(MBB) << "Failed to get BB0";

  SmallVector<MachineInstr *, 4> Instrs;
  for (MachineInstr &MI : *MBB)
    if (!MI.isTerminator())
      Instrs.push_back(&MI);
  ASSERT_EQ(Instrs.size(), 3u);

  MachineInstr *MI0 = Instrs[0]; // GLOBAL_LOAD_DWORD
  MachineInstr *MI1 = Instrs[1]; // S_LOAD_DWORD_IMM
  MachineInstr *MI2 = Instrs[2]; // DS_READ_B32

  AMDGPU::CounterType LoadCntr = AMDGPU::LoadCnt();
  AMDGPU::CounterType KmCntr = AMDGPU::KmCnt();
  AMDGPU::CounterType DsCntr = AMDGPU::DsCnt();

  // Test 1: Entries with distinct counters are all kept, in counter ID order.
  {
    AMDGPU::WaitDescriptors Vec;
    Vec.emplace(MI2, DsCntr, 1);
    Vec.emplace(MI0, LoadCntr, 2);
    Vec.emplace(MI1, KmCntr, 3);
    ASSERT_EQ(Vec.size(), 3u);

    auto It = Vec.begin();
    EXPECT_EQ(It->Cntr, LoadCntr);
    EXPECT_EQ(It->Wait, 2u);
    EXPECT_EQ(It->MI, MI0);
    ++It;
    EXPECT_EQ(It->Cntr, DsCntr);
    EXPECT_EQ(It->Wait, 1u);
    EXPECT_EQ(It->MI, MI2);
    ++It;
    EXPECT_EQ(It->Cntr, KmCntr);
    EXPECT_EQ(It->Wait, 3u);
    EXPECT_EQ(It->MI, MI1);
  }

  // Test 2: Duplicate counter keeps the minimum wait value.
  {
    AMDGPU::WaitDescriptors Vec;
    Vec.emplace(MI0, LoadCntr, 5);
    Vec.emplace(MI1, LoadCntr, 2);
    ASSERT_EQ(Vec.size(), 1u);
    EXPECT_EQ(Vec.begin()->Cntr, LoadCntr);
    EXPECT_EQ(Vec.begin()->Wait, 2u);
    EXPECT_EQ(Vec.begin()->MI, MI1);
  }

  // Test 3: Duplicate counter with higher wait is ignored.
  {
    AMDGPU::WaitDescriptors Vec;
    Vec.emplace(MI0, LoadCntr, 2);
    Vec.emplace(MI1, LoadCntr, 5);
    ASSERT_EQ(Vec.size(), 1u);
    EXPECT_EQ(Vec.begin()->Wait, 2u);
    EXPECT_EQ(Vec.begin()->MI, MI0);
  }

  // Test 4: Multiple duplicates across different counters.
  {
    AMDGPU::WaitDescriptors Vec;
    Vec.emplace(MI0, LoadCntr, 3);
    Vec.emplace(MI1, KmCntr, 4);
    Vec.emplace(MI2, LoadCntr, 1);
    Vec.emplace(MI0, KmCntr, 2);
    ASSERT_EQ(Vec.size(), 2u);

    auto It = Vec.begin();
    EXPECT_EQ(It->Cntr, LoadCntr);
    EXPECT_EQ(It->Wait, 1u);
    EXPECT_EQ(It->MI, MI2);
    ++It;
    EXPECT_EQ(It->Cntr, KmCntr);
    EXPECT_EQ(It->Wait, 2u);
    EXPECT_EQ(It->MI, MI0);
  }

  // Test 5: insert() (not emplace) also deduplicates.
  {
    AMDGPU::WaitDescriptors Vec;
    Vec.insert(AMDGPU::WaitDescriptor(MI0, LoadCntr, 5));
    Vec.insert(AMDGPU::WaitDescriptor(MI1, LoadCntr, 3));
    ASSERT_EQ(Vec.size(), 1u);
    EXPECT_EQ(Vec.begin()->Wait, 3u);
    EXPECT_EQ(Vec.begin()->MI, MI1);
  }

  // Test 6: getCounterAndWait() returns matching entries.
  {
    AMDGPU::WaitDescriptors Vec;
    Vec.emplace(MI0, LoadCntr, 2);
    Vec.emplace(MI1, KmCntr, 4);
    Vec.emplace(MI2, LoadCntr, 1); // updates LoadCnt to wait=1

    ASSERT_EQ(Vec.size(), 2u);
    auto CIt = Vec.begin();
    EXPECT_EQ(CIt->Cntr, AMDGPU::LoadCnt());
    EXPECT_EQ(CIt->Wait, 1u);
    ++CIt;
    EXPECT_EQ(CIt->Cntr, AMDGPU::KmCnt());
    EXPECT_EQ(CIt->Wait, 4u);
  }

  // Test 7: get() finds existing counter.
  {
    AMDGPU::WaitDescriptors Vec;
    Vec.emplace(MI0, LoadCntr, 2);
    Vec.emplace(MI1, KmCntr, 4);

    AMDGPU::WaitDescriptor *Result = Vec.get(LoadCntr);
    ASSERT_TRUE(Result != nullptr);
    EXPECT_EQ(Result->Cntr, LoadCntr);
    EXPECT_EQ(Result->Wait, 2u);
    EXPECT_EQ(Result->MI, MI0);

    Result = Vec.get(KmCntr);
    ASSERT_TRUE(Result != nullptr);
    EXPECT_EQ(Result->Cntr, KmCntr);
    EXPECT_EQ(Result->Wait, 4u);
    EXPECT_EQ(Result->MI, MI1);
  }

  // Test 8: get() returns nullopt for missing counter.
  {
    AMDGPU::WaitDescriptors Vec;
    Vec.emplace(MI0, LoadCntr, 2);

    EXPECT_EQ(Vec.get(KmCntr), nullptr);
    EXPECT_EQ(Vec.get(DsCntr), nullptr);
  }

  // Test 9: get() on empty vec.
  {
    AMDGPU::WaitDescriptors Vec;
    EXPECT_EQ(Vec.get(LoadCntr), nullptr);
  }
}

TEST_F(AMDGPUResourceTrackerTest, AsyncBuffer) {
  StringRef MIRString = R"MIR(
---
name: test
body: |
  bb.0:
    $vgpr0 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $vgpr1 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 4, 0, implicit $exec
    $vgpr2 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 8, 0, implicit $exec
    $vgpr3 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 12, 0, implicit $exec
    $vgpr4 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 16, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1200", MIRString);
  ASSERT_TRUE(MBB);

  SmallVector<MachineInstr *, 5> Instrs;
  for (MachineInstr &MI : *MBB) {
    if (MI.getOpcode() == AMDGPU::GLOBAL_LOAD_DWORD)
      Instrs.push_back(&MI);
  }
  ASSERT_EQ(Instrs.size(), 5u);

  // Shared sequence number for AsyncBuffer construction in tests.
  uint64_t SeqNum = 0;

  // pushBack and getTopIndex: each call creates a new slot.
  {
    AMDGPU::AsyncBuffer Buf(SeqNum);
    EXPECT_EQ(Buf.getTopIndex(), 0u);
    EXPECT_TRUE(Buf.empty());
    Buf.pushBack(Instrs[0]);
    EXPECT_EQ(Buf.getTopIndex(), 1u);
    Buf.pushBack(Instrs[1]);
    Buf.pushBack(Instrs[2]);
    EXPECT_EQ(Buf.getTopIndex(), 3u);
    EXPECT_FALSE(Buf.empty());
  }

  // Duplicate entries: same instruction can appear at multiple positions.
  {
    AMDGPU::AsyncBuffer Buf(SeqNum);
    Buf.pushBack(Instrs[0]);
    Buf.pushBack(Instrs[1]);
    Buf.pushBack(Instrs[0]); // second occurrence of Instrs[0]
    EXPECT_EQ(Buf.getTopIndex(), 3u);
    EXPECT_TRUE(Buf.contains(Instrs[0]));
    EXPECT_TRUE(Buf.contains(Instrs[1]));
    EXPECT_EQ(Buf.numInstrs(), 3u);
  }

  // contains() is O(1) and reflects the ref-count correctly.
  {
    AMDGPU::AsyncBuffer Buf(SeqNum);
    EXPECT_FALSE(Buf.contains(Instrs[0]));
    Buf.pushBack(Instrs[0]);
    EXPECT_TRUE(Buf.contains(Instrs[0]));
    EXPECT_FALSE(Buf.contains(Instrs[1]));
    Buf.pushBack(Instrs[0]); // second occurrence
    EXPECT_TRUE(Buf.contains(Instrs[0]));
    // After popFront removes the first slot, still present in the second.
    Buf.popFront(1);
    EXPECT_TRUE(Buf.contains(Instrs[0]));
    // After popFront removes the second slot, no longer present.
    Buf.popFront(1);
    EXPECT_FALSE(Buf.contains(Instrs[0]));
  }

  // hasUnknown() detects nullptr entries.
  {
    AMDGPU::AsyncBuffer Buf(SeqNum);
    EXPECT_FALSE(Buf.hasUnknown());
    Buf.pushBack(Instrs[0]);
    EXPECT_FALSE(Buf.hasUnknown());
    Buf.pushBack(nullptr); // unknown entry
    EXPECT_TRUE(Buf.hasUnknown());
    Buf.popFront(1); // removes Instrs[0]
    EXPECT_TRUE(Buf.hasUnknown());
    Buf.popFront(1); // removes nullptr
    EXPECT_FALSE(Buf.hasUnknown());
  }

  // popFront removes oldest slots and adjusts getTopIndex.
  {
    AMDGPU::AsyncBuffer Buf(SeqNum);
    Buf.pushBack(Instrs[0]);
    Buf.pushBack(Instrs[1]);
    Buf.pushBack(Instrs[2]);
    Buf.popFront(1);
    EXPECT_EQ(Buf.getTopIndex(), 2u);
    EXPECT_FALSE(Buf.contains(Instrs[0]));
    EXPECT_TRUE(Buf.contains(Instrs[1]));
    EXPECT_TRUE(Buf.contains(Instrs[2]));
    Buf.popFront(2);
    EXPECT_EQ(Buf.getTopIndex(), 0u);
    EXPECT_TRUE(Buf.empty());
  }

  // getNthFromEnd: 0 = most recently pushed, N = Nth-from-end.
  {
    AMDGPU::AsyncBuffer Buf(SeqNum);
    Buf.pushBack(Instrs[0]);
    Buf.pushBack(Instrs[1]);
    Buf.pushBack(Instrs[2]);
    EXPECT_TRUE(Buf.getNthFromEnd(0).contains(AMDGPU::TrackedInstr(Instrs[2])));
    EXPECT_TRUE(Buf.getNthFromEnd(1).contains(AMDGPU::TrackedInstr(Instrs[1])));
    EXPECT_TRUE(Buf.getNthFromEnd(2).contains(AMDGPU::TrackedInstr(Instrs[0])));
    EXPECT_TRUE(Buf.getNthFromEnd(3).empty()); // out of range
  }

  // back() returns the set at the top slot.
  {
    AMDGPU::AsyncBuffer Buf(SeqNum);
    EXPECT_TRUE(Buf.back().empty());
    Buf.pushBack(Instrs[0]);
    EXPECT_TRUE(Buf.back().contains(AMDGPU::TrackedInstr(Instrs[0])));
    Buf.pushBack(Instrs[1]);
    EXPECT_TRUE(Buf.back().contains(AMDGPU::TrackedInstr(Instrs[1])));
    EXPECT_FALSE(Buf.back().contains(AMDGPU::TrackedInstr(Instrs[0])));
  }

  // numInstrs() counts all occurrences across all slots.
  {
    AMDGPU::AsyncBuffer Buf(SeqNum);
    EXPECT_EQ(Buf.numInstrs(), 0u);
    Buf.pushBack(Instrs[0]);
    EXPECT_EQ(Buf.numInstrs(), 1u);
    Buf.pushBack(Instrs[0]); // duplicate
    EXPECT_EQ(Buf.numInstrs(), 2u);
    Buf.pushBack(Instrs[1]);
    EXPECT_EQ(Buf.numInstrs(), 3u);
    Buf.popFront(1);
    EXPECT_EQ(Buf.numInstrs(), 2u);
  }

  // removeIf removes matching instructions from all slots.
  {
    AMDGPU::AsyncBuffer Buf(SeqNum);
    Buf.pushBack(Instrs[0]);
    Buf.pushBack(Instrs[1]);
    Buf.pushBack(Instrs[0]); // second occurrence
    Buf.removeIf([&](const AMDGPU::TrackedInstr &TI) {
      return TI.getMI() == Instrs[0];
    });
    EXPECT_FALSE(Buf.contains(Instrs[0]));
    EXPECT_TRUE(Buf.contains(Instrs[1]));
    EXPECT_EQ(Buf.numInstrs(), 1u);
    // getTopIndex should be trimmed: only the middle slot (Instrs[1]) remains.
    EXPECT_EQ(Buf.getTopIndex(), 1u);
  }

  // clear() empties the buffer completely.
  {
    AMDGPU::AsyncBuffer Buf(SeqNum);
    Buf.pushBack(Instrs[0]);
    Buf.pushBack(Instrs[1]);
    Buf.clear();
    EXPECT_TRUE(Buf.empty());
    EXPECT_EQ(Buf.getTopIndex(), 0u);
    EXPECT_FALSE(Buf.contains(Instrs[0]));
    EXPECT_FALSE(Buf.contains(Instrs[1]));
    EXPECT_EQ(Buf.numInstrs(), 0u);
    // Can push again after clear.
    Buf.pushBack(Instrs[2]);
    EXPECT_EQ(Buf.getTopIndex(), 1u);
    EXPECT_TRUE(Buf.contains(Instrs[2]));
  }

  // Equality: two empty buffers are equal.
  {
    AMDGPU::AsyncBuffer Buf1(SeqNum), Buf2(SeqNum);
    EXPECT_TRUE(Buf1 == Buf2);
    EXPECT_FALSE(Buf1 != Buf2);
  }

  // Equality: same instructions in same order.
  {
    AMDGPU::AsyncBuffer Buf1(SeqNum), Buf2(SeqNum);
    Buf1.pushBack(Instrs[0]);
    Buf1.pushBack(Instrs[1]);
    Buf2.pushBack(Instrs[0]);
    Buf2.pushBack(Instrs[1]);
    EXPECT_TRUE(Buf1 == Buf2);
  }

  // Inequality: different sizes.
  {
    AMDGPU::AsyncBuffer Buf1(SeqNum), Buf2(SeqNum);
    Buf1.pushBack(Instrs[0]);
    Buf2.pushBack(Instrs[0]);
    Buf2.pushBack(Instrs[1]);
    EXPECT_FALSE(Buf1 == Buf2);
    EXPECT_TRUE(Buf1 != Buf2);
  }

  // Inequality: same size, different instructions.
  {
    AMDGPU::AsyncBuffer Buf1(SeqNum), Buf2(SeqNum);
    Buf1.pushBack(Instrs[0]);
    Buf2.pushBack(Instrs[1]);
    EXPECT_FALSE(Buf1 == Buf2);
  }

  // Equality: duplicates must match position-by-position.
  {
    AMDGPU::AsyncBuffer Buf1(SeqNum), Buf2(SeqNum);
    Buf1.pushBack(Instrs[0]);
    Buf1.pushBack(Instrs[0]); // [I0, I0]
    Buf2.pushBack(Instrs[0]);
    Buf2.pushBack(Instrs[0]);
    EXPECT_TRUE(Buf1 == Buf2);
    // [I0, I0] != [I0, I1]
    AMDGPU::AsyncBuffer Buf3(SeqNum), Buf4(SeqNum);
    Buf3.pushBack(Instrs[0]);
    Buf3.pushBack(Instrs[0]);
    Buf4.pushBack(Instrs[0]);
    Buf4.pushBack(Instrs[1]);
    EXPECT_FALSE(Buf3 == Buf4);
  }

  // merge() always uses max semantics: result size = max(A, B), aligned at top.
  {
    AMDGPU::AsyncBuffer BufA(SeqNum), BufB(SeqNum);
    BufA.pushBack(Instrs[0]);
    BufA.pushBack(Instrs[1]);
    BufA.pushBack(Instrs[2]); // size 3
    BufB.pushBack(Instrs[3]);
    BufB.pushBack(Instrs[4]); // size 2
    BufA.merge(BufB);
    EXPECT_EQ(BufA.getTopIndex(), 3u);
    // Top slot (fromEnd=0): {I2, I4}
    auto Top = BufA.getNthFromEnd(0);
    EXPECT_TRUE(Top.contains(AMDGPU::TrackedInstr(Instrs[2])));
    EXPECT_TRUE(Top.contains(AMDGPU::TrackedInstr(Instrs[4])));
    // Second slot (fromEnd=1): {I1, I3}
    auto Mid = BufA.getNthFromEnd(1);
    EXPECT_TRUE(Mid.contains(AMDGPU::TrackedInstr(Instrs[1])));
    EXPECT_TRUE(Mid.contains(AMDGPU::TrackedInstr(Instrs[3])));
    // Bottom slot (fromEnd=2): {I0} only (BufB had no entry here)
    auto Bot = BufA.getNthFromEnd(2);
    EXPECT_TRUE(Bot.contains(AMDGPU::TrackedInstr(Instrs[0])));
    EXPECT_EQ(Bot.size(), 1u);
    // contains() reflects merged state
    EXPECT_TRUE(BufA.contains(Instrs[0]));
    EXPECT_TRUE(BufA.contains(Instrs[3]));
  }

  // merge() always uses max semantics: result size = max(A, B).
  // Same data as the UseMax=true test above: BufA has size 3, BufB has size 2.
  // merge() retains all entries; I0 is kept in the bottom slot.
  {
    AMDGPU::AsyncBuffer BufA(SeqNum), BufB(SeqNum);
    BufA.pushBack(Instrs[0]);
    BufA.pushBack(Instrs[1]);
    BufA.pushBack(Instrs[2]); // size 3
    BufB.pushBack(Instrs[3]);
    BufB.pushBack(Instrs[4]); // size 2
    BufA.merge(BufB);
    EXPECT_EQ(BufA.getTopIndex(), 3u);
    auto Top = BufA.getNthFromEnd(0);
    EXPECT_TRUE(Top.contains(AMDGPU::TrackedInstr(Instrs[2])));
    EXPECT_TRUE(Top.contains(AMDGPU::TrackedInstr(Instrs[4])));
    auto Mid = BufA.getNthFromEnd(1);
    EXPECT_TRUE(Mid.contains(AMDGPU::TrackedInstr(Instrs[1])));
    EXPECT_TRUE(Mid.contains(AMDGPU::TrackedInstr(Instrs[3])));
    // I0 is retained in the bottom slot.
    EXPECT_TRUE(BufA.contains(Instrs[0]));
  }

  // merge with empty other: no change.
  {
    AMDGPU::AsyncBuffer BufA(SeqNum), BufEmpty(SeqNum);
    BufA.pushBack(Instrs[0]);
    BufA.pushBack(Instrs[1]);
    BufA.merge(BufEmpty);
    EXPECT_EQ(BufA.getTopIndex(), 2u);
    EXPECT_TRUE(BufA.contains(Instrs[0]));
    EXPECT_TRUE(BufA.contains(Instrs[1]));
  }

  // merge into empty this (always max): Other's entries are copied in.
  // Simulates building the entry-state accumulator for a block whose first
  // predecessor has marks but the accumulator starts empty.
  {
    AMDGPU::AsyncBuffer BufEmpty(SeqNum), BufOther(SeqNum);
    BufOther.pushBack(Instrs[0]);
    BufOther.pushBack(Instrs[1]);
    BufEmpty.merge(BufOther);
    EXPECT_EQ(BufEmpty.getTopIndex(), 2u);
    EXPECT_TRUE(BufEmpty.contains(Instrs[0]));
    EXPECT_TRUE(BufEmpty.contains(Instrs[1]));
  }

  // iterator covers all slots bottom-to-top.
  {
    AMDGPU::AsyncBuffer Buf(SeqNum);
    Buf.pushBack(Instrs[0]);
    Buf.pushBack(Instrs[1]);
    Buf.pushBack(Instrs[2]);
    SmallVector<SmallDenseSet<AMDGPU::TrackedInstr, 2>> Slots;
    for (const auto &Slot : Buf)
      Slots.push_back(Slot);
    ASSERT_EQ(Slots.size(), 3u);
    EXPECT_TRUE(Slots[0].contains(AMDGPU::TrackedInstr(Instrs[0])));
    EXPECT_TRUE(Slots[1].contains(AMDGPU::TrackedInstr(Instrs[1])));
    EXPECT_TRUE(Slots[2].contains(AMDGPU::TrackedInstr(Instrs[2])));
  }

  // instrsUnordered() returns every TrackedInstr across all slots.
  {
    AMDGPU::AsyncBuffer Buf(SeqNum);
    Buf.pushBack(Instrs[0]);
    Buf.pushBack(Instrs[1]);
    Buf.pushBack(Instrs[0]); // duplicate
    SmallVector<AMDGPU::TrackedInstr> All = Buf.instrsUnordered();
    EXPECT_EQ(All.size(), 3u);
    // Count occurrences of I0 and I1.
    unsigned I0Count = llvm::count_if(
        All, [&](const AMDGPU::TrackedInstr &TI) { return TI.getMI() == Instrs[0]; });
    unsigned I1Count = llvm::count_if(
        All, [&](const AMDGPU::TrackedInstr &TI) { return TI.getMI() == Instrs[1]; });
    EXPECT_EQ(I0Count, 2u);
    EXPECT_EQ(I1Count, 1u);
  }

  // getWaitForNthMarked: tested via the AsyncBuffer_IsAsyncMarked and
  // AsyncCounter_GetNthMostRecentMarkedAmong tests which use real MIR with
  // ASYNCMARK so isAsyncMarked() fires correctly in pushBack().
}

TEST_F(AMDGPUResourceTrackerTest, AsyncBuffer_IsAsyncMarked) {
  // Test isAsyncMarked() on gfx900: only BUFFER_LOAD_*_LDS with IsAsync=1
  // (not regular VMEM loads) followed by ASYNCMARK should return true.
  StringRef MIRString = R"MIR(
---
name: test
machineFunctionInfo:
  isEntryFunction: true
body: |
  bb.0:
    liveins: $vgpr0_vgpr1, $sgpr0_sgpr1_sgpr2_sgpr3
    ; Regular VMEM load followed by ASYNCMARK — code-motion barrier only.
    $vgpr0 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    ASYNCMARK implicit $exec
    ; Async LDS DMA load (IsAsync=1) followed by ASYNCMARK — real async mark.
    BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, 1, implicit $exec, implicit $m0
    ASYNCMARK implicit $exec
    ; Async LDS DMA load NOT followed by ASYNCMARK.
    BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 4, 0, 0, 1, implicit $exec, implicit $m0
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx900", MIRString);
  ASSERT_TRUE(MBB);

  SmallVector<MachineInstr *> AllInstrs;
  for (MachineInstr &MI : *MBB)
    AllInstrs.push_back(&MI);

  // Find the instructions by opcode.
  MachineInstr *GlobalLoad = nullptr, *AsyncLoad1 = nullptr,
               *AsyncLoad2 = nullptr;
  for (MachineInstr *MI : AllInstrs) {
    if (MI->getOpcode() == AMDGPU::GLOBAL_LOAD_DWORD)
      GlobalLoad = MI;
    else if (MI->getOpcode() == AMDGPU::BUFFER_LOAD_DWORD_LDS_OFFEN) {
      if (!AsyncLoad1)
        AsyncLoad1 = MI;
      else
        AsyncLoad2 = MI;
    }
  }
  ASSERT_TRUE(GlobalLoad);
  ASSERT_TRUE(AsyncLoad1);
  ASSERT_TRUE(AsyncLoad2);

  // Regular VMEM load followed by ASYNCMARK: NOT an async DMA candidate.
  EXPECT_FALSE(AMDGPU::AsyncBuffer::isAsyncMarked(GlobalLoad));

  // Async LDS DMA (IsAsync=1) followed by ASYNCMARK: IS a candidate.
  EXPECT_TRUE(AMDGPU::AsyncBuffer::isAsyncMarked(AsyncLoad1));

  // Async LDS DMA NOT followed by ASYNCMARK: NOT marked.
  EXPECT_FALSE(AMDGPU::AsyncBuffer::isAsyncMarked(AsyncLoad2));

  // nullptr: NOT marked.
  EXPECT_FALSE(AMDGPU::AsyncBuffer::isAsyncMarked(nullptr));
}

TEST_F(AMDGPUResourceTrackerTest, AsyncBuffer_SeqNum) {
  // Use gfx900 MIR: three async DMA loads each followed by ASYNCMARK.
  // Load0, Load1, Load2 are all marked. GlobalSeqNum increments only for
  // marked pushes, so seqnums are 0, 1, 2 respectively.
  StringRef MIRString = R"MIR(
---
name: test
machineFunctionInfo:
  isEntryFunction: true
body: |
  bb.0:
    liveins: $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3
    BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, 1, implicit $exec, implicit $m0
    ASYNCMARK implicit $exec
    BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 4, 0, 0, 1, implicit $exec, implicit $m0
    ASYNCMARK implicit $exec
    BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 8, 0, 0, 1, implicit $exec, implicit $m0
    ASYNCMARK implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx900", MIRString);
  ASSERT_TRUE(MBB);
  SmallVector<MachineInstr *, 3> Instrs;
  for (MachineInstr &MI : *MBB)
    if (MI.getOpcode() == AMDGPU::BUFFER_LOAD_DWORD_LDS_OFFEN)
      Instrs.push_back(&MI);
  ASSERT_EQ(Instrs.size(), 3u);

  uint64_t SeqNum = 0;
  AMDGPU::AsyncBuffer Buf(SeqNum);

  // Seqnums are only assigned to marked slots (DMA followed by ASYNCMARK).
  // In straight-line code each slot has exactly one seqnum.
  Buf.pushBack(Instrs[0]); // slot 0, marked → seqnum 0
  EXPECT_THAT(Buf.getSeqNumsForIndex(0), UnorderedElementsAre(0u));
  Buf.pushBack(Instrs[1]); // slot 1, marked → seqnum 1
  EXPECT_THAT(Buf.getSeqNumsForIndex(1), UnorderedElementsAre(1u));
  Buf.pushBack(Instrs[2]); // slot 2, marked → seqnum 2
  EXPECT_THAT(Buf.getSeqNumsForIndex(2), UnorderedElementsAre(2u));
  EXPECT_EQ(SeqNum, 3u);

  // Re-pushing the same instruction creates a new slot with a new seqnum.
  Buf.pushBack(Instrs[0]); // slot 3, marked → seqnum 3
  EXPECT_THAT(Buf.getSeqNumsForIndex(3), UnorderedElementsAre(3u));
  // Original slot 0 still has its old seqnum.
  EXPECT_THAT(Buf.getSeqNumsForIndex(0), UnorderedElementsAre(0u));
  EXPECT_EQ(SeqNum, 4u);

  // Two AsyncBuffers sharing the same GlobalSeqNum produce ordered timestamps.
  uint64_t SharedSeqNum = 0;
  AMDGPU::AsyncBuffer BufA(SharedSeqNum);
  AMDGPU::AsyncBuffer BufB(SharedSeqNum);
  BufA.pushBack(Instrs[0]); // slot 0, seqnum 0
  BufB.pushBack(Instrs[1]); // slot 0, seqnum 1
  BufA.pushBack(Instrs[2]); // slot 1, seqnum 2
  EXPECT_LT(*BufA.getSeqNumsForIndex(0).begin(),
            *BufB.getSeqNumsForIndex(0).begin());
  EXPECT_LT(*BufB.getSeqNumsForIndex(0).begin(),
            *BufA.getSeqNumsForIndex(1).begin());
  EXPECT_EQ(SharedSeqNum, 3u);
}

TEST_F(AMDGPUResourceTrackerTest, AsyncCounter_GetNthMostRecentMarkedAmong) {
  // Use gfx900 MIR: Load0 (marked) → ASYNCMARK → Load1 (unmarked) →
  //                 Load2 (marked) → ASYNCMARK
  // isAsyncMarked() fires automatically in pushBack() for loads followed by
  // ASYNCMARK, so MarkedInstrs is populated correctly.
  StringRef MIRString = R"MIR(
---
name: test
machineFunctionInfo:
  isEntryFunction: true
body: |
  bb.0:
    liveins: $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3
    BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, 1, implicit $exec, implicit $m0
    ASYNCMARK implicit $exec
    BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 4, 0, 0, 1, implicit $exec, implicit $m0
    BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 8, 0, 0, 1, implicit $exec, implicit $m0
    ASYNCMARK implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx900", MIRString);
  ASSERT_TRUE(MBB);

  SmallVector<MachineInstr *> AsyncLoads;
  for (MachineInstr &MI : *MBB)
    if (MI.getOpcode() == AMDGPU::BUFFER_LOAD_DWORD_LDS_OFFEN)
      AsyncLoads.push_back(&MI);
  ASSERT_EQ(AsyncLoads.size(), 3u);
  // AsyncLoads[0] = Load0 (followed by ASYNCMARK → marked)
  // AsyncLoads[1] = Load1 (not followed by ASYNCMARK → unmarked)
  // AsyncLoads[2] = Load2 (followed by ASYNCMARK → marked)

  // Single counter: verify ordering by seqnum.
  {
    uint64_t SeqNum = 0;
    AMDGPU::AsyncCounter C(AMDGPU::VmCnt(), /*MaxSize=*/0, SeqNum);
    C.insert({AsyncLoads[0]}); // seqnum=0, marked
    C.insert({AsyncLoads[1]}); // seqnum=1, unmarked
    C.insert({AsyncLoads[2]}); // seqnum=2, marked

    SmallVector<AMDGPU::AsyncCounter *> Counters = {&C};
    // N=0: most recently marked = Load2 (seqnum=2, fromEnd=0).
    // Returns {&C, 0}: counter C, index_in_counter=0 (newest marked slot).
    auto R0 = AMDGPU::AsyncCounter::getNthMostRecentMarkedAmong(Counters, 0);
    ASSERT_TRUE(!R0.empty());
    EXPECT_EQ(R0[0].first, &C);
    EXPECT_EQ(R0[0].second, 0u); // fromEnd=0: newest marked entry (Load2)
    // N=1: Load0 (seqnum=0, fromEnd=2). Load1 is NOT marked.
    auto R1 = AMDGPU::AsyncCounter::getNthMostRecentMarkedAmong(Counters, 1);
    ASSERT_TRUE(!R1.empty());
    EXPECT_EQ(R1[0].first, &C);
    EXPECT_EQ(R1[0].second, 2u); // fromEnd=2: oldest marked entry (Load0)
    // N=2: no more marked entries
    EXPECT_TRUE(
        AMDGPU::AsyncCounter::getNthMostRecentMarkedAmong(Counters, 2)
            .empty());;
  }

  // Two counters sharing GlobalSeqNum: cross-counter ordering by seqnum.
  {
    uint64_t SeqNum = 0;
    AMDGPU::AsyncCounter C1(AMDGPU::VmCnt(), /*MaxSize=*/0, SeqNum);
    AMDGPU::AsyncCounter C2(AMDGPU::VmCnt(), /*MaxSize=*/0, SeqNum);
    C1.insert({AsyncLoads[0]}); // seqnum=0, marked (Load0 → ASYNCMARK)
    C2.insert({AsyncLoads[1]}); // seqnum=1, NOT marked (next is Load2, same type)
    C1.insert({AsyncLoads[2]}); // seqnum=2, marked (Load2 → ASYNCMARK)

    SmallVector<AMDGPU::AsyncCounter *> Counters = {&C1, &C2};
    // N=0: most recently marked across both = Load2 in C1 (seqnum=2, fromEnd=0).
    auto R0 = AMDGPU::AsyncCounter::getNthMostRecentMarkedAmong(Counters, 0);
    ASSERT_TRUE(!R0.empty());
    EXPECT_EQ(R0[0].first, &C1);
    EXPECT_EQ(R0[0].second, 0u); // fromEnd=0 in C1: Load2 (newest)
    // N=1: Load0 in C1 (seqnum=0, fromEnd=1). C1 has 2 entries:
    // Load2@slot1(fromEnd=0) and Load0@slot0(fromEnd=1). Load1 is NOT marked.
    auto R1 = AMDGPU::AsyncCounter::getNthMostRecentMarkedAmong(Counters, 1);
    ASSERT_TRUE(!R1.empty());
    EXPECT_EQ(R1[0].first, &C1);
    EXPECT_EQ(R1[0].second, 1u); // fromEnd=1 in C1: Load0 (oldest marked)
    // N=2: no more marked entries
    EXPECT_TRUE(
        AMDGPU::AsyncCounter::getNthMostRecentMarkedAmong(Counters, 2)
            .empty());;
  }
}

// Test getNthMostRecentMarkedAmong when the Nth entry has seqnum=0.
// This exercises the bug where the skip condition `Seq <= BestSeq` (with
// BestSeq initialized to 0) incorrectly skips entries with seqnum=0.
// Setup: Load0 (seqnum=0, marked) and Load1 (seqnum=1, marked).
// N=1 (2nd most recent) should find Load0 (seqnum=0), not return nullopt.
TEST_F(AMDGPUResourceTrackerTest, AsyncCounter_GetNthMostRecentMarked_SeqNum0) {
  StringRef MIRString = R"MIR(
---
name: test
machineFunctionInfo:
  isEntryFunction: true
body: |
  bb.0:
    liveins: $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3
    BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 0, 0, 0, 1, implicit $exec, implicit $m0
    ASYNCMARK implicit $exec
    BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr0, $sgpr0_sgpr1_sgpr2_sgpr3, 0, 4, 0, 0, 1, implicit $exec, implicit $m0
    ASYNCMARK implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx900", MIRString);
  ASSERT_TRUE(MBB);

  SmallVector<MachineInstr *> AsyncLoads;
  for (MachineInstr &MI : *MBB)
    if (MI.getOpcode() == AMDGPU::BUFFER_LOAD_DWORD_LDS_OFFEN)
      AsyncLoads.push_back(&MI);
  ASSERT_EQ(AsyncLoads.size(), 2u);

  uint64_t SeqNum = 0;
  AMDGPU::AsyncCounter C(AMDGPU::VmCnt(), /*MaxSize=*/0, SeqNum);
  C.insert({AsyncLoads[0]}); // seqnum=0, marked (followed by ASYNCMARK)
  C.insert({AsyncLoads[1]}); // seqnum=1, marked (followed by ASYNCMARK)

  // Verify seqnums: Load0=0, Load1=1.
  ASSERT_EQ(SeqNum, 2u);

  SmallVector<AMDGPU::AsyncCounter *> Counters = {&C};

  // N=0: most recently marked = Load1 (seqnum=1).
  auto R0 = AMDGPU::AsyncCounter::getNthMostRecentMarkedAmong(Counters, 0);
  ASSERT_TRUE(!R0.empty());
  EXPECT_EQ(R0[0].first, &C);
  EXPECT_EQ(R0[0].second, 0u); // fromEnd=0: newest (Load1)

  // N=1: 2nd most recently marked = Load0 (seqnum=0).
  // This exercises the bug where seqnum=0 was incorrectly skipped.
  auto R1 = AMDGPU::AsyncCounter::getNthMostRecentMarkedAmong(Counters, 1);
  ASSERT_TRUE(!R1.empty()) << "seqnum=0 entry must not be skipped";
  EXPECT_EQ(R1[0].first, &C);
  EXPECT_EQ(R1[0].second, 1u); // fromEnd=1: oldest (Load0)

  // N=2: no more marked entries.
  EXPECT_TRUE(
      AMDGPU::AsyncCounter::getNthMostRecentMarkedAmong(Counters, 2)
          .empty());;
}

// Test AsyncBuffer::getAsyncMarkFor(MI):
// - Returns the ASYNCMARK immediately after an async DMA candidate.
// - Skips async DMA instructions of a different counter type.
// - Stops (unreachable in practice; only called on marked MIs) at same-type.
TEST_F(AMDGPUResourceTrackerTest, AsyncBuffer_GetAsyncMarkFor) {
  StringRef MIRString = R"MIR(
---
name: test
machineFunctionInfo:
  isEntryFunction: true
body: |
  bb.0:
    liveins: $vgpr2, $vgpr0_vgpr1, $sgpr16_sgpr17_sgpr18_sgpr19, $sgpr20_sgpr21_sgpr22_sgpr23_sgpr24_sgpr25_sgpr26_sgpr27
    ; Async then ASYNCMARK — getAsyncMarkFor(Async) returns this ASYNCMARK.
    GLOBAL_LOAD_ASYNC_TO_LDS_B32 $vgpr2, $vgpr0_vgpr1, 0, 0, implicit-def dead $asynccnt, implicit $exec, implicit $asynccnt
    ASYNCMARK implicit $exec
    ; Tensor then ASYNCMARK — getAsyncMarkFor(Tensor) returns this ASYNCMARK.
    TENSOR_LOAD_TO_LDS_d2 $sgpr16_sgpr17_sgpr18_sgpr19, $sgpr20_sgpr21_sgpr22_sgpr23_sgpr24_sgpr25_sgpr26_sgpr27, 0, 0, implicit-def dead $tensorcnt, implicit $exec, implicit $tensorcnt
    ASYNCMARK implicit $exec
    ; Async, then Tensor (different type), then ASYNCMARK —
    ; getAsyncMarkFor(Async) skips Tensor and returns this ASYNCMARK.
    GLOBAL_LOAD_ASYNC_TO_LDS_B32 $vgpr2, $vgpr0_vgpr1, 0, 0, implicit-def dead $asynccnt, implicit $exec, implicit $asynccnt
    TENSOR_LOAD_TO_LDS_d2 $sgpr16_sgpr17_sgpr18_sgpr19, $sgpr20_sgpr21_sgpr22_sgpr23_sgpr24_sgpr25_sgpr26_sgpr27, 0, 0, implicit-def dead $tensorcnt, implicit $exec, implicit $tensorcnt
    ASYNCMARK implicit $exec
    ; Async, then a non-async instruction (S_MOV_B32), then ASYNCMARK —
    ; getAsyncMarkFor skips non-async instructions and returns the ASYNCMARK.
    GLOBAL_LOAD_ASYNC_TO_LDS_B32 $vgpr2, $vgpr0_vgpr1, 0, 0, implicit-def dead $asynccnt, implicit $exec, implicit $asynccnt
    $sgpr4 = S_MOV_B32 0
    ASYNCMARK implicit $exec
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1250", MIRString);
  ASSERT_TRUE(MBB);

  SmallVector<MachineInstr *> Asyncs, Tensors, Marks;
  for (MachineInstr &MI : *MBB) {
    if (MI.getOpcode() == AMDGPU::GLOBAL_LOAD_ASYNC_TO_LDS_B32)
      Asyncs.push_back(&MI);
    else if (MI.getOpcode() == AMDGPU::TENSOR_LOAD_TO_LDS_d2)
      Tensors.push_back(&MI);
    else if (MI.getOpcode() == AMDGPU::ASYNCMARK)
      Marks.push_back(&MI);
  }
  ASSERT_EQ(Asyncs.size(), 3u);
  ASSERT_EQ(Tensors.size(), 2u);
  ASSERT_EQ(Marks.size(), 4u);

  // Async[0] → Mark[0]: ASYNCMARK immediately follows.
  EXPECT_EQ(AMDGPU::AsyncBuffer::getAsyncMarkFor(Asyncs[0]), Marks[0]);
  // Tensor[0] → Mark[1]: ASYNCMARK immediately follows.
  EXPECT_EQ(AMDGPU::AsyncBuffer::getAsyncMarkFor(Tensors[0]), Marks[1]);
  // Tensor[1] → Mark[2]: Async[1] (different counter type) is skipped.
  EXPECT_EQ(AMDGPU::AsyncBuffer::getAsyncMarkFor(Tensors[1]), Marks[2]);
  // Async[1] → Mark[2]: Tensor[1] (different counter type) is skipped.
  EXPECT_EQ(AMDGPU::AsyncBuffer::getAsyncMarkFor(Asyncs[1]), Marks[2]);
  // Async[2] → Mark[3]: non-async instruction (V_MOV_B32) in between is skipped.
  EXPECT_EQ(AMDGPU::AsyncBuffer::getAsyncMarkFor(Asyncs[2]), Marks[3]);
}

TEST_F(AMDGPUResourceTrackerTest, AsyncBuffer_IsAsyncMarked_MultiCounter) {
  // Test that isAsyncMarked correctly identifies both an async load and a
  // tensor load when ASYNCMARK follows only the tensor load:
  //
  //   GLOBAL_LOAD_ASYNC_TO_LDS  ← goes into AsyncCnt, NOT immediately before ASYNCMARK
  //   TENSOR_LOAD_TO_LDS        ← goes into TensorCnt, immediately before ASYNCMARK
  //   ASYNCMARK
  //
  // Bug: isAsyncMarked(GLOBAL_LOAD_ASYNC) = false (next is TENSOR, not ASYNCMARK).
  // The ASYNCMARK should mark the most recent instruction in EACH async counter,
  // not just the instruction immediately before it.
  StringRef MIRString = R"MIR(
---
name: test
machineFunctionInfo:
  isEntryFunction: true
body: |
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3, $vgpr0, $vgpr1, $sgpr16_sgpr17_sgpr18_sgpr19_sgpr20_sgpr21_sgpr22_sgpr23
    GLOBAL_LOAD_ASYNC_TO_LDS_B32 $vgpr1, $vgpr0_vgpr1, 0, 0, implicit-def dead $asynccnt, implicit $exec, implicit $asynccnt
    TENSOR_LOAD_TO_LDS_d2 $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr16_sgpr17_sgpr18_sgpr19_sgpr20_sgpr21_sgpr22_sgpr23, 0, 0, implicit-def dead $tensorcnt, implicit $exec, implicit $tensorcnt
    ASYNCMARK
    S_ENDPGM 0
...
)MIR";

  MachineBasicBlock *MBB = parseMIRGetBB("gfx1250", MIRString);
  ASSERT_TRUE(MBB);

  MachineInstr *AsyncLoad = nullptr, *TensorLoad = nullptr,
               *AsyncMark = nullptr;
  for (MachineInstr &MI : *MBB) {
    if (MI.getOpcode() == AMDGPU::GLOBAL_LOAD_ASYNC_TO_LDS_B32)
      AsyncLoad = &MI;
    else if (MI.getOpcode() == AMDGPU::TENSOR_LOAD_TO_LDS_d2)
      TensorLoad = &MI;
    else if (MI.getOpcode() == AMDGPU::ASYNCMARK)
      AsyncMark = &MI;
  }
  ASSERT_TRUE(AsyncLoad);
  ASSERT_TRUE(TensorLoad);
  ASSERT_TRUE(AsyncMark);

  // Pattern: Async → Tensor → ASYNCMARK
  // TensorLoad is immediately before ASYNCMARK → marked.
  EXPECT_TRUE(AMDGPU::AsyncBuffer::isAsyncMarked(TensorLoad));
  // AsyncLoad has TensorLoad (another async DMA) then ASYNCMARK → also marked.
  EXPECT_TRUE(AMDGPU::AsyncBuffer::isAsyncMarked(AsyncLoad));

  // Pattern: ASYNCMARK itself → not an async DMA → not marked.
  EXPECT_FALSE(AMDGPU::AsyncBuffer::isAsyncMarked(AsyncMark));

  // Pattern: Tensor → Async → ASYNCMARK (reversed order).
  StringRef MIRString2 = R"MIR(
---
name: test
machineFunctionInfo:
  isEntryFunction: true
body: |
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3, $vgpr0, $vgpr1, $sgpr16_sgpr17_sgpr18_sgpr19_sgpr20_sgpr21_sgpr22_sgpr23
    TENSOR_LOAD_TO_LDS_d2 $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr16_sgpr17_sgpr18_sgpr19_sgpr20_sgpr21_sgpr22_sgpr23, 0, 0, implicit-def dead $tensorcnt, implicit $exec, implicit $tensorcnt
    GLOBAL_LOAD_ASYNC_TO_LDS_B32 $vgpr1, $vgpr0_vgpr1, 0, 0, implicit-def dead $asynccnt, implicit $exec, implicit $asynccnt
    ASYNCMARK
    S_ENDPGM 0
...
)MIR";
  MachineBasicBlock *MBB2 = parseMIRGetBB("gfx1250", MIRString2);
  ASSERT_TRUE(MBB2);
  MachineInstr *TensorLoad2 = nullptr, *AsyncLoad2 = nullptr;
  for (MachineInstr &MI : *MBB2) {
    if (MI.getOpcode() == AMDGPU::TENSOR_LOAD_TO_LDS_d2)
      TensorLoad2 = &MI;
    else if (MI.getOpcode() == AMDGPU::GLOBAL_LOAD_ASYNC_TO_LDS_B32)
      AsyncLoad2 = &MI;
  }
  ASSERT_TRUE(TensorLoad2);
  ASSERT_TRUE(AsyncLoad2);
  // AsyncLoad2 is immediately before ASYNCMARK → marked.
  EXPECT_TRUE(AMDGPU::AsyncBuffer::isAsyncMarked(AsyncLoad2));
  // TensorLoad2 has AsyncLoad2 (async DMA) then ASYNCMARK → also marked.
  EXPECT_TRUE(AMDGPU::AsyncBuffer::isAsyncMarked(TensorLoad2));

  // Pattern: nullptr → not marked.
  EXPECT_FALSE(AMDGPU::AsyncBuffer::isAsyncMarked(nullptr));
}
