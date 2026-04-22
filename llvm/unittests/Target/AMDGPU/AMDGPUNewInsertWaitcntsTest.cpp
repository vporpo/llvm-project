//===-- AMDGPUNewInsertWaitcntsTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUNewInsertWaitcnts.h"
#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnitTests.h"
#include "GCNSubtarget.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;
using ::testing::Not;
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

TEST_F(AMDGPUTestBase, PromoteSoftWaitInstrs_PreGFX12) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1030", "");
  ASSERT_TRUE(TM) << "No target machine";

  // Pre-GFX12 soft wait instructions: S_WAITCNT_soft, S_WAITCNT_VSCNT_soft
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $vgpr2
    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    S_WAITCNT_soft 0
    S_WAITCNT_VSCNT_soft undef $sgpr_null, 0
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

  // Find the soft wait instructions before promotion.
  MachineInstr *WaitcntSoft = nullptr;
  MachineInstr *VscntSoft = nullptr;
  for (MachineInstr &MI : *MBB) {
    if (MI.getOpcode() == AMDGPU::S_WAITCNT_soft)
      WaitcntSoft = &MI;
    else if (MI.getOpcode() == AMDGPU::S_WAITCNT_VSCNT_soft)
      VscntSoft = &MI;
  }
  ASSERT_TRUE(WaitcntSoft) << "S_WAITCNT_soft not found";
  ASSERT_TRUE(VscntSoft) << "S_WAITCNT_VSCNT_soft not found";

  // Set up InsertWaitcnt and call promoteSoftWaitInstrs directly.
  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  AMDGPU::InsertWaitcnt IW;
  IW.TII = ST->getInstrInfo();
  bool Changed = IW.promoteSoftWaitInstrs(*MBB);

  EXPECT_TRUE(Changed) << "promoteSoftWaitInstrs should return true";
  EXPECT_EQ(WaitcntSoft->getOpcode(), AMDGPU::S_WAITCNT)
      << "S_WAITCNT_soft should be promoted to S_WAITCNT";
  EXPECT_EQ(VscntSoft->getOpcode(), AMDGPU::S_WAITCNT_VSCNT)
      << "S_WAITCNT_VSCNT_soft should be promoted to S_WAITCNT_VSCNT";
}

TEST_F(AMDGPUTestBase, PromoteSoftWaitInstrs_GFX12) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  // GFX12+ soft wait instructions
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $vgpr2
    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    S_WAITCNT_soft 0
    S_WAIT_LOADCNT_soft 0
    S_WAIT_STORECNT_soft 0
    S_WAIT_SAMPLECNT_soft 0
    S_WAIT_BVHCNT_soft 0
    S_WAIT_DSCNT_soft 0
    S_WAIT_KMCNT_soft 0
    S_WAIT_XCNT_soft 0
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

  // Find the soft wait instructions before promotion.
  MachineInstr *WaitcntSoft = nullptr;
  MachineInstr *LoadcntSoft = nullptr;
  MachineInstr *StorecntSoft = nullptr;
  MachineInstr *SamplecntSoft = nullptr;
  MachineInstr *BvhcntSoft = nullptr;
  MachineInstr *DscntSoft = nullptr;
  MachineInstr *KmcntSoft = nullptr;
  MachineInstr *XcntSoft = nullptr;
  for (MachineInstr &MI : *MBB) {
    switch (MI.getOpcode()) {
    case AMDGPU::S_WAITCNT_soft:
      WaitcntSoft = &MI;
      break;
    case AMDGPU::S_WAIT_LOADCNT_soft:
      LoadcntSoft = &MI;
      break;
    case AMDGPU::S_WAIT_STORECNT_soft:
      StorecntSoft = &MI;
      break;
    case AMDGPU::S_WAIT_SAMPLECNT_soft:
      SamplecntSoft = &MI;
      break;
    case AMDGPU::S_WAIT_BVHCNT_soft:
      BvhcntSoft = &MI;
      break;
    case AMDGPU::S_WAIT_DSCNT_soft:
      DscntSoft = &MI;
      break;
    case AMDGPU::S_WAIT_KMCNT_soft:
      KmcntSoft = &MI;
      break;
    case AMDGPU::S_WAIT_XCNT_soft:
      XcntSoft = &MI;
      break;
    }
  }
  ASSERT_TRUE(WaitcntSoft) << "S_WAITCNT_soft not found";
  ASSERT_TRUE(LoadcntSoft) << "S_WAIT_LOADCNT_soft not found";
  ASSERT_TRUE(StorecntSoft) << "S_WAIT_STORECNT_soft not found";
  ASSERT_TRUE(SamplecntSoft) << "S_WAIT_SAMPLECNT_soft not found";
  ASSERT_TRUE(BvhcntSoft) << "S_WAIT_BVHCNT_soft not found";
  ASSERT_TRUE(DscntSoft) << "S_WAIT_DSCNT_soft not found";
  ASSERT_TRUE(KmcntSoft) << "S_WAIT_KMCNT_soft not found";
  ASSERT_TRUE(XcntSoft) << "S_WAIT_XCNT_soft not found";

  // Set up InsertWaitcnt and call promoteSoftWaitInstrs directly.
  const GCNSubtarget *ST = &MF->getSubtarget<GCNSubtarget>();
  AMDGPU::InsertWaitcnt IW;
  IW.TII = ST->getInstrInfo();
  bool Changed = IW.promoteSoftWaitInstrs(*MBB);

  EXPECT_TRUE(Changed) << "promoteSoftWaitInstrs should return true";
  EXPECT_EQ(WaitcntSoft->getOpcode(), AMDGPU::S_WAITCNT)
      << "S_WAITCNT_soft should be promoted to S_WAITCNT";
  EXPECT_EQ(LoadcntSoft->getOpcode(), AMDGPU::S_WAIT_LOADCNT)
      << "S_WAIT_LOADCNT_soft should be promoted to S_WAIT_LOADCNT";
  EXPECT_EQ(StorecntSoft->getOpcode(), AMDGPU::S_WAIT_STORECNT)
      << "S_WAIT_STORECNT_soft should be promoted to S_WAIT_STORECNT";
  EXPECT_EQ(SamplecntSoft->getOpcode(), AMDGPU::S_WAIT_SAMPLECNT)
      << "S_WAIT_SAMPLECNT_soft should be promoted to S_WAIT_SAMPLECNT";
  EXPECT_EQ(BvhcntSoft->getOpcode(), AMDGPU::S_WAIT_BVHCNT)
      << "S_WAIT_BVHCNT_soft should be promoted to S_WAIT_BVHCNT";
  EXPECT_EQ(DscntSoft->getOpcode(), AMDGPU::S_WAIT_DSCNT)
      << "S_WAIT_DSCNT_soft should be promoted to S_WAIT_DSCNT";
  EXPECT_EQ(KmcntSoft->getOpcode(), AMDGPU::S_WAIT_KMCNT)
      << "S_WAIT_KMCNT_soft should be promoted to S_WAIT_KMCNT";
  // With the soft-xcnt fence optimization enabled (the default), soft xcnt fences
  // are deliberately left soft so the block walk can decide per fence whether a
  // later data-counter wait makes them redundant; the walk promotes any it keeps.
  EXPECT_EQ(XcntSoft->getOpcode(), AMDGPU::S_WAIT_XCNT_soft)
      << "S_WAIT_XCNT_soft should be left soft for the fence optimization";
}

TEST_F(AMDGPUTestBase, EmitWaitInstr_PreGFX12) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1030", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0
    $vgpr1 = V_MOV_B32_e32 0, implicit $exec
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
  AMDGPU::InsertWaitcnt IW;
  IW.ST = ST;
  IW.TII = ST->getInstrInfo();

  AMDGPU::IsaVersion IV = AMDGPU::getIsaVersion(ST->getCPU());

  // Helper to check wait instruction opcode and decoded counter values.
  // For S_WAITCNT, decodes the immediate and checks each counter.
  // For S_WAITCNT_VSCNT (SOPK format), checks operand 1.
  auto CheckWait = [&](MachineInstr *WaitMI, unsigned ExpectedOpcode,
                       std::initializer_list<
                           std::pair<AMDGPU::InstCounterType, unsigned>>
                           Expected) -> bool {
    if (!WaitMI || WaitMI->getOpcode() != ExpectedOpcode)
      return false;
    if (ExpectedOpcode == AMDGPU::S_WAITCNT) {
      AMDGPU::Waitcnt Decoded =
          AMDGPU::decodeWaitcnt(IV, WaitMI->getOperand(0).getImm());
      for (auto [CT, Val] : Expected)
        if (Decoded.get(CT) != Val)
          return false;
    } else if (ExpectedOpcode == AMDGPU::S_WAITCNT_VSCNT) {
      for (auto [CT, Val] : Expected)
        if (WaitMI->getOperand(1).getImm() != (int64_t)Val)
          return false;
    }
    WaitMI->eraseFromParent();
    return true;
  };

  // Test 1: VmCnt only
  {
    AMDGPU::WaitDescriptors CAWs = {{AMDGPU::VmCnt(), 3}};
    auto WaitRange = IW.emitWaitInstr(*MBB, MBB->instr_begin(), CAWs);
    ASSERT_FALSE(WaitRange.empty());
    EXPECT_TRUE(
        CheckWait(&*WaitRange.begin(), AMDGPU::S_WAITCNT, {{AMDGPU::LOAD_CNT, 3u}}));
  }

  // Test 2: ExpCnt only
  {
    AMDGPU::WaitDescriptors CAWs = {{AMDGPU::ExpCnt(), 2}};
    auto WaitRange = IW.emitWaitInstr(*MBB, MBB->instr_begin(), CAWs);
    ASSERT_FALSE(WaitRange.empty());
    EXPECT_TRUE(
        CheckWait(&*WaitRange.begin(), AMDGPU::S_WAITCNT, {{AMDGPU::EXP_CNT, 2u}}));
  }

  // Test 3: LgkmCnt only
  {
    AMDGPU::WaitDescriptors CAWs = {{AMDGPU::LgkmCnt(), 5}};
    auto WaitRange = IW.emitWaitInstr(*MBB, MBB->instr_begin(), CAWs);
    ASSERT_FALSE(WaitRange.empty());
    EXPECT_TRUE(
        CheckWait(&*WaitRange.begin(), AMDGPU::S_WAITCNT, {{AMDGPU::DS_CNT, 5u}}));
  }

  // Test 4: VsCnt only
  {
    AMDGPU::WaitDescriptors CAWs = {{AMDGPU::VsCnt(), 7}};
    auto WaitRange = IW.emitWaitInstr(*MBB, MBB->instr_begin(), CAWs);
    ASSERT_FALSE(WaitRange.empty());
    EXPECT_TRUE(CheckWait(&*WaitRange.begin(), AMDGPU::S_WAITCNT_VSCNT,
                          {{AMDGPU::STORE_CNT, 7u}}));
  }

  // Test 5: VmCnt + ExpCnt + LgkmCnt
  {
    AMDGPU::WaitDescriptors CAWs = {
        {AMDGPU::VmCnt(), 1},
        {AMDGPU::ExpCnt(), 2},
        {AMDGPU::LgkmCnt(), 3},
    };
    auto WaitRange = IW.emitWaitInstr(*MBB, MBB->instr_begin(), CAWs);
    ASSERT_FALSE(WaitRange.empty());
    EXPECT_TRUE(CheckWait(
        &*WaitRange.begin(), AMDGPU::S_WAITCNT,
        {{AMDGPU::LOAD_CNT, 1u}, {AMDGPU::EXP_CNT, 2u}, {AMDGPU::DS_CNT, 3u}}));
  }

  // Test 6: VmCnt + LgkmCnt + VsCnt (both S_WAITCNT and S_WAITCNT_VSCNT)
  {
    AMDGPU::WaitDescriptors CAWs = {
        {AMDGPU::VmCnt(), 0},
        {AMDGPU::LgkmCnt(), 0},
        {AMDGPU::VsCnt(), 0},
    };
    MachineBasicBlock::instr_iterator InsertPt = MBB->instr_begin();
    auto WaitRange = IW.emitWaitInstr(*MBB, InsertPt, CAWs);

    // Should emit two instructions
    ASSERT_FALSE(WaitRange.empty());
    auto It = WaitRange.begin();
    MachineInstr *FirstMI = &*It++;
    ASSERT_NE(It, WaitRange.end());
    MachineInstr *SecondMI = &*It;

    // S_WAITCNT should be first, S_WAITCNT_VSCNT second
    EXPECT_EQ(FirstMI->getOpcode(), AMDGPU::S_WAITCNT);
    EXPECT_EQ(SecondMI->getOpcode(), AMDGPU::S_WAITCNT_VSCNT);

    FirstMI->eraseFromParent();
    SecondMI->eraseFromParent();
  }

  // Test 7: Empty CAWs (no instruction emitted)
  {
    AMDGPU::WaitDescriptors CAWs = {};
    MachineBasicBlock::instr_iterator InsertPt = MBB->instr_begin();
    auto WaitRange = IW.emitWaitInstr(*MBB, InsertPt, CAWs);

    EXPECT_TRUE(WaitRange.empty())
        << "Empty CAWs should not emit any instruction";
  }
}

TEST_F(AMDGPUTestBase, EmitWaitInstr_GFX12) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0
    $vgpr1 = V_MOV_B32_e32 0, implicit $exec
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
  AMDGPU::InsertWaitcnt IW;
  IW.ST = ST;
  IW.TII = ST->getInstrInfo();

  // Helper to check GFX12 SOPK-format wait instructions (operand 0 is value).
  auto CheckWait = [&](MachineInstr *WaitMI, unsigned ExpectedOpcode,
                       unsigned ExpectedVal) -> bool {
    if (!WaitMI || WaitMI->getOpcode() != ExpectedOpcode)
      return false;
    if (WaitMI->getOperand(0).getImm() != (int64_t)ExpectedVal)
      return false;
    WaitMI->eraseFromParent();
    return true;
  };

  // Test 1: LoadCnt only
  {
    AMDGPU::WaitDescriptors CAWs = {{AMDGPU::LoadCnt(), 3}};
    auto WaitRange = IW.emitWaitInstr(*MBB, MBB->instr_begin(), CAWs);
    ASSERT_FALSE(WaitRange.empty());
    EXPECT_TRUE(CheckWait(&*WaitRange.begin(), AMDGPU::S_WAIT_LOADCNT, 3));
  }

  // Test 2: StoreCnt only
  {
    AMDGPU::WaitDescriptors CAWs = {{AMDGPU::StoreCnt(), 2}};
    auto WaitRange = IW.emitWaitInstr(*MBB, MBB->instr_begin(), CAWs);
    ASSERT_FALSE(WaitRange.empty());
    EXPECT_TRUE(CheckWait(&*WaitRange.begin(), AMDGPU::S_WAIT_STORECNT, 2));
  }

  // Test 3: DsCnt only
  {
    AMDGPU::WaitDescriptors CAWs = {{AMDGPU::DsCnt(), 4}};
    auto WaitRange = IW.emitWaitInstr(*MBB, MBB->instr_begin(), CAWs);
    ASSERT_FALSE(WaitRange.empty());
    EXPECT_TRUE(CheckWait(&*WaitRange.begin(), AMDGPU::S_WAIT_DSCNT, 4));
  }

  // Test 4: ExpCnt only
  {
    AMDGPU::WaitDescriptors CAWs = {{AMDGPU::ExpCnt(), 1}};
    auto WaitRange = IW.emitWaitInstr(*MBB, MBB->instr_begin(), CAWs);
    ASSERT_FALSE(WaitRange.empty());
    EXPECT_TRUE(CheckWait(&*WaitRange.begin(), AMDGPU::S_WAIT_EXPCNT, 1));
  }

  // Test 5: SampleCnt only
  {
    AMDGPU::WaitDescriptors CAWs = {{AMDGPU::SampleCnt(), 5}};
    auto WaitRange = IW.emitWaitInstr(*MBB, MBB->instr_begin(), CAWs);
    ASSERT_FALSE(WaitRange.empty());
    EXPECT_TRUE(CheckWait(&*WaitRange.begin(), AMDGPU::S_WAIT_SAMPLECNT, 5));
  }

  // Test 6: BvhCnt only
  {
    AMDGPU::WaitDescriptors CAWs = {{AMDGPU::BvhCnt(), 6}};
    auto WaitRange = IW.emitWaitInstr(*MBB, MBB->instr_begin(), CAWs);
    ASSERT_FALSE(WaitRange.empty());
    EXPECT_TRUE(CheckWait(&*WaitRange.begin(), AMDGPU::S_WAIT_BVHCNT, 6));
  }

  // Test 7: KmCnt only
  {
    AMDGPU::WaitDescriptors CAWs = {{AMDGPU::KmCnt(), 7}};
    auto WaitRange = IW.emitWaitInstr(*MBB, MBB->instr_begin(), CAWs);
    ASSERT_FALSE(WaitRange.empty());
    EXPECT_TRUE(CheckWait(&*WaitRange.begin(), AMDGPU::S_WAIT_KMCNT, 7));
  }

  // Test 8: LoadCnt + DsCnt (combined S_WAIT_LOADCNT_DSCNT)
  {
    AMDGPU::WaitDescriptors CAWs = {
        {AMDGPU::LoadCnt(), 1},
        {AMDGPU::DsCnt(), 2},
    };
    MachineBasicBlock::instr_iterator InsertPt = MBB->instr_begin();
    auto WaitRange = IW.emitWaitInstr(*MBB, InsertPt, CAWs);

    ASSERT_FALSE(WaitRange.empty());
    EXPECT_EQ(WaitRange.begin()->getOpcode(), AMDGPU::S_WAIT_LOADCNT_DSCNT);

    WaitRange.begin()->eraseFromParent();
  }

  // Test 9: StoreCnt + DsCnt (combined S_WAIT_STORECNT_DSCNT)
  {
    AMDGPU::WaitDescriptors CAWs = {
        {AMDGPU::StoreCnt(), 1},
        {AMDGPU::DsCnt(), 2},
    };
    MachineBasicBlock::instr_iterator InsertPt = MBB->instr_begin();
    auto WaitRange = IW.emitWaitInstr(*MBB, InsertPt, CAWs);

    ASSERT_FALSE(WaitRange.empty());
    EXPECT_EQ(WaitRange.begin()->getOpcode(), AMDGPU::S_WAIT_STORECNT_DSCNT);

    WaitRange.begin()->eraseFromParent();
  }

  // Test 10: LoadCnt + StoreCnt + DsCnt (LoadCnt+DsCnt combined, StoreCnt separate)
  {
    AMDGPU::WaitDescriptors CAWs = {
        {AMDGPU::LoadCnt(), 1},
        {AMDGPU::StoreCnt(), 2},
        {AMDGPU::DsCnt(), 3},
    };
    MachineBasicBlock::instr_iterator InsertPt = MBB->instr_begin();
    auto WaitRange = IW.emitWaitInstr(*MBB, InsertPt, CAWs);

    ASSERT_FALSE(WaitRange.empty());

    // Should emit S_WAIT_LOADCNT_DSCNT and S_WAIT_STORECNT
    bool FoundLoadDsCnt = false;
    bool FoundStoreCnt = false;
    for (auto I = MBB->instr_begin(); I != InsertPt; ++I) {
      if (I->getOpcode() == AMDGPU::S_WAIT_LOADCNT_DSCNT)
        FoundLoadDsCnt = true;
      if (I->getOpcode() == AMDGPU::S_WAIT_STORECNT)
        FoundStoreCnt = true;
    }
    EXPECT_TRUE(FoundLoadDsCnt);
    EXPECT_TRUE(FoundStoreCnt);

    while (MBB->instr_begin() != InsertPt)
      MBB->instr_begin()->eraseFromParent();
  }

  // Test 11: Empty CAWs (no instruction emitted)
  {
    AMDGPU::WaitDescriptors CAWs = {};
    MachineBasicBlock::instr_iterator InsertPt = MBB->instr_begin();
    auto WaitRange = IW.emitWaitInstr(*MBB, InsertPt, CAWs);

    EXPECT_TRUE(WaitRange.empty())
        << "Empty CAWs should not emit any instruction";
  }

  // Test 12: Multiple individual counters (LoadCnt + StoreCnt + KmCnt)
  {
    AMDGPU::WaitDescriptors CAWs = {
        {AMDGPU::LoadCnt(), 1},
        {AMDGPU::StoreCnt(), 2},
        {AMDGPU::KmCnt(), 3},
    };
    MachineBasicBlock::instr_iterator InsertPt = MBB->instr_begin();
    auto WaitRange = IW.emitWaitInstr(*MBB, InsertPt, CAWs);

    ASSERT_FALSE(WaitRange.empty());

    // Count the emitted wait instructions
    int Count = 0;
    for (auto I = MBB->instr_begin(); I != InsertPt; ++I) {
      if (SIInstrInfo::isWaitcnt(I->getOpcode()))
        ++Count;
    }
    EXPECT_EQ(Count, 3) << "Should emit 3 individual wait instructions";

    while (MBB->instr_begin() != InsertPt)
      MBB->instr_begin()->eraseFromParent();
  }
}

TEST_F(AMDGPUTestBase, ImpliesXcntSync) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1250", "");
  ASSERT_TRUE(TM) << "No target machine";

  // Test VMEM followed by SMEM, SMEM followed by VMEM, and same-type sequences.
  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0_vgpr1, $vgpr2_vgpr3, $sgpr0_sgpr1, $sgpr2_sgpr3
    $vgpr10 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    $sgpr10 = S_LOAD_DWORD_IMM $sgpr0_sgpr1, 0, 0
    $vgpr11 = GLOBAL_LOAD_DWORD $vgpr2_vgpr3, 0, 0, implicit $exec
    $sgpr11 = S_LOAD_DWORD_IMM $sgpr2_sgpr3, 0, 0
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
  ASSERT_TRUE(ST->hasWaitXcnt()) << "gfx1250 should have XCnt";

  uint64_t TestSeqNum = 0;
  AMDGPU::ResourceTracker RTracker(ST, nullptr,
                                   AMDGPU::SchedulingMode::NoExpert);

  // Find the instructions.
  MachineInstr *VmemLoad1 = nullptr;
  MachineInstr *SmemLoad1 = nullptr;
  MachineInstr *VmemLoad2 = nullptr;
  MachineInstr *SmemLoad2 = nullptr;
  for (MachineInstr &MI : *MBB) {
    if (MI.getOpcode() == AMDGPU::GLOBAL_LOAD_DWORD) {
      if (!VmemLoad1)
        VmemLoad1 = &MI;
      else
        VmemLoad2 = &MI;
    } else if (MI.getOpcode() == AMDGPU::S_LOAD_DWORD_IMM) {
      if (!SmemLoad1)
        SmemLoad1 = &MI;
      else
        SmemLoad2 = &MI;
    }
  }
  ASSERT_TRUE(VmemLoad1 && SmemLoad1 && VmemLoad2 && SmemLoad2);

  // Initially, no pending instructions - no sync needed.
  EXPECT_FALSE(RTracker.impliesXcntSync(*VmemLoad1))
      << "No pending instrs, no sync needed";

  // Track VMEM load 1.
  RTracker.track(*VmemLoad1);

  // SMEM after VMEM - sync needed.
  EXPECT_TRUE(RTracker.impliesXcntSync(*SmemLoad1))
      << "SMEM after pending VMEM should need sync";

  // Process the implicit sync (clear XCnt).
  RTracker.drainCounters({{AMDGPU::XCnt(), 0}});

  // Track SMEM load 1.
  RTracker.track(*SmemLoad1);

  // VMEM after SMEM - sync needed.
  EXPECT_TRUE(RTracker.impliesXcntSync(*VmemLoad2))
      << "VMEM after pending SMEM should need sync";

  // Process the implicit sync.
  RTracker.drainCounters({{AMDGPU::XCnt(), 0}});

  // Track VMEM load 2.
  RTracker.track(*VmemLoad2);

  // SMEM after VMEM - sync needed again.
  EXPECT_TRUE(RTracker.impliesXcntSync(*SmemLoad2))
      << "SMEM after pending VMEM should need sync";

  // Test same-type sequence (no sync needed).
  RTracker.drainCounters({{AMDGPU::XCnt(), 0}});
  RTracker.track(*VmemLoad1);

  // VMEM after VMEM - no sync needed.
  EXPECT_FALSE(RTracker.impliesXcntSync(*VmemLoad2))
      << "VMEM after pending VMEM should not need sync";
}

TEST_F(AMDGPUTestBase, DecodeWaitMI_PreGFX12) {
  // gfx1030: pre-GFX12 (packs Vm/Exp/Lgkm into one S_WAITCNT) but has a separate
  // VsCnt counter (S_WAITCNT_VSCNT), so it exercises both wait forms.
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1030", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0
    $vgpr1 = V_MOV_B32_e32 0, implicit $exec
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
  AMDGPU::InsertWaitcnt IW;
  IW.ST = ST;
  IW.TII = ST->getInstrInfo();

  // Builds a wait via emitWaitInstr, decodes it back, then erases it.
  auto RoundTrip =
      [&](const AMDGPU::WaitDescriptors &CAWs) -> AMDGPU::WaitDescriptors {
    auto WaitRange = IW.emitWaitInstr(*MBB, MBB->instr_begin(), CAWs);
    AMDGPU::WaitDescriptors Decoded;
    for (MachineInstr &WaitMI : WaitRange)
      for (const auto &CAW :
           AMDGPU::InsertWaitcnt::getCountersAndWaits(WaitMI, *ST))
        Decoded.emplace(CAW.Cntr, CAW.Wait);
    for (MachineInstr &WaitMI : make_early_inc_range(WaitRange))
      WaitMI.eraseFromParent();
    return Decoded;
  };

  // A combined S_WAITCNT with a single counter set must decode to exactly that
  // counter - not to all of Vmcnt/Expcnt/Lgkmcnt. The unset fields are encoded
  // at their bit-mask ("no wait") and must be omitted.
  {
    auto D = RoundTrip({{AMDGPU::LgkmCnt(), 0}});
    EXPECT_THAT(D, ElementsAre(AMDGPU::WaitDescriptor{AMDGPU::LgkmCnt(), 0}));
  }
  {
    auto D = RoundTrip({{AMDGPU::VmCnt(), 0}});
    EXPECT_THAT(D, ElementsAre(AMDGPU::WaitDescriptor{AMDGPU::VmCnt(), 0}));
  }
  {
    auto D = RoundTrip({{AMDGPU::ExpCnt(), 0}});
    EXPECT_THAT(D, ElementsAre(AMDGPU::WaitDescriptor{AMDGPU::ExpCnt(), 0}));
  }

  // A non-zero wait value is preserved, and unset counters are still omitted.
  {
    auto D = RoundTrip({{AMDGPU::VmCnt(), 3}});
    EXPECT_THAT(D, ElementsAre(AMDGPU::WaitDescriptor{AMDGPU::VmCnt(), 3}));
  }

  // Multiple counters set: all are decoded.
  {
    auto D = RoundTrip(
        {{AMDGPU::VmCnt(), 1}, {AMDGPU::ExpCnt(), 2}, {AMDGPU::LgkmCnt(), 3}});
    EXPECT_THAT(D,
                testing::UnorderedElementsAre(
                    AMDGPU::WaitDescriptor{AMDGPU::VmCnt(), 1},
                    AMDGPU::WaitDescriptor{AMDGPU::ExpCnt(), 2},
                    AMDGPU::WaitDescriptor{AMDGPU::LgkmCnt(), 3}));
  }

  // VsCnt is a separate S_WAITCNT_VSCNT instruction (single counter).
  {
    auto D = RoundTrip({{AMDGPU::VsCnt(), 0}});
    EXPECT_THAT(D, ElementsAre(AMDGPU::WaitDescriptor{AMDGPU::VsCnt(), 0}));
  }
}

TEST_F(AMDGPUTestBase, DecodeWaitMI_GFX12) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0
    $vgpr1 = V_MOV_B32_e32 0, implicit $exec
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
  AMDGPU::InsertWaitcnt IW;
  IW.ST = ST;
  IW.TII = ST->getInstrInfo();

  auto RoundTrip =
      [&](const AMDGPU::WaitDescriptors &CAWs) -> AMDGPU::WaitDescriptors {
    auto WaitRange = IW.emitWaitInstr(*MBB, MBB->instr_begin(), CAWs);
    AMDGPU::WaitDescriptors Decoded;
    for (MachineInstr &WaitMI : WaitRange)
      for (const auto &CAW : AMDGPU::InsertWaitcnt::getCountersAndWaits(WaitMI, *ST))
        Decoded.emplace(CAW.Cntr, CAW.Wait);
    for (MachineInstr &WaitMI : make_early_inc_range(WaitRange))
      WaitMI.eraseFromParent();
    return Decoded;
  };

  // On GFX12+ each counter is its own single-counter wait instruction, so a
  // single counter decodes to exactly itself.
  {
    auto D = RoundTrip({{AMDGPU::LoadCnt(), 0}});
    EXPECT_THAT(D, ElementsAre(AMDGPU::WaitDescriptor{AMDGPU::LoadCnt(), 0}));
  }
  {
    auto D = RoundTrip({{AMDGPU::KmCnt(), 5}});
    EXPECT_THAT(D, ElementsAre(AMDGPU::WaitDescriptor{AMDGPU::KmCnt(), 5}));
  }
  {
    auto D = RoundTrip({{AMDGPU::StoreCnt(), 0}});
    EXPECT_THAT(D, ElementsAre(AMDGPU::WaitDescriptor{AMDGPU::StoreCnt(), 0}));
  }

  // S_WAITCNT_DEPCTR decodes to VaVdst and/or VmVsrc.
  IW.SchedMode = AMDGPU::SchedulingMode::ExpertMode2;
  {
    auto D = RoundTrip({{AMDGPU::VaVdst(), 0}});
    EXPECT_THAT(D, ElementsAre(AMDGPU::WaitDescriptor{AMDGPU::VaVdst(), 0}));
  }
  {
    auto D = RoundTrip({{AMDGPU::VmVsrc(), 0}});
    EXPECT_THAT(D, ElementsAre(AMDGPU::WaitDescriptor{AMDGPU::VmVsrc(), 0}));
  }
}

TEST_F(AMDGPUTestBase, DecodeWaitMI_GFX1250) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1250", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $vgpr0
    $vgpr1 = V_MOV_B32_e32 0, implicit $exec
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
  AMDGPU::InsertWaitcnt IW;
  IW.ST = ST;
  IW.TII = ST->getInstrInfo();

  auto RoundTrip =
      [&](const AMDGPU::WaitDescriptors &CAWs) -> AMDGPU::WaitDescriptors {
    auto WaitRange = IW.emitWaitInstr(*MBB, MBB->instr_begin(), CAWs);
    AMDGPU::WaitDescriptors Decoded;
    for (MachineInstr &WaitMI : WaitRange)
      for (const auto &CAW :
           AMDGPU::InsertWaitcnt::getCountersAndWaits(WaitMI, *ST))
        Decoded.emplace(CAW.Cntr, CAW.Wait);
    for (MachineInstr &WaitMI : make_early_inc_range(WaitRange))
      WaitMI.eraseFromParent();
    return Decoded;
  };

  // S_WAIT_ASYNCCNT decodes to AsyncCnt.
  {
    auto D = RoundTrip({{AMDGPU::AsyncCnt(), 3}});
    EXPECT_THAT(D, ElementsAre(AMDGPU::WaitDescriptor{AMDGPU::AsyncCnt(), 3}));
  }
  // S_WAIT_TENSORCNT decodes to TensorCnt.
  {
    auto D = RoundTrip({{AMDGPU::TensorCnt(), 3}});
    EXPECT_THAT(D, ElementsAre(AMDGPU::WaitDescriptor{AMDGPU::TensorCnt(), 3}));
  }
}

// ---------------------------------------------------------------------------
// Tests for getEmittedWaits() via CollectEmittedWaits=true.
// ---------------------------------------------------------------------------

TEST_F(AMDGPUTestBase, GetEmittedWaits_SmemThenVmem) {
  // GFX12: SMEM load followed by VMEM load using the SMEM result as address.
  // InsertWaitcnt must insert a kmcnt(0) wait between them.
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-amdhsa"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $sgpr0_sgpr1, $vgpr0_vgpr1
    $sgpr4_sgpr5_sgpr6_sgpr7_sgpr8_sgpr9_sgpr10_sgpr11 = S_LOAD_DWORDX8_IMM $sgpr0_sgpr1, 0, 0
    $vgpr2 = GLOBAL_LOAD_DWORD $vgpr0_vgpr1, 0, 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";
  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  TargetLibraryInfoImpl TLIImpl(TM->getTargetTriple());
  TargetLibraryInfo TLI(TLIImpl);
  llvm::AAResults AA(TLI);
  MachineDominatorTree MDT(*MF);
  MachineLoopInfo MLI(MDT);

  AMDGPU::InsertWaitcnt IW(/*CollectEmittedWaits=*/true);
  IW.run(*MF, MLI, AA);

  SmallVector<MachineInstr *> Waits(IW.getEmittedWaits());
  ASSERT_FALSE(Waits.empty()) << "Expected at least one emitted wait";

  // Collect all wait instructions inserted into the function in program order.
  SmallVector<MachineInstr *> InsertedWaits;
  for (MachineBasicBlock &MBB : *MF)
    for (MachineInstr &MI : MBB)
      if (SIInstrInfo::isWaitcnt(MI.getOpcode()))
        InsertedWaits.push_back(&MI);

  EXPECT_THAT(Waits, UnorderedElementsAreArray(InsertedWaits));
}

TEST_F(AMDGPUTestBase, GetEmittedWaits_PreexistingEntryWaits_PreGFX12) {
  // Pre-GFX12: non-entry function with a pre-existing S_WAITCNT 0 at entry.
  // The pass should not emit additional entry waits since all counters are
  // already covered, so getEmittedWaits() must not include any entry-block
  // waits.
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-amdhsa"), "gfx908", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
machineFunctionInfo:
  isEntryFunction: false
body:             |
  bb.0:
    S_WAITCNT 0
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";
  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  TargetLibraryInfoImpl TLIImpl(TM->getTargetTriple());
  TargetLibraryInfo TLI(TLIImpl);
  llvm::AAResults AA(TLI);
  MachineDominatorTree MDT(*MF);
  MachineLoopInfo MLI(MDT);

  AMDGPU::InsertWaitcnt IW(/*CollectEmittedWaits=*/true);
  IW.run(*MF, MLI, AA);

  // No new waits should have been emitted; pre-existing S_WAITCNT 0 covers all
  // pre-GFX12 counters.
  SmallVector<MachineInstr *> Waits(IW.getEmittedWaits());
  EXPECT_THAT(Waits, IsEmpty());
}

TEST_F(AMDGPUTestBase, GetEmittedWaits_PreexistingEntryWaits_GFX12) {
  // GFX12: non-entry function with pre-existing GFX12-native waits covering
  // LoadCnt+DsCnt and KmCnt. The pass should only emit the remaining counters
  // (ExpCnt, SampleCnt, BvhCnt), so getEmittedWaits() must contain exactly
  // those.
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-amdhsa"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
machineFunctionInfo:
  isEntryFunction: false
body:             |
  bb.0:
    S_WAIT_LOADCNT_DSCNT 0
    S_WAIT_KMCNT 0
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";
  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  TargetLibraryInfoImpl TLIImpl(TM->getTargetTriple());
  TargetLibraryInfo TLI(TLIImpl);
  llvm::AAResults AA(TLI);
  MachineDominatorTree MDT(*MF);
  MachineLoopInfo MLI(MDT);

  AMDGPU::InsertWaitcnt IW(/*CollectEmittedWaits=*/true);
  IW.run(*MF, MLI, AA);

  // Collect the opcodes of newly emitted waits.
  SmallVector<unsigned> EmittedOpcodes;
  for (MachineInstr *MI : IW.getEmittedWaits())
    EmittedOpcodes.push_back(MI->getOpcode());

  // LoadCnt+DsCnt and KmCnt were pre-existing; only ExpCnt, SampleCnt, BvhCnt
  // should be newly emitted.
  EXPECT_THAT(EmittedOpcodes,
              testing::UnorderedElementsAre(AMDGPU::S_WAIT_EXPCNT,
                                            AMDGPU::S_WAIT_SAMPLECNT,
                                            AMDGPU::S_WAIT_BVHCNT));
}

TEST_F(AMDGPUTestBase, GetEmittedWaits_CombinedWaits) {
  // GFX12: SMEM load followed by a DS store using the SMEM result, then a
  // VMEM load using the DS address. The pass inserts adjacent S_WAIT_KMCNT and
  // S_WAIT_DSCNT waits which combineWaitInstrs() folds into a single
  // S_WAIT_LOADCNT_DSCNT. getEmittedWaits() must return the combined
  // instruction, not the originals that were erased.
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-amdhsa"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
body:             |
  bb.0:
    liveins: $sgpr0_sgpr1, $vgpr0_vgpr1, $vgpr2
    $sgpr4 = S_LOAD_DWORD_IMM $sgpr0_sgpr1, 0, 0
    DS_WRITE_B32 $vgpr0, $vgpr2, 0, 0, implicit $m0, implicit $exec
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

  TargetLibraryInfoImpl TLIImpl(TM->getTargetTriple());
  TargetLibraryInfo TLI(TLIImpl);
  llvm::AAResults AA(TLI);
  MachineDominatorTree MDT(*MF);
  MachineLoopInfo MLI(MDT);

  AMDGPU::InsertWaitcnt IW(/*CollectEmittedWaits=*/true);
  IW.run(*MF, MLI, AA);

  // getEmittedWaits() must exactly match the wait instructions present in
  // the final MIR — combined instructions replace their originals.
  SmallVector<MachineInstr *> Waits(IW.getEmittedWaits());
  SmallVector<MachineInstr *> InsertedWaits;
  for (MachineBasicBlock &MBB : *MF)
    for (MachineInstr &MI : MBB)
      if (SIInstrInfo::isWaitcnt(MI.getOpcode()))
        InsertedWaits.push_back(&MI);

  EXPECT_THAT(Waits, UnorderedElementsAreArray(InsertedWaits));
}

TEST_F(AMDGPUTestBase, GetEmittedWaits_NoWaitsForPureVALU) {
  // Pure VALU: InsertWaitcnt should emit no waits.
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-amdhsa"), "gfx1200", "");
  ASSERT_TRUE(TM) << "No target machine";

  StringRef MIRString = R"MIR(
---
name:            test
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    $vgpr0 = V_MOV_B32_e32 1, implicit $exec
    $vgpr1 = V_MOV_B32_e32 2, implicit $exec
    $vgpr2 = V_ADD_F32_e32 $vgpr0, $vgpr1, implicit $mode, implicit $exec
    S_ENDPGM 0
...
)MIR";

  LLVMContext Context;
  MachineModuleInfo MMI(TM.get());
  auto M = parseMIR(Context, *TM, MIRString, "test", MMI);
  ASSERT_TRUE(M) << "Failed to parse MIR";
  auto *MF = MMI.getMachineFunction(*M->getFunction("test"));
  ASSERT_TRUE(MF) << "Failed to get MachineFunction";

  TargetLibraryInfoImpl TLIImpl(TM->getTargetTriple());
  TargetLibraryInfo TLI(TLIImpl);
  llvm::AAResults AA(TLI);
  MachineDominatorTree MDT(*MF);
  MachineLoopInfo MLI(MDT);

  AMDGPU::InsertWaitcnt IW(/*CollectEmittedWaits=*/true);
  IW.run(*MF, MLI, AA);

  EXPECT_TRUE(IW.getEmittedWaits().empty());
}
