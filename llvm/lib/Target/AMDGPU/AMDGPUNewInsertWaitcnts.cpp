//===- AMDGPUNewInsertWaitcnts.cpp - New Wait Instruction Insertion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Insert wait instructions.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUNewInsertWaitcnts.h"
#include "AMDGPU.h"
#include "AMDGPUWaitcntUtils.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace llvm::AMDGPU;
using RegAccessType = ResourceTracker::RegAccessType;

#define DEBUG_TYPE "amdgpu-new-insert-waitcnts"

// Defined in SIInsertWaitcnts.cpp; shared between both passes.
extern cl::opt<bool> ExpertSchedulingModeFlag;

static cl::opt<bool>
    EnableDataflow("amdgpu-new-insert-waitcnts-dataflow",
                   cl::desc("Enable cross-block dataflow analysis for "
                            "wait insertion"),
                   cl::init(true), cl::Hidden);

// Split combined waits to allow partial hoisting to preheader. This is for
// compatibility with the old pass behavior, not for performance - splitting
// waits doesn't reduce the number of wait instructions in the loop.
static cl::opt<bool> EnableSplitWaitHoist(
    "amdgpu-new-insert-waitcnts-split-wait-hoist",
    cl::desc("Split combined waits to allow partial hoisting to preheader"),
    cl::init(true), cl::Hidden);

// Hoist the SMEM (KmCnt) wait for a loop-invariant scalar load into the loop
// preheader. This is a valid optimization (the loop has no SMEM to re-arm
// KmCnt, so one preheader wait covers every iteration), but the old pass never
// flushes KmCnt in a preheader - it only flushes VmCnt/DsCnt. Off by default to
// match the old pass; enable to drop the redundant in-loop wait.
static cl::opt<bool> EnableKmCntPreheaderFlush(
    "amdgpu-new-insert-waitcnts-kmcnt-preheader-flush",
    cl::desc(
        "Hoist a loop-invariant SMEM (KmCnt) wait into the loop preheader"),
    cl::init(false), cl::Hidden);

// Hoist the pre-gfx12 SMEM/DS (LgkmCnt) wait for a loop-invariant operation
// into the loop preheader. The old pass never hoists LgkmCnt, but the hoist is
// valid when the loop doesn't re-increment the counter.
static cl::opt<bool> EnableLgkmCntPreheaderFlush(
    "amdgpu-new-insert-waitcnts-lgkmcnt-preheader-flush",
    cl::desc("Hoist a loop-invariant LgkmCnt wait into the loop preheader"),
    cl::init(false), cl::Hidden);

static cl::opt<bool> EnableDepctrPreheaderFlush(
    "amdgpu-new-insert-waitcnts-depctr-preheader-flush",
    cl::desc("Hoist loop-invariant VaVdst/VmVsrc depctr waits to preheader"),
    cl::init(false), cl::Hidden);

static cl::opt<bool>
    WaitHoistOpt("amdgpu-new-insert-waitcnts-wait-hoist-opt",
                 cl::desc("In Expert Sched 2 try to hoist to avoid blocking "
                          "the co-execution window"),
                 cl::init(false), cl::Hidden);

// Preserve explicit (non-soft) waits already present in the input verbatim -
// e.g. a user s_waitcnt intrinsic - excluding them from redundancy removal, as
// the original pass does. Disable to also simplify such explicit waits.
static cl::opt<bool> KeepExplicitWaits(
    "amdgpu-new-insert-waitcnts-keep-explicit-waits",
    cl::desc("Preserve explicit (non-soft) input waits, excluding them from "
             "redundant-wait removal"),
    cl::init(true), cl::Hidden);

// Restrict XCnt fence simplification to soft (memory-legalizer) xcnt waits,
// matching the original pass which only simplifies soft/required waits. When
// false, a pass-inserted xcnt wait is also dropped when a later consuming op's
// data wait provably drains the same address translations.
// Defined in SIInsertWaitcnts.cpp; shared between both passes.
extern cl::opt<bool> ForceEmitZeroLoadFlag;
// Defined in SIInsertWaitcnts.cpp; forces all waits to zero, used by
// expand-waitcnt-profiling to exercise the staircase expansion path.
extern cl::opt<bool> ForceEmitZeroFlag;

static cl::opt<bool> SimplifyOnlySoftXCnt(
    "amdgpu-new-insert-waitcnts-simplify-only-soft-xcnt",
    cl::desc("Only simplify soft (memory-legalizer) xcnt fences, not "
             "pass-inserted xcnt waits"),
    cl::init(true), cl::Hidden);

// On GFX11+, send a DEALLOC_VGPRS message (or S_ALLOC_VGPR 0 in dynamic VGPR
// mode) before an S_ENDPGM that may complete with outstanding stores, releasing
// the VGPRs early instead of waiting for the stores. Matches the original pass.
static cl::opt<bool> EnableDeallocVGPRs(
    "amdgpu-new-insert-waitcnts-dealloc-vgprs",
    cl::desc("Release VGPRs before S_ENDPGM with outstanding stores on GFX11+"),
    cl::init(true), cl::Hidden);

// Off by default to match the old pass behavior.
static cl::opt<bool> EnableEntryBlockWaitDebugLoc(
    "amdgpu-new-insert-waitcnts-entry-wait-debug-loc",
    cl::desc("Attach debug location to entry block wait instructions"),
    cl::init(false), cl::Hidden);

namespace {

class AMDGPUNewInsertWaitcntsLegacy : public MachineFunctionPass {
  InsertWaitcnt IW;

public:
  static char ID;

  AMDGPUNewInsertWaitcntsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    return IW.run(MF, MLI, AA);
  }

  StringRef getPassName() const override {
    return "AMDGPU New Insert Waitcnts";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS(AMDGPUNewInsertWaitcntsLegacy, DEBUG_TYPE,
                "AMDGPU New Insert Waitcnts", false, false)

char AMDGPUNewInsertWaitcntsLegacy::ID = 0;

char &llvm::AMDGPUNewInsertWaitcntsID = AMDGPUNewInsertWaitcntsLegacy::ID;

FunctionPass *llvm::createAMDGPUNewInsertWaitcntsPass() {
  return new AMDGPUNewInsertWaitcntsLegacy();
}

PreservedAnalyses
AMDGPUNewInsertWaitcntsPass::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  auto &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
  auto &FAM = MFAM.getResult<FunctionAnalysisManagerMachineFunctionProxy>(MF)
                  .getManager();
  auto &AA = FAM.getResult<AAManager>(MF.getFunction());
  if (!IW.run(MF, MLI, AA))
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses()
      .preserveSet<CFGAnalyses>()
      .preserve<AAManager>();
}

/// Returns true if MI is a wait instruction, including S_WAITCNT_DEPCTR
/// which is not in SIInstrInfo::isWaitcnt() since it's for VALU hazards,
/// and S_WAIT_ASYNCCNT which is for async LDS DMA operations.
static bool isWaitInstr(const MachineInstr &MI) {
  return SIInstrInfo::isWaitcnt(MI.getOpcode()) ||
         MI.getOpcode() == AMDGPU::S_WAITCNT_DEPCTR ||
         MI.getOpcode() == AMDGPU::S_WAIT_ASYNCCNT ||
         MI.getOpcode() == AMDGPU::S_WAIT_TENSORCNT;
}

// A VMEM access is a VMEM or FLAT instruction that goes through VMEM. Address
// translation for these completes in-order, so an XCnt wait before such an
// instruction is unnecessary: by the time this instruction's address
// translation is queued, any prior VMEM address translation has finished.
static bool isVmemAccess(const MachineInstr &MI, const SIInstrInfo &TII) {
  return (TII.isFLAT(MI) && TII.mayAccessVMEMThroughFlat(MI)) ||
         (SIInstrInfo::isVMEM(MI) &&
          !AMDGPU::getMUBUFIsBufferInv(MI.getOpcode()));
}

// A KmCnt dependency on an S_BARRIER_SIGNAL_ISFIRST_IMM's asynchronous SCC
// write (see getCountersForInstr) can be ignored at an SCC reader if an
// S_BARRIER_WAIT on the same barrier sits between them: that wait guarantees
// the SCC write has landed, so no s_wait_kmcnt is needed. Returns true if such
// a barrier wait is found between \p SrcMI (the isfirst signal) and \p Reader
// in the same block. Mirrors tryClearSCCWriteEvent in the original
// SIInsertWaitcnts pass.
static bool sccWriteLandedViaBarrierWait(const MachineInstr &SrcMI,
                                         const MachineInstr &Reader) {
  if (SrcMI.getOpcode() != AMDGPU::S_BARRIER_SIGNAL_ISFIRST_IMM)
    return false;
  if (SrcMI.getParent() != Reader.getParent())
    return false;
  int64_t BarrierId = SrcMI.getOperand(0).getImm();
  for (auto It = std::next(SrcMI.getIterator()), E = Reader.getIterator();
       It != E; ++It) {
    if (It->getOpcode() == AMDGPU::S_BARRIER_WAIT &&
        It->getOperand(0).getImm() == BarrierId)
      return true;
  }
  return false;
}

static unsigned getWaitOpcode(const CounterType &C) {
  thread_local static DenseMap<CounterType, unsigned> CounterToOpcodeMap;
  if (CounterToOpcodeMap.empty()) {
    CounterToOpcodeMap[AMDGPU::LoadCnt()] = AMDGPU::S_WAIT_LOADCNT;
    CounterToOpcodeMap[AMDGPU::DsCnt()] = AMDGPU::S_WAIT_DSCNT;
    CounterToOpcodeMap[AMDGPU::ExpCnt()] = AMDGPU::S_WAIT_EXPCNT;
    CounterToOpcodeMap[AMDGPU::StoreCnt()] = AMDGPU::S_WAIT_STORECNT;
    CounterToOpcodeMap[AMDGPU::SampleCnt()] = AMDGPU::S_WAIT_SAMPLECNT;
    CounterToOpcodeMap[AMDGPU::BvhCnt()] = AMDGPU::S_WAIT_BVHCNT;
    CounterToOpcodeMap[AMDGPU::KmCnt()] = AMDGPU::S_WAIT_KMCNT;
    CounterToOpcodeMap[AMDGPU::XCnt()] = AMDGPU::S_WAIT_XCNT;
    CounterToOpcodeMap[AMDGPU::AsyncCnt()] = AMDGPU::S_WAIT_ASYNCCNT;
    CounterToOpcodeMap[AMDGPU::TensorCnt()] = AMDGPU::S_WAIT_TENSORCNT;
    CounterToOpcodeMap[AMDGPU::VaVdst()] = AMDGPU::S_WAITCNT_DEPCTR;
    CounterToOpcodeMap[AMDGPU::VmVsrc()] = AMDGPU::S_WAITCNT_DEPCTR;
  }
  return CounterToOpcodeMap.at(C);
}

static Waitcnt getWaitcntFor(const WaitDescriptors &CAWs) {
  Waitcnt Wcnt;
  for (auto [MI, Cntr, Wait] : CAWs) {
    InstCounterType T = InstCounters::getLegacyInstCounterType(Cntr);
    Wcnt.set(T, Wait);
  }
  return Wcnt;
}

#ifndef NDEBUG
void DisambiguatedWaitAsyncmark::dump() const {
  print(dbgs());
  dbgs() << "\n";
}

void InsertWaitcnt::dumpDisambiguatedWaitAsyncmarks() const {
  if (WaitAsyncmarkToDisambiguatedMap.empty()) {
    dbgs() << "No disambiguated WAIT_ASYNCMARKs\n";
    return;
  }
  const MachineFunction *MF =
      WaitAsyncmarkToDisambiguatedMap.begin()->first->getMF();
  dbgs() << "=== Disambiguated WAIT_ASYNCMARKs in " << MF->getName()
         << " ===\n";
  for (const MachineBasicBlock &MBB : *MF) {
    MBB.printName(dbgs());
    dbgs() << ":\n";
    for (const MachineInstr &MI : MBB) {
      dbgs().indent(4);
      MI.print(dbgs(), /*IsStandalone=*/false, /*SkipOpers=*/false,
               /*SkipDebugLoc=*/true);
      if (MI.getOpcode() == AMDGPU::WAIT_ASYNCMARK) {
        auto It = WaitAsyncmarkToDisambiguatedMap.find(&MI);
        if (It != WaitAsyncmarkToDisambiguatedMap.end()) {
          dbgs() << "  ; disambiguated: [";
          interleave(
              It->second, dbgs(),
              [](const DisambiguatedWaitAsyncmark &DWA) { dbgs() << DWA; },
              ",");
          dbgs() << "]";
        }
      }
      dbgs() << "\n";
    }
  }
}
#endif

void InsertWaitcnt::tryDisambiguateWaitAsyncmark(MachineInstr &MI) {
  assert(MI.getOpcode() == AMDGPU::WAIT_ASYNCMARK && "Expected AsyncMark!");
  // Check if we have already disambiguated this wait.
  auto It = WaitAsyncmarkToDisambiguatedMap.find(&MI);
  if (It != WaitAsyncmarkToDisambiguatedMap.end())
    return;

  unsigned N = MI.getOperand(0).getImm();
  // Find the Nth mark event across all async counters. Returns one entry per
  // counter that belongs to the same ASYNCMARK event. Clamping of N when
  // MaxAsyncMarks is exceeded is handled inside getNthMostRecentMarkedAmong.
  for (auto [AC, EffectiveN] : AsyncCounter::getNthMostRecentMarkedAmong(
           RTracker->getAsyncCounters(), N))
    WaitAsyncmarkToDisambiguatedMap[&MI].insert(
        DisambiguatedWaitAsyncmark(AC->getType(), EffectiveN));
}

WaitDescriptors InsertWaitcnt::getCountersAndWaits(const MachineInstr &WaitMI,
                                                     const GCNSubtarget &ST) {
  AMDGPU::IsaVersion IV = AMDGPU::getIsaVersion(ST.getCPU());

  const bool IsGFX12Plus = ST.getGeneration() >= AMDGPUSubtarget::GFX12;

  unsigned Opc = WaitMI.getOpcode();
  if (Opc == AMDGPU::S_WAITCNT) {
    Waitcnt Decoded = AMDGPU::decodeWaitcnt(IV, WaitMI.getOperand(0).getImm());
    // Legacy S_WAITCNT only waits for VMcnt, Expcnt, LGKMcnt.
    // Map to generation-appropriate counters.
    //
    // A combined S_WAITCNT immediate has a fixed field per counter; a field at
    // its bit-mask value means "no wait" for that counter. decodeWaitcnt
    // returns that bit-mask (not ~0u), so compare against the per-field mask to
    // add only the counters that are actually being waited on.
    WaitDescriptors Result;
    auto MaybeAdd = [&](const CounterType &Cntr, unsigned Wait,
                        unsigned NoWait) {
      if (Wait != NoWait)
        Result.emplace(Cntr, Wait);
    };
    unsigned VmNoWait = AMDGPU::getVmcntBitMask(IV);
    unsigned ExpNoWait = AMDGPU::getExpcntBitMask(IV);
    unsigned LgkmNoWait = AMDGPU::getLgkmcntBitMask(IV);
    if (IsGFX12Plus) {
      MaybeAdd(LoadCnt(), Decoded.get(AMDGPU::LOAD_CNT), VmNoWait);
      MaybeAdd(ExpCnt(), Decoded.get(AMDGPU::EXP_CNT), ExpNoWait);
      MaybeAdd(DsCnt(), Decoded.get(AMDGPU::DS_CNT), LgkmNoWait);
    } else {
      MaybeAdd(VmCnt(), Decoded.get(AMDGPU::LOAD_CNT), VmNoWait);
      MaybeAdd(ExpCnt(), Decoded.get(AMDGPU::EXP_CNT), ExpNoWait);
      MaybeAdd(LgkmCnt(), Decoded.get(AMDGPU::DS_CNT), LgkmNoWait);
    }
    return Result;
  }

  if (!WaitMI.getNumOperands())
    return WaitDescriptors();

  // SOPK wait instructions have format (reg, imm) - wait value in operand 1.
  auto MakeSOPKResult = [&](const CounterType &Cntr) -> WaitDescriptors {
    WaitDescriptors Result;
    Result.emplace(Cntr, static_cast<unsigned>(WaitMI.getOperand(1).getImm()));
    return Result;
  };
  switch (Opc) {
  case AMDGPU::S_WAITCNT_VSCNT:
  case AMDGPU::S_WAITCNT_VSCNT_soft:
    return MakeSOPKResult(VsCnt());
  case AMDGPU::S_WAITCNT_VMCNT:
    return MakeSOPKResult(VmCnt());
  case AMDGPU::S_WAITCNT_EXPCNT:
    return MakeSOPKResult(ExpCnt());
  case AMDGPU::S_WAITCNT_LGKMCNT:
    return MakeSOPKResult(LgkmCnt());
  default:
    break;
  }

  unsigned WaitValue = WaitMI.getOperand(0).getImm();
  WaitDescriptors Result;

  switch (Opc) {
  case AMDGPU::S_WAIT_LOADCNT:
    Result.emplace(LoadCnt(), WaitValue);
    break;
  case AMDGPU::S_WAIT_STORECNT:
    Result.emplace(StoreCnt(), WaitValue);
    break;
  case AMDGPU::S_WAIT_SAMPLECNT:
    Result.emplace(SampleCnt(), WaitValue);
    break;
  case AMDGPU::S_WAIT_BVHCNT:
    Result.emplace(BvhCnt(), WaitValue);
    break;
  case AMDGPU::S_WAIT_DSCNT:
    Result.emplace(DsCnt(), WaitValue);
    break;
  case AMDGPU::S_WAIT_KMCNT:
    Result.emplace(KmCnt(), WaitValue);
    break;
  case AMDGPU::S_WAIT_XCNT:
    Result.emplace(XCnt(), WaitValue);
    break;
  case AMDGPU::S_WAIT_EXPCNT:
    Result.emplace(ExpCnt(), WaitValue);
    break;
  case AMDGPU::S_WAIT_LOADCNT_DSCNT: {
    Waitcnt Decoded = AMDGPU::decodeLoadcntDscnt(IV, WaitValue);
    Result.emplace(LoadCnt(), Decoded.get(AMDGPU::LOAD_CNT));
    Result.emplace(DsCnt(), Decoded.get(AMDGPU::DS_CNT));
    break;
  }
  case AMDGPU::S_WAIT_STORECNT_DSCNT: {
    Waitcnt Decoded = AMDGPU::decodeStorecntDscnt(IV, WaitValue);
    Result.emplace(StoreCnt(), Decoded.get(AMDGPU::STORE_CNT));
    Result.emplace(DsCnt(), Decoded.get(AMDGPU::DS_CNT));
    break;
  }
  case AMDGPU::S_WAITCNT_DEPCTR: {
    // Decode VaVdst and VmVsrc from DEPCTR encoding.
    unsigned VaVdstVal = AMDGPU::DepCtr::decodeFieldVaVdst(WaitValue);
    unsigned VmVsrcVal = AMDGPU::DepCtr::decodeFieldVmVsrc(WaitValue);
    // Only add if the field is not "don't wait" (max value for the field).
    if (VaVdstVal != AMDGPU::DepCtr::getVaVdstBitMask())
      Result.emplace(VaVdst(), VaVdstVal);
    if (VmVsrcVal != AMDGPU::DepCtr::getVmVsrcBitMask())
      Result.emplace(VmVsrc(), VmVsrcVal);
    break;
  }
  case AMDGPU::S_WAIT_ASYNCCNT:
    Result.emplace(AsyncCnt(), WaitValue);
    break;
  case AMDGPU::S_WAIT_TENSORCNT:
    Result.emplace(TensorCnt(), WaitValue);
    break;
  default:
    break;
  }
  return Result;
}

iterator_range<MachineBasicBlock::instr_iterator>
InsertWaitcnt::emitWaitInstr(MachineBasicBlock &MBB,
                             MachineBasicBlock::instr_iterator InsertPt,
                             const WaitDescriptors &CAWs,
                             MachineBasicBlock::instr_iterator ForMI) const {
  LLVM_DEBUG(dbgs() << "emitWaitInstr ENTRY with " << CAWs.size() << " CAWs\n");
  MachineBasicBlock::instr_iterator FirstMI = InsertPt;
  const DebugLoc &DbgLoc = MBB.findDebugLoc(InsertPt);
  AMDGPU::IsaVersion IsaVer(AMDGPU::getIsaVersion(ST->getCPU()));

  auto EmitMI = [&](MachineInstrBuilder MIB) {
    MachineInstr *MI = MIB;
    LLVM_DEBUG(dbgs() << "emitWaitInstr(): " << *MI;);
    if (FirstMI == InsertPt)
      FirstMI = MI->getIterator();
  };

  if (ST->hasExtendedWaitCounts()) {
    // gfx12+: Check for combined instruction opportunities.
    bool EmittedDepctr = false;
    std::optional<unsigned> LoadCntWait, StoreCntWait, DsCntWait;
    for (const auto &[MI, Cntr, Wait] : CAWs) {
      if (Cntr == LoadCnt())
        LoadCntWait = Wait;
      else if (Cntr == StoreCnt())
        StoreCntWait = Wait;
      else if (Cntr == DsCnt())
        DsCntWait = Wait;
    }

    // Emit combined S_WAIT_LOADCNT_DSCNT if both LoadCnt and DsCnt are present.
    bool EmittedLoadDsCnt = false;
    if (LoadCntWait && DsCntWait) {
      Waitcnt Wait;
      Wait.set(LOAD_CNT, *LoadCntWait);
      Wait.set(DS_CNT, *DsCntWait);
      unsigned Enc = AMDGPU::encodeLoadcntDscnt(IsaVer, Wait);
      EmitMI(
          BuildMI(MBB, InsertPt, DbgLoc, TII->get(AMDGPU::S_WAIT_LOADCNT_DSCNT))
              .addImm(Enc));
      EmittedLoadDsCnt = true;
    }

    // Emit combined S_WAIT_STORECNT_DSCNT if both StoreCnt and DsCnt are
    // present and we haven't already emitted a combined instruction with DsCnt.
    bool EmittedStoreDsCnt = false;
    if (StoreCntWait && DsCntWait && !EmittedLoadDsCnt) {
      Waitcnt Wait;
      Wait.set(STORE_CNT, *StoreCntWait);
      Wait.set(DS_CNT, *DsCntWait);
      unsigned Enc = AMDGPU::encodeStorecntDscnt(IsaVer, Wait);
      EmitMI(BuildMI(MBB, InsertPt, DbgLoc,
                     TII->get(AMDGPU::S_WAIT_STORECNT_DSCNT))
                 .addImm(Enc));
      EmittedStoreDsCnt = true;
    }

    // Emit individual wait instructions for remaining counters.
    for (const auto &[MI, Cntr, Wait] : CAWs) {
      // Skip counters already handled by combined instructions.
      if (Cntr == LoadCnt() && EmittedLoadDsCnt)
        continue;
      if (Cntr == DsCnt() && (EmittedLoadDsCnt || EmittedStoreDsCnt))
        continue;
      if (Cntr == StoreCnt() && EmittedStoreDsCnt)
        continue;

      // Expert mode counters (VaVdst, VmVsrc) use S_WAITCNT_DEPCTR encoding.
      // Combine both into a single instruction when present.
      if (Cntr == VaVdst() || Cntr == VmVsrc()) {
        assert(SchedMode == SchedulingMode::ExpertMode2 &&
               "Expert counters require ExpertMode2");
        if (!EmittedDepctr) {
          unsigned Enc = AMDGPU::DepCtr::encodeFieldVaVdst(~0u, *ST);
          Enc = AMDGPU::DepCtr::encodeFieldVmVsrc(Enc, ~0u);
          for (const auto &[_MI, C, W] : CAWs) {
            if (C == VaVdst())
              Enc = AMDGPU::DepCtr::encodeFieldVaVdst(Enc, W);
            else if (C == VmVsrc())
              Enc = AMDGPU::DepCtr::encodeFieldVmVsrc(Enc, W);
          }
          EmitMI(
              BuildMI(MBB, InsertPt, DbgLoc, TII->get(AMDGPU::S_WAITCNT_DEPCTR))
                  .addImm(Enc));
          EmittedDepctr = true;
        }
        continue;
      }
      EmitMI(BuildMI(MBB, InsertPt, DbgLoc, TII->get(getWaitOpcode(Cntr)))
                 .addImm(Wait));
    }
  } else {
    // Pre-gfx12: Emit S_WAITCNT for VM_CNT/EXP_CNT/LGKM_CNT and
    // S_WAITCNT_VSCNT separately for VS_CNT.
    std::optional<unsigned> VsCntWait;
    WaitDescriptors NonVsCntCAWs;
    for (const auto &[MI, Cntr, Wait] : CAWs) {
      if (Cntr == VsCnt())
        VsCntWait = Wait;
      else
        NonVsCntCAWs.emplace(Cntr, Wait);
    }

    // Emit S_WAITCNT for non-VsCnt counters if any are present.
    if (!NonVsCntCAWs.empty()) {
      EmitMI(BuildMI(MBB, InsertPt, DbgLoc, TII->get(AMDGPU::S_WAITCNT))
                 .addImm(AMDGPU::encodeWaitcnt(IsaVer,
                                               getWaitcntFor(NonVsCntCAWs))));
    }

    // Emit S_WAITCNT_VSCNT for VsCnt if present and target has VsCnt.
    // Pre-GFX10 targets don't have a separate VsCnt counter.
    if (VsCntWait && ST->hasVscnt()) {
      EmitMI(BuildMI(MBB, InsertPt, DbgLoc, TII->get(AMDGPU::S_WAITCNT_VSCNT))
                 .addReg(AMDGPU::SGPR_NULL, RegState::Undef)
                 .addImm(*VsCntWait));
    }
  }
  LLVM_DEBUG(dbgs() << "emitWaitInstr range size="
                    << std::distance(FirstMI, InsertPt) << "\n";);

  if (CollectEmittedWaits) {
    // Scan backward from ForMI collecting pre-existing wait coverage. ForMI
    // is the instruction that required the waits; pre-existing waits sit
    // between InsertPt and ForMI. Stop at the first instruction already in
    // EmittedWaits — that marks the boundary of the previous emitWaitInstr().
    WaitDescriptors PreexistingCAWs;
    for (MachineInstr &PreMI :
         make_range(MachineBasicBlock::reverse_instr_iterator(ForMI),
                    MachineBasicBlock::reverse_instr_iterator(InsertPt))) {
      if (EmittedWaits.contains(&PreMI))
        break;
      if (isWaitInstr(PreMI))
        for (const auto &CAW : getCountersAndWaits(PreMI, *ST))
          PreexistingCAWs.insert(CAW);
    }
    for (MachineInstr &MI : make_range(FirstMI, InsertPt)) {
      bool IsNew = false;
      for (const auto &[_MI, Cntr, Wait] : getCountersAndWaits(MI, *ST)) {
        const WaitDescriptor *Pre = PreexistingCAWs.get(Cntr);
        if (!Pre || Wait < Pre->Wait) {
          IsNew = true;
          break;
        }
      }
      if (IsNew)
        EmittedWaits.insert(&MI);
    }
  }

  return make_range(FirstMI, InsertPt);
}

/// Recomputes the vccz bit where it may be stale. There are two reasons vccz
/// can be wrong, both repaired by writing vcc back to itself (any write to vcc
/// refreshes vccz):
///   i. VCCZ bug (SI/CI, hasReadVCCZBug()): an in-flight SMEM load can corrupt
///      vccz. The matching s_waitcnt lgkmcnt(0) is emitted in
///      addMissingWaits(); here we only emit the vcc recompute.
///   ii. Partial vcc writes (!partialVCCWritesUpdateVCCZ()): writing only
///      VCC_LO/VCC_HI does not update vccz, so it must be recomputed. No
///      waitcnt is needed in this case.
/// This mirrors the VCCZWorkaround class in the original SIInsertWaitcnts pass.
/// maybeEmit() must be called on every instruction in program order, since it
/// tracks state across the block.
class VCCZRecompute {
  const ResourceTracker &RT;
  const GCNSubtarget &ST;
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  const bool VCCZCorruptionBug;
  const bool VCCZNotUpdatedByPartialWrites;
  /// vccz could be incorrect at a basic block boundary if a predecessor wrote
  /// to vcc and then issued an SMEM load, so initialize to true.
  bool MustRecomputeVCCZ = true;

  static bool partiallyWritesToVCC(const MachineInstr &MI) {
    return MI.definesRegister(AMDGPU::VCC_LO, /*TRI=*/nullptr) ||
           MI.definesRegister(AMDGPU::VCC_HI, /*TRI=*/nullptr);
  }

public:
  VCCZRecompute(const ResourceTracker &RT, const GCNSubtarget &ST,
                const SIInstrInfo &TII, const SIRegisterInfo &TRI)
      : RT(RT), ST(ST), TII(TII), TRI(TRI),
        VCCZCorruptionBug(ST.hasReadVCCZBug()),
        VCCZNotUpdatedByPartialWrites(!ST.partialVCCWritesUpdateVCCZ()) {}

  /// If \p MI reads vccz and vccz may be stale, emit a vccz recompute before
  /// \p MI. Returns true if it modified the IR.
  bool maybeEmit(MachineInstr &MI) {
    if (!VCCZCorruptionBug && !VCCZNotUpdatedByPartialWrites)
      return false;

    // An in-flight SMEM read could complete and clobber vccz at any time.
    bool PendingSmem = VCCZCorruptionBug &&
                       llvm::any_of(RT.getCounter(LgkmCnt()).instrsUnordered(),
                                    [](const MachineInstr *M) {
                                      return SIInstrInfo::isSMRD(*M);
                                    });
    MustRecomputeVCCZ |= VCCZCorruptionBug && TII.isSMRD(MI);

    bool PartiallyWritesVCC = partiallyWritesToVCC(MI);
    if (VCCZNotUpdatedByPartialWrites)
      MustRecomputeVCCZ |= PartiallyWritesVCC;

    // If MI is a vcc write with no pending SMEM (or the target has no vccz
    // corruption bug), the write itself recomputes vccz correctly.
    if (!PendingSmem || !VCCZCorruptionBug) {
      bool FullyWritesVCC = !PartiallyWritesVCC &&
                            MI.definesRegister(AMDGPU::VCC, /*TRI=*/nullptr);
      bool UpdatesVCCZ = FullyWritesVCC ||
                         (!VCCZNotUpdatedByPartialWrites && PartiallyWritesVCC);
      if (UpdatesVCCZ)
        MustRecomputeVCCZ = false;
    }

    if (SIInstrInfo::isCBranchVCCZRead(MI) && MustRecomputeVCCZ) {
      // Restore vccz by reading vcc and writing it back.
      BuildMI(*MI.getParent(), MI, MI.getDebugLoc(),
              TII.get(ST.isWave32() ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64),
              TRI.getVCC())
          .addReg(TRI.getVCC());
      MustRecomputeVCCZ = false;
      return true;
    }
    return false;
  }
};

/// Check if MBB is a loop preheader for any of its successors.
static bool isLoopPreheader(const MachineBasicBlock &MBB,
                            MachineLoopInfo *MLI) {
  for (const MachineBasicBlock *Succ : MBB.successors()) {
    MachineLoop *SuccLoop = MLI->getLoopFor(Succ);
    if (SuccLoop && SuccLoop->getHeader() == Succ &&
        SuccLoop->getLoopPreheader() == &MBB)
      return true;
  }
  return false;
}

MachineBasicBlock::instr_iterator
InsertWaitcnt::getWaitInsertPoint(MachineInstr &MI,
                                  const WaitDescriptors &FinalCAWs) const {
  auto DefaultIt = MI.getIterator();
  if (!WaitHoistOpt)
    return DefaultIt;

  // WMMA (matrix) instructions have a long co-execution window: independent
  // work can run in parallel with them. Emitting a wait in that window between
  // the WMMA and MI forces the wave to stall and collapses the window. If a
  // WMMA precedes MI and every instruction MI actually depends on was already
  // issued before that WMMA, the wait is equally correct above the WMMA, where
  // it no longer blocks the window. Hoist it there.
  //
  // Limit how far back we search so this stays cheap; the window of interest is
  // short in practice.
  static constexpr unsigned MaxWalkDistance = 8;
  MachineBasicBlock *MBB = MI.getParent();

  // Find the nearest WMMA before MI in this block, within MaxWalkDistance.
  MachineInstr *Wmma = nullptr;
  unsigned Walked = 0;
  for (MachineInstr &WalkMI :
       reverse(make_range(MBB->instr_begin(), MI.getIterator()))) {
    if (Walked++ >= MaxWalkDistance)
      break;
    if (SIInstrInfo::isWMMA(WalkMI)) {
      Wmma = &WalkMI;
      break;
    }
  }
  if (!Wmma)
    return DefaultIt;

  // Collect the instructions in the half-open window [WMMA, MI). Hoisting is
  // only safe if none of MI's dependencies live in this window.
  SmallPtrSet<const MachineInstr *, 8> InWindow;
  for (MachineInstr &Win : make_range(Wmma->getIterator(), MI.getIterator()))
    InWindow.insert(&Win);

  // Determine which counters we are about to wait on.
  SmallPtrSet<const Counter *, 4> WaitCounters;
  for (const auto &[_MI, Cntr, Wait] : FinalCAWs)
    WaitCounters.insert(&RTracker->getCounter(Cntr));

  // Re-query MI's actual register dependencies from the tracker. Only an
  // instruction MI truly depends on can pin the wait below the WMMA. An
  // instruction that merely shares a counter with MI (e.g. the WMMA itself,
  // which is a VALU tracked on VaVdst) but that MI does not depend on must not
  // block hoisting. If any true dependency on a counter we are waiting for
  // lives in the window, keep the wait below the WMMA.
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.getReg().isValid())
      continue;
    RegAccessType Access = MO.isDef() ? RegAccessType::Def : RegAccessType::Use;
    for (auto &CntAndWait : RTracker->getWaitFor(MO.getReg(), MI, Access)) {
      if (!WaitCounters.contains(&RTracker->getCounter(CntAndWait.Cntr)))
        continue;
      if (InWindow.contains(CntAndWait.MI))
        return DefaultIt;
    }
  }

  return Wmma->getIterator();
}

WaitDescriptors
InsertWaitcnt::getWaitsBasedOnRegDeps(MachineInstr &MI) const {
  WaitDescriptors Waits;
  // Collect all waits for all registers read/written.
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    if (!MO.getReg().isValid())
      continue;
    // If the instruction does not read tied source, skip the operand.
    if (MO.isTied() && MO.isUse() && TII->doesNotReadTiedSource(MI))
      continue;
    Register Reg = MO.getReg();
    // Implicit VGPR defs and uses are never a part of the memory
    // instructions description and usually present to account for
    // super-register liveness.
    const bool IsVGPR = TRI->isVectorRegister(*MRI, Reg);
    if (IsVGPR && MO.isImplicit() && MI.mayLoadOrStore())
      continue;
    // A call's (including a tail call's) implicit operands are its arguments
    // (both VGPR and SGPR). The callee waits for their readiness itself via the
    // waits inserted at its function entry (see tryInsertWaitsAtEntryBlock), so
    // we do not wait for them here. Only the call-target address register (an
    // explicit operand) needs a wait. Matches the old pass, which only waits
    // for the call-target address register.
    if (MO.isImplicit() && MI.isCall())
      continue;
    // Get the dependencies by querying the tracker.
    RegAccessType Access = MO.isDef() ? RegAccessType::Def : RegAccessType::Use;
    for (auto &CAW : RTracker->getWaitFor(Reg, MI, Access)) {
      // XCnt waits are only needed when DstMI defines the register. For RAW
      // hazards (DstMI uses), other counters (LoadCnt, KmCnt) handle the wait.
      if (CAW.Cntr == XCnt() && !MO.isDef())
        continue;
      // An asynchronous SCC write from S_BARRIER_SIGNAL_ISFIRST_IMM is tracked
      // on KmCnt, but a matching S_BARRIER_WAIT in between guarantees it has
      // landed, so the kmcnt wait is unnecessary here.
      if (CAW.Cntr == KmCnt() && Reg == AMDGPU::SCC &&
          sccWriteLandedViaBarrierWait(*CAW.MI, MI))
        continue;
      Waits.insert(CAW);
    }
  }
  LLVM_DEBUG(dbgs() << "getWaitsBasedOnRegDeps(): MI: " << MI
                    << " RegWaits: " << Waits << "\n");
  return Waits;
}

void InsertWaitcnt::removeRedundantWaits(WaitDescriptors &CAWs,
                                         MachineInstr &MI, bool &Change) const {
  if (SchedMode == SchedulingMode::ExpertMode2) {
    // In expert scheduling mode, hardware handles VALU→VGPR→VALU hazards
    // automatically via pipeline interlocks. A VaVdst wait is only needed
    // before non-VALU consumers (stores, returns, etc.) that lack this
    // interlock.
    if (SIInstrInfo::isVALU(MI, /*AllowLDSDMA=*/true))
      CAWs.erase_if(
          [](const WaitDescriptor &CAW) { return CAW.Cntr == VaVdst(); });

    // Don't emit VaVdst/VmVsrc depctr waits if a pre-existing S_WAITCNT_DEPCTR
    // for the same counter exists immediately before MI (adjacent, separated
    // only by other waits or meta instructions). The existing wait was placed
    // by a prior dataflow iteration; emitting a second one causes
    // combineWaitInstrs to pick the more conservative value.
    if (!CAWs.empty() && MI.getIterator() != MI.getParent()->instr_begin()) {
      auto HasAdjacentDepctr = [&](const CounterType &Cntr) {
        auto It = std::prev(MI.getIterator());
        auto Begin = MI.getParent()->instr_begin();
        while (true) {
          if (It->getOpcode() == AMDGPU::S_WAITCNT_DEPCTR) {
            for (const auto &[_MI, C, W] : getCountersAndWaits(*It, *ST))
              if (C == Cntr)
                return true;
          }
          if (!It->isMetaInstruction() && !isWaitInstr(*It))
            break;
          if (It == Begin)
            break;
          --It;
        }
        return false;
      };
      CAWs.erase_if([&](const WaitDescriptor &CAW) {
        return (CAW.Cntr == VaVdst() || CAW.Cntr == VmVsrc()) &&
               HasAdjacentDepctr(CAW.Cntr);
      });
    }
  }

  // VMEM address translation completes in program order. If this instruction is
  // itself a VMEM access, an XCnt wait for a prior VMEM operation is
  // unnecessary: by the time this instruction's address translation is queued,
  // the prior VMEM's translation has already finished. We still apply the wait
  // to the tracker (so later non-VMEM instructions account for it) but suppress
  // the emitted S_WAIT_XCNT. This mirrors the original pass.
  if (isVmemAccess(MI, *TII)) {
    auto XCntWaitOpt = CAWs.get(XCnt());
    if (XCntWaitOpt)
      RTracker->drainCounters({*XCntWaitOpt});
    CAWs.erase_if(
        [](const WaitDescriptor &CAW) { return CAW.Cntr == XCnt(); });
  }

  // Some counters imply others. Waiting for TriggerCntr=0 also implies that
  // ImpliedCntr has already drained.
  // For example: KmCnt=0 drains XCnt entries from SMEM ops.
  if (ST->hasWaitXcnt()) {
    // This is the table listing all TriggerWait->ImpliedWait pairs.
    static const SmallDenseMap<CounterType, CounterType> CounterImpliesCounter =
        {
            // {TriggerWait, ImpliedWait}
            {KmCnt(), XCnt()},  ///> KMCnt=0   implies XCnt=0
            {LoadCnt(), XCnt()} ///> LoadCnt=0 impleix XCnt=0
        };

    for (const WaitDescriptor &TriggerWait : CAWs) {
      if (TriggerWait.Wait != 0)
        continue;
      auto It = CounterImpliesCounter.find(TriggerWait.Cntr);
      if (It == CounterImpliesCounter.end())
        continue;
      const CounterType &ImpliedCntr = It->second;
      const Counter &ImpliedCntrState = RTracker->getCounter(ImpliedCntr);
      // Only suppress ImpliedCntr if every pending instruction in it also
      // contributes to TriggerCntr. For example:
      //   $vgpr0 = GLOBAL_LOAD_DWORD ...  ; VMEM: LoadCnt++, XCnt++
      //   $sgpr0 = S_LOAD_DWORD ...       ; SMEM: KmCnt++,   XCnt++
      //
      //   S_WAIT_LOADCNT 0                ; drains VMEM XCnt entry only
      //
      // In this example XCnt include entries from both VMEM and SMEM
      // instructions. Wait LoadCnt=0 only drains the VMEM entries.
      // The SMEM address reads may still be outstanding, so there is no
      // guaranteed that XCnt will have drained. So we still need S_WAIT_XCNT 0.
      bool AllInstrsInImpliedCntrContributeToTrigger =
          !ImpliedCntrState.empty() &&
          llvm::all_of(ImpliedCntrState.instrsUnordered(),
                       [&](const MachineInstr *I) {
                         return llvm::is_contained(
                             Counter::getCountersForInstr(*I, *ST, SchedMode),
                             TriggerWait.Cntr);
                       });
      if (AllInstrsInImpliedCntrContributeToTrigger) {
        const WaitDescriptor *ImpliedWait = CAWs.get(ImpliedCntr);
        if (ImpliedWait)
          RTracker->drainCounters({*ImpliedWait});
        CAWs.erase_if([&](const WaitDescriptor &CAW) {
          return CAW.Cntr == ImpliedCntr;
        });
      }
    }
  }

  // Try to merge an expcnt wait into a VINTERP instruction's waitexp field.
  // VINTERP instructions have a built-in wait for LDS_PARAM_LOAD results.
  // On success, removes the expcnt entry from Waits and updates the counter.
  if (SIInstrInfo::isVINTERP(MI)) {
    int WaitExpIdx =
        AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::waitexp);
    if (WaitExpIdx >= 0) {
      // Find the expcnt wait value.
      if (WaitDescriptor *ExpcntWait = CAWs.get(ExpCnt())) {
        // Update the waitexp operand only if the new value is smaller.
        MachineOperand &WaitExpOp = MI.getOperand(WaitExpIdx);
        unsigned CurrentWaitExp = WaitExpOp.getImm();
        unsigned NewWaitExp = std::min(CurrentWaitExp, ExpcntWait->Wait);
        if (NewWaitExp != CurrentWaitExp) {
          WaitExpOp.setImm(NewWaitExp);
          Change = true;
        }

        // Remove the expcnt entry.
        CAWs.erase_if(
            [](const WaitDescriptor &CAW) { return CAW.Cntr == ExpCnt(); });

        // The waitexp field folded into the VINTERP performs the expcnt wait in
        // hardware, so apply it to the tracker. Otherwise a later VINTERP
        // reading the same LDS params would think they are still pending and
        // re-wait.
        RTracker->drainCounters({{ExpCnt(), NewWaitExp}});
      }
    }
  }

  // Debug/profiling flag: force all load counter waits to zero so every load
  // stalls until completion, making PC-sampling profiles easier to interpret.
  if (ForceEmitZeroLoadFlag) {
    if (WaitDescriptor *VmCntWait = CAWs.get(VmCnt()))
      VmCntWait->Wait = 0;
    if (WaitDescriptor *LoadCntWait = CAWs.get(LoadCnt()))
      LoadCntWait->Wait = 0;
  }
}

void InsertWaitcnt::addNonDepCAWs(WaitDescriptors &CAWs,
                                  MachineInstr &MI) const {
  // Add missing waits required, but only if the counter has pending
  // instructions. This avoids emitting redundant waits when a preexisting
  // waitcnt has already cleared the counter.

  // SI/CI VCCZ bug: an in-flight SMEM load can corrupt the vccz bit. Before an
  // instruction that reads vccz (e.g. S_CBRANCH_VCCZ/VCCNZ), force the SMEM
  // counter to 0 so the load completes and can no longer clobber vccz. The
  // matching vccz recompute (s_mov vcc, vcc) is emitted separately during the
  // block walk; see VCCZRecompute. hasReadVCCZBug() is pre-GFX12 only, so the
  // SMEM counter is always LgkmCnt here.
  if (ST->hasReadVCCZBug() && SIInstrInfo::isCBranchVCCZRead(MI)) {
    const Counter &C = RTracker->getCounter(LgkmCnt());
    bool PendingSmem =
        llvm::any_of(C.instrsUnordered(), [](const MachineInstr *PendingMI) {
          return SIInstrInfo::isSMRD(*PendingMI);
        });
    if (PendingSmem) {
      if (!RTracker->getCounter(LgkmCnt()).empty()) {
        WaitDescriptor WaitForVCCZBug(nullptr, LgkmCnt(), 0);
        CAWs.insert(WaitForVCCZBug);
      }
    }
  }

  /// Returns the wait required for special instructions like buffer
  /// invalidation that need all pending loads to complete.
  auto GetWaitForSpecialInstr = [&]() -> std::optional<WaitDescriptor> {
    switch (MI.getOpcode()) {
    case AMDGPU::BUFFER_WBINVL1:
    case AMDGPU::BUFFER_WBINVL1_SC:
    case AMDGPU::BUFFER_WBINVL1_VOL:
    case AMDGPU::BUFFER_GL0_INV:
    case AMDGPU::BUFFER_GL1_INV:
      if (ST->getGeneration() >= AMDGPUSubtarget::GFX12)
        return WaitDescriptor(nullptr, LoadCnt(), 0);
      return WaitDescriptor(nullptr, VmCnt(), 0);
    default:
      break;
    }
    return std::nullopt;
  };
  if (auto SpecialWait = GetWaitForSpecialInstr()) {
    if (!RTracker->getCounter(SpecialWait->Cntr).empty())
      CAWs.insert(*SpecialWait);
  }

  // An EXEC modification while exports, GDS, or LDSDIR instructions are
  // pending on ExpCnt requires waiting for ExpCnt to reach 0. VMEM stores
  // also track on ExpCnt (pre-gfx10) but don't hold GPR locks that conflict.
  if (MI.modifiesRegister(AMDGPU::EXEC, TRI)) {
    const Counter &ExCntr = RTracker->getCounter(ExpCnt());
    if (!ExCntr.empty()) {
      bool NeedsWait = llvm::any_of(
          ExCntr.instrsUnordered(), [this](const MachineInstr *Pending) {
            return SIInstrInfo::isEXP(*Pending) ||
                   TII->isAlwaysGDS(Pending->getOpcode()) ||
                   (TII->isDS(*Pending) &&
                    TII->hasModifiersSet(*Pending, AMDGPU::OpName::gds));
          });
      if (NeedsWait)
        CAWs.emplace(ExpCnt(), 0);
    }
  }

  // Return instructions require all pending operations to complete before
  // returning to the caller. Tail calls and S_ENDPGM are excluded.
  if (MI.isReturn() && !MI.isCall()) {
    unsigned Opc = MI.getOpcode();
    bool IsEndPgm = Opc == AMDGPU::S_ENDPGM ||
                    Opc == AMDGPU::S_ENDPGM_SAVED ||
                    Opc == AMDGPU::S_ENDPGM_ORDERED_PS_DONE;
    if (!IsEndPgm) {
      const bool IsGFX12Plus = ST->getGeneration() >= AMDGPUSubtarget::GFX12;
      for (const auto &[_MI, Cntr, Wait] : getAllZeroWaitCAWs()) {
        if (Cntr == VsCnt() || Cntr == StoreCnt())
          continue;
        const Counter &C = RTracker->getCounter(Cntr);
        if (C.empty())
          continue;
        // On GFX12+, cache invalidate/writeback instructions increment LoadCnt
        // but produce no result the caller consumes. Skip if those are the only
        // pending LoadCnt ops.
        if (IsGFX12Plus && Cntr == LoadCnt() &&
            llvm::all_of(C.instrsUnordered(),
                         [](const MachineInstr *PendingMI) {
                           return SIInstrInfo::isGFX12CacheInvOrWBInst(
                               PendingMI->getOpcode());
                         }))
          continue;
        CAWs.emplace(Cntr, Wait);
      }
    }
  }

  // Force all counters to zero for profiling so every instruction gets a wait.
  if (ForceEmitZeroFlag && !MI.isTerminator() && IsFirstVisit) {
    for (const auto &[_MI2, Cntr, Wait] : getAllZeroWaitCAWs()) {
      if (Cntr == StoreCnt() || Cntr == VsCnt())
        continue;
      CAWs.emplace(Cntr, Wait);
    }
  }
}

bool InsertWaitcnt::insertWaitsFor(MachineInstr &MI) const {
  bool Change = false;
  if (MI.isMetaInstruction())
    return Change;

  // Get waits based on MI's register dependencies.
  WaitDescriptors AllWaits = getWaitsBasedOnRegDeps(MI);

  // Add cross-unit memory dependencies (VMEM-to-LDS -> DS).
  for (const auto &CAW : RTracker->getWaitForMemory(MI))
    AllWaits.insert(CAW);

  // Add wait entries to AllWaits that don't have to do with dependencies.
  addNonDepCAWs(AllWaits, MI);

  // Optimize away some waits.
  removeRedundantWaits(AllWaits, MI, Change);

  // Emit the actual wait instructions.
  if (!AllWaits.empty()) {
    MachineBasicBlock::instr_iterator Where = getWaitInsertPoint(MI, AllWaits);
    emitWaitInstr(*MI.getParent(), Where, AllWaits, MI.getIterator());
    // Update the counters.
    RTracker->drainCounters(AllWaits);
    Change = true;
  }
  return Change;
}

bool InsertWaitcnt::tryInsertWaitsAtEntryBlock(MachineFunction &MF) {
  EntryBlockWaits.clear();

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  if (MFI->isEntryFunction())
    return false;
  if (MF.getFunction().hasFnAttribute(Attribute::Naked))
    return false;

  MachineBasicBlock &EntryMBB = MF.front();
  // Wait for any outstanding memory operations that the input registers may
  // depend on. We can't track them and it's better to do the wait after the
  // costly call sequence.

  // TODO: Could insert earlier and schedule more liberally with operations
  // that only use caller preserved registers.
  MachineBasicBlock::instr_iterator I = EntryMBB.instr_begin();
  while (I != EntryMBB.instr_end() &&
         (I->isMetaInstruction() || ExpertModeSetRegs.contains(&*I)))
    ++I;

  // Build a WaitDescriptors with wait 0 for all relevant counters.
  WaitDescriptors CAWs;
  const bool IsGFX12Plus = ST->getGeneration() >= AMDGPUSubtarget::GFX12;
  for (const auto &CT : ICounters.get(*ST, SchedMode)) {
    // StoreCnt/VsCnt, XCnt, AsyncCnt, TensorCnt are not waited on at entry.
    // TensorCnt only exists in the counter list on GFX1250+; the check is safe
    // on other targets since those counter lists don't include TensorCnt.
    if (CT == StoreCnt() || CT == VsCnt() || CT == XCnt() || CT == AsyncCnt() ||
        CT == TensorCnt())
      continue;
    // Expert counters (VaVdst, VmVsrc) are for VALU hazards, not memory waits.
    if (CT == VaVdst() || CT == VmVsrc())
      continue;
    // On GFX12+, skip image counters if image instructions are not supported.
    // For older GPUs, the S_WAITCNT encoding always includes all counters
    // so we don't skip any.
    if (IsGFX12Plus && !ST->hasImageInsts() &&
        (CT == ExpCnt() || CT == SampleCnt() || CT == BvhCnt()))
      continue;
    CAWs.emplace(CT, 0);
  }
  // Remove counters already covered at wait=0 by pre-existing wait instructions
  // at the insertion point so we don't emit redundant waits. Also compute ForMI
  // (the first non-wait instruction after the pre-existing waits) for correct
  // EmittedWaits attribution in emitWaitInstr.
  auto ForMI = I;
  while (ForMI != EntryMBB.instr_end() && isWaitInstr(*ForMI)) {
    // On GFX12+, legacy S_WAITCNT doesn't map to the fine-grained counter
    // model (see processExistingWait), so skip it here as well.
    if (!(IsGFX12Plus && ForMI->getOpcode() == AMDGPU::S_WAITCNT)) {
      for (const auto &[MI, Cntr, Wait] : getCountersAndWaits(*ForMI, *ST)) {
        if (Wait == 0) {
          const CounterType CntrCopy = Cntr;
          CAWs.erase_if([&, CntrCopy](const WaitDescriptor &CAW) {
            return CAW.Cntr == CntrCopy;
          });
        }
      }
    }
    ++ForMI;
  }
  if (CAWs.empty())
    return false;
  for (MachineInstr &WaitMI : emitWaitInstr(EntryMBB, I, CAWs, ForMI)) {
    // The old pass uses an empty DebugLoc for entry block waits.
    if (!EnableEntryBlockWaitDebugLoc)
      WaitMI.setDebugLoc(DebugLoc());
    EntryBlockWaits.insert(&WaitMI);
    LLVM_DEBUG(dbgs() << "Added to EntryBlockWaits: " << WaitMI);
  }

  // The store counter is intentionally not waited on here: at function entry we
  // don't force completion of the caller's stores. But the caller's stores may
  // still be outstanding, and the memory legalizer can insert a soft StoreCnt
  // fence (e.g. before a seq_cst atomic) that must order after them. That fence
  // is preserved by a different mechanism: insertWaitsInBlock seeds the store
  // counter with unknown incoming state for non-entry functions, which keeps
  // the fence from being treated as redundant (and propagates into loops via
  // merge).

  return true;
}

bool InsertWaitcnt::removeRedundantSoftXcnts(MachineBasicBlock &MBB) {
  if (!ST->hasWaitXcnt() || MBB.size() <= 1)
    return false;

  bool Modified = false;
  MachineInstr *LastAtomicWithSoftXcnt = nullptr;

  auto *MF = MBB.getParent();
  auto &ST = MF->getSubtarget<GCNSubtarget>();
  bool TgSplit =
      ST.hasTgSplitSupport() && AMDGPU::isTgSplitEnabled(MF->getFunction());

  for (MachineInstr &MI : drop_begin(MBB)) {
    bool IsLDS = TII->isDS(MI) ||
                 (TII->isFLAT(MI) && TII->mayAccessLDSThroughFlat(MI, TgSplit));
    if (!IsLDS && (MI.mayLoad() ^ MI.mayStore()))
      LastAtomicWithSoftXcnt = nullptr;

    bool IsAtomicRMW = (MI.getDesc().TSFlags & SIInstrFlags::maybeAtomic) &&
                       MI.mayLoad() && MI.mayStore();
    MachineInstr &PrevMI = *MI.getPrevNode();
    if (PrevMI.getOpcode() == AMDGPU::S_WAIT_XCNT_soft && IsAtomicRMW) {
      if (LastAtomicWithSoftXcnt) {
        eraseWait(PrevMI);
        Modified = true;
      }
      LastAtomicWithSoftXcnt = &MI;
    }
  }
  return Modified;
}

/// Promote soft wait instructions to their hard equivalents.
bool InsertWaitcnt::promoteSoftWaitInstrs(MachineBasicBlock &MBB) {
  bool Changed = false;
  // Use instr_iterator so waits inside bundles are seen too: the block walk
  // (insertWaitsInBlock) processes bundled waits, so they must be promoted and
  // recorded in ExplicitWaits here as well.
  for (MachineInstr &MI : make_range(MBB.instr_begin(), MBB.instr_end())) {
    if (!isWaitInstr(MI))
      continue;
    // With the soft-only xcnt optimization on, leave soft xcnt fences soft: the
    // block walk decides per fence whether a later consuming op's data wait
    // makes it redundant. The soft opcode is the marker that distinguishes a
    // memory- legalizer fence (eligible for removal) from a pass-inserted
    // dependency xcnt (always kept); the walk promotes it to hard when it
    // decides to keep it. With the optimization off, or at -O0 (where
    // preexisting soft waits are kept verbatim), fall through and promote it
    // like any other wait.
    if (SimplifyOnlySoftXCnt && !OptNone &&
        MI.getOpcode() == AMDGPU::S_WAIT_XCNT_soft)
      continue;
    unsigned NewOpc = SIInstrInfo::getNonSoftWaitcntOpcode(MI.getOpcode());
    if (NewOpc != MI.getOpcode()) {
      MI.setDesc(TII->get(NewOpc));
      Changed = true;
    } else {
      // Already hard before promotion: this is an explicit wait from the input
      // (e.g. a user s_waitcnt intrinsic). Record it so redundancy removal
      // preserves it verbatim. Waits the pass inserts later are not recorded
      // and remain eligible for simplification.
      ExplicitWaits.insert(&MI);
    }
  }
  return Changed;
}

bool InsertWaitcnt::combineWaitInstrs(MachineBasicBlock &MBB) {
  bool Changed = false;
  const bool IsGFX12Plus = ST->getGeneration() >= AMDGPUSubtarget::GFX12;

  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E;) {
    MachineInstr &MI = *I++;

    if (!isWaitInstr(MI))
      continue;

    // On GFX12+, skip legacy S_WAITCNT instructions - they were placed by
    // the user or a previous pass and should be preserved.
    bool IsLegacyWaitcnt = MI.getOpcode() == AMDGPU::S_WAITCNT;
    if (IsGFX12Plus && IsLegacyWaitcnt)
      continue;

    // Don't combine S_WAIT_ASYNCCNT. This function merges several adjacent wait
    // instructions into one packed S_WAITCNT immediate, but that immediate has
    // no field for the async counter - async LDS DMA has its own dedicated
    // S_WAIT_ASYNCCNT instruction. Merging an S_WAIT_ASYNCCNT in would drop the
    // async wait entirely. It already corresponds to exactly one instruction,
    // so there is nothing to merge - leave it untouched.
    if (MI.getOpcode() == AMDGPU::S_WAIT_ASYNCCNT)
      continue;

    // Collect the run of waitcnt instructions to combine.
    SmallVector<MachineInstr *, 4> WaitInstrs;
    WaitInstrs.push_back(&MI);

    for (; I != E; ++I) {
      // Meta instructions (ASYNCMARK, WAIT_ASYNC, IMPLICIT_DEF, KILL, CFI,
      // ...) have no execution effect, so a wait may move across them; treat
      // them as transparent so waits separated only by meta instructions still
      // combine, matching the old pass.
      if (I->isMetaInstruction())
        continue;
      if (!isWaitInstr(*I))
        break;
      // S_WAIT_ASYNCCNT is not foldable into the packed immediate (see above);
      // stop the run before it so it is left as its own instruction.
      if (I->getOpcode() == AMDGPU::S_WAIT_ASYNCCNT)
        break;
      bool IsNextLegacy = I->getOpcode() == AMDGPU::S_WAITCNT;
      // Don't mix legacy S_WAITCNT with new GFX12+ wait instructions.
      if (IsGFX12Plus && IsNextLegacy)
        break;
      WaitInstrs.push_back(&*I);
    }

    if (WaitInstrs.size() < 2)
      continue;

    LLVM_DEBUG(dbgs() << "combineWaitInstrs: combining " << WaitInstrs.size()
                      << " wait instructions\n";);

    // Combine all consecutive waitcnt instructions into a single
    // WaitDescriptors, keeping the minimum wait value per counter.
    WaitDescriptors CombinedCAWs;
    for (MachineInstr *WaitMI : WaitInstrs) {
      LLVM_DEBUG(dbgs() << "combineWaitInstrs: decoding " << *WaitMI);
      for (const auto &[MI, Cntr, Wait] : getCountersAndWaits(*WaitMI, *ST)) {
        LLVM_DEBUG(dbgs() << "combineWaitInstrs: counter " << Cntr.getName()
                          << " Wait=" << Wait << "\n");
        CombinedCAWs.insert({Cntr, Wait});
      }
    }
    LLVM_DEBUG(dbgs() << "combineWaitInstrs: emitting " << CombinedCAWs.size()
                      << " combined waits\n";);

    // Emit the combined wait at the first original wait's position ahead of any
    // skipped meta, matching the old pass. ForMI is after the last original so
    // the scan in emitWaitInstr covers all originals as pre-existing candidates.
    auto ForMI = std::next(WaitInstrs.back()->getIterator());
    emitWaitInstr(MBB, MI.getIterator(), CombinedCAWs, ForMI);
    // Erase the originals.
    for (MachineInstr *WaitMI : WaitInstrs)
      eraseWait(*WaitMI);
    Changed = true;
  }

  return Changed;
}

WaitDescriptors InsertWaitcnt::getAllZeroWaitCAWs() const {
  const bool IsGFX12Plus = ST->getGeneration() >= AMDGPUSubtarget::GFX12;
  if (IsGFX12Plus)
    return {{LoadCnt(), 0},   {DsCnt(), 0},  {ExpCnt(), 0}, {StoreCnt(), 0},
            {SampleCnt(), 0}, {BvhCnt(), 0}, {KmCnt(), 0}};
  return {{VmCnt(), 0}, {ExpCnt(), 0}, {LgkmCnt(), 0}, {VsCnt(), 0}};
}

InsertWaitcnt::WaitSimplifyResult
InsertWaitcnt::simplifyRedundantWait(MachineInstr &MI) {
  if (!isWaitInstr(MI))
    return WaitSimplifyResult::Kept;

  // Don't touch entry block waits - they handle unknown caller state.
  if (EntryBlockWaits.contains(&MI)) {
    LLVM_DEBUG(dbgs() << "Keeping entry block wait: " << MI);
    return WaitSimplifyResult::Kept;
  }

  // Preserve explicit (non-soft) waits verbatim: a wait already hard in the
  // input (e.g. a user s_waitcnt intrinsic) is an instruction the programmer
  // wrote and must be honored as-is. Only waits that were soft before promotion
  // are eligible for redundancy removal. This matches the old pass.
  if (KeepExplicitWaits && ExplicitWaits.contains(&MI)) {
    LLVM_DEBUG(dbgs() << "Keeping explicit input wait: " << MI);
    return WaitSimplifyResult::Kept;
  }

  // At -O0, preserve all preexisting waits (soft waits inserted by memory
  // legalizer for fence semantics). This matches the old pass behavior.
  if (OptNone)
    return WaitSimplifyResult::Kept;

  // Keep only the counters that are still needed (have pending operations).
  WaitDescriptors CAWs = getCountersAndWaits(MI, *ST);
  WaitDescriptors Needed;
  for (const auto &CAW : CAWs)
    if (!RTracker->getCounter(CAW.Cntr).empty() ||
        // When forcing zero waits, preserve waits even for empty counters so
        // every instruction gets a wait for profiling purposes.
        ForceEmitZeroFlag)
      Needed.insert(CAW);

  // Nothing needed: the whole wait is redundant (also covers a wait that
  // decodes to no counters at all, e.g. an all-"no-wait" immediate).
  if (Needed.empty()) {
    eraseWait(MI);
    return WaitSimplifyResult::Erased;
  }

  // Everything needed: leave the wait unchanged.
  if (Needed.size() == CAWs.size())
    return WaitSimplifyResult::Kept;

  // Partially redundant: a combined S_WAITCNT (pre-GFX12) packs Vmcnt/Expcnt/
  // Lgkmcnt into one immediate. Rewrite it in place to wait only on the needed
  // counters. Rewriting in place (vs. erase + re-emit) keeps the same
  // instruction so dataflow converges. Single-counter wait opcodes never reach
  // here (their CAWs set has one element, handled above).
  AMDGPU::IsaVersion IV = AMDGPU::getIsaVersion(ST->getCPU());
  MI.getOperand(0).setImm(AMDGPU::encodeWaitcnt(IV, getWaitcntFor(Needed)));
  return WaitSimplifyResult::Trimmed;
}

bool InsertWaitcnt::insertWaitsBeforeBarrier(MachineInstr &MI) {
  if (MI.getOpcode() != AMDGPU::S_BARRIER)
    return false;

  bool Changed = false;

  // S_BARRIER requires all memory operations to complete on subtargets
  // that don't have auto-waitcnt-before-barrier or back-off-barrier.
  if (!ST->hasAutoWaitcntBeforeBarrier() && !ST->hasBackOffBarrier()) {
    // Only wait for counters that have pending operations.
    WaitDescriptors CAWs;
    for (const auto &[MI, Cntr, Wait] : getAllZeroWaitCAWs()) {
      if (!RTracker->getCounter(Cntr).empty())
        CAWs.emplace(Cntr, Wait);
    }
    if (!CAWs.empty()) {
      MachineBasicBlock &MBB = *MI.getParent();
      emitWaitInstr(MBB, MI.getIterator(), CAWs);
      Changed = true;
      RTracker->drainCounters(CAWs);
    }
  }

  return Changed;
}

bool InsertWaitcnt::tryHandlePseudoWaitcnt(MachineInstr &MI) {
  bool Change = false;
  // S_WAITCNT_lds_direct is a pseudo inserted by the memory legalizer to mark
  // points where we need to wait for VMEM-to-LDS loads to complete.
  if (MI.getOpcode() == AMDGPU::S_WAITCNT_lds_direct) {
    // Wait only for outstanding VMEM-to-LDS loads. They land on the load
    // counter (LoadCnt on GFX12+, VmCnt on pre-GFX12), but other ops on that
    // counter must not trigger this wait: pre-GFX12 has no store counter, so
    // VMEM stores (e.g. register spills) also sit on VmCnt - and a store is not
    // a VMEM-to-LDS load. Wait for the counter only when such a load is
    // pending. These waits may be combined with adjacent waits by
    // combineWaitInstrs().
    const bool IsGFX12Plus = ST->getGeneration() >= AMDGPUSubtarget::GFX12;
    const Counter &Cntr = RTracker->getCounter(
        IsGFX12Plus ? CounterType(LoadCnt()) : CounterType(VmCnt()));
    bool HasPendingLDSDMALoad =
        llvm::any_of(Cntr.instrsUnordered(), [](const MachineInstr *MI) {
          return SIInstrInfo::mayWriteLDSThroughDMA(*MI);
        });
    if (HasPendingLDSDMALoad) {
      WaitDescriptors CAWs;
      CAWs.emplace(Cntr.getType(), 0);
      emitWaitInstr(*MI.getParent(), MI.getIterator(), CAWs);
      RTracker->drainCounters(CAWs);
    }
    eraseWait(MI);
    Change = true;
  }

  if (MI.getOpcode() == AMDGPU::WAIT_ASYNCMARK) {
    // WAIT_ASYNCMARK(N) waits for the Nth async mark from the end among
    // AsyncCnt and TensorCnt on GFX1250, VmCnt/LoadCnt on older targets.
    // Disambiguate what wait(s) this is.
    // Note: use dumpDisambiguatedWaitAsyncmarks() to see the disambiguated
    // waits as comments next to the waits.
    tryDisambiguateWaitAsyncmark(MI);

    auto It = WaitAsyncmarkToDisambiguatedMap.find(&MI);
    if (It != WaitAsyncmarkToDisambiguatedMap.end()) {
      WaitDescriptors Waits;
      for (const DisambiguatedWaitAsyncmark &DWA : It->second) {
        // Check if we need to wait.
        unsigned EffectiveN = DWA.getEffectiveN();
        const Counter &Cntr = RTracker->getCounter(DWA.getCounterType());
        if (EffectiveN >= Cntr.getDepth())
          continue;
        Waits.insert(DWA.getWaitDescriptor());
      }

      if (!Waits.empty()) {
        // A WAIT_ASYNCMARK behaves like a use (or the destination of a
        // dependency in general) of the value produced by the ASYNCMARK, so
        // insert the necessary waits before WAIT_ASYNCMARK. Just like with
        // other waits, on subsequent dataflow iterations, processExistingWait()
        // will process the emitted waits and drain the counters before the wait
        // is reached again. But in assembly tests expect the waits to show up
        // below the wait_asyncmarks, so we have finalFixups() after dataflow
        // that moves the waits to their expected location.
        emitWaitInstr(*MI.getParent(), MI.getIterator(), Waits);
        RTracker->drainCounters(Waits);
        Change = true;
      }
    }
    // Don't delete the WAIT_ASYNCMARK instruction - it will be emitted as a
    // comment in the assembly output to show the original parameter N.
  }

  return Change;
}


bool InsertWaitcnt::insertWaitsWithDataflow(MachineFunction &MF) {
  bool Change = false;

  // Store exit state for each block. Used to propagate state to successors.
  // Use unique_ptr because ResourceTracker has no default ctor or copy assign.
  DenseMap<MachineBasicBlock *, std::unique_ptr<ResourceTracker>>
      MBBResourceTrackers;

  // Initialize exit states for all blocks.
  for (MachineBasicBlock &MBB : MF)
    MBBResourceTrackers[&MBB] =
        std::make_unique<ResourceTracker>(ST, AA, SchedMode);

  // Compute reverse post-order for faster convergence.
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  SmallVector<MachineBasicBlock *, 16> RPOBlocks(RPOT.begin(), RPOT.end());

  bool Convergence = false;
  unsigned Iteration = 0;
  // Fixed-point iteration: process all blocks until the state of the counters
  // stops changing.
  while (!Convergence) {
    ++Iteration;
    LLVM_DEBUG(dbgs() << "Dataflow iteration " << Iteration << "\n");

    if (Iteration > 1000)
      report_fatal_error("Dataflow failed to converge!");

    Convergence = true;

    for (MachineBasicBlock *MBB : RPOBlocks) {
      LLVM_DEBUG(dbgs() << "Processing BB#" << MBB->getNumber() << "\n");
      IsFirstVisit = DataFlowVisitedBlocks.insert(MBB).second;
      RTracker = MBBResourceTrackers[MBB].get();

      ResourceTracker::ConvergenceState LastState =
          RTracker->getConvergenceState();

      // Compute entry state by merging exit states of all predecessors.
      // Use a separate tracker to avoid clearing the stored exit state that
      // back-edges need to merge from. Always use max so that pending
      // dependencies from any predecessor path are preserved at the join.
      ResourceTracker EntryState(ST, AA, SchedMode);
      for (MachineBasicBlock *Pred : MBB->predecessors()) {
        const auto &OtherTracker = *MBBResourceTrackers[Pred];
        EntryState.merge(OtherTracker);
      }
      *RTracker = std::move(EntryState);

      // Process MBB and insert wait instructions.
      LLVM_DEBUG({
        dbgs() << "After merge, pending counters:\n";
        for (const auto &CT : ICounters.get(*ST, SchedMode)) {
          const Counter &C = RTracker->getCounter(CT);
          dbgs() << "  " << CT.getName() << ": ";
          if (C.empty())
            dbgs() << "(empty)\n";
          else
            C.print(dbgs());
        }
      });

      Change |= insertWaitsInBlock(*MBB);

      auto State = RTracker->getConvergenceState();
      if (State != LastState)
        Convergence = false;
      LastState = State;
    }
  }

  LLVM_DEBUG(dbgs() << "Dataflow converged after " << Iteration
                    << " iterations\n");

  return Change;
}

bool InsertWaitcnt::insertWaitsNoDataflow(MachineFunction &MF) {
  // Without dataflow, we don't know what's pending from predecessors, so
  // conservatively wait for all counters at each block boundary.
  ResourceTracker Tracker(ST, AA, SchedMode);
  RTracker = &Tracker;
  bool Change = false;
  for (MachineBasicBlock &MBB : MF) {
    RTracker->clear();
    // Insert waits for all counters at the start of non-entry blocks.
    // Add them to EntryBlockWaits so they won't be removed as redundant.
    if (&MBB != &MF.front()) {
      auto Range = emitWaitInstr(MBB, MBB.instr_begin(), getAllZeroWaitCAWs());
      for (MachineInstr &MI : Range)
        EntryBlockWaits.insert(&MI);
      Change |= !Range.empty();
    }
    Change |= insertWaitsInBlock(MBB);
  }
  return Change;
}

bool InsertWaitcnt::processExistingWait(MachineInstr &MI, bool IsGFX12Plus,
                                        MachineInstr *&DeferredSoftXCnt) {
  LLVM_DEBUG(dbgs() << "Processing wait instr: " << MI);
  bool Change = false;

  // On GFX12+, legacy S_WAITCNT waits for all outstanding memory
  // operations to complete rather than waiting for specific counter
  // values. This doesn't map to the new fine-grained counter model, so
  // we don't update the tracker and continue inserting new-style waits.
  if (IsGFX12Plus && MI.getOpcode() == AMDGPU::S_WAITCNT)
    return false;

  auto AWG = RTracker->getApplyWaitsGuard(getCountersAndWaits(MI, *ST));

  // A soft xcnt fence (S_WAIT_XCNT_soft) is a memory-legalizer fence whose
  // removal depends on whether a later consuming memory op's data wait drains
  // the address translations it guards. (It only reaches here with the
  // optimization on; otherwise it was promoted to a hard wait up front.)
  if (MI.getOpcode() == AMDGPU::S_WAIT_XCNT_soft) {
    if (!RTracker->getCounter(XCnt()).empty()) {
      // Live address translations: defer the decision to the next consuming
      // op. A prior pending fence (e.g. two fences with no memory op between
      // them) has no consumer to cover it, so keep it before deferring this.
      if (DeferredSoftXCnt) {
        DeferredSoftXCnt->setDesc(TII->get(SIInstrInfo::getNonSoftWaitcntOpcode(
            DeferredSoftXCnt->getOpcode())));
        RTracker->drainCounters(getCountersAndWaits(*DeferredSoftXCnt, *ST));
      }
      DeferredSoftXCnt = &MI;
      return false;
    }
    // XCnt empty: the fence is plainly redundant. Erase it directly, since
    // simplifyRedundantWait/decodeWaitMI do not understand the soft opcode.
    eraseWait(MI);
    return true;
  }

  WaitSimplifyResult Res = simplifyRedundantWait(MI);
  if (Res == WaitSimplifyResult::Erased) {
    LLVM_DEBUG(dbgs() << "Removed as redundant\n");
    return true;
  }
  // Kept or Trimmed: the instruction still exists. Update the tracker to
  // reflect the (possibly reduced) wait it applies, so insertWaitsFor
  // does not re-emit the counters it covers.
  if (Res == WaitSimplifyResult::Trimmed)
    Change = true;
  assert(
      (Res == WaitSimplifyResult::Kept || Res == WaitSimplifyResult::Trimmed) &&
      "Erased wait must not be processed: the instruction is gone");
  // A wait following a deferred xcnt fence may itself drain the guarded
  // address translations (e.g. a soft kmcnt 0 after the soft xcnt). Snapshot
  // XCnt before processing this wait: if it goes from non-empty to empty the
  // fence is redundant and erased, otherwise it is honored.
  bool XCntWasEmpty = DeferredSoftXCnt && RTracker->getCounter(XCnt()).empty();

  // This is where the main update of the counters takes place.
  auto Waits = getCountersAndWaits(MI, *ST);
  RTracker->drainCounters(Waits);

  if (DeferredSoftXCnt) {
    if (!XCntWasEmpty && RTracker->getCounter(XCnt()).empty()) {
      eraseWait(*DeferredSoftXCnt);
      Change = true;
    } else {
      // Keeping the fence promotes it to a hard S_WAIT_XCNT and updates the
      // tracker (no structural MIR change beyond the opcode), so it must not
      // set Change - that would prevent dataflow from converging.
      DeferredSoftXCnt->setDesc(TII->get(
          SIInstrInfo::getNonSoftWaitcntOpcode(DeferredSoftXCnt->getOpcode())));
      RTracker->drainCounters(getCountersAndWaits(*DeferredSoftXCnt, *ST));
    }
    DeferredSoftXCnt = nullptr;
  }
  return Change;
}

bool InsertWaitcnt::postTrackUpdates(MachineInstr &MI) {
  bool Change = false;
  // Record "always GDS" instructions for the post-pass and drain the DS
  // counter so successor blocks see it as completed.
  if (TII->isAlwaysGDS(MI.getOpcode())) {
    const bool IsGFX12Plus = ST->getGeneration() >= AMDGPUSubtarget::GFX12;
    CounterType DsCounter =
        IsGFX12Plus ? CounterType(DsCnt()) : CounterType(LgkmCnt());
    if (IsFirstVisit)
      AlwaysGDSInstrs.push_back(&MI);
    RTracker->drainCounters({{DsCounter, 0}});
  }

  // In precise memory mode, record this memory op for the post-pass and
  // drain its counters so the dataflow sees them as completed.
  if (ST->isPreciseMemoryEnabled() && MI.mayLoadOrStore()) {
    auto Counters = Counter::getCountersForInstr(MI, *ST, SchedMode);
    if (!Counters.empty()) {
      PreciseMemoryInstrs.push_back(&MI);
      WaitDescriptors PreciseCAWs;
      for (const CounterType &CT : Counters)
        PreciseCAWs.emplace(CT, 0);
      RTracker->drainCounters(PreciseCAWs);
    }
  }
  return Change;
}

bool InsertWaitcnt::insertWaitsInBlock(MachineBasicBlock &MBB) {
  bool Change = false;
  const bool IsGFX12Plus = ST->getGeneration() >= AMDGPUSubtarget::GFX12;

  // At the entry of a non-entry function the caller's stores may still be
  // outstanding on the store counter. Seed it as unknown so a memory-legalizer
  // seq_cst store fence is preserved here and, via merge(), in any block
  // reached from entry (e.g. an atomic retry loop). A wait-for-zero clears it.
  // The readiness counters instead get explicit synthesized entry waits, so
  // only the store counter needs seeding here.
  const MachineFunction *MF = MBB.getParent();
  const SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  if (&MBB == &MF->front() && !MFI->isEntryFunction() &&
      !MF->getFunction().hasFnAttribute(Attribute::Naked))
    RTracker->setCounterIncomingUnknown(IsGFX12Plus ? CounterType(StoreCnt())
                                                    : CounterType(VsCnt()));

  // Repairs the vccz bit before vccz-reading branches on targets where it can
  // become stale (SI/CI VCCZ bug, or partial vcc writes). Constructed fresh per
  // block; its MustRecomputeVCCZ state assumes vccz may be dirty at block
  // entry.
  VCCZRecompute VCCZRecomp(*RTracker, *ST, *TII, *TRI);

  // A soft S_WAIT_XCNT_soft fence whose removal decision is deferred until the
  // next consuming memory op (or following wait): if that op's data wait drains
  // the fence's address translations it is redundant and erased, otherwise it
  // is kept. Keeping a fence promotes it to a hard S_WAIT_XCNT and applies it
  // to the tracker, so the final MIR never carries a soft opcode. nullptr when
  // none is pending.
  MachineInstr *DeferredSoftXCnt = nullptr;

  // Use instr_iterator to iterate over all instructions including those inside
  // bundles. This is necessary because bundles may contain memory operations
  // that need waitcnts inserted before them. Use make_early_inc_range to handle
  // iterator invalidation when removing redundant waits.
  for (MachineInstr &MI :
       make_early_inc_range(make_range(MBB.instr_begin(), MBB.instr_end()))) {
    // Record DEALLOC_VGPRS eligibility for an S_ENDPGM reached after MI,
    // reading the store counter before MI (a coupled trailing store wait)
    // drains it. DEALLOC_VGPRS is a GFX11+ optimization, gated by a cl::opt.
    if (EnableDeallocVGPRs && ST->getGeneration() >= AMDGPUSubtarget::GFX11)
      recordEndPgmDealloc(MI);

    // Handle preexisting waitcnt instructions. Remove if redundant, otherwise
    // update the tracker so we don't insert redundant waits later.
    if (isWaitInstr(MI)) {
      Change |= processExistingWait(MI, IsGFX12Plus, DeferredSoftXCnt);
      continue;
    }
    // Handle pseudo waitcnt instructions inserted by other passes.
    if (tryHandlePseudoWaitcnt(MI)) {
      Change = true;
      continue;
    }

    // S_BARRIER requires all memory operations to complete. Insert waits and
    // clear the tracker so we don't emit redundant waits after the barrier.
    if (insertWaitsBeforeBarrier(MI))
      continue;

    // A deferred soft xcnt fence is kept or erased depending on whether MI's
    // waits drain the XCnt counter. Snapshot its emptiness before insertWaitsFor
    // so we can detect the transition from non-empty to empty afterwards.
    bool XCntWasEmpty =
        DeferredSoftXCnt && RTracker->getCounter(XCnt()).empty();

    // The main insertion of waits, based on dependencies and other rules.
    Change |= insertWaitsFor(MI);

    // Resolve a deferred xcnt fence at this consuming op: it is redundant if
    // MI's wait emptied the (previously non-empty) XCnt counter, otherwise
    // honored.
    if (DeferredSoftXCnt) {
      if (!XCntWasEmpty && RTracker->getCounter(XCnt()).empty()) {
        eraseWait(*DeferredSoftXCnt);
        Change = true;
      } else {
        DeferredSoftXCnt->setDesc(TII->get(SIInstrInfo::getNonSoftWaitcntOpcode(
            DeferredSoftXCnt->getOpcode())));
        RTracker->drainCounters(getCountersAndWaits(*DeferredSoftXCnt, *ST));
      }
      DeferredSoftXCnt = nullptr;
    }

    // Update the counters that get affected by MI.
    RTracker->track(MI);

    Change |= postTrackUpdates(MI);

    // Emit a vccz recompute if MI reads vccz and it may be stale.
    Change |= VCCZRecomp.maybeEmit(MI);
  }

  // A fence still pending at block end has no consumer to cover it; keep it by
  // promoting it to a hard S_WAIT_XCNT and applying it to the tracker.
  if (DeferredSoftXCnt) {
    DeferredSoftXCnt->setDesc(TII->get(
        SIInstrInfo::getNonSoftWaitcntOpcode(DeferredSoftXCnt->getOpcode())));
    RTracker->drainCounters(getCountersAndWaits(*DeferredSoftXCnt, *ST));
    DeferredSoftXCnt = nullptr;
  }

  // Emit preheader flush if this block is a loop preheader.
  if (isLoopPreheader(MBB, MLI))
    Change |= tryEmitPreheaderFlush(MBB);

  return Change;
}

bool InsertWaitcnt::tryEmitPreheaderFlush(MachineBasicBlock &MBB) {
  assert(isLoopPreheader(MBB, MLI) && "Expected a loop preheader");

  // Only emit preheader flush on first visit to avoid infinite dataflow loop.
  // Subsequent visits will see the flush we emitted and process it via
  // applyWait, which clears the counter state.
  if (!IsFirstVisit)
    return false;

  // Find the loop we're a preheader for.
  MachineLoop *Loop = nullptr;
  for (MachineBasicBlock *Succ : MBB.successors()) {
    MachineLoop *SuccLoop = MLI->getLoopFor(Succ);
    if (SuccLoop && SuccLoop->getHeader() == Succ &&
        SuccLoop->getLoopPreheader() == &MBB) {
      Loop = SuccLoop;
      break;
    }
  }
  if (!Loop)
    return false;

  // Collect registers used in the loop.
  SmallDenseSet<MCRegUnit, 64> LoopUsedRegUnits;
  for (MachineBasicBlock *LoopBB : Loop->blocks()) {
    for (MachineInstr &MI : *LoopBB) {
      for (const MachineOperand &MO : MI.uses()) {
        if (!MO.isReg() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit RU : TRI->regunits(MO.getReg().asMCReg()))
          LoopUsedRegUnits.insert(RU);
      }
    }
  }

  // Check each counter for pending operations whose results are used in loop.
  WaitDescriptors FlushCAWs;
  for (const CounterType &CT : ICounters.get(*ST, SchedMode)) {
    const Counter &Cntr = RTracker->getCounter(CT);
    if (Cntr.empty())
      continue;

    // Check if any pending instruction writes a register used in the loop.
    // Also note whether such a producer is a FLAT instruction: a FLAT load
    // lands on both the VMEM and DS counters, but its result readiness is
    // governed by the VMEM counter. When that VMEM wait stays in the loop, the
    // DS-counter wait for the same load must stay with it rather than being
    // hoisted, or the wait gets needlessly split across the preheader and the
    // loop.
    bool HasLiveInResult = false;
    bool LiveInResultFromFlat = false;
    for (const MachineInstr *PendingMI : Cntr.instrsUnordered()) {
      bool IsLiveIn = false;
      for (const MachineOperand &MO : PendingMI->all_defs()) {
        if (!MO.isReg() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit RU : TRI->regunits(MO.getReg().asMCReg())) {
          if (LoopUsedRegUnits.contains(RU)) {
            IsLiveIn = true;
            break;
          }
        }
        if (IsLiveIn)
          break;
      }
      if (IsLiveIn) {
        HasLiveInResult = true;
        if (SIInstrInfo::isFLAT(*PendingMI))
          LiveInResultFromFlat = true;
      }
    }

    if (!HasLiveInResult)
      continue;

    // Async LDS DMA completion is managed entirely by WAIT_ASYNCMARK, not by
    // preheader flushing: in a software-pipelined loop the async loads are
    // meant to stay in flight across iterations, with WAIT_ASYNCMARK enforcing
    // the staggered per-iteration waits. So never flush the async counters
    // here.
    if (CT == AsyncCnt() || CT == TensorCnt())
      continue;

    // XCnt is implied by KmCnt/LoadCnt, skip it.
    if (CT == XCnt())
      continue;

    // The old pass never flushes the SMEM counter (KmCnt, or LgkmCnt on
    // pre-gfx12) into a preheader - it only flushes VmCnt/DsCnt. Hoisting a
    // loop-invariant SMEM load's wait here is valid but diverges from the old
    // pass, so it is off by default.
    if (CT == KmCnt() && !EnableKmCntPreheaderFlush)
      continue;
    if (CT == LgkmCnt() && !EnableLgkmCntPreheaderFlush)
      continue;

    // Don't hoist the DS counter for a FLAT load whose result is used in the
    // loop: its VMEM-counter wait stays in the loop, so keep the DS-counter
    // wait there too instead of splitting it into the preheader.
    if ((CT == DsCnt() || CT == LgkmCnt()) && LiveInResultFromFlat)
      continue;

    // Expert-mode counters (VaVdst, VmVsrc) are not hoisted by the old pass.
    if ((CT == VaVdst() || CT == VmVsrc()) && !EnableDepctrPreheaderFlush)
      continue;

    // Don't flush if the loop has operations that invalidate preheader flush
    // (e.g., stores that increment the counter and need to be waited inside).
    if (loopInvalidatesPreheaderFlush(Loop, CT))
      continue;

    LLVM_DEBUG(dbgs() << "tryEmitPreheaderFlush: BB#" << MBB.getNumber()
                      << " flushing " << CT.getName()
                      << " (results live-in to loop)\n");
    FlushCAWs.emplace(CT, 0);
  }

  if (FlushCAWs.empty())
    return false;

  // Emit flush before the terminator.
  MachineBasicBlock::instr_iterator InsertPt =
      MBB.getFirstTerminator().getInstrIterator();
  emitWaitInstr(MBB, InsertPt, FlushCAWs);

  // Update tracker state.
  RTracker->drainCounters(FlushCAWs);

  return true;
}

/// Determine if counter should be flushed in the preheader based on
/// generation-specific rules. Returns true if flush should be applied.
///
/// Some counters have generation-specific behavior due to hardware differences:
/// - VmCnt on GFX9: tracks both loads and stores. Flush only when loop has
///   store but no load (store won't interfere with preheader load wait).
/// - VmCnt on GFX10/GFX11: Vscnt separates stores from VmCnt. Flush if loop
///   has VMEM load (loads complete in order with VmemWriteVgprInOrder).
bool InsertWaitcnt::shouldFlushCounterBasedOnGeneration(
    const CounterType &Cntr, const MachineLoop *Loop) const {
  // VmCnt has generation-specific preheader flush logic.
  if (Cntr == VmCnt()) {
    bool HasVMemLoad = false;
    bool HasVMemStore = false;
    bool VMemLoadUsedInLoop = false;

    // Collect registers defined by VMEM loads in the loop.
    SmallDenseSet<MCRegUnit, 16> VgprDefVMEM;
    SmallDenseSet<MCRegUnit, 16> VgprUse;

    for (MachineBasicBlock *MBB : Loop->blocks()) {
      for (MachineInstr &MI : *MBB) {
        bool IsVMEM = SIInstrInfo::isVMEM(MI) ||
                      SIInstrInfo::isFLATGlobal(MI) ||
                      SIInstrInfo::isFLATScratch(MI) || SIInstrInfo::isFLAT(MI);
        if (IsVMEM) {
          if (MI.mayLoad())
            HasVMemLoad = true;
          if (MI.mayStore())
            HasVMemStore = true;
        }

        // Check if any VMEM load def is used in the loop (either order).
        // This invalidates the preheader flush optimization.
        for (const MachineOperand &Op : MI.all_uses()) {
          if (!Op.isReg() || !Op.getReg().isPhysical())
            continue;
          for (MCRegUnit RU : TRI->regunits(Op.getReg().asMCReg())) {
            if (VgprDefVMEM.contains(RU))
              VMemLoadUsedInLoop = true;
            VgprUse.insert(RU);
          }
        }

        if (IsVMEM && MI.mayLoad()) {
          for (const MachineOperand &Op : MI.all_defs()) {
            if (!Op.isReg() || !Op.getReg().isPhysical())
              continue;
            for (MCRegUnit RU : TRI->regunits(Op.getReg().asMCReg())) {
              if (VgprUse.contains(RU))
                VMemLoadUsedInLoop = true;
              VgprDefVMEM.insert(RU);
            }
          }
        }
      }
    }

    // If a VMEM load result is used in the same loop, preheader flush is
    // invalid - the wait must happen inside the loop for correctness.
    if (VMemLoadUsedInLoop) {
      LLVM_DEBUG(
          dbgs() << "  VmCnt: VMEM load result used in loop, skip flush\n");
      return false;
    }

    // The original pass logic for VmCnt preheader flush is:
    // (!hasVscnt && HasVMemStore && !HasVMemLoad) ||
    // (HasVMemLoad && hasVmemWriteVgprInOrder)
    //
    // - First condition: pre-GFX10 with store but no load in loop
    // - Second condition: loop has load and in-order completion is guaranteed
    bool GFX9Condition = !ST->hasVscnt() && HasVMemStore && !HasVMemLoad;
    bool InOrderCondition = HasVMemLoad && ST->hasVmemWriteVgprInOrder();
    bool ShouldFlush = GFX9Condition || InOrderCondition;

    LLVM_DEBUG(dbgs() << "  VmCnt: HasVMemStore=" << HasVMemStore
                      << " HasVMemLoad=" << HasVMemLoad << " hasVscnt="
                      << ST->hasVscnt() << " hasVmemWriteVgprInOrder="
                      << ST->hasVmemWriteVgprInOrder() << " -> "
                      << (ShouldFlush ? "flush" : "skip") << "\n");
    return ShouldFlush;
  }

  // Other counters: no generation-specific logic, use default behavior.
  return false;
}

/// Check if the loop contains operations that invalidate preheader flush
/// for the given counter. For load counters (DsCnt, LoadCnt), stores
/// invalidate because we need to wait for them inside the loop. Loads
/// don't invalidate because they complete in order and prefetch patterns
/// use loop loads for next iteration only.
bool InsertWaitcnt::loopInvalidatesPreheaderFlush(
    const MachineLoop *Loop, const CounterType &Cntr) const {
  // Use insert to get iterator for both lookup and store in one operation.
  auto Key = std::make_pair(Loop, Cntr);
  auto [It, Inserted] = LoopInvalidatesFlushCache.insert({Key, false});
  if (!Inserted)
    return It->second;

  auto ComputeResult = [&]() {
    const bool IsGFX12Plus = ST->getGeneration() >= AMDGPUSubtarget::GFX12;

    LLVM_DEBUG(dbgs() << "loopInvalidatesPreheaderFlush: checking "
                      << Cntr.getName() << " in loop with "
                      << Loop->getNumBlocks() << " blocks\n");

    // The preheader flush optimization moves a wait from inside the loop to the
    // preheader. This is valid when:
    // 1. Pending operations from preheader have results live-in to the loop
    // 2. Loop operations don't invalidate the flush
    //
    // For load counters (DsCnt, LoadCnt, KmCnt), loop loads don't invalidate
    // because loads complete in order - the preheader load will complete before
    // any loop loads. However, loop stores DO invalidate because they increment
    // the counter and may need to be waited for inside the loop.
    //
    // For store counters (StoreCnt/VsCnt), the situation is different - stores
    // don't produce register results that are "waited for" in the same sense.
    // Any store in the loop would increment the counter.

    // VmCnt (pre-GFX12) has generation-specific preheader flush logic.
    // Handle it separately since we need to scan the entire loop first.
    if (!IsGFX12Plus && Cntr == VmCnt()) {
      // Return true to skip flush, false to allow flush.
      return !shouldFlushCounterBasedOnGeneration(Cntr, Loop);
    }

    for (MachineBasicBlock *MBB : Loop->blocks()) {
      for (MachineInstr &MI : *MBB) {
        // DsCnt (GFX12+) or LgkmCnt (pre-GFX12) for DS operations.
        // DS stores invalidate preheader flush because we need to wait for them
        // inside the loop. DS loads don't invalidate because:
        // 1. Loads complete in order - preheader load completes before loop
        // loads
        // 2. Prefetch pattern: loop loads are for NEXT iteration, current
        // iteration
        //    uses values from preheader (first iter) or previous iteration's
        //    prefetch
        // FLAT operations also increment DsCnt (GFX12+) or LgkmCnt (pre-GFX12).
        if (Cntr == DsCnt() || (!IsGFX12Plus && Cntr == LgkmCnt())) {
          if (SIInstrInfo::isDS(MI) && MI.mayStore()) {
            LLVM_DEBUG(dbgs() << "  Found DS store: " << MI);
            return true;
          }
          // FLAT stores also increment DsCnt/LgkmCnt.
          if (SIInstrInfo::isFLAT(MI) && MI.mayStore()) {
            LLVM_DEBUG(dbgs() << "  Found FLAT store (increments "
                              << Cntr.getName() << "): " << MI);
            return true;
          }
        }

        // LoadCnt (GFX12+) for VMEM loads.
        // Skip preheader flush - GFX12+ doesn't have generation-specific logic
        // implemented yet.
        if (Cntr == LoadCnt())
          return true;

        // StoreCnt (GFX12+) or VsCnt (pre-GFX12) for VMEM stores.
        // Any store increments the counter.
        if (Cntr == StoreCnt() || (!IsGFX12Plus && Cntr == VsCnt())) {
          if ((SIInstrInfo::isVMEM(MI) || SIInstrInfo::isFLATGlobal(MI) ||
               SIInstrInfo::isFLATScratch(MI) || SIInstrInfo::isFLAT(MI)) &&
              MI.mayStore()) {
            LLVM_DEBUG(dbgs() << "  Found VMEM store: " << MI);
            return true;
          }
        }

        // KmCnt (GFX12+) for SMEM operations.
        // SMEM is read-only (no stores), so this counter is never invalidated
        // by loop operations. Skip the check for KmCnt.
        // Note: LgkmCnt (pre-GFX12) also covers DS, handled above.

        // ExpCnt for exports, LDS_DIRECT, and (on pre-gfx10) VMEM/scratch
        // stores. On pre-gfx10, VMEM/scratch stores use ExpCnt to track when
        // their data source registers are safe to overwrite
        // (vmemWriteNeedsExpWaitcnt), so any such store in the loop
        // re-increments ExpCnt each iteration.
        if (Cntr == ExpCnt()) {
          if (SIInstrInfo::isEXP(MI) || SIInstrInfo::isLDSDIR(MI)) {
            LLVM_DEBUG(dbgs() << "  Found EXP/LDSDIR: " << MI);
            return true;
          }
          if (ST->vmemWriteNeedsExpWaitcnt() && MI.mayStore() &&
              (SIInstrInfo::isVMEM(MI) || SIInstrInfo::isFLAT(MI))) {
            LLVM_DEBUG(dbgs()
                       << "  Found VMEM/FLAT store (pre-gfx10 ExpCnt): " << MI);
            return true;
          }
        }

        // SampleCnt and BvhCnt for image operations.
        if (Cntr == SampleCnt()) {
          if (SIInstrInfo::isMIMG(MI) &&
              AMDGPU::getMIMGBaseOpcode(MI.getOpcode())->Sampler) {
            LLVM_DEBUG(dbgs() << "  Found sampler: " << MI);
            return true;
          }
        }
        if (Cntr == BvhCnt()) {
          if (SIInstrInfo::isMIMG(MI) &&
              AMDGPU::getMIMGBaseOpcode(MI.getOpcode())->BVH) {
            LLVM_DEBUG(dbgs() << "  Found BVH: " << MI);
            return true;
          }
        }
      }
    }
    return false;
  };

  It->second = ComputeResult();
  return It->second;
}

/// The preheader flush optimization moves a wait from inside the loop to the
/// preheader to reduce per-iteration overhead.
///
/// Without optimization:
///   bb.0 (preheader):
///     DS_READ vgpr10    ; DS_CNT = 1
///     S_BRANCH bb.1
///
///   bb.1 (loop):
///     S_WAIT_DSCNT 0    ; Wait every iteration
///     V_ADD vgpr10      ; Use DS-loaded value
///     S_CBRANCH bb.1
///
/// With optimization:
///   bb.0 (preheader):
///     DS_READ vgpr10    ; DS_CNT = 1
///     S_WAIT_DSCNT 0    ; Wait once before entering loop
///     S_BRANCH bb.1
///
///   bb.1 (loop):
///                       ; No wait needed - DS_CNT is already 0
///     V_ADD vgpr10      ; Use DS-loaded value
///     S_CBRANCH bb.1
bool InsertWaitcnt::tryFlushInPreheader(MachineBasicBlock &PreheaderMBB) {
  assert(isLoopPreheader(PreheaderMBB, MLI) && "Expected a loop preheader");
  bool Change = false;

  // Check all successors to find loop headers where we're the preheader.
  for (MachineBasicBlock *Succ : PreheaderMBB.successors()) {
    MachineLoop *SuccLoop = MLI->getLoopFor(Succ);
    if (!SuccLoop || SuccLoop->getHeader() != Succ)
      continue;

    if (SuccLoop->getLoopPreheader() != &PreheaderMBB)
      continue;

    // Found: PreheaderMBB is a preheader for SuccLoop. Find wait instructions
    // in the loop that wait for counters not incremented by the loop. These
    // waits can be hoisted to the preheader.
    //
    // After dataflow, waits have been emitted inside the loop. We scan for
    // them and move eligible ones to the preheader. This is more reliable than
    // checking counter state, which gets cleared by drainCounters during
    // dataflow.
    SmallVector<MachineInstr *, 4> WaitsToRemove;
    SmallVector<std::pair<MachineInstr *, WaitDescriptors>, 4> WaitsToReplace;
    WaitDescriptors CAWs;

    AMDGPU::IsaVersion IV = AMDGPU::getIsaVersion(ST->getCPU());
    auto GetMaxVal = [&](const CounterType &Cntr) -> unsigned {
      if (Cntr == VmCnt() || Cntr == LoadCnt())
        return AMDGPU::getVmcntBitMask(IV);
      if (Cntr == ExpCnt())
        return AMDGPU::getExpcntBitMask(IV);
      if (Cntr == LgkmCnt() || Cntr == DsCnt())
        return AMDGPU::getLgkmcntBitMask(IV);
      if (Cntr == VsCnt() || Cntr == StoreCnt())
        return AMDGPU::getStorecntBitMask(IV);
      if (Cntr == SampleCnt())
        return AMDGPU::getSamplecntBitMask(IV);
      if (Cntr == BvhCnt())
        return AMDGPU::getBvhcntBitMask(IV);
      if (Cntr == KmCnt())
        return AMDGPU::getKmcntBitMask(IV);
      if (Cntr == XCnt())
        return AMDGPU::getXcntBitMask(IV);
      return ~0u;
    };

    for (MachineBasicBlock *LoopBB : SuccLoop->blocks()) {
      for (MachineInstr &MI : *LoopBB) {
        if (!isWaitInstr(MI))
          continue;

        // Decode the wait instruction to get counter/value pairs.
        WaitDescriptors CAWs = getCountersAndWaits(MI, *ST);
        if (CAWs.empty())
          continue;

        // Filter out counters at max value ("don't wait" values).
        WaitDescriptors FilteredWaits;
        for (const auto &[MI, Cntr, Wait] : CAWs) {
          if (Wait != GetMaxVal(Cntr))
            FilteredWaits.emplace(Cntr, Wait);
        }

        if (FilteredWaits.empty())
          continue;

        // Check each counter individually for hoistability.
        WaitDescriptors HoistableWaits;
        WaitDescriptors RemainingWaits;

        // Determine which counters are incremented by operations in the
        // preheader. We use Counter::getCountersForInstr() which returns the
        // counters that an instruction increments (i.e., the instruction
        // creates a pending operation tracked by those counters).
        SmallDenseSet<CounterType, 4> PreheaderCounters;
        for (MachineInstr &PHI : PreheaderMBB) {
          for (const CounterType &CT :
               Counter::getCountersForInstr(PHI, *ST, SchedMode))
            PreheaderCounters.insert(CT);
        }

        // A FLAT load lands on both the VMEM and DS counters. If the VMEM half
        // of this wait cannot be hoisted, the DS half must stay with it too:
        // hoisting only the DS counter would split a single FLAT load's wait
        // across the preheader and the loop. Detect a non-hoistable VMEM
        // counter up front so the DS counter can be pinned to the loop below.
        // The old pass only hoists VmCnt/DsCnt into the preheader, not ExpCnt.
        // XCnt is implied by KmCnt/LoadCnt and is never hoisted.
        // LgkmCnt/KmCnt hoisting is gated by cl::opts (default off).
        auto IsHoistable = [&](const CounterType &Cntr, unsigned Wait) {
          if (Cntr == XCnt() || Cntr == ExpCnt())
            return false;
          if (Cntr == KmCnt() && !EnableKmCntPreheaderFlush)
            return false;
          if (Cntr == LgkmCnt() && !EnableLgkmCntPreheaderFlush)
            return false;
          if (Cntr == VaVdst() || Cntr == VmVsrc())
            return false;
          return Wait == 0 && PreheaderCounters.contains(Cntr) &&
                 !loopInvalidatesPreheaderFlush(SuccLoop, Cntr);
        };
        bool VMemPinnedToLoop = false;
        for (const auto &[_MI, Cntr, Wait] : FilteredWaits)
          if ((Cntr == VmCnt() || Cntr == LoadCnt()) &&
              !IsHoistable(Cntr, Wait))
            VMemPinnedToLoop = true;

        for (const auto &[_MI2, Cntr, Wait] : FilteredWaits) {
          bool CanHoistThis = IsHoistable(Cntr, Wait);

          // Keep the DS counter in the loop alongside a non-hoistable VMEM
          // counter (see above): don't split a FLAT load's combined wait.
          if (CanHoistThis && VMemPinnedToLoop &&
              (Cntr == LgkmCnt() || Cntr == DsCnt())) {
            LLVM_DEBUG(dbgs()
                       << "Cannot hoist " << Cntr.getName()
                       << " wait: keeping with non-hoistable VMEM wait\n");
            CanHoistThis = false;
          }

          if (CanHoistThis)
            HoistableWaits.emplace(Cntr, Wait);
          else
            RemainingWaits.emplace(Cntr, Wait);
        }

        if (HoistableWaits.empty())
          continue;

        // If split hoisting is disabled, only hoist if ALL counters can be
        // hoisted.
        if (!EnableSplitWaitHoist && !RemainingWaits.empty())
          continue;

        LLVM_DEBUG(dbgs() << "Hoisting " << HoistableWaits.size()
                          << " counters to preheader BB#"
                          << PreheaderMBB.getNumber() << " from: " << MI);

        for (const auto &CAW : HoistableWaits)
          CAWs.insert(CAW);

        if (RemainingWaits.empty()) {
          WaitsToRemove.push_back(&MI);
        } else {
          // Replace original wait with one that only waits for remaining
          // counters.
          WaitsToReplace.emplace_back(&MI, std::move(RemainingWaits));
        }
      }
    }

    if (CAWs.empty())
      continue;

    // Check if preheader already has wait=0 for any of the counters we want to
    // hoist. If so, skip those counters since they're already flushed.
    SmallDenseSet<CounterType, 4> AlreadyFlushedCounters;
    for (MachineInstr &MI : PreheaderMBB) {
      if (!isWaitInstr(MI))
        continue;
      for (const auto &[_MI, Cntr, Wait] : getCountersAndWaits(MI, *ST)) {
        if (Wait == 0)
          AlreadyFlushedCounters.insert(Cntr);
      }
    }
    CAWs.erase_if([&](const WaitDescriptor &CAW) {
      return AlreadyFlushedCounters.contains(CAW.Cntr);
    });
    if (CAWs.empty())
      continue;

    // Emit hoisted wait instructions in the preheader (before terminator).
    MachineBasicBlock::instr_iterator InsertPt =
        PreheaderMBB.getFirstTerminator().getInstrIterator();
    emitWaitInstr(PreheaderMBB, InsertPt, CAWs);

    // Remove waits that were fully hoisted.
    for (MachineInstr *MI : WaitsToRemove)
      eraseWait(*MI);

    // Replace waits that were partially hoisted with waits for remaining
    // counters.
    for (auto &[MI, RemainingWaits] : WaitsToReplace) {
      MachineBasicBlock *MBB = MI->getParent();
      MachineBasicBlock::instr_iterator It = MI->getIterator();
      emitWaitInstr(*MBB, It, RemainingWaits);
      eraseWait(*MI);
    }

    Change = true;
  }
  return Change;
}

bool InsertWaitcnt::finalFixups(MachineFunction &MF) {
  bool Change = false;
  // Waits for WAIT_ASYNCMARK are emitted before it to drain the counters by the
  // time we reach WAIT_ASYNCMARK during dataflow. But tests expect them after
  // WAIT_ASYNCMARK. This moves any wait instructions immediately preceding a
  // WAIT_ASYNCMARK to immediately after it, preserving their relative order.
  // NOTE: This is not required for correctness.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() != AMDGPU::WAIT_ASYNCMARK)
        continue;
      // Scan backward to find the start of the wait run before MI.
      auto RunStart = MI.getIterator();
      for (auto It = MI.getIterator(); It != MBB.instr_begin();) {
        --It;
        if (It->isMetaInstruction())
          continue;
        if (!isWaitInstr(*It))
          break;
        RunStart = It;
      }
      if (RunStart == MI.getIterator())
        continue;
      // Forward-splice each wait in the run to after MI, preserving order.
      MachineBasicBlock::instr_iterator InsertPt = std::next(MI.getIterator());
      for (auto It = RunStart; It != MI.getIterator();) {
        auto Next = std::next(It);
        if (isWaitInstr(*It)) {
          MBB.splice(InsertPt, &MBB, It);
          Change = true;
        }
        It = Next;
      }
    }
  }
  return Change;
}

bool InsertWaitcnt::emitGlobalPrefetch(MachineFunction &MF) {
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  MachineBasicBlock &EntryBB = MF.front();
  if (MFI->isEntryFunction() && ST->hasRequiresInitialUnclausedVmem()) {
    // Hardware entrypoints must begin with a specific sequence:
    //   GLOBAL_PREFETCH_B8 V0, S[0:1] SCOPE:SCOPE_SE
    //   V_NOP
    MachineBasicBlock::iterator I = EntryBB.begin();
    BuildMI(EntryBB, I, DebugLoc(), TII->get(AMDGPU::GLOBAL_PREFETCH_B8_SADDR))
        .addReg(AMDGPU::SGPR0_SGPR1, RegState::Undef)
        .addReg(AMDGPU::VGPR0, RegState::Undef)
        .addImm(0)
        .addImm(AMDGPU::CPol::SCOPE_SE | AMDGPU::CPol::TH_RT);
    BuildMI(EntryBB, I, DebugLoc(), TII->get(AMDGPU::V_NOP_e32));
    return true;
  }
  return false;
}

bool InsertWaitcnt::run(MachineFunction &MF, MachineLoopInfo &InMLI,
                        AAResults &InAA) {
  ST = &MF.getSubtarget<GCNSubtarget>();
  TII = ST->getInstrInfo();
  TRI = ST->getRegisterInfo();
  MRI = &MF.getRegInfo();
  MLI = &InMLI;
  AA = &InAA;
  OptNone = MF.getFunction().hasOptNone() ||
            MF.getTarget().getOptLevel() == CodeGenOptLevel::None;

  // Clear caches from previous function.
  LoopInvalidatesFlushCache.clear();
  ExpertModeSetRegs.clear();
  DataFlowVisitedBlocks.clear();
  ExplicitWaits.clear();
  EndPgmInstsWithDealloc.clear();
  PreciseMemoryInstrs.clear();
  AlwaysGDSInstrs.clear();
  WaitAsyncmarkToDisambiguatedMap.clear();

  // Determine the scheduling mode either by the ExpertSchedulingModeFlag flag
  // or by the function attribute.
  SchedMode = (ExpertSchedulingModeFlag ||
               MF.getFunction()
                   .getFnAttribute("amdgpu-expert-scheduling-mode")
                   .getValueAsBool())
                  ? SchedulingMode::ExpertMode2
                  : SchedulingMode::NoExpert;

  bool Change = false;

  tryInsertWaitsAtEntryBlock(MF);

  // Remove redundant soft xcnt waits between consecutive atomic RMWs before
  // promotion, so the dataflow walk doesn't see them as hard waits.
  for (MachineBasicBlock &MBB : MF)
    Change |= removeRedundantSoftXcnts(MBB);

  // Promote soft waits to hard waits first, so applyWait can decode them.
  for (MachineBasicBlock &MBB : MF)
    Change |= promoteSoftWaitInstrs(MBB);

  if (EnableDataflow)
    Change |= insertWaitsWithDataflow(MF);
  else
    Change |= insertWaitsNoDataflow(MF);

  for (MachineBasicBlock &MBB : MF)
    Change |= combineWaitInstrs(MBB);

  // Preheader flush optimization: after waits are combined, hoist eligible
  // waits from loop bodies to preheaders. This moves waits from per-iteration
  // to once-before-loop, reducing overhead.
  for (MachineBasicBlock &MBB : MF) {
    if (isLoopPreheader(MBB, MLI))
      Change |= tryFlushInPreheader(MBB);
  }

  if (ST->isPreciseMemoryEnabled())
    Change |= insertPreciseMemoryWaits(MF);

  Change |= insertAlwaysGDSWaits();

  // Enable/disable expert scheduling mode. This runs after wait insertion so
  // the S_SETREG instructions wrap the already-emitted waits: the entry enable
  // comes before entry waits, the disable-before-return comes after the
  // return's VaVdst wait. Matches the old pass's ordering.
  Change |= trySetExpertSchedulingMode(MF);

  // Release VGPRs before any S_ENDPGM recorded as completing with outstanding
  // stores. DEALLOC_VGPRS is a GFX11+ feature, gated by a cl::opt.
  if (EnableDeallocVGPRs && ST->getGeneration() >= AMDGPUSubtarget::GFX11)
    Change |= insertDeallocVGPRs(MF);

  if (MF.getFunction().hasFnAttribute("amdgpu-expand-waitcnt-profiling"))
    Change |= expandWaitcntProfiling(MF);

  Change |= finalFixups(MF);

  Change |= emitGlobalPrefetch(MF);
  return Change;
}

bool InsertWaitcnt::expandWaitcntProfiling(MachineFunction &MF) {
  bool Changed = false;
  AMDGPU::IsaVersion IV(AMDGPU::getIsaVersion(ST->getCPU()));

  for (MachineBasicBlock &MBB : MF) {
    DenseMap<CounterType, unsigned> Outstanding;
    // Track whether each counter has any out-of-order instructions pending
    // (SMEM or stores on a shared load/store counter). If so, skip expansion.
    DenseMap<CounterType, bool> OutOfOrder;

    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (!isWaitInstr(MI) && !MI.isMetaInstruction()) {
        for (const auto &CT :
             Counter::getCountersForInstr(MI, *ST, SchedMode)) {
          ++Outstanding[CT];
          if (SIInstrInfo::isSMRD(MI) || MI.mayStore())
            OutOfOrder[CT] = true;
        }
        continue;
      }

      if (!isWaitInstr(MI))
        continue;

      WaitDescriptors CAWs = getCountersAndWaits(MI, *ST);
      if (CAWs.empty())
        continue;

      MachineBasicBlock::iterator InsertPt = MI.getIterator();

      for (const auto &[MI, Cntr, Target] : CAWs) {
        unsigned Outst = Outstanding.lookup(Cntr);
        if (Outst <= Target)
          continue;
        if (OutOfOrder.lookup(Cntr))
          continue;

        if (ST->hasExtendedWaitCounts()) {
          for (unsigned I = Outst - 1; I > Target && I != ~0u; --I)
            BuildMI(MBB, InsertPt, DebugLoc(), TII->get(getWaitOpcode(Cntr)))
                .addImm(I);
        } else {
          InstCounterType T = InstCounters::getLegacyInstCounterType(Cntr);
          if (Cntr == VsCnt()) {
            for (unsigned I = Outst - 1; I > Target && I != ~0u; --I)
              BuildMI(MBB, InsertPt, DebugLoc(),
                      TII->get(AMDGPU::S_WAITCNT_VSCNT))
                  .addReg(AMDGPU::SGPR_NULL, RegState::Undef)
                  .addImm(I);
          } else {
            for (unsigned I = Outst - 1; I > Target && I != ~0u; --I) {
              AMDGPU::Waitcnt W;
              W.set(T, I);
              BuildMI(MBB, InsertPt, DebugLoc(), TII->get(AMDGPU::S_WAITCNT))
                  .addImm(AMDGPU::encodeWaitcnt(IV, W));
            }
          }
        }
        Changed = true;
      }

      for (const auto &[MI, Cntr, Target] : CAWs) {
        unsigned &O = Outstanding[Cntr];
        if (Target < O)
          O = Target;
        if (O == 0)
          OutOfOrder.erase(Cntr);
      }
    }
  }
  return Changed;
}

void InsertWaitcnt::setSchedulingMode(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator InsertPt,
                                      bool Enable) {
  const unsigned EncodedReg = AMDGPU::Hwreg::HwregEncoding::encode(
      AMDGPU::Hwreg::ID_SCHED_MODE, AMDGPU::Hwreg::HwregOffset::Default, 2);
  MachineInstr *MI =
      BuildMI(MBB, InsertPt, DebugLoc(), TII->get(AMDGPU::S_SETREG_IMM32_B32))
          .addImm(Enable ? 2 : 0)
          .addImm(EncodedReg);
  ExpertModeSetRegs.insert(MI);
}

bool InsertWaitcnt::trySetExpertSchedulingMode(MachineFunction &MF) {
  if (SchedMode != SchedulingMode::ExpertMode2)
    return false;

  // Enable expert scheduling on function entry.
  MachineBasicBlock &EntryBB = MF.front();
  MachineBasicBlock::iterator I = EntryBB.begin();
  while (I != EntryBB.end() && I->isMetaInstruction())
    ++I;
  setSchedulingMode(EntryBB, I, true);

  // Disable before calls and re-enable after: the callee runs with its own
  // scheduling mode. Disable before non-endpgm returns to satisfy ABI.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.isCall() && !MI.isReturn()) {
        setSchedulingMode(MBB, MI.getIterator(), false);
        setSchedulingMode(MBB, std::next(MI.getIterator()), true);
      } else if (MI.isReturn()) {
        unsigned Opc = MI.getOpcode();
        bool IsEndPgm = Opc == AMDGPU::S_ENDPGM ||
                        Opc == AMDGPU::S_ENDPGM_SAVED ||
                        Opc == AMDGPU::S_ENDPGM_ORDERED_PS_DONE;
        if (!IsEndPgm)
          setSchedulingMode(MBB, MI.getIterator(), false);
      }
    }
  }

  return true;
}

// Returns the S_ENDPGM reached from \p Start by skipping only meta and wait
// instructions, or nullptr if a real instruction (or block end) intervenes. The
// scan stops at the first real instruction, so it only spans the trailing run
// of meta/wait instructions; a small cap guards against a pathological all-meta
// tail.
static MachineInstr *endPgmAfter(MachineInstr &Start) {
  // The trailing run before an endpgm is a handful of waits/meta in practice (a
  // store wait, a few counter waits, debug values). Cap the scan well above
  // that so a degenerate all-meta tail cannot make this walk the whole block.
  unsigned Budget = 16;
  for (MachineInstr &MI : make_range(std::next(Start.getIterator()),
                                     Start.getParent()->instr_end())) {
    if (MI.isMetaInstruction() || isWaitInstr(MI)) {
      if (Budget-- == 0)
        return nullptr;
      continue;
    }
    unsigned Opc = MI.getOpcode();
    if (Opc == AMDGPU::S_ENDPGM || Opc == AMDGPU::S_ENDPGM_SAVED)
      return &MI;
    return nullptr;
  }
  return nullptr;
}

void InsertWaitcnt::recordEndPgmDealloc(MachineInstr &MI) {
  // A DEALLOC_VGPRS message releases the VGPRs while letting outstanding stores
  // keep draining. It is recorded for an S_ENDPGM that is reached with pending
  // non-scratch stores (so there is something to overlap with) but no pending
  // scratch store and no store carried across a call (those must complete
  // normally / cannot be tracked). The store counter is StoreCnt on GFX12+ and
  // VsCnt on GFX11.
  const bool IsGFX12Plus = ST->getGeneration() >= AMDGPUSubtarget::GFX12;
  const Counter &StoreCntr = RTracker->getCounter(
      IsGFX12Plus ? CounterType(StoreCnt()) : CounterType(VsCnt()));

  // Identify the endpgm this read applies to. Either MI is a store-draining
  // wait coupled to a following endpgm (read the counter now, before MI drains
  // it), or MI is the endpgm itself with no preceding store wait. In both cases
  // the counter currently holds the pre-drain state the old pass evaluates.
  MachineInstr *EndPgm = nullptr;
  unsigned Opc = MI.getOpcode();
  if (Opc == AMDGPU::S_ENDPGM || Opc == AMDGPU::S_ENDPGM_SAVED) {
    // MI is the endpgm itself, with the store still pending at it (no draining
    // wait in between), e.g.:
    //   GLOBAL_STORE_DWORD ...
    //   S_ENDPGM 0            <- MI; store counter still non-empty here
    //
    // If instead a store-draining wait precedes this endpgm, that wait already
    // recorded the eligibility from the pre-drain counter; the counter is now
    // drained, so re-reading here would wrongly erase it. Only handle the
    // endpgm directly when no such coupled wait exists.
    for (MachineInstr &Prev : make_range(std::next(MI.getReverseIterator()),
                                         MI.getParent()->instr_rend())) {
      if (Prev.isMetaInstruction())
        continue;
      if (isWaitInstr(Prev)) {
        CounterType SC =
            IsGFX12Plus ? CounterType(StoreCnt()) : CounterType(VsCnt());
        if (llvm::any_of(getCountersAndWaits(Prev, *ST),
                         [&](const WaitDescriptor &CAW) {
                           return CAW.Cntr == SC && CAW.Wait == 0;
                         }))
          return; // coupled wait owns the decision
      }
      break;
    }
    EndPgm = &MI;
  } else if (isWaitInstr(MI)) {
    // MI is a store-draining wait coupled to a following endpgm (only
    // meta/waits between them), e.g.:
    //   GLOBAL_STORE_DWORD ...
    //   S_WAITCNT_VSCNT 0    <- MI; read the store counter now, before it
    //   drains S_ENDPGM 0
    //
    // Only the wait that actually drains the store counter is the coupled one;
    // a redundant second wait sees an already-empty counter and is skipped
    // here.
    if (StoreCntr.empty())
      return;
    CounterType SC =
        IsGFX12Plus ? CounterType(StoreCnt()) : CounterType(VsCnt());
    bool DrainsStore = llvm::any_of(getCountersAndWaits(MI, *ST),
                                    [&](const WaitDescriptor &CAW) {
                                      return CAW.Cntr == SC && CAW.Wait == 0;
                                    });
    if (!DrainsStore)
      return;
    EndPgm = endPgmAfter(MI);
  }
  if (!EndPgm)
    return;

  // A store carried across a call (IncomingUnknown) may be a scratch store the
  // callee left outstanding, so it must complete normally - no dealloc.
  // Likewise any pending scratch store in this function.
  bool ScratchPending =
      StoreCntr.hasIncomingUnknown() ||
      llvm::any_of(StoreCntr.instrsUnordered(), [&](const MachineInstr *S) {
        return TII->mayAccessScratch(*S);
      });
  // The flag can flip across dataflow iterations; keep the set in sync.
  if (!StoreCntr.empty() && !ScratchPending)
    EndPgmInstsWithDealloc.insert(EndPgm);
  else
    EndPgmInstsWithDealloc.erase(EndPgm);
}

bool InsertWaitcnt::insertAlwaysGDSWaits() {
  const bool IsGFX12Plus = ST->getGeneration() >= AMDGPUSubtarget::GFX12;
  CounterType DsCounter =
      IsGFX12Plus ? CounterType(DsCnt()) : CounterType(LgkmCnt());
  bool Change = !AlwaysGDSInstrs.empty();
  MachineBasicBlock *PrevMBB = nullptr;
  for (MachineInstr *MI : AlwaysGDSInstrs) {
    MachineBasicBlock &MBB = *MI->getParent();
    auto InsertPt = std::next(MI->getIterator());
    auto WaitRange = emitWaitInstr(MBB, InsertPt, {{DsCounter, 0}});
    // S_ENDPGM may terminate the wavefront before the GDS operation's
    // side effects land. Insert S_NOP if S_ENDPGM immediately follows
    // the emitted wait (skipping only meta instructions).
    auto ScanPt = WaitRange.end();
    while (ScanPt != MBB.instr_end() && ScanPt->isMetaInstruction())
      ++ScanPt;
    if (ScanPt != MBB.instr_end() && ScanPt->getOpcode() == AMDGPU::S_ENDPGM)
      BuildMI(MBB, ScanPt, MI->getDebugLoc(), TII->get(AMDGPU::S_NOP))
          .addImm(0);
    if (&MBB != PrevMBB)
      combineWaitInstrs(MBB);
    PrevMBB = &MBB;
  }
  AlwaysGDSInstrs.clear();
  return Change;
}

bool InsertWaitcnt::insertPreciseMemoryWaits(MachineFunction &MF) {
  bool Change = false;
  for (MachineInstr *MI : PreciseMemoryInstrs) {
    MachineBasicBlock &MBB = *MI->getParent();
    auto Counters = Counter::getCountersForInstr(*MI, *ST, SchedMode);
    WaitDescriptors PreciseCAWs;
    for (const CounterType &CT : Counters)
      PreciseCAWs.emplace(CT, 0);
    emitWaitInstr(MBB, std::next(MI->getIterator()), PreciseCAWs);
    Change = true;
  }
  PreciseMemoryInstrs.clear();
  if (Change)
    for (MachineBasicBlock &MBB : MF)
      Change |= combineWaitInstrs(MBB);
  return Change;
}

bool InsertWaitcnt::insertDeallocVGPRs(MachineFunction &MF) {
  if (OptNone)
    return false;

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  // In dynamic VGPR mode, release the VGPRs explicitly with S_ALLOC_VGPR 0
  // before every S_ENDPGM unconditionally (not just when stores are pending).
  // The hardware would do this on its own, but explicitly is faster.
  if (MFI->isDynamicVGPREnabled()) {
    bool Changed = false;
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : MBB) {
        unsigned Opc = MI.getOpcode();
        if (Opc == AMDGPU::S_ENDPGM || Opc == AMDGPU::S_ENDPGM_SAVED) {
          LLVM_DEBUG(dbgs() << "insertDeallocVGPRs: emitting S_ALLOC_VGPR 0 "
                               "before "
                            << MI);
          BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(AMDGPU::S_ALLOC_VGPR))
              .addImm(0);
          Changed = true;
        }
      }
    return Changed;
  }

  if (EndPgmInstsWithDealloc.empty())
    return false;

  // Otherwise send DEALLOC_VGPRS. Skip if the kernel is waveslot-limited rather
  // than VGPR-limited (deallocation would slow a short waveslot-limited
  // kernel), i.e. only do it when the function has calls or is
  // VGPR-occupancy-limited.
  if (!MF.getFrameInfo().hasCalls() &&
      ST->getOccupancyWithNumVGPRs(
          TRI->getNumUsedPhysRegs(*MRI, AMDGPU::VGPR_32RegClass),
          /*DynamicVGPRBlockSize=*/0) >= AMDGPU::IsaInfo::getMaxWavesPerEU(*ST))
    return false;

  for (MachineInstr *MI : EndPgmInstsWithDealloc) {
    LLVM_DEBUG(
        dbgs() << "insertDeallocVGPRs: emitting DEALLOC_VGPRS message"
               << (ST->requiresNopBeforeDeallocVGPRs() ? " (with S_NOP)" : "")
               << " before " << *MI);
    // Some subtargets require an S_NOP before the dealloc message.
    if (ST->requiresNopBeforeDeallocVGPRs())
      BuildMI(*MI->getParent(), MI, MI->getDebugLoc(), TII->get(AMDGPU::S_NOP))
          .addImm(0);
    BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
            TII->get(AMDGPU::S_SENDMSG))
        .addImm(AMDGPU::SendMsg::ID_DEALLOC_VGPRS_GFX11Plus);
  }
  return true;
}
