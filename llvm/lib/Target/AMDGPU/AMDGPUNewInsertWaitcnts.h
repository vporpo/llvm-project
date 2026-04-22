//===--- AMDGPUNewInsertWaitcnts.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
// Resource-based dependency tracking:
//
// 1. Track written state.
//   - if instr is of the kind that updates counters, then
//     - go over resources written (reg or mem)
//     - keep track of mapping from resource to instr
//     - insert this instr to the counter buffer
//
// 2. Determine wait counter/value for read state.
//   - go over resources read/written (reg or mem)
//     - get the instr that last wrote the resource and
//     - get the corresponding counter and the wait amount
//
// Counter Design
// ==============
// Each counter contains the instructions that update the value of the counter
// in a FIFO-like data structure.
// This allows us to know not only how much we need to wait, but also to inspect
// which instructions we are waiting for, which can be helpful not only for
// debugging but for implementing optimizations without introducing new data
// structures for tracking the instructions.
//
// We have two types of Counters: the de-duplicating and the non-dedup one.
// The reason for this is that for most counters the dependencies are carried
// through registers, so there is never a need to differentiate between
// instances of the same instruction in the counter. This is why most counters
// are de-dup ones.
//
// This is not the case for dependencies marked through ASYNCMARK/WAIT_ASYNC.
// In these cases a wait can point to a specific instruction among multiple
// identical ones in the counter. So for such counters we are using the
// non-deduplicating counters.
//
// AsyncCnt / TensorCnt have the extra challenge that they may include
// both marked (by ASYNCMARK) and non-marked instructions. To get the
// wait value for a WAIT_ASYNC(N) we need to count N marked instructions
// in AsyncCnt (or TensorCnt) and skip the unmarked ones.
//
// AsyncWait/AsyncMark Design
// ==========================
// Background: WAIT_ASYNC(N) pseudo instruction points to the Nth previous (from
// a dataflow perspective) ASYNCMARK pseudo instruction which by itself points
// to the last (from a dataflow persepctive) aync candidate pending instruction.
// Note that it is perfectly valid for a WAIT_ASYNC to point to an ASYNCMARK
// from a previous loop iteration.
//
// Upon encountering a WAIT_ASYNCMARK(N) we use
// AsyncCounter::getNthMostRecentMarkedAmong() to find the Nth most recently
// marked instruction across all async counters (AsyncCnt and TensorCnt on
// GFX1250, VmCnt/LoadCnt on older targets). We then call getWaitForNthMarked()
// on the owning counter to determine the required hardware wait value, and emit
// the appropriate wait instruction before the WAIT_ASYNCMARK.
//
// Dataflow convergence
// ====================
// The state used to check for convergence is the number of elements per
// counter. If this does not change we have converged.
// Note that checking for MIR change is not sufficient because of WAIT_ASYNC(N)
// handling: An N pointing to a prior loop iteration may not result in emitting
// a wait instruction right away, which would lead to false early convergence.
//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUNEWINSERTWAITCNTS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUNEWINSERTWAITCNTS_H

#include "AMDGPUResourceTracker.h"
#include "AMDGPUWaitcntUtils.h"
#include "llvm/CodeGen/MachinePassManager.h"

class AMDGPUTestBase_PromoteSoftWaitInstrs_PreGFX12_Test;
class AMDGPUTestBase_PromoteSoftWaitInstrs_GFX12_Test;
class AMDGPUTestBase_EmitWaitInstr_PreGFX12_Test;
class AMDGPUTestBase_EmitWaitInstr_GFX12_Test;
class AMDGPUTestBase_DecodeWaitMI_PreGFX12_Test;
class AMDGPUTestBase_DecodeWaitMI_GFX12_Test;
class AMDGPUTestBase_DecodeWaitMI_GFX1250_Test;
namespace llvm {

class MachineRegisterInfo;
class SIInstrInfo;
class SIRegisterInfo;
class GCNSubtarget;

namespace AMDGPU {

/// An WAIT_ASYNCMARK can point to either AsyncCnt or TensorCnt or both. This
/// class holds the actual counter an WAIT_ASYNCMARK is pointing to and the
/// effective N value within that counter.
class DisambiguatedWaitAsyncmark {
  CounterType CntrType;
  unsigned EffectiveN;

public:
  explicit DisambiguatedWaitAsyncmark(CounterType CntrType, unsigned EffectiveN)
      : CntrType(CntrType), EffectiveN(EffectiveN) {}
  CounterType getCounterType() const { return CntrType; }
  unsigned getEffectiveN() const { return EffectiveN; }
  WaitDescriptor getWaitDescriptor() const {
    return WaitDescriptor(CntrType, EffectiveN);
  }
  bool operator==(const DisambiguatedWaitAsyncmark &Other) const {
    return CntrType == Other.CntrType && EffectiveN == Other.EffectiveN;
  }
#ifndef NDEBUG
  void print(raw_ostream &OS) const {
    OS << CntrType.getName() << " " << EffectiveN;
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const DisambiguatedWaitAsyncmark &DWA) {
    DWA.print(OS);
    return OS;
  }
  LLVM_DUMP_METHOD void dump() const;
#endif
};

} // namespace AMDGPU

template <> struct DenseMapInfo<AMDGPU::DisambiguatedWaitAsyncmark> {
  using T = AMDGPU::DisambiguatedWaitAsyncmark;
  static T getEmptyKey() {
    return T(DenseMapInfo<AMDGPU::CounterType>::getEmptyKey(), ~0u);
  }
  static T getTombstoneKey() {
    return T(DenseMapInfo<AMDGPU::CounterType>::getTombstoneKey(), ~0u);
  }
  static unsigned getHashValue(const T &V) {
    return hash_combine(
        DenseMapInfo<AMDGPU::CounterType>::getHashValue(V.getCounterType()),
        V.getEffectiveN());
  }
  static bool isEqual(const T &LHS, const T &RHS) { return LHS == RHS; }
};

namespace AMDGPU {
/// This is the main class of the pass.
class InsertWaitcnt {
  InstCounters ICounters;
  const GCNSubtarget *ST = nullptr;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  const MachineRegisterInfo *MRI = nullptr;
  MachineLoopInfo *MLI = nullptr;
  AAResults *AA = nullptr;
  SchedulingMode SchedMode = SchedulingMode::NoExpert;
  /// Monotonically increasing sequence number shared across all ResourceTracker
  /// instances for the entire function. Ensures seqnums are globally ordered
  /// across all dataflow iterations for correct WAIT_ASYNCMARK(N) computation.
  uint64_t GlobalSeqNum = 0;

  /// If this is true then we collect the waits we emit in EmittedWaits.
  bool CollectEmittedWaits = false;

  /// True if the function has OptNone or the target optimization level is None.
  /// At -O0, preexisting soft waits (after promotion) are not removed even if
  /// redundant, matching the old pass behavior.
  bool OptNone = false;

  ResourceTracker *RTracker = nullptr;

  /// Waits inserted at function entry that should not be removed.
  DenseSet<MachineInstr *> EntryBlockWaits;

  /// Explicit wait instructions present in the input that were already hard
  /// (not soft) before promotion - e.g. a user s_waitcnt intrinsic. These are
  /// preserved verbatim and never simplified. Pass-inserted and soft-promoted
  /// waits are not in this set and remain eligible for redundancy removal.
  DenseSet<MachineInstr *> ExplicitWaits;

  /// S_SETREG instructions inserted for expert scheduling mode.
  DenseSet<MachineInstr *> ExpertModeSetRegs;

  /// S_ENDPGM instructions before which a DEALLOC_VGPRS message may be sent:
  /// those reached with outstanding non-scratch VMEM stores but no outstanding
  /// scratch store. Recorded during the block walk - at the trailing store wait
  /// coupled to the endpgm (read before it drains the counter), or at the endpgm
  /// itself when no such wait precedes it - and consumed by insertDeallocVGPRs.
  /// The flag can flip across dataflow iterations, so entries are erased when no
  /// longer eligible. Cleared per function.
  DenseSet<MachineInstr *> EndPgmInstsWithDealloc;

  /// Memory instructions that need precise-memory zero-waits. Collected during
  /// the dataflow walk (where drainCounters drains the counters) and consumed by
  /// insertPreciseMemoryWaits to emit the actual wait instructions.
  SmallVector<MachineInstr *, 16> PreciseMemoryInstrs;

  /// "Always GDS" instructions that need a zero-wait on the DS counter.
  /// Collected during the dataflow walk and consumed by insertAlwaysGDSWaits.
  SmallVector<MachineInstr *, 4> AlwaysGDSInstrs;

  /// Cache for loopInvalidatesPreheaderFlush results to avoid rescanning loops.
  /// Cleared at the start of each function.
  mutable DenseMap<std::pair<const MachineLoop *, CounterType>, bool>
      LoopInvalidatesFlushCache;

  /// Blocks that have been visited during dataflow iteration.
  /// Used to process WAIT_ASYNCMARK only on first visit, avoiding incorrect
  /// wait values from VmCnt positions corrupted by drainCounters.
  DenseSet<MachineBasicBlock *> DataFlowVisitedBlocks;

  /// True when processing a block for the first time during dataflow.
  /// Set before calling insertWaitsInBlock in dataflow mode.
  bool IsFirstVisit = true;

  /// Collect emitted waits for use by external tools.
  mutable DenseSet<MachineInstr *> EmittedWaits;

  /// Maps each WAIT_ASYNCMARK instruction to its disambiguated object.
  DenseMap<MachineInstr *, SmallDenseSet<DisambiguatedWaitAsyncmark, 1>>
      WaitAsyncmarkToDisambiguatedMap;

#ifndef NDEBUG
  LLVM_DUMP_METHOD void dumpDisambiguatedWaitAsyncmarks() const;
#endif

  /// WAIT_ASYNCMARKs may point to either an AsyncCnt or TensorCnt entry. This
  /// function tries to disambiguate WAIT_ASYNCMARK \p MI if the counters have
  /// been filled in adequately due to dataflow. This populates
  /// WaitAsyncmarkToDisambiguatedMap.
  void tryDisambiguateWaitAsyncmark(MachineInstr &MI);

  /// Erase \p MI from the MIR and, if collecting, remove it from EmittedWaits.
  void eraseWait(MachineInstr &MI) {
    if (CollectEmittedWaits)
      EmittedWaits.erase(&MI);
    MI.eraseFromParent();
  }

  /// Decodes a wait instruction \p WaitMI into the counters it waits on and
  /// their wait values. Counters that the instruction does not wait on (a
  /// combined S_WAITCNT field at its "no wait" bit-mask) are omitted.
  static WaitDescriptors getCountersAndWaits(const MachineInstr &WaitMI,
                                               const GCNSubtarget &ST);

  /// Emits wait instructions for the given counters before \p InsertPt.
  /// Returns an iterator range covering all emitted instructions in
  /// top-to-bottom program order (may be empty if no waits were needed).
  /// Uses instr_iterator to support insertion inside bundles.
  ///
  /// \p ForMI is the instruction that requires the waits. When
  /// CollectEmittedWaits is true, it is used to identify pre-existing wait
  /// instructions in [\p InsertPt, \p ForMI) so that combined replacements
  /// which merely re-emit already-present coverage are not attributed to the
  /// pass in EmittedWaits. Pass \p InsertPt when there are no pre-existing
  /// waits between the insertion point and the requiring instruction.
  // TODO: ForMI could be dropped if we update combineWaitInstrs() to emit the
  // new waits after the pre-existing ones, but that would change the output
  // compared to the old pass.
  iterator_range<MachineBasicBlock::instr_iterator>
  emitWaitInstr(MachineBasicBlock &MBB, MachineBasicBlock::instr_iterator InsertPt,
                const WaitDescriptors &CAWs,
                MachineBasicBlock::instr_iterator ForMI) const;

  iterator_range<MachineBasicBlock::instr_iterator>
  emitWaitInstr(MachineBasicBlock &MBB, MachineBasicBlock::instr_iterator InsertPt,
                const WaitDescriptors &CAWs) const {
    return emitWaitInstr(MBB, InsertPt, CAWs, InsertPt);
  }

  /// Returns the wait insert point for wait. This is MI.getIterator() in most
  /// cases but it could also be an earlier point if we decide that it is better
  /// for performance.
  MachineBasicBlock::instr_iterator
  getWaitInsertPoint(MachineInstr &MI,
                     const WaitDescriptors &FinalCAWs) const;

  /// Goes over MI's operands, finds its register dependencies and returns the
  /// waits needed based on the counter that is tracking the definition/use of
  /// each dependent register.
  WaitDescriptors getWaitsBasedOnRegDeps(MachineInstr &MI) const;

  /// Add to \p CAWs waits that are not related to dependencies, like for
  /// example for the special needs of instruction kinds or to handle bugs.
  void addNonDepCAWs(WaitDescriptors &CAWs, MachineInstr &MI) const;

  /// Reduce the number of waits in \p CAWs. This may also update the counters
  /// that need to be updated.
  void removeRedundantWaits(WaitDescriptors &CAWs, MachineInstr &MI,
                            bool &Change) const;

  /// Checks MI's dependencies and inserts waits accordingly.
  bool insertWaitsFor(MachineInstr &MI) const;

  /// Returns a WaitDescriptors with wait 0 for all counters.
  WaitDescriptors getAllZeroWaitCAWs() const;

  /// Outcome of simplifying a preexisting wait against current tracker state.
  enum class WaitSimplifyResult {
    Kept,    ///< Unchanged: every counter is still needed.
    Trimmed, ///< Some redundant counters dropped; the instruction remains.
    Erased,  ///< Every counter redundant; the instruction was erased.
  };

  /// Drops redundant counters (those with no pending operations) from \p MI,
  /// erasing it if all are redundant.
  WaitSimplifyResult simplifyRedundantWait(MachineInstr &MI);

  /// Process a preexisting wait instruction: remove if redundant, otherwise
  /// update the tracker. Handles soft xcnt fence deferral via
  /// \p DeferredSoftXCnt. Returns true if the MIR was modified.
  bool processExistingWait(MachineInstr &MI, bool IsGFX12Plus,
                           MachineInstr *&DeferredSoftXCnt);

  /// Inserts waits for all counters before S_BARRIER on subtargets that require
  /// it. Also clears the tracker since the barrier synchronizes all operations.
  bool insertWaitsBeforeBarrier(MachineInstr &MI);

  /// Handles pseudo waitcnt instructions that need to be consumed by this pass.
  /// Returns true if MIR was modified.
  bool tryHandlePseudoWaitcnt(MachineInstr &MI);

  /// Inserts waits for the entry block.
  bool tryInsertWaitsAtEntryBlock(MachineFunction &MF);

  /// Combines consecutive wait instructions into a single instruction.
  bool combineWaitInstrs(MachineBasicBlock &MBB);

  /// Remove redundant soft xcnt waits between consecutive atomic RMWs.
  bool removeRedundantSoftXcnts(MachineBasicBlock &MBB);

  /// Promotes soft wait instructions to their hard equivalents.
  bool promoteSoftWaitInstrs(MachineBasicBlock &MBB);

  /// Tracker updates that need to run after we track(MI).
  bool postTrackUpdates(MachineInstr &MI);

  /// Process a single basic block: insert waits, track instructions, handle
  /// preexisting waits. Returns true if any changes were made.
  bool insertWaitsInBlock(MachineBasicBlock &MBB);

  /// Emit flush waits at the end of a preheader block (before terminator).
  /// This ensures pending operations from the preheader complete before
  /// the loop starts, preventing dataflow merge issues with backedge state.
  bool tryEmitPreheaderFlush(MachineBasicBlock &MBB);

  /// Process all blocks using dataflow analysis to propagate counter state
  /// across block boundaries. Iterates until fixed-point convergence.
  /// Returns true if any changes were made.
  bool insertWaitsWithDataflow(MachineFunction &MF);

  /// Process all blocks without dataflow analysis. Conservatively waits for
  /// all counters at each block boundary. Returns true if any changes were
  /// made.
  bool insertWaitsNoDataflow(MachineFunction &MF);

  /// Check if the loop contains operations that invalidate preheader flush
  /// for the given counter. For load counters (DsCnt, LoadCnt), stores
  /// invalidate because we need to wait for them inside the loop. Loads
  /// don't invalidate because they complete in order (preheader load
  /// completes before loop loads) and prefetch patterns use loop loads
  /// for next iteration only.
  bool loopInvalidatesPreheaderFlush(const MachineLoop *Loop,
                                     const CounterType &Cntr) const;

  /// Determine if counter should be flushed in the preheader based on
  /// generation-specific rules.
  bool shouldFlushCounterBasedOnGeneration(const CounterType &Cntr,
                                           const MachineLoop *Loop) const;

  /// Flush counters at the end of a preheader block (before terminator).
  /// Flushes all counters whose pending operations have results live-in to
  /// the loop and whose counters won't be incremented inside the loop.
  bool tryFlushInPreheader(MachineBasicBlock &MBB);

  /// Emit S_SETREG_IMM32_B32 to enable or disable expert scheduling mode.
  void setSchedulingMode(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator InsertPt, bool Enable);

  /// Enable expert mode at function entry, disable before returns.
  bool trySetExpertSchedulingMode(MachineFunction &MF);

  /// In precise memory mode, emit a zero-waitcnt after every memory operation.
  /// Only waits for counters the instruction actually uses. Runs as a post-pass
  /// after dataflow converges.
  bool insertPreciseMemoryWaits(MachineFunction &MF);

  /// Emit zero-waits after "always GDS" instructions collected during the
  /// dataflow walk. Inserts S_NOP before S_ENDPGM when needed.
  bool insertAlwaysGDSWaits();

  /// During the block walk, record whether the S_ENDPGM reached after \p MI may
  /// complete with outstanding stores, reading the store counter before \p MI (a
  /// trailing store wait coupled to the endpgm) drains it. Also handles an
  /// endpgm with no preceding store wait. Updates EndPgmInstsWithDealloc.
  void recordEndPgmDealloc(MachineInstr &MI);

  /// On GFX11+, release VGPRs before the S_ENDPGM instructions recorded in
  /// EndPgmInstsWithDealloc, by emitting S_SENDMSG(DEALLOC_VGPRS) (or
  /// S_ALLOC_VGPR 0 in dynamic VGPR mode).
  bool insertDeallocVGPRs(MachineFunction &MF);

  /// Expand wait instructions into staircases for PC-sampling profiling.
  /// Instead of a single waitcnt(target), emit waitcnt(N-1), waitcnt(N-2),
  /// ..., waitcnt(target) so profilers can identify which operation stalls.
  bool expandWaitcntProfiling(MachineFunction &MF);

  /// Changes in the code that don't depend on anything else and can be done as
  /// late as possible.
  bool finalFixups(MachineFunction &MF);

  // Global prefetch instructions insertions copied verbatim from the old pass.
  // TODO: Shouldn't this be in a separate pass ?
  bool emitGlobalPrefetch(MachineFunction &MF);

  friend class ::AMDGPUTestBase_PromoteSoftWaitInstrs_PreGFX12_Test;
  friend class ::AMDGPUTestBase_PromoteSoftWaitInstrs_GFX12_Test;
  friend class ::AMDGPUTestBase_EmitWaitInstr_PreGFX12_Test;
  friend class ::AMDGPUTestBase_EmitWaitInstr_GFX12_Test;
  friend class ::AMDGPUTestBase_DecodeWaitMI_PreGFX12_Test;
  friend class ::AMDGPUTestBase_DecodeWaitMI_GFX12_Test;
  friend class ::AMDGPUTestBase_DecodeWaitMI_GFX1250_Test;

public:
  InsertWaitcnt(bool CollectEmittedWaits = false)
      : CollectEmittedWaits(CollectEmittedWaits) {}
  bool run(MachineFunction &MF, MachineLoopInfo &MLI, AAResults &AA);
  /// \Returns the range of emitted waits (if constructed with
  /// CollectEmittedWaits=true).
  auto getEmittedWaits() const {
    return make_range(EmittedWaits.begin(), EmittedWaits.end());
  }
};

} // namespace AMDGPU

class AMDGPUNewInsertWaitcntsPass
    : public PassInfoMixin<AMDGPUNewInsertWaitcntsPass> {
  AMDGPU::InsertWaitcnt IW;

public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUNEWINSERTWAITCNTS_H
