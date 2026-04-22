//===- AMDGPUResourceTracker.cpp - Resource Dependency Tracking ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Resource-based dependency tracking for AMDGPU wait insertion.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUResourceTracker.h"
#include "AMDGPU.h"
#include "AMDGPUWaitcntUtils.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

static cl::opt<bool> EnableFlatSameCounterOpt(
    "amdgpu-flat-same-counter-opt", cl::Hidden,
    cl::desc("Enable optimization for pure FLAT when DstMI uses same counter "
             "(provides implicit in-order semantics via counter FIFO)"),
    cl::init(false));

static cl::opt<bool> EnableFlatInOrderOpt(
    "amdgpu-flat-in-order-opt", cl::Hidden,
    cl::desc("Enable optimization for pure FLAT on gfx10+ where FLAT completes "
             "in-order with hasFlatLgkmVMemCountInOrder"),
    cl::init(false));

static cl::opt<bool> EnableCrossCounterFlatPending(
    "amdgpu-cross-counter-flat-pending", cl::Hidden,
    cl::desc("Check the sibling counter for pending FLATs when applying the "
             "pre-gfx10 early completion workaround, matching the old pass"),
    cl::init(true));

static cl::opt<bool> EnableSmemVmemDepAA(
    "amdgpu-smem-vmem-dep-aa", cl::Hidden,
    cl::desc("Use alias analysis for SMEM/VMEM memory dependency detection "
             "instead of exact Value* pointer match"),
    cl::init(false));

static cl::opt<bool> EnableStrictMixedFlatCheck(
    "amdgpu-strict-mixed-flat-check", cl::Hidden,
    cl::desc("When FLAT is pending with VMEM-only or DS instructions, force "
             "wait for all (cnt 0) instead of position-based waits. This is "
             "the correct behavior but differs from the old pass."),
    cl::init(false));

// TODO: This is only used for replicating the old pass behavior. Should be
// removed once we migrate to the new pass.
static cl::opt<unsigned> MaxAsyncMarks(
    "amdgpu-max-async-marks", cl::Hidden,
    cl::desc("Maximum number of ASYNCMARK positions tracked for WAIT_ASYNCMARK(N). "
             "N is clamped to MaxAsyncMarks-1, replicating the old si-insert-waitcnts "
             "pass behavior which capped AsyncMarks at 16 entries. "
             "Set to 0 to disable clamping (unlimited)."),
    cl::init(16));

using namespace llvm::AMDGPU;

#define DEBUG_TYPE "amdgpu-resource-tracker"

// Enumerate different types of result-returning VMEM operations. Although
// s_waitcnt orders them all with a single vmcnt counter, in the absence of
// s_waitcnt only instructions of the same VmemType are guaranteed to write
// their results in order -- so there is no need to insert an s_waitcnt between
// two instructions of the same type that write the same vgpr.
enum VmemType {
  // BUF instructions and MIMG instructions without a sampler.
  VMEM_NOSAMPLER,
  // MIMG instructions with a sampler.
  VMEM_SAMPLER,
  // BVH instructions
  VMEM_BVH,
};

// Returns true if the instruction only updates VMEM counters (LOAD_CNT,
// SAMPLE_CNT, BVH_CNT).
static bool updateVMCntOnly(const MachineInstr &Inst) {
  return (SIInstrInfo::isVMEM(Inst) && !SIInstrInfo::isFLAT(Inst)) ||
         SIInstrInfo::isFLATGlobal(Inst) || SIInstrInfo::isFLATScratch(Inst);
}

static VmemType getVmemType(const MachineInstr &Inst) {
  assert(updateVMCntOnly(Inst));
  if (!SIInstrInfo::isImage(Inst))
    return VMEM_NOSAMPLER;
  const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Inst.getOpcode());
  const AMDGPU::MIMGBaseOpcodeInfo *BaseInfo =
      AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);

  if (BaseInfo->BVH)
    return VMEM_BVH;

  // We have to make an additional check for isVSAMPLE here since some
  // instructions don't have a sampler, but are still classified as sampler
  // instructions for the purposes of e.g. waitcnt.
  if (BaseInfo->Sampler || BaseInfo->MSAA || SIInstrInfo::isVSAMPLE(Inst))
    return VMEM_SAMPLER;

  return VMEM_NOSAMPLER;
}

/// Returns true if the instruction has Point Sample Acceleration, which can
/// change its effective VmemType from SAMPLER to NOSAMPLER.
static bool hasPointSampleAccel(const MachineInstr &Inst,
                                const GCNSubtarget &ST) {
  if (!ST.hasPointSampleAccel() || !SIInstrInfo::isMIMG(Inst))
    return false;
  const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Inst.getOpcode());
  const AMDGPU::MIMGBaseOpcodeInfo *BaseInfo =
      AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);
  return BaseInfo->PointSampleAccel;
}

/// Returns true if \p Cntr is a VMEM counter (VmCnt, LoadCnt, SampleCnt, BvhCnt).
static bool isVmemCounter(const CounterType &Cntr) {
  return Cntr == VmCnt() || Cntr == LoadCnt() || Cntr == SampleCnt() ||
         Cntr == BvhCnt();
}

/// Returns true if a WAW hazard between \p SrcMI and \p DstMI can be skipped
/// because hardware guarantees in-order VGPR writes for VMEM instructions.
/// This requires hasVmemWriteVgprInOrder, DstMI being VMEM-only, and either:
/// - SrcMI is not VMEM-only (e.g., FLAT), or
/// - Both have the same VmemType AND neither has Point Sample Acceleration
///   issues (PSA can cause SAMPLER instructions to complete as NOSAMPLER).
static bool canSkipWawHazard(const MachineInstr &SrcMI,
                             const MachineInstr &DstMI,
                             const GCNSubtarget &ST) {
  if (!ST.hasVmemWriteVgprInOrder() || !updateVMCntOnly(DstMI))
    return false;
  // SrcMI is not VMEM-only (e.g., FLAT): safe to skip because FLAT doesn't
  // participate in VMEM in-order write guarantees.
  if (!updateVMCntOnly(SrcMI))
    return true;
  // Both are VMEM-only: check VmemType and Point Sample Acceleration.
  VmemType SrcVT = getVmemType(SrcMI);
  VmemType DstVT = getVmemType(DstMI);
  // Different VmemTypes can complete out of order.
  if (SrcVT != DstVT)
    return false;
  // Point Sample Acceleration: if SrcMI has PSA, it can complete as NOSAMPLER,
  // breaking same-type assumption. If DstMI has PSA and SrcMI is non-NOSAMPLER,
  // DstMI could be accelerated and complete before SrcMI.
  bool SrcHasPSA = hasPointSampleAccel(SrcMI, ST);
  bool DstHasPSA = hasPointSampleAccel(DstMI, ST);
  if (SrcHasPSA || (DstHasPSA && SrcVT != VMEM_NOSAMPLER))
    return false;
  return true;
}

/// Returns true if \p MI keeps its VGPR source operand(s) live on the export
/// counter (ExpCnt): the instruction reads those registers some time after it is
/// issued and does not decrement ExpCnt until it has done so. Overwriting such a
/// source register is a write-after-read hazard that needs a wait for ExpCnt to
/// reach 0.
///
/// This covers:
///  - EXP (exports) and LDSDIR (LDS direct loads), which read their VGPR sources
///    via the export pathway and hold them on ExpCnt.
///  - On pre-SEA_ISLANDS targets (gfx6), VMEM/FLAT stores (and return atomics),
///    whose data source is held on ExpCnt because those targets have no
///    dedicated store counter.
///
/// Note this only tells you that *some* source is held on ExpCnt; which operands
/// those are differs by instruction (all VGPR uses for EXP/LDSDIR, only the data
/// operand for a store), so callers that need the specific operands handle that
/// distinction themselves.
static bool srcsHeldOnExpCnt(const MachineInstr &MI, const GCNSubtarget &ST) {
  if (SIInstrInfo::isEXP(MI) || SIInstrInfo::isLDSDIR(MI))
    return true;
  if (!ST.vmemWriteNeedsExpWaitcnt())
    return false;
  if (!SIInstrInfo::isVMEM(MI) && !SIInstrInfo::isFLAT(MI))
    return false;
  return MI.mayStore() || SIInstrInfo::isAtomicRet(MI);
}

/// Returns true if \p MI holds the specific register \p Reg on the export
/// counter (ExpCnt) as a source read, so that a later overwrite of \p Reg is a
/// write-after-read hazard ordered by ExpCnt. This is the per-register refinement
/// of srcsHeldOnExpCnt: track() records every read register, and this decides
/// which of them ExpCnt actually orders.
///  - EXP/LDSDIR hold their VGPR source operands (not implicit $exec/$m0 reads).
///  - A gfx6 store / return atomic holds only its data operand; its address
///    inputs (buffer descriptor, offset) are read but not ordered by ExpCnt.
static bool regHeldOnExpCnt(const MachineInstr &MI, MCRegister Reg,
                            const GCNSubtarget &ST) {
  if (!srcsHeldOnExpCnt(MI, ST))
    return false;
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  if (SIInstrInfo::isEXP(MI) || SIInstrInfo::isLDSDIR(MI))
    return TRI->isVGPRPhysReg(Reg);
  // gfx6 store / return atomic: only the data operand is held on ExpCnt. Use
  // isSubRegisterEq (not regsOverlap) so that when the address register pair
  // contains the data register as a sub-register (e.g. vaddr=$vgpr3_vgpr4,
  // vdata=$vgpr4), querying the address pair does NOT report it as held — only
  // the data register itself (or a sub-register of it) qualifies.
  const MachineOperand *DataOp = TII->getNamedOperand(MI, AMDGPU::OpName::vdata);
  if (!DataOp)
    DataOp = TII->getNamedOperand(MI, AMDGPU::OpName::data);
  return DataOp && DataOp->getReg().isValid() &&
         TRI->isSubRegisterEq(DataOp->getReg(), Reg);
}

/// Returns true if \p MI is a GDS (Global Data Share) instruction.
/// GDS instructions include DS_ORDERED_COUNT, DS_ADD_GS_REG_RTN, GWS ops,
/// and any DS instruction with the gds modifier bit set.
static bool isGDS(const MachineInstr &MI, const SIInstrInfo &TII) {
  return TII.isAlwaysGDS(MI.getOpcode()) ||
         TII.hasModifiersSet(MI, AMDGPU::OpName::gds);
}

#ifndef NDEBUG
void TrackedInstr::print(raw_ostream &OS) const {
  if (isUnknown())
    OS << "<unknown>";
  else
    OS << *getMI();
}

void TrackedInstr::dump() const { print(dbgs()); }
#endif

void InstrBuffer::pushBack(ArrayRef<MachineInstr *> MIs) {
  std::optional<unsigned> NewTopIdxInternal;
  for (MachineInstr *MI : MIs) {
    auto It = InstrToIdxMap.find(MI);
    if (It != InstrToIdxMap.end()) {
      // If the instruction already exists "move" it to its new index.
      unsigned OldIdx = It->second;
      unsigned LastIdx = TopIdxInternal - 1;
      if (OldIdx == LastIdx)
        continue;
      std::swap(IdxToInstrMap[OldIdx], IdxToInstrMap[LastIdx]);
      for (TrackedInstr Swapped : IdxToInstrMap[OldIdx])
        InstrToIdxMap[Swapped] = OldIdx;
      for (TrackedInstr Swapped : IdxToInstrMap[LastIdx])
        InstrToIdxMap[Swapped] = LastIdx;
      continue;
    }
    assert(TopIdxInternal <
               std::numeric_limits<decltype(TopIdxInternal)>::max() - 1 &&
           "TopIdxInternal overflow or TombstoneKey collision!");
    InstrToIdxMap[MI] = TopIdxInternal;
    IdxToInstrMap[TopIdxInternal].insert(MI);
    NewTopIdxInternal = TopIdxInternal + 1;
    Hash = getHash<Action::Add>(MI);
  }
  if (NewTopIdxInternal)
    TopIdxInternal = *NewTopIdxInternal;
}

void InstrBuffer::popFront(unsigned NumIndices) {
  for (unsigned IdxToRemove :
       seq<unsigned>(BottomIdxInternal, BottomIdxInternal + NumIndices)) {
    auto It = IdxToInstrMap.find(IdxToRemove);
    if (It == IdxToInstrMap.end())
      continue;
    for (TrackedInstr TI : make_early_inc_range(It->second)) {
      InstrToIdxMap.erase(TI);
      Hash = getHash<Action::Remove>(TI);
    }
    IdxToInstrMap.erase(It);
  }
  BottomIdxInternal += NumIndices;
  assert(BottomIdxInternal <= TopIdxInternal && "Broken indices!");
}

void InstrBuffer::removeIf(function_ref<bool(const TrackedInstr &)> Pred) {
  SmallVector<TrackedInstr, 4> ToRemove;
  for (const auto &[TI, InternalIdx] : InstrToIdxMap)
    if (Pred(TI))
      ToRemove.push_back(TI);
  for (TrackedInstr TI : ToRemove) {
    unsigned InternalIdx = InstrToIdxMap[TI];
    auto IdxIt = IdxToInstrMap.find(InternalIdx);
    if (IdxIt != IdxToInstrMap.end()) {
      IdxIt->second.erase(TI);
      if (IdxIt->second.empty())
        IdxToInstrMap.erase(IdxIt);
    }
    InstrToIdxMap.erase(TI);
    Hash = getHash<Action::Remove>(TI);
  }
  // Trim leading gaps so BottomIdxInternal points at a live index (or equals
  // TopIdxInternal when the buffer is now empty).
  while (BottomIdxInternal < TopIdxInternal &&
         !IdxToInstrMap.count(BottomIdxInternal))
    ++BottomIdxInternal;
  // Trim trailing gaps.
  while (TopIdxInternal > BottomIdxInternal &&
         !IdxToInstrMap.count(TopIdxInternal - 1))
    --TopIdxInternal;
}

bool InstrBuffer::operator==(const InstrBufferBase &OtherBase) const {
  const InstrBuffer *Other = dyn_cast<InstrBuffer>(&OtherBase);
  assert(Other && "OtherBase is expected to be an InstrBuffer!");
  if (Hash != Other->Hash)
    return false;
  if (InstrToIdxMap.size() != Other->InstrToIdxMap.size())
    return false;
  for (const auto &[TI, InternalIdx] : InstrToIdxMap) {
    auto It = Other->InstrToIdxMap.find(TI);
    if (It == Other->InstrToIdxMap.end())
      return false;
    unsigned ThisUserIdx = InternalIdx - BottomIdxInternal;
    unsigned OtherUserIdx = It->second - Other->BottomIdxInternal;
    if (ThisUserIdx != OtherUserIdx)
      return false;
  }
  return true;
}

void InstrBuffer::merge(const InstrBufferBase &OtherBase) {
  // We can only merge with same type of buffer so cast to InstrBuffer is safe.
  const InstrBuffer &Other = static_cast<const InstrBuffer &>(OtherBase);

  // When merging we align the buffers such that their most recent entries
  // (the ones with highest indices) get merged on the same index. For example:
  //
  //  This    Other  after merge:  This
  //  0. I1   0. I2                0.I2
  //          1. I3                1.I1 I3

  unsigned MergedTopIdxInternal = std::max(TopIdxInternal, Other.TopIdxInternal);
  unsigned ThisOffset = MergedTopIdxInternal - TopIdxInternal;
  unsigned OtherOffset = MergedTopIdxInternal - Other.TopIdxInternal;
  unsigned MergedBotIdxInternal = MergedTopIdxInternal;

  // TODO: Try to reuse one the existing maps.
  DenseMap<TrackedInstr, unsigned> MergedInstrToIdxMap;
  DenseMap<unsigned, SmallDenseSet<TrackedInstr, 2>> MergedIdxToInstrMap;
  for (auto [ThisTI, ThisIdxInternal] : InstrToIdxMap) {
    unsigned MergedThisIdx = ThisIdxInternal + ThisOffset;
    MergedIdxToInstrMap[MergedThisIdx].insert(ThisTI);
    MergedInstrToIdxMap[ThisTI] = MergedThisIdx;
  }
  for (auto [OtherTI, OtherIdxInternal] : Other.InstrToIdxMap) {
    unsigned MergedOtherIdx = OtherIdxInternal + OtherOffset;
    auto It = MergedInstrToIdxMap.find(OtherTI);
    if (It != MergedInstrToIdxMap.end()) {
      // OtherTI already exists. In this case we keep the one with the highest
      // index.
      unsigned ExistingIdx = It->second;
      if (ExistingIdx < MergedOtherIdx) {
        // Replace existing with OtherTI.
        MergedInstrToIdxMap.erase(It);
        MergedInstrToIdxMap[OtherTI] = MergedOtherIdx;
        auto &InstrSet = MergedIdxToInstrMap[ExistingIdx];
        InstrSet.erase(OtherTI);
        if (InstrSet.empty())
          MergedIdxToInstrMap.erase(ExistingIdx);
      }
      else {
        continue;
      }
    }
    MergedInstrToIdxMap[OtherTI] = MergedOtherIdx;
    MergedIdxToInstrMap[MergedOtherIdx].insert(OtherTI);
  }

  // TODO: Avoid this second loop just to get the BottomIdxInternal.
  for (auto [MergedIdxInternal, TI] : MergedIdxToInstrMap)
    MergedBotIdxInternal = std::min(MergedBotIdxInternal, MergedIdxInternal);

  InstrToIdxMap = std::move(MergedInstrToIdxMap);
  IdxToInstrMap = std::move(MergedIdxToInstrMap);
  TopIdxInternal = MergedTopIdxInternal;
  BottomIdxInternal = MergedBotIdxInternal;
}

SmallDenseSet<TrackedInstr, 2> InstrBuffer::back() const {
  if (empty())
    return {};
  return IdxToInstrMap.at(TopIdxInternal - 1);
}

const SmallDenseSet<TrackedInstr, 2> &
InstrBuffer::getNthFromEnd(unsigned N) const {
  static const SmallDenseSet<TrackedInstr, 2> Empty;
  if (N >= getTopIndex())
    return Empty;
  auto It = IdxToInstrMap.find(TopIdxInternal - N - 1);
  if (It == IdxToInstrMap.end())
    return Empty;
  return It->second;
}

void InstrBuffer::clear() {
  InstrToIdxMap.clear();
  IdxToInstrMap.clear();
  BottomIdxInternal = 0;
  TopIdxInternal = 0;
  Hash = 0;
}

const SmallDenseSet<TrackedInstr, 2> AsyncBuffer::EmptySet;

MachineInstr *AsyncBuffer::getLastMarkedInstr() const {
  if (SeqNumToInternalIdxs.empty())
    return nullptr;
  for (unsigned InternalIdx : SeqNumToInternalIdxs.at(MaxMarkedSeqNum)) {
    auto SlotIt = IdxToInstrMap.find(InternalIdx);
    if (SlotIt == IdxToInstrMap.end())
      return nullptr;
    for (TrackedInstr TI : SlotIt->second)
      if (!TI.isUnknown())
        return TI.getMI();
  }
  return nullptr;
}

/// Returns true if \p MI is an async LDS DMA instruction (i.e. a candidate
/// for being recorded at an ASYNCMARK position).
static bool isAsyncDMACandidate(const MachineInstr &MI) {
  if (SIInstrInfo::usesASYNC_CNT(MI) || SIInstrInfo::usesTENSOR_CNT(MI))
    return true;
  if (SIInstrInfo::isLDSDMA(MI))
    return true;
  int IsAsyncIdx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::IsAsync);
  return IsAsyncIdx >= 0 && MI.getOperand(IsAsyncIdx).getImm();
}

bool AsyncBuffer::isAsyncMarked(const MachineInstr *MI) {
  if (!MI || !isAsyncDMACandidate(*MI))
    return false;
  // Scan forward from MI looking for ASYNCMARK. ASYNCMARK marks all in-flight
  // async DMA streams simultaneously, so non-async instructions and async DMA
  // from different counter types may appear between MI and the ASYNCMARK.
  // Stop (return false) if we encounter another async DMA of the same counter
  // type as MI — that belongs to a different batch and breaks the association.
  bool MIIsAsync = SIInstrInfo::usesASYNC_CNT(*MI);
  bool MIIsTensor = SIInstrInfo::usesTENSOR_CNT(*MI);
  auto SameCounterType = [&](const MachineInstr &Other) {
    if (MIIsAsync || SIInstrInfo::usesASYNC_CNT(Other))
      return MIIsAsync == SIInstrInfo::usesASYNC_CNT(Other);
    if (MIIsTensor || SIInstrInfo::usesTENSOR_CNT(Other))
      return MIIsTensor == SIInstrInfo::usesTENSOR_CNT(Other);
    return true; // both are the same legacy LDS DMA type
  };
  for (const MachineInstr *Next = MI->getNextNode(); Next;
       Next = Next->getNextNode()) {
    if (Next->getOpcode() == AMDGPU::ASYNCMARK)
      return true;
    // Another async DMA of the same type means a new batch — stop here.
    if (isAsyncDMACandidate(*Next) && SameCounterType(*Next))
      return false;
  }
  return false;
}

void AsyncBuffer::pushBack(ArrayRef<MachineInstr *> MIs) {
  if (MIs.empty())
    return;
  assert(TopIdxInternal <
             std::numeric_limits<decltype(TopIdxInternal)>::max() - 1 &&
         "TopIdxInternal overflow!");
  for (MachineInstr *MI : MIs) {
    TrackedInstr TI(MI);
    IdxToInstrMap[TopIdxInternal].insert(TI);
    InstrToIdxsMap[TI].insert(TopIdxInternal);
    Hash = getHash<Action::Add>(TI);
    if (AsyncBuffer::isAsyncMarked(MI)) {
      uint64_t Seq = GlobalSeqNum++;
      SeqNumToInternalIdxs[Seq].insert(TopIdxInternal);
      InternalIdxToSeqNums[TopIdxInternal].insert(Seq);
      if (SeqNumToInternalIdxs.size() == 1)
        MinMarkedSeqNum = Seq;
      MaxMarkedSeqNum = Seq;
    }
  }
  ++TopIdxInternal;
  MergedAndCapped = false;
}

unsigned AsyncBuffer::getTopIndex() const {
  return TopIdxInternal - BottomIdxInternal;
}

unsigned AsyncBuffer::getIndex(MachineInstr *MI) const {
  auto It = InstrToIdxsMap.find(TrackedInstr(MI));
  assert(It != InstrToIdxsMap.end() && "MI not in AsyncBuffer");
  const auto &Indices = It->second;
  assert(!Indices.empty() && "Expected non empty indices vector!");
  // Find the maximum index (i.e., the most recent).
  unsigned MaxIndex = *max_element(Indices);
  return MaxIndex - BottomIdxInternal;
}

void AsyncBuffer::popFront(unsigned NumIndices) {
  for (unsigned IdxToRemove :
       seq(BottomIdxInternal, BottomIdxInternal + NumIndices)) {
    auto It = IdxToInstrMap.find(IdxToRemove);
    if (It == IdxToInstrMap.end())
      continue;
    for (TrackedInstr TI : It->second) {
      Hash = getHash<Action::Remove>(TI);
      auto &Idxs = InstrToIdxsMap[TI];
      Idxs.erase(IdxToRemove);
      if (Idxs.empty())
        InstrToIdxsMap.erase(TI);
    }
    IdxToInstrMap.erase(It);
    // If this index was marked, evict it from the seqnum maps.
    auto SeqIt = InternalIdxToSeqNums.find(IdxToRemove);
    if (SeqIt != InternalIdxToSeqNums.end()) {
      // Copy seqnums before erasing to avoid iterator invalidation.
      SmallVector<uint64_t> Seqs(SeqIt->second.begin(), SeqIt->second.end());
      InternalIdxToSeqNums.erase(SeqIt);
      for (uint64_t Seq : Seqs) {
        SeqNumToInternalIdxs.erase(Seq);
        if (Seq == MinMarkedSeqNum) {
          if (SeqNumToInternalIdxs.empty()) {
            MinMarkedSeqNum = std::numeric_limits<uint64_t>::max();
            MaxMarkedSeqNum = 0;
          } else {
            do {
              ++MinMarkedSeqNum;
            } while (!SeqNumToInternalIdxs.count(MinMarkedSeqNum));
          }
        }
      }
    }
  }
  BottomIdxInternal += NumIndices;
}

void AsyncBuffer::removeIf(function_ref<bool(const TrackedInstr &)> Pred) {
  // Collect matching instructions via the reverse map to avoid scanning
  // IdxToInstrMap. This makes removeIf() O(M) where M = removed instrs.
  SmallVector<TrackedInstr> ToRemove;
  for (auto &[TI, Idxs] : InstrToIdxsMap)
    if (Pred(TI))
      ToRemove.push_back(TI);
  for (TrackedInstr TI : ToRemove) {
    for (unsigned Idx : InstrToIdxsMap[TI]) {
      Hash = getHash<Action::Remove>(TI);
      // If this slot was marked, remove its seqnum entries.
      auto SeqIt = InternalIdxToSeqNums.find(Idx);
      if (SeqIt != InternalIdxToSeqNums.end()) {
        for (auto InternalIdx : SeqIt->second)
          SeqNumToInternalIdxs.erase(InternalIdx);
        InternalIdxToSeqNums.erase(SeqIt);
      }
      auto It = IdxToInstrMap.find(Idx);
      if (It == IdxToInstrMap.end())
        continue;
      It->second.erase(TI);
      if (It->second.empty())
        IdxToInstrMap.erase(It);
    }
    InstrToIdxsMap.erase(TI);
  }
  // Recompute Min/MaxMarkedSeqNum after arbitrary removals.
  if (SeqNumToInternalIdxs.empty()) {
    MinMarkedSeqNum = std::numeric_limits<uint64_t>::max();
    MaxMarkedSeqNum = 0;
  } else {
    MinMarkedSeqNum = MaxMarkedSeqNum = SeqNumToInternalIdxs.begin()->first;
    for (const auto &[Seq, _] : SeqNumToInternalIdxs) {
      MinMarkedSeqNum = std::min(MinMarkedSeqNum, Seq);
      MaxMarkedSeqNum = std::max(MaxMarkedSeqNum, Seq);
    }
  }
  // Trim leading/trailing gaps.
  while (BottomIdxInternal < TopIdxInternal &&
         !IdxToInstrMap.count(BottomIdxInternal))
    ++BottomIdxInternal;
  while (TopIdxInternal > BottomIdxInternal &&
         !IdxToInstrMap.count(TopIdxInternal - 1))
    --TopIdxInternal;
}

bool AsyncBuffer::operator==(const InstrBufferBase &OtherBase) const {
  const AsyncBuffer *Other = dyn_cast<AsyncBuffer>(&OtherBase);
  if (!Other || Hash != Other->Hash)
    return false;
  unsigned ThisSize = getTopIndex();
  if (ThisSize != Other->getTopIndex())
    return false;
  for (unsigned I = 0; I < ThisSize; ++I) {
    if (getNthFromEnd(I) != Other->getNthFromEnd(I))
      return false;
  }
  return true;
}

void AsyncBuffer::merge(const InstrBufferBase &OtherBase) {
  const AsyncBuffer &Other = cast<AsyncBuffer>(OtherBase);
  if (Other.empty())
    return;
  // When merging we align the buffers such that their most recent entries
  // (the ones with highest indices) get merged on the same index. For example:
  //
  //  This      Other   after merge:  This
  //  ----      ------                ----
  //  Idx=0:I1  Idx=1:I2              Idx=1:I2
  //            Idx=0:I3              Idx=0:I1 I3

  unsigned MergedTopIdxInternal = std::max(TopIdxInternal, Other.TopIdxInternal);
  unsigned ThisOffset = MergedTopIdxInternal - TopIdxInternal;
  unsigned OtherOffset = MergedTopIdxInternal - Other.TopIdxInternal;
  unsigned MergedBotIdxInternal = MergedTopIdxInternal;

  // TODO: Try to reuse one the existing maps.
  DenseMap<TrackedInstr, SmallDenseSet<unsigned, 2>> MergedInstrToIdxsMap;
  DenseMap<unsigned, SmallDenseSet<TrackedInstr, 2>> MergedIdxToInstrMap;

  SmallDenseSet<unsigned, 16> MergedIdxsWithSeqNum;

  for (const auto &[ThisTI, ThisIdxsInternal] : InstrToIdxsMap) {
    auto &MergedIdxs = MergedInstrToIdxsMap[ThisTI];
    for (unsigned ThisIdxInternal : ThisIdxsInternal) {
      unsigned MergedThisIdx = ThisIdxInternal + ThisOffset;
      MergedIdxToInstrMap[MergedThisIdx].insert(ThisTI);
      MergedIdxs.insert(MergedThisIdx);
    }
  }
  for (auto [OtherTI, OtherIdxsInternal] : Other.InstrToIdxsMap) {
    for (unsigned OtherIdxInternal : OtherIdxsInternal) {
      unsigned MergedOtherIdx = OtherIdxInternal + OtherOffset;
      MergedInstrToIdxsMap[OtherTI].insert(MergedOtherIdx);
      MergedIdxToInstrMap[MergedOtherIdx].insert(OtherTI);
    }
  }

  // We also merge the sequence numbers. Since at each BB the seqnums start from
  // 0 we should just merge them without aligning them. In this way we maintain
  // the "global" ordering across counters.
  // Example:
  //         This                   Other                     Merged.This
  //         ----                   -----                     -----------
  //     Async      Tensor        Async      Tensor       Async        Tensor
  //  Idx:3 SN:1  Idx:0 SN:2   Idx:4                   Idx:4
  //  Idx:2                    Idx:3 SN:2              Idx:3 SN:1,2
  //  Idx:1                    Idx:2 SN:1              Idx:2 SN:1
  //  Idx:0 SN:0               Idx:1 SN:0              Idx:1 SN:0
  //                           Idx:0                   Idx:0 SN:0    Idx:0 SN:2
  //
  //
  // Remap This's seqnum maps if its entries were shifted.
  if (ThisOffset != 0) {
    DenseMap<uint64_t, SmallSet<unsigned, 1>> RemappedSeqToIdxs;
    DenseMap<unsigned, SmallSet<uint64_t, 1>> RemappedIdxToSeqs;
    for (const auto &[Seq, ThisInternalIdxSet] : SeqNumToInternalIdxs)
      for (unsigned ThisInternalIdx : ThisInternalIdxSet) {
        unsigned NewThisInternalIdx = ThisInternalIdx + ThisOffset;
        RemappedSeqToIdxs[Seq].insert(NewThisInternalIdx);
        RemappedIdxToSeqs[NewThisInternalIdx].insert(Seq);
      }
    SeqNumToInternalIdxs = std::move(RemappedSeqToIdxs);
    InternalIdxToSeqNums = std::move(RemappedIdxToSeqs);
  }
  // Merge Other's seqnum entries.
  for (const auto &[OtherSeqNum, OtherIdxInternalSet] : Other.SeqNumToInternalIdxs) {
    for (unsigned OtherIdxInternal : OtherIdxInternalSet) {
      unsigned MergedOtherIdx = OtherIdxInternal + OtherOffset;
      SeqNumToInternalIdxs[OtherSeqNum].insert(MergedOtherIdx);
      InternalIdxToSeqNums[MergedOtherIdx].insert(OtherSeqNum);
    }
  }

  // TODO: Try to avoid this second loop.
  for (auto [MergedIdxInternal, TI] : MergedIdxToInstrMap)
    MergedBotIdxInternal = std::min(MergedBotIdxInternal, MergedIdxInternal);

  InstrToIdxsMap = std::move(MergedInstrToIdxsMap);
  IdxToInstrMap = std::move(MergedIdxToInstrMap);
  TopIdxInternal = MergedTopIdxInternal;
  BottomIdxInternal = MergedBotIdxInternal;

  // Recompute Min/MaxMarkedSeqNum from the merged seqnum map.
  if (SeqNumToInternalIdxs.empty()) {
    MinMarkedSeqNum = std::numeric_limits<uint64_t>::max();
    MaxMarkedSeqNum = 0;
  } else {
    MinMarkedSeqNum = std::min(MinMarkedSeqNum, Other.MinMarkedSeqNum);
    MaxMarkedSeqNum = std::max(MaxMarkedSeqNum, Other.MaxMarkedSeqNum);
  }

  // Advance GlobalSeqNum past the maximum seqnum used by the merged
  // AsyncBuffers. This ensures that subsequent pushBacks use a monotonically
  // increasing seqnum.
  if (!Other.SeqNumToInternalIdxs.empty())
    GlobalSeqNum = std::max(GlobalSeqNum, Other.MaxMarkedSeqNum + 1);

  // Signal that a CFG merge occurred. getWaitForNthMarked() uses this to clamp
  // N to MaxAsyncMarks-1, replicating the old pass's merge-time cap behavior.
  MergedAndCapped = true;
}

unsigned AsyncBuffer::numInstrs() const {
  unsigned Count = 0;
  for (const auto &[_, Set] : IdxToInstrMap)
    Count += Set.size();
  return Count;
}

bool AsyncBuffer::contains(MachineInstr *MI) const {
  return InstrToIdxsMap.count(TrackedInstr(MI));
}

bool AsyncBuffer::hasUnknown() const {
  return InstrToIdxsMap.count(TrackedInstr(nullptr));
}

SmallVector<TrackedInstr> AsyncBuffer::instrsUnordered() const {
  SmallVector<TrackedInstr> Result;
  for (const auto &[_, Set] : IdxToInstrMap)
    for (TrackedInstr TI : Set)
      Result.push_back(TI);
  return Result;
}

SmallDenseSet<TrackedInstr, 2> AsyncBuffer::back() const {
  if (empty())
    return {};
  auto It = IdxToInstrMap.find(TopIdxInternal - 1);
  return It != IdxToInstrMap.end() ? It->second : SmallDenseSet<TrackedInstr, 2>{};
}

const SmallDenseSet<TrackedInstr, 2> &
AsyncBuffer::getNthFromEnd(unsigned N) const {
  if (N >= getTopIndex())
    return EmptySet;
  unsigned InternalIdx = TopIdxInternal - 1 - N;
  auto It = IdxToInstrMap.find(InternalIdx);
  return It != IdxToInstrMap.end() ? It->second : EmptySet;
}

void AsyncBuffer::clear() {
  IdxToInstrMap.clear();
  InstrToIdxsMap.clear();
  SeqNumToInternalIdxs.clear();
  InternalIdxToSeqNums.clear();
  MinMarkedSeqNum = std::numeric_limits<uint64_t>::max();
  MaxMarkedSeqNum = 0;
  BottomIdxInternal = 0;
  TopIdxInternal = 0;
  Hash = 0;
}

std::optional<unsigned>
AsyncBuffer::getWaitForNthMarked(unsigned N) const {
  // Clamp N to MaxAsyncMarks-1 after a CFG merge (MergedAndCapped), replicating
  // the old pass's behavior: the AsyncMarks array was capped at MaxAsyncMarks
  // entries after merge, so N >= MaxAsyncMarks refers to an evicted entry and
  // must be clamped to the oldest retained mark. In straightline code the buffer
  // is unbounded, so no clamping occurs.
  if (MergedAndCapped && MaxAsyncMarks > 0 && N >= MaxAsyncMarks)
    N = MaxAsyncMarks - 1;
  // Scan slots from newest to oldest, skipping unmarked entries, and return
  // the wait value for the Nth marked entry (N=0 = most recently marked).
  LLVM_DEBUG(dbgs() << "getWaitForNthMarked(N=" << N
                    << "): TopIdx=" << getTopIndex()
                    << " marked=" << SeqNumToInternalIdxs.size() << "\n");

  // Iterate from the most recent to the oldest sequence number (i.e., max to
  // min) and count the ones that match this buffer.
  // TODO: This should work OK for now because We don't have too many counters
  // containing marked instructions but we could use a linked list to speed this
  // up.
  unsigned Cnt = 0;
  for (uint64_t SeqNum :
       reverse(seq<uint64_t>(MinMarkedSeqNum, MaxMarkedSeqNum + 1))) {
    auto It = SeqNumToInternalIdxs.find(SeqNum);
    if (It == SeqNumToInternalIdxs.end())
      continue;
    if (Cnt == N) {
      // Get the minimum wait across marks.
      unsigned MinIdx = std::numeric_limits<unsigned>::max();
      for (unsigned InternalIdx : It->second) {
        unsigned Idx = getExternalIdxFromInternal(InternalIdx);
        MinIdx = std::min(MinIdx, Idx);
      }
      return MinIdx;
    }
    ++Cnt;
  }
  return std::nullopt;
}

SmallDenseSet<TrackedInstr, 2> AsyncBuffer::IteratorImpl::deref() const {
  auto It = Buf->IdxToInstrMap.find(Idx);
  return It != Buf->IdxToInstrMap.end() ? It->second
                                        : SmallDenseSet<TrackedInstr, 2>{};
}

InstrBufferBase::iterator AsyncBuffer::begin() const {
  return iterator(std::make_unique<IteratorImpl>(this, BottomIdxInternal));
}

InstrBufferBase::iterator AsyncBuffer::end() const {
  return iterator(std::make_unique<IteratorImpl>(this, TopIdxInternal));
}

#ifndef NDEBUG

void InstrBufferBase::dump() const {
  print(dbgs(), /*Verbose=*/true);
  dbgs() << "\n";
}

void InstrBuffer::print(raw_ostream &OS, bool Verbose) const {
  if (Verbose) {
    OS << "TopIdxInternal:    " << TopIdxInternal << "\n";
    OS << "BottomIdxInternal: " << BottomIdxInternal << "\n";
    OS << "Hash: " << Hash << "\n";
    OS << "IdxToInstrMap:\n";
  }
  static constexpr const int Indent = 2;
  SmallVector<std::pair<unsigned, SmallDenseSet<TrackedInstr, 2>>>
      SortedIdxToInstrSetVec;
  for (auto &Pair : IdxToInstrMap)
    SortedIdxToInstrSetVec.push_back(Pair);
  sort(SortedIdxToInstrSetVec, [](const auto &Pair1, const auto &Pair2) {
    return Pair1.first < Pair2.first;
  });

  formatted_raw_ostream FOS(OS);
  for (const auto &[InternalIdx, TISet] : SortedIdxToInstrSetVec) {
    FOS.indent(Indent) << getExternalIdxFromInternal(InternalIdx)
                       << " (internal=" << InternalIdx << ") : ";
    for (const TrackedInstr &TI : TISet) {
      FOS.PadToColumn(20);
      FOS << TI;
    }
  }
}

void AsyncBuffer::print(raw_ostream &OS, bool Verbose) const {
  if (Verbose) {
    OS << "AsyncBuffer TopIdxInternal:    " << TopIdxInternal << "\n";
    OS << "AsyncBuffer BottomIdxInternal: " << BottomIdxInternal << "\n";
    OS << "AsyncBuffer Hash: " << Hash << "\n";
    OS << "IdxToInstrMap:\n";
  }
  int Indent = 2;
  SmallVector<std::pair<unsigned, SmallDenseSet<TrackedInstr, 2>>>
      SortedIdxToInstrSetVec;
  for (auto &Pair : IdxToInstrMap)
    SortedIdxToInstrSetVec.push_back(Pair);
  sort(SortedIdxToInstrSetVec, [](const auto &Pair1, const auto &Pair2) {
    return Pair1.first < Pair2.first;
  });

  formatted_raw_ostream FOS(OS);
  for (const auto &[InternalIdx, TISet] : SortedIdxToInstrSetVec) {
    FOS.indent(Indent) << getExternalIdxFromInternal(InternalIdx)
                       << " (internal=" << InternalIdx << ") : ";
    auto SeqIt = InternalIdxToSeqNums.find(InternalIdx);
    for (const TrackedInstr &TI : TISet) {
      FOS.PadToColumn(20);
      if (SeqIt != InternalIdxToSeqNums.end()) {
        FOS << " (SeqNum=";
        interleave(
            SeqIt->second, FOS, [&FOS](uint64_t SeqNum) { FOS << SeqNum; },
            ",");
        FOS << ") MARK ";
      } else {
        FOS << "  ";
      }
      FOS << TI;
    }
  }
}

void Counter::print(raw_ostream &OS) const {
  OS << getName() << "\n";
  OS << std::string(getName().size(), '-') << "\n";
  Instrs->print(OS, /*Verbose=*/false);
}

void AsyncCounter::print(raw_ostream &OS) const { Counter::print(OS); }

#endif // NDEBUG

MachineInstr *AsyncBuffer::getAsyncMarkFor(MachineInstr *MI) {
  assert(MI && isAsyncDMACandidate(*MI) &&
         "getAsyncMarkFor called on non-async-DMA instruction");
  bool MIIsAsync = SIInstrInfo::usesASYNC_CNT(*MI);
  bool MIIsTensor = SIInstrInfo::usesTENSOR_CNT(*MI);
  for (MachineInstr *Next = MI->getNextNode(); Next;
       Next = Next->getNextNode()) {
    if (Next->getOpcode() == AMDGPU::ASYNCMARK)
      return Next;
    if (!isAsyncDMACandidate(*Next))
      continue;
    // Same counter type, new batch.
    bool NextIsAsync = SIInstrInfo::usesASYNC_CNT(*Next);
    bool NextIsTensor = SIInstrInfo::usesTENSOR_CNT(*Next);
    if ((MIIsAsync && NextIsAsync) || (MIIsTensor && NextIsTensor) ||
        (!MIIsAsync && !MIIsTensor && !NextIsAsync && !NextIsTensor))
      break;
    // Different counter type, so skip and continue scanning.
  }
  llvm_unreachable("getAsyncMarkFor: no ASYNCMARK found for marked MI");
}

SmallVector<std::pair<unsigned, unsigned>>
AsyncBuffer::getNthMostRecentMarkedAmong(ArrayRef<AsyncBuffer *> Bufs,
                                         unsigned N) {
  // Clamp N when any buffer has been capped at a CFG merge: replicates the old
  // pass's behavior of capping the AsyncMarks array at MaxAsyncMarks entries.
  // If N >= MaxAsyncMarks and any buffer is MergedAndCapped, the requested
  // event was evicted — clamp to the oldest retained mark (MaxAsyncMarks-1).
  if (MaxAsyncMarks > 0 && N >= MaxAsyncMarks) {
    bool AnyCapped = llvm::any_of(Bufs, [](const AsyncBuffer *B) {
      return B->MergedAndCapped;
    });
    if (AnyCapped)
      N = MaxAsyncMarks - 1;
  }
  // Find the global seqnum range across all buffers.
  uint64_t GlobalSeqMax = 0;
  uint64_t GlobalSeqMin = std::numeric_limits<uint64_t>::max();
  for (const AsyncBuffer *Buf : Bufs) {
    if (Buf->SeqNumToInternalIdxs.empty())
      continue;
    GlobalSeqMax = std::max(GlobalSeqMax, Buf->MaxMarkedSeqNum);
    GlobalSeqMin = std::min(GlobalSeqMin, Buf->MinMarkedSeqNum);
  }
  if (GlobalSeqMin > GlobalSeqMax)
    return {}; // No marked entries in any buffer.

  auto GetMarkedMI = [](AsyncBuffer *Buf, unsigned Idx) -> MachineInstr * {
    unsigned InternalIdx = Buf->getInternalIdxFromExternal(Idx);
    auto It = Buf->IdxToInstrMap.find(InternalIdx);
    if (It != Buf->IdxToInstrMap.end())
      for (TrackedInstr TI : It->second)
        if (!TI.isUnknown())
          return TI.getMI();
    llvm_unreachable("Should have found the marked instruction by now!");
  };

  auto GetBufAndIdxWithSeqNum =
      [&Bufs](uint64_t SeqNum) -> SmallVector<std::pair<unsigned, unsigned>> {
    for (auto [BufIdx, Buf] : llvm::enumerate(Bufs)) {
      auto It = Buf->SeqNumToInternalIdxs.find(SeqNum);
      if (It == Buf->SeqNumToInternalIdxs.end())
        continue;
      SmallVector<std::pair<unsigned, unsigned>> Result;
      for (unsigned InternalIdx : It->second) {
        unsigned Idx = Buf->getExternalIdxFromInternal(InternalIdx);
        Result.push_back({BufIdx, Idx});
      }
      return Result;
    }
    return {};
  };

  SmallVector<std::pair<unsigned, unsigned>> Result;

  // Iterate over entries from the latest to the earliest by iterating from
  // GlobalSeqMax to GlobalSeqMin across all marked instructions across
  // all AsyncBuffers.
  unsigned EntryCount = 0;
  uint64_t MarkedSeqNum = 0;
  // TODO: Introduce an iterator for accessing the entries across buffers.
  for (uint64_t SeqNum :
       reverse(seq<uint64_t>(GlobalSeqMin, GlobalSeqMax + 1))) {
    auto BufAndIdxVec = GetBufAndIdxWithSeqNum(SeqNum);
    if (BufAndIdxVec.empty())
      continue;
    if (EntryCount == N) {
      for (auto [BufIdx, Idx] : BufAndIdxVec) {
        Result.emplace_back(BufIdx, Idx);
        MarkedSeqNum = SeqNum;
      }
      break;
    }
    ++EntryCount;
  }
  if (Result.empty())
    return Result;

  // At this point we would normally return the result but an ASYNCMARK
  // instruction could point to more than one instruction across buffers
  // (i.e., an Async DMA and a Tensor).
  auto [BufIdx0, Idx0] = Result[0];
  MachineInstr *MarkedMI = GetMarkedMI(Bufs[BufIdx0], Idx0);
  MachineInstr *AsyncMark0 = getAsyncMarkFor(MarkedMI);

  // TODO: For now search all entries but we could improve this with a cyclic
  // +-1 search around MarkedSeqNum.
  for (uint64_t SeqNum : seq<uint64_t>(GlobalSeqMin, GlobalSeqMax)) {
    // Skip the one we have already found.
    if (SeqNum == MarkedSeqNum)
      continue;
    // If we can't find a marked instr for SeqNum, we are done.
    auto BufAndIdxVec = GetBufAndIdxWithSeqNum(SeqNum);
    if (BufAndIdxVec.empty())
      break;

    for (auto [BufIdx, Idx] : BufAndIdxVec) {
      MachineInstr *MarkedMI = GetMarkedMI(Bufs[BufIdx], Idx);
      MachineInstr *AsyncMark = getAsyncMarkFor(MarkedMI);
      if (AsyncMark != AsyncMark0)
        continue;

      Result.emplace_back((unsigned)BufIdx, Idx);
    }
  }
  return Result;
}

SmallVector<std::pair<AsyncCounter *, unsigned>>
AsyncCounter::getNthMostRecentMarkedAmong(ArrayRef<AsyncCounter *> Counters,
                                          unsigned N) {
  SmallVector<AsyncBuffer *, 2> Bufs;
  for (AsyncCounter *C : Counters)
    Bufs.push_back(cast<AsyncBuffer>(C->Instrs.get()));
  SmallVector<std::pair<AsyncCounter *, unsigned>> Results;
  for (auto [BufIdx, Idx] : AsyncBuffer::getNthMostRecentMarkedAmong(Bufs, N))
    Results.emplace_back(Counters[BufIdx], Idx);
  return Results;
}

#ifndef NDEBUG

void Counter::dump() const {
  print(dbgs());
  dbgs() << "\n";
}

void WaitDescriptor::print(raw_ostream &OS) const {
  formatted_raw_ostream FOS(OS);
  FOS << "Counter: " << Cntr.getName();
  FOS.PadToColumn(28);
  FOS << " Wait: " << Wait << " ";
  if (MI)
    FOS << *MI;
  else
    FOS << "<unknown>\n";
}

void WaitDescriptor::dump() const {
  print(dbgs());
  dbgs() << "\n";
}

void AllCounters::print(raw_ostream &OS) const {
  SmallVector<const Counter *> SortedCounters;
  for (const auto &[Key, Cntr] : Map)
    SortedCounters.push_back(Cntr.get());
  sort(SortedCounters, [](const Counter *C1, const Counter *C2) {
    return C1->getType().getId() < C2->getType().getId();
  });
  for (const Counter *Cntr : SortedCounters) {
    Cntr->print(OS);
    OS << "\n";
  }
}

void AllCounters::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif

bool Counter::hasMixedEventTypes(const SIInstrInfo &TII) const {
  if (size() == 0)
    return false;

  // For XCnt, check if we have any SMEM instructions pending.
  // SMEM operations always complete out of order, even a single one.
  // This is different from other counters where we need 2+ instructions
  // of different types to have out-of-order completion.
  // VMEM-only XCnt operations complete in order.
  if (T == XCnt()) {
    for (const MachineInstr *MI : instrsUnordered()) {
      if (SIInstrInfo::isSMRD(*MI))
        return true;
    }
    return false;
  }

  // For other counters, a single pending instruction completes in order.
  if (size() <= 1)
    return false;

  // For DS_CNT (gfx12+), check if we have both LDS and GDS operations pending.
  // LDS and GDS can complete out of order relative to each other.
  // Note: LDS reads and writes are the same event type (LDS_ACCESS in the old
  // pass) so they complete in order. Same for GDS reads/writes.
  if (T == DsCnt()) {
    bool HasLDS = false;
    bool HasGDS = false;
    for (const MachineInstr *MI : instrsUnordered()) {
      if (isGDS(*MI, TII))
        HasGDS = true;
      else
        HasLDS = true;
      if (HasLDS && HasGDS)
        return true;
    }
    return false;
  }

  // For VMEM counters (LoadCnt on gfx12+, VmCnt on pre-gfx12), check if we
  // have multiple VmemTypes pending. Different VmemTypes (NOSAMPLER, SAMPLER,
  // BVH) can complete out of order relative to each other.
  // Also check for FLAT instructions which can take either VMEM or LDS path.
  if (T == LoadCnt() || T == VmCnt()) {
    std::optional<VmemType> FirstType;
    for (const MachineInstr *MI : instrsUnordered()) {
      // FLAT instructions (non-segment-specific) can take either VMEM or LDS
      // path, so they can complete out of order relative to VMEM-only instrs.
      // This check is guarded by EnableStrictMixedFlatCheck to match old pass.
      if (EnableStrictMixedFlatCheck && SIInstrInfo::isFLAT(*MI) &&
          !SIInstrInfo::isSegmentSpecificFLAT(*MI))
        return true;
      // Skip non-VMEM instructions.
      if (!updateVMCntOnly(*MI))
        continue;
      VmemType Type = getVmemType(*MI);
      if (!FirstType) {
        FirstType = Type;
      } else if (*FirstType != Type) {
        return true; // Multiple VmemTypes pending
      }
    }
    return false;
  }

  // For LgkmCnt (pre-gfx12), check if we have mixed DS/SMEM/FLAT types.
  // DS, SMEM, and pure FLAT (LDS path) all use lgkmcnt but can complete
  // out of order relative to each other.
  // SMEM operations always complete out of order (even a single one), similar
  // to XCnt behavior.
  // The FLAT check is guarded by EnableStrictMixedFlatCheck to match old pass.
  if (T == LgkmCnt()) {
    bool HasDS = false;
    bool HasSMEM = false;
    bool HasFLAT = false;
    for (const MachineInstr *MI : instrsUnordered()) {
      if (TII.isDS(*MI))
        HasDS = true;
      else if (SIInstrInfo::isSMRD(*MI))
        HasSMEM = true;
      else if (EnableStrictMixedFlatCheck && SIInstrInfo::isFLAT(*MI))
        HasFLAT = true;
    }
    // SMEM always completes out of order, so any SMEM pending means mixed
    // (even if it's the only pending instruction, it can complete out of
    // order relative to the source instruction waiting on lgkmcnt).
    if (HasSMEM)
      return true;
    // If more than one type is pending, they can complete out of order.
    if ((HasDS + HasSMEM + HasFLAT) > 1)
      return true;
    return false;
  }

  return false;
}

bool Counter::hasPendingFlat(const SIInstrInfo &TII) const {
  for (const MachineInstr *MI : instrsUnordered()) {
    // Check for "pure" FLAT instructions (FLAT_LOAD_*, FLAT_STORE_*, etc.).
    // Segment-specific FLAT instructions (GLOBAL_*, SCRATCH_*) are excluded
    // because they access a known memory segment and don't have the early
    // completion issue that pure FLAT instructions have on pre-GFX10.
    if (!SIInstrInfo::isFLAT(*MI) || SIInstrInfo::isSegmentSpecificFLAT(*MI))
      continue;

    // The FLAT early completion issue only applies when the instruction can
    // access both global memory (VMEM) and LDS. If the address space is known
    // from memory operands, only one path is possible and the hardware can
    // track completion correctly.
    auto *MF = MI->getParent()->getParent();
    auto &ST = MF->getSubtarget<GCNSubtarget>();
    bool TgSplit =
        ST.hasTgSplitSupport() && AMDGPU::isTgSplitEnabled(MF->getFunction());
    if (!TII.mayAccessVMEMThroughFlat(*MI) ||
        !TII.mayAccessLDSThroughFlat(*MI, TgSplit))
      continue;

    return true;
  }
  return false;
}

bool Counter::hasNonNosamplerVmemType() const {
  for (const MachineInstr *MI : instrsUnordered()) {
    if (!updateVMCntOnly(*MI))
      continue;
    if (getVmemType(*MI) != VMEM_NOSAMPLER)
      return true;
  }
  return false;
}

AllCounters::AllCounters(const GCNSubtarget &ST, SchedulingMode SchedMode,
                         const InstCounters *ICounters)
    : ICounters(ICounters) {
  for (const CounterType &T : ICounters->get(ST, SchedMode)) {
    // ExpCnt drops its oldest entries on overflow (an export past the counter
    // depth has finished reading its sources, so it needs no wait). Every other
    // counter keeps overflowed entries and clamps the wait value instead,
    // because an overflowed producer's result may still be pending.
    bool DropOnOverflow = T == ExpCnt();
    unsigned Size = CounterType::getCounterSize(T, ST);
    // Counters that track ASYNCMARK'd instructions need AsyncCounter (no-dedup
    // buffer) so that getWaitForNthMarked works correctly across loop
    // iterations and with interleaved unmarked loads.
    bool NeedsNoDedup = T == CounterType::getAsyncCounter(ST) ||
                        T == TensorCnt();
    if (NeedsNoDedup)
      Map.try_emplace(T, std::make_unique<AsyncCounter>(T, Size, GlobalSeqNum));
    else
      Map.try_emplace(T, std::make_unique<Counter>(T, Size, DropOnOverflow));
  }
}

WaitDescriptors AllCounters::get(MachineInstr &MI) const {
  WaitDescriptors QueryResults;
  // TODO: Avoid linear search.
  for (const auto &[Key, Ctr] : Map) {
    auto WaitOpt = Ctr->getWaitFor(MI);
    if (!WaitOpt)
      continue;
    QueryResults.emplace(&MI, Ctr->getType(), *WaitOpt);
  }
  return QueryResults;
}

WaitDescriptors AllCounters::get() const {
  WaitDescriptors Waits;
  for (const auto &[Key, Ctr] : Map)
    Waits.emplace(Ctr->getType(), Ctr->getConvergenceDepth());
  return Waits;
}

void AllCounters::merge(const AllCounters &Other) {
  // TODO: Assert that counter sets match before the loop.
  for (const auto &[OtherInstCounter, OtherCounter] : Other.Map) {
    auto It = Map.find(OtherInstCounter);
    assert(It != Map.end() && "CounterArrays have mismatched counter sets");
    It->second->merge(*OtherCounter);
  }
}

void AllCounters::clear() {
  for (auto &[Key, Ctr] : Map)
    Ctr->clear();
}

#ifndef NDEBUG
StringLiteral AMDGPU::regDepTypeToStr(RegDepType Dep) {
  switch (Dep) {
  case RegDepType::RAW:
    return "RAW";
  case RegDepType::WAR:
    return "WAR";
  case RegDepType::WAW:
    return "WAW";
  case RegDepType::RAR:
    return "RAR";
  }
  llvm_unreachable("Unhandled switch case");
}
#endif // NDEBUG

void WaitDescriptors::insertOrUpdate(MachineInstr *MI, CounterType Cntr,
                                       unsigned Wait) {
  unsigned Id = Cntr.getId();
  auto It =
      llvm::lower_bound(Vec, Id, [](const WaitDescriptor &E, unsigned Id) {
        return E.Cntr.getId() < Id;
      });
  if (It != Vec.end() && It->Cntr.getId() == Id) {
    if (Wait < It->Wait) {
      It->MI = MI;
      It->Wait = Wait;
    }
    return;
  }
  Vec.insert(It, WaitDescriptor(MI, Cntr, Wait));
}

WaitDescriptor *WaitDescriptors::get(CounterType Cntr) {
  unsigned Id = Cntr.getId();
  auto It =
      llvm::lower_bound(Vec, Id, [](const WaitDescriptor &E, unsigned Id) {
        return E.Cntr.getId() < Id;
      });
  if (It != Vec.end() && It->Cntr.getId() == Id)
    return &*It;
  return nullptr;
}

#ifndef NDEBUG
void WaitDescriptors::dump() const {
  print(dbgs());
  dbgs() << "\n";
}

void ResourceTracker::print(raw_ostream &OS) const {
  const TargetRegisterInfo *TRI = ST->getRegisterInfo();
  OS << "== RegUnit to instr map ==\n";
  SmallVector<MCRegUnit, 32> SortedRegUnits;
  for (const auto &[RU, MIInfos] : RegUnitToInstrsMap)
    SortedRegUnits.push_back(RU);
  sort(SortedRegUnits);
  for (MCRegUnit RU : SortedRegUnits) {
    const auto &MIInfos = RegUnitToInstrsMap.at(RU);
    OS << printRegUnit(RU, TRI) << " :\n";
    for (const RegUnitInfo &Info : MIInfos)
      OS << "  [" << (Info.Access == RegAccessType::Def ? "Def" : "Use") << " "
         << Info.Counter.getName() << "] " << *Info.MI << "\n";
  }
  OS << "\n";
  OS << "== Counters ==\n";
  Counters.print(OS);
}

void ResourceTracker::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SmallVector<CounterType, 4>
Counter::getCountersForInstr(const MachineInstr &MI, const GCNSubtarget &ST,
                             SchedulingMode SchedMode) {
  const SIInstrInfo *TII = ST.getInstrInfo();
  const bool IsGFX12Plus = ST.getGeneration() >= AMDGPUSubtarget::GFX12;

  // Counter aliases for different generations.
  CounterType DsCounter =
      IsGFX12Plus ? CounterType(AMDGPU::DsCnt()) : CounterType(AMDGPU::LgkmCnt());
  CounterType LoadCounter =
      IsGFX12Plus ? CounterType(AMDGPU::LoadCnt()) : CounterType(AMDGPU::VmCnt());
  CounterType StoreCounter =
      IsGFX12Plus ? CounterType(AMDGPU::StoreCnt()) : CounterType(AMDGPU::VsCnt());
  CounterType SmemCounter =
      IsGFX12Plus ? CounterType(AMDGPU::KmCnt()) : CounterType(AMDGPU::LgkmCnt());

  SmallVector<CounterType, 2> Result;

  // Tensor load/store DMA instructions (gfx1250+) use TensorCnt.
  if (IsGFX12Plus && SIInstrInfo::usesTENSOR_CNT(MI))
    return {AMDGPU::TensorCnt()};

  // DS instructions use DsCnt/LgkmCnt for LDS, or DsCnt/LgkmCnt + ExpCnt for GDS.
  if (TII->isDS(MI) && TII->usesLGKM_CNT(MI)) {
    if (TII->isAlwaysGDS(MI.getOpcode()) ||
        TII->hasModifiersSet(MI, AMDGPU::OpName::gds)) {
      // GDS uses DsCnt/LgkmCnt and also ExpCnt (for GDS_GPR_LOCK).
      Result = {DsCounter, AMDGPU::ExpCnt()};
    } else {
      Result = {DsCounter};
    }
  }
  // FLAT instructions can access both VMEM and LDS.
  else if (TII->isFLAT(MI)) {
    // GLOBAL_INV/WB/WBINV (GFX12+) are special cache operations. They don't
    // have mayLoad/mayStore but do increment a single counter. They perform no
    // address translation, so they must not be tracked on XCnt (nor on DsCnt),
    // and return immediately with just that counter.
    switch (MI.getOpcode()) {
    case AMDGPU::GLOBAL_INV:
      return {LoadCounter};
    case AMDGPU::GLOBAL_WB:
    case AMDGPU::GLOBAL_WBINV:
      return {StoreCounter};
    default:
      break;
    }
    if (SIInstrInfo::usesASYNC_CNT(MI)) {
      // Async LDS DMA instructions (gfx1250+) use AsyncCnt.
      Result = {AMDGPU::AsyncCnt()};
    } else if (TII->mayAccessVMEMThroughFlat(MI)) {
      // FLAT VMEM access uses LoadCnt/VmCnt for reads, StoreCnt/VsCnt for
      // writes. However, VsCnt only exists on GFX10+. On earlier targets,
      // stores also use VmCnt.
      // Note: FLAT doesn't use SampleCnt or BvhCnt.
      if (ST.hasVscnt() && MI.mayStore() &&
          (!MI.mayLoad() || SIInstrInfo::isAtomicNoRet(MI)))
        Result.push_back(StoreCounter);
      else
        Result.push_back(LoadCounter);
    }
    auto *MF = MI.getParent()->getParent();
    auto &ST = MF->getSubtarget<GCNSubtarget>();
    bool TgSplit =
        ST.hasTgSplitSupport() && AMDGPU::isTgSplitEnabled(MF->getFunction());
    if (!SIInstrInfo::usesASYNC_CNT(MI) &&
        TII->mayAccessLDSThroughFlat(MI, TgSplit))
      Result.push_back(DsCounter);
    // XCnt tracks address translation on gfx1250+ for VMEM access.
    if (ST.hasWaitXcnt() && TII->mayAccessVMEMThroughFlat(MI))
      Result.push_back(AMDGPU::XCnt());
  }
  // VMEM instructions (MUBUF, MTBUF, MIMG).
  else if (SIInstrInfo::isVMEM(MI)) {
    // Buffer invalidation instructions are encoded as MUBUF but don't update
    // any counters - they don't produce results that need to be waited for.
    switch (MI.getOpcode()) {
    case AMDGPU::BUFFER_WBINVL1:
    case AMDGPU::BUFFER_WBINVL1_SC:
    case AMDGPU::BUFFER_WBINVL1_VOL:
    case AMDGPU::BUFFER_GL0_INV:
    case AMDGPU::BUFFER_GL1_INV:
    case AMDGPU::BUFFER_INV:
    case AMDGPU::BUFFER_INVL2:
      return {};
    default:
      break;
    }

    // Store-only and atomic-no-return instructions use StoreCnt/VsCnt.
    // VsCnt only exists on GFX10+; on earlier targets stores use VmCnt.
    if (ST.hasVscnt() && MI.mayStore() &&
        (!MI.mayLoad() || SIInstrInfo::isAtomicNoRet(MI))) {
      Result = {StoreCounter};
    }
    // Image instructions with sampler/BVH use dedicated counters on GFX12+.
    else if (IsGFX12Plus && SIInstrInfo::isImage(MI)) {
      const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(MI.getOpcode());
      if (Info) {
        const AMDGPU::MIMGBaseOpcodeInfo *BaseInfo =
            AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);
        if (BaseInfo && BaseInfo->BVH)
          Result = {AMDGPU::BvhCnt()};
        // VSAMPLE instructions are also classified as sampler instructions.
        else if (BaseInfo &&
            (BaseInfo->Sampler || BaseInfo->MSAA || SIInstrInfo::isVSAMPLE(MI)))
          Result = {AMDGPU::SampleCnt()};
        else
          Result = {LoadCounter};
      } else {
        Result = {LoadCounter};
      }
    } else {
      Result = {LoadCounter};
    }
    // XCnt tracks address translation on gfx1250+ for VMEM access.
    if (ST.hasWaitXcnt())
      Result.push_back(AMDGPU::XCnt());
  }
  // SMEM instructions use KmCnt on GFX12+, LgkmCnt on pre-GFX12.
  else if (TII->isSMRD(MI)) {
    Result = {SmemCounter};
    // XCnt tracks address translation on gfx1250+ for SMEM access.
    if (ST.hasWaitXcnt())
      Result.push_back(AMDGPU::XCnt());
  }
  // EXP instructions use ExpCnt.
  else if (SIInstrInfo::isEXP(MI)) {
    Result = {AMDGPU::ExpCnt()};
  }
  // LDSDIR instructions use ExpCnt.
  else if (SIInstrInfo::isLDSDIR(MI)) {
    Result = {AMDGPU::ExpCnt()};
  }
  // On GFX12+, some barrier instructions (S_BARRIER_LEAVE,
  // S_BARRIER_SIGNAL_ISFIRST_*) write SCC asynchronously: the SCC result lands
  // via KmCnt rather than immediately. Track them on KmCnt so a later SCC read
  // waits for the write to land. These opcodes only exist on GFX12+, where
  // SmemCounter is KmCnt.
  else if (SIInstrInfo::isSBarrierSCCWrite(MI.getOpcode())) {
    Result = {SmemCounter};
  }
  // S_SENDMSG instructions use KmCnt/LgkmCnt.
  else {
    switch (MI.getOpcode()) {
    case AMDGPU::S_SENDMSG:
    case AMDGPU::S_SENDMSG_RTN_B32:
    case AMDGPU::S_SENDMSG_RTN_B64:
    case AMDGPU::S_SENDMSGHALT:
    case AMDGPU::S_MEMTIME:
    case AMDGPU::S_MEMREALTIME:
    case AMDGPU::S_GET_BARRIER_STATE_M0:
    case AMDGPU::S_GET_BARRIER_STATE_IMM:
      Result = {SmemCounter};
      break;
    // ASYNCMARK is a meta instruction that does not update any hardware counter.
    // WAIT_ASYNCMARK lowering (getNthMostRecentMarkedAmong) reads the async
    // counters directly; no special counter tracking is needed here.
    case AMDGPU::ASYNCMARK:
      break;
    default:
      break;
    }
  }

  // On gfx6, a VMEM/FLAT store also tracks ExpCnt to protect its data source
  // register(s) against a later overwrite (WAR hazard). EXP/LDSDIR already get
  // ExpCnt above, so this only needs to add it for stores. See srcsHeldOnExpCnt.
  if (ST.vmemWriteNeedsExpWaitcnt() &&
      (SIInstrInfo::isVMEM(MI) || SIInstrInfo::isFLAT(MI)) &&
      (MI.mayStore() || SIInstrInfo::isAtomicRet(MI)))
    Result.push_back(AMDGPU::ExpCnt());

  // Expert mode: VALU instructions that write VGPRs also use VaVdst.
  if (SchedMode == SchedulingMode::ExpertMode2 &&
      SIInstrInfo::isVALU(MI, /*AllowLDSDMA=*/true))
    Result.push_back(AMDGPU::VaVdst());

  // Expert mode: VMEM, FLAT, and DS instructions hold their VGPR sources on
  // VmVsrc until the address/data is read. Track it so a later VGPR overwrite
  // emits a depctr_vm_vsrc wait.
  if (SchedMode == SchedulingMode::ExpertMode2 &&
      (SIInstrInfo::isVMEM(MI) || TII->isFLAT(MI) || TII->isDS(MI) ||
       SIInstrInfo::isVIMAGE(MI) || SIInstrInfo::isVSAMPLE(MI)))
    Result.push_back(AMDGPU::VmVsrc());

  return Result;
}

bool Counter::needsFlatEarlyCompletionWorkaround(const GCNSubtarget &ST) const {
  if (ST.hasFlatLgkmVMemCountInOrder())
    return false;
  bool IsVmemOrLgkmCnt = T == VmCnt() || T == LoadCnt() ||
                         T == LgkmCnt() || T == DsCnt();
  return IsVmemOrLgkmCnt && hasPendingFlat(*ST.getInstrInfo());
}

bool Counter::isNonZeroWaitLegal(const MachineInstr &SrcMI,
                                 const MachineInstr &DstMI,
                                 const GCNSubtarget &ST,
                                 SchedulingMode SchedMode) const {
  if (needsFlatEarlyCompletionWorkaround(ST))
    return false;

  const SIInstrInfo *TII = ST.getInstrInfo();

  // Helper to check for "pure" FLAT instructions (FLAT_LOAD_*, FLAT_STORE_*).
  auto isPureFLAT = [](const MachineInstr &MI) {
    return SIInstrInfo::isFLAT(MI) && !SIInstrInfo::isSegmentSpecificFLAT(MI);
  };
  const CounterType &Cntr = T;
  // DS_CNT (gfx12+) tracks both DS reads and DS writes. When both are pending,
  // the counter can decrement out of order. However, when only reads or only
  // writes are pending (no mixed types), DS operations complete in order.
  if (Cntr == DsCnt()) {
    if (hasMixedEventTypes(*TII)) {
      LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: DsCnt mixed types, returning false\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: DsCnt, returning true\n");
    return true;
  }

  // LgkmCnt (pre-gfx12) is shared by DS, SMEM, and FLAT (LDS path) operations.
  // DS operations complete in order when no mixed types (LDS + GDS).
  // Pure FLAT uses lgkmcnt for its LDS path, and on gfx10+ with
  // hasFlatLgkmVMemCountInOrder, FLAT operations also complete in order.
  if (Cntr == LgkmCnt()) {
    if (TII->isDS(SrcMI)) {
      if (hasMixedEventTypes(*TII)) {
        LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: LgkmCnt DS mixed types, returning false\n");
        return false;
      }
      LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: LgkmCnt DS src, returning true\n");
      return true;
    }
    // Pure FLAT on lgkmcnt: in-order on gfx10+ with hasFlatLgkmVMemCountInOrder.
    // Pre-GFX10 FLAT can report early completion (counter decrements before
    // memory op actually completes), so position-based waits are unsafe.
    if (isPureFLAT(SrcMI) && ST.hasFlatLgkmVMemCountInOrder()) {
      if (hasMixedEventTypes(*TII)) {
        LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: LgkmCnt FLAT mixed types, returning false\n");
        return false;
      }
      LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: LgkmCnt FLAT src, returning true\n");
      return true;
    }
    LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: LgkmCnt, returning false\n");
    return false;
  }

  // EXP_CNT tracks exports and LDS_PARAM_LOAD instructions (on GFX11+).
  // LDS_PARAM_LOAD operations complete in order for register dependencies.
  if (Cntr == ExpCnt()) {
    LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: ExpCnt, returning true\n");
    return true;
  }

  // X_CNT tracks address translation for VMEM/SMEM on gfx1250+.
  // VMEM address translation completes in order, but SMEM can complete out of
  // order. Position-based waits are only legal if:
  // 1. The specific instruction we're waiting for (SrcMI) is VMEM, AND
  // 2. There are no SMEM instructions pending on XCnt (which would make the
  //    counter decrement out-of-order).
  // In multi-block scenarios, SMEM might be pending from a different path,
  // so we must check hasMixedEventTypes.
  if (Cntr == XCnt()) {
    if (SIInstrInfo::isSMRD(SrcMI)) {
      LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: XCnt SrcMI is SMEM, "
                        << "returning false\n");
      return false;
    }
    // Even if SrcMI is VMEM, if there's any SMEM pending on XCnt, the counter
    // can decrement out of order, making position-based waits unsafe.
    if (hasMixedEventTypes(*TII)) {
      LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: XCnt has mixed VMEM/SMEM, "
                        << "returning false\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: XCnt SrcMI is VMEM, "
                      << "returning true\n");
    return true;
  }

  // Expert-mode counters (VaVdst, VmVsrc) track in-order VALU/VMEM pipelines.
  // Position-based waits are always safe for these.
  if (Cntr == VaVdst() || Cntr == VmVsrc()) {
    assert(SchedMode == SchedulingMode::ExpertMode2 &&
           "Expert counters should only be present in ExpertMode2");
    LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: expert counter, returning true\n");
    return true;
  }

  // In-order completion only applies to VMEM counters (load/sample/bvh).
  if (!isVmemCounter(Cntr)) {
    LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: NOT vmem counter, returning false\n");
    return false;
  }

  // Check if DstMI is on the vmcnt counter (VMEM-only or pure FLAT).
  bool DstOnVmcnt = updateVMCntOnly(DstMI) || isPureFLAT(DstMI);

  // VMEM-only instructions (BUFFER, GLOBAL, SCRATCH) access only global memory.
  // They complete in-order with each other when they have the same VmemType.
  if (updateVMCntOnly(SrcMI)) {
    // Check if there are FLAT instructions pending which can complete out of
    // order relative to VMEM-only instructions.
    // This check is guarded by EnableStrictMixedFlatCheck to match old pass.
    if (EnableStrictMixedFlatCheck && hasPendingFlat(*TII)) {
      LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: SrcMI VMEM-only, FLAT pending, "
                        << "returning false\n");
      return false;
    }
    // If DstMI is not on vmcnt and not a memory instruction (e.g., VALU reading
    // the result), SrcMI's completion order doesn't depend on DstMI.
    if (!DstOnVmcnt) {
      LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: SrcMI VMEM-only, DstMI not on vmcnt, "
                        << "returning true\n");
      return true;
    }
    // VMEM-only -> pure FLAT: a pure FLAT can access LDS, causing counter
    // early completion on pre-gfx10. But this only matters when there's a
    // pending pure FLAT on the counter that could cause the early completion.
    // On gfx10+ (hasFlatLgkmVMemCountInOrder), FLAT completes in order and
    // this concern doesn't apply.
    if (!ST.hasFlatLgkmVMemCountInOrder() && isPureFLAT(DstMI) &&
        hasPendingFlat(*TII)) {
      LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: VMEM-only->pure FLAT with "
                           "pending FLAT, returning false\n");
      return false;
    }
    // If DstMI is a pure store it consumes SrcMI's value but produces no VMEM
    // result of its own, so there is no write-ordering (WAW) concern between
    // SrcMI and DstMI. The remaining question - whether SrcMI completes in a
    // predictable position - is only at risk when differently-completing VMEM
    // ops share the counter. Different VmemTypeS (NOSAMPLER/SAMPLER/BVH) are a
    // single in-order event on this counter pre-gfx12, and live on separate
    // counters on gfx12+, so they never make this counter out-of-order. (FLAT
    // early completion was already handled above.) The position-based wait is
    // therefore legal.
    if (DstMI.mayStore() && !DstMI.mayLoad()) {
      LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: VMEM-only -> store, returning true\n");
      return true;
    }
    // A pure FLAT DstMI (load or atomic) with no pending FLAT: no early
    // completion risk, position-based waits are safe.
    if (isPureFLAT(DstMI)) {
      LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: VMEM-only -> pure FLAT (no "
                           "pending FLAT), returning true\n");
      return true;
    }
    // Both are VMEM-only: check VmemType and Point Sample Acceleration.
    assert(updateVMCntOnly(DstMI));
    if (hasPointSampleAccel(SrcMI, ST) && getVmemType(DstMI) != VMEM_NOSAMPLER) {
      LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: SrcMI has point sample accel, "
                        << "DstMI not NOSAMPLER, returning false\n");
      return false;
    }
    if (hasPointSampleAccel(DstMI, ST) && getVmemType(SrcMI) != VMEM_NOSAMPLER) {
      LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: DstMI has point sample accel, "
                        << "SrcMI not NOSAMPLER, returning false\n");
      return false;
    }
    bool SameType = getVmemType(SrcMI) == getVmemType(DstMI);
    LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: both VMEM-only, same type="
                      << SameType << ", returning " << SameType << "\n");
    return SameType;
  }

  // Pure FLAT on vmcnt: in-order on gfx10+ with hasFlatLgkmVMemCountInOrder.
  // Pre-GFX10 FLAT can report early completion relative to other memory ops,
  // but for FLAT→VALU (non-memory consumer), position-based waits are safe.
  if (isPureFLAT(SrcMI)) {
    bool IsPreGFX12 = ST.getGeneration() < AMDGPUSubtarget::GFX12;
    auto DstCounters = Counter::getCountersForInstr(DstMI, ST, SchedMode);
    bool DstUsesCounter = llvm::is_contained(DstCounters, Cntr);

    if (!ST.hasFlatLgkmVMemCountInOrder()) {
      // Pre-GFX10: FLAT can complete out of order with other memory ops.
      // If DstMI doesn't use the counter (e.g., VALU), position-based waits
      // are safe because there's no counter interaction.
      if (!DstUsesCounter) {
        LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: pure FLAT, DstMI doesn't use counter, "
                          << "returning true\n");
        return true;
      }
      // DstMI uses the same counter (e.g., FLAT→GLOBAL). Since DstMI must wait
      // for SrcMI's counter slot to be freed before it can execute, SrcMI's
      // result is guaranteed to be ready. Enabled by default for pre-GFX12 to
      // match old pass; for GFX12+ can be enabled via -amdgpu-flat-same-counter-opt.
      if (EnableFlatSameCounterOpt || IsPreGFX12) {
        LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: pure FLAT, DstMI uses same counter "
                          << Cntr.getName() << ", returning true\n");
        return true;
      }
      LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: pure FLAT, no FlatLgkmVMemCountInOrder, "
                        << "returning false\n");
      return false;
    }
    // On gfx10+, pure FLAT completes in-order with other FLAT and VMEM-only
    // instructions for vmcnt (and lgkmcnt for FLAT's LDS path).
    // If DstMI doesn't use the counter (e.g., VALU), position-based waits are
    // safe. If DstMI uses the same vmem counter (e.g., FLAT→GLOBAL), DstMI
    // implicitly waits for SrcMI's counter slot, so SrcMI's result is ready.
    // This is disabled by default for GFX12+ to match old pass behavior.
    // Can be enabled via -amdgpu-flat-in-order-opt.
    if (!DstUsesCounter || EnableFlatInOrderOpt || IsPreGFX12) {
      LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: pure FLAT src with FlatLgkmVMemCountInOrder, "
                        << "DstUsesCounter=" << DstUsesCounter << ", returning true\n");
      return true;
    }
    LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: pure FLAT src, DstMI uses same counter, "
                      << "FlatInOrderOpt disabled for GFX12+, returning false\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "isNonZeroWaitLegal: fallthrough, returning false\n");
  return false;
}

bool ResourceTracker::impliesXcntSync(const MachineInstr &MI) const {
  if (!ST->hasWaitXcnt())
    return false;

  bool IsSmem = SIInstrInfo::isSMRD(MI);
  bool IsVmem = SIInstrInfo::isVMEM(MI) || SIInstrInfo::isFLAT(MI);
  if (!IsSmem && !IsVmem)
    return false;

  const Counter &XCntr = Counters[XCnt()];
  for (const MachineInstr *PendingMI : XCntr.instrsUnordered()) {
    bool PendingIsSmem = SIInstrInfo::isSMRD(*PendingMI);
    bool PendingIsVmem =
        SIInstrInfo::isVMEM(*PendingMI) || SIInstrInfo::isFLAT(*PendingMI);
    if ((IsSmem && PendingIsVmem) || (IsVmem && PendingIsSmem))
      return true;
  }
  return false;
}

void ResourceTracker::applyHWImpliedWaits(MachineInstr &MI) {
  // Hardware inserts an implicit XCnt wait between interleaved SMEM and VMEM
  // operations, which clears the XCnt state. Check and apply before inserting
  // MI into counters so that MI's own entry is not cleared.
  if (impliesXcntSync(MI)) {
    LLVM_DEBUG(dbgs() << "Implicit XCnt sync for SMEM/VMEM interleave\n");
    drainCounters({{XCnt(), 0}});
  }

  // S_BARRIER_WAIT guarantees that a matching S_BARRIER_SIGNAL_ISFIRST's
  // asynchronous SCC write has landed. Drain KmCnt so the pending signal
  // is no longer tracked and downstream soft waits can be simplified away.
  if (MI.getOpcode() == AMDGPU::S_BARRIER_WAIT) {
    const Counter &KmCntr = getCounter(KmCnt());
    for (const MachineInstr *Pending : KmCntr.instrsUnordered()) {
      if (Pending->getOpcode() == AMDGPU::S_BARRIER_SIGNAL_ISFIRST_IMM &&
          Pending->getOperand(0).getImm() == MI.getOperand(0).getImm()) {
        drainCounters({{KmCnt(), 0}});
        break;
      }
    }
  }
}

void ResourceTracker::drainCountersPerCallingConvention(MachineInstr &MI) {
  const bool IsGFX12Plus = ST->getGeneration() >= AMDGPUSubtarget::GFX12;
  // A call acts as a wait on all readiness counters: control only returns once
  // the callee (which waits in its prolog) has completed, so any of the
  // caller's memory ops issued before the call have finished by then. Drain
  // those counters so a later use of a pre-call result does not re-emit a
  // redundant wait. Mirrors the old pass's applyWaitcnt(getAllZeroWaitcnt(
  // IncludeVSCnt=false)) at a call. The counters left untouched match that
  // blanket wait's exclusions:
  //  - StoreCnt/VsCnt: the callee's stores may still be outstanding and the
  //    caller cannot track them; seed it unknown (below) so a following
  //    memory-legalizer seq_cst store fence is preserved.
  //  - XCnt, AsyncCnt, TensorCnt: not part of the blanket call wait.
  //  - VaVdst/VmVsrc: expert-mode VALU hazard counters, not memory waits.
  for (const CounterType &CT : ICounters.get(*ST, SchedMode)) {
    if (CT == StoreCnt() || CT == VsCnt() || CT == XCnt() || CT == AsyncCnt() ||
        CT == TensorCnt() || CT == VaVdst() || CT == VmVsrc())
      continue;
    Counters[CT].applyWait(0);
  }
  setCounterIncomingUnknown(IsGFX12Plus ? CounterType(StoreCnt())
                                        : CounterType(VsCnt()));
}

SmallVector<AsyncCounter *, 2> ResourceTracker::getAsyncCounters() {
  SmallVector<AsyncCounter *, 2> Result;
  CounterType AsyncCntrType = CounterType::getAsyncCounter(*ST);
  Result.push_back(&static_cast<AsyncCounter &>(getCounter(AsyncCntrType)));
  if (ST->getGeneration() >= AMDGPUSubtarget::GFX12)
    Result.push_back(&static_cast<AsyncCounter &>(getCounter(TensorCnt())));
  return Result;
}

void ResourceTracker::track(MachineInstr &MI) {
  LLVM_DEBUG(dbgs() << "ResourceTracker::track(): " << MI;);

  // Apply any hardware implied waits for MI.
  applyHWImpliedWaits(MI);

  auto CountersVec = Counter::getCountersForInstr(MI, *ST, SchedMode);
  // If the instruction is not affecting any counters then we don't need to
  // track it.
  if (!CountersVec.empty()) {
    const SIRegisterInfo *TRI = ST->getRegisterInfo();

    // We track by register unit so that a super-register access (e.g.,
    // vgpr0_vgpr1_vgpr2_vgpr3) can be found when querying for any of its
    // sub-registers (e.g., vgpr0).
    //
    // For each operand, record the register units it accesses together with the
    // counter that orders that access (see RegUnitInfo): a DEF on each result
    // counter (RAW/WAW), and a USE on a WAR counter that orders the read (XCnt
    // for any address read, ExpCnt for a register the producer holds on it). A
    // read that no counter orders is not a hazard and is not recorded. A
    // register may be both a DEF and a USE of the same instruction (e.g. a
    // return atomic), yielding entries on different counters that coexist.
    SmallDenseMap<MCRegUnit, SmallVector<RegUnitInfo, 2>, 8> RUToAccessMap;

    auto AddEntries = [&](MCRegister Reg, RegAccessType Access,
                          CounterType Counter) {
      for (MCRegUnit RU : TRI->regunits(Reg)) {
        SmallVectorImpl<RegUnitInfo> &Infos = RUToAccessMap[RU];
        RegUnitInfo Entry{&MI, Access, Counter};
        if (!llvm::is_contained(Infos, Entry))
          Infos.push_back(Entry);
      }
    };

    // LDSDIR delivers its result on ExpCnt, so for it ExpCnt orders a DEF
    // (RAW/WAW) in addition to the WAR reads ExpCnt normally tracks.
    bool ResultOnExpCnt = SIInstrInfo::isLDSDIR(MI);

    for (MachineOperand &MO : MI.operands()) {
      if (!MO.isReg())
        continue;
      Register Reg = MO.getReg();
      if (!Reg.isValid())
        continue;
      MCRegister MCReg = Reg.asMCReg();

      if (MO.isDef()) {
        // A memory op's real register result, if any, is an explicit def (e.g.
        // the loaded value, or a return-atomic's result). An implicit register
        // def on a memory op is not a real value - it is bookkeeping. The
        // motivating case is a GlobalISel multi-register (tuple) spill/reload,
        // where the first access gets an implicit-def of the whole super-register
        // to hold its live range together across the lowered sequence, e.g.:
        //   $vgpr0 = BUFFER_LOAD_DWORD_OFFSET ..., implicit-def $vgpr0_vgpr1_vgpr2_vgpr3
        //   $vgpr2 = ...    ; not really written by the load
        //   BUFFER_STORE_DWORD_OFFSET $vgpr5, ..., implicit-def $vgpr5_vgpr6_vgpr7_vgpr8
        //   BUFFER_STORE_DWORD_OFFSET $vgpr6, ...   ; reads $vgpr6 from that tuple
        // Recording the implicit-def as a result would make a later write or read
        // of one of those registers look like a WAW/RAW hazard and emit a bogus
        // wait. So for a memory op, skip implicit DEF operands; explicit defs
        // (the real results) are still tracked, and the op's own memory ordering
        // is tracked separately further down.
        if (MI.mayLoadOrStore() && MO.isImplicit())
          continue;
        for (const CounterType &C : CountersVec) {
          // A result counter orders a DEF for RAW/WAW. These are the non-WAR
          // counters, plus ExpCnt for LDSDIR.
          bool IsResultCounter =
              !C.isWarCounter() || (C == ExpCnt() && ResultOnExpCnt);
          if (IsResultCounter)
            AddEntries(MCReg, RegAccessType::Def, C);
        }
      } else if (MO.isUse()) {
        // A gfx6 store's ExpCnt data source is its explicit vdata operand. Its
        // implicit operands are liveness bookkeeping only (e.g. the whole-tuple
        // implicit use a GISel spill attaches); recording them on ExpCnt would
        // claim sibling dwords this store does not order, letting a later store
        // wrongly own a sibling's ExpCnt entry and under-count a WAR wait.
        bool IsStoreImplicitOperand = MO.isImplicit() && MI.mayStore();
        for (const CounterType &C : CountersVec) {
          if (C == ExpCnt() && IsStoreImplicitOperand)
            continue;
          // A WAR counter orders a USE so a later overwrite is a WAR hazard.
          // XCnt orders address translation, so any read register qualifies;
          // ExpCnt orders only the specific registers the producer holds on it;
          // VmVsrc orders VMEM source VGPR reads in expert scheduling mode
          // (SGPRs are read at issue and don't need VmVsrc protection).
          if (C.isWarCounter() &&
              (C == XCnt() ||
               (C == VmVsrc() && TRI->isVGPRClass(TRI->getPhysRegBaseClass(MCReg))) ||
               (C == ExpCnt() && regHeldOnExpCnt(MI, MCReg, *ST))))
            AddEntries(MCReg, RegAccessType::Use, C);
        }
      }
    }

    // Commit the resolved accesses, replacing any prior block-local accessor
    // for each unit. Within a single block each register unit has at most one
    // accessor instruction per (access, counter) (the latest one), but that
    // instruction may contribute several entries (e.g. a DEF result and an
    // ExpCnt data USE).
    for (auto &[RU, Infos] : RUToAccessMap)
      RegUnitToInstrsMap[RU] = std::move(Infos);

    // Insert MI into all counters that it updates.
    for (const CounterType &T : CountersVec) {
      LLVM_DEBUG(dbgs() << "ResourceTracker::track(): " << T.getName()
                        << "\n";);
      Counters[T].insert({&MI});
    }
  }

  if (MI.isCall())
    drainCountersPerCallingConvention(MI);

  // Some instructions drain XCnt.
  if (ST->hasWaitXcnt() && SIInstrInfo::isXcntDrain(MI))
    Counters[XCnt()].clear();
}

void ResourceTracker::drainCounters(const WaitDescriptors &CAWs) {
  for (const auto &CAW : CAWs) {
    LLVM_DEBUG(dbgs() << "drainCounters: " << CAW.Cntr.getName()
                      << " wait=" << CAW.Wait
                      << " size_before=" << Counters[CAW.Cntr].size());
    Counters[CAW.Cntr].applyWait(CAW.Wait);
    LLVM_DEBUG(dbgs() << " size_after=" << Counters[CAW.Cntr].size() << "\n");
  }
  // XCnt tracks address translation which completes before data arrives.
  // When waiting for a data counter with wait=0, the corresponding XCnt
  // entries are also complete (address translation finishes before data).
  // However, XCnt is a shared counter for both VMEM and SMEM operations,
  // so we can only clear it when the OTHER type is not pending.
  if (ST->hasWaitXcnt()) {
    const Counter &XCntr = Counters[XCnt()];
    bool HasPendingSmem = false;
    bool HasPendingVmem = false;
    for (const MachineInstr *MI : XCntr.instrsUnordered()) {
      if (SIInstrInfo::isSMRD(*MI))
        HasPendingSmem = true;
      else if (SIInstrInfo::isVMEM(*MI) || SIInstrInfo::isFLAT(*MI))
        HasPendingVmem = true;
    }
    bool Mixed = HasPendingSmem && HasPendingVmem;
    for (const auto &CAW : CAWs) {
      // KmCnt=0 means all SMEM address translation is complete.
      if (CAW.Wait == 0 && CAW.Cntr == KmCnt() && HasPendingSmem) {
        if (!Mixed) {
          // XCnt holds SMEM only: clear it entirely.
          Counters[XCnt()].applyWait(0);
        } else {
          // XCnt holds both SMEM and VMEM. The SMEM entries are now complete,
          // so drop them as hazards. The VMEM entries keep their positions so
          // VMEM is once again the only (in-order) type pending, enabling
          // position-based XCnt waits. Mirrors the original pass.
          Counters[XCnt()].removeIf(
              [](MachineInstr *MI) { return SIInstrInfo::isSMRD(*MI); });
        }
        break;
      }
      // LoadCnt=0 means all VMEM address translation is complete (stores don't
      // wait on LoadCnt, so only safe when no VMEM store is pending). Only clear
      // when XCnt is VMEM-only. When SMEM is also pending (Mixed), leave the
      // entries in place: the old pass keeps the per-register XCnt hazard and,
      // because the remaining SMEM keeps the counter out-of-order, a later WAR
      // on a VMEM address register still forces xcnt 0. Matching that keeps the
      // two passes in agreement.
      LLVM_DEBUG(dbgs() << "drainCounters: checking LoadCnt=0 XCnt clear: "
                        << "Wait=" << CAW.Wait
                        << " Cntr=" << CAW.Cntr.getName()
                        << " HasPendingSmem=" << HasPendingSmem
                        << " StoreCnt.size=" << Counters[StoreCnt()].size()
                        << "\n");
      if (CAW.Wait == 0 && CAW.Cntr == LoadCnt() && HasPendingVmem && !Mixed &&
          !Counters[StoreCnt()].size()) {
        LLVM_DEBUG(dbgs() << "drainCounters: clearing XCnt due to LoadCnt=0\n");
        Counters[XCnt()].applyWait(0);
        break;
      }
    }
  }
}

ArrayRef<ResourceTracker::RegUnitInfo>
ResourceTracker::getInstrsFor(MCRegUnit RU) const {
  auto It = RegUnitToInstrsMap.find(RU);
  if (It == RegUnitToInstrsMap.end())
    return {};
  return It->second;
}

MCRegister ResourceTracker::getEffectiveDepReg(Register Reg,
                                               const MachineInstr &DstMI) const {
  const SIRegisterInfo *TRI = ST->getRegisterInfo();
  MCRegister MCReg = Reg.asMCReg();
  // The two 16-bit halves of a VGPR are tracked as separate register units, so
  // accesses to lo16 and hi16 are normally independent. On targets with
  // hasD16Writes32BitVgpr, however, a D16 VALU instruction writes the whole
  // 32-bit VGPR, so a VALU access to one half can depend on a pending access to
  // the other half. For that case widen to the enclosing 32-bit register so both
  // halves' units are scanned. A D16 memory op writes only its own half, so a
  // memory access stays narrow (e.g. a store of lo16 must not wait for a load
  // that wrote hi16). Mirrors determineVGPR16Dependency in the original pass.
  //
  // Only physical registers in an allocatable class have a base class to query;
  // special registers like $exec/$scc/$m0 don't, so skip them.
  if (!ST->hasD16Writes32BitVgpr() || !Reg.isPhysical() ||
      !SIInstrInfo::isVALU(DstMI, /*AllowLDSDMA=*/true))
    return MCReg;
  const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(MCReg);
  if (!RC || !TRI->isVGPRClass(RC) || TRI->getRegSizeInBits(*RC) != 16)
    return MCReg;
  MCRegister Reg32 = TRI->get32BitRegister(MCReg);
  if (!Reg32)
    return MCReg;

  // Widen only when the other half actually has a pending op; otherwise there is
  // no cross-half hazard and the narrow register suffices.
  MCRegister OtherHalf = TRI->getSubReg(
      Reg32, AMDGPU::isHi16Reg(MCReg, *TRI) ? AMDGPU::lo16 : AMDGPU::hi16);
  for (MCRegUnit RU : TRI->regunits(OtherHalf))
    if (!getInstrsFor(RU).empty())
      return Reg32;
  return MCReg;
}

WaitDescriptors ResourceTracker::getWaitForReg(Register Reg) const {
  const SIRegisterInfo *TRI = ST->getRegisterInfo();
  WaitDescriptors Result;
  SmallPtrSet<MachineInstr *, 4> SeenMIs;
  for (MCRegUnit RU : TRI->regunits(Reg.asMCReg())) {
    for (const RegUnitInfo &Info : getInstrsFor(RU)) {
      if (!SeenMIs.insert(Info.MI).second)
        continue;
      for (auto &CW : Counters.get(*Info.MI))
        Result.insert(CW);
    }
  }
  return Result;
}

/// On pre-GFX10 targets, FLAT operations can report early completion (the
/// counter decrements before the memory operation actually completes). When
/// there's a pending FLAT on vmcnt/lgkmcnt and the target doesn't have the
/// hasFlatLgkmVMemCountInOrder feature, position-based wait values are unsafe
/// and we must wait for 0.
WaitDescriptors ResourceTracker::getWaitFor(Register Reg,
                                                    MachineInstr &DstMI,
                                                    RegAccessType DstAccess) const {
  const SIRegisterInfo *TRI = ST->getRegisterInfo();
  WaitDescriptors Result;
  // Dedup on (SrcMI, Counter): a super-register access surfaces the same entry
  // via several register units, but distinct (access, counter) entries of the
  // same source MI (e.g. a return atomic's DEF and ExpCnt USE) must all process.
  SmallDenseSet<std::pair<MachineInstr *, unsigned>, 4> SeenEntries;

  // On hasD16Writes32BitVgpr targets a 16-bit VGPR VALU access can depend on a
  // pending access to the other half of the 32-bit register, so scan the
  // enclosing register's units. See getEffectiveDepReg.
  MCRegister EffectiveReg = getEffectiveDepReg(Reg, DstMI);

  // Iterate over all register units of EffectiveReg.
  for (MCRegUnit RU : TRI->regunits(EffectiveReg)) {
    // Iterate over all (instruction, access, counter) entries the unit depends
    // on. Each entry already names the counter that orders it, so there is no
    // counter selection to do here - just classify the hazard and look up the
    // wait value on that counter.
    for (const RegUnitInfo &Info : getInstrsFor(RU)) {
      MachineInstr *SrcMI = Info.MI;
      RegDepType DepType = getDepType(Info.Access, DstAccess);
      // Skip read-after-read (no hazard).
      if (DepType == RegDepType::RAR)
        continue;
      // A USE entry orders a WAR hazard (a later write); skip it for a read.
      // A DEF entry orders RAW/WAW; skip it for nothing here. A USE meeting a
      // read is RAR, already skipped above, so only RAW/WAW/WAR remain.
      if (!SeenEntries.insert({SrcMI, Info.Counter.getId()}).second)
        continue;

      const Counter &Cntr = Counters[Info.Counter];
      std::optional<unsigned> Wait = Cntr.getWaitFor(*SrcMI);
      if (!Wait)
        continue;

      // For WAW hazards on VMEM counters, skip the wait if hardware guarantees
      // in-order VGPR writes. This check must come before isNonZeroWaitLegal
      // because the WAW skip applies regardless of position-based wait safety
      // (e.g., even when FLAT is pending and would normally force wait=0).
      if (DepType == RegDepType::WAW && isVmemCounter(Info.Counter) &&
          canSkipWawHazard(*SrcMI, DstMI, *ST)) {
        LLVM_DEBUG(dbgs() << "getWaitFor: skipping due to in-order WAW\n");
        continue;
      }
      // Check if position-based waits are safe for this dependency.
      bool InOrderCompletion =
          Cntr.isNonZeroWaitLegal(*SrcMI, DstMI, *ST, SchedMode);
      // TODO: This is a workaround to help match the old pass behavior.
      // A FLAT increments both VmCnt and LgkmCnt. If this counter's FLAT entry
      // was drained but the sibling counter still has a pending FLAT, the FLAT
      // may not have truly completed.
      if (EnableCrossCounterFlatPending && InOrderCompletion &&
          !ST->hasFlatLgkmVMemCountInOrder()) {
        const SIInstrInfo &TII = *ST->getInstrInfo();
        auto CheckSibling = [&](const CounterType &Sibling) {
          return Counters[Sibling].hasPendingFlat(TII);
        };
        if ((Info.Counter == VmCnt() && CheckSibling(LgkmCnt())) ||
            (Info.Counter == LgkmCnt() && CheckSibling(VmCnt())) ||
            (Info.Counter == LoadCnt() && CheckSibling(DsCnt())) ||
            (Info.Counter == DsCnt() && CheckSibling(LoadCnt())))
          InOrderCompletion = false;
      }
      LLVM_DEBUG(dbgs() << "getWaitFor: Counter " << Info.Counter.getName()
                        << " Wait=" << *Wait
                        << " InOrderCompletion=" << InOrderCompletion
                        << " DepType=" << regDepTypeToStr(DepType) << "\n");
      // Position-based waits are not safe when completion is out-of-order: force
      // wait=0.
      Result.emplace(SrcMI, Cntr.getType(), InOrderCompletion ? *Wait : 0);
    }
  }
  return Result;
}

WaitDescriptors ResourceTracker::getWaitForMemory(const MachineInstr &MI) const {
  // Check for SMEM/VMEM memory dependencies: a store on one unit may alias
  // a pending load on the other. The load must complete before the store
  // overwrites the data (WAR). Check both directions:
  //   1. VMEM store vs pending SMEM load (LgkmCnt/KmCnt)
  //   2. SMEM store vs pending VMEM load (VmCnt/LoadCnt)
  if (MI.mayStore()) {
    const bool IsGFX12Plus = ST->getGeneration() >= AMDGPUSubtarget::GFX12;
    bool IsSmemStore = SIInstrInfo::isSMRD(MI);
    CounterType PendingCntr =
        IsSmemStore ? (IsGFX12Plus ? CounterType(LoadCnt()) : CounterType(VmCnt()))
                    : (IsGFX12Plus ? CounterType(KmCnt()) : CounterType(LgkmCnt()));
    const Counter &Cntr = Counters[PendingCntr];
    for (MachineInstr *Pending : Cntr.instrsUnordered()) {
      if (!Pending->mayLoad())
        continue;
      // For VMEM store vs SMEM load: only SMEM loads (isSMRD) read global
      // memory. LgkmCnt/KmCnt also tracks DS/FLAT which access local memory
      // and can't alias with a VMEM store.
      if (!IsSmemStore && !SIInstrInfo::isSMRD(*Pending))
        continue;
      // Invariant loads read from memory guaranteed not to change (e.g. kernel
      // arguments), so no store can create a WAR hazard against them.
      if (llvm::any_of(Pending->memoperands(),
                       [](const MachineMemOperand *MMO) {
                         return MMO->isInvariant();
                       }))
        continue;
      if (EnableSmemVmemDepAA) {
        if (!AA || MI.mayAlias(AA, *Pending, /*UseTBAA=*/true)) {
          WaitDescriptors Result;
          Result.emplace(Pending, PendingCntr, 0);
          return Result;
        }
      } else {
        for (const MachineMemOperand *StoreMMO : MI.memoperands()) {
          if (!StoreMMO->isStore())
            continue;
          const Value *StorePtr = StoreMMO->getValue();
          if (!StorePtr)
            continue;
          for (const MachineMemOperand *LoadMMO : Pending->memoperands()) {
            if (LoadMMO->getValue() == StorePtr) {
              WaitDescriptors Result;
              Result.emplace(Pending, PendingCntr, 0);
              return Result;
            }
          }
        }
      }
    }
  }

  // Check for cross-unit LDS memory dependencies.
  // If MI is a DS operation and there are pending VMEM-to-LDS operations in
  // the vmcnt counter, we need a vmcnt wait to ensure the VMEM writes complete.
  if (!TII->isDS(MI))
    return {};

  const bool IsGFX12Plus = ST->getGeneration() >= AMDGPUSubtarget::GFX12;
  CounterType VmemCounter =
      IsGFX12Plus ? CounterType(LoadCnt()) : CounterType(VmCnt());
  const Counter &VmemCntr = Counters[VmemCounter];

  // Check if LDSDMA instruction is async (uses ASYNCMARK mechanism).
  // Async LDSDMA is handled via WAIT_ASYNCMARK, not via normal aliasing check.
  auto IsAsyncLdsDma = [&](const MachineInstr &Instr) {
    if (SIInstrInfo::usesASYNC_CNT(Instr))
      return true;
    const MachineOperand *Async =
        TII->getNamedOperand(Instr, AMDGPU::OpName::IsAsync);
    return Async && Async->getImm();
  };

  // Find the minimum wait value across all aliasing VMEM-to-LDS DMAs.
  // A smaller wait value means a longer wait; we need the newest aliasing DMA
  // (smallest getWaitFor) to have completed. Returns nullopt if no aliasing DMA.
  auto GetMinWaitForAliasingLdsDma =
      [&]() -> std::optional<std::pair<MachineInstr *, unsigned>> {
    MachineInstr *MinMI = nullptr;
    std::optional<unsigned> MinWait;
    LLVM_DEBUG(dbgs() << "getWaitForMemory: checking " << VmemCntr.size()
                      << " pending instrs for DS: " << MI);
    for (MachineInstr *Instr : VmemCntr.instrsUnordered()) {
      if (!SIInstrInfo::isLDSDMA(*Instr))
        continue;
      if (IsAsyncLdsDma(*Instr)) {
        LLVM_DEBUG(dbgs() << "  Skipping async LDSDMA: " << *Instr);
        continue;
      }
      LLVM_DEBUG(dbgs() << "  Checking LDSDMA: " << *Instr);
      if (!AA) {
        LLVM_DEBUG(dbgs() << "    AA is null, assuming aliasing\n");
        return {{Instr, 0}};
      }
      // Without alias scope metadata on the LDS store memoperand, mayAlias
      // can't reliably disambiguate different LDS arrays (it falls back to
      // range-overlap on the same Value, which is unsound for poison pointers).
      bool HasAliasScope = false;
      for (const MachineMemOperand *MMO : Instr->memoperands()) {
        if (MMO->isStore() && MMO->getAddrSpace() == AMDGPUAS::LOCAL_ADDRESS) {
          auto AAI = MMO->getAAInfo();
          if (AAI && AAI.Scope)
            HasAliasScope = true;
          break;
        }
      }
      if (!HasAliasScope) {
        LLVM_DEBUG(dbgs() << "    No alias scope, assuming aliasing\n");
        return {{Instr, 0}};
      }
      bool Aliases = MI.mayAlias(AA, *Instr, /*UseTBAA=*/true);
      LLVM_DEBUG(dbgs() << "    mayAlias=" << Aliases << "\n");
      if (!Aliases)
        continue;
      auto Wait = VmemCntr.getWaitFor(*Instr);
      if (!Wait)
        return {{Instr, 0}};
      if (!MinWait || *Wait < *MinWait) {
        MinWait = *Wait;
        MinMI = Instr;
      }
    }
    if (!MinWait)
      return std::nullopt;
    return {{MinMI, *MinWait}};
  };

  auto Result = GetMinWaitForAliasingLdsDma();
  if (!Result)
    return {};

  WaitDescriptors Ret;
  Ret.emplace(Result->first, VmemCounter, Result->second);
  return Ret;
}

void ResourceTracker::merge(const ResourceTracker &Other) {
  Counters.merge(Other.Counters);
  // Merge RegUnitToInstrsMap: keep all mappings from both. A register unit can
  // have several entries for the same block (one per access/counter), so dedup
  // on the full entry rather than the block: append an entry from Other only if
  // this map does not already have it. This also handles loop back-edges, where
  // Other's entries may already be present from a prior iteration.
  for (const auto &[RU, OtherMIInfos] : Other.RegUnitToInstrsMap) {
    auto &ThisMIInfos = RegUnitToInstrsMap[RU];
    for (const RegUnitInfo &MIInfo : OtherMIInfos)
      if (!llvm::is_contained(ThisMIInfos, MIInfo))
        ThisMIInfos.push_back(MIInfo);
  }
}

#ifndef NDEBUG
void ResourceTracker::ConvergenceState::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif

void ResourceTracker::clear() {
  Counters.clear();
  RegUnitToInstrsMap.clear();
}
