//===- AMDGPUCounters.h - The AMDGPU hardware counters --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// AMDGPU Hardware counters.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUCOUNTERS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUCOUNTERS_H

#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace llvm {

namespace AMDGPU {

/// Base class for all AMDGPU counters.
class CounterType {
  StringLiteral Name;
  unsigned Id = 0;

public:
  CounterType(StringLiteral Name, unsigned Id) : Name(Name), Id(Id) {}
  StringLiteral getName() const { return Name; }
  unsigned getId() const { return Id; }
  bool operator==(const CounterType &Other) const { return Id == Other.Id; }
  bool operator!=(const CounterType &Other) const { return !(*this == Other); }

  /// Returns true if this is a WAR counter: one that orders a producer's read
  /// of a source register, so that a later overwrite of that register is a
  /// write-after-read hazard. XCnt orders address translation; ExpCnt orders
  /// EXP/LDSDIR sources and gfx6 store/atomic data. Every other counter tracks
  /// when a result is ready (RAW/WAW), not a source read.
  bool isWarCounter() const;

  /// Returns the hardware counter size limit for this counter type.
  /// Returns 0 if the counter has no hardware limit (e.g., pseudo-counters).
  static unsigned getCounterSize(const CounterType &T, const GCNSubtarget &ST);

  /// Returns the counter used for async LDS DMA operations on the given target.
  /// gfx1250 uses AsyncCnt, gfx12 uses LoadCnt, pre-gfx12 uses VmCnt.
  static CounterType getAsyncCounter(const GCNSubtarget &ST);
#ifndef NDEBUG
  void print(raw_ostream &OS) const { OS << Name; }
  void dump() const;
#endif
};

#define DEFINE_AMDGPU_COUNTER(CounterClass)                                    \
  class CounterClass : public CounterType {                                    \
  public:                                                                      \
    CounterClass() : CounterType(#CounterClass, __LINE__) {}                   \
  }; // Note: We are using __LINE__ as a unique ID!

// The definition order determines the counter ID (__LINE__), which controls
// the sorted emission order in WaitDescriptors. The order must match the
// canonical wait emission order of the old pass (SIInsertWaitcnts).

// LOAD_CNT: VMcnt prior to gfx12, LoadCnt for gfx12+.
DEFINE_AMDGPU_COUNTER(VmCnt)
DEFINE_AMDGPU_COUNTER(LoadCnt)

// DS_CNT: LgkmCnt prior to gfx12, DsCnt for gfx12+.
DEFINE_AMDGPU_COUNTER(LgkmCnt)
DEFINE_AMDGPU_COUNTER(DsCnt)

// EXP_CNT
DEFINE_AMDGPU_COUNTER(ExpCnt)

// SAMPLE_CNT: gfx12+ only.
DEFINE_AMDGPU_COUNTER(SampleCnt)

// BVH_CNT: gfx12+ only.
DEFINE_AMDGPU_COUNTER(BvhCnt)

// KM_CNT: gfx12+ only.
DEFINE_AMDGPU_COUNTER(KmCnt)

// STORE_CNT: VsCnt in gfx10/gfx11, StoreCnt for gfx12+.
DEFINE_AMDGPU_COUNTER(StoreCnt)
DEFINE_AMDGPU_COUNTER(VsCnt)

// X_CNT: gfx1250 only.
DEFINE_AMDGPU_COUNTER(XCnt)

// ASYNC_CNT: gfx1250 only.
DEFINE_AMDGPU_COUNTER(AsyncCnt)
// Used by tensor load/store DMA operations.
DEFINE_AMDGPU_COUNTER(TensorCnt)

// VA_VDST: gfx12+ expert mode only.
DEFINE_AMDGPU_COUNTER(VaVdst)

// VM_VSRC: gfx12+ expert mode only.
DEFINE_AMDGPU_COUNTER(VmVsrc)

#undef DEFINE_AMDGPU_COUNTER

enum class SchedulingMode {
  NoExpert,
  ExpertMode2,
};

/// Provides access to all counters per target.
class InstCounters {
  static SmallVector<CounterType> concat(SmallVector<CounterType> A,
                                         const SmallVector<CounterType> &B) {
    A.append(B);
    return A;
  }

  // Counters for pre-gfx12 (gfx10/gfx11).
  SmallVector<CounterType> PreGfx12Counters{VmCnt(), LgkmCnt(), ExpCnt(),
                                            VsCnt()};

  // Expert mode counters (gfx12+ only).
  SmallVector<CounterType> ExpertCounters{VaVdst(), VmVsrc()};

  // InstCounters for gfx12 (excluding gfx1250).
  SmallVector<CounterType> Gfx12Counters{
      LoadCnt(), DsCnt(), ExpCnt(), StoreCnt(), SampleCnt(), BvhCnt(),
      KmCnt()};
  // InstCounters for gfx12 with expert scheduling.
  SmallVector<CounterType> Gfx12ExpertCounters =
      concat(Gfx12Counters, ExpertCounters);

  // InstCounters for gfx1250.
  SmallVector<CounterType> Gfx1250Counters{LoadCnt(), DsCnt(),    ExpCnt(),
                                            StoreCnt(), SampleCnt(), BvhCnt(),
                                            KmCnt(),    XCnt(),      AsyncCnt(),
                                            TensorCnt()};

  // InstCounters for gfx1250 with expert scheduling.
  SmallVector<CounterType> Gfx1250ExpertCounters =
      concat(Gfx1250Counters, ExpertCounters);

public:
  InstCounters() {
#ifndef NDEBUG
    // Make sure the IDs are unique.
    DenseSet<unsigned> IDs;
    auto CheckNoDuplicates = [&IDs](ArrayRef<CounterType> Ctrs) {
      IDs.clear();
      for (const CounterType &C : Ctrs) {
        bool Inserted = IDs.insert(C.getId()).second;
        assert(Inserted && "Duplicate ID!");
      }
    };
    CheckNoDuplicates(PreGfx12Counters);
    CheckNoDuplicates(ExpertCounters);
    CheckNoDuplicates(Gfx12Counters);
    CheckNoDuplicates(Gfx12ExpertCounters);
    CheckNoDuplicates(Gfx1250Counters);
    CheckNoDuplicates(Gfx1250ExpertCounters);
#endif
  }
  /// \Returns the hardware counters for \p ST.
  const SmallVector<CounterType> &get(const GCNSubtarget &ST,
                                      SchedulingMode SchedMode) const {
    if (ST.getGeneration() >= AMDGPUSubtarget::GFX12) {
      if (ST.hasGFX1250Insts())
        return SchedMode == SchedulingMode::ExpertMode2 ? Gfx1250ExpertCounters
                                                        : Gfx1250Counters;
      return SchedMode == SchedulingMode::ExpertMode2 ? Gfx12ExpertCounters
                                                      : Gfx12Counters;
    }
    assert(SchedMode != SchedulingMode::ExpertMode2 &&
           "Expert scheduling only supported on gfx12+");
    return PreGfx12Counters;
  }

  /// Returns the list of async counters for \p ST.
  static SmallVector<CounterType, 2>
  getAsyncCounterTypes(const GCNSubtarget &ST) {
    SmallVector<CounterType, 2> AsyncCounters;
    CounterType AsyncCntrType = CounterType::getAsyncCounter(ST);
    AsyncCounters.push_back(AsyncCntrType);
    if (ST.getGeneration() >= AMDGPUSubtarget::GFX12)
      AsyncCounters.push_back(TensorCnt());
    return AsyncCounters;
  }

  /// TODO: Remove this once we migrate all code to the new counters.
  static InstCounterType getLegacyInstCounterType(const CounterType &T) {
    thread_local static DenseMap<CounterType, InstCounterType> CounterToICTMap;
    if (CounterToICTMap.empty()) {
      CounterToICTMap[VmCnt()] = AMDGPU::LOAD_CNT;
      CounterToICTMap[LoadCnt()] = AMDGPU::LOAD_CNT;
      CounterToICTMap[LgkmCnt()] = AMDGPU::DS_CNT;
      CounterToICTMap[DsCnt()] = AMDGPU::DS_CNT;
      CounterToICTMap[ExpCnt()] = AMDGPU::EXP_CNT;
      CounterToICTMap[VsCnt()] = AMDGPU::STORE_CNT;
      CounterToICTMap[StoreCnt()] = AMDGPU::STORE_CNT;
      CounterToICTMap[SampleCnt()] = AMDGPU::SAMPLE_CNT;
      CounterToICTMap[BvhCnt()] = AMDGPU::BVH_CNT;
      CounterToICTMap[KmCnt()] = AMDGPU::KM_CNT;
      CounterToICTMap[XCnt()] = AMDGPU::X_CNT;
      CounterToICTMap[AsyncCnt()] = AMDGPU::ASYNC_CNT;
      CounterToICTMap[TensorCnt()] = AMDGPU::TENSOR_CNT;
      CounterToICTMap[VaVdst()] = AMDGPU::VA_VDST_WR;
      CounterToICTMap[VmVsrc()] = AMDGPU::VM_VSRC;
    }
    return CounterToICTMap[T];
  }
};

} // namespace AMDGPU

template <> struct DenseMapInfo<AMDGPU::CounterType> {
  static inline AMDGPU::CounterType getEmptyKey() {
    return AMDGPU::CounterType("EMPTY", ~0U);
  }
  static inline AMDGPU::CounterType getTombstoneKey() {
    return AMDGPU::CounterType("TOMB", ~0U - 1);
  }
  static unsigned getHashValue(const AMDGPU::CounterType &C) {
    return DenseMapInfo<unsigned>::getHashValue(C.getId());
  }
  static bool isEqual(const AMDGPU::CounterType &LHS,
                      const AMDGPU::CounterType &RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUCOUNTERS_H
