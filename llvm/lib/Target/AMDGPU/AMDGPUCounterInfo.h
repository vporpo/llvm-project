//===-- AMDGPUCounterInfo.h - Per-target wait counter info --------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the AMDGPU counters using classes, which helps
// encapsulate target-dependent info like the hardware counter size in it.
// These classes are lightweight and cheap to default-construct.
//
// AMDGPUCounterInfo is initialized for a given subtarget and its main job is to
// provide the range of counters available on the subtarget.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUCOUNTERINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUCOUNTERINFO_H

#include "GCNSubtarget.h"
#include "llvm/ADT/ArrayRef.h"
#include <array>

namespace llvm::AMDGPU {

class AMDGPUCounterInfo;

/// A short lightweight counter descriptor. This is used instead of an enum.
class CounterDescr {
  /// Used for printing.
  StringRef Name;
  /// Unique ID of this counter. Used for comparing CounterDescrs.
  unsigned ID;
  /// The size of the hardware counter.
  // Note: It gets populated by the first call to getSize() and not during
  // construction because we want to have a lightweight default constructor.
  unsigned HwSize = 0;

protected:
  CounterDescr(StringRef Name, unsigned ID) : Name(Name), ID(ID) {}

  friend class AMDGPUCounterInfo; // For constructor and ID.

public:
  CounterDescr() = default;
  bool operator==(const CounterDescr &Other) const {
    bool Same = ID == Other.ID;
    assert((!Same || (Name == Other.Name && HwSize == Other.HwSize)) &&
           "Expected same Name and HwSize!");
    return Same;
  }

  bool operator!=(const CounterDescr &Other) const { return !(*this == Other); }

  StringRef getName() const { return Name; }

  unsigned getSize(const AMDGPUCounterInfo &CI);

  friend raw_ostream &operator<<(raw_ostream &OS, const CounterDescr &C) {
    C.print(OS);
    return OS;
  }

#ifndef NDEBUG
  void print(raw_ostream &OS) const { OS << Name; }

  void dump() const;
#endif
};

#define DEFINE_AMDGPU_COUNTER_DESCR(CounterClass, ID)                          \
  class CounterClass : public CounterDescr {                                   \
  public:                                                                      \
    CounterClass() : CounterDescr(#CounterClass, ID) {}                        \
  };

// clang-format off
DEFINE_AMDGPU_COUNTER_DESCR(VmCnt,     /*ID=*/0)
DEFINE_AMDGPU_COUNTER_DESCR(LoadCnt,   /*ID=*/1)
DEFINE_AMDGPU_COUNTER_DESCR(LgkmCnt,   /*ID=*/2)
DEFINE_AMDGPU_COUNTER_DESCR(DsCnt,     /*ID=*/3)
DEFINE_AMDGPU_COUNTER_DESCR(ExpCnt,    /*ID=*/4)
DEFINE_AMDGPU_COUNTER_DESCR(SampleCnt, /*ID=*/5)
DEFINE_AMDGPU_COUNTER_DESCR(BvhCnt,    /*ID=*/6)
DEFINE_AMDGPU_COUNTER_DESCR(KmCnt,     /*ID=*/7)
DEFINE_AMDGPU_COUNTER_DESCR(StoreCnt,  /*ID=*/8)
DEFINE_AMDGPU_COUNTER_DESCR(VsCnt,     /*ID=*/9)
DEFINE_AMDGPU_COUNTER_DESCR(XCnt,      /*ID=*/10)
DEFINE_AMDGPU_COUNTER_DESCR(AsyncCnt,  /*ID=*/11)
DEFINE_AMDGPU_COUNTER_DESCR(TensorCnt, /*ID=*/12)
DEFINE_AMDGPU_COUNTER_DESCR(VaVdst,    /*ID=*/13)
DEFINE_AMDGPU_COUNTER_DESCR(VmVsrc,    /*ID=*/14)
// clang-format on

#undef DEFINE_AMDGPU_COUNTER_DESCR

/// Targets can have a different range of counters. This class gets initialized
/// with the subtarget and provides the range of counter descriptors for that
/// target.
class AMDGPUCounterInfo {
  /// Helper for concatenating counters in initialization list.
  static SmallVector<CounterDescr> concat(const SmallVector<CounterDescr> &A,
                                          const SmallVector<CounterDescr> &B) {
    auto NewVec = A;
    NewVec.append(B);
    return NewVec;
  }

  SmallVector<CounterDescr> PreGfx12Counters{VmCnt(), LgkmCnt(),
                                                    ExpCnt(), VsCnt()};
  // Expert mode counters (gfx12+ only).
  SmallVector<CounterDescr> ExpertCounters{VaVdst(), VmVsrc()};
  // InstCounters for gfx12 (excluding gfx1250).
  SmallVector<CounterDescr> Gfx12Counters{
      LoadCnt(), DsCnt(), ExpCnt(), StoreCnt(), SampleCnt(), BvhCnt(), KmCnt()};
  // InstCounters for gfx12 with expert scheduling.
  SmallVector<CounterDescr> Gfx12ExpertCounters =
      concat(Gfx12Counters, ExpertCounters);
  // InstCounters for gfx1250.
  SmallVector<CounterDescr> Gfx1250Counters{
      LoadCnt(), DsCnt(), ExpCnt(), StoreCnt(), SampleCnt(),
      BvhCnt(),  KmCnt(), XCnt(),   AsyncCnt(), TensorCnt()};
  // InstCounters for gfx1250 with expert scheduling.
  SmallVector<CounterDescr> Gfx1250ExpertCounters =
      concat(Gfx1250Counters, ExpertCounters);

  /// Points to the appropriate counter array for the current subtarget.
  ArrayRef<CounterDescr> CountersRef;

  const GCNSubtarget &ST;

  unsigned getCounterSize(const CounterDescr &C) const;
  friend unsigned CounterDescr::getSize(const AMDGPUCounterInfo &);

public:
  // TODO: Use an enum class instead of bool ExpertMode2.
  AMDGPUCounterInfo(const GCNSubtarget &ST, bool ExpertMode2) : ST(ST) {
    if (ST.getGeneration() >= AMDGPUSubtarget::GFX12) {
      if (ST.hasGFX1250Insts())
        CountersRef = ExpertMode2 ? Gfx1250ExpertCounters : Gfx1250Counters;
      CountersRef = ExpertMode2 ? Gfx12ExpertCounters : Gfx12Counters;
    }
    assert(ExpertMode2 && "Expert scheduling only supported on gfx12+");
    CountersRef = PreGfx12Counters;
#ifndef NDEBUG
    // Check that all counters have unique IDs.
    SmallDenseSet<unsigned, 16> VisitedIDs;
    for (const CounterDescr &C : CountersRef)
      assert(VisitedIDs.insert(C.ID).second && "ID already exists!");
#endif
  }

  ArrayRef<CounterDescr> getCounters() const { return CountersRef; }
};

} // namespace llvm::AMDGPU

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUCOUNTERINFO_H
