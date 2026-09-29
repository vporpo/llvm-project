//===-- AMDGPUCounterInfo.h - Per-target wait counter info --------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the AMDGPU counters using thin descriptor classes.
//
// AMDGPUCounterInfo is initialized for a given subtarget and its main job is to
// provide target dependent information for the counter descriptors, like the
// counter limits.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUCOUNTERINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUCOUNTERINFO_H

#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Debug.h"

namespace llvm::AMDGPU {

class AMDGPUCounterInfo;

/// A short lightweight counter descriptor. This is used instead of an enum.
class CounterDescr {
  unsigned ID;
  StringRef Name;
  constexpr CounterDescr(unsigned ID, StringRef Name) : ID(ID), Name(Name) {}
  friend class AMDGPUCounterInfo;
public:
  bool operator==(const CounterDescr &Other) const {
    return ID == Other.ID;
  }
#ifndef NDEBUG
  void print(raw_ostream &OS) const { OS << Name; }
  LLVM_DUMP_METHOD void dump() const;
#endif
};

/// Targets can have a different range of counters. This class gets initialized
/// with the subtarget and provides the range of counter descriptors for that
/// target.
class AMDGPUCounterInfo {
public:
  // TODO: Add checks within its factory function to check if the target
  // supports this particular counter.
#define DEFINE_GET_COUNTER(NAME, ID)                                           \
  constexpr CounterDescr get##NAME() const { return CounterDescr(ID, #NAME); }

  // Define get<NAME>() functions, for example getVmCnt().
  DEFINE_GET_COUNTER(VmCnt, 0)
  DEFINE_GET_COUNTER(LoadCnt, 1)
  DEFINE_GET_COUNTER(DsCnt, 2)
  DEFINE_GET_COUNTER(LgkmCnt, 3)
  DEFINE_GET_COUNTER(ExpCnt, 4)
  DEFINE_GET_COUNTER(SampleCnt, 5)
  DEFINE_GET_COUNTER(BvhCnt, 6)
  DEFINE_GET_COUNTER(KmCnt, 7)
  DEFINE_GET_COUNTER(StoreCnt, 8)
  DEFINE_GET_COUNTER(VsCnt, 9)
  DEFINE_GET_COUNTER(XCnt, 10)
  DEFINE_GET_COUNTER(AsyncCnt, 11)
  DEFINE_GET_COUNTER(TensorCnt, 12)
  DEFINE_GET_COUNTER(VaVdstCnt, 13)
  DEFINE_GET_COUNTER(VmVsrcCnt, 14)

private:
  static SmallVector<CounterDescr>
  concat(const SmallVector<CounterDescr> &Vec1,
         const SmallVector<CounterDescr> &Vec2) {
    auto NewVec = Vec1;
    NewVec.append(Vec2);
    return NewVec;
  }
  SmallVector<CounterDescr> PreGfx12Counters{getVmCnt(), getLgkmCnt(),
                                             getExpCnt(), getVsCnt()};
  // Expert mode counters (gfx12+ only).
  SmallVector<CounterDescr> ExpertCounters{getVaVdstCnt(), getVmVsrcCnt()};
  // InstCounters for gfx12 (excluding gfx1250).
  SmallVector<CounterDescr> Gfx12Counters{
      getLoadCnt(),   getDsCnt(),  getExpCnt(), getStoreCnt(),
      getSampleCnt(), getBvhCnt(), getKmCnt()};
  // InstCounters for gfx12 with expert scheduling.
  SmallVector<CounterDescr> Gfx12ExpertCounters =
      concat(Gfx12Counters, ExpertCounters);
  // InstCounters for gfx1250.
  SmallVector<CounterDescr> Gfx1250Counters{
      getLoadCnt(), getDsCnt(), getExpCnt(), getStoreCnt(), getSampleCnt(),
      getBvhCnt(),  getKmCnt(), getXCnt(),   getAsyncCnt(), getTensorCnt()};
  // InstCounters for gfx1250 with expert scheduling.
  SmallVector<CounterDescr> Gfx1250ExpertCounters =
      concat(Gfx1250Counters, ExpertCounters);

  /// Points to the appropriate counter array for the current subtarget.
  ArrayRef<CounterDescr> CountersRef;

  const GCNSubtarget &ST;

  unsigned getCounterSize(const CounterDescr &C) const;

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
  }

  ArrayRef<CounterDescr> getCounters() const { return CountersRef; }

  unsigned getCounterSize(CounterDescr C) const;

  InstCounterType getLegacyID(const CounterDescr &C) const;
};

} // namespace llvm::AMDGPU

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUCOUNTERINFO_H
