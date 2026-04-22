//===- AMDGPUCounters.cpp - The AMDGPU hardware counters ------------------===//
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

#include "AMDGPUCounters.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace llvm::AMDGPU;

static cl::opt<bool> EnforceCounterLimits(
    "amdgpu-waitcnt-enforce-counter-limits",
    cl::desc("Enforce hardware counter size limits during tracking"),
    cl::init(true), cl::Hidden);

bool CounterType::isWarCounter() const {
  return *this == XCnt() || *this == ExpCnt() || *this == VmVsrc();
}

unsigned CounterType::getCounterSize(const CounterType &T,
                                     const GCNSubtarget &ST) {
  if (!EnforceCounterLimits)
    return 0;

  IsaVersion IV = getIsaVersion(ST.getCPU());
  bool IsGFX12Plus = IV.Major >= 12;

  if (T == VmCnt() || T == LoadCnt())
    return IsGFX12Plus ? getLoadcntBitMask(IV) : getVmcntBitMask(IV);
  if (T == LgkmCnt() || T == DsCnt())
    return IsGFX12Plus ? getDscntBitMask(IV) : getLgkmcntBitMask(IV);
  if (T == ExpCnt())
    return getExpcntBitMask(IV);
  if (T == VsCnt() || T == StoreCnt())
    return getStorecntBitMask(IV);
  if (T == SampleCnt())
    return getSamplecntBitMask(IV);
  if (T == BvhCnt())
    return getBvhcntBitMask(IV);
  if (T == KmCnt())
    return getKmcntBitMask(IV);
  if (T == XCnt())
    return getXcntBitMask(IV);
  if (T == AsyncCnt())
    return getAsynccntBitMask(IV);
  if (T == TensorCnt())
    return getTensorcntBitMask(IV);
  if (T == VaVdst())
    return DepCtr::getVaVdstBitMask();
  if (T == VmVsrc())
    return DepCtr::getVmVsrcBitMask();

  llvm_unreachable("Unknown counter type");
}

CounterType CounterType::getAsyncCounter(const GCNSubtarget &ST) {
  if (ST.hasGFX1250Insts())
    return AsyncCnt();
  if (ST.getGeneration() >= AMDGPUSubtarget::GFX12)
    return LoadCnt();
  return VmCnt();
}

#ifndef NDEBUG
void CounterType::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif
