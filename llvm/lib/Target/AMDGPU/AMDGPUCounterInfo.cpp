//===--- AMDGPUCounterInfo.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUCounterInfo.h"

using namespace llvm::AMDGPU;

unsigned CounterDescr::getSize(const AMDGPUCounterInfo &CI) {
  if (HwSize == 0)
    HwSize = CI.getCounterSize(*this);
  return HwSize;
}

void CounterDescr::dump() const {
  print(dbgs());
  dbgs() << "\n";
}

unsigned AMDGPUCounterInfo::getCounterSize(const CounterDescr &C) const {
  const bool IsGFX12Plus = ST.getGeneration() >= AMDGPUSubtarget::GFX12;
  AMDGPU::IsaVersion Isa = AMDGPU::getIsaVersion(ST.getCPU());

  if (C == VmCnt() || C == LoadCnt())
    return IsGFX12Plus ? getLoadcntBitMask(Isa) : getVmcntBitMask(Isa);
  if (C == LgkmCnt() || C == DsCnt())
    return IsGFX12Plus ? getDscntBitMask(Isa) : getLgkmcntBitMask(Isa);
  if (C == ExpCnt())
    return getExpcntBitMask(Isa);
  if (C == VsCnt() || C == StoreCnt())
    return getStorecntBitMask(Isa);
  if (C == SampleCnt())
    return getSamplecntBitMask(Isa);
  if (C == BvhCnt())
    return getBvhcntBitMask(Isa);
  if (C == KmCnt())
    return getKmcntBitMask(Isa);
  if (C == XCnt())
    return getXcntBitMask(Isa);
  if (C == AsyncCnt())
    return getAsynccntBitMask(Isa);
  if (C == TensorCnt())
    return getTensorcntBitMask(Isa);
  if (C == VaVdst())
    return DepCtr::getVaVdstBitMask();
  if (C == VmVsrc())
    return DepCtr::getVmVsrcBitMask();
}
