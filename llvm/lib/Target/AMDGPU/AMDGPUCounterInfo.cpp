//===--- AMDGPUCounterInfo.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUCounterInfo.h"

using namespace llvm::AMDGPU;

#ifndef NDEBUG
void CounterDescr::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif

unsigned AMDGPUCounterInfo::getCounterSize(const CounterDescr &C) const {
  const bool IsGFX12Plus = ST.getGeneration() >= AMDGPUSubtarget::GFX12;
  AMDGPU::IsaVersion Isa = AMDGPU::getIsaVersion(ST.getCPU());

  if (C == getVmCnt() || C == getLoadCnt())
    return IsGFX12Plus ? getLoadcntBitMask(Isa) : getVmcntBitMask(Isa);
  if (C == getLgkmCnt() || C == getDsCnt())
    return IsGFX12Plus ? getDscntBitMask(Isa) : getLgkmcntBitMask(Isa);
  if (C == getExpCnt())
    return getExpcntBitMask(Isa);
  if (C == getVsCnt() || C == getStoreCnt())
    return getStorecntBitMask(Isa);
  if (C == getSampleCnt())
    return getSamplecntBitMask(Isa);
  if (C == getBvhCnt())
    return getBvhcntBitMask(Isa);
  if (C == getKmCnt())
    return getKmcntBitMask(Isa);
  if (C == getXCnt())
    return getXcntBitMask(Isa);
  if (C == getAsyncCnt())
    return getAsynccntBitMask(Isa);
  if (C == getTensorCnt())
    return getTensorcntBitMask(Isa);
  if (C == getVaVdstCnt())
    return DepCtr::getVaVdstBitMask();
  if (C == getVmVsrcCnt())
    return DepCtr::getVmVsrcBitMask();
}

InstCounterType AMDGPUCounterInfo::getLegacyID(const CounterDescr &C) const {
  if (C == getVmCnt())
    return LOAD_CNT;
  if (C == getLoadCnt())
    return LOAD_CNT;
  if (C == getDsCnt())
    return DS_CNT;
  if (C == getLgkmCnt())
    return DS_CNT;
  if (C == getExpCnt())
    return EXP_CNT;
  if (C == getSampleCnt())
    return SAMPLE_CNT;
  if (C == getBvhCnt())
    return BVH_CNT;
  if (C == getKmCnt())
    return KM_CNT;
  if (C == getStoreCnt())
    return STORE_CNT;
  if (C == getVsCnt())
    return STORE_CNT;
  if (C == getXCnt())
    return X_CNT;
  if (C == getAsyncCnt())
    return ASYNC_CNT;
  if (C == getTensorCnt())
    return TENSOR_CNT;
  if (C == getVaVdstCnt())
    return VA_VDST_RD;
  if (C == getVmVsrcCnt())
    return VM_VSRC;

  llvm_unreachable("Unhandled counter!");
}
