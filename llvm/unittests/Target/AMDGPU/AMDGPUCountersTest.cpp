//===-- AMDGPUCountersTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUCounters.h"
#include "AMDGPUUnitTests.h"
#include "GCNSubtarget.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::AMDGPU;

class AMDGPUCountersTest : public AMDGPUTestBase {};

TEST_F(AMDGPUCountersTest, GetCounterSize_GFX9) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx900", "");
  ASSERT_TRUE(TM);
  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);

  // gfx9: vmcnt has 6 bits (4 low + 2 high), max = 63
  EXPECT_EQ(CounterType::getCounterSize(VmCnt(), ST), 63u);
  EXPECT_EQ(CounterType::getCounterSize(LoadCnt(), ST), 63u);

  // gfx9: lgkmcnt has 4 bits, max = 15
  EXPECT_EQ(CounterType::getCounterSize(LgkmCnt(), ST), 15u);
  EXPECT_EQ(CounterType::getCounterSize(DsCnt(), ST), 15u);

  // gfx9: expcnt has 3 bits, max = 7
  EXPECT_EQ(CounterType::getCounterSize(ExpCnt(), ST), 7u);

  // gfx9: no vscnt, returns 0
  EXPECT_EQ(CounterType::getCounterSize(VsCnt(), ST), 0u);
  EXPECT_EQ(CounterType::getCounterSize(StoreCnt(), ST), 0u);

  // gfx12+ only counters return 0 on gfx9
  EXPECT_EQ(CounterType::getCounterSize(SampleCnt(), ST), 0u);
  EXPECT_EQ(CounterType::getCounterSize(BvhCnt(), ST), 0u);
  EXPECT_EQ(CounterType::getCounterSize(KmCnt(), ST), 0u);
}

TEST_F(AMDGPUCountersTest, GetCounterSize_GFX10) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1010", "");
  ASSERT_TRUE(TM);
  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);

  // gfx10: vmcnt has 6 bits (4 low + 2 high), max = 63
  EXPECT_EQ(CounterType::getCounterSize(VmCnt(), ST), 63u);
  EXPECT_EQ(CounterType::getCounterSize(LoadCnt(), ST), 63u);

  // gfx10: lgkmcnt has 6 bits, max = 63
  EXPECT_EQ(CounterType::getCounterSize(LgkmCnt(), ST), 63u);
  EXPECT_EQ(CounterType::getCounterSize(DsCnt(), ST), 63u);

  // gfx10: expcnt has 3 bits, max = 7
  EXPECT_EQ(CounterType::getCounterSize(ExpCnt(), ST), 7u);

  // gfx10: vscnt has 6 bits, max = 63
  EXPECT_EQ(CounterType::getCounterSize(VsCnt(), ST), 63u);
  EXPECT_EQ(CounterType::getCounterSize(StoreCnt(), ST), 63u);
}

TEST_F(AMDGPUCountersTest, GetCounterSize_GFX12) {
  auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1200", "");
  ASSERT_TRUE(TM);
  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);

  // gfx12: loadcnt has 6 bits, max = 63
  EXPECT_EQ(CounterType::getCounterSize(VmCnt(), ST), 63u);
  EXPECT_EQ(CounterType::getCounterSize(LoadCnt(), ST), 63u);

  // gfx12: dscnt has 6 bits, max = 63
  EXPECT_EQ(CounterType::getCounterSize(LgkmCnt(), ST), 63u);
  EXPECT_EQ(CounterType::getCounterSize(DsCnt(), ST), 63u);

  // gfx12: expcnt has 3 bits, max = 7
  EXPECT_EQ(CounterType::getCounterSize(ExpCnt(), ST), 7u);

  // gfx12: storecnt has 6 bits, max = 63
  EXPECT_EQ(CounterType::getCounterSize(VsCnt(), ST), 63u);
  EXPECT_EQ(CounterType::getCounterSize(StoreCnt(), ST), 63u);

  // gfx12: samplecnt has 6 bits, max = 63
  EXPECT_EQ(CounterType::getCounterSize(SampleCnt(), ST), 63u);

  // gfx12: bvhcnt has 3 bits, max = 7
  EXPECT_EQ(CounterType::getCounterSize(BvhCnt(), ST), 7u);

  // gfx12: kmcnt has 5 bits, max = 31
  EXPECT_EQ(CounterType::getCounterSize(KmCnt(), ST), 31u);
}

TEST_F(AMDGPUCountersTest, GetAsyncCounter) {
  auto TestGetAsyncCounter = [this](StringRef CPU, CounterType Expected) {
    auto TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), CPU, "");
    ASSERT_TRUE(TM);
    GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                    std::string(TM->getTargetFeatureString()), *TM);
    EXPECT_EQ(CounterType::getAsyncCounter(ST), Expected)
        << "Failed for CPU: " << CPU.str();
  };

  // Pre-gfx12 uses VmCnt for async LDS DMA.
  TestGetAsyncCounter("gfx900", VmCnt());
  TestGetAsyncCounter("gfx1010", VmCnt());

  // gfx12 (non-1250) uses LoadCnt for async LDS DMA.
  TestGetAsyncCounter("gfx1200", LoadCnt());

  // gfx1250 uses AsyncCnt for async LDS DMA.
  TestGetAsyncCounter("gfx1250", AsyncCnt());
}
