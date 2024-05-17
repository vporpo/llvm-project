//===- SBRegionTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/SBRegion.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Utils.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

struct SBRegionTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;
  TargetLibraryInfoImpl TLII;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("SandboxIRTest", errs());
  }
  BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
    for (BasicBlock &BB : F)
      if (BB.getName() == Name)
        return &BB;
    llvm_unreachable("Expected to find basic block!");
  }
};

TEST_F(SBRegionTest, Basic) {
  parseIR(C, R"IR(
define void @foo(i32 %v1) {
  %node = add i32 %v1, %v1
  %root = add i32 %node, %node
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  BasicBlock *BB = &*F.begin();
  auto BBIt = BB->begin();
  auto *I1 = &*BBIt++;
  auto *I2 = &*BBIt++;
  auto *Ret = &*BBIt++;

  TargetLibraryInfo TLI(TLII);
  AAResults AA(TLI);
  SBContext Ctxt(C, AA);
  SBBasicBlock &SBBB = *Ctxt.createSBBasicBlock(BB);
  DataLayout DL(M.get());
  TargetTransformInfo TTI(DL);

  auto It = SBBB.begin();
  auto *SBI1 = &*It++;
  auto *SBI2 = &*It++;
  auto *SBRet = &*It++;
  SBValBundle Roots({SBI2});
  SBRegion Rgn0(SBBB, Ctxt, TTI);
  const auto &Kind = SBRegion::MDKind;
  auto *RegionMDN = I2->getMetadata(Kind);
  EXPECT_EQ(RegionMDN, nullptr);
  Rgn0.add(SBI1);
  EXPECT_EQ(I2->getMetadata(Kind), nullptr);
  EXPECT_NE(I1->getMetadata(Kind), nullptr);
  RegionMDN = I1->getMetadata(Kind);
  EXPECT_NE(RegionMDN, nullptr);
  EXPECT_EQ(cast<MDString>(RegionMDN->getOperand(SBRegion::TLRegionStrOpIdx))
                ->getString(),
            SBRegion::MDStrRegion);
  Rgn0.add(SBRet);
  // Check operator==
#ifndef NDEBUG
  EXPECT_TRUE(Rgn0 == Rgn0);
#endif
  auto *RetMDN = Ret->getMetadata(Kind);
  EXPECT_NE(RetMDN, nullptr);

  // Check roundtrip: create a region by parsing the metadata and check we get
  // an equivalent region.
  SBRegionBuilderFromMD Builder(Ctxt, TTI);
  auto Regions = Builder.createRegionsFromMD(SBBB);
  EXPECT_EQ(Regions.size(), 1u);
#ifndef NDEBUG
  SBRegion &Rgn1 = *Regions.front();
  EXPECT_TRUE(Rgn1 == Rgn0);
#endif
}
