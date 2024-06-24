//===- RegionTest.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Region.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIRVec.h"
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
  sandboxir::SBVecContext Ctx(C, AA);
  sandboxir::BasicBlock &SBBB = *Ctx.createBasicBlock(BB);
  DataLayout DL(M.get());
  TargetTransformInfo TTI(DL);

  auto It = SBBB.begin();
  auto *SBI1 = &*It++;
  auto *SBI2 = &*It++;
  auto *SBRet = &*It++;
  DmpVector<sandboxir::Value *> Roots({SBI2});
  sandboxir::Region Rgn0(SBBB, Ctx, TTI);
  const auto &Kind = sandboxir::Region::MDKind;
  auto *RegionMDN = I2->getMetadata(Kind);
  EXPECT_EQ(RegionMDN, nullptr);
  Rgn0.add(SBI1);
  EXPECT_EQ(I2->getMetadata(Kind), nullptr);
  EXPECT_NE(I1->getMetadata(Kind), nullptr);
  RegionMDN = I1->getMetadata(Kind);
  EXPECT_NE(RegionMDN, nullptr);
  EXPECT_EQ(cast<MDString>(
                RegionMDN->getOperand(sandboxir::Region::TLRegionStrOpIdx))
                ->getString(),
            sandboxir::Region::MDStrRegion);
  Rgn0.add(SBRet);
  // Check operator==
#ifndef NDEBUG
  EXPECT_TRUE(Rgn0 == Rgn0);
#endif
  auto *RetMDN = Ret->getMetadata(Kind);
  EXPECT_NE(RetMDN, nullptr);

  // Check roundtrip: create a region by parsing the metadata and check we get
  // an equivalent region.
  sandboxir::RegionBuilderFromMD Builder(Ctx, TTI);
  auto Regions = Builder.createRegionsFromMD(SBBB);
  EXPECT_EQ(Regions.size(), 1u);
#ifndef NDEBUG
  sandboxir::Region &Rgn1 = *Regions.front();
  EXPECT_TRUE(Rgn1 == Rgn0);
#endif
}
