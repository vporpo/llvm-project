//===- VecUtilsTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/VecUtils.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("VecUtilsTest", errs());
  return Mod;
}

static BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    if (BB.getName() == Name)
      return &BB;
  llvm_unreachable("Expected to find basic block!");
}

TEST(Utils, AreConsecutive_gep_float) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep0 = getelementptr inbounds float, ptr %ptr, i64 0
  %gep1 = getelementptr inbounds float, ptr %ptr, i64 1
  %gep2 = getelementptr inbounds float, ptr %ptr, i64 2
  %gep3 = getelementptr inbounds float, ptr %ptr, i64 3

  %ld0 = load float, ptr %gep0
  %ld1 = load float, ptr %gep1
  %ld2 = load float, ptr %gep2
  %ld3 = load float, ptr %gep3

  %v2ld0 = load <2 x float>, ptr %gep0
  %v2ld1 = load <2 x float>, ptr %gep1
  %v2ld2 = load <2 x float>, ptr %gep2
  %v2ld3 = load <2 x float>, ptr %gep3

  %v3ld0 = load <3 x float>, ptr %gep0
  %v3ld1 = load <3 x float>, ptr %gep1
  %v3ld2 = load <3 x float>, ptr %gep2
  %v3ld3 = load <3 x float>, ptr %gep3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  LoopInfo LI(DT);
  ScalarEvolution SE(F, TLI, AC, DT, LI);

  sandboxir::Context Ctx(C);

  auto &SBF = *Ctx.createFunction(&F);
  auto &BB = *SBF.begin();
  auto It = std::next(BB.begin(), 4);
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *L3 = cast<sandboxir::LoadInst>(&*It++);

  auto *V2L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L3 = cast<sandboxir::LoadInst>(&*It++);

  auto *V3L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *V3L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *V3L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *V3L3 = cast<sandboxir::LoadInst>(&*It++);

  // Scalar
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(L0, L1, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(L1, L2, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(L2, L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L1, L0, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L2, L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L3, L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L0, L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L0, L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L1, L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L2, L0, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L3, L1, SE, DL));

  // Check 2-wide loads
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V2L0, V2L2, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V2L1, V2L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L0, V2L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L1, V2L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L2, V2L3, SE, DL));

  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L3, V2L1, SE, DL));

  // Check 3-wide loads
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V3L0, V3L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L0, V3L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L1, V3L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L2, V3L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L1, V3L0, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L2, V3L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L3, V3L2, SE, DL));

  // Check mixes of vectors and scalar
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(L0, V2L1, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(L1, V2L2, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V2L0, L2, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V3L0, L3, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V2L0, V3L2, SE, DL));

  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L0, V2L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L0, V3L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L0, V2L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L0, V3L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L0, L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L0, L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L0, V2L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L0, V2L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L1, L0, SE, DL));
}

TEST(Utils, AreConsecutive_gep_i8) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep0 = getelementptr inbounds i8, ptr %ptr, i64 0
  %gep1 = getelementptr inbounds i8, ptr %ptr, i64 4
  %gep2 = getelementptr inbounds i8, ptr %ptr, i64 8
  %gep3 = getelementptr inbounds i8, ptr %ptr, i64 12

  %ld0 = load float, ptr %gep0
  %ld1 = load float, ptr %gep1
  %ld2 = load float, ptr %gep2
  %ld3 = load float, ptr %gep3

  %v2ld0 = load <2 x float>, ptr %gep0
  %v2ld1 = load <2 x float>, ptr %gep1
  %v2ld2 = load <2 x float>, ptr %gep2
  %v2ld3 = load <2 x float>, ptr %gep3

  %v3ld0 = load <3 x float>, ptr %gep0
  %v3ld1 = load <3 x float>, ptr %gep1
  %v3ld2 = load <3 x float>, ptr %gep2
  %v3ld3 = load <3 x float>, ptr %gep3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  LoopInfo LI(DT);
  ScalarEvolution SE(F, TLI, AC, DT, LI);

  sandboxir::Context Ctx(C);

  auto &SBF = *Ctx.createFunction(&F);
  auto &BB = *SBF.begin();
  auto It = std::next(BB.begin(), 4);
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *L3 = cast<sandboxir::LoadInst>(&*It++);

  auto *V2L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L3 = cast<sandboxir::LoadInst>(&*It++);

  auto *V3L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *V3L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *V3L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *V3L3 = cast<sandboxir::LoadInst>(&*It++);

  // Scalar
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(L0, L1, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(L1, L2, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(L2, L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L1, L0, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L2, L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L3, L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L0, L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L0, L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L1, L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L2, L0, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L3, L1, SE, DL));

  // Check 2-wide loads
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V2L0, V2L2, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V2L1, V2L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L0, V2L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L1, V2L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L2, V2L3, SE, DL));

  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L3, V2L1, SE, DL));

  // Check 3-wide loads
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V3L0, V3L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L0, V3L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L1, V3L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L2, V3L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L1, V3L0, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L2, V3L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L3, V3L2, SE, DL));

  // Check mixes of vectors and scalar
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(L0, V2L1, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(L1, V2L2, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V2L0, L2, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V3L0, L3, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V2L0, V3L2, SE, DL));

  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L0, V2L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L0, V3L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L0, V2L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L0, V3L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L0, L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L0, L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L0, V2L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L0, V2L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L1, L0, SE, DL));
}

TEST(Utils, AreConsecutive_gep_i1) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep0 = getelementptr inbounds i1, ptr %ptr, i64 0
  %gep1 = getelementptr inbounds i2, ptr %ptr, i64 4
  %gep2 = getelementptr inbounds i3, ptr %ptr, i64 8
  %gep3 = getelementptr inbounds i7, ptr %ptr, i64 12

  %ld0 = load float, ptr %gep0
  %ld1 = load float, ptr %gep1
  %ld2 = load float, ptr %gep2
  %ld3 = load float, ptr %gep3

  %v2ld0 = load <2 x float>, ptr %gep0
  %v2ld1 = load <2 x float>, ptr %gep1
  %v2ld2 = load <2 x float>, ptr %gep2
  %v2ld3 = load <2 x float>, ptr %gep3

  %v3ld0 = load <3 x float>, ptr %gep0
  %v3ld1 = load <3 x float>, ptr %gep1
  %v3ld2 = load <3 x float>, ptr %gep2
  %v3ld3 = load <3 x float>, ptr %gep3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  LoopInfo LI(DT);
  ScalarEvolution SE(F, TLI, AC, DT, LI);

  sandboxir::Context Ctx(C);

  auto &SBF = *Ctx.createFunction(&F);
  auto &BB = *SBF.begin();
  auto It = std::next(BB.begin(), 4);
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *L3 = cast<sandboxir::LoadInst>(&*It++);

  auto *V2L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L3 = cast<sandboxir::LoadInst>(&*It++);

  auto *V3L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *V3L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *V3L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *V3L3 = cast<sandboxir::LoadInst>(&*It++);

  // Scalar
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(L0, L1, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(L1, L2, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(L2, L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L1, L0, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L2, L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L3, L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L0, L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L0, L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L1, L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L2, L0, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L3, L1, SE, DL));

  // Check 2-wide loads
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V2L0, V2L2, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V2L1, V2L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L0, V2L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L1, V2L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L2, V2L3, SE, DL));

  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L3, V2L1, SE, DL));

  // Check 3-wide loads
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V3L0, V3L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L0, V3L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L1, V3L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L2, V3L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L1, V3L0, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L2, V3L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L3, V3L2, SE, DL));

  // Check mixes of vectors and scalar
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(L0, V2L1, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(L1, V2L2, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V2L0, L2, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V3L0, L3, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::areConsecutive(V2L0, V3L2, SE, DL));

  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L0, V2L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L0, V3L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(L0, V2L3, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L0, V3L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L0, L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L0, L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L0, V2L1, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V3L0, V2L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::areConsecutive(V2L1, L0, SE, DL));
}

TEST(Utils, ComesBeforeInMem) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep0 = getelementptr inbounds float, ptr %ptr, i64 0
  %gep1 = getelementptr inbounds float, ptr %ptr, i64 1
  %gep2 = getelementptr inbounds float, ptr %ptr, i64 2
  %gep3 = getelementptr inbounds float, ptr %ptr, i64 3

  %ld0 = load float, ptr %gep0
  %ld1 = load float, ptr %gep1
  %ld2 = load float, ptr %gep2
  %ld3 = load float, ptr %gep3

  %v2ld0 = load <2 x float>, ptr %gep0
  %v2ld1 = load <2 x float>, ptr %gep1
  %v2ld2 = load <2 x float>, ptr %gep2
  %v2ld3 = load <2 x float>, ptr %gep3

  %v3ld0 = load <3 x float>, ptr %gep0
  %v3ld1 = load <3 x float>, ptr %gep1
  %v3ld2 = load <3 x float>, ptr %gep2
  %v3ld3 = load <3 x float>, ptr %gep3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  LoopInfo LI(DT);
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  ScalarEvolution SE(F, TLI, AC, DT, LI);

  sandboxir::Context Ctx(C);
  auto &SBF = *Ctx.createFunction(&F);
  auto &BB = *SBF.begin();
  auto It = std::next(BB.begin(), 4);
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *L3 = cast<sandboxir::LoadInst>(&*It++);

  auto *V2L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L3 = cast<sandboxir::LoadInst>(&*It++);
  (void)V2L3;

  auto *V3L0 = cast<sandboxir::LoadInst>(&*It++);
  (void)V3L0;
  auto *V3L1 = cast<sandboxir::LoadInst>(&*It++);
  (void)V3L1;
  auto *V3L2 = cast<sandboxir::LoadInst>(&*It++);
  (void)V3L2;
  auto *V3L3 = cast<sandboxir::LoadInst>(&*It++);
  (void)V3L3;

  EXPECT_TRUE(sandboxir::VecUtils::comesBeforeInMem(L0, L1, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::comesBeforeInMem(L0, L2, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::comesBeforeInMem(L0, L0, SE, DL));
  EXPECT_FALSE(sandboxir::VecUtils::comesBeforeInMem(L1, L0, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::comesBeforeInMem(L0, V2L1, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::comesBeforeInMem(L0, V2L2, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::comesBeforeInMem(V2L0, L1, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::comesBeforeInMem(V2L0, L2, SE, DL));
  EXPECT_TRUE(sandboxir::VecUtils::comesBeforeInMem(V2L0, L3, SE, DL));
}

TEST(VecUtils, GetNumElements) {
  LLVMContext C;
  auto *ElemTy = Type::getInt32Ty(C);
  EXPECT_EQ(sandboxir::VecUtils::getNumElements(ElemTy), 1);
  auto *VTy = FixedVectorType::get(ElemTy, 2);
  EXPECT_EQ(sandboxir::VecUtils::getNumElements(VTy), 2);
  auto *VTy1 = FixedVectorType::get(ElemTy, 1);
  EXPECT_EQ(sandboxir::VecUtils::getNumElements(VTy1), 1);
}

TEST(VecUtils, GetElementType) {
  LLVMContext C;
  auto *ElemTy = Type::getInt32Ty(C);
  EXPECT_EQ(sandboxir::VecUtils::getElementType(ElemTy), ElemTy);
  auto *VTy = FixedVectorType::get(ElemTy, 2);
  EXPECT_EQ(sandboxir::VecUtils::getElementType(VTy), ElemTy);
}

TEST(VecUtils, GetExpectedValue) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define float @foo(float %v, ptr %ptr) {
  store float %v, ptr %ptr
  ret float %v
}
)IR");
  Function &F = *M->getFunction("foo");
  BasicBlock &BB = *F.begin();
  auto It = BB.begin();
  auto *S0 = cast<StoreInst>(&*It++);
  auto *Ret = cast<ReturnInst>(&*It++);
  EXPECT_EQ(sandboxir::VecUtils::getExpectedValue(S0), S0->getValueOperand());
  EXPECT_EQ(sandboxir::VecUtils::getExpectedValue(Ret),
            Ret->getReturnValue());
}

TEST(Utils, GetExpectedType) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define float @foo(float %v, ptr %ptr) {
  store float %v, ptr %ptr
  ret float %v
}
)IR");
  Function &F = *M->getFunction("foo");
  BasicBlock &BB = *F.begin();
  auto It = BB.begin();
  auto *S0 = cast<StoreInst>(&*It++);
  auto *Ret = cast<ReturnInst>(&*It++);
  EXPECT_EQ(sandboxir::VecUtils::getExpectedType(S0),
            S0->getValueOperand()->getType());
  EXPECT_EQ(sandboxir::VecUtils::getExpectedType(Ret),
            Ret->getReturnValue()->getType());
}

TEST(Utils, GetNumLanes) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define <4 x float> @foo(float %v, <2 x float> %v2, <4 x float> %ret, ptr %ptr) {
  store float %v, ptr %ptr
  store <2 x float> %v2, ptr %ptr
  ret <4 x float> %ret
}
)IR");
  Function &F = *M->getFunction("foo");
  BasicBlock &BB = *F.begin();
  auto It = BB.begin();
  auto *S0 = cast<StoreInst>(&*It++);
  auto *S1 = cast<StoreInst>(&*It++);
  auto *Ret = cast<ReturnInst>(&*It++);
  EXPECT_EQ(sandboxir::VecUtils::getNumLanes(S0), 1);
  EXPECT_EQ(sandboxir::VecUtils::getNumLanes(S1), 2);
  EXPECT_EQ(sandboxir::VecUtils::getNumLanes(Ret), 4);
}

TEST(VecUtils, GetExpectedTypeReturnVoid) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo() {
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  BasicBlock &BB = *F.begin();
  auto It = BB.begin();
  auto *Ret = cast<ReturnInst>(&*It++);
  EXPECT_EQ(sandboxir::VecUtils::getExpectedType(Ret), Type::getVoidTy(C));
}

TEST(VecUtils, CeilPowerOf2) {
  EXPECT_EQ(sandboxir::VecUtils::getCeilPowerOf2(0), 0u);
  EXPECT_EQ(sandboxir::VecUtils::getCeilPowerOf2(1 << 0), 1u << 0);
  EXPECT_EQ(sandboxir::VecUtils::getCeilPowerOf2(1 << 1), 1u << 1);
  EXPECT_EQ(sandboxir::VecUtils::getCeilPowerOf2(1 << 31), 1u << 31);
  EXPECT_EQ(sandboxir::VecUtils::getCeilPowerOf2(1), 1u);
  EXPECT_EQ(sandboxir::VecUtils::getCeilPowerOf2(3), 4u);
  EXPECT_EQ(sandboxir::VecUtils::getCeilPowerOf2(5), 8u);
  EXPECT_EQ(sandboxir::VecUtils::getCeilPowerOf2((1 << 30) + 1), 1u << 31);
  EXPECT_EQ(sandboxir::VecUtils::getCeilPowerOf2((1 << 31) + 1), 0u);
}

TEST(VecUtils, FloorPowerOf2) {
  EXPECT_EQ(sandboxir::VecUtils::getFloorPowerOf2(0), 0u);
  EXPECT_EQ(sandboxir::VecUtils::getFloorPowerOf2(1 << 0), 1u << 0);
  EXPECT_EQ(sandboxir::VecUtils::getFloorPowerOf2(3), 2u);
  EXPECT_EQ(sandboxir::VecUtils::getFloorPowerOf2(4), 4u);
  EXPECT_EQ(sandboxir::VecUtils::getFloorPowerOf2(5), 4u);
  EXPECT_EQ(sandboxir::VecUtils::getFloorPowerOf2(7), 4u);
  EXPECT_EQ(sandboxir::VecUtils::getFloorPowerOf2(8), 8u);
  EXPECT_EQ(sandboxir::VecUtils::getFloorPowerOf2(9), 8u);
}

TEST(VecUtils, GetPointerDiffInBytes) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep0 = getelementptr inbounds float, ptr %ptr, i64 0
  %gep1 = getelementptr inbounds float, ptr %ptr, i64 1
  %gep2 = getelementptr inbounds float, ptr %ptr, i64 2
  %gep3 = getelementptr inbounds float, ptr %ptr, i64 3

  %ld0 = load float, ptr %gep0
  %ld1 = load float, ptr %gep1
  %ld2 = load float, ptr %gep2
  %ld3 = load float, ptr %gep3

  %v2ld0 = load <2 x float>, ptr %gep0
  %v2ld1 = load <2 x float>, ptr %gep1
  %v2ld2 = load <2 x float>, ptr %gep2
  %v2ld3 = load <2 x float>, ptr %gep3

  %v3ld0 = load <3 x float>, ptr %gep0
  %v3ld1 = load <3 x float>, ptr %gep1
  %v3ld2 = load <3 x float>, ptr %gep2
  %v3ld3 = load <3 x float>, ptr %gep3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  LoopInfo LI(DT);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  sandboxir::Context Ctx(C);

  auto &SBF = *Ctx.createFunction(&F);
  auto &BB = *SBF.begin();
  auto It = std::next(BB.begin(), 4);
  auto *L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *L3 = cast<sandboxir::LoadInst>(&*It++);
  (void)L3;

  auto *V2L0 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L2 = cast<sandboxir::LoadInst>(&*It++);
  auto *V2L3 = cast<sandboxir::LoadInst>(&*It++);

  auto *V3L0 = cast<sandboxir::LoadInst>(&*It++);
  (void)V3L0;
  auto *V3L1 = cast<sandboxir::LoadInst>(&*It++);
  auto *V3L2 = cast<sandboxir::LoadInst>(&*It++);
  (void)V3L2;
  auto *V3L3 = cast<sandboxir::LoadInst>(&*It++);
  (void)V3L3;

  EXPECT_EQ(
      *sandboxir::VecUtilsPrivileged::getPointerDiffInBytes(L0, L1, SE, DL),
      4);
  EXPECT_EQ(
      *sandboxir::VecUtilsPrivileged::getPointerDiffInBytes(L0, L2, SE, DL),
      8);
  EXPECT_EQ(
      *sandboxir::VecUtilsPrivileged::getPointerDiffInBytes(L1, L0, SE, DL),
      -4);
  EXPECT_EQ(
      *sandboxir::VecUtilsPrivileged::getPointerDiffInBytes(L0, V2L0, SE, DL),
      0);

  EXPECT_EQ(
      *sandboxir::VecUtilsPrivileged::getPointerDiffInBytes(L0, V2L1, SE, DL),
      4);
  EXPECT_EQ(
      *sandboxir::VecUtilsPrivileged::getPointerDiffInBytes(L0, V3L1, SE, DL),
      4);
  EXPECT_EQ(*sandboxir::VecUtilsPrivileged::getPointerDiffInBytes(V2L0, V2L2,
                                                                    SE, DL),
            8);
  EXPECT_EQ(*sandboxir::VecUtilsPrivileged::getPointerDiffInBytes(V2L0, V2L3,
                                                                    SE, DL),
            12);
  EXPECT_EQ(*sandboxir::VecUtilsPrivileged::getPointerDiffInBytes(V2L3, V2L0,
                                                                    SE, DL),
            -12);
}

TEST(VecUtils, GetInsertPointAfter_SameBB) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr0, ptr %ptr1) {
bb0:
  %ld0 = load float, ptr %ptr0
  %ld1 = load float, ptr %ptr1
  store float %ld0, ptr %ptr0
  store float %ld1, ptr %ptr1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  BasicBlock *BB0 = getBasicBlockByName(F, "bb0");
  auto It = BB0->begin();
  auto *L0 = &*It++;
  auto *L1 = &*It++;
  DmpVector<Value *> Bndl({L0, L1});
  BasicBlock::iterator ResIt =
      sandboxir::VecUtils::getInsertPointAfter(Bndl, BB0);
  EXPECT_EQ(ResIt, std::next(L1->getIterator()));
}

TEST(Utils, GetInsertPointAfter_DiffBBs) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr0, ptr %ptr1) {
bb0:
  %ld0 = load float, ptr %ptr0
  br label %bb1

bb1:
  %ld1 = load float, ptr %ptr1
  store float %ld0, ptr %ptr0
  store float %ld1, ptr %ptr1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  BasicBlock *BB0 = getBasicBlockByName(F, "bb0");
  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  auto It0 = BB0->begin();
  auto *L0 = &*It0++;
  auto It1 = BB1->begin();
  auto *L1 = &*It1++;
  DmpVector<Value *> Bndl({L0, L1});
  BasicBlock::iterator ResIt =
      sandboxir::VecUtils::getInsertPointAfter(Bndl, BB1);
  EXPECT_EQ(ResIt, std::next(L1->getIterator()));
}

TEST(Utils, GetInsertPointAfter_CrossBBs) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr0, ptr %ptr1) {
bb0:
  %ld0 = load float, ptr %ptr0
  %ld1 = load float, ptr %ptr1
  br label %bb1

bb1:
  store float %ld0, ptr %ptr0
  store float %ld1, ptr %ptr1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  BasicBlock *BB0 = getBasicBlockByName(F, "bb0");
  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  auto It = BB0->begin();
  auto *L0 = &*It++;
  auto *L1 = &*It++;
  DmpVector<Value *> Bndl({L0, L1});
  BasicBlock::iterator ResIt =
      sandboxir::VecUtils::getInsertPointAfter(Bndl, BB1);
  EXPECT_EQ(ResIt, BB1->begin());
}

TEST(Utils, GetInsertPointAfter_CrossBBs_WithPHis) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr0, ptr %ptr1) {
bb0:
  %ld0 = load float, ptr %ptr0
  %ld1 = load float, ptr %ptr1
  br label %bb1

bb1:
  %phi = phi i32 [ 0, %bb0 ], [ 1, %bb1 ]
  store float %ld0, ptr %ptr0
  store float %ld1, ptr %ptr1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  BasicBlock *BB0 = getBasicBlockByName(F, "bb0");
  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  auto It = BB0->begin();
  auto *L0 = &*It++;
  auto *L1 = &*It++;
  DmpVector<Value *> Bndl({L0, L1});
  BasicBlock::iterator ResIt =
      sandboxir::VecUtils::getInsertPointAfter(Bndl, BB1);
  EXPECT_EQ(ResIt, BB1->getFirstNonPHI()->getIterator());
}

TEST(Utils, GetInsertPointAfter_SkipPHIs) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr0) {
bb0:
  br label %bb1

bb1:
  %phi1 = phi i32 [ 1, %bb0 ], [ 1, %bb1 ]
  %phi2 = phi i32 [ 2, %bb0 ], [ 2, %bb1 ]
  %phi3 = phi i32 [ 3, %bb0 ], [ 3, %bb1 ]
  store float 0.0, ptr %ptr0
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  auto It = BB1->begin();
  auto *PHI1 = &*It++;
  auto *PHI2 = &*It++;
  auto *PHI3 = &*It++;
  auto *St0 = &*It++;
  {
    BasicBlock::iterator WhereIt = sandboxir::VecUtils::getInsertPointAfter(
        {PHI1, PHI2}, BB1, /*SkipPHIs=*/true);
    EXPECT_EQ(WhereIt, St0->getIterator());
  }
  {
    BasicBlock::iterator WhereIt = sandboxir::VecUtils::getInsertPointAfter(
        {PHI1, PHI2}, BB1, /*SkipPHIs=*/false);
    EXPECT_EQ(WhereIt, PHI3->getIterator());
  }
}

TEST(VecUtils, GetLowest_GetHighest) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(i8 %v) {
bb0:
  %A = add i8 %v, %v
  %B = add i8 %v, %v
  %C = add i8 %v, %v
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  LoopInfo LI(DT);
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  ScalarEvolution SE(F, TLI, AC, DT, LI);

  sandboxir::Context Ctx(C);
  auto &SBF = *Ctx.createFunction(&F);
  auto &BB = *SBF.begin();
  auto It = BB.begin();
  auto *IA = &*It++;
  auto *IB = &*It++;
  auto *IC = &*It++;
  DmpVector<sandboxir::Value *> ABC({IA, IB, IC});
  EXPECT_EQ(sandboxir::VecUtils::getLowest(ABC), IC);
  EXPECT_EQ(sandboxir::VecUtils::getHighest(ABC), IA);
  DmpVector<sandboxir::Value *> ACB({IA, IC, IB});
  EXPECT_EQ(sandboxir::VecUtils::getLowest(ACB), IC);
  EXPECT_EQ(sandboxir::VecUtils::getHighest(ACB), IA);
  DmpVector<sandboxir::Value *> CAB({IC, IA, IB});
  EXPECT_EQ(sandboxir::VecUtils::getLowest(CAB), IC);
  EXPECT_EQ(sandboxir::VecUtils::getHighest(CAB), IA);
  DmpVector<sandboxir::Value *> CBA({IC, IB, IA});
  EXPECT_EQ(sandboxir::VecUtils::getLowest(CBA), IC);
  EXPECT_EQ(sandboxir::VecUtils::getHighest(CBA), IA);
}
