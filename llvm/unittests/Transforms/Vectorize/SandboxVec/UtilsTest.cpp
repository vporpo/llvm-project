//===- UtilsTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Utils.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("UtilsTest", errs());
  return Mod;
}

static BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    if (BB.getName() == Name)
      return &BB;
  llvm_unreachable("Expected to find basic block!");
}

TEST(Utils, GetNumElements) {
  LLVMContext C;
  auto *ElemTy = Type::getInt32Ty(C);
  EXPECT_EQ(SBUtils::getNumElements(ElemTy), 1);
  auto *VTy = FixedVectorType::get(ElemTy, 2);
  EXPECT_EQ(SBUtils::getNumElements(VTy), 2);
  auto *VTy1 = FixedVectorType::get(ElemTy, 1);
  EXPECT_EQ(SBUtils::getNumElements(VTy1), 1);
}

TEST(Utils, GetElementType) {
  LLVMContext C;
  auto *ElemTy = Type::getInt32Ty(C);
  EXPECT_EQ(SBUtils::getElementType(ElemTy), ElemTy);
  auto *VTy = FixedVectorType::get(ElemTy, 2);
  EXPECT_EQ(SBUtils::getElementType(VTy), ElemTy);
}

TEST(Utils, GetExpectedValue) {
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
  EXPECT_EQ(SBUtils::getExpectedValue(S0), S0->getValueOperand());
  EXPECT_EQ(SBUtils::getExpectedValue(Ret), Ret->getReturnValue());
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
  EXPECT_EQ(SBUtils::getExpectedType(S0), S0->getValueOperand()->getType());
  EXPECT_EQ(SBUtils::getExpectedType(Ret), Ret->getReturnValue()->getType());
}

TEST(Utils, GetExpectedTypeReturnVoid) {
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
  EXPECT_EQ(SBUtils::getExpectedType(Ret), Type::getVoidTy(C));
}

TEST(Utils, GetPointerDiffInBytes) {
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
  SBContext Ctxt(C, AA);

  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &BB = *SBF.begin();
  auto It = std::next(BB.begin(), 4);
  auto *L0 = cast<SBLoadInstruction>(&*It++);
  auto *L1 = cast<SBLoadInstruction>(&*It++);
  auto *L2 = cast<SBLoadInstruction>(&*It++);
  auto *L3 = cast<SBLoadInstruction>(&*It++);
  (void)L3;

  auto *V2L0 = cast<SBLoadInstruction>(&*It++);
  auto *V2L1 = cast<SBLoadInstruction>(&*It++);
  auto *V2L2 = cast<SBLoadInstruction>(&*It++);
  auto *V2L3 = cast<SBLoadInstruction>(&*It++);

  auto *V3L0 = cast<SBLoadInstruction>(&*It++);
  (void)V3L0;
  auto *V3L1 = cast<SBLoadInstruction>(&*It++);
  auto *V3L2 = cast<SBLoadInstruction>(&*It++);
  (void)V3L2;
  auto *V3L3 = cast<SBLoadInstruction>(&*It++);
  (void)V3L3;

  EXPECT_EQ(*SBUtils::getPointerDiffInBytes(L0, L1, SE, DL), 4);
  EXPECT_EQ(*SBUtils::getPointerDiffInBytes(L0, L2, SE, DL), 8);
  EXPECT_EQ(*SBUtils::getPointerDiffInBytes(L1, L0, SE, DL), -4);
  EXPECT_EQ(*SBUtils::getPointerDiffInBytes(L0, V2L0, SE, DL), 0);

  EXPECT_EQ(*SBUtils::getPointerDiffInBytes(L0, V2L1, SE, DL), 4);
  EXPECT_EQ(*SBUtils::getPointerDiffInBytes(L0, V3L1, SE, DL), 4);
  EXPECT_EQ(*SBUtils::getPointerDiffInBytes(V2L0, V2L2, SE, DL), 8);
  EXPECT_EQ(*SBUtils::getPointerDiffInBytes(V2L0, V2L3, SE, DL), 12);
  EXPECT_EQ(*SBUtils::getPointerDiffInBytes(V2L3, V2L0, SE, DL), -12);
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

  SBContext Ctxt(C, AA);

  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &BB = *SBF.begin();
  auto It = std::next(BB.begin(), 4);
  auto *L0 = cast<SBLoadInstruction>(&*It++);
  auto *L1 = cast<SBLoadInstruction>(&*It++);
  auto *L2 = cast<SBLoadInstruction>(&*It++);
  auto *L3 = cast<SBLoadInstruction>(&*It++);

  auto *V2L0 = cast<SBLoadInstruction>(&*It++);
  auto *V2L1 = cast<SBLoadInstruction>(&*It++);
  auto *V2L2 = cast<SBLoadInstruction>(&*It++);
  auto *V2L3 = cast<SBLoadInstruction>(&*It++);

  auto *V3L0 = cast<SBLoadInstruction>(&*It++);
  auto *V3L1 = cast<SBLoadInstruction>(&*It++);
  auto *V3L2 = cast<SBLoadInstruction>(&*It++);
  auto *V3L3 = cast<SBLoadInstruction>(&*It++);

  // Scalar
  EXPECT_TRUE(SBUtils::areConsecutive(L0, L1, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(L1, L2, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(L2, L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L1, L0, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L2, L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L3, L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L0, L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L0, L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L1, L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L2, L0, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L3, L1, SE, DL));

  // Check 2-wide loads
  EXPECT_TRUE(SBUtils::areConsecutive(V2L0, V2L2, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(V2L1, V2L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L0, V2L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L1, V2L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L2, V2L3, SE, DL));

  EXPECT_FALSE(SBUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L3, V2L1, SE, DL));

  // Check 3-wide loads
  EXPECT_TRUE(SBUtils::areConsecutive(V3L0, V3L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L0, V3L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L1, V3L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L2, V3L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L1, V3L0, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L2, V3L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L3, V3L2, SE, DL));

  // Check mixes of vectors and scalar
  EXPECT_TRUE(SBUtils::areConsecutive(L0, V2L1, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(L1, V2L2, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(V2L0, L2, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(V3L0, L3, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(V2L0, V3L2, SE, DL));

  EXPECT_FALSE(SBUtils::areConsecutive(L0, V2L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L0, V3L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L0, V2L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L0, V3L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L0, L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L0, L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L0, V2L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L0, V2L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L1, L0, SE, DL));
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

  SBContext Ctxt(C, AA);

  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &BB = *SBF.begin();
  auto It = std::next(BB.begin(), 4);
  auto *L0 = cast<SBLoadInstruction>(&*It++);
  auto *L1 = cast<SBLoadInstruction>(&*It++);
  auto *L2 = cast<SBLoadInstruction>(&*It++);
  auto *L3 = cast<SBLoadInstruction>(&*It++);

  auto *V2L0 = cast<SBLoadInstruction>(&*It++);
  auto *V2L1 = cast<SBLoadInstruction>(&*It++);
  auto *V2L2 = cast<SBLoadInstruction>(&*It++);
  auto *V2L3 = cast<SBLoadInstruction>(&*It++);

  auto *V3L0 = cast<SBLoadInstruction>(&*It++);
  auto *V3L1 = cast<SBLoadInstruction>(&*It++);
  auto *V3L2 = cast<SBLoadInstruction>(&*It++);
  auto *V3L3 = cast<SBLoadInstruction>(&*It++);

  // Scalar
  EXPECT_TRUE(SBUtils::areConsecutive(L0, L1, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(L1, L2, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(L2, L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L1, L0, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L2, L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L3, L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L0, L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L0, L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L1, L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L2, L0, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L3, L1, SE, DL));

  // Check 2-wide loads
  EXPECT_TRUE(SBUtils::areConsecutive(V2L0, V2L2, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(V2L1, V2L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L0, V2L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L1, V2L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L2, V2L3, SE, DL));

  EXPECT_FALSE(SBUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L3, V2L1, SE, DL));

  // Check 3-wide loads
  EXPECT_TRUE(SBUtils::areConsecutive(V3L0, V3L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L0, V3L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L1, V3L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L2, V3L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L1, V3L0, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L2, V3L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L3, V3L2, SE, DL));

  // Check mixes of vectors and scalar
  EXPECT_TRUE(SBUtils::areConsecutive(L0, V2L1, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(L1, V2L2, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(V2L0, L2, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(V3L0, L3, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(V2L0, V3L2, SE, DL));

  EXPECT_FALSE(SBUtils::areConsecutive(L0, V2L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L0, V3L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L0, V2L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L0, V3L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L0, L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L0, L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L0, V2L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L0, V2L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L1, L0, SE, DL));
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

  SBContext Ctxt(C, AA);

  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &BB = *SBF.begin();
  auto It = std::next(BB.begin(), 4);
  auto *L0 = cast<SBLoadInstruction>(&*It++);
  auto *L1 = cast<SBLoadInstruction>(&*It++);
  auto *L2 = cast<SBLoadInstruction>(&*It++);
  auto *L3 = cast<SBLoadInstruction>(&*It++);

  auto *V2L0 = cast<SBLoadInstruction>(&*It++);
  auto *V2L1 = cast<SBLoadInstruction>(&*It++);
  auto *V2L2 = cast<SBLoadInstruction>(&*It++);
  auto *V2L3 = cast<SBLoadInstruction>(&*It++);

  auto *V3L0 = cast<SBLoadInstruction>(&*It++);
  auto *V3L1 = cast<SBLoadInstruction>(&*It++);
  auto *V3L2 = cast<SBLoadInstruction>(&*It++);
  auto *V3L3 = cast<SBLoadInstruction>(&*It++);

  // Scalar
  EXPECT_TRUE(SBUtils::areConsecutive(L0, L1, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(L1, L2, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(L2, L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L1, L0, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L2, L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L3, L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L0, L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L0, L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L1, L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L2, L0, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L3, L1, SE, DL));

  // Check 2-wide loads
  EXPECT_TRUE(SBUtils::areConsecutive(V2L0, V2L2, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(V2L1, V2L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L0, V2L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L1, V2L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L2, V2L3, SE, DL));

  EXPECT_FALSE(SBUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L3, V2L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L3, V2L1, SE, DL));

  // Check 3-wide loads
  EXPECT_TRUE(SBUtils::areConsecutive(V3L0, V3L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L0, V3L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L1, V3L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L2, V3L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L1, V3L0, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L2, V3L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L3, V3L2, SE, DL));

  // Check mixes of vectors and scalar
  EXPECT_TRUE(SBUtils::areConsecutive(L0, V2L1, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(L1, V2L2, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(V2L0, L2, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(V3L0, L3, SE, DL));
  EXPECT_TRUE(SBUtils::areConsecutive(V2L0, V3L2, SE, DL));

  EXPECT_FALSE(SBUtils::areConsecutive(L0, V2L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L0, V3L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(L0, V2L3, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L0, V3L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L0, L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L0, L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L0, V2L1, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V3L0, V2L2, SE, DL));
  EXPECT_FALSE(SBUtils::areConsecutive(V2L1, L0, SE, DL));
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

  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &BB = *SBF.begin();
  auto It = std::next(BB.begin(), 4);
  auto *L0 = cast<SBLoadInstruction>(&*It++);
  auto *L1 = cast<SBLoadInstruction>(&*It++);
  auto *L2 = cast<SBLoadInstruction>(&*It++);
  auto *L3 = cast<SBLoadInstruction>(&*It++);

  auto *V2L0 = cast<SBLoadInstruction>(&*It++);
  auto *V2L1 = cast<SBLoadInstruction>(&*It++);
  auto *V2L2 = cast<SBLoadInstruction>(&*It++);
  auto *V2L3 = cast<SBLoadInstruction>(&*It++);
  (void)V2L3;

  auto *V3L0 = cast<SBLoadInstruction>(&*It++);
  (void)V3L0;
  auto *V3L1 = cast<SBLoadInstruction>(&*It++);
  (void)V3L1;
  auto *V3L2 = cast<SBLoadInstruction>(&*It++);
  (void)V3L2;
  auto *V3L3 = cast<SBLoadInstruction>(&*It++);
  (void)V3L3;

  EXPECT_TRUE(SBUtils::comesBeforeInMem(L0, L1, SE, DL));
  EXPECT_TRUE(SBUtils::comesBeforeInMem(L0, L2, SE, DL));
  EXPECT_FALSE(SBUtils::comesBeforeInMem(L0, L0, SE, DL));
  EXPECT_FALSE(SBUtils::comesBeforeInMem(L1, L0, SE, DL));
  EXPECT_TRUE(SBUtils::comesBeforeInMem(L0, V2L1, SE, DL));
  EXPECT_TRUE(SBUtils::comesBeforeInMem(L0, V2L2, SE, DL));
  EXPECT_TRUE(SBUtils::comesBeforeInMem(V2L0, L1, SE, DL));
  EXPECT_TRUE(SBUtils::comesBeforeInMem(V2L0, L2, SE, DL));
  EXPECT_TRUE(SBUtils::comesBeforeInMem(V2L0, L3, SE, DL));
}

TEST(Utils, GetLowest_GetHighest) {
  LLVMContext LLVMC;
  std::unique_ptr<Module> M = parseIR(LLVMC, R"IR(
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

  SBContext Ctxt(LLVMC, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &BB = *SBF.begin();
  auto It = BB.begin();
  auto *A = &*It++;
  auto *B = &*It++;
  auto *C = &*It++;
  SBValBundle ABC({A, B, C});
  EXPECT_EQ(SBUtils::getLowest(ABC), C);
  EXPECT_EQ(SBUtils::getHighest(ABC), A);
  SBValBundle ACB({A, C, B});
  EXPECT_EQ(SBUtils::getLowest(ACB), C);
  EXPECT_EQ(SBUtils::getHighest(ACB), A);
  SBValBundle CAB({C, A, B});
  EXPECT_EQ(SBUtils::getLowest(CAB), C);
  EXPECT_EQ(SBUtils::getHighest(CAB), A);
  SBValBundle CBA({C, B, A});
  EXPECT_EQ(SBUtils::getLowest(CBA), C);
  EXPECT_EQ(SBUtils::getHighest(CBA), A);
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
  EXPECT_EQ(SBUtils::getNumLanes(S0), 1);
  EXPECT_EQ(SBUtils::getNumLanes(S1), 2);
  EXPECT_EQ(SBUtils::getNumLanes(Ret), 4);
}

TEST(Utils, HasNUsersOrMore) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define float @foo(ptr %ptr) {
  %ld0 = load float, ptr %ptr
  %ld1 = load float, ptr %ptr
  %add0 = fadd float %ld0, %ld0
  %add1 = fadd float %ld1, %ld1
  ret float %ld1
}
)IR");
  Function &F = *M->getFunction("foo");
  BasicBlock &BB = *F.begin();
  auto It = BB.begin();
  auto *L0 = &*It++;
  auto *L1 = &*It++;
  auto *Add0 = &*It++;
  (void)Add0;
  auto *Add1 = &*It++;
  (void)Add1;
  EXPECT_TRUE(SBUtils::hasNUsersOrMore(L0, 1));
  EXPECT_FALSE(SBUtils::hasNUsersOrMore(L0, 2));
  EXPECT_TRUE(SBUtils::hasNUsersOrMore(L1, 2));
  EXPECT_FALSE(SBUtils::hasNUsersOrMore(L1, 3));
}

TEST(Utils, CeilPowerOf2) {
  EXPECT_EQ(SBUtils::getCeilPowerOf2(0), 0u);
  EXPECT_EQ(SBUtils::getCeilPowerOf2(1 << 0), 1u << 0);
  EXPECT_EQ(SBUtils::getCeilPowerOf2(1 << 1), 1u << 1);
  EXPECT_EQ(SBUtils::getCeilPowerOf2(1 << 31), 1u << 31);
  EXPECT_EQ(SBUtils::getCeilPowerOf2(1), 1u);
  EXPECT_EQ(SBUtils::getCeilPowerOf2(3), 4u);
  EXPECT_EQ(SBUtils::getCeilPowerOf2(5), 8u);
  EXPECT_EQ(SBUtils::getCeilPowerOf2((1 << 30) + 1), 1u << 31);
  EXPECT_EQ(SBUtils::getCeilPowerOf2((1 << 31) + 1), 0u);
}

TEST(Utils, FloorPowerOf2) {
  EXPECT_EQ(SBUtils::getFloorPowerOf2(0), 0u);
  EXPECT_EQ(SBUtils::getFloorPowerOf2(1 << 0), 1u << 0);
  EXPECT_EQ(SBUtils::getFloorPowerOf2(3), 2u);
  EXPECT_EQ(SBUtils::getFloorPowerOf2(4), 4u);
  EXPECT_EQ(SBUtils::getFloorPowerOf2(5), 4u);
  EXPECT_EQ(SBUtils::getFloorPowerOf2(7), 4u);
  EXPECT_EQ(SBUtils::getFloorPowerOf2(8), 8u);
  EXPECT_EQ(SBUtils::getFloorPowerOf2(9), 8u);
}

TEST(Utils, GetInsertPointAfter_SameBB) {
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
  ValueBundle Bndl({L0, L1});
  BasicBlock::iterator ResIt = SBUtils::getInsertPointAfter(Bndl, BB0);
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
  ValueBundle Bndl({L0, L1});
  BasicBlock::iterator ResIt = SBUtils::getInsertPointAfter(Bndl, BB1);
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
  ValueBundle Bndl({L0, L1});
  BasicBlock::iterator ResIt = SBUtils::getInsertPointAfter(Bndl, BB1);
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
  ValueBundle Bndl({L0, L1});
  BasicBlock::iterator ResIt = SBUtils::getInsertPointAfter(Bndl, BB1);
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
    BasicBlock::iterator WhereIt =
        SBUtils::getInsertPointAfter({PHI1, PHI2}, BB1, /*SkipPHIs=*/true);
    EXPECT_EQ(WhereIt, St0->getIterator());
  }
  {
    BasicBlock::iterator WhereIt =
        SBUtils::getInsertPointAfter({PHI1, PHI2}, BB1, /*SkipPHIs=*/false);
    EXPECT_EQ(WhereIt, PHI3->getIterator());
  }
}

TEST(Utils, ShuffleMask) {
  SmallVector<int> IMask{0, 1, 2, 3};
  ShuffleMask IdentityMask(IMask);
  EXPECT_EQ(IdentityMask.size(), 4u);
  EXPECT_TRUE(IdentityMask.isIdentity());
  EXPECT_EQ(ShuffleMask::getIdentity(4), IdentityMask);
  EXPECT_TRUE(IdentityMask.isInOrder());
  for (int Idx = 0; Idx != 4; ++Idx)
    EXPECT_EQ(IdentityMask[Idx], Idx);

  SmallVector<int> IOMask{1, 2};
  ShuffleMask InOrderMask(IOMask);
  EXPECT_EQ(InOrderMask.size(), 2u);
  EXPECT_FALSE(InOrderMask.isIdentity());
  EXPECT_NE(InOrderMask, IdentityMask);
  EXPECT_NE(IdentityMask, InOrderMask);
  EXPECT_TRUE(InOrderMask.isInOrder());
  for (auto [Idx, Val] : enumerate(IOMask))
    EXPECT_EQ(InOrderMask[Idx], Val);

  SmallVector<int> IncOMask{1, 3, 4};
  ShuffleMask IncreasingOrderMask(IncOMask);
  EXPECT_EQ(IncreasingOrderMask.size(), 3u);
  EXPECT_FALSE(IncreasingOrderMask.isIdentity());
  EXPECT_NE(IncreasingOrderMask, InOrderMask);
  EXPECT_FALSE(IncreasingOrderMask.isInOrder());
  EXPECT_TRUE(IncreasingOrderMask.isIncreasingOrder());
  for (auto [Idx, Val] : enumerate(IncOMask))
    EXPECT_EQ(IncreasingOrderMask[Idx], Val);

  ShuffleMask NotIncreasingOrderMask(ArrayRef<int>{2, 2, 3});
  EXPECT_EQ(NotIncreasingOrderMask.size(), 3u);
  EXPECT_FALSE(NotIncreasingOrderMask.isIdentity());
  EXPECT_NE(NotIncreasingOrderMask, IncreasingOrderMask);
  EXPECT_FALSE(NotIncreasingOrderMask.isInOrder());
  EXPECT_FALSE(NotIncreasingOrderMask.isIncreasingOrder());

  ShuffleMask NotIncreasingOrderMask2(ArrayRef<int>{2, 1, 3, 4});
  EXPECT_EQ(NotIncreasingOrderMask2.size(), 4u);
  EXPECT_FALSE(NotIncreasingOrderMask2.isIdentity());
  EXPECT_NE(NotIncreasingOrderMask2, NotIncreasingOrderMask);
  EXPECT_FALSE(NotIncreasingOrderMask2.isInOrder());
  EXPECT_FALSE(NotIncreasingOrderMask2.isIncreasingOrder());
}

TEST(Utils, ShuffleMask_Inverse_Combine) {
  ShuffleMask Mask({1, 2, 3, 0});
  ShuffleMask FlippedMask = Mask.getInverse();
  EXPECT_TRUE(FlippedMask == ShuffleMask({3, 0, 1, 2}));
  ShuffleMask CombinedMask = FlippedMask.combine(Mask);
  EXPECT_TRUE(CombinedMask.isIdentity());
  ShuffleMask CombinedMaskRev = Mask.combine(FlippedMask);
  EXPECT_TRUE(CombinedMaskRev.isIdentity());

  ShuffleMask MaskA({0, 2, 1, 3});
  ShuffleMask MaskB({1, 2, 3, 0});
  EXPECT_EQ(MaskA.combine(MaskB), ShuffleMask({1, 3, 2, 0}));
  EXPECT_EQ(MaskB.combine(MaskA), ShuffleMask({2, 1, 3, 0}));
}
