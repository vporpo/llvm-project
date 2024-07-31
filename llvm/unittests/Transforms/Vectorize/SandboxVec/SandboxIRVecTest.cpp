//===- SandboxIRVecTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIRVec.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

struct SandboxIRVecTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;
  TargetLibraryInfoImpl TLII;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("SandboxIRVecTest", errs());
  }
  BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
    for (BasicBlock &BB : F)
      if (BB.getName() == Name)
        return &BB;
    llvm_unreachable("Expected to find basic block!");
  }
};

TEST_F(SandboxIRVecTest, ShuffleMask) {
  SmallVector<int> IMask{0, 1, 2, 3};
  sandboxir::ShuffleMask IdentityMask(IMask);
  EXPECT_EQ(IdentityMask.size(), 4u);
  EXPECT_TRUE(IdentityMask.isIdentity());
  EXPECT_EQ(sandboxir::ShuffleMask::getIdentity(4), IdentityMask);
  EXPECT_TRUE(IdentityMask.isInOrder());
  for (int Idx = 0; Idx != 4; ++Idx)
    EXPECT_EQ(IdentityMask[Idx], Idx);

  SmallVector<int> IOMask{1, 2};
  sandboxir::ShuffleMask InOrderMask(IOMask);
  EXPECT_EQ(InOrderMask.size(), 2u);
  EXPECT_FALSE(InOrderMask.isIdentity());
  EXPECT_NE(InOrderMask, IdentityMask);
  EXPECT_NE(IdentityMask, InOrderMask);
  EXPECT_TRUE(InOrderMask.isInOrder());
  for (auto [Idx, Val] : enumerate(IOMask))
    EXPECT_EQ(InOrderMask[Idx], Val);

  SmallVector<int> IncOMask{1, 3, 4};
  sandboxir::ShuffleMask IncreasingOrderMask(IncOMask);
  EXPECT_EQ(IncreasingOrderMask.size(), 3u);
  EXPECT_FALSE(IncreasingOrderMask.isIdentity());
  EXPECT_NE(IncreasingOrderMask, InOrderMask);
  EXPECT_FALSE(IncreasingOrderMask.isInOrder());
  EXPECT_TRUE(IncreasingOrderMask.isIncreasingOrder());
  for (auto [Idx, Val] : enumerate(IncOMask))
    EXPECT_EQ(IncreasingOrderMask[Idx], Val);

  sandboxir::ShuffleMask NotIncreasingOrderMask(ArrayRef<int>{2, 2, 3});
  EXPECT_EQ(NotIncreasingOrderMask.size(), 3u);
  EXPECT_FALSE(NotIncreasingOrderMask.isIdentity());
  EXPECT_NE(NotIncreasingOrderMask, IncreasingOrderMask);
  EXPECT_FALSE(NotIncreasingOrderMask.isInOrder());
  EXPECT_FALSE(NotIncreasingOrderMask.isIncreasingOrder());

  sandboxir::ShuffleMask NotIncreasingOrderMask2(ArrayRef<int>{2, 1, 3, 4});
  EXPECT_EQ(NotIncreasingOrderMask2.size(), 4u);
  EXPECT_FALSE(NotIncreasingOrderMask2.isIdentity());
  EXPECT_NE(NotIncreasingOrderMask2, NotIncreasingOrderMask);
  EXPECT_FALSE(NotIncreasingOrderMask2.isInOrder());
  EXPECT_FALSE(NotIncreasingOrderMask2.isIncreasingOrder());
}

TEST_F(SandboxIRVecTest, ShuffleMask_Inverse_Combine) {
  sandboxir::ShuffleMask Mask({1, 2, 3, 0});
  sandboxir::ShuffleMask FlippedMask = Mask.getInverse();
  EXPECT_TRUE(FlippedMask == sandboxir::ShuffleMask({3, 0, 1, 2}));
  sandboxir::ShuffleMask CombinedMask = FlippedMask.combine(Mask);
  EXPECT_TRUE(CombinedMask.isIdentity());
  sandboxir::ShuffleMask CombinedMaskRev = Mask.combine(FlippedMask);
  EXPECT_TRUE(CombinedMaskRev.isIdentity());

  sandboxir::ShuffleMask MaskA({0, 2, 1, 3});
  sandboxir::ShuffleMask MaskB({1, 2, 3, 0});
  EXPECT_EQ(MaskA.combine(MaskB), sandboxir::ShuffleMask({1, 3, 2, 0}));
  EXPECT_EQ(MaskB.combine(MaskA), sandboxir::ShuffleMask({2, 1, 3, 0}));
}

TEST_F(SandboxIRVecTest, IteratorsMultiInstr) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, i32 %v2, i32 %v3) {
  %Pack0 = insertelement <4 x i32> poison, i32 %v0, i64 0
  %Pack1 = insertelement <4 x i32> %Pack0, i32 %v2, i64 1
  %Pack2 = insertelement <4 x i32> %Pack1, i32 %v1, i64 2
  %Pack3 = insertelement <4 x i32> %Pack2, i32 %v2, i64 3
  %add0 = add i32 %v0, %v0
  %add1 = add i32 %v1, %v1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto BBIt = BB->begin();
  Instruction *Pack0 = &*BBIt++;
  Instruction *Pack1 = &*BBIt++;
  Instruction *Pack2 = &*BBIt++;
  Instruction *Pack3 = &*BBIt++;
  Instruction *Add0 = &*BBIt++;
  Instruction *Add1 = &*BBIt++;
  Instruction *Ret = &*BBIt++;
  auto &SBBB = *Ctx.createBasicBlock(BB);
  auto It = SBBB.begin();

  auto *I0 = &*It++;
  EXPECT_EQ(I0, Ctx.getValue(Pack0));
  EXPECT_EQ(I0, Ctx.getValue(Pack1));
  EXPECT_EQ(I0, Ctx.getValue(Pack2));
  EXPECT_EQ(I0, Ctx.getValue(Pack3));
  auto *I1 = &*It++;
  EXPECT_EQ(I1, Ctx.getValue(Add0));
  auto *I2 = &*It++;
  EXPECT_EQ(I2, Ctx.getValue(Add1));
  auto *I3 = &*It++;
  EXPECT_EQ(I3, Ctx.getValue(Ret));
  EXPECT_EQ(It, SBBB.end());
#ifndef NDEBUG
  EXPECT_DEATH(++It, "Already.*");
#endif
  --It;
  EXPECT_EQ(&*It--, Ctx.getValue(Ret));
  EXPECT_EQ(&*It--, Ctx.getValue(Add1));
  EXPECT_EQ(&*It--, Ctx.getValue(Add0));
  EXPECT_EQ(&*It, Ctx.getValue(Pack0));
  EXPECT_EQ(It, SBBB.begin());
#ifndef NDEBUG
  EXPECT_DEATH(--It, "Already.*");
#endif

  // Check iterator equality.
  auto *Pack = cast<sandboxir::PackInst>(Ctx.getValue(Pack3));
  EXPECT_TRUE(SBBB.begin() == Pack->getIterator());
}

TEST_F(SandboxIRVecTest, PackUseUser_WhenFoldedConstant) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld = load i32, ptr %ptr
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto It = BB->begin();
  Instruction *Ld = &*It++;
  Constant *C1 = Constant::getIntegerValue(Type::getInt32Ty(C), APInt(32, 42));
  auto &SBBB = *Ctx.createBasicBlock(BB);
  sandboxir::Value *SBL = Ctx.getValue(Ld);
  auto *SBC = Ctx.getOrCreateConstant(C1);
  DmpVector<sandboxir::Value *> SBPackInstructions({SBC, SBL});
  auto WhereIt =
      sandboxir::VecUtils::getInsertPointAfter(SBPackInstructions, &SBBB);
  auto *Pack = cast<sandboxir::PackInst>(
      sandboxir::PackInst::create(SBPackInstructions, WhereIt, &SBBB, Ctx));

  auto *LLVMI = cast<llvm::InsertElementInst>(&*std::next(BB->begin()));
  EXPECT_EQ(Pack, Ctx.getValue(LLVMI));

  // When a pack is reading from a folded constant vector, like <poison, i8 42>
  // Then the pack's operand is `const` inside the vector constant, not the
  // vector constant. So the LLVM use is `i8 42` -> <poison, i8 42>.
  // But the SandboxIR Use should be `i8 42`->Pack
  sandboxir::Use Use0 = Pack->getOperandUse(0);
  EXPECT_EQ(Use0.getUser(), Pack);
}

// Check that we drop references once we erase a Pack.
TEST_F(SandboxIRVecTest, SBErasePackDropReferences) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ptr0 = getelementptr float, ptr %ptr, i32 0
  %VecL = load <2 x float>, ptr %ptr0, align 4
  %Unpack = extractelement <2 x float> %VecL, i64 0
  %Pack = insertelement <2 x float> poison, float %Unpack, i64 0
  %Pack1 = insertelement <2 x float> %Pack, float %Unpack, i64 1
  store <2 x float> %VecL, ptr %ptr0, align 4
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();
  It++; // Skip over %ptr0.
  It++; // Skip over %VecL.
  auto *Unpack = cast<sandboxir::UnpackInst>(&*It++);
  auto *Pack = cast<sandboxir::PackInst>(&*It++);
  Pack->eraseFromParent();

  EXPECT_TRUE(Unpack->users().empty());
}

TEST_F(SandboxIRVecTest, SBOperandUseIterator_Pack) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1) {
  %Pack0 = insertelement <2 x i32> poison, i32 %v0, i64 0
  %Pack1 = insertelement <2 x i32> %Pack0, i32 %v1, i64 1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto BBIt = BB->begin();
  Instruction *I0 = &*BBIt++;
  (void)I0;
  Instruction *I1 = &*BBIt++;
  (void)I1;
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *Ctx.getBasicBlock(BB);
  auto *SBArg0 = SBF.getArg(0);
  auto *SBArg1 = SBF.getArg(1);
  auto It = SBBB.begin();
  auto *Pack0 = cast<sandboxir::PackInst>(&*It++);
  sandboxir::OperandUseIterator UseIt = Pack0->op_begin();
  EXPECT_EQ(*UseIt, SBArg0);
  ++UseIt;
  EXPECT_EQ(*UseIt, SBArg1);
  ++UseIt;
  EXPECT_EQ(UseIt, Pack0->op_end());
}

TEST_F(SandboxIRVecTest, SBOperandUseIterator_Unpack) {
  parseIR(C, R"IR(
define void @foo(<2 x i32> %Vec) {
  %Unpack = extractelement <2 x i32> %Vec, i64 0
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto BBIt = BB->begin();
  Instruction *I0 = &*BBIt++;
  (void)I0;
  Instruction *I1 = &*BBIt++;
  (void)I1;
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *Ctx.getBasicBlock(BB);
  auto *SBArg0 = SBF.getArg(0);
  auto It = SBBB.begin();
  auto *Unpack0 = cast<sandboxir::UnpackInst>(&*It++);
  sandboxir::OperandUseIterator UseIt = Unpack0->op_begin();
  EXPECT_EQ(*UseIt, SBArg0);
  ++UseIt;
  EXPECT_EQ(UseIt, Unpack0->op_end());
}

TEST_F(SandboxIRVecTest, SBOperandUseIterator_Shuffle) {
  parseIR(C, R"IR(
define void @foo(<2 x i32> %vec) {
  %Shuffle = shufflevector <2 x i32> %vec, <2 x i32> poison, <2 x i32> <i32 1, i32 0>
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto BBIt = BB->begin();
  Instruction *I0 = &*BBIt++;
  (void)I0;
  Instruction *I1 = &*BBIt++;
  (void)I1;
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *Ctx.getBasicBlock(BB);
  auto *SBArg0 = SBF.getArg(0);
  auto It = SBBB.begin();
  auto *Shuffle0 = cast<sandboxir::ShuffleInst>(&*It++);
  sandboxir::OperandUseIterator UseIt = Shuffle0->op_begin();
  EXPECT_EQ(*UseIt, SBArg0);
  ++UseIt;
  EXPECT_EQ(UseIt, Shuffle0->op_end());
}

#ifndef NDEBUG
TEST_F(SandboxIRVecTest, MoveBefore_Multi) {
  parseIR(C, R"IR(
define void @foo(i8 %v0, i8 %v1, i8 %p0, i8 %p1) {
  %add0 = add i8 %v0, %v0
  %ins0 = insertelement <2 x i8> poison, i8 %p0, i32 0
  %ins1 = insertelement <2 x i8> %ins0, i8 %p1, i32 1
  %add1 = add i8 %v1, %v1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto RawIt = BB->begin();
  auto *Add0 = &*RawIt++;
  auto *Ins0 = &*RawIt++;
  auto *Ins1 = &*RawIt++;
  auto *Add1 = &*RawIt++;
  auto *Ret = &*RawIt++;

  auto &SBBB = *Ctx.createBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  auto It = SBBB.begin();
  auto *SBAdd0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *Pack = cast<sandboxir::PackInst>(&*It++);
  auto *SBAdd1 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *SBRet = &*It++;

  Ctx.disableCallbacks();

  // Test single-IR instruction move before multi-IR instruction
  SBAdd1->moveBefore(Pack);
  It = SBBB.begin();
  EXPECT_EQ(&*It++, SBAdd0);
  EXPECT_EQ(&*It++, SBAdd1);
  EXPECT_EQ(&*It++, Pack);
  EXPECT_EQ(&*It++, SBRet);

  RawIt = BB->begin();
  EXPECT_EQ(&*RawIt++, Add0);
  EXPECT_EQ(&*RawIt++, Add1);
  EXPECT_EQ(&*RawIt++, Ins0);
  EXPECT_EQ(&*RawIt++, Ins1);
  EXPECT_EQ(&*RawIt++, Ret);

  // Test multi-IR instruction move before single-IR instruction
  Pack->moveBefore(SBAdd1);
  It = SBBB.begin();
  EXPECT_EQ(&*It++, SBAdd0);
  EXPECT_EQ(&*It++, Pack);
  EXPECT_EQ(&*It++, SBAdd1);
  EXPECT_EQ(&*It++, SBRet);

  RawIt = BB->begin();
  EXPECT_EQ(&*RawIt++, Add0);
  EXPECT_EQ(&*RawIt++, Ins0);
  EXPECT_EQ(&*RawIt++, Ins1);
  EXPECT_EQ(&*RawIt++, Add1);
  EXPECT_EQ(&*RawIt++, Ret);
}
#endif

#ifndef NDEBUG
TEST_F(SandboxIRVecTest, MoveAfter_Multi) {
  parseIR(C, R"IR(
define void @foo(i8 %v0, i8 %v1, i8 %p0, i8 %p1) {
  %add0 = add i8 %v0, %v0
  %ins0 = insertelement <2 x i8> poison, i8 %p0, i32 0
  %ins1 = insertelement <2 x i8> %ins0, i8 %p1, i32 1
  %add1 = add i8 %v1, %v1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto RawIt = BB->begin();
  auto *Add0 = &*RawIt++;
  auto *Ins0 = &*RawIt++;
  auto *Ins1 = &*RawIt++;
  auto *Add1 = &*RawIt++;
  auto *Ret = &*RawIt++;

  auto &SBBB = *Ctx.createBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  auto It = SBBB.begin();
  auto *SBAdd0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *Pack = cast<sandboxir::PackInst>(&*It++);
  auto *SBAdd1 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *SBRet = &*It++;

  Ctx.disableCallbacks();

  // Move multi-IR instruction after single-IR
  Pack->moveAfter(SBAdd1);
  It = SBBB.begin();
  EXPECT_EQ(&*It++, SBAdd0);
  EXPECT_EQ(&*It++, SBAdd1);
  EXPECT_EQ(&*It++, Pack);
  EXPECT_EQ(&*It++, SBRet);

  RawIt = BB->begin();
  EXPECT_EQ(&*RawIt++, Add0);
  EXPECT_EQ(&*RawIt++, Add1);
  EXPECT_EQ(&*RawIt++, Ins0);
  EXPECT_EQ(&*RawIt++, Ins1);
  EXPECT_EQ(&*RawIt++, Ret);

  // Move single-IR instruction after multi-IR
  SBAdd1->moveAfter(Pack);
  It = SBBB.begin();
  EXPECT_EQ(&*It++, SBAdd0);
  EXPECT_EQ(&*It++, Pack);
  EXPECT_EQ(&*It++, SBAdd1);
  EXPECT_EQ(&*It++, SBRet);

  RawIt = BB->begin();
  EXPECT_EQ(&*RawIt++, Add0);
  EXPECT_EQ(&*RawIt++, Ins0);
  EXPECT_EQ(&*RawIt++, Ins1);
  EXPECT_EQ(&*RawIt++, Add1);
  EXPECT_EQ(&*RawIt++, Ret);
}
#endif

TEST_F(SandboxIRVecTest, Pack_GetOperandUseIdx) {
  parseIR(C, R"IR(
define <4 x float> @foo(float %arg0, <2 x float> %arg1, float %arg2, <2 x float> %arg3, <4 x float> %arg4) {
  %ins0 = insertelement <4 x float> poison, float %arg0, i32 0
  %extr0 = extractelement <2 x float> %arg1, i32 0
  %ins1 = insertelement <4 x float> %ins0, float %extr0, i32 1
  %extr1 = extractelement <2 x float> %arg1, i32 1
  %ins2 = insertelement <4 x float> %ins1, float %extr1, i32 2
  %ins3 = insertelement <4 x float> %ins2, float %arg2, i32 3
  ret <4 x float> %ins3
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto *SBF = Ctx.createFunction(&F);
  (void)SBF;
  auto *SBBB = Ctx.getBasicBlock(BB);
  auto SBIt = SBBB->begin();
  auto *Pack = cast<sandboxir::PackInst>(&*SBIt++);
  auto *SBRet = &*SBIt++;

  auto It = BB->begin();
  Instruction *Ins0 = &*It++;
  Instruction *Extr0 = &*It++;
  Instruction *Ins1 = &*It++;
  (void)Ins1;
  Instruction *Extr1 = &*It++;
  (void)Extr1;
  Instruction *Ins2 = &*It++;
  (void)Ins2;
  Instruction *Ins3 = &*It++;
  (void)Ins3;
  Instruction *Ret = &*It++;
  (void)Ret;

  const Use &Use0 = Ins0->getOperandUse(1);
  const Use &Use1_a = Extr0->getOperandUse(0);
  const Use &Use1_b = Extr1->getOperandUse(0);
  const Use &Use2 = Ins3->getOperandUse(1);
  const Use &RetUse = Ret->getOperandUse(0);
  EXPECT_EQ(sandboxir::UserAttorney::getOperandUseIdx(Pack, Use0), 0u);
  EXPECT_EQ(sandboxir::UserAttorney::getOperandUseIdx(Pack, Use1_a), 1u);
  EXPECT_EQ(sandboxir::UserAttorney::getOperandUseIdx(Pack, Use1_b), 1u);
  EXPECT_EQ(sandboxir::UserAttorney::getOperandUseIdx(Pack, Use2), 2u);
  EXPECT_EQ(sandboxir::UserAttorney::getOperandUseIdx(SBRet, RetUse), 0u);

#ifndef NDEBUG
  // Some invalid uses:
  const Use &UseIns1_0 = Ins1->getOperandUse(0);
  const Use &UseIns1_1 = Ins1->getOperandUse(1);
  const Use &UseIns2_0 = Ins2->getOperandUse(0);
  const Use &UseIns2_1 = Ins2->getOperandUse(1);
  EXPECT_DEATH(sandboxir::UserAttorney::getOperandUseIdx(Pack, UseIns1_0),
               ".*not found.*");
  EXPECT_DEATH(sandboxir::UserAttorney::getOperandUseIdx(Pack, UseIns1_1),
               ".*not found.*");
  EXPECT_DEATH(sandboxir::UserAttorney::getOperandUseIdx(Pack, UseIns2_0),
               ".*not found.*");
  EXPECT_DEATH(sandboxir::UserAttorney::getOperandUseIdx(Pack, UseIns2_1),
               ".*not found.*");
  EXPECT_DEATH(sandboxir::UserAttorney::getOperandUseIdx(SBRet, UseIns2_1),
               ".*not found.*");
#endif
}

// This checks that the Use::getOperandNo() for Packs does not time out.
TEST_F(SandboxIRVecTest, PackOperandNo_Timeout) {
  parseIR(C, R"IR(
define void @foo(i8 %v0,i8 %v1,i8 %v2,i8 %v3,i8 %v4,i8 %v5,i8 %v6,i8 %v7,i8 %v8,i8 %v9,i8 %v10,i8 %v11,i8 %v12,i8 %v13,i8 %v14,i8 %v15,i8 %v16,i8 %v17,i8 %v18,i8 %v19,i8 %v20,i8 %v21,i8 %v22,i8 %v23,i8 %v24,i8 %v25,i8 %v26,i8 %v27,i8 %v28,i8 %v29,i8 %v30,i8 %v31,i8 %v32,i8 %v33,i8 %v34,i8 %v35,i8 %v36,i8 %v37,i8 %v38,i8 %v39,i8 %v40,i8 %v41,i8 %v42,i8 %v43,i8 %v44,i8 %v45,i8 %v46,i8 %v47,i8 %v48,i8 %v49,i8 %v50,i8 %v51,i8 %v52,i8 %v53,i8 %v54,i8 %v55,i8 %v56,i8 %v57,i8 %v58,i8 %v59,i8 %v60,i8 %v61,i8 %v62,i8 %v63,i8 %v64,i8 %v65,i8 %v66,i8 %v67,i8 %v68,i8 %v69,i8 %v70,i8 %v71,i8 %v72,i8 %v73,i8 %v74,i8 %v75,i8 %v76,i8 %v77,i8 %v78,i8 %v79,i8 %v80,i8 %v81,i8 %v82,i8 %v83,i8 %v84,i8 %v85,i8 %v86,i8 %v87,i8 %v88,i8 %v89,i8 %v90,i8 %v91,i8 %v92,i8 %v93,i8 %v94,i8 %v95,i8 %v96,i8 %v97,i8 %v98,i8 %v99,i8 %v100,i8 %v101,i8 %v102,i8 %v103,i8 %v104,i8 %v105,i8 %v106,i8 %v107,i8 %v108,i8 %v109,i8 %v110,i8 %v111,i8 %v112,i8 %v113,i8 %v114,i8 %v115,i8 %v116,i8 %v117,i8 %v118,i8 %v119,i8 %v120,i8 %v121,i8 %v122,i8 %v123,i8 %v124,i8 %v125,i8 %v126,i8 %v127) {
%Pack0 = insertelement <128 x i8> poison, i8 %v0, i64 0
%Pack1 = insertelement <128 x i8> %Pack0, i8 %v1, i64 1
%Pack2 = insertelement <128 x i8> %Pack1, i8 %v2, i64 2
%Pack3 = insertelement <128 x i8> %Pack2, i8 %v3, i64 3
%Pack4 = insertelement <128 x i8> %Pack3, i8 %v4, i64 4
%Pack5 = insertelement <128 x i8> %Pack4, i8 %v5, i64 5
%Pack6 = insertelement <128 x i8> %Pack5, i8 %v6, i64 6
%Pack7 = insertelement <128 x i8> %Pack6, i8 %v7, i64 7
%Pack8 = insertelement <128 x i8> %Pack7, i8 %v8, i64 8
%Pack9 = insertelement <128 x i8> %Pack8, i8 %v9, i64 9
%Pack10 = insertelement <128 x i8> %Pack9, i8 %v10, i64 10
%Pack11 = insertelement <128 x i8> %Pack10, i8 %v11, i64 11
%Pack12 = insertelement <128 x i8> %Pack11, i8 %v12, i64 12
%Pack13 = insertelement <128 x i8> %Pack12, i8 %v13, i64 13
%Pack14 = insertelement <128 x i8> %Pack13, i8 %v14, i64 14
%Pack15 = insertelement <128 x i8> %Pack14, i8 %v15, i64 15
%Pack16 = insertelement <128 x i8> %Pack15, i8 %v16, i64 16
%Pack17 = insertelement <128 x i8> %Pack16, i8 %v17, i64 17
%Pack18 = insertelement <128 x i8> %Pack17, i8 %v18, i64 18
%Pack19 = insertelement <128 x i8> %Pack18, i8 %v19, i64 19
%Pack20 = insertelement <128 x i8> %Pack19, i8 %v20, i64 20
%Pack21 = insertelement <128 x i8> %Pack20, i8 %v21, i64 21
%Pack22 = insertelement <128 x i8> %Pack21, i8 %v22, i64 22
%Pack23 = insertelement <128 x i8> %Pack22, i8 %v23, i64 23
%Pack24 = insertelement <128 x i8> %Pack23, i8 %v24, i64 24
%Pack25 = insertelement <128 x i8> %Pack24, i8 %v25, i64 25
%Pack26 = insertelement <128 x i8> %Pack25, i8 %v26, i64 26
%Pack27 = insertelement <128 x i8> %Pack26, i8 %v27, i64 27
%Pack28 = insertelement <128 x i8> %Pack27, i8 %v28, i64 28
%Pack29 = insertelement <128 x i8> %Pack28, i8 %v29, i64 29
%Pack30 = insertelement <128 x i8> %Pack29, i8 %v30, i64 30
%Pack31 = insertelement <128 x i8> %Pack30, i8 %v31, i64 31
%Pack32 = insertelement <128 x i8> %Pack31, i8 %v32, i64 32
%Pack33 = insertelement <128 x i8> %Pack32, i8 %v33, i64 33
%Pack34 = insertelement <128 x i8> %Pack33, i8 %v34, i64 34
%Pack35 = insertelement <128 x i8> %Pack34, i8 %v35, i64 35
%Pack36 = insertelement <128 x i8> %Pack35, i8 %v36, i64 36
%Pack37 = insertelement <128 x i8> %Pack36, i8 %v37, i64 37
%Pack38 = insertelement <128 x i8> %Pack37, i8 %v38, i64 38
%Pack39 = insertelement <128 x i8> %Pack38, i8 %v39, i64 39
%Pack40 = insertelement <128 x i8> %Pack39, i8 %v40, i64 40
%Pack41 = insertelement <128 x i8> %Pack40, i8 %v41, i64 41
%Pack42 = insertelement <128 x i8> %Pack41, i8 %v42, i64 42
%Pack43 = insertelement <128 x i8> %Pack42, i8 %v43, i64 43
%Pack44 = insertelement <128 x i8> %Pack43, i8 %v44, i64 44
%Pack45 = insertelement <128 x i8> %Pack44, i8 %v45, i64 45
%Pack46 = insertelement <128 x i8> %Pack45, i8 %v46, i64 46
%Pack47 = insertelement <128 x i8> %Pack46, i8 %v47, i64 47
%Pack48 = insertelement <128 x i8> %Pack47, i8 %v48, i64 48
%Pack49 = insertelement <128 x i8> %Pack48, i8 %v49, i64 49
%Pack50 = insertelement <128 x i8> %Pack49, i8 %v50, i64 50
%Pack51 = insertelement <128 x i8> %Pack50, i8 %v51, i64 51
%Pack52 = insertelement <128 x i8> %Pack51, i8 %v52, i64 52
%Pack53 = insertelement <128 x i8> %Pack52, i8 %v53, i64 53
%Pack54 = insertelement <128 x i8> %Pack53, i8 %v54, i64 54
%Pack55 = insertelement <128 x i8> %Pack54, i8 %v55, i64 55
%Pack56 = insertelement <128 x i8> %Pack55, i8 %v56, i64 56
%Pack57 = insertelement <128 x i8> %Pack56, i8 %v57, i64 57
%Pack58 = insertelement <128 x i8> %Pack57, i8 %v58, i64 58
%Pack59 = insertelement <128 x i8> %Pack58, i8 %v59, i64 59
%Pack60 = insertelement <128 x i8> %Pack59, i8 %v60, i64 60
%Pack61 = insertelement <128 x i8> %Pack60, i8 %v61, i64 61
%Pack62 = insertelement <128 x i8> %Pack61, i8 %v62, i64 62
%Pack63 = insertelement <128 x i8> %Pack62, i8 %v63, i64 63
%Pack64 = insertelement <128 x i8> %Pack63, i8 %v64, i64 64
%Pack65 = insertelement <128 x i8> %Pack64, i8 %v65, i64 65
%Pack66 = insertelement <128 x i8> %Pack65, i8 %v66, i64 66
%Pack67 = insertelement <128 x i8> %Pack66, i8 %v67, i64 67
%Pack68 = insertelement <128 x i8> %Pack67, i8 %v68, i64 68
%Pack69 = insertelement <128 x i8> %Pack68, i8 %v69, i64 69
%Pack70 = insertelement <128 x i8> %Pack69, i8 %v70, i64 70
%Pack71 = insertelement <128 x i8> %Pack70, i8 %v71, i64 71
%Pack72 = insertelement <128 x i8> %Pack71, i8 %v72, i64 72
%Pack73 = insertelement <128 x i8> %Pack72, i8 %v73, i64 73
%Pack74 = insertelement <128 x i8> %Pack73, i8 %v74, i64 74
%Pack75 = insertelement <128 x i8> %Pack74, i8 %v75, i64 75
%Pack76 = insertelement <128 x i8> %Pack75, i8 %v76, i64 76
%Pack77 = insertelement <128 x i8> %Pack76, i8 %v77, i64 77
%Pack78 = insertelement <128 x i8> %Pack77, i8 %v78, i64 78
%Pack79 = insertelement <128 x i8> %Pack78, i8 %v79, i64 79
%Pack80 = insertelement <128 x i8> %Pack79, i8 %v80, i64 80
%Pack81 = insertelement <128 x i8> %Pack80, i8 %v81, i64 81
%Pack82 = insertelement <128 x i8> %Pack81, i8 %v82, i64 82
%Pack83 = insertelement <128 x i8> %Pack82, i8 %v83, i64 83
%Pack84 = insertelement <128 x i8> %Pack83, i8 %v84, i64 84
%Pack85 = insertelement <128 x i8> %Pack84, i8 %v85, i64 85
%Pack86 = insertelement <128 x i8> %Pack85, i8 %v86, i64 86
%Pack87 = insertelement <128 x i8> %Pack86, i8 %v87, i64 87
%Pack88 = insertelement <128 x i8> %Pack87, i8 %v88, i64 88
%Pack89 = insertelement <128 x i8> %Pack88, i8 %v89, i64 89
%Pack90 = insertelement <128 x i8> %Pack89, i8 %v90, i64 90
%Pack91 = insertelement <128 x i8> %Pack90, i8 %v91, i64 91
%Pack92 = insertelement <128 x i8> %Pack91, i8 %v92, i64 92
%Pack93 = insertelement <128 x i8> %Pack92, i8 %v93, i64 93
%Pack94 = insertelement <128 x i8> %Pack93, i8 %v94, i64 94
%Pack95 = insertelement <128 x i8> %Pack94, i8 %v95, i64 95
%Pack96 = insertelement <128 x i8> %Pack95, i8 %v96, i64 96
%Pack97 = insertelement <128 x i8> %Pack96, i8 %v97, i64 97
%Pack98 = insertelement <128 x i8> %Pack97, i8 %v98, i64 98
%Pack99 = insertelement <128 x i8> %Pack98, i8 %v99, i64 99
%Pack100 = insertelement <128 x i8> %Pack99, i8 %v100, i64 100
%Pack101 = insertelement <128 x i8> %Pack100, i8 %v101, i64 101
%Pack102 = insertelement <128 x i8> %Pack101, i8 %v102, i64 102
%Pack103 = insertelement <128 x i8> %Pack102, i8 %v103, i64 103
%Pack104 = insertelement <128 x i8> %Pack103, i8 %v104, i64 104
%Pack105 = insertelement <128 x i8> %Pack104, i8 %v105, i64 105
%Pack106 = insertelement <128 x i8> %Pack105, i8 %v106, i64 106
%Pack107 = insertelement <128 x i8> %Pack106, i8 %v107, i64 107
%Pack108 = insertelement <128 x i8> %Pack107, i8 %v108, i64 108
%Pack109 = insertelement <128 x i8> %Pack108, i8 %v109, i64 109
%Pack110 = insertelement <128 x i8> %Pack109, i8 %v110, i64 110
%Pack111 = insertelement <128 x i8> %Pack110, i8 %v111, i64 111
%Pack112 = insertelement <128 x i8> %Pack111, i8 %v112, i64 112
%Pack113 = insertelement <128 x i8> %Pack112, i8 %v113, i64 113
%Pack114 = insertelement <128 x i8> %Pack113, i8 %v114, i64 114
%Pack115 = insertelement <128 x i8> %Pack114, i8 %v115, i64 115
%Pack116 = insertelement <128 x i8> %Pack115, i8 %v116, i64 116
%Pack117 = insertelement <128 x i8> %Pack116, i8 %v117, i64 117
%Pack118 = insertelement <128 x i8> %Pack117, i8 %v118, i64 118
%Pack119 = insertelement <128 x i8> %Pack118, i8 %v119, i64 119
%Pack120 = insertelement <128 x i8> %Pack119, i8 %v120, i64 120
%Pack121 = insertelement <128 x i8> %Pack120, i8 %v121, i64 121
%Pack122 = insertelement <128 x i8> %Pack121, i8 %v122, i64 122
%Pack123 = insertelement <128 x i8> %Pack122, i8 %v123, i64 123
%Pack124 = insertelement <128 x i8> %Pack123, i8 %v124, i64 124
%Pack125 = insertelement <128 x i8> %Pack124, i8 %v125, i64 125
%Pack126 = insertelement <128 x i8> %Pack125, i8 %v126, i64 126
%Pack127 = insertelement <128 x i8> %Pack126, i8 %v127, i64 127
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();
  auto *Pack = cast<sandboxir::PackInst>(&*It++);
  auto Use127 = Pack->getOperandUse(127);
  EXPECT_EQ(Use127.getOperandNo(), 127u);
  auto Use0 = Pack->getOperandUse(0);
  EXPECT_EQ(Use0.getOperandNo(), 0u);
}

TEST_F(SandboxIRVecTest, PackOperandNo_Check2) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, <2 x i32> %v1, i32 %v2) {
  %Pack0 = insertelement <4 x i32> poison, i32 %v0, i64 0

  %Extr1.0 = extractelement <2 x i32> %v1, i32 0
  %Pack1 = insertelement <4 x i32> %Pack0, i32 %Extr1.0, i64 1
  %Extr1.1 = extractelement <2 x i32> %v1, i32 1
  %Pack2 = insertelement <4 x i32> %Pack1, i32 %Extr1.1, i64 2

  %Pack3 = insertelement <4 x i32> %Pack2, i32 %v2, i64 3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto *Op0 = SBF.getArg(0);
  auto *Op1 = SBF.getArg(1);
  auto *Op2 = SBF.getArg(2);
  auto It = SBBB.begin();
  auto *Pack = cast<sandboxir::PackInst>(&*It++);
  auto Use0 = Pack->getOperandUse(0);
  EXPECT_EQ(Use0.get(), Op0);
  EXPECT_EQ(Use0.getOperandNo(), 0u);

  auto Use1 = Pack->getOperandUse(1);
  EXPECT_EQ(Use1.get(), Op1);
  EXPECT_EQ(Use1.getOperandNo(), 1u);

  auto Use2 = Pack->getOperandUse(2);
  EXPECT_EQ(Use2.get(), Op2);
  EXPECT_EQ(Use2.getOperandNo(), 2u);
}

TEST_F(SandboxIRVecTest, PackDetection_Canonical1) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, i32 %v2, <2 x i32> %vec) {
  ; PackA
  %PackA0 = insertelement <2 x i32> poison, i32 %v0, i64 0
  %PackA1 = insertelement <2 x i32> %PackA0, i32 %v1, i64 1

  ; PackB
  %PackB0 = insertelement <2 x i32> poison, i32 %v0, i64 0
  %PackB1 = insertelement <2 x i32> %PackB0, i32 %v1, i64 1

  ; An extract with a constant index is an Unpack.
  %Unpack =  extractelement <2 x i32> %vec, i32 0

  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  auto *PackA = cast<sandboxir::PackInst>(&*It++);
  EXPECT_EQ(PackA->getOpcode(), sandboxir::Instruction::Opcode::Pack);
  auto *PackB = cast<sandboxir::PackInst>(&*It++);
  EXPECT_EQ(PackB->getOpcode(), sandboxir::Instruction::Opcode::Pack);
  auto *Unpack = cast<sandboxir::UnpackInst>(&*It++);
  EXPECT_EQ(Unpack->getOpcode(), sandboxir::Instruction::Opcode::Unpack);
  auto *Ret = cast<sandboxir::RetInst>(&*It++);
  EXPECT_EQ(Ret->getOpcode(), sandboxir::Instruction::Opcode::Ret);
}

TEST_F(SandboxIRVecTest, PackDetection_Canonical2) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, i32 %v2, <2 x i32> %vec) {
  ; Yet another pack
  %PackIns0 = insertelement <3 x i32> poison, i32 %v0, i32 0
  %PackExtr0 = extractelement <2 x i32> %vec, i32 0
  %PackIns1 = insertelement <3 x i32> %PackIns0, i32 %PackExtr0, i32 1
  %PackExtr1 = extractelement <2 x i32> %vec, i32 1
  %PackIns2 = insertelement <3 x i32> %PackIns1, i32 %PackExtr1, i32 2

  ; PackB
  %PackB0 = insertelement <2 x i32> poison, i32 %v0, i64 0
  %PackB1 = insertelement <2 x i32> %PackB0, i32 %v1, i64 1

  ; An extract with a constant index is an Unpack.
  %Unpack =  extractelement <2 x i32> %vec, i32 0

  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  auto *PackA = cast<sandboxir::PackInst>(&*It++);
  EXPECT_EQ(PackA->getOpcode(), sandboxir::Instruction::Opcode::Pack);
  auto *PackB = cast<sandboxir::PackInst>(&*It++);
  EXPECT_EQ(PackB->getOpcode(), sandboxir::Instruction::Opcode::Pack);
  auto *Unpack = cast<sandboxir::UnpackInst>(&*It++);
  EXPECT_EQ(Unpack->getOpcode(), sandboxir::Instruction::Opcode::Unpack);
  auto *Ret = cast<sandboxir::RetInst>(&*It++);
  EXPECT_EQ(Ret->getOpcode(), sandboxir::Instruction::Opcode::Ret);
}

TEST_F(SandboxIRVecTest, PackDetection_Canonical_FullExtracts) {
  parseIR(C, R"IR(
define void @foo(<2 x i32> %vec) {
  %PackExtr0 = extractelement <2 x i32> %vec, i32 0
  %PackIns0 = insertelement <2 x i32> poison, i32 %PackExtr0, i32 0
  %PackExtr1 = extractelement <2 x i32> %vec, i32 1
  %PackIns1 = insertelement <2 x i32> %PackIns0, i32 %PackExtr1, i32 1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<sandboxir::PackInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::RetInst>(&*It++));
}

TEST_F(SandboxIRVecTest, PackDetection_Canonical_ExtractGroups) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, <2 x i32> %vecA, <2 x i32> %vecB) {
  %PackExtrA0 = extractelement <2 x i32> %vecA, i32 0
  %PackIns0 = insertelement <5 x i32> poison, i32 %PackExtrA0, i32 0
  %PackExtrA1 = extractelement <2 x i32> %vecA, i32 1
  %PackIns1 = insertelement <5 x i32> %PackIns0, i32 %PackExtrA1, i32 1

  %PackIns2 = insertelement <5 x i32> %PackIns1, i32 %v0, i32 2

  %PackExtrB0 = extractelement <2 x i32> %vecB, i32 0
  %PackIns3 = insertelement <5 x i32> %PackIns2, i32 %PackExtrB0, i32 3
  %PackExtrB1 = extractelement <2 x i32> %vecB, i32 1
  %PackIns4 = insertelement <5 x i32> %PackIns3, i32 %PackExtrB1, i32 4

  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<sandboxir::PackInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::RetInst>(&*It++));
}

TEST_F(SandboxIRVecTest, PackDetection_SimpleIRPattern_WithExtracts) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, <2 x i32> %v1, i32 %v2) {
  %Pack0 = insertelement <4 x i32> poison, i32 %v0, i64 0

  %Extr1.0 = extractelement <2 x i32> %v1, i32 0
  %Pack1 = insertelement <4 x i32> %Pack0, i32 %Extr1.0, i64 1
  %Extr1.1 = extractelement <2 x i32> %v1, i32 1
  %Pack2 = insertelement <4 x i32> %Pack1, i32 %Extr1.1, i64 2

  %Pack3 = insertelement <4 x i32> %Pack2, i32 %v2, i64 3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  Instruction *Ret = BB->getTerminator();

  auto &SBBB = *Ctx.createBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  auto *BotInsert = cast<InsertElementInst>(BB->getTerminator()->getPrevNode());
  auto *Pack = cast<sandboxir::PackInst>(Ctx.getValue(BotInsert));
#ifndef NDEBUG
  Pack->verify();
#endif
  // Make sure the extracts are part of the Pack, not separate unpacks.
  auto *SBV0 = Ctx.getValue(F.getArg(0));
  auto *SBV1 = Ctx.getValue(F.getArg(1));
  auto *SBV2 = Ctx.getValue(F.getArg(2));
  EXPECT_EQ(Pack->getOperand(0), SBV0);
  EXPECT_EQ(Pack->getOperand(1), SBV1);
  EXPECT_EQ(Pack->getOperand(2), SBV2);
  EXPECT_EQ(Pack->getNumOperands(), 3u);

  for (sandboxir::Value &SBV : SBBB)
    EXPECT_TRUE(!isa<sandboxir::UnpackInst>(&SBV));

  // Check eraseFromParent().
  Pack->eraseFromParent();
  ASSERT_EQ(BB->size(), 1u);
  ASSERT_EQ(&*BB->rbegin(), Ret);
}

TEST_F(SandboxIRVecTest, PackNotCanonical_BadExtractPosition) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, i32 %v2, <2 x i32> %vec) {
  ; The extracts should be positioned right before their user Insert.
  %PackIns0 = insertelement <3 x i32> poison, i32 %v0, i32 0
  %PackExtr0 = extractelement <2 x i32> %vec, i32 0
  %PackExtr1 = extractelement <2 x i32> %vec, i32 1
  %PackIns1 = insertelement <3 x i32> %PackIns0, i32 %PackExtr0, i32 1
  %PackIns2 = insertelement <3 x i32> %PackIns1, i32 %PackExtr1, i32 2
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::UnpackInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::UnpackInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::RetInst>(&*It++));
}

TEST_F(SandboxIRVecTest, PackNotCanonical_MissingExtract1) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, <2 x i32> %vec) {
  ; The extracts should be complete
  %PackIns0 = insertelement <3 x i32> poison, i32 %v0, i32 0
  %PackIns1 = insertelement <3 x i32> %PackIns0, i32 %v1, i32 1
  %PackExtr1 = extractelement <2 x i32> %vec, i32 1
  %PackIns2 = insertelement <3 x i32> %PackIns1, i32 %PackExtr1, i32 2
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::UnpackInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::RetInst>(&*It++));
}

TEST_F(SandboxIRVecTest, PackNotCanonical_MissingExtract2) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, <2 x i32> %vec) {
  ; The extracts should be complete
  %PackIns0 = insertelement <3 x i32> poison, i32 %v0, i32 0
  %PackExtr1 = extractelement <2 x i32> %vec, i32 1
  %PackIns1 = insertelement <3 x i32> %PackIns0, i32 %PackExtr1, i32 1
  %PackIns2 = insertelement <3 x i32> %PackIns1, i32 %v1, i32 2
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::UnpackInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::RetInst>(&*It++));
}

TEST_F(SandboxIRVecTest, PackNotCanonical_ExtractIndicesOutOfOrder) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, i32 %v2, <2 x i32> %vec) {
  ; The extracts indices should increment top-down
  %PackIns0 = insertelement <3 x i32> poison, i32 %v0, i32 0
  %PackExtr0 = extractelement <2 x i32> %vec, i32 1
  %PackIns1 = insertelement <3 x i32> %PackIns0, i32 %PackExtr0, i32 1
  %PackExtr1 = extractelement <2 x i32> %vec, i32 0
  %PackIns2 = insertelement <3 x i32> %PackIns1, i32 %PackExtr1, i32 2
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::UnpackInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::UnpackInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::RetInst>(&*It++));
}

TEST_F(SandboxIRVecTest, PackNotCanonical_ExtractsFromDifferentVectors) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, i32 %v2, <2 x i32> %vec, <2 x i32> %vec2) {
  ; The extracts should read from the same operand vector
  %PackIns0 = insertelement <3 x i32> poison, i32 %v0, i32 0
  %PackExtr0 = extractelement <2 x i32> %vec, i32 0
  %PackIns1 = insertelement <3 x i32> %PackIns0, i32 %PackExtr0, i32 1
  %PackExtr1 = extractelement <2 x i32> %vec2, i32 1
  %PackIns2 = insertelement <3 x i32> %PackIns1, i32 %PackExtr1, i32 2
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::UnpackInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::UnpackInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::RetInst>(&*It++));
}

TEST_F(SandboxIRVecTest, PackNotCanonical_ExtractVectorLarger) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, i32 %v2, <3 x i32> %vec) {
  ; The extracts vector should be <2 x i32>
  %PackIns0 = insertelement <3 x i32> poison, i32 %v0, i32 0
  %PackExtr0 = extractelement <3 x i32> %vec, i32 0
  %PackIns1 = insertelement <3 x i32> %PackIns0, i32 %PackExtr0, i32 1
  %PackExtr1 = extractelement <3 x i32> %vec, i32 1
  %PackIns2 = insertelement <3 x i32> %PackIns1, i32 %PackExtr1, i32 2
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::UnpackInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::UnpackInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::RetInst>(&*It++));
}

TEST_F(SandboxIRVecTest, PackNotCanonical_ExtractVectorSmaller) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, <2 x i32> %vec) {
  ; The extracts vector should be <2 x i32>
  %PackIns0 = insertelement <4 x i32> poison, i32 %v0, i32 0
  %PackExtr0 = extractelement <2 x i32> %vec, i32 0
  %PackIns1 = insertelement <4 x i32> %PackIns0, i32 %PackExtr0, i32 1
  %PackExtr1 = extractelement <2 x i32> %vec, i32 0
  %PackIns2 = insertelement <4 x i32> %PackIns1, i32 %PackExtr1, i32 2
  %PackExtr2 = extractelement <2 x i32> %vec, i32 1
  %PackIns3 = insertelement <4 x i32> %PackIns2, i32 %PackExtr2, i32 3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::UnpackInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::UnpackInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::UnpackInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::RetInst>(&*It++));
}

TEST_F(SandboxIRVecTest, PackNotCanonical_InsertIndicesOutOffOrder) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, i32 %v2, i32 %v3) {
  ; The insert indexes should increase top-down
  %Pack0 = insertelement <4 x i32> poison, i32 %v0, i64 0
  %Pack2 = insertelement <4 x i32> %Pack0, i32 %v2, i64 2
  %Pack1 = insertelement <4 x i32> %Pack2, i32 %v1, i64 1
  %Pack3 = insertelement <4 x i32> %Pack1, i32 %v3, i64 3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::RetInst>(&*It++));
}

TEST_F(SandboxIRVecTest, PackNotCanonical_BadInsertPattern) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, i32 %v2, i32 %v3) {
  ; Bad insert operands.
  %Pack0 = insertelement <4 x i32> poison, i32 %v0, i64 0
  %Pack1 = insertelement <4 x i32> %Pack0, i32 %v2, i64 1
  %Pack2 = insertelement <4 x i32> %Pack0, i32 %v1, i64 2
  %Pack3 = insertelement <4 x i32> %Pack2, i32 %v3, i64 3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::RetInst>(&*It++));
}

TEST_F(SandboxIRVecTest, PackNotCanonical_NonPoisonOperand) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, i32 %v2, i32 %v3, <4 x i32> %NotPoison) {
  ; Non-poison operand.
  %Pack0 = insertelement <4 x i32> %NotPoison, i32 %v0, i64 0
  %Pack1 = insertelement <4 x i32> %Pack0, i32 %v2, i64 1
  %Pack2 = insertelement <4 x i32> %Pack1, i32 %v1, i64 2
  %Pack3 = insertelement <4 x i32> %Pack2, i32 %v3, i64 3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::RetInst>(&*It++));
}

TEST_F(SandboxIRVecTest, PackNotCanonical_BrokenChain) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, i32 %v2, i32 %v3) {
  ; The insert-chain is interrupted by a poison value.
  %Pack0 = insertelement <4 x i32> poison, i32 %v0, i64 0
  %Pack1 = insertelement <4 x i32> %Pack0, i32 %v2, i64 1
  %Pack2 = insertelement <4 x i32> poison, i32 %v1, i64 2
  %Pack3 = insertelement <4 x i32> %Pack2, i32 %v3, i64 3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::RetInst>(&*It++));
}

TEST_F(SandboxIRVecTest, PackNotCanonical_BadChain) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, i32 %v2, i32 %v3) {
  %Pack0 = insertelement <4 x i32> poison, i32 %v0, i64 0
  %Pack1 = insertelement <4 x i32> %Pack0, i32 %v1, i64 1
  %Pack2 = insertelement <4 x i32> %Pack1, i32 %v2, i64 2
  %Pack3 = insertelement <4 x i32> %Pack1, i32 %v2, i64 3 ; Uses Pack1 instead of Pack2
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto &SBBB = *Ctx.createBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  EXPECT_TRUE(none_of(SBBB, [](sandboxir::Value &N) {
    return isa<sandboxir::PackInst>(&N);
  }));
}

TEST_F(SandboxIRVecTest, PackNotCanonical_FoldedConsts) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, i32 %v2, i32 %v3) {
  %Pack0 = insertelement <4 x i32> <i32 poison, i32 0, i32 poison, i32 poison>, i32 %v0, i64 0
  %Pack2 = insertelement <4 x i32> %Pack0, i32 %v2, i64 2
  %Pack3 = insertelement <4 x i32> %Pack2, i32 %v3, i64 3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::InsertElementInst>(&*It++));
  EXPECT_TRUE(isa<sandboxir::RetInst>(&*It++));
}

// This used to crash.
TEST_F(SandboxIRVecTest, PackNotCanonical_OneInsertAtStartOfBB) {
  parseIR(C, R"IR(
define void @foo(<2 x double> %vec, double %val) {
  %Bad = insertelement <2 x double> %vec, double %val, i32 1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();
  EXPECT_TRUE(!isa<sandboxir::PackInst>(&*It++));
}

TEST_F(SandboxIRVecTest, PackNotCanonical_MissingTopExtract) {
  parseIR(C, R"IR(
define void @foo(i32 %PackExtr0, <2 x i32> %vec) {
  %PackIns0 = insertelement <2 x i32> poison, i32 %PackExtr0, i32 0
  %PackExtr1 = extractelement <2 x i32> %vec, i32 1
  %PackIns1 = insertelement <2 x i32> %PackIns0, i32 %PackExtr1, i32 1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();
  EXPECT_TRUE(!isa<sandboxir::PackInst>(&*It++));
  EXPECT_TRUE(!isa<sandboxir::PackInst>(&*It++));
  EXPECT_TRUE(!isa<sandboxir::PackInst>(&*It++));
}

TEST_F(SandboxIRVecTest, PackNotCanonical_MissingOperands) {
  parseIR(C, R"IR(
define void @foo(i8 %arg0) {
  %ins0 = insertelement <4 x i8> poison, i8 %arg0, i32 3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto *SBF = Ctx.createFunction(&F);
  (void)SBF;
  auto *SBBB = Ctx.getBasicBlock(BB);
  auto It = SBBB->begin();
  auto *NotPack = &*It++;
  EXPECT_FALSE(isa<sandboxir::PackInst>(NotPack));
}

TEST_F(SandboxIRVecTest, PackNotCanonical_PoisonOperand) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep = getelementptr inbounds i8, ptr %ptr, i64 42
  %Pack = insertelement <2 x ptr> poison, ptr %gep, i64 1
  store <2 x ptr> %Pack, ptr %ptr
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  auto *BB = Ctx.createBasicBlock(&*F.begin());
#ifndef NDEBUG
  BB->verify();
#endif
  auto It = BB->begin();
  auto *Gep = &*It++;
  (void)Gep;
  auto *NotPack = &*It++;
  EXPECT_FALSE(isa<sandboxir::PackInst>(NotPack));
  auto *St = cast<sandboxir::StoreInst>(&*It++);
  EXPECT_EQ(St->getOperand(0), NotPack);
}

TEST_F(SandboxIRVecTest, Opcodes) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1, i32 %v2, <2 x i32> %vec, <2 x i32> %vec2) {
  ; Yet another pack
  %PackIns0 = insertelement <3 x i32> poison, i32 %v0, i32 0
  %PackExtr0 = extractelement <2 x i32> %vec, i32 0
  %PackIns1 = insertelement <3 x i32> %PackIns0, i32 %PackExtr0, i32 1
  %PackExtr1 = extractelement <2 x i32> %vec, i32 1
  %PackIns2 = insertelement <3 x i32> %PackIns1, i32 %PackExtr1, i32 2

  ; This sequence of inserts is a single SBPack
  %Pack0 = insertelement <2 x i32> poison, i32 %v0, i64 0
  %Pack1 = insertelement <2 x i32> %Pack0, i32 %v1, i64 1

  ; A roque insert is Opaque.
  %InsertOpq1 =  insertelement <2 x i32> poison, i32 %v0, i64 0

  ; An insert with a non-constant index is Opaque
  %InsertOpq2 =  insertelement <2 x i32> poison, i32 %v0, i32 %v2

  ; An extract with a constant index is an Unpack.
  %Unpack =  extractelement <2 x i32> %vec, i32 0
  ; An extract with a non-constant index is Extract
  %ExtractOpq1 =  extractelement <2 x i32> %vec, i32 %v2

  ; A SB-IR-style shuffle.
  %Shuffle = shufflevector <2 x i32> %vec, <2 x i32> poison, <2 x i32> <i32 1, i32 0>
  ; A blend is ShuffleVec for now.
  %ShuffleBlend = shufflevector <2 x i32> %vec, <2 x i32> %vec2, <2 x i32> <i32 2, i32 0>

  ; A call is Opaque for now.
  call void @foo(i32 %v0, i32 %v1, i32 %v2, <2 x i32> %vec, <2 x i32> %vec2)

  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  auto *Pack1 = cast<sandboxir::PackInst>(&*It++);
  EXPECT_EQ(Pack1->getOpcode(), sandboxir::Instruction::Opcode::Pack);

  auto *Pack2 = cast<sandboxir::PackInst>(&*It++);
  EXPECT_EQ(Pack2->getOpcode(), sandboxir::Instruction::Opcode::Pack);

  auto *InsertOpq1 = &*It++;
  EXPECT_EQ(InsertOpq1->getOpcode(), sandboxir::Instruction::Opcode::Insert);
  auto *InsertOpq2 = &*It++;
  EXPECT_EQ(InsertOpq2->getOpcode(), sandboxir::Instruction::Opcode::Insert);

  auto *Unpack = &*It++;
  EXPECT_EQ(Unpack->getOpcode(), sandboxir::Instruction::Opcode::Unpack);
  auto *ExtractOpq1 = &*It++;
  EXPECT_EQ(ExtractOpq1->getOpcode(),
            sandboxir::Instruction::Opcode::Extract);

  auto *Shuffle = &*It++;
  EXPECT_EQ(Shuffle->getOpcode(), sandboxir::Instruction::Opcode::Shuffle);
  auto *ShuffleBlend = &*It++;
  EXPECT_EQ(ShuffleBlend->getOpcode(),
            sandboxir::Instruction::Opcode::ShuffleVec);

  auto *Call = &*It++;
  EXPECT_EQ(Call->getOpcode(), sandboxir::Instruction::Opcode::Call);

  auto *Ret = &*It++;
  EXPECT_EQ(Ret->getOpcode(), sandboxir::Instruction::Opcode::Ret);
}

TEST_F(SandboxIRVecTest, PackOperands) {
  parseIR(C, R"IR(
define <4 x float> @foo(float %arg0, <2 x float> %arg1, float %arg2, <2 x float> %arg3, <4 x float> %arg4) {
  %ins0 = insertelement <4 x float> poison, float %arg0, i32 0
  %extr0 = extractelement <2 x float> %arg1, i32 0
  %ins1 = insertelement <4 x float> %ins0, float %extr0, i32 1
  %extr1 = extractelement <2 x float> %arg1, i32 1
  %ins2 = insertelement <4 x float> %ins1, float %extr1, i32 2
  %ins3 = insertelement <4 x float> %ins2, float %arg2, i32 3
  ret <4 x float> %ins3
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto *SBF = Ctx.createFunction(&F);
  auto *SBBB = Ctx.getBasicBlock(BB);
  auto It = SBBB->begin();
  auto *Pack = cast<sandboxir::PackInst>(&*It++);
  auto *Ret = &*It++;

  auto *Arg0 = SBF->getArg(0);
  auto *Arg1 = SBF->getArg(1);
  auto *Arg2 = SBF->getArg(2);
  auto *Arg3 = SBF->getArg(3);
  auto *Arg4 = SBF->getArg(4);

  // Check getOperand()
  SmallVector<sandboxir::Value *> ExpectedOperands{Arg0, Arg1, Arg2};
  unsigned Cnt = 0;
  for (auto [OpIdx, Op] : enumerate(Pack->operands())) {
    EXPECT_EQ(Pack->getOperand(OpIdx), Op);
    EXPECT_EQ(ExpectedOperands[OpIdx], Op);
    ++Cnt;
  }
  EXPECT_EQ(Cnt, ExpectedOperands.size());

  // Check RAUW for one of Pack operands
  Arg1->replaceAllUsesWith(Arg3);
  {
    SmallVector<sandboxir::Value *> ExpectedOperands{Arg0, Arg3, Arg2};
    for (auto [OpIdx, Op] : enumerate(Pack->operands())) {
      EXPECT_EQ(Pack->getOperand(OpIdx), Op);
      EXPECT_EQ(ExpectedOperands[OpIdx], Op);
    }
  }

  // Check Pack RUOW
  Pack->replaceUsesOfWith(Arg1, Arg3);
  EXPECT_EQ(Pack->getOperand(1), Arg3);

  Pack->replaceUsesWithIf(Arg4, [](sandboxir::Use Use) { return true; });
  EXPECT_EQ(Ret->getOperand(0), Arg4);
}

TEST_F(SandboxIRVecTest, PackInstructionCreate) {
  parseIR(C, R"IR(
define void @foo(i8 %val1, i8 %val2) {
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  auto *SBF = Ctx.createFunction(&F);
  auto *BB = Ctx.getBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *Ret = cast<sandboxir::RetInst>(&*It++);
  auto *ArgVal1 = SBF->getArg(0);
  auto *ArgVal2 = SBF->getArg(1);
  DmpVector<sandboxir::Value *> Ops{ArgVal1, ArgVal2};
  auto *Pack1 = cast<sandboxir::PackInst>(
      sandboxir::PackInst::create(Ops, BB, Ctx));
  EXPECT_EQ(&*BB->rbegin(), Pack1);

  auto *Pack2 = cast<sandboxir::PackInst>(
      sandboxir::PackInst::create(Ops, /*InsertBefore=*/Ret, Ctx));
  EXPECT_EQ(Ret->getPrevNode(), Pack2);

  auto WhereIt = BB->end();
  auto *Pack3 = cast<sandboxir::PackInst>(
      sandboxir::PackInst::create(Ops, WhereIt, BB, Ctx));
  EXPECT_EQ(&*BB->rbegin(), Pack3);
}

// When packing constant vectors the constant elements may finally show up as
// individual constants in the pack instruction. Check that the SBConstants
// for these constans have been created, otherwise getOperand() returns null.
TEST_F(SandboxIRVecTest, PackConstantVector) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %val1, i8 %val2) {
  store <2 x i8> <i8 0, i8 1>, ptr %ptr
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  auto *SBF = Ctx.createFunction(&F);
  auto *BB = Ctx.getBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *St = cast<sandboxir::StoreInst>(&*It++);

  auto *SBConstVec = cast<sandboxir::Constant>(St->getValueOperand());
  auto *ArgVal1 = SBF->getArg(1);
  auto *ArgVal2 = SBF->getArg(2);
  DmpVector<sandboxir::Value *> Ops{ArgVal1, ArgVal2, SBConstVec};
  auto WhereIt = sandboxir::VecUtils::getInsertPointAfter(Ops, BB);
  auto *Pack = cast<sandboxir::PackInst>(
      sandboxir::PackInst::create(Ops, WhereIt, BB, Ctx));

  // We expect non-null operands.
  EXPECT_TRUE(Pack->getOperand(0) != nullptr);
  EXPECT_TRUE(Pack->getOperand(1) != nullptr);
  EXPECT_TRUE(Pack->getOperand(2) != nullptr);
  EXPECT_TRUE(Pack->getOperand(3) != nullptr);
}

TEST_F(SandboxIRVecTest, PackExternalFacingOperands) {
  parseIR(C, R"IR(
define void @foo(float %arg0, <2 x float> %arg1, float %arg2) {
  %ins0 = insertelement <4 x float> poison, float %arg0, i32 0
  %extr0 = extractelement <2 x float> %arg1, i32 0
  %ins1 = insertelement <4 x float> %ins0, float %extr0, i32 1
  %extr1 = extractelement <2 x float> %arg1, i32 1
  %ins2 = insertelement <4 x float> %ins1, float %extr1, i32 2
  %ins3 = insertelement <4 x float> %ins2, float %arg2, i32 3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto *SBF = Ctx.createFunction(&F);
  (void)SBF;
  auto *SBBB = Ctx.getBasicBlock(BB);
  auto It = SBBB->begin();
  auto *Pack = cast<sandboxir::PackInst>(&*It++);

  auto BBIt = BB->begin();
  auto *Ins0 = &*BBIt++;
  auto *Extr0 = &*BBIt++;
  ++BBIt; // Ins1
  auto *Extr1 = &*BBIt++;
  ++BBIt; // Ins2
  auto *Ins3 = &*BBIt++;

  auto ExtInstrs =
      sandboxir::PackInstAttorney::getLLVMInstrsWithExternalOperands(
          Pack);
  EXPECT_EQ(ExtInstrs, SmallVector<Instruction *>({Ins3, Extr1, Extr0, Ins0}));
}

TEST_F(SandboxIRVecTest, PackGetIRInstrs_InProgramOrder) {
  parseIR(C, R"IR(
define void @foo(float %arg0, <2 x float> %arg1) {
  %ins0 = insertelement <3 x float> poison, float %arg0, i32 0
  %extr0 = extractelement <2 x float> %arg1, i32 0
  %ins1 = insertelement <3 x float> %ins0, float %extr0, i32 1
  %extr1 = extractelement <2 x float> %arg1, i32 1
  %ins2 = insertelement <3 x float> %ins1, float %extr1, i32 2
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto *SBF = Ctx.createFunction(&F);
  (void)SBF;
  auto *SBBB = Ctx.getBasicBlock(BB);
  auto It = SBBB->begin();
  auto *Pack = cast<sandboxir::PackInst>(&*It++);
  auto IRInstrs = sandboxir::PackInstAttorney::getLLVMInstrs(Pack);
  // Expect program order
  for (auto Idx : seq<unsigned>(1, IRInstrs.size()))
    EXPECT_TRUE(IRInstrs[Idx - 1]->comesBefore(IRInstrs[Idx]));
}

TEST_F(SandboxIRVecTest, PrevNode_Pack) {
  parseIR(C, R"IR(
define void @foo(i8 %v0, i8 %v1) {
  %ins0 = insertelement <2 x i8> poison, i8 %v0, i32 0
  %ins1 = insertelement <2 x i8> %ins0, i8 %v1, i32 1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);
  BasicBlock *BB = &*F.begin();
  sandboxir::BasicBlock &SBBB = *Ctx.createBasicBlock(BB);
  auto It = SBBB.begin();
  auto *Pack = cast<sandboxir::PackInst>(&*It++);
  EXPECT_EQ(Pack->getPrevNode(), nullptr);
}

// Check getPrevNode() and getNextNode()
TEST_F(SandboxIRVecTest, MultiIRPrevNextNode) {
  parseIR(C, R"IR(
define void @foo(i8 %v0, i8 %v1, i8 %v2, i8 %v3) {
  %ins0 = insertelement <2 x i8> poison, i8 %v0, i32 0
  %ins1 = insertelement <2 x i8> %ins0, i8 %v1, i32 1
  %ins10 = insertelement <2 x i8> poison, i8 %v2, i32 0
  %ins11 = insertelement <2 x i8> %ins10, i8 %v3, i32 1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto RawIt = BB->begin();

  auto *Ins0 = &*RawIt++;
  auto *Ins1 = &*RawIt++;
  auto *Ins10 = &*RawIt++;
  auto *Ins11 = &*RawIt++;
  auto *Ret = &*RawIt++;
  (void)Ret;

  auto *SBF = Ctx.createFunction(&F);
  (void)SBF;
  auto &SBBB = *Ctx.getBasicBlock(BB);
  auto It = SBBB.begin();
  auto *Pack0 = cast<sandboxir::PackInst>(&*It++);
  auto *Pack1 = cast<sandboxir::PackInst>(&*It++);
  auto *SBRet = &*It++;

#ifndef NDEBUG
  SBBB.verify();
#endif

  EXPECT_EQ(Pack0, Ctx.getValue(Ins0));
  EXPECT_EQ(Pack0, Ctx.getValue(Ins1));
  EXPECT_EQ(Pack1, Ctx.getValue(Ins10));
  EXPECT_EQ(Pack1, Ctx.getValue(Ins11));

  EXPECT_EQ(SBRet->getPrevNode(), Pack1);
  EXPECT_EQ(Pack1->getPrevNode(), Pack0);
  EXPECT_EQ(Pack0->getPrevNode(), nullptr);

  EXPECT_EQ(Pack0->getNextNode(), Pack1);
  EXPECT_EQ(Pack1->getNextNode(), SBRet);
  EXPECT_EQ(SBRet->getNextNode(), nullptr);
}

TEST_F(SandboxIRVecTest, InsertBefore) {
  parseIR(C, R"IR(
define void @foo(i8 %v0, i8 %v1, i8 %v2, i8 %v3) {
  %ins0 = insertelement <2 x i8> poison, i8 %v0, i32 0
  %ins1 = insertelement <2 x i8> %ins0, i8 %v1, i32 1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto RawIt = BB->begin();
  auto *Ins0 = &*RawIt++;
  auto *Ins1 = &*RawIt++;
  auto *Ret = &*RawIt++;

  auto *SBF = Ctx.createFunction(&F);
  auto &SBBB = *Ctx.getBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  auto It = SBBB.begin();
  auto *Pack = &*It++;
  auto *SBRet = &*It++;

  DmpVector<sandboxir::Value *> SBPackInstructions(
      {SBF->getArg(2), SBF->getArg(3)});
  auto WhereIt =
      sandboxir::VecUtils::getInsertPointAfter(SBPackInstructions, &SBBB);
  auto *NewPack =
      cast<sandboxir::PackInst>(sandboxir::PackInst::create(
          SBPackInstructions, WhereIt, &SBBB, Ctx));
  NewPack->removeFromParent();

  // Insert multi-IR instruction before multi-IR
  NewPack->insertBefore(Pack);
  It = SBBB.begin();
  EXPECT_EQ(&*It++, NewPack);
  EXPECT_EQ(&*It++, Pack);
  EXPECT_EQ(&*It++, SBRet);

  RawIt = BB->begin();
  RawIt++;
  RawIt++;
  EXPECT_EQ(&*RawIt++, Ins0);
  EXPECT_EQ(&*RawIt++, Ins1);
  EXPECT_EQ(&*RawIt++, Ret);
}

TEST_F(SandboxIRVecTest, InsertAfter) {
  parseIR(C, R"IR(
define void @foo(i8 %v0, i8 %v1, i8 %v2, i8 %v3) {
  %ins0 = insertelement <2 x i8> poison, i8 %v0, i32 0
  %ins1 = insertelement <2 x i8> %ins0, i8 %v1, i32 1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto RawIt = BB->begin();
  auto *Ins0 = &*RawIt++;
  auto *Ins1 = &*RawIt++;
  auto *Ret = &*RawIt++;

  auto *SBF = Ctx.createFunction(&F);
  auto &SBBB = *Ctx.getBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  auto It = SBBB.begin();
  auto *Pack = &*It++;
  auto *SBRet = &*It++;

  DmpVector<sandboxir::Value *> SBPackInstructions(
      {SBF->getArg(2), SBF->getArg(3)});
  auto WhereIt =
      sandboxir::VecUtils::getInsertPointAfter(SBPackInstructions, &SBBB);
  auto *NewPack =
      cast<sandboxir::PackInst>(sandboxir::PackInst::create(
          SBPackInstructions, WhereIt, &SBBB, Ctx));
  NewPack->removeFromParent();

  // Insert multi-IR instruction after multi-IR
  NewPack->insertAfter(Pack);
  It = SBBB.begin();
  EXPECT_EQ(&*It++, Pack);
  EXPECT_EQ(&*It++, NewPack);
  EXPECT_EQ(&*It++, SBRet);

  RawIt = BB->begin();
  EXPECT_EQ(&*RawIt++, Ins0);
  EXPECT_EQ(&*RawIt++, Ins1);
  RawIt++;
  RawIt++;
  EXPECT_EQ(&*RawIt++, Ret);
}

TEST_F(SandboxIRVecTest, Pack_WithVecOp) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i32 %v0, <2 x i32> %v1, i32 %v2, <3 x i32> %v3) {
  %ld0 = load i32, ptr %ptr
  %ld1 = load <2 x i32>, ptr %ptr
  %ld2 = load i32, ptr %ptr
  %ld3 = load <3 x i32>, ptr %ptr
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  int ArgIdx = 1;
  Argument *Arg0 = F.getArg(ArgIdx++);
  Argument *Arg1 = F.getArg(ArgIdx++);
  Argument *Arg2 = F.getArg(ArgIdx++);
  Argument *Arg3 = F.getArg(ArgIdx++);

  auto It = BB->begin();
  Instruction *Ld0 = &*It++;
  Instruction *Ld1 = &*It++;
  Instruction *Ld2 = &*It++;
  Instruction *Ld3 = &*It++;

  auto &SBBB = *Ctx.createBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  sandboxir::Value *SBArg0 = Ctx.getOrCreateArgument(Arg0);
  sandboxir::Value *SBArg1 = Ctx.getOrCreateArgument(Arg1);
  sandboxir::Value *SBArg2 = Ctx.getOrCreateArgument(Arg2);
  sandboxir::Value *SBArg3 = Ctx.getOrCreateArgument(Arg3);
  DmpVector<sandboxir::Value *> SBPackInstructions(
      {SBArg0, SBArg1, SBArg2, SBArg3});
  auto WhereIt =
      sandboxir::VecUtils::getInsertPointAfter(SBPackInstructions, &SBBB);
  auto *Pack =
      cast<sandboxir::PackInst>(sandboxir::PackInst::create(
          SBPackInstructions, WhereIt, &SBBB, Ctx));
  EXPECT_EQ(Pack->getNumOperands(), 4u);
  EXPECT_EQ(Pack->getOperand(0), SBArg0);
  EXPECT_EQ(Pack->getOperand(1), SBArg1);
  EXPECT_EQ(Pack->getOperand(2), SBArg2);
  EXPECT_EQ(Pack->getOperand(3), SBArg3);

  sandboxir::Value *SBL0 = Ctx.getValue(Ld0);
  sandboxir::Value *SBL1 = Ctx.getValue(Ld1);
  sandboxir::Value *SBL2 = Ctx.getValue(Ld2);
  sandboxir::Value *SBL3 = Ctx.getValue(Ld3);
  Pack->setOperand(0, SBL0);
  EXPECT_EQ(Pack->getOperand(0), SBL0);
  Pack->setOperand(1, SBL1);
  EXPECT_EQ(Pack->getOperand(1), SBL1);
  Pack->setOperand(2, SBL2);
  EXPECT_EQ(Pack->getOperand(2), SBL2);
  Pack->setOperand(3, SBL3);
  EXPECT_EQ(Pack->getOperand(3), SBL3);

#ifndef NDEBUG
  Pack->verify();
#endif
  // Check that we crash if we try to set an operand of the wrong type
  EXPECT_NE(SBL0->getType(), SBL1->getType());
#ifndef NDEBUG
  EXPECT_DEATH(Pack->setOperand(0, SBL1), ".*wrong type.*");
#endif
}

TEST_F(SandboxIRVecTest, SBPackInstruction_FoldedConst) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld = load i32, ptr %ptr
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto It = BB->begin();
  Instruction *Ld = &*It++;
  Constant *C1 = Constant::getIntegerValue(Type::getInt32Ty(C), APInt(32, 42));
  auto &SBBB = *Ctx.createBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  sandboxir::Value *SBL = Ctx.getValue(Ld);
  auto *SBC = Ctx.getOrCreateConstant(C1);
  DmpVector<sandboxir::Value *> SBPackInstructions({SBC, SBL});
  auto WhereIt =
      sandboxir::VecUtils::getInsertPointAfter(SBPackInstructions, &SBBB);
  auto *Pack =
      cast<sandboxir::PackInst>(sandboxir::PackInst::create(
          SBPackInstructions, WhereIt, &SBBB, Ctx));
  EXPECT_EQ(Pack->getNumOperands(), 2u);
  EXPECT_EQ(Pack->getOperand(0), SBC);
  EXPECT_EQ(Pack->getOperand(1), SBL);
#ifndef NDEBUG
  Pack->verify();
#endif

  // Check that we can update a folded constant operand.
  Constant *C2 = Constant::getIntegerValue(Type::getInt32Ty(C), APInt(32, 43));
  auto *SBC2 = Ctx.getOrCreateConstant(C2);
  Pack->setOperand(0, SBC2);
  EXPECT_EQ(Pack->getOperand(0), SBC2);
}

TEST_F(SandboxIRVecTest, SBPackInstruction_SimpleIRPattern_InOrder) {
  parseIR(C, R"IR(
define <4 x i32> @foo(i32 %v0, i32 %v1, i32 %v2, i32 %v3) {
  %Pack0 = insertelement <4 x i32> poison, i32 %v0, i64 0
  %Pack1 = insertelement <4 x i32> %Pack0, i32 %v1, i64 1
  %Pack2 = insertelement <4 x i32> %Pack1, i32 %v2, i64 2
  %Pack3 = insertelement <4 x i32> %Pack2, i32 %v3, i64 3
  ret <4 x i32> %Pack3
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto &SBBB = *Ctx.createBasicBlock(BB);
  (void)SBBB;
#ifndef NDEBUG
  SBBB.verify();
#endif
  int ArgIdx = 0;
  Argument *Arg0 = F.getArg(ArgIdx++);
  Argument *Arg1 = F.getArg(ArgIdx++);
  Argument *Arg2 = F.getArg(ArgIdx++);
  Argument *Arg3 = F.getArg(ArgIdx++);
  sandboxir::Argument *SBArg0 = Ctx.getArgument(Arg0);
  sandboxir::Argument *SBArg1 = Ctx.getArgument(Arg1);
  sandboxir::Argument *SBArg2 = Ctx.getArgument(Arg2);
  sandboxir::Argument *SBArg3 = Ctx.getArgument(Arg3);
  auto *BotInsert = cast<InsertElementInst>(BB->getTerminator()->getPrevNode());
  auto *Ret = BB->getTerminator();

  auto *Pack = cast<sandboxir::PackInst>(Ctx.getValue(BotInsert));
  auto *SBRet = cast<sandboxir::Instruction>(Ctx.getValue(Ret));

#ifndef NDEBUG
  Pack->verify();
#endif

  // Check the operands
  EXPECT_EQ(Pack->getNumOperands(), 4u);
  EXPECT_EQ(Pack->getOperand(0), SBArg0);
  EXPECT_EQ(Pack->getOperand(1), SBArg1);
  EXPECT_EQ(Pack->getOperand(2), SBArg2);
  EXPECT_EQ(Pack->getOperand(3), SBArg3);
  // Check users
  EXPECT_EQ(Pack->getNumUsers(), 1u);
  EXPECT_EQ(*Pack->users().begin(), SBRet);

  SBRet->eraseFromParent();
  // Check that eraseFromParent() erases the instructions.
  Pack->eraseFromParent();
  ASSERT_EQ(BB->size(), 0u);
}

TEST_F(SandboxIRVecTest, PackEraseFromParent_DropAllUses) {
  parseIR(C, R"IR(
define void @foo(i32 %arg0, <2 x i32> %arg1, i32 %arg2) {
  %Pack = insertelement <4 x i32> poison, i32 %arg0, i64 0
  %XPack = extractelement <2 x i32> %arg1, i64 0
  %Pack1 = insertelement <4 x i32> %Pack, i32 %XPack, i64 1
  %XPack2 = extractelement <2 x i32> %arg1, i64 1
  %Pack3 = insertelement <4 x i32> %Pack1, i32 %XPack2, i64 2
  %Pack4 = insertelement <4 x i32> %Pack3, i32 %arg2, i64 3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  Instruction *Ret = BB->getTerminator();
  (void)Ret;
  auto *SBF = Ctx.createFunction(&F);
  (void)SBF;
  auto *SBBB = Ctx.getBasicBlock(BB);
  auto It = SBBB->begin();
  auto *Pack = &*It++;

  unsigned ArgIdx = 0;
  auto *Arg0 = F.getArg(ArgIdx++);
  auto *Arg1 = F.getArg(ArgIdx++);
  auto *Arg2 = F.getArg(ArgIdx++);

  Pack->eraseFromParent();
  // Check that erasing the Pack also drops all its operand Uses.
  EXPECT_TRUE(Arg0->users().empty());
  EXPECT_TRUE(Arg1->users().empty());
  EXPECT_TRUE(Arg2->users().empty());
}

TEST_F(SandboxIRVecTest,
       SBPackInstruction_DestructorRemoveFromLLVMValueToValueMap) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
bb0:
  %ld = load <2 x i32>, ptr %ptr
  br label %bb1

bb1:
  %extr0 = extractelement <2 x i32> %ld, i32 0
  %ins0 = insertelement <2 x i32> poison, i32 %extr0, i32 0
  %extr1 = extractelement <2 x i32> %ld, i32 1
  %ins1 = insertelement <2 x i32> %ins0, i32 %extr1, i32 1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  // Create an sandboxir::BasicBlock for BB1. This should generate a
  // sandboxir::PackInst.
  {
    BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
    auto &SBBB1 = *Ctx.createBasicBlock(BB1);
#ifndef NDEBUG
    SBBB1.verify();
#endif
    auto *BotInsert =
        cast<InsertElementInst>(BB1->getTerminator()->getPrevNode());
    auto *Pack = cast<sandboxir::PackInst>(Ctx.getValue(BotInsert));
    (void)Pack;
#ifndef NDEBUG
    Pack->verify();
#endif
    SBBB1.detachFromLLVMIR();
  }
  // Now SBBB1 has been deleted. Create a new SBBB0 for BB0 and check that
  // the load has a null user (which is in SBBB1 that has now been deleted).
  {
    BasicBlock *BB0 = getBasicBlockByName(F, "bb0");
    auto &SBBB0 = *Ctx.createBasicBlock(BB0);
    (void)SBBB0;
#ifndef NDEBUG
    SBBB0.verify();
#endif
    auto It = BB0->begin();
    Instruction *Ld = &*It++;
    auto *SBL = Ctx.getValue(Ld);
    sandboxir::Use Use0 = *SBL->use_begin();
    EXPECT_EQ(Use0.getUser(), nullptr);
    EXPECT_TRUE(SBL->users().begin() != SBL->users().end());
#ifndef NDEBUG
    // This shouldn't crash
    std::string Str;
    raw_string_ostream SS(Str);
    SBL->dump(SS);
#endif
  }
}

TEST_F(SandboxIRVecTest, UnpackFromConstant) {
  parseIR(C, R"IR(
define i32 @foo() {
  %extr0 = extractelement <2 x i32> <i32 42, i32 43>, i32 0
  ret i32 %extr0
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB0 = &*F.begin();
  auto *Extract = &*BB0->begin();
  auto *CVec = cast<Constant>(Extract->getOperand(0));
  auto &SBBB = *Ctx.createBasicBlock(BB0);
  auto It = SBBB.begin();
  auto *Unpack = &*It++;
  (void)Unpack;
  auto *Ret = &*It++;
  (void)Ret;
  auto *SBCVec = Ctx.getConstant(CVec);
  EXPECT_EQ(Unpack->getOperand(0), SBCVec);
}

TEST_F(SandboxIRVecTest, UnpackInstructionCreate) {
  parseIR(C, R"IR(
define void @foo(<2 x i8> %vec) {
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  auto *SBF = Ctx.createFunction(&F);
  auto *BB = Ctx.getBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *Ret = cast<sandboxir::RetInst>(&*It++);
  auto *Arg = SBF->getArg(0);
  auto *Unpack1 = cast<sandboxir::UnpackInst>(
      sandboxir::UnpackInst::create(Arg, 0, 1, BB, Ctx));
  EXPECT_EQ(&*BB->rbegin(), Unpack1);

  auto *Pack2 = cast<sandboxir::UnpackInst>(
      sandboxir::UnpackInst::create(Arg, 0, 1, Ret, Ctx));
  EXPECT_EQ(Ret->getPrevNode(), Pack2);

  auto WhereIt = BB->end();
  auto *Pack3 = cast<sandboxir::UnpackInst>(
      sandboxir::UnpackInst::create(Arg, 0, 1, WhereIt, BB, Ctx));
  EXPECT_EQ(&*BB->rbegin(), Pack3);
}

// Check that a vector unpack (shuffle) is recognized correctly.
TEST_F(SandboxIRVecTest, VectorUnpackDetection) {
  parseIR(C, R"IR(
define void @foo(<4 x i8> %v) {
  %Op = add <4 x i8> %v, %v
  %Unpack0 = shufflevector <4 x i8> poison, <4 x i8> %Op, <2 x i32> <i32 4, i32 5>
  %Unpack1 = shufflevector <4 x i8> poison, <4 x i8> %Op, <2 x i32> <i32 5, i32 6>
  %Unpack2 = shufflevector <4 x i8> poison, <4 x i8> %Op, <1 x i32> <i32 6>
  %NotUnpack0 = shufflevector <4 x i8> poison, <4 x i8> %Op, <2 x i32> <i32 5, i32 4>
  %NotUnpack1 = shufflevector <4 x i8> poison, <4 x i8> %Op, <2 x i32> <i32 4, i32 6>
  %NotUnpack2 = shufflevector <4 x i8> poison, <4 x i8> %Op, <2 x i32> <i32 0, i32 4>
  %NotUnpack3 = shufflevector <4 x i8> poison, <4 x i8> %Op, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
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
  sandboxir::SBVecContext Ctx(C, AA);
  auto *SBF = Ctx.createFunction(&F);
  auto *BB = &*SBF->begin();
  auto It = BB->begin();
  auto *Op = &*It++;
  (void)Op;
  auto *Unpack0 = cast<sandboxir::UnpackInst>(&*It++);
  EXPECT_EQ(Unpack0->getUnpackLane(), 0u);
  auto *Unpack1 = cast<sandboxir::UnpackInst>(&*It++);
  EXPECT_EQ(Unpack1->getUnpackLane(), 1u);
  auto *Unpack2 = cast<sandboxir::UnpackInst>(&*It++);
  EXPECT_EQ(Unpack2->getUnpackLane(), 2u);
  EXPECT_EQ(sandboxir::VecUtils::getNumLanes(Unpack2), 1u);
  auto *NotUnpack0 = dyn_cast<sandboxir::UnpackInst>(&*It++);
  EXPECT_EQ(NotUnpack0, nullptr);
  auto *NotUnpack1 = dyn_cast<sandboxir::UnpackInst>(&*It++);
  EXPECT_EQ(NotUnpack1, nullptr);
  auto *NotUnpack2 = dyn_cast<sandboxir::UnpackInst>(&*It++);
  EXPECT_EQ(NotUnpack2, nullptr);
  auto *NotUnpack3 = dyn_cast<sandboxir::UnpackInst>(&*It++);
  EXPECT_EQ(NotUnpack3, nullptr);

  auto BeginIt = Unpack0->op_begin();
  auto EndIt = Unpack0->op_end();
  unsigned Cnt = 0;
  for (auto It = BeginIt; It != EndIt; ++It)
    ++Cnt;
  EXPECT_EQ(Cnt, 1u);
#ifndef NDEBUG
  BB->verify();
#endif
}

// When creating a new instruction with a constant operand, its corresponding
// SBConstant may need to be created. Check if it is missing.
TEST_F(SandboxIRVecTest, NoNullOperands) {
  parseIR(C, R"IR(
define void @foo(<4 x i8> %v) {
  %Op = add <4 x i8> %v, %v
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
  sandboxir::SBVecContext Ctx(C, AA);
  auto *SBF = Ctx.createFunction(&F);
  auto *BB = &*SBF->begin();
  auto It = BB->begin();
  auto *Op = &*It++;
  auto WhereIt = sandboxir::VecUtils::getInsertPointAfter({Op}, BB);
  auto *Unpack =
      cast<sandboxir::Instruction>(sandboxir::UnpackInst::create(
          Op, /*Lane=*/0, /*LanesToUnpack=*/1, WhereIt, BB, Ctx));
  for (sandboxir::Value *Op : Unpack->operands())
    EXPECT_NE(Op, nullptr);
#ifndef NDEBUG
  BB->verify();
#endif
}

TEST_F(SandboxIRVecTest, PackConstants) {
  parseIR(C, R"IR(
define <2 x i32> @foo() {
  %ins0 = insertelement <2 x i32> poison, i32 42, i32 0
  %ins1 = insertelement <2 x i32> %ins0, i32 43, i32 1
  ret <2 x i32> %ins1
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB0 = &*F.begin();
  auto It = BB0->begin();
  auto *Ins0 = &*It++;
  auto *Ins1 = &*It++;
  auto *C0 = cast<Constant>(Ins0->getOperand(1));
  auto *C1 = cast<Constant>(Ins1->getOperand(1));

  auto &SBBB = *Ctx.createBasicBlock(BB0);
  auto It2 = SBBB.begin();
  auto *Pack = &*It2++;
  (void)Pack;
  auto *Ret = &*It2++;
  (void)Ret;
  auto *SBC0 = Ctx.getConstant(C0);
  auto *SBC1 = Ctx.getConstant(C1);
  EXPECT_EQ(Pack->getOperand(0), SBC0);
  EXPECT_EQ(Pack->getOperand(1), SBC1);
#ifndef NDEBUG
  SBBB.verify();
#endif
}

// Check that a vector operand to a pack that is used as a whole, is counted as
// a single operand.
//
//  I (2xwide)
//  |
// Pack
//
TEST_F(SandboxIRVecTest, DuplicateUsesIntoPacks) {
  parseIR(C, R"IR(
define void @foo(<2 x i8> %v) {
  %Op = add <2 x i8> %v, %v
  %Extr0 = extractelement <2 x i8> %Op, i64 0
  %Pack0 = insertelement <2 x i8> poison, i8 %Extr0, i64 0
  %Extr1 = extractelement <2 x i8> %Op, i64 1
  %Pack1 = insertelement <2 x i8> %Pack0, i8 %Extr1, i64 1
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
  sandboxir::SBVecContext Ctx(C, AA);
  auto *SBF = Ctx.createFunction(&F);
  auto *BB = &*SBF->begin();
  auto It = BB->begin();
  auto *Op = &*It++;
  auto *Pack = cast<sandboxir::PackInst>(&*It++);
  // Check Operands/Users.
  unsigned CntUsers = 0u;
  for (auto *User : Op->users()) {
    (void)User;
    ++CntUsers;
  }
  EXPECT_EQ(CntUsers, 1u);
  unsigned CntOperands = 0u;
  for (sandboxir::Value *Op : Pack->operands()) {
    (void)Op;
    ++CntOperands;
  }
  EXPECT_EQ(CntOperands, 1u);

  // Check OperandUses/Uses.
  unsigned CntUses = 0u;
  for (const sandboxir::Use &Use : Op->uses()) {
    (void)Use;
    ++CntUses;
  }
  EXPECT_EQ(CntUses, 1u);

  unsigned CntOpUses = 0u;
  for (const sandboxir::Use &OpUse : Pack->operands()) {
    (void)OpUse;
    ++CntOpUses;
  }
  EXPECT_EQ(CntOpUses, 1u);
#ifndef NDEBUG
  BB->verify();
#endif
}

TEST_F(SandboxIRVecTest, UsesIntoUnpack) {
  parseIR(C, R"IR(
define void @foo(<4 x i8> %v) {
  %Op = add <4 x i8> %v, %v
  %Shuff = shufflevector <4 x i8> poison, <4 x i8> %Op, <2 x i32> <i32 4, i32 5>
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
  sandboxir::SBVecContext Ctx(C, AA);
  auto *SBF = Ctx.createFunction(&F);
  auto *BB = &*SBF->begin();
  auto It = BB->begin();
  auto *Op = &*It++;
  auto *Unpack = cast<sandboxir::UnpackInst>(&*It++);
  // Check Operands/Users.
  unsigned CntUsers = 0u;
  for (auto *User : Op->users()) {
    (void)User;
    ++CntUsers;
  }
  EXPECT_EQ(CntUsers, 1u);
  unsigned CntOperands = 0u;
  for (sandboxir::Value *Op : Unpack->operands()) {
    (void)Op;
    ++CntOperands;
  }
  EXPECT_EQ(CntOperands, 1u);

  // Check OperandUses/Uses.
  unsigned CntUses = 0u;
  for (const sandboxir::Use &Use : Op->uses()) {
    (void)Use;
    ++CntUses;
  }
  EXPECT_EQ(CntUses, 1u);

  unsigned CntOpUses = 0u;
  for (const sandboxir::Use &OpUse : Unpack->operands()) {
    (void)OpUse;
    ++CntOpUses;
  }
  EXPECT_EQ(CntOpUses, 1u);
#ifndef NDEBUG
  BB->verify();
#endif
}

TEST_F(SandboxIRVecTest, UsesIntoShuffle) {
  parseIR(C, R"IR(
define void @foo(<4 x i8> %v) {
  %Op = add <4 x i8> %v, %v
  %Shuff = shufflevector <4 x i8> %Op, <4 x i8> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
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
  sandboxir::SBVecContext Ctx(C, AA);
  auto *SBF = Ctx.createFunction(&F);
  auto *BB = &*SBF->begin();
  auto It = BB->begin();
  auto *Op = &*It++;
  auto *Shuff = cast<sandboxir::ShuffleInst>(&*It++);
  // Check Operands/Users.
  unsigned CntUsers = 0u;
  for (auto *User : Op->users()) {
    (void)User;
    ++CntUsers;
  }
  EXPECT_EQ(CntUsers, 1u);
  unsigned CntOperands = 0u;
  for (sandboxir::Value *Op : Shuff->operands()) {
    (void)Op;
    ++CntOperands;
  }
  EXPECT_EQ(CntOperands, 1u);

  // Check OperandUses/Uses.
  unsigned CntUses = 0u;
  for (const sandboxir::Use &Use : Op->uses()) {
    (void)Use;
    ++CntUses;
  }
  EXPECT_EQ(CntUses, 1u);

  unsigned CntOpUses = 0u;
  for (const sandboxir::Use &OpUse : Shuff->operands()) {
    (void)OpUse;
    ++CntOpUses;
  }
  EXPECT_EQ(CntOpUses, 1u);
#ifndef NDEBUG
  BB->verify();
#endif
}

TEST_F(SandboxIRVecTest, SHuffleInstructionCreate) {
  parseIR(C, R"IR(
define void @foo(<2 x i8> %vec) {
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  auto *SBF = Ctx.createFunction(&F);
  auto *BB = Ctx.getBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *Ret = cast<sandboxir::RetInst>(&*It++);
  auto *Arg = SBF->getArg(0);
  sandboxir::ShuffleMask Mask{1, 0};
  auto *Shuff1 = cast<sandboxir::ShuffleInst>(
      sandboxir::ShuffleInst::create(Arg, Mask, BB, Ctx));
  EXPECT_EQ(&*BB->rbegin(), Shuff1);

  auto *Shuff2 = cast<sandboxir::ShuffleInst>(
      sandboxir::ShuffleInst::create(Arg, Mask, Ret, Ctx));
  EXPECT_EQ(Ret->getPrevNode(), Shuff2);

  auto WhereIt = BB->end();
  auto *Shuff3 = cast<sandboxir::ShuffleInst>(
      sandboxir::ShuffleInst::create(Arg, Mask, WhereIt, BB, Ctx));
  EXPECT_EQ(&*BB->rbegin(), Shuff3);
}

// Checks detaching an sandboxir::BasicBlock from its underlying BB.
TEST_F(SandboxIRVecTest, SBBasicBlockDestruction_WithPack) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld = load <2 x i32>, ptr %ptr
  %extr0 = extractelement <2 x i32> %ld, i32 0
  %ins0 = insertelement <2 x i32> poison, i32 %extr0, i32 0
  %extr1 = extractelement <2 x i32> %ld, i32 1
  %ins1 = insertelement <2 x i32> %ins0, i32 %extr1, i32 1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  sandboxir::SBVecContext Ctx(C, AA);

  BasicBlock *BB = &*F.begin();
  auto It = BB->begin();
  auto *Ld = &*It++;
  auto *Extr0 = &*It++;
  auto *Ins0 = &*It++;
  auto *Extr1 = &*It++;
  auto *Ins1 = &*It++;
  auto *Ret = &*It++;
  unsigned BBSize = BB->size();
  {
    auto &SBBB = *Ctx.createBasicBlock(BB);
#ifndef NDEBUG
    SBBB.verify();
#endif
    Ctx.getTracker().start(&SBBB);
    auto It = SBBB.begin();
    auto *SBLd = Ctx.getValue(Ld);
    EXPECT_EQ(&*It++, SBLd);
    auto *Pack = cast<sandboxir::PackInst>(Ctx.getValue(Extr0));
    EXPECT_EQ(&*It++, Pack);
    auto *SBRet = Ctx.getValue(Ret);
    EXPECT_EQ(&*It++, SBRet);
  }
  // Check that BB is still intact.
  EXPECT_EQ(BBSize, BB->size());
  It = BB->begin();
  EXPECT_EQ(&*It++, Ld);
  EXPECT_EQ(&*It++, Extr0);
  EXPECT_EQ(&*It++, Ins0);
  EXPECT_EQ(&*It++, Extr1);
  EXPECT_EQ(&*It++, Ins1);
  EXPECT_EQ(&*It++, Ret);
  // Expect that clearing a BB does not track changes.
  EXPECT_TRUE(Ctx.getTracker().empty());
  Ctx.getTracker().accept();
}
