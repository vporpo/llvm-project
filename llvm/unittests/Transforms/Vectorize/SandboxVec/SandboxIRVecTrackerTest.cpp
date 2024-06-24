//===- SandboxIRVecTrackerTest.cpp
//-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIRVec.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"
#include "gtest/gtest.h"

using namespace llvm;

struct SandboxIRVecTrackerTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;
  TargetLibraryInfoImpl TLII;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("SandboxIRVecTrackerTest", errs());
  }
  BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
    for (BasicBlock &BB : F)
      if (BB.getName() == Name)
        return &BB;
    llvm_unreachable("Expected to find basic block!");
  }
};

TEST_F(SandboxIRVecTrackerTest, CreatePackAndRevert) {
  parseIR(C, R"IR(
define void @foo(i32 %arg0, <2 x i32> %arg1, i32 %arg2) {
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
  BasicBlock *BB = &*F.begin();
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  sandboxir::Instruction *SBRet = &*It++;

  // Create an instruction and check the changes.
  auto *Arg0 = SBF->getArg(0);
  auto *Arg1 = SBF->getArg(1);
  auto *Arg2 = SBF->getArg(2);
  Ctx.getTracker().start(SBBB);
  DmpVector<sandboxir::Value *> Vals{Arg0, Arg1, Arg2};
  auto WhereIt = sandboxir::VecUtils::getInsertPointAfter(Vals, SBBB);
  auto *Pack = sandboxir::PackInst::create(Vals, WhereIt, SBBB, Ctx);
  (void)Pack;
  // Check the changes appended by create().
  EXPECT_EQ(Ctx.getTracker().size(), 1u);
#ifndef NDEBUG
  unsigned Idx = 0;
  EXPECT_TRUE(
      isa<sandboxir::CreateAndInsertInstr>(Ctx.getTracker().getChange(Idx++)));
#endif

  Ctx.getTracker().revert();
  // Check that revert() removes the IR instr from the BB.
  EXPECT_EQ(BB->size(), 1u);
  It = SBBB->begin();
  EXPECT_EQ(&*It++, SBRet);
  // Check that the operands have been dropped.
  EXPECT_TRUE(F.getArg(0)->users().empty());
  EXPECT_TRUE(F.getArg(1)->users().empty());
  EXPECT_TRUE(F.getArg(2)->users().empty());
}

TEST_F(SandboxIRVecTrackerTest, RUWIfTest_Pack) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i32 %v0, i32 %v1) {
  %add0 = add i32 %v0, %v0
  %add1 = add i32 %v1, %v1
  %Pack0 = insertelement <2 x i32> poison, i32 %add0, i64 0
  %Pack1 = insertelement <2 x i32> %Pack0, i32 %add1, i64 1
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
  auto *SBBB = &*SBF->begin();
  Ctx.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBAdd0 = &*It++;
  auto *SBAdd1 = &*It++;
  auto *SBPack = cast<sandboxir::PackInst>(&*It++);
  SBAdd0->replaceUsesWithIf(SBAdd1, [](sandboxir::Use Use) { return true; });
  EXPECT_EQ(SBPack->getOperand(0), SBAdd1);
  EXPECT_EQ(SBPack->getOperand(1), SBAdd1);
  Ctx.getTracker().revert();
  EXPECT_EQ(SBPack->getOperand(0), SBAdd0);
  EXPECT_EQ(SBPack->getOperand(1), SBAdd1);
  Ctx.getTracker().stop();
}

TEST_F(SandboxIRVecTrackerTest, RAUW_Pack_SameOperand) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i32 %v0, i32 %v1) {
  %add0 = add i32 %v0, %v0
  %add1 = add i32 %v1, %v1
  %Pack0 = insertelement <2 x i32> poison, i32 %add0, i64 0
  %Pack1 = insertelement <2 x i32> %Pack0, i32 %add0, i64 1
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
  auto *SBBB = &*SBF->begin();
  Ctx.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBAdd0 = &*It++;
  auto *SBAdd1 = &*It++;
  auto *SBPack = cast<sandboxir::PackInst>(&*It++);
  SBAdd0->replaceAllUsesWith(SBAdd1);
  EXPECT_EQ(SBPack->getOperand(0), SBAdd1);
  EXPECT_EQ(SBPack->getOperand(1), SBAdd1);
  Ctx.getTracker().revert();
  EXPECT_EQ(SBPack->getOperand(0), SBAdd0);
  EXPECT_EQ(SBPack->getOperand(1), SBAdd0);
  Ctx.getTracker().stop();
}

TEST_F(SandboxIRVecTrackerTest, SetOperand_Pack) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i32 %v0, i32 %v1) {
  %add0 = add i32 %v0, %v0
  %add1 = add i32 %v1, %v1
  %Pack0 = insertelement <2 x i32> poison, i32 %add0, i64 0
  %Pack1 = insertelement <2 x i32> %Pack0, i32 %add1, i64 1
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
  auto *SBBB = &*SBF->begin();
  Ctx.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBAdd0 = &*It++;
  auto *SBAdd1 = &*It++;
  auto *Pack = cast<sandboxir::PackInst>(&*It++);
  Pack->setOperand(0, SBAdd1);
  Pack->setOperand(1, SBAdd0);
  Ctx.getTracker().revert();
  EXPECT_EQ(Pack->getOperand(0), SBAdd0);
  EXPECT_EQ(Pack->getOperand(1), SBAdd1);
  Ctx.getTracker().stop();
}

// Packs are more challenging to revert-erase because all inserts/extracts need
// to be re-inserted in the right order.
TEST_F(SandboxIRVecTrackerTest, RevertErasePack) {
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
  auto BBIt = BB->begin();
  auto *Ld = &*BBIt++;
  auto *Extr0 = &*BBIt++;
  auto *Ins0 = &*BBIt++;
  auto *Extr1 = &*BBIt++;
  auto *Ins1 = &*BBIt++;
  auto *Ret = &*BBIt++;

  auto *SBF = Ctx.createFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctx.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBLd = &*It++;
  auto *Pack = cast<sandboxir::PackInst>(&*It++);
  auto *SBRet = &*It;
  Pack->eraseFromParent();

  Ctx.getTracker().revert();
  {
    auto It = SBBB->begin();
    EXPECT_EQ(&*It++, SBLd);
    EXPECT_EQ(&*It++, Pack);
    EXPECT_EQ(&*It++, SBRet);
  }
  {
    auto It = F.begin()->begin();
    EXPECT_EQ(&*It++, Ld);
    EXPECT_EQ(&*It++, Extr0);
    EXPECT_EQ(&*It++, Ins0);
    EXPECT_EQ(&*It++, Extr1);
    EXPECT_EQ(&*It++, Ins1);
    EXPECT_EQ(&*It++, Ret);
  }
  Ctx.getTracker().stop();
}

// This checks that creating an SBInstruction registers the right callbacks,
// and that reverting works.
TEST_F(SandboxIRVecTrackerTest, RevertCreate_Unpack) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, <2 x i32> %v0) {
  %add0 = add <2 x i32> %v0, %v0
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *SBF = Ctx.createFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctx.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *Add0 = &*It++;
  auto *Ret = &*It++;
  auto *New =
      cast<sandboxir::Instruction>(sandboxir::BinaryOperator::create(
          sandboxir::Instruction::Opcode::Add, Add0, Add0, Ret, Ctx, "New"));
  EXPECT_EQ(Ctx.getTracker().size(), 1u);
#ifndef NDEBUG
  unsigned Idx = 0;
  EXPECT_TRUE(
      isa<sandboxir::CreateAndInsertInstr>(Ctx.getTracker().getChange(Idx++)));
#endif
  It = SBBB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, New);
  EXPECT_EQ(&*It++, Ret);

  Ctx.getTracker().revert();
  It = SBBB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Ret);
  Ctx.getTracker().stop();
}

TEST_F(SandboxIRVecTrackerTest, TestCallbacks) {
  parseIR(C, R"IR(
define void @foo(i32 %v1, ptr %ptr) {
  %add0 = add i32 %v1, %v1
  %add1 = add i32 %add0, %add0
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
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  sandboxir::Instruction *SBAdd0 = &*It++;
  sandboxir::Instruction *SBAdd1 = &*It++;
  sandboxir::Instruction *SBRet = &*It++;

  // Check that we get callbacks when we erase a sandboxir::Instruction
  SmallVector<sandboxir::Instruction *> ErasedIRInstrs;
  auto *Sched = Ctx.getScheduler(SBBB);
  for (auto &SBI : reverse(*SBBB))
    Sched->trySchedule({&SBI});
  auto *CB = Ctx.registerRemoveInstrCallback(
      [&ErasedIRInstrs](sandboxir::Instruction *SBI) {
        ErasedIRInstrs.push_back(SBI);
      });
  Ctx.getTracker().start(SBBB);
  // Check the DAG's state.
  const sandboxir::InstrInterval &View = Sched->getDAG().getView();
  const sandboxir::InstrInterval &DAGInterval =
      Sched->getDAG().getDAGInterval();
  EXPECT_EQ(View.from(), SBAdd0);
  EXPECT_EQ(View.to(), SBRet);
  EXPECT_EQ(DAGInterval.from(), SBAdd0);
  EXPECT_EQ(DAGInterval.to(), SBRet);

  SBAdd1->eraseFromParent();
  // Check the changes appended by eraseFromParent().
  EXPECT_EQ(Ctx.getTracker().size(), 1u);
#ifndef NDEBUG
  unsigned Idx = 0;
  EXPECT_TRUE(
      isa<sandboxir::EraseFromParent>(Ctx.getTracker().getChange(Idx++)));
#endif
  // Check that the callback worked.
  EXPECT_EQ(ErasedIRInstrs.size(), 1u);
  EXPECT_EQ(ErasedIRInstrs[0], SBAdd1);
  // Check the DAG
  EXPECT_EQ(View.from(), SBAdd0);
  EXPECT_EQ(View.to(), SBRet);
  EXPECT_EQ(DAGInterval.from(), SBAdd0);
  EXPECT_EQ(DAGInterval.to(), SBRet);

  // Now unregister the callback and check if callbacks run.
  ErasedIRInstrs.clear();
  Ctx.unregisterRemoveInstrCallback(CB);
  SBAdd0->eraseFromParent();
  // Check that callback removal worked.
  EXPECT_TRUE(ErasedIRInstrs.empty());
  // Check the changes appended by eraseFromParent().
  EXPECT_EQ(Ctx.getTracker().size(), 2u);
#ifndef NDEBUG
  Idx = 0;
  EXPECT_TRUE(
      isa<sandboxir::EraseFromParent>(Ctx.getTracker().getChange(Idx++)));
  EXPECT_TRUE(
      isa<sandboxir::EraseFromParent>(Ctx.getTracker().getChange(Idx++)));
#endif
  // Check the DAG
  EXPECT_EQ(View.from(), SBRet);
  EXPECT_EQ(View.to(), SBRet);
  EXPECT_EQ(DAGInterval.from(), SBRet);
  EXPECT_EQ(DAGInterval.to(), SBRet);

  // Create an instruction and check the changes.
  auto *Ptr = SBF->getArg(1);
  auto *NewSBI = sandboxir::StoreInst::create(
      SBAdd0, Ptr, /*Align=*/std::nullopt, SBRet, Ctx);
  ErasedIRInstrs.clear();
  Ctx.registerRemoveInstrCallback(
      [&ErasedIRInstrs](sandboxir::Instruction *SBI) {
        ErasedIRInstrs.push_back(SBI);
      });
  // Check the changes appended by create().
  EXPECT_EQ(Ctx.getTracker().size(), 3u);
#ifndef NDEBUG
  Idx = 0;
  EXPECT_TRUE(
      isa<sandboxir::EraseFromParent>(Ctx.getTracker().getChange(Idx++)));
  EXPECT_TRUE(
      isa<sandboxir::EraseFromParent>(Ctx.getTracker().getChange(Idx++)));
  EXPECT_TRUE(
      isa<sandboxir::CreateAndInsertInstr>(Ctx.getTracker().getChange(Idx++)));
#endif
  // Check the DAG
  EXPECT_EQ(View.from(), NewSBI);
  EXPECT_EQ(View.to(), SBRet);
  EXPECT_EQ(DAGInterval.from(), NewSBI);
  EXPECT_EQ(DAGInterval.to(), SBRet);

  // Check that we get callbacks on tracker revert().
  Ctx.getTracker().revert();
  ASSERT_EQ(ErasedIRInstrs.size(), 1u);
  EXPECT_EQ(ErasedIRInstrs[0], NewSBI);
  // Check the DAG
  EXPECT_EQ(View.from(), SBAdd0);
  EXPECT_EQ(View.to(), SBRet);
  EXPECT_EQ(DAGInterval.from(), SBAdd0);
  EXPECT_EQ(DAGInterval.to(), SBRet);
}
