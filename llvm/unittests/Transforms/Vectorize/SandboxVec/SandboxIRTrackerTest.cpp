//===- SandboxIRTrackerTest.cpp ----------------------------------------------===//
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
#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"
#include "gtest/gtest.h"

using namespace llvm;

struct SandboxIRTrackerTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;
  TargetLibraryInfoImpl TLII;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("SandboxIRTrackerTest", errs());
  }
  BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
    for (BasicBlock &BB : F)
      if (BB.getName() == Name)
        return &BB;
    llvm_unreachable("Expected to find basic block!");
  }
};

TEST_F(SandboxIRTrackerTest, RUWIf) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep0 = getelementptr float, ptr %ptr, i32 0
  %gep1 = getelementptr float, ptr %ptr, i32 1
  %ld0 = load float, ptr %gep0
  store float undef, ptr %gep0
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
  SBContext Ctxt(C, AA);
  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctxt.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBGep0 = &*It++;
  auto *SBGep1 = &*It++;
  auto *SBLd = &*It++;
  auto *SBSt = &*It++;
  SBGep0->replaceUsesWithIf(
      SBGep1,
      [SBSt](SBUser *DstU, unsigned OpIdx) { return DstU == SBSt; });
  EXPECT_EQ(SBSt->getOperand(1), SBGep1);
  EXPECT_EQ(SBLd->getOperand(0), SBGep0);
  Ctxt.getTracker().revert();
  EXPECT_EQ(SBSt->getOperand(1), SBGep0);
  EXPECT_EQ(SBLd->getOperand(0), SBGep0);
  Ctxt.getTracker().stop();
}

TEST_F(SandboxIRTrackerTest, RUWIfTest_Pack) {
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
  SBContext Ctxt(C, AA);
  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctxt.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBAdd0 = &*It++;
  auto *SBAdd1 = &*It++;
  auto *SBPack = &*It++;
  SBAdd0->replaceUsesWithIf(
      SBAdd1, [](SBUser *DstU, unsigned OpIdx) { return true; });
  EXPECT_EQ(SBPack->getOperand(0), SBAdd1);
  EXPECT_EQ(SBPack->getOperand(1), SBAdd1);
  Ctxt.getTracker().revert();
  EXPECT_EQ(SBPack->getOperand(0), SBAdd0);
  EXPECT_EQ(SBPack->getOperand(1), SBAdd1);
  Ctxt.getTracker().stop();
}

TEST_F(SandboxIRTrackerTest, RAUW) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep0 = getelementptr float, ptr %ptr, i32 0
  %gep1 = getelementptr float, ptr %ptr, i32 1
  %ld0 = load float, ptr %gep0
  store float undef, ptr %gep0
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
  SBContext Ctxt(C, AA);
  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctxt.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBGep0 = &*It++;
  auto *SBGep1 = &*It++;
  auto *SBLd = &*It++;
  auto *SBSt = &*It++;
  SBGep0->replaceAllUsesWith(SBGep1);
  EXPECT_EQ(SBSt->getOperand(1), SBGep1);
  EXPECT_EQ(SBLd->getOperand(0), SBGep1);
  Ctxt.getTracker().revert();
  EXPECT_EQ(SBSt->getOperand(1), SBGep0);
  EXPECT_EQ(SBLd->getOperand(0), SBGep0);
  Ctxt.getTracker().stop();
}

TEST_F(SandboxIRTrackerTest, RAUW_Pack_SameOperand) {
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
  SBContext Ctxt(C, AA);
  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctxt.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBAdd0 = &*It++;
  auto *SBAdd1 = &*It++;
  auto *SBPack = &*It++;
  SBAdd0->replaceAllUsesWith(SBAdd1);
  EXPECT_EQ(SBPack->getOperand(0), SBAdd1);
  EXPECT_EQ(SBPack->getOperand(1), SBAdd1);
  Ctxt.getTracker().revert();
  EXPECT_EQ(SBPack->getOperand(0), SBAdd0);
  EXPECT_EQ(SBPack->getOperand(1), SBAdd0);
  Ctxt.getTracker().stop();
}

TEST_F(SandboxIRTrackerTest, RUOW) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep0 = getelementptr float, ptr %ptr, i32 0
  %gep1 = getelementptr float, ptr %ptr, i32 1
  %ld0 = load float, ptr %gep0
  store float undef, ptr %gep0
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
  SBContext Ctxt(C, AA);
  SBFunction *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctxt.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBGep0 = &*It++;
  auto *SBGep1 = &*It++;
  auto *SBLd = &*It++;
  (void)SBLd;
  auto *SBSt = &*It++;
  SBSt->replaceUsesOfWith(SBGep0, SBGep1);
  EXPECT_EQ(SBSt->getOperand(1), SBGep1);
  Ctxt.getTracker().revert();
  EXPECT_EQ(SBSt->getOperand(1), SBGep0);
  Ctxt.getTracker().stop();
}

TEST_F(SandboxIRTrackerTest, SetOperand) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep0 = getelementptr float, ptr %ptr, i32 0
  %gep1 = getelementptr float, ptr %ptr, i32 1
  %ld0 = load float, ptr %gep0
  store float undef, ptr %gep0
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
  SBContext Ctxt(C, AA);
  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctxt.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBGep0 = &*It++;
  auto *SBGep1 = &*It++;
  auto *SBLd = &*It++;
  auto *SBSt = &*It++;
  SBSt->setOperand(0, SBLd);
  SBSt->setOperand(1, SBGep1);
  SBLd->setOperand(0, SBGep1);
  EXPECT_EQ(SBSt->getOperand(0), SBLd);
  EXPECT_EQ(SBSt->getOperand(1), SBGep1);
  EXPECT_EQ(SBLd->getOperand(0), SBGep1);
  Ctxt.getTracker().revert();
  EXPECT_NE(SBSt->getOperand(0), SBLd);
  EXPECT_EQ(SBSt->getOperand(1), SBGep0);
  EXPECT_EQ(SBLd->getOperand(0), SBGep0);
  Ctxt.getTracker().stop();
}

TEST_F(SandboxIRTrackerTest, SetOperand_Pack) {
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
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctxt.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBAdd0 = &*It++;
  auto *SBAdd1 = &*It++;
  auto *Pack = &*It++;
  Pack->setOperand(0, SBAdd1);
  Pack->setOperand(1, SBAdd0);
  Ctxt.getTracker().revert();
  EXPECT_EQ(Pack->getOperand(0), SBAdd0);
  EXPECT_EQ(Pack->getOperand(1), SBAdd1);
  Ctxt.getTracker().stop();
}

TEST_F(SandboxIRTrackerTest, RevertErase) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i32 %v0) {
  %add0 = add i32 %v0, %v0
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
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctxt.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBAdd0 = &*It++;
  auto *SBAdd1 = &*It++;
  auto *SBRet = &*It++;
  SBAdd1->eraseFromParent();
  Ctxt.getTracker().revert();
  It = SBBB->begin();
  EXPECT_EQ(&*It++, SBAdd0);
  EXPECT_EQ(&*It++, SBAdd1);
  EXPECT_EQ(&*It++, SBRet);

  // Check revert removeFromParent().
  SBAdd0->removeFromParent();
  EXPECT_EQ(&*SBBB->begin(), SBAdd1);
  Ctxt.getTracker().revert();
  It = SBBB->begin();
  EXPECT_EQ(&*It++, SBAdd0);
  EXPECT_EQ(&*It++, SBAdd1);
  EXPECT_EQ(SBAdd1->getOperand(0), SBAdd0);
  EXPECT_EQ(SBAdd1->getOperand(1), SBAdd0);
  EXPECT_EQ(&*It++, SBRet);

  Ctxt.getTracker().stop();
}

// Packs are more challenging to revert-erase because all inserts/extracts need
// to be re-inserted in the right order.
TEST_F(SandboxIRTrackerTest, RevertErasePack) {
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
  SBContext Ctxt(C, AA);
  BasicBlock *BB = &*F.begin();
  auto BBIt = BB->begin();
  auto *Ld = &*BBIt++;
  auto *Extr0 = &*BBIt++;
  auto *Ins0 = &*BBIt++;
  auto *Extr1 = &*BBIt++;
  auto *Ins1 = &*BBIt++;
  auto *Ret = &*BBIt++;

  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctxt.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBLd = &*It++;
  auto *Pack = &*It++;
  auto *SBRet = &*It;
  Pack->eraseFromParent();

  Ctxt.getTracker().revert();
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
  Ctxt.getTracker().stop();
}

// This checks that creating an SBInstruction registers the right callbacks,
// and that reverting works.
TEST_F(SandboxIRTrackerTest, RevertCreate) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, <2 x i32> %v0) {
  %add0 = add <2 x i32> %v0, %v0
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
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctxt.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBAdd0 = &*It++;
  auto *SBRet = &*It++;
  auto *New = cast<SBInstruction>(
      Ctxt.createSBUnpackInstruction(SBAdd0, 0, SBBB, 2));
  EXPECT_EQ(Ctxt.getTracker().size(), 1u);
#ifndef NDEBUG
  unsigned Idx = 0;
  EXPECT_TRUE(isa<CreateAndInsertInstr>(Ctxt.getTracker().getChange(Idx++)));
#endif
  It = SBBB->begin();
  EXPECT_EQ(&*It++, SBAdd0);
  EXPECT_EQ(&*It++, New);
  EXPECT_EQ(&*It++, SBRet);

  Ctxt.getTracker().revert();
  It = SBBB->begin();
  EXPECT_EQ(&*It++, SBAdd0);
  EXPECT_EQ(&*It++, SBRet);
  Ctxt.getTracker().stop();
}

TEST_F(SandboxIRTrackerTest, CreateInstrAndRevert) {
  parseIR(C, R"IR(
define void @foo(i32 %v1, ptr %ptr) {
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
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
  BasicBlock *BB = &*F.begin();
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  SBInstruction *SBRet = &*It++;

  // Create an instruction and check the changes.
  auto *Val = SBF->getArg(0);
  auto *Ptr = SBF->getArg(1);
  Ctxt.getTracker().start(SBBB);
  auto *NewSBI = SBStoreInstruction::create(
      Val, Ptr, /*Align=*/std::nullopt, SBRet, Ctxt);
  (void)NewSBI;
  // Check the changes appended by create().
  EXPECT_EQ(Ctxt.getTracker().size(), 1u);
#ifndef NDEBUG
  unsigned Idx = 0;
  EXPECT_TRUE(isa<CreateAndInsertInstr>(Ctxt.getTracker().getChange(Idx++)));
#endif

  Ctxt.getTracker().revert();
  // Check that revert() removes the IR instr from the BB.
  EXPECT_EQ(BB->size(), 1u);
  It = SBBB->begin();
  EXPECT_EQ(&*It++, SBRet);
  // Check that the operands have been dropped.
  EXPECT_TRUE(F.getArg(0)->users().empty());
  EXPECT_TRUE(F.getArg(1)->users().empty());
}

TEST_F(SandboxIRTrackerTest, CreatePackAndRevert) {
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
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
  BasicBlock *BB = &*F.begin();
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  SBInstruction *SBRet = &*It++;

  // Create an instruction and check the changes.
  auto *Arg0 = SBF->getArg(0);
  auto *Arg1 = SBF->getArg(1);
  auto *Arg2 = SBF->getArg(2);
  Ctxt.getTracker().start(SBBB);
  auto *Pack = Ctxt.createSBPackInstruction({Arg0, Arg1, Arg2}, SBBB);
  (void)Pack;
  // Check the changes appended by create().
  EXPECT_EQ(Ctxt.getTracker().size(), 1u);
#ifndef NDEBUG
  unsigned Idx = 0;
  EXPECT_TRUE(isa<CreateAndInsertInstr>(Ctxt.getTracker().getChange(Idx++)));
#endif

  Ctxt.getTracker().revert();
  // Check that revert() removes the IR instr from the BB.
  EXPECT_EQ(BB->size(), 1u);
  It = SBBB->begin();
  EXPECT_EQ(&*It++, SBRet);
  // Check that the operands have been dropped.
  EXPECT_TRUE(F.getArg(0)->users().empty());
  EXPECT_TRUE(F.getArg(1)->users().empty());
  EXPECT_TRUE(F.getArg(2)->users().empty());
}

TEST_F(SandboxIRTrackerTest, EraseInstrAndRevert) {
  parseIR(C, R"IR(
define void @foo(i32 %v1) {
  %add0 = add i32 %v1, %v1
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
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
  BasicBlock *BB = &*F.begin();
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  SBInstruction *SBAdd0 = &*It++;

  // Create an instruction and check the changes.
  Ctxt.getTracker().start(SBBB);
  SBAdd0->eraseFromParent();
  // Check the changes appended by create().
  EXPECT_EQ(Ctxt.getTracker().size(), 1u);
#ifndef NDEBUG
  unsigned Idx = 0;
  EXPECT_TRUE(isa<EraseFromParent>(Ctxt.getTracker().getChange(Idx++)));
#endif

  Ctxt.getTracker().revert();
  // Check that revert() works.
  EXPECT_EQ(BB->size(), 2u);
  EXPECT_EQ(&SBBB->front(), SBAdd0);
  EXPECT_EQ(SBAdd0->getOperand(0), SBF->getArg(0));
  EXPECT_EQ(SBAdd0->getOperand(1), SBF->getArg(0));
}

TEST_F(SandboxIRTrackerTest, RemoveInstrCallbacks) {
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
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  SBInstruction *SBAdd0 = &*It++;
  SBInstruction *SBAdd1 = &*It++;
  SBInstruction *SBRet = &*It++;

  // Check that we get callbacks when we erase a SBInstruction
  SmallVector<SBInstruction *> ErasedIRInstrs;
  auto *Sched = Ctxt.getScheduler(SBBB);
  for (auto &SBI : reverse(*SBBB))
    Sched->trySchedule({&SBI});
  auto *CB = Ctxt.registerRemoveInstrCallback(
      [&ErasedIRInstrs](SBInstruction *SBI) {
        ErasedIRInstrs.push_back(SBI);
      });
  Ctxt.getTracker().start(SBBB);
  // Check the DAG's state.
  const InstrRange &View = Sched->getDAG().getView();
  const InstrRange &DAGRange = Sched->getDAG().getDAGRange();
  EXPECT_EQ(View.from(), SBAdd0);
  EXPECT_EQ(View.to(), SBRet);
  EXPECT_EQ(DAGRange.from(), SBAdd0);
  EXPECT_EQ(DAGRange.to(), SBRet);

  SBAdd1->eraseFromParent();
  // Check the changes appended by eraseFromParent().
  EXPECT_EQ(Ctxt.getTracker().size(), 1u);
#ifndef NDEBUG
  unsigned Idx = 0;
  EXPECT_TRUE(isa<EraseFromParent>(Ctxt.getTracker().getChange(Idx++)));
#endif
  // Check that the callback worked.
  EXPECT_EQ(ErasedIRInstrs.size(), 1u);
  EXPECT_EQ(ErasedIRInstrs[0], SBAdd1);
  // Check the DAG
  EXPECT_EQ(View.from(), SBAdd0);
  EXPECT_EQ(View.to(), SBRet);
  EXPECT_EQ(DAGRange.from(), SBAdd0);
  EXPECT_EQ(DAGRange.to(), SBRet);

  // Now unregister the callback and check if callbacks run.
  ErasedIRInstrs.clear();
  Ctxt.unregisterRemoveInstrCallback(CB);
  SBAdd0->eraseFromParent();
  // Check that callback removal worked.
  EXPECT_TRUE(ErasedIRInstrs.empty());
  // Check the changes appended by eraseFromParent().
  EXPECT_EQ(Ctxt.getTracker().size(), 2u);
#ifndef NDEBUG
  Idx = 0;
  EXPECT_TRUE(isa<EraseFromParent>(Ctxt.getTracker().getChange(Idx++)));
  EXPECT_TRUE(isa<EraseFromParent>(Ctxt.getTracker().getChange(Idx++)));
#endif
  // Check the DAG
  EXPECT_EQ(View.from(), SBRet);
  EXPECT_EQ(View.to(), SBRet);
  EXPECT_EQ(DAGRange.from(), SBRet);
  EXPECT_EQ(DAGRange.to(), SBRet);

  // Create an instruction and check the changes.
  auto *Ptr = SBF->getArg(1);
  auto *NewSBI = SBStoreInstruction::create(
      SBAdd0, Ptr, /*Align=*/std::nullopt, SBRet, Ctxt);
  ErasedIRInstrs.clear();
  Ctxt.registerRemoveInstrCallback([&ErasedIRInstrs](SBInstruction *SBI) {
    ErasedIRInstrs.push_back(SBI);
  });
  // Check the changes appended by create().
  EXPECT_EQ(Ctxt.getTracker().size(), 3u);
#ifndef NDEBUG
  Idx = 0;
  EXPECT_TRUE(isa<EraseFromParent>(Ctxt.getTracker().getChange(Idx++)));
  EXPECT_TRUE(isa<EraseFromParent>(Ctxt.getTracker().getChange(Idx++)));
  EXPECT_TRUE(isa<CreateAndInsertInstr>(Ctxt.getTracker().getChange(Idx++)));
#endif
  // Check the DAG
  EXPECT_EQ(View.from(), NewSBI);
  EXPECT_EQ(View.to(), SBRet);
  EXPECT_EQ(DAGRange.from(), NewSBI);
  EXPECT_EQ(DAGRange.to(), SBRet);

  // Check that we get callbacks on tracker revert().
  Ctxt.getTracker().revert();
  ASSERT_EQ(ErasedIRInstrs.size(), 1u);
  EXPECT_EQ(ErasedIRInstrs[0], NewSBI);
  // Check the DAG
  EXPECT_EQ(View.from(), SBAdd0);
  EXPECT_EQ(View.to(), SBRet);
  EXPECT_EQ(DAGRange.from(), SBAdd0);
  EXPECT_EQ(DAGRange.to(), SBRet);
}

TEST_F(SandboxIRTrackerTest, InsertInstrCallbacks) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = cast<SBBasicBlock>(Ctxt.getSBValue(BB));
  auto It = SBBB->begin();
  SBInstruction *SBAdd0 = &*It++;
  SBInstruction *SBAdd1 = &*It++;
  (void)SBAdd1;
  SBInstruction *SBRet = &*It++;

  // Check that we get callbacks on tracker revert().
  SmallVector<SBInstruction *> InsertedInstrs;
  Ctxt.registerInsertInstrCallback([&InsertedInstrs](SBInstruction *SBI) {
    InsertedInstrs.push_back(SBI);
  });
  SBArgument *Ptr = SBF->getArg(1);
  Ctxt.getTracker().start(SBBB);
  auto *NewSBI = SBStoreInstruction::create(
      SBAdd0, Ptr, /*Align=*/std::nullopt, SBRet, Ctxt);
  ASSERT_EQ(InsertedInstrs.size(), 1u);
  EXPECT_EQ(InsertedInstrs[0], NewSBI);
}
