//===- SandboxIRTrackerTest.cpp
//----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIRVec.h"
#include "gtest/gtest.h"

using namespace llvm;

struct SandboxIRTrackerTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

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
  sandboxir::Context Ctx(C);
  auto *SBF = Ctx.createFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctx.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBGep0 = &*It++;
  auto *SBGep1 = &*It++;
  auto *SBLd = &*It++;
  auto *SBSt = &*It++;
  SBGep0->replaceUsesWithIf(
      SBGep1, [SBSt](sandboxir::Use Use) { return Use.getUser() == SBSt; });
  EXPECT_EQ(SBSt->getOperand(1), SBGep1);
  EXPECT_EQ(SBLd->getOperand(0), SBGep0);
  Ctx.getTracker().revert();
  EXPECT_EQ(SBSt->getOperand(1), SBGep0);
  EXPECT_EQ(SBLd->getOperand(0), SBGep0);
  Ctx.getTracker().stop();
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
  sandboxir::Context Ctx(C);
  auto *SBF = Ctx.createFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctx.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBGep0 = &*It++;
  auto *SBGep1 = &*It++;
  auto *SBLd = &*It++;
  auto *SBSt = &*It++;
  SBGep0->replaceAllUsesWith(SBGep1);
  EXPECT_EQ(SBSt->getOperand(1), SBGep1);
  EXPECT_EQ(SBLd->getOperand(0), SBGep1);
  Ctx.getTracker().revert();
  EXPECT_EQ(SBSt->getOperand(1), SBGep0);
  EXPECT_EQ(SBLd->getOperand(0), SBGep0);
  Ctx.getTracker().stop();
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
  sandboxir::Context Ctx(C);
  sandboxir::Function *SBF = Ctx.createFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctx.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBGep0 = &*It++;
  auto *SBGep1 = &*It++;
  auto *SBLd = &*It++;
  (void)SBLd;
  auto *SBSt = &*It++;
  SBSt->replaceUsesOfWith(SBGep0, SBGep1);
  EXPECT_EQ(SBSt->getOperand(1), SBGep1);
  Ctx.getTracker().revert();
  EXPECT_EQ(SBSt->getOperand(1), SBGep0);
  Ctx.getTracker().stop();
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
  sandboxir::Context Ctx(C);
  auto *SBF = Ctx.createFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctx.getTracker().start(SBBB);
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
  Ctx.getTracker().revert();
  EXPECT_NE(SBSt->getOperand(0), SBLd);
  EXPECT_EQ(SBSt->getOperand(1), SBGep0);
  EXPECT_EQ(SBLd->getOperand(0), SBGep0);
  Ctx.getTracker().stop();
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
  sandboxir::Context Ctx(C);

  auto *SBF = Ctx.createFunction(&F);
  auto *SBBB = &*SBF->begin();
  Ctx.getTracker().start(SBBB);
  auto It = SBBB->begin();
  auto *SBAdd0 = &*It++;
  auto *SBAdd1 = &*It++;
  auto *SBRet = &*It++;
  SBAdd1->eraseFromParent();
  Ctx.getTracker().revert();
  It = SBBB->begin();
  EXPECT_EQ(&*It++, SBAdd0);
  EXPECT_EQ(&*It++, SBAdd1);
  EXPECT_EQ(&*It++, SBRet);

  // Check revert removeFromParent().
  SBAdd0->removeFromParent();
  EXPECT_EQ(&*SBBB->begin(), SBAdd1);
  Ctx.getTracker().revert();
  It = SBBB->begin();
  EXPECT_EQ(&*It++, SBAdd0);
  EXPECT_EQ(&*It++, SBAdd1);
  EXPECT_EQ(SBAdd1->getOperand(0), SBAdd0);
  EXPECT_EQ(SBAdd1->getOperand(1), SBAdd0);
  EXPECT_EQ(&*It++, SBRet);

  Ctx.getTracker().stop();
}

TEST_F(SandboxIRTrackerTest, CreateInstrAndRevert) {
  parseIR(C, R"IR(
define void @foo(i32 %v1, ptr %ptr) {
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *SBF = Ctx.createFunction(&F);
  BasicBlock *BB = &*F.begin();
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  sandboxir::Instruction *SBRet = &*It++;

  // Create an instruction and check the changes.
  auto *Val = SBF->getArg(0);
  auto *Ptr = SBF->getArg(1);
  Ctx.getTracker().start(SBBB);
  auto *NewSBI = sandboxir::StoreInst::create(Val, Ptr, /*Align=*/std::nullopt,
                                              SBRet, Ctx);
  (void)NewSBI;
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
}

TEST_F(SandboxIRTrackerTest, EraseInstrAndRevert) {
  parseIR(C, R"IR(
define void @foo(i32 %v1) {
  %add0 = add i32 %v1, %v1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *SBF = Ctx.createFunction(&F);
  BasicBlock *BB = &*F.begin();
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  sandboxir::Instruction *SBAdd0 = &*It++;

  // Create an instruction and check the changes.
  Ctx.getTracker().start(SBBB);
  SBAdd0->eraseFromParent();
  // Check the changes appended by create().
  EXPECT_EQ(Ctx.getTracker().size(), 1u);
#ifndef NDEBUG
  unsigned Idx = 0;
  EXPECT_TRUE(
      isa<sandboxir::EraseFromParent>(Ctx.getTracker().getChange(Idx++)));
#endif

  Ctx.getTracker().revert();
  // Check that revert() works.
  EXPECT_EQ(BB->size(), 2u);
  EXPECT_EQ(&*SBBB->begin(), SBAdd0);
  EXPECT_EQ(SBAdd0->getOperand(0), SBF->getArg(0));
  EXPECT_EQ(SBAdd0->getOperand(1), SBF->getArg(0));
}

TEST_F(SandboxIRTrackerTest, EraseCallbacks) {
  parseIR(C, R"IR(
define void @foo(i32 %v1, ptr %ptr) {
  %add0 = add i32 %v1, %v1
  %add1 = add i32 %add0, %add0
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *SBF = Ctx.createFunction(&F);
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  sandboxir::Instruction *SBAdd0 = &*It++;
  sandboxir::Instruction *SBAdd1 = &*It++;
  sandboxir::Instruction *SBRet = &*It++;

  // Check that we get callbacks when we erase a sandboxir::Instruction
  SmallVector<sandboxir::Instruction *> ErasedIRInstrs;
  auto *CB = Ctx.registerRemoveInstrCallback(
      [&ErasedIRInstrs](sandboxir::Instruction *SBI) {
        ErasedIRInstrs.push_back(SBI);
      });
  Ctx.getTracker().start(SBBB);

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

  // Check that we get callbacks on tracker revert().
  Ctx.getTracker().revert();
  ASSERT_EQ(ErasedIRInstrs.size(), 1u);
  EXPECT_EQ(ErasedIRInstrs[0], NewSBI);
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
  sandboxir::Context Ctx(C);

  BasicBlock *BB = &*F.begin();
  auto *SBF = Ctx.createFunction(&F);
  auto *SBBB = cast<sandboxir::BasicBlock>(Ctx.getValue(BB));
  auto It = SBBB->begin();
  sandboxir::Instruction *SBAdd0 = &*It++;
  sandboxir::Instruction *SBAdd1 = &*It++;
  (void)SBAdd1;
  sandboxir::Instruction *SBRet = &*It++;

  // Check that we get callbacks on tracker revert().
  SmallVector<sandboxir::Instruction *> InsertedInstrs;
  Ctx.registerInsertInstrCallback(
      [&InsertedInstrs](sandboxir::Instruction *SBI) {
        InsertedInstrs.push_back(SBI);
      });
  sandboxir::Argument *Ptr = SBF->getArg(1);
  Ctx.getTracker().start(SBBB);
  auto *NewSBI = sandboxir::StoreInst::create(
      SBAdd0, Ptr, /*Align=*/std::nullopt, SBRet, Ctx);
  ASSERT_EQ(InsertedInstrs.size(), 1u);
  EXPECT_EQ(InsertedInstrs[0], NewSBI);
  Ctx.getTracker().accept();
}

TEST_F(SandboxIRTrackerTest, BranchInst) {
  parseIR(C, R"IR(
define void @foo(i1 %cond0, i1 %cond1) {
bb0:
  br i1 %cond0, label %bb0, label %bb1
bb1:
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(&LLVMF);
  auto *Cond0 = F->getArg(0);
  auto *Cond1 = F->getArg(1);
  auto *BB0 = Ctx.getBasicBlock(getBasicBlockByName(LLVMF, "bb0"));
  auto *BB1 = Ctx.getBasicBlock(getBasicBlockByName(LLVMF, "bb1"));
  auto &Tracker = Ctx.getTracker();
  Tracker.start(BB0);
  auto It = BB0->begin();
  auto *Br0 = cast<sandboxir::BranchInst>(&*It++);
  // Check setCondition()
  EXPECT_EQ(Br0->getCondition(), Cond0);
  Br0->setCondition(Cond1);
  EXPECT_EQ(Br0->getCondition(), Cond1);
  Tracker.revert();
  EXPECT_EQ(Br0->getCondition(), Cond0);

  // Check setSuccessor()
  Tracker.start(BB0);
  EXPECT_EQ(Br0->getSuccessor(0), BB0);
  EXPECT_EQ(Br0->getSuccessor(1), BB1);
  Br0->setSuccessor(0, BB1);
  EXPECT_EQ(Br0->getSuccessor(0), BB1);
  EXPECT_EQ(Br0->getSuccessor(1), BB1);
  Tracker.revert();
  EXPECT_EQ(Br0->getSuccessor(0), BB0);
  EXPECT_EQ(Br0->getSuccessor(1), BB1);

  // Check swapSuccessors()
  Tracker.start(BB0);
  Br0->swapSuccessors();
  EXPECT_EQ(Br0->getSuccessor(0), BB1);
  EXPECT_EQ(Br0->getSuccessor(1), BB0);
  Tracker.revert();
  EXPECT_EQ(Br0->getSuccessor(0), BB0);
  EXPECT_EQ(Br0->getSuccessor(1), BB1);

  Ctx.getTracker().stop();
}

TEST_F(SandboxIRTrackerTest, UseSet) {
  parseIR(C, R"IR(
define void @foo(i8 %arg0, i8 %arg1) {
  %add = add i8 %arg0, %arg1
  ret void
}
)IR");
  llvm::Function &LLVMF = *M->getFunction("foo");
  llvm::BasicBlock *LLVMBB = &*LLVMF.begin();
  llvm::Instruction *LLVMI = &*LLVMBB->begin();
  llvm::Argument *LLVMArg0 = LLVMF.getArg(0);
  llvm::Argument *LLVMArg1 = LLVMF.getArg(1);
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(&LLVMF);
  auto *Arg0 = F->getArg(0);
  auto *Arg1 = F->getArg(1);
  auto *BB = Ctx.getBasicBlock(LLVMBB);
  auto *Add = &*BB->begin();
  auto &Tracker = Ctx.getTracker();
  Tracker.start(BB);
  sandboxir::Use Use0 = Add->getOperandUse(0);
  sandboxir::Use Use1 = Add->getOperandUse(1);
  EXPECT_EQ(Use0.get(), Arg0);
  EXPECT_EQ(Use1.get(), Arg1);
  Use0.set(Arg1);
  Use1.set(Arg0);
  EXPECT_EQ(Use0.get(), Arg1);
  EXPECT_EQ(Use1.get(), Arg0);
  EXPECT_EQ(LLVMI->getOperand(0), LLVMArg1);
  EXPECT_EQ(LLVMI->getOperand(1), LLVMArg0);

  Tracker.revert();
  EXPECT_EQ(Use0.get(), Arg0);
  EXPECT_EQ(Use1.get(), Arg1);
  EXPECT_EQ(LLVMI->getOperand(0), LLVMArg0);
  EXPECT_EQ(LLVMI->getOperand(1), LLVMArg1);

  // Check Use::operator=(sandboxir::Value *)
  Tracker.start(BB);
  Use0 = Arg1;
  Use1 = Arg0;
  EXPECT_EQ(Use0.get(), Arg1);
  EXPECT_EQ(Use1.get(), Arg0);
  EXPECT_EQ(LLVMI->getOperand(0), LLVMArg1);
  EXPECT_EQ(LLVMI->getOperand(1), LLVMArg0);

  Tracker.revert();
  EXPECT_EQ(Use0.get(), Arg0);
  EXPECT_EQ(Use1.get(), Arg1);
  EXPECT_EQ(LLVMI->getOperand(0), LLVMArg0);
  EXPECT_EQ(LLVMI->getOperand(1), LLVMArg1);

  Ctx.getTracker().stop();
}

TEST_F(SandboxIRTrackerTest, UseSwap) {
  parseIR(C, R"IR(
define void @foo(i8 %arg0, i8 %arg1, i8 %arg2, i8 %arg3) {
  %add0 = add i8 %arg0, %arg1
  %add1 = add i8 %arg2, %arg3
  ret void
}
)IR");
  llvm::Function &LLVMF = *M->getFunction("foo");
  llvm::BasicBlock *LLVMBB = &*LLVMF.begin();
  auto LLVMIt = LLVMBB->begin();
  llvm::Instruction *LLVMI0 = &*LLVMIt++;
  llvm::Instruction *LLVMI1 = &*LLVMIt++;
  (void)LLVMI1;
  llvm::Argument *LLVMArg0 = LLVMF.getArg(0);
  llvm::Argument *LLVMArg1 = LLVMF.getArg(1);
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(&LLVMF);
  int ArgIdx = 0;
  auto *Arg0 = F->getArg(ArgIdx++);
  auto *Arg1 = F->getArg(ArgIdx++);
  auto *Arg2 = F->getArg(ArgIdx++);
  auto *Arg3 = F->getArg(ArgIdx++);
  (void)Arg3;
  auto *BB = Ctx.getBasicBlock(LLVMBB);
  auto It = BB->begin();
  auto *Add0 = &*It++;
  auto *Add1 = &*It++;
  auto &Tracker = Ctx.getTracker();
  Tracker.start(BB);
  sandboxir::Use Use0 = Add0->getOperandUse(0);
  sandboxir::Use Use1 = Add0->getOperandUse(1);
  EXPECT_EQ(Use0.get(), Arg0);
  EXPECT_EQ(Use1.get(), Arg1);
  Use0.swap(Use1);
  EXPECT_EQ(Use0.get(), Arg1);
  EXPECT_EQ(Use1.get(), Arg0);
  EXPECT_EQ(LLVMI0->getOperand(0), LLVMArg1);
  EXPECT_EQ(LLVMI0->getOperand(1), LLVMArg0);

  Tracker.revert();
  EXPECT_EQ(Use0.get(), Arg0);
  EXPECT_EQ(Use1.get(), Arg1);
  EXPECT_EQ(LLVMI0->getOperand(0), LLVMArg0);
  EXPECT_EQ(LLVMI0->getOperand(1), LLVMArg1);

  // Check reverting Uses with different users.
  sandboxir::Use Use2 = Add1->getOperandUse(0);
  Tracker.start(BB);
  Use0.swap(Use2);
  EXPECT_EQ(Use0.get(), Arg2);
  EXPECT_EQ(Use0.getUser(), Add0);
  EXPECT_EQ(Use2.get(), Arg0);
  EXPECT_EQ(Use2.getUser(), Add1);

  Tracker.revert();
  EXPECT_EQ(Use0.get(), Arg0);
  EXPECT_EQ(Use0.getUser(), Add0);
  EXPECT_EQ(Use2.get(), Arg2);
  EXPECT_EQ(Use2.getUser(), Add1);

  Ctx.getTracker().stop();
}

TEST_F(SandboxIRTrackerTest, BinaryOperator_swapOperands) {
  parseIR(C, R"IR(
define i32 @foo(i32 %v0, i32 %v1) {
  %add0 = add i32 %v0, %v1
  ret i32 %add0
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *LLVMBB = &*LLVMF.begin();
  auto LLVMBBIt = LLVMBB->begin();
  Instruction *LLVMI0 = &*LLVMBBIt++;
  auto *LLVMV0 = LLVMF.getArg(0);
  auto *LLVMV1 = LLVMF.getArg(1);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto &BB = *Ctx.getBasicBlock(LLVMBB);
  auto *V0 = F.getArg(0);
  auto *V1 = F.getArg(1);
  auto It = BB.begin();
  auto *I0 = cast<sandboxir::BinaryOperator>(&*It++);

  auto &Tracker = Ctx.getTracker();
  Tracker.start(&BB);

  EXPECT_EQ(I0->getOperand(0), V0);
  EXPECT_EQ(I0->getOperand(1), V1);
  I0->swapOperands();
  EXPECT_EQ(I0->getOperand(0), V1);
  EXPECT_EQ(I0->getOperand(1), V0);
  EXPECT_EQ(LLVMI0->getOperand(0), LLVMV1);
  EXPECT_EQ(LLVMI0->getOperand(1), LLVMV0);

  Tracker.revert();
  EXPECT_EQ(I0->getOperand(0), V0);
  EXPECT_EQ(I0->getOperand(1), V1);
  EXPECT_EQ(LLVMI0->getOperand(0), LLVMV0);
  EXPECT_EQ(LLVMI0->getOperand(1), LLVMV1);

  Tracker.stop();
}
