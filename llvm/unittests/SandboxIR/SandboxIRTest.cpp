//===- SandboxIRTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

struct SandboxIRTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

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

TEST_F(SandboxIRTest, IteratorsSimple) {
  parseIR(C, R"IR(
define void @foo(i32 %v1) {
  %add0 = add i32 %v1, %v1
  %add1 = add i32 %add0, %add0
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *BB = &*F.begin();
  auto BBIt = BB->begin();
  Instruction *Add0 = &*BBIt++;
  Instruction *Add1 = &*BBIt++;
  Instruction *Ret = &*BBIt++;
  sandboxir::BasicBlock &SBBB = *Ctx.createBasicBlock(BB);

  auto *SBAdd0 = Ctx.getValue(Add0);
  auto *SBAdd1 = Ctx.getValue(Add1);
  auto *SBRet = Ctx.getValue(Ret);

  auto It = SBBB.begin();
  EXPECT_EQ(&*It++, SBAdd0);
  EXPECT_EQ(&*It++, SBAdd1);
  EXPECT_EQ(&*It++, SBRet);
  EXPECT_EQ(It, SBBB.end());
#ifndef NDEBUG
  EXPECT_DEATH(++It, "Already.*");
#endif
  --It;
  EXPECT_EQ(&*It--, Ctx.getValue(Ret));
  EXPECT_EQ(&*It--, Ctx.getValue(Add1));
  EXPECT_EQ(&*It, Ctx.getValue(Add0));
  EXPECT_EQ(It, SBBB.begin());
#ifndef NDEBUG
  EXPECT_DEATH(--It, "Already.*");
  EXPECT_DEATH(It--, "Already.*");
#endif

  {
    // bidirectional: +1 and -1
    auto It = SBBB.begin();
    std::advance(It, 1);
    EXPECT_EQ(&*It, SBAdd1);
    std::advance(It, -1);
    EXPECT_EQ(&*It, SBAdd0);
  }
}

TEST_F(SandboxIRTest, Use_Simple) {
  parseIR(C, R"IR(
define i32 @foo(i32 %v0, i32 %v1) {
  %add0 = add i32 %v0, %v1
  %add1 = add i32 %add0, %add0
  ret i32 %add0
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

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
  auto *SBI0 = &*It++;
  auto *SBI1 = &*It++;
  (void)SBI1;
  auto *SBRet = &*It++;

  SmallVector<sandboxir::Argument *> Args{SBArg0, SBArg1};
  unsigned OpIdx = 0;
  for (sandboxir::Use Use : SBI0->operands()) {
    EXPECT_EQ(Use.getOperandNo(), OpIdx);
    EXPECT_EQ(Use.get(), Args[OpIdx]);
    ++OpIdx;
  }
  EXPECT_EQ(OpIdx, 2u);

  // Check sandboxir::Use iterators when the value has no uses.
  unsigned Cnt = 0;
  for (auto It = SBRet->use_begin(), ItE = SBRet->use_end(); It != ItE; ++It)
    ++Cnt;
  EXPECT_EQ(Cnt, 0u);
}

TEST_F(SandboxIRTest, Use_set) {
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

  auto Use0 = I0->getOperandUse(0);
  EXPECT_EQ(Use0.get(), V0);
  EXPECT_EQ(LLVMI0->getOperand(0), LLVMV0);
  auto Use1 = I0->getOperandUse(1);
  EXPECT_EQ(Use1.get(), V1);
  EXPECT_EQ(LLVMI0->getOperand(1), LLVMV1);

  Use0.set(V1);
  EXPECT_EQ(Use0.get(), V1);
  EXPECT_EQ(I0->getOperand(0), V1);
  EXPECT_EQ(I0->getOperand(1), V1);
  EXPECT_EQ(LLVMI0->getOperand(0), LLVMV1);
  EXPECT_EQ(LLVMI0->getOperand(1), LLVMV1);

  // Check Use::operator=(sandboxir::Value *)
  Use0 = V0;
  EXPECT_EQ(Use0.get(), V0);
  EXPECT_EQ(I0->getOperand(0), V0);
  EXPECT_EQ(I0->getOperand(1), V1);
  EXPECT_EQ(LLVMI0->getOperand(0), LLVMV0);
  EXPECT_EQ(LLVMI0->getOperand(1), LLVMV1);
}

TEST_F(SandboxIRTest, Use_swap) {
  parseIR(C, R"IR(
define i32 @foo(i32 %v0, i32 %v1, i32 %v2 ,i32 %v3) {
  %add0 = add i32 %v0, %v1
  %add1 = add i32 %v2, %v3
  ret i32 %add0
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *LLVMBB = &*LLVMF.begin();
  auto LLVMBBIt = LLVMBB->begin();
  Instruction *LLVMI0 = &*LLVMBBIt++;
  Instruction *LLVMI1 = &*LLVMBBIt++;
  int ArgIdx = 0;
  auto *LLVMV0 = LLVMF.getArg(ArgIdx++);
  auto *LLVMV1 = LLVMF.getArg(ArgIdx++);
  auto *LLVMV2 = LLVMF.getArg(ArgIdx++);
  auto *LLVMV3 = LLVMF.getArg(ArgIdx++);
  (void)LLVMV3;
  auto &F = *Ctx.createFunction(&LLVMF);
  auto &BB = *Ctx.getBasicBlock(LLVMBB);
  ArgIdx = 0;
  auto *V0 = F.getArg(ArgIdx++);
  auto *V1 = F.getArg(ArgIdx++);
  auto *V2 = F.getArg(ArgIdx++);
  auto It = BB.begin();
  auto *I0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *I1 = cast<sandboxir::BinaryOperator>(&*It++);

  // Check swap with common user.
  auto Use0 = I0->getOperandUse(0);
  EXPECT_EQ(Use0.get(), V0);
  EXPECT_EQ(LLVMI0->getOperand(0), LLVMV0);
  auto Use1 = I0->getOperandUse(1);
  EXPECT_EQ(Use1.get(), V1);
  EXPECT_EQ(LLVMI0->getOperand(1), LLVMV1);

  Use0.swap(Use1);
  EXPECT_EQ(I0->getOperand(0), V1);
  EXPECT_EQ(I0->getOperand(1), V0);
  EXPECT_EQ(LLVMI0->getOperand(0), LLVMV1);
  EXPECT_EQ(LLVMI0->getOperand(1), LLVMV0);

  Use0.swap(Use1);
  EXPECT_EQ(I0->getOperand(0), V0);
  EXPECT_EQ(I0->getOperand(1), V1);
  EXPECT_EQ(LLVMI0->getOperand(0), LLVMV0);
  EXPECT_EQ(LLVMI0->getOperand(1), LLVMV1);

  llvm::Use &LLVMUse0 = LLVMI0->getOperandUse(0);
  llvm::Use &LLVMUse2 = LLVMI1->getOperandUse(0);

  // Check swap of Uses that point to different users.
  auto Use2 = I1->getOperandUse(0);
  Use0.swap(Use2);
  EXPECT_EQ(LLVMUse0.get(), LLVMV2);
  EXPECT_EQ(LLVMUse2.get(), LLVMV0);
  EXPECT_EQ(LLVMUse0.getUser(), LLVMI0);
  EXPECT_EQ(LLVMUse2.getUser(), LLVMI1);
  EXPECT_EQ(I0->getOperand(0), V2);
  EXPECT_EQ(I1->getOperand(0), V0);
  EXPECT_EQ(Use0.get(), V2);
  EXPECT_EQ(Use2.get(), V0);
  EXPECT_EQ(Use0.getUser(), I0);
  EXPECT_EQ(Use2.getUser(), I1);
}

TEST_F(SandboxIRTest, UseIterator_Simple) {
  parseIR(C, R"IR(
define i32 @foo(i32 %v0, i32 %v1) {
  %add0 = add i32 %v0, %v1
  %add1 = add i32 %add0, %add0
  ret i32 %add0
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

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
  auto *SBI0 = &*It++;
  auto *SBI1 = &*It++;

  sandboxir::OperandUseIterator UseIt0 = SBI0->op_begin();
  // Check operator==
  EXPECT_TRUE(UseIt0 == SBI0->op_begin());
  // Check SBUse
  sandboxir::Use Use0 = *UseIt0;
  EXPECT_EQ(Use0.get(), SBArg0);
  EXPECT_EQ(Use0.getUser(), SBI0);
  ++UseIt0;
  Use0 = *UseIt0;
  EXPECT_EQ(Use0.get(), SBArg1);
  EXPECT_EQ(Use0.getUser(), SBI0);
  ++UseIt0;
  EXPECT_EQ(UseIt0, SBI0->op_end());

  sandboxir::OperandUseIterator UseIt1 = SBI1->op_begin();
  EXPECT_TRUE(UseIt1 != UseIt0);
  sandboxir::Use Use1 = *UseIt1;
  EXPECT_EQ(Use1.get(), SBI0);
  EXPECT_EQ(Use1.getUser(), SBI1);
  ++UseIt1;
  Use1 = *UseIt1;
  EXPECT_EQ(Use1.get(), SBI0);
  EXPECT_EQ(Use1.getUser(), SBI1);
  ++UseIt1;
}

TEST_F(SandboxIRTest, UserUse_Simple) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1) {
  %add0 = add i32 %v0, %v1
  %add1 = add i32 %add0, %add0
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *BB = &*F.begin();
  auto BBIt = BB->begin();
  Instruction *I0 = &*BBIt++;
  (void)I0;
  Instruction *I1 = &*BBIt++;
  (void)I1;
  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();
  auto *SBI0 = &*It++;
  auto *SBI1 = &*It++;

  sandboxir::Value::use_iterator UseIt0 = SBI0->use_begin();
  const sandboxir::Use &Use = *UseIt0;
  EXPECT_EQ(Use.getUser(), SBI1);
  EXPECT_EQ(Use.get(), SBI0);
  ++UseIt0;
  const sandboxir::Use &Use1 = *UseIt0;
  EXPECT_EQ(Use1.getUser(), SBI1);
  EXPECT_EQ(Use1.get(), SBI0);
  ++UseIt0;
  EXPECT_EQ(UseIt0, SBI0->use_end());
}

TEST_F(SandboxIRTest, HasNUsersOrMore) {
  parseIR(C, R"IR(
define float @foo(ptr %ptr) {
  %ld0 = load float, ptr %ptr
  %ld1 = load float, ptr %ptr
  %add0 = fadd float %ld0, %ld0
  %add1 = fadd float %ld1, %ld1
  ret float %ld1
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *SBF = Ctx.createFunction(&F);
  sandboxir::BasicBlock &BB = *SBF->begin();
  auto It = BB.begin();
  auto *L0 = &*It++;
  auto *L1 = &*It++;
  auto *Add0 = &*It++;
  (void)Add0;
  auto *Add1 = &*It++;
  (void)Add1;
  EXPECT_TRUE(L0->hasNUsersOrMore(1));
  EXPECT_FALSE(L0->hasNUsersOrMore(2));
  EXPECT_TRUE(L1->hasNUsersOrMore(2));
  EXPECT_FALSE(L1->hasNUsersOrMore(3));
}

TEST_F(SandboxIRTest, IsMem) {
  parseIR(C, R"IR(
declare void @bar()
declare void @llvm.sideeffect()
define void @foo(i8 %v, ptr %ptr) {
  %ld = load i8, ptr %ptr
  store i8 %v, ptr %ptr
  call void @bar()
  call void @llvm.sideeffect()
  call void @llvm.pseudoprobe(i64 0, i64 0, i32 0, i64 0)
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &SBF = *Ctx.createFunction(&F);
  auto &BB = *SBF.begin();
  auto It = BB.begin();
  auto *I0 = &*It++;
  auto *I1 = &*It++;
  auto *I2 = &*It++;
  auto *I3 = &*It++;
  auto *I4 = &*It++;
  EXPECT_TRUE(I0->isMemInst());
  EXPECT_TRUE(I1->isMemInst());
  EXPECT_TRUE(I2->isMemInst());
  EXPECT_FALSE(I3->isMemInst());
  EXPECT_FALSE(I4->isMemInst());
}

TEST_F(SandboxIRTest, SBFunction_Simple) {
  parseIR(C, R"IR(
define void @foo(i8 %arg1, i32 %arg2) {
bb0:
  br label %bb1

bb1:
  br label %bb2

bb2:
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *BB0 = getBasicBlockByName(F, "bb0");
  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  BasicBlock *BB2 = getBasicBlockByName(F, "bb2");

  auto *SBF = Ctx.createFunction(&F);
  EXPECT_EQ(SBF->arg_size(), 2u);
  EXPECT_EQ(SBF->getArg(0), Ctx.getValue(F.getArg(0)));
  EXPECT_EQ(SBF->getArg(1), Ctx.getValue(F.getArg(1)));
  auto *SBBB0 = Ctx.getBasicBlock(BB0);
  auto *SBBB1 = Ctx.getBasicBlock(BB1);
  auto *SBBB2 = Ctx.getBasicBlock(BB2);
  SmallVector<sandboxir::BasicBlock *> SBBBs;
  for (auto &SBBB : *SBF)
    SBBBs.push_back(&SBBB);

  EXPECT_EQ(SBBBs[0], SBBB0);
  EXPECT_EQ(SBBBs[1], SBBB1);
  EXPECT_EQ(SBBBs[2], SBBB2);
}

TEST_F(SandboxIRTest, SBFunction_Args) {
  parseIR(C, R"IR(
define void @foo(i8 %arg0, i32 %arg1) {
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  unsigned Idx = 0;
  auto *Arg0 = F.getArg(Idx++);
  auto *Arg1 = F.getArg(Idx++);
  auto *SBF = Ctx.createFunction(&F);
  EXPECT_EQ(SBF->getArg(0), Ctx.getValue(Arg0));
  EXPECT_EQ(SBF->getArg(1), Ctx.getValue(Arg1));
  unsigned ArgIdx = 0;
  for (const sandboxir::Argument &Arg : SBF->args()) {
    EXPECT_EQ(&Arg, SBF->getArg(ArgIdx));
    ++ArgIdx;
  }
}

TEST_F(SandboxIRTest, SBFunction_detachFromLLVMIR) {
  parseIR(C, R"IR(
define void @foo() {
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *BB = &*F.begin();

  auto &SBF = *Ctx.createFunction(&F);
  auto *SBBB = cast<sandboxir::BasicBlock>(Ctx.getValue(BB));
  (void)SBBB;
  EXPECT_NE(Ctx.getNumValues(), 0u);
  SBF.detachFromLLVMIR();
  EXPECT_EQ(Ctx.getNumValues(), 0u);
}

TEST_F(SandboxIRTest, PrevNode_WhenPrevIsDetached) {
  parseIR(C, R"IR(
define void @foo(i32 %v1) {
  %add0 = add i32 %v1, %v1
  %add1 = add i32 %add0, %add0
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  BasicBlock *BB = &*F.begin();

  sandboxir::BasicBlock &SBBB = *Ctx.createBasicBlock(BB);
  auto It = SBBB.begin();
  auto *SBAdd0 = &*It++;
  auto *SBAdd1 = &*It++;
  SBAdd0->removeFromParent();
  It = SBAdd1->getIterator();
#ifndef NDEBUG
  EXPECT_DEATH(--It, ".*begin.*");
#endif
  EXPECT_EQ(SBAdd1->getPrevNode(), nullptr);

  SBAdd0->insertBefore(SBAdd1);
  EXPECT_EQ(SBAdd1->getPrevNode(), SBAdd0);
}

TEST_F(SandboxIRTest, PrevNode_WhenSBBBisNotComplete) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1) {
  %add1 = add i32 %v1, %v1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  BasicBlock *BB = &*F.begin();

  sandboxir::BasicBlock &SBBB = *Ctx.createBasicBlock(BB);
  auto It = SBBB.begin();
  auto *SBAdd1 = &*It++;

  Argument *Arg0 = F.getArg(0);
  Instruction *Add1 = &*BB->begin();
  Instruction *Add0 =
      BinaryOperator::Create(Instruction::Add, Arg0, Arg0, "Add0", Add1);
  (void)Add0;
  EXPECT_FALSE(SBAdd1->getIterator().atBegin());
  EXPECT_EQ(SBAdd1->getPrevNode(), nullptr);
}

TEST_F(SandboxIRTest, SBBasicBlock_detachFromLLVMIR) {
  parseIR(C, R"IR(
define void @foo() {
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *BB = &*F.begin();
  Instruction *Ret = BB->getTerminator();

  auto &SBBB = *Ctx.createBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  auto *SBI = cast<sandboxir::Instruction>(Ctx.getValue(Ret));
  (void)SBI;
  SBBB.detachFromLLVMIR();
  EXPECT_EQ(Ctx.getNumValues(), 0u);
}

// Check that SandboxIR Instructions in sandboxir::BasicBlock get erased in
// the right order when context goes out of scope
TEST_F(SandboxIRTest, ContextDestruction) {
  parseIR(C, R"IR(
define i32 @foo(i32 %v) {
  %add0 = add i32 %v, 1
  %add1 = add i32 %add0, 1
  %add2 = add i32 %add1, 1
  %add3 = add i32 %add2, 1
  ret i32 %add3
}
)IR");
  Function &F = *M->getFunction("foo");
  {
    sandboxir::Context Ctx(C);
    BasicBlock *BB = &*F.begin();
    Instruction *Ret = BB->getTerminator();
    (void)Ret;
    auto &SBBB = *Ctx.createBasicBlock(BB);
    (void)SBBB;
#ifndef NDEBUG
    SBBB.verify();
#endif
  }
}

/// Check that sandboxir::BasicBlock is registered with LLVMValueToValueMap
TEST_F(SandboxIRTest, SBBBLLVMValueToValueMap) {
  parseIR(C, R"IR(
define i32 @foo(i32 %v) {
bb0:
  br label %bb1
bb1:
  ret i32 %v
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *BB0 = getBasicBlockByName(F, "bb0");
  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  auto &SBBB0 = *Ctx.createBasicBlock(BB0);
  EXPECT_EQ(&SBBB0, Ctx.getBasicBlock(BB0));
#ifndef NDEBUG
  SBBB0.verify();
#endif
  auto &SBBB1 = *Ctx.createBasicBlock(BB1);
#ifndef NDEBUG
  SBBB1.verify();
#endif
  EXPECT_EQ(&SBBB1, Ctx.getBasicBlock(BB1));
}

// Check sandboxir::Instruction::getParent()
TEST_F(SandboxIRTest, SBInstructionGetParent) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1) {
bb0:
  %add0 = add i32 %v0, %v0
  br label %bb1
bb1:
  %add1 = add i32 %v1, %v1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *BB0 = getBasicBlockByName(F, "bb0");
  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  auto *SBBB0 = Ctx.createBasicBlock(BB0);
  auto *SBI0 = &*SBBB0->begin();
  EXPECT_EQ(SBI0->getParent(), SBBB0);
  auto *SBBB1 = Ctx.createBasicBlock(BB1);
  auto *SBI1 = &*SBBB1->begin();
  EXPECT_EQ(SBI1->getParent(), SBBB1);

  SBI0->moveBefore(SBI1);
  EXPECT_EQ(SBI0->getParent(), SBI1->getParent());
}

// Check sandboxir::BasicBlock::getParent()
TEST_F(SandboxIRTest, SBBasicBlockGetParent) {
  parseIR(C, R"IR(
define void @foo() {
bb0:
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *SBF = Ctx.createFunction(&F);
  BasicBlock *BB0 = getBasicBlockByName(F, "bb0");
  auto *SBBB0 = Ctx.getBasicBlock(BB0);
  EXPECT_EQ(SBBB0->getParent(), SBF);
}

TEST_F(SandboxIRTest, PHIs) {
  parseIR(C, R"IR(
define void @foo(float %v, ptr noalias %ptr1, ptr noalias %ptr2) {
bb1:
  %ld1 = load float, ptr %ptr1
  br label %bb1

bb2:
  %phi = phi float [ %ld1, %bb1 ], [ 0.0, %bb2 ]
  %add = fadd float %phi, %ld1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  BasicBlock *BB2 = getBasicBlockByName(F, "bb2");
  auto &SBBB1 = *Ctx.createBasicBlock(BB1);
  (void)SBBB1;
#ifndef NDEBUG
  SBBB1.verify();
#endif
  auto &SBBB2 = *Ctx.createBasicBlock(BB2);
  (void)SBBB2;
#ifndef NDEBUG
  SBBB2.verify();
#endif
  auto It = SBBB2.begin();
  auto *PHI = cast<sandboxir::PHINode>(&*It++);
  (void)PHI;
  auto *BinOp = cast<sandboxir::BinaryOperator>(&*It++);
  (void)BinOp;
}

TEST_F(SandboxIRTest, BranchInstLabelOperands) {
  parseIR(C, R"IR(
define void @foo(float %v, ptr noalias %ptr1) {
bb1:
  %ld1 = load float, ptr %ptr1
  %cmp = fcmp oeq float %ld1, 0.0
  br i1 %cmp, label %bb2, label %bb3

bb2:
  br label %bb2

bb3:
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  BasicBlock *BB2 = getBasicBlockByName(F, "bb2");
  BasicBlock *BB3 = getBasicBlockByName(F, "bb3");

  auto &SBBB1 = *Ctx.createBasicBlock(BB1);
  (void)SBBB1;
#ifndef NDEBUG
  SBBB1.verify();
#endif
  auto &SBBB2 = *Ctx.createBasicBlock(BB2);
  (void)SBBB2;
#ifndef NDEBUG
  SBBB2.verify();
#endif
  auto &SBBB3 = *Ctx.createBasicBlock(BB3);
  (void)SBBB3;
#ifndef NDEBUG
  SBBB3.verify();
#endif
}

TEST_F(SandboxIRTest, InvokeInstLabelOperands) {
  parseIR(C, R"IR(
declare void @bar()
define void @foo(float %v, ptr noalias %ptr1) {
bb1:
  %ld1 = load float, ptr %ptr1
  invoke void @bar() to label %bb2 unwind label %bb3
  %cmp = fcmp oeq float %ld1, 0.0
  br i1 %cmp, label %bb2, label %bb3

bb2:
  br label %bb2

bb3:
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");

  auto &SBBB1 = *Ctx.createBasicBlock(BB1);
  (void)SBBB1;
#ifndef NDEBUG
  SBBB1.verify();
#endif
}

TEST_F(SandboxIRTest, BinaryOperator_Create) {
  parseIR(C, R"IR(
define void @foo(i8 %val0, i8 %val1) {
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto &SBF = *Ctx.createFunction(&F);
  auto *Arg0 = SBF.getArg(0);
  auto *Arg1 = SBF.getArg(1);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();
  sandboxir::Instruction *Ret = &*It++;
  auto *SBAdd =
      cast<sandboxir::BinaryOperator>(sandboxir::BinaryOperator::create(
          sandboxir::Instruction::Opcode::Add, Arg0, Arg1, Ret, Ctx, "Test"));
  EXPECT_EQ(SBAdd->getOpcode(), sandboxir::Instruction::Opcode::Add);
}

TEST_F(SandboxIRTest, CheckInstructionTypes) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld = load i32, ptr %ptr
  store i32 %ld, ptr %ptr
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *BB = &*F.begin();

  auto It = BB->begin();
  auto *Ld = &*It++;
  auto *St = &*It++;
  auto &SBBB = *Ctx.createBasicBlock(BB);
  (void)SBBB;
#ifndef NDEBUG
  SBBB.verify();
#endif
  auto *SBL = Ctx.getValue(Ld);
  auto *SBS = Ctx.getValue(St);
  auto *SBPtr = Ctx.getOrCreateValue(F.getArg(0));

  EXPECT_TRUE(isa<sandboxir::LoadInst>(SBL));
  EXPECT_EQ(cast<sandboxir::LoadInst>(SBL)->getPointerOperand(), SBPtr);
  EXPECT_TRUE(isa<sandboxir::StoreInst>(SBS));
  EXPECT_EQ(cast<sandboxir::StoreInst>(SBS)->getValueOperand(), SBL);
  EXPECT_EQ(cast<sandboxir::StoreInst>(SBS)->getPointerOperand(), SBPtr);
}

// Checks detaching an sandboxir::BasicBlock from its underlying BB.
TEST_F(SandboxIRTest, SBBasicBlockDestruction) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, <2 x i32> %v) {
  %add = add <2 x i32> %v, %v
  %extr0 = extractelement <2 x i32> %add, i32 0
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *BB = &*F.begin();
  auto It = BB->begin();
  auto *Add = &*It++;
  auto *Extr = &*It++;
  auto *Ret = &*It++;
  unsigned BBSize = BB->size();
  {
    auto &SBBB = *Ctx.createBasicBlock(BB);
#ifndef NDEBUG
    SBBB.verify();
#endif
    Ctx.getTracker().start(&SBBB);
    auto It = SBBB.begin();
    auto *SBAdd = Ctx.getValue(Add);
    EXPECT_EQ(&*It++, SBAdd);
    auto *SBExt = cast<sandboxir::ExtractElementInst>(Ctx.getValue(Extr));
    EXPECT_EQ(&*It++, SBExt);
  }
  // Check that BB is still intact.
  EXPECT_EQ(BBSize, BB->size());
  It = BB->begin();
  EXPECT_EQ(&*It++, Add);
  EXPECT_EQ(&*It++, Extr);
  EXPECT_EQ(&*It++, Ret);
  // Expect that clearing a BB does not track changes.
  EXPECT_TRUE(Ctx.getTracker().empty());
  Ctx.getTracker().accept();
}

TEST_F(SandboxIRTest, RAW_and_RUWIf_DiffBB) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
bb0:
  %ld0 = load float, ptr %ptr
  %ld1 = load float, ptr %ptr
  br label %bb1

bb1:
  store float %ld0, ptr %ptr
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *BB0 = getBasicBlockByName(F, "bb0");
  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  auto &SBBB0 = *Ctx.createBasicBlock(BB0);
  Ctx.getTracker().start(&SBBB0);
  auto It = BB0->begin();
  Instruction *Ld0 = &*It++;
  Instruction *Ld1 = &*It++;
  It = BB1->begin();
  Instruction *St0 = &*It++;
  auto *SBLd0 = cast<sandboxir::Instruction>(Ctx.getValue(Ld0));
  auto *SBLd1 = cast<sandboxir::Instruction>(Ctx.getValue(Ld1));
  auto DoRAWIf = [&]() {
    SBLd0->replaceUsesWithIf(SBLd1, [](sandboxir::Use Use) { return true; });
  };
  // The user is in BB1 but there is no SBBB1. Make sure it doesn't crash.
  DoRAWIf();
  // Now create the SBBB1 and try again.
  Ctx.createBasicBlock(BB1);
  DoRAWIf();

  EXPECT_EQ(St0->getOperand(0), Ld1);
  SBLd1->replaceAllUsesWith(SBLd0);
  EXPECT_EQ(St0->getOperand(0), Ld0);
  Ctx.getTracker().accept();
}

TEST_F(SandboxIRTest, SBConstant) {
  parseIR(C, R"IR(
define void @foo() {
  %add0 = add i32 42, 42
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *BB0 = &*F.begin();
  auto &SBBB = *Ctx.createBasicBlock(BB0);
  Ctx.getTracker().start(&SBBB);
  auto It = BB0->begin();
  Instruction *Add = &*It++;
  Instruction *Ret = &*It++;
  auto *C42 = cast<Constant>(Add->getOperand(0));
  sandboxir::Instruction *SBAdd =
      cast<sandboxir::Instruction>(Ctx.getValue(Add));
  auto *SBRet = cast<sandboxir::Instruction>(Ctx.getValue(Ret));
  (void)SBRet;
  sandboxir::Constant *SBC42 = cast<sandboxir::Constant>(SBAdd->getOperand(0));
  EXPECT_EQ(SBC42, SBAdd->getOperand(1));
  EXPECT_EQ(Ctx.getValue(C42), SBC42);
}

// Check that SandboxIR creation handles BlockAddress
TEST_F(SandboxIRTest, BlockAddress) {
  parseIR(C, R"IR(
define void @foo() {
bb0:
  %gep = getelementptr inbounds i8, ptr blockaddress(@foo, %bb1), i64 0
  br label %bb1
bb1:
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *SBF = Ctx.createFunction(&F);
  auto *BB0 = Ctx.getBasicBlock(getBasicBlockByName(F, "bb0"));
  auto *BB1 = Ctx.getBasicBlock(getBasicBlockByName(F, "bb1"));

  auto It = BB0->begin();
  auto *Gep = &*It++;
  auto *BlockAddress = cast<sandboxir::Constant>(Gep->getOperand(0));
  auto *FOp = BlockAddress->getOperand(0);
  EXPECT_EQ(FOp, SBF);
  auto *BBOp = BlockAddress->getOperand(1);
  EXPECT_EQ(BBOp, BB1);
#ifndef NDEBUG
  BB0->verify();
#endif
}

TEST_F(SandboxIRTest, BlockAddressWithMissingSBBBAndSBF) {
  parseIR(C, R"IR(
define void @foo() {
bb0:
  %gep = getelementptr inbounds i8, ptr blockaddress(@foo, %bb1), i64 0
  br label %bb1
bb1:
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *BB0 = Ctx.createBasicBlock(getBasicBlockByName(F, "bb0"));

  auto It = BB0->begin();
  auto *Gep = &*It++;
  auto *BlockAddress = cast<sandboxir::Constant>(Gep->getOperand(0));
  assert(isa<sandboxir::User>(BlockAddress) && "Constants are users");
  auto *FOp = BlockAddress->getOperand(0);
  EXPECT_NE(FOp, nullptr);
  auto *BBOp = BlockAddress->getOperand(1);
  EXPECT_EQ(BBOp, nullptr);
#ifndef NDEBUG
  BB0->verify();
#endif
}

TEST_F(SandboxIRTest, SelfReferencingType) {
  parseIR(C, R"IR(
@bar = global [1 x ptr] [ptr @bar]
define void @foo() {
  %a = getelementptr i8, ptr @bar, i64 0
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *SBF = Ctx.createFunction(&F);
  (void)SBF;
}

TEST_F(SandboxIRTest, SBFunctionIsaSBConstant) {
  parseIR(C, R"IR(
define void @bar() {
  ret void
}
@g = global [1 x ptr] [ptr @foo]
define void @foo() {
  %a = getelementptr i8, ptr @g, i64 0
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *SBF = Ctx.createFunction(&F);
  (void)SBF;
  auto *BB = &*F.begin();
  auto *GEP = &*BB->begin();
  auto *GepOpFn = GEP->getOperand(0);
  EXPECT_TRUE(isa<Constant>(GepOpFn)); // In LLVM IR a function is a constant
  auto *SBPtr = Ctx.getOrCreateValue(GepOpFn);
  EXPECT_TRUE(isa<sandboxir::Constant>(SBF));
  (void)SBPtr;
}

// Check that the operands/users are counted correctly.
//  I1
// /  \
// \  /
//  I2
TEST_F(SandboxIRTest, DuplicateUses) {
  parseIR(C, R"IR(
define void @foo(i8 %v) {
  %I1 = add i8 %v, %v
  %I2 = add i8 %I1, %I1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *SBF = Ctx.createFunction(&F);
  auto *BB = &*SBF->begin();
  auto It = BB->begin();
  auto *I1 = &*It++;
  auto *I2 = &*It++;
  unsigned CntI1Users = 0u;
  for (auto *User : I1->users()) {
    (void)User;
    ++CntI1Users;
  }
  EXPECT_EQ(CntI1Users, 2u);
  unsigned CntI2Operands = 0u;
  for (sandboxir::Value *Op : I2->operands()) {
    (void)Op;
    ++CntI2Operands;
  }
  EXPECT_EQ(CntI2Operands, 2u);
#ifndef NDEBUG
  BB->verify();
#endif
}

TEST_F(SandboxIRTest, CheckInsertAndRemoveInstrCallback) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %val) {
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  sandboxir::Argument *Ptr = SBF.getArg(0);
  sandboxir::Argument *Val = SBF.getArg(1);
  sandboxir::Instruction *Ret = &*SBBB.rbegin();
  SmallVector<sandboxir::Instruction *> Inserted;
  SmallVector<sandboxir::Instruction *> Removed;
  Ctx.registerInsertInstrCallback(
      [&Inserted](sandboxir::Instruction *SBI) { Inserted.push_back(SBI); });
  Ctx.registerRemoveInstrCallback(
      [&Removed](sandboxir::Instruction *SBI) { Removed.push_back(SBI); });

  auto *NewI =
      sandboxir::StoreInst::create(Val, Ptr, /*Align=*/std::nullopt, Ret, Ctx);
  EXPECT_EQ(Inserted.size(), 1u);
  EXPECT_EQ(Inserted[0], NewI);
  EXPECT_EQ(Removed.size(), 0u);

  Ret->eraseFromParent();
  EXPECT_EQ(Removed.size(), 1u);
  EXPECT_EQ(Removed[0], Ret);
  EXPECT_EQ(Inserted.size(), 1u);

  NewI->eraseFromParent();
  EXPECT_EQ(Removed.size(), 2u);
  EXPECT_EQ(Removed[1], NewI);
  EXPECT_EQ(Inserted.size(), 1u);
#ifndef NDEBUG
  SBBB.verify();
#endif
}

TEST_F(SandboxIRTest, CheckInsertAndRemovePerBBInstrCallback) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %val) {
bb0:
  br label %bb1
bb1:
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto &SBF = *Ctx.createFunction(&F);
  auto &BB0 = *Ctx.getBasicBlock(getBasicBlockByName(F, "bb0"));
  auto &BB1 = *Ctx.getBasicBlock(getBasicBlockByName(F, "bb1"));
  sandboxir::Argument *Ptr = SBF.getArg(0);
  sandboxir::Argument *Val = SBF.getArg(1);

  sandboxir::Instruction *Br = &*BB0.rbegin();
  SmallVector<sandboxir::Instruction *> InsertedBB0;
  SmallVector<sandboxir::Instruction *> RemovedBB0;
  SmallVector<sandboxir::Instruction *> MovedBB0;
  Ctx.registerInsertInstrCallbackBB(
      BB0, [&InsertedBB0](sandboxir::Instruction *SBI) {
        InsertedBB0.push_back(SBI);
      });
  Ctx.registerRemoveInstrCallbackBB(
      BB0, [&RemovedBB0](sandboxir::Instruction *SBI) {
        RemovedBB0.push_back(SBI);
      });
  Ctx.registerMoveInstrCallbackBB(
      BB0, [&MovedBB0](sandboxir::Instruction *SBI, sandboxir::BasicBlock &BB,
                       const sandboxir::BBIterator &It) {
        MovedBB0.push_back(SBI);
      });

  Ctx.registerInsertInstrCallbackBB(BB1, [](sandboxir::Instruction *SBI) {
    llvm_unreachable("Shouldn't run!");
  });
  Ctx.registerRemoveInstrCallbackBB(BB1, [](sandboxir::Instruction *SBI) {
    llvm_unreachable("Shouldn't run!");
  });
  Ctx.registerRemoveInstrCallbackBB(BB1, [](sandboxir::Instruction *SBI) {
    llvm_unreachable("Shouldn't run!");
  });

  auto *NewI =
      sandboxir::StoreInst::create(Val, Ptr, /*Align=*/std::nullopt, Br, Ctx);
  EXPECT_EQ(InsertedBB0.size(), 1u);
  EXPECT_EQ(InsertedBB0[0], NewI);
  EXPECT_EQ(RemovedBB0.size(), 0u);
  EXPECT_EQ(MovedBB0.size(), 0u);

  // TODO: This causes an assertion failure in the DAG
  // NewI->moveBefore(BB0, BB0.end());
  // EXPECT_EQ(InsertedBB0.size(), 1u);
  // EXPECT_EQ(InsertedBB0[0], NewI);
  // EXPECT_EQ(RemovedBB0.size(), 0u);
  // EXPECT_EQ(MovedBB0.size(), 1u);
  // EXPECT_EQ(MovedBB0[0], NewI);

  Br->eraseFromParent();
  EXPECT_EQ(RemovedBB0.size(), 1u);
  EXPECT_EQ(RemovedBB0[0], Br);
  EXPECT_EQ(InsertedBB0.size(), 1u);

  NewI->eraseFromParent();
  EXPECT_EQ(RemovedBB0.size(), 2u);
  EXPECT_EQ(RemovedBB0[1], NewI);
  EXPECT_EQ(InsertedBB0.size(), 1u);
#ifndef NDEBUG
  BB0.verify();
#endif
}

TEST_F(SandboxIRTest, SBBasicBlock_GraphTraits) {
  parseIR(C, R"IR(
define void @foo(i1 %cond) {
bb0:
  br i1 %cond, label %bb1, label %bb2
bb1:
  br label %bb3
bb2:
  br label %bb3
bb3:
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto &SBF = *Ctx.createFunction(&F);
  (void)SBF;
  auto *SBBB0 = Ctx.getBasicBlock(getBasicBlockByName(F, "bb0"));
  auto *SBBB1 = Ctx.getBasicBlock(getBasicBlockByName(F, "bb1"));
  auto *SBBB2 = Ctx.getBasicBlock(getBasicBlockByName(F, "bb2"));
  auto *SBBB3 = Ctx.getBasicBlock(getBasicBlockByName(F, "bb3"));

  EXPECT_EQ(SBBB0, GraphTraits<sandboxir::BasicBlock *>::getEntryNode(SBBB0));
  auto ChildIt = GraphTraits<sandboxir::BasicBlock *>::child_begin(SBBB0);
  EXPECT_EQ(SBBB1, *ChildIt++);
  EXPECT_EQ(SBBB2, *ChildIt++);
  EXPECT_EQ(ChildIt, GraphTraits<sandboxir::BasicBlock *>::child_end(SBBB0));
  EXPECT_EQ(*GraphTraits<sandboxir::BasicBlock *>::child_begin(SBBB1), SBBB3);
}

TEST_F(SandboxIRTest, InstrNumbers) {
  parseIR(C, R"IR(
define void @foo(i8 %val0, i8 %val1) {
  %add = add i8 %val0, %val1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  sandboxir::Argument *Val0 = SBF.getArg(0);
  sandboxir::Argument *Val1 = SBF.getArg(1);
  auto It = SBBB.begin();
  sandboxir::Instruction *Add = &*It++;
  sandboxir::Instruction *Ret = &*It++;

  EXPECT_EQ(Add->getInstrNumber(), 0);
  EXPECT_EQ(Ret->getInstrNumber(), sandboxir::BasicBlock::InstrNumberingStep);
  int Div = 2;
  while (sandboxir::BasicBlock::InstrNumberingStep / Div >= 1) {
    auto *NewI = cast<sandboxir::Instruction>(
        sandboxir::CmpInst::create(CmpInst::ICMP_NE, Val0, Val1, Ret, Ctx));
    EXPECT_EQ(NewI->getInstrNumber(),
              sandboxir::BasicBlock::InstrNumberingStep -
                  sandboxir::BasicBlock::InstrNumberingStep / Div);
    Div = Div * 2;
  }
  // This should trigger re-numbering.
  sandboxir::CmpInst::create(CmpInst::ICMP_NE, Val0, Val1, Ret, Ctx);
  for (auto [Idx, I] : enumerate(SBBB))
    EXPECT_EQ(I.getInstrNumber(),
              (int64_t)Idx * sandboxir::BasicBlock::InstrNumberingStep);

  // Now check inserting an instruction before the first one.
  auto *NewI0 = cast<sandboxir::Instruction>(
      sandboxir::CmpInst::create(CmpInst::ICMP_NE, Val0, Val1, Add, Ctx));
  EXPECT_EQ(NewI0->getInstrNumber(),
            -sandboxir::BasicBlock::InstrNumberingStep);

  // Check moveBefore
  NewI0->moveBefore(SBBB, Add->getNextNode()->getIterator());
  EXPECT_EQ(NewI0->getInstrNumber(),
            Add->getInstrNumber() +
                sandboxir::BasicBlock::InstrNumberingStep / 2);

  NewI0->eraseFromParent();
#ifndef NDEBUG
  SBBB.verify();
#endif
  Add->removeFromParent();
#ifndef NDEBUG
  SBBB.verify();
#endif
  Add->insertBefore(Ret);
#ifndef NDEBUG
  SBBB.verify();
#endif
}

TEST_F(SandboxIRTest, ApproxInstrDistance) {
  parseIR(C, R"IR(
define void @foo(i8 %val0, i8 %val1) {
  %add0 = add i8 %val0, %val1
  %add1 = add i8 %val0, %val1
  %add2 = add i8 %val0, %val1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto &SBF = *Ctx.createFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();
  sandboxir::Instruction *I0 = &*It++;
  sandboxir::Instruction *I1 = &*It++;
  sandboxir::Instruction *I2 = &*It++;
  sandboxir::Instruction *I3 = &*It++;

  EXPECT_EQ(I0->getApproximateDistanceTo(I0), 0u);
  EXPECT_EQ(I0->getApproximateDistanceTo(I1), 1u);
  EXPECT_EQ(I1->getApproximateDistanceTo(I0), 1u);
  EXPECT_EQ(I0->getApproximateDistanceTo(I2), 2u);
  EXPECT_EQ(I0->getApproximateDistanceTo(I3), 3u);
}

// This used to crash the InstrRnage notifier.
TEST_F(SandboxIRTest, MoveToEnd) {
  parseIR(C, R"IR(
define void @foo(i8 %val0, i8 %val1) {
  %add0 = add i8 %val0, %val1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto &SBF = *Ctx.createFunction(&F);
  auto *Arg0 = SBF.getArg(0);
  auto *Arg1 = SBF.getArg(1);
  auto &BB = *SBF.begin();
  auto It = BB.begin();
  sandboxir::Instruction *I0 = &*It++;

  auto *Add = cast<sandboxir::BinaryOperator>(sandboxir::BinaryOperator::create(
      sandboxir::Instruction::Opcode::Add, Arg0, Arg1, I0, Ctx, "Test"));
  Add->moveBefore(BB, BB.end());
  EXPECT_EQ(&*BB.rbegin(), Add);
}

TEST_F(SandboxIRTest, GetOperandBundle) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr1, ptr %ptr2) {
  store i8 0, ptr %ptr1
  store i8 1, ptr %ptr2
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  BasicBlock *BB = &*F.begin();
  sandboxir::Function &SBF = *Ctx.createFunction(&F);
  sandboxir::BasicBlock &SBBB = *Ctx.getBasicBlock(BB);
  auto *Arg0 = SBF.getArg(0);
  auto *Arg1 = SBF.getArg(1);
  auto It = SBBB.begin();
  auto *S0 = &*It++;
  auto *S1 = &*It++;
  auto *C0 = S0->getOperand(0);
  auto *C1 = S1->getOperand(0);
  // Check getOperandBundle()
  DmpVector<sandboxir::Instruction *> SBVec({S0, S1});
  DmpVector<sandboxir::Value *> OpVec = getOperandBundle(SBVec, 1);
  EXPECT_EQ(OpVec, DmpVector<sandboxir::Value *>({Arg0, Arg1}));
  // Check getOpreandBundles()
  auto OpVecVec = getOperandBundles(SBVec);
  EXPECT_EQ(OpVecVec.size(), 2u);
  EXPECT_EQ(OpVecVec[0], DmpVector<sandboxir::Value *>({C0, C1}));
  EXPECT_EQ(OpVecVec[1], DmpVector<sandboxir::Value *>({Arg0, Arg1}));
}

TEST_F(SandboxIRTest, InsertElement) {
  parseIR(C, R"IR(
define void @foo(i8 %v0, i8 %v1) {
  %ins0 = insertelement <2 x i8> poison, i8 %v0, i32 0
  %ins1 = insertelement <2 x i8> %ins0, i8 %v1, i32 1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &SBF = *Ctx.createFunction(&F);
  unsigned ArgIdx = 0;
  auto *Arg0 = SBF.getArg(ArgIdx++);
  auto *Arg1 = SBF.getArg(ArgIdx++);
  auto *SBB = &*SBF.begin();
  auto It = SBB->begin();
  auto *Ins0 = cast<sandboxir::InsertElementInst>(&*It++);
  auto *Ins1 = cast<sandboxir::InsertElementInst>(&*It++);
  auto *Ret = &*It++;

  EXPECT_EQ(Ins0->getOpcode(), sandboxir::Instruction::Opcode::Insert);
  EXPECT_EQ(Ins0->getOperand(1), Arg0);
  EXPECT_EQ(Ins1->getOperand(1), Arg1);
  EXPECT_EQ(Ins1->getOperand(0), Ins0);
  auto *Poison = Ins0->getOperand(0);
  auto *Idx = Ins0->getOperand(2);
  auto *NewI1 =
      cast<sandboxir::InsertElementInst>(sandboxir::InsertElementInst::create(
          Poison, Arg0, Idx, Ret, Ctx, "NewIns1"));
  EXPECT_EQ(NewI1->getOperand(0), Poison);

  auto *NewI2 =
      cast<sandboxir::InsertElementInst>(sandboxir::InsertElementInst::create(
          Poison, Arg0, Idx, SBB, Ctx, "NewIns2"));
  EXPECT_EQ(NewI2->getPrevNode(), Ret);
}

TEST_F(SandboxIRTest, ExtractElement) {
  parseIR(C, R"IR(
define void @foo(<2 x i8> %vec) {
  %ext0 = extractelement <2 x i8> %vec, i32 0
  %ext1 = extractelement <2 x i8> %vec, i32 1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &SBF = *Ctx.createFunction(&F);
  unsigned ArgIdx = 0;
  auto *Arg0 = SBF.getArg(ArgIdx++);
  auto *SBB = &*SBF.begin();
  auto It = SBB->begin();
  auto *Ext0 = cast<sandboxir::ExtractElementInst>(&*It++);
  auto *Ext1 = cast<sandboxir::ExtractElementInst>(&*It++);
  auto *Ret = &*It++;

  EXPECT_EQ(Ext0->getOpcode(), sandboxir::Instruction::Opcode::Extract);
  EXPECT_EQ(Ext0->getOperand(0), Arg0);
  EXPECT_EQ(Ext1->getOperand(0), Arg0);
  auto *Idx0 = Ext0->getOperand(1);
  EXPECT_TRUE(isa<sandboxir::Constant>(Idx0));
  auto *Idx1 = Ext1->getOperand(1);
  EXPECT_TRUE(isa<sandboxir::Constant>(Idx1));
  auto *NewI1 = cast<sandboxir::ExtractElementInst>(
      sandboxir::ExtractElementInst::create(Arg0, Idx0, Ret, Ctx, "NewExt1"));
  EXPECT_EQ(NewI1->getOperand(0), Arg0);
  EXPECT_EQ(NewI1->getOperand(1), Idx0);
  auto *NewI2 = cast<sandboxir::ExtractElementInst>(
      sandboxir::ExtractElementInst::create(Arg0, Idx1, SBB, Ctx, "NewExt2"));
  EXPECT_EQ(NewI2->getPrevNode(), Ret);
}

TEST_F(SandboxIRTest, ShuffleVector) {
  parseIR(C, R"IR(
define void @foo(<2 x i8> %vec) {
  %shuff0 = shufflevector <2 x i8> %vec, <2 x i8> poison, <2 x i32> <i32 0, i32 1>
  %shuff1 = shufflevector <2 x i8> %vec, <2 x i8> poison, <2 x i32> <i32 1, i32 0>
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &SBF = *Ctx.createFunction(&F);
  unsigned ArgIdx = 0;
  auto *Arg0 = SBF.getArg(ArgIdx++);
  auto *SBB = &*SBF.begin();
  auto It = SBB->begin();
  auto *Shuff0 = cast<sandboxir::ShuffleVectorInst>(&*It++);
  auto *Shuff1 = cast<sandboxir::ShuffleVectorInst>(&*It++);
  auto *Ret = &*It++;

  EXPECT_EQ(Shuff0->getOpcode(), sandboxir::Instruction::Opcode::ShuffleVec);
  EXPECT_EQ(Shuff0->getOperand(0), Arg0);
  EXPECT_EQ(Shuff1->getOperand(0), Arg0);
  auto Mask0 = Shuff0->getShuffleMask();
  EXPECT_EQ(Mask0, SmallVector<int>({0, 1}));
  auto Mask1 = Shuff1->getShuffleMask();
  EXPECT_EQ(Mask1, SmallVector<int>({1, 0}));
  auto *NewI1 =
      cast<sandboxir::ShuffleVectorInst>(sandboxir::ShuffleVectorInst::create(
          Arg0, Arg0, Mask0, Ret, Ctx, "NewShuf1"));
  EXPECT_EQ(NewI1->getOperand(0), Arg0);
  EXPECT_EQ(NewI1->getOperand(1), Arg0);
  EXPECT_EQ(NewI1->getShuffleMask(), Mask0);
  auto *NewI2 =
      cast<sandboxir::ShuffleVectorInst>(sandboxir::ShuffleVectorInst::create(
          Arg0, Arg0, Mask1, SBB, Ctx, "NewShuf2"));
  EXPECT_EQ(NewI2->getOperand(0), Arg0);
  EXPECT_EQ(NewI2->getOperand(1), Arg0);
  EXPECT_EQ(NewI2->getShuffleMask(), Mask1);
}

TEST_F(SandboxIRTest, ReturnInstruction) {
  parseIR(C, R"IR(
define <2 x i8> @foo(<2 x i8> %vec) {
  ret <2 x i8> %vec
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &SBF = *Ctx.createFunction(&F);
  unsigned ArgIdx = 0;
  auto *Arg0 = SBF.getArg(ArgIdx++);
  auto *SBB = &*SBF.begin();
  auto It = SBB->begin();
  auto *Ret = cast<sandboxir::RetInst>(&*It++);
  EXPECT_EQ(Ret->getReturnValue(), Arg0);
  EXPECT_EQ(Ret->getNumOperands(), 1u);
  EXPECT_EQ(Ret->getOpcode(), sandboxir::Instruction::Opcode::Ret);
  auto *NewRet1 =
      cast<sandboxir::RetInst>(sandboxir::RetInst::create(nullptr, SBB, Ctx));
  EXPECT_EQ(NewRet1->getReturnValue(), nullptr);
  auto *NewRet2 =
      cast<sandboxir::RetInst>(sandboxir::RetInst::create(Arg0, SBB, Ctx));
  EXPECT_EQ(NewRet2->getReturnValue(), Arg0);
}

TEST_F(SandboxIRTest, CallInstruction) {
  parseIR(C, R"IR(
define <2 x i8> @foo(<2 x i8> %vec) {
  %call = call <2 x i8> @foo(<2 x i8> %vec)
  ret <2 x i8> %vec
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &SBF = *Ctx.createFunction(&F);
  unsigned ArgIdx = 0;
  auto *Arg0 = SBF.getArg(ArgIdx++);
  auto *SBB = &*SBF.begin();
  auto It = SBB->begin();
  auto *Call = cast<sandboxir::CallInst>(&*It++);
  auto *Ret = cast<sandboxir::RetInst>(&*It++);
  EXPECT_EQ(Call->getNumOperands(), 2u);
  EXPECT_EQ(Ret->getOpcode(), sandboxir::Instruction::Opcode::Ret);
  FunctionType *FTy = F.getFunctionType();
  SmallVector<sandboxir::Value *, 1> Args;
  Args.push_back(Arg0);
  auto *NewCall1 = cast<sandboxir::CallInst>(sandboxir::CallInst::create(
      FTy, &SBF, Args, Ret->getIterator(), SBB, Ctx));
  EXPECT_EQ(NewCall1->getNextNode(), Ret);

  auto *NewCall2 = cast<sandboxir::CallInst>(
      sandboxir::CallInst::create(FTy, &SBF, Args, Ret, Ctx));
  EXPECT_EQ(NewCall2->getNextNode(), Ret);

  auto *NewCall3 = cast<sandboxir::CallInst>(
      sandboxir::CallInst::create(FTy, &SBF, Args, SBB, Ctx));
  EXPECT_EQ(NewCall3->getPrevNode(), Ret);
}

TEST_F(SandboxIRTest, GetElementPtrInstruction) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep = getelementptr i8, ptr %ptr, i32 0
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &SBF = *Ctx.createFunction(&F);
  unsigned ArgIdx = 0;
  auto *Arg0 = SBF.getArg(ArgIdx++);
  auto *SBB = &*SBF.begin();
  auto It = SBB->begin();
  auto *GEP = cast<sandboxir::GetElementPtrInst>(&*It++);
  auto *Ret = cast<sandboxir::RetInst>(&*It++);
  EXPECT_EQ(GEP->getNumOperands(), 2u);
  EXPECT_EQ(Ret->getOpcode(), sandboxir::Instruction::Opcode::Ret);
  FunctionType *FTy = F.getFunctionType();
  SmallVector<sandboxir::Value *, 1> Args;
  Args.push_back(Arg0);
  auto *NewGEP1 =
      cast<sandboxir::GetElementPtrInst>(sandboxir::GetElementPtrInst::create(
          FTy, &SBF, Args, Ret->getIterator(), SBB, Ctx));
  EXPECT_EQ(NewGEP1->getNextNode(), Ret);

  auto *NewGEP2 = cast<sandboxir::GetElementPtrInst>(
      sandboxir::GetElementPtrInst::create(FTy, &SBF, Args, Ret, Ctx));
  EXPECT_EQ(NewGEP2->getNextNode(), Ret);

  auto *NewGEP3 = cast<sandboxir::GetElementPtrInst>(
      sandboxir::GetElementPtrInst::create(FTy, &SBF, Args, SBB, Ctx));
  EXPECT_EQ(NewGEP3->getPrevNode(), Ret);
}

TEST_F(SandboxIRTest, BranchInst) {
  parseIR(C, R"IR(
define void @foo(i1 %cond0, i1 %cond1) {
bb0:
  br label %bb1
bb1:
  br i1 %cond0, label %bb0, label %bb1
bb2:
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &SBF = *Ctx.createFunction(&F);
  unsigned ArgIdx = 0;
  auto *Cond0 = SBF.getArg(ArgIdx++);
  auto *Cond1 = SBF.getArg(ArgIdx++);

  auto *LLVMBB0 = getBasicBlockByName(F, "bb0");
  auto *LLVMBB1 = getBasicBlockByName(F, "bb1");
  auto *LLVMBB2 = getBasicBlockByName(F, "bb2");
  auto *BB0 = Ctx.getBasicBlock(LLVMBB0);
  auto *BB1 = Ctx.getBasicBlock(LLVMBB1);
  auto *BB2 = Ctx.getBasicBlock(LLVMBB2);
  (void)BB2;
  auto *Br0 = cast<sandboxir::BranchInst>(&*BB0->begin());
  EXPECT_FALSE(Br0->isConditional());
  EXPECT_TRUE(Br0->isUnconditional());
  EXPECT_EQ(Br0->getNumSuccessors(), 1u);
  EXPECT_EQ(Br0->getSuccessor(0), BB1);
  {
    auto SuccRange = Br0->successors();
    EXPECT_EQ(*SuccRange.begin(), BB1);
  }

  auto *Br1 = cast<sandboxir::BranchInst>(&*BB1->begin());
  EXPECT_TRUE(Br1->isConditional());
  EXPECT_FALSE(Br1->isUnconditional());
  EXPECT_EQ(Br1->getNumSuccessors(), 2u);
  EXPECT_EQ(Br1->getSuccessor(0), BB0);
  EXPECT_EQ(Br1->getSuccessor(1), BB1);
  EXPECT_EQ(Br1->getCondition(), Cond0);
  Br1->setCondition(Cond1);
  EXPECT_EQ(Br1->getCondition(), Cond1);
  {
    auto SuccRange = Br1->successors();
    auto SuccIt = SuccRange.begin();

    auto *LLVMBr1 = cast<llvm::BranchInst>(LLVMBB1->getTerminator());
    auto LLVMSuccRange = LLVMBr1->successors();
    auto LLVMSuccIt = LLVMSuccRange.begin();
    EXPECT_EQ(*SuccIt++,
              cast<sandboxir::BasicBlock>(Ctx.getValue(*LLVMSuccIt++)));
    EXPECT_EQ(*SuccIt++,
              cast<sandboxir::BasicBlock>(Ctx.getValue(*LLVMSuccIt++)));
  }
  {
    // Same but for const successors()
    auto SuccRange =
        static_cast<const sandboxir::BranchInst *>(Br1)->successors();
    auto SuccIt = SuccRange.begin();

    auto *LLVMBr1 = static_cast<const llvm::BranchInst *>(
        cast<llvm::BranchInst>(LLVMBB1->getTerminator()));
    auto LLVMSuccRange = LLVMBr1->successors();
    auto LLVMSuccIt = LLVMSuccRange.begin();
    EXPECT_EQ(*SuccIt++,
              cast<sandboxir::BasicBlock>(Ctx.getValue(*LLVMSuccIt++)));
    EXPECT_EQ(*SuccIt++,
              cast<sandboxir::BasicBlock>(Ctx.getValue(*LLVMSuccIt++)));
  }

  sandboxir::BranchInst *NewBr = sandboxir::BranchInst::create(BB1, BB1, Ctx);
  EXPECT_FALSE(NewBr->isConditional());
  EXPECT_TRUE(NewBr->isUnconditional());
#ifndef NDEBUG
  EXPECT_DEATH(NewBr->getCondition(), ".*");
#endif
  EXPECT_EQ(NewBr->getNumSuccessors(), 1u);
  EXPECT_EQ(NewBr->getSuccessor(0), BB1);

  // Check swapSuccessors()
  Br1->swapSuccessors();
  EXPECT_EQ(Br1->getCondition(), Cond1);
  EXPECT_EQ(Br1->getSuccessor(0), BB1);
  EXPECT_EQ(Br1->getSuccessor(1), BB0);
  Br1->swapSuccessors();
  EXPECT_EQ(Br1->getCondition(), Cond1);
  EXPECT_EQ(Br1->getSuccessor(0), BB0);
  EXPECT_EQ(Br1->getSuccessor(1), BB1);
}

TEST_F(SandboxIRTest, BinaryOperator_swapOperands) {
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

  EXPECT_EQ(I0->getOperand(0), V0);
  EXPECT_EQ(I0->getOperand(1), V1);
  I0->swapOperands();
  EXPECT_EQ(I0->getOperand(0), V1);
  EXPECT_EQ(I0->getOperand(1), V0);
  EXPECT_EQ(LLVMI0->getOperand(0), LLVMV1);
  EXPECT_EQ(LLVMI0->getOperand(1), LLVMV0);
  I0->swapOperands();
  EXPECT_EQ(I0->getOperand(0), V0);
  EXPECT_EQ(I0->getOperand(1), V1);
  EXPECT_EQ(LLVMI0->getOperand(0), LLVMV0);
  EXPECT_EQ(LLVMI0->getOperand(1), LLVMV1);
}
