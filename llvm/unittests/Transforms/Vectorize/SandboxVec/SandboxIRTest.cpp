//===- SandboxIRTest.cpp --------------------------------------------===//
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
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"
#include "gtest/gtest.h"

using namespace llvm;

struct SandboxIRTest : public testing::Test {
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

TEST_F(SandboxIRTest, IteratorsSimple) {
  parseIR(C, R"IR(
define void @foo(i32 %v1) {
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
  auto BBIt = BB->begin();
  Instruction *Add0 = &*BBIt++;
  Instruction *Add1 = &*BBIt++;
  Instruction *Ret = &*BBIt++;
  SBBasicBlock &SBBB = *Ctxt.createSBBasicBlock(BB);

  auto *SBAdd0 = Ctxt.getSBValue(Add0);
  auto *SBAdd1 = Ctxt.getSBValue(Add1);
  auto *SBRet = Ctxt.getSBValue(Ret);

  auto It = SBBB.begin();
  EXPECT_EQ(&*It++, SBAdd0);
  EXPECT_EQ(&*It++, SBAdd1);
  EXPECT_EQ(&*It++, SBRet);
  EXPECT_EQ(It, SBBB.end());
#ifndef NDEBUG
  EXPECT_DEATH(++It, "Already.*");
#endif
  --It;
  EXPECT_EQ(&*It--, Ctxt.getSBValue(Ret));
  EXPECT_EQ(&*It--, Ctxt.getSBValue(Add1));
  EXPECT_EQ(&*It, Ctxt.getSBValue(Add0));
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

TEST_F(SandboxIRTest, IteratorsMultiInstr) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto BBIt = BB->begin();
  Instruction *Pack0 = &*BBIt++;
  Instruction *Pack1 = &*BBIt++;
  Instruction *Pack2 = &*BBIt++;
  Instruction *Pack3 = &*BBIt++;
  Instruction *Add0 = &*BBIt++;
  Instruction *Add1 = &*BBIt++;
  Instruction *Ret = &*BBIt++;
  auto &SBBB = *Ctxt.createSBBasicBlock(BB);
  auto It = SBBB.begin();

  auto *I0 = &*It++;
  EXPECT_EQ(I0, Ctxt.getSBValue(Pack0));
  EXPECT_EQ(I0, Ctxt.getSBValue(Pack1));
  EXPECT_EQ(I0, Ctxt.getSBValue(Pack2));
  EXPECT_EQ(I0, Ctxt.getSBValue(Pack3));
  auto *I1 = &*It++;
  EXPECT_EQ(I1, Ctxt.getSBValue(Add0));
  auto *I2 = &*It++;
  EXPECT_EQ(I2, Ctxt.getSBValue(Add1));
  auto *I3 = &*It++;
  EXPECT_EQ(I3, Ctxt.getSBValue(Ret));
  EXPECT_EQ(It, SBBB.end());
#ifndef NDEBUG
  EXPECT_DEATH(++It, "Already.*");
#endif
  --It;
  EXPECT_EQ(&*It--, Ctxt.getSBValue(Ret));
  EXPECT_EQ(&*It--, Ctxt.getSBValue(Add1));
  EXPECT_EQ(&*It--, Ctxt.getSBValue(Add0));
  EXPECT_EQ(&*It, Ctxt.getSBValue(Pack0));
  EXPECT_EQ(It, SBBB.begin());
#ifndef NDEBUG
  EXPECT_DEATH(--It, "Already.*");
#endif

  // Check iterator equality.
  auto *Pack = cast<SBPackInstruction>(Ctxt.getSBValue(Pack3));
  EXPECT_TRUE(SBBB.begin() == Pack->getIterator());
}

TEST_F(SandboxIRTest, SBUse_Simple) {
  parseIR(C, R"IR(
define i32 @foo(i32 %v0, i32 %v1) {
  %add0 = add i32 %v0, %v1
  %add1 = add i32 %add0, %add0
  ret i32 %add0
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
  Instruction *I0 = &*BBIt++;
  (void)I0;
  Instruction *I1 = &*BBIt++;
  (void)I1;
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *Ctxt.getSBBasicBlock(BB);
  auto *SBArg0 = SBF.getArg(0);
  auto *SBArg1 = SBF.getArg(1);
  auto It = SBBB.begin();
  auto *SBI0 = &*It++;
  auto *SBI1 = &*It++;
  (void)SBI1;
  auto *SBRet = &*It++;

  SmallVector<SBArgument *> Args{SBArg0, SBArg1};
  unsigned OpIdx = 0;
  for (SBUse Use : SBI0->operands()) {
    EXPECT_EQ(Use.getOperandNo(), OpIdx);
    EXPECT_EQ(Use.get(), Args[OpIdx]);
    ++OpIdx;
  }
  EXPECT_EQ(OpIdx, 2u);

  // Check SBUse iterators when the value has no uses.
  unsigned Cnt = 0;
  for (auto It = SBRet->use_begin(), ItE = SBRet->use_end(); It != ItE;
       ++It)
    ++Cnt;
  EXPECT_EQ(Cnt, 0u);
}

TEST_F(SandboxIRTest, SBUseIterator_Simple) {
  parseIR(C, R"IR(
define i32 @foo(i32 %v0, i32 %v1) {
  %add0 = add i32 %v0, %v1
  %add1 = add i32 %add0, %add0
  ret i32 %add0
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
  Instruction *I0 = &*BBIt++;
  (void)I0;
  Instruction *I1 = &*BBIt++;
  (void)I1;
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *Ctxt.getSBBasicBlock(BB);
  auto *SBArg0 = SBF.getArg(0);
  auto *SBArg1 = SBF.getArg(1);
  auto It = SBBB.begin();
  auto *SBI0 = &*It++;
  auto *SBI1 = &*It++;

  SBOperandUseIterator UseIt0 = SBI0->op_begin();
  // Check operator==
  EXPECT_TRUE(UseIt0 == SBI0->op_begin());
  // Check SBUse
  SBUse Use0 = *UseIt0;
  EXPECT_EQ(Use0.get(), SBArg0);
  EXPECT_EQ(Use0.getUser(), SBI0);
  ++UseIt0;
  Use0 = *UseIt0;
  EXPECT_EQ(Use0.get(), SBArg1);
  EXPECT_EQ(Use0.getUser(), SBI0);
  ++UseIt0;
  EXPECT_EQ(UseIt0, SBI0->op_end());

  SBOperandUseIterator UseIt1 = SBI1->op_begin();
  EXPECT_TRUE(UseIt1 != UseIt0);
  SBUse Use1 = *UseIt1;
  EXPECT_EQ(Use1.get(), SBI0);
  EXPECT_EQ(Use1.getUser(), SBI1);
  ++UseIt1;
  Use1 = *UseIt1;
  EXPECT_EQ(Use1.get(), SBI0);
  EXPECT_EQ(Use1.getUser(), SBI1);
  ++UseIt1;
}

TEST_F(SandboxIRTest, SBOperandUseIterator_Pack) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto BBIt = BB->begin();
  Instruction *I0 = &*BBIt++;
  (void)I0;
  Instruction *I1 = &*BBIt++;
  (void)I1;
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *Ctxt.getSBBasicBlock(BB);
  auto *SBArg0 = SBF.getArg(0);
  auto *SBArg1 = SBF.getArg(1);
  auto It = SBBB.begin();
  auto *Pack0 = &*It++;
  SBOperandUseIterator UseIt = Pack0->op_begin();
  EXPECT_EQ(*UseIt, SBArg0);
  ++UseIt;
  EXPECT_EQ(*UseIt, SBArg1);
  ++UseIt;
  EXPECT_EQ(UseIt, Pack0->op_end());
}

TEST_F(SandboxIRTest, SBOperandUseIterator_Unpack) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto BBIt = BB->begin();
  Instruction *I0 = &*BBIt++;
  (void)I0;
  Instruction *I1 = &*BBIt++;
  (void)I1;
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *Ctxt.getSBBasicBlock(BB);
  auto *SBArg0 = SBF.getArg(0);
  auto It = SBBB.begin();
  auto *Unpack0 = &*It++;
  SBOperandUseIterator UseIt = Unpack0->op_begin();
  EXPECT_EQ(*UseIt, SBArg0);
  ++UseIt;
  EXPECT_EQ(UseIt, Unpack0->op_end());
}

TEST_F(SandboxIRTest, SBOperandUseIterator_Shuffle) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto BBIt = BB->begin();
  Instruction *I0 = &*BBIt++;
  (void)I0;
  Instruction *I1 = &*BBIt++;
  (void)I1;
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *Ctxt.getSBBasicBlock(BB);
  auto *SBArg0 = SBF.getArg(0);
  auto It = SBBB.begin();
  auto *Shuffle0 = &*It++;
  SBOperandUseIterator UseIt = Shuffle0->op_begin();
  EXPECT_EQ(*UseIt, SBArg0);
  ++UseIt;
  EXPECT_EQ(UseIt, Shuffle0->op_end());
}

TEST_F(SandboxIRTest, SBUserUse_Simple) {
  parseIR(C, R"IR(
define void @foo(i32 %v0, i32 %v1) {
  %add0 = add i32 %v0, %v1
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
  auto BBIt = BB->begin();
  Instruction *I0 = &*BBIt++;
  (void)I0;
  Instruction *I1 = &*BBIt++;
  (void)I1;
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();
  auto *SBI0 = &*It++;
  auto *SBI1 = &*It++;

  SBValue::use_iterator UseIt0 = SBI0->use_begin();
  const SBUse &Use = *UseIt0;
  EXPECT_EQ(Use.getUser(), SBI1);
  EXPECT_EQ(Use.get(), SBI0);
  ++UseIt0;
  const SBUse &Use1 = *UseIt0;
  EXPECT_EQ(Use1.getUser(), SBI1);
  EXPECT_EQ(Use1.get(), SBI0);
  ++UseIt0;
  EXPECT_EQ(UseIt0, SBI0->use_end());
}

TEST_F(SandboxIRTest, PackDetection_Canonical1) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  auto *PackA = cast<SBPackInstruction>(&*It++);
  EXPECT_EQ(PackA->getOpcode(), SBInstruction::Opcode::Pack);
  auto *PackB = cast<SBPackInstruction>(&*It++);
  EXPECT_EQ(PackB->getOpcode(), SBInstruction::Opcode::Pack);
  auto *Unpack = cast<SBUnpackInstruction>(&*It++);
  EXPECT_EQ(Unpack->getOpcode(), SBInstruction::Opcode::Unpack);
  auto *Ret = cast<SBOpaqueInstruction>(&*It++);
  EXPECT_EQ(Ret->getOpcode(), SBInstruction::Opcode::Opaque);
}

TEST_F(SandboxIRTest, PackDetection_Canonical2) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  auto *PackA = cast<SBPackInstruction>(&*It++);
  EXPECT_EQ(PackA->getOpcode(), SBInstruction::Opcode::Pack);
  auto *PackB = cast<SBPackInstruction>(&*It++);
  EXPECT_EQ(PackB->getOpcode(), SBInstruction::Opcode::Pack);
  auto *Unpack = cast<SBUnpackInstruction>(&*It++);
  EXPECT_EQ(Unpack->getOpcode(), SBInstruction::Opcode::Unpack);
  auto *Ret = cast<SBOpaqueInstruction>(&*It++);
  EXPECT_EQ(Ret->getOpcode(), SBInstruction::Opcode::Opaque);
}

TEST_F(SandboxIRTest, PackDetection_Canonical_FullExtracts) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<SBPackInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
}

TEST_F(SandboxIRTest, PackDetection_Canonical_ExtractGroups) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<SBPackInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
}

TEST_F(SandboxIRTest, PackDetection_SimpleIRPattern_WithExtracts) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  Instruction *Ret = BB->getTerminator();

  auto &SBBB = *Ctxt.createSBBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  auto *BotInsert = cast<InsertElementInst>(BB->getTerminator()->getPrevNode());
  auto *Pack = cast<SBPackInstruction>(Ctxt.getSBValue(BotInsert));
#ifndef NDEBUG
  Pack->verify();
#endif
  // Make sure the extracts are part of the Pack, not separate unpacks.
  auto *SBV0 = Ctxt.getSBValue(F.getArg(0));
  auto *SBV1 = Ctxt.getSBValue(F.getArg(1));
  auto *SBV2 = Ctxt.getSBValue(F.getArg(2));
  EXPECT_EQ(Pack->getOperand(0), SBV0);
  EXPECT_EQ(Pack->getOperand(1), SBV1);
  EXPECT_EQ(Pack->getOperand(2), SBV2);
  EXPECT_EQ(Pack->getNumOperands(), 3u);

  for (SBValue &SBV : SBBB)
    EXPECT_TRUE(!isa<SBUnpackInstruction>(&SBV));

  // Check eraseFromParent().
  Pack->eraseFromParent();
  ASSERT_EQ(BB->size(), 1u);
  ASSERT_EQ(&BB->back(), Ret);
}

TEST_F(SandboxIRTest, PackNotCanonical_BadExtractPosition) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBUnpackInstruction>(&*It++));
  EXPECT_TRUE(isa<SBUnpackInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
}

TEST_F(SandboxIRTest, PackNotCanonical_MissingExtract1) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBUnpackInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
}

TEST_F(SandboxIRTest, PackNotCanonical_MissingExtract2) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBUnpackInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
}

TEST_F(SandboxIRTest, PackNotCanonical_ExtractIndicesOutOfOrder) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBUnpackInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBUnpackInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
}

TEST_F(SandboxIRTest, PackNotCanonical_ExtractsFromDifferentVectors) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBUnpackInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBUnpackInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
}

TEST_F(SandboxIRTest, PackNotCanonical_ExtractVectorLarger) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBUnpackInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBUnpackInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
}

TEST_F(SandboxIRTest, PackNotCanonical_ExtractVectorSmaller) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBUnpackInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBUnpackInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBUnpackInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
}

TEST_F(SandboxIRTest, PackNotCanonical_InsertIndicesOutOffOrder) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
}

TEST_F(SandboxIRTest, PackNotCanonical_BadInsertPattern) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
}

TEST_F(SandboxIRTest, PackNotCanonical_NonPoisonOperand) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
}

TEST_F(SandboxIRTest, PackNotCanonical_BrokenChain) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
}

TEST_F(SandboxIRTest, PackNotCanonical_BadChain) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto &SBBB = *Ctxt.createSBBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  EXPECT_TRUE(none_of(
      SBBB, [](SBValue &N) { return isa<SBPackInstruction>(&N); }));
}

TEST_F(SandboxIRTest, PackNotCanonical_FoldedConsts) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
  EXPECT_TRUE(isa<SBOpaqueInstruction>(&*It++));
}

// This used to crash.
TEST_F(SandboxIRTest, PackNotCanonical_OneInsertAtStartOfBB) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();
  EXPECT_TRUE(!isa<SBPackInstruction>(&*It++));
}

TEST_F(SandboxIRTest, PackNotCanonical_MissingTopExtract) {
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();
  EXPECT_TRUE(!isa<SBPackInstruction>(&*It++));
  EXPECT_TRUE(!isa<SBPackInstruction>(&*It++));
  EXPECT_TRUE(!isa<SBPackInstruction>(&*It++));
}

TEST_F(SandboxIRTest, PackNotCanonical_MissingOperands) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto *SBF = Ctxt.createSBFunction(&F);
  (void)SBF;
  auto *SBBB = Ctxt.getSBBasicBlock(BB);
  auto It = SBBB->begin();
  auto *NotPack = &*It++;
  EXPECT_FALSE(isa<SBPackInstruction>(NotPack));
}

TEST_F(SandboxIRTest, PackNotCanonical_PoisonOperand) {
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
  SBContext Ctxt(C, AA);

  auto *BB = Ctxt.createSBBasicBlock(&*F.begin());
#ifndef NDEBUG
  BB->verify();
#endif
  auto It = BB->begin();
  auto *Gep = &*It++;
  (void)Gep;
  auto *NotPack = &*It++;
  EXPECT_FALSE(isa<SBPackInstruction>(NotPack));
  auto *St = cast<SBStoreInstruction>(&*It++);
  EXPECT_EQ(St->getOperand(0), NotPack);
}

TEST_F(SandboxIRTest, Opcodes) {
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
  ; An extract with a non-constant index is Opaque
  %ExtractOpq1 =  extractelement <2 x i32> %vec, i32 %v2

  ; A SB-IR-style shuffle.
  %Shuffle = shufflevector <2 x i32> %vec, <2 x i32> poison, <2 x i32> <i32 1, i32 0>
  ; A blend is Opaque for now.
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
  SBContext Ctxt(C, AA);
  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  auto It = SBBB.begin();

  auto *Pack1 = cast<SBPackInstruction>(&*It++);
  EXPECT_EQ(Pack1->getOpcode(), SBInstruction::Opcode::Pack);

  auto *Pack2 = cast<SBPackInstruction>(&*It++);
  EXPECT_EQ(Pack2->getOpcode(), SBInstruction::Opcode::Pack);

  auto *InsertOpq1 = &*It++;
  EXPECT_EQ(InsertOpq1->getOpcode(), SBInstruction::Opcode::Opaque);
  auto *InsertOpq2 = &*It++;
  EXPECT_EQ(InsertOpq2->getOpcode(), SBInstruction::Opcode::Opaque);

  auto *Unpack = &*It++;
  EXPECT_EQ(Unpack->getOpcode(), SBInstruction::Opcode::Unpack);
  auto *ExtractOpq1 = &*It++;
  EXPECT_EQ(ExtractOpq1->getOpcode(), SBInstruction::Opcode::Opaque);

  auto *Shuffle = &*It++;
  EXPECT_EQ(Shuffle->getOpcode(), SBInstruction::Opcode::Shuffle);
  auto *ShuffleBlend = &*It++;
  EXPECT_EQ(ShuffleBlend->getOpcode(), SBInstruction::Opcode::Opaque);

  auto *Call = &*It++;
  EXPECT_EQ(Call->getOpcode(), SBInstruction::Opcode::Opaque);

  auto *Ret = &*It++;
  EXPECT_EQ(Ret->getOpcode(), SBInstruction::Opcode::Opaque);
}

TEST_F(SandboxIRTest, PackOperands) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = Ctxt.getSBBasicBlock(BB);
  auto It = SBBB->begin();
  auto *Pack = cast<SBPackInstruction>(&*It++);
  auto *Ret = &*It++;

  auto *Arg0 = SBF->getArg(0);
  auto *Arg1 = SBF->getArg(1);
  auto *Arg2 = SBF->getArg(2);
  auto *Arg3 = SBF->getArg(3);
  auto *Arg4 = SBF->getArg(4);

  // Check getOperand()
  SmallVector<SBValue *> ExpectedOperands{Arg0, Arg1, Arg2};
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
    SmallVector<SBValue *> ExpectedOperands{Arg0, Arg3, Arg2};
    for (auto [OpIdx, Op] : enumerate(Pack->operands())) {
      EXPECT_EQ(Pack->getOperand(OpIdx), Op);
      EXPECT_EQ(ExpectedOperands[OpIdx], Op);
    }
  }

  // Check Pack RUOW
  Pack->replaceUsesOfWith(Arg1, Arg3);
  EXPECT_EQ(Pack->getOperand(1), Arg3);

  Pack->replaceUsesWithIf(Arg4,
                          [](SBUser *DstU, unsigned OpIdx) { return true; });
  EXPECT_EQ(Ret->getOperand(0), Arg4);
}

// When packing constant vectors the constant elements may finally show up as
// individual constants in the pack instruction. Check that the SBConstants
// for these constans have been created, otherwise getOperand() returns null.
TEST_F(SandboxIRTest, PackConstantVector) {
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
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
  auto *BB = Ctxt.getSBBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *St = cast<SBStoreInstruction>(&*It++);

  auto *SBConstVec = cast<SBConstant>(St->getValueOperand());
  auto *ArgVal1 = SBF->getArg(1);
  auto *ArgVal2 = SBF->getArg(2);
  SBValBundle Ops{ArgVal1, ArgVal2, SBConstVec};
  auto *Pack =
      cast<SBPackInstruction>(Ctxt.createSBPackInstruction(Ops, BB));

  // We expect non-null operands.
  EXPECT_TRUE(Pack->getOperand(0) != nullptr);
  EXPECT_TRUE(Pack->getOperand(1) != nullptr);
  EXPECT_TRUE(Pack->getOperand(2) != nullptr);
  EXPECT_TRUE(Pack->getOperand(3) != nullptr);
}

TEST_F(SandboxIRTest, PackExternalFacingOperands) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto *SBF = Ctxt.createSBFunction(&F);
  (void)SBF;
  auto *SBBB = Ctxt.getSBBasicBlock(BB);
  auto It = SBBB->begin();
  auto *Pack = cast<SBPackInstruction>(&*It++);

  auto BBIt = BB->begin();
  auto *Ins0 = &*BBIt++;
  auto *Extr0 = &*BBIt++;
  ++BBIt; // Ins1
  auto *Extr1 = &*BBIt++;
  ++BBIt; // Ins2
  auto *Ins3 = &*BBIt++;

  auto ExtInstrs = SBPackInstructionAttorney::getExternalFacingIRInstrs(Pack);
  EXPECT_EQ(ExtInstrs, SmallVector<Instruction *>({Ins3, Extr1, Extr0, Ins0}));
}

TEST_F(SandboxIRTest, PackGetIRInstrs_InReverseProgramOrder) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto *SBF = Ctxt.createSBFunction(&F);
  (void)SBF;
  auto *SBBB = Ctxt.getSBBasicBlock(BB);
  auto It = SBBB->begin();
  auto *Pack = cast<SBPackInstruction>(&*It++);
  auto IRInstrs = SBPackInstructionAttorney::getIRInstrs(Pack);
  // Expect reverse program order
  for (auto Idx : seq<unsigned>(1, IRInstrs.size()))
    EXPECT_TRUE(IRInstrs[Idx]->comesBefore(IRInstrs[Idx - 1]));
}

TEST_F(SandboxIRTest, GetOperandUseIdx) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto *SBF = Ctxt.createSBFunction(&F);
  (void)SBF;
  auto *SBBB = Ctxt.getSBBasicBlock(BB);
  auto SBIt = SBBB->begin();
  auto *Pack = &*SBIt++;
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
  EXPECT_EQ(SBUserAttorney::getOperandUseIdx(Pack, Use0), 0u);
  EXPECT_EQ(SBUserAttorney::getOperandUseIdx(Pack, Use1_a), 1u);
  EXPECT_EQ(SBUserAttorney::getOperandUseIdx(Pack, Use1_b), 1u);
  EXPECT_EQ(SBUserAttorney::getOperandUseIdx(Pack, Use2), 2u);
  EXPECT_EQ(SBUserAttorney::getOperandUseIdx(SBRet, RetUse), 0u);

#ifndef NDEBUG
  // Some invalid uses:
  const Use &UseIns1_0 = Ins1->getOperandUse(0);
  const Use &UseIns1_1 = Ins1->getOperandUse(1);
  const Use &UseIns2_0 = Ins2->getOperandUse(0);
  const Use &UseIns2_1 = Ins2->getOperandUse(1);
  EXPECT_DEATH(SBUserAttorney::getOperandUseIdx(Pack, UseIns1_0),
               ".*not found.*");
  EXPECT_DEATH(SBUserAttorney::getOperandUseIdx(Pack, UseIns1_1),
               ".*not found.*");
  EXPECT_DEATH(SBUserAttorney::getOperandUseIdx(Pack, UseIns2_0),
               ".*not found.*");
  EXPECT_DEATH(SBUserAttorney::getOperandUseIdx(Pack, UseIns2_1),
               ".*not found.*");
  EXPECT_DEATH(SBUserAttorney::getOperandUseIdx(SBRet, UseIns2_1),
               ".*not found.*");
#endif
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
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  SBContext Ctxt(C, AA);

  BasicBlock *BB0 = getBasicBlockByName(F, "bb0");
  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  BasicBlock *BB2 = getBasicBlockByName(F, "bb2");

  auto *SBF = Ctxt.createSBFunction(&F);
  EXPECT_EQ(SBF->arg_size(), 2u);
  EXPECT_EQ(SBF->getArg(0), Ctxt.getSBValue(F.getArg(0)));
  EXPECT_EQ(SBF->getArg(1), Ctxt.getSBValue(F.getArg(1)));
  auto *SBBB0 = Ctxt.getSBBasicBlock(BB0);
  auto *SBBB1 = Ctxt.getSBBasicBlock(BB1);
  auto *SBBB2 = Ctxt.getSBBasicBlock(BB2);
  SmallVector<SBBasicBlock *> SBBBs;
  for (auto &SBBB : *SBF)
    SBBBs.push_back(&SBBB);

  EXPECT_EQ(SBBBs[0], SBBB0);
  EXPECT_EQ(SBBBs[1], SBBB1);
  EXPECT_EQ(SBBBs[2], SBBB2);
}

TEST_F(SandboxIRTest, SBFunction_detachFromLLVMIR) {
  parseIR(C, R"IR(
define void @foo() {
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

  auto &SBF = *Ctxt.createSBFunction(&F);
  auto *SBBB = cast<SBBasicBlock>(Ctxt.getSBValue(BB));
  (void)SBBB;
  EXPECT_NE(Ctxt.getNumValues(), 0u);
  SBF.detachFromLLVMIR();
  EXPECT_EQ(Ctxt.getNumValues(), 0u);
}

#ifndef NDEBUG
TEST_F(SandboxIRTest, MoveBefore) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto RawIt = BB->begin();
  auto *Add0 = &*RawIt++;
  auto *Ins0 = &*RawIt++;
  auto *Ins1 = &*RawIt++;
  auto *Add1 = &*RawIt++;
  auto *Ret = &*RawIt++;

  auto &SBBB = *Ctxt.createSBBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  auto It = SBBB.begin();
  auto *SBAdd0 = &*It++;
  auto *Pack = &*It++;
  auto *SBAdd1 = &*It++;
  auto *SBRet = &*It++;

  Ctxt.disableCallbacks();

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
TEST_F(SandboxIRTest, MoveAfter) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto RawIt = BB->begin();
  auto *Add0 = &*RawIt++;
  auto *Ins0 = &*RawIt++;
  auto *Ins1 = &*RawIt++;
  auto *Add1 = &*RawIt++;
  auto *Ret = &*RawIt++;

  auto &SBBB = *Ctxt.createSBBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  auto It = SBBB.begin();
  auto *SBAdd0 = &*It++;
  auto *Pack = &*It++;
  auto *SBAdd1 = &*It++;
  auto *SBRet = &*It++;

  Ctxt.disableCallbacks();

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

TEST_F(SandboxIRTest, PrevNode_WhenPrevIsDetached) {
  parseIR(C, R"IR(
define void @foo(i32 %v1) {
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

  SBBasicBlock &SBBB = *Ctxt.createSBBasicBlock(BB);
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
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  SBContext Ctxt(C, AA);
  BasicBlock *BB = &*F.begin();

  SBBasicBlock &SBBB = *Ctxt.createSBBasicBlock(BB);
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

TEST_F(SandboxIRTest, PrevNode_Pack) {
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
  SBContext Ctxt(C, AA);
  BasicBlock *BB = &*F.begin();
  SBBasicBlock &SBBB = *Ctxt.createSBBasicBlock(BB);
  auto It = SBBB.begin();
  auto *Pack = cast<SBPackInstruction>(&*It++);
  EXPECT_EQ(Pack->getPrevNode(), nullptr);
}

// Check getPrevNode() and getNextNode()
TEST_F(SandboxIRTest, MultiIRPrevNextNode) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto RawIt = BB->begin();

  auto *Ins0 = &*RawIt++;
  auto *Ins1 = &*RawIt++;
  auto *Ins10 = &*RawIt++;
  auto *Ins11 = &*RawIt++;
  auto *Ret = &*RawIt++;
  (void)Ret;

  auto *SBF = Ctxt.createSBFunction(&F);
  (void)SBF;
  auto &SBBB = *Ctxt.getSBBasicBlock(BB);
  auto It = SBBB.begin();
  auto *Pack0 = cast<SBPackInstruction>(&*It++);
  auto *Pack1 = cast<SBPackInstruction>(&*It++);
  auto *SBRet = &*It++;

#ifndef NDEBUG
  SBBB.verify();
#endif

  EXPECT_EQ(Pack0, Ctxt.getSBValue(Ins0));
  EXPECT_EQ(Pack0, Ctxt.getSBValue(Ins1));
  EXPECT_EQ(Pack1, Ctxt.getSBValue(Ins10));
  EXPECT_EQ(Pack1, Ctxt.getSBValue(Ins11));

  EXPECT_EQ(SBRet->getPrevNode(), Pack1);
  EXPECT_EQ(Pack1->getPrevNode(), Pack0);
  EXPECT_EQ(Pack0->getPrevNode(), nullptr);

  EXPECT_EQ(Pack0->getNextNode(), Pack1);
  EXPECT_EQ(Pack1->getNextNode(), SBRet);
  EXPECT_EQ(SBRet->getNextNode(), nullptr);
}

TEST_F(SandboxIRTest, InsertBefore) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto RawIt = BB->begin();
  auto *Ins0 = &*RawIt++;
  auto *Ins1 = &*RawIt++;
  auto *Ret = &*RawIt++;

  auto *SBF = Ctxt.createSBFunction(&F);
  auto &SBBB = *Ctxt.getSBBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  auto It = SBBB.begin();
  auto *Pack = &*It++;
  auto *SBRet = &*It++;

  SBValBundle SBPackInstructions({SBF->getArg(2), SBF->getArg(3)});
  auto *NewPack = cast<SBPackInstruction>(
      Ctxt.createSBPackInstruction(SBPackInstructions, &SBBB));
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

TEST_F(SandboxIRTest, SBBasicBlock_detachFromLLVMIR) {
  parseIR(C, R"IR(
define void @foo() {
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
  Instruction *Ret = BB->getTerminator();

  auto &SBBB = *Ctxt.createSBBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  auto *SBI = cast<SBInstruction>(Ctxt.getSBValue(Ret));
  (void)SBI;
  SBBB.detachFromLLVMIR();
  EXPECT_EQ(Ctxt.getNumValues(), 0u);
}

// Check that SandboxIR Instructions in SBBasicBlock get erased in the right
// order when context goes out of scope
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
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  {
    SBContext Ctxt(C, AA);
    BasicBlock *BB = &*F.begin();
    Instruction *Ret = BB->getTerminator();
    (void)Ret;
    auto &SBBB = *Ctxt.createSBBasicBlock(BB);
    (void)SBBB;
#ifndef NDEBUG
    SBBB.verify();
#endif
  }
}

/// Check that SBBasicBlock is registered with LLVMValueToSBValueMap
TEST_F(SandboxIRTest, SBBBLLVMValueToSBValueMap) {
  parseIR(C, R"IR(
define i32 @foo(i32 %v) {
bb0:
  br label %bb1
bb1:
  ret i32 %v
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

  BasicBlock *BB0 = getBasicBlockByName(F, "bb0");
  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  auto &SBBB0 = *Ctxt.createSBBasicBlock(BB0);
  EXPECT_EQ(&SBBB0, Ctxt.getSBBasicBlock(BB0));
#ifndef NDEBUG
  SBBB0.verify();
#endif
  auto &SBBB1 = *Ctxt.createSBBasicBlock(BB1);
#ifndef NDEBUG
  SBBB1.verify();
#endif
  EXPECT_EQ(&SBBB1, Ctxt.getSBBasicBlock(BB1));
}

// Check SBInstruction::getParent()
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
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  SBContext Ctxt(C, AA);

  BasicBlock *BB0 = getBasicBlockByName(F, "bb0");
  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  auto *SBBB0 = Ctxt.createSBBasicBlock(BB0);
  auto *SBI0 = &*SBBB0->begin();
  EXPECT_EQ(SBI0->getParent(), SBBB0);
  auto *SBBB1 = Ctxt.createSBBasicBlock(BB1);
  auto *SBI1 = &*SBBB1->begin();
  EXPECT_EQ(SBI1->getParent(), SBBB1);

  SBI0->moveBefore(SBI1);
  EXPECT_EQ(SBI0->getParent(), SBI1->getParent());
}

// Check SBBasicBlock::getParent()
TEST_F(SandboxIRTest, SBBasicBlockGetParent) {
  parseIR(C, R"IR(
define void @foo() {
bb0:
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
  BasicBlock *BB0 = getBasicBlockByName(F, "bb0");
  auto *SBBB0 = Ctxt.getSBBasicBlock(BB0);
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
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  SBContext Ctxt(C, AA);

  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  BasicBlock *BB2 = getBasicBlockByName(F, "bb2");
  auto &SBBB1 = *Ctxt.createSBBasicBlock(BB1);
  (void)SBBB1;
#ifndef NDEBUG
  SBBB1.verify();
#endif
  auto &SBBB2 = *Ctxt.createSBBasicBlock(BB2);
  (void)SBBB2;
#ifndef NDEBUG
  SBBB2.verify();
#endif
  auto It = SBBB2.begin();
  auto *PHI = cast<SBPHINode>(&*It++);
  (void)PHI;
  auto *BinOp = cast<SBBinaryOperator>(&*It++);
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
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  SBContext Ctxt(C, AA);

  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  BasicBlock *BB2 = getBasicBlockByName(F, "bb2");
  BasicBlock *BB3 = getBasicBlockByName(F, "bb3");

  auto &SBBB1 = *Ctxt.createSBBasicBlock(BB1);
  (void)SBBB1;
#ifndef NDEBUG
  SBBB1.verify();
#endif
  auto &SBBB2 = *Ctxt.createSBBasicBlock(BB2);
  (void)SBBB2;
#ifndef NDEBUG
  SBBB2.verify();
#endif
  auto &SBBB3 = *Ctxt.createSBBasicBlock(BB3);
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
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  SBContext Ctxt(C, AA);

  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");

  auto &SBBB1 = *Ctxt.createSBBasicBlock(BB1);
  (void)SBBB1;
#ifndef NDEBUG
  SBBB1.verify();
#endif
}

TEST_F(SandboxIRTest, Pack_WithVecOp) {
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
  SBContext Ctxt(C, AA);

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

  auto &SBBB = *Ctxt.createSBBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  SBValue *SBArg0 = Ctxt.getOrCreateSBArgument(Arg0);
  SBValue *SBArg1 = Ctxt.getOrCreateSBArgument(Arg1);
  SBValue *SBArg2 = Ctxt.getOrCreateSBArgument(Arg2);
  SBValue *SBArg3 = Ctxt.getOrCreateSBArgument(Arg3);
  SBValBundle SBPackInstructions({SBArg0, SBArg1, SBArg2, SBArg3});
  auto *Pack = cast<SBPackInstruction>(
      Ctxt.createSBPackInstruction(SBPackInstructions, &SBBB));
  EXPECT_EQ(Pack->getNumOperands(), 4u);
  EXPECT_EQ(Pack->getOperand(0), SBArg0);
  EXPECT_EQ(Pack->getOperand(1), SBArg1);
  EXPECT_EQ(Pack->getOperand(2), SBArg2);
  EXPECT_EQ(Pack->getOperand(3), SBArg3);

  SBValue *SBL0 = Ctxt.getSBValue(Ld0);
  SBValue *SBL1 = Ctxt.getSBValue(Ld1);
  SBValue *SBL2 = Ctxt.getSBValue(Ld2);
  SBValue *SBL3 = Ctxt.getSBValue(Ld3);
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

TEST_F(SandboxIRTest, CheckInstructionTypes) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld = load i32, ptr %ptr
  store i32 %ld, ptr %ptr
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

  auto It = BB->begin();
  auto *Ld = &*It++;
  auto *St = &*It++;
  auto &SBBB = *Ctxt.createSBBasicBlock(BB);
  (void)SBBB;
#ifndef NDEBUG
  SBBB.verify();
#endif
  auto *SBL = Ctxt.getSBValue(Ld);
  auto *SBS = Ctxt.getSBValue(St);
  auto *SBPtr = Ctxt.getOrCreateSBValue(F.getArg(0));

  EXPECT_TRUE(isa<SBLoadInstruction>(SBL));
  EXPECT_EQ(cast<SBLoadInstruction>(SBL)->getPointerOperand(), SBPtr);
  EXPECT_TRUE(isa<SBStoreInstruction>(SBS));
  EXPECT_EQ(cast<SBStoreInstruction>(SBS)->getValueOperand(), SBL);
  EXPECT_EQ(cast<SBStoreInstruction>(SBS)->getPointerOperand(), SBPtr);
}

TEST_F(SandboxIRTest, SBPackInstruction_FoldedConst) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto It = BB->begin();
  Instruction *Ld = &*It++;
  Constant *C1 = Constant::getIntegerValue(Type::getInt32Ty(C), APInt(32, 42));
  auto &SBBB = *Ctxt.createSBBasicBlock(BB);
#ifndef NDEBUG
  SBBB.verify();
#endif
  SBValue *SBL = Ctxt.getSBValue(Ld);
  auto *SBC = Ctxt.getOrCreateSBConstant(C1);
  SBValBundle SBPackInstructions({SBC, SBL});
  auto *Pack = cast<SBPackInstruction>(
      Ctxt.createSBPackInstruction(SBPackInstructions, &SBBB));
  EXPECT_EQ(Pack->getNumOperands(), 2u);
  EXPECT_EQ(Pack->getOperand(0), SBC);
  EXPECT_EQ(Pack->getOperand(1), SBL);
#ifndef NDEBUG
  Pack->verify();
#endif

  // Check that we can update a folded constant operand.
  Constant *C2 = Constant::getIntegerValue(Type::getInt32Ty(C), APInt(32, 43));
  auto *SBC2 = Ctxt.getOrCreateSBConstant(C2);
  Pack->setOperand(0, SBC2);
  EXPECT_EQ(Pack->getOperand(0), SBC2);
}

TEST_F(SandboxIRTest, SBPackInstruction_SimpleIRPattern_InOrder) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  auto &SBBB = *Ctxt.createSBBasicBlock(BB);
  (void)SBBB;
#ifndef NDEBUG
  SBBB.verify();
#endif
  int ArgIdx = 0;
  Argument *Arg0 = F.getArg(ArgIdx++);
  Argument *Arg1 = F.getArg(ArgIdx++);
  Argument *Arg2 = F.getArg(ArgIdx++);
  Argument *Arg3 = F.getArg(ArgIdx++);
  SBArgument *SBArg0 = Ctxt.getSBArgument(Arg0);
  SBArgument *SBArg1 = Ctxt.getSBArgument(Arg1);
  SBArgument *SBArg2 = Ctxt.getSBArgument(Arg2);
  SBArgument *SBArg3 = Ctxt.getSBArgument(Arg3);
  auto *BotInsert = cast<InsertElementInst>(BB->getTerminator()->getPrevNode());
  auto *Ret = BB->getTerminator();

  auto *Pack = cast<SBPackInstruction>(Ctxt.getSBValue(BotInsert));
  auto *SBRet = cast<SBInstruction>(Ctxt.getSBValue(Ret));

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

TEST_F(SandboxIRTest, PackEraseFromParent_DropAllUses) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB = &*F.begin();
  Instruction *Ret = BB->getTerminator();
  (void)Ret;
  auto *SBF = Ctxt.createSBFunction(&F);
  (void)SBF;
  auto *SBBB = Ctxt.getSBBasicBlock(BB);
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

TEST_F(SandboxIRTest,
       SBPackInstruction_DestructorRemoveFromLLVMValueToSBValueMap) {
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
  SBContext Ctxt(C, AA);

  // Create an SBBasicBlock for BB1. This should generate a
  // SBPackInstruction.
  {
    BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
    auto &SBBB1 = *Ctxt.createSBBasicBlock(BB1);
#ifndef NDEBUG
    SBBB1.verify();
#endif
    auto *BotInsert =
        cast<InsertElementInst>(BB1->getTerminator()->getPrevNode());
    auto *Pack = cast<SBPackInstruction>(Ctxt.getSBValue(BotInsert));
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
    auto &SBBB0 = *Ctxt.createSBBasicBlock(BB0);
    (void)SBBB0;
#ifndef NDEBUG
    SBBB0.verify();
#endif
    auto It = BB0->begin();
    Instruction *Ld = &*It++;
    auto *SBL = Ctxt.getSBValue(Ld);
    SBUse Use0 = *SBL->use_begin();
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

// Checks detaching an SBBasicBlock from its underlying BB.
TEST_F(SandboxIRTest, SBBasicBlockDestruction) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, <2 x i32> %v) {
  %add = add <2 x i32> %v, %v
  %extr0 = extractelement <2 x i32> %add, i32 0
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
  auto It = BB->begin();
  auto *Add = &*It++;
  auto *Extr = &*It++;
  auto *Ret = &*It++;
  unsigned BBSize = BB->size();
  {
    auto &SBBB = *Ctxt.createSBBasicBlock(BB);
#ifndef NDEBUG
    SBBB.verify();
#endif
    Ctxt.getTracker().start(&SBBB);
    auto It = SBBB.begin();
    auto *SBAdd = Ctxt.getSBValue(Add);
    EXPECT_EQ(&*It++, SBAdd);
    auto *SBUnpk = cast<SBUnpackInstruction>(Ctxt.getSBValue(Extr));
    EXPECT_EQ(&*It++, SBUnpk);
  }
  // Check that BB is still intact.
  EXPECT_EQ(BBSize, BB->size());
  It = BB->begin();
  EXPECT_EQ(&*It++, Add);
  EXPECT_EQ(&*It++, Extr);
  EXPECT_EQ(&*It++, Ret);
  // Expect that clearing a BB does not track changes.
  EXPECT_TRUE(Ctxt.getTracker().empty());
  Ctxt.getTracker().accept();
}

// Checks detaching an SBBasicBlock from its underlying BB.
TEST_F(SandboxIRTest, SBBasicBlockDestruction_WithPack) {
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
  auto It = BB->begin();
  auto *Ld = &*It++;
  auto *Extr0 = &*It++;
  auto *Ins0 = &*It++;
  auto *Extr1 = &*It++;
  auto *Ins1 = &*It++;
  auto *Ret = &*It++;
  unsigned BBSize = BB->size();
  {
    auto &SBBB = *Ctxt.createSBBasicBlock(BB);
#ifndef NDEBUG
    SBBB.verify();
#endif
    Ctxt.getTracker().start(&SBBB);
    auto It = SBBB.begin();
    auto *SBLd = Ctxt.getSBValue(Ld);
    EXPECT_EQ(&*It++, SBLd);
    auto *Pack = Ctxt.getSBValue(Extr0);
    EXPECT_EQ(&*It++, Pack);
    auto *SBRet = Ctxt.getSBValue(Ret);
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
  EXPECT_TRUE(Ctxt.getTracker().empty());
  Ctxt.getTracker().accept();
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
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  SBContext Ctxt(C, AA);

  BasicBlock *BB0 = getBasicBlockByName(F, "bb0");
  BasicBlock *BB1 = getBasicBlockByName(F, "bb1");
  auto &SBBB0 = *Ctxt.createSBBasicBlock(BB0);
  Ctxt.getTracker().start(&SBBB0);
  auto It = BB0->begin();
  Instruction *Ld0 = &*It++;
  Instruction *Ld1 = &*It++;
  It = BB1->begin();
  Instruction *St0 = &*It++;
  auto *SBLd0 = cast<SBInstruction>(Ctxt.getSBValue(Ld0));
  auto *SBLd1 = cast<SBInstruction>(Ctxt.getSBValue(Ld1));
  auto DoRAWIf = [&]() {
    SBLd0->replaceUsesWithIf(
        SBLd1, [](SBUser *DstU, unsigned OpIdx) { return true; });
  };
  // The user is in BB1 but there is no SBBB1. Make sure it doesn't crash.
  DoRAWIf();
  // Now create the SBBB1 and try again.
  Ctxt.createSBBasicBlock(BB1);
  DoRAWIf();

  EXPECT_EQ(St0->getOperand(0), Ld1);
  SBLd1->replaceAllUsesWith(SBLd0);
  EXPECT_EQ(St0->getOperand(0), Ld0);
  Ctxt.getTracker().accept();
}

TEST_F(SandboxIRTest, SBConstant) {
  parseIR(C, R"IR(
define void @foo() {
  %add0 = add i32 42, 42
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

  BasicBlock *BB0 = &*F.begin();
  auto &SBBB = *Ctxt.createSBBasicBlock(BB0);
  Ctxt.getTracker().start(&SBBB);
  auto It = BB0->begin();
  Instruction *Add = &*It++;
  Instruction *Ret = &*It++;
  auto *C42 = cast<Constant>(Add->getOperand(0));
  SBInstruction *SBAdd = cast<SBInstruction>(Ctxt.getSBValue(Add));
  auto *SBRet = cast<SBInstruction>(Ctxt.getSBValue(Ret));
  (void)SBRet;
  SBConstant *SBC42 = cast<SBConstant>(SBAdd->getOperand(0));
  EXPECT_EQ(SBC42, SBAdd->getOperand(1));
  EXPECT_EQ(Ctxt.getSBValue(C42), SBC42);
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
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
  auto *BB0 = Ctxt.getSBBasicBlock(getBasicBlockByName(F, "bb0"));
  auto *BB1 = Ctxt.getSBBasicBlock(getBasicBlockByName(F, "bb1"));

  auto It = BB0->begin();
  auto *Gep = &*It++;
  auto *BlockAddress = cast<SBConstant>(Gep->getOperand(0));
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
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  SBContext Ctxt(C, AA);

  auto *BB0 = Ctxt.createSBBasicBlock(getBasicBlockByName(F, "bb0"));

  auto It = BB0->begin();
  auto *Gep = &*It++;
  auto *BlockAddress = cast<SBConstant>(Gep->getOperand(0));
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
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  SBContext Ctxt(C, AA);
  auto *SBF = Ctxt.createSBFunction(&F);
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
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  SBContext Ctxt(C, AA);
  auto *SBF = Ctxt.createSBFunction(&F);
  (void)SBF;
  auto *BB = &*F.begin();
  auto *GEP = &*BB->begin();
  auto *GepOpFn = GEP->getOperand(0);
  EXPECT_TRUE(isa<Constant>(GepOpFn)); // In LLVM IR a function is a constant
  auto *SBPtr = Ctxt.getOrCreateSBValue(GepOpFn);
  EXPECT_TRUE(isa<SBConstant>(SBF));
  (void)SBPtr;
}

TEST_F(SandboxIRTest, UnpackFromConstant) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB0 = &*F.begin();
  auto *Extract = &*BB0->begin();
  auto *CVec = cast<Constant>(Extract->getOperand(0));
  auto &SBBB = *Ctxt.createSBBasicBlock(BB0);
  auto It = SBBB.begin();
  auto *Unpack = &*It++;
  (void)Unpack;
  auto *Ret = &*It++;
  (void)Ret;
  auto *SBCVec = Ctxt.getSBConstant(CVec);
  EXPECT_EQ(Unpack->getOperand(0), SBCVec);
}

// Check that a vector unpack (shuffle) is recognized correctly.
TEST_F(SandboxIRTest, VectorUnpackDetection) {
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
  SBContext Ctxt(C, AA);
  auto *SBF = Ctxt.createSBFunction(&F);
  auto *BB = &*SBF->begin();
  auto It = BB->begin();
  auto *Op = &*It++;
  (void)Op;
  auto *Unpack0 = cast<SBUnpackInstruction>(&*It++);
  EXPECT_EQ(Unpack0->getUnpackLane(), 0u);
  auto *Unpack1 = cast<SBUnpackInstruction>(&*It++);
  EXPECT_EQ(Unpack1->getUnpackLane(), 1u);
  auto *Unpack2 = cast<SBUnpackInstruction>(&*It++);
  EXPECT_EQ(Unpack2->getUnpackLane(), 2u);
  EXPECT_EQ(SBUtils::getNumLanes(Unpack2), 1u);
  auto *NotUnpack0 = dyn_cast<SBUnpackInstruction>(&*It++);
  EXPECT_EQ(NotUnpack0, nullptr);
  auto *NotUnpack1 = dyn_cast<SBUnpackInstruction>(&*It++);
  EXPECT_EQ(NotUnpack1, nullptr);
  auto *NotUnpack2 = dyn_cast<SBUnpackInstruction>(&*It++);
  EXPECT_EQ(NotUnpack2, nullptr);
  auto *NotUnpack3 = dyn_cast<SBUnpackInstruction>(&*It++);
  EXPECT_EQ(NotUnpack3, nullptr);

  auto BeginIt = Unpack0->op_begin();
  auto EndIt = Unpack0->op_end();
  unsigned Cnt = 0;
  for (auto It = BeginIt; It != EndIt; ++It)
    ++Cnt;
  EXPECT_EQ(Cnt, 1u);
}

// When creating a new instruction with a constant operand, its corresponding
// SBConstant may need to be created. Check if it is missing.
TEST_F(SandboxIRTest, NoNullOperands) {
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
  SBContext Ctxt(C, AA);
  auto *SBF = Ctxt.createSBFunction(&F);
  auto *BB = &*SBF->begin();
  auto It = BB->begin();
  auto *Op = &*It++;
  auto *Unpack = cast<SBInstruction>(Ctxt.createSBUnpackInstruction(
      Op, /*Lane=*/0, BB, /*LanesToUnpack=*/1));
  for (SBValue *Op : Unpack->operands())
    EXPECT_NE(Op, nullptr);
}

TEST_F(SandboxIRTest, PackConstants) {
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
  SBContext Ctxt(C, AA);

  BasicBlock *BB0 = &*F.begin();
  auto It = BB0->begin();
  auto *Ins0 = &*It++;
  auto *Ins1 = &*It++;
  auto *C0 = cast<Constant>(Ins0->getOperand(1));
  auto *C1 = cast<Constant>(Ins1->getOperand(1));

  auto &SBBB = *Ctxt.createSBBasicBlock(BB0);
  auto It2 = SBBB.begin();
  auto *Pack = &*It2++;
  (void)Pack;
  auto *Ret = &*It2++;
  (void)Ret;
  auto *SBC0 = Ctxt.getSBConstant(C0);
  auto *SBC1 = Ctxt.getSBConstant(C1);
  EXPECT_EQ(Pack->getOperand(0), SBC0);
  EXPECT_EQ(Pack->getOperand(1), SBC1);
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
  DominatorTree DT(F);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  SBContext Ctxt(C, AA);
  auto *SBF = Ctxt.createSBFunction(&F);
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
  for (SBValue *Op : I2->operands()) {
    (void)Op;
    ++CntI2Operands;
  }
  EXPECT_EQ(CntI2Operands, 2u);
}

// Check that a vector operand to a pack that is used as a whole, is counted as
// a single operand.
//
//  I (2xwide)
//  |
// Pack
//
TEST_F(SandboxIRTest, DuplicateUsesIntoPacks) {
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
  SBContext Ctxt(C, AA);
  auto *SBF = Ctxt.createSBFunction(&F);
  auto *BB = &*SBF->begin();
  auto It = BB->begin();
  auto *Op = &*It++;
  auto *Pack = cast<SBPackInstruction>(&*It++);
  // Check Operands/Users.
  unsigned CntUsers = 0u;
  for (auto *User : Op->users()) {
    (void)User;
    ++CntUsers;
  }
  EXPECT_EQ(CntUsers, 1u);
  unsigned CntOperands = 0u;
  for (SBValue *Op : Pack->operands()) {
    (void)Op;
    ++CntOperands;
  }
  EXPECT_EQ(CntOperands, 1u);

  // Check OperandUses/Uses.
  unsigned CntUses = 0u;
  for (const SBUse &Use : Op->uses()) {
    (void)Use;
    ++CntUses;
  }
  EXPECT_EQ(CntUses, 1u);

  unsigned CntOpUses = 0u;
  for (const SBUse &OpUse : Pack->operands()) {
    (void)OpUse;
    ++CntOpUses;
  }
  EXPECT_EQ(CntOpUses, 1u);
}

TEST_F(SandboxIRTest, UsesIntoUnpack) {
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
  SBContext Ctxt(C, AA);
  auto *SBF = Ctxt.createSBFunction(&F);
  auto *BB = &*SBF->begin();
  auto It = BB->begin();
  auto *Op = &*It++;
  auto *Unpack = cast<SBUnpackInstruction>(&*It++);
  // Check Operands/Users.
  unsigned CntUsers = 0u;
  for (auto *User : Op->users()) {
    (void)User;
    ++CntUsers;
  }
  EXPECT_EQ(CntUsers, 1u);
  unsigned CntOperands = 0u;
  for (SBValue *Op : Unpack->operands()) {
    (void)Op;
    ++CntOperands;
  }
  EXPECT_EQ(CntOperands, 1u);

  // Check OperandUses/Uses.
  unsigned CntUses = 0u;
  for (const SBUse &Use : Op->uses()) {
    (void)Use;
    ++CntUses;
  }
  EXPECT_EQ(CntUses, 1u);

  unsigned CntOpUses = 0u;
  for (const SBUse &OpUse : Unpack->operands()) {
    (void)OpUse;
    ++CntOpUses;
  }
  EXPECT_EQ(CntOpUses, 1u);
}

TEST_F(SandboxIRTest, UsesIntoShuffle) {
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
  SBContext Ctxt(C, AA);
  auto *SBF = Ctxt.createSBFunction(&F);
  auto *BB = &*SBF->begin();
  auto It = BB->begin();
  auto *Op = &*It++;
  auto *Shuff = cast<SBShuffleInstruction>(&*It++);
  // Check Operands/Users.
  unsigned CntUsers = 0u;
  for (auto *User : Op->users()) {
    (void)User;
    ++CntUsers;
  }
  EXPECT_EQ(CntUsers, 1u);
  unsigned CntOperands = 0u;
  for (SBValue *Op : Shuff->operands()) {
    (void)Op;
    ++CntOperands;
  }
  EXPECT_EQ(CntOperands, 1u);

  // Check OperandUses/Uses.
  unsigned CntUses = 0u;
  for (const SBUse &Use : Op->uses()) {
    (void)Use;
    ++CntUses;
  }
  EXPECT_EQ(CntUses, 1u);

  unsigned CntOpUses = 0u;
  for (const SBUse &OpUse : Shuff->operands()) {
    (void)OpUse;
    ++CntOpUses;
  }
  EXPECT_EQ(CntOpUses, 1u);
}

TEST_F(SandboxIRTest, CheckInsertAndRemoveInstrCallback) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %val) {
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

  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &SBBB = *SBF.begin();
  SBArgument *Ptr = SBF.getArg(0);
  SBArgument *Val = SBF.getArg(1);
  SBInstruction *Ret = &SBBB.back();
  SmallVector<SBInstruction *> Inserted;
  SmallVector<SBInstruction *> Removed;
  Ctxt.registerInsertInstrCallback(
      [&Inserted](SBInstruction *SBI) { Inserted.push_back(SBI); });
  Ctxt.registerRemoveInstrCallback(
      [&Removed](SBInstruction *SBI) { Removed.push_back(SBI); });

  auto *NewI =
      SBStoreInstruction::create(Val, Ptr, /*Align=*/std::nullopt, Ret, Ctxt);
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
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  SBContext Ctxt(C, AA);

  auto &SBF = *Ctxt.createSBFunction(&F);
  auto &BB0 = *Ctxt.getSBBasicBlock(getBasicBlockByName(F, "bb0"));
  auto &BB1 = *Ctxt.getSBBasicBlock(getBasicBlockByName(F, "bb1"));
  SBArgument *Ptr = SBF.getArg(0);
  SBArgument *Val = SBF.getArg(1);

  SBInstruction *Br = &BB0.back();
  SmallVector<SBInstruction *> InsertedBB0;
  SmallVector<SBInstruction *> RemovedBB0;
  SmallVector<SBInstruction *> MovedBB0;
  Ctxt.registerInsertInstrCallbackBB(
      BB0,
      [&InsertedBB0](SBInstruction *SBI) { InsertedBB0.push_back(SBI); });
  Ctxt.registerRemoveInstrCallbackBB(
      BB0,
      [&RemovedBB0](SBInstruction *SBI) { RemovedBB0.push_back(SBI); });
  Ctxt.registerMoveInstrCallbackBB(
      BB0,
      [&MovedBB0](SBInstruction *SBI, SBBasicBlock &BB,
                  const SBBBIterator &It) { MovedBB0.push_back(SBI); });

  Ctxt.registerInsertInstrCallbackBB(
      BB1, [](SBInstruction *SBI) { llvm_unreachable("Shouldn't run!"); });
  Ctxt.registerRemoveInstrCallbackBB(
      BB1, [](SBInstruction *SBI) { llvm_unreachable("Shouldn't run!"); });
  Ctxt.registerRemoveInstrCallbackBB(
      BB1, [](SBInstruction *SBI) { llvm_unreachable("Shouldn't run!"); });

  auto *NewI =
      SBStoreInstruction::create(Val, Ptr, /*Align=*/std::nullopt, Br, Ctxt);
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
  DominatorTree DT(F);
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  AssumptionCache AC(F);
  BasicAAResult BAA(DL, F, TLI, AC, &DT);
  AAResults AA(TLI);
  AA.addAAResult(BAA);
  SBContext Ctxt(C, AA);

  auto &SBF = *Ctxt.createSBFunction(&F);
  (void)SBF;
  auto *SBBB0 = Ctxt.getSBBasicBlock(getBasicBlockByName(F, "bb0"));
  auto *SBBB1 = Ctxt.getSBBasicBlock(getBasicBlockByName(F, "bb1"));
  auto *SBBB2 = Ctxt.getSBBasicBlock(getBasicBlockByName(F, "bb2"));
  auto *SBBB3 = Ctxt.getSBBasicBlock(getBasicBlockByName(F, "bb3"));

  EXPECT_EQ(SBBB0, GraphTraits<SBBasicBlock *>::getEntryNode(SBBB0));
  auto ChildIt = GraphTraits<SBBasicBlock *>::child_begin(SBBB0);
  EXPECT_EQ(SBBB1, *ChildIt++);
  EXPECT_EQ(SBBB2, *ChildIt++);
  EXPECT_EQ(ChildIt, GraphTraits<SBBasicBlock *>::child_end(SBBB0));
  EXPECT_EQ(*GraphTraits<SBBasicBlock *>::child_begin(SBBB1), SBBB3);
}
