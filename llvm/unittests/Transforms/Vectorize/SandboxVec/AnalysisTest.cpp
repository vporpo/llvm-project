//===- AnalysisTest.cpp ---------------------------------------------===//
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
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/SandboxVec/InstrMaps.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("AnalysisTest", errs());
  return Mod;
}

BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    if (BB.getName() == Name)
      return &BB;
  llvm_unreachable("Expected to find basic block!");
}

TEST(Analysis, TopDown_UnpackTopDown) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptrA, ptr %ptrB) {
  %ptrA0 = getelementptr float, ptr %ptrA, i32 0
  %ptrA1 = getelementptr float, ptr %ptrA, i32 1
  %ldA0 = load float, ptr %ptrA0
  %ldA1 = load float, ptr %ptrA1
  store float %ldA0, ptr %ptrA0
  store float %ldA1, ptr %ptrB
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
  TargetTransformInfo TTI(DL);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = cast<SBBasicBlock>(&*SBF->begin());
  auto It = SBBB->begin();
  auto *Gep0 = &*It++;
  (void)Gep0;
  auto *Gep1 = &*It++;
  (void)Gep1;
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *St0 = &*It++;
  auto *St1 = &*It++;

  InstructionMaps InstrMaps;
  SBAnalysis Analysis(SE, DL);
  SmallPtrSet<SBInstruction *, 4> ProbablyDead;
  Analysis.init(*SBBB);
  auto *Res = Analysis.getBndlAnalysis({Ld0, Ld1}, InstrMaps, ProbablyDead,
                                       /*TopDown=*/true);
  EXPECT_TRUE(Res->getSubclassID() == ResultID::SimpleWiden);

  Res = Analysis.getBndlAnalysis({St0, St1}, InstrMaps, ProbablyDead,
                                 /*TopDown=*/true);
  EXPECT_TRUE(Res->getSubclassID() == ResultID::UnpackTopDown);
}

TEST(Analysis, TopDown_CrossBB) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptrA, ptr %ptrB) {
bb0:
  %ptrA0 = getelementptr float, ptr %ptrA, i32 0
  %ptrA1 = getelementptr float, ptr %ptrA, i32 1
  %ldA0 = load float, ptr %ptrA0
  %ldA1 = load float, ptr %ptrA1
  br label %bb1

bb1:
  store float %ldA0, ptr %ptrA0
  store float %ldA1, ptr %ptrB
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
  TargetTransformInfo TTI(DL);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
  (void)SBF;
  auto *BB0 = getBasicBlockByName(F, "bb0");
  auto *BB1 = getBasicBlockByName(F, "bb1");
  auto *SBBB0 = Ctxt.getSBBasicBlock(BB0);
  auto It = SBBB0->begin();
  auto *Gep0 = &*It++;
  (void)Gep0;
  auto *Gep1 = &*It++;
  (void)Gep1;
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;

  auto *SBBB1 = Ctxt.getSBBasicBlock(BB1);
  It = SBBB1->begin();
  auto *St0 = &*It++;
  auto *St1 = &*It++;

  InstructionMaps InstrMaps;
  SBAnalysis Analysis(SE, DL);
  SmallPtrSet<SBInstruction *, 4> ProbablyDead;
  Analysis.init(*SBBB0);
  auto *Res = Analysis.getBndlAnalysis({Ld0, Ld1}, InstrMaps, ProbablyDead,
                                       /*TopDown=*/true);
  EXPECT_TRUE(Res->getSubclassID() == ResultID::SimpleWiden);

  Res = Analysis.getBndlAnalysis({St0, St1}, InstrMaps, ProbablyDead,
                                 /*TopDown=*/true);
  // For now we don't allow crossing BBs.
  EXPECT_TRUE(Res->getSubclassID() == ResultID::UnpackTopDown);
}
