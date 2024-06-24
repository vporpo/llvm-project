//===- VectorizeScalarsTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Passes/VectorizeScalars.h"
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
#include "llvm/Transforms/Vectorize/SandboxVec/Region.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIRVec.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("SchedulerTest", errs());
  return Mod;
}

/// \Returns true if there is a dependency path of any type FromN->...->ToN.
static bool dependencyPath(sandboxir::DependencyGraph::Node *FromN,
                           sandboxir::DependencyGraph::Node *ToN) {
  std::list<sandboxir::DependencyGraph::Node *> Worklist;
  for (auto *PredN : ToN->preds())
    Worklist.push_back(PredN);

  while (!Worklist.empty()) {
    auto *N = Worklist.front();
    Worklist.pop_front();

    if (N == FromN)
      return true;

    for (auto *PredN : N->preds())
      Worklist.push_back(PredN);
  }
  return false;
}

// Check that we emit the necessary unpacks that connect a vector instruction to
// its scalar external users while vectorizing bottom-up.
TEST(VectorizeScalars, UnpacksForExternalUsers) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptrA, ptr %ptrB) {
  %ptrA0 = getelementptr float, ptr %ptrA, i32 0
  %ptrA1 = getelementptr float, ptr %ptrA, i32 1

  %ldA0 = load float, ptr %ptrA0
  %ldA1 = load float, ptr %ptrA1

  %extUse = fadd float %ldA0, %ldA0

  store float %ldA0, ptr %ptrA0
  store float %ldA1, ptr %ptrA1
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
  sandboxir::SBVecContext Ctx(C, AA);

  auto *SBF = Ctx.createFunction(&F);
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  auto *Gep0 = &*It++;
  (void)Gep0;
  auto *Gep1 = &*It++;
  (void)Gep1;
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *ExtUse = &*It++;
  auto *St0 = &*It++;
  auto *St1 = &*It++;

  sandboxir::VectorizeFromSeeds Vec(SBBB, Ctx, SE, DL, TTI);
  SmallPtrSet<sandboxir::Instruction *, 4> EraseCandidates;
  sandboxir::Analysis &Analysis = Vec.getAnalysis();
  sandboxir::Region DummyRgn(*SBBB, Ctx, TTI);
  auto &InstrMaps = sandboxir::VectorizeFromSeedsAttorney::getInstrMaps(Vec);
  InstrMaps.setRegion(DummyRgn);

  auto VecBundle = [&Analysis, &InstrMaps, &EraseCandidates,
                    &Vec](DmpVector<sandboxir::Value *> Bndl) {
    auto *AnalysisResult = Analysis.getBndlAnalysis(
        Bndl, InstrMaps, EraseCandidates, /*TopDown=*/false);
    return sandboxir::VectorizeFromSeedsAttorney::vectorizeRec(
        Vec, Bndl, std::move(AnalysisResult), EraseCandidates,
        /*TopDown=*/false);
  };
  auto *VecSt = cast<sandboxir::Instruction>(
      VecBundle(DmpVector<sandboxir::Value *>({St0, St1})));
  (void)VecSt;
  auto *VecLd = cast<sandboxir::Instruction>(
      VecBundle(DmpVector<sandboxir::Value *>({Ld0, Ld1})));
  auto *Sched = Ctx.getScheduler(SBBB);
  auto &DAG = Sched->getDAG();

  auto *Ld0N = DAG.getNode(Ld0);
  auto *ExtN = DAG.getNode(ExtUse);
  EXPECT_FALSE(ExtN->dependsOn(Ld0N));
  auto *VecLdN = DAG.getNode(VecLd);
  EXPECT_TRUE(dependencyPath(VecLdN, ExtN));
}

// Check that no unpacks are emitted for internal users while vectorizing
// bottom-up.
TEST(VectorizeScalars, NoUnpacksForInternalUsers) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptrA, ptr %ptrB) {
  %ptrA0 = getelementptr float, ptr %ptrA, i32 0
  %ptrA1 = getelementptr float, ptr %ptrA, i32 1

  %ldA0 = load float, ptr %ptrA0
  %ldA1 = load float, ptr %ptrA1
  store float %ldA0, ptr %ptrA0
  store float %ldA1, ptr %ptrA1
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
  sandboxir::SBVecContext Ctx(C, AA);

  auto *SBF = Ctx.createFunction(&F);
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  auto *Gep0 = &*It++;
  (void)Gep0;
  auto *Gep1 = &*It++;
  (void)Gep1;
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *St0 = &*It++;
  auto *St1 = &*It++;

  sandboxir::VectorizeFromSeeds Vec(SBBB, Ctx, SE, DL, TTI);
  SmallPtrSet<sandboxir::Instruction *, 4> EraseCandidates;
  sandboxir::Analysis &Analysis = Vec.getAnalysis();
  auto &InstrMaps = sandboxir::VectorizeFromSeedsAttorney::getInstrMaps(Vec);
  sandboxir::Region DummyRgn(*SBBB, Ctx, TTI);
  InstrMaps.setRegion(DummyRgn);

  auto VecBundle = [&Analysis, &InstrMaps, &EraseCandidates,
                    &Vec](DmpVector<sandboxir::Value *> Bndl) {
    auto AnalysisResult = Analysis.getBndlAnalysis(
        Bndl, InstrMaps, EraseCandidates, /*TopDown=*/false);
    return sandboxir::VectorizeFromSeedsAttorney::vectorizeRec(
        Vec, Bndl, std::move(AnalysisResult), EraseCandidates,
        /*TopDown=*/false);
  };
  auto *VecSt = cast<sandboxir::Instruction>(
      VecBundle(DmpVector<sandboxir::Value *>({St0, St1})));
  (void)VecSt;
  auto *VecLd = cast<sandboxir::Instruction>(
      VecBundle(DmpVector<sandboxir::Value *>({Ld0, Ld1})));
  EXPECT_EQ(VecLd->getNumUsers(), 1u);
  EXPECT_EQ(VecSt->getOperand(0), VecLd);
}

TEST(VectorizeScalars, ResetViewBeforeWeTryVectorize) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ptr0 = getelementptr float, ptr %ptr, i32 0
  %ptr1 = getelementptr float, ptr %ptr, i32 1

  %ld0 = load float, ptr %ptr0
  %ld1 = load float, ptr %ptr1

  store float %ld0, ptr %ptr0
  store float %ld1, ptr %ptr1
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
  sandboxir::SBVecContext Ctx(C, AA);

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *Gep0 = &*It++;
  (void)Gep0;
  auto *Gep1 = &*It++;
  (void)Gep1;
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *St0 = &*It++;
  auto *St1 = &*It++;

  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  Sched.startTracking(BB);
  sandboxir::VectorizeFromSeeds Vectorizer(BB, Ctx, SE, DL, TTI);
  SmallPtrSet<sandboxir::Instruction *, 4> EraseCandidates;
  DmpVector<sandboxir::Value *> LdSeeds{Ld0, Ld1};
  {
    sandboxir::Region Rgn(*BB, Ctx, TTI);
    Vectorizer.tryVectorize(LdSeeds, Rgn, EraseCandidates);
  }
  {
    DmpVector<sandboxir::Value *> StSeeds{St0, St1};
    sandboxir::Region Rgn(*BB, Ctx, TTI);
    Vectorizer.tryVectorize(StSeeds, Rgn, EraseCandidates);
  }
}

TEST(VectorizeScalars, DontCrashOnOpaqueFPMath) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
declare <2 x float> @bar()
define void @foo(float %v) {
  %bar0 = call <2 x float> @bar()
  %opq = insertelement <2 x float> <float poison, float 0.000000e+00>, float %v, i64 0
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
  sandboxir::SBVecContext Ctx(C, AA);

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *Opq0 = &*It++;
  auto *Opq1 = &*It++;

  ASSERT_TRUE(isa<sandboxir::CallInst>(Opq0));
  ASSERT_TRUE(isa<sandboxir::InsertElementInst>(Opq1));

  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  Sched.startTracking(BB);
  sandboxir::VectorizeFromSeeds Vectorizer(BB, Ctx, SE, DL, TTI);
  SmallPtrSet<sandboxir::Instruction *, 4> EraseCandidates;
  DmpVector<sandboxir::Value *> Seeds{Opq0, Opq1};
  sandboxir::Region Rgn(*BB, Ctx, TTI);
  EXPECT_FALSE(Vectorizer.tryVectorize(Seeds, Rgn, EraseCandidates));
}

TEST(VectorizeScalars, GetUsesBundles_Simple) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptrA0, ptr %ptrA1) {
  %ldA0 = load i32, ptr %ptrA0
  %ldA1 = load i32, ptr %ptrA1
  %sub0 = sub i32 %ldA0, 1
  %sub1 = sub i32 %ldA1, 2
  %add0 = sub i32 %ldA1, 3
  %add1 = sub i32 %ldA0, 4
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
  sandboxir::SBVecContext Ctx(C, AA);

  auto *SBF = Ctx.createFunction(&F);
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *Sub0 = &*It++;
  auto *Sub1 = &*It++;
  auto *Add0 = &*It++;
  auto *Add1 = &*It++;
  DmpVector<sandboxir::Value *> Bndl({Ld0, Ld1});
  auto Pair = getUsesBundlesPicky(Bndl);
  bool Fail = Pair.second;
  EXPECT_FALSE(Fail);
  auto &Vec = Pair.first;
  EXPECT_EQ(Vec.size(), 2u);
  {
    auto &UseVec = Vec[0];
    EXPECT_EQ(UseVec[0].getUser(), Add1);
    EXPECT_EQ(UseVec[1].getUser(), Add0);
  }
  {
    auto &UseVec = Vec[1];
    EXPECT_EQ(UseVec[0].getUser(), Sub0);
    EXPECT_EQ(UseVec[1].getUser(), Sub1);
  }
}

TEST(VectorizeScalars, GetUsesBundles_Fail_SameUser) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptrA0, ptr %ptrA1) {
  %ldA0 = load i32, ptr %ptrA0
  %ldA1 = load i32, ptr %ptrA1
  %sub0 = sub i32 %ldA0, 1
  %sub1 = sub i32 %ldA0, 2
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
  sandboxir::SBVecContext Ctx(C, AA);

  auto *SBF = Ctx.createFunction(&F);
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  DmpVector<sandboxir::Value *> Bndl({Ld0, Ld1});
  auto Pair = getUsesBundlesPicky(Bndl);
  bool Fail = Pair.second;
  EXPECT_TRUE(Fail);
}

TEST(VectorizeScalars, GetUsesBundles_Fail_DifferentOperandNo) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptrA0, ptr %ptrA1) {
  %ldA0 = load i32, ptr %ptrA0
  %ldA1 = load i32, ptr %ptrA1
  %sub0 = sub i32 %ldA0, 1
  %sub1 = sub i32 2, %ldA1
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
  sandboxir::SBVecContext Ctx(C, AA);

  auto *SBF = Ctx.createFunction(&F);
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  DmpVector<sandboxir::Value *> Bndl({Ld0, Ld1});
  auto Pair = getUsesBundlesPicky(Bndl);
  bool Fail = Pair.second;
  EXPECT_TRUE(Fail);
}

TEST(VectorizeScalars, SimpleTopDown) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptrA, ptr %ptrB) {
  %ptrA0 = getelementptr float, ptr %ptrA, i32 0
  %ptrA1 = getelementptr float, ptr %ptrA, i32 1

  %ldA0 = load float, ptr %ptrA0
  %ldA1 = load float, ptr %ptrA1

  store float %ldA0, ptr %ptrA0
  store float %ldA1, ptr %ptrA1
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
  sandboxir::SBVecContext Ctx(C, AA);

  auto *SBF = Ctx.createFunction(&F);
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  auto *Gep0 = &*It++;
  (void)Gep0;
  auto *Gep1 = &*It++;
  (void)Gep1;
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *St0 = &*It++;
  auto *St1 = &*It++;

  sandboxir::VectorizeFromSeeds Vec(SBBB, Ctx, SE, DL, TTI);
  SmallPtrSet<sandboxir::Instruction *, 4> EraseCandidates;
  sandboxir::Analysis &Analysis = Vec.getAnalysis();
  sandboxir::Region DummyRgn(*SBBB, Ctx, TTI);
  auto &InstrMaps = sandboxir::VectorizeFromSeedsAttorney::getInstrMaps(Vec);
  InstrMaps.setRegion(DummyRgn);

  auto VecBundle = [&Analysis, &InstrMaps, &EraseCandidates,
                    &Vec](DmpVector<sandboxir::Value *> Bndl) {
    auto *AnalysisResult = Analysis.getBndlAnalysis(
        Bndl, InstrMaps, EraseCandidates, /*TopDown=*/true);
    return sandboxir::VectorizeFromSeedsAttorney::vectorizeRec(
        Vec, Bndl, std::move(AnalysisResult), EraseCandidates,
        /*TopDown=*/true);
  };
  auto *VecLd = cast<sandboxir::Instruction>(
      VecBundle(DmpVector<sandboxir::Value *>({Ld0, Ld1})));
  EXPECT_NE(VecLd, nullptr);
  EXPECT_TRUE(isa<VectorType>(VecLd->getType()));
  auto *VecSt = cast<sandboxir::Instruction>(
      VecBundle(DmpVector<sandboxir::Value *>({St0, St1})));
  EXPECT_NE(VecSt, nullptr);
  EXPECT_TRUE(isa<VectorType>(cast<sandboxir::StoreInst>(VecSt)
                                  ->getValueOperand()
                                  ->getType()));
}
