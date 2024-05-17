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
#include "llvm/Transforms/Vectorize/SandboxVec/SBRegion.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"
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
static bool dependencyPath(DependencyGraph::Node *FromN,
                           DependencyGraph::Node *ToN) {
  std::list<DependencyGraph::Node *> Worklist;
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
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
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

  VectorizeFromSeeds Vec(SBBB, Ctxt, SE, DL, TTI);
  SmallPtrSet<SBInstruction *, 4> EraseCandidates;
  SBAnalysis &Analysis = Vec.getAnalysis();
  SBRegion DummyRgn(*SBBB, Ctxt, TTI);
  auto &InstrMaps = VectorizeFromSeedsAttorney::getInstrMaps(Vec);
  InstrMaps.setRegion(DummyRgn);

  auto VecBundle = [&Analysis, &InstrMaps, &EraseCandidates,
                    &Vec](SBValBundle Bndl) {
    auto *AnalysisResult = Analysis.getBndlAnalysis(
        Bndl, InstrMaps, EraseCandidates, /*TopDown=*/false);
    return VectorizeFromSeedsAttorney::vectorizeRec(
        Vec, Bndl, std::move(AnalysisResult), EraseCandidates,
        /*TopDown=*/false);
  };
  auto *VecSt = cast<SBInstruction>(VecBundle(SBValBundle({St0, St1})));
  (void)VecSt;
  auto *VecLd = cast<SBInstruction>(VecBundle(SBValBundle({Ld0, Ld1})));
  auto *Sched = Ctxt.getScheduler(SBBB);
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
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
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

  VectorizeFromSeeds Vec(SBBB, Ctxt, SE, DL, TTI);
  SmallPtrSet<SBInstruction *, 4> EraseCandidates;
  SBAnalysis &Analysis = Vec.getAnalysis();
  auto &InstrMaps = VectorizeFromSeedsAttorney::getInstrMaps(Vec);
  SBRegion DummyRgn(*SBBB, Ctxt, TTI);
  InstrMaps.setRegion(DummyRgn);

  auto VecBundle = [&Analysis, &InstrMaps, &EraseCandidates,
                    &Vec](SBValBundle Bndl) {
    auto AnalysisResult = Analysis.getBndlAnalysis(
        Bndl, InstrMaps, EraseCandidates, /*TopDown=*/false);
    return VectorizeFromSeedsAttorney::vectorizeRec(
        Vec, Bndl, std::move(AnalysisResult), EraseCandidates,
        /*TopDown=*/false);
  };
  auto *VecSt = cast<SBInstruction>(VecBundle(SBValBundle({St0, St1})));
  (void)VecSt;
  auto *VecLd = cast<SBInstruction>(VecBundle(SBValBundle({Ld0, Ld1})));
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
  SBContext Ctxt(C, AA);

  auto *BB = Ctxt.createSBBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *Gep0 = &*It++;
  (void)Gep0;
  auto *Gep1 = &*It++;
  (void)Gep1;
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *St0 = &*It++;
  auto *St1 = &*It++;

  Scheduler &Sched = *Ctxt.getScheduler(BB);
  Sched.startTracking(BB);
  VectorizeFromSeeds Vectorizer(BB, Ctxt, SE, DL, TTI);
  SmallPtrSet<SBInstruction *, 4> EraseCandidates;
  SBValBundle LdSeeds{Ld0, Ld1};
  {
    SBRegion Rgn(*BB, Ctxt, TTI);
    Vectorizer.tryVectorize(LdSeeds, Rgn, EraseCandidates);
  }
  {
    SBValBundle StSeeds{St0, St1};
    SBRegion Rgn(*BB, Ctxt, TTI);
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
  SBContext Ctxt(C, AA);

  auto *BB = Ctxt.createSBBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *Opq0 = &*It++;
  auto *Opq1 = &*It++;

  // If calls are no longer Opaque, we should probably drop this test.
  ASSERT_TRUE(isa<SBOpaqueInstruction>(Opq0));
  ASSERT_TRUE(isa<SBOpaqueInstruction>(Opq1));

  Scheduler &Sched = *Ctxt.getScheduler(BB);
  Sched.startTracking(BB);
  VectorizeFromSeeds Vectorizer(BB, Ctxt, SE, DL, TTI);
  SmallPtrSet<SBInstruction *, 4> EraseCandidates;
  SBValBundle Seeds{Opq0, Opq1};
  SBRegion Rgn(*BB, Ctxt, TTI);
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
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *Sub0 = &*It++;
  auto *Sub1 = &*It++;
  auto *Add0 = &*It++;
  auto *Add1 = &*It++;
  SBValBundle Bndl({Ld0, Ld1});
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
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  SBValBundle Bndl({Ld0, Ld1});
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
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
  auto *SBBB = &*SBF->begin();
  auto It = SBBB->begin();
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  SBValBundle Bndl({Ld0, Ld1});
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
  SBContext Ctxt(C, AA);

  auto *SBF = Ctxt.createSBFunction(&F);
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

  VectorizeFromSeeds Vec(SBBB, Ctxt, SE, DL, TTI);
  SmallPtrSet<SBInstruction *, 4> EraseCandidates;
  SBAnalysis &Analysis = Vec.getAnalysis();
  SBRegion DummyRgn(*SBBB, Ctxt, TTI);
  auto &InstrMaps = VectorizeFromSeedsAttorney::getInstrMaps(Vec);
  InstrMaps.setRegion(DummyRgn);

  auto VecBundle = [&Analysis, &InstrMaps, &EraseCandidates,
                    &Vec](SBValBundle Bndl) {
    auto *AnalysisResult = Analysis.getBndlAnalysis(
        Bndl, InstrMaps, EraseCandidates, /*TopDown=*/true);
    return VectorizeFromSeedsAttorney::vectorizeRec(
        Vec, Bndl, std::move(AnalysisResult), EraseCandidates,
        /*TopDown=*/true);
  };
  auto *VecLd = cast<SBInstruction>(VecBundle(SBValBundle({Ld0, Ld1})));
  EXPECT_NE(VecLd, nullptr);
  EXPECT_TRUE(isa<VectorType>(VecLd->getType()));
  auto *VecSt = cast<SBInstruction>(VecBundle(SBValBundle({St0, St1})));
  EXPECT_NE(VecSt, nullptr);
  EXPECT_TRUE(isa<VectorType>(
      cast<SBStoreInstruction>(VecSt)->getValueOperand()->getType()));
}
