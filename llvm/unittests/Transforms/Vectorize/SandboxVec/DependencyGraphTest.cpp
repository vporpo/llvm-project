//===- DependencyGraphTest.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/DependencyGraph.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
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

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("DependencyGraphTest", errs());
  return Mod;
}

#ifndef NDEBUG
static BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    if (BB.getName() == Name)
      return &BB;
  llvm_unreachable("Expected to find basic block!");
}
#endif

#ifndef NDEBUG
// TODO: Replace this with Node::dependsOn()
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
#endif

/// Checks that \p I's predecessors match \p ExpectedPredInstrs.
static void
expect_deps(const sandboxir::DependencyGraph &DAG, sandboxir::Instruction *I,
            const SmallVector<sandboxir::Instruction *> &ExpectedPredInstrs) {
  auto *N = DAG.getNode(I);
  EXPECT_NE(N, nullptr);
  SmallVector<sandboxir::DependencyGraph::Node *> ExpectedPreds;
  ExpectedPreds.reserve(ExpectedPredInstrs.size());
  for (auto *PredI : ExpectedPredInstrs) {
    auto *PredN = DAG.getNode(PredI);
    EXPECT_NE(PredN, nullptr);
    ExpectedPreds.push_back(PredN);
  }
  SmallVector<sandboxir::DependencyGraph::Node *> Preds(N->preds().begin(),
                                                        N->preds().end());
  bool IsPermutation =
      Preds.size() == ExpectedPreds.size() &&
      std::is_permutation(Preds.begin(), Preds.end(), ExpectedPreds.begin());
  if (!IsPermutation) {
#ifndef NDEBUG
    errs() << "Instruction:\n" << *I << "\n";
    errs() << "Expected Preds:\n";
    for (auto *N : ExpectedPreds)
      errs().indent(4) << *N->getInstruction() << "\n";
    errs() << "Actual Preds:\n";
    for (auto *N : Preds)
      errs().indent(4) << *N->getInstruction() << "\n";
#endif
  }
  EXPECT_TRUE(IsPermutation);
}

TEST(DependencyGraph, NodeTest1) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v, ptr noalias %ptr1, ptr noalias %ptr2, ptr noalias %ptr3) {
  %ld = load float, ptr %ptr1
  store float %ld, ptr %ptr2
  store float %ld, ptr %ptr3
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
  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto *Ld1 = &*std::next(BB->begin(), 0);
  auto *St1 = &*std::next(BB->begin(), 1);
  auto *St2 = &*std::next(BB->begin(), 2);
  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});

  auto *L1 = DAG.getNode(Ld1);
  auto *S1 = DAG.getNode(St1);
  auto *S2 = DAG.getNode(St2);

  DenseSet<sandboxir::DependencyGraph::Node *> Succs;
  for (auto *SuccN : L1->succs())
    Succs.insert(SuccN);
  EXPECT_EQ(Succs.size(), 2u);
  EXPECT_TRUE(Succs.count(S1));
  EXPECT_TRUE(Succs.count(S2));

  DenseSet<sandboxir::DependencyGraph::Node *> S1Preds;
  for (auto *PredN : S1->preds())
    S1Preds.insert(PredN);
  EXPECT_EQ(S1Preds.size(), 1u);
  EXPECT_TRUE(S1Preds.count(L1));

  DenseSet<sandboxir::DependencyGraph::Node *> S2Preds;
  for (auto *PredN : S2->preds())
    S2Preds.insert(PredN);
  EXPECT_EQ(S2Preds.size(), 1u);
  EXPECT_TRUE(S2Preds.count(L1));
}

TEST(DependencyGraph, NodeTest2) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v, ptr noalias %ptr1, ptr noalias %ptr2) {
  %ld1 = load float, ptr %ptr1
  %ld2 = load float, ptr %ptr2
  %add = fadd float %ld1, %ld2
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto *Ld1 = &*std::next(BB->begin(), 0);
  auto *Ld2 = &*std::next(BB->begin(), 1);
  auto *Add = &*std::next(BB->begin(), 2);
  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});

  auto *L1 = DAG.getNode(Ld1);
  auto *L2 = DAG.getNode(Ld2);
  auto *A = DAG.getNode(Add);

  DenseSet<sandboxir::DependencyGraph::Node *> Preds;
  for (auto *PredN : A->preds())
    Preds.insert(PredN);
  EXPECT_EQ(Preds.size(), 2u);
  EXPECT_TRUE(Preds.count(L1));
  EXPECT_TRUE(Preds.count(L2));

  DenseSet<sandboxir::DependencyGraph::Node *> L1Succs;
  for (auto *SuccN : L1->succs())
    L1Succs.insert(SuccN);
  EXPECT_EQ(L1Succs.size(), 1u);
  EXPECT_TRUE(L1Succs.count(A));

  DenseSet<sandboxir::DependencyGraph::Node *> L2Succs;
  for (auto *SuccN : L2->succs())
    L2Succs.insert(SuccN);
  EXPECT_EQ(L2Succs.size(), 1u);
  EXPECT_TRUE(L2Succs.count(A));
}

#ifndef NDEBUG
TEST(DependencyGraph, SimpleWithCall) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
declare void @bar1()
declare void @bar2()
define void @foo(float %v1, float %v2) {
  call void @bar1()
  %add = fadd float %v1, %v2
  call void @bar2()
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

  sandboxir::BasicBlock *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  DAG.extend(DmpVector<sandboxir::Value *>{
      &*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *Call1 = &*It++;
  auto *Add = &*It++;
  auto *Call2 = &*It++;
  expect_deps(DAG, Call1, {});
  expect_deps(DAG, Add, {});
  expect_deps(DAG, Call2, {Call1});
}

TEST(DependencyGraph, SimpleLoadStore) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v, ptr %ptr1, ptr %ptr2) {
  store float %v, ptr %ptr1
  %ld1 = load float, ptr %ptr1
  %ld2 = load float, ptr %ptr2
  store float %v, ptr %ptr2
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

  sandboxir::BasicBlock *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  DAG.extend(DmpVector<sandboxir::Value *>{
      &*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *St1 = &*It++;
  auto *Ld1 = &*It++;
  auto *Ld2 = &*It++;
  auto *St2 = &*It++;

  expect_deps(DAG, St1, {});
  expect_deps(DAG, Ld1, {St1});
  expect_deps(DAG, Ld2, {St1});
  expect_deps(DAG, St2, {Ld1, Ld2, St1});
}

// Same as above but with noalias %ptr1 and %ptr2
TEST(DependencyGraph, SimpleLoadStoreNoAlias) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v, ptr noalias %ptr1, ptr noalias %ptr2) {
  store float %v, ptr %ptr1
  %ld1 = load float, ptr %ptr1
  %ld2 = load float, ptr %ptr2
  store float %v, ptr %ptr2
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

  sandboxir::BasicBlock *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  DAG.extend(DmpVector<sandboxir::Value *>{
      &*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *St1 = &*It++;
  auto *Ld1 = &*It++;
  auto *Ld2 = &*It++;
  auto *St2 = &*It++;

  expect_deps(DAG, St1, {});
  expect_deps(DAG, Ld1, {St1});
  expect_deps(DAG, Ld2, {});
  expect_deps(DAG, St2, {Ld2});
}

// Make sure there is a dependency between volatile loads.
TEST(DependencyGraph, VolatileLoads) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v, ptr noalias %ptr1, ptr noalias %ptr2) {
  %ld1 = load volatile float, ptr %ptr1
  %ld2 = load volatile float, ptr %ptr2
  %add = fadd float %ld1, %ld2
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

  sandboxir::BasicBlock *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Ld1 = &*It++;
  auto *Ld2 = &*It++;
  auto *Add = &*It++;
  DAG.extend(DmpVector<sandboxir::Value *>{Ld1, Add});

  expect_deps(DAG, Ld1, {});
  expect_deps(DAG, Ld2, {Ld1});
  expect_deps(DAG, Add, {Ld1, Ld2});
}

// Make sure there is a dependency between volatile stores.
TEST(DependencyGraph, VolatileStores) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v, ptr noalias %ptr1, ptr noalias %ptr2) {
  store volatile float %v, ptr %ptr1, align 4
  store volatile float %v, ptr %ptr2, align 4
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

  sandboxir::BasicBlock *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *St1 = &*It++;
  auto *St2 = &*It++;
  DAG.extend(DmpVector<sandboxir::Value *>{St1, St2});

  expect_deps(DAG, St1, {});
  expect_deps(DAG, St2, {St1});
}

TEST(DependencyGraph, CallDeps) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
declare void @bar()
define void @foo(float %v, ptr noalias %ptr1, ptr noalias %ptr2) {
  store float %v, ptr %ptr1, align 4
  call void @bar()
  store float %v, ptr %ptr2, align 4
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

  sandboxir::BasicBlock *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  DAG.extend(DmpVector<sandboxir::Value *>{&*BB->begin(), &*BB->rbegin()});

  auto It = BB->begin();
  auto *St1 = &*It++;
  auto *Call = &*It++;
  auto *St2 = &*It++;
  auto *Ret = &*It++;

  expect_deps(DAG, St1, {});
  expect_deps(DAG, Call, {St1});
  expect_deps(DAG, St2, {Call});
  expect_deps(DAG, Ret, {});
}

// Don't add memory dep if there is already a use-def edge to the same node.
TEST(DependencyGraph, DuplicateEdges) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
declare float @bar()

define void @foo(ptr %ptr) {
  %call = call float @bar()
  store volatile float %call, ptr %ptr, align 4
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

  sandboxir::BasicBlock *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  DAG.extend(DmpVector<sandboxir::Value *>{
      &*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *Call = &*It++;
  auto *St = &*It++;

  expect_deps(DAG, Call, {});
  expect_deps(DAG, St, {Call});
}

// The DAG should not include any debug instructions.
TEST(DependencyGraph, SkipDebug) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
declare void @llvm.dbg.value(metadata, metadata, metadata)

define void @foo(float %v, ptr noalias %ptr1, ptr noalias %ptr2) {
  store volatile float %v, ptr %ptr1, align 4
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !13
  store volatile float %v, ptr %ptr2, align 4
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t2.c", directory: "foo")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocalVariable(name: "x", scope: !8, file: !1, line: 2, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 2, column: 7, scope: !8)
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

  sandboxir::BasicBlock *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  DAG.extend(DmpVector<sandboxir::Value *>{
      &*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *St1 = &*It++;
  sandboxir::Instruction *I = &*It++;
  sandboxir::Instruction *St2;
  if (I->isDbgInfo()) {
    // This won't happen with DebugRecords.
    St2 = &*It++;
  } else {
    St2 = I;
  }
  expect_deps(DAG, St1, {});
  expect_deps(DAG, St2, {St1});
}
#endif // NDEBUG

// Check that an inalloc alloca instruction depends on stacksave and
// stackrestore
TEST(DependencyGraph, StackSaveRestoreInAlloca) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
declare ptr @llvm.stacksave()
declare void @llvm.stackrestore(ptr %ptr)

define void @foo() {
  %stack0 = call ptr @llvm.stacksave()        ; Should depend on store
  %alloca0 = alloca inalloca i8               ; Should depend on stacksave
  call void @llvm.stackrestore(ptr %stack0)   ; Should depend transiently on %alloca0
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Stacksave = &*It++;
  auto *Alloca = &*It++;
  auto *Stackrestore = &*It++;
  DAG.extend(DmpVector<sandboxir::Value *>{
      &*BB->begin(), BB->getTerminator()->getPrevNode()});
  EXPECT_TRUE(DAG.getNode(Stackrestore)->dependsOn(DAG.getNode(Alloca)));
  EXPECT_TRUE(DAG.getNode(Alloca)->dependsOn(DAG.getNode(Stacksave)));
}

// Checks that stacksave and stackrestore depend on other mem instrs.
TEST(DependencyGraph, StackSaveRestoreDependOnOtherMem) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
declare ptr @llvm.stacksave()
declare void @llvm.stackrestore(ptr %ptr)

define void @foo(i8 %v0, i8 %v1, ptr %ptr) {
  store volatile i8 %v0, ptr %ptr, align 4
  %stack0 = call ptr @llvm.stacksave()       ; Should depend on store
  call void @llvm.stackrestore(ptr %stack0)  ; Should depend on stacksave
  store volatile i8 %v1, ptr %ptr, align 4   ; Should depend on stackrestore
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Store0 = &*It++;
  auto *Stacksave = &*It++;
  auto *Stackrestore = &*It++;
  auto *Store1 = &*It++;
  DAG.extend(DmpVector<sandboxir::Value *>{
      &*BB->begin(), BB->getTerminator()->getPrevNode()});
  EXPECT_TRUE(DAG.getNode(Stacksave)->dependsOn(DAG.getNode(Store0)));
  EXPECT_TRUE(DAG.getNode(Stackrestore)->dependsOn(DAG.getNode(Stacksave)));
  EXPECT_TRUE(DAG.getNode(Store1)->dependsOn(DAG.getNode(Stackrestore)));
}

// Make sure there is a dependency between a stackrestore and an alloca.
TEST(DependencyGraph, StackrestoreAndInAlloca) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
declare void @llvm.stackrestore(ptr %ptr)

define void @foo(ptr %ptr) {
  call void @llvm.stackrestore(ptr %ptr)
  %alloca0 = alloca inalloca i8              ; Should depend on stackrestore
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Stackrestore = &*It++;
  auto *Alloca = &*It++;
  DAG.extend(DmpVector<sandboxir::Value *>{
      &*BB->begin(), BB->getTerminator()->getPrevNode()});
  EXPECT_TRUE(DAG.getNode(Alloca)->dependsOn(DAG.getNode(Stackrestore)));
}

// Make sure there is a dependency between the alloca and stacksave
TEST(DependencyGraph, StacksaveAndInAlloca) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
declare ptr @llvm.stacksave()

define void @foo(ptr %ptr) {
  %alloca0 = alloca inalloca i8              ; Should depend on stackrestore
  %stack0 = call ptr @llvm.stacksave()       ; Should depend on alloca0
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Alloca = &*It++;
  auto *Stacksave = &*It++;
  DAG.extend(DmpVector<sandboxir::Value *>{
      &*BB->begin(), BB->getTerminator()->getPrevNode()});
  EXPECT_TRUE(DAG.getNode(Stacksave)->dependsOn(DAG.getNode(Alloca)));
}

// A non-InAlloca in a stacksave-stackrestore region does not need extra
// dependencies.
TEST(DependencyGraph, StackSaveRestoreNoInAlloca) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
declare ptr @llvm.stacksave()
declare void @llvm.stackrestore(ptr %ptr)
declare void @use(ptr %ptr)

define void @foo() {
  %stack = call ptr @llvm.stacksave()
  %alloca1 = alloca i8                         ; No dependency
  call void @llvm.stackrestore(ptr %stack)
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  DAG.extend(DmpVector<sandboxir::Value *>{
      &*BB->begin(), BB->getTerminator()->getPrevNode()});
  auto It = BB->begin();
  auto *Stacksave = &*It++;
  auto *Alloca = &*It++;
  auto *Stackrestore = &*It++;
  std::string Str;
  EXPECT_FALSE(DAG.getNode(Alloca)->dependsOn(DAG.getNode(Stacksave)));
  EXPECT_FALSE(DAG.getNode(Stackrestore)->dependsOn(DAG.getNode(Alloca)));
}

#ifndef NDEBUG
TEST(DependencyGraph, InlineAsmSideeffects) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(i8 %v, ptr %ptr1, ptr %ptr2) {
entry:
  store volatile i8 %v, ptr %ptr1, align 4
  call void asm sideeffect "", ""()
  store volatile i8 %v, ptr %ptr2, align 4
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  DAG.extend(DmpVector<sandboxir::Value *>{
      &*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *St1 = &*It++;
  auto *Asm = &*It++;
  auto *St2 = &*It++;

  expect_deps(DAG, St1, {});
  expect_deps(DAG, Asm, {St1});
  expect_deps(DAG, St2, {Asm, St1});
}

TEST(DependencyGraph, Loop) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(i32 %val) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv_next, %loop ]
  %iv_next = add i32 %iv, 1
  %slt = icmp slt i32 %iv_next, %val
  br i1 %slt, label %loop, label %exit

exit:
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

  auto *BB = Ctx.createBasicBlock(getBasicBlockByName(F, "loop"));
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});

  auto It = BB->begin();
  auto *Phi = &*It++;
  auto *Add = &*It++;
  auto *ICmp = &*It++;
  auto *Br = &*It++;

  expect_deps(DAG, Phi, {Add});
  expect_deps(DAG, Add, {Phi});
  expect_deps(DAG, ICmp, {Add});
  expect_deps(DAG, Br, {ICmp});
}

TEST(DependencyGraph, AAQueryBudget1) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(i8 %v, ptr noalias %ptr1, ptr noalias %ptr2, ptr noalias %ptr3, ptr noalias %ptr4) {
  store i8 %v, ptr %ptr1
  store i8 %v, ptr %ptr2
  store i8 %v, ptr %ptr3
  store i8 %v, ptr %ptr4
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

  std::string Str;
  raw_string_ostream SS(Str);

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  DAG.setAAQueryBudget(1);
  DAG.extend(DmpVector<sandboxir::Value *>{
      &*BB->begin(), BB->getTerminator()->getPrevNode()});

  auto It = BB->begin();
  auto *St1 = &*It++;
  auto *St2 = &*It++;
  auto *St3 = &*It++;
  auto *St4 = &*It++;

  expect_deps(DAG, St1, {});
  expect_deps(DAG, St2, {});
  expect_deps(DAG, St3, {St1});
  expect_deps(DAG, St4, {St1, St2});
}

TEST(DependencyGraph, AAQueryBudget2) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(i8 %v0, i8 %v1, i8 %v2, i8 %v3, ptr noalias %ptrA, ptr noalias %ptrB) {
  store i8 %v0, ptr %ptrA
  store i8 %v1, ptr %ptrA
  store i8 %v2, ptr %ptrB
  store i8 %v3, ptr %ptrB
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

  std::string Str;
  raw_string_ostream SS(Str);

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  {
    // AA Budget = 9999
    DAG.setAAQueryBudget(9999);
    DAG.extend(DmpVector<sandboxir::Value *>{
        &*BB->begin(), BB->getTerminator()->getPrevNode()});
    auto It = BB->begin();
    auto *St0 = &*It++;
    auto *St1 = &*It++;
    auto *St2 = &*It++;
    auto *St3 = &*It++;

    expect_deps(DAG, St0, {});
    expect_deps(DAG, St1, {St0});
    expect_deps(DAG, St2, {});
    expect_deps(DAG, St3, {St2});
  }

  {
    // AA Budget = 1
    DAG.clear();
    DAG.setAAQueryBudget(1);
    DAG.extend(DmpVector<sandboxir::Value *>{
        &*BB->begin(), BB->getTerminator()->getPrevNode()});
    auto It = BB->begin();
    auto *St0 = &*It++;
    auto *St1 = &*It++;
    auto *St2 = &*It++;
    auto *St3 = &*It++;

    expect_deps(DAG, St0, {});
    expect_deps(DAG, St1, {St0});
    expect_deps(DAG, St2, {St0});
    expect_deps(DAG, St3, {St0, St1, St2});
  }

  {
    // AA Budget = 0
    DAG.clear();
    DAG.setAAQueryBudget(0);
    DAG.extend(DmpVector<sandboxir::Value *>{
        &*BB->begin(), BB->getTerminator()->getPrevNode()});
    auto It = BB->begin();
    auto *St0 = &*It++;
    auto *St1 = &*It++;
    auto *St2 = &*It++;
    auto *St3 = &*It++;

    expect_deps(DAG, St0, {});
    expect_deps(DAG, St1, {St0});
    expect_deps(DAG, St2, {St0, St1});
    expect_deps(DAG, St3, {St0, St1, St2});
  }
}

TEST(DependencyGraph, MultipleExtends) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v, float %v0, ptr noalias %ptr1, ptr noalias %ptr2) {
  store float %v0, ptr %ptr1
  %ld1 = load float, ptr %ptr1
  %ld2 = load float, ptr %ptr2
  store float %v, ptr %ptr1
  store float %v, ptr %ptr2
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

  static constexpr const int AABudget = 10000;
  sandboxir::DependencyGraph DAG(Ctx, AA, AABudget);
  std::string Str;
  raw_string_ostream SS(Str);

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *St0 = &*It++;
  auto *Ld1 = &*It++;
  auto *Ld2 = &*It++;
  auto *St1 = &*It++;
  auto *St2 = &*It++;

  DAG.extend(DmpVector<sandboxir::Value *>{Ld2, Ld2});
  expect_deps(DAG, Ld2, {});

  DAG.extend(DmpVector<sandboxir::Value *>{Ld1, Ld1});
  expect_deps(DAG, Ld1, {});
  expect_deps(DAG, Ld2, {});

  DAG.extend(DmpVector<sandboxir::Value *>{St1, St1});
  expect_deps(DAG, Ld1, {});
  expect_deps(DAG, Ld2, {});
  expect_deps(DAG, St1, {Ld1});

  DAG.extend(DmpVector<sandboxir::Value *>{St2, St2});
  expect_deps(DAG, Ld1, {});
  expect_deps(DAG, Ld2, {});
  expect_deps(DAG, St1, {Ld1});
  expect_deps(DAG, St2, {Ld2});

  DAG.extend(DmpVector<sandboxir::Value *>{St0, St0});
  expect_deps(DAG, St0, {});
  expect_deps(DAG, Ld1, {St0});
  expect_deps(DAG, Ld2, {});
  expect_deps(DAG, St1, {Ld1, St0});
  expect_deps(DAG, St2, {Ld2});
}

TEST(DependencyGraph, MultipleExtendsWAR_Down) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v, ptr noalias %ptr) {
  %ld = load float, ptr %ptr, align 4
  store float %v, ptr %ptr, align 4
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

  std::string Str;
  raw_string_ostream SS(Str);

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Ld1 = &*It++;
  auto *St1 = &*It++;

  DAG.extend(DmpVector<sandboxir::Value *>{Ld1, Ld1});
  expect_deps(DAG, Ld1, {});

  DAG.extend(DmpVector<sandboxir::Value *>{St1, St1});
  expect_deps(DAG, Ld1, {});
  expect_deps(DAG, St1, {Ld1});
}

TEST(DependencyGraph, MultipleExtendsWAR_Up) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v, ptr noalias %ptr) {
  %ld = load float, ptr %ptr, align 4
  store float %v, ptr %ptr, align 4
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

  std::string Str;
  raw_string_ostream SS(Str);

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Ld1 = &*It++;
  auto *St1 = &*It++;

  DAG.extend(DmpVector<sandboxir::Value *>{St1, St1});
  expect_deps(DAG, St1, {});

  DAG.extend(DmpVector<sandboxir::Value *>{Ld1, Ld1});
  expect_deps(DAG, Ld1, {});
  expect_deps(DAG, St1, {Ld1});
}

TEST(DependencyGraph, MultipleExtendsWithGapsAndOverlap) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v, float %v0, ptr noalias %ptr1, ptr noalias %ptr2) {
  store float %v0, ptr %ptr1
  %ld1 = load float, ptr %ptr1
  %ld2 = load float, ptr %ptr2
  store float %v, ptr %ptr1
  store float %v, ptr %ptr2
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

  std::string Str;
  raw_string_ostream SS(Str);

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *St0 = &*It++;
  auto *Ld1 = &*It++;
  auto *Ld2 = &*It++;
  auto *St1 = &*It++;
  auto *St2 = &*It++;
  // Gap between St1-St2 and St0-St0
  DAG.extend(DmpVector<sandboxir::Value *>{St1, St2});
  expect_deps(DAG, St1, {});
  expect_deps(DAG, St2, {});

  DAG.extend(DmpVector<sandboxir::Value *>{St0, St0});
  expect_deps(DAG, St0, {});
  expect_deps(DAG, Ld1, {St0});
  expect_deps(DAG, Ld2, {});
  expect_deps(DAG, St1, {Ld1, St0});
  expect_deps(DAG, St2, {Ld2});

  // Overlap between St0-St1 and Ld1-St2
  sandboxir::DependencyGraph DAG2(Ctx, AA);
  DAG2.extend(DmpVector<sandboxir::Value *>{St0, St1});
  expect_deps(DAG2, St0, {});
  expect_deps(DAG2, Ld1, {St0});
  expect_deps(DAG2, Ld2, {});
  expect_deps(DAG2, St1, {St0, Ld1});

  DAG2.extend(DmpVector<sandboxir::Value *>{Ld1, St2});
  expect_deps(DAG2, St0, {});
  expect_deps(DAG2, Ld1, {St0});
  expect_deps(DAG2, Ld2, {});
  expect_deps(DAG2, St1, {Ld1, St0});
  expect_deps(DAG2, St2, {Ld2});

  // Full overlap Ld2-Ld2 and St0-St2
  sandboxir::DependencyGraph DAG3(Ctx, AA);
  DAG3.extend(DmpVector<sandboxir::Value *>{Ld2, Ld2});
  expect_deps(DAG3, Ld2, {});

  DAG3.extend(DmpVector<sandboxir::Value *>{St0});
  DAG3.extend(DmpVector<sandboxir::Value *>{St2});
  expect_deps(DAG3, St0, {});
  expect_deps(DAG3, Ld1, {St0});
  expect_deps(DAG3, Ld2, {});
  expect_deps(DAG3, St1, {Ld1, St0});
  expect_deps(DAG3, St2, {Ld2});

  // Full overlap St0-St2 and Ld2->Ld2
  sandboxir::DependencyGraph DAG4(Ctx, AA);
  DAG4.extend(DmpVector<sandboxir::Value *>{St0, St2});
  expect_deps(DAG4, St0, {});
  expect_deps(DAG4, Ld1, {St0});
  expect_deps(DAG4, Ld2, {});
  expect_deps(DAG4, St1, {Ld1, St0});
  expect_deps(DAG4, St2, {Ld2});

  DAG4.extend(DmpVector<sandboxir::Value *>{Ld2, Ld2});
  expect_deps(DAG4, St0, {});
  expect_deps(DAG4, Ld1, {St0});
  expect_deps(DAG4, Ld2, {});
  expect_deps(DAG4, St1, {Ld1, St0});
  expect_deps(DAG4, St2, {Ld2});
}
#endif // NDEBUG

TEST(DependencyGraph, MultipleExtends_NewUnderOld) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v1, float %v2, ptr %ptr) {
  %ld1 = load float, ptr %ptr
  store float %v1, ptr %ptr
  %dep_load = load float, ptr %ptr
  store float %v2, ptr %ptr
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Ld1 = &*It++;
  auto *St1 = &*It++;
  auto *DepLoad = &*It++;
  auto *St2 = &*It++;

  DAG.extend(DmpVector<sandboxir::Value *>{Ld1});

  DAG.extend(DmpVector<sandboxir::Value *>{St1, St2});
  auto *Ld1N = DAG.getNode(Ld1);
  auto *St1N = DAG.getNode(St1);
  auto *DepLoadN = DAG.getNode(DepLoad);
  auto *St2N = DAG.getNode(St2);
  // Check dependencies within the lower extension region.
  EXPECT_TRUE(St2N->dependsOn(DepLoadN));
  EXPECT_TRUE(DepLoadN->dependsOn(St1N));
  // Check dependencies with upper region.
  EXPECT_TRUE(St1N->dependsOn(Ld1N));
}

TEST(DependencyGraph, MultipleExtends_NewOverOld) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v1, float %v2, ptr %ptr) {
  %ld1 = load float, ptr %ptr
  store float %v1, ptr %ptr
  %dep_load = load float, ptr %ptr
  store float %v2, ptr %ptr
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Ld1 = &*It++;
  auto *St1 = &*It++;
  auto *DepLoad = &*It++;
  auto *St2 = &*It++;

  DAG.extend(DmpVector<sandboxir::Value *>{St2});

  DAG.extend(DmpVector<sandboxir::Value *>{Ld1, DepLoad});
  auto *Ld1N = DAG.getNode(Ld1);
  auto *St1N = DAG.getNode(St1);
  auto *DepLoadN = DAG.getNode(DepLoad);
  auto *St2N = DAG.getNode(St2);
  // Check dependencies within the upper extension region.
  EXPECT_TRUE(St1N->dependsOn(Ld1N));
  EXPECT_TRUE(DepLoadN->dependsOn(St1N));
  // Check dependencies with lower region.
  EXPECT_TRUE(St2N->dependsOn(DepLoadN));
}

// Check that the roots are correct for each DAG region.
TEST(DependencyGraph, Roots) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1) {
  %ld0 = load i8, ptr %ptr0
  %ld1 = load i8, ptr %ptr1
  store i8 %ld0, ptr %ptr0
  store i8 %ld1, ptr %ptr1
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *St0 = &*It++;
  auto *St1 = &*It++;
  {
    // Extend downwards
    DAG.extend(DmpVector<sandboxir::Value *>{Ld0});
    auto Roots = DAG.getRoots();
    EXPECT_EQ(Roots.size(), 1u);
    EXPECT_TRUE(find(Roots, DAG.getNode(Ld0)) != Roots.end());

    DAG.extend(DmpVector<sandboxir::Value *>{St0});
    Roots = DAG.getRoots();
    EXPECT_EQ(Roots.size(), 2u);
    EXPECT_TRUE(find(Roots, DAG.getNode(Ld1)) != Roots.end());
    EXPECT_TRUE(find(Roots, DAG.getNode(St0)) != Roots.end());

    DAG.extend(DmpVector<sandboxir::Value *>{St1});
    Roots = DAG.getRoots();
    EXPECT_EQ(Roots.size(), 2u);
    EXPECT_TRUE(find(Roots, DAG.getNode(St0)) != Roots.end());
    EXPECT_TRUE(find(Roots, DAG.getNode(St1)) != Roots.end());
  }

  {
    // Extend upwards
    DAG.clear();
    DAG.extend(DmpVector<sandboxir::Value *>{St0});
    auto Roots = DAG.getRoots();
    EXPECT_EQ(Roots.size(), 1u);
    EXPECT_TRUE(find(Roots, DAG.getNode(St0)) != Roots.end());

    DAG.extend(DmpVector<sandboxir::Value *>{Ld1});
    Roots = DAG.getRoots();
    EXPECT_EQ(Roots.size(), 2u);
    EXPECT_TRUE(find(Roots, DAG.getNode(Ld1)) != Roots.end());
    EXPECT_TRUE(find(Roots, DAG.getNode(St0)) != Roots.end());

    DAG.extend(DmpVector<sandboxir::Value *>{Ld0});
    Roots = DAG.getRoots();
    EXPECT_EQ(Roots.size(), 2u);
    EXPECT_TRUE(find(Roots, DAG.getNode(Ld1)) != Roots.end());
    EXPECT_TRUE(find(Roots, DAG.getNode(St0)) != Roots.end());

    DAG.extend(DmpVector<sandboxir::Value *>{St1});
    Roots = DAG.getRoots();
    EXPECT_EQ(Roots.size(), 2u);
    EXPECT_TRUE(find(Roots, DAG.getNode(St0)) != Roots.end());
    EXPECT_TRUE(find(Roots, DAG.getNode(St1)) != Roots.end());
  }
}

// Check that the roots are correct for each DAG region.
TEST(DependencyGraph, Roots2) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(i8 %v0, i8 %v1) {
  %add0 = add i8 %v0, %v1
  %add1 = add i8 %add0, %v1
  %add2 = add i8 %v0, %v1
  %add3 = add i8 %add1, %add0
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Add0 = &*It++;
  auto *Add1 = &*It++;
  auto *Add2 = &*It++;
  auto *Add3 = &*It++;
  // Extend downwards
  DAG.extend(DmpVector<sandboxir::Value *>{Add0});
  auto Roots = DAG.getRoots();
  EXPECT_EQ(Roots.size(), 1u);
  EXPECT_TRUE(find(Roots, DAG.getNode(Add0)) != Roots.end());

  DAG.extend(DmpVector<sandboxir::Value *>{Add1});
  Roots = DAG.getRoots();
  EXPECT_EQ(Roots.size(), 1u);
  EXPECT_TRUE(find(Roots, DAG.getNode(Add1)) != Roots.end());

  DAG.extend(DmpVector<sandboxir::Value *>{Add2});
  Roots = DAG.getRoots();
  EXPECT_EQ(Roots.size(), 2u);
  EXPECT_TRUE(find(Roots, DAG.getNode(Add1)) != Roots.end());
  EXPECT_TRUE(find(Roots, DAG.getNode(Add2)) != Roots.end());

  DAG.extend(DmpVector<sandboxir::Value *>{Add3});
  Roots = DAG.getRoots();
  EXPECT_EQ(Roots.size(), 2u);
  EXPECT_TRUE(find(Roots, DAG.getNode(Add2)) != Roots.end());
  EXPECT_TRUE(find(Roots, DAG.getNode(Add3)) != Roots.end());
}

// Check that the roots are correct after extending the DAG. This tests mem-only
// dependencies.
TEST(DependencyGraph, Roots_MemOnly) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  store i8 0, ptr %ptr
  store i8 1, ptr %ptr
  store i8 2, ptr %ptr
  store i8 3, ptr %ptr
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *St0 = &*It++;
  auto *St1 = &*It++;
  auto *St2 = &*It++;
  auto *St3 = &*It++;

  DAG.extend(DmpVector<sandboxir::Value *>{St0});
  auto Roots = DAG.getRoots();
  EXPECT_EQ(Roots.size(), 1u);
  EXPECT_TRUE(find(Roots, DAG.getNode(St0)) != Roots.end());

  Roots.clear();
  DAG.extend(DmpVector<sandboxir::Value *>{St1});
  Roots = DAG.getRoots();
  EXPECT_EQ(Roots.size(), 1u);
  EXPECT_TRUE(find(Roots, DAG.getNode(St1)) != Roots.end());

  Roots.clear();
  DAG.extend(DmpVector<sandboxir::Value *>{St2});
  Roots = DAG.getRoots();
  EXPECT_EQ(Roots.size(), 1u);
  EXPECT_TRUE(find(Roots, DAG.getNode(St2)) != Roots.end());

  Roots.clear();
  DAG.extend(DmpVector<sandboxir::Value *>{St3});
  Roots = DAG.getRoots();
  EXPECT_EQ(Roots.size(), 1u);
  EXPECT_TRUE(find(Roots, DAG.getNode(St3)) != Roots.end());
}

// Check ViewRange.
TEST(DependencyGraph, View) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld0 = load i8, ptr %ptr
  store i8 0, ptr %ptr
  store i8 1, ptr %ptr
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Ld0 = &*It++;
  auto *St0 = &*It++;
  auto *St1 = &*It++;

  DAG.extend(DmpVector<sandboxir::Value *>{St1});
  auto Roots = DAG.getRoots();
  EXPECT_EQ(Roots.size(), 1u);
  EXPECT_TRUE(find(Roots, DAG.getNode(St1)) != Roots.end());

  DAG.extend(DmpVector<sandboxir::Value *>{Ld0});
  Roots = DAG.getRoots();
  EXPECT_EQ(Roots.size(), 1u);
  EXPECT_TRUE(find(Roots, DAG.getNode(St1)) != Roots.end());

  // DAG resetView
  DAG.resetView();
  Roots = DAG.getRoots();
  EXPECT_EQ(Roots.size(), 0u);

  DAG.extend(DmpVector<sandboxir::Value *>{Ld0});
  Roots = DAG.getRoots();
  EXPECT_EQ(Roots.size(), 1u);
  EXPECT_TRUE(find(Roots, DAG.getNode(Ld0)) != Roots.end());

  DAG.extend(DmpVector<sandboxir::Value *>{St0});
  Roots = DAG.getRoots();
  EXPECT_EQ(Roots.size(), 1u);
  EXPECT_TRUE(find(Roots, DAG.getNode(St0)) != Roots.end());

  // Check allSuccsReady().
  DAG.resetView();
  DAG.extend(DmpVector<sandboxir::Value *>{Ld0});
  auto *Ld0N = DAG.getNode(Ld0);
  EXPECT_TRUE(Ld0N->allSuccsReady());
}

// Check the return value of DAG.extend().
TEST(DependencyGraph, DagExtendReturn) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld0 = load i8, ptr %ptr
  store i8 0, ptr %ptr
  store i8 1, ptr %ptr
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Ld0 = &*It++;
  auto *St0 = &*It++;
  auto *St1 = &*It++;
  auto *Ret = &*It++;

  {
    auto Rgn = DAG.extend(DmpVector<sandboxir::Value *>{St1});
    EXPECT_TRUE(Rgn == sandboxir::InstrInterval(St1, St1));
  }
  {
    auto Rgn = DAG.extend(DmpVector<sandboxir::Value *>{St0});
    EXPECT_TRUE(Rgn == sandboxir::InstrInterval(St0, St0));
  }
  {
    DAG.resetView();
    auto Rgn = DAG.extend(DmpVector<sandboxir::Value *>{Ld0, Ret});
    EXPECT_TRUE(Rgn == sandboxir::InstrInterval(Ld0, Ret));
  }
}

TEST(DependencyGraph, View2) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1) {
  %ld0 = load i8, ptr %ptr0
  %ld1 = load i8, ptr %ptr1
  store i8 %ld0, ptr %ptr0
  store i8 %ld1, ptr %ptr1
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *St0 = &*It++;
  auto *St1 = &*It++;

  DAG.extend(DmpVector<sandboxir::Value *>{Ld0, Ld1, St0, St1});
  DAG.resetView();

  DAG.extend(DmpVector<sandboxir::Value *>{Ld1});
  DAG.extend(DmpVector<sandboxir::Value *>{St1});
#if !defined NDEBUG && defined EXPENSIVE_CHECKS
  DAG.verify();
#endif
}

#ifndef NDEBUG
// Dependencies to the BB terminator are skipped. We handle them in the
// scheduler.
TEST(DependencyGraph, DepsToTerminator) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(i8 %v1, i8 %v2) {
  %add1 = add i8 %v1, %v1
  %add2 = add i8 %v2, %v2
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});

  auto It = BB->begin();
  auto *Add1 = &*It++;
  auto *Add2 = &*It++;
  auto *Ret = &*It++;
  expect_deps(DAG, Add1, {});
  expect_deps(DAG, Add2, {});
  expect_deps(DAG, Ret, {});
}

TEST(DependencyGraph, PHIs) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1, i8 %val0) {
entry:
  br label %loop

loop:
  %phi = phi i8 [ 0, %entry ], [ %val0, %loop ]
  %add = add i8 %val0, %val0
  %ld0 = load i8, ptr %ptr0
  store i8 %ld0, ptr %ptr0
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

  auto *BB = Ctx.createBasicBlock(getBasicBlockByName(F, "loop"));
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *PHI = &*It++;
  auto *Add0 = &*It++;
  auto *Ld0 = &*It++;
  auto *St0 = &*It++;
  auto *Ret = &*It++;

  std::string Str;
  raw_string_ostream SS(Str);

  {
    // Build the DAG in one go.
    DAG.clear();
    DAG.extend(
        DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});
    auto *PHIN = DAG.getNode(PHI);
    for (sandboxir::Instruction &I : drop_begin(*BB)) {
      auto *N = DAG.getNode(&I);
      EXPECT_FALSE(dependencyPath(PHIN, N));
    }
    expect_deps(DAG, PHI, {});
    expect_deps(DAG, Add0, {});
    expect_deps(DAG, Ld0, {});
    expect_deps(DAG, St0, {Ld0});
    expect_deps(DAG, Ret, {});
  }

  {
    // Build the DAG in sections
    DAG.clear();
    DAG.extend({Add0});
    DAG.extend({PHI});
    DAG.extend(DmpVector<sandboxir::Value *>{Ld0, BB->getTerminator()});
    auto *PHIN = DAG.getNode(PHI);
    for (sandboxir::Instruction &I : drop_begin(*BB)) {
      auto *N = DAG.getNode(&I);
      EXPECT_FALSE(dependencyPath(PHIN, N));
    }
  }
}
#endif // NDEBUG

// Checks DAG.erase().
TEST(DependencyGraph, EraseInstr) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v, ptr %ptr1, ptr %ptr2) {
  %ld = load float, ptr %ptr1
  store float %v, ptr %ptr1
  store float %v, ptr %ptr2
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto *Ld1 = &*std::next(BB->begin(), 0);
  auto *St1 = &*std::next(BB->begin(), 1);
  auto *St2 = &*std::next(BB->begin(), 2);
  auto *Ret = &*std::next(BB->begin(), 3);
  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});

  auto *L1 = DAG.getNode(Ld1);
  auto *S1 = DAG.getNode(St1);
  auto *S2 = DAG.getNode(St2);
  auto *RN = DAG.getNode(Ret);
  EXPECT_TRUE(S1->dependsOn(L1));
  // Now erase St1
  DAG.erase(St1);
  St1->eraseFromParent();
  EXPECT_EQ(DAG.getNode(St1), nullptr);
  EXPECT_TRUE(S2->dependsOn(L1));

  // Remove Ret, check that it is no longer a root
  auto Roots = DAG.getRoots();
  EXPECT_NE(find(Roots, RN), Roots.end());
  DAG.erase(Ret);
  Ret->eraseFromParent();
  Roots = DAG.getRoots();
  EXPECT_EQ(find(Roots, RN), Roots.end());
}

// Checks Node::inheritDeps()
TEST(DependencyGraph, NodeInheritDeps_AvoidDepsToSelf) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v1, float %v2, float %v3, <2 x float> %vec, ptr noalias %ptr1, ptr noalias %ptr2) {
  store float %v1, ptr %ptr1
  store float %v2, ptr %ptr1
  store float %v3, ptr %ptr2
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

  sandboxir::Function *SBF = Ctx.createFunction(&F);
  auto *BB = Ctx.getBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  // Explicitly enable tracking
  DAG.enableTracking();
  int Idx = 0;
  auto *St1 = &*std::next(BB->begin(), Idx++);
  auto *St2 = &*std::next(BB->begin(), Idx++);
  auto *St3 = &*std::next(BB->begin(), Idx++);
  auto *Ret = &*std::next(BB->begin(), Idx++);
  (void)Ret;
  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});

  auto *S1 = DAG.getNode(St1);
  auto *S2 = DAG.getNode(St2);
  auto *S3 = DAG.getNode(St3);

  EXPECT_TRUE(S2->dependsOn(S1));
  EXPECT_FALSE(S2->dependsOn(S3));
  EXPECT_FALSE(S3->dependsOn(S2));

  // Now create a new vector store that replaces {S2,S3}.
  auto *Vec = SBF->getArg(3);
  auto *Ptr = cast<sandboxir::StoreInst>(St2)->getPointerOperand();
  auto *NewI =
      sandboxir::StoreInst::create(Vec, Ptr, /*Align=*/std::nullopt,
                                            /*InsertBefore=*/Ret, Ctx);
  DAG.notifyInsert(NewI);
  auto *NewN = DAG.getNode(NewI);
  EXPECT_TRUE(NewN->dependsOn(S1));
  EXPECT_TRUE(NewN->dependsOn(S2));
  EXPECT_FALSE(NewN->dependsOn(NewN));
}

TEST(DependencyGraph, DAGInsertAndAddDeps_Simple) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v1, float %v2, float %v3, ptr %ptr) {
  store float %v1, ptr %ptr
  store float %v3, ptr %ptr
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

  sandboxir::Function *SBF = Ctx.createFunction(&F);
  auto *BB = Ctx.getBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  DAG.enableTracking();
  auto It = BB->begin();
  auto *St1 = &*It++;
  auto *St3 = &*It++;

  DAG.notifyInsert(St1);
  auto *S1 = DAG.getNode(St1);
  DAG.notifyInsert(St3);
  auto *S3 = DAG.getNode(St3);
  EXPECT_TRUE(S3->dependsOn(S1));

  auto *Ptr = cast<sandboxir::StoreInst>(St1)->getPointerOperand();
  auto *St2 =
      sandboxir::StoreInst::create(SBF->getArg(1), Ptr, /*Align=*/
                                            std::nullopt,
                                            /*InsertBefore=*/St3, Ctx);
  DAG.notifyInsert(St2);
  auto *S2 = DAG.getNode(St2);
  EXPECT_TRUE(S2->dependsOn(S1));
  EXPECT_TRUE(S3->dependsOn(S2));
}

// Checks DAG.notifyInsert(). The new instruction is in the DAG region.
TEST(DependencyGraph, DAGInsertAndAddDeps_InInstrInterval) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v1, float %v2, float %v3, float %v4, <2 x float> %vec,
                 ptr noalias %ptr1, ptr noalias %ptr2) {
  store float %v1, ptr %ptr1
  store float %v2, ptr %ptr1
  store float %v3, ptr %ptr2
  store float %v4, ptr %ptr2
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

  sandboxir::Function *SBF = Ctx.createFunction(&F);
  auto *BB = Ctx.getBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *St1 = &*It++;
  auto *St2 = &*It++;
  auto *St3 = &*It++;
  auto *St4 = &*It++;
  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});

  auto *S1 = DAG.getNode(St1);
  auto *S2 = DAG.getNode(St2);
  auto *S3 = DAG.getNode(St3);
  auto *S4 = DAG.getNode(St4);

  EXPECT_TRUE(S2->dependsOn(S1));
  EXPECT_TRUE(S4->dependsOn(S3));
  // We are going to use S2,S3 as a bundle, so no deps.
  EXPECT_FALSE(S3->dependsOn(S2));

  // Now create a new vector store that replaces {S2,S3}.
  auto *Vec = SBF->getArg(4);
  auto *Ptr = cast<sandboxir::StoreInst>(St2)->getPointerOperand();
  auto *NewI =
      sandboxir::StoreInst::create(Vec, Ptr, /*Align=*/std::nullopt,
                                            /*InsertBefore=*/St4, Ctx);
  DAG.notifyInsert(NewI);
  auto *NewN = DAG.getNode(NewI);
  EXPECT_TRUE(NewN != nullptr);
  EXPECT_TRUE(NewN->dependsOn(S1));
}

// Checks DAG.notifyInsert() when adding an instruction not in the DAG.
TEST(DependencyGraph, DAGInsertAndAddDeps_NotInInstrInterval) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v1, float %v2, float %v3, float %v4, <2 x float> %vec,
                 ptr noalias %ptr1, ptr noalias %ptr2) {
  store float %v1, ptr %ptr1
  store float %v2, ptr %ptr1
  store float %v3, ptr %ptr2
  store float %v4, ptr %ptr2
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
  auto *BB = Ctx.getBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *St1 = &*It++;
  auto *St2 = &*It++;
  auto *St3 = &*It++;
  auto *St4 = &*It++;
  auto *Ret = &*It++;
  DAG.extend(DmpVector<sandboxir::Value *>{St1, St4});

  auto *S1 = DAG.getNode(St1);
  auto *S2 = DAG.getNode(St2);
  auto *S3 = DAG.getNode(St3);
  auto *S4 = DAG.getNode(St4);

  EXPECT_TRUE(S2->dependsOn(S1));
  EXPECT_TRUE(S4->dependsOn(S3));
  // We are going to use S2,S3 as a bundle, so no deps.
  EXPECT_FALSE(S3->dependsOn(S2));

  // Now create a new vector store that replaces {S2,S3}.
  auto *Vec = SBF->getArg(4);
  auto *Ptr = cast<sandboxir::StoreInst>(St2)->getPointerOperand();
  auto *NewI =
      sandboxir::StoreInst::create(Vec, Ptr, /*Align=*/std::nullopt,
                                            /*InsertBefore=*/Ret, Ctx);
  DAG.notifyInsert(NewI);
  auto *NewN = DAG.getNode(NewI);
  EXPECT_TRUE(NewN != nullptr);
  EXPECT_TRUE(NewN->dependsOn(S1));
}

// Checks DAG.notifyInsert() when adding an instr that has a dependency with
// an instr in the DAGInterval but not in the current ViewRange.
//
// DAG  View
//  -     -
//  |     |  o New
//  |     -
//  |
//  |        o Dep
//  -
TEST(DependencyGraph, DAGInsertAndAddDeps_DependencyOutsideView_DepBelow) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v1, float %v2, float %v3, float %new, ptr noalias %ptr1, ptr noalias %ptr2) {
  store float %v1, ptr %ptr1
  store float %v2, ptr %ptr2
  store float %v3, ptr %ptr1
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
  auto *BB = Ctx.getBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *St1 = &*It++;
  auto *St2 = &*It++;
  auto *St3 = &*It++;
  // Create a DAG that spans St1 - St3.
  DAG.extend(DmpVector<sandboxir::Value *>{St1, St3});
  // Now restrict the View region to St1 - St2 while
  // the DAG region still remaining St1 - St3.
  DAG.resetView();
  EXPECT_TRUE(DAG.getView().empty());
  DAG.extend(DmpVector<sandboxir::Value *>{St1, St2});
  // Check the regions.
  EXPECT_TRUE(DAG.getView().contains(St1));
  EXPECT_TRUE(DAG.getView().contains(St2));
  EXPECT_FALSE(DAG.getView().contains(St3));
  for (auto *SBI : {St1, St2, St3})
    EXPECT_TRUE(DAG.getDAGInterval().contains(SBI));

  // Now create a new instruction within the View region that has a dependency
  // with St3 which is outside the View region, but in the DAG region.
  auto *Val = SBF->getArg(3);
  auto *Ptr1 = SBF->getArg(4);
  auto *NewI =
      sandboxir::StoreInst::create(Val, Ptr1, /*Align=*/std::nullopt,
                                            /*InsertBefore=*/St2, Ctx);

  // Now call the function that updates the DAG.
  DAG.notifyInsert(NewI);

  auto *St1N = DAG.getNode(St1);
  auto *NewN = DAG.getNode(NewI);
  auto *St2N = DAG.getNode(St2);
  auto *St3N = DAG.getNode(St3);

  // Check the dependencies within the View region.
  EXPECT_TRUE(NewN->dependsOn(St1N));
  EXPECT_FALSE(St2N->dependsOn(NewN));
  // Now check for deps outside the View region.
  EXPECT_TRUE(St3N->dependsOn(NewN));
}

// Checks DAG.notifyInsert() when adding an instr that has a dependency with
// an instr in the DAGInterval but not in the current ViewRange.
//
// DAG  View
//  -
//  |        o Dep
//  |
//  |     -
//  |     |  o New
//  -     -
TEST(DependencyGraph, DAGInsertAndAddDeps_DependencyOutsideView_DepAbove) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v1, float %v2, float %v3, float %new, ptr noalias %ptr1, ptr noalias %ptr2) {
  store float %v1, ptr %ptr1
  store float %v2, ptr %ptr2
  store float %v3, ptr %ptr1
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
  auto *BB = Ctx.getBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *St1 = &*It++;
  auto *St2 = &*It++;
  auto *St3 = &*It++;
  // Create a DAG that spans St1 - St3.
  DAG.extend(DmpVector<sandboxir::Value *>{St1, St3});
  // Now restrict the View region to St2 - St3 while
  // the DAG region still remaining St1 - St3.
  DAG.resetView();
  EXPECT_TRUE(DAG.getView().empty());
  DAG.extend(DmpVector<sandboxir::Value *>{St2, St3});
  // Check the regions.
  EXPECT_FALSE(DAG.getView().contains(St1));
  EXPECT_TRUE(DAG.getView().contains(St2));
  EXPECT_TRUE(DAG.getView().contains(St3));
  for (auto *SBI : {St1, St2, St3})
    EXPECT_TRUE(DAG.getDAGInterval().contains(SBI));

  // Now create a new instruction within the View region that has a dependency
  // with St1 which is outside the View region, but in the DAG region.
  auto *Val = SBF->getArg(3);
  auto *Ptr1 = SBF->getArg(4);
  auto *NewI =
      sandboxir::StoreInst::create(Val, Ptr1, /*Align=*/std::nullopt,
                                            /*InsertBefore=*/St3, Ctx);

  // Now call the function that updates the DAG.
  DAG.notifyInsert(NewI);

  auto *St1N = DAG.getNode(St1);
  auto *St2N = DAG.getNode(St2);
  auto *NewN = DAG.getNode(NewI);
  auto *St3N = DAG.getNode(St3);

  // Check the dependencies within the View region.
  EXPECT_FALSE(NewN->dependsOn(St2N));
  EXPECT_TRUE(St3N->dependsOn(NewN));
  // Now check for deps outside the View region.
  EXPECT_TRUE(NewN->dependsOn(St1N));
}

// Checks DAG.notifyInsert() when adding an instr outside the View that has
// a dependency with an instr also outside the View
//
// DAG  View
//  -     -
//  |     |
//  |     -
//  |        o Dep
//  |        o New
//  -
TEST(DependencyGraph, DAGInsertAndAddDeps_NewAndDepOutsideView) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v1, float %v2, float %new, ptr noalias %ptr1, ptr noalias %ptr2) {
  store float %v1, ptr %ptr1
  store float %v2, ptr %ptr2
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
  auto *BB = Ctx.getBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *St1 = &*It++;
  auto *St2 = &*It++;
  auto *Ret = &*It++;
  // Create a DAG that spans St1 - St2.
  DAG.extend(DmpVector<sandboxir::Value *>{St1, St2});
  // Now restrict the View region to St1 while
  // the DAG region still remaining St1 - St2.
  DAG.resetView();
  EXPECT_TRUE(DAG.getView().empty());
  DAG.extend(DmpVector<sandboxir::Value *>{St1, St1});
  // Check the regions.
  EXPECT_TRUE(DAG.getView().contains(St1));
  EXPECT_FALSE(DAG.getView().contains(St2));
  for (auto *SBI : {St1, St2})
    EXPECT_TRUE(DAG.getDAGInterval().contains(SBI));

  // Now create a new instruction within the View region that has a dependency
  // with St2 which is outside the View region, but within the DAG region.
  auto *Val = SBF->getArg(2);
  auto *Ptr2 = SBF->getArg(4);
  auto *NewI =
      sandboxir::StoreInst::create(Val, Ptr2, /*Align=*/std::nullopt,
                                            /*InsertBefore=*/Ret, Ctx);

  // Now call the function that updates the DAG.
  DAG.notifyInsert(NewI);

  auto *St1N = DAG.getNode(St1);
  (void)St1N;
  auto *St2N = DAG.getNode(St2);
  auto *NewN = DAG.getNode(NewI);

  // Check the dependency in the DAG region
  EXPECT_TRUE(NewN->dependsOn(St2N));
}

// Checks DAG.notifyInsert() when adding an instr outside the View that has
// a dependency with an instr also outside the View
//
// DAG  View
//  -     -
//  |     |
//  |     -
//  |        o New
//  |        o Dep
//  -
TEST(DependencyGraph, DAGInsertAndAddDeps_NewAndDepOutsideView_NewAboveDep) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v1, float %v2, float %new, ptr noalias %ptr1, ptr noalias %ptr2) {
  store float %v1, ptr %ptr1
  store float %v2, ptr %ptr2
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
  auto *BB = Ctx.getBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *St1 = &*It++;
  auto *St2 = &*It++;
  // Create a DAG that spans St1 - St2.
  DAG.extend(DmpVector<sandboxir::Value *>{St1, St2});
  // Now restrict the View region to St1 while
  // the DAG region still remaining St1 - St2.
  DAG.resetView();
  EXPECT_TRUE(DAG.getView().empty());
  DAG.extend(DmpVector<sandboxir::Value *>{St1, St1});
  // Check the regions.
  EXPECT_TRUE(DAG.getView().contains(St1));
  EXPECT_FALSE(DAG.getView().contains(St2));
  for (auto *SBI : {St1, St2})
    EXPECT_TRUE(DAG.getDAGInterval().contains(SBI));

  // Now create a new instruction within the View region that has a dependency
  // with St2 which is outside the View region, but within the DAG region.
  auto *Val = SBF->getArg(2);
  auto *Ptr2 = SBF->getArg(4);
  auto *NewI =
      sandboxir::StoreInst::create(Val, Ptr2, /*Align=*/std::nullopt,
                                            /*InsertBefore=*/St2, Ctx);

  // Now call the function that updates the DAG.
  DAG.notifyInsert(NewI);

  auto *St1N = DAG.getNode(St1);
  (void)St1N;
  auto *St2N = DAG.getNode(St2);
  auto *NewN = DAG.getNode(NewI);

  // Check the dependency in the DAG region
  EXPECT_TRUE(St2N->dependsOn(NewN));
}

// We are erasing node N that was the intermediate node that was the reason to
// skip dependency B->A.
// B
// |\
// N |
// |/
// A
//
TEST(DependencyGraph, NotifyEraseTransientNode) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %B, i8 %A, i8 %N) {
  store i8 %B, ptr %ptr
  store i8 %N, ptr %ptr
  store i8 %A, ptr %ptr
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *StB = &*It++;
  auto *StN = &*It++;
  auto *StA = &*It++;

  DAG.extend(DmpVector<sandboxir::Value *>{StB, StA});
  auto *B = DAG.getNode(StB);
  auto *N = DAG.getNode(StN);
  auto *A = DAG.getNode(StA);

  EXPECT_TRUE(N->hasImmPred(B));
  EXPECT_TRUE(A->hasImmPred(N));
  EXPECT_TRUE(A->hasImmPred(B)); // Transient

  // Check that there is a dependency: B->A.
  EXPECT_TRUE(A->hasImmPred(B));
}

// We are erasing node N that was the intermediate node that was the reason to
// skip dependency B->A.
//        B
// Use-Def|\
//        N |Mem
//     Mem|/
//        A
//
TEST(DependencyGraph, NotifyEraseTransientNodeUseDef) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr, i8 %A) {
  %B = load i8, ptr %ptr ; B
  store i8 %B, ptr %ptr  ; N
  store i8 %A, ptr %ptr  ; A
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *LdB = &*It++;
  auto *StN = &*It++;
  auto *StA = &*It++;

  DAG.extend(DmpVector<sandboxir::Value *>{LdB, StA});
  auto *B = DAG.getNode(LdB);
  auto *N = DAG.getNode(StN);
  auto *A = DAG.getNode(StA);

  EXPECT_TRUE(N->hasImmPred(B));
  EXPECT_FALSE(N->hasMemPred(B)); // Expected Use-Def dep
  EXPECT_TRUE(A->hasImmPred(N));
  EXPECT_TRUE(A->hasMemPred(N)); // Expected Mem dep
  EXPECT_TRUE(A->hasImmPred(B)); // Transient

  // Check that there is a dependency: B->A.
  EXPECT_TRUE(A->hasImmPred(B));
}

TEST(DependencyGraph, LLVMMemcpy) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
declare void @llvm.memcpy.p0.p0.i8(ptr %dst, ptr %src, i64 %len, i1 %volatile)
define void @foo(ptr %dst, ptr %src, i8 %val) {
  %ld = load i8, ptr %dst
  call void @llvm.memcpy.p0.p0.i8(ptr %dst, ptr %src, i64 8, i1 false)
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Ld = &*It++;
  auto *Memcpy = &*It++;

  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});
  expect_deps(DAG, Ld, {});
  expect_deps(DAG, Memcpy, {Ld});
}

// Make sure the dependencies are accurate after we re erase a node.
TEST(DependencyGraph, AccurateDependenciesAfterErase) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
declare void @bar()
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1, i8 %v) {
  store i8 %v, ptr %ptr0
  call void @bar() ; Dependencies with both stores due to side-effects
  store i8 %v, ptr %ptr1 ; No dependency with store 0
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *St0 = &*It++;
  auto *Call = &*It++;
  auto *St1 = &*It++;
  auto *Ret = &*It++;

  DAG.extend(DmpVector<sandboxir::Value *>{St0, Ret});
  auto *St0N = DAG.getNode(St0);
  auto *CallN = DAG.getNode(Call);
  auto *St1N = DAG.getNode(St1);
  EXPECT_TRUE(St1N->dependsOn(CallN));
  EXPECT_TRUE(CallN->dependsOn(St0N));

  Call->eraseFromParent();

  EXPECT_FALSE(St1N->dependsOn(St0N));
}

TEST(DependencyGraph, NodeIterators) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v1, i8 %v2, i8 %v3) {
  %add1 = add i8 %v1, %v1
  store i8 %v1, ptr %ptr
  store i8 %v2, ptr %ptr
  %add2 = add i8 %v2, %v2
  store i8 %v3, ptr %ptr
  %add3 = add i8 %v3, %v3
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  auto It = BB->begin();
  auto *Add1 = &*It++;
  auto *St1 = &*It++;
  auto *St2 = &*It++;
  auto *Add2 = &*It++;
  auto *St3 = &*It++;
  auto *Add3 = &*It++;
  auto *Ret = &*It++;

  DAG.extend(DmpVector<sandboxir::Value *>{Add1, Ret});
  auto *Add1N = DAG.getNode(Add1);
  auto *St1N = DAG.getNode(St1);
  auto *St2N = DAG.getNode(St2);
  auto *Add2N = DAG.getNode(Add2);
  (void)Add2N;
  auto *St3N = DAG.getNode(St3);
  auto *Add3N = DAG.getNode(Add3);
  (void)Add3N;
  auto *RetN = DAG.getNode(Ret);

  // Check single-element range
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (auto &N : DAG.makeRange(Add1N, Add1N))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 1u);
    EXPECT_EQ(Nodes[0], Add1N);
  }

  // Check multi-element range
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (auto &N : DAG.makeRange(Add1N, St1N))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 2u);
    EXPECT_EQ(Nodes[0], Add1N);
    EXPECT_EQ(Nodes[1], St1N);
  }
  // Check whole BB
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (auto &N : DAG.makeRange(Add1N, RetN))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 7u);
    for (auto [Idx, IRef] : enumerate(*BB)) {
      auto *N = DAG.getNode(&IRef);
      EXPECT_EQ(N, Nodes[Idx]);
    }
  }
  // Check whole BB in reverse
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (auto &N : reverse(DAG.makeRange(Add1N, RetN)))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 7u);
    for (auto [Idx, IRef] : enumerate(*BB)) {
      auto *N = DAG.getNode(&IRef);
      EXPECT_EQ(N, Nodes[Nodes.size() - Idx - 1]);
    }
  }

  // Check creating the node from Instructions.
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (auto &N : DAG.makeRange(Add1, St1))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 2u);
    EXPECT_EQ(Nodes[0], Add1N);
    EXPECT_EQ(Nodes[1], St1N);
  }
  // Check reverse iteration
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (sandboxir::DependencyGraph::Node &N :
         reverse(DAG.makeRange(Add1N, St1N)))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 2u);
    EXPECT_EQ(Nodes[0], St1N);
    EXPECT_EQ(Nodes[1], Add1N);
  }
  // Check drop_begin
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (sandboxir::DependencyGraph::Node &N :
         drop_begin(DAG.makeRange(Add1N, St1N)))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 1u);
    EXPECT_EQ(Nodes[0], St1N);
  }
  // Check assertion if Top > Bot
#ifndef NDEBUG
  EXPECT_DEATH(DAG.makeRange(St1N, Add1N), ".*Top before Bot.*");
#endif

  // Check MemRange single-element
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (sandboxir::DependencyGraph::Node &N : DAG.makeMemRange(St1N, St1N))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 1u);
    EXPECT_EQ(Nodes[0], St1N);
  }
  // Check MemRange three-element
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (sandboxir::DependencyGraph::Node &N : DAG.makeMemRange(St1N, St3N))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 3u);
    EXPECT_EQ(Nodes[0], St1N);
    EXPECT_EQ(Nodes[1], St2N);
    EXPECT_EQ(Nodes[2], St3N);
  }
  // Check death if Top > Bot
#ifndef NDEBUG
  EXPECT_DEATH(DAG.makeMemRange(St2N, St1N), ".*Top before Bot.*");
#endif
  // Check MemRange three-element
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (sandboxir::DependencyGraph::Node &N : DAG.makeMemRange(St1N, St3N))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 3u);
    EXPECT_EQ(Nodes[0], St1N);
    EXPECT_EQ(Nodes[1], St2N);
    EXPECT_EQ(Nodes[2], St3N);
  }
  // Check MemRange created from instructions
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (sandboxir::DependencyGraph::Node &N : DAG.makeMemRange(St1, St3))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 3u);
    EXPECT_EQ(Nodes[0], St1N);
    EXPECT_EQ(Nodes[1], St2N);
    EXPECT_EQ(Nodes[2], St3N);
  }
  // Check MemRange created from non-mem instructions
#ifndef NDEBUG
  EXPECT_DEATH(DAG.makeMemRange(Add1, St1), ".*mem.*");
  EXPECT_DEATH(DAG.makeMemRange(St1, Add2), ".*mem.*");
#endif
  // Check MemRange reverse
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (sandboxir::DependencyGraph::Node &N :
         reverse(DAG.makeMemRange(St1N, St3N)))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 3u);
    EXPECT_EQ(Nodes[0], St3N);
    EXPECT_EQ(Nodes[1], St2N);
    EXPECT_EQ(Nodes[2], St1N);
  }
  // Check MemRange reverse
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (sandboxir::DependencyGraph::Node &N :
         reverse(DAG.makeMemRange(St1N, St3N)))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 3u);
    EXPECT_EQ(Nodes[0], St3N);
    EXPECT_EQ(Nodes[1], St2N);
    EXPECT_EQ(Nodes[2], St1N);
  }

  // Check MemRangeFromNonMem with 2 mem in between
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (sandboxir::DependencyGraph::Node &N :
         DAG.makeMemRangeFromNonMem(Add1, Add2))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 2u);
    EXPECT_EQ(Nodes[0], St1N);
    EXPECT_EQ(Nodes[1], St2N);
  }
  // Check MemRangeFromNonMem with 1 mem instruction in between
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (sandboxir::DependencyGraph::Node &N :
         DAG.makeMemRangeFromNonMem(Add2, Add3))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 1u);
    EXPECT_EQ(Nodes[0], St3N);
  } // Check MemRangeFromNonMem when one is already mem
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (sandboxir::DependencyGraph::Node &N :
         DAG.makeMemRangeFromNonMem(Add1, St2))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 2u);
    EXPECT_EQ(Nodes[0], St1N);
    EXPECT_EQ(Nodes[1], St2N);
  }
  // Check MemRangeFromNonMem when both are already mem
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (sandboxir::DependencyGraph::Node &N :
         DAG.makeMemRangeFromNonMem(St1, St2))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 2u);
    EXPECT_EQ(Nodes[0], St1N);
    EXPECT_EQ(Nodes[1], St2N);
  }
  // Check MemRangeFromNonMem with no mem instructions in between
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (sandboxir::DependencyGraph::Node &N :
         DAG.makeMemRangeFromNonMem(Add3, Ret))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 0u);
  }
  // Check MemRangeFromNonMem with no mem instructions in between and both same
  {
    SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
    for (sandboxir::DependencyGraph::Node &N :
         DAG.makeMemRangeFromNonMem(Add3, Add3))
      Nodes.push_back(&N);
    ASSERT_EQ(Nodes.size(), 0u);
  }
  // Check death when Top after Bot
#ifndef NDEBUG
  EXPECT_DEATH(DAG.makeMemRangeFromNonMem(Add3, Add2), ".*Top before Bot.*");
#endif
}

//  Op
//  | \
//  Pack
// There should be a single dependency successor Op->Pack, and a single
// dependency predecessor from Pack to Op.
TEST(DependencyGraph, CheckPredsSuccs_Pack) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
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

  auto *BB = Ctx.createBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *Op = &*It++;
  auto *Pack = cast<sandboxir::PackInst>(&*It++);

  auto &DAG = Ctx.getScheduler(BB)->getDAG();
  DAG.extend({Op, Pack});
  auto *OpN = DAG.getNode(Op);
  auto *PackN = DAG.getNode(Pack);
  unsigned SuccCnt = 0;
  for (auto *SuccN : OpN->succs()) {
    (void)SuccN;
    ++SuccCnt;
  }
  EXPECT_EQ(SuccCnt, 1u);
  unsigned PredCnt = 0;
  for (auto *PredN : PackN->preds()) {
    (void)PredN;
    ++PredCnt;
  }
  EXPECT_EQ(PredCnt, 1u);
}
