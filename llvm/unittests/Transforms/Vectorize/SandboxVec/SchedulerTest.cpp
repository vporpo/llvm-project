//===- SchedulerTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIRVec.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("SchedulerTest", errs());
  return Mod;
}

/// \Returns true if the Node that corresponds to \p SBI is in the scheduler's
/// schedule list.
static bool inList(sandboxir::Instruction *SBI,
                   const sandboxir::Scheduler &Sched) {
  if (Sched.scheduleEmpty())
    return false;
  const sandboxir::DependencyGraph::Node *N = Sched.getDAG().getNode(SBI);
  for (auto *SB = Sched.getTop(); SB != nullptr; SB = SB->getNext()) {
    if (SB->contains(N))
      return true;
  }
  return false;
}

/// \Returns true if \p I1 and \p I2 belong to the same bundle.
static sandboxir::SchedBundle *areBundle(sandboxir::Instruction *I1,
                                         sandboxir::Instruction *I2,
                                         const sandboxir::Scheduler &Sched,
                                         uint64_t CarryBundleSize = 0) {
  auto *N1 = Sched.getDAG().getNode(I1);
  auto *N2 = Sched.getDAG().getNode(I2);
  assert(N1->getBundle() != nullptr && "Expected non-null Bundle");
  assert(N2->getBundle() != nullptr && "Expected non-null Bundle");
  if (N1->getBundle() != N2->getBundle())
    return nullptr;
  // The bundle contains more nodes than those in the arguments.
  if (CarryBundleSize + 2 != N1->getBundle()->size())
    return nullptr;
  return N1->getBundle();
}

/// \Returns true if \p Instrs... all belong to the same bundle.
template <typename... Ts>
static sandboxir::SchedBundle *
areBundle(Ts... Instrs, sandboxir::Instruction *ILast,
          const sandboxir::Scheduler &Sched, uint64_t CarryBundleSize = 0) {
  sandboxir::SchedBundle *InstrsBundle =
      areBundle(Instrs..., Sched, CarryBundleSize + 1);
  if (InstrsBundle == nullptr)
    return nullptr;
  return Sched.getDAG().getNode(ILast)->getBundle() != InstrsBundle
             ? InstrsBundle
             : nullptr;
}

/// \Returns true if \p I1 is scheduled before \p I2.
static bool isScheduledBefore(sandboxir::Instruction *I1,
                              sandboxir::Instruction *I2,
                              sandboxir::Scheduler &Sched) {
  auto *N1 = Sched.getDAG().getNode(I1);
  auto *N2 = Sched.getDAG().getNode(I2);
  bool Is = N1->getBundle()->comesBefore(N2->getBundle());
  assert(Is == I1->comesBefore(I2) && "Schedule and BB out-of-sync!");
  return Is;
}

static uint64_t getNumScheduledNodes(const sandboxir::Scheduler &Sched) {
  uint64_t Cnt = 0;
  for (sandboxir::SchedBundle *SB = Sched.getTop(); SB != nullptr;
       SB = SB->getNext())
    Cnt += SB->size();
  return Cnt;
}

#ifndef NDEBUG
// Checks the text representation of a BB at construction and at `now()`.
class BBIRChecker {
  sandboxir::BasicBlock *SBBB;
  std::string Str;
  raw_string_ostream OS;

public:
  explicit BBIRChecker(sandboxir::BasicBlock *SBBB) : SBBB(SBBB), OS(Str) {
    OS << *SBBB;
  }
  const std::string &before() const { return Str; }
  const std::string now() const {
    std::string Str2;
    raw_string_ostream OS2(Str2);
    OS2 << *SBBB;
    return Str2;
  }
};
#endif

static BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    if (BB.getName() == Name)
      return &BB;
  llvm_unreachable("Expected to find basic block!");
}

// It is the scheduler's responsibility to schedule PHIs before other
// instructions and the terminator as the last instruction in the BB.
TEST(Scheduler, PHIsAndTerminator) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(i8 %v0, i8 %v1, i8 %v2) {
bb0:
  %add0 = add i8 %v0, %v0
  br label %bb1

bb1:
  %phi0 = phi i8 [ 0, %bb0 ], [ 1, %bb1 ]
  %add1 = add i8 %v1, %v1
  %add2 = add i8 %v2, %v2
  br label %bb1

bb2:
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

  auto *BB0 = Ctx.createBasicBlock(getBasicBlockByName(F, "bb0"));
  auto It = BB0->begin();
  auto *Add0 = &*It++;
  auto *Br0 = &*It++;

  auto *BB1 = Ctx.createBasicBlock(getBasicBlockByName(F, "bb1"));
  It = BB1->begin();
  auto *Phi0 = &*It++;
  auto *Add1 = &*It++;
  (void)Add1;
  auto *Add2 = &*It++;
  (void)Add2;
  auto *Br1 = &*It++;
  {
#ifndef NDEBUG
    // Check that we crash if we use a scheduler for a different BB
    sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB1);
    EXPECT_DEATH(Sched.trySchedule({Add0}), ".*different BB.*");
#endif
  }
  {
    sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB0);
    Sched.startTracking(BB0);
    // Scheduling Add0 should automatically schedule Br0 below it.
    EXPECT_TRUE(Sched.trySchedule({Add0}));
    EXPECT_TRUE(Add0->comesBefore(Br0));
    Sched.accept();
  }

  {
    sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB1);
    Sched.startTracking(BB1);
    EXPECT_TRUE(Sched.trySchedule({Phi0}));
    EXPECT_EQ(&*BB1->begin(), Phi0);
    EXPECT_EQ(&*BB1->rbegin(), Br1);
    Sched.accept();
  }
}

// It is the scheduler's responsibility to schedule landingpads as the first
// non-PHI instrs.
TEST(Scheduler, Landingpad) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
declare void @bar()
declare void @baz(ptr)
define void @foo(ptr %ptr, ptr %ptr1) personality ptr @bar {
entry:
  invoke void @baz(ptr %ptr) to label %label unwind label %lpad
  ret void

label:
  ret void

lpad:
  %phi0 = phi i8 [ 0, %entry ], [ 1, %lpad ]
  %pad = landingpad { ptr, i32 }
           catch ptr null
  store {ptr, i32} %pad, ptr %ptr1
  store i8 %phi0, ptr %ptr
  br label %lpad
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
  (void)SBF;
  auto *LpadBB = Ctx.getBasicBlock(getBasicBlockByName(F, "lpad"));
  auto It = LpadBB->begin();
  auto *Phi = &*It++;
  auto *Pad = &*It++;
  auto *St0 = &*It++;
  (void)St0;
  auto *St1 = &*It++;
  (void)St1;
  auto *Br = &*It++;
  (void)Br;
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(LpadBB);
  Sched.startTracking(LpadBB);

  EXPECT_TRUE(Sched.trySchedule({Phi}));
  It = LpadBB->begin();
  EXPECT_EQ(&*It++, Phi);
  EXPECT_EQ(&*It++, Pad);
#ifndef NDEBUG
  LpadBB->verify();
#endif
}

TEST(Scheduler, BundleEraseNode) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr) {
  %ldA = load float, ptr %ptr
  %ldB1 = load float, ptr %ptr
  %ldB2 = load float, ptr %ptr
  %ldC = load float, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  sandboxir::DependencyGraph &DAG = Sched.getDAG();
  auto It = BB->begin();
  auto *LdA = &*It++;
  auto *LdB1 = &*It++;
  auto *LdB2 = &*It++;
  auto *LdC = &*It++;

  EXPECT_TRUE(Sched.trySchedule({LdC}));
  EXPECT_TRUE(Sched.trySchedule({LdB1, LdB2}));
  EXPECT_TRUE(Sched.trySchedule({LdA}));
  sandboxir::SchedBundle *B1 = Sched.getBundle(LdA);
  (void)B1;
  sandboxir::SchedBundle *B2 = Sched.getBundle(LdB1);
  sandboxir::SchedBundle *B3 = Sched.getBundle(LdC);
  (void)B3;

  auto *NA = DAG.getNode(LdA);
  (void)NA;
  auto *NB1 = DAG.getNode(LdB1);
  auto *NB2 = DAG.getNode(LdB2);
  auto *NC = DAG.getNode(LdC);
  (void)NC;
  EXPECT_EQ((*B2)[0], NB1);
  EXPECT_EQ((*B2)[1], NB2);
  EXPECT_EQ(*B2, *B2);
  B2->remove(NB2);
  EXPECT_EQ(B2->size(), 1u);
  EXPECT_EQ(*B2->begin(), NB1);

  // Check that B2 is removed from List.
  B2->remove(NB1);
  EXPECT_FALSE(inList(LdB2, Sched));
}

// Make sure UnscheduledSuccs are reset correctly with schedule revert.
TEST(Scheduler, UnscheduledSuccsWithRevert) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld0 = load i8, ptr %ptr
  store i8 %ld0, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  auto &DAG = Sched.getDAG();
#ifndef NDEBUG
  BBIRChecker BBChk(BB);
#endif
  auto It = BB->begin();
  auto *Ld = &*It++;
  auto *St = &*It++;
  Sched.startTracking(BB);
  Sched.startScheduling();
  DAG.extend({St});
  auto *StN = DAG.getNode(St);
  EXPECT_EQ(StN->getNumUnscheduledSuccs(), 0u);

  DAG.extend({Ld});
  auto *LdN = DAG.getNode(Ld);
  EXPECT_EQ(LdN->getNumUnscheduledSuccs(), 1u);

  DAG.resetView();
  EXPECT_TRUE(Sched.trySchedule({St}));
  EXPECT_EQ(getNumScheduledNodes(Sched), 1u);
  EXPECT_EQ(StN->getNumUnscheduledSuccs(), 0u);

  // Reset the schedule and re-extend the view.
  // Make sure the UnscheduledSuccs are correct.
  Sched.revert();
  Sched.startScheduling();
  DAG.extend(DmpVector<sandboxir::Value *>{St, Ld});
  EXPECT_EQ(LdN->getNumUnscheduledSuccs(), 1u);
  EXPECT_EQ(StN->getNumUnscheduledSuccs(), 0u);
}

TEST(Scheduler, SimpleBottomUp) {
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
#ifndef NDEBUG
  BBIRChecker BBChk(BB);
#endif
  auto It = BB->begin();
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *St0 = &*It++;
  auto *St1 = &*It++;
  Sched.startTracking(BB);
  EXPECT_TRUE(Sched.trySchedule({St0, St1}));
  EXPECT_EQ(getNumScheduledNodes(Sched), 2u);
  EXPECT_TRUE(areBundle(St0, St1, Sched));
  EXPECT_TRUE(inList(St0, Sched));
  EXPECT_TRUE(inList(St1, Sched));
  EXPECT_TRUE(Sched.trySchedule({Ld0, Ld1}));
  EXPECT_TRUE(areBundle(Ld0, Ld1, Sched));
  EXPECT_EQ(getNumScheduledNodes(Sched), 4u);
  EXPECT_TRUE(inList(St0, Sched));
  EXPECT_TRUE(inList(St1, Sched));
  EXPECT_TRUE(inList(Ld0, Sched));
  EXPECT_TRUE(inList(Ld1, Sched));
  EXPECT_TRUE(isScheduledBefore(Ld0, St0, Sched));
  EXPECT_TRUE(isScheduledBefore(Ld1, St1, Sched));

  // Check that `revert()` works.
  Sched.revert();
  EXPECT_EQ(Sched.scheduleSize(), 0u);
  EXPECT_EQ(Sched.getDAG().getNode(Ld0)->getBundle(), nullptr);
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif

  EXPECT_TRUE(Sched.trySchedule({St0, St1}));
  EXPECT_TRUE(areBundle(St0, St1, Sched));
  EXPECT_EQ(getNumScheduledNodes(Sched), 2u);
  EXPECT_TRUE(inList(St0, Sched));
  EXPECT_TRUE(inList(St1, Sched));
  EXPECT_TRUE(Sched.trySchedule({Ld0, Ld1}));
  EXPECT_TRUE(areBundle(St0, St1, Sched));
  EXPECT_EQ(getNumScheduledNodes(Sched), 4u);
  EXPECT_TRUE(inList(St0, Sched));
  EXPECT_TRUE(inList(St1, Sched));
  EXPECT_TRUE(inList(Ld0, Sched));
  EXPECT_TRUE(inList(Ld1, Sched));
  EXPECT_TRUE(isScheduledBefore(Ld0, St0, Sched));
  EXPECT_TRUE(isScheduledBefore(Ld1, St1, Sched));
  Sched.revert();
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif

  // Check that the DAG roots are calculated correctly.
  EXPECT_TRUE(Sched.trySchedule({Ld0, Ld1}));
  EXPECT_TRUE(areBundle(Ld0, Ld1, Sched));
  EXPECT_EQ(getNumScheduledNodes(Sched), 2u);
  EXPECT_TRUE(inList(Ld0, Sched));
  EXPECT_TRUE(inList(Ld1, Sched));
  Sched.revert();
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif

  // There is no dependency between St0-Ld1, so it should be legal.
  EXPECT_TRUE(Sched.trySchedule({St0, Ld1}));
  EXPECT_TRUE(areBundle(St0, Ld1, Sched));
  Sched.revert();
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif
  // There is no dependency between St1-Ld0, so it should be legal.
  EXPECT_TRUE(Sched.trySchedule({St1, Ld0}));
  EXPECT_TRUE(areBundle(St1, Ld0, Sched));
  Sched.revert();
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif
  // Check that we can't schedule instructions that depend on each other.
  EXPECT_FALSE(Sched.trySchedule({St0, Ld0}));
  Sched.revert();
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif
  EXPECT_FALSE(Sched.trySchedule({St1, Ld1}));
  Sched.revert();
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif
  EXPECT_FALSE(Sched.trySchedule({St0, St1, Ld0}));
  Sched.revert();
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif
  EXPECT_FALSE(Sched.trySchedule({St0, St1, Ld1}));
  Sched.revert();
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif
  EXPECT_FALSE(Sched.trySchedule({St0, St1, Ld0, Ld1}));
  Sched.revert();
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif

  // Check that we don't extend a region on both directions.
  EXPECT_TRUE(Sched.trySchedule({St0}));
#ifndef NDEBUG
  EXPECT_DEATH(Sched.trySchedule({St1, Ld1}), ".*");
#endif
}

//  Ld
//  | \
// Sub |
//  | /
//  Add
TEST(Scheduler, DiamondDeps) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1) {
  %ld0 = load i8, ptr %ptr0
  %ld1 = load i8, ptr %ptr1
  %sub0 = sub i8 %ld0, 0
  %sub1 = sub i8 %ld1, 1
  %add0 = add i8 %sub0, %ld0
  %add1 = add i8 %sub1, %ld1
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
#ifndef NDEBUG
  BBIRChecker BBChk(BB);
#endif
  auto It = BB->begin();
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *Sub0 = &*It++;
  auto *Sub1 = &*It++;
  auto *Add0 = &*It++;
  auto *Add1 = &*It++;
  Sched.startTracking(BB);

  EXPECT_TRUE(Sched.trySchedule({Add0, Add1}));
  EXPECT_TRUE(Sched.trySchedule({Ld0, Ld1}));
  EXPECT_TRUE(isScheduledBefore(Ld0, Sub0, Sched));
  EXPECT_TRUE(isScheduledBefore(Sub0, Add0, Sched));

  EXPECT_TRUE(Sched.trySchedule({Sub0, Sub1}));
  EXPECT_TRUE(isScheduledBefore(Sub0, Add0, Sched));
  EXPECT_TRUE(Ld0->comesBefore(Sub0));

  Sched.revert();
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif

  EXPECT_TRUE(Sched.trySchedule({Add0, Add1}));
  EXPECT_TRUE(Sched.trySchedule({Sub0, Sub1}));
  EXPECT_TRUE(Sched.trySchedule({Ld0, Ld1}));
  EXPECT_TRUE(isScheduledBefore(Ld0, Sub0, Sched));
  EXPECT_TRUE(isScheduledBefore(Sub0, Add0, Sched));
  Sched.revert();
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif

  EXPECT_TRUE(Sched.trySchedule({Add0, Add1}));
  EXPECT_TRUE(Sched.trySchedule({Sub0}));
  EXPECT_TRUE(Sched.trySchedule({Sub1}));
  EXPECT_TRUE(Sched.trySchedule({Ld0, Ld1}));
}

//  Ld
//  | \
// Sub |
//  |  |
// Add |
//  | /
//  Mul
TEST(Scheduler, DeeperTriangularDeps) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1) {
  %ld0 = load i8, ptr %ptr0
  %ld1 = load i8, ptr %ptr1
  %sub0 = sub i8 %ld0, 0
  %sub1 = sub i8 %ld1, 1
  %add0 = add i8 %sub0, 0
  %add1 = add i8 %sub1, 1
  %mul0 = mul i8 %add0, %ld0
  %mul1 = mul i8 %add1, %ld1
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
#ifndef NDEBUG
  BBIRChecker BBChk(BB);
#endif
  auto It = BB->begin();
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *Sub0 = &*It++;
  auto *Sub1 = &*It++;
  auto *Add0 = &*It++;
  auto *Add1 = &*It++;
  auto *Mul0 = &*It++;
  auto *Mul1 = &*It++;
  Sched.startTracking(BB);

  EXPECT_TRUE(Sched.trySchedule({Mul0, Mul1}));
  EXPECT_TRUE(Sched.trySchedule({Ld0, Ld1}));
  EXPECT_TRUE(Sched.trySchedule({Add0, Add1}));
  EXPECT_TRUE(Sched.trySchedule({Sub0, Sub1}));
  EXPECT_TRUE(Ld0->comesBefore(Sub0));
  EXPECT_TRUE(isScheduledBefore(Sub0, Add0, Sched));
  EXPECT_TRUE(isScheduledBefore(Add0, Mul0, Sched));
  Sched.revert();
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif

  EXPECT_TRUE(Sched.trySchedule({Mul0, Mul1}));
  EXPECT_TRUE(Sched.trySchedule({Ld0, Ld1}));
  EXPECT_TRUE(Sched.trySchedule({Add0, Add1}));
  // TODO: It is a bit strange that trySchedule() succeeds after {Add0,Add1}
  //       though it does get scheduled correctly.
  EXPECT_TRUE(Sched.trySchedule({Sub0, Sub1}));
  EXPECT_TRUE(Ld0->comesBefore(Sub0));
  EXPECT_TRUE(isScheduledBefore(Sub0, Add0, Sched));
  EXPECT_TRUE(isScheduledBefore(Add0, Mul0, Sched));
  Sched.revert();
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif
}

// o Scheduled
// |
// New
// |
// o Scheduled
TEST(Scheduler, NewInstrThatDependsOnScheduled) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(i8 %v, ptr %ptr0, ptr %ptr1, ptr %ptr2) {
  store i8 %v, ptr %ptr0
  store i8 %v, ptr %ptr1
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
#ifndef NDEBUG
  BBIRChecker BBChk(BB);
#endif
  auto It = BB->begin();
  auto *St0 = &*It++;
  auto *St1 = &*It++;
  Sched.startTracking(BB);
  auto *V = SBF->getArg(0);
  auto *Ptr3 = SBF->getArg(3);

  // Schedule all instrs.
  EXPECT_TRUE(Sched.trySchedule({St1}));
  EXPECT_TRUE(Sched.trySchedule({St0}));
  auto *NewI = sandboxir::StoreInst::create(
      V, Ptr3, /*Align=*/std::nullopt, St1, Ctx);
  (void)NewI;
  auto &DAG = Sched.getDAG();
  auto *St0N = DAG.getNode(St0);
  auto *NewN = DAG.getNode(NewI);
  auto *St1N = DAG.getNode(St1);
  EXPECT_TRUE(NewN->dependsOn(St0N));
  EXPECT_TRUE(St1N->dependsOn(NewN));
  EXPECT_TRUE(St0N->allSuccsReady());

  Sched.accept();
}

// Once NewN is added to the DAG, we extend the DAG until NewN.
// This exposes a bug where N0 is scheduled while being in the ReadyList.
//
// NewN
// |
// o N0 Not-scheduled but Ready
// |
// o N1 Scheduled
TEST(Scheduler, DontCreateBundleOfNodeInReadyList) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(i8 %v0, i8 %v1, i8 %v2, ptr %ptr) {
  store i8 %v0, ptr %ptr
  store i8 %v1, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
#ifndef NDEBUG
  BBIRChecker BBChk(BB);
#endif
  auto It = BB->begin();
  auto *St0 = &*It++;
  auto *St1 = &*It++;
  Sched.startTracking(BB);
  unsigned Idx = 0;
  auto *V0 = SBF->getArg(Idx++);
  (void)V0;
  auto *V1 = SBF->getArg(Idx++);
  (void)(V1);
  auto *V2 = SBF->getArg(Idx++);
  auto *Ptr = SBF->getArg(Idx++);

  // Schedule all instrs.
  Sched.startScheduling();
  auto &DAG = Sched.getDAG();
  sandboxir::SchedulerAttorney::extendRegionAndUpdateReadyList(Sched,
                                                               {St0, St1});
  EXPECT_TRUE(Sched.trySchedule({St1}));
  auto *St0N = DAG.getNode(St0);
  EXPECT_TRUE(Sched.getReadyList().contains(St0N));
  auto *NewI = sandboxir::StoreInst::create(
      V2, Ptr, /*Align=*/std::nullopt, St0, Ctx);
  EXPECT_FALSE(Sched.getReadyList().contains(St0N));
  (void)NewI;
  Sched.accept();
}

// TEST(Scheduler, DISABLED_BottomUpAndTopDown) {
//   LLVMContext C;
//   std::unique_ptr<Module> M = parseIR(C, R"IR(
// define void @foo(ptr %ptr) {
//   store i8 0, ptr %ptr
//   store i8 1, ptr %ptr
//   store i8 2, ptr %ptr
//   store i8 3, ptr %ptr
//   ret void
// }
// )IR");
//   Function &F = *M->getFunction("foo");

//   DominatorTree DT(F);
//   TargetLibraryInfoImpl TLII;
//   TargetLibraryInfo TLI(TLII);
//   DataLayout DL(M.get());
//   AssumptionCache AC(F);
//   BasicAAResult BAA(DL, F, TLI, AC, &DT);
//   AAResults AA(TLI);
//   AA.addAAResult(BAA);

//   sandboxir::SBVecContext Ctx(C, AA);
//   BasicBlock *BB = &*F.begin();
//   sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
//   BBIRChecker BBChk(BB);
//   auto It = BB->begin();
//   Instruction *St0 = &*It++;
//   Instruction *St1 = &*It++;
//   Instruction *St2 = &*It++;
//   Instruction *St3 = &*It++;
//   (void)St1;
//   Sched.startTracking(BB);

//   EXPECT_TRUE(Sched.trySchedule({St3}));
//   EXPECT_TRUE(Sched.trySchedule({St2}));
//   EXPECT_TRUE(isScheduledBefore(St2, St3, Sched));
//   // St0 should be BottomUp, not TopDown.
//   EXPECT_DEATH(Sched.trySchedule({St0}, TopDown), ".*");
//   Sched.revert();
//   EXPECT_EQ(BBChk.before(), BBChk.now());

//   EXPECT_TRUE(Sched.trySchedule({St0}));
//   // St1 should be TopDown, not BottomUp.
//   EXPECT_DEATH(Sched.trySchedule({St1}), ".*");
//   Sched.revert();
//   EXPECT_EQ(BBChk.before(), BBChk.now());

//   EXPECT_TRUE(Sched.trySchedule({St2}));
//   EXPECT_TRUE(Sched.trySchedule({St0}));
//   // `St1` has already been scheduled so this should succeed.
//   EXPECT_TRUE(Sched.trySchedule({St1}));
//   EXPECT_TRUE(Sched.trySchedule({St3}, TopDown));
//   EXPECT_TRUE(isScheduledBefore(St0, St1, Sched));
//   EXPECT_TRUE(isScheduledBefore(St1, St2, Sched));
//   EXPECT_TRUE(isScheduledBefore(St2, St3, Sched));
//   Sched.revert();
//   EXPECT_EQ(BBChk.before(), BBChk.now());
// }

TEST(Scheduler, TopDown) {
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
#ifndef NDEBUG
  BBIRChecker BBChk(BB);
#endif
  auto It = BB->begin();
  sandboxir::Instruction *Ld0 = &*It++;
  sandboxir::Instruction *Ld1 = &*It++;
  sandboxir::Instruction *St0 = &*It++;
  sandboxir::Instruction *St1 = &*It++;
  Sched.startTracking(BB);

  bool TopDown = true;
  // Schedule downwards
  EXPECT_TRUE(Sched.trySchedule({Ld0, Ld1}, TopDown));
  EXPECT_TRUE(Sched.trySchedule({St0, St1}, TopDown));
  EXPECT_TRUE(Sched.getReadyList().empty());
  EXPECT_TRUE(Sched.getReadyList().empty());
  Sched.revert();
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif

  EXPECT_TRUE(Sched.trySchedule({Ld1}, TopDown));
  EXPECT_TRUE(Sched.trySchedule({St1}, TopDown));
#ifndef NDEBUG
  EXPECT_DEATH(Sched.trySchedule({St0}, /*TopDown=*/!TopDown), ".*");
#endif
  Sched.revert();
#ifndef NDEBUG
  EXPECT_EQ(BBChk.before(), BBChk.now());
#endif
}

TEST(Scheduler, Apply) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr0, ptr noalias %ptr1) {
  %ld0 = load i8, ptr %ptr0
  store i8 %ld0, ptr %ptr0
  %ld1 = load i8, ptr %ptr1
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  auto It = BB->begin();
  auto *Ld0 = &*It++;
  auto *St0 = &*It++;
  auto *Ld1 = &*It++;
  auto *St1 = &*It++;
  Sched.startTracking(BB);

  EXPECT_TRUE(Sched.trySchedule({St0, St1}));
  EXPECT_TRUE(Sched.trySchedule({Ld0, Ld1}));
  Sched.accept();
  It = BB->begin();
  EXPECT_EQ(&*It++, Ld0);
  EXPECT_EQ(&*It++, Ld1);
  EXPECT_EQ(&*It++, St0);
  EXPECT_EQ(&*It++, St1);
}

TEST(Scheduler, EraseInstrs) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr) {
  %ld0 = load i8, ptr %ptr
  store i8 %ld0, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  auto It = BB->begin();
  auto *Ld = &*It++;
  auto *St = &*It++;
  auto *Ret = &*It++;
  Sched.startTracking(BB);

  EXPECT_TRUE(Sched.trySchedule({Ret}));
  EXPECT_TRUE(Sched.trySchedule({St}));
  EXPECT_TRUE(Sched.trySchedule({Ld}));
  Sched.accept();

  Sched.notifyRemove(St);
  St->eraseFromParent();
  Sched.accept();
  It = BB->begin();
  EXPECT_EQ(&*It++, Ld);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
}

TEST(Scheduler, EraseInstrs2) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr) {
  %ld0 = load i8, ptr %ptr
  store i8 %ld0, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  auto It = BB->begin();
  auto *Ld = &*It++;
  auto *St = &*It++;
  auto *Ret = &*It++;
  Sched.startTracking(BB);

  EXPECT_TRUE(Sched.trySchedule({Ret}));
  EXPECT_TRUE(Sched.trySchedule({St}));
  EXPECT_TRUE(Sched.trySchedule({Ld}));
  Ctx.getTracker().accept();
  Sched.startFresh(BB);

  Sched.notifyRemove(St);
  St->eraseFromParent();
  It = BB->begin();
  EXPECT_EQ(&*It++, Ld);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
}

TEST(Scheduler, BundleNodesInInitOrder) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr) {
  %ld0 = load i8, ptr %ptr
  %ld1 = load i8, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  auto It = BB->begin();
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  Sched.startTracking(BB);

  sandboxir::DependencyGraph &DAG = Sched.getDAG();
  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});
  {
    sandboxir::SchedBundle B1({Ld0, Ld1}, Sched);
    EXPECT_EQ((*std::next(B1.begin(), 0))->getInstruction(), Ld0);
    EXPECT_EQ((*std::next(B1.begin(), 1))->getInstruction(), Ld1);
  }
  {
    sandboxir::SchedBundle B1({Ld1, Ld0}, Sched);
    EXPECT_EQ((*std::next(B1.begin(), 0))->getInstruction(), Ld1);
    EXPECT_EQ((*std::next(B1.begin(), 1))->getInstruction(), Ld0);
  }
}

TEST(Scheduler, Cluster) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr, ptr noalias %ptr2) {
  %ld0 = load i8, ptr %ptr
  store i8 %ld0, ptr %ptr2
  %ld1 = load i8, ptr %ptr
  store i8 %ld1, ptr %ptr2
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  auto It = BB->begin();
  auto *Ld0 = &*It++;
  auto *St0 = &*It++;
  auto *Ld1 = &*It++;
  auto *St1 = &*It++;
  auto *Ret = &*It++;
  Sched.startTracking(BB);

  sandboxir::DependencyGraph &DAG = Sched.getDAG();
  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});

  {
    // Cluster() will move all instructions just before the lowest instruction
    // in the bundle.
    sandboxir::SchedBundle B1({Ld0, Ld1}, Sched);
    B1.cluster();
    It = BB->begin();
    EXPECT_EQ(&*It++, St0);
    EXPECT_EQ(&*It++, Ld0);
    EXPECT_EQ(&*It++, Ld1);
    EXPECT_EQ(&*It++, St1);
    EXPECT_EQ(&*It++, Ret);
  }

  Ld0->moveBefore(St0);
  DAG.clear();
  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});

  {
    // Cluster(It) will cluster them just before It.
    sandboxir::SchedBundle B1({Ld0, Ld1}, Sched);
    B1.cluster(Ret->getIterator(), BB);
    It = BB->begin();
    EXPECT_EQ(&*It++, St0);
    EXPECT_EQ(&*It++, St1);
    EXPECT_EQ(&*It++, Ld0);
    EXPECT_EQ(&*It++, Ld1);
    EXPECT_EQ(&*It++, Ret);
  }

  Ld0->moveBefore(St0);
  Ld1->moveBefore(St1);
  DAG.clear();
  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});

  {
    // Cluster(end)
    sandboxir::SchedBundle B1({Ld0, Ld1, Ret}, Sched);
    B1.cluster(BB->end(), BB);
    It = BB->begin();
    EXPECT_EQ(&*It++, St0);
    EXPECT_EQ(&*It++, St1);
    EXPECT_EQ(&*It++, Ld0);
    EXPECT_EQ(&*It++, Ld1);
    EXPECT_EQ(&*It++, Ret);
  }

  Ld0->moveBefore(St0);
  Ld1->moveBefore(St1);
  DAG.clear();
  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});

  {
    // Cluster(It) where It is a bundle It.
    sandboxir::SchedBundle B1({Ld0, Ld1}, Sched);
    B1.cluster(Ld1->getIterator(), BB);
    It = BB->begin();
    EXPECT_EQ(&*It++, St0);
    EXPECT_EQ(&*It++, Ld0);
    EXPECT_EQ(&*It++, Ld1);
    EXPECT_EQ(&*It++, St1);
    EXPECT_EQ(&*It++, Ret);
  }

  Ld0->moveBefore(St0);
  Ld1->moveBefore(St1);
  DAG.clear();
  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});

  {
    // Cluster(It) where It is a bundle instr.
    sandboxir::SchedBundle B1({Ld0, Ld1}, Sched);
    B1.cluster(Ld0->getIterator(), BB);
    It = BB->begin();
    EXPECT_EQ(&*It++, Ld0);
    EXPECT_EQ(&*It++, Ld1);
    EXPECT_EQ(&*It++, St0);
    EXPECT_EQ(&*It++, St1);
    EXPECT_EQ(&*It++, Ret);
  }
}

TEST(Scheduler, AutoAddNewNode) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr1, ptr noalias %ptr2, ptr noalias %ptr3,
                 i8 %s1, i8 %s2, i8 %s3,
                 <2 x i8> %vec) {
  %ld0 = load i8, ptr %ptr1
  store i8 %s1, ptr %ptr1
  store i8 %s2, ptr %ptr2
  store i8 %s3, ptr %ptr1
  %ld1 = load i8, ptr %ptr1
  %ld2 = load i8, ptr %ptr1
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  auto It = BB->begin();
  auto *Ld0 = &*It++;
  auto *St1 = &*It++;
  auto *St2 = &*It++;
  auto *St3 = &*It++;
  auto *Ld1 = &*It++;
  auto *Ld2 = &*It++;

  EXPECT_TRUE(Sched.trySchedule({Ld2}));
  EXPECT_TRUE(Sched.trySchedule({Ld1}));
  EXPECT_TRUE(Sched.trySchedule({St3}));
  EXPECT_TRUE(Sched.trySchedule({St1, St2}));
  EXPECT_TRUE(Sched.trySchedule({Ld0}));

  Sched.startTracking(BB);
  auto *Vec = SBF->getArg(6);
  EXPECT_TRUE(Vec->getType()->isVectorTy());
  auto *Ptr = cast<sandboxir::StoreInst>(St2)->getPointerOperand();
  auto *NewSI = sandboxir::StoreInst::create(
      Vec, Ptr, /*Align=*/std::nullopt, Ld2, Ctx);
  sandboxir::SchedBundle *SB_NewI = Sched.getBundle(NewSI);
  EXPECT_EQ(SB_NewI->getPrev(), Sched.getBundle(Ld1));
  EXPECT_EQ(SB_NewI->getNext(), Sched.getBundle(Ld2));
  // The newly added node should be marked as 'Scheduled'.
  EXPECT_TRUE(Sched.getDAG().getNode(NewSI)->isScheduled());

  sandboxir::SchedBundle *SB_Ld0 = Sched.getBundle(Ld0);
  sandboxir::SchedBundle *SB_St1 = Sched.getBundle(St1);
  sandboxir::SchedBundle *SB_St2 = Sched.getBundle(St2);
  sandboxir::SchedBundle *SB_St3 = Sched.getBundle(St3);
  sandboxir::SchedBundle *SB_Ld1 = Sched.getBundle(Ld1);
  sandboxir::SchedBundle *SB_Ld2 = Sched.getBundle(Ld2);
  (void)SB_Ld2;
  EXPECT_EQ(SB_St1, SB_St2);
  sandboxir::SchedBundle *SB_St1St2 = SB_St1;

  EXPECT_EQ(SB_Ld0->getNext(), SB_St1St2);
  EXPECT_EQ(SB_St1St2->getNext(), SB_St3);
  EXPECT_EQ(SB_St3->getNext(), SB_Ld1);
  EXPECT_EQ(SB_Ld1->getNext(), SB_NewI);
  EXPECT_EQ(SB_NewI->getNext(), SB_Ld2);
  Sched.accept();
}

TEST(Scheduler, AddNew) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr1, ptr noalias %ptr2, ptr noalias %ptr3,
                 i8 %s1, i8 %s2, i8 %s3,
                 <2 x i8> %vec) {
  %ld0 = load i8, ptr %ptr1
  %add0 = add i8 %ld0, %ld0
  %sub0 = sub i8 %ld0, %add0

  %ld1 = load i8, ptr %ptr2
  %add1 = add i8 %ld1, %ld1
  %sub1 = sub i8 %ld1, %add1
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  auto It = BB->begin();
  auto *Ld0 = &*It++;
  auto *Add0 = &*It++;
  auto *Sub0 = &*It++;
  auto *Ld1 = &*It++;
  auto *Add1 = &*It++;
  auto *Sub1 = &*It++;
  Sched.startTracking(BB);

  EXPECT_TRUE(Sched.trySchedule({Sub0, Sub1}));
  auto *Ptr = SBF->getArg(0);
  auto *NewI = sandboxir::StoreInst::create(
      Ld0, Ptr, /*Align=*/std::nullopt, Add0, Ctx);

  It = BB->begin();
  EXPECT_EQ(&*It++, Ld0);
  EXPECT_EQ(&*It++, NewI);
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Ld1);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(&*It++, Sub0);
  EXPECT_EQ(&*It++, Sub1);
}

TEST(Scheduler, ReadyListInsertOrder_Term) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld0 = load i8, ptr %ptr
  %ld1 = load i8, ptr %ptr
  %ld2 = load i8, ptr %ptr
  %ld3 = load i8, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  auto &DAG = Sched.getDAG();
  auto It = BB->begin();
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *Ld2 = &*It++;
  auto *Ld3 = &*It++;
  auto *Ret = &*It++;

  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});
  auto &ReadyList =
      const_cast<sandboxir::ReadyListContainer &>(Sched.getReadyList());
  auto *LN0 = DAG.getNode(Ld0);
  auto *LN1 = DAG.getNode(Ld1);
  auto *LN2 = DAG.getNode(Ld2);
  auto *LN3 = DAG.getNode(Ld3);
  auto *RN = DAG.getNode(Ret);
  using Node = sandboxir::DependencyGraph::Node;
  SmallVector<Node *> ExpectedOrder;
  for (auto *N : {RN, LN3, LN2, LN1, LN0}) {
    ReadyList.insert(N);
    ExpectedOrder.push_back(N);
    EXPECT_EQ(ReadyList.getContents(), ExpectedOrder);
  }
}

TEST(Scheduler, ReadyListInsertOrder_PHIs) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
bb0:
  br label %bb1
bb1:
  %phi0 = phi i8 [ 0, %bb0 ], [ 1, %bb1 ]
  %phi1 = phi i8 [ 0, %bb0 ], [ 1, %bb1 ]
  %phi2 = phi i8 [ 0, %bb0 ], [ 1, %bb1 ]
  %ld0 = load i8, ptr %ptr
  %ld1 = load i8, ptr %ptr
  %ld2 = load i8, ptr %ptr
  %ld3 = load i8, ptr %ptr
  br label %bb0
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
  auto *BB1 = Ctx.createBasicBlock(getBasicBlockByName(F, "bb1"));
  auto &DAG = Ctx.getScheduler(BB1)->getDAG();
  auto It = BB1->begin();
  auto *Phi0 = &*It++;
  auto *Phi1 = &*It++;
  auto *Phi2 = &*It++;
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *Ld2 = &*It++;
  auto *Ld3 = &*It++;
  auto *Ret = &*It++;

  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB1->begin(), BB1->getTerminator()});
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB1);
  auto &ReadyList =
      const_cast<sandboxir::ReadyListContainer &>(Sched.getReadyList());
  auto *PN0 = DAG.getNode(Phi0);
  auto *PN1 = DAG.getNode(Phi1);
  auto *PN2 = DAG.getNode(Phi2);
  auto *LN0 = DAG.getNode(Ld0);
  auto *LN1 = DAG.getNode(Ld1);
  auto *LN2 = DAG.getNode(Ld2);
  auto *LN3 = DAG.getNode(Ld3);
  auto *RN = DAG.getNode(Ret);
  using Node = sandboxir::DependencyGraph::Node;

  SmallVector<Node *> ExpectedOrder;
  for (auto *N : {RN, LN3, LN2, LN1, LN0, PN2, PN1, PN0}) {
    ReadyList.insert(N);
    ExpectedOrder.push_back(N);
    EXPECT_EQ(ReadyList.getContents(), ExpectedOrder);
  }
}

TEST(Scheduler, AddSuccsToReady) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr, <2 x i8> %vec) {
  %ptr0 = getelementptr i8, ptr %ptr, i32 0
  %ptr1 = getelementptr i8, ptr %ptr, i32 1
  %ld1 = load i8, ptr %ptr0
  %ld2 = load i8, ptr %ptr1
  store i8 %ld1, ptr %ptr0
  store i8 %ld2, ptr %ptr1
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  auto It = BB->begin();
  auto *Gep0 = &*It++;
  (void)Gep0;
  auto *Gep1 = &*It++;
  (void)Gep1;
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *St0 = &*It++;
  auto *St1 = &*It++;
  auto *Ret = &*It++;
  auto *Vec = SBF->getArg(1);
  Sched.startTracking(BB);
  Sched.startScheduling();

  sandboxir::SchedulerAttorney::extendRegionAndUpdateReadyList(
      Sched, DmpVector<sandboxir::Value *>{&*BB->begin(), St1});
  EXPECT_TRUE(Sched.trySchedule({St0, St1}));

  // At this point the loads should be ready
  auto *LN0 = Sched.getDAG().getNode(Ld0);
  auto *LN1 = Sched.getDAG().getNode(Ld1);
  EXPECT_TRUE(LN0->allSuccsReady());
  EXPECT_TRUE(LN1->allSuccsReady());
#ifndef NDEBUG
  auto &ReadyList =
      const_cast<sandboxir::ReadyListContainer &>(Sched.getReadyList());
  EXPECT_DEATH(ReadyList.insert(LN0), ".*Already in ready list.*");
  EXPECT_DEATH(ReadyList.insert(LN1), ".*Already in ready list.*");
#endif
  EXPECT_TRUE(Sched.getReadyList().contains(LN0));
  EXPECT_TRUE(Sched.getReadyList().contains(LN1));

  auto *Ptr = cast<sandboxir::StoreInst>(St0)->getPointerOperand();
  auto *NewSI = sandboxir::StoreInst::create(
      Vec, Ptr, /*Align=*/std::nullopt, Ret, Ctx);
  sandboxir::SchedBundle *NewSB = Sched.getBundle(NewSI);
  (void)NewSB;
  // The newly added node should be marked as 'Scheduled'.
  auto *NewN = Sched.getDAG().getNode(NewSI);
  EXPECT_TRUE(NewN->isScheduled());

  // NewN depends on the loads
  EXPECT_TRUE(NewN->dependsOn(LN0));

  // But Since LN0 is in the ready list we expect
  EXPECT_TRUE(LN0->allSuccsReady());
}

// Check that scheduled instructions are not moved outside the DAG view.
TEST(Scheduler, ScheduleWithinDAGView) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v) {
  %ptr0 = getelementptr i8, ptr %ptr, i32 0
  %ptr1 = getelementptr i8, ptr %ptr, i32 1

  %ldA0 = load i8, ptr %ptr0
  %ldA1 = load i8, ptr %ptr1

  %ldB0 = load i8, ptr %ptr0
  %ldB1 = load i8, ptr %ptr1

  ; Some random instruction
  %add = add i8 %v, %v

  store i8 %add, ptr %ptr0
  store i8 %add, ptr %ptr1
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
  auto *Gep0 = &*It++;
  (void)Gep0;
  auto *Gep1 = &*It++;
  (void)Gep1;
  auto *LdA0 = &*It++;
  auto *LdA1 = &*It++;
  auto *LdB0 = &*It++;
  auto *LdB1 = &*It++;
  auto *Add = &*It++;
  auto *St0 = &*It++;
  auto *St1 = &*It++;

  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  Sched.startTracking(BB);
  EXPECT_TRUE(Sched.trySchedule({St0, St1}));
  EXPECT_TRUE(Sched.trySchedule({LdA0, LdA1}));
  // Now all instructions between LdA1 and St0 have been scheduled as
  // single-bundles. This triggers re-scheduling.
  EXPECT_TRUE(Sched.trySchedule({LdB0, LdB1}));
  EXPECT_EQ(LdB1->getNextNode(), Add);
}

TEST(Scheduler, ReschedulingClearsReadyList) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld0 = load ptr, ptr %ptr
  %ldA = load i8, ptr %ld0
  %gep = getelementptr ptr, ptr %ld0, i32 0
  store ptr %gep, ptr %ld0
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
  auto *Ld0 = &*It++;
  auto *LdA = &*It++;
  auto *Gep = &*It++;
  auto *St0 = &*It++;

  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  Sched.startTracking(BB);
  const auto &ReadyList = Sched.getReadyList();
  EXPECT_TRUE(Sched.trySchedule({St0}));
  EXPECT_TRUE(Sched.trySchedule({Ld0}));
  // Now all instructions between `Ld0` and `St0` have been scheduled as
  // single-bundles.
  // Schedule {Gep, LdA}. This triggers re-scheduling and clears the DAG View.
  EXPECT_TRUE(Sched.trySchedule({Gep, LdA}));
  // The DAG View should only contain LdB0 and LdB1 and the readylist should
  // be empty.
  EXPECT_TRUE(ReadyList.empty());
}

// Trying to schedule nodes that some of them have already been scheduled used
// to cause a crash because some of the nodes were missing from the DAG.
TEST(Scheduler, ReschedulingMissingFromDAG) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr0, ptr %ptr1, ptr %ptr2) {
  %ld0 = load i8, ptr %ptr0
  %ld1 = load i8, ptr %ptr1
  %ld2 = load i8, ptr %ptr2
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
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *Ld2 = &*It++;

  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  Sched.startTracking(BB);
  EXPECT_TRUE(Sched.trySchedule({Ld2}));
  EXPECT_TRUE(Sched.trySchedule({Ld1}));
  EXPECT_TRUE(Sched.trySchedule({Ld0, Ld1}));
}

// Checks that creating/deleting SandboxIR also automatically updates the DAG.
TEST(Scheduler, AutoDAGFormationWhenSandboxIRGetsUpdated) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %val) {
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
  FunctionAnalysisManager FAM;

  auto &SBF = *Ctx.createFunction(&F);
  auto *BB = &*SBF.begin();
  auto *Ptr = SBF.getArg(0);
  auto *Val = SBF.getArg(1);
  auto *Ret = &*BB->rbegin();

  // Check the DAG's state.
  const sandboxir::InstrInterval &View = Ctx.getDAG(BB).getView();
  const sandboxir::InstrInterval &DAGInterval =
      Ctx.getDAG(BB).getDAGInterval();
  auto *Sched = Ctx.getScheduler(BB);
  for (auto &SBI : *BB)
    Sched->trySchedule({&SBI});

  EXPECT_EQ(View.from(), Ret);
  EXPECT_EQ(View.to(), Ret);
  EXPECT_EQ(DAGInterval.from(), Ret);
  EXPECT_EQ(DAGInterval.to(), Ret);

  auto *NewSBI = sandboxir::StoreInst::create(
      Val, Ptr, /*Align=*/std::nullopt, Ret, Ctx);
  auto *NewI = &*BB->begin();
  EXPECT_EQ(NewI, NewSBI);
  EXPECT_EQ(View.from(), NewSBI);
  EXPECT_EQ(View.to(), Ret);
  EXPECT_EQ(DAGInterval.from(), NewSBI);
  EXPECT_EQ(DAGInterval.to(), Ret);

  Ret->eraseFromParent();
  EXPECT_EQ(View.from(), NewSBI);
  EXPECT_EQ(View.to(), NewSBI);
  EXPECT_EQ(DAGInterval.from(), NewSBI);
  EXPECT_EQ(DAGInterval.to(), NewSBI);

  NewSBI->eraseFromParent();
  EXPECT_EQ(View.from(), nullptr);
  EXPECT_EQ(View.to(), nullptr);
  EXPECT_EQ(DAGInterval.from(), nullptr);
  EXPECT_EQ(DAGInterval.to(), nullptr);
}

TEST(Scheduler, ScheduleAlreadyScheduled) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld0 = load i8, ptr %ptr
  %ld1 = load i8, ptr %ptr
  %ld2 = load i8, ptr %ptr
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
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *Ld2 = &*It++;

  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  Sched.startTracking(BB);
  EXPECT_TRUE(Sched.trySchedule({Ld2}));
  EXPECT_TRUE(Sched.trySchedule({Ld0}));
  // This should have scheduled all Ld0, Ld1, Ld2
  // Now try to schedule Ld1. We already have a schedule for this.
  DmpVector<sandboxir::Value *> LdBndl({Ld1});
  EXPECT_EQ(sandboxir::SchedulerAttorney::getBndlSchedState(Sched, LdBndl),
            sandboxir::SchedulerAttorney::BndlSchedState::FullyScheduled);
  EXPECT_TRUE(Sched.trySchedule(LdBndl));
  EXPECT_TRUE(Sched.getBundle(Ld1));

  // Now try to schedule {Ld1, Ld0} in the same scheduling bundle.
  // This should trigger re-scheduling.
  DmpVector<sandboxir::Value *> NewBndl({Ld1, Ld0});
  EXPECT_EQ(sandboxir::SchedulerAttorney::getBndlSchedState(Sched, NewBndl),
            sandboxir::SchedulerAttorney::BndlSchedState::
                PartiallyOrDifferentlyScheduled);
  // Check that trimming works as expected.
  sandboxir::SchedulerAttorney::trimSchedule(Sched, NewBndl);
  auto *TopSB = Sched.getTop();
  EXPECT_EQ(TopSB->getTopI(), Ld2);
  auto &DAG = Sched.getDAG();
  EXPECT_TRUE(DAG.inView(Ld2));
  EXPECT_FALSE(DAG.inView(Ld1));
  EXPECT_FALSE(DAG.inView(Ld0));
  EXPECT_TRUE(DAG.getNode(Ld2)->isScheduled());

  // Check that re-scheduling works.
  EXPECT_TRUE(Sched.trySchedule({Ld2}));
  EXPECT_TRUE(Sched.trySchedule({Ld0, Ld1}));
}

// Check that trimSchedule for instrs that span multiple bundles doesn't break
// TopSB. So given the bundles:
//
// Bunldes
// -------
// Ld0
// Ld1,Ld2
// Ret
//
// If we trimSchedule(Ld0,Ld1) we should get a valid TopSB == Ret.
//
TEST(Scheduler, TrimSchedule_TopSB) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld0 = load i8, ptr %ptr
  %ld1 = load i8, ptr %ptr
  %ld2 = load i8, ptr %ptr
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
  auto *Ld0 = &*It++;
  auto *Ld1 = &*It++;
  auto *Ld2 = &*It++;
  auto *Ret = &*It++;

  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  auto &DAG = Sched.getDAG();
  (void)DAG;

  EXPECT_TRUE(Sched.trySchedule({Ret}));
  EXPECT_TRUE(Sched.trySchedule({Ld1, Ld2}));
  EXPECT_TRUE(Sched.trySchedule({Ld0}));
  // Now trim the schedule section that includes Ld0 and Ld1.
  // So we trim part of bundle (Ld1, Ld2).
  sandboxir::SchedulerAttorney::trimSchedule(Sched, {Ld0, Ld1});
  auto *TopSB = Sched.getTop();
  EXPECT_TRUE(TopSB != nullptr);
  EXPECT_EQ(TopSB, Sched.getBundle(Ret));
}

// Check that we update the InReadyList flag when we insert/remove.
TEST(Scheduler, InReadyList) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld0 = load ptr, ptr %ptr
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
  auto *Ld0 = &*It++;

  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  auto &DAG = Sched.getDAG();
  Sched.startTracking(BB);
  auto &ReadyList =
      *const_cast<sandboxir::ReadyListContainer *>(&Sched.getReadyList());
  DAG.extend({Ld0});
  auto *N = DAG.getNode(Ld0);
  EXPECT_FALSE(N->isInReadyList());
  ReadyList.insert(N);
  EXPECT_TRUE(N->isInReadyList());
  // Check remove()
  ReadyList.remove(N);
  EXPECT_FALSE(N->isInReadyList());
  // Check clear()
  ReadyList.insert(N);
  EXPECT_TRUE(N->isInReadyList());
  ReadyList.clear();
  EXPECT_FALSE(N->isInReadyList());
  // Check pop()
  ReadyList.insert(N);
  EXPECT_TRUE(N->isInReadyList());
  ReadyList.pop();
  EXPECT_FALSE(N->isInReadyList());
}

// Make sure UnscheduledSuccs gets updated correctly in TopDown.
TEST(Scheduler, UnscheduledSuccsAndNotify_TopDown) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v0, i8 %v1) {
  %foo0 = add i8 %v0, %v0
  store i8 %v1, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  auto &DAG = Sched.getDAG();

  auto *ArgV0 = SBF->getArg(1);
  auto It = BB->begin();
  auto *Foo0 = &*It++;
  auto *St1 = cast<sandboxir::StoreInst>(&*It++);

  auto *Ptr = St1->getPointerOperand();
  Sched.trySchedule({Foo0}, /*TopDown=*/true);
  DAG.extend({St1});
  auto *St1N = DAG.getNode(St1);
  EXPECT_EQ(St1N->getNumUnscheduledSuccs(), 0u);

  auto *St0 = sandboxir::StoreInst::create(
      ArgV0, Ptr, /*Align=*/std::nullopt, St1, Ctx);
  auto *St0N = DAG.getNode(St0);
  EXPECT_EQ(St0N->getNumUnscheduledSuccs(), 0u);
  EXPECT_TRUE(St0N->isScheduled());
  // St1N's counter was incremented when St0N->St1N dependency was created and
  // then decremented when St0N was scheduled.
  EXPECT_EQ(St1N->getNumUnscheduledSuccs(), 0u);
}

// Make sure UnscheduledSuccs gets updated correctly in BottomUp
TEST(Scheduler, UnscheduledSuccsAndNotify_BottomUp) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v0, i8 %v1) {
  store i8 %v1, ptr %ptr
  %foo0 = add i8 %v0, %v0
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  auto &DAG = Sched.getDAG();
  Sched.startScheduling();

  auto *ArgV0 = SBF->getArg(1);
  auto It = BB->begin();
  auto *St1 = cast<sandboxir::StoreInst>(&*It++);
  auto *Foo0 = &*It++;

  auto *Ptr = St1->getPointerOperand();
  Sched.trySchedule({Foo0}, /*TopDown=*/false);

  auto *St0 = sandboxir::StoreInst::create(
      ArgV0, Ptr, /*Align=*/std::nullopt, St1, Ctx);
  auto *St0N = DAG.getNode(St0);
  auto *St1N = DAG.getNode(St1);
  EXPECT_EQ(St1N->getNumUnscheduledSuccs(), 0u);
  EXPECT_TRUE(St0N->isScheduled());
  EXPECT_TRUE(St1N->isScheduled());
  EXPECT_EQ(St0N->getNumUnscheduledSuccs(), 0u);
  EXPECT_EQ(St1N->getNumUnscheduledSuccs(), 0u);
}

TEST(Scheduler, DAGInsertAndAddDeps_UpdateUnschedSuccsOfPreds_UseDef) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v0, ptr %ptr) {
  %add = fadd float %v0, %v0
  store float %add, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  sandboxir::DependencyGraph &DAG = Sched.getDAG();
  Sched.startScheduling();

  auto It = BB->begin();
  auto *Add = &*It++;
  auto *St = &*It++;
  auto *Ret = &*It++;
  DAG.extend(DmpVector<sandboxir::Value *>{Add, BB->getTerminator()});

  auto *AddN = DAG.getNode(Add);
  auto *StN = DAG.getNode(St);
  EXPECT_TRUE(StN->dependsOn(AddN));
  EXPECT_EQ(AddN->getNumUnscheduledSuccs(), 1u);

  // Now create a new instruction with Add as its operand.
  auto *Ptr = SBF->getArg(1);
  auto *NewI =
      sandboxir::StoreInst::create(Add, Ptr, /*Align=*/std::nullopt,
                                            /*InsertBefore=*/Ret, Ctx);
  auto *NewN = DAG.getNode(NewI);
  EXPECT_TRUE(NewN != nullptr);
  EXPECT_TRUE(NewN->dependsOn(AddN));
  // Check the UnscheduledSuccs counter of the Add: The new store should have
  // been marked as "scheduled", so we only have one 1 unscheduled succ.
  EXPECT_EQ(AddN->getNumUnscheduledSuccs(), 1u);
}

TEST(Scheduler, DAGInsertAndAddDeps_UpdateUnschedSuccsOfPreds_MemAndUseDef) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr1, ptr noalias %ptr2) {
  %ld = load float, ptr %ptr1
  store float %ld, ptr %ptr2
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  sandboxir::DependencyGraph &DAG = Sched.getDAG();
  Sched.startScheduling();

  auto It = BB->begin();
  auto *Add = &*It++;
  auto *St = &*It++;
  auto *Ret = &*It++;
  DAG.extend(DmpVector<sandboxir::Value *>{Add, BB->getTerminator()});

  auto *LdN = DAG.getNode(Add);
  auto *StN = DAG.getNode(St);
  EXPECT_TRUE(StN->dependsOn(LdN));
  EXPECT_EQ(LdN->getNumUnscheduledSuccs(), 1u);

  Sched.startScheduling();

  // Now create a store with both Mem and Use-Def dependencies with the Ld.
  auto *Ptr = SBF->getArg(0);
  auto *NewI =
      sandboxir::StoreInst::create(Add, Ptr, /*Align=*/std::nullopt,
                                            /*InsertBefore=*/Ret, Ctx);
  auto *NewN = DAG.getNode(NewI);
  EXPECT_TRUE(NewN != nullptr);
  EXPECT_TRUE(NewN->dependsOn(LdN));
  EXPECT_TRUE(StN->dependsOn(LdN));
  // Check the UnscheduledSuccs counter of LdN. NewI is marked "scheduled".
  EXPECT_EQ(LdN->getNumUnscheduledSuccs(), 1u);
}

TEST(Scheduler, DAGInsertAndAddDeps_UpdateUnschedSuccsOfPreds_MemOnly) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v0, float %v1, ptr noalias %ptr0, ptr noalias %ptr1, float %vNew) {
  store float %v0, ptr %ptr0
  store float %v1, ptr %ptr1
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  sandboxir::DependencyGraph &DAG = Sched.getDAG();

  auto It = BB->begin();
  auto *St0 = &*It++;
  auto *St1 = &*It++;
  DAG.extend(DmpVector<sandboxir::Value *>{St0, BB->getTerminator()});

  auto *St0N = DAG.getNode(St0);
  auto *St1N = DAG.getNode(St1);
  EXPECT_FALSE(St1N->dependsOn(St0N));
  EXPECT_EQ(St0N->getNumUnscheduledSuccs(), 0u);

  Sched.startScheduling();

  // Now create a store with only a mem dependency to St0N
  auto *Val = SBF->getArg(4);
  auto *Ptr = SBF->getArg(2);
  auto *NewI =
      sandboxir::StoreInst::create(Val, Ptr, /*Align=*/std::nullopt,
                                            /*InsertBefore=*/St1, Ctx);
  auto *NewN = DAG.getNode(NewI);
  EXPECT_TRUE(NewN != nullptr);
  EXPECT_TRUE(NewN->dependsOn(St0N));
  // Check the UnscheduledSuccs counter of St0N. `NewN` is "scheduled".
  EXPECT_EQ(St0N->getNumUnscheduledSuccs(), 0u);
}

// Check that erasing a node updates UnscheduledSuccs of its predecessors
//  St0 (1)        St0 (1)
//   |              |
//  St1 (1)    =>  St1 (0)
//   |
//  St2 (0)
TEST(Scheduler, UnscheduledSuccsErase) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v0, i8 %v1, i8 %v2) {
  store i8 %v0, ptr %ptr
  store i8 %v1, ptr %ptr
  store i8 %v2, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  sandboxir::DependencyGraph &DAG = Sched.getDAG();

  auto It = BB->begin();
  auto *St0 = &*It++;
  auto *St1 = &*It++;
  auto *St2 = &*It++;

  Sched.startScheduling();

  DAG.extend(DmpVector<sandboxir::Value *>{St0, St2});
  auto *St0N = DAG.getNode(St0);
  auto *St1N = DAG.getNode(St1);
  auto *St2N = DAG.getNode(St2);
  EXPECT_EQ(St0N->getNumUnscheduledSuccs(), 2u);
  EXPECT_EQ(St1N->getNumUnscheduledSuccs(), 1u);
  EXPECT_EQ(St2N->getNumUnscheduledSuccs(), 0u);

  DAG.erase(St2N);
  EXPECT_EQ(St0N->getNumUnscheduledSuccs(), 1u);
  EXPECT_EQ(St1N->getNumUnscheduledSuccs(), 0u);

  // Check that extending after resetView() resets UnscheduledSuccs to the
  // correct value.
  DAG.resetView();
  DAG.extend({St0, St1});
  EXPECT_EQ(St0N->getNumUnscheduledSuccs(), 1u);
  EXPECT_EQ(St1N->getNumUnscheduledSuccs(), 0u);
}

// Check if UnscheduledSuccs get updated correctly with DAG.extend().
TEST(Scheduler, UnscheduledSuccsAndExtend) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v0, i8 %v1, i8 %v2, i8 %v3) {
  store i8 %v0, ptr %ptr
  store i8 %v1, ptr %ptr
  store i8 %v2, ptr %ptr
  store i8 %v3, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  sandboxir::DependencyGraph &DAG = Sched.getDAG();
  Sched.startScheduling();

  auto It = BB->begin();
  auto *St0 = &*It++;
  auto *St1 = &*It++;
  auto *St2 = &*It++;
  auto *St3 = &*It++;
  DAG.extend(DmpVector<sandboxir::Value *>{St2, St3});
  auto *St2N = DAG.getNode(St2);
  auto *St3N = DAG.getNode(St3);
  EXPECT_EQ(St2N->getNumUnscheduledSuccs(), 1u);
  EXPECT_EQ(St3N->getNumUnscheduledSuccs(), 0u);

  // We clear the view.
  DAG.resetView();
  // Now extend the view to include a node not created by the previous extend().
  DAG.extend({St1});
  auto *St1N = DAG.getNode(St1);
  EXPECT_EQ(St1N->getNumUnscheduledSuccs(), 0u);

  DAG.extend({St0});
  auto *St0N = DAG.getNode(St0);
  EXPECT_EQ(St0N->getNumUnscheduledSuccs(), 1u);
  EXPECT_EQ(St1N->getNumUnscheduledSuccs(), 0u);

  DAG.resetView();

  DAG.extend(DmpVector<sandboxir::Value *>{St2, St3});
  EXPECT_EQ(St2N->getNumUnscheduledSuccs(), 1u);
  EXPECT_EQ(St3N->getNumUnscheduledSuccs(), 0u);

  DAG.extend({St0});
  EXPECT_EQ(St0N->getNumUnscheduledSuccs(), 3u);
  EXPECT_EQ(St1N->getNumUnscheduledSuccs(), 2u);
  EXPECT_EQ(St2N->getNumUnscheduledSuccs(), 1u);
  EXPECT_EQ(St3N->getNumUnscheduledSuccs(), 0u);
}

TEST(Scheduler, UnscheduledSuccsAndExtend_TopDown) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v0, i8 %v1) {
  %foo = add i8 %v0, %v0
  store i8 %v0, ptr %ptr
  store i8 %v1, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  sandboxir::DependencyGraph &DAG = Sched.getDAG();

  auto It = BB->begin();
  auto *Foo = &*It++;
  auto *St0 = &*It++;
  auto *St1 = &*It++;

  // Shcedule Foo node just to set the scheduler in top-down mode.
  Sched.trySchedule({Foo}, /*TopDown=*/true);

  DAG.extend(DmpVector<sandboxir::Value *>{St0});
  auto *St0N = DAG.getNode(St0);
  EXPECT_EQ(St0N->getNumUnscheduledSuccs(), 0u);

  DAG.extend(DmpVector<sandboxir::Value *>{St1});
  auto *St1N = DAG.getNode(St1);
  EXPECT_EQ(St1N->getNumUnscheduledSuccs(), 1u);
}

TEST(Scheduler, UnscheduledSuccsAndExtend_BottomUp) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v0, i8 %v1) {
  store i8 %v0, ptr %ptr
  store i8 %v1, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  sandboxir::DependencyGraph &DAG = Sched.getDAG();
  Sched.startScheduling();

  auto It = BB->begin();
  auto *St0 = &*It++;
  auto *St1 = &*It++;

  DAG.extend(DmpVector<sandboxir::Value *>{St1});
  auto *St1N = DAG.getNode(St1);
  EXPECT_EQ(St1N->getNumUnscheduledSuccs(), 0u);

  DAG.extend(DmpVector<sandboxir::Value *>{St0});
  auto *St0N = DAG.getNode(St0);
  EXPECT_EQ(St0N->getNumUnscheduledSuccs(), 1u);
}

// Make sure UnscheduledSuccs are set correctly when there is a use-def edge.
TEST(Scheduler, UnscheduledSuccsUseDef) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld0 = load i8, ptr %ptr
  store i8 %ld0, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  sandboxir::DependencyGraph &DAG = Sched.getDAG();
  Sched.startScheduling();

  auto It = BB->begin();
  auto *Ld = &*It++;
  auto *St = &*It++;

  DAG.extend({St});
  auto *StN = DAG.getNode(St);
  EXPECT_EQ(StN->getNumUnscheduledSuccs(), 0u);

  DAG.extend({Ld});
  auto *LdN = DAG.getNode(Ld);
  EXPECT_EQ(LdN->getNumUnscheduledSuccs(), 1u);
  EXPECT_EQ(StN->getNumUnscheduledSuccs(), 0u);
}

// Make sure UnscheduledSuccs are set correctly when there is a use-def edge.
TEST(Scheduler, UnscheduledSuccsUseDefNoMem) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v) {
  %add = add i8 %v, %v
  store i8 %add, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  sandboxir::DependencyGraph &DAG = Sched.getDAG();
  Sched.startScheduling();

  auto It = BB->begin();
  auto *Add = &*It++;
  auto *St = &*It++;

  DAG.extend({St});
  auto *StN = DAG.getNode(St);
  EXPECT_EQ(StN->getNumUnscheduledSuccs(), 0u);

  DAG.extend({Add});
  auto *AddN = DAG.getNode(Add);
  EXPECT_EQ(AddN->getNumUnscheduledSuccs(), 1u);
  EXPECT_EQ(StN->getNumUnscheduledSuccs(), 0u);
}

// If scheduling is off, then UnscheduledSuccs are not being updated.
TEST(Scheduler, UnscheduledSuccs_WithoutStartScheduling) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v) {
  %add = add i8 %v, %v
  store i8 %add, ptr %ptr
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
  sandboxir::Scheduler &Sched = *Ctx.getScheduler(BB);
  sandboxir::DependencyGraph &DAG = Sched.getDAG();
  // Sched.startScheduling();

  auto It = BB->begin();
  auto *Add = &*It++;
  auto *St = &*It++;

  DAG.extend({St, Add});
  auto *StN = DAG.getNode(St);
  auto *AddN = DAG.getNode(Add);
  EXPECT_EQ(StN->getNumUnscheduledSuccs(), 0u);
  EXPECT_EQ(AddN->getNumUnscheduledSuccs(), 0u);
  EXPECT_EQ(StN->getNumUnscheduledSuccs(), 0u);
}
