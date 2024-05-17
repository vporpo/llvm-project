//===- InstrRangeTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/InstrRange.h"
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
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("InstrRangeTest", errs());
  return Mod;
}

TEST(InstrRange, Basic) {
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
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AAResults AA(TLI);
  SBContext Ctxt(C, AA);
  SBBasicBlock *BB = Ctxt.createSBBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *I0 = &*It++;
  auto *I1 = &*It++;
  auto *I2 = &*It++;
  auto *I3 = &*It++;
  auto *I4 = &*It++;
  {
    InstrRange InstrRange(I0, I1);
    EXPECT_TRUE(InstrRange.from()->comesBefore(InstrRange.to()));
    EXPECT_EQ(InstrRange.from(), I0);
    EXPECT_EQ(InstrRange.to(), I1);
  }
  {
    InstrRange InstrRange(I0, I0);
    EXPECT_EQ(InstrRange.from(), InstrRange.to());
    EXPECT_EQ(InstrRange.from(), I0);
    EXPECT_EQ(InstrRange.to(), I0);
  }
  {
    InstrRange InstrRange(I1, I0);
    EXPECT_TRUE(InstrRange.from()->comesBefore(InstrRange.to()));
    EXPECT_EQ(InstrRange.from(), I0);
    EXPECT_EQ(InstrRange.to(), I1);
  }

  // Check ArrayRef constructor.
  {
    InstrRange InstrRange({I0, I1, I4});
    EXPECT_TRUE(InstrRange.from()->comesBefore(InstrRange.to()));
    EXPECT_EQ(InstrRange.from(), I0);
    EXPECT_EQ(InstrRange.to(), I4);
  }
  {
    InstrRange InstrRange({I4, I0, I2, I2});
    EXPECT_TRUE(InstrRange.from()->comesBefore(InstrRange.to()));
    EXPECT_EQ(InstrRange.from(), I0);
    EXPECT_EQ(InstrRange.to(), I4);
  }

  // Check contains(Instruction).
  {
    InstrRange InstrRange(I1, I1);
    EXPECT_TRUE(InstrRange.contains(I1));
    EXPECT_FALSE(InstrRange.contains(I0));
    EXPECT_FALSE(InstrRange.contains(I2));
    EXPECT_FALSE(InstrRange.contains(I3));
    EXPECT_FALSE(InstrRange.contains(I4));
  }
  {
    InstrRange InstrRange(I1, I4);
    EXPECT_TRUE(InstrRange.contains(I1));
    EXPECT_TRUE(InstrRange.contains(I2));
    EXPECT_TRUE(InstrRange.contains(I3));
    EXPECT_TRUE(InstrRange.contains(I4));
    EXPECT_FALSE(InstrRange.contains(I0));
  }

  // Check contains(InstrRange).
  {
    InstrRange InstrRange1(I0, I2);
    InstrRange InstrRange2(I0, I1);
    EXPECT_TRUE(InstrRange1.contains(InstrRange2));
    InstrRange InstrRange3(I0, I2);
    EXPECT_TRUE(InstrRange1.contains(InstrRange3));
    InstrRange InstrRange4(I1, I2);
    EXPECT_TRUE(InstrRange1.contains(InstrRange4));
    InstrRange InstrRange5(I2, I3);
    EXPECT_FALSE(InstrRange1.contains(InstrRange5));
    InstrRange InstrRange6(I3, I4);
    EXPECT_FALSE(InstrRange1.contains(InstrRange6));
  }

  // Check operator== and operator!=.
  {
    InstrRange InstrRangeEmpty;
    InstrRange InstrRange1(I0, I0);
    EXPECT_FALSE(InstrRange1 == InstrRangeEmpty);
    EXPECT_FALSE(InstrRangeEmpty == InstrRange1);
    EXPECT_TRUE(InstrRange1 != InstrRangeEmpty);
    EXPECT_TRUE(InstrRangeEmpty != InstrRange1);
    InstrRange InstrRange2(I0, I1);
    EXPECT_TRUE(InstrRange1 != InstrRange2);
    EXPECT_TRUE(InstrRange2 != InstrRange1);
    InstrRange InstrRangeEmpty2;
    EXPECT_TRUE(InstrRangeEmpty == InstrRangeEmpty2);
    EXPECT_TRUE(InstrRangeEmpty2 == InstrRangeEmpty);
    InstrRange InstrRange3(I0, I1);
    EXPECT_TRUE(InstrRange2 == InstrRange3);
    EXPECT_TRUE(InstrRange3 == InstrRange2);
  }

  // Check getUnion().
  {
    InstrRange InstrRangeEmpty;
    InstrRange InstrRange1(I0, I0);
    auto UnionE1 = InstrRangeEmpty.getUnionSingleSpan(InstrRange1);
    EXPECT_TRUE(UnionE1 == InstrRange1);
    auto Union1E = InstrRange1.getUnionSingleSpan(InstrRange1);
    EXPECT_TRUE(Union1E == InstrRange1);
    InstrRange InstrRange2(I1, I1);
    auto Union12 = InstrRange1.getUnionSingleSpan(InstrRange2);
    EXPECT_TRUE(Union12 == InstrRange2.getUnionSingleSpan(InstrRange1));
    EXPECT_EQ(Union12.from(), I0);
    EXPECT_EQ(Union12.to(), I1);

    InstrRange InstrRange3(I4, I4);
    auto Union13 = InstrRange1.getUnionSingleSpan(InstrRange3);
    EXPECT_EQ(Union13.from(), I0);
    EXPECT_EQ(Union13.to(), I4);
  }

  // Check getIntersection().
  {
    InstrRange InstrRangeEmpty;
    InstrRange InstrRange1(I0, I0);
    EXPECT_TRUE(InstrRangeEmpty.getIntersection(InstrRange1).empty());
    EXPECT_TRUE(InstrRange1.getIntersection(InstrRangeEmpty).empty());

    InstrRange InstrRange2(I1, I1);
    EXPECT_TRUE(InstrRange1.getIntersection(InstrRange2).empty());
    EXPECT_TRUE(InstrRange2.getIntersection(InstrRange1).empty());

    InstrRange InstrRange3(I0, I3);
    InstrRange InstrRange4(I1, I2);
    auto Intersection34 = InstrRange3.getIntersection(InstrRange4);
    EXPECT_TRUE(InstrRange4.getIntersection(InstrRange3) == Intersection34);
    EXPECT_EQ(Intersection34.from(), I1);
    EXPECT_EQ(Intersection34.to(), I2);

    InstrRange InstrRange5(I2, I4);
    auto Intersection35 = InstrRange3.getIntersection(InstrRange5);
    EXPECT_TRUE(InstrRange5.getIntersection(InstrRange3) == Intersection35);
    EXPECT_EQ(Intersection35.from(), I2);
    EXPECT_EQ(Intersection35.to(), I3);
  }

  // Check the difference operator-().
  {
    // Same FromI
    InstrRange InstrRange1(I0, I3);
    InstrRange InstrRange2(I0, I2);
    auto Diff12Vec = InstrRange1 - InstrRange2;
    EXPECT_EQ(Diff12Vec.size(), 1u);
    auto Diff12 = Diff12Vec.back();
    EXPECT_EQ(Diff12.from(), I3);
    EXPECT_EQ(Diff12.to(), I3);

    // Same ToI
    InstrRange InstrRange3(I2, I3);
    auto Diff13Vec = InstrRange1 - InstrRange3;
    EXPECT_EQ(Diff13Vec.size(), 1u);
    auto Diff13 = Diff13Vec.back();
    EXPECT_EQ(Diff13.from(), I0);
    EXPECT_EQ(Diff13.to(), I1);

    // Disjoint
    InstrRange InstrRange4(I4, I4);
    auto Diff14Vec = InstrRange1 - InstrRange4;
    EXPECT_EQ(Diff14Vec.size(), 1u);
    EXPECT_TRUE(Diff14Vec.back() == InstrRange1);

    // Overlap
    InstrRange InstrRange5(I2, I4);
    auto Diff15Vec = InstrRange1 - InstrRange5;
    EXPECT_EQ(Diff15Vec.size(), 1u);
    EXPECT_TRUE(Diff15Vec.back() == InstrRange(I0, I1));

    // 2 results
    InstrRange InstrRange6(I2, I2);
    auto Diff16Vec = InstrRange1 - InstrRange6;
    EXPECT_EQ(Diff16Vec.size(), 2u);
    EXPECT_TRUE(Diff16Vec[0] == InstrRange(I0, I1));
    EXPECT_TRUE(Diff16Vec[1] == InstrRange(I3, I3));

    // A - A
    auto Diff11Vec = InstrRange1 - InstrRange1;
    EXPECT_EQ(Diff11Vec.size(), 1u);
    EXPECT_TRUE(Diff11Vec.back().empty());
    InstrRange InstrRangeEmpty;

    // A - Empty
    auto Diff1E = InstrRange1 - InstrRangeEmpty;
    EXPECT_EQ(Diff1E.size(), 1u);
    EXPECT_TRUE(Diff1E.back() == InstrRange1);
  }

  // InstrRange iterator
  {
    InstrRange InstrRange(I0, I1);
    SmallVector<SBInstruction *> Instrs;
    for (SBInstruction &IRef : InstrRange)
      Instrs.push_back(&IRef);
    EXPECT_EQ(Instrs.size(), 2u);
    EXPECT_EQ(Instrs[0], I0);
    EXPECT_EQ(Instrs[1], I1);

    Instrs.clear();
    const auto &ConstInstrRange = InstrRange;
    for (SBInstruction &IRef : ConstInstrRange)
      Instrs.push_back(&IRef);
    EXPECT_EQ(Instrs.size(), 2u);
    EXPECT_EQ(Instrs[0], I0);
    EXPECT_EQ(Instrs[1], I1);
  }
  // InstrRange reverse iterator
  {
    InstrRange InstrRange(I0, I1);
    SmallVector<SBInstruction *> Instrs;
    for (SBInstruction &IRef : reverse(InstrRange))
      Instrs.push_back(&IRef);
    EXPECT_EQ(Instrs.size(), 2u);
    EXPECT_EQ(Instrs[0], I1);
    EXPECT_EQ(Instrs[1], I0);
  }

  // Check end()-- when InstrRange.ToI is the BB terminator.
  {
    InstrRange InstrRange(I4);
    auto It = InstrRange.end();
    --It;
    EXPECT_EQ(&*It, I4);
  }

  // Check extend(I)
  {
    InstrRange InstrRange(I2);
    InstrRange.extend(I4);
    EXPECT_EQ(InstrRange.from(), I2);
    EXPECT_EQ(InstrRange.to(), I4);

    InstrRange.extend(I1);
    EXPECT_EQ(InstrRange.from(), I1);
    EXPECT_EQ(InstrRange.to(), I4);

    InstrRange.extend(I3);
    EXPECT_EQ(InstrRange.from(), I1);
    EXPECT_EQ(InstrRange.to(), I4);
  }

  // Erase an Instruction that is not in the region.
  {
    InstrRange InstrRange(I2);
#ifndef NDEBUG
    EXPECT_DEATH(InstrRange.erase(I1), ".*not in region.*");
#endif
    InstrRange.erase(I1, /*CheckContained=*/false);
    EXPECT_EQ(InstrRange.from(), I2);
    EXPECT_EQ(InstrRange.to(), I2);
  }
  {
    InstrRange InstrRange(I2, I3);
#ifndef NDEBUG
    EXPECT_DEATH(InstrRange.erase(I1), ".*not in region.*");
#endif
    InstrRange.erase(I1, /*CheckContained=*/false);
    EXPECT_EQ(InstrRange.from(), I2);
    EXPECT_EQ(InstrRange.to(), I3);
  }
}

TEST(InstrRange, ContainsIt) {
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
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AAResults AA(TLI);
  SBContext Ctxt(C, AA);
  SBBasicBlock *BB = Ctxt.createSBBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *I0 = &*It++;
  (void)I0;
  auto *I1 = &*It++;
  auto *I2 = &*It++;
  auto *I3 = &*It++;
  auto *I4 = &*It++;

  {
    InstrRange R(I3, I3);
    EXPECT_TRUE(R.contains(I3->getIterator()));
    EXPECT_TRUE(R.contains(I4->getIterator()));
    EXPECT_FALSE(R.contains(I2->getIterator()));
    EXPECT_FALSE(R.contains(BB->end()));
  }
  {
    InstrRange R(I2, I4);
    EXPECT_TRUE(R.contains(I4->getIterator()));
    EXPECT_TRUE(R.contains(BB->end()));
    EXPECT_FALSE(R.contains(I1->getIterator()));
  }
}

TEST(InstrRange, ExtendEmpty) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  store i8 0, ptr %ptr
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AAResults AA(TLI);
  SBContext Ctxt(C, AA);
  SBBasicBlock *BB = Ctxt.createSBBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *I0 = &*It++;

  InstrRange R;
  R.extend(I0);
  EXPECT_TRUE(R.contains(I0));
}

#ifndef NDEBUG
TEST(InstrRange, InstrRangeNotify) {
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
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AAResults AA(TLI);
  SBContext Ctxt(C, AA);
  Function &F = *M->getFunction("foo");

  SBBasicBlock *BB = Ctxt.createSBBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *I0 = &*It++;
  auto *I1 = &*It++;
  auto *I2 = &*It++;
  auto *I3 = &*It++;
  auto *I4 = &*It++;
  {
    Ctxt.disableCallbacks();
    InstrRange InstrRange(I0, I4);
    auto WhereIt = I0->getNextNode()->getIterator();
    // Move FromI to itself.
    InstrRange.notifyMoveInstr(I0, WhereIt, BB);
    I1->moveBefore(*BB, WhereIt);
    EXPECT_EQ(InstrRange.from(), &BB->front());
    EXPECT_EQ(InstrRange.to(), &BB->back());

    // Move ToI to itself.
    WhereIt = BB->end();
    InstrRange.notifyMoveInstr(I4, WhereIt, BB);
    I4->moveBefore(*BB, WhereIt);
    EXPECT_EQ(InstrRange.from(), &BB->front());
    EXPECT_EQ(InstrRange.to(), &BB->back());

    // Move instruction to FromI.
    InstrRange.notifyMoveInstr(I1, I0->getIterator(), BB);
    I1->moveBefore(I0);
    EXPECT_EQ(InstrRange.from(), &BB->front());
    EXPECT_EQ(InstrRange.to(), &BB->back());

    // Move FromI.
    InstrRange.notifyMoveInstr(I1, I2->getIterator(), BB);
    I1->moveBefore(I2);
    EXPECT_EQ(InstrRange.from(), &BB->front());
    EXPECT_EQ(InstrRange.to(), &BB->back());

    // Move instruction to ToI.
    InstrRange.notifyMoveInstr(I2, BB->end(), BB);
    I2->moveBefore(*BB, BB->end());
    EXPECT_EQ(InstrRange.from(), &BB->front());
    EXPECT_EQ(InstrRange.to(), &BB->back());

    // Move ToI to FromI.
    InstrRange.notifyMoveInstr(I2, I1->getIterator(), BB);
    I2->moveBefore(I1);
    EXPECT_EQ(InstrRange.from(), &BB->front());
    EXPECT_EQ(InstrRange.to(), &BB->back());

    // Move internal instructions.
    InstrRange.notifyMoveInstr(I3, I4->getIterator(), BB);
    I3->moveBefore(I4);
    EXPECT_EQ(InstrRange.from(), &BB->front());
    EXPECT_EQ(InstrRange.to(), &BB->back());
  }
}
#endif

#ifndef NDEBUG
TEST(InstrRange, InstrRangeNotify2) {
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
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AAResults AA(TLI);
  SBContext Ctxt(C, AA);

  Function &F = *M->getFunction("foo");
  SBBasicBlock *BB = Ctxt.createSBBasicBlock(&*F.begin());
  auto It = BB->begin();
  auto *I0 = &*It++;
  (void)I0;
  auto *I1 = &*It++;
  auto *I2 = &*It++;
  auto *I3 = &*It++;
  auto *I4 = &*It++;
  (void)I4;

  Ctxt.disableCallbacks();
  {
    // Single-instruction region.
    InstrRange InstrRange(I1, I1);
    // Trying to move I1 out of the region should crash!
    EXPECT_DEATH(InstrRange.notifyMoveInstr(I1, I3->getIterator(), BB), ".*");
    // Move to itself should be a nop.
    InstrRange.notifyMoveInstr(I1, std::next(I1->getIterator()), BB);
    I1->moveBefore(*BB, std::next(I1->getIterator()));
    EXPECT_EQ(InstrRange.from(), I1);
    EXPECT_EQ(InstrRange.to(), I1);
  }
  {
    InstrRange InstrRange(I2, I4);
    // To help debug the scheduler, trying to move I1, an external instruction,
    // into the region, should crash. If the scheduler wants to move new
    // instructions into the scheduled region it should first extend the DAG's
    // region to include them.
    EXPECT_DEATH(InstrRange.notifyMoveInstr(I1, I4->getIterator(), BB), ".*");
  }
  {
    // Moving I2 before I2 should not change the region.
    InstrRange InstrRange(I1, I2);
    InstrRange.notifyMoveInstr(I2, I2->getIterator(), BB);
    I2->moveBefore(*BB, I2->getIterator());
    EXPECT_EQ(InstrRange.from(), I1);
    EXPECT_EQ(InstrRange.to(), I2);
  }
  {
    // Moving I2 before I1 should change the region to {I2, I1}.
    InstrRange InstrRange(I1, I2);
    InstrRange.notifyMoveInstr(I2, I1->getIterator(), BB);
    I2->moveBefore(*BB, I1->getIterator());
    EXPECT_EQ(InstrRange.from(), I2);
    EXPECT_EQ(InstrRange.to(), I1);
    // Revert IR
    I2->moveAfter(I1);
  }
  {
    // Moving I1 after I2 should change the region to {I2, I1}.
    InstrRange InstrRange(I1, I2);
    InstrRange.notifyMoveInstr(I1, std::next(I2->getIterator()), BB);
    I1->moveBefore(*BB, std::next(I2->getIterator()));
    EXPECT_EQ(InstrRange.from(), I2);
    EXPECT_EQ(InstrRange.to(), I1);
  }
}
#endif
