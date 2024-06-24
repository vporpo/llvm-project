//===- SeedCollectorTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/SeedCollector.h"
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
#include "llvm/SandboxIR/SandboxIR.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("SeedCollectorTest", errs());
  return Mod;
}

static BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    if (BB.getName() == Name)
      return &BB;
  llvm_unreachable("Expected to find basic block!");
}

TEST(SeedCollectorTest, SeedContainer) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(float %v0, float %v1) {
bb:
  %add0 = fadd float %v0, %v1
  %add1 = fadd float %v0, %v1
  %add2 = fadd float %v0, %v1
  %add3 = fadd float %v0, %v1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  BasicBlock *BB = getBasicBlockByName(F, "bb");
  sandboxir::Context Ctx(C);
  sandboxir::BasicBlock &SBBB = *Ctx.createBasicBlock(BB);
  auto It = SBBB.begin();
  auto *I0 = cast<sandboxir::Instruction>(&*It++);
  auto *I1 = cast<sandboxir::Instruction>(&*It++);
  auto *I2 = cast<sandboxir::Instruction>(&*It++);
  auto *I3 = cast<sandboxir::Instruction>(&*It++);
  sandboxir::SeedContainer SC;
  // Check begin() end() when empty.
  EXPECT_EQ(SC.begin(), SC.end());
  // Crash if we are attempting to insert a value twice.
#ifndef NDEBUG
  EXPECT_DEATH(SC.insert({I0, I0}), ".*more than once.*");
  EXPECT_DEATH(SC.insert({I0, I1, I0}), ".*more than once.*");
#endif

  SC.insert({I0, I1});
  SC.insert({I2, I3});

  unsigned Cnt = 0;
  SmallVector<sandboxir::SeedBundle *> Bndls;
  for (auto &SeedBndl : SC) {
    EXPECT_EQ(SeedBndl.size(), 2u);
    ++Cnt;
    Bndls.push_back(&SeedBndl);
  }
  EXPECT_EQ(Cnt, 2u);

  // Mark them "Used" to check if operator++ skips them in the next loop.
  for (auto *SeedBndl : Bndls)
    for (auto Lane : seq<unsigned>(SeedBndl->size()))
      SeedBndl->setUsed(Lane);
  // Check if iterator::operator++ skips used lanes.
  Cnt = 0;
  for (auto &SeedBndl : SC) {
    (void)SeedBndl;
    ++Cnt;
  }
  EXPECT_EQ(Cnt, 0u);
}

TEST(SeedCollectorTest, MemSeedContainer) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptrA, float %val, ptr %ptrB) {
bb:
  %gepA0 = getelementptr float, ptr %ptrA, i32 0
  %gepA1 = getelementptr float, ptr %ptrA, i32 1
  %gepB0 = getelementptr float, ptr %ptrB, i32 0
  %gepB1 = getelementptr float, ptr %ptrB, i32 1
  store float %val, ptr %gepA0
  store float %val, ptr %gepA1
  store float %val, ptr %gepB0
  store float %val, ptr %gepB1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");

  DominatorTree DT(F);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  LoopInfo LI(DT);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);

  BasicBlock *BB = getBasicBlockByName(F, "bb");
  sandboxir::Context Ctx(C);
  sandboxir::BasicBlock &SBBB = *Ctx.createBasicBlock(BB);
  auto It = std::next(SBBB.begin(), 4);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *S2 = cast<sandboxir::StoreInst>(&*It++);
  auto *S3 = cast<sandboxir::StoreInst>(&*It++);
  auto *S4 = cast<sandboxir::StoreInst>(&*It++);
  sandboxir::MemSeedContainer MSC(DL, SE);
  // Check begin() end() when empty.
  EXPECT_EQ(MSC.begin(), MSC.end());

  MSC.insert(S1);
  MSC.insert(S2);
  MSC.insert(S3);
  MSC.insert(S4);
  unsigned Cnt = 0;
  SmallVector<sandboxir::SeedBundle *> Bndls;
  for (auto &SeedBndl : MSC) {
    EXPECT_EQ(SeedBndl.size(), 2u);
    ++Cnt;
    Bndls.push_back(&SeedBndl);
  }
  EXPECT_EQ(Cnt, 2u);

  // Mark them "Used" to check if operator++ skips them in the next loop.
  for (auto *SeedBndl : Bndls)
    for (auto Lane : seq<unsigned>(SeedBndl->size()))
      SeedBndl->setUsed(Lane);
  // Check if iterator::operator++ skips used lanes.
  Cnt = 0;
  for (auto &SeedBndl : MSC) {
    (void)SeedBndl;
    ++Cnt;
  }
  EXPECT_EQ(Cnt, 0u);
}

TEST(SeedCollectorTest, ConsecutiveStores) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr, float %val) {
bb:
  %ptr0 = getelementptr float, ptr %ptr, i32 0
  %ptr1 = getelementptr float, ptr %ptr, i32 1
  %ptr2 = getelementptr float, ptr %ptr, i32 2
  %ptr3 = getelementptr float, ptr %ptr, i32 3
  store float %val, ptr %ptr0
  store float %val, ptr %ptr2
  store float %val, ptr %ptr1
  store float %val, ptr %ptr3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");

  DominatorTree DT(F);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  LoopInfo LI(DT);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);

  BasicBlock *BB = getBasicBlockByName(F, "bb");
  sandboxir::Context Ctx(C);
  sandboxir::BasicBlock &SBBB = *Ctx.createBasicBlock(BB);
  sandboxir::SeedCollector SC(&SBBB, DL, SE);

  auto It = std::next(BB->begin(), 4);
  Instruction *St0 = &*It++;
  Instruction *St2 = &*It++;
  Instruction *St1 = &*It++;
  Instruction *St3 = &*It++;

  auto *SBSt0 = Ctx.getValue(St0);
  auto *SBSt1 = Ctx.getValue(St1);
  auto *SBSt2 = Ctx.getValue(St2);
  auto *SBSt3 = Ctx.getValue(St3);

  auto StoreSeedsRange = SC.getStoreSeeds();
  sandboxir::SeedBundle &SB = *StoreSeedsRange.begin();
  EXPECT_TRUE(std::next(StoreSeedsRange.begin()) == StoreSeedsRange.end());
  EXPECT_EQ(SB, DmpVector<sandboxir::Value *>({SBSt0, SBSt1, SBSt2, SBSt3}));
}

TEST(SeedCollectorTest, StoresWithGaps) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr, float %val) {
bb:
  %ptr0 = getelementptr float, ptr %ptr, i32 0
  %ptr1 = getelementptr float, ptr %ptr, i32 3
  %ptr2 = getelementptr float, ptr %ptr, i32 5
  %ptr3 = getelementptr float, ptr %ptr, i32 7
  store float %val, ptr %ptr0
  store float %val, ptr %ptr2
  store float %val, ptr %ptr1
  store float %val, ptr %ptr3
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");

  DominatorTree DT(F);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  LoopInfo LI(DT);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);

  BasicBlock *BB = getBasicBlockByName(F, "bb");
  sandboxir::Context Ctx(C);
  sandboxir::BasicBlock &SBBB = *Ctx.createBasicBlock(BB);
  sandboxir::SeedCollector SC(&SBBB, DL, SE);

  auto It = std::next(BB->begin(), 4);
  Instruction *St0 = &*It++;
  Instruction *St2 = &*It++;
  Instruction *St1 = &*It++;
  Instruction *St3 = &*It++;

  auto *SBSt0 = Ctx.getValue(St0);
  auto *SBSt1 = Ctx.getValue(St1);
  auto *SBSt2 = Ctx.getValue(St2);
  auto *SBSt3 = Ctx.getValue(St3);

  auto StoreSeedsRange = SC.getStoreSeeds();
  sandboxir::SeedBundle &SB = *StoreSeedsRange.begin();
  EXPECT_TRUE(std::next(StoreSeedsRange.begin()) == StoreSeedsRange.end());
  EXPECT_EQ(SB, DmpVector<sandboxir::Value *>({SBSt0, SBSt1, SBSt2, SBSt3}));
}

TEST(SeedCollectorTest, VectorStores) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr, <2 x float> %val) {
bb:
  %ptr0 = getelementptr float, ptr %ptr, i32 0
  %ptr2 = getelementptr float, ptr %ptr, i32 2
  store <2 x float> %val, ptr %ptr2
  store <2 x float> %val, ptr %ptr0
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");

  DominatorTree DT(F);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  LoopInfo LI(DT);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);

  BasicBlock *BB = getBasicBlockByName(F, "bb");
  sandboxir::Context Ctx(C);
  sandboxir::BasicBlock &SBBB = *Ctx.createBasicBlock(BB);
  sandboxir::SeedCollector SC(&SBBB, DL, SE);

  auto It = std::next(BB->begin(), 2);
  Instruction *St2 = &*It++;
  Instruction *St0 = &*It++;

  auto *SBSt2 = Ctx.getValue(St2);
  auto *SBSt0 = Ctx.getValue(St0);

  auto StoreSeedsRange = SC.getStoreSeeds();
  sandboxir::SeedBundle &SB = *StoreSeedsRange.begin();
  EXPECT_TRUE(std::next(StoreSeedsRange.begin()) == StoreSeedsRange.end());
  EXPECT_EQ(SB, DmpVector<sandboxir::Value *>({SBSt0, SBSt2}));
}

TEST(SeedCollectorTest, MixedScalarVectors) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr noalias %ptr, float %v, <2 x float> %val) {
bb:
  %ptr0 = getelementptr float, ptr %ptr, i32 0
  %ptr1 = getelementptr float, ptr %ptr, i32 1
  %ptr3 = getelementptr float, ptr %ptr, i32 3
  store float %v, ptr %ptr0
  store float %v, ptr %ptr3
  store <2 x float> %val, ptr %ptr1
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");

  DominatorTree DT(F);
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  DataLayout DL(M.get());
  LoopInfo LI(DT);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);

  BasicBlock *BB = getBasicBlockByName(F, "bb");
  sandboxir::Context Ctx(C);
  sandboxir::BasicBlock &SBBB = *Ctx.createBasicBlock(BB);
  sandboxir::SeedCollector SC(&SBBB, DL, SE);

  auto It = std::next(BB->begin(), 3);
  Instruction *St0 = &*It++;
  Instruction *St3 = &*It++;
  Instruction *St1 = &*It++;
  auto *SBSt0 = Ctx.getValue(St0);
  auto *SBSt3 = Ctx.getValue(St3);
  auto *SBSt1 = Ctx.getValue(St1);

  sandboxir::SeedBundle &SB = *SC.getStoreSeeds().begin();
  EXPECT_TRUE(std::next(SC.getStoreSeeds().begin()) ==
              SC.getStoreSeeds().end());
  EXPECT_EQ(SB, DmpVector<sandboxir::Value *>({SBSt0, SBSt1, SBSt3}));
}
