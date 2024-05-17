//===- BundleTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Bundle.h"

#include "llvm/ADT/DenseMap.h"
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
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"
#include "gtest/gtest.h"

using namespace llvm;
static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("BundleTest", errs());
  return Mod;
}

TEST(Bundle, Bundle) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  store i8 0, ptr %ptr
  store i8 1, ptr %ptr
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");

  BasicBlock *BB = &*F.begin();
  auto It = BB->begin();
  Instruction *I0 = &*It++;
  Instruction *I1 = &*It++;

  {
    ValueBundle Bndl({I0, I1});
    EXPECT_EQ(Bndl[0], I0);
    EXPECT_EQ(Bndl[1], I1);
    EXPECT_EQ(Bndl.size(), 2u);
    EXPECT_EQ(*Bndl.begin(), I0);
    EXPECT_EQ(Bndl.front(), I0);
    EXPECT_EQ(*std::prev(Bndl.end()), I1);
    EXPECT_EQ(Bndl.back(), I1);
    EXPECT_EQ(*Bndl.ibegin(), I0);
    auto HashBefore = Bndl.hash();

    Bndl.erase(std::next(Bndl.begin()));
    auto HashAfter = Bndl.hash();
    EXPECT_NE(HashBefore, HashAfter);
    EXPECT_EQ(Bndl.size(), 1u);
    Bndl.push_back(I1);
  }
  {
    // Check single-element constructor.
    ValueBundle Bndl(I0);
    EXPECT_EQ(Bndl.size(), 1u);
    EXPECT_EQ(Bndl[0], I0);
  }
  {
    // Check reserve constructor.
    ValueBundle Bndl(2);
    EXPECT_TRUE(Bndl.empty());
  }
  {
    // Check the DenseMap traits.
    ValueBundle Bndl0(I0);
    ValueBundle Bndl1(I1);
    DenseMap<ValueBundle, int> Map;
    Map[Bndl0] = 0;
    Map[Bndl1] = 1;
    EXPECT_EQ(Map.size(), 2u);
    EXPECT_NE(Map.find(Bndl0), Map.end());
    EXPECT_NE(Map.find(Bndl1), Map.end());
    EXPECT_EQ(Map.find(Bndl0)->second, 0);
    EXPECT_EQ(Map.find(Bndl1)->second, 1);
  }
}

TEST(Bundle, SBInstrBundle) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
define void @foo(ptr %ptr1, ptr %ptr2) {
  store i8 0, ptr %ptr1
  store i8 1, ptr %ptr2
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
  BasicBlock *BB = &*F.begin();
  SBFunction &SBF = *Ctxt.createSBFunction(&F);
  SBBasicBlock &SBBB = *Ctxt.getSBBasicBlock(BB);
  auto *Arg0 = SBF.getArg(0);
  auto *Arg1 = SBF.getArg(1);
  auto It = SBBB.begin();
  auto *S0 = &*It++;
  auto *S1 = &*It++;
  auto *Ret = &*It++;
  {
    // Check default constructor.
    SBInstrBundle Tmp;
    EXPECT_TRUE(Tmp.empty());
  }
  {
    // Check constructor with initializer_list
    SBInstrBundle SBBndl({S0, S1, Ret});
    EXPECT_EQ(SBBndl.size(), 3u);
  }
  {
    // Check constructor with ArrayRef.
    SmallVector<SBInstruction *> Vec({S0, S1, Ret});
    SBInstrBundle SBBndl(Vec);
    EXPECT_EQ(SBBndl.size(), 3u);
    EXPECT_EQ(SBBndl[0], S0);
    EXPECT_EQ(SBBndl[1], S1);
    EXPECT_EQ(SBBndl[2], Ret);
  }
  {
    // Check copy constructor
    SBInstrBundle SBBndl({S0, S1, Ret});
    SBInstrBundle SBBndlCpy(SBBndl);
    EXPECT_EQ(SBBndl, SBBndlCpy);
  }
  {
    // Check move constructor
    SBInstrBundle SBBndl({S0, S1, Ret});
    SBInstrBundle SBBndlMv(std::move(SBBndl));
    EXPECT_EQ(SBBndlMv.size(), 3u);
    EXPECT_EQ(SBBndlMv[0], S0);
    EXPECT_EQ(SBBndlMv[1], S1);
    EXPECT_EQ(SBBndlMv[2], Ret);
  }
  {
    // Check reserve constructor
    SBInstrBundle SBBndl(10);
    EXPECT_EQ(SBBndl.size(), 0u);
  }
  {
    // Check copy assignment
    SBInstrBundle SBBndl({S0, S1, Ret});
    SBInstrBundle SBBndlCpy = SBBndl;
    EXPECT_EQ(SBBndl, SBBndlCpy);
  }
  {
    // Check range constructor
    SBInstrBundle SBBndl({S0, S1, Ret});
    SBInstrBundle SBBndlCpy(SBBndl.begin(), SBBndl.end());
    EXPECT_EQ(SBBndl, SBBndlCpy);
  }
  {
    // Check getOperandBundle()
    SBInstrBundle SBBndl({S0, S1});
    SBValBundle OpBndl = SBBndl.getOperandBundle(1);
    EXPECT_EQ(OpBndl, SBValBundle({Arg0, Arg1}));
  }
}
