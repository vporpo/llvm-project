//===- BundleTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/DmpVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
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
    Err.print("DmpVectorTest", errs());
  return Mod;
}

TEST(DmpVector, Basic) {
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
    DmpVector<Value *> Vec({I0, I1});
    EXPECT_EQ(Vec[0], I0);
    EXPECT_EQ(Vec[1], I1);
    EXPECT_EQ(Vec.size(), 2u);
    EXPECT_EQ(*Vec.begin(), I0);
    EXPECT_EQ(Vec.front(), I0);
    EXPECT_EQ(*std::prev(Vec.end()), I1);
    EXPECT_EQ(Vec.back(), I1);
    EXPECT_EQ(*Vec.ibegin(), I0);
    auto HashBefore = Vec.hash();

    Vec.erase(std::next(Vec.begin()));
    auto HashAfter = Vec.hash();
    EXPECT_NE(HashBefore, HashAfter);
    EXPECT_EQ(Vec.size(), 1u);
    Vec.push_back(I1);

    EXPECT_FALSE(Vec.empty());
    Vec.clear();
    EXPECT_TRUE(Vec.empty());
  }
  {
    // Check single-element constructor.
    DmpVector<Value *> Vec(I0);
    EXPECT_EQ(Vec.size(), 1u);
    EXPECT_EQ(Vec[0], I0);
  }
  {
    // Check reserve constructor.
    DmpVector<Value *> Vec(2);
    EXPECT_TRUE(Vec.empty());
  }
  {
    // Check the DenseMap traits.
    DmpVector<Value *> Vec0(I0);
    DmpVector<Value *> Vec1(I1);
    DenseMap<DmpVector<Value *>, int> Map;
    Map[Vec0] = 0;
    Map[Vec1] = 1;
    EXPECT_EQ(Map.size(), 2u);
    EXPECT_NE(Map.find(Vec0), Map.end());
    EXPECT_NE(Map.find(Vec1), Map.end());
    EXPECT_EQ(Map.find(Vec0)->second, 0);
    EXPECT_EQ(Map.find(Vec1)->second, 1);
  }
}

TEST(DmpVector, DmpVector_SBValue_instrRange) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
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
  (void)SBF;
  sandboxir::BasicBlock &SBBB = *Ctx.getBasicBlock(BB);
  auto It = SBBB.begin();
  auto *S0 = &*It++;
  auto *S1 = &*It++;
  auto *Ret = &*It++;
  DmpVector<sandboxir::Value *> Vec({S0, S1, Ret});
  sandboxir::Instruction *TmpI = S0;
  for (sandboxir::Instruction *I : Vec.instrRange()) {
    EXPECT_EQ(I, TmpI);
    TmpI = TmpI->getNextNode();
  }
}

TEST(DmpVector, DmpVector_SBInstruction) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
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
  (void)SBF;
  sandboxir::BasicBlock &SBBB = *Ctx.getBasicBlock(BB);
  auto It = SBBB.begin();
  auto *S0 = &*It++;
  auto *S1 = &*It++;
  auto *Ret = &*It++;
  {
    // Check default constructor.
    DmpVector<sandboxir::Instruction *> Tmp;
    EXPECT_TRUE(Tmp.empty());
  }
  {
    // Check constructor with initializer_list
    DmpVector<sandboxir::Instruction *> SBVec({S0, S1, Ret});
    EXPECT_EQ(SBVec.size(), 3u);
  }
  {
    // Check constructor with ArrayRef.
    SmallVector<sandboxir::Instruction *> Vec({S0, S1, Ret});
    DmpVector<sandboxir::Instruction *> SBVec(Vec);
    EXPECT_EQ(SBVec.size(), 3u);
    EXPECT_EQ(SBVec[0], S0);
    EXPECT_EQ(SBVec[1], S1);
    EXPECT_EQ(SBVec[2], Ret);
  }
  {
    // Check copy constructor
    DmpVector<sandboxir::Instruction *> SBVec({S0, S1, Ret});
    DmpVector<sandboxir::Instruction *> SBVecCpy(SBVec);
    EXPECT_EQ(SBVec, SBVecCpy);
  }
  {
    // Check move constructor
    DmpVector<sandboxir::Instruction *> SBVec({S0, S1, Ret});
    DmpVector<sandboxir::Instruction *> SBVecMv(std::move(SBVec));
    EXPECT_EQ(SBVecMv.size(), 3u);
    EXPECT_EQ(SBVecMv[0], S0);
    EXPECT_EQ(SBVecMv[1], S1);
    EXPECT_EQ(SBVecMv[2], Ret);
  }
  {
    // Check reserve constructor
    DmpVector<sandboxir::Instruction *> SBVec(10);
    EXPECT_EQ(SBVec.size(), 0u);
  }
  {
    // Check copy assignment
    DmpVector<sandboxir::Instruction *> SBVec({S0, S1, Ret});
    DmpVector<sandboxir::Instruction *> SBVecCpy = SBVec;
    EXPECT_EQ(SBVec, SBVecCpy);
  }
  {
    // Check range constructor
    DmpVector<sandboxir::Instruction *> SBVec({S0, S1, Ret});
    DmpVector<sandboxir::Instruction *> SBVecCpy(SBVec.begin(), SBVec.end());
    EXPECT_EQ(SBVec, SBVecCpy);
  }
}

TEST(DmpVector, DmpVectorView) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"IR(
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
  (void)SBF;
  sandboxir::BasicBlock &SBBB = *Ctx.getBasicBlock(BB);
  auto It = SBBB.begin();
  auto *S0 = &*It++;
  auto *S1 = &*It++;
  auto *Ret = &*It++;
  DmpVector<sandboxir::Instruction *> SBVec({S0, S1, Ret});
  {
    // Default constructor.
    DmpVectorView<sandboxir::Instruction *> SBVecView;
    EXPECT_TRUE(SBVecView.empty());
  }
  {
    // DmpVector consturctor.
    DmpVectorView<sandboxir::Instruction *> SBVecView(SBVec);
    EXPECT_EQ(SBVecView.size(), SBVec.size());
    for (auto [Idx, I] : enumerate(SBVecView))
      EXPECT_EQ(SBVec[Idx], I);
  }
  {
    // Range constructor.
    DmpVectorView<sandboxir::Instruction *> SBVecView(SBVec.begin(),
                                                        std::prev(SBVec.end()));
    EXPECT_EQ(SBVecView.size(), SBVec.size() - 1);
    for (auto [Idx, I] : enumerate(SBVecView))
      EXPECT_EQ(SBVec[Idx], I);
  }
  {
    // ArrayRef constructor.
    ArrayRef<sandboxir::Instruction *> Array(SBVec.begin(), SBVec.end());
    DmpVectorView<sandboxir::Instruction *> SBVecView(Array);
    EXPECT_EQ(SBVecView.size(), Array.size());
    for (auto [Idx, I] : enumerate(SBVecView))
      EXPECT_EQ(Array[Idx], I);
  }
  {
    // Implicit conversion from DmpVector
    auto ImplConv = [](DmpVectorView<sandboxir::Instruction *> View) {};
    ImplConv(SBVec);
  }
  // Check that some of the inherited member functions that return ArrayRef can
  // be converted to DmpVectorView.
  {
    // slice(N, M)
    DmpVectorView<sandboxir::Instruction *> SBVecView(SBVec);
    DmpVectorView<sandboxir::Instruction *> SliceView = SBVecView.slice(1, 2);
    EXPECT_EQ(SliceView.size(), 2u);
    EXPECT_EQ(SliceView[0], SBVec[1]);
    EXPECT_EQ(SliceView[1], SBVec[2]);
  }
  {
    // slice(N)
    DmpVectorView<sandboxir::Instruction *> SBVecView(SBVec);
    DmpVectorView<sandboxir::Instruction *> SliceView = SBVecView.slice(1);
    EXPECT_EQ(SliceView.size(), 2u);
    EXPECT_EQ(SliceView[0], SBVec[1]);
    EXPECT_EQ(SliceView[1], SBVec[2]);
  }
  {
    // drop_back(N)
    DmpVectorView<sandboxir::Instruction *> SBVecView(SBVec);
    DmpVectorView<sandboxir::Instruction *> SliceView =
        SBVecView.drop_back(2);
    EXPECT_EQ(SliceView.size(), 1u);
    EXPECT_EQ(SliceView[0], SBVec[0]);
  }
}
