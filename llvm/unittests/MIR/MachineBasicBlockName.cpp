//===- MachineBasicBlockName.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "gtest/gtest.h"

using namespace llvm;

extern cl::opt<bool> PersistentMBBNames;

TEST(MachineBasicBlockNameTest, BasicTest) {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  PersistentMBBNames.setValue(true);
  LLVMContext C;
  Module M("Test", C);
  auto *FType = FunctionType::get(Type::getVoidTy(C), false);
  M.setDataLayout("e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-"
                  "f80:128-n8:16:32:64-S128");
  Triple TargetTriple("x86_64--");
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
  if (!T)
    GTEST_SKIP();
  auto *F = Function::Create(FType, GlobalValue::ExternalLinkage, "Test", &M);
  TargetOptions Options;
  auto Machine = std::unique_ptr<TargetMachine>(
      T->createTargetMachine(TargetTriple, "", "", Options, std::nullopt,
                             std::nullopt, CodeGenOptLevel::Aggressive));
  const TargetSubtargetInfo &STI = *Machine->getSubtargetImpl(*F);
  auto MMI = std::make_unique<MachineModuleInfo>(Machine.get());
  auto MF = std::make_unique<MachineFunction>(*F, *Machine, STI,
                                              MMI->getContext(), 42);

  auto CreateMBB = [&](Twine BBName) {
    auto *BB = BasicBlock::Create(C, BBName, F);
    IRBuilder<> IRB(BB);
    IRB.CreateRetVoid();
    auto *MBB = MF->CreateMachineBasicBlock(BB);
    MF->insert(MF->end(), MBB);
    return MBB;
  };

  auto GetMBBName = [](MachineBasicBlock *MBB) {
    std::string Str;
    raw_string_ostream SS(Str);
    MBB->printName(SS);
    return Str;
  };

  auto *MBB0 = CreateMBB("entry");
  auto *MBB1 = CreateMBB("foo");
  auto *MBB2 = CreateMBB("bar");

  EXPECT_EQ(GetMBBName(MBB0), "bb.0.entry");
  EXPECT_EQ(GetMBBName(MBB1), "bb.1.foo");
  EXPECT_EQ(GetMBBName(MBB2), "bb.2.bar");
  // Check MBB numbers before renumbering.
  EXPECT_EQ(MBB0->getNumber(), 0);
  EXPECT_EQ(MBB1->getNumber(), 1);
  EXPECT_EQ(MBB2->getNumber(), 2);
  // Check MBB names before renumbering.
  EXPECT_EQ(GetMBBName(MBB0), "bb.0.entry");
  EXPECT_EQ(GetMBBName(MBB1), "bb.1.foo");
  EXPECT_EQ(GetMBBName(MBB2), "bb.2.bar");

  // Now erase MBB1, renumber and check again.
  MBB1->eraseFromParent();
  MF->RenumberBlocks();

  // Check MBB numbers after renumbering.
  EXPECT_EQ(MBB0->getNumber(), 0);
  EXPECT_EQ(MBB2->getNumber(), 1);
  // Check MBB names after renumbering.
  EXPECT_EQ(GetMBBName(MBB0), "bb.0.entry");
  EXPECT_EQ(GetMBBName(MBB2), "bb.2.bar");
}
