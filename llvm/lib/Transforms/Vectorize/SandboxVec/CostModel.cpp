//===- CostModel.cpp - Cost estimation ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/CostModel.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Transforms/Vectorize/SandboxVec/InstrInterval.h"
#include "llvm/SandboxIR/SandboxIR.h"

using namespace llvm;

InstructionCost
sandboxir::CostModel::getCost(sandboxir::Instruction *SBI) const {
  InstructionCost Cost = 0;
  for (llvm::Instruction *I : SBI->getLLVMInstrs()) {
    SmallVector<const llvm::Value *> Operands(I->operands());
    Cost += TTI.getInstructionCost(I, Operands, CostKind);
  }
  return Cost;
}
