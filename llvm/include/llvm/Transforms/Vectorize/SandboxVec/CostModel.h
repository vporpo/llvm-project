//===- CostModel.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_COSTMODEL_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_COSTMODEL_H

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/InstructionCost.h"

namespace llvm {
namespace sandboxir {

class Instruction;

class CostModel {
  TargetTransformInfo &TTI;
  constexpr static TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;

public:
  CostModel(TargetTransformInfo &TTI) : TTI(TTI) {}
  /// \Returns the cost of \p SBI. Internally this goes over all LLVM IR
  /// instructions that make up \p SBI and adds their cost.
  InstructionCost getCost(sandboxir::Instruction *SBI) const;
};

} // namespace sandboxir
} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_COSTMODEL_H
