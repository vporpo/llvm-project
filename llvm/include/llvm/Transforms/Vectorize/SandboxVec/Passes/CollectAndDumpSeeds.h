//===- CollectAndDumpSeeds.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_COLLECTANDDUMPSEEDS_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_COLLECTANDDUMPSEEDS_H

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Pass.h"

namespace llvm {
namespace sandboxir {

class Context;

class CollectAndDumpSeeds : public sandboxir::SBBBPass {
  ScalarEvolution &SE;
  const DataLayout &DL;

public:
  CollectAndDumpSeeds(sandboxir::Context &Ctx, ScalarEvolution &SE,
                      const DataLayout &DL, TargetTransformInfo &TTI)
      : sandboxir::SBBBPass("CollectAndDumpSeeds", "collect-and-dump-seeds"),
        SE(SE), DL(DL) {}
  bool runOnSBBasicBlock(sandboxir::BasicBlock &SBBB) final;
};

} // namespace sandboxir
} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_COLLECTANDDUMPSEEDS_H
