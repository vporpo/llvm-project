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
#include "llvm/Transforms/Vectorize/SandboxVec/SBPass.h"

namespace llvm {

class SBContext;

class CollectAndDumpSeeds : public SBBBPass {
  ScalarEvolution &SE;
  const DataLayout &DL;

public:
  CollectAndDumpSeeds(SBContext &Ctxt, ScalarEvolution &SE,
                      const DataLayout &DL, TargetTransformInfo &TTI)
      : SBBBPass("CollectAndDumpSeeds", "collect-and-dump-seeds"), SE(SE),
        DL(DL) {}
  bool runOnSBBasicBlock(SBBasicBlock &SBBB) final;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_COLLECTANDDUMPSEEDS_H
