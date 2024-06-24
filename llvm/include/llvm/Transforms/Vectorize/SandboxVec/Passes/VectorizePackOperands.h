//===- VectorizePackOperands.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_VECTORIZEPACKOPERANDS_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_VECTORIZEPACKOPERANDS_H

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Pass.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIRVec.h"

namespace llvm {
namespace sandboxir {
class Context;

class VectorizePackOperands : public sandboxir::RegionPass {
  sandboxir::SBVecContext &Ctx;
  ScalarEvolution &SE;
  const DataLayout &DL;
  TargetTransformInfo &TTI;

public:
  VectorizePackOperands(sandboxir::SBVecContext &Ctx, ScalarEvolution &SE,
                        const DataLayout &DL, TargetTransformInfo &TTI)
      : sandboxir::RegionPass("VectorizePackOperands", "vectorize-packs"),
        Ctx(Ctx), SE(SE), DL(DL), TTI(TTI) {}
  bool runOnRegion(sandboxir::Region &Rgn) final;
};

} // namespace sandboxir
} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_VECTORIZEPACKOPERANDS_H
