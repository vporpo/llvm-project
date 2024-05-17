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
#include "llvm/Transforms/Vectorize/SandboxVec/SBPass.h"

namespace llvm {
class SBContext;

class VectorizePackOperands : public SBRegionPass {
  SBContext &Ctxt;
  ScalarEvolution &SE;
  const DataLayout &DL;
  TargetTransformInfo &TTI;

public:
  VectorizePackOperands(SBContext &Ctxt, ScalarEvolution &SE,
                        const DataLayout &DL, TargetTransformInfo &TTI)
      : SBRegionPass("VectorizePackOperands", "vectorize-packs"), Ctxt(Ctxt),
        SE(SE), DL(DL), TTI(TTI) {}
  bool runOnRegion(SBRegion &Rgn) final;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_VECTORIZEPACKOPERANDS_H
