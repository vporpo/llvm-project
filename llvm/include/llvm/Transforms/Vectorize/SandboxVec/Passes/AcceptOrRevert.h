//===- AcceptOrRevertPass.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_ACCEPTORREVERT_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_ACCEPTORREVERT_H

#include "llvm/Transforms/Vectorize/SandboxVec/SBPass.h"

namespace llvm {

class SBContext;

class AcceptOrRevert : public SBRegionPass {
  SBContext &Ctxt;

public:
  AcceptOrRevert(SBContext &Ctxt)
      : SBRegionPass("AcceptOrRevert", "accept-or-revert"), Ctxt(Ctxt) {}
  bool runOnRegion(SBRegion &Rgn) final;
};

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_ACCEPTORREVERT_H
