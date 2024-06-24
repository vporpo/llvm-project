//===- AcceptOrRevertPass.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_ACCEPTORREVERT_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_ACCEPTORREVERT_H

#include "llvm/Transforms/Vectorize/SandboxVec/Pass.h"

namespace llvm {
namespace sandboxir {

class SBVecContext;

class AcceptOrRevert : public sandboxir::RegionPass {
  sandboxir::SBVecContext &Ctx;

public:
  AcceptOrRevert(sandboxir::SBVecContext &Ctx)
      : sandboxir::RegionPass("AcceptOrRevert", "accept-or-revert"),
        Ctx(Ctx) {}
  bool runOnRegion(sandboxir::Region &Rgn) final;
};

} // namespace sandboxir
} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_ACCEPTORREVERT_H
