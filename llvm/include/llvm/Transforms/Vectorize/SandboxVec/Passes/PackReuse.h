//===- PackReuse.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_PACKREUSE_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_PACKREUSE_H

#include "llvm/Transforms/Vectorize/SandboxVec/Pass.h"

namespace llvm {
namespace sandboxir {

class PackReuse : public sandboxir::RegionPass {
public:
  PackReuse() : sandboxir::RegionPass("PackReuse", "pack-reuse") {}
  bool runOnRegion(sandboxir::Region &Rgn) final;
};

} // namespace sandboxir
} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_PACKREUSE_H
