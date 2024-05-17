//===- DumpRegion.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_DUMPREGION_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_DUMPREGION_H

#include "llvm/Transforms/Vectorize/SandboxVec/SBPass.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SBRegion.h"

namespace llvm {

/// A helper pass that dumps an SB region with a given root.
class DumpRegion : public SBRegionPass {
public:
  DumpRegion() : SBRegionPass("DumpRegion", "dump-region") {}
  bool runOnRegion(SBRegion &Rgn) final;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_DUMPREGION_H
