//===- Pass.cpp - Passes that operate on the SBBasicBlock -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVec/PassManager.h"

using namespace llvm;

sandboxir::PassManager *sandboxir::SBPass::asPM() {
  switch (getSubclassID()) {
  case ClassID::BBPassManager:
    return cast<SBBBPassManager>(this);
  case ClassID::RegionPassManager:
    return cast<sandboxir::RegionPassManager>(this);
  default:
    return nullptr;
  }
}

#ifndef NDEBUG
void sandboxir::SBPass::dump(raw_ostream &OS) const {
  OS << Name << " " << Flag;
}

void sandboxir::SBPass::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG
