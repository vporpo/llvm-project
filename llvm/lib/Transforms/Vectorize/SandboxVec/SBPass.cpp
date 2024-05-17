//===- SBPass.cpp - Passes that operate on the SBBasicBlock
//------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/SBPass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SBPassManager.h"

using namespace llvm;

SBPassManager *SBPass::asPM() {
  switch (getSubclassID()) {
  case ClassID::BBPassManager:
    return cast<SBBBPassManager>(this);
  case ClassID::RegionPassManager:
    return cast<SBRegionPassManager>(this);
  default:
    return nullptr;
  }
}

#ifndef NDEBUG
void SBPass::dump(raw_ostream &OS) const { OS << Name << " " << Flag; }

void SBPass::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG
