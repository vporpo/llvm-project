//===- CollectAndDumpSeeds.cpp - Helper pass for testing seed collection --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Passes/CollectAndDumpSeeds.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SeedCollector.h"

#define DEBUG_TYPE "SBVec"

using namespace llvm;

bool sandboxir::CollectAndDumpSeeds::runOnSBBasicBlock(
    sandboxir::BasicBlock &SBBB) {
  // Used for testing the seed collector in lit tests.
  sandboxir::SeedCollector SC(&SBBB, DL, SE);
#ifndef NDEBUG
  SC.dump(dbgs());
#else
  dbgs() << "Dumps enabled in Debug build!\n";
#endif // NDEBUG
  return false;
}
