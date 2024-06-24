//===- DumpRegion.cpp - Helper pass that prints an SBRegion ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Passes/DumpRegion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "SBVec"

using namespace llvm;

bool sandboxir::DumpRegion::runOnRegion(sandboxir::Region &Rgn) {
#ifndef NDEBUG
  dbgs() << Rgn << "\n";
#else
  llvm_unreachable("Requires Debug build!");
#endif
  return false;
}
