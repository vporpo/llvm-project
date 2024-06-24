//===- AcceptOrRevertPass.cpp - Check cost and accept/revert the region ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Passes/AcceptOrRevert.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"

using namespace llvm;

extern llvm::cl::opt<int> CostThreshold;

bool sandboxir::AcceptOrRevert::runOnRegion(sandboxir::Region &Rgn) {
  InstructionCost CostAfterMinusBefore = Rgn.getVectorMinusScalarCost();
  if (CostAfterMinusBefore < -CostThreshold) {
    Ctx.getTracker().accept();
    return false;
  }
  auto &Sched = *Ctx.getScheduler(Rgn.getBB());
  // Don't maintain the view when reverting.
  Sched.getDAG().resetView();
  // We don't maintain the scheduler when reverting.
  Sched.clearState();
  // Revert the IR.
  Ctx.getTracker().revert();
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  Sched.getDAG().verify(/*CheckReadyCnt=*/false);
#endif
  sandboxir::RegionAttorney::dropMetadata(Rgn);
  return true;
}
