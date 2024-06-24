//===- VectorizePackOperands.cpp - SB Pass that vectorizes pack operands --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Passes/VectorizePackOperands.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Analysis.h"
#include "llvm/Transforms/Vectorize/SandboxVec/CostModel.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Passes/VectorizeScalars.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"

using namespace llvm;

// TODO: This is currently a single pass, but we may introduce new candidate
// Packs, so ideally we should iterate until now more candidates are found.
bool sandboxir::VectorizePackOperands::runOnRegion(sandboxir::Region &Rgn) {
  bool Change = false;
  // Collect Pack candidates that didn't get vectorized for the right reason
  SmallVector<sandboxir::PackInst *, 8> PackCandidates;
  for (auto *SBI : Rgn) {
    auto *PackI = dyn_cast<sandboxir::PackInst>(SBI);
    if (PackI == nullptr)
      continue;
    PackCandidates.push_back(PackI);
  }

  SmallPtrSet<sandboxir::Instruction *, 4> EraseCandidates;

  if (PackCandidates.empty())
    return false;

  auto &SBBB = *Rgn.getParent();

  // Analyze each Pack and extend vectorization graph if possible.
  for (sandboxir::PackInst *Pack : PackCandidates) {
    auto PackOpsRange = Pack->operands();
    DmpVector<sandboxir::Value *> PackOps(PackOpsRange.begin(),
                                            PackOpsRange.end());
    // Look for subsets of same opcode.
    MapVector<sandboxir::Instruction::Opcode,
              std::pair<DmpVector<sandboxir::Value *>, SmallVector<unsigned>>>
        OpcodesToBndlAndLanes;
    for (auto [Lane, SBV] : enumerate(PackOps)) {
      if (!isa<sandboxir::Instruction>(SBV))
        continue;
      auto Opcode = cast<sandboxir::Instruction>(SBV)->getOpcode();
      auto &Pair = OpcodesToBndlAndLanes[Opcode];
      Pair.first.push_back(SBV);
      Pair.second.push_back(Lane);
    }
    // Try to extend the vectorization graph.
    for (auto &Pair : OpcodesToBndlAndLanes) {
      auto &[NewBndl, PackLanes] = Pair.second;
      if (NewBndl.size() <= 1)
        continue;
      // We just found a bundle that we can vectorize!
      VectorizeFromSeeds Vec(&SBBB, Ctx, SE, DL, TTI);
      // TODO: Try different slices of NewBndl
      sandboxir::Instruction *NewI =
          Vec.tryVectorize(NewBndl, Rgn, EraseCandidates);
      if (NewI == nullptr)
        continue;
      if (sandboxir::VecUtils::getNumLanes(NewI) ==
          sandboxir::VecUtils::getNumLanes(Pack)) {
        // Optimal case: Pack is no longer needed.
        Pack->replaceAllUsesWith(NewI);
        Pack->eraseFromParent();
      } else {
        // We also need Unpack nodes to extract from NewI and feed into Pack.
        for (auto Lane : seq<unsigned>(0, NewBndl.size())) {
          unsigned PackLane = PackLanes[Lane];
          unsigned Lanes = sandboxir::VecUtils::getNumElements(
              Pack->getOperand(PackLane)->getType());
          auto WhereIt =
              sandboxir::VecUtils::getInsertPointAfter({NewI}, &SBBB);
          auto *Unpack = sandboxir::UnpackInst::create(
              NewI, Lane, Lanes, WhereIt, &SBBB, Ctx);
          Pack->setOperand(PackLane, Unpack);
        }
      }
      Change = true;
    }
  }
  return Change;
}
