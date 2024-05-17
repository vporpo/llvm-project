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
#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Passes/VectorizeScalars.h"

using namespace llvm;

// TODO: This is currently a single pass, but we may introduce new candidate
// PackNodes, so ideally we should iterate until now more candidates are found.
bool VectorizePackOperands::runOnRegion(SBRegion &Rgn) {
  bool Change = false;
  // Collect PackNode candidates that didn't get vectorized for the right reason
  SmallVector<SBPackInstruction *, 8> PackNodeCandidates;
  for (auto *SBI : Rgn) {
    auto *PackI = dyn_cast<SBPackInstruction>(SBI);
    if (PackI == nullptr)
      continue;
    PackNodeCandidates.push_back(PackI);
  }

  SmallPtrSet<SBInstruction *, 4> EraseCandidates;

  if (PackNodeCandidates.empty())
    return false;

  auto &SBBB = *Rgn.getParent();

  // Analyze each PackNode and extend vectorization graph if possible.
  for (SBPackInstruction *Pack : PackNodeCandidates) {
    auto PackOpsRange = Pack->operands();
    SBValBundle PackOps(PackOpsRange.begin(), PackOpsRange.end());
    // Look for subsets of same opcode.
    MapVector<SBInstruction::Opcode,
              std::pair<SBValBundle, SmallVector<unsigned>>>
        OpcodesToBndlAndLanes;
    for (auto [Lane, SBV] : enumerate(PackOps)) {
      if (!isa<SBInstruction>(SBV))
        continue;
      auto Opcode = cast<SBInstruction>(SBV)->getOpcode();
      auto &Pair = OpcodesToBndlAndLanes[Opcode];
      Pair.first.push_back(SBV);
      Pair.second.push_back(Lane);
    }
    // Try to generate the new nodes and extend the vectorization graph.
    for (auto &Pair : OpcodesToBndlAndLanes) {
      auto &[NewBndl, PackLanes] = Pair.second;
      if (NewBndl.size() <= 1)
        continue;
      // We just found a bundle that we can vectorize!
      VectorizeFromSeeds Vec(&SBBB, Ctxt, SE, DL, TTI);
      // TODO: Try different slices of NewBndl
      SBInstruction *NewN = Vec.tryVectorize(NewBndl, Rgn, EraseCandidates);
      if (NewN == nullptr)
        continue;
      if (NewN->lanes() == Pack->lanes()) {
        // Optimal case: PackN is no longer needed.
        Pack->replaceAllUsesWith(NewN);
        Pack->eraseFromParent();
      } else {
        // We also need Unpack nodes to extract from NewN and feed into PackN.
        for (auto Lane : seq<unsigned>(0, NewBndl.size())) {
          unsigned PackLane = PackLanes[Lane];
          unsigned Lanes =
              SBUtils::getNumElements(Pack->getOperand(PackLane)->getType());
          auto *UnpackN =
              Ctxt.createSBUnpackInstruction(NewN, Lane, &SBBB, Lanes);
          Pack->setOperand(PackLane, UnpackN);
        }
      }
      Change = true;
    }
  }
  VectorizeFromSeeds::tryEraseDeadInstrs(EraseCandidates);
  return Change;
}
