//===- PackReuse.cpp - Deletes redundant packs ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Passes/PackReuse.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"

using namespace llvm;

bool PackReuse::runOnRegion(SBRegion &Rgn) {
  bool Change = false;
  DenseMap<SBValBundle, SBPackInstruction *> PackOpsToPackMap;
  for (SBValue *SBV : make_early_inc_range(Rgn)) {
    if (auto *Pack = dyn_cast<SBPackInstruction>(SBV)) {
      auto PackOpsRange = Pack->operands();
      SBValBundle PackOps(PackOpsRange.begin(), PackOpsRange.end());
      auto Pair = PackOpsToPackMap.insert({PackOps, Pack});
      if (!Pair.second) {
        auto *MatchingPack = Pair.first->second;
        if (MatchingPack->comesBefore(Pack)) {
          Pack->replaceAllUsesWith(MatchingPack);
        } else {
          MatchingPack->replaceAllUsesWith(Pack);
          // Replace MatchingPack with Pack in the map.
          Pair.first->second = Pack;
          // TODO: is it safe to erase while iterating?
          MatchingPack->eraseFromParent();
        }
        Change = true;
      }
    }
  }
  return Change;
}
