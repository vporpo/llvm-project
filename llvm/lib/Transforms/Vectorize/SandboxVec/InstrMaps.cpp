//===- InstructionMaps.cpp - Maps scalars to vectors and reverse ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/InstrMaps.h"
#include "llvm/Transforms/Vectorize/SandboxVec/InstrRange.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SBRegion.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"

using namespace llvm;

bool InstructionMaps::empty() const {
  return ScalarToVectorInstrMap.empty() && VectorToScalarsMap.empty();
}

void InstructionMaps::clear() {
  ScalarToVectorInstrMap.clear();
  VectorToScalarsMap.clear();
}

void InstructionMaps::eraseVector(SBInstruction *Vec) {
  auto It = VectorToScalarsMap.find(Vec);
  if (It == VectorToScalarsMap.end())
    return;
  VectorToScalarsMap.erase(It);
  for (auto *Scalar : getScalarsFor(Vec))
    ScalarToVectorInstrMap.erase(Scalar);
}

void InstructionMaps::eraseScalar(SBInstruction *Scalar) {
  auto It = ScalarToVectorInstrMap.find(Scalar);
  if (It == ScalarToVectorInstrMap.end())
    return;
  auto *Vec = getVectorForScalar(Scalar);
  VectorToScalarsMap.erase(Vec);
  ScalarToVectorInstrMap.erase(It);
}

void InstructionMaps::registerVector(SBValue *Vec,
                                     const SBValBundle &Scalars) {
  for (auto [Lane, SBV] : enumerate(Scalars)) {
    auto *SBI = cast<SBInstruction>(SBV);
    ScalarToVectorInstrMap.try_emplace(SBI, Vec);
  }
  VectorToScalarsMap.try_emplace(Vec, Scalars);
}

SBValue *InstructionMaps::getVectorForScalar(SBValue *Scalar) const {
  auto It = ScalarToVectorInstrMap.find(Scalar);
  return It != ScalarToVectorInstrMap.end() ? It->second : nullptr;
}

unsigned InstructionMaps::getScalarLane(SBValue *Vec,
                                        SBValue *Scalar) const {
  auto MapIt = VectorToScalarsMap.find(Vec);
  assert(MapIt != VectorToScalarsMap.end() && "Not registered!");
  const auto &Bndl = MapIt->second;
  auto ScalarIt = find(Bndl, Scalar);
  // Not all "scalar" elements are actually scalar. For example we may have
  // a vector like: {<2 x double>, double}.
  // In that case if we are looking for the lane of the 2nd element we should
  // return 2, not 1.
  unsigned CntLanes = 0;
  for (auto It = Bndl.begin(); It != ScalarIt; ++It)
    CntLanes += SBUtils::getNumLanes(*It);
  return CntLanes;
}

const SBValBundle &InstructionMaps::getScalarsFor(SBValue *Vec) const {
  auto It = VectorToScalarsMap.find(Vec);
  static SBValBundle EmptyBndl;
  return It != VectorToScalarsMap.end() ? It->second : EmptyBndl;
}

std::pair<DenseSet<SBValue *>, bool>
InstructionMaps::getVectorsThatCombinedScalars(
    const SBValBundle &Bndl) const {
  DenseSet<SBValue *> Matches;
  bool SetIsIncomplete = false;
  for (SBValue *SBV : Bndl) {
    auto *SBI = dyn_cast<SBInstruction>(SBV);
    if (SBI == nullptr) {
      SetIsIncomplete = true;
      continue;
    }
    if (auto *NewMatch = getVectorForScalar(SBI)) {
      Matches.insert(NewMatch);
      continue;
    }
    SetIsIncomplete = true;
  }
  return {Matches, SetIsIncomplete};
}

std::optional<ShuffleMask>
InstructionMaps::getShuffleMask(SBValue *Vec,
                                const SBValBundle &OtherBndl) const {
  const SBValBundle &Bndl = getScalarsFor(Vec);
  // Collect this lane indices of this node's values.
  DenseMap<SBValue *, int> IndexMap;
  for (auto [Idx, N] : enumerate(Bndl))
    IndexMap[N] = Idx;
  ShuffleMask::IndicesVecT ShuffleIndices;
  ShuffleIndices.reserve(SBUtils::getNumLanes(Vec));
  // Now go over OtherBndl's values and collect the indices that each value
  // corresponds to.
  for (SBValue *OtherV : OtherBndl) {
    auto It = IndexMap.find(OtherV);
    if (It == IndexMap.end())
      // Not found in Bndl.
      return std::nullopt;
    auto Idx = It->second;
    // For vector values we need to push `Idx` Lanes times.
    for (unsigned Cnt = 0, Lanes = SBUtils::getNumLanes(OtherV); Cnt != Lanes;
         ++Cnt)
      ShuffleIndices.push_back(Idx);
  }
  return ShuffleMask(std::move(ShuffleIndices));
}

#ifndef NDEBUG
void InstructionMaps::dump(raw_ostream &OS) const {
  OS << "\nScalarToVector:\n";
  for (const auto &Pair : ScalarToVectorInstrMap)
    OS.indent(2) << *Pair.first << " : " << *Pair.second << "\n";

  OS << "\nVectorToScalars:\n";
  for (const auto &Pair : VectorToScalarsMap) {
    OS.indent(2) << *Pair.first << " : [";
    for (auto *SBV : Pair.second)
      OS << *SBV << ", ";
    OS << "]\n";
  }
}

void InstructionMaps::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif
