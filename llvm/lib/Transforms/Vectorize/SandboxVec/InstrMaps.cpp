//===- InstructionMaps.cpp - Maps scalars to vectors and reverse ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/InstrMaps.h"
#include "llvm/Transforms/Vectorize/SandboxVec/InstrInterval.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Region.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIRVec.h"

using namespace llvm;

bool sandboxir::InstructionMaps::empty() const {
  return ScalarToVectorInstrMap.empty() && VectorToScalarsMap.empty();
}

void sandboxir::InstructionMaps::clear() {
  ScalarToVectorInstrMap.clear();
  VectorToScalarsMap.clear();
}

void sandboxir::InstructionMaps::eraseVector(sandboxir::Instruction *Vec) {
  auto It = VectorToScalarsMap.find(Vec);
  if (It == VectorToScalarsMap.end())
    return;
  VectorToScalarsMap.erase(It);
  for (auto *Scalar : getScalarsFor(Vec))
    ScalarToVectorInstrMap.erase(Scalar);
}

void sandboxir::InstructionMaps::eraseScalar(sandboxir::Instruction *Scalar) {
  auto It = ScalarToVectorInstrMap.find(Scalar);
  if (It == ScalarToVectorInstrMap.end())
    return;
  auto *Vec = getVectorForScalar(Scalar);
  VectorToScalarsMap.erase(Vec);
  ScalarToVectorInstrMap.erase(It);
}

void sandboxir::InstructionMaps::registerVector(
    sandboxir::Value *Vec, const DmpVector<sandboxir::Value *> &Scalars) {
  for (auto [Lane, SBV] : enumerate(Scalars)) {
    auto *SBI = cast<sandboxir::Instruction>(SBV);
    ScalarToVectorInstrMap.try_emplace(SBI, Vec);
  }
  VectorToScalarsMap.try_emplace(Vec, Scalars);
}

sandboxir::Value *sandboxir::InstructionMaps::getVectorForScalar(
    sandboxir::Value *Scalar) const {
  auto It = ScalarToVectorInstrMap.find(Scalar);
  return It != ScalarToVectorInstrMap.end() ? It->second : nullptr;
}

unsigned
sandboxir::InstructionMaps::getScalarLane(sandboxir::Value *Vec,
                                          sandboxir::Value *Scalar) const {
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
    CntLanes += sandboxir::VecUtils::getNumLanes(*It);
  return CntLanes;
}

const DmpVector<sandboxir::Value *> &
sandboxir::InstructionMaps::getScalarsFor(sandboxir::Value *Vec) const {
  auto It = VectorToScalarsMap.find(Vec);
  static DmpVector<sandboxir::Value *> EmptyBndl;
  return It != VectorToScalarsMap.end() ? It->second : EmptyBndl;
}

std::pair<DenseSet<sandboxir::Value *>, bool>
sandboxir::InstructionMaps::getVectorsThatCombinedScalars(
    const DmpVector<sandboxir::Value *> &Bndl) const {
  DenseSet<sandboxir::Value *> Matches;
  bool SetIsIncomplete = false;
  for (sandboxir::Value *SBV : Bndl) {
    auto *SBI = dyn_cast<sandboxir::Instruction>(SBV);
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

std::optional<sandboxir::ShuffleMask>
sandboxir::InstructionMaps::getShuffleMask(
    sandboxir::Value *Vec,
    const DmpVector<sandboxir::Value *> &OtherBndl) const {
  const DmpVector<sandboxir::Value *> &Bndl = getScalarsFor(Vec);
  // Collect this lane indices of this node's values.
  DenseMap<sandboxir::Value *, int> IndexMap;
  for (auto [Idx, N] : enumerate(Bndl))
    IndexMap[N] = Idx;
  sandboxir::ShuffleMask::IndicesVecT ShuffleIndices;
  ShuffleIndices.reserve(sandboxir::VecUtils::getNumLanes(Vec));
  // Now go over OtherBndl's values and collect the indices that each value
  // corresponds to.
  for (sandboxir::Value *OtherV : OtherBndl) {
    auto It = IndexMap.find(OtherV);
    if (It == IndexMap.end())
      // Not found in Bndl.
      return std::nullopt;
    auto Idx = It->second;
    // For vector values we need to push `Idx` Lanes times.
    for (unsigned Cnt = 0, Lanes = sandboxir::VecUtils::getNumLanes(OtherV);
         Cnt != Lanes; ++Cnt)
      ShuffleIndices.push_back(Idx);
  }
  return sandboxir::ShuffleMask(std::move(ShuffleIndices));
}

#ifndef NDEBUG
void sandboxir::InstructionMaps::dump(raw_ostream &OS) const {
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

void sandboxir::InstructionMaps::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif
