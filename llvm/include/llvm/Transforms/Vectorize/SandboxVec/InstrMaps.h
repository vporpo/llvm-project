//===- InstructionMaps.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_INSTRUCTIONMAPS_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_INSTRUCTIONMAPS_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Utils.h"

namespace llvm {

class SBInstruction;
class SBGenericInstruction;
class SBRegion;

/// Helper for mapped_iterator. \Returns the \p Pair.first.
static SBInstruction *
pairFirst(const std::pair<SBValue *, SBValue *> &Pair) {
  return cast<SBInstruction>(Pair.first);
}

class InstructionMaps {
  /// A map from all scalars that got combined into vectors to the vector
  /// instruction.
  DenseMap<SBValue *, SBValue *> ScalarToVectorInstrMap;
  /// Map from the vector instruction to the scalars that got vectorized.
  DenseMap<SBValue *, SBValBundle> VectorToScalarsMap;
  /// We maintain this region
  SBRegion *Rgn = nullptr;

public:
  using ScalarsIterator =
      mapped_iterator<DenseMap<SBValue *, SBValue *>::const_iterator,
                      decltype(*pairFirst)>;
  /// \Returns the range of scalars that are in the maps.
  iterator_range<ScalarsIterator> getScalars() const {
    ScalarsIterator Begin(ScalarToVectorInstrMap.begin(), pairFirst);
    ScalarsIterator End(ScalarToVectorInstrMap.end(), pairFirst);
    return make_range(Begin, End);
  }
  bool empty() const;
  void clear();
  void eraseVector(SBInstruction *Vec);
  void eraseScalar(SBInstruction *Scalar);
  void registerVector(SBValue *Vec, const SBValBundle &Scalars);
  SBValue *getVectorForScalar(SBValue *Scalar) const;
  /// This can handle "scalar" elements that are actually vectors.
  /// Like in `{<2 x double>, double}` it would return 2 for the lane of the
  /// second element, not 1.
  unsigned getScalarLane(SBValue *Vec, SBValue *Scalar) const;
  /// \Returns an empty bundle if not found.
  const SBValBundle &getScalarsFor(SBValue *Vec) const;
  /// \Returns: (i) the set of vector values that correspond to \p Bndl and (ii)
  /// whether there are scalar values that do not correspond to any vector
  /// value.
  std::pair<DenseSet<SBValue *>, bool>
  getVectorsThatCombinedScalars(const SBValBundle &Bndl) const;
  std::optional<ShuffleMask>
  getShuffleMask(SBValue *Vec, const SBValBundle &OtherBndl) const;
  void setRegion(SBRegion &Region) { Rgn = &Region; }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_INSTRUCTIONMAPS_H
