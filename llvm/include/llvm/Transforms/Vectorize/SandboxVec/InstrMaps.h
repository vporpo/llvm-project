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
#include "llvm/Support/Casting.h"
#include "llvm/SandboxIR/DmpVector.h"

namespace llvm {
namespace sandboxir {

class Instruction;
class SBGenericInstruction;
class Region;
class Value;
class ShuffleMask;

/// Helper for mapped_iterator. \Returns the \p Pair.first.
static sandboxir::Instruction *
pairFirst(const std::pair<sandboxir::Value *, sandboxir::Value *> &Pair) {
  return cast<sandboxir::Instruction>(Pair.first);
}

class InstructionMaps {
  /// A map from all scalars that got combined into vectors to the vector
  /// instruction.
  DenseMap<sandboxir::Value *, sandboxir::Value *> ScalarToVectorInstrMap;
  /// Map from the vector instruction to the scalars that got vectorized.
  DenseMap<sandboxir::Value *, DmpVector<sandboxir::Value *>>
      VectorToScalarsMap;
  /// We maintain this region
  sandboxir::Region *Rgn = nullptr;

public:
  using ScalarsIterator = mapped_iterator<
      DenseMap<sandboxir::Value *, sandboxir::Value *>::const_iterator,
      decltype(*pairFirst)>;
  /// \Returns the range of scalars that are in the maps.
  iterator_range<ScalarsIterator> getScalars() const {
    ScalarsIterator Begin(ScalarToVectorInstrMap.begin(), pairFirst);
    ScalarsIterator End(ScalarToVectorInstrMap.end(), pairFirst);
    return make_range(Begin, End);
  }
  bool empty() const;
  void clear();
  void eraseVector(sandboxir::Instruction *Vec);
  void eraseScalar(sandboxir::Instruction *Scalar);
  void registerVector(sandboxir::Value *Vec,
                      const DmpVector<sandboxir::Value *> &Scalars);
  sandboxir::Value *getVectorForScalar(sandboxir::Value *Scalar) const;
  /// This can handle "scalar" elements that are actually vectors.
  /// Like in `{<2 x double>, double}` it would return 2 for the lane of the
  /// second element, not 1.
  unsigned getScalarLane(sandboxir::Value *Vec,
                         sandboxir::Value *Scalar) const;
  /// \Returns an empty bundle if not found.
  const DmpVector<sandboxir::Value *> &
  getScalarsFor(sandboxir::Value *Vec) const;
  /// \Returns: (i) the set of vector values that correspond to \p Bndl and (ii)
  /// whether there are scalar values that do not correspond to any vector
  /// value.
  std::pair<DenseSet<sandboxir::Value *>, bool> getVectorsThatCombinedScalars(
      const DmpVector<sandboxir::Value *> &Bndl) const;
  std::optional<ShuffleMask>
  getShuffleMask(sandboxir::Value *Vec,
                 const DmpVector<sandboxir::Value *> &OtherBndl) const;
  void setRegion(sandboxir::Region &Region) { Rgn = &Region; }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

} // namespace sandboxir
} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_INSTRUCTIONMAPS_H
