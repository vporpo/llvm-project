//===- InstrInterval.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_INSTRINTERVAL_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_INSTRINTERVAL_H

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/SandboxIR/DmpVector.h"

namespace llvm {

namespace sandboxir {
class Instruction;
class BBIterator;
class BasicBlock;
class Context;

/// A simple iterator for iterating the region.
template <typename DerefType, typename InstrIntervalType>
class InstrIntervalIterator {
  sandboxir::Instruction *I;
  InstrIntervalType &R;

public:
  using difference_type = std::ptrdiff_t;
  using value_type = sandboxir::Instruction;
  using pointer = value_type *;
  using reference = sandboxir::Instruction &;
  using iterator_category = std::bidirectional_iterator_tag;

  InstrIntervalIterator(sandboxir::Instruction *I, InstrIntervalType &R)
      : I(I), R(R) {}
  bool operator==(const InstrIntervalIterator &Other) const {
    assert(&R == &Other.R && "Iterators belong to different regions!");
    return Other.I == I;
  }
  bool operator!=(const InstrIntervalIterator &Other) const {
    return !(*this == Other);
  }
  InstrIntervalIterator &operator++();
  InstrIntervalIterator operator++(int);
  InstrIntervalIterator &operator--();
  InstrIntervalIterator operator--(int);
  template <typename T =
                std::enable_if<std::is_same<DerefType, Instruction *&>::value>>
  sandboxir::Instruction &operator*() {
    return *I;
  }
  DerefType operator*() const { return *I; }
};

/// An instruction interval.
class InstrInterval {
  sandboxir::Instruction *FromI;
  sandboxir::Instruction *ToI;
  template <typename DerefType, typename InstrIntervalType>
  friend class InstrIntervalIterator;

  template <typename IntervalT> void init(IntervalT Instrs);

public:
  InstrInterval() : FromI(nullptr), ToI(nullptr) {}
  /// Create a region spanning I1 to I2. Instruction order does not matter.
  InstrInterval(sandboxir::Instruction *I1, sandboxir::Instruction *I2);
  InstrInterval(const DmpVector<sandboxir::Value *> &SBVals);
  /// Initialize region with \p Instrs. Instruciton order does not matter.
  template <typename IntervalT> InstrInterval(IntervalT Instrs) {
    init(Instrs);
  }
  InstrInterval(std::initializer_list<sandboxir::Instruction *> Instrs)
      : InstrInterval(ArrayRef<sandboxir::Instruction *>{Instrs.begin(),
                                                           Instrs.end()}) {}
  InstrInterval(sandboxir::Instruction *I) : InstrInterval({I}) {}
  /// \Returns the union of this and \p Other.
  /// WARNING: This is not the usual group union: this always returns a single
  /// region that spans both this and \p Other, even if they are disjoint!
  // For example:
  // |---|        this
  //        |---| Other
  // |----------| this->getUnion(Other)
  InstrInterval getUnionSingleSpan(const InstrInterval &Other) const;
  /// \Returns a region with instructions that can be found in both this and
  /// \ Other.
  // Example:
  // |----|   this
  //    |---| Other
  //    |-|   this->getIntersection(Other)
  InstrInterval getIntersection(const InstrInterval &Other) const;
  /// The diffence operation.
  // Example:
  // |--------| this
  //    |-|     Other
  // |-|   |--| this - Other
  SmallVector<InstrInterval, 2> operator-(const InstrInterval &Other) const;
  /// Just like operator- but we expect a single region in the result.
  InstrInterval getSingleDifference(const InstrInterval &Other) const;
  sandboxir::Instruction *from() const { return FromI; }
  void setFrom(sandboxir::Instruction *I) { FromI = I; }
  sandboxir::Instruction *to() const { return ToI; }
  void setTo(sandboxir::Instruction *I) { ToI = I; }
  /// \Returns true if \p It points to anywher in the region or right after the
  /// bottom or at the top.
  bool contains(const sandboxir::BBIterator &It) const;
  bool contains(sandboxir::Instruction *I) const;

  /// Extend region to include \p I.
  void extend(sandboxir::Instruction *I);
  bool empty() const;
  bool operator==(const InstrInterval &Other) const {
    return FromI == Other.FromI && ToI == Other.ToI;
  }
  bool operator!=(const InstrInterval &Other) const {
    return !(*this == Other);
  }
  /// \Returns true if this fully contains (inclusive) \p Other.
  bool contains(const InstrInterval &Other) const;
  /// \Returns true if this and \p Other have no instructions in common.
  bool disjoint(const InstrInterval &Other) const;
  void clear();

  using iterator =
      InstrIntervalIterator<sandboxir::Instruction &, InstrInterval>;
  using const_iterator = InstrIntervalIterator<const sandboxir::Instruction &,
                                               const InstrInterval>;
  // using const_iterator = Iterator<const Instruction *>;
  iterator begin() { return iterator(FromI, *this); }
  iterator end();
  const_iterator begin() const { return const_iterator(FromI, *this); }
  const_iterator end() const;
  /// Update InstrInterval before \p I gets erased from its parent.
  void erase(sandboxir::Instruction *I, bool CheckContained = true);

  /// \p I is about to be moved at \p BeforeIt. Update the region accordingly.
  void notifyMoveInstr(sandboxir::Instruction *I,
                       const sandboxir::BBIterator &BeforeIt,
                       sandboxir::BasicBlock *BB);
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  friend raw_ostream &operator<<(raw_ostream &OS, const InstrInterval &R) {
    R.dump(OS);
    return OS;
  }
#endif
};
} // namespace sandboxir
} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_INSTRINTERVAL_H
