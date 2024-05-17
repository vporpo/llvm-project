//===- InstrRange.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_INSTRRANGE_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_INSTRRANGE_H

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"

namespace llvm {

class SBValBundle;
class SBInstruction;
class SBBBIterator;
class SBBasicBlock;
class SBContext;

/// A simple iterator for iterating the region.
template <typename DerefType, typename InstrRangeType>
class InstrRangeIterator {
  SBInstruction *I;
  InstrRangeType &R;

public:
  using difference_type = std::ptrdiff_t;
  using value_type = SBInstruction;
  using pointer = value_type *;
  using reference = SBInstruction &;
  using iterator_category = std::bidirectional_iterator_tag;

  InstrRangeIterator(SBInstruction *I, InstrRangeType &R) : I(I), R(R) {}
  bool operator==(const InstrRangeIterator &Other) const {
    assert(&R == &Other.R && "Iterators belong to different regions!");
    return Other.I == I;
  }
  bool operator!=(const InstrRangeIterator &Other) const {
    return !(*this == Other);
  }
  InstrRangeIterator &operator++();
  InstrRangeIterator operator++(int);
  InstrRangeIterator &operator--();
  InstrRangeIterator operator--(int);
  template <typename T =
                std::enable_if<std::is_same<DerefType, Instruction *&>::value>>
  SBInstruction &operator*() {
    return *I;
  }
  DerefType operator*() const { return *I; }
};

/// A range of instructions.
class InstrRange {
  SBInstruction *FromI;
  SBInstruction *ToI;
  template <typename DerefType, typename InstrRangeType>
  friend class InstrRangeIterator;

  template <typename RangeT> void init(RangeT Instrs);

public:
  InstrRange() : FromI(nullptr), ToI(nullptr) {}
  /// Create a region spanning I1 to I2. Instruction order does not matter.
  InstrRange(SBInstruction *I1, SBInstruction *I2);
  InstrRange(const SBValBundle &SBVals);
  /// Initialize region with \p Instrs. Instruciton order does not matter.
  template <typename RangeT> InstrRange(RangeT Instrs) { init(Instrs); }
  InstrRange(std::initializer_list<SBInstruction *> Instrs)
      : InstrRange(ArrayRef<SBInstruction *>{Instrs.begin(), Instrs.end()}) {}
  InstrRange(SBInstruction *I) : InstrRange({I}) {}
  /// \Returns the union of this and \p Other.
  /// WARNING: This is not the usual group union: this always returns a single
  /// region that spans both this and \p Other, even if they are disjoint!
  // For example:
  // |---|        this
  //        |---| Other
  // |----------| this->getUnion(Other)
  InstrRange getUnionSingleSpan(const InstrRange &Other) const;
  /// \Returns a region with instructions that can be found in both this and
  /// \ Other.
  // Example:
  // |----|   this
  //    |---| Other
  //    |-|   this->getIntersection(Other)
  InstrRange getIntersection(const InstrRange &Other) const;
  /// The diffence operation.
  // Example:
  // |--------| this
  //    |-|     Other
  // |-|   |--| this - Other
  SmallVector<InstrRange, 2> operator-(const InstrRange &Other) const;
  /// Just like operator- but we expect a single region in the result.
  InstrRange getSingleDifference(const InstrRange &Other) const;
  SBInstruction *from() const { return FromI; }
  void setFrom(SBInstruction *I) { FromI = I; }
  SBInstruction *to() const { return ToI; }
  void setTo(SBInstruction *I) { ToI = I; }
  /// \Returns true if \p It points to anywher in the region or right after the
  /// bottom or at the top.
  bool contains(const SBBBIterator &It) const;
  bool contains(SBInstruction *I) const;

  /// Extend region to include \p I.
  void extend(SBInstruction *I);
  bool empty() const;
  bool operator==(const InstrRange &Other) const {
    return FromI == Other.FromI && ToI == Other.ToI;
  }
  bool operator!=(const InstrRange &Other) const { return !(*this == Other); }
  /// \Returns true if this fully contains (inclusive) \p Other.
  bool contains(const InstrRange &Other) const;
  /// \Returns true if this and \p Other have no instructions in common.
  bool disjoint(const InstrRange &Other) const;
  void clear();

  using iterator = InstrRangeIterator<SBInstruction &, InstrRange>;
  using const_iterator =
      InstrRangeIterator<const SBInstruction &, const InstrRange>;
  // using const_iterator = Iterator<const Instruction *>;
  iterator begin() { return iterator(FromI, *this); }
  iterator end();
  const_iterator begin() const { return const_iterator(FromI, *this); }
  const_iterator end() const;
  /// Update InstrRange before \p I gets erased from its parent.
  void erase(SBInstruction *I, bool CheckContained = true);

  /// \p I is about to be moved at \p BeforeIt. Update the region accordingly.
  void notifyMoveInstr(SBInstruction *I, const SBBBIterator &BeforeIt,
                       SBBasicBlock *BB);
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  friend raw_ostream &operator<<(raw_ostream &OS, const InstrRange &R) {
    R.dump(OS);
    return OS;
  }
#endif
};
} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_INSTRRANGE_H
