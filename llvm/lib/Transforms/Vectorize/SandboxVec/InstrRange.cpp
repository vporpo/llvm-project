//===- InstrRange.cpp
//---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/InstrRange.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Utils.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"

using namespace llvm;

template <typename DerefType, typename InstrRangeType>
InstrRangeIterator<DerefType, InstrRangeType> &
InstrRangeIterator<DerefType, InstrRangeType>::operator++() {
  assert(I != nullptr && "already at end()!");
  I = I->getNextNode();
  return *this;
}
template <typename DerefType, typename InstrRangeType>
InstrRangeIterator<DerefType, InstrRangeType>
InstrRangeIterator<DerefType, InstrRangeType>::operator++(int) {
  auto ItCopy = *this;
  ++*this;
  return ItCopy;
}
template <typename DerefType, typename InstrRangeType>
InstrRangeIterator<DerefType, InstrRangeType> &
InstrRangeIterator<DerefType, InstrRangeType>::operator--() {
  // `I` is nullptr for end() when ToI is the BB terminator.
  I = I != nullptr ? I->getPrevNode() : R.ToI;
  return *this;
}

template <typename DerefType, typename InstrRangeType>
InstrRangeIterator<DerefType, InstrRangeType>
InstrRangeIterator<DerefType, InstrRangeType>::operator--(int) {
  auto ItCopy = *this;
  --*this;
  return ItCopy;
}

InstrRange::InstrRange(SBInstruction *I1, SBInstruction *I2) {
  assert(!I1->isDbgInfo() && !I2->isDbgInfo() &&
         "No debug instructions allowed!");
  if (I1 != I2 && I2->comesBefore(I1))
    std::swap(I1, I2);
  FromI = I1;
  ToI = I2;
  assert((FromI == ToI || FromI->comesBefore(ToI)) &&
         "Expected FromI before ToI or equal");
}

// Explicit instantiation.
namespace llvm {
template class InstrRangeIterator<SBInstruction &, InstrRange>;
template class InstrRangeIterator<SBInstruction const &, InstrRange const>;
} // namespace llvm

template <typename RangeT> void InstrRange::init(RangeT Instrs) {
  // Find the first and last instr among `Instrs`.
  SBInstruction *TopI = cast<SBInstruction>(*Instrs.begin());
  SBInstruction *BotI = TopI;
  for (SBValue *SBV : drop_begin(Instrs)) {
    auto *I = cast<SBInstruction>(SBV);
    if (I->comesBefore(TopI))
      TopI = I;
    if (BotI->comesBefore(I))
      BotI = I;
  }
  FromI = TopI;
  ToI = BotI;
  assert((FromI == ToI || FromI->comesBefore(ToI)) &&
         "Expected FromI before ToI!");
}

// Explicit instantiations.
template InstrRange::InstrRange(SBValBundle);
template InstrRange::InstrRange(ArrayRef<SBValue *>);
template InstrRange::InstrRange(ArrayRef<SBInstruction *>);
template void InstrRange::init(ArrayRef<SBValue *>);
template void InstrRange::init(ArrayRef<SBInstruction *>);

InstrRange::InstrRange(const SBValBundle &SBVals) { init(SBVals); }

InstrRange InstrRange::getUnionSingleSpan(const InstrRange &Other) const {
  if (empty())
    return Other;
  if (Other.empty())
    return *this;
  auto *NewFromI = FromI->comesBefore(Other.FromI) ? FromI : Other.FromI;
  auto *NewToI = ToI->comesBefore(Other.ToI) ? Other.ToI : ToI;
  return {NewFromI, NewToI};
}

InstrRange InstrRange::getIntersection(const InstrRange &Other) const {
  if (empty())
    return *this; // empty
  if (Other.empty())
    return InstrRange();
  // 1. No overlap
  // A---B      this
  //       C--D Other
  if (ToI->comesBefore(Other.FromI) || Other.ToI->comesBefore(FromI))
    return InstrRange();
  // 2. Overlap.
  // A---B   this
  //   C--D  Other
  auto NewFromI = FromI->comesBefore(Other.FromI) ? Other.FromI : FromI;
  auto NewToI = ToI->comesBefore(Other.ToI) ? ToI : Other.ToI;
  return InstrRange(NewFromI, NewToI);
}

SmallVector<InstrRange, 2>
InstrRange::operator-(const InstrRange &Other) const {
  if (disjoint(Other))
    return {*this};
  if (Other.empty())
    return {*this};
  if (*this == Other)
    return {InstrRange()};
  InstrRange Intersection = getIntersection(Other);
  SmallVector<InstrRange, 2> Result;
  // Part 1, skip if empty.
  if (FromI != Intersection.FromI)
    Result.emplace_back(FromI, Intersection.FromI->getPrevNode());
  // Part 2, skip if empty.
  if (Intersection.ToI != ToI)
    Result.emplace_back(Intersection.ToI->getNextNode(), ToI);
  return Result;
}

InstrRange InstrRange::getSingleDifference(const InstrRange &Other) const {
  auto Diffs = *this - Other;
  if (Diffs.empty())
    return {};
  assert(Diffs.size() == 1 &&
         "Expected up to one region in the difference operation!");
  return Diffs[0];
}

bool InstrRange::contains(const SBBBIterator &It) const {
  assert(!empty() && "Expected a non-empty region!");
  SBBasicBlock *BB = from()->getParent();
  if (It == BB->end())
    return to() == &BB->back();
  SBInstruction *I = &*It;
  return contains(I) || I == to()->getNextNode();
}

bool InstrRange::contains(SBInstruction *I) const {
  if (empty())
    return false;
  return (FromI == I || FromI->comesBefore(I)) &&
         (I == ToI || I->comesBefore(ToI));
}

void InstrRange::extend(SBInstruction *I) {
  if (empty()) {
    FromI = I;
    ToI = I;
    return;
  }
  if (contains(I))
    return;
  if (I->comesBefore(FromI))
    FromI = I;
  if (ToI->comesBefore(I))
    ToI = I;
}

bool InstrRange::empty() const {
  assert(((FromI == nullptr && ToI == nullptr) ||
          (FromI != nullptr && ToI != nullptr)) &&
         "Either none or both should be null");
  return FromI == nullptr;
}

bool InstrRange::contains(const InstrRange &Other) const {
  if (Other.empty())
    return true;
  return (FromI == Other.FromI || FromI->comesBefore(Other.FromI)) &&
         (ToI == Other.ToI || Other.ToI->comesBefore(ToI));
}

bool InstrRange::disjoint(const InstrRange &Other) const {
  if (Other.empty())
    return true;
  if (empty())
    return true;
  return Other.ToI->comesBefore(FromI) || ToI->comesBefore(Other.FromI);
}

InstrRange::iterator InstrRange::end() {
  return iterator(ToI != nullptr ? ToI->getNextNode() : nullptr, *this);
}

InstrRange::const_iterator InstrRange::end() const {
  return const_iterator(ToI != nullptr ? ToI->getNextNode() : nullptr, *this);
}

void InstrRange::erase(SBInstruction *I, bool CheckContained) {
  assert((!CheckContained || contains(I)) && "Instruction not in region!");
  if (empty())
    return;
  if (FromI == ToI) {
    // Corner case: if the region contains only one node
    if (I == FromI) {
      FromI = nullptr;
      ToI = nullptr;
    }
    return;
  }
  if (I == FromI)
    FromI = FromI->getNextNode();
  if (I == ToI)
    ToI = ToI->getPrevNode();
  assert((FromI == ToI || FromI->comesBefore(ToI)) && "Malformed region!");
}

void InstrRange::notifyMoveInstr(SBInstruction *I,
                                 const SBBBIterator &BeforeIt,
                                 SBBasicBlock *BB) {
  assert(!empty() && "Expect a non-empty region!");
  assert(contains(I) && contains(BeforeIt) &&
         "This function can only handle intra-region instruction movement, "
         "which is what we expect from the scheduler.");
  // `I` doesn't move so early return.
  if (std::next(I->getIterator()) == BeforeIt)
    return;

  // If `I` is at the region's boundaries we need to move the boundaries to
  // the next/prev bundle accordingly.
  if (I == FromI) {
    assert(I != ToI && "This is equivalent to moving to itself, should have "
                       "early returned earlier!");
    FromI = I->getNextNode();
  } else if (I == ToI) {
    assert(I != FromI && "This is equivalent to moving to itself, should have "
                         "early returned earlier!");
    ToI = I->getPrevNode();
  }
  // If the destination is before/after the boundaries,
  if (BeforeIt == FromI->getIterator()) {
    // Destination is just above FromI, so update FromI.
    assert(BeforeIt != std::next(ToI->getIterator()) &&
           "Should have been handled earlier!");
    FromI = I;
    return;
  }
  if (BeforeIt == std::next(ToI->getIterator())) {
    // Destination is just below ToI, so update ToI.
    ToI = I;
    return;
  }
}

void InstrRange::clear() {
  FromI = nullptr;
  ToI = nullptr;
}

#ifndef NDEBUG
void InstrRange::dump(raw_ostream &OS) const {
  if (empty()) {
    OS << "Empty\n";
    return;
  }
  OS << "FromI:";
  if (FromI != nullptr)
    OS << *FromI;
  else
    OS << "NULL";
  OS << "\n";

  OS << "ToI:  ";
  if (ToI != nullptr)
    OS << *ToI;
  else
    OS << "NULL";
  OS << "\n";

  if (FromI != nullptr && ToI != nullptr) {
    if (FromI != ToI && !FromI->comesBefore(ToI)) {
      OS << "ERROR: FromI does not come before ToI !\n";
      return;
    }
    for (SBInstruction *I = FromI, *IE = ToI->getNextNode(); I != IE;
         I = I->getNextNode())
      OS << *I << "\n";
  }
}
LLVM_DUMP_METHOD void InstrRange::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

#endif
