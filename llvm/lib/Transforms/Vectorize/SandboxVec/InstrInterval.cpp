//===- InstrInterval.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/InstrInterval.h"
#include "llvm/SandboxIR/SandboxIR.h"

using namespace llvm;

template <typename DerefType, typename InstrIntervalType>
sandboxir::InstrIntervalIterator<DerefType, InstrIntervalType> &
sandboxir::InstrIntervalIterator<DerefType, InstrIntervalType>::operator++() {
  assert(I != nullptr && "already at end()!");
  I = I->getNextNode();
  return *this;
}
template <typename DerefType, typename InstrIntervalType>
sandboxir::InstrIntervalIterator<DerefType, InstrIntervalType>
sandboxir::InstrIntervalIterator<DerefType, InstrIntervalType>::operator++(
    int) {
  auto ItCopy = *this;
  ++*this;
  return ItCopy;
}
template <typename DerefType, typename InstrIntervalType>
sandboxir::InstrIntervalIterator<DerefType, InstrIntervalType> &
sandboxir::InstrIntervalIterator<DerefType, InstrIntervalType>::operator--() {
  // `I` is nullptr for end() when ToI is the BB terminator.
  I = I != nullptr ? I->getPrevNode() : R.ToI;
  return *this;
}

template <typename DerefType, typename InstrIntervalType>
sandboxir::InstrIntervalIterator<DerefType, InstrIntervalType>
sandboxir::InstrIntervalIterator<DerefType, InstrIntervalType>::operator--(
    int) {
  auto ItCopy = *this;
  --*this;
  return ItCopy;
}

sandboxir::InstrInterval::InstrInterval(sandboxir::Instruction *I1,
                                        sandboxir::Instruction *I2) {
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
namespace sandboxir {
template class sandboxir::InstrIntervalIterator<sandboxir::Instruction &,
                                                sandboxir::InstrInterval>;
template class sandboxir::InstrIntervalIterator<
    sandboxir::Instruction const &, sandboxir::InstrInterval const>;
} // namespace sandboxir
} // namespace llvm

template <typename IntervalT>
void sandboxir::InstrInterval::init(IntervalT Instrs) {
  // Find the first and last instr among `Instrs`.
  sandboxir::Instruction *TopI =
      cast<sandboxir::Instruction>(*Instrs.begin());
  sandboxir::Instruction *BotI = TopI;
  for (sandboxir::Value *SBV : drop_begin(Instrs)) {
    auto *I = cast<sandboxir::Instruction>(SBV);
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
template sandboxir::InstrInterval::InstrInterval(
    DmpVector<sandboxir::Value *>);
template sandboxir::InstrInterval::InstrInterval(
    ArrayRef<sandboxir::Value *>);
template sandboxir::InstrInterval::InstrInterval(
    ArrayRef<sandboxir::Instruction *>);
template void sandboxir::InstrInterval::init(ArrayRef<sandboxir::Value *>);
template void
    sandboxir::InstrInterval::init(ArrayRef<sandboxir::Instruction *>);

sandboxir::InstrInterval::InstrInterval(
    const DmpVector<sandboxir::Value *> &SBVals) {
  init(SBVals);
}

sandboxir::InstrInterval
sandboxir::InstrInterval::getUnionSingleSpan(const InstrInterval &Other) const {
  if (empty())
    return Other;
  if (Other.empty())
    return *this;
  auto *NewFromI = FromI->comesBefore(Other.FromI) ? FromI : Other.FromI;
  auto *NewToI = ToI->comesBefore(Other.ToI) ? Other.ToI : ToI;
  return {NewFromI, NewToI};
}

sandboxir::InstrInterval
sandboxir::InstrInterval::getIntersection(const InstrInterval &Other) const {
  if (empty())
    return *this; // empty
  if (Other.empty())
    return InstrInterval();
  // 1. No overlap
  // A---B      this
  //       C--D Other
  if (ToI->comesBefore(Other.FromI) || Other.ToI->comesBefore(FromI))
    return InstrInterval();
  // 2. Overlap.
  // A---B   this
  //   C--D  Other
  auto NewFromI = FromI->comesBefore(Other.FromI) ? Other.FromI : FromI;
  auto NewToI = ToI->comesBefore(Other.ToI) ? ToI : Other.ToI;
  return InstrInterval(NewFromI, NewToI);
}

SmallVector<sandboxir::InstrInterval, 2> sandboxir::InstrInterval::operator-(
    const sandboxir::InstrInterval &Other) const {
  if (disjoint(Other))
    return {*this};
  if (Other.empty())
    return {*this};
  if (*this == Other)
    return {InstrInterval()};
  InstrInterval Intersection = getIntersection(Other);
  SmallVector<InstrInterval, 2> Result;
  // Part 1, skip if empty.
  if (FromI != Intersection.FromI)
    Result.emplace_back(FromI, Intersection.FromI->getPrevNode());
  // Part 2, skip if empty.
  if (Intersection.ToI != ToI)
    Result.emplace_back(Intersection.ToI->getNextNode(), ToI);
  return Result;
}

sandboxir::InstrInterval sandboxir::InstrInterval::getSingleDifference(
    const sandboxir::InstrInterval &Other) const {
  auto Diffs = *this - Other;
  if (Diffs.empty())
    return {};
  assert(Diffs.size() == 1 &&
         "Expected up to one interval in the difference operation!");
  return Diffs[0];
}

bool sandboxir::InstrInterval::contains(
    const sandboxir::BBIterator &It) const {
  assert(!empty() && "Expected a non-empty interval!");
  sandboxir::BasicBlock *BB = from()->getParent();
  if (It == BB->end())
    return to() == &*BB->rbegin();
  sandboxir::Instruction *I = &*It;
  return contains(I) || I == to()->getNextNode();
}

bool sandboxir::InstrInterval::contains(sandboxir::Instruction *I) const {
  if (empty())
    return false;
  return (FromI == I || FromI->comesBefore(I)) &&
         (I == ToI || I->comesBefore(ToI));
}

void sandboxir::InstrInterval::extend(sandboxir::Instruction *I) {
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

bool sandboxir::InstrInterval::empty() const {
  assert(((FromI == nullptr && ToI == nullptr) ||
          (FromI != nullptr && ToI != nullptr)) &&
         "Either none or both should be null");
  return FromI == nullptr;
}

bool sandboxir::InstrInterval::contains(
    const sandboxir::InstrInterval &Other) const {
  if (Other.empty())
    return true;
  return (FromI == Other.FromI || FromI->comesBefore(Other.FromI)) &&
         (ToI == Other.ToI || Other.ToI->comesBefore(ToI));
}

bool sandboxir::InstrInterval::disjoint(
    const sandboxir::InstrInterval &Other) const {
  if (Other.empty())
    return true;
  if (empty())
    return true;
  return Other.ToI->comesBefore(FromI) || ToI->comesBefore(Other.FromI);
}

sandboxir::InstrInterval::iterator sandboxir::InstrInterval::end() {
  return iterator(ToI != nullptr ? ToI->getNextNode() : nullptr, *this);
}

sandboxir::InstrInterval::const_iterator sandboxir::InstrInterval::end() const {
  return const_iterator(ToI != nullptr ? ToI->getNextNode() : nullptr, *this);
}

void sandboxir::InstrInterval::erase(sandboxir::Instruction *I,
                                     bool CheckContained) {
  assert((!CheckContained || contains(I)) && "Instruction not in interval!");
  if (empty())
    return;
  if (FromI == ToI) {
    // Corner case: if the interval contains only one node
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
  assert((FromI == ToI || FromI->comesBefore(ToI)) && "Malformed interval!");
}

void sandboxir::InstrInterval::notifyMoveInstr(
    sandboxir::Instruction *I, const sandboxir::BBIterator &BeforeIt,
    sandboxir::BasicBlock *BB) {
  assert(contains(I) && contains(BeforeIt) &&
         "This function can only handle intra-interval instruction movement, "
         "which is what we expect from the scheduler.");
  // `I` doesn't move, so early return.
  if (std::next(I->getIterator()) == BeforeIt)
    return;

  // If `I` is at the interval's boundaries we need to move the boundaries to
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

void sandboxir::InstrInterval::clear() {
  FromI = nullptr;
  ToI = nullptr;
}

#ifndef NDEBUG
void sandboxir::InstrInterval::dump(raw_ostream &OS) const {
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
    for (sandboxir::Instruction *I = FromI, *IE = ToI->getNextNode(); I != IE;
         I = I->getNextNode())
      OS << *I << "\n";
  }
}
LLVM_DUMP_METHOD void sandboxir::InstrInterval::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

#endif
