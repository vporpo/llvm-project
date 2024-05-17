//===- Bundle.cpp - Bunlde helper classes ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Bundle.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"

using namespace llvm;

SBValBundle::SBValBundle(const ValueBundle &Bndl, const SBBasicBlock &SBBB) {
  Vals.reserve(Bndl.size());
  for (Value *V : Bndl) {
    SBValue *SBV = SBBB.getContext().getSBValue(V);
    assert(SBV != nullptr && "Can't get SBValue for V!");
    Vals.push_back(SBV);
  }
}

SBValBundle SBValBundle::getOperandBundle(unsigned OpIdx) const {
  SBValBundle OpBndl;
  OpBndl.reserve(size());
  for (auto *SBV : *this) {
    auto *SBI = cast<SBInstruction>(SBV);
    assert(OpIdx < SBI->getNumOperands() && "Out of bounds!");
    OpBndl.push_back(SBI->getOperand(OpIdx));
  }
  return OpBndl;
}

SmallVector<SBValBundle, 2> SBValBundle::getOperandBundles() const {
  SmallVector<SBValBundle, 2> OpBndls;
#ifndef NDEBUG
  unsigned NumOps = cast<SBInstruction>(Vals[0])->getNumOperands();
  assert(all_of(drop_begin(Vals),
                [NumOps](auto *V) {
                  return cast<SBInstruction>(V)->getNumOperands() == NumOps;
                }) &&
         "Expected same number of operands!");
#endif
  for (unsigned OpIdx :
       seq<unsigned>(cast<SBInstruction>(Vals[0])->getNumOperands()))
    OpBndls.push_back(getOperandBundle(OpIdx));
  return OpBndls;
}

ValueBundle SBValBundle::getValueBundle() const {
  ValueBundle Bndl(size());
  for (auto *N : *this)
    Bndl.push_back(ValueAttorney::getValue(N));
  return Bndl;
}

SBInstrBundle::SBInstrBundle(const ValueBundle &Bndl,
                                 const SBBasicBlock &SBBB) {
  Vals.reserve(Bndl.size());
  for (Instruction *I : Bndl.instrRange()) {
    auto *SBV = SBBB.getContext().getSBValue(I);
    assert(SBV != nullptr && "Can't get SBValue for V!");
    assert(isa<SBInstruction>(SBV) && "Not a SBInstruction!");
    Vals.push_back(cast<SBInstruction>(SBV));
  }
}

SBValBundle SBInstrBundle::getOperandBundle(unsigned OpIdx) const {
  SBValBundle OpBndl;
  OpBndl.reserve(size());
  for (auto *SBI : *this) {
    assert(OpIdx < SBI->getNumOperands() && "Out of bounds!");
    OpBndl.push_back(SBI->getOperand(OpIdx));
  }
  return OpBndl;
}

ValueBundle ValueBundle::create(const SBValBundle &SBBndl) {
  ValueBundle Bndl;
  Bndl.reserve(SBBndl.size());
  for (const auto *SBV : SBBndl)
    Bndl.push_back(ValueAttorney::getValue(SBV));
  return Bndl;
}
