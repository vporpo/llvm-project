//===- DmpVector.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/DmpVector.h"
#include "llvm/SandboxIR/SandboxIR.h"

using namespace llvm;

void DmpVector<sandboxir::Value *>::init(
    const DmpVector<Value *> &Vec, const sandboxir::BasicBlock &SBBB) {
  reserve(Vec.size());
  for (Value *V : Vec) {
    sandboxir::Value *SBV = SBBB.getContext().getValue(V);
    assert(SBV != nullptr && "Can't get SBValue for V!");
    push_back(SBV);
  }
}

DmpVector<Value *> DmpVector<sandboxir::Value *>::getLLVMValueVector() const {
  DmpVector<Value *> Vec(size());
  for (auto *N : *this)
    Vec.push_back(sandboxir::ValueAttorney::getValue(N));
  return Vec;
}

void DmpVector<sandboxir::Instruction *>::init(
    const DmpVector<Value *> &Vec, const sandboxir::BasicBlock &SBBB) {
  reserve(Vec.size());
  for (Instruction *I : Vec.instrRange()) {
    auto *SBV = SBBB.getContext().getValue(I);
    assert(SBV != nullptr && "Can't get SBValue for V!");
    assert(isa<sandboxir::Instruction>(SBV) &&
           "Not a sandboxir::Instruction!");
    push_back(cast<sandboxir::Instruction>(SBV));
  }
}

DmpVector<Value *>
DmpVector<Value *>::create(const DmpVector<sandboxir::Value *> &SBVec) {
  DmpVector<Value *> Vec;
  Vec.reserve(SBVec.size());
  for (const auto *SBV : SBVec)
    Vec.push_back(sandboxir::ValueAttorney::getValue(SBV));
  return Vec;
}
