//===- VecUtils.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Helpers for the Sandbox Vectorizer
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_VECUTILS_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_VECUTILS_H

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/SandboxIR/DmpVector.h"

namespace llvm {
namespace sandboxir {

class Instruction;
class BBIterator;
class Value;

class VecUtils {
public:
  /// \Returns true if \p I1 is accessing a prior memory location than \p I2.
  template <typename LoadOrStoreT>
  static bool comesBeforeInMem(LoadOrStoreT *I1, LoadOrStoreT *I2,
                               ScalarEvolution &SE, const DataLayout &DL);
  /// \Returns true if \p I1 and \p I2 are load/stores accessing consecutive
  /// memory addresses.
  template <typename LoadOrStoreT>
  static bool areConsecutive(LoadOrStoreT *I1, LoadOrStoreT *I2,
                             ScalarEvolution &SE, const DataLayout &DL);

  template <typename LoadOrStoreT>
  static bool areConsecutive(const DmpVector<sandboxir::Value *> &SBBndl,
                             ScalarEvolution &SE, const DataLayout &DL);

  /// If \p SBI is a load it returns its type. If a store it returns its value
  /// operand type.
  template <typename LoadOrStoreT>
  static Type *getLoadStoreType(LoadOrStoreT *SBI);
  /// \Returns the number of elements in \p Ty, that is the number of lanes if
  /// vector or 1 if scalar.
  static int getNumElements(Type *Ty) {
    return Ty->isVectorTy() ? cast<FixedVectorType>(Ty)->getNumElements() : 1;
  }
  /// Returns \p Ty if scalar or its element type if vector.
  static Type *getElementType(Type *Ty) {
    assert((!isa<VectorType>(Ty) || isa<FixedVectorType>(Ty)) &&
           "Expected only Fixed vector types!");
    return Ty->isVectorTy() ? cast<FixedVectorType>(Ty)->getElementType() : Ty;
  }
  /// \Returns the scalar type shared among Nodes in Bndl. \Returns nullptr if
  /// they don't share a common scalar type.
  static Type *getCommonScalarType(const DmpVector<sandboxir::Value *> &Bndl);
  /// Same as getCommonScalarType() but expects that there is a common scalar
  /// type. If not it will crash in a DEBUG build.
  static Type *
  getCommonScalarTypeFast(const DmpVector<sandboxir::Value *> &Bndl);

  /// Unlike other instructions, getting the value of a store requires a
  /// function call.
  static llvm::Value *getExpectedValue(llvm::Instruction *I);
  static sandboxir::Value *
  getExpectedValue(const sandboxir::Instruction *I);

  /// A store's type is void, which is rather useless. This function does the
  /// right thing and returns the type of the stored value.
  static Type *getExpectedType(llvm::Value *V);
  static Type *getExpectedType(const sandboxir::Value *V);

  template <typename ValT> static bool isVector(ValT *V) {
    return isa<FixedVectorType>(sandboxir::VecUtils::getExpectedType(V));
  }

  /// \Returns the number of vector lanes of \p Ty or 1 if not a vector.
  /// NOTE: It crashes if \p V is a scalable vector.
  static int getNumLanes(Type *Ty) {
    assert(!isa<ScalableVectorType>(Ty) && "Expect fixed vector");
    if (!isa<FixedVectorType>(Ty))
      return 1;
    return cast<FixedVectorType>(Ty)->getNumElements();
  }

  /// \Returns the expected vector lanes of \p V or 1 if not a vector.
  /// NOTE: It crashes if \p V is a scalable vector.
  static int getNumLanes(llvm::Value *V) {
    return sandboxir::VecUtils::getNumLanes(getExpectedType(V));
  }
  static unsigned getNumLanes(const sandboxir::Value *SBV);
  /// This works even if Bndl contains Nodes with vector type.
  static unsigned getNumLanes(const DmpVector<sandboxir::Value *> &Bndl);

  static unsigned getNumBits(sandboxir::Value *SBV, const DataLayout &DL);
  static unsigned getNumBits(const DmpVector<sandboxir::Instruction *> &Bndl,
                             const DataLayout &DL);
  static unsigned getNumBits(const DmpVector<sandboxir::Value *> &Bndl,
                             const DataLayout &DL);

  // Returns the next power of 2.
  static unsigned getCeilPowerOf2(unsigned Num) {
    if (Num == 0)
      return Num;
    Num--;
    for (int ShiftBy = 1; ShiftBy < 32; ShiftBy <<= 1)
      Num |= Num >> ShiftBy;
    return Num + 1;
  }
  static unsigned getFloorPowerOf2(unsigned Num) {
    if (Num == 0)
      return Num;
    unsigned Mask = Num;
    Mask >>= 1;
    for (int ShiftBy = 1; ShiftBy < 32; ShiftBy <<= 1)
      Mask |= Mask >> ShiftBy;
    return Num & ~Mask;
  }
  static bool isPowerOf2(unsigned Num) { return getFloorPowerOf2(Num) == Num; }

  /// \Returns the a type that is \p NumElts times wider than \p ElemTy.
  /// It works for both scalar and vector \p ElemTy.
  static Type *getWideType(Type *ElemTy, uint32_t NumElts);

  static bool areInSameBB(const DmpVector<llvm::Value *> &Instrs);
  static bool areInSameBB(const DmpVector<sandboxir::Value *> &SBInstrs);

  /// \Returns the iterator right after the lowest instruction in \p Bndl. If \p
  /// Bndl contains only non-instructions, or if the instructions in BB are at
  /// different blocks, other than \p BB, it returns the beginning of \p BB.
  static llvm::BasicBlock::iterator
  getInsertPointAfter(const DmpVector<llvm::Value *> &Bndl,
                      llvm::BasicBlock *BB, bool SkipPHIs = true,
                      bool SkipPads = true);

  static sandboxir::BBIterator
  getInsertPointAfter(const DmpVector<sandboxir::Value *> &Bndl,
                      sandboxir::BasicBlock *BB, bool SkipPHIs = true,
                      bool SkipPads = true);

  static std::pair<sandboxir::BasicBlock *, sandboxir::BBIterator>
  getInsertPointAfterInstrs(const DmpVector<sandboxir::Value *> &InstrRange);

  template <typename BuilderT, typename InstrRangeT>
  static void setInsertPointAfter(const InstrRangeT &Instrs,
                                  llvm::BasicBlock *BB, BuilderT &Builder,
                                  bool SkipPHIs = true) {
    auto WhereIt = getInsertPointAfter(Instrs, BB, SkipPHIs);
    Builder.SetInsertPoint(BB, WhereIt);
  }

  /// \Returns the number that corresponds to the index operand of \p InsertI.
  static std::optional<int> getInsertLane(llvm::InsertElementInst *InsertI);
  /// \Returns the number that corresponds to the index operand of \p ExtractI.
  static std::optional<int> getExtractLane(llvm::ExtractElementInst *ExtractI);
  /// \Returns the constant index lane of an insert or extract, or nullopt if
  /// not a constant.
  static std::optional<int>
  getConstantIndex(llvm::Instruction *InsertOrExtractI);

  static bool differentMathFlags(const DmpVector<sandboxir::Value *> &SBBndl);
  static bool differentWrapFlags(const DmpVector<sandboxir::Value *> &SBBndl);

  /// \Returns the lowest in BB among \p Instrs.
  static sandboxir::Instruction *
  getLowest(const DmpVector<sandboxir::Value *> &Instrs);
  static sandboxir::Instruction *
  getHighest(const DmpVector<sandboxir::Value *> &Instrs);

  static unsigned getAddrOperandIdx(sandboxir::Instruction *LoadOrStore);
};

/// TODO: These are utility functions that can access LLVM IR through
/// ValueAttorney. They should be removed at some point.
class VecUtilsPrivileged {
public:
  /// Proxy for llvm::propagateMetadata().
  static void propagateMetadata(sandboxir::Instruction *SBI,
                                const DmpVector<sandboxir::Value *> &SBVals);
  /// \Returns the number gap between the memory locations accessed by \p I1 and
  /// \p I2 in bytes.
  template <typename LoadOrStoreT>
  static std::optional<int>
  getPointerDiffInBytes(LoadOrStoreT *I1, LoadOrStoreT *I2, ScalarEvolution &SE,
                        const DataLayout &DL);
};

} // namespace sandboxir
} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_VECUTILS_H
