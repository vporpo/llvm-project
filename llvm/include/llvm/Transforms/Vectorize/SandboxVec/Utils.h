//===- Utils.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Helpers for the Sandbox Vectorizer.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_UTILS_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_UTILS_H

#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Bundle.h"

namespace llvm {

class SBBasicBlock;
class SBBBIterator;
class SBContext;
class SBLoadInstruction;
class SBStoreInstruction;
class DataLayout;

class SBUtils {
public:
  /// \Returns the number of elements in \p Ty, that is the number of lanes if
  /// vector or 1 if scalar.
  static int getNumElements(Type *Ty) {
    return Ty->isVectorTy() ? cast<FixedVectorType>(Ty)->getNumElements() : 1;
  }
  /// Returns \p Ty if scalar or its element type if vector.
  static Type *getElementType(Type *Ty) {
    return Ty->isVectorTy() ? cast<FixedVectorType>(Ty)->getElementType() : Ty;
  }

  static unsigned getMaxVFForTarget(Type *ElemTy, const DataLayout &DL,
                                    TargetTransformInfo &TTI) {
    assert(!ElemTy->isVectorTy() && "Expect scalar type!");
    unsigned VecBits =
        TTI.getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector)
            .getFixedValue();
    unsigned TyBits = DL.getTypeSizeInBits(ElemTy);
    if (VecBits < TyBits) {
      // Most probably target flags not set!
      return 1u;
    }
    return VecBits / TyBits;
  }

  /// Unlike other instructions, getting the value of a store requires a
  /// function call.
  static Value *getExpectedValue(Instruction *I) {
    if (auto *SI = dyn_cast<StoreInst>(I))
      return SI->getValueOperand();
    if (auto *RI = dyn_cast<ReturnInst>(I))
      return RI->getReturnValue();
    return I;
  }

  /// A store's type is void, which is rather useless. This function does the
  /// right thing and returns the type of the stored value.
  static Type *getExpectedType(Value *V) {
    if (isa<Instruction>(V)) {
      // A Return's value operand can be null if it returns void.
      if (auto *RI = dyn_cast<ReturnInst>(V)) {
        if (RI->getReturnValue() == nullptr)
          return RI->getType();
      }
      return getExpectedValue(cast<Instruction>(V))->getType();
    }
    return V->getType();
  }

  static Type *getExpectedType(SBValue *SBV);

  template <typename ValT> static bool isVector(ValT *V) {
    return isa<FixedVectorType>(getExpectedType(V));
  }

  /// If \p SBI is a load it returns its type. If a store it returns its value
  /// operand type.
  template <typename LoadOrStoreT>
  static Type *getLoadStoreType(LoadOrStoreT *SBI);

  /// \Returns the number gap between the memory locations accessed by \p I1 and
  /// \p I2 in bytes.
  template <typename LoadOrStoreT>
  static std::optional<int>
  getPointerDiffInBytes(LoadOrStoreT *I1, LoadOrStoreT *I2, ScalarEvolution &SE,
                        const DataLayout &DL);

  /// \Returns true if \p I1 and \p I2 are load/stores accessing consecutive
  /// memory addresses.
  template <typename LoadOrStoreT>
  static bool areConsecutive(LoadOrStoreT *I1, LoadOrStoreT *I2,
                             ScalarEvolution &SE, const DataLayout &DL);

  template <typename LoadOrStoreT>
  static bool areConsecutive(const SBValBundle &SBBndl, ScalarEvolution &SE,
                             const DataLayout &DL);

  /// \Returns true if \p I1 is accessing a prior memory location than \p I2.
  template <typename LoadOrStoreT>
  static bool comesBeforeInMem(LoadOrStoreT *I1, LoadOrStoreT *I2,
                               ScalarEvolution &SE, const DataLayout &DL);

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
  static int getNumLanes(Value *V) { return getNumLanes(getExpectedType(V)); }
  static unsigned getNumLanes(SBValue *SBV);
  static unsigned getNumBits(SBValue *SBV, const DataLayout &DL);

  /// \Returns true if \p V has at least \p N number of users.
  /// Please note that this is different from Value::hasNUsesOrMore(), as we are
  /// counting Users, not Uses.
  /// WARNING: This can be expensive as it is linear to the number of users.
  /// Please note that this is different from `hasNUsesOrMore()`, as we count
  /// the actual User and not the Use edges.
  static bool hasNUsersOrMore(Value *V, unsigned N);

  static bool isMem(Instruction *I) {
    // TODO: This is copied from SLPVectorizer.cpp initScheduleData().
    //       Is this correct?
    return I->mayReadOrWriteMemory() &&
           (!isa<IntrinsicInst>(I) ||
            (cast<IntrinsicInst>(I)->getIntrinsicID() !=
                 Intrinsic::sideeffect &&
             cast<IntrinsicInst>(I)->getIntrinsicID() !=
                 Intrinsic::pseudoprobe));
  }

  /// \Returns the first memory instruction before I in the instr chain, or
  /// nullptr if not found.
  static Instruction *getPrevMem(Instruction *I) {
    for (I = I->getPrevNode(); I != nullptr; I = I->getPrevNode()) {
      if (isMem(I))
        return I;
    }
    return nullptr;
  }
  static Instruction *getNextMem(Instruction *I) {
    for (I = I->getNextNode(); I != nullptr; I = I->getNextNode()) {
      if (isMem(I))
        return I;
    }
    return nullptr;
  }

  /// \Returns the a type that is \p NumElts times wider than \p ElemTy.
  /// It works for both scalar and vector \p ElemTy.
  static Type *getWideType(Type *ElemTy, uint32_t NumElts);
  /// This works even if Bndl contains Nodes with vector type.
  static unsigned getNumLanes(const SBValBundle &Bndl);
  static unsigned getNumBits(const SBInstrBundle &Bndl, const DataLayout &DL);
  static unsigned getNumBits(const SBValBundle &Bndl, const DataLayout &DL);
  /// \Returns the scalar type shared among Nodes in Bndl. \Returns nullptr if
  /// they don't share a common scalar type.
  static Type *getCommonScalarType(const SBValBundle &Bndl);
  /// Same as getCommonScalarType() but expects that there is a common scalar
  /// type. If not it will crash in a DEBUG build.
  static Type *getCommonScalarTypeFast(const SBValBundle &Bndl);

  static bool areInSameBB(const ValueBundle &Instrs) {
    if (Instrs.empty())
      return true;
    auto *I0 = cast<Instruction>(Instrs[0]);
    return all_of(drop_begin(Instrs.instrRange()), [I0](Instruction *I) {
      return I->getParent() == I0->getParent();
    });
  }
  static bool areInSameBB(const SBValBundle &SBInstrs);
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

  /// \Returns the iterator right after the lowest instruction in \p Bndl. If \p
  /// Bndl contains only non-instructions, or if the instructions in BB are at
  /// different blocks, other than \p BB, it returns the beginning of \p BB.
  static BasicBlock::iterator getInsertPointAfter(const ValueBundle &Bndl,
                                                  BasicBlock *BB,
                                                  bool SkipPHIs = true,
                                                  bool SkipPads = true);

  /// \Returns the number that corresponds to the index operand of \p InsertI.
  static std::optional<int> getInsertLane(InsertElementInst *InsertI) {
    auto *IdxOp = InsertI->getOperand(2);
    if (!isa<ConstantInt>(IdxOp))
      return std::nullopt;
    return cast<ConstantInt>(IdxOp)->getZExtValue();
  }
  /// \Returns the number that corresponds to the index operand of \p ExtractI.
  static std::optional<int> getExtractLane(ExtractElementInst *ExtractI) {
    auto *IdxOp = ExtractI->getIndexOperand();
    if (!isa<ConstantInt>(IdxOp))
      return std::nullopt;
    return cast<ConstantInt>(IdxOp)->getZExtValue();
  }

  /// \Returns the constant index lane of an insert or extract, or nullopt if
  /// not a constant.
  static std::optional<int> getConstantIndex(Instruction *InsertOrExtractI) {
    if (auto *InsertI = dyn_cast<InsertElementInst>(InsertOrExtractI))
      return getInsertLane(InsertI);
    if (auto *ExtractI = dyn_cast<ExtractElementInst>(InsertOrExtractI))
      return getExtractLane(ExtractI);
    llvm_unreachable("Expect Insert or Extract only!");
  }

  static bool differentMathFlags(const SBValBundle &SBBndl);
  static bool differentWrapFlags(const SBValBundle &SBBndl);

  template <typename BuilderT, typename InstrRangeT>
  static void setInsertPointAfter(const InstrRangeT &Instrs, BasicBlock *BB,
                                  BuilderT &Builder, bool SkipPHIs = true) {
    auto WhereIt = getInsertPointAfter(Instrs, BB, SkipPHIs);
    Builder.SetInsertPoint(BB, WhereIt);
  }

  static std::pair<SBBasicBlock *, SBBBIterator>
  getInsertPointAfterInstrs(const SBValBundle &InstrRange);

  /// \Returns the lowest in BB among \p Instrs.
  static SBInstruction *getLowest(const SBValBundle &Instrs);
  static SBInstruction *getHighest(const SBValBundle &Instrs);

  static std::string stripComments(const std::string &Str);

  /// Proxy for llvm::propagateMetadata().
  static void propagateMetadata(SBInstruction *SBI,
                                const SBValBundle &SBVals);

  /// \Returns the number of users, but is capped to \p Limit to save
  /// compile-time, because getNumUsers() is linear to the number of edges.
  static unsigned getNumUsersWithLimit(SBValue *SBV, unsigned Limit);

  static unsigned getAddrOperandIdx(SBInstruction *LoadOrStore);
};

class ShuffleMask {
public:
  using IndicesVecT = SmallVector<int, 8>;

private:
  IndicesVecT Indices;

public:
  ShuffleMask(IndicesVecT &&Indices) : Indices(std::move(Indices)) {}
  ShuffleMask(std::initializer_list<int> Indices) : Indices(Indices) {}
  explicit ShuffleMask(ArrayRef<int> Indices) : Indices(Indices) {}
  static ShuffleMask getIdentity(unsigned Lanes) {
    IndicesVecT Vec;
    Vec.reserve(Lanes);
    for (unsigned Idx = 0; Idx != Lanes; ++Idx)
      Vec.push_back(Idx);
    return ShuffleMask(std::move(Vec));
  }
  /// \Returns true if the mask is a perfect identity mask with consecutive
  /// indices, i.e., performs no lane shuffling, like 0,1,2,3...
  bool isIdentity() const;
  bool isInOrder() const;
  bool isIncreasingOrder() const;
  bool operator==(const ShuffleMask &Other) const;
  bool operator!=(const ShuffleMask &Other) const { return !(*this == Other); }
  size_t size() const { return Indices.size(); }
  int operator[](int Idx) const { return Indices[Idx]; }
  operator ArrayRef<int>() const { return Indices; }
  /// \Returns the inverse mask which when applied to the original gives us an
  /// identity mask. This can also be thought of as the horizontally flipped
  /// mask.
  /// For example:
  /// Mask0 (3,0,1,2) represents shuffling from BCDA to ABCD. This returns Mask1
  /// (1,2,3,0) which corresponds to the reverse shuffle from ABCD to BCDA.
  /// If you combine Mask0 and Mask1 you get the identity mask: (0,1,2,3).
  ShuffleMask getInverse() const {
    IndicesVecT NewIndices;
    NewIndices.resize(Indices.size());
    for (auto [Cnt, OrigIdx] : enumerate(Indices))
      NewIndices[OrigIdx] = Cnt;
    return ShuffleMask(std::move(NewIndices));
  }
  /// \Returns a mask that is equivalent of applying \p Other on top of this,
  /// back to back.
  /// For example, given MaskA (0,2,1,3) and MaskB (1,2,3,0), then
  /// MaskA.combine(MaskB) returns (1,3,2,0) and
  /// MaskB.combine(MaskA) returns (2,1,3,0).
  ShuffleMask combine(const ShuffleMask &Other) const {
    IndicesVecT NewIndices;
    NewIndices.resize(Indices.size());
    for (auto [Cnt, OrigIdx] : enumerate(Indices))
      NewIndices[Cnt] = Other[OrigIdx];
    return ShuffleMask(std::move(NewIndices));
  }
  hash_code hash() const {
    return hash_combine_range(Indices.begin(), Indices.end());
  }
  friend hash_code hash_value(const ShuffleMask &M) { return M.hash(); }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, const ShuffleMask &M) {
    M.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  void verify() const;
#endif
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_UTILS_H
