//===- VecUtils.cpp - Sandbox Vectorizer Utilities ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/VecUtils.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Region.h"

using namespace llvm;

template <typename LoadOrStoreT>
std::optional<int> sandboxir::VecUtilsPrivileged::getPointerDiffInBytes(
    LoadOrStoreT *I1, LoadOrStoreT *I2, ScalarEvolution &SE,
    const DataLayout &DL) {
  static_assert(
      std::is_same<LoadOrStoreT, sandboxir::LoadInst>::value ||
          std::is_same<LoadOrStoreT, sandboxir::StoreInst>::value,
      "Expected sandboxir::SBLoad or sandboxir::SBStore!");
  llvm::Value *PtrOp1 = ValueAttorney::getValue(I1->getPointerOperand());
  llvm::Value *PtrOp2 = ValueAttorney::getValue(I2->getPointerOperand());
  llvm::Value *Ptr1 = getUnderlyingObject(PtrOp1);
  llvm::Value *Ptr2 = getUnderlyingObject(PtrOp2);
  if (Ptr1 != Ptr2)
    return false;
  Type *ElemTy = Type::getInt8Ty(SE.getContext());
  // getPointersDiff(arg1, arg2) computes the difference arg2-arg1
  return getPointersDiff(ElemTy, PtrOp1, ElemTy, PtrOp2, DL, SE,
                         /*StrictCheck=*/false, /*CheckType=*/false);
}

template std::optional<int>
sandboxir::VecUtilsPrivileged::getPointerDiffInBytes<
    sandboxir::LoadInst>(sandboxir::LoadInst *,
                                  sandboxir::LoadInst *,
                                  ScalarEvolution &, const DataLayout &);
template std::optional<int>
sandboxir::VecUtilsPrivileged::getPointerDiffInBytes<
    sandboxir::StoreInst>(sandboxir::StoreInst *,
                                   sandboxir::StoreInst *,
                                   ScalarEvolution &, const DataLayout &);

template <typename LoadOrStoreT>
bool sandboxir::VecUtils::comesBeforeInMem(LoadOrStoreT *I1, LoadOrStoreT *I2,
                                             ScalarEvolution &SE,
                                             const DataLayout &DL) {
  auto Diff =
      sandboxir::VecUtilsPrivileged::getPointerDiffInBytes(I1, I2, SE, DL);
  if (!Diff)
    return false;
  return *Diff > 0;
}

template bool
sandboxir::VecUtils::comesBeforeInMem<sandboxir::LoadInst>(
    sandboxir::LoadInst *, sandboxir::LoadInst *,
    ScalarEvolution &, const DataLayout &);
template bool
sandboxir::VecUtils::comesBeforeInMem<sandboxir::StoreInst>(
    sandboxir::StoreInst *, sandboxir::StoreInst *,
    ScalarEvolution &, const DataLayout &);

template <typename LoadOrStoreT>
bool sandboxir::VecUtils::areConsecutive(LoadOrStoreT *I1, LoadOrStoreT *I2,
                                           ScalarEvolution &SE,
                                           const DataLayout &DL) {
  static_assert(
      std::is_same<LoadOrStoreT, sandboxir::LoadInst>::value ||
          std::is_same<LoadOrStoreT, sandboxir::StoreInst>::value,
      "Expected sandboxir::SBLoad or sandboxir::SBStore!");
  auto Diff =
      sandboxir::VecUtilsPrivileged::getPointerDiffInBytes(I1, I2, SE, DL);
  if (!Diff)
    return false;
  Type *ElmTy = getLoadStoreType(I1);
  int ElmBytes = DL.getTypeSizeInBits(ElmTy) / 8;
  return *Diff == ElmBytes;
}

template bool
sandboxir::VecUtils::areConsecutive<sandboxir::LoadInst>(
    sandboxir::LoadInst *, sandboxir::LoadInst *,
    ScalarEvolution &, const DataLayout &);
template bool
sandboxir::VecUtils::areConsecutive<sandboxir::StoreInst>(
    sandboxir::StoreInst *, sandboxir::StoreInst *,
    ScalarEvolution &, const DataLayout &);

template <typename LoadOrStoreT>
bool sandboxir::VecUtils::areConsecutive(
    const DmpVector<sandboxir::Value *> &SBBndl, ScalarEvolution &SE,
    const DataLayout &DL) {
  static_assert(
      std::is_same<LoadOrStoreT, sandboxir::LoadInst>::value ||
          std::is_same<LoadOrStoreT, sandboxir::StoreInst>::value,
      "Expected sandboxir::SBLoad or sandboxir::SBStore!");
  assert(isa<LoadOrStoreT>(SBBndl[0]) &&
         "Expected sandboxir::SBStoreInst or sandboxir::SBLoadInst!");
  auto *LastS = cast<LoadOrStoreT>(SBBndl[0]);
  for (sandboxir::Value *V : drop_begin(SBBndl)) {
    assert(isa<LoadOrStoreT>(V) && "Unimplemented: we only support StoreInst!");
    auto *S = cast<LoadOrStoreT>(V);
    if (!sandboxir::VecUtils::areConsecutive(LastS, S, SE, DL))
      return false;
    LastS = S;
  }
  return true;
}

template bool
sandboxir::VecUtils::areConsecutive<sandboxir::LoadInst>(
    const DmpVector<sandboxir::Value *> &, ScalarEvolution &,
    const DataLayout &);
template bool
sandboxir::VecUtils::areConsecutive<sandboxir::StoreInst>(
    const DmpVector<sandboxir::Value *> &, ScalarEvolution &,
    const DataLayout &);

template <typename LoadOrStoreT>
Type *sandboxir::VecUtils::getLoadStoreType(LoadOrStoreT *SBI) {
  static_assert(
      std::is_same<LoadOrStoreT, sandboxir::LoadInst>::value ||
          std::is_same<LoadOrStoreT, sandboxir::StoreInst>::value,
      "Expected sandboxir::SBLoad or sandboxir::SBStore!");
  if constexpr (std::is_same<LoadOrStoreT,
                             sandboxir::LoadInst>::value) {
    return cast<sandboxir::LoadInst>(SBI)->getType();
  } else if constexpr (std::is_same<LoadOrStoreT,
                                    sandboxir::StoreInst>::value) {
    return cast<sandboxir::StoreInst>(SBI)
        ->getValueOperand()
        ->getType();
  } else {
    llvm_unreachable("Expected sandboxir::LoadInst or "
                     "sandboxir::StoreInst");
  }
}

template Type *
sandboxir::VecUtils::getLoadStoreType<sandboxir::LoadInst>(
    sandboxir::LoadInst *);
template Type *
sandboxir::VecUtils::getLoadStoreType<sandboxir::StoreInst>(
    sandboxir::StoreInst *);

Type *sandboxir::VecUtils::getCommonScalarType(
    const DmpVector<sandboxir::Value *> &Bndl) {
  sandboxir::Value *V0 = Bndl[0];
  Type *Ty0 = sandboxir::VecUtils::getExpectedType(V0);
  Type *ScalarTy = sandboxir::VecUtils::getElementType(Ty0);
  for (auto *V : drop_begin(Bndl)) {
    Type *NTy = sandboxir::VecUtils::getExpectedType(V);
    Type *NScalarTy = sandboxir::VecUtils::getElementType(NTy);
    if (NScalarTy != ScalarTy)
      return nullptr;
  }
  return ScalarTy;
}

Type *sandboxir::VecUtils::getCommonScalarTypeFast(
    const DmpVector<sandboxir::Value *> &Bndl) {
  sandboxir::Value *V0 = Bndl[0];
  Type *Ty0 = sandboxir::VecUtils::getExpectedType(V0);
  Type *ScalarTy = sandboxir::VecUtils::getElementType(Ty0);
  assert(getCommonScalarType(Bndl) && "Expected common scalar type!");
  return ScalarTy;
}

llvm::Value *sandboxir::VecUtils::getExpectedValue(llvm::Instruction *I) {
  if (auto *SI = dyn_cast<llvm::StoreInst>(I))
    return SI->getValueOperand();
  if (auto *RI = dyn_cast<llvm::ReturnInst>(I))
    return RI->getReturnValue();
  return I;
}

sandboxir::Value *
sandboxir::VecUtils::getExpectedValue(const sandboxir::Instruction *I) {
  if (auto *SI = dyn_cast<sandboxir::StoreInst>(I))
    return SI->getValueOperand();
  if (auto *RI = dyn_cast<sandboxir::RetInst>(I))
    return RI->getReturnValue();
  return const_cast<sandboxir::Instruction *>(I);
}

Type *sandboxir::VecUtils::getExpectedType(llvm::Value *V) {
  if (isa<llvm::Instruction>(V)) {
    // A Return's value operand can be null if it returns void.
    if (auto *RI = dyn_cast<llvm::ReturnInst>(V)) {
      if (RI->getReturnValue() == nullptr)
        return RI->getType();
    }
    return getExpectedValue(cast<llvm::Instruction>(V))->getType();
  }
  return V->getType();
}

Type *sandboxir::VecUtils::getExpectedType(const sandboxir::Value *V) {
  if (isa<sandboxir::Instruction>(V)) {
    // A Return's value operand can be null if it returns void.
    if (auto *RI = dyn_cast<sandboxir::RetInst>(V)) {
      if (RI->getReturnValue() == nullptr)
        return RI->getType();
    }
    return getExpectedValue(cast<sandboxir::Instruction>(V))->getType();
  }
  return V->getType();
}

unsigned sandboxir::VecUtils::getNumLanes(const sandboxir::Value *SBV) {
  Type *Ty = sandboxir::VecUtils::getExpectedType(SBV);
  return isa<FixedVectorType>(Ty) ? cast<FixedVectorType>(Ty)->getNumElements()
                                  : 1;
}

unsigned sandboxir::VecUtils::getNumLanes(
    const DmpVector<sandboxir::Value *> &Bndl) {
  unsigned Lanes = 0;
  for (sandboxir::Value *SBV : Bndl)
    Lanes += getNumLanes(SBV);
  return Lanes;
}

unsigned sandboxir::VecUtils::getNumBits(sandboxir::Value *SBV,
                                           const DataLayout &DL) {
  Type *Ty = sandboxir::VecUtils::getExpectedType(SBV);
  return DL.getTypeSizeInBits(Ty);
}
template <typename BndlT>
static unsigned getNumBitsCommon(const BndlT &Bndl, const DataLayout &DL) {
  unsigned Bits = 0;
  for (sandboxir::Value *SBV : Bndl)
    Bits += sandboxir::VecUtils::getNumBits(SBV, DL);
  return Bits;
}
unsigned
sandboxir::VecUtils::getNumBits(const DmpVector<sandboxir::Value *> &Bndl,
                                  const DataLayout &DL) {
  return getNumBitsCommon(Bndl, DL);
}

unsigned sandboxir::VecUtils::getNumBits(
    const DmpVector<sandboxir::Instruction *> &Bndl, const DataLayout &DL) {
  return getNumBitsCommon(Bndl, DL);
}

Type *sandboxir::VecUtils::getWideType(Type *ElemTy, uint32_t NumElts) {
  if (ElemTy->isVectorTy()) {
    auto *VecTy = cast<FixedVectorType>(ElemTy);
    ElemTy = VecTy->getElementType();
    NumElts = VecTy->getNumElements() * NumElts;
  }
  return FixedVectorType::get(ElemTy, NumElts);
}

bool sandboxir::VecUtils::areInSameBB(
    const DmpVector<llvm::Value *> &Instrs) {
  if (Instrs.empty())
    return true;
  auto *I0 = cast<llvm::Instruction>(Instrs[0]);
  return all_of(drop_begin(Instrs.instrRange()), [I0](llvm::Instruction *I) {
    return I->getParent() == I0->getParent();
  });
}
bool sandboxir::VecUtils::areInSameBB(
    const DmpVector<sandboxir::Value *> &SBInstrs) {
  if (SBInstrs.empty())
    return true;
  auto *I0 = cast<sandboxir::Instruction>(SBInstrs[0]);
  return all_of(drop_begin(SBInstrs), [I0](sandboxir::Value *SBV) {
    return cast<sandboxir::Instruction>(SBV)->getParent() == I0->getParent();
  });
}

/// \Returns the next iterator after \p I, but will also skip PHIs if \p I is
/// a PHINode.
template <typename BBT, typename InstrT, typename PHIT>
static typename BBT::iterator getNextIteratorSkippingPHIs(InstrT *I) {
  auto NextIt = std::next(I->getIterator());
  typename BBT::iterator ItE = I->getParent()->end();
  while (NextIt != ItE && isa<PHIT>(&*NextIt))
    ++NextIt;
  return NextIt;
}

BasicBlock::iterator
sandboxir::VecUtils::getInsertPointAfter(const DmpVector<llvm::Value *> &Bndl,
                                           llvm::BasicBlock *BB, bool SkipPHIs,
                                           bool SkipPads) {
  llvm::Instruction *LowestI = nullptr;
  for (llvm::Value *V : Bndl) {
    if (V == nullptr)
      continue;
    if (!isa<llvm::Instruction>(V))
      continue;
    llvm::Instruction *I = cast<llvm::Instruction>(V);
    // A nullptr instruction means that we are at the top of BB.
    llvm::Instruction *WhereI = I->getParent() == BB ? I : nullptr;
    if (LowestI == nullptr ||
        // If WhereI == null then a non-null LowestI will always come after it.
        (WhereI != nullptr && LowestI->comesBefore(WhereI)))
      LowestI = WhereI;
  }

  llvm::BasicBlock::iterator It;
  if (LowestI == nullptr)
    It = SkipPHIs ? BB->getFirstNonPHIIt() : BB->begin();
  else
    It = SkipPHIs
             ? getNextIteratorSkippingPHIs<llvm::BasicBlock, llvm::Instruction,
                                           llvm::PHINode>(LowestI)
             : std::next(LowestI->getIterator());
  if (SkipPads) {
    if (It != BB->end()) {
      llvm::Instruction *I = &*It;
      if (LLVM_UNLIKELY(isa<LandingPadInst>(I) || isa<CatchPadInst>(I) ||
                        isa<CleanupPadInst>(I)))
        ++It;
    }
  }
  return It;
}

sandboxir::BasicBlock::iterator sandboxir::VecUtils::getInsertPointAfter(
    const DmpVector<sandboxir::Value *> &Bndl, sandboxir::BasicBlock *BB,
    bool SkipPHIs, bool SkipPads) {
  sandboxir::Instruction *LowestI = nullptr;
  for (sandboxir::Value *V : Bndl) {
    if (V == nullptr)
      continue;
    if (!isa<sandboxir::Instruction>(V))
      continue;
    sandboxir::Instruction *I = cast<sandboxir::Instruction>(V);
    // A nullptr instruction means that we are at the top of BB.
    sandboxir::Instruction *WhereI = I->getParent() == BB ? I : nullptr;
    if (LowestI == nullptr ||
        // If WhereI == null then a non-null LowestI will always come after it.
        (WhereI != nullptr && LowestI->comesBefore(WhereI)))
      LowestI = WhereI;
  }

  sandboxir::BasicBlock::iterator It;
  if (LowestI == nullptr)
    It = SkipPHIs ? BB->getFirstNonPHIIt() : BB->begin();
  else
    It = SkipPHIs ? getNextIteratorSkippingPHIs<sandboxir::BasicBlock,
                                                sandboxir::Instruction,
                                                sandboxir::PHINode>(LowestI)
                  : std::next(LowestI->getIterator());
  if (SkipPads) {
    if (It != BB->end()) {
      sandboxir::Instruction *I = &*It;
      if (I->isPad())
        ++It;
    }
  }
  return It;
}

std::pair<sandboxir::BasicBlock *, sandboxir::BasicBlock::iterator>
sandboxir::VecUtils::getInsertPointAfterInstrs(
    const DmpVector<sandboxir::Value *> &InstrRange) {
  // Find the instr that is lowest in the BB.
  sandboxir::Instruction *LastI = nullptr;
  for (auto *SBV : InstrRange) {
    auto *I = cast<sandboxir::Instruction>(SBV);
    if (LastI == nullptr || LastI->comesBefore(I))
      LastI = I;
  }
  // If Bndl contains Arguments or Constants, use the beginning of the BB.
  sandboxir::BasicBlock::iterator WhereIt = std::next(LastI->getIterator());
  sandboxir::BasicBlock *WhereBB = LastI->getParent();
  return {WhereBB, WhereIt};
}

std::optional<int>
sandboxir::VecUtils::getInsertLane(llvm::InsertElementInst *InsertI) {
  auto *IdxOp = InsertI->getOperand(2);
  if (!isa<ConstantInt>(IdxOp))
    return std::nullopt;
  return cast<ConstantInt>(IdxOp)->getZExtValue();
}

std::optional<int>
sandboxir::VecUtils::getExtractLane(llvm::ExtractElementInst *ExtractI) {
  auto *IdxOp = ExtractI->getIndexOperand();
  if (!isa<llvm::ConstantInt>(IdxOp))
    return std::nullopt;
  return cast<ConstantInt>(IdxOp)->getZExtValue();
}
std::optional<int>
sandboxir::VecUtils::getConstantIndex(llvm::Instruction *InsertOrExtractI) {
  if (auto *InsertI = dyn_cast<llvm::InsertElementInst>(InsertOrExtractI))
    return sandboxir::VecUtils::getInsertLane(InsertI);
  if (auto *ExtractI = dyn_cast<llvm::ExtractElementInst>(InsertOrExtractI))
    return sandboxir::VecUtils::getExtractLane(ExtractI);
  llvm_unreachable("Expect Insert or Extract only!");
}

bool sandboxir::VecUtils::differentMathFlags(
    const DmpVector<sandboxir::Value *> &SBBndl) {
  FastMathFlags FMF0 =
      cast<sandboxir::Instruction>(SBBndl[0])->getFastMathFlags();
  return any_of(drop_begin(SBBndl), [FMF0](auto *SBV) {
    return cast<sandboxir::Instruction>(SBV)->getFastMathFlags() != FMF0;
  });
}

bool sandboxir::VecUtils::differentWrapFlags(
    const DmpVector<sandboxir::Value *> &SBBndl) {
  bool NUW0 = cast<sandboxir::Instruction>(SBBndl[0])->hasNoUnsignedWrap();
  bool NSW0 = cast<sandboxir::Instruction>(SBBndl[0])->hasNoSignedWrap();
  return any_of(drop_begin(SBBndl), [NUW0, NSW0](auto *SBV) {
    return cast<sandboxir::Instruction>(SBV)->hasNoUnsignedWrap() != NUW0 ||
           cast<sandboxir::Instruction>(SBV)->hasNoSignedWrap() != NSW0;
  });
}

sandboxir::Instruction *sandboxir::VecUtils::getLowest(
    const DmpVector<sandboxir::Value *> &Instrs) {
  sandboxir::Instruction *LowestI =
      cast<sandboxir::Instruction>(Instrs.front());
  for (auto *SBV : drop_begin(Instrs)) {
    auto *SBI = cast<sandboxir::Instruction>(SBV);
    if (LowestI->comesBefore(SBI))
      LowestI = SBI;
  }
  return LowestI;
}

sandboxir::Instruction *sandboxir::VecUtils::getHighest(
    const DmpVector<sandboxir::Value *> &Instrs) {
  sandboxir::Instruction *HighestI =
      cast<sandboxir::Instruction>(Instrs.front());
  for (auto *SBV : drop_begin(Instrs)) {
    auto *SBI = cast<sandboxir::Instruction>(SBV);
    if (HighestI->comesAfter(SBI))
      HighestI = SBI;
  }
  return HighestI;
}

unsigned sandboxir::VecUtils::getAddrOperandIdx(
    sandboxir::Instruction *LoadOrStore) {
  if (isa<sandboxir::LoadInst>(LoadOrStore))
    return 0u;
  assert(isa<sandboxir::StoreInst>(LoadOrStore) &&
         "Expected only load or store!");
  return 1u;
}

void sandboxir::VecUtilsPrivileged::propagateMetadata(
    sandboxir::Instruction *SBI,
    const DmpVector<sandboxir::Value *> &SBVals) {
  auto *I = cast<llvm::Instruction>(ValueAttorney::getValue(SBI));
  // llvm::propagateMetadata() will propagate sandboxir::Region metadata too,
  // but we don't want this to happen. So save the metadata here and set them
  // later.
  auto *SavedSBRegionMD = I->getMetadata(sandboxir::Region::MDKind);
  SmallVector<llvm::Value *> Vals;
  Vals.reserve(SBVals.size());
  for (auto *SBV : SBVals)
    Vals.push_back(ValueAttorney::getValue(SBV));
  llvm::propagateMetadata(I, Vals);
  // Override sandboxir::Region meteadata with the value before
  // propagateMetadata().
  I->setMetadata(sandboxir::Region::MDKind, SavedSBRegionMD);

  // Now go over !dbg metadata. We copy the first !dbg metadata in `SBVals`.
  MDNode *DbgMD = nullptr;
  for (auto *SBV : SBVals)
    if (auto *I = dyn_cast<llvm::Instruction>(ValueAttorney::getValue(SBV)))
      if ((DbgMD = I->getMetadata(LLVMContext::MD_dbg)))
        break;
  if (DbgMD != nullptr)
    I->setMetadata(LLVMContext::MD_dbg, DbgMD);
}
