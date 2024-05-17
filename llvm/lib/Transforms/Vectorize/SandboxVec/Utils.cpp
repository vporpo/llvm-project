//===- Utils.cpp - Sandbox Vectorizer Utilities ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Utils.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SBRegion.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"
#include <sstream>

using namespace llvm;

template <typename LoadOrStoreT>
Type *SBUtils::getLoadStoreType(LoadOrStoreT *SBI) {
  static_assert(std::is_same<LoadOrStoreT, SBLoadInstruction>::value ||
                    std::is_same<LoadOrStoreT, SBStoreInstruction>::value,
                "Expected SBLoad or SBStore!");
  if constexpr (std::is_same<LoadOrStoreT, SBLoadInstruction>::value) {
    return cast<SBLoadInstruction>(SBI)->getType();
  } else if constexpr (std::is_same<LoadOrStoreT,
                                    SBStoreInstruction>::value) {
    return cast<SBStoreInstruction>(SBI)->getValueOperand()->getType();
  } else {
    llvm_unreachable("Expected SBLoadInstruction or SBStoreInstruction");
  }
}

template Type *
SBUtils::getLoadStoreType<SBLoadInstruction>(SBLoadInstruction *);
template Type *
SBUtils::getLoadStoreType<SBStoreInstruction>(SBStoreInstruction *);

template <typename LoadOrStoreT>
std::optional<int>
SBUtils::getPointerDiffInBytes(LoadOrStoreT *I1, LoadOrStoreT *I2,
                                ScalarEvolution &SE, const DataLayout &DL) {
  static_assert(std::is_same<LoadOrStoreT, SBLoadInstruction>::value ||
                    std::is_same<LoadOrStoreT, SBStoreInstruction>::value,
                "Expected SBLoad or SBStore!");
  Value *PtrOp1 = ValueAttorney::getValue(I1->getPointerOperand());
  Value *PtrOp2 = ValueAttorney::getValue(I2->getPointerOperand());
  Value *Ptr1 = getUnderlyingObject(PtrOp1);
  Value *Ptr2 = getUnderlyingObject(PtrOp2);
  if (Ptr1 != Ptr2)
    return false;
  Type *ElemTy = Type::getInt8Ty(SE.getContext());
  // getPointersDiff(arg1, arg2) computes the difference arg2-arg1
  return getPointersDiff(ElemTy, PtrOp1, ElemTy, PtrOp2, DL, SE,
                         /*StrictCheck=*/false, /*CheckType=*/false);
}

template std::optional<int>
SBUtils::getPointerDiffInBytes<SBLoadInstruction>(SBLoadInstruction *,
                                                     SBLoadInstruction *,
                                                     ScalarEvolution &,
                                                     const DataLayout &);
template std::optional<int>
SBUtils::getPointerDiffInBytes<SBStoreInstruction>(SBStoreInstruction *,
                                                      SBStoreInstruction *,
                                                      ScalarEvolution &,
                                                      const DataLayout &);

template <typename LoadOrStoreT>
bool SBUtils::areConsecutive(LoadOrStoreT *I1, LoadOrStoreT *I2,
                              ScalarEvolution &SE, const DataLayout &DL) {
  static_assert(std::is_same<LoadOrStoreT, SBLoadInstruction>::value ||
                    std::is_same<LoadOrStoreT, SBStoreInstruction>::value,
                "Expected SBLoad or SBStore!");
  auto Diff = getPointerDiffInBytes(I1, I2, SE, DL);
  if (!Diff)
    return false;
  Type *ElmTy = getLoadStoreType(I1);
  int ElmBytes = DL.getTypeSizeInBits(ElmTy) / 8;
  return *Diff == ElmBytes;
}

template bool SBUtils::areConsecutive<SBLoadInstruction>(
    SBLoadInstruction *, SBLoadInstruction *, ScalarEvolution &,
    const DataLayout &);
template bool SBUtils::areConsecutive<SBStoreInstruction>(
    SBStoreInstruction *, SBStoreInstruction *, ScalarEvolution &,
    const DataLayout &);

template <typename LoadOrStoreT>
bool SBUtils::areConsecutive(const SBValBundle &SBBndl,
                              ScalarEvolution &SE, const DataLayout &DL) {
  static_assert(std::is_same<LoadOrStoreT, SBLoadInstruction>::value ||
                    std::is_same<LoadOrStoreT, SBStoreInstruction>::value,
                "Expected SBLoad or SBStore!");
  assert(isa<LoadOrStoreT>(SBBndl[0]) &&
         "Expected SBStoreInst or SBLoadInst!");
  auto *LastS = cast<LoadOrStoreT>(SBBndl[0]);
  for (SBValue *V : drop_begin(SBBndl)) {
    assert(isa<LoadOrStoreT>(V) && "Unimplemented: we only support StoreInst!");
    auto *S = cast<LoadOrStoreT>(V);
    if (!areConsecutive(LastS, S, SE, DL))
      return false;
    LastS = S;
  }
  return true;
}

template bool SBUtils::areConsecutive<SBLoadInstruction>(
    const SBValBundle &, ScalarEvolution &, const DataLayout &);
template bool SBUtils::areConsecutive<SBStoreInstruction>(
    const SBValBundle &, ScalarEvolution &, const DataLayout &);

template <typename LoadOrStoreT>
bool SBUtils::comesBeforeInMem(LoadOrStoreT *I1, LoadOrStoreT *I2,
                                ScalarEvolution &SE, const DataLayout &DL) {
  auto Diff = getPointerDiffInBytes(I1, I2, SE, DL);
  if (!Diff)
    return false;
  return *Diff > 0;
}

template bool SBUtils::comesBeforeInMem<SBLoadInstruction>(
    SBLoadInstruction *, SBLoadInstruction *, ScalarEvolution &,
    const DataLayout &);
template bool SBUtils::comesBeforeInMem<SBStoreInstruction>(
    SBStoreInstruction *, SBStoreInstruction *, ScalarEvolution &,
    const DataLayout &);

Type *SBUtils::getExpectedType(SBValue *SBV) {
  return getExpectedType(ValueAttorney::getValue(SBV));
}

Type *SBUtils::getWideType(Type *ElemTy, uint32_t NumElts) {
  if (ElemTy->isVectorTy()) {
    auto *VecTy = cast<FixedVectorType>(ElemTy);
    ElemTy = VecTy->getElementType();
    NumElts = VecTy->getNumElements() * NumElts;
  }
  return FixedVectorType::get(ElemTy, NumElts);
}

unsigned SBUtils::getNumLanes(SBValue *SBV) {
  Type *Ty = SBUtils::getExpectedType(SBV);
  return isa<FixedVectorType>(Ty) ? cast<FixedVectorType>(Ty)->getNumElements()
                                  : 1;
}

unsigned SBUtils::getNumLanes(const SBValBundle &Bndl) {
  unsigned Lanes = 0;
  for (SBValue *SBV : Bndl)
    Lanes += getNumLanes(SBV);
  return Lanes;
}

unsigned SBUtils::getNumBits(SBValue *SBV, const DataLayout &DL) {
  Type *Ty = SBUtils::getExpectedType(SBV);
  return DL.getTypeSizeInBits(Ty);
}

template <typename BndlT>
static unsigned getNumBitsCommon(const BndlT &Bndl, const DataLayout &DL) {
  unsigned Bits = 0;
  for (SBValue *SBV : Bndl)
    Bits += SBUtils::getNumBits(SBV, DL);
  return Bits;
}
unsigned SBUtils::getNumBits(const SBValBundle &Bndl, const DataLayout &DL) {
  return getNumBitsCommon(Bndl, DL);
}

unsigned SBUtils::getNumBits(const SBInstrBundle &Bndl,
                              const DataLayout &DL) {
  return getNumBitsCommon(Bndl, DL);
}

Type *SBUtils::getCommonScalarType(const SBValBundle &Bndl) {
  SBValue *N0 = Bndl[0];
  Type *Ty0 = N0->getExpectedType();
  Type *ScalarTy = SBUtils::getElementType(Ty0);
  for (auto *N : drop_begin(Bndl)) {
    Type *NTy = N->getExpectedType();
    Type *NScalarTy = SBUtils::getElementType(NTy);
    if (NScalarTy != ScalarTy)
      return nullptr;
  }
  return ScalarTy;
}

Type *SBUtils::getCommonScalarTypeFast(const SBValBundle &Bndl) {
  SBValue *N0 = Bndl[0];
  Type *Ty0 = N0->getExpectedType();
  Type *ScalarTy = SBUtils::getElementType(Ty0);
  assert(getCommonScalarType(Bndl) && "Expected common scalar type!");
  return ScalarTy;
}

bool SBUtils::hasNUsersOrMore(Value *V, unsigned N) {
  DenseSet<User *> Users;
  for (User *U : V->users()) {
    Users.insert(U);
    if (Users.size() == N)
      return true;
  }
  return false;
}

bool SBUtils::areInSameBB(const SBValBundle &SBInstrs) {
  if (SBInstrs.empty())
    return true;
  auto *I0 = cast<SBInstruction>(SBInstrs[0]);
  return all_of(drop_begin(SBInstrs), [I0](SBValue *SBV) {
    return cast<SBInstruction>(SBV)->getParent() == I0->getParent();
  });
}

/// \Returns the next iterator after \p I, but will also skip PHIs if \p I is a
/// PHINode.
static BasicBlock::iterator getNextIteratorSkippingPHIs(Instruction *I) {
  auto NextIt = std::next(I->getIterator());
  BasicBlock::iterator ItE = I->getParent()->end();
  while (NextIt != ItE && isa<PHINode>(&*NextIt))
    ++NextIt;
  return NextIt;
}

BasicBlock::iterator SBUtils::getInsertPointAfter(const ValueBundle &Bndl,
                                                   BasicBlock *BB,
                                                   bool SkipPHIs,
                                                   bool SkipPads) {
  Instruction *LowestI = nullptr;
  for (Value *V : Bndl) {
    if (V == nullptr)
      continue;
    if (!isa<Instruction>(V))
      continue;
    Instruction *I = cast<Instruction>(V);
    // A nullptr instruction means that we are at the top of BB.
    Instruction *WhereI = I->getParent() == BB ? I : nullptr;
    if (LowestI == nullptr ||
        // If WhereI == null then a non-null LowestI will always come after it.
        (WhereI != nullptr && LowestI->comesBefore(WhereI)))
      LowestI = WhereI;
  }

  BasicBlock::iterator It;
  if (LowestI == nullptr)
    It = SkipPHIs ? BB->getFirstNonPHIIt() : BB->begin();
  else
    It = SkipPHIs ? getNextIteratorSkippingPHIs(LowestI)
                  : std::next(LowestI->getIterator());
  if (SkipPads) {
    if (It != BB->end()) {
      Instruction *I = &*It;
      if (LLVM_UNLIKELY(isa<LandingPadInst>(I) || isa<CatchPadInst>(I) ||
                        isa<CleanupPadInst>(I)))
        ++It;
    }
  }
  return It;
}

bool SBUtils::differentMathFlags(const SBValBundle &SBBndl) {
  FastMathFlags FMF0 = cast<SBInstruction>(SBBndl[0])->getFastMathFlags();
  return any_of(drop_begin(SBBndl), [FMF0](auto *SBV) {
    return cast<SBInstruction>(SBV)->getFastMathFlags() != FMF0;
  });
}

bool SBUtils::differentWrapFlags(const SBValBundle &SBBndl) {
  bool NUW0 = cast<SBInstruction>(SBBndl[0])->hasNoUnsignedWrap();
  bool NSW0 = cast<SBInstruction>(SBBndl[0])->hasNoSignedWrap();
  return any_of(drop_begin(SBBndl), [NUW0, NSW0](auto *SBV) {
    return cast<SBInstruction>(SBV)->hasNoUnsignedWrap() != NUW0 ||
           cast<SBInstruction>(SBV)->hasNoSignedWrap() != NSW0;
  });
}

std::pair<SBBasicBlock *, SBBasicBlock::iterator>
SBUtils::getInsertPointAfterInstrs(const SBValBundle &InstrRange) {
  // Find the instr that is lowest in the BB.
  SBInstruction *LastI = nullptr;
  for (auto *SBV : InstrRange) {
    auto *I = cast<SBInstruction>(SBV);
    if (LastI == nullptr || LastI->comesBefore(I))
      LastI = I;
  }
  // If Bndl contains Arguments or Constants, use the beginning of the BB.
  SBBasicBlock::iterator WhereIt = std::next(LastI->getIterator());
  SBBasicBlock *WhereBB = LastI->getParent();
  return {WhereBB, WhereIt};
}

SBInstruction *SBUtils::getLowest(const SBValBundle &Instrs) {
  SBInstruction *LowestI = cast<SBInstruction>(Instrs.front());
  for (auto *SBV : drop_begin(Instrs)) {
    auto *SBI = cast<SBInstruction>(SBV);
    if (LowestI->comesBefore(SBI))
      LowestI = SBI;
  }
  return LowestI;
}

SBInstruction *SBUtils::getHighest(const SBValBundle &Instrs) {
  SBInstruction *HighestI = cast<SBInstruction>(Instrs.front());
  for (auto *SBV : drop_begin(Instrs)) {
    auto *SBI = cast<SBInstruction>(SBV);
    if (HighestI->comesAfter(SBI))
      HighestI = SBI;
  }
  return HighestI;
}

std::string SBUtils::stripComments(const std::string &Str) {
  std::stringstream InSS(Str);
  std::stringstream OutSS;
  std::string Line;
  while (std::getline(InSS, Line)) {
    auto CommentChar = Line.find(" ;");
    OutSS << Line.substr(0, CommentChar) << "\n";
  }
  return OutSS.str();
}

void SBUtils::propagateMetadata(SBInstruction *SBI,
                                 const SBValBundle &SBVals) {
  auto *I = cast<Instruction>(ValueAttorney::getValue(SBI));
  // llvm::propagateMetadata() will propagate SBRegion metadata too, but we
  // don't want this to happen. So save the metadata here and set them later.
  auto *SavedSBRegionMD = I->getMetadata(SBRegion::MDKind);
  SmallVector<Value *> Vals;
  Vals.reserve(SBVals.size());
  for (auto *SBV : SBVals)
    Vals.push_back(ValueAttorney::getValue(SBV));
  llvm::propagateMetadata(I, Vals);
  // Override SBRegion meteadata with the value before propagateMetadata().
  I->setMetadata(SBRegion::MDKind, SavedSBRegionMD);

  // Now go over !dbg metadata. We copy the first !dbg metadata in `SBVals`.
  MDNode *DbgMD = nullptr;
  for (auto *SBV : SBVals)
    if (auto *I = dyn_cast<Instruction>(ValueAttorney::getValue(SBV)))
      if ((DbgMD = I->getMetadata(LLVMContext::MD_dbg)))
        break;
  if (DbgMD != nullptr)
    I->setMetadata(LLVMContext::MD_dbg, DbgMD);
}

unsigned SBUtils::getNumUsersWithLimit(SBValue *SBV, unsigned Limit) {
  unsigned Cnt = 0;
  for (SBUser *U : SBV->users()) {
    (void)U;
    ++Cnt;
    if (Cnt == Limit)
      break;
  }
  return Cnt;
}

unsigned SBUtils::getAddrOperandIdx(SBInstruction *LoadOrStore) {
  if (isa<SBLoadInstruction>(LoadOrStore))
    return 0u;
  assert(isa<SBStoreInstruction>(LoadOrStore) &&
         "Expected only load or store!");
  return 1u;
}

bool ShuffleMask::isIdentity() const {
  if (Indices.empty())
    return true;
  for (auto [Lane, Idx] : enumerate(Indices))
    if (Idx != (int)Lane)
      return false;
  return true;
}

bool ShuffleMask::isInOrder() const {
  if (Indices.empty())
    return true;
  int LastIdx = Indices.front();
  for (int Idx : drop_begin(Indices)) {
    if (Idx != LastIdx + 1)
      return false;
    LastIdx = Idx;
  }
  return true;
}

bool ShuffleMask::isIncreasingOrder() const {
  if (Indices.empty())
    return true;
  int LastIdx = Indices.front();
  for (int Idx : drop_begin(Indices)) {
    if (Idx <= LastIdx)
      return false;
    LastIdx = Idx;
  }
  return true;
}

bool ShuffleMask::operator==(const ShuffleMask &Other) const {
  return equal(Indices, Other.Indices);
}

#ifndef NDEBUG
void ShuffleMask::dump(raw_ostream &OS) const {
  for (auto [Lane, ShuffleIdx] : enumerate(Indices)) {
    if (Lane != 0)
      OS << ", ";
    OS << ShuffleIdx;
  }
}

void ShuffleMask::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

void ShuffleMask::verify() const {
  auto NumLanes = (int)Indices.size();
  for (auto Idx : Indices)
    assert(Idx < NumLanes && "Bad index!");
}
#endif // NDEBUG
