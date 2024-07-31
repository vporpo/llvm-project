//===- SandboxIRVec.cpp - Vectorization-specific SandboxIR ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIRVec.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"

using namespace llvm;

bool sandboxir::ShuffleMask::isIdentity() const {
  if (Indices.empty())
    return true;
  for (auto [Lane, Idx] : enumerate(Indices))
    if (Idx != (int)Lane)
      return false;
  return true;
}

bool sandboxir::ShuffleMask::isInOrder() const {
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

bool sandboxir::ShuffleMask::isIncreasingOrder() const {
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

bool sandboxir::ShuffleMask::operator==(
    const sandboxir::ShuffleMask &Other) const {
  return equal(Indices, Other.Indices);
}

#ifndef NDEBUG
void sandboxir::ShuffleMask::dump(raw_ostream &OS) const {
  for (auto [Lane, ShuffleIdx] : enumerate(Indices)) {
    if (Lane != 0)
      OS << ", ";
    OS << ShuffleIdx;
  }
}

void sandboxir::ShuffleMask::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

void sandboxir::ShuffleMask::verify() const {
  auto NumLanes = (int)Indices.size();
  for (auto Idx : Indices)
    assert(Idx < NumLanes && "Bad index!");
}
#endif // NDEBUG

void sandboxir::SBVecContext::SchedulerDeleter::operator()(
    sandboxir::Scheduler *Ptr) const {
  delete Ptr;
}

void sandboxir::SBVecContext::quickFlush() {
  InQuickFlush = true;
  SchedForBB.clear();

  RemoveInstrCallbacks.clear();
  InsertInstrCallbacks.clear();
  MoveInstrCallbacks.clear();

  RemoveInstrCallbacksBB.clear();
  InsertInstrCallbacksBB.clear();
  MoveInstrCallbacksBB.clear();

  Sched = nullptr;
  LLVMValueToValueMap.clear();
  MultiInstrMap.clear();
  InQuickFlush = false;
}

sandboxir::Scheduler *
sandboxir::SBVecContext::getScheduler(sandboxir::BasicBlock *SBBB) const {
  auto It = SchedForBB.find(SBBB);
  return It != SchedForBB.end() ? It->second.get() : nullptr;
}

const sandboxir::DependencyGraph &
sandboxir::SBVecContext::getDAG(sandboxir::BasicBlock *SBBB) const {
  return getScheduler(SBBB)->getDAG();
}

void sandboxir::SBVecContext::createdBasicBlock(sandboxir::BasicBlock &BB) {
  // Create a scheduler object for this particular BB.
  // Note: This should be done *after* we populate the BB.
  auto Pair = SchedForBB.try_emplace(
      &BB, std::unique_ptr<sandboxir::Scheduler, SchedulerDeleter>(
               new sandboxir::Scheduler(BB, AA, *this)));
  (void)Pair;
  assert(Pair.second && "Creating a scheduler for SBBB for the second time!");
}

#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
void sandboxir::SBVecContext::afterMoveInstrHook(sandboxir::BasicBlock &BB) {
  if (!getTracker().inRevert() && getScheduler(&BB) != nullptr)
    getScheduler(&BB)->getDAG().verify();
}
#endif

// Pack
sandboxir::PackInst *sandboxir::SBVecContext::createPackInst(
    const DmpVector<llvm::Value *> &PackInstrs) {
  assert(all_of(PackInstrs,
                [](llvm::Value *V) {
                  return isa<llvm::InsertElementInst>(V) ||
                         isa<llvm::ExtractElementInst>(V);
                }) &&
         "Expected inserts or extracts!");
  auto NewPtr = std::unique_ptr<sandboxir::PackInst>(
      new sandboxir::PackInst(PackInstrs, *this));
  return cast<sandboxir::PackInst>(registerValue(std::move(NewPtr)));
}

// Unpack
sandboxir::UnpackInst *sandboxir::SBVecContext::getUnpackInst(
    llvm::ExtractElementInst *ExtractI) const {
  auto *SBV = getValue(ExtractI);
  return SBV ? cast<sandboxir::UnpackInst>(SBV) : nullptr;
}

sandboxir::UnpackInst *
sandboxir::SBVecContext::createUnpackInst(
    llvm::ExtractElementInst *ExtractI) {
  assert(getUnpackInst(ExtractI) == nullptr && "Already exists!");
  auto *Op = getValue(ExtractI->getVectorOperand());
  assert(Op != nullptr &&
         "Please create the operand SBValue before calling this function!");
  llvm::Value *Idx = ExtractI->getIndexOperand();
  assert(isa<llvm::ConstantInt>(Idx) && "Can only handle constant int index!");
  auto Lane = cast<llvm::ConstantInt>(Idx)->getSExtValue();
  auto NewPtr = std::unique_ptr<sandboxir::UnpackInst>(
      new sandboxir::UnpackInst(ExtractI, Op, Lane, *this));
  assert(NewPtr->getOperand(0) == Op && "Bad operand!");
  return cast<sandboxir::UnpackInst>(
      registerValue(std::move(NewPtr)));
}

sandboxir::UnpackInst *
sandboxir::SBVecContext::getOrCreateUnpackInst(
    llvm::ExtractElementInst *ExtractI) {
  if (auto *Unpack = getUnpackInst(ExtractI))
    return Unpack;
  return createUnpackInst(ExtractI);
}

sandboxir::UnpackInst *sandboxir::SBVecContext::getUnpackInst(
    llvm::ShuffleVectorInst *ShuffleI) const {
  auto *SBV = getValue(ShuffleI);
  return SBV ? cast<sandboxir::UnpackInst>(SBV) : nullptr;
}

sandboxir::UnpackInst *
sandboxir::SBVecContext::createUnpackInst(
    llvm::ShuffleVectorInst *ShuffleI) {
  assert(getUnpackInst(ShuffleI) == nullptr && "Already exists!");
  auto *Op = getValue(ShuffleI->getOperand(1));
  assert(Op != nullptr &&
         "Please create the operand SBValue before calling this function!");
  auto Lane = sandboxir::UnpackInst::getShuffleLane(ShuffleI);
  auto NewPtr = std::unique_ptr<sandboxir::UnpackInst>(
      new sandboxir::UnpackInst(ShuffleI, Op, Lane, *this));
  assert(NewPtr->getOperand(0) == Op && "Bad operand!");
  return cast<sandboxir::UnpackInst>(
      registerValue(std::move(NewPtr)));
}

sandboxir::UnpackInst *
sandboxir::SBVecContext::getOrCreateUnpackInst(
    llvm::ShuffleVectorInst *ShuffleI) {
  if (auto *Unpack = getUnpackInst(ShuffleI))
    return Unpack;
  return createUnpackInst(ShuffleI);
}

// Shuffle
sandboxir::ShuffleInst *
sandboxir::SBVecContext::getShuffleInst(
    llvm::ShuffleVectorInst *ShuffleI) const {
  return cast_or_null<sandboxir::ShuffleInst>(getValue(ShuffleI));
}

sandboxir::ShuffleInst *
sandboxir::SBVecContext::createShuffleInst(
    llvm::ShuffleVectorInst *ShuffleI) {
  assert(getShuffleInst(ShuffleI) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<sandboxir::ShuffleInst>(
      new sandboxir::ShuffleInst(ShuffleI, *this));
  return cast<sandboxir::ShuffleInst>(
      registerValue(std::move(NewPtr)));
}

sandboxir::ShuffleInst *
sandboxir::SBVecContext::getOrCreateShuffleInst(
    llvm::ShuffleVectorInst *ShuffleI) {
  if (auto *Shuffle = getShuffleInst(ShuffleI))
    return Shuffle;
  return createShuffleInst(ShuffleI);
}

sandboxir::Value *sandboxir::SBVecContext::createValueFromExtractElement(
    llvm::ExtractElementInst *ExtractI, int Depth) {
  // Check that all indices are ConstantInts.
  if (!isa<llvm::ConstantInt>(ExtractI->getIndexOperand()))
    return getOrCreateExtractElementInst(ExtractI);
  getOrCreateValueInternal(ExtractI->getVectorOperand(), Depth + 1);
  // ExtractI could be a member of either Unpack or Pack from vectors.
  if (sandboxir::Value *Extract = getValue(ExtractI))
    return Extract;
  return createUnpackInst(ExtractI);
}

#ifndef NDEBUG
static std::optional<int> getPoisonVectorLanes(llvm::Value *V) {
  if (!isa<llvm::PoisonValue>(V))
    return std::nullopt;
  auto *Ty = V->getType();
  if (!isa<FixedVectorType>(Ty))
    return std::nullopt;
  return cast<FixedVectorType>(Ty)->getNumElements();
}
#endif

/// Checks if \p PackBottomInsert contains the last insert of an insert/extract
/// packing pattern, and if so returns the packed values in order, or an empty
/// vector, along with their operands.
// The simplest pattern is:
//   %i0 = insert poison, %v0, 0
//   %i1 = insert %i0,    %v1, 1
//   %i2 = insert %i1,    %v2, 2
//   ...
//   %iN = insert %iN-1,  %vN, N  ; <-- This is `PackBottomInsert`
static std::pair<DmpVector<llvm::Value *>, DmpVector<llvm::Value *>>
matchPackAndGetPackInstrs(llvm::InsertElementInst *PackBottomInsert) {
  // All instructions in the pattern must be in canonical form and back-to-back.
  // The canonical form rules:
  // 1. The bottom instruction must be an insert to the last lane.
  // 2. The rest of the inserts must have constant indexes in increasing order.
  // 3. If the pack pattern includes vector operands, the extracts from the
  //    vector are also part of the pattern. They must:
  //    a. have constant indexes in order, starting from 0 at the bottom.
  //    b. be positioned right before the insert instruction that uses it.
  //    c. have a single user: the pattern's insert.
  //    d. All extracts in the group extract from the same vector.
  // 4. No gaps (i.e. other instrs) between the instructions that form the pack.
  // 5. The topmost insert's vector value operand must be Poison.
  // 6. All pattern instrs (except the bottom most one) must have a single-use
  //    and it must be the next instruction after it and a member of the pattern
  // 7. Inserts should cover all lanes of the vector.
  // If the pattern is not in a canonical form, matching will fail.
  //
  // Example 1:
  //  %Pack0 = insertelement <2 x i32> poison, i32 %v0, i64 0
  //  %Pack1 = insertelement <2 x i32> %Pack0, i32 %v1, i64 1
  //
  // Example 2:
  //  %PackIns0 = insertelement <3 x i32> poison, i32 %v0, i32 0
  //  %PackExtr0 = extractelement <2 x i32> %vec, i32 0
  //  %PackIns1 = insertelement <3 x i32> %PackIns0, i32 %PackExtr0, i32 1
  //  %PackExtr1 = extractelement <2 x i32> %vec, i32 1
  //  %PackIns2 = insertelement <3 x i32> %PackIns1, i32 %PackExtr1, i32 2
  //
  // Example 3:
  //  %PackExtr0 = extractelement <2 x i32> %vec, i32 0
  //  %PackIns0 = insertelement <2 x i32> poison, i32 %PackExtr0, i32 0
  //  %PackExtr1 = extractelement <2 x i32> %vec, i32 1
  //  %PackIns1 = insertelement <2 x i32> %PackIns0, i32 %PackExtr1, i32 1
  //
  // Example 4:
  //  %Pack = insertelement <2 x i32> poison, i32 %v, i64 1
  int ExpectedExtractLane = 0;
  DmpVector<llvm::Value *> PackInstrs;
  int TotalLanes = sandboxir::VecUtils::getNumLanes(PackBottomInsert);
  int ExpectedInsertLane = TotalLanes - 1;
  llvm::InsertElementInst *LastInsert = nullptr;
  llvm::ExtractElementInst *LastExtractInGroup = nullptr;
  // Walk the chain bottom-up collecting the matched instrs into `PackInstrs`
  for (llvm::Instruction *CurrI = PackBottomInsert;
       CurrI != nullptr && (ExpectedInsertLane >= 0 || ExpectedExtractLane > 0);
       ExpectedInsertLane -= (isa<llvm::InsertElementInst>(CurrI) ? 1 : 0),
                         CurrI = CurrI->getPrevNode()) {
    // Checks for both Insert and Extract:
    bool IsAtBottom = PackInstrs.empty();
    if (IsAtBottom) {
      // The bottom instr must be an Insert (Rule 1).
      if (!isa<llvm::InsertElementInst>(CurrI))
        return {};
    } else {
      // If not the last instruction and it does not have a single user then
      // discard it (Rule 6).
      if (!CurrI->hasOneUse())
        return {};
      // Discard user is not the previous instr in the pattern (Rule 6).
      llvm::User *SingleUser = *CurrI->users().begin();
      if (SingleUser != LastInsert)
        return {};
      assert(isa<llvm::InsertElementInst>(SingleUser) &&
             "The user must be an Inset");
    }

    // We expect a constant lane that matches ExpectedInsertLane (Rules 1,2,7).
    if (auto InsertI = dyn_cast<llvm::InsertElementInst>(CurrI)) {
      auto LaneOpt = sandboxir::VecUtils::getConstantIndex(CurrI);
      if (!LaneOpt || *LaneOpt != ExpectedInsertLane)
        return {};
      assert(
          (IsAtBottom || cast<llvm::InsertElementInst>(*CurrI->users().begin())
                                 ->getOperand(0) == CurrI) &&
          "CurrI must be the user's vector operand!");
      LastInsert = InsertI;
    } else if (auto *ExtractI = dyn_cast<llvm::ExtractElementInst>(CurrI)) {
      // The extract's lane must be constant (Rule 3a).
      auto ExtractLaneOpt = sandboxir::VecUtils::getExtractLane(ExtractI);
      if (!ExtractLaneOpt)
        return {};
      int ExtractLanes =
          cast<FixedVectorType>(ExtractI->getVectorOperand()->getType())
              ->getNumElements();
      bool IsFirstExtractInGroup = ExpectedExtractLane == 0;
      ExpectedExtractLane =
          IsFirstExtractInGroup ? ExtractLanes - 1 : ExpectedExtractLane - 1;
      assert(ExpectedExtractLane >= 0 && "Bad calculation of expected lane");
      // The extract's lane must be in order (Rule 3a).
      if (*ExtractLaneOpt != ExpectedExtractLane)
        return {};
      // Make sure all extracts groups extract from the same vector (Rule 3d).
      if (LastExtractInGroup != nullptr &&
          ExtractI->getVectorOperand() !=
              LastExtractInGroup->getVectorOperand())
        return {};
      assert(!IsAtBottom && "An extract should not be at the pattern bottom!");
      assert(cast<llvm::InsertElementInst>(*CurrI->users().begin())
                     ->getOperand(1) == CurrI &&
             "CurrI must be the user's scalar operand!");
      LastExtractInGroup = ExpectedExtractLane > 0 ? ExtractI : nullptr;
    } else {
      // No non-pattern instructions allowed (Rule 4).
      return {};
    }

    // Collect CurrI, it looks good.
    PackInstrs.push_back(CurrI);
  }
  // Missing insert.
  if (ExpectedInsertLane != -1)
    return {};
  // Missing extract.
  if (ExpectedExtractLane != 0)
    return {};
#ifndef NDEBUG
  llvm::Instruction *TopI = cast<llvm::Instruction>(PackInstrs.back());
  assert((getPoisonVectorLanes(TopI->getOperand(0)) ||
          *sandboxir::VecUtils::getConstantIndex(TopI) == 0) &&
         "TopI is pointing to the wrong instruction!");
#endif // NDEBUG
  // If this is the top-most insert, its operand must be poison (Rule 5).
  if (!isa<llvm::PoisonValue>(LastInsert->getOperand(0)))
    return {};

  // Collect operands.
  DmpVector<llvm::Value *> Operands;
  for (unsigned Idx = 0, E = PackInstrs.size(); Idx < E; ++Idx) {
    llvm::Value *V = PackInstrs[Idx];
    if (isa<llvm::InsertElementInst>(V)) {
      Operands.push_back(cast<llvm::InsertElementInst>(V)->getOperand(1));
      continue;
    }
    assert(isa<llvm::ExtractElementInst>(V) && "Expected Extract!");
    auto *Extract = cast<llvm::ExtractElementInst>(V);
    llvm::Value *Op = Extract->getVectorOperand();
    Operands.push_back(Op);
    // Now we need to skip all Inserts and Extracts reading `Extract`.
    unsigned Skip = sandboxir::VecUtils::getNumLanes(Op) * 2 - 1;
    Idx += Skip;
  }
  return {PackInstrs, Operands};
}

sandboxir::Value *sandboxir::SBVecContext::createValueFromInsertElement(
    llvm::InsertElementInst *InsertI, int Depth) {
  if (auto *Insert = getValue(InsertI))
    return Insert;
  // Check if this is the bottom of an InsertElementInst packing pattern.
  auto [PackInstrs, PackOperands] = matchPackAndGetPackInstrs(InsertI);
  if (PackInstrs.empty())
    return getOrCreateInsertElementInst(InsertI);
  // Else create a new SBPackInstruction.
  return createPackInst(PackInstrs);
}

sandboxir::Value *sandboxir::SBVecContext::createValueFromShuffleVector(
    llvm::ShuffleVectorInst *ShuffleI, int Depth) {
  // Check that we are only using the first operand.
  // TODO: Is a poison/undef operand always 2nd operand when canonicalized?
  if (sandboxir::UnpackInst::isUnpack(ShuffleI)) {
    getOrCreateValueInternal(ShuffleI->getOperand(0), Depth + 1);
    getOrCreateValueInternal(ShuffleI->getOperand(1), Depth + 1);
    return getOrCreateUnpackInst(ShuffleI);
  }
  if (sandboxir::ShuffleInst::isShuffle(ShuffleI))
    return getOrCreateShuffleInst(ShuffleI);
  return getOrCreateShuffleVectorInst(ShuffleI);
}

sandboxir::PackInstrBundle::PackInstrBundle(
    const DmpVector<llvm::Value *> &PackInstrsBndl) {
  PackInstrs.reserve(PackInstrsBndl.size());
  copy(PackInstrsBndl.instrRange(), std::back_inserter(PackInstrs));
  // Sort in reverse program order.
  sort(PackInstrs, [](auto *I1, auto *I2) { return I2->comesBefore(I1); });
}

sandboxir::Value *sandboxir::PackInst::create(
    const DmpVector<sandboxir::Value *> &PackOps,
    sandboxir::BasicBlock::iterator WhereIt, sandboxir::BasicBlock *WhereBB,
    sandboxir::SBVecContext &SBCtx) {
  std::variant<DmpVector<llvm::Value *>, llvm::Constant *> BorC =
      sandboxir::PackInst::createIR(PackOps, WhereIt, WhereBB);
  // CreateIR packed constants which resulted in a single folded Constant.
  if (llvm::Constant **CPtr = std::get_if<llvm::Constant *>(&BorC))
    return SBCtx.getOrCreateValue(*CPtr);

  for (llvm::Value *V : std::get<DmpVector<llvm::Value *>>(BorC))
    SBCtx.createMissingConstantOperands(V);
  // If we created Instructions then create and return a Pack.
  sandboxir::Value *NewSBV =
      SBCtx.createPackInst(std::get<DmpVector<llvm::Value *>>(BorC));
  return NewSBV;
}

sandboxir::Value *sandboxir::PackInst::create(
    const DmpVector<sandboxir::Value *> &PackOps,
    sandboxir::Instruction *InsertBefore, sandboxir::SBVecContext &SBCtx) {
  return create(PackOps, InsertBefore->getIterator(), InsertBefore->getParent(),
                SBCtx);
}

sandboxir::Value *sandboxir::PackInst::create(
    const DmpVector<sandboxir::Value *> &PackOps,
    sandboxir::BasicBlock *InsertAtEnd, sandboxir::SBVecContext &SBCtx) {
  return create(PackOps, InsertAtEnd->end(), InsertAtEnd, SBCtx);
}

sandboxir::Use
sandboxir::PackInst::getOperandUseInternal(unsigned OperandIdx,
                                                    bool Verify) const {
  assert((!Verify || OperandIdx < getNumOperands()) && "Out of bounds!");
  llvm::Use &LLVMUse = PackInstrBundle::getBndlOperandUse(OperandIdx);
  return sandboxir::Use(
      &LLVMUse, const_cast<sandboxir::PackInst *>(this), Ctx);
}

bool sandboxir::PackInst::isRealOperandUse(llvm::Use &OpUse) const {
  bool IsReal = false;
  bool Found = true;
  doOnOperands([&OpUse, &IsReal, &Found](llvm::Use &Use, bool IsRealOp) {
    if (&Use != &OpUse)
      return false; // Don't break
    IsReal = IsRealOp;
    Found = true;
    return true; // Break
  });
  assert(Found && "OpUse not found!");
  return IsReal;
}

void sandboxir::PackInst::setOperand(unsigned OperandIdx,
                                              sandboxir::Value *Operand) {
  assert(OperandIdx < getNumOperands() && "Out of bounds!");
  assert(Operand->getType() ==
             sandboxir::User::getOperand(OperandIdx)->getType() &&
         "Operand of wrong type!");
  llvm::Value *NewOp = ValueAttorney::getValue(Operand);
  unsigned RealOpIdx = 0;
  doOnOperands([NewOp, OperandIdx, &RealOpIdx, this](llvm::Use &Use,
                                                     bool IsRealOp) {
    if (RealOpIdx == OperandIdx) {
      auto &Tracker = getTracker();
      if (Tracker.tracking())
        Tracker.track(std::make_unique<SetOperand>(this, OperandIdx, Tracker));
      Use.set(NewOp);
    }
    if (IsRealOp)
      ++RealOpIdx;
    // Break once we are done updating the operands.
    if (RealOpIdx > OperandIdx)
      return true; // Break
    return false;  // Don't break
  });
}

InsertElementInst *sandboxir::PackInst::getBottomInsert(
    const DmpVector<llvm::Value *> &Instrs) const {
  // Get the bottom insert by removing the vector operands from the set until we
  // have only the bottom left.
  DenseSet<llvm::Value *> AllPackInstrs(Instrs.begin(), Instrs.end());
  for (auto *PackI : Instrs) {
    assert((isa<llvm::InsertElementInst>(PackI) ||
            isa<llvm::ExtractElementInst>(PackI)) &&
           "Expected Insert or Extract");
    AllPackInstrs.erase(cast<llvm::Instruction>(PackI)->getOperand(0));
    AllPackInstrs.erase(cast<llvm::Instruction>(PackI)->getOperand(1));
  }
  assert(AllPackInstrs.size() == 1 && "Unexpected pack instruction structure");
  return cast<llvm::InsertElementInst>(*AllPackInstrs.begin());
}

bool sandboxir::PackInst::classof(const sandboxir::Value *From) {
  return From->getSubclassID() == ClassID::Pack;
}

std::variant<DmpVector<llvm::Value *>, llvm::Constant *>
sandboxir::PackInst::createIR(
    const DmpVector<sandboxir::Value *> &ToPack,
    sandboxir::BasicBlock::iterator WhereIt,
    sandboxir::BasicBlock *WhereBB) {
  // A Pack should be placed after the latest packed value.
  auto &LLVMIRBuilder = WhereBB->getContext().getLLVMIRBuilder();
  auto *LLVMBB = sandboxir::BasicBlockAttorney::getBB(WhereBB);
  if (WhereIt != WhereBB->end())
    LLVMIRBuilder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    LLVMIRBuilder.SetInsertPoint(LLVMBB);

  Type *ScalarTy = sandboxir::VecUtils::getCommonScalarTypeFast(ToPack);
  unsigned Lanes = sandboxir::VecUtils::getNumLanes(ToPack);
  auto *VecTy = sandboxir::VecUtils::getWideType(ScalarTy, Lanes);

  // Create a series of pack instructions.
  DmpVector<llvm::Value *> AllPackInstrs;
  llvm::Value *LastInsert = PoisonValue::get(VecTy);

  auto Collect = [&AllPackInstrs](llvm::Value *NewV) {
    assert(isa<llvm::Instruction>(NewV) && "Expected instruction!");
    auto *I = cast<llvm::Instruction>(NewV);
    AllPackInstrs.push_back(I);
  };

  unsigned InsertIdx = 0;
  for (sandboxir::Value *SBV : ToPack) {
    llvm::Value *Elm = ValueAttorney::getValue(SBV);
    if (Elm->getType()->isVectorTy()) {
      unsigned NumElms =
          cast<FixedVectorType>(Elm->getType())->getNumElements();
      for (auto ExtrLane : seq<int>(0, NumElms)) {
        // This may return a Constant if Elm is a Constant.
        auto *ExtrI =
            LLVMIRBuilder.CreateExtractElement(Elm, ExtrLane, "XPack");
        if (auto *ExtrC = dyn_cast<llvm::Constant>(ExtrI))
          WhereBB->getContext().getOrCreateConstant(ExtrC);
        else
          Collect(ExtrI);
        // This may also return a Constant if ExtrI is a Constant.
        LastInsert = LLVMIRBuilder.CreateInsertElement(LastInsert, ExtrI,
                                                       InsertIdx++, "Pack");
        if (auto *C = dyn_cast<llvm::Constant>(LastInsert)) {
          if (InsertIdx == Lanes)
            return C;
          WhereBB->getContext().getOrCreateValue(C);
        } else
          Collect(LastInsert);
      }
    } else {
      // This may be folded into a Constant if LastInsert is a Constant. In that
      // case we only collect the last constant.
      LastInsert = LLVMIRBuilder.CreateInsertElement(LastInsert, Elm,
                                                     InsertIdx++, "Pack");
      if (auto *C = dyn_cast<llvm::Constant>(LastInsert)) {
        if (InsertIdx == Lanes)
          return C;
        WhereBB->getContext().getOrCreateValue(C);
      } else
        Collect(LastInsert);
    }
  }
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  WhereBB->verifyLLVMIR();
#endif
  return AllPackInstrs;
}

llvm::Use &sandboxir::PackInstrBundle::getExternalFacingOperandUse(
    llvm::InsertElementInst *InsertI) const {
  // Get the Insert's edge and check if its source is a Pack Extract. If it is,
  // then don't use the Insert's edge, but rather the Extract's edge.
  llvm::Use &OpUse = InsertI->getOperandUse(1);
  llvm::Value *Op = OpUse.get();
  if (!isa<llvm::ExtractElementInst>(Op) ||
      find(PackInstrs, cast<llvm::ExtractElementInst>(Op)) == PackInstrs.end())
    return OpUse;
  // This is an extract used in the pack-from-vector pattern.
  return cast<llvm::ExtractElementInst>(Op)->getOperandUse(0);
}

llvm::InsertElementInst *
sandboxir::PackInstrBundle::getInsertAtLane(int Lane) const {
  auto It = find_if(PackInstrs, [Lane](llvm::Value *V) {
    return isa<llvm::InsertElementInst>(V) &&
           *sandboxir::VecUtils::getInsertLane(
               cast<llvm::InsertElementInst>(V)) == Lane;
  });
  return It != PackInstrs.end() ? cast<llvm::InsertElementInst>(*It) : nullptr;
}

llvm::InsertElementInst *sandboxir::PackInstrBundle::getTopInsert() const {
  auto Range = reverse(PackInstrs);
  auto It =
      find_if(Range, [](auto *I) { return isa<llvm::InsertElementInst>(I); });
  assert(It != Range.end() && "Not found!");
  return cast<llvm::InsertElementInst>(*It);
}

llvm::InsertElementInst *sandboxir::PackInstrBundle::getBotInsert() const {
  auto It = find_if(PackInstrs,
                    [](auto *I) { return isa<llvm::InsertElementInst>(I); });
  assert(It != PackInstrs.end() && "Not found!");
  return cast<llvm::InsertElementInst>(*It);
}

static bool isSingleUseEdge(llvm::Use &ExtFacingUse) {
  return !isa<llvm::ExtractElementInst>(ExtFacingUse.getUser());
}

static bool isLastOfMultiUseEdge(llvm::Use &ExtFacingUse) {
  llvm::User *U = ExtFacingUse.getUser();
  assert(isa<llvm::ExtractElementInst>(U) &&
         "A multi-Use edge must have an Extract operand!");
  // If the user is not an extract, then this is a single-Use edge.
  auto *ExtractI = cast<llvm::ExtractElementInst>(U);
  auto ExtrIdx = *sandboxir::VecUtils::getExtractLane(ExtractI);
  int Lanes = cast<FixedVectorType>(ExtractI->getVectorOperand()->getType())
                  ->getNumElements();
  return ExtrIdx == Lanes - 1;
}

void sandboxir::PackInstrBundle::doOnOperands(
    function_ref<bool(llvm::Use &, bool)> DoOnOpFn) const {
  // Constant operands may be folded into the poison vector, or poison operands
  // can also be folded into a single vector poison value.
  auto *TopInsertI = getTopInsert();
  auto *PoisonVal = cast<llvm::Constant>(TopInsertI->getOperand(0));
  auto *PoisonConstantVec = dyn_cast<llvm::ConstantVector>(PoisonVal);
  auto Lanes = cast<FixedVectorType>(PoisonVal->getType())->getNumElements();
  assert((PoisonConstantVec == nullptr ||
          PoisonConstantVec->getNumOperands() == Lanes) &&
         "Bad Lanes or PoisonConstantVec!");
  for (auto Lane : seq<unsigned>(0, Lanes)) {
    llvm::InsertElementInst *InsertI = getInsertAtLane(Lane);
    // A missing insert means that the operand was folded into the poison vector
    llvm::Use *OpUsePtr = nullptr;
    if (InsertI != nullptr) {
      OpUsePtr = &getExternalFacingOperandUse(InsertI);
    } else if (PoisonConstantVec != nullptr) {
      OpUsePtr = &PoisonConstantVec->getOperandUse(Lane);
    } else {
      auto *TopInsertI = cast<llvm::InsertElementInst>(PackInstrs.front());
      OpUsePtr = &TopInsertI->getOperandUse(0);
    }
    llvm::Use &OpUse = *OpUsePtr;
    // insert %val0  0  <- Single edge
    // insert %extr0 1
    // insert %extr0 2  <- Last of 2-wide Multi-edge
    // insert %val1  3  <- Single edge
    bool IsOnlyOrLastUse =
        isSingleUseEdge(OpUse) || isLastOfMultiUseEdge(OpUse);
    bool Break = DoOnOpFn(OpUse, IsOnlyOrLastUse);
    if (Break)
      break;
  }
}

unsigned sandboxir::PackInst::getOperandUseIdx(
    const llvm::Use &UseToMatch) const {
#ifndef NDEBUG
  verifyUserOfLLVMUse(UseToMatch);
#endif
  unsigned RealOpIdx = 0;
  bool Found = false;
  doOnOperands(
      [&UseToMatch, &RealOpIdx, &Found](llvm::Use &Use, bool IsRealOp) {
        if (&Use == &UseToMatch) {
          Found = true;
          return true; // Ask to break
        }
        if (IsRealOp)
          ++RealOpIdx;
        return false; // Don't break
      });
  assert(Found && "Use not found in external facing operands!");
  return RealOpIdx;
}

llvm::Use &
sandboxir::PackInstrBundle::getBndlOperandUse(unsigned OperandIdx) const {
  unsigned RealOpIdx = 0;
  // Special case for op_end().
  if (OperandIdx == getNumOperands())
    return *getBotInsert()->op_end();

  llvm::Use *ReturnUse = nullptr;
  doOnOperands(
      [&RealOpIdx, &ReturnUse, OperandIdx](llvm::Use &Use, bool IsRealOp) {
        if (IsRealOp) {
          if (RealOpIdx == OperandIdx) {
            ReturnUse = &Use;
            return true; // Ask to break
          }
          ++RealOpIdx;
        }
        return false; // Don't break
      });
  assert(ReturnUse != nullptr && "Expected non-null operand!");
  return *ReturnUse;
}

unsigned sandboxir::PackInstrBundle::getNumOperands() const {
  // Not breaking for any operand will give us the total number of operands.
  unsigned RealOpCnt = 0;
  doOnOperands([&RealOpCnt](llvm::Use &Use, bool IsRealOp) {
    if (IsRealOp)
      ++RealOpCnt;
    return false; // Don't break
  });
  return RealOpCnt;
}

#ifndef NDEBUG
void sandboxir::PackInstrBundle::verifyInstrBundle() const {
  // Make sure that the consecutive Extracts that make up the
  // pack-from-vector pattern have the same operand. This could break during
  // a SBPackInstruction::setOperand() operation.
  llvm::ExtractElementInst *LastExtractI = nullptr;
  doOnOperands([&LastExtractI](llvm::Use &Use, bool IsRealOp) -> bool {
    if (IsRealOp && LastExtractI != nullptr) {
      // We expect an extract that extracts from the same vector as
      // LastExtractI, but the next lane.
      assert(isa<llvm::ExtractElementInst>(Use.getUser()) && "Expect Extract");
      auto *ExtractI = cast<llvm::ExtractElementInst>(Use.getUser());
      assert(Use.get() == ExtractI->getVectorOperand() && "Sanity check");
      // Skip <1 x type>
      if (cast<FixedVectorType>(ExtractI->getVectorOperand()->getType())
              ->getNumElements() != 1u) {
        assert(ExtractI->getVectorOperand() ==
                   LastExtractI->getVectorOperand() &&
               "Most likely setOperand() did not update all Extracts!");
        assert(ExtractI->getIndexOperand() != LastExtractI->getIndexOperand() &&
               "Expected different indices");
      }
      LastExtractI = nullptr;
    } else {
      LastExtractI = dyn_cast<llvm::ExtractElementInst>(Use.getUser());
    }
    return false;
  });
}
#endif

sandboxir::PackInst::PackInst(const DmpVector<llvm::Value *> &Instrs,
                              sandboxir::Context &SBCtx)
    : PackInstrBundle(Instrs),
      sandboxir::Instruction(ClassID::Pack, Opcode::Pack,
                             getBottomInsert(Instrs), SBCtx) {
  assert(all_of(PackInstrs,
                [](llvm::Value *V) {
                  return isa<llvm::InsertElementInst>(V) ||
                         isa<llvm::ExtractElementInst>(V);
                }) &&
         "Expected inserts or extracts!");
#ifndef NDEBUG
  for (auto Idx : seq<unsigned>(1, PackInstrs.size()))
    assert(this->PackInstrs[Idx]->comesBefore(this->PackInstrs[Idx - 1]) &&
           "Expecte reverse program order!");
  assert(all_of(drop_begin(this->PackInstrs),
                [this](auto *I) {
                  return I->comesBefore(cast<llvm::Instruction>(Val));
                }) &&
         "Val should be the bottom instruction!");
#endif
}

DmpVector<llvm::Instruction *>
sandboxir::PackInst::getLLVMInstrs() const {
#ifndef NDEBUG
  for (auto Idx : seq<unsigned>(1, PackInstrs.size())) {
    auto *I1 = PackInstrs[Idx - 1];
    auto *I2 = PackInstrs[Idx];
    assert(((!I1->getParent() || I2->getParent()) || I2->comesBefore(I1)) &&
           "Expected reverse program order!");
  }
#endif
  // TODO: Perhaps change the order in PackInstrs?
  auto Range = reverse(PackInstrs);
  DmpVector<llvm::Instruction *> PackInstrsInProgramOrder(Range.begin(),
                                                          Range.end());
  return PackInstrsInProgramOrder;
}

DmpVector<llvm::Instruction *>
sandboxir::PackInst::getLLVMInstrsWithExternalOperands() const {
  SmallVector<llvm::Instruction *> IRInstrs;
  for (llvm::Instruction *I : PackInstrs) {
    if (auto *InsertI = dyn_cast<llvm::InsertElementInst>(I)) {
      // If this is an internal insert, it must have an Extract operand, which
      // is the external facing IR instruction.
      auto *ExtractOp =
          dyn_cast<llvm::ExtractElementInst>(InsertI->getOperand(1));
      if (ExtractOp != nullptr &&
          find(PackInstrs, ExtractOp) != PackInstrs.end()) {
        // ExtractOp is the out-facing instruction, not the insert.
        IRInstrs.push_back(ExtractOp);
      } else {
        // This is an external-facing Insert.
        IRInstrs.push_back(InsertI);
      }
    }
  }
  return IRInstrs;
}

unsigned sandboxir::PackInst::getUseOperandNo(
    const sandboxir::Use &SBUse) const {
  unsigned OpNo = 0;
  llvm::Use *UseToMatch = SBUse.LLVMUse;
  doOnOperands([&OpNo, UseToMatch](llvm::Use &LLVMUse, bool IsRealOp) -> bool {
    if (&LLVMUse == UseToMatch)
      return true; // break
    if (IsRealOp)
      ++OpNo;
    return false; // don't break
  });
  return OpNo;
}

sandboxir::User::op_iterator sandboxir::PackInst::op_begin() {
  return op_iterator(getOperandUseInternal(0, /*Verify=*/false));
}

sandboxir::User::op_iterator sandboxir::PackInst::op_end() {
  return op_iterator(getOperandUseInternal(getNumOperands(), /*Verify=*/false));
}

sandboxir::User::const_op_iterator
sandboxir::PackInst::op_begin() const {
  return const_cast<sandboxir::PackInst *>(this)->op_begin();
}

sandboxir::User::const_op_iterator
sandboxir::PackInst::op_end() const {
  return const_cast<sandboxir::PackInst *>(this)->op_end();
}

void sandboxir::PackInst::detachExtras() {
  auto *PackV = ValueAttorney::getValue(this);
  for (auto *PI : getPackInstrs())
    if (PI != PackV) // Skip the bottom value, gets detached later
      Ctx.detachValue(PI);
}

#ifndef NDEBUG
void sandboxir::PackInst::verify() const {
  if (any_of(operands(), [](sandboxir::Value *Op) {
        return sandboxir::VecUtils::isVector(Op);
      }))
    assert((isa<FixedVectorType>(getOperand(0)->getType()) ||
            getNumOperands() < cast<FixedVectorType>(
                                   sandboxir::VecUtils::getExpectedType(this))
                                   ->getNumElements()) &&
           "This has vector operands. We expect fewer operands than lanes");
  verifyInstrBundle();
}

void sandboxir::PackInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  OS.indent(2) << "PackInstrs:\n";
  for (auto *I : PackInstrs)
    OS.indent(2) << *I << "\n";
  dumpCommonFooter(OS);
}
void sandboxir::PackInst::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}

void sandboxir::PackInst::dump(raw_ostream &OS) const {
  // Sort pack instructions in program order.
  auto SortedInstrs = getLLVMInstrs();
  if (all_of(SortedInstrs, [](llvm::Instruction *I) { return I->getParent(); }))
    sort(SortedInstrs, [](auto *I1, auto *I2) { return I1->comesBefore(I2); });
  else
    OS << "** Error: Not all IR Instrs have a parent! **";

  auto ExtFacing = getLLVMInstrsWithExternalOperands();
  unsigned NumOperands = getNumOperands();

  for (auto [Idx, I] : enumerate(SortedInstrs)) {
    OS << *I;
    dumpCommonSuffix(OS);
    // Print the lane.
    bool IsExt = find(ExtFacing, I) != ExtFacing.end();
    if (IsExt) {
      llvm::Use *OpUse = nullptr;
      if (auto *InsertI = dyn_cast<llvm::InsertElementInst>(I)) {
        OpUse = &InsertI->getOperandUse(1);
      } else if (auto *ExtractI = dyn_cast<llvm::ExtractElementInst>(I)) {
        OpUse = &ExtractI->getOperandUse(0);
      }
      OS << " OpIdx=";
      if (OpUse == nullptr)
        OS << "** ERROR: Can't get OpIdx! **";
      else
        OS << getOperandUseIdx(*OpUse) << "/" << NumOperands - 1 << " ";
    }
    OS << (Idx + 1 != SortedInstrs.size() ? "\n" : "");
  }
}
void sandboxir::PackInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

sandboxir::ShuffleInst::ShuffleInst(sandboxir::Value *Op,
                                    const ShuffleMask &Mask,
                                    sandboxir::BasicBlock::iterator WhereIt,
                                    sandboxir::BasicBlock *WhereBB)
    : sandboxir::ShuffleInst(createIR(Op, Mask, WhereIt, WhereBB),
                             WhereBB->getContext()) {
  Ctx.createMissingConstantOperands(Val);
  assert(Val != nullptr && "Shuffle was folded!");
}

sandboxir::ShuffleInst *sandboxir::ShuffleInst::create(
    sandboxir::Value *Op, ShuffleMask &Mask,
    sandboxir::BasicBlock::iterator WhereIt, sandboxir::BasicBlock *WhereBB,
    sandboxir::SBVecContext &SBCtx) {
  auto NewPtr = std::unique_ptr<sandboxir::ShuffleInst>(
      new sandboxir::ShuffleInst(Op, Mask, WhereIt, WhereBB));
  return cast<sandboxir::ShuffleInst>(
      SBCtx.registerValue(std::move(NewPtr)));
}

sandboxir::ShuffleInst *sandboxir::ShuffleInst::create(
    sandboxir::Value *Op, ShuffleMask &Mask,
    sandboxir::Instruction *InsertBefore, sandboxir::SBVecContext &SBCtx) {
  return sandboxir::ShuffleInst::create(
      Op, Mask, InsertBefore->getIterator(), InsertBefore->getParent(), SBCtx);
}

sandboxir::ShuffleInst *sandboxir::ShuffleInst::create(
    sandboxir::Value *Op, ShuffleMask &Mask,

    sandboxir::BasicBlock *InsertAtEnd, sandboxir::SBVecContext &SBCtx) {
  return sandboxir::ShuffleInst::create(Op, Mask, InsertAtEnd->end(),
                                                 InsertAtEnd, SBCtx);
}

sandboxir::User::op_iterator sandboxir::ShuffleInst::op_begin() {
  return op_iterator(getOperandUseInternal(0, /*Verify=*/false));
}
sandboxir::User::op_iterator sandboxir::ShuffleInst::op_end() {
  return op_iterator(getOperandUseInternal(getNumOperands(), /*Verify=*/false));
}
sandboxir::User::const_op_iterator
sandboxir::ShuffleInst::op_begin() const {
  return const_cast<sandboxir::ShuffleInst *>(this)->op_begin();
}
sandboxir::User::const_op_iterator
sandboxir::ShuffleInst::op_end() const {
  return const_cast<sandboxir::ShuffleInst *>(this)->op_end();
}

void sandboxir::ShuffleInst::setOperand(unsigned OperandIdx,
                                                 sandboxir::Value *Operand) {
  assert(OperandIdx == 0 && "A SBShuffleInstruction has exactly 1 operand!");
  sandboxir::User::setOperand(OperandIdx, Operand);
}

bool sandboxir::ShuffleInst::classof(const sandboxir::Value *From) {
  return From->getSubclassID() == ClassID::Shuffle;
}

ShuffleVectorInst *sandboxir::ShuffleInst::createIR(
    sandboxir::Value *Op, const ShuffleMask &Mask,
    sandboxir::BasicBlock::iterator WhereIt,
    sandboxir::BasicBlock *WhereBB) {
  llvm::Value *Vec = ValueAttorney::getValue(Op);
  auto &LLVMIRBuilder = WhereBB->getContext().getLLVMIRBuilder();
  auto *LLVMBB = sandboxir::BasicBlockAttorney::getBB(WhereBB);
  if (WhereIt != WhereBB->end())
    LLVMIRBuilder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    LLVMIRBuilder.SetInsertPoint(LLVMBB);
  auto *Shuffle = cast_or_null<llvm::ShuffleVectorInst>(
      LLVMIRBuilder.CreateShuffleVector(Vec, Mask, "Shuf"));
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  WhereBB->verifyLLVMIR();
#endif
  return Shuffle;
}

#ifndef NDEBUG
void sandboxir::ShuffleInst::verify() const {
  assert(getMask().size() == sandboxir::VecUtils::getNumLanes(this) &&
         "Expected same number of indices as lanes.");
  assert((int)sandboxir::VecUtils::getNumLanes(this) ==
             sandboxir::VecUtils::getNumLanes(getOperand(0)->getType()) &&
         "A SBShuffle should not unpack, it should only reorder lanes!");
  getMask().verify();
  assert(getNumOperands() == 1 && "Expected a single operand");
}
void sandboxir::ShuffleInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "  " << getMask() << "\n";
  dumpCommonFooter(OS);
}
void sandboxir::ShuffleInst::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
void sandboxir::ShuffleInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
  OS << " ; " << getMask();
}
void sandboxir::ShuffleInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

sandboxir::UnpackInst::UnpackInst(llvm::ExtractElementInst *ExtractI,
                                  sandboxir::Value *UnpackOp,
                                  unsigned UnpackLane,
                                  sandboxir::Context &SBCtx)
    : sandboxir::Instruction(ClassID::Unpack, Opcode::Unpack, ExtractI,
                             SBCtx) {
  assert(getOperand(0) == UnpackOp && "Bad operand!");
}

sandboxir::UnpackInst::UnpackInst(llvm::ShuffleVectorInst *ShuffleI,
                                  sandboxir::Value *UnpackOp,
                                  unsigned UnpackLane,
                                  sandboxir::Context &SBCtx)
    : sandboxir::Instruction(ClassID::Unpack, Opcode::Unpack, ShuffleI,
                             SBCtx) {
  assert(getOperand(0) == UnpackOp && "Bad operand!");
}

sandboxir::Value *sandboxir::UnpackInst::create(
    sandboxir::Value *Op, unsigned UnpackLane, unsigned NumLanesToUnpack,
    sandboxir::BasicBlock::iterator WhereIt, sandboxir::BasicBlock *WhereBB,
    sandboxir::SBVecContext &SBCtx) {
  llvm::Value *V = sandboxir::UnpackInst::createIR(
      Op, UnpackLane, NumLanesToUnpack, WhereIt, WhereBB);
  SBCtx.createMissingConstantOperands(V);
  auto *NewSBV = SBCtx.getOrCreateValue(V);
  return NewSBV;
}

Value *sandboxir::UnpackInst::createIR(
    sandboxir::Value *UnpackOp, unsigned Lane, unsigned Lanes,
    sandboxir::BasicBlock::iterator WhereIt,
    sandboxir::BasicBlock *WhereBB) {
  auto &LLVMIRBuilder = WhereBB->getContext().getLLVMIRBuilder();
  llvm::Value *OpVec = ValueAttorney::getValue(UnpackOp);
  if (WhereIt != WhereBB->end())
    LLVMIRBuilder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    LLVMIRBuilder.SetInsertPoint(
        sandboxir::BasicBlockAttorney::getBB(WhereBB));
  llvm::Value *Unpack;
  if (Lanes == 1) {
    // If we are unpacking a scalar, we can use an ExtractElementInst.
    Unpack = LLVMIRBuilder.CreateExtractElement(OpVec, Lane, "Unpack");
  } else {
    // If we are unpacking a vector element, we need to use a Shuffle.
    ShuffleMask::IndicesVecT ShuffleIndices;
    ShuffleIndices.reserve(Lanes);
    for (auto Ln : seq<unsigned>(Lane, Lane + Lanes))
      ShuffleIndices.push_back(Ln);
    ShuffleMask Mask(std::move(ShuffleIndices));
    Unpack = LLVMIRBuilder.CreateShuffleVector(OpVec, Mask, "Unpack");
  }
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  WhereBB->verifyLLVMIR();
#endif
  return Unpack;
}

sandboxir::Value *sandboxir::UnpackInst::create(
    sandboxir::Value *Op, unsigned UnpackLane, unsigned NumLanesToUnpack,
    sandboxir::Instruction *InsertBefore, sandboxir::SBVecContext &SBCtx) {
  return sandboxir::UnpackInst::create(
      Op, UnpackLane, NumLanesToUnpack, InsertBefore->getIterator(),
      InsertBefore->getParent(), SBCtx);
}

sandboxir::Value *sandboxir::UnpackInst::create(
    sandboxir::Value *Op, unsigned UnpackLane, unsigned NumLanesToUnpack,
    sandboxir::BasicBlock *InsertAtEnd, sandboxir::SBVecContext &SBCtx) {
  return sandboxir::UnpackInst::create(
      Op, UnpackLane, NumLanesToUnpack, InsertAtEnd->end(), InsertAtEnd,
      SBCtx);
}

bool sandboxir::UnpackInst::classof(const sandboxir::Value *From) {
  return From->getSubclassID() == ClassID::Unpack;
}

#ifndef NDEBUG
void sandboxir::UnpackInst::verify() const {
  // TODO:
}
void sandboxir::UnpackInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << " " << "lane:" << getUnpackLane() << "\n";
  dumpCommonFooter(OS);
}
void sandboxir::UnpackInst::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
void sandboxir::UnpackInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
  OS << " lane:" << getUnpackLane();
}
void sandboxir::UnpackInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif
