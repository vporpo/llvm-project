//===- SandboxIRTracker.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIRTracker.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Vectorize/SandboxVec/InstrRange.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"
#include <sstream>

using namespace llvm;

#ifndef NDEBUG
std::string IRChecker::dumpIR(BasicBlock *BB) const {
  std::string TmpStr;
  raw_string_ostream SS(TmpStr);
  BB->print(SS, /*AssemblyAnnotationWriter=*/nullptr);
  return TmpStr;
}

void IRChecker::save() { OrigIR = dumpIR(BB); }

void IRChecker::stripMetadata(std::string &Line) const {
  std::smatch Match;
  // Erase ', !sb !42'
  if (std::regex_search(Line, Match, SBMDRegex)) {
    assert(Match.size() == 1 && "Expected exactly one region MD match!");
    auto Begin = Line.begin() + Match.prefix().length();
    auto End = Begin + Match.str().length();
    Line.erase(Begin, End);
  }
  // Erase '!42'
  for (std::smatch Match; std::regex_search(Line, Match, MDNumRegex);) {
    auto Begin = Line.begin() + Match.prefix().length();
    auto End = Begin + Match.str().length();
    Line.erase(Begin, End);
  }
}

bool IRChecker::diff(const std::string &OrigIR,
                     const std::string &CurrIR) const {
  bool Differ = false;
  // Show the first line that differes.
  std::stringstream OrigSS(OrigIR);
  std::stringstream CurrSS(CurrIR);
  std::string OrigLine;
  std::string CurrLine;
  SmallVector<std::string> Context;
  static constexpr const uint32_t MaxContext = 3;
  while (OrigSS.good() && CurrSS.good()) {
    std::getline(OrigSS, OrigLine);
    std::getline(CurrSS, CurrLine);
    std::string OrigLineStripped(OrigLine);
    std::string CurrLineStripped(CurrLine);
    // Strip metadata to avoid false positives.
    // This removes the !sb region metadata and all !<num>.
    stripMetadata(OrigLineStripped);
    stripMetadata(CurrLineStripped);
    if (CurrLineStripped != OrigLineStripped) {
      Differ = true;
      // Print context.
      for (const std::string &ContextLine : Context)
        dbgs() << "  " << ContextLine << "\n";
      // Print the line difference.
      dbgs() << "- " << OrigLine << "\n";
      dbgs() << "+ " << CurrLine << "\n";
    } else {
      // Lazy way to maintain context. Performance of this code does not matter.
      Context.push_back(OrigLine);
      if (Context.size() > MaxContext)
        Context.erase(Context.begin());
    }
  }
  // If one file is larger than the other print line in the larger one.
  if (!OrigSS.good() && CurrSS.good()) {
    Differ = true;
    std::getline(CurrSS, CurrLine);
    dbgs() << "+ " << CurrLine << "\n";
  }
  if (OrigSS.good() && !CurrSS.good()) {
    Differ = true;
    std::getline(OrigSS, OrigLine);
    dbgs() << "+ " << OrigLine << "\n";
  }
  return Differ;
}

void IRChecker::expectNoDiff() const {
  std::string CurrIR = dumpIR(BB);
  assert(!OrigIR.empty() && "OrigIR not created!");
  if (diff(OrigIR, CurrIR)) {
    llvm_unreachable(
        "Original and current IR differ! Possibly a Checkpointing bug.");
  }
}
#endif

IRChangeBase::IRChangeBase(TrackID ID, SandboxIRTracker &Parent)
    : ID(ID), Parent(Parent) {
#ifndef NDEBUG
  Idx = Parent.size();

  assert(!Parent.InMiddleOfCreatingChange &&
         "We are in the middle of creating another change!");
  if (Parent.tracking())
    Parent.InMiddleOfCreatingChange = true;
#endif // NDEBUG
}

bool IRChangeBase::isCompulsory() const {
  switch (ID) {
  case TrackID::DeleteOnAccept:
    return true;
  case TrackID::InstrRemoveFromParent:
  case TrackID::InsertToBB:
  case TrackID::MoveInstr:
  case TrackID::ReplaceAllUsesWith:
  case TrackID::ReplaceUsesOfWith:
  case TrackID::SetOperand:
  case TrackID::CreateAndInsertInstr:
  case TrackID::EraseFromParent:
  case TrackID::ClearDAGRange:
    return false;
  }
}

DeleteOnAccept::DeleteOnAccept(SBInstruction *SBI, SandboxIRTracker &Tracker)
    : IRChangeBase(TrackID::DeleteOnAccept, Tracker),
      Instrs(SBI->getIRInstrs()) {
  auto *BotI = cast<Instruction>(ValueAttorney::getValue(SBI));
  BeforeI = BotI->getNextNode();
  BB = BotI->getParent();
}

void DeleteOnAccept::apply() {
  for (auto *I : Instrs)
    I->deleteValue();
}

#ifndef NDEBUG
void DeleteOnAccept::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

InstrRemoveFromParent::InstrRemoveFromParent(SBInstruction *I,
                                             SandboxIRTracker &Tracker)
    : IRChangeBase(TrackID::InstrRemoveFromParent, Tracker), I(I),
      BB(I->getParent()), BeforeI(I->getNextNode()) {}

void InstrRemoveFromParent::revert() {
  if (BeforeI)
    I->insertBefore(BeforeI);
  else
    I->insertInto(BB, BB->end());
}

#ifndef NDEBUG
void InstrRemoveFromParent::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

void InsertToBB::revert() {
  // Run the callbacks before we actually erase the instruction.
  // TODO: Ctxt.runRemoveInstrCallbacks(I);
  I->removeFromParent();
}

InsertToBB::InsertToBB(SBInstruction *I, SBBasicBlock *BB,
                       SandboxIRTracker &Tracker)
    : IRChangeBase(TrackID::InsertToBB, Tracker), I(I) {}

#ifndef NDEBUG
void InsertToBB::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

MoveInstr::MoveInstr(SBInstruction *I, SandboxIRTracker &Tracker)
    : IRChangeBase(TrackID::MoveInstr, Tracker), I(I), BB(I->getParent()),
      NextI(I->getNextNode()) {}
void MoveInstr::revert() {
  if (NextI != nullptr)
    I->moveBefore(NextI);
  else
    I->moveBefore(*BB, BB->end());
}

#ifndef NDEBUG
void MoveInstr::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

ReplaceAllUsesWith::ReplaceAllUsesWith(SBValue *V, SandboxIRTracker &Tracker)
    : IRChangeBase(TrackID::ReplaceAllUsesWith, Tracker) {
  auto CollectUses = [this](Value *V) {
    for (Use &U : V->uses()) {
      User *Usr = U.getUser();
      Value *Op = U.get();
      unsigned OpIdx = U.getOperandNo();
      OrigUserState.push_back({Usr, Op, OpIdx});
    }
  };
  if (auto *SBI = dyn_cast<SBInstruction>(V)) {
    for (Instruction *I : SBI->getIRInstrs()) {
      CollectUses(I);
    }
  } else {
    CollectUses(ValueAttorney::getValue(V));
  }
}

void ReplaceAllUsesWith::revert() {
  for (const auto [Usr, Op, OpIdx] : OrigUserState)
    Usr->setOperand(OpIdx, Op);
}

#ifndef NDEBUG
void ReplaceAllUsesWith::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

ReplaceUsesOfWith::ReplaceUsesOfWith(SBUser *U, SBValue *CurrOp,
                                     SBValue *NewOp, SandboxIRTracker &Tracker)
    : IRChangeBase(TrackID::ReplaceUsesOfWith, Tracker) {
  auto CollectUses = [this](User *U) {
    for (Use &U : U->operands()) {
      Value *Op = U.get();
      unsigned OpIdx = U.getOperandNo();
      OrigOperandState.push_back({Op, OpIdx, U.getUser()});
    }
  };
  if (auto *SBI = dyn_cast<SBInstruction>(U)) {
    for (Instruction *I : SBI->getIRInstrs())
      CollectUses(I);
  } else {
    CollectUses(cast<User>(ValueAttorney::getValue(U)));
  }
}

void ReplaceUsesOfWith::revert() {
  for (const auto [Op, OpIdx, U] : OrigOperandState)
    U->setOperand(OpIdx, Op);
}

#ifndef NDEBUG
void ReplaceUsesOfWith::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

SetOperand::SetOperand(SBUser *U, unsigned OpIdx, SandboxIRTracker &Tracker)
    : IRChangeBase(TrackID::SetOperand, Tracker), U(U), OpIdx(OpIdx) {
  Op = U->getOperand(OpIdx);
}

void SetOperand::revert() { U->setOperand(OpIdx, Op); }

#ifndef NDEBUG
void SetOperand::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

void CreateAndInsertInstr::revert() { I->eraseFromParent(); }

#ifndef NDEBUG
void CreateAndInsertInstr::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

EraseFromParent::EraseFromParent(std::unique_ptr<SBValue> &&IPtr,
                                 SBContext &Ctxt, SandboxIRTracker &Tracker)
    : IRChangeBase(TrackID::EraseFromParent, Tracker), IPtr(std::move(IPtr)),
      Ctxt(Ctxt) {
  auto *SBI = cast<SBInstruction>(this->IPtr.get());
  auto IRInstrs = SBI->getIRInstrs();
  // Sort them in reverse program order.
  sort(IRInstrs, [](auto *I1, auto *I2) { return I2->comesBefore(I1); });
  for (auto *I : IRInstrs) {
    SmallVector<OpData> OpDataVec;
    for (auto [OpIdx, Use] : enumerate(I->operands()))
      OpDataVec.push_back({OpIdx, Use.get()});
    InstrData.push_back({I, OpDataVec});
  }
#ifndef NDEBUG
  for (auto Idx : seq<unsigned>(1, InstrData.size()))
    assert(InstrData[Idx].I->comesBefore(InstrData[Idx - 1].I) &&
           "Expected reverse program order!");
#endif
  auto *BotI = cast<Instruction>(ValueAttorney::getValue(SBI));
  NextI = BotI->getNextNode();
  BB = BotI->getParent();
}

void EraseFromParent::apply() {
  for (const auto &IData : InstrData)
    IData.I->deleteValue();
}

void EraseFromParent::revert() {
  auto [BotI, OpData] = InstrData[0];
  if (NextI)
    BotI->insertBefore(NextI);
  else
    BotI->insertInto(BB, BB->end());
  for (auto [OpIdx, Op] : OpData)
    BotI->setOperand(OpIdx, Op);

  for (auto [I, OpData] : drop_begin(InstrData)) {
    I->insertBefore(BotI);
    for (auto [OpIdx, Op] : OpData)
      I->setOperand(OpIdx, Op);
    BotI = I;
  }
  Ctxt.registerSBValue(std::move(IPtr));
}

#ifndef NDEBUG
void EraseFromParent::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

#ifndef NDEBUG
void SandboxIRTracker::dump(raw_ostream &OS) const {
  for (const auto &ChangePtr : Changes) {
    ChangePtr->dump(OS);
    OS << "\n";
  }
}
void SandboxIRTracker::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SandboxIRTracker::~SandboxIRTracker() {
#ifndef NDEBUG
  // If tracking is disabled, then the only changes that we should have
  // collected are the compulsory ones.
  assert((Changes.empty() || all_of(Changes,
                                    [](const auto &ChngPtr) {
                                      return ChngPtr->isCompulsory();
                                    })) &&
         "Please revert() or accept() before destroying the tracker!");
#endif
  // All changes tracked are compulsory, so simply accept them.
  accept();
}

void SandboxIRTracker::track(std::unique_ptr<IRChangeBase> &&Change) {
#ifndef NDEBUG
  assert(!InRevert && "No changes should be tracked during revert()!");
  assert((Tracking || Change->isCompulsory()) &&
         "If we are not tracking, this should be a compulsory change!");
#endif
  // #if defined(EXPENSIVE_CHECKS) && !defined(NDEBUG)
  auto *Chng = Change.get();
  (void)Chng;
  Changes.push_back(std::move(Change));

#ifndef NDEBUG
  InMiddleOfCreatingChange = false;
#endif // NDEBUG
}

void SandboxIRTracker::start(SBBasicBlock *SBBB) {
  Tracking = true;
#ifndef NDEBUG
#ifdef SBVEC_EXPENSIVE_CHECKS
  // TODO: Remove ValueAttorney
  IRVerifier.enable(cast<BasicBlock>(ValueAttorney::getValue(SBBB)));
  // Dump the IR to make sure revert() restores it to its original state.
  if (Changes.empty())
    IRVerifier.save();
#endif
#endif
}

void SandboxIRTracker::revert() {
  bool SvTracking = Tracking;
  Tracking = false;
  InRevert = true;
  for (auto &Change : reverse(Changes)) {
    Change->revert();
  }
  Changes.clear();
#ifndef NDEBUG
#ifdef SBVEC_EXPENSIVE_CHECKS
  if (IRVerifier.enabled())
    IRVerifier.expectNoDiff();
#endif
#endif // NDEBUG
  InRevert = false;
  Tracking = SvTracking;
}

void SandboxIRTracker::accept() {
  bool SvTracking = Tracking;
  Tracking = false;
  for (auto &Change : Changes)
    Change->apply();
  Changes.clear();
#ifndef NDEBUG
  IRVerifier.disable();
#endif // NDEBUG
  Tracking = SvTracking;
}
