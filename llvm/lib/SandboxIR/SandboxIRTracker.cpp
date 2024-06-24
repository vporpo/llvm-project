//===- SandboxIRTracker.cpp
//------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/SandboxIRTracker.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include <sstream>

using namespace llvm;

#ifndef NDEBUG
std::string sandboxir::IRChecker::dumpIR(llvm::BasicBlock *BB) const {
  std::string TmpStr;
  raw_string_ostream SS(TmpStr);
  BB->print(SS, /*AssemblyAnnotationWriter=*/nullptr);
  return TmpStr;
}

void sandboxir::IRChecker::save() { OrigIR = dumpIR(BB); }

void sandboxir::IRChecker::stripMetadata(std::string &Line) const {
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

bool sandboxir::IRChecker::diff(const std::string &OrigIR,
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

void sandboxir::IRChecker::expectNoDiff() const {
  std::string CurrIR = dumpIR(BB);
  assert(!OrigIR.empty() && "OrigIR not created!");
  if (diff(OrigIR, CurrIR)) {
    llvm_unreachable(
        "Original and current IR differ! Possibly a Checkpointing bug.");
  }
}
#endif

sandboxir::IRChangeBase::IRChangeBase(sandboxir::TrackID ID,
                                      sandboxir::SandboxIRTracker &Parent)
    : ID(ID), Parent(Parent) {
#ifndef NDEBUG
  Idx = Parent.size();

  assert(!Parent.InMiddleOfCreatingChange &&
         "We are in the middle of creating another change!");
  if (Parent.tracking())
    Parent.InMiddleOfCreatingChange = true;
#endif // NDEBUG
}

bool sandboxir::IRChangeBase::isCompulsory() const {
  switch (ID) {
  case sandboxir::TrackID::DeleteOnAccept:
    return true;
  case sandboxir::TrackID::InstrRemoveFromParent:
  case sandboxir::TrackID::InsertToBB:
  case sandboxir::TrackID::MoveInstr:
  case sandboxir::TrackID::ReplaceAllUsesWith:
  case sandboxir::TrackID::ReplaceUsesOfWith:
  case sandboxir::TrackID::SetOperand:
  case sandboxir::TrackID::CreateAndInsertInstr:
  case sandboxir::TrackID::EraseFromParent:
  case sandboxir::TrackID::ClearDAGInterval:
  case sandboxir::TrackID::UseSet:
  case sandboxir::TrackID::UseSwap:
    return false;
  }
}

sandboxir::DeleteOnAccept::DeleteOnAccept(sandboxir::Instruction *SBI,
                                          sandboxir::SandboxIRTracker &Tracker)
    : IRChangeBase(sandboxir::TrackID::DeleteOnAccept, Tracker),
      Instrs(SBI->getLLVMInstrs()) {
  auto *BotI = cast<llvm::Instruction>(ValueAttorney::getValue(SBI));
  BeforeI = BotI->getNextNode();
  BB = BotI->getParent();
}

void sandboxir::DeleteOnAccept::accept() {
  // Delete instructions bottom-up to avoid deleting instrs with attached users.
  for (auto *I : reverse(Instrs))
    I->deleteValue();
}

#ifndef NDEBUG
void sandboxir::DeleteOnAccept::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

sandboxir::InstrRemoveFromParent::InstrRemoveFromParent(
    sandboxir::Instruction *I, sandboxir::SandboxIRTracker &Tracker)
    : IRChangeBase(sandboxir::TrackID::InstrRemoveFromParent, Tracker), I(I),
      BB(I->getParent()), BeforeI(I->getNextNode()) {}

void sandboxir::InstrRemoveFromParent::revert() {
  if (BeforeI)
    I->insertBefore(BeforeI);
  else
    I->insertInto(BB, BB->end());
}

#ifndef NDEBUG
void sandboxir::InstrRemoveFromParent::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

void sandboxir::InsertToBB::revert() {
  // Run the callbacks before we actually erase the instruction.
  // TODO: Ctx.runRemoveInstrCallbacks(I);
  I->removeFromParent();
}

sandboxir::InsertToBB::InsertToBB(sandboxir::Instruction *I,
                                  sandboxir::BasicBlock *BB,
                                  sandboxir::SandboxIRTracker &Tracker)
    : IRChangeBase(sandboxir::TrackID::InsertToBB, Tracker), I(I) {}

#ifndef NDEBUG
void sandboxir::InsertToBB::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

sandboxir::MoveInstr::MoveInstr(sandboxir::Instruction *I,
                                sandboxir::SandboxIRTracker &Tracker)
    : IRChangeBase(sandboxir::TrackID::MoveInstr, Tracker), I(I),
      BB(I->getParent()), NextI(I->getNextNode()) {}
void sandboxir::MoveInstr::revert() {
  if (NextI != nullptr)
    I->moveBefore(NextI);
  else
    I->moveBefore(*BB, BB->end());
}

#ifndef NDEBUG
void sandboxir::MoveInstr::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

sandboxir::ReplaceAllUsesWith::ReplaceAllUsesWith(
    sandboxir::Value *V, sandboxir::SandboxIRTracker &Tracker)
    : IRChangeBase(sandboxir::TrackID::ReplaceAllUsesWith, Tracker) {
  auto CollectUses = [this](llvm::Value *V) {
    for (llvm::Use &U : V->uses()) {
      llvm::User *Usr = U.getUser();
      llvm::Value *Op = U.get();
      unsigned OpIdx = U.getOperandNo();
      OrigUserState.push_back({Usr, Op, OpIdx});
    }
  };
  if (auto *SBI = dyn_cast<sandboxir::Instruction>(V)) {
    for (llvm::Instruction *I : SBI->getLLVMInstrs()) {
      CollectUses(I);
    }
  } else {
    CollectUses(ValueAttorney::getValue(V));
  }
}

void sandboxir::ReplaceAllUsesWith::revert() {
  for (const auto [Usr, Op, OpIdx] : OrigUserState)
    Usr->setOperand(OpIdx, Op);
}

#ifndef NDEBUG
void sandboxir::ReplaceAllUsesWith::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

sandboxir::ReplaceUsesOfWith::ReplaceUsesOfWith(
    sandboxir::User *U, sandboxir::Value *CurrOp, sandboxir::Value *NewOp,
    sandboxir::SandboxIRTracker &Tracker)
    : IRChangeBase(sandboxir::TrackID::ReplaceUsesOfWith, Tracker) {
  auto CollectUses = [this](llvm::User *U) {
    for (llvm::Use &U : U->operands()) {
      llvm::Value *Op = U.get();
      unsigned OpIdx = U.getOperandNo();
      OrigOperandState.push_back({Op, OpIdx, U.getUser()});
    }
  };
  if (auto *SBI = dyn_cast<sandboxir::Instruction>(U)) {
    for (llvm::Instruction *I : SBI->getLLVMInstrs())
      CollectUses(I);
  } else {
    CollectUses(cast<llvm::User>(ValueAttorney::getValue(U)));
  }
}

void sandboxir::ReplaceUsesOfWith::revert() {
  for (const auto [Op, OpIdx, U] : OrigOperandState)
    U->setOperand(OpIdx, Op);
}

#ifndef NDEBUG
void sandboxir::ReplaceUsesOfWith::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

sandboxir::SetOperand::SetOperand(sandboxir::User *U, unsigned OpIdx,
                                  sandboxir::SandboxIRTracker &Tracker)
    : IRChangeBase(sandboxir::TrackID::SetOperand, Tracker), U(U),
      OpIdx(OpIdx) {
  Op = U->getOperand(OpIdx);
}

void sandboxir::SetOperand::revert() { U->setOperand(OpIdx, Op); }

#ifndef NDEBUG
void sandboxir::SetOperand::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

void sandboxir::CreateAndInsertInstr::revert() { I->eraseFromParent(); }

#ifndef NDEBUG
void sandboxir::CreateAndInsertInstr::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

sandboxir::EraseFromParent::EraseFromParent(
    std::unique_ptr<sandboxir::Value> &&IPtr, sandboxir::Context &Ctx,
    sandboxir::SandboxIRTracker &Tracker)
    : IRChangeBase(sandboxir::TrackID::EraseFromParent, Tracker),
      IPtr(std::move(IPtr)), Ctx(Ctx) {
  auto *SBI = cast<sandboxir::Instruction>(this->IPtr.get());
  auto IRInstrs = SBI->getLLVMInstrs();
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
  auto *BotI = cast<llvm::Instruction>(ValueAttorney::getValue(SBI));
  NextI = BotI->getNextNode();
  BB = BotI->getParent();
}

void sandboxir::EraseFromParent::accept() {
  for (const auto &IData : InstrData)
    IData.I->deleteValue();
}

void sandboxir::EraseFromParent::revert() {
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
  Ctx.registerValue(std::move(IPtr));
}

#ifndef NDEBUG
void sandboxir::EraseFromParent::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

#ifndef NDEBUG
void sandboxir::UseSet::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

#ifndef NDEBUG
void sandboxir::UseSwap::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

#ifndef NDEBUG
void sandboxir::SandboxIRTracker::dump(raw_ostream &OS) const {
  for (const auto &ChangePtr : Changes) {
    ChangePtr->dump(OS);
    OS << "\n";
  }
}
void sandboxir::SandboxIRTracker::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

sandboxir::SandboxIRTracker::~SandboxIRTracker() {
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

void sandboxir::SandboxIRTracker::track(
    std::unique_ptr<IRChangeBase> &&Change) {
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
#endif
}

void sandboxir::SandboxIRTracker::start(sandboxir::BasicBlock *SBBB) {
  Tracking = true;
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  // TODO: Remove ValueAttorney
  IRVerifier.enable(cast<llvm::BasicBlock>(ValueAttorney::getValue(SBBB)));
  // Dump the IR to make sure revert() restores it to its original state.
  if (Changes.empty())
    IRVerifier.save();
#endif
}

void sandboxir::SandboxIRTracker::revert() {
  bool SvTracking = Tracking;
  Tracking = false;
  InRevert = true;
  for (auto &Change : reverse(Changes)) {
    Change->revert();
  }
  Changes.clear();
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  if (IRVerifier.enabled())
    IRVerifier.expectNoDiff();
#endif
  InRevert = false;
  Tracking = SvTracking;
}

void sandboxir::SandboxIRTracker::accept() {
  bool SvTracking = Tracking;
  Tracking = false;
  for (auto &Change : Changes)
    Change->accept();
  Changes.clear();
#ifndef NDEBUG
  IRVerifier.disable();
#endif // NDEBUG
  Tracking = SvTracking;
}
