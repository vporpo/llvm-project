//===- SandboxIRTracker.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tracks changes made to SandboxIR (and as a result to LLVM IR too) so that we
// can revert the IR's state if needed.
//

#ifndef LLVM_TRANSFORMS_SANDBOXIR_IRCHANGETRACKER_H
#define LLVM_TRANSFORMS_SANDBOXIR_IRCHANGETRACKER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/SandboxIR/DmpVector.h"
#include "llvm/Support/Debug.h"
#include <memory>
#include <regex>

namespace llvm {
namespace sandboxir {

class Value;
class Context;
class Instruction;
class BasicBlock;
class User;
class InstrRange;

#ifndef NDEBUG
/// Helper class that saves the textual representation of the IR upon
/// construction and compares against it when `expectNoChange()` is called.
class IRChecker {
  llvm::BasicBlock *BB = nullptr;
  std::string OrigIR;
  std::regex SBMDRegex;
  std::regex MDNumRegex;
  void stripMetadata(std::string &Line) const;
  /// \Returns true if \p OrigIR and \p CurrIR differ, and prints diff.
  bool diff(const std::string &OrigIR, const std::string &CurrIR) const;

public:
  IRChecker() : SBMDRegex(", !sb ![0-9]+"), MDNumRegex("![0-9]+") {}
  /// This enables the verifier for \p BB.
  void enable(llvm::BasicBlock *BB) { this->BB = BB; }
  void disable() { BB = nullptr; }
  bool enabled() const { return BB != nullptr; }
  void save();
  /// \Returns the IR dump in a string.
  std::string dumpIR(llvm::BasicBlock *BB) const;
  /// Crashes if there is a difference between the original and current IR.
  void expectNoDiff() const;
};
#endif // NDEBUG

enum class TrackID {
  CreateAndInsertInstr,
  EraseFromParent,
  InsertToBB,
  MoveInstr,
  ReplaceAllUsesWith,
  ReplaceUsesOfWith,
  SetOperand,
  DeleteOnAccept,
  InstrRemoveFromParent,
  ClearDAGInterval,
  UseSet,
  UseSwap,
};
#ifndef NDEBUG
static const char *trackIDToStr(TrackID ID) {
  switch (ID) {
  case TrackID::CreateAndInsertInstr:
    return "CreateAndInsertInstr";
  case TrackID::EraseFromParent:
    return "EraseFromParent";
  case TrackID::DeleteOnAccept:
    return "DeleteOnAccept";
  case TrackID::InstrRemoveFromParent:
    return "InstrRemoveFromParent";
  case TrackID::InsertToBB:
    return "InsertToBB";
  case TrackID::MoveInstr:
    return "MoveInstr";
  case TrackID::ReplaceAllUsesWith:
    return "ReplaceAllUsesWith";
  case TrackID::ReplaceUsesOfWith:
    return "ReplaceUsesOfWith";
  case TrackID::SetOperand:
    return "SetOperand";
  case TrackID::ClearDAGInterval:
    return "ClearDAGInterval";
  case TrackID::UseSet:
    return "UseSet";
  case TrackID::UseSwap:
    return "UseSwap";
  }
  llvm_unreachable("Unimplemented ID");
}
#endif // NDEBUG

class SandboxIRTracker;

class IRChangeBase {
protected:
#ifndef NDEBUG
  unsigned Idx = 0;
#endif
  const TrackID ID;
  SandboxIRTracker &Parent;

public:
  IRChangeBase(TrackID ID, SandboxIRTracker &Parent);
  TrackID getTrackID() const { return ID; }
  /// What to do when we want to revert the change.
  virtual void revert() = 0;
  /// Some cleanup that the change may need to fully apply the change.
  virtual void accept() = 0;
  virtual ~IRChangeBase() = default;
#ifndef NDEBUG
  void dumpCommon(raw_ostream &OS) const {
    OS << Idx << ". " << trackIDToStr(ID);
  }
  virtual void dump(raw_ostream &OS) const = 0;
  LLVM_DUMP_METHOD virtual void dump() const = 0;
#endif
  /// \Returns true if tracking this change is compulsory for the correct
  /// operation of SandboxIR. This includes changes like eraseFromParent because
  /// without it we may leak memory.
  bool isCompulsory() const;
};

/// This is a special change, used to make sure we don't leak memory when we
/// eraseFromParent a SandboxIR instruction and accept the changes.
class DeleteOnAccept : public IRChangeBase {
  DmpVector<llvm::Instruction *> Instrs;
  llvm::Instruction *BeforeI = nullptr;
  llvm::BasicBlock *BB = nullptr;

public:
  DeleteOnAccept(sandboxir::Instruction *I, SandboxIRTracker &Tracker);
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::DeleteOnAccept;
  }
  void revert() final {}
  void accept() final;
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const DeleteOnAccept &C) {
    C.dump(OS);
    return OS;
  }
#endif // NDEBUG
};

class InstrRemoveFromParent : public IRChangeBase {
  sandboxir::Instruction *I = nullptr;
  sandboxir::BasicBlock *BB = nullptr;
  /// If null we insert at the end of BB.
  sandboxir::Instruction *BeforeI = nullptr;

public:
  InstrRemoveFromParent(sandboxir::Instruction *I, SandboxIRTracker &Tracker);
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::InstrRemoveFromParent;
  }
  void revert() final;
  void accept() final {};
  sandboxir::Instruction *getInstruction() const { return I; }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const InstrRemoveFromParent &C) {
    C.dump(OS);
    return OS;
  }
#endif // NDEBUG
};

class InsertToBB : public IRChangeBase {
  sandboxir::Instruction *I = nullptr;

public:
  InsertToBB(sandboxir::Instruction *I, sandboxir::BasicBlock *BB,
             SandboxIRTracker &Tracker);
  sandboxir::Instruction *getInstruction() const { return I; }
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::InsertToBB;
  }
  void revert() final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const InsertToBB &C) {
    C.dump(OS);
    return OS;
  }
#endif // NDEBUG
};

class MoveInstr : public IRChangeBase {
  /// The instruction that moved.
  sandboxir::Instruction *I;
  /// `I`'s parent before moving.
  sandboxir::BasicBlock *BB;
  /// `I`'s next instruction in the instruction list or nullptr if at the end.
  sandboxir::Instruction *NextI;

public:
  MoveInstr(sandboxir::Instruction *I, SandboxIRTracker &Tracker);
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::MoveInstr;
  }
  void revert() final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const MoveInstr &C) {
    C.dump(OS);
    return OS;
  }
#endif // NDEBUG
};

class ReplaceAllUsesWith : public IRChangeBase {
  struct UserState {
    llvm::User *Usr;
    llvm::Value *Op;
    unsigned OpIdx;
  };
  SmallVector<UserState> OrigUserState;

public:
  ReplaceAllUsesWith(sandboxir::Value *V, SandboxIRTracker &Tracker);
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::ReplaceAllUsesWith;
  }
  void revert() final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const ReplaceAllUsesWith &C) {
    C.dump(OS);
    return OS;
  }
#endif // NDEBUG
};

class ReplaceUsesOfWith : public IRChangeBase {
  struct OperandState {
    llvm::Value *OrigOp;
    unsigned OpIdx;
    llvm::User *U;
  };
  SmallVector<OperandState> OrigOperandState;

public:
  ReplaceUsesOfWith(sandboxir::User *U, sandboxir::Value *CurrOp,
                    sandboxir::Value *NewOp, SandboxIRTracker &Tracker);
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::ReplaceUsesOfWith;
  }
  void revert() final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const ReplaceUsesOfWith &C) {
    C.dump(OS);
    return OS;
  }
#endif // NDEBUG
};

class SetOperand : public IRChangeBase {
  sandboxir::User *U;
  unsigned OpIdx;
  sandboxir::Value *Op;

public:
  SetOperand(sandboxir::User *U, unsigned OpIdx, SandboxIRTracker &Tracker);
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::SetOperand;
  }
  void revert() final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const SetOperand &C) {
    C.dump(OS);
    return OS;
  }
#endif // NDEBUG
};

class CreateAndInsertInstr : public IRChangeBase {
  sandboxir::Instruction *I = nullptr;

public:
  CreateAndInsertInstr(sandboxir::Instruction *I, SandboxIRTracker &Tracker)
      : IRChangeBase(TrackID::CreateAndInsertInstr, Tracker), I(I) {}
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::CreateAndInsertInstr;
  }
  void revert() final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const CreateAndInsertInstr &C) {
    C.dump(OS);
    return OS;
  }
#endif
};

class EraseFromParent : public IRChangeBase {
  struct OpData {
    unsigned long OpIdx;
    llvm::Value *Op;
  };
  struct InstrData {
    llvm::Instruction *I;
    SmallVector<OpData> OpDataVec;
  };
  /// The instruction data is in revere program order, which helps create the
  /// original program order during revert().
  SmallVector<InstrData> InstrData;
  // TODO: We actually need only one of NextI and BB.
  llvm::Instruction *NextI;
  llvm::BasicBlock *BB;
  std::unique_ptr<sandboxir::Value> IPtr;
  sandboxir::Context &Ctx;

public:
  EraseFromParent(std::unique_ptr<sandboxir::Value> &&IPtr,
                  sandboxir::Context &Ctx, SandboxIRTracker &Tracker);
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::EraseFromParent;
  }
  void revert() final;
  void accept() final;
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const EraseFromParent &C) {
    C.dump(OS);
    return OS;
  }
#endif
};

class UseSet : public IRChangeBase {
  llvm::Use &Use;
  llvm::Value *OrigV = nullptr;

public:
  UseSet(llvm::Use &Use, SandboxIRTracker &Tracker)
      : IRChangeBase(sandboxir::TrackID::UseSet, Tracker), Use(Use),
        OrigV(Use.get()) {}
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::UseSet;
  }
  void revert() final { Use.set(OrigV); }
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const UseSet &C) {
    C.dump(OS);
    return OS;
  }
#endif
};

class UseSwap : public IRChangeBase {
  llvm::Use &ThisUse;
  llvm::Use &OtherUse;

public:
  UseSwap(llvm::Use &ThisUse, llvm::Use &OtherUse, SandboxIRTracker &Tracker)
      : IRChangeBase(sandboxir::TrackID::UseSet, Tracker), ThisUse(ThisUse),
        OtherUse(OtherUse) {}
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::UseSet;
  }
  void revert() final { ThisUse.swap(OtherUse); }
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const UseSwap &C) {
    C.dump(OS);
    return OS;
  }
#endif
};

class SandboxIRTracker {
  SmallVector<std::unique_ptr<IRChangeBase>> Changes;
  /// This is true when we are tracking changes.
  bool Tracking = false;
  bool InRevert = false;
#ifndef NDEBUG
  IRChecker IRVerifier;
#endif
public:
  // Note: We are using std::function instead of llvm::function_ref because we
  // need ownership of the callable.
  using RemoveCBTy = std::function<void(llvm::Instruction *)>;
  using InsertCBTy = std::function<void(llvm::Instruction *)>;

#ifndef NDEBUG
  /// Helps catch bugs where we are creating new change objects while in the
  /// middle of creating other change objects.
  bool InMiddleOfCreatingChange = false;
#endif // NDEBUG

protected:
  /// Vector of callbacks called when an IR Instruction is about to get erased.
  SmallVector<std::unique_ptr<RemoveCBTy>> RemoveInstrCallbacks;
  SmallVector<std::unique_ptr<InsertCBTy>> InsertInstrCallbacks;

public:
  SandboxIRTracker() = default;
  ~SandboxIRTracker();
  void track(std::unique_ptr<IRChangeBase> &&Change);
  /// \Returns true if the user has started tracking.
  bool tracking() const { return Tracking; }
  /// \Returns true while we are in the middle of a revert().
  bool inRevert() const { return InRevert; }
  /// Turns on change tracking.
  void start(sandboxir::BasicBlock *SBBB);
  void stop() {
    assert(Tracking && "Stopping without having started tracking first!");
    Tracking = false;
  }
  void revert();
  unsigned size() const { return Changes.size(); }
  bool empty() const { return Changes.empty(); }
  void accept();

#ifndef NDEBUG
  /// \Returns the \p Idx'th change. This is used for testing.
  IRChangeBase *getChange(unsigned Idx) const { return Changes[Idx].get(); }
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  friend raw_ostream &operator<<(raw_ostream &OS, const SandboxIRTracker &C) {
    C.dump(OS);
    return OS;
  }
#endif // NDEBUG
};
} // namespace sandboxir
} // namespace llvm

#endif // LLVM_TRANSFORMS_SANDBOXIR_IRCHANGETRACKER_H
