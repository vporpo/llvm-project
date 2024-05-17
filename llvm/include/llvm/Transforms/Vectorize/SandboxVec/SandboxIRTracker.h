//===- SandboxIRTracker.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tracks changes made to SandboxIR (and as a result to LLVM IR too) so that we can
// revert the IR's state if needed.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_IRCHANGETRACKER_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_IRCHANGETRACKER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Bundle.h"
#include <memory>
#include <regex>

namespace llvm {

class SBValue;
class SBContext;
class SBInstruction;
class SBBasicBlock;
class SBUser;
class InstrRange;

#ifndef NDEBUG
/// Helper class that saves the textual representation of the IR upon
/// construction and compares against it when `expectNoChange()` is called.
class IRChecker {
  BasicBlock *BB = nullptr;
  std::string OrigIR;
  std::regex SBMDRegex;
  std::regex MDNumRegex;
  void stripMetadata(std::string &Line) const;
  /// \Returns true if \p OrigIR and \p CurrIR differ, and prints diff.
  bool diff(const std::string &OrigIR, const std::string &CurrIR) const;

public:
  IRChecker() : SBMDRegex(", !sb ![0-9]+"), MDNumRegex("![0-9]+") {}
  /// This enables the verifier for \p BB.
  void enable(BasicBlock *BB) { this->BB = BB; }
  void disable() { BB = nullptr; }
  bool enabled() const { return BB != nullptr; }
  void save();
  /// \Returns the IR dump in a string.
  std::string dumpIR(BasicBlock *BB) const;
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
  ClearDAGRange,
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
  case TrackID::ClearDAGRange:
    return "ClearDAGRange";
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
  virtual void apply() = 0;
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
  Bundle<Instruction *> Instrs;
  Instruction *BeforeI = nullptr;
  BasicBlock *BB = nullptr;

public:
  DeleteOnAccept(SBInstruction *I, SandboxIRTracker &Tracker);
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::DeleteOnAccept;
  }
  void revert() final {}
  void apply() final;
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
  SBInstruction *I = nullptr;
  SBBasicBlock *BB = nullptr;
  /// If null we insert at the end of BB.
  SBInstruction *BeforeI = nullptr;

public:
  InstrRemoveFromParent(SBInstruction *I, SandboxIRTracker &Tracker);
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::InstrRemoveFromParent;
  }
  void revert() final;
  void apply() final {};
  SBInstruction *getInstruction() const { return I; }
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
  SBInstruction *I = nullptr;

public:
  InsertToBB(SBInstruction *I, SBBasicBlock *BB, SandboxIRTracker &Tracker);
  SBInstruction *getInstruction() const { return I; }
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::InsertToBB;
  }
  void revert() final;
  void apply() final {}
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
  SBInstruction *I;
  /// `I`'s parent before moving.
  SBBasicBlock *BB;
  /// `I`'s next instruction in the instruction list or nullptr if at the end.
  SBInstruction *NextI;

public:
  MoveInstr(SBInstruction *I, SandboxIRTracker &Tracker);
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::MoveInstr;
  }
  void revert() final;
  void apply() final {}
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
    User *Usr;
    Value *Op;
    unsigned OpIdx;
  };
  SmallVector<UserState> OrigUserState;

public:
  ReplaceAllUsesWith(SBValue *V, SandboxIRTracker &Tracker);
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::ReplaceAllUsesWith;
  }
  void revert() final;
  void apply() final {}
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
    Value *OrigOp;
    unsigned OpIdx;
    User *U;
  };
  SmallVector<OperandState> OrigOperandState;

public:
  ReplaceUsesOfWith(SBUser *U, SBValue *CurrOp, SBValue *NewOp,
                    SandboxIRTracker &Tracker);
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::ReplaceUsesOfWith;
  }
  void revert() final;
  void apply() final {}
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
  SBUser *U;
  unsigned OpIdx;
  SBValue *Op;

public:
  SetOperand(SBUser *U, unsigned OpIdx, SandboxIRTracker &Tracker);
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::SetOperand;
  }
  void revert() final;
  void apply() final {}
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
  SBInstruction *I = nullptr;

public:
  CreateAndInsertInstr(SBInstruction *I, SandboxIRTracker &Tracker)
      : IRChangeBase(TrackID::CreateAndInsertInstr, Tracker), I(I) {}
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::CreateAndInsertInstr;
  }
  void revert() final;
  void apply() final {}
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
    Value *Op;
  };
  struct InstrData {
    Instruction *I;
    SmallVector<OpData> OpDataVec;
  };
  /// The instruction data is in revere program order, which helps create the
  /// original program order during revert().
  SmallVector<InstrData> InstrData;
  // TODO: We actually need only one of NextI and BB.
  Instruction *NextI;
  BasicBlock *BB;
  std::unique_ptr<SBValue> IPtr;
  SBContext &Ctxt;

public:
  EraseFromParent(std::unique_ptr<SBValue> &&IPtr, SBContext &Ctxt,
                  SandboxIRTracker &Tracker);
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::EraseFromParent;
  }
  void revert() final;
  void apply() final;
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const EraseFromParent &C) {
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
  using RemoveCBTy = std::function<void(Instruction *)>;
  using InsertCBTy = std::function<void(Instruction *)>;

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
  void start(SBBasicBlock *SBBB);
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
} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_IRCHANGETRACKER_H
