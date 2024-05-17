//===- SandboxIR.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The vectorizer does not operate on LLVM IR directly, but rather on a custom
// thin overlay IR, named SandboxIR, consisting of SBBasicBlocks and SBsInstrs.
// SandboxIR has several features that make it very easy to work with in the
// context of vectorization:
// - It supports coarser instructions (e.g., Pack) that map to a sequence of
//   LLVM IR instructions.
// - It is transactional: you can save/restore it at any point.
// - When modified it automatically updates the DAG and scheduler state.
// - It is always in sync with the underlying LLVM IR. This has two benefits:
//    a. There is no need to replicate data-structures that already exist in
//       LLVM IR, like def-use chains, Types etc. This makes it a thin layer.
//    b. Simplifies development and debugging as what you see is what you get.
//
// SandboxIR forms a simple class hierarcy that resembles that of LLVM IR:
//
//          +- SBArgument   +- SBConstant     +- SBOpaqueInstruction
//          |               |                 |
// SBValue -+- SBUser ------+- SBInstruction -+- SBPackInstruction
//          |                                 |
//          +- SBBasicBlock                   +- SBShuffleInstruction
//                                            |
//                                            +- SBUnpackInstruction
//                                            |
//                                            +- SBStoreInstruction
//                                            |
//                                            +- SBLoadInstruction
//                                            |
//                                            +- SBCmpInstruction
//                                            |
//                                            +- SBCastInstruction
//                                            |
//                                            +- SBPHINode
//                                            |
//                                            +- SBSelectInstruction
//                                            |
//                                            +- SBBinaryOperator
//                                            |
//                                            +- SBUnaryOperator
//
// SBUse
//
//
// Pack: A perfect pack of scalar or vector SandboxIR values.
// Shuffle: A single-operand shuffle with a constant mask.
// Unpack: Extracts a scalar from a vector.
//
// Development notes
// -----------------
// - When adding a new SandboxIR instruction (or when writing a test) that requires
//   the creation of LLVM IR, you must use SBContext::getLLVMIRBuilder()
//   otherwise IR creation does not get tracked and checkpointing will break!

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SANDBOXIR_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SANDBOXIR_H

#if !defined(NDEBUG) && defined(EXPENSIVE_CHECKS)
#define SBVEC_EXPENSIVE_CHECKS
#endif

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIRTracker.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Analysis.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Bundle.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Utils.h"
#include <iterator>
#include <variant>

using namespace llvm::PatternMatch;

namespace llvm {

class SBBasicBlock;
class SBValue;
class SBUser;
class SBPackInstruction;
class SBContext;
class SBFunction;
class Scheduler;
class DependencyGraph;

/// Represents a Def-use/Use-def edge in SandboxIR.
/// NOTE: Unlike llvm::Use, this is not an integral part of the use-def chains.
/// It is also not uniqued and is currently passed by value, so you can have to
/// SBUse objects for the same use-def edge.
class SBUse {
  llvm::Use *LLVMUse;
  SBUser *User;
  SBContext *Ctxt;

  /// Don't allow the user to create a SBUse directly.
  SBUse(llvm::Use *LLVMUse, SBUser *User, SBContext &Ctxt)
      : LLVMUse(LLVMUse), User(User), Ctxt(&Ctxt) {}
  SBUse() : LLVMUse(nullptr), Ctxt(nullptr) {}

  friend class SBUser;               // For constructor
  friend class SBValue;              // For constructor
  friend class SBPackInstruction;    // For constructor
  friend class SBOperandUseIterator; // For constructor
  friend class SBUserUseIterator;    // For constructor
  friend class SBShuffleInstruction; // For constructor
  friend class SBUnpackInstruction;  // For constructor

public:
  operator SBValue *() const { return get(); }
  SBValue *get() const;
  SBUser *getUser() const { return User; }
  unsigned getOperandNo() const;
  SBContext *getContext() const { return Ctxt; }
  bool operator==(const SBUse &Other) const {
    assert(Ctxt == Other.Ctxt && "Contexts differ!");
    return LLVMUse == Other.LLVMUse && User == Other.User;
  }
  bool operator!=(const SBUse &Other) const { return !(*this == Other); }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  void dump() const;
#endif // NDEBUG
};

/// Returns the operand edge when dereferenced.
class SBOperandUseIterator {
  SBUse Use;
  /// Don't let the user create a non-empty SBOperandUseIterator.
  SBOperandUseIterator(const SBUse &Use) : Use(Use) {}
  friend class SBUser;               // For constructor
  friend class SBPackInstruction;    // For constructor
  friend class SBShuffleInstruction; // For constructor
  friend class SBUnpackInstruction;  // For constructor

public:
  using difference_type = std::ptrdiff_t;
  using value_type = SBUse;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::input_iterator_tag;

  SBOperandUseIterator() {}
  value_type operator*() const;
  SBOperandUseIterator &operator++();
  bool operator==(const SBOperandUseIterator &Other) const {
    return Use == Other.Use;
  }
  bool operator!=(const SBOperandUseIterator &Other) const {
    return !(*this == Other);
  }
};

/// Returns user edge when dereferenced.
class SBUserUseIterator {
  SBUse Use;
  /// Don't let the user create a non-empty SBUserUseIterator.
  SBUserUseIterator(const SBUse &Use) : Use(Use) {}
  friend class SBValue;              // For constructor
  friend class SBPackInstruction;    // For constructor
  friend class SBShuffleInstruction; // For constructor

public:
  using difference_type = std::ptrdiff_t;
  using value_type = SBUse;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::input_iterator_tag;

  SBUserUseIterator() {}
  value_type operator*() const;
  SBUserUseIterator &operator++();
  bool operator==(const SBUserUseIterator &Other) const {
    return Use == Other.Use;
  }
  bool operator!=(const SBUserUseIterator &Other) const {
    return !(*this == Other);
  }
};

/// Simple adaptor class for SBUserUseIterator and SBOperandUseIterator that
/// returns \p RetTy* when dereferenced, that is SBUser* or SBValue*.
template <typename RetTy, typename ItTy> class RetTyAdaptor {
  ItTy It;

public:
  using difference_type = std::ptrdiff_t;
  using value_type = RetTy;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::input_iterator_tag;
  RetTyAdaptor(ItTy It) : It(It) {}
  RetTyAdaptor() = default;
  RetTyAdaptor &operator++() {
    ++It;
    return *this;
  }
  pointer operator*() const {
    static_assert(std::is_same<ItTy, SBUserUseIterator>::value ||
                      std::is_same<ItTy, SBOperandUseIterator>::value,
                  "Unsupported ItTy!");
    if constexpr (std::is_same<ItTy, SBUserUseIterator>::value) {
      return (*It).getUser();
    } else if constexpr (std::is_same<ItTy, SBOperandUseIterator>::value) {
      return (*It).get();
    }
  }
  bool operator==(const RetTyAdaptor &Other) const { return It == Other.It; }
  bool operator!=(const RetTyAdaptor &Other) const { return !(*this == Other); }
};

/// A SBValue has users. This is the base class.
class SBValue {
public:
  enum class ClassID : unsigned {
    Pack,
    Shuffle,
    Unpack,
    OpaqueInstr,
    Argument,
    User,
    Constant,
    Block,
    Store,
    Load,
    Cast,
    PHI,
    Select,
    BinOp,
    UnOp,
    Cmp,
    Function,
  };

protected:
  static const char *getSubclassIDStr(ClassID ID) {
    switch (ID) {
    case ClassID::Pack:
      return "Pack";
    case ClassID::Shuffle:
      return "Shuffle";
    case ClassID::Unpack:
      return "Unpack";
    case ClassID::OpaqueInstr:
      return "OpaqueInstr";
    case ClassID::Argument:
      return "Argument";
    case ClassID::User:
      return "User";
    case ClassID::Constant:
      return "Constant";
    case ClassID::Block:
      return "Block";
    case ClassID::Store:
      return "Store";
    case ClassID::Load:
      return "Load";
    case ClassID::Cast:
      return "Cast";
    case ClassID::PHI:
      return "PHI";
    case ClassID::Select:
      return "Select";
    case ClassID::BinOp:
      return "BinOp";
    case ClassID::UnOp:
      return "UnOp";
    case ClassID::Cmp:
      return "Cmp";
    case ClassID::Function:
      return "Function";
    }
    llvm_unreachable("Unimplemented ID");
  }

  /// For isa/dyn_cast.
  ClassID SubclassID;
#ifndef NDEBUG
  /// A unique ID used for forming the name (used for debugging).
  unsigned UID;
#endif
  /// The LLVM Value that corresponds to this SBValue.
  /// NOTE: Some SBInstructions, like Packs, may include more than one value.
  Value *Val = nullptr;
  friend class ValueAttorney; // For Val

  /// All values point to the context.
  SBContext &Ctxt;
  // This is used by eraseFromParent().
  void clearValue() { Val = nullptr; }
  template <typename ItTy, typename SBTy> friend class LLVMOpUserItToSBTy;

public:
  SBValue(ClassID SubclassID, Value *Val, SBContext &Ctxt);
  virtual ~SBValue() = default;
  ClassID getSubclassID() const { return SubclassID; }

  using use_iterator = SBUserUseIterator;
  using const_use_iterator = SBUserUseIterator;

  use_iterator use_begin();
  const_use_iterator use_begin() const {
    return const_cast<SBValue *>(this)->use_begin();
  }
  use_iterator use_end() {
    return use_iterator(SBUse(nullptr, nullptr, Ctxt));
  }
  const_use_iterator use_end() const {
    return const_cast<SBValue *>(this)->use_end();
  }

  iterator_range<use_iterator> uses() {
    return make_range<use_iterator>(use_begin(), use_end());
  }
  iterator_range<const_use_iterator> uses() const {
    return make_range<const_use_iterator>(use_begin(), use_end());
  }

  using user_iterator = RetTyAdaptor<SBUser, SBUserUseIterator>;
  using const_user_iterator = user_iterator;

  user_iterator user_begin();
  user_iterator user_end() {
    return user_iterator(SBUse(nullptr, nullptr, Ctxt));
  }
  const_user_iterator user_begin() const {
    return const_cast<SBValue *>(this)->user_begin();
  }
  const_user_iterator user_end() const {
    return const_cast<SBValue *>(this)->user_end();
  }

  iterator_range<user_iterator> users() {
    return make_range<user_iterator>(user_begin(), user_end());
  }
  iterator_range<const_user_iterator> users() const {
    return make_range<const_user_iterator>(user_begin(), user_end());
  }
  /// \Returns the number of unique users.
  /// WARNING: This is a linear-time operation.
  unsigned getNumUsers() const;
  /// \Returns the number of user edges (not necessarily to unique users).
  /// WARNING: This is a linear-time operation.
  unsigned getNumUses() const;
  /// WARNING: This can be expensive, as it is linear to the number of users.
  bool hasNUsersOrMore(unsigned Num) const;
  bool hasNUsesOrMore(unsigned Num) const {
    unsigned Cnt = 0;
    for (auto It = use_begin(), ItE = use_end(); It != ItE; ++It) {
      if (++Cnt >= Num)
        return true;
    }
    return false;
  }
  bool hasNUses(unsigned Num) const {
    unsigned Cnt = 0;
    for (auto It = use_begin(), ItE = use_end(); It != ItE; ++It) {
      if (++Cnt > Num)
        return true;
    }
    return Cnt == Num;
  }

  SBValue *getSingleUser() const;

  Type *getExpectedType() const { return SBUtils::getExpectedType(Val); }
  Type *getType() const { return Val->getType(); }
  bool isVector() const { return isa<FixedVectorType>(getExpectedType()); }

  SBContext &getContext() const;
  SandboxIRTracker &getTracker();
  virtual hash_code hashCommon() const {
    return hash_combine(SubclassID, &Ctxt, Val);
  }
  /// WARNING: DstU can be nullptr if it is in a BB that is not in SandboxIR!
  void replaceUsesWithIf(
      SBValue *OtherV,
      llvm::function_ref<bool(SBUser *DstU, unsigned OpIdx)> ShouldReplace);
  void replaceAllUsesWith(SBValue *Other);
  unsigned lanes() const { return SBUtils::getNumLanes(Val); }
  virtual hash_code hash() const = 0;
  friend hash_code hash_value(const SBValue &SBV) { return SBV.hash(); }
#ifndef NDEBUG
  /// Returns the name in the form 'T<number>.' like 'T1.'
  std::string getName() const;
  virtual void dumpCommonHeader(raw_ostream &OS) const;
  void dumpCommonFooter(raw_ostream &OS) const;
  void dumpCommonPrefix(raw_ostream &OS) const;
  void dumpCommonSuffix(raw_ostream &OS) const;
  void printAsOperandCommon(raw_ostream &OS) const;
  friend raw_ostream &operator<<(raw_ostream &OS, const SBValue &SBV) {
    SBV.dump(OS);
    return OS;
  }
  virtual void dump(raw_ostream &OS) const = 0;
  LLVM_DUMP_METHOD virtual void dump() const = 0;
  virtual void dumpVerbose(raw_ostream &OS) const = 0;
  LLVM_DUMP_METHOD virtual void dumpVerbose() const = 0;
#endif
};

/// Helper Attorney-Client class that gives access to the underlying IR.
class ValueAttorney {
private:
  static Value *getValue(const SBValue *SBV) { return SBV->Val; }
  friend class DependencyGraph;
  friend class SBValBundle;
  friend class ValueBundle;
  friend class SBPackInstruction;
  friend class SBUnpackInstruction;
  friend class SBShuffleInstruction;
  friend class SBBasicBlock;
  friend class SBAnalysis;
  friend class SBPassManager;
  friend class SBContext;
  friend class SBUser;
  friend class MemSeedContainer; // TODO: Removeme
  friend class SBInstruction;
  friend class SandboxIRTracker; // TODO: Removeme
  friend class SchedBundle;     // TODO: Removeme
  friend class SBStoreInstruction;
  friend class SBLoadInstruction;
  friend class SBCastInstruction;
  friend class SBPHINode;
  friend class SBSelectInstruction;
  friend class SBBinaryOperator;
  friend class SBUnaryOperator;
  friend class SBCmpInstruction;
  friend class SBRegionBuilderFromMD;
  friend class SBRegion;
  friend void SBUtils::propagateMetadata(SBInstruction *SBI,
                                          const SBValBundle &SBVals);
  friend Type *SBUtils::getExpectedType(SBValue *);
  friend std::optional<int>
  SBUtils::getPointerDiffInBytes(SBLoadInstruction *, SBLoadInstruction *,
                                  ScalarEvolution &, const DataLayout &);
  friend std::optional<int>
  SBUtils::getPointerDiffInBytes(SBStoreInstruction *,
                                  SBStoreInstruction *, ScalarEvolution &,
                                  const DataLayout &);
  friend bool SBUtils::areConsecutive(SBLoadInstruction *,
                                       SBLoadInstruction *, ScalarEvolution &,
                                       const DataLayout &);
  friend bool SBUtils::areConsecutive(SBStoreInstruction *,
                                       SBStoreInstruction *,
                                       ScalarEvolution &, const DataLayout &);
  friend void
  SBValue::replaceUsesWithIf(SBValue *,
                               llvm::function_ref<bool(SBUser *, unsigned)>);
  friend class Scheduler;
  friend class SBOperandUseIterator;
  friend class SBBBIterator;
  friend ReplaceAllUsesWith::ReplaceAllUsesWith(SBValue *, SandboxIRTracker &);
  friend ReplaceUsesOfWith::ReplaceUsesOfWith(SBUser *, SBValue *,
                                              SBValue *, SandboxIRTracker &);
  friend class DeleteOnAccept;
  friend class CreateAndInsertInstr;
  friend class EraseFromParent;
};

/// A function argument.
class SBArgument : public SBValue {
  SBArgument(Argument *Arg, SBContext &SBCtxt);
  friend class SBContext; // for createSBArgument()

public:
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBArgument &TArg) { return TArg.hash(); }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, const SBArgument &TArg) {
    TArg.dump(OS);
    return OS;
  }
  void printAsOperand(raw_ostream &OS) const;
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

/// A SBValue with operands.
class SBUser : public SBValue {
protected:
  SBUser(ClassID ID, Value *V, SBContext &SBCtxt);
  friend class SBInstruction; // For constructors.

  /// \Returns the SBUse edge that corresponds to \p OpIdx.
  /// Note: This is the default implementation that works for instructions that
  /// match the underlying LLVM instruction. All others should use a different
  /// implementation.
  SBUse getOperandUseDefault(unsigned OpIdx, bool Verify) const;
  virtual SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const = 0;
  friend class SBOperandUseIterator; // for getOperandUseInternal()

  /// \Returns true if \p Use should be considered as an edge to its SandboxIR
  /// operand. Most instructions should return true.
  /// Currently it is only Uses from Vectors into Packs that return false.
  virtual bool isRealOperandUse(Use &Use) const = 0;
  friend class SBUserUseIterator; // for isRealOperandUse()

  /// The default implementation works only for single-LLVMIR-instruction
  /// SBUsers and only if they match exactly the LLVM instruction.
  unsigned getUseOperandNoDefault(const SBUse &Use) const {
    return Use.LLVMUse->getOperandNo();
  }

public:
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  using op_iterator = SBOperandUseIterator;
  using const_op_iterator = SBOperandUseIterator;
  using op_range = iterator_range<op_iterator>;
  using const_op_range = iterator_range<const_op_iterator>;

  virtual op_iterator op_begin();
  virtual op_iterator op_end();
  virtual const_op_iterator op_begin() const;
  virtual const_op_iterator op_end() const;

  op_range operands() { return make_range<op_iterator>(op_begin(), op_end()); }
  const_op_range operands() const {
    return make_range<const_op_iterator>(op_begin(), op_end());
  }
  hash_code hashCommon() const override {
    auto Hash = SBValue::hashCommon();
    for (SBValue *Op : operands())
      Hash = hash_combine(Hash, Op);
    return Hash;
  }
  SBValue *getOperand(unsigned OpIdx) const {
    return getOperandUse(OpIdx).get();
  }
  /// \Returns the operand edge for \p OpIdx. NOTE: This should also work for
  /// OpIdx == getNumOperands(), which is used for op_end().
  SBUse getOperandUse(unsigned OpIdx) const {
    return getOperandUseInternal(OpIdx, /*Verify=*/true);
  }
  /// \Returns the operand index of \p Use.
  virtual unsigned getUseOperandNo(const SBUse &Use) const = 0;
  SBValue *getSingleOperand() const;
  virtual void setOperand(unsigned OperandIdx, SBValue *Operand);
  virtual unsigned getNumOperands() const {
    return isa<User>(Val) ? cast<User>(Val)->getNumOperands() : 0;
  }
  /// Replaces any operands that match \p FromV with \p ToV. Returns whether any
  /// operands were replaced.
  /// WARNING: This will replace even uses that are not in SandboxIR!
  bool replaceUsesOfWith(SBValue *FromV, SBValue *ToV);

#ifndef NDEBUG
  void dumpCommonHeader(raw_ostream &OS) const final;
#endif

protected:
  /// \Returns the operand index that corresponds to \p UseToMatch.
  virtual unsigned getOperandUseIdx(const Use &UseToMatch) const;
  friend class SBUserAttorney; // For testing
  friend void
  SBValue::replaceUsesWithIf(SBValue *,
                               llvm::function_ref<bool(SBUser *, unsigned)>);
};

/// A simple client-attorney class that exposes some protected members of
/// SBUser for use in tests.
class SBUserAttorney {
public:
  // For testing.
  static unsigned getOperandUseIdx(const SBUser *SBU,
                                   const Use &UseToMatch) {
    return SBU->getOperandUseIdx(UseToMatch);
  }
};

class SBConstant : public SBUser {
  /// Use SBContext::createSBConstant() instead.
  SBConstant(Constant *C, SBContext &SBCtxt);
  friend class SBContext; // For constructor.
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(Use &Use) const final { return true; }

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  SBContext &getParent() const { return getContext(); }
  hash_code hashCommon() const final { return SBUser::hashCommon(); }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBConstant &SBC) {
    return SBC.hash();
  }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, const SBConstant &SBC) {
    SBC.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBInstruction;

/// The SBBasicBlock::iterator.
class SBBBIterator {
public:
  using difference_type = std::ptrdiff_t;
  using value_type = SBInstruction;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::bidirectional_iterator_tag;

private:
  BasicBlock *BB;
  /// This should always point to the bottom IR instruction of a multi-IR
  /// SBInstruction.
  BasicBlock::iterator It;
  SBContext *SBCtxt;
  pointer getSBI(BasicBlock::iterator It) const;

public:
  SBBBIterator() : BB(nullptr), SBCtxt(nullptr) {}
  SBBBIterator(BasicBlock *BB, BasicBlock::iterator It, SBContext *SBCtxt)
      : BB(BB), It(It), SBCtxt(SBCtxt) {}
  reference operator*() const { return *getSBI(It); }
  SBBBIterator &operator++();
  SBBBIterator operator++(int) {
    auto Copy = *this;
    ++*this;
    return Copy;
  }
  SBBBIterator &operator--();
  SBBBIterator operator--(int) {
    auto Copy = *this;
    --*this;
    return Copy;
  }
  bool operator==(const SBBBIterator &Other) const {
    assert(SBCtxt == Other.SBCtxt &&
           "SBBBIterators in different context!");
    return It == Other.It;
  }
  bool operator!=(const SBBBIterator &Other) const {
    return !(*this == Other);
  }
  /// \Returns true if the internal iterator is at the beginning of the IR BB.
  /// NOTE: This is meant to be used internally, during the construction of a
  /// SBBB, during which SBBB->begin() fails due to the missing mapping of
  /// BB->begin() to SandboxIR.
  bool atBegin() const;
  /// \Returns the SBInstruction that corresponds to this iterator, or null if
  /// the instruction is not found in the IR-to-SandboxIR tables.
  pointer get() const { return getSBI(It); }
};

/// A SBUser with operands and opcode.
class SBInstruction : public SBUser {
protected:
  /// Don't create objects of this class. Use a sub-class instead.
  SBInstruction(ClassID ID, Instruction *I, SBContext &SBCtxt);
  friend class SBPackInstruction;
  friend class SBUnpackInstruction;
  friend class SBShuffleInstruction;
  friend class SBOpaqueInstruction;
  friend class SBStoreInstruction;
  friend class SBLoadInstruction;
  friend class SBCastInstruction;
  friend class SBPHINode;
  friend class SBCmpInstruction;
  friend class SBSelectInstruction;
  friend class SBBinaryOperator;
  friend class SBUnaryOperator;

  /// A SBInstruction may map to multiple IR Instruction. This returns its
  /// topmost IR instruction.
  Instruction *getTopmostIRInstruction() const;

  /// \Returns all IR instructions that make up this SBInstruction in reverse
  /// program order.
  Bundle<Instruction *> getIRInstrs() const;
  friend class SBCostModel; // For getIRInstrs().
  // TODO: Make this pure virtual.
  /// \Returns all IR instructions with external operands. Note: This is useful
  /// for multi-IR instructions like Packs, that are composed of both
  /// internal-only and external-facing IR Instructions.
  virtual Bundle<Instruction *> getExternalFacingIRInstrs() const;
  friend void DeleteOnAccept::apply();
  friend ReplaceAllUsesWith::ReplaceAllUsesWith(SBValue *, SandboxIRTracker &);
  friend ReplaceUsesOfWith::ReplaceUsesOfWith(SBUser *, SBValue *,
                                              SBValue *, SandboxIRTracker &);
  friend bool SBUser::replaceUsesOfWith(SBValue *, SBValue *);
  friend class EraseFromParent;
  friend class DeleteOnAccept;

public:
  enum class Opcode {
    // Vector-related
    Shuffle,
    Pack,
    Unpack,
    // Casts
    ZExt,
    SExt,
    FPToUI,
    FPToSI,
    FPExt,
    PtrToInt,
    IntToPtr,
    SIToFP,
    UIToFP,
    Trunc,
    FPTrunc,
    BitCast,
    // Cmp
    FCmp,
    ICmp,
    // Select
    Select,
    // Unary
    FNeg,
    // BinOp
    Add,
    FAdd,
    Sub,
    FSub,
    Mul,
    FMul,
    UDiv,
    SDiv,
    FDiv,
    URem,
    SRem,
    FRem,
    Shl,
    LShr,
    AShr,
    And,
    Or,
    Xor,
    // Mem
    Load,
    Store,
    // Opaque for everything else
    Opaque,
  };

protected:
  /// Maps SBInstruction::Opcode to its corresponding IR opcode, if it exists.
  static Instruction::UnaryOps getIRUnaryOp(Opcode Opc);
  static Instruction::BinaryOps getIRBinaryOp(Opcode Opc);
  static Instruction::CastOps getIRCastOp(Opcode Opc);

  // Metadata is LLMV IR, so protect it. Access this via the
  // SBInstructionAttorney class.
  MDNode *getMetadata(unsigned KindID) const {
    return cast<Instruction>(Val)->getMetadata(KindID);
  }
  MDNode *getMetadata(StringRef Kind) const {
    return cast<Instruction>(Val)->getMetadata(Kind);
  }
  friend class SBInstructionAttorney;

public:
  static const char *getOpcodeName(Opcode Opc);
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, Opcode Opc) {
    OS << getOpcodeName(Opc);
    return OS;
  }
#endif
  /// This is used by SBBasicBlcok::iterator.
  virtual unsigned getNumOfIRInstrs() const = 0;
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  SBBBIterator getIterator() const;
  SBInstruction *getNextNode() const;
  SBInstruction *getPrevNode() const;
  /// \Returns the opcode of the Instruction contained.
  Opcode getOpcode() const;
  /// Detach this from its parent SBBasicBlock without deleting it.
  void removeFromParent();
  /// Detach this SBValue from its parent and delete it.
  void eraseFromParent();
  /// \Returns the parent graph or null if there is no parent graph, i.e., when
  /// it holds a Constant.
  SBBasicBlock *getParent() const;
  bool isFPMath() const { return isa<FPMathOperator>(Val); }
  FastMathFlags getFastMathFlags() const {
    return cast<Instruction>(Val)->getFastMathFlags();
  }
  bool canHaveWrapFlags() const {
    return isa<OverflowingBinaryOperator>(Val) || isa<TruncInst>(Val);
  }
  bool hasNoUnsignedWrap() const {
    if (!canHaveWrapFlags())
      return false;
    return cast<Instruction>(Val)->hasNoUnsignedWrap();
  }
  bool hasNoSignedWrap() const {
    if (!canHaveWrapFlags())
      return false;
    return cast<Instruction>(Val)->hasNoSignedWrap();
  }
  /// \Returns true if this is a landingpad, a catchpad or a cleanuppadd
  bool isPad() const {
    return isa<LandingPadInst>(Val) || isa<CatchPadInst>(Val) ||
           isa<CleanupPadInst>(Val);
  }
  bool isFenceLike() const { return cast<Instruction>(Val)->isFenceLike(); }
  bool comesBefore(SBInstruction *Other) {
    // TODO: Replace this with a better algorithm and don't rely on IR.
    // TODO: Once we don't rely on IR remove ValueAttorney.
    return cast<Instruction>(Val)->comesBefore(
        cast<Instruction>(ValueAttorney::getValue(Other)));
  }
  bool comesAfter(SBInstruction *Other) { return Other->comesBefore(this); }
  void moveBefore(SBBasicBlock &SBBB, const SBBBIterator &WhereIt);
  void moveBefore(SBInstruction *Before) {
    moveBefore(*Before->getParent(), Before->getIterator());
  }
  void moveAfter(SBInstruction *After) {
    moveBefore(*After->getParent(), std::next(After->getIterator()));
  }
  hash_code hashCommon() const override {
    return hash_combine(SBUser::hashCommon(), getParent());
  }
  void insertBefore(SBInstruction *BeforeI);
  void insertInto(SBBasicBlock *SBBB, const SBBBIterator &WhereIt);

  bool mayWriteToMemory() const {
    return cast<Instruction>(Val)->mayWriteToMemory();
  }
  bool mayReadFromMemory() const {
    return cast<Instruction>(Val)->mayReadFromMemory();
  }
  bool isTerminator() const { return cast<Instruction>(Val)->isTerminator(); }

  bool isStackRelated() const {
    auto IsInAlloca = [](Instruction *I) {
      return isa<AllocaInst>(I) && cast<AllocaInst>(I)->isUsedWithInAlloca();
    };
    auto *I = cast<Instruction>(Val);
    return match(I, m_Intrinsic<Intrinsic::stackrestore>()) ||
           match(I, m_Intrinsic<Intrinsic::stacksave>()) || IsInAlloca(I);
  }
  /// We consider \p I as a Mem instruction if it accesses memory or if it is
  /// stack-related. This is used to determine whether this instruction needs
  /// dependency edges.
  bool isMemInst() const {
    auto *I = cast<Instruction>(Val);
    return SBUtils::isMem(I) || isStackRelated();
  }
  bool isDbgInfo() const {
    auto *I = cast<Instruction>(Val);
    return isa<DbgInfoIntrinsic>(I);
  }
  /// \Returns the number of successors that this terminator instruction has.
  unsigned getNumSuccessors() const LLVM_READONLY {
    return cast<Instruction>(Val)->getNumSuccessors();
  }
  SBBasicBlock *getSuccessor(unsigned Idx) const LLVM_READONLY;

#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBInstruction &SBI) {
    SBI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

/// A client-attorney class for SBInstruction.
class SBInstructionAttorney {
public:
  friend class SBRegionBuilderFromMD;
  static MDNode *getMetadata(const SBInstruction *SBI, unsigned KindID) {
    return SBI->getMetadata(KindID);
  }
  static MDNode *getMetadata(const SBInstruction *SBI, StringRef Kind) {
    return SBI->getMetadata(Kind);
  }
};

class SBCmpInstruction : public SBInstruction {
  /// Use SBBasicBlock::createSBCmpInstruction(). Don't call the
  /// constructor directly.
  SBCmpInstruction(CmpInst *CI, SBContext &Ctxt)
      : SBInstruction(ClassID::Cmp, CI, Ctxt) {}
  friend SBContext; // for SBCmpInstruction()

  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(Use &Use) const final { return true; }

public:
  static SBValue *create(CmpInst::Predicate Pred, SBValue *LHS,
                           SBValue *RHS, SBInstruction *InsertBefore,
                           SBContext &SBCtxt, const Twine &Name = "",
                           MDNode *FPMathTag = nullptr);
  static SBValue *create(CmpInst::Predicate Pred, SBValue *LHS,
                           SBValue *RHS, SBBasicBlock *InsertAtEnd,
                           SBContext &SBCtxt, const Twine &Name = "",
                           MDNode *FPMathTag = nullptr);
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBCmpInstruction &SBSI) {
    return SBSI.hash();
  }
  auto getPredicate() const { return cast<CmpInst>(Val)->getPredicate(); }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBCmpInstruction &SBSI) {
    SBSI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

/// Traits for DenseMap.
template <> struct DenseMapInfo<SBInstruction::Opcode> {
  static inline SBInstruction::Opcode getEmptyKey() {
    return (SBInstruction::Opcode)-1;
  }
  static inline SBInstruction::Opcode getTombstoneKey() {
    return (SBInstruction::Opcode)-2;
  }
  static unsigned getHashValue(const SBInstruction::Opcode &B) {
    return (unsigned)B;
  }
  static bool isEqual(const SBInstruction::Opcode &B1,
                      const SBInstruction::Opcode &B2) {
    return B1 == B2;
  }
};

class SBStoreInstruction : public SBInstruction {
  /// Use SBBasicBlock::createSBStoreInstruction(). Don't call the
  /// constructor directly.
  SBStoreInstruction(StoreInst *SI, SBContext &Ctxt)
      : SBInstruction(ClassID::Store, SI, Ctxt) {}
  friend SBContext; // for SBStoreInstruction()
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(Use &Use) const final { return true; }

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static SBStoreInstruction *create(SBValue *V, SBValue *Ptr,
                                      MaybeAlign Align,
                                      SBInstruction *InsertBefore,
                                      SBContext &SBCtxt);
  static SBStoreInstruction *create(SBValue *V, SBValue *Ptr,
                                      MaybeAlign Align,
                                      SBBasicBlock *InsertAtEnd,
                                      SBContext &SBCtxt);
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBStoreInstruction &SBSI) {
    return SBSI.hash();
  }
  SBValue *getValueOperand() const;
  SBValue *getPointerOperand() const;
  Align getAlign() const { return cast<StoreInst>(Val)->getAlign(); }
  bool isSimple() const { return cast<StoreInst>(Val)->isSimple(); }
  bool isUnordered() const { return cast<StoreInst>(Val)->isUnordered(); }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBStoreInstruction &SBSI) {
    SBSI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBLoadInstruction : public SBInstruction {
  /// Use SBBasicBlock::createSBLoadInstruction(). Don't call the
  /// constructor directly.
  SBLoadInstruction(LoadInst *LI, SBContext &Ctxt)
      : SBInstruction(ClassID::Load, LI, Ctxt) {}
  friend SBContext; // for SBLoadInstruction()
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(Use &Use) const final { return true; }

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }

  unsigned getNumOfIRInstrs() const final { return 1u; }
  static SBLoadInstruction *create(Type *Ty, SBValue *Ptr, MaybeAlign Align,
                                     SBInstruction *InsertBefore,
                                     SBContext &SBCtxt,
                                     const Twine &Name = "");
  static SBLoadInstruction *create(Type *Ty, SBValue *Ptr, MaybeAlign Align,
                                     SBBasicBlock *InsertAtEnd,
                                     SBContext &SBCtxt,
                                     const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBLoadInstruction &SBLI) {
    return SBLI.hash();
  }
  SBValue *getPointerOperand() const;
  Align getAlign() const { return cast<LoadInst>(Val)->getAlign(); }
  bool isUnordered() const { return cast<LoadInst>(Val)->isUnordered(); }
  bool isSimple() const { return cast<LoadInst>(Val)->isSimple(); }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBLoadInstruction &SBLI) {
    SBLI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBCastInstruction : public SBInstruction {
  /// Use SBBasicBlock::createSBCastInstruction(). Don't call the
  /// constructor directly.
  SBCastInstruction(CastInst *CI, SBContext &Ctxt)
      : SBInstruction(ClassID::Cast, CI, Ctxt) {}
  friend SBContext; // for SBCastInstruction()
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(Use &Use) const final { return true; }

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static SBValue *create(Type *Ty, Opcode Op, SBValue *Operand,
                           SBInstruction *InsertBefore, SBContext &SBCtxt,
                           const Twine &Name = "");
  static SBValue *create(Type *Ty, Opcode Op, SBValue *Operand,
                           SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                           const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBCastInstruction &SBCI) {
    return SBCI.hash();
  }
  Instruction::CastOps getOpcode() const {
    return cast<CastInst>(Val)->getOpcode();
  }
  Type *getSrcTy() const { return cast<CastInst>(Val)->getSrcTy(); }
  Type *getDestTy() const { return cast<CastInst>(Val)->getDestTy(); }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBCastInstruction &SBCI) {
    SBCI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBPHINode : public SBInstruction {
  /// Use SBBasicBlock::createSBPHINode(). Don't call the
  /// constructor directly.
  SBPHINode(PHINode *PHI, SBContext &Ctxt)
      : SBInstruction(ClassID::PHI, PHI, Ctxt) {}
  friend SBContext; // for SBPHINode()
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(Use &Use) const final { return true; }

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static SBValue *create(Type *Ty, unsigned NumReservedValues,
                           SBInstruction *InsertBefore, SBContext &SBCtxt,
                           const Twine &Name = "");
  static SBValue *create(Type *Ty, unsigned NumReservedValues,
                           SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                           const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBPHINode &SBCI) {
    return SBCI.hash();
  }
  Type *getSrcTy() const { return cast<CastInst>(Val)->getSrcTy(); }
  Type *getDestTy() const { return cast<CastInst>(Val)->getDestTy(); }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, const SBPHINode &SBCI) {
    SBCI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBSelectInstruction : public SBInstruction {
  /// Use SBBasicBlock::createSBSelectInstruction(). Don't call the
  /// constructor directly.
  SBSelectInstruction(SelectInst *CI, SBContext &Ctxt)
      : SBInstruction(ClassID::Select, CI, Ctxt) {}
  friend SBContext; // for SBSelectInstruction()
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(Use &Use) const final { return true; }

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static SBValue *create(SBValue *Cond, SBValue *True, SBValue *False,
                           SBInstruction *InsertBefore, SBContext &SBCtxt,
                           const Twine &Name = "");
  static SBValue *create(SBValue *Cond, SBValue *True, SBValue *False,
                           SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
                           const Twine &Name = "");
  SBValue *getCondition() { return getOperand(0); }
  SBValue *getTrueValue() { return getOperand(1); }
  SBValue *getFalseValue() { return getOperand(2); }

  void setCondition(SBValue *New) { setOperand(0, New); }
  void setTrueValue(SBValue *New) { setOperand(1, New); }
  void setFalseValue(SBValue *New) { setOperand(2, New); }
  void swapValues() { cast<SelectInst>(Val)->swapValues(); }
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBSelectInstruction &SBSI) {
    return SBSI.hash();
  }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBSelectInstruction &SBSI) {
    SBSI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBBinaryOperator : public SBInstruction {
  /// Use SBBasicBlock::createSBBinaryOperator(). Don't call the
  /// constructor directly.
  SBBinaryOperator(BinaryOperator *BO, SBContext &Ctxt)
      : SBInstruction(ClassID::BinOp, BO, Ctxt) {}
  friend SBContext; // for SBSelectInstruction()
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(Use &Use) const final { return true; }

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static SBValue *createWithCopiedFlags(SBInstruction::Opcode Op,
                                          SBValue *LHS, SBValue *RHS,
                                          SBValue *CopyFrom,
                                          SBInstruction *InsertBefore,
                                          SBContext &SBCtxt,
                                          const Twine &Name = "");
  static SBValue *createWithCopiedFlags(SBInstruction::Opcode Op,
                                          SBValue *LHS, SBValue *RHS,
                                          SBValue *CopyFrom,
                                          SBBasicBlock *InsertAtEnd,
                                          SBContext &SBCtxt,
                                          const Twine &Name = "");

  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBBinaryOperator &SBBO) {
    return SBBO.hash();
  }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBBinaryOperator &SBBO) {
    SBBO.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBUnaryOperator : public SBInstruction {
  /// Use SBBasicBlock::createSBUnaryOperator(). Don't call the
  /// constructor directly.
  SBUnaryOperator(UnaryOperator *UO, SBContext &Ctxt)
      : SBInstruction(ClassID::UnOp, UO, Ctxt) {}
  friend SBContext; // for SBSelectInstruction()
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(Use &Use) const final { return true; }

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static SBValue *createWithCopiedFlags(SBInstruction::Opcode Op,
                                          SBValue *OpV, SBValue *CopyFrom,
                                          SBInstruction *InsertBefore,
                                          SBContext &SBCtxt,
                                          const Twine &Name = "");
  static SBValue *createWithCopiedFlags(SBInstruction::Opcode Op,
                                          SBValue *OpV, SBValue *CopyFrom,
                                          SBBasicBlock *InsertAtEnd,
                                          SBContext &SBCtxt,
                                          const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBUnaryOperator &SBUO) {
    return SBUO.hash();
  }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBUnaryOperator &SBUO) {
    SBUO.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SBOpaqueInstruction : public SBInstruction {
  /// Use SBBasicBlock::createSBOpaqueInstruction(). Don't call the
  /// constructor directly.
  SBOpaqueInstruction(Instruction *I, SBContext &Ctxt)
      : SBInstruction(ClassID::OpaqueInstr, I, Ctxt) {}
  SBOpaqueInstruction(ClassID SubclassID, Instruction *I, SBContext &Ctxt)
      : SBInstruction(SubclassID, I, Ctxt) {}
  friend class SBBasicBlock;
  friend class SBContext; // For creating SB constants.
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(Use &Use) const final { return true; }

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SBOpaqueInstruction &SBGI) {
    return SBGI.hash();
  }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBOpaqueInstruction &SBGI) {
    SBGI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

/// The InsertElementInsts that make up a pack.
/// NOTE: We are using a seprate class for it because we need to get it
/// initialized before the call to createIR().
class PackInstrBundle {
protected:
  /// Contains all instructions in the packing pattern, including Inserts into
  /// the final vector and also Extracts from vector operands.
  Bundle<Instruction *> PackInstrs;

  /// Given an \p InsertI return either its use accessing its immediate
  /// operand, or the use of the extract providing the insert's operand, if
  /// part of a pack-from-vector patter. In any way this returns the use
  /// pointing to the external operand.
  Use &getExternalFacingOperandUse(InsertElementInst *InsertI) const;
  /// \Returns the Pack Insert at \p Lane or nullptr.
  InsertElementInst *getInsertAtLane(int Lane) const;
  /// Iterate over operands and call:
  ///   Fn(Use, IsRealOp)
  void doOnOperands(function_ref<bool(Use &, bool)> DoOnOpFn) const;
  /// \Returns the operand at \p OperandIdx. This works for both insert-only and
  /// extract/insert pack instructions.
  Use &getBndlOperandUse(unsigned OperandIdx) const;
  /// \Returns the number of operands. This works for both insert-only and
  /// extract/insert pack instructions.
  /// NOTE: This is linear to the number of entries in PackInstrs.
  unsigned getNumOperands() const;
  /// \Returns the top-most InsertElement instruction. This is the one that
  /// inserts into poison.
  InsertElementInst *getTopInsert() const;
  /// \Returns the bottom InsertElement instr.
  InsertElementInst *getBotInsert() const;

public:
  PackInstrBundle() = default;
  PackInstrBundle(const ValueBundle &PackInstrs);
#ifndef NDEBUG
  void verify() const;
#endif
};

/// Packs multiple scalar values into a vector.
class SBPackInstruction : public PackInstrBundle, public SBInstruction {
  friend SBContext; // for eraseBundleInstrs().
  /// Use SBBasicBlock::createSBPackInstruction(). Don't call the
  /// constructor directly.
  /// Create a Pack that packs \p ToPack.
  SBPackInstruction(const SBValBundle &ToPack, SBBasicBlock *Parent);
  /// Create a Pack from its LLVM IR values.
  SBPackInstruction(const ValueBundle &Instrs, SBContext &SBCtxt);
  friend class SBBasicBlock;
  InsertElementInst *getBottomInsert(const ValueBundle &Instrs) const;
  /// \Returns pack instrs and constants.
  static std::variant<ValueBundle, Constant *>
  createIR(const SBValBundle &ToPack, SBBasicBlock *Parent,
           SBInstruction *BeforeI = nullptr);
  /// \Returns all the IR instructions that make up this Pack in reverse program
  /// order.
  Bundle<Instruction *> getIRInstrsInternal() const;
  // Friend for getIRInstrsInternal().
  friend Bundle<Instruction *> SBInstruction::getIRInstrs() const;
  virtual Bundle<Instruction *> getExternalFacingIRInstrs() const final;
  friend void SBInstruction::eraseFromParent();

protected:
#ifndef NDEBUG
  // Public for testing
public:
#endif
  unsigned getOperandUseIdx(const Use &UseToMatch) const override;
  SBUse getOperandUseInternal(unsigned OperandIdx, bool Verify) const final;
  bool isRealOperandUse(Use &OpUse) const final;

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    for (auto [OpIdx, OpUse] : enumerate(operands()))
      if (Use == OpUse)
        return OpIdx;
    llvm_unreachable("Can't find Use!");
  }
  unsigned getNumOfIRInstrs() const final { return PackInstrs.size(); }
  // Since a Pack corresponds to a sequence of insertelement instructions,
  // the internals of which we don't care too much from the vectorizer's
  // persctive, we need to make sure the operands work as expected.
  SBUser::op_iterator op_begin() final;
  SBUser::op_iterator op_end() final;
  SBUser::const_op_iterator op_begin() const final;
  SBUser::const_op_iterator op_end() const final;

  void setOperand(unsigned OperandIdx, SBValue *Operand) final;
  unsigned getNumOperands() const final {
    return PackInstrBundle::getNumOperands();
  }
  /// \Returns the Inserts that do the packing.
  const auto &getPackInstrs() const { return PackInstrs; }
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const final {
    auto Hash = SBValue::hashCommon();
    for (SBValue *Op : operands())
      Hash = hash_combine(Hash, Op);
    return Hash;
  }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
  void dumpVerbose(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dumpVerbose() const final;
#endif
  friend class SBPackInstructionAttorney;
};

/// Client-attorney class for accessing SBPackInstruction's protected members.
class SBPackInstructionAttorney {
public:
  // For tests
  static auto getExternalFacingIRInstrs(SBPackInstruction *Pack) {
    return Pack->getExternalFacingIRInstrs();
  }
  // For tests
  static auto getIRInstrs(SBPackInstruction *Pack) {
    return Pack->getIRInstrs();
  }
};

/// Reorders the lanes of its operand.
class SBShuffleInstruction : public SBInstruction {
private:
  /// Use SBBasicBlock::createSBShuffleInstruction(). Don't call the
  /// constructor directly.
  SBShuffleInstruction(ShuffleVectorInst *ShuffleI, SBContext &Ctxt)
      : SBInstruction(ClassID::Shuffle, ShuffleI, Ctxt) {}
  SBShuffleInstruction(const ShuffleMask &Mask, SBValue *Op,
                         SBBasicBlock *Parent);
  friend class SBContext;
  ShuffleVectorInst *createIR(const ShuffleMask &Mask, SBValue *Op,
                              SBBasicBlock *Parent);
  SBUse getOperandUseInternal(unsigned OperandIdx, bool Verify) const final {
    return getOperandUseDefault(OperandIdx, Verify);
  }
  bool isRealOperandUse(Use &Use) const final { return true; }

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
#ifndef NDEBUG
    assert(Use.getUser() == this && "Use does not point to this!");
    assert(Use.LLVMUse == &cast<User>(Val)->getOperandUse(0) &&
           "Use does not match!");
#endif
    return 0u;
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  // Since a Shuffle is a specific single-input SBInstruction,
  // we need to make sure the operands work as expected.
  SBUser::op_iterator op_begin() final;
  SBUser::op_iterator op_end() final;
  SBUser::const_op_iterator op_begin() const final;
  SBUser::const_op_iterator op_end() const final;

  void setOperand(unsigned OperandIdx, SBValue *Operand) final;
  unsigned getNumOperands() const final { return 1; }

  ShuffleMask getMask() const {
    return ShuffleMask(cast<ShuffleVectorInst>(Val)->getShuffleMask());
  }
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const final { return hash_combine(getMask(), hashCommon()); }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
  void dumpVerbose(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dumpVerbose() const final;
#endif
};

/// Extracts a scalar or a vector from a vector. Scalars are extracted with an
/// `extreactelement`, while vectors with a `shufflevector`.
class SBUnpackInstruction : public SBInstruction {
  SBUse getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    llvm::Use *LLVMUse;
    if (OpIdx != getNumOperands()) {
      unsigned LLVMOpIdx = isa<ExtractElementInst>(Val) ? OpIdx : OpIdx + 1;
      LLVMUse = &cast<User>(Val)->getOperandUse(LLVMOpIdx);
    } else
      LLVMUse = cast<User>(Val)->op_end();
    return SBUse(LLVMUse, const_cast<SBUnpackInstruction *>(this), Ctxt);
  }
  bool isRealOperandUse(Use &Use) const final { return true; }

public:
  unsigned getUseOperandNo(const SBUse &Use) const final {
    unsigned LLVMOperandNo = Use.LLVMUse->getOperandNo();
    return isa<ExtractElementInst>(Val) ? LLVMOperandNo : LLVMOperandNo - 1;
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  /// Use SBBasicBlock::createSBUnpackInstruction(). Don't call the
  /// constructor directly.
  SBUnpackInstruction(ExtractElementInst *ExtractI, SBValue *UnpackOp,
                        unsigned UnpackLane, SBContext &SBCtxt);
  SBUnpackInstruction(ShuffleVectorInst *ShuffleI, SBValue *UnpackOp,
                        unsigned UnpackLane, SBContext &SBCtxt);
  friend class SBBasicBlock;

  static Value *createIR(SBValue *UnpackOp, SBBasicBlock *Parent,
                         unsigned Lane, unsigned Lanes);
  /// \Returns true if \p ShuffleI is an unpack.
  static bool isUnpack(ShuffleVectorInst *ShuffleI) {
    auto Mask = ShuffleI->getShuffleMask();
    auto NumInputElms =
        SBUtils::getNumElements(ShuffleI->getOperand(0)->getType());
    if (!ShuffleVectorInst::isSingleSourceMask(Mask, NumInputElms))
      return false;
    if (!ShuffleI->changesLength())
      return false;
    if (Mask.size() == 0 || (int)Mask.size() == NumInputElms)
      return false;
    if (!isa<PoisonValue>(ShuffleI->getOperand(0)))
      return false;
    // We expect element indexes in order.
    int Mask0 = Mask[0];
    if (Mask0 < NumInputElms)
      return false;
    int LastElm = Mask0;
    for (auto Elm : drop_begin(Mask)) {
      if (Elm != LastElm + 1)
        return false;
      LastElm = Elm;
    }
    return true;
  }
  /// Helper that returns the lane unpacked by \p ShuffleI.
  static int getShuffleLane(ShuffleVectorInst *ShuffleI) {
    assert(isUnpack(ShuffleI) && "Expected an unpack!");
    int TotalElms = SBUtils::getNumLanes(ShuffleI->getOperand(0)->getType());
    int Elm = ShuffleI->getMaskValue(0);
    int Lane = Elm - TotalElms;
    assert(Lane >= 0 && "Expected non-negative!");
    return Lane;
  }
  unsigned getUnpackLane() const {
    ConstantInt *IdxC = nullptr;
    if (auto *Extract = dyn_cast<ExtractElementInst>(Val)) {
      IdxC = cast<ConstantInt>(Extract->getIndexOperand());
      return IdxC->getSExtValue();
    }
    return getShuffleLane(cast<ShuffleVectorInst>(Val));
  }
  unsigned getNumOperands() const final { return 1u; }
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  hash_code hash() const final {
    return hash_combine(getUnpackLane(), hashCommon());
  }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
  void dumpVerbose(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dumpVerbose() const final;
#endif
};

class SBContext;

class SBBasicBlock : public SBValue {
  // Needs to call getOrCreateSBValue()
  friend std::variant<ValueBundle, Constant *>
  SBPackInstruction::createIR(const SBValBundle &, SBBasicBlock *,
                                SBInstruction *);

public:
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From);
  /// Builds a graph that contains all values in \p BB in their original form
  /// i.e., no vectorization is taking place here.
  void buildSBBasicBlockFromIR(BasicBlock *BB);
  /// \Returns the iterator to the first non-PHI instruction.
  SBBBIterator getFirstNonPHIIt();

private:
  friend void SBValue::replaceUsesWithIf(
      SBValue *,
      llvm::function_ref<bool(SBUser *, unsigned)>);     // for ChangeTracker.
  friend void SBValue::replaceAllUsesWith(SBValue *);  // for ChangeTracker.
  friend void SBUser::setOperand(unsigned, SBValue *); // for ChangeTracker

  /// Detach SBBasicBlock from the underlying BB. This is called by the
  /// destructor.
  void detach();
  /// Use SBContext::createSBBasicBlock().
  SBBasicBlock(BasicBlock *BB, SBContext &SBCtxt);
  friend class SBContext; // For createSBBasicBlock().
  friend class SBBasicBlockAttorney;

public:
  ~SBBasicBlock();
  SBFunction *getParent() const;
  /// Detaches the block and its instructions from LLVM IR.
  void detachFromLLVMIR();
  using iterator = SBBBIterator;
  iterator begin() const;
  iterator end() const {
    auto *BB = cast<BasicBlock>(Val);
    return iterator(BB, BB->end(), &Ctxt);
  }
  SBContext &getContext() const { return Ctxt; }
  SandboxIRTracker &getTracker();
  SBInstruction *getTerminator() const;
  auto LLVMSize() const { return cast<BasicBlock>(Val)->size(); }

  hash_code hash() const final {
    return hash_combine(SBValue::hashCommon(),
                        hash_combine_range(begin(), end()));
  }
  friend hash_code hash_value(const SBBasicBlock &SBBB) {
    return SBBB.hash();
  }

  bool empty() const { return begin() == end(); }
  SBInstruction &front() const;
  SBInstruction &back() const;

#ifndef NDEBUG
  /// Verifies LLVM IR.
  void verifyFunctionIR() const {
    assert(!verifyFunction(*cast<BasicBlock>(Val)->getParent(), &errs()));
  }
  void verify();
  /// A simple LLVM IR verifier that checks that:
  /// (i)  definitions dominate uses, and
  /// (ii) PHIs are grouped at top.
  void verifyIR() const;
  // void verifyIR(const SBValBundle &Instrs) const;
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const SBBasicBlock &SBBB) {
    SBBB.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
  void dumpVerbose(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dumpVerbose() const final;
  /// Dump a range of instructions near \p SBV.
  LLVM_DUMP_METHOD void dumpInstrs(SBValue *SBV, int Num) const;
#endif
};

/// A client-attorney class for SBBasicBlock that allows access to selected
/// private members.
class SBBasicBlockAttorney {
  static BasicBlock *getBB(SBBasicBlock *SBBB) {
    return cast<BasicBlock>(SBBB->Val);
  }
  friend class SBCostModel;
  friend class SBShuffleInstruction;
  friend class SBPackInstruction;
  friend class SBUnpackInstruction;
  friend class SBCastInstruction;
  friend class SBPHINode;
  friend class SBSelectInstruction;
  friend class SBLoadInstruction;
  friend class SBStoreInstruction;
  friend class SBBinaryOperator;
  friend class SBUnaryOperator;
  friend class SBCmpInstruction;
};

class SBContext {
public:
  using RemoveCBTy = std::function<void(SBInstruction *)>;
  using InsertCBTy = std::function<void(SBInstruction *)>;
  using MoveCBTy = std::function<void(SBInstruction *, SBBasicBlock &,
                                      const SBBBIterator &)>;

private:
  LLVMContext &LLVMCtxt;
  AliasAnalysis &AA;
  SandboxIRTracker ChangeTracker;
  IRBuilder<ConstantFolder> LLVMIRBuilder;
  /// Helper deleter that allows us to use std::unique_ptr<Schduler> here,
  /// where it is not defined (due to cyclic header file dependencies).
  struct SchedulerDeleter {
    void operator()(Scheduler *) const;
  };
  DenseMap<SBBasicBlock *,
           std::unique_ptr<Scheduler, SchedulerDeleter>>
      SchedForSBBB;

  /// Vector of callbacks called when an IR Instruction is about to get erased.
  SmallVector<std::unique_ptr<RemoveCBTy>> RemoveInstrCallbacks;
  DenseMap<SBBasicBlock *, SmallVector<std::unique_ptr<RemoveCBTy>>>
      RemoveInstrCallbacksBB;
  SmallVector<std::unique_ptr<InsertCBTy>> InsertInstrCallbacks;
  DenseMap<SBBasicBlock *, SmallVector<std::unique_ptr<InsertCBTy>>>
      InsertInstrCallbacksBB;
  SmallVector<std::unique_ptr<MoveCBTy>> MoveInstrCallbacks;
  DenseMap<SBBasicBlock *, SmallVector<std::unique_ptr<MoveCBTy>>>
      MoveInstrCallbacksBB;

  /// Maps LLVM Value to the corresponding SBValue. Owns all SandboxIR objects.
  DenseMap<Value *, std::unique_ptr<SBValue>> LLVMValueToSBValueMap;
  /// In SandboxIR some instructions correspond to multiple IR Instructions,
  /// like Packs. For such cases we map the IR instructions to the key used in
  /// LLVMValueToSBValueMap.
  DenseMap<Value *, Value *> MultiInstrMap;

  /// This holds a pointer to the scheduler that is currently active.
  /// It helps avoid passing the scheduler as argument to all SandboxIR modifying
  /// functions. It gets set by the Scheduler's constructor.
  Scheduler *Sched = nullptr;

  friend SBBasicBlock::~SBBasicBlock(); // For removing the scheduler.
  /// This is true during quickFlush(). It helps with some assertions that would
  /// otherwise trigger.
  bool InQuickFlush = false;

  void setScheduler(Scheduler &NewSched) { Sched = &NewSched; }
  void clearScheduler() { Sched = nullptr; }
  friend class SBContextAttorney; // for setScheduler(), clearScheduler()
  /// Removes \p V from the maps and returns the unique_ptr.
  std::unique_ptr<SBValue> detachValue(Value *V);

  friend void SBInstruction::eraseFromParent();
  friend void SBInstruction::removeFromParent();
  friend void SBInstruction::moveBefore(SBBasicBlock &,
                                          const SBBBIterator &);

  void runRemoveInstrCallbacks(SBInstruction *I);
  void runInsertInstrCallbacks(SBInstruction *I);
  void runMoveInstrCallbacks(SBInstruction *I, SBBasicBlock &SBBB,
                             const SBBBIterator &WhereIt);

  /// Helper for avoiding recursion loop when creating SBConstants.
  SmallDenseSet<Constant *, 8> VisitedConstants;
  SBValue *getOrCreateSBValueInternal(Value *V, int Depth,
                                          User *U = nullptr);

public:
  SBContext(LLVMContext &LLVMCtxt, AliasAnalysis &AA);
  SandboxIRTracker &getTracker() { return ChangeTracker; }
  auto &getLLVMIRBuilder() { return LLVMIRBuilder; }
  Scheduler *getScheduler(SBBasicBlock *SBBB) const;
  const DependencyGraph &getDAG(SBBasicBlock *SBBB) const;
  size_t getNumValues() const {
    return LLVMValueToSBValueMap.size() + MultiInstrMap.size();
  }
  /// Remove \p SBV from all SandboxIR maps and stop owning it. This effectively
  /// detaches \p SBV from the underlying IR.
  std::unique_ptr<SBValue> detach(SBValue *SBV);
  SBValue *registerSBValue(std::unique_ptr<SBValue> &&SBVPtr);

  SBValue *getSBValue(Value *V) const;

  SBConstant *getSBConstant(Constant *C) const;
  SBConstant *getOrCreateSBConstant(Constant *C);

  SBValue *getOrCreateSBValue(Value *V);

  /// Helper function called when we create SBInstructions that create new
  /// constant operands. It goes through V's operands and creates SBConstants.
  void createMissingConstantOperands(Value *V);

  // Pack
  /// Please note that this may return a constant if folded.
  SBValue *createSBPackInstruction(const SBValBundle &PackOps,
                                       SBBasicBlock *SBBB,
                                       SBInstruction *BeforeI = nullptr);
  SBPackInstruction *createSBPackInstruction(const ValueBundle &PackInstrs);

  // Arguments
  SBArgument *getSBArgument(Argument *Arg) const;
  SBArgument *createSBArgument(Argument *Arg);
  SBArgument *getOrCreateSBArgument(Argument *Arg);

  // OpaqueInstr
  SBOpaqueInstruction *getSBOpaqueInstruction(Instruction *I) const;
  SBOpaqueInstruction *createSBOpaqueInstruction(Instruction *I);
  SBOpaqueInstruction *getOrCreateSBOpaqueInstruction(Instruction *I);

  // Unpack
  /// Please note that this may return a constant if folded.
  SBValue *createSBUnpackInstruction(SBValue *Op, unsigned Lane,
                                         SBBasicBlock *SBBB,
                                         unsigned LanesToUnpack = 1u);
  SBUnpackInstruction *
  getSBUnpackInstruction(ExtractElementInst *ExtractI) const;
  SBUnpackInstruction *
  createSBUnpackInstruction(ExtractElementInst *ExtractI);
  SBUnpackInstruction *
  getOrCreateSBUnpackInstruction(ExtractElementInst *ExtractI);

  SBUnpackInstruction *
  getSBUnpackInstruction(ShuffleVectorInst *ShuffleI) const;
  SBUnpackInstruction *
  createSBUnpackInstruction(ShuffleVectorInst *ShuffleI);
  SBUnpackInstruction *
  getOrCreateSBUnpackInstruction(ShuffleVectorInst *ShuffleI);

  // Store
  SBStoreInstruction *getSBStoreInstruction(StoreInst *SI) const;
  SBStoreInstruction *createSBStoreInstruction(StoreInst *SI);
  SBStoreInstruction *getOrCreateSBStoreInstruction(StoreInst *SI);

  // Load
  SBLoadInstruction *getSBLoadInstruction(LoadInst *LI) const;
  SBLoadInstruction *createSBLoadInstruction(LoadInst *LI);
  SBLoadInstruction *getOrCreateSBLoadInstruction(LoadInst *LI);

  // Cast
  SBCastInstruction *getSBCastInstruction(CastInst *CI) const;
  SBCastInstruction *createSBCastInstruction(CastInst *CI);
  SBCastInstruction *getOrCreateSBCastInstruction(CastInst *CI);

  // PHI
  SBPHINode *getSBPHINode(PHINode *PHI) const;
  SBPHINode *createSBPHINode(PHINode *PHI);
  SBPHINode *getOrCreateSBPHINode(PHINode *PHI);

  // Select
  SBSelectInstruction *getSBSelectInstruction(SelectInst *SI) const;
  SBSelectInstruction *createSBSelectInstruction(SelectInst *SI);
  SBSelectInstruction *getOrCreateSBSelectInstruction(SelectInst *SI);

  // BinaryOperator
  SBBinaryOperator *getSBBinaryOperator(BinaryOperator *BO) const;
  SBBinaryOperator *createSBBinaryOperator(BinaryOperator *BO);
  SBBinaryOperator *getOrCreateSBBinaryOperator(BinaryOperator *BO);

  // UnaryOperator
  SBUnaryOperator *getSBUnaryOperator(UnaryOperator *UO) const;
  SBUnaryOperator *createSBUnaryOperator(UnaryOperator *UO);
  SBUnaryOperator *getOrCreateSBUnaryOperator(UnaryOperator *UO);

  // Cmp
  SBCmpInstruction *getSBCmpInstruction(CmpInst *CI) const;
  SBCmpInstruction *createSBCmpInstruction(CmpInst *CI);
  SBCmpInstruction *getOrCreateSBCmpInstruction(CmpInst *CI);

  // Shuffle
  SBShuffleInstruction *createSBShuffleInstruction(ShuffleMask &Mask,
                                                       SBValue *Op,
                                                       SBBasicBlock *SBBB);
  SBShuffleInstruction *
  getSBShuffleInstruction(ShuffleVectorInst *ShuffleI) const;
  SBShuffleInstruction *
  createSBShuffleInstruction(ShuffleVectorInst *ShuffleI);
  SBShuffleInstruction *
  getOrCreateSBShuffleInstruction(ShuffleVectorInst *ShuffleI);

  // Block
  SBBasicBlock *getSBBasicBlock(BasicBlock *BB) const;
  SBBasicBlock *createSBBasicBlock(BasicBlock *BB);

  // Function
  SBFunction *getSBFunction(Function *F) const;
  SBFunction *createSBFunction(Function *F, bool CreateBBs = true);

  /// Register a callback that gets called when a SandboxIR instruction is about
  /// to be removed from its parent. Please not that this will also be called
  /// when reverting the creation of an instruction.
  /// \Returns the function pointer, which can be used later to remove it from
  /// the callback list.
  RemoveCBTy *registerRemoveInstrCallback(RemoveCBTy CB);
  void unregisterRemoveInstrCallback(RemoveCBTy *CB);

  InsertCBTy *registerInsertInstrCallback(InsertCBTy CB);
  void unregisterInsertInstrCallback(InsertCBTy *CB);

  MoveCBTy *registerMoveInstrCallback(MoveCBTy CB);
  void unregisterMoveInstrCallback(MoveCBTy *CB);

  /// Register a callback that gets called if the instruction is removed from a
  /// specific BB.
  RemoveCBTy *registerRemoveInstrCallbackBB(SBBasicBlock &BB, RemoveCBTy CB);
  void unregisterRemoveInstrCallbackBB(SBBasicBlock &BB, RemoveCBTy *CB);

  InsertCBTy *registerInsertInstrCallbackBB(SBBasicBlock &BB, InsertCBTy CB);
  void unregisterInsertInstrCallbackBB(SBBasicBlock &BB, InsertCBTy *CB);

  MoveCBTy *registerMoveInstrCallbackBB(SBBasicBlock &BB, MoveCBTy CB);
  void unregisterMoveInstrCallbackBB(SBBasicBlock &BB, MoveCBTy *CB);

  /// Clears state for the whole context quickly. This is to speed up
  /// destruction of the whole SandboxIR.
  void quickFlush();

#ifndef NDEBUG
  /// Used in tests
  void disableCallbacks() { CallbacksDisabled = true; }
#endif

protected:
#ifndef NDEBUG
  bool CallbacksDisabled = false;
#endif
  friend class SBContextAttorney;
};

/// A client-attorney class for SBContext.
class SBContextAttorney {
  friend class SBRegion;
  friend class SBRegionBuilderFromMD;

public:
  static LLVMContext &getLLVMContext(SBContext &Ctxt) {
    return Ctxt.LLVMCtxt;
  }
};

class SBFunction : public SBValue {
  Function *getFunction() const { return cast<Function>(Val); }

public:
  SBFunction(Function *F, SBContext &Ctxt)
      : SBValue(ClassID::Function, F, Ctxt) {}
  /// For isa/dyn_cast.
  static bool classof(const SBValue *From) {
    return From->getSubclassID() == ClassID::Function;
  }

  /// Iterates over SBBasicBlocks
  class iterator {
    Function::iterator It;
#ifndef NDEBUG
    Function *F;
#endif
    SBContext *Ctxt;

  public:
    using difference_type = std::ptrdiff_t;
    using value_type = SBBasicBlock;
    using pointer = SBBasicBlock *;
    using reference = value_type &;
    using iterator_category = std::bidirectional_iterator_tag;

#ifndef NDEBUG
    iterator() : F(nullptr), Ctxt(nullptr) {}
    iterator(Function::iterator It, Function *F, SBContext &Ctxt)
        : It(It), F(F), Ctxt(&Ctxt) {}
#else
    iterator() : Ctxt(nullptr) {}
    iterator(Function::iterator It, SBContext &Ctxt) : It(It), Ctxt(&Ctxt) {}
#endif

    bool operator==(const iterator &Other) const {
      assert(F == Other.F && "Comparing iterators of different functions!");
      return It == Other.It;
    }
    bool operator!=(const iterator &Other) const { return !(*this == Other); }
    iterator &operator++() {
      assert(It != F->end() && "Already at end!");
      ++It;
      return *this;
    }
    iterator operator++(int) {
      auto Copy = *this;
      ++*this;
      return Copy;
    }
    iterator &operator--() {
      assert(It != F->begin() && "Already at begin!");
      --It;
      return *this;
    }
    reference operator*() {
      assert(It != F->end() && "Dereferencing end()!");
      return *cast<SBBasicBlock>(Ctxt->getSBValue(&*It));
    }
    const SBBasicBlock &operator*() const {
      assert(It != F->end() && "Dereferencing end()!");
      return *cast<SBBasicBlock>(Ctxt->getSBValue(&*It));
    }
  };

  SBArgument *getArg(unsigned Idx) const {
    Argument *Arg = getFunction()->getArg(Idx);
    return cast<SBArgument>(Ctxt.getSBValue(Arg));
  }

  size_t arg_size() const { return getFunction()->arg_size(); }
  bool arg_empty() const { return getFunction()->arg_empty(); }

  SBBasicBlock &getEntryBlock() const {
    BasicBlock &EntryBB = getFunction()->getEntryBlock();
    return *cast<SBBasicBlock>(Ctxt.getSBValue(&EntryBB));
  }

  iterator begin() const {
    Function *F = getFunction();
#ifndef NDEBUG
    return iterator(F->begin(), F, Ctxt);
#else
    return iterator(F->begin(), Ctxt);
#endif
  }
  iterator end() const {
    Function *F = getFunction();
#ifndef NDEBUG
    return iterator(F->end(), F, Ctxt);
#else
    return iterator(F->end(), Ctxt);
#endif
  }
  /// Detaches the function, its blocks and its instructions from LLVM IR.
  void detachFromLLVMIR();

  hash_code hash() const final {
    auto Hash = hash_combine(SBValue::hashCommon(),
                             hash_combine_range(begin(), end()));
    for (auto ArgIdx : seq<unsigned>(0, arg_size()))
      Hash = hash_combine(Hash, getArg(ArgIdx));
    return Hash;
  }
  friend hash_code hash_value(const SBFunction &SBF) {
    return SBF.hash();
  }

#ifndef NDEBUG
  void dumpNameAndArgs(raw_ostream &OS) const;
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
  void dumpVerbose(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dumpVerbose() const final;
#endif
};

using sb_succ_iterator = SuccIterator<SBInstruction, SBBasicBlock>;
using const_sb_succ_iterator =
    SuccIterator<const SBInstruction, const SBBasicBlock>;
using sb_succ_range = iterator_range<sb_succ_iterator>;
using const_sb_succ_range = iterator_range<const_sb_succ_iterator>;

inline sb_succ_iterator succ_begin(SBInstruction *I) {
  return sb_succ_iterator(I);
}
inline const_sb_succ_iterator succ_begin(const SBInstruction *I) {
  return const_sb_succ_iterator(I);
}
inline sb_succ_iterator succ_end(SBInstruction *I) {
  return sb_succ_iterator(I, true);
}
inline const_sb_succ_iterator succ_end(const SBInstruction *I) {
  return const_sb_succ_iterator(I, true);
}
inline bool succ_empty(const SBInstruction *I) {
  return succ_begin(I) == succ_end(I);
}
inline unsigned succ_size(const SBInstruction *I) {
  return std::distance(succ_begin(I), succ_end(I));
}
inline sb_succ_range successors(SBInstruction *I) {
  return sb_succ_range(succ_begin(I), succ_end(I));
}
inline const_sb_succ_range successors(const SBInstruction *I) {
  return const_sb_succ_range(succ_begin(I), succ_end(I));
}

inline sb_succ_iterator succ_begin(SBBasicBlock *BB) {
  return sb_succ_iterator(BB->getTerminator());
}
inline const_sb_succ_iterator succ_begin(const SBBasicBlock *BB) {
  return const_sb_succ_iterator(BB->getTerminator());
}
inline sb_succ_iterator succ_end(SBBasicBlock *BB) {
  return sb_succ_iterator(BB->getTerminator(), true);
}
inline const_sb_succ_iterator succ_end(const SBBasicBlock *BB) {
  return const_sb_succ_iterator(BB->getTerminator(), true);
}
inline bool succ_empty(const SBBasicBlock *BB) {
  return succ_begin(BB) == succ_end(BB);
}
inline unsigned succ_size(const SBBasicBlock *BB) {
  return std::distance(succ_begin(BB), succ_end(BB));
}
inline sb_succ_range successors(SBBasicBlock *BB) {
  return sb_succ_range(succ_begin(BB), succ_end(BB));
}
inline const_sb_succ_range successors(const SBBasicBlock *BB) {
  return const_sb_succ_range(succ_begin(BB), succ_end(BB));
}

// GraphTraits for SBBasicBlock.
template <> struct GraphTraits<SBBasicBlock *> {
  using NodeRef = SBBasicBlock *;
  using ChildIteratorType = sb_succ_iterator;
  static NodeRef getEntryNode(SBBasicBlock *BB) { return BB; }
  static ChildIteratorType child_begin(NodeRef N) { return succ_begin(N); }
  static ChildIteratorType child_end(NodeRef N) { return succ_end(N); }
};
template <> struct GraphTraits<const SBBasicBlock *> {
  using NodeRef = const SBBasicBlock *;
  using ChildIteratorType = const_sb_succ_iterator;
  static NodeRef getEntryNode(const SBBasicBlock *BB) { return BB; }
  static ChildIteratorType child_begin(NodeRef N) { return succ_begin(N); }
  static ChildIteratorType child_end(NodeRef N) { return succ_end(N); }
};

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SANDBOXIR_H
