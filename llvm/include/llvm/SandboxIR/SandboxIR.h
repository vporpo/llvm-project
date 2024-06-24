//===- SandboxIR.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sandbox IR is a lightweight overlay transactional IR on top of LLVM IR.
// Features:
// - You can save/rollback the state of the IR at any time.
// - Any changes made to Sandbox IR will automatically update the underlying
//   LLVM IR so both IRs are always in sync.
// - Feels like LLVM IR, similar API.
//
// SandboxIR forms a class hierarchy that resembles that of LLVM IR
// but is in the `sandboxir` namespace:
//
// namespace sandboxir {
//
//        +- Argument                   +- BinaryOperator
//        |                             |
// Value -+- BasicBlock                 +- BranchInst
//        |                             |
//        +- Function   +- Constant     +- CastInst
//        |             |               |
//        +- User ------+- Instruction -+- CallInst
//                                      |
//                                      +- CmpInst
//                                      |
//                                      +- ExtractElementInst
//                                      |
//                                      +- GetElementPtrInst
//                                      |
//                                      +- InsertElementInst
//                                      |
//                                      +- LoadInst
//                                      |
//                                      +- OpaqueInst
//                                      |
//                                      +- PHINode
//                                      |
//                                      +- RetInst
//                                      |
//                                      +- SelectInst
//                                      |
//                                      +- ShuffleVectorInst
//                                      |
//                                      +- StoreInst
//                                      |
//                                      +- UnaryOperator
//
// Use
//
// } // namespace sandboxir
//
//
// SandboxIR Internals
// ===================
// Design Principles
// -----------------
// - SandoxIR is maintains no state in its own data-structures. Its state should
//   always reflect that of the underlying LLVM IR.
// - SandboxIR is always in sync with LLVM IR and requires no lowering.
// - Any change in the state of SandboxIR is tracked and can be reverted.
// - No API function of SandboxIR should give access to LLVM IR objects, except
//   Type which is shared across SandboxIR and LLVM IR. Otherwise the user could
//   easily break transactions.
//
// Implementation Details
// ----------------------
// Internally SandboxIR maintains a map from
// llvm::Value* to sandboxir::Value* namely:
// `sandboxir::Context::LLVMValueToValueMap`. The SandboxIR Value objects are
// thin wrappers of the llvm::Value objects they point to. This means that they
// maintain almost no state, and they rely on the state of the LLVM objects.
//
// Example 1
// ---------
// sandboxir::Instruction::getOperand(N) works by:
//  (i)  Getting llvm::Value *Op = LLVMInst->getOperand(N) of the llvm instr
//       pointed to by this sandboxir instr.
//  (ii) Returning the corresponding sandboxir::Value by looking up the llvm
//      `Op` in `sandboxir::Context::LLVMValueToValueMap`.
// Example 2
// ---------
// sandboxir::Instruction::setOperand(N, NewV) works by:
//  (i)  Getting the llvm value `NewLLVMV` pointed to by `NewV`.
//  (ii) Setting the operand of the LLVM instr `LLVMInst` pointed to by this
//       sandboxir instr: LLVMInst->setOperand(NewLLVMV).
//
#ifndef LLVM_TRANSFORMS_SANDBOXIR_SANDBOXIR_H
#define LLVM_TRANSFORMS_SANDBOXIR_SANDBOXIR_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/SandboxIR/DmpVector.h"
#include "llvm/SandboxIR/SandboxIRTracker.h"
#include <iterator>

using namespace llvm::PatternMatch;

namespace llvm {
namespace sandboxir {

class BasicBlock;
class Value;
class User;
class PackInst;
class Context;
class Function;
class DependencyGraph;
class UseAttorney;
class OperandUseIterator;
class UserUseIterator;
class ValueAttorney;
class Analysis;
class PassManager;
class MemSeedContainer;
class RegionBuilderFromMD;
class Region;
class VecUtilsPrivileged;
class Scheduler;
class BBIterator;
class UserAttorney;
class CostModel;
class InstructionAttorney;
class BasicBlockAttorney;
class ContextAttorney;

// Forward declare all classes in the .def file so we can friend them later.
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS) class CLASS;
#include "llvm/SandboxIR/SandboxIRValues.def"

/// Represents a Def-use/Use-def edge in SandboxIR.
/// NOTE: Unlike llvm::Use, this is not an integral part of the use-def chains.
/// It is also not uniqued and is currently passed by value, so you can have to
/// SBUse objects for the same use-def edge.
/// WARNING: Use::operator(Use &) doesn't work like in LLVM IR, it won't set it.
///          Instead it just copies the contents of the Use object.
class Use {
  llvm::Use *LLVMUse;
  friend class UseAttorney; // For LLVMUse
  User *Usr;
  Context *Ctx;

  /// Don't allow the user to create a sandboxir::Use directly.
  Use(llvm::Use *LLVMUse, class User *Usr, Context &Ctx)
      : LLVMUse(LLVMUse), Usr(Usr), Ctx(&Ctx) {}
  Use() : LLVMUse(nullptr), Ctx(nullptr) {}

  friend class User;               // For constructor
  friend class Value;              // For constructor
  friend class OperandUseIterator; // For constructor
  friend class UserUseIterator;    // For constructor
  // Several instructions need access to the Use() constructor for
  // their implementation of getOperandUseInternal().
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS; // For constructor
#include "llvm/SandboxIR/SandboxIRValues.def"

public:
  operator Value *() const { return get(); }
  Value *get() const;
  class User *getUser() const { return Usr; }
  unsigned getOperandNo() const;
  Context *getContext() const { return Ctx; }
  void swap(Use &Other);
  bool operator==(const Use &Other) const {
    assert(Ctx == Other.Ctx && "Contexts differ!");
    return LLVMUse == Other.LLVMUse && Usr == Other.Usr;
  }
  bool operator!=(const Use &Other) const { return !(*this == Other); }
  void set(Value *Val);
  inline Value *operator=(Value *Val) {
    set(Val);
    return Val;
  }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  void dump() const;
#endif // NDEBUG
};

/// A client-attorney class for Use.
class UseAttorney {
  static llvm::Use *getLLVMUse(Use &Use) { return Use.LLVMUse; }
  friend class BasicBlock; // For getLLVMUse()
};

/// Returns the operand edge when dereferenced.
class OperandUseIterator {
  sandboxir::Use Use;
  /// Don't let the user create a non-empty SBOperandUseIterator.
  OperandUseIterator(const class Use &Use) : Use(Use) {}
  friend class User; // For constructor
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS; // For constructor
#include "llvm/SandboxIR/SandboxIRValues.def"

public:
  using difference_type = std::ptrdiff_t;
  using value_type = class Use;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::input_iterator_tag;

  OperandUseIterator() {}
  value_type operator*() const;
  OperandUseIterator &operator++();
  bool operator==(const OperandUseIterator &Other) const {
    return Use == Other.Use;
  }
  bool operator!=(const OperandUseIterator &Other) const {
    return !(*this == Other);
  }
};

/// Returns user edge when dereferenced.
class UserUseIterator {
  sandboxir::Use Use;
  /// Don't let the user create a non-empty UserUseIterator.
  UserUseIterator(const class Use &Use) : Use(Use) {}
  friend class Value; // For constructor

#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS; // For constructor
#include "llvm/SandboxIR/SandboxIRValues.def"

public:
  using difference_type = std::ptrdiff_t;
  using value_type = class Use;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::input_iterator_tag;

  UserUseIterator() {}
  value_type operator*() const;
  UserUseIterator &operator++();
  bool operator==(const UserUseIterator &Other) const {
    return Use == Other.Use;
  }
  bool operator!=(const UserUseIterator &Other) const {
    return !(*this == Other);
  }
};

/// Simple adaptor class for UserUseIterator and SBOperandUseIterator that
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
    static_assert(std::is_same<ItTy, UserUseIterator>::value ||
                      std::is_same<ItTy, OperandUseIterator>::value,
                  "Unsupported ItTy!");
    if constexpr (std::is_same<ItTy, UserUseIterator>::value) {
      return (*It).getUser();
    } else if constexpr (std::is_same<ItTy, OperandUseIterator>::value) {
      return (*It).get();
    }
  }
  bool operator==(const RetTyAdaptor &Other) const { return It == Other.It; }
  bool operator!=(const RetTyAdaptor &Other) const { return !(*this == Other); }
};

/// A SandboxIR Value has users. This is an abstract class.
class Value {
public:
  enum class ClassID : unsigned {
#define DEF_VALUE(ID, CLASS) ID,
#define DEF_USER(ID, CLASS) ID,
#define DEF_INSTR(ID, OPC, CLASS) ID,
#include "llvm/SandboxIR/SandboxIRValues.def"
  };

protected:
  static const char *getSubclassIDStr(ClassID ID) {
    switch (ID) {
#define DEF_VALUE(ID, CLASS)                                                   \
  case ClassID::ID:                                                            \
    return #ID;
#define DEF_USER(ID, CLASS)                                                    \
  case ClassID::ID:                                                            \
    return #ID;
#define DEF_INSTR(ID, OPC, CLASS)                                              \
  case ClassID::ID:                                                            \
    return #ID;
#include "llvm/SandboxIR/SandboxIRValues.def"
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
  llvm::Value *Val = nullptr;
  friend class ValueAttorney; // For accessing llvm::Value *Val

  /// All values point to the context.
  Context &Ctx;
  // This is used by eraseFromParent().
  void clearValue() { Val = nullptr; }
  template <typename ItTy, typename SBTy> friend class LLVMOpUserItToSBTy;

public:
  Value(ClassID SubclassID, llvm::Value *Val, Context &Ctx);
  virtual ~Value() = default;
  ClassID getSubclassID() const { return SubclassID; }

  using use_iterator = UserUseIterator;
  using const_use_iterator = UserUseIterator;

  use_iterator use_begin();
  const_use_iterator use_begin() const {
    return const_cast<Value *>(this)->use_begin();
  }
  use_iterator use_end() { return use_iterator(Use(nullptr, nullptr, Ctx)); }
  const_use_iterator use_end() const {
    return const_cast<Value *>(this)->use_end();
  }

  iterator_range<use_iterator> uses() {
    return make_range<use_iterator>(use_begin(), use_end());
  }
  iterator_range<const_use_iterator> uses() const {
    return make_range<const_use_iterator>(use_begin(), use_end());
  }

  using user_iterator = RetTyAdaptor<User, sandboxir::UserUseIterator>;
  using const_user_iterator = user_iterator;

  user_iterator user_begin();
  user_iterator user_end() { return user_iterator(Use(nullptr, nullptr, Ctx)); }
  const_user_iterator user_begin() const {
    return const_cast<Value *>(this)->user_begin();
  }
  const_user_iterator user_end() const {
    return const_cast<Value *>(this)->user_end();
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
        return false;
    }
    return Cnt == Num;
  }

  Value *getSingleUser() const;

  Type *getType() const { return Val->getType(); }

  Context &getContext() const;
  SandboxIRTracker &getTracker();
  virtual hash_code hashCommon() const {
    return hash_combine(SubclassID, &Ctx, Val);
  }
  /// WARNING: DstU can be nullptr if it is in a BB that is not in SandboxIR!
  void replaceUsesWithIf(Value *OtherV,
                         llvm::function_ref<bool(Use)> ShouldReplace);
  void replaceAllUsesWith(Value *Other);
  virtual hash_code hash() const = 0;
  friend hash_code hash_value(const Value &SBV) { return SBV.hash(); }
#ifndef NDEBUG
  /// Should crash if there is something wrong with the instruction.
  virtual void verify() const = 0;
  /// Returns the name in the form 'T<number>.' like 'T1.'
  std::string getName() const;
  virtual void dumpCommonHeader(raw_ostream &OS) const;
  void dumpCommonFooter(raw_ostream &OS) const;
  void dumpCommonPrefix(raw_ostream &OS) const;
  void dumpCommonSuffix(raw_ostream &OS) const;
  void printAsOperandCommon(raw_ostream &OS) const;
  friend raw_ostream &operator<<(raw_ostream &OS, const Value &SBV) {
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
  static llvm::Value *getValue(const Value *SBV) { return SBV->Val; }

#define DEF_VALUE(ID, CLASS) friend class CLASS;
#define DEF_USER(ID, CLASS) friend class CLASS;
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS;
#include "llvm/SandboxIR/SandboxIRValues.def"

  friend class Instruction;
  friend class DependencyGraph;
  template <typename T> friend class llvm::DmpVector;
  friend class Analysis;
  friend class PassManager;
  friend class Context;
  friend class User;
  friend class Use;
  friend class MemSeedContainer;
  friend class SandboxIRTracker;
  friend class RegionBuilderFromMD;
  friend class Region;
  friend class VecUtilsPrivileged;

  friend void Value::replaceUsesWithIf(Value *, llvm::function_ref<bool(Use)>);
  friend class Scheduler;
  friend class OperandUseIterator;
  friend class BBIterator;
  friend ReplaceAllUsesWith::ReplaceAllUsesWith(Value *, SandboxIRTracker &);
  friend ReplaceUsesOfWith::ReplaceUsesOfWith(User *, Value *, Value *,
                                              SandboxIRTracker &);
  friend class DeleteOnAccept;
  friend class CreateAndInsertInstr;
  friend class EraseFromParent;
};

/// A function argument.
class Argument : public Value {
  Argument(llvm::Argument *Arg, Context &Ctx);
  friend class Context; // for createArgument()

public:
  static bool classof(const Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const Argument &TArg) { return TArg.hash(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::Argument>(Val) && "Expected Argument!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const Argument &TArg) {
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
class User : public Value {
protected:
  User(ClassID ID, llvm::Value *V, Context &Ctx);
  friend class Instruction; // For constructors.

  /// \Returns the SBUse edge that corresponds to \p OpIdx.
  /// Note: This is the default implementation that works for instructions that
  /// match the underlying LLVM instruction. All others should use a different
  /// implementation.
  Use getOperandUseDefault(unsigned OpIdx, bool Verify) const;
  virtual Use getOperandUseInternal(unsigned OpIdx, bool Verify) const = 0;
  friend class OperandUseIterator; // for getOperandUseInternal()

  /// \Returns true if \p Use should be considered as an edge to its SandboxIR
  /// operand. Most instructions should return true.
  /// Currently it is only Uses from Vectors into Packs that return false.
  virtual bool isRealOperandUse(llvm::Use &Use) const = 0;
  friend class UserUseIterator; // for isRealOperandUse()

  /// The default implementation works only for single-LLVMIR-instruction
  /// SBUsers and only if they match exactly the LLVM instruction.
  unsigned getUseOperandNoDefault(const Use &Use) const {
    return Use.LLVMUse->getOperandNo();
  }

  void swapOperandsInternal(unsigned OpIdxA, unsigned OpIdxB) {
    assert(OpIdxA < getNumOperands() && "OpIdxA out of bounds!");
    assert(OpIdxB < getNumOperands() && "OpIdxB out of bounds!");
    auto UseA = getOperandUse(OpIdxA);
    auto UseB = getOperandUse(OpIdxB);
    UseA.swap(UseB);
  }

#ifndef NDEBUG
  void verifyUserOfLLVMUse(const llvm::Use &Use) const;
#endif

public:
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  using op_iterator = OperandUseIterator;
  using const_op_iterator = OperandUseIterator;
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
    auto Hash = Value::hashCommon();
    for (Value *Op : operands())
      Hash = hash_combine(Hash, Op);
    return Hash;
  }
  Value *getOperand(unsigned OpIdx) const { return getOperandUse(OpIdx).get(); }
  /// \Returns the operand edge for \p OpIdx. NOTE: This should also work for
  /// OpIdx == getNumOperands(), which is used for op_end().
  Use getOperandUse(unsigned OpIdx) const {
    return getOperandUseInternal(OpIdx, /*Verify=*/true);
  }
  /// \Returns the operand index of \p Use.
  virtual unsigned getUseOperandNo(const Use &Use) const = 0;
  Value *getSingleOperand() const;
  virtual void setOperand(unsigned OperandIdx, Value *Operand);
  virtual unsigned getNumOperands() const {
    return isa<llvm::User>(Val) ? cast<llvm::User>(Val)->getNumOperands() : 0;
  }
  /// Replaces any operands that match \p FromV with \p ToV. Returns whether any
  /// operands were replaced.
  /// WARNING: This will replace even uses that are not in SandboxIR!
  bool replaceUsesOfWith(Value *FromV, Value *ToV);

#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::User>(Val) && "Expected User!");
  }
  void dumpCommonHeader(raw_ostream &OS) const final;
#endif

protected:
  /// \Returns the operand index that corresponds to \p UseToMatch.
  virtual unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const = 0;
  friend class UserAttorney; // For testing
  friend void Value::replaceUsesWithIf(Value *, llvm::function_ref<bool(Use)>);
};

/// A simple client-attorney class that exposes some protected members of
/// User for use in tests.
class UserAttorney {
public:
  // For testing.
  static unsigned getOperandUseIdx(const User *SBU,
                                   const llvm::Use &UseToMatch) {
    return SBU->getOperandUseIdx(UseToMatch);
  }
};

class Constant : public User {
  /// Use Context::createConstant() instead.
  Constant(llvm::Constant *C, Context &Ctx);
  friend class Context; // For constructor.
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }

public:
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  Context &getParent() const { return getContext(); }
  hash_code hashCommon() const final { return User::hashCommon(); }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const Constant &SBC) { return SBC.hash(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::Constant>(Val) && "Expected Constant!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const Constant &SBC) {
    SBC.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class Instruction;

/// The SBBasicBlock::iterator.
class BBIterator {
public:
  using difference_type = std::ptrdiff_t;
  using value_type = Instruction;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::bidirectional_iterator_tag;

private:
  llvm::BasicBlock *BB;
  /// This should always point to the bottom IR instruction of a multi-IR
  /// Instruction.
  llvm::BasicBlock::iterator It;
  Context *Ctx;
  pointer getI(llvm::BasicBlock::iterator It) const;

public:
  BBIterator() : BB(nullptr), Ctx(nullptr) {}
  BBIterator(llvm::BasicBlock *BB, llvm::BasicBlock::iterator It, Context *Ctx)
      : BB(BB), It(It), Ctx(Ctx) {}
  reference operator*() const { return *getI(It); }
  BBIterator &operator++();
  BBIterator operator++(int) {
    auto Copy = *this;
    ++*this;
    return Copy;
  }
  BBIterator &operator--();
  BBIterator operator--(int) {
    auto Copy = *this;
    --*this;
    return Copy;
  }
  bool operator==(const BBIterator &Other) const {
    assert(Ctx == Other.Ctx && "SBBBIterators in different context!");
    return It == Other.It;
  }
  bool operator!=(const BBIterator &Other) const { return !(*this == Other); }
  /// \Returns true if the internal iterator is at the beginning of the IR BB.
  /// NOTE: This is meant to be used internally, during the construction of a
  /// SBBB, during which SBBB->begin() fails due to the missing mapping of
  /// BB->begin() to SandboxIR.
  bool atBegin() const;
  /// \Returns the SBInstruction that corresponds to this iterator, or null if
  /// the instruction is not found in the IR-to-SandboxIR tables.
  pointer get() const { return getI(It); }
};

/// A SBUser with operands and opcode.
class Instruction : public User {
public:
  enum class Opcode {
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define OP(OPC) OPC,
#define OPCODES(...) __VA_ARGS__
#define DEF_INSTR(ID, OPC, CLASS) OPC
#include "llvm/SandboxIR/SandboxIRValues.def"
  };

protected:
  /// Don't create objects of this class. Use a sub-class instead.
  Instruction(ClassID ID, Opcode Opc, llvm::Instruction *I, Context &Ctx);

#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS;
#include "llvm/SandboxIR/SandboxIRValues.def"

  /// Any extra actions that need to be performed upon detach.
  virtual void detachExtras() = 0;
  friend class Context; // For detachExtras()

  /// A SBInstruction may map to multiple IR Instruction. This returns its
  /// topmost IR instruction.
  llvm::Instruction *getTopmostLLVMInstruction() const;

  /// \Returns all IR instructions that make up this SBInstruction in reverse
  /// program order.
  virtual DmpVector<llvm::Instruction *> getLLVMInstrs() const = 0;
  friend class CostModel; // For getLLVMInstrs().
  /// \Returns all IR instructions with external operands. Note: This is useful
  /// for multi-IR instructions like Packs, that are composed of both
  /// internal-only and external-facing IR Instructions.
  virtual DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const = 0;
  friend void DeleteOnAccept::accept();
  friend ReplaceAllUsesWith::ReplaceAllUsesWith(Value *, SandboxIRTracker &);
  friend ReplaceUsesOfWith::ReplaceUsesOfWith(User *, Value *, Value *,
                                              SandboxIRTracker &);
  friend bool User::replaceUsesOfWith(Value *, Value *);
  friend class EraseFromParent;
  friend class DeleteOnAccept;

  Opcode Opc;
  /// Maps SBInstruction::Opcode to its corresponding IR opcode, if it exists.
  static llvm::Instruction::UnaryOps getIRUnaryOp(Opcode Opc);
  static llvm::Instruction::BinaryOps getIRBinaryOp(Opcode Opc);
  static llvm::Instruction::CastOps getIRCastOp(Opcode Opc);

  // Metadata is LLMV IR, so protect it. Access this via the
  // SBInstructionAttorney class.
  MDNode *getMetadata(unsigned KindID) const {
    return cast<llvm::Instruction>(Val)->getMetadata(KindID);
  }
  MDNode *getMetadata(StringRef Kind) const {
    return cast<llvm::Instruction>(Val)->getMetadata(Kind);
  }
  friend class InstructionAttorney;

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
  static bool classof(const Value *From);
  BBIterator getIterator() const;
  Instruction *getNextNode() const;
  Instruction *getPrevNode() const;
  /// \Returns the opcode of the Instruction contained.
  Opcode getOpcode() const { return Opc; }
  /// Detach this from its parent BasicBlock without deleting it.
  void removeFromParent();
  /// Detach this Value from its parent and delete it.
  void eraseFromParent();
  /// \Returns the parent graph or null if there is no parent graph, i.e., when
  /// it holds a Constant.
  BasicBlock *getParent() const;
  bool isFPMath() const { return isa<FPMathOperator>(Val); }
  FastMathFlags getFastMathFlags() const {
    return cast<llvm::Instruction>(Val)->getFastMathFlags();
  }
  bool canHaveWrapFlags() const {
    return isa<OverflowingBinaryOperator>(Val) || isa<TruncInst>(Val);
  }
  bool hasNoUnsignedWrap() const {
    if (!canHaveWrapFlags())
      return false;
    return cast<llvm::Instruction>(Val)->hasNoUnsignedWrap();
  }
  bool hasNoSignedWrap() const {
    if (!canHaveWrapFlags())
      return false;
    return cast<llvm::Instruction>(Val)->hasNoSignedWrap();
  }
  /// \Returns true if this is a landingpad, a catchpad or a cleanuppadd
  bool isPad() const {
    return isa<LandingPadInst>(Val) || isa<CatchPadInst>(Val) ||
           isa<CleanupPadInst>(Val);
  }
  bool isFenceLike() const {
    return cast<llvm::Instruction>(Val)->isFenceLike();
  }
  int64_t getInstrNumber() const;
  bool comesBefore(Instruction *Other) const {
    return getInstrNumber() < Other->getInstrNumber();
  }
  bool comesAfter(Instruction *Other) { return Other->comesBefore(this); }
  /// \Returns a (very) approximate absolute distance between this instruction
  /// and \p ToI. This is a constant-time operation.
  uint64_t getApproximateDistanceTo(Instruction *ToI) const;
  void moveBefore(BasicBlock &SBBB, const BBIterator &WhereIt);
  void moveBefore(Instruction *Before) {
    moveBefore(*Before->getParent(), Before->getIterator());
  }
  void moveAfter(Instruction *After) {
    moveBefore(*After->getParent(), std::next(After->getIterator()));
  }
  hash_code hashCommon() const override {
    return hash_combine(User::hashCommon(), getParent());
  }
  void insertBefore(Instruction *BeforeI);
  void insertAfter(Instruction *AfterI);
  void insertInto(BasicBlock *SBBB, const BBIterator &WhereIt);

  bool mayWriteToMemory() const {
    return cast<llvm::Instruction>(Val)->mayWriteToMemory();
  }
  bool mayReadFromMemory() const {
    return cast<llvm::Instruction>(Val)->mayReadFromMemory();
  }
  bool isTerminator() const {
    return cast<llvm::Instruction>(Val)->isTerminator();
  }

  bool isStackRelated() const {
    auto IsInAlloca = [](llvm::Instruction *I) {
      return isa<AllocaInst>(I) && cast<AllocaInst>(I)->isUsedWithInAlloca();
    };
    auto *I = cast<llvm::Instruction>(Val);
    return match(I, m_Intrinsic<Intrinsic::stackrestore>()) ||
           match(I, m_Intrinsic<Intrinsic::stacksave>()) || IsInAlloca(I);
  }
  /// We consider \p I as a Mem instruction if it accesses memory or if it is
  /// stack-related. This is used to determine whether this instruction needs
  /// dependency edges.
  bool isMemInst() const {
    auto IsMem = [](llvm::Instruction *I) {
      return I->mayReadOrWriteMemory() &&
             (!isa<IntrinsicInst>(I) ||
              (cast<IntrinsicInst>(I)->getIntrinsicID() !=
                   Intrinsic::sideeffect &&
               cast<IntrinsicInst>(I)->getIntrinsicID() !=
                   Intrinsic::pseudoprobe));
    };
    return IsMem(cast<llvm::Instruction>(Val)) || isStackRelated();
  }
  bool isDbgInfo() const {
    auto *I = cast<llvm::Instruction>(Val);
    return isa<DbgInfoIntrinsic>(I);
  }
  /// \Returns the number of successors that this terminator instruction has.
  unsigned getNumSuccessors() const LLVM_READONLY {
    return cast<llvm::Instruction>(Val)->getNumSuccessors();
  }
  BasicBlock *getSuccessor(unsigned Idx) const LLVM_READONLY;

#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, const Instruction &SBI) {
    SBI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

/// A client-attorney class for Instruction.
class InstructionAttorney {
public:
  friend class RegionBuilderFromMD;
  static MDNode *getMetadata(const Instruction *SBI, unsigned KindID) {
    return SBI->getMetadata(KindID);
  }
  static MDNode *getMetadata(const Instruction *SBI, StringRef Kind) {
    return SBI->getMetadata(Kind);
  }
};

class CmpInst : public Instruction {
  static Opcode getCmpOpcode(unsigned CmpOp) {
    switch (CmpOp) {
    case llvm::Instruction::FCmp:
      return Opcode::FCmp;
    case llvm::Instruction::ICmp:
      return Opcode::ICmp;
    }
    llvm_unreachable("Unhandled CmpOp!");
  }

  /// Use Context::createCmpInst(). Don't call the
  /// constructor directly.
  CmpInst(llvm::CmpInst *CI, Context &Ctx)
      : Instruction(ClassID::Cmp, getCmpOpcode(CI->getOpcode()), CI, Ctx) {
    assert((Opc == Opcode::FCmp || Opc == Opcode::ICmp) && "Bad Opcode!");
  }
  friend class Context; // for CmpInst()
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  static Value *create(llvm::CmpInst::Predicate Pred, Value *LHS, Value *RHS,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "", MDNode *FPMathTag = nullptr);
  static Value *create(llvm::CmpInst::Predicate Pred, Value *LHS, Value *RHS,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "", MDNode *FPMathTag = nullptr);
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const CmpInst &SBSI) { return SBSI.hash(); }
  auto getPredicate() const { return cast<llvm::CmpInst>(Val)->getPredicate(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::CmpInst>(Val) && "Expected CmpInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const CmpInst &SBSI) {
    SBSI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};
} // namespace sandboxir

/// Traits for DenseMap.
template <> struct DenseMapInfo<sandboxir::Instruction::Opcode> {
  static inline sandboxir::Instruction::Opcode getEmptyKey() {
    return (sandboxir::Instruction::Opcode)-1;
  }
  static inline sandboxir::Instruction::Opcode getTombstoneKey() {
    return (sandboxir::Instruction::Opcode)-2;
  }
  static unsigned getHashValue(const sandboxir::Instruction::Opcode &B) {
    return (unsigned)B;
  }
  static bool isEqual(const sandboxir::Instruction::Opcode &B1,
                      const sandboxir::Instruction::Opcode &B2) {
    return B1 == B2;
  }
};

namespace sandboxir {

class BranchInst : public Instruction {
  /// Use Context::createBranchInst(). Don't call the constructor directly.
  BranchInst(llvm::BranchInst *BI, Context &Ctx)
      : Instruction(ClassID::Br, Opcode::Br, BI, Ctx) {}
  friend Context; // for BranchInst()
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static BranchInst *create(BasicBlock *IfTrue, Instruction *InsertBefore,
                            Context &Ctx);
  static BranchInst *create(BasicBlock *IfTrue, BasicBlock *InsertAtEnd,
                            Context &Ctx);
  static BranchInst *create(BasicBlock *IfTrue, BasicBlock *IfFalse,
                            Value *Cond, Instruction *InsertBefore,
                            Context &Ctx);
  static BranchInst *create(BasicBlock *IfTrue, BasicBlock *IfFalse,
                            Value *Cond, BasicBlock *InsertAtEnd, Context &Ctx);
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const BranchInst &BI) { return BI.hash(); }
  bool isUnconditional() const {
    return cast<llvm::BranchInst>(Val)->isUnconditional();
  }
  bool isConditional() const {
    return cast<llvm::BranchInst>(Val)->isConditional();
  }
  Value *getCondition() const;
  void setCondition(Value *V) { setOperand(0, V); }
  unsigned getNumSuccessors() const { return 1 + isConditional(); }
  BasicBlock *getSuccessor(unsigned i) const;
  void setSuccessor(unsigned Idx, BasicBlock *NewSucc);
  void swapSuccessors() { swapOperandsInternal(1, 2); }

private:
  struct LLVMBBToSBBB {
    Context &Ctx;
    LLVMBBToSBBB(Context &Ctx) : Ctx(Ctx) {}
    BasicBlock *operator()(llvm::BasicBlock *BB) const;
  };

  struct ConstLLVMBBToSBBB {
    Context &Ctx;
    ConstLLVMBBToSBBB(Context &Ctx) : Ctx(Ctx) {}
    const BasicBlock *operator()(const llvm::BasicBlock *BB) const;
  };

public:
  using sb_succ_op_iterator =
      mapped_iterator<llvm::BranchInst::succ_op_iterator, LLVMBBToSBBB>;
  iterator_range<sb_succ_op_iterator> successors() {
    iterator_range<llvm::BranchInst::succ_op_iterator> LLVMRange =
        cast<llvm::BranchInst>(Val)->successors();
    LLVMBBToSBBB BBMap(Ctx);
    sb_succ_op_iterator MappedBegin = map_iterator(LLVMRange.begin(), BBMap);
    sb_succ_op_iterator MappedEnd = map_iterator(LLVMRange.end(), BBMap);
    return make_range(MappedBegin, MappedEnd);
  }

  using const_sb_succ_op_iterator =
      mapped_iterator<llvm::BranchInst::const_succ_op_iterator,
                      ConstLLVMBBToSBBB>;
  iterator_range<const_sb_succ_op_iterator> successors() const {
    iterator_range<llvm::BranchInst::const_succ_op_iterator> ConstLLVMRange =
        static_cast<const llvm::BranchInst *>(cast<llvm::BranchInst>(Val))
            ->successors();
    ConstLLVMBBToSBBB ConstBBMap(Ctx);
    const_sb_succ_op_iterator ConstMappedBegin =
        map_iterator(ConstLLVMRange.begin(), ConstBBMap);
    const_sb_succ_op_iterator ConstMappedEnd =
        map_iterator(ConstLLVMRange.end(), ConstBBMap);
    return make_range(ConstMappedBegin, ConstMappedEnd);
  }

#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::BranchInst>(Val) && "Expected BranchInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const BranchInst &BI) {
    BI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class StoreInst : public Instruction {
  /// Use Context::createStoreInst(). Don't call the
  /// constructor directly.
  StoreInst(llvm::StoreInst *SI, Context &Ctx)
      : Instruction(ClassID::Store, Opcode::Store, SI, Ctx) {}
  friend Context; // for StoreInst()
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static StoreInst *create(Value *V, Value *Ptr, MaybeAlign Align,
                           Instruction *InsertBefore, Context &Ctx);
  static StoreInst *create(Value *V, Value *Ptr, MaybeAlign Align,
                           BasicBlock *InsertAtEnd, Context &Ctx);
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const StoreInst &SBSI) { return SBSI.hash(); }
  Value *getValueOperand() const;
  Value *getPointerOperand() const;
  Align getAlign() const { return cast<llvm::StoreInst>(Val)->getAlign(); }
  bool isSimple() const { return cast<llvm::StoreInst>(Val)->isSimple(); }
  bool isUnordered() const { return cast<llvm::StoreInst>(Val)->isUnordered(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::StoreInst>(Val) && "Expected StoreInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const StoreInst &SBSI) {
    SBSI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class LoadInst : public Instruction {
  /// Use Context::createLoadInst(). Don't call the
  /// constructor directly.
  LoadInst(llvm::LoadInst *LI, Context &Ctx)
      : Instruction(ClassID::Load, Opcode::Load, LI, Ctx) {}
  friend Context; // for LoadInst()
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }

  unsigned getNumOfIRInstrs() const final { return 1u; }
  static LoadInst *create(Type *Ty, Value *Ptr, MaybeAlign Align,
                          Instruction *InsertBefore, Context &Ctx,
                          const Twine &Name = "");
  static LoadInst *create(Type *Ty, Value *Ptr, MaybeAlign Align,
                          BasicBlock *InsertAtEnd, Context &Ctx,
                          const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const LoadInst &SBLI) { return SBLI.hash(); }
  Value *getPointerOperand() const;
  Align getAlign() const { return cast<llvm::LoadInst>(Val)->getAlign(); }
  bool isUnordered() const { return cast<llvm::LoadInst>(Val)->isUnordered(); }
  bool isSimple() const { return cast<llvm::LoadInst>(Val)->isSimple(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::LoadInst>(Val) && "Expected LoadInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const LoadInst &SBLI) {
    SBLI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class CastInst : public Instruction {
  static Opcode getCastOpcode(llvm::Instruction::CastOps CastOp) {
    switch (CastOp) {
    case llvm::Instruction::ZExt:
      return Opcode::ZExt;
    case llvm::Instruction::SExt:
      return Opcode::SExt;
    case llvm::Instruction::FPToUI:
      return Opcode::FPToUI;
    case llvm::Instruction::FPToSI:
      return Opcode::FPToSI;
    case llvm::Instruction::FPExt:
      return Opcode::FPExt;
    case llvm::Instruction::PtrToInt:
      return Opcode::PtrToInt;
    case llvm::Instruction::IntToPtr:
      return Opcode::IntToPtr;
    case llvm::Instruction::SIToFP:
      return Opcode::SIToFP;
    case llvm::Instruction::UIToFP:
      return Opcode::UIToFP;
    case llvm::Instruction::Trunc:
      return Opcode::Trunc;
    case llvm::Instruction::FPTrunc:
      return Opcode::FPTrunc;
    case llvm::Instruction::BitCast:
      return Opcode::BitCast;
    case llvm::Instruction::AddrSpaceCast:
      return Opcode::AddrSpaceCast;
    case llvm::Instruction::CastOpsEnd:
      llvm_unreachable("Bad CastOp!");
    }
    llvm_unreachable("Unhandled CastOp!");
  }
  /// Use Context::createCastInst(). Don't call the
  /// constructor directly.
  CastInst(llvm::CastInst *CI, Context &Ctx)
      : Instruction(ClassID::Cast, getCastOpcode(CI->getOpcode()), CI, Ctx) {}
  friend Context; // for SBCastInstruction()
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static Value *create(Type *Ty, Opcode Op, Value *Operand,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Type *Ty, Opcode Op, Value *Operand,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const CastInst &SBCI) { return SBCI.hash(); }
  llvm::Instruction::CastOps getOpcode() const {
    return cast<llvm::CastInst>(Val)->getOpcode();
  }
  Type *getSrcTy() const { return cast<llvm::CastInst>(Val)->getSrcTy(); }
  Type *getDestTy() const { return cast<llvm::CastInst>(Val)->getDestTy(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::CastInst>(Val) && "Expected CastInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const CastInst &SBCI) {
    SBCI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class PHINode : public Instruction {
  /// Use Context::createPHINode(). Don't call the
  /// constructor directly.
  PHINode(llvm::PHINode *PHI, Context &Ctx)
      : Instruction(ClassID::PHI, Opcode::PHI, PHI, Ctx) {}
  friend Context; // for SBPHINode()
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static Value *create(Type *Ty, unsigned NumReservedValues,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Type *Ty, unsigned NumReservedValues,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const PHINode &SBCI) { return SBCI.hash(); }
  Type *getSrcTy() const { return cast<llvm::CastInst>(Val)->getSrcTy(); }
  Type *getDestTy() const { return cast<llvm::CastInst>(Val)->getDestTy(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::PHINode>(Val) && "Expected PHINode!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const PHINode &SBCI) {
    SBCI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SelectInst : public Instruction {
  /// Use Context::createSelectInst(). Don't call the
  /// constructor directly.
  SelectInst(llvm::SelectInst *CI, Context &Ctx)
      : Instruction(ClassID::Select, Opcode::Select, CI, Ctx) {}
  friend Context; // for SelectInst()
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static Value *create(Value *Cond, Value *True, Value *False,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Value *Cond, Value *True, Value *False,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  Value *getCondition() { return getOperand(0); }
  Value *getTrueValue() { return getOperand(1); }
  Value *getFalseValue() { return getOperand(2); }

  void setCondition(Value *New) { setOperand(0, New); }
  void setTrueValue(Value *New) { setOperand(1, New); }
  void setFalseValue(Value *New) { setOperand(2, New); }
  void swapValues() { cast<llvm::SelectInst>(Val)->swapValues(); }
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const SelectInst &SBSI) { return SBSI.hash(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::SelectInst>(Val) && "Expected SelectInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const SelectInst &SBSI) {
    SBSI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class BinaryOperator : public Instruction {
  static Opcode getBinOpOpcode(llvm::Instruction::BinaryOps BinOp) {
    switch (BinOp) {
    case llvm::Instruction::Add:
      return Opcode::Add;
    case llvm::Instruction::FAdd:
      return Opcode::FAdd;
    case llvm::Instruction::Sub:
      return Opcode::Sub;
    case llvm::Instruction::FSub:
      return Opcode::FSub;
    case llvm::Instruction::Mul:
      return Opcode::Mul;
    case llvm::Instruction::FMul:
      return Opcode::FMul;
    case llvm::Instruction::UDiv:
      return Opcode::UDiv;
    case llvm::Instruction::SDiv:
      return Opcode::SDiv;
    case llvm::Instruction::FDiv:
      return Opcode::FDiv;
    case llvm::Instruction::URem:
      return Opcode::URem;
    case llvm::Instruction::SRem:
      return Opcode::SRem;
    case llvm::Instruction::FRem:
      return Opcode::FRem;
    case llvm::Instruction::Shl:
      return Opcode::Shl;
    case llvm::Instruction::LShr:
      return Opcode::LShr;
    case llvm::Instruction::AShr:
      return Opcode::AShr;
    case llvm::Instruction::And:
      return Opcode::And;
    case llvm::Instruction::Or:
      return Opcode::Or;
    case llvm::Instruction::Xor:
      return Opcode::Xor;
    case llvm::Instruction::BinaryOpsEnd:
      llvm_unreachable("Bad BinOp!");
    }
    llvm_unreachable("Unhandled BinOp!");
  }
  /// Use Context::createBinaryOperator(). Don't call the
  /// constructor directly.
  BinaryOperator(llvm::BinaryOperator *BO, Context &Ctx)
      : Instruction(ClassID::BinOp, getBinOpOpcode(BO->getOpcode()), BO, Ctx) {}
  friend Context; // for SelectInst()
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static Value *create(Instruction::Opcode Op, Value *LHS, Value *RHS,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Instruction::Opcode Op, Value *LHS, Value *RHS,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *LHS,
                                      Value *RHS, Value *CopyFrom,
                                      Instruction *InsertBefore, Context &Ctx,
                                      const Twine &Name = "");
  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *LHS,
                                      Value *RHS, Value *CopyFrom,
                                      BasicBlock *InsertAtEnd, Context &Ctx,
                                      const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  void swapOperands() { swapOperandsInternal(0, 1); }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const BinaryOperator &SBBO) {
    return SBBO.hash();
  }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::BinaryOperator>(Val) && "Expected BinaryOperator!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const BinaryOperator &SBBO) {
    SBBO.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class UnaryOperator : public Instruction {
  static Opcode getUnaryOpcode(llvm::Instruction::UnaryOps UnOp) {
    switch (UnOp) {
    case llvm::Instruction::FNeg:
      return Opcode::FNeg;
    case llvm::Instruction::UnaryOpsEnd:
      llvm_unreachable("Bad UnOp!");
    }
    llvm_unreachable("Unhandled UnOp!");
  }
  /// Use Context::createUnaryOperator(). Don't call the
  /// constructor directly.
  UnaryOperator(llvm::UnaryOperator *UO, Context &Ctx)
      : Instruction(ClassID::UnOp, getUnaryOpcode(UO->getOpcode()), UO, Ctx) {}
  friend Context; // for SelectInst()
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *OpV,
                                      Value *CopyFrom,
                                      Instruction *InsertBefore, Context &Ctx,
                                      const Twine &Name = "");
  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *OpV,
                                      Value *CopyFrom, BasicBlock *InsertAtEnd,
                                      Context &Ctx, const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const UnaryOperator &SBUO) { return SBUO.hash(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::UnaryOperator>(Val) && "Expected UnaryOperator!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const UnaryOperator &SBUO) {
    SBUO.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class InsertElementInst : public Instruction {
  /// Use Context::createInsertElementInst(). Don't call
  /// the constructor directly.
  InsertElementInst(llvm::Instruction *I, Context &Ctx)
      : Instruction(ClassID::Insert, Opcode::Insert, I, Ctx) {}
  InsertElementInst(ClassID SubclassID, llvm::Instruction *I, Context &Ctx)
      : Instruction(SubclassID, Opcode::Insert, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  static Value *create(Value *Vec, Value *NewElt, Value *Idx,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Value *Vec, Value *NewElt, Value *Idx,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Insert;
  }
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const InsertElementInst &SBGI) {
    return SBGI.hash();
  }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const InsertElementInst &SBGI) {
    SBGI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class ExtractElementInst : public Instruction {
  /// Use Context::createExtractElementInst(). Don't call
  /// the constructor directly.
  ExtractElementInst(llvm::Instruction *I, Context &Ctx)
      : Instruction(ClassID::Extract, Opcode::Extract, I, Ctx) {}
  ExtractElementInst(ClassID SubclassID, llvm::Instruction *I, Context &Ctx)
      : Instruction(SubclassID, Opcode::Extract, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  static Value *create(Value *Vec, Value *Idx, Instruction *InsertBefore,
                       Context &Ctx, const Twine &Name = "");
  static Value *create(Value *Vec, Value *Idx, BasicBlock *InsertAtEnd,
                       Context &Ctx, const Twine &Name = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Extract;
  }
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const ExtractElementInst &SBGI) {
    return SBGI.hash();
  }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const ExtractElementInst &SBGI) {
    SBGI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class ShuffleVectorInst : public Instruction {
  /// Use Context::createShuffleVectorInst(). Don't call
  /// the constructor directly.
  ShuffleVectorInst(llvm::Instruction *I, Context &Ctx)
      : Instruction(ClassID::ShuffleVec, Opcode::ShuffleVec, I, Ctx) {}
  ShuffleVectorInst(ClassID SubclassID, llvm::Instruction *I, Context &Ctx)
      : Instruction(SubclassID, Opcode::ShuffleVec, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  static Value *create(Value *V1, Value *V2, Value *Mask,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Value *V1, Value *V2, Value *Mask,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Value *V1, Value *V2, ArrayRef<int> Mask,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Value *V1, Value *V2, ArrayRef<int> Mask,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ShuffleVec;
  }
  SmallVector<int> getShuffleMask() const {
    SmallVector<int> Mask;
    cast<llvm::ShuffleVectorInst>(Val)->getShuffleMask(Mask);
    return Mask;
  }
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const ShuffleVectorInst &SBGI) {
    return SBGI.hash();
  }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const ShuffleVectorInst &SBGI) {
    SBGI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class RetInst : public Instruction {
  /// Use Context::createRetInst(). Don't call the
  /// constructor directly.
  RetInst(llvm::Instruction *I, Context &Ctx)
      : Instruction(ClassID::Ret, Opcode::Ret, I, Ctx) {}
  RetInst(ClassID SubclassID, llvm::Instruction *I, Context &Ctx)
      : Instruction(SubclassID, Opcode::Ret, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  static Value *create(Value *RetVal, Instruction *InsertBefore, Context &Ctx);
  static Value *create(Value *RetVal, BasicBlock *InsertAtEnd, Context &Ctx);
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Ret;
  }
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  /// \Returns null if there is no return value.
  Value *getReturnValue() const;
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const RetInst &I) { return I.hash(); }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS, const RetInst &I) {
    I.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class CallInst : public Instruction {
  /// Use Context::createCallInst(). Don't call the
  /// constructor directly.
  CallInst(llvm::Instruction *I, Context &Ctx)
      : Instruction(ClassID::Call, Opcode::Call, I, Ctx) {}
  CallInst(ClassID SubclassID, llvm::Instruction *I, Context &Ctx)
      : Instruction(SubclassID, Opcode::Call, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  static CallInst *create(FunctionType *FTy, Value *Func,
                          ArrayRef<Value *> Args, BBIterator WhereIt,
                          BasicBlock *WhereBB, Context &Ctx,
                          const Twine &NameStr = "");
  static CallInst *create(FunctionType *FTy, Value *Func,
                          ArrayRef<Value *> Args, Instruction *InsertBefore,
                          Context &Ctx, const Twine &NameStr = "");
  static CallInst *create(FunctionType *FTy, Value *Func,
                          ArrayRef<Value *> Args, BasicBlock *InsertAtEnd,
                          Context &Ctx, const Twine &NameStr = "");

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Call;
  }
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const CallInst &I) { return I.hash(); }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS, const CallInst &I) {
    I.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class GetElementPtrInst : public Instruction {
  /// Use Context::createGetElementPtrInst(). Don't call
  /// the constructor directly.
  GetElementPtrInst(llvm::Instruction *I, Context &Ctx)
      : Instruction(ClassID::GetElementPtr, Opcode::GetElementPtr, I, Ctx) {}
  GetElementPtrInst(ClassID SubclassID, llvm::Instruction *I, Context &Ctx)
      : Instruction(SubclassID, Opcode::GetElementPtr, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  static Value *create(Type *Ty, Value *Ptr, ArrayRef<Value *> IdxList,
                       BBIterator WhereIt, BasicBlock *WhereBB, Context &Ctx,
                       const Twine &NameStr = "");
  static Value *create(Type *Ty, Value *Ptr, ArrayRef<Value *> IdxList,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &NameStr = "");
  static Value *create(Type *Ty, Value *Ptr, ArrayRef<Value *> IdxList,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &NameStr = "");

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::GetElementPtr;
  }
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const GetElementPtrInst &I) { return I.hash(); }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS, const GetElementPtrInst &I) {
    I.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class OpaqueInst : public Instruction {
  /// Use Context::createOpaqueInstruction(). Don't call the
  /// constructor directly.
  OpaqueInst(llvm::Instruction *I, Context &Ctx)
      : Instruction(ClassID::Opaque, Opcode::Opaque, I, Ctx) {}
  OpaqueInst(ClassID SubclassID, llvm::Instruction *I, Context &Ctx)
      : Instruction(SubclassID, Opcode::Opaque, I, Ctx) {}
  friend class BasicBlock;
  friend class Context; // For creating SB constants.
  void detachExtras() final {}
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  bool isRealOperandUse(llvm::Use &Use) const final { return true; }
  unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const final {
#ifndef NDEBUG
    verifyUserOfLLVMUse(UseToMatch);
#endif
    return UseToMatch.getOperandNo();
  }
  DmpVector<llvm::Instruction *> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }
  DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const final {
    return getLLVMInstrs();
  }

public:
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Opaque;
  }
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const OpaqueInst &SBGI) { return SBGI.hash(); }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS, const OpaqueInst &SBGI) {
    SBGI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class BasicBlock : public Value {
  /// Assigns an ordering number to instructions in the block. This is used for
  /// quick comesBefore() lookups or for a rough estimate of distance.
  DenseMap<Instruction *, int64_t> InstrNumberMap;
  /// When we first assign numbers to instructions we use this step. This allows
  /// us to insert new instructions in between without renumbering the whole
  /// block.
public:
  static constexpr const int64_t InstrNumberingStep = 64;

private:
  void renumberInstructions();
  /// This is called after \p I has been inserted into its parent block.
  void assignInstrNumber(Instruction *I);
  void removeInstrNumber(Instruction *I);
  friend void Instruction::moveBefore(BasicBlock &, const BBIterator &);
  friend void Instruction::insertBefore(Instruction *);
  friend void Instruction::insertInto(BasicBlock *, const BBIterator &);
  friend void Instruction::eraseFromParent();
  friend void Instruction::removeFromParent();

public:
  int64_t getInstrNumber(const Instruction *I) const {
    auto It = InstrNumberMap.find(I);
    assert(It != InstrNumberMap.end() && "Missing InstrNumber!");
    return It->second;
  }
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  /// Builds a graph that contains all values in \p BB in their original form
  /// i.e., no vectorization is taking place here.
  void buildBasicBlockFromIR(llvm::BasicBlock *BB);
  /// \Returns the iterator to the first non-PHI instruction.
  BBIterator getFirstNonPHIIt();

private:
  friend void
  Value::replaceUsesWithIf(Value *,
                           llvm::function_ref<bool(Use)>); // for ChangeTracker.
  friend void Value::replaceAllUsesWith(Value *);   // for ChangeTracker.
  friend void User::setOperand(unsigned,
                               Value *); // for ChangeTracker

  /// Detach BasicBlock from the underlying BB. This is called by
  /// the destructor.
  void detach();
  /// Use Context::createBasicBlock().
  BasicBlock(llvm::BasicBlock *BB, Context &Ctx);
  friend class Context; // For createBasicBlock().
  friend class BasicBlockAttorney;

public:
  ~BasicBlock();
  Function *getParent() const;
  /// Detaches the block and its instructions from LLVM IR.
  void detachFromLLVMIR();
  using iterator = BBIterator;
  iterator begin() const;
  iterator end() const {
    auto *BB = cast<llvm::BasicBlock>(Val);
    return iterator(BB, BB->end(), &Ctx);
  }
  std::reverse_iterator<iterator> rbegin() const {
    return std::make_reverse_iterator(end());
  }
  std::reverse_iterator<iterator> rend() const {
    return std::make_reverse_iterator(begin());
  }
  Context &getContext() const { return Ctx; }
  SandboxIRTracker &getTracker();
  Instruction *getTerminator() const;
  auto LLVMSize() const { return cast<llvm::BasicBlock>(Val)->size(); }

  hash_code hash() const final {
    return hash_combine(Value::hashCommon(),
                        hash_combine_range(begin(), end()));
  }
  friend hash_code hash_value(const BasicBlock &SBBB) { return SBBB.hash(); }

  bool empty() const { return begin() == end(); }
  Instruction &front() const;
  Instruction &back() const;

#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::BasicBlock>(Val) && "Expected BasicBlock!");
  }
  /// Verifies LLVM IR.
  void verifyFunctionIR() const {
    assert(!verifyFunction(*cast<llvm::BasicBlock>(Val)->getParent(), &errs()));
  }
  void verify();
  /// A simple LLVM IR verifier that checks that:
  /// (i)  definitions dominate uses, and
  /// (ii) PHIs are grouped at top.
  void verifyLLVMIR() const;
  // void verifyIR(const DmpVector<Value *> &Instrs) const;
  friend raw_ostream &operator<<(raw_ostream &OS, const BasicBlock &SBBB) {
    SBBB.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
  void dumpVerbose(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dumpVerbose() const final;
  /// Dump a range of instructions near \p SBV.
  LLVM_DUMP_METHOD void dumpInstrs(Value *SBV, int Num) const;
#endif
};

/// A client-attorney class for BasicBlock that allows access to
/// selected private members.
class BasicBlockAttorney {
  static llvm::BasicBlock *getBB(BasicBlock *SBBB) {
    return cast<llvm::BasicBlock>(SBBB->Val);
  }
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS;
#include "llvm/SandboxIR/SandboxIRValues.def"
};

class Context {
public:
  using RemoveCBTy = std::function<void(Instruction *)>;
  using InsertCBTy = std::function<void(Instruction *)>;
  using MoveCBTy =
      std::function<void(Instruction *, BasicBlock &, const BBIterator &)>;

  friend class PackInst; // For detachValue()

protected:
  LLVMContext &LLVMCtx;
  SandboxIRTracker ChangeTracker;
  IRBuilder<ConstantFolder> LLVMIRBuilder;

  /// Vector of callbacks called when an IR Instruction is about to get erased.
  SmallVector<std::unique_ptr<RemoveCBTy>> RemoveInstrCallbacks;
  DenseMap<BasicBlock *, SmallVector<std::unique_ptr<RemoveCBTy>>>
      RemoveInstrCallbacksBB;
  SmallVector<std::unique_ptr<InsertCBTy>> InsertInstrCallbacks;
  DenseMap<BasicBlock *, SmallVector<std::unique_ptr<InsertCBTy>>>
      InsertInstrCallbacksBB;
  SmallVector<std::unique_ptr<MoveCBTy>> MoveInstrCallbacks;
  DenseMap<BasicBlock *, SmallVector<std::unique_ptr<MoveCBTy>>>
      MoveInstrCallbacksBB;

  /// Maps LLVM Value to the corresponding Value. Owns all
  /// SandboxIR objects.
  DenseMap<llvm::Value *, std::unique_ptr<Value>> LLVMValueToValueMap;
  /// In SandboxIR some instructions correspond to multiple IR Instructions,
  /// like Packs. For such cases we map the IR instructions to the key used in
  /// LLVMValueToValueMap.
  DenseMap<llvm::Value *, llvm::Value *> MultiInstrMap;

  friend BasicBlock::~BasicBlock(); // For removing the
                                    // scheduler.
  /// This is true during quickFlush(). It helps with some assertions that would
  /// otherwise trigger.
  bool InQuickFlush = false;

  /// This is true during the initial creation of SandboxIR. This helps select
  /// different code paths during/after creation of SandboxIR.
  bool DontNumberInstrs = false;

  friend class ContextAttorney; // for setScheduler(),
                                // clearScheduler()
  /// Removes \p V from the maps and returns the unique_ptr.
  std::unique_ptr<Value> detachValue(llvm::Value *V);

  friend void Instruction::eraseFromParent();
  friend void Instruction::removeFromParent();
  friend void Instruction::moveBefore(BasicBlock &, const BBIterator &);

  void runRemoveInstrCallbacks(Instruction *I);
  void runInsertInstrCallbacks(Instruction *I);
  void runMoveInstrCallbacks(Instruction *I, BasicBlock &SBBB,
                             const BBIterator &WhereIt);

  virtual Value *
  createValueFromExtractElement(llvm::ExtractElementInst *ExtractI, int Depth) {
    return getOrCreateExtractElementInst(ExtractI);
  }
  Value *getValueFromExtractElement(llvm::ExtractElementInst *ExtractI) const;
  Value *getOrCreateValueFromExtractElement(llvm::ExtractElementInst *ExtractI,
                                            int Depth);

  virtual Value *createValueFromInsertElement(llvm::InsertElementInst *InsertI,
                                              int Depth) {
    return getOrCreateInsertElementInst(InsertI);
  }
  Value *getValueFromInsertElement(llvm::InsertElementInst *InsertI) const;
  Value *getOrCreateValueFromInsertElement(llvm::InsertElementInst *InsertI,
                                           int Depth);

  virtual Value *createValueFromShuffleVector(llvm::ShuffleVectorInst *ShuffleI,
                                              int Depth) {
    return getOrCreateShuffleVectorInst(ShuffleI);
  }

  Value *getValueFromShuffleVector(llvm::ShuffleVectorInst *ShuffleI) const;
  Value *getOrCreateValueFromShuffleVector(llvm::ShuffleVectorInst *ShuffleI,
                                           int Depth);

  /// This runs right after \p SBB has been created.
  virtual void createdBasicBlock(BasicBlock &BB) {}

#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  /// Runs right after an instruction has moved in \p BB. This is used for
  /// testing the DAG and Scheduler by SBVecContext.
  virtual void afterMoveInstrHook(BasicBlock &BB) {}
#endif
  /// This is called by the BasicBlock's destructor.
  virtual void destroyingBB(BasicBlock &BB) {}

  /// Helper for avoiding recursion loop when creating SBConstants.
  SmallDenseSet<llvm::Constant *, 8> VisitedConstants;
  Value *getOrCreateValueInternal(llvm::Value *V, int Depth,
                                  llvm::User *U = nullptr);

public:
  Context(LLVMContext &LLVMCtx);
  virtual ~Context() {}
  SandboxIRTracker &getTracker() { return ChangeTracker; }
  auto &getLLVMIRBuilder() { return LLVMIRBuilder; }
  size_t getNumValues() const {
    return LLVMValueToValueMap.size() + MultiInstrMap.size();
  }

  /// Remove \p SBV from all SandboxIR maps and stop owning it. This effectively
  /// detaches \p SBV from the underlying IR.
  std::unique_ptr<Value> detach(Value *SBV);
  Value *registerValue(std::unique_ptr<Value> &&SBVPtr);

  Value *getValue(llvm::Value *V) const;
  const Value *getValue(const llvm::Value *V) const {
    return getValue(const_cast<llvm::Value *>(V));
  }

  Constant *getConstant(llvm::Constant *C) const;
  Constant *getOrCreateConstant(llvm::Constant *C);

  Value *getOrCreateValue(llvm::Value *V);

  /// Helper function called when we create Instructions that
  /// create new constant operands. It goes through V's operands and creates
  /// Constants.
  void createMissingConstantOperands(llvm::Value *V);

  // Arguments
  Argument *getArgument(llvm::Argument *Arg) const;
  Argument *createArgument(llvm::Argument *Arg);
  Argument *getOrCreateArgument(llvm::Argument *Arg);

  // InsertElementInstruction
  InsertElementInst *getInsertElementInst(llvm::InsertElementInst *I) const;
  InsertElementInst *createInsertElementInst(llvm::InsertElementInst *I);
  InsertElementInst *getOrCreateInsertElementInst(llvm::InsertElementInst *I);

  // InsertElementInstruction
  ExtractElementInst *getExtractElementInst(llvm::ExtractElementInst *I) const;
  ExtractElementInst *createExtractElementInst(llvm::ExtractElementInst *I);
  ExtractElementInst *
  getOrCreateExtractElementInst(llvm::ExtractElementInst *I);

  // ShuffleVectorInstruction
  ShuffleVectorInst *getShuffleVectorInst(llvm::ShuffleVectorInst *I) const;
  ShuffleVectorInst *createShuffleVectorInst(llvm::ShuffleVectorInst *I);
  ShuffleVectorInst *getOrCreateShuffleVectorInst(llvm::ShuffleVectorInst *I);

  // Return
  RetInst *getRetInst(llvm::ReturnInst *I) const;
  RetInst *createRetInst(llvm::ReturnInst *I);
  RetInst *getOrCreateRetInst(llvm::ReturnInst *I);

  // Call
  CallInst *getCallInst(llvm::CallInst *I) const;
  CallInst *createCallInst(llvm::CallInst *I);
  CallInst *getOrCreateCallInst(llvm::CallInst *I);

  // GEP
  GetElementPtrInst *getGetElementPtrInst(llvm::GetElementPtrInst *I) const;
  GetElementPtrInst *createGetElementPtrInst(llvm::GetElementPtrInst *I);
  GetElementPtrInst *getOrCreateGetElementPtrInst(llvm::GetElementPtrInst *I);

  // OpaqueInstr
  OpaqueInst *getOpaqueInstruction(llvm::Instruction *I) const;
  OpaqueInst *createOpaqueInstruction(llvm::Instruction *I);
  OpaqueInst *getOrCreateOpaqueInstruction(llvm::Instruction *I);

  // BranchInst
  BranchInst *getBranchInst(llvm::BranchInst *BI) const;
  BranchInst *createBranchInst(llvm::BranchInst *BI);
  BranchInst *getOrCreateBranchInst(llvm::BranchInst *BI);

  // Store
  StoreInst *getStoreInst(llvm::StoreInst *SI) const;
  StoreInst *createStoreInst(llvm::StoreInst *SI);
  StoreInst *getOrCreateStoreInst(llvm::StoreInst *SI);

  // Load
  LoadInst *getLoadInst(llvm::LoadInst *LI) const;
  LoadInst *createLoadInst(llvm::LoadInst *LI);
  LoadInst *getOrCreateLoadInst(llvm::LoadInst *LI);

  // Cast
  CastInst *getCastInst(llvm::CastInst *CI) const;
  CastInst *createCastInst(llvm::CastInst *CI);
  CastInst *getOrCreateCastInst(llvm::CastInst *CI);

  // PHI
  PHINode *getPHINode(llvm::PHINode *PHI) const;
  PHINode *createPHINode(llvm::PHINode *PHI);
  PHINode *getOrCreatePHINode(llvm::PHINode *PHI);

  // Select
  SelectInst *getSelectInst(llvm::SelectInst *SI) const;
  SelectInst *createSelectInst(llvm::SelectInst *SI);
  SelectInst *getOrCreateSelectInst(llvm::SelectInst *SI);

  // BinaryOperator
  BinaryOperator *getBinaryOperator(llvm::BinaryOperator *BO) const;
  BinaryOperator *createBinaryOperator(llvm::BinaryOperator *BO);
  BinaryOperator *getOrCreateBinaryOperator(llvm::BinaryOperator *BO);

  // UnaryOperator
  UnaryOperator *getUnaryOperator(llvm::UnaryOperator *UO) const;
  UnaryOperator *createUnaryOperator(llvm::UnaryOperator *UO);
  UnaryOperator *getOrCreateUnaryOperator(llvm::UnaryOperator *UO);

  // Cmp
  CmpInst *getCmpInst(llvm::CmpInst *CI) const;
  CmpInst *createCmpInst(llvm::CmpInst *CI);
  CmpInst *getOrCreateCmpInst(llvm::CmpInst *CI);

  // Block
  BasicBlock *getBasicBlock(llvm::BasicBlock *BB) const;
  BasicBlock *createBasicBlock(llvm::BasicBlock *BB);

  // Function
  Function *getFunction(llvm::Function *F) const;
  Function *createFunction(llvm::Function *F, bool CreateBBs = true);

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
  RemoveCBTy *registerRemoveInstrCallbackBB(BasicBlock &BB, RemoveCBTy CB);
  void unregisterRemoveInstrCallbackBB(BasicBlock &BB, RemoveCBTy *CB);

  InsertCBTy *registerInsertInstrCallbackBB(BasicBlock &BB, InsertCBTy CB);
  void unregisterInsertInstrCallbackBB(BasicBlock &BB, InsertCBTy *CB);

  MoveCBTy *registerMoveInstrCallbackBB(BasicBlock &BB, MoveCBTy CB);
  void unregisterMoveInstrCallbackBB(BasicBlock &BB, MoveCBTy *CB);

  /// Clears state for the whole context quickly. This is to speed up
  /// destruction of the whole SandboxIR.
  virtual void quickFlush();

#ifndef NDEBUG
  /// Used in tests
  void disableCallbacks() { CallbacksDisabled = true; }
#endif

protected:
#ifndef NDEBUG
  bool CallbacksDisabled = false;
#endif
  friend class ContextAttorney;
};

/// A client-attorney class for Context.
class ContextAttorney {
  friend class Region;
  friend class RegionBuilderFromMD;

public:
  static LLVMContext &getLLVMContext(Context &Ctx) { return Ctx.LLVMCtx; }
};

class Function : public Value {
  llvm::Function *getFunction() const { return cast<llvm::Function>(Val); }

public:
  Function(llvm::Function *F, Context &Ctx)
      : Value(ClassID::Function, F, Ctx) {}
  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Function;
  }

  /// Iterates over BasicBlocks
  class iterator {
    llvm::Function::iterator It;
#ifndef NDEBUG
    llvm::Function *F;
#endif
    Context *Ctx;

  public:
    using difference_type = std::ptrdiff_t;
    using value_type = BasicBlock;
    using pointer = BasicBlock *;
    using reference = value_type &;
    using iterator_category = std::bidirectional_iterator_tag;

#ifndef NDEBUG
    iterator() : F(nullptr), Ctx(nullptr) {}
    iterator(llvm::Function::iterator It, llvm::Function *F, Context &Ctx)
        : It(It), F(F), Ctx(&Ctx) {}
#else
    iterator() : Ctx(nullptr) {}
    iterator(llvm::Function::iterator It, Context &Ctx) : It(It), Ctx(&Ctx) {}
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
      return *cast<BasicBlock>(Ctx->getValue(&*It));
    }
    const BasicBlock &operator*() const {
      assert(It != F->end() && "Dereferencing end()!");
      return *cast<BasicBlock>(Ctx->getValue(&*It));
    }
  };

  Argument *getArg(unsigned Idx) const {
    llvm::Argument *Arg = getFunction()->getArg(Idx);
    return cast<Argument>(Ctx.getValue(Arg));
  }

  size_t arg_size() const { return getFunction()->arg_size(); }
  bool arg_empty() const { return getFunction()->arg_empty(); }

  struct LLVMArgToSBArgConst {
    Context &Ctx;
    LLVMArgToSBArgConst(Context &Ctx) : Ctx(Ctx) {}
    const Argument &operator()(const llvm::Argument &Arg) const {
      return *cast<Argument>(Ctx.getValue(const_cast<llvm::Argument *>(&Arg)));
    }
  };
  using const_arg_iterator =
      mapped_iterator<llvm::Function::const_arg_iterator, LLVMArgToSBArgConst>;

  const_arg_iterator arg_begin() const {
    LLVMArgToSBArgConst GetSBArg(Ctx);
    const llvm::Function *F = cast<llvm::Function>(Val);
    return map_iterator(F->arg_begin(), GetSBArg);
  }
  const_arg_iterator arg_end() const {
    LLVMArgToSBArgConst GetSBArg(Ctx);
    const llvm::Function *F = cast<llvm::Function>(Val);
    return map_iterator(F->arg_end(), GetSBArg);
  }
  iterator_range<const_arg_iterator> args() const {
    return make_range(arg_begin(), arg_end());
  }

  struct LLVMArgToSBArg {
    Context &Ctx;
    LLVMArgToSBArg(Context &Ctx) : Ctx(Ctx) {}
    Argument &operator()(llvm::Argument &Arg) const {
      return *cast<Argument>(Ctx.getValue(&Arg));
    }
  };
  using arg_iterator =
      mapped_iterator<llvm::Function::arg_iterator, LLVMArgToSBArg>;

  arg_iterator arg_begin() {
    LLVMArgToSBArg GetSBArg(Ctx);
    llvm::Function *F = cast<llvm::Function>(Val);
    return map_iterator(F->arg_begin(), GetSBArg);
  }
  arg_iterator arg_end() {
    LLVMArgToSBArg GetSBArg(Ctx);
    llvm::Function *F = cast<llvm::Function>(Val);
    return map_iterator(F->arg_end(), GetSBArg);
  }
  iterator_range<arg_iterator> args() {
    return make_range(arg_begin(), arg_end());
  }

  BasicBlock &getEntryBlock() const {
    llvm::BasicBlock &EntryBB = getFunction()->getEntryBlock();
    return *cast<BasicBlock>(Ctx.getValue(&EntryBB));
  }

  iterator begin() const {
    llvm::Function *F = getFunction();
#ifndef NDEBUG
    return iterator(F->begin(), F, Ctx);
#else
    return iterator(F->begin(), Ctx);
#endif
  }
  iterator end() const {
    llvm::Function *F = getFunction();
#ifndef NDEBUG
    return iterator(F->end(), F, Ctx);
#else
    return iterator(F->end(), Ctx);
#endif
  }

  /// Detaches the function, its blocks and its instructions from LLVM IR.
  void detachFromLLVMIR();

  hash_code hash() const final {
    auto Hash =
        hash_combine(Value::hashCommon(), hash_combine_range(begin(), end()));
    for (auto ArgIdx : seq<unsigned>(0, arg_size()))
      Hash = hash_combine(Hash, getArg(ArgIdx));
    return Hash;
  }
  friend hash_code hash_value(const Function &SBF) { return SBF.hash(); }

#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::Function>(Val) && "Expected Function!");
  }
  void dumpNameAndArgs(raw_ostream &OS) const;
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
  void dumpVerbose(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dumpVerbose() const final;
#endif
};

using sb_succ_iterator = SuccIterator<Instruction, BasicBlock>;
using const_sb_succ_iterator =
    SuccIterator<const Instruction, const BasicBlock>;
using sb_succ_range = iterator_range<sb_succ_iterator>;
using const_sb_succ_range = iterator_range<const_sb_succ_iterator>;

inline sb_succ_iterator succ_begin(Instruction *I) {
  return sb_succ_iterator(I);
}
inline const_sb_succ_iterator succ_begin(const Instruction *I) {
  return const_sb_succ_iterator(I);
}
inline sb_succ_iterator succ_end(Instruction *I) {
  return sb_succ_iterator(I, true);
}
inline const_sb_succ_iterator succ_end(const Instruction *I) {
  return const_sb_succ_iterator(I, true);
}
inline bool succ_empty(const Instruction *I) {
  return succ_begin(I) == succ_end(I);
}
inline unsigned succ_size(const Instruction *I) {
  return std::distance(succ_begin(I), succ_end(I));
}
inline sb_succ_range successors(Instruction *I) {
  return sb_succ_range(succ_begin(I), succ_end(I));
}
inline const_sb_succ_range successors(const Instruction *I) {
  return const_sb_succ_range(succ_begin(I), succ_end(I));
}

inline sb_succ_iterator succ_begin(BasicBlock *BB) {
  return sb_succ_iterator(BB->getTerminator());
}
inline const_sb_succ_iterator succ_begin(const BasicBlock *BB) {
  return const_sb_succ_iterator(BB->getTerminator());
}
inline sb_succ_iterator succ_end(BasicBlock *BB) {
  return sb_succ_iterator(BB->getTerminator(), true);
}
inline const_sb_succ_iterator succ_end(const BasicBlock *BB) {
  return const_sb_succ_iterator(BB->getTerminator(), true);
}
inline bool succ_empty(const BasicBlock *BB) {
  return succ_begin(BB) == succ_end(BB);
}
inline unsigned succ_size(const BasicBlock *BB) {
  return std::distance(succ_begin(BB), succ_end(BB));
}
inline sb_succ_range successors(BasicBlock *BB) {
  return sb_succ_range(succ_begin(BB), succ_end(BB));
}
inline const_sb_succ_range successors(const BasicBlock *BB) {
  return const_sb_succ_range(succ_begin(BB), succ_end(BB));
}
} // namespace sandboxir

// GraphTraits for BasicBlock.
template <> struct GraphTraits<sandboxir::BasicBlock *> {
  using NodeRef = sandboxir::BasicBlock *;
  using ChildIteratorType = sandboxir::sb_succ_iterator;
  static NodeRef getEntryNode(sandboxir::BasicBlock *BB) { return BB; }
  static ChildIteratorType child_begin(NodeRef N) { return succ_begin(N); }
  static ChildIteratorType child_end(NodeRef N) { return succ_end(N); }
};
template <> struct GraphTraits<const sandboxir::BasicBlock *> {
  using NodeRef = const sandboxir::BasicBlock *;
  using ChildIteratorType = sandboxir::const_sb_succ_iterator;
  static NodeRef getEntryNode(const sandboxir::BasicBlock *BB) { return BB; }
  static ChildIteratorType child_begin(NodeRef N) { return succ_begin(N); }
  static ChildIteratorType child_end(NodeRef N) { return succ_end(N); }
};

namespace sandboxir {

template <typename RangeT>
DmpVector<Value *> getOperandBundle(const RangeT &Bndl, unsigned OpIdx) {
  DmpVector<Value *> OpVec;
  OpVec.reserve(Bndl.size());
  for (auto *SBV : Bndl) {
    auto *SBI = cast<Instruction>(SBV);
    assert(OpIdx < SBI->getNumOperands() && "Out of bounds!");
    OpVec.push_back(SBI->getOperand(OpIdx));
  }
  return OpVec;
}

template <typename RangeT>
SmallVector<DmpVector<Value *>, 2> getOperandBundles(const RangeT &Bndl) {
  SmallVector<DmpVector<Value *>, 2> OpVecs;
#ifndef NDEBUG
  unsigned NumOps = cast<Instruction>(Bndl[0])->getNumOperands();
  assert(all_of(drop_begin(Bndl),
                [NumOps](auto *V) {
                  return cast<Instruction>(V)->getNumOperands() == NumOps;
                }) &&
         "Expected same number of operands!");
#endif
  for (unsigned OpIdx :
       seq<unsigned>(cast<Instruction>(Bndl[0])->getNumOperands()))
    OpVecs.push_back(getOperandBundle(Bndl, OpIdx));
  return OpVecs;
}

} // namespace sandboxir
} // namespace llvm
#endif // LLVM_TRANSFORMS_SANDBOXIR_SANDBOXIR_H
