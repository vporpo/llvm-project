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
// `sandboxir::Context::LLVMValueToSBValueMap`. The SandboxIR Value objects are
// thin wrappers of the llvm::Value objects they point to. This means that they
// maintain almost no state, and they rely on the state of the LLVM objects.
//
// Example 1
// ---------
// sandboxir::Instruction::getOperand(N) works by:
//  (i)  Getting llvm::Value *Op = LLVMInst->getOperand(N) of the llvm instr
//       pointed to by this sandboxir instr.
//  (ii) Returning the corresponding sandboxir::Value by looking up the llvm
//      `Op` in `sandboxir::Context::LLVMValueToSBValueMap`.
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
  friend class sandboxir::UseAttorney; // For LLVMUse
  sandboxir::User *User;
  sandboxir::Context *Ctx;

  /// Don't allow the user to create a sandboxir::Use directly.
  Use(llvm::Use *LLVMUse, sandboxir::User *User, sandboxir::Context &Ctx)
      : LLVMUse(LLVMUse), User(User), Ctx(&Ctx) {}
  Use() : LLVMUse(nullptr), Ctx(nullptr) {}

  friend class sandboxir::User;               // For constructor
  friend class sandboxir::Value;              // For constructor
  friend class sandboxir::OperandUseIterator; // For constructor
  friend class sandboxir::UserUseIterator;    // For constructor
  // Several instructions need access to the sandboxir::Use() constructor for
  // their implementation of getOperandUseInternal().
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS; // For constructor
#include "llvm/SandboxIR/SandboxIRValues.def"

public:
  operator sandboxir::Value *() const { return get(); }
  sandboxir::Value *get() const;
  sandboxir::User *getUser() const { return User; }
  unsigned getOperandNo() const;
  sandboxir::Context *getContext() const { return Ctx; }
  void swap(sandboxir::Use &Other);
  bool operator==(const sandboxir::Use &Other) const {
    assert(Ctx == Other.Ctx && "Contexts differ!");
    return LLVMUse == Other.LLVMUse && User == Other.User;
  }
  bool operator!=(const sandboxir::Use &Other) const {
    return !(*this == Other);
  }
  void set(sandboxir::Value *Val);
  inline sandboxir::Value *operator=(sandboxir::Value *Val) {
    set(Val);
    return Val;
  }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  void dump() const;
#endif // NDEBUG
};

/// A client-attorney class for sandboxir::Use.
class UseAttorney {
  static llvm::Use *getLLVMUse(sandboxir::Use &Use) { return Use.LLVMUse; }
  friend class sandboxir::BasicBlock; // For getLLVMUse()
};

/// Returns the operand edge when dereferenced.
class OperandUseIterator {
  sandboxir::Use Use;
  /// Don't let the user create a non-empty SBOperandUseIterator.
  OperandUseIterator(const sandboxir::Use &Use) : Use(Use) {}
  friend class sandboxir::User; // For constructor
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS)                                              \
  friend class sandboxir::CLASS; // For constructor
#include "llvm/SandboxIR/SandboxIRValues.def"

public:
  using difference_type = std::ptrdiff_t;
  using value_type = sandboxir::Use;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::input_iterator_tag;

  OperandUseIterator() {}
  value_type operator*() const;
  OperandUseIterator &operator++();
  bool operator==(const sandboxir::OperandUseIterator &Other) const {
    return Use == Other.Use;
  }
  bool operator!=(const sandboxir::OperandUseIterator &Other) const {
    return !(*this == Other);
  }
};

/// Returns user edge when dereferenced.
class UserUseIterator {
  sandboxir::Use Use;
  /// Don't let the user create a non-empty UserUseIterator.
  UserUseIterator(const sandboxir::Use &Use) : Use(Use) {}
  friend class sandboxir::Value; // For constructor

#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS; // For constructor
#include "llvm/SandboxIR/SandboxIRValues.def"

public:
  using difference_type = std::ptrdiff_t;
  using value_type = sandboxir::Use;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::input_iterator_tag;

  UserUseIterator() {}
  value_type operator*() const;
  UserUseIterator &operator++();
  bool operator==(const sandboxir::UserUseIterator &Other) const {
    return Use == Other.Use;
  }
  bool operator!=(const sandboxir::UserUseIterator &Other) const {
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
    static_assert(std::is_same<ItTy, sandboxir::UserUseIterator>::value ||
                      std::is_same<ItTy, sandboxir::OperandUseIterator>::value,
                  "Unsupported ItTy!");
    if constexpr (std::is_same<ItTy, sandboxir::UserUseIterator>::value) {
      return (*It).getUser();
    } else if constexpr (std::is_same<ItTy,
                                      sandboxir::OperandUseIterator>::value) {
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
  friend class sandboxir::ValueAttorney; // For accessing llvm::Value *Val

  /// All values point to the context.
  sandboxir::Context &Ctx;
  // This is used by eraseFromParent().
  void clearValue() { Val = nullptr; }
  template <typename ItTy, typename SBTy> friend class LLVMOpUserItToSBTy;

public:
  Value(ClassID SubclassID, llvm::Value *Val, sandboxir::Context &Ctx);
  virtual ~Value() = default;
  ClassID getSubclassID() const { return SubclassID; }

  using use_iterator = sandboxir::UserUseIterator;
  using const_use_iterator = sandboxir::UserUseIterator;

  use_iterator use_begin();
  const_use_iterator use_begin() const {
    return const_cast<sandboxir::Value *>(this)->use_begin();
  }
  use_iterator use_end() {
    return use_iterator(sandboxir::Use(nullptr, nullptr, Ctx));
  }
  const_use_iterator use_end() const {
    return const_cast<sandboxir::Value *>(this)->use_end();
  }

  iterator_range<use_iterator> uses() {
    return make_range<use_iterator>(use_begin(), use_end());
  }
  iterator_range<const_use_iterator> uses() const {
    return make_range<const_use_iterator>(use_begin(), use_end());
  }

  using user_iterator =
      RetTyAdaptor<sandboxir::User, sandboxir::UserUseIterator>;
  using const_user_iterator = user_iterator;

  user_iterator user_begin();
  user_iterator user_end() {
    return user_iterator(sandboxir::Use(nullptr, nullptr, Ctx));
  }
  const_user_iterator user_begin() const {
    return const_cast<sandboxir::Value *>(this)->user_begin();
  }
  const_user_iterator user_end() const {
    return const_cast<sandboxir::Value *>(this)->user_end();
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

  sandboxir::Value *getSingleUser() const;

  Type *getType() const { return Val->getType(); }

  sandboxir::Context &getContext() const;
  SandboxIRTracker &getTracker();
  virtual hash_code hashCommon() const {
    return hash_combine(SubclassID, &Ctx, Val);
  }
  /// WARNING: DstU can be nullptr if it is in a BB that is not in SandboxIR!
  void replaceUsesWithIf(
      sandboxir::Value *OtherV,
      llvm::function_ref<bool(sandboxir::User *DstU, unsigned OpIdx)>
          ShouldReplace);
  void replaceAllUsesWith(sandboxir::Value *Other);
  virtual hash_code hash() const = 0;
  friend hash_code hash_value(const sandboxir::Value &SBV) {
    return SBV.hash();
  }
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
  friend raw_ostream &operator<<(raw_ostream &OS, const sandboxir::Value &SBV) {
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
  static llvm::Value *getValue(const sandboxir::Value *SBV) { return SBV->Val; }

#define DEF_VALUE(ID, CLASS) friend class CLASS;
#define DEF_USER(ID, CLASS) friend class CLASS;
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS;
#include "llvm/SandboxIR/SandboxIRValues.def"

  friend class sandboxir::Instruction;
  friend class sandboxir::DependencyGraph;
  template <typename T> friend class llvm::DmpVector;
  friend class sandboxir::Analysis;
  friend class sandboxir::PassManager;
  friend class sandboxir::Context;
  friend class sandboxir::User;
  friend class sandboxir::Use;
  friend class sandboxir::MemSeedContainer;
  friend class sandboxir::SandboxIRTracker;
  friend class sandboxir::RegionBuilderFromMD;
  friend class sandboxir::Region;
  friend class sandboxir::VecUtilsPrivileged;

  friend void sandboxir::Value::replaceUsesWithIf(
      sandboxir::Value *,
      llvm::function_ref<bool(sandboxir::User *, unsigned)>);
  friend class sandboxir::Scheduler;
  friend class sandboxir::OperandUseIterator;
  friend class sandboxir::BBIterator;
  friend ReplaceAllUsesWith::ReplaceAllUsesWith(sandboxir::Value *,
                                                SandboxIRTracker &);
  friend ReplaceUsesOfWith::ReplaceUsesOfWith(sandboxir::User *,
                                              sandboxir::Value *,
                                              sandboxir::Value *,
                                              SandboxIRTracker &);
  friend class sandboxir::DeleteOnAccept;
  friend class sandboxir::CreateAndInsertInstr;
  friend class sandboxir::EraseFromParent;
};

/// A function argument.
class Argument : public sandboxir::Value {
  Argument(llvm::Argument *Arg, sandboxir::Context &SBCtx);
  friend class sandboxir::Context; // for createArgument()

public:
  static bool classof(const sandboxir::Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::Argument &TArg) {
    return TArg.hash();
  }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::Argument>(Val) && "Expected Argument!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::Argument &TArg) {
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
class User : public sandboxir::Value {
protected:
  User(ClassID ID, llvm::Value *V, sandboxir::Context &SBCtx);
  friend class sandboxir::Instruction; // For constructors.

  /// \Returns the SBUse edge that corresponds to \p OpIdx.
  /// Note: This is the default implementation that works for instructions that
  /// match the underlying LLVM instruction. All others should use a different
  /// implementation.
  sandboxir::Use getOperandUseDefault(unsigned OpIdx, bool Verify) const;
  virtual sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                               bool Verify) const = 0;
  friend class sandboxir::OperandUseIterator; // for getOperandUseInternal()

  /// \Returns true if \p Use should be considered as an edge to its SandboxIR
  /// operand. Most instructions should return true.
  /// Currently it is only Uses from Vectors into Packs that return false.
  virtual bool isRealOperandUse(llvm::Use &Use) const = 0;
  friend class sandboxir::UserUseIterator; // for isRealOperandUse()

  /// The default implementation works only for single-LLVMIR-instruction
  /// SBUsers and only if they match exactly the LLVM instruction.
  unsigned getUseOperandNoDefault(const sandboxir::Use &Use) const {
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
  static bool classof(const sandboxir::Value *From);
  using op_iterator = sandboxir::OperandUseIterator;
  using const_op_iterator = sandboxir::OperandUseIterator;
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
    auto Hash = sandboxir::Value::hashCommon();
    for (sandboxir::Value *Op : operands())
      Hash = hash_combine(Hash, Op);
    return Hash;
  }
  sandboxir::Value *getOperand(unsigned OpIdx) const {
    return getOperandUse(OpIdx).get();
  }
  /// \Returns the operand edge for \p OpIdx. NOTE: This should also work for
  /// OpIdx == getNumOperands(), which is used for op_end().
  sandboxir::Use getOperandUse(unsigned OpIdx) const {
    return getOperandUseInternal(OpIdx, /*Verify=*/true);
  }
  /// \Returns the operand index of \p Use.
  virtual unsigned getUseOperandNo(const sandboxir::Use &Use) const = 0;
  sandboxir::Value *getSingleOperand() const;
  virtual void setOperand(unsigned OperandIdx, sandboxir::Value *Operand);
  virtual unsigned getNumOperands() const {
    return isa<llvm::User>(Val) ? cast<llvm::User>(Val)->getNumOperands() : 0;
  }
  /// Replaces any operands that match \p FromV with \p ToV. Returns whether any
  /// operands were replaced.
  /// WARNING: This will replace even uses that are not in SandboxIR!
  bool replaceUsesOfWith(sandboxir::Value *FromV, sandboxir::Value *ToV);

#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::User>(Val) && "Expected User!");
  }
  void dumpCommonHeader(raw_ostream &OS) const final;
#endif

protected:
  /// \Returns the operand index that corresponds to \p UseToMatch.
  virtual unsigned getOperandUseIdx(const llvm::Use &UseToMatch) const = 0;
  friend class sandboxir::UserAttorney; // For testing
  friend void sandboxir::Value::replaceUsesWithIf(
      sandboxir::Value *,
      llvm::function_ref<bool(sandboxir::User *, unsigned)>);
};

/// A simple client-attorney class that exposes some protected members of
/// sandboxir::User for use in tests.
class UserAttorney {
public:
  // For testing.
  static unsigned getOperandUseIdx(const sandboxir::User *SBU,
                                   const llvm::Use &UseToMatch) {
    return SBU->getOperandUseIdx(UseToMatch);
  }
};

class Constant : public sandboxir::User {
  /// Use Context::createConstant() instead.
  Constant(llvm::Constant *C, sandboxir::Context &SBCtx);
  friend class sandboxir::Context; // For constructor.
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From);
  sandboxir::Context &getParent() const { return getContext(); }
  hash_code hashCommon() const final { return sandboxir::User::hashCommon(); }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::Constant &SBC) {
    return SBC.hash();
  }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::Constant>(Val) && "Expected Constant!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::Constant &SBC) {
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
  using value_type = sandboxir::Instruction;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::bidirectional_iterator_tag;

private:
  llvm::BasicBlock *BB;
  /// This should always point to the bottom IR instruction of a multi-IR
  /// sandboxir::Instruction.
  llvm::BasicBlock::iterator It;
  sandboxir::Context *SBCtx;
  pointer getI(llvm::BasicBlock::iterator It) const;

public:
  BBIterator() : BB(nullptr), SBCtx(nullptr) {}
  BBIterator(llvm::BasicBlock *BB, llvm::BasicBlock::iterator It,
             sandboxir::Context *SBCtx)
      : BB(BB), It(It), SBCtx(SBCtx) {}
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
  bool operator==(const sandboxir::BBIterator &Other) const {
    assert(SBCtx == Other.SBCtx && "SBBBIterators in different context!");
    return It == Other.It;
  }
  bool operator!=(const sandboxir::BBIterator &Other) const {
    return !(*this == Other);
  }
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
class Instruction : public sandboxir::User {
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
  Instruction(ClassID ID, Opcode Opc, llvm::Instruction *I,
              sandboxir::Context &SBCtx);

#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS;
#include "llvm/SandboxIR/SandboxIRValues.def"

  /// Any extra actions that need to be performed upon detach.
  virtual void detachExtras() = 0;
  friend class sandboxir::Context; // For detachExtras()

  /// A SBInstruction may map to multiple IR Instruction. This returns its
  /// topmost IR instruction.
  llvm::Instruction *getTopmostIRInstruction() const;

  /// \Returns all IR instructions that make up this SBInstruction in reverse
  /// program order.
  virtual DmpVector<llvm::Instruction *> getLLVMInstrs() const = 0;
  friend class sandboxir::CostModel; // For getLLVMInstrs().
  /// \Returns all IR instructions with external operands. Note: This is useful
  /// for multi-IR instructions like Packs, that are composed of both
  /// internal-only and external-facing IR Instructions.
  virtual DmpVector<llvm::Instruction *>
  getLLVMInstrsWithExternalOperands() const = 0;
  friend void DeleteOnAccept::apply();
  friend ReplaceAllUsesWith::ReplaceAllUsesWith(sandboxir::Value *,
                                                SandboxIRTracker &);
  friend ReplaceUsesOfWith::ReplaceUsesOfWith(sandboxir::User *,
                                              sandboxir::Value *,
                                              sandboxir::Value *,
                                              SandboxIRTracker &);
  friend bool sandboxir::User::replaceUsesOfWith(sandboxir::Value *,
                                                 sandboxir::Value *);
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
  friend class sandboxir::InstructionAttorney;

public:
  static const char *getOpcodeName(Opcode Opc);
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, Opcode Opc) {
    OS << getOpcodeName(Opc);
    return OS;
  }
#endif
  /// This is used by sandboxir::SBBasicBlcok::iterator.
  virtual unsigned getNumOfIRInstrs() const = 0;
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From);
  sandboxir::BBIterator getIterator() const;
  sandboxir::Instruction *getNextNode() const;
  sandboxir::Instruction *getPrevNode() const;
  /// \Returns the opcode of the Instruction contained.
  Opcode getOpcode() const { return Opc; }
  /// Detach this from its parent sandboxir::BasicBlock without deleting it.
  void removeFromParent();
  /// Detach this sandboxir::Value from its parent and delete it.
  void eraseFromParent();
  /// \Returns the parent graph or null if there is no parent graph, i.e., when
  /// it holds a Constant.
  sandboxir::BasicBlock *getParent() const;
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
  bool comesBefore(sandboxir::Instruction *Other) const {
    return getInstrNumber() < Other->getInstrNumber();
  }
  bool comesAfter(sandboxir::Instruction *Other) {
    return Other->comesBefore(this);
  }
  /// \Returns a (very) approximate absolute distance between this instruction
  /// and \p ToI. This is a constant-time operation.
  uint64_t getApproximateDistanceTo(sandboxir::Instruction *ToI) const;
  void moveBefore(sandboxir::BasicBlock &SBBB,
                  const sandboxir::BBIterator &WhereIt);
  void moveBefore(sandboxir::Instruction *Before) {
    moveBefore(*Before->getParent(), Before->getIterator());
  }
  void moveAfter(sandboxir::Instruction *After) {
    moveBefore(*After->getParent(), std::next(After->getIterator()));
  }
  hash_code hashCommon() const override {
    return hash_combine(sandboxir::User::hashCommon(), getParent());
  }
  void insertBefore(sandboxir::Instruction *BeforeI);
  void insertAfter(sandboxir::Instruction *AfterI);
  void insertInto(sandboxir::BasicBlock *SBBB,
                  const sandboxir::BBIterator &WhereIt);

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
  sandboxir::BasicBlock *getSuccessor(unsigned Idx) const LLVM_READONLY;

#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::Instruction &SBI) {
    SBI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

/// A client-attorney class for sandboxir::Instruction.
class InstructionAttorney {
public:
  friend class sandboxir::RegionBuilderFromMD;
  static MDNode *getMetadata(const sandboxir::Instruction *SBI,
                             unsigned KindID) {
    return SBI->getMetadata(KindID);
  }
  static MDNode *getMetadata(const sandboxir::Instruction *SBI,
                             StringRef Kind) {
    return SBI->getMetadata(Kind);
  }
};

class CmpInst : public sandboxir::Instruction {
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
  CmpInst(llvm::CmpInst *CI, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::Cmp, getCmpOpcode(CI->getOpcode()), CI,
                               Ctx) {
    assert((Opc == Opcode::FCmp || Opc == Opcode::ICmp) && "Bad Opcode!");
  }
  friend class sandboxir::Context; // for sandboxir::CmpInst()
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  static sandboxir::Value *create(llvm::CmpInst::Predicate Pred,
                                  sandboxir::Value *LHS, sandboxir::Value *RHS,
                                  sandboxir::Instruction *InsertBefore,
                                  sandboxir::Context &SBCtx,
                                  const Twine &Name = "",
                                  MDNode *FPMathTag = nullptr);
  static sandboxir::Value *create(llvm::CmpInst::Predicate Pred,
                                  sandboxir::Value *LHS, sandboxir::Value *RHS,
                                  sandboxir::BasicBlock *InsertAtEnd,
                                  sandboxir::Context &SBCtx,
                                  const Twine &Name = "",
                                  MDNode *FPMathTag = nullptr);
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::CmpInst &SBSI) {
    return SBSI.hash();
  }
  auto getPredicate() const { return cast<llvm::CmpInst>(Val)->getPredicate(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::CmpInst>(Val) && "Expected CmpInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::CmpInst &SBSI) {
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

class BranchInst : public sandboxir::Instruction {
  /// Use Context::createBranchInst(). Don't call the constructor directly.
  BranchInst(llvm::BranchInst *BI, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::Br, Opcode::Br, BI, Ctx) {}
  friend sandboxir::Context; // for sandboxir::BranchInst()
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static sandboxir::BranchInst *create(sandboxir::BasicBlock *IfTrue,
                                       sandboxir::Instruction *InsertBefore,
                                       sandboxir::Context &SBCtx);
  static sandboxir::BranchInst *create(sandboxir::BasicBlock *IfTrue,
                                       sandboxir::BasicBlock *InsertAtEnd,
                                       sandboxir::Context &SBCtx);
  static sandboxir::BranchInst *create(sandboxir::BasicBlock *IfTrue,
                                       sandboxir::BasicBlock *IfFalse,
                                       sandboxir::Value *Cond,
                                       sandboxir::Instruction *InsertBefore,
                                       sandboxir::Context &SBCtx);
  static sandboxir::BranchInst *create(sandboxir::BasicBlock *IfTrue,
                                       sandboxir::BasicBlock *IfFalse,
                                       sandboxir::Value *Cond,
                                       sandboxir::BasicBlock *InsertAtEnd,
                                       sandboxir::Context &SBCtx);
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::BranchInst &BI) {
    return BI.hash();
  }
  bool isUnconditional() const {
    return cast<llvm::BranchInst>(Val)->isUnconditional();
  }
  bool isConditional() const {
    return cast<llvm::BranchInst>(Val)->isConditional();
  }
  sandboxir::Value *getCondition() const;
  void setCondition(sandboxir::Value *V) { setOperand(0, V); }
  unsigned getNumSuccessors() const { return 1 + isConditional(); }
  sandboxir::BasicBlock *getSuccessor(unsigned i) const;
  void setSuccessor(unsigned Idx, sandboxir::BasicBlock *NewSucc);
  void swapSuccessors() { swapOperandsInternal(1, 2); }

private:
  struct LLVMBBToSBBB {
    sandboxir::Context &Ctx;
    LLVMBBToSBBB(sandboxir::Context &Ctx) : Ctx(Ctx) {}
    sandboxir::BasicBlock *operator()(llvm::BasicBlock *BB) const;
  };

  struct ConstLLVMBBToSBBB {
    sandboxir::Context &Ctx;
    ConstLLVMBBToSBBB(sandboxir::Context &Ctx) : Ctx(Ctx) {}
    const sandboxir::BasicBlock *operator()(const llvm::BasicBlock *BB) const;
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
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::BranchInst &BI) {
    BI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class StoreInst : public sandboxir::Instruction {
  /// Use Context::createStoreInst(). Don't call the
  /// constructor directly.
  StoreInst(llvm::StoreInst *SI, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::Store, Opcode::Store, SI, Ctx) {}
  friend sandboxir::Context; // for sandboxir::StoreInst()
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static sandboxir::StoreInst *create(sandboxir::Value *V,
                                      sandboxir::Value *Ptr, MaybeAlign Align,
                                      sandboxir::Instruction *InsertBefore,
                                      sandboxir::Context &SBCtx);
  static sandboxir::StoreInst *create(sandboxir::Value *V,
                                      sandboxir::Value *Ptr, MaybeAlign Align,
                                      sandboxir::BasicBlock *InsertAtEnd,
                                      sandboxir::Context &SBCtx);
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::StoreInst &SBSI) {
    return SBSI.hash();
  }
  sandboxir::Value *getValueOperand() const;
  sandboxir::Value *getPointerOperand() const;
  Align getAlign() const { return cast<llvm::StoreInst>(Val)->getAlign(); }
  bool isSimple() const { return cast<llvm::StoreInst>(Val)->isSimple(); }
  bool isUnordered() const { return cast<llvm::StoreInst>(Val)->isUnordered(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::StoreInst>(Val) && "Expected StoreInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::StoreInst &SBSI) {
    SBSI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class LoadInst : public sandboxir::Instruction {
  /// Use Context::createLoadInst(). Don't call the
  /// constructor directly.
  LoadInst(llvm::LoadInst *LI, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::Load, Opcode::Load, LI, Ctx) {}
  friend sandboxir::Context; // for sandboxir::LoadInst()
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }

  unsigned getNumOfIRInstrs() const final { return 1u; }
  static sandboxir::LoadInst *create(Type *Ty, sandboxir::Value *Ptr,
                                     MaybeAlign Align,
                                     sandboxir::Instruction *InsertBefore,
                                     sandboxir::Context &SBCtx,
                                     const Twine &Name = "");
  static sandboxir::LoadInst *create(Type *Ty, sandboxir::Value *Ptr,
                                     MaybeAlign Align,
                                     sandboxir::BasicBlock *InsertAtEnd,
                                     sandboxir::Context &SBCtx,
                                     const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::LoadInst &SBLI) {
    return SBLI.hash();
  }
  sandboxir::Value *getPointerOperand() const;
  Align getAlign() const { return cast<llvm::LoadInst>(Val)->getAlign(); }
  bool isUnordered() const { return cast<llvm::LoadInst>(Val)->isUnordered(); }
  bool isSimple() const { return cast<llvm::LoadInst>(Val)->isSimple(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::LoadInst>(Val) && "Expected LoadInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::LoadInst &SBLI) {
    SBLI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class CastInst : public sandboxir::Instruction {
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
  CastInst(llvm::CastInst *CI, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::Cast, getCastOpcode(CI->getOpcode()),
                               CI, Ctx) {}
  friend sandboxir::Context; // for SBCastInstruction()
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static sandboxir::Value *create(Type *Ty, Opcode Op,
                                  sandboxir::Value *Operand,
                                  sandboxir::Instruction *InsertBefore,
                                  sandboxir::Context &SBCtx,
                                  const Twine &Name = "");
  static sandboxir::Value *create(Type *Ty, Opcode Op,
                                  sandboxir::Value *Operand,
                                  sandboxir::BasicBlock *InsertAtEnd,
                                  sandboxir::Context &SBCtx,
                                  const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::CastInst &SBCI) {
    return SBCI.hash();
  }
  llvm::Instruction::CastOps getOpcode() const {
    return cast<llvm::CastInst>(Val)->getOpcode();
  }
  Type *getSrcTy() const { return cast<llvm::CastInst>(Val)->getSrcTy(); }
  Type *getDestTy() const { return cast<llvm::CastInst>(Val)->getDestTy(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::CastInst>(Val) && "Expected CastInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::CastInst &SBCI) {
    SBCI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class PHINode : public sandboxir::Instruction {
  /// Use sandboxir::Context::createPHINode(). Don't call the
  /// constructor directly.
  PHINode(llvm::PHINode *PHI, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::PHI, Opcode::PHI, PHI, Ctx) {}
  friend sandboxir::Context; // for SBPHINode()
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static sandboxir::Value *create(Type *Ty, unsigned NumReservedValues,
                                  sandboxir::Instruction *InsertBefore,
                                  sandboxir::Context &SBCtx,
                                  const Twine &Name = "");
  static sandboxir::Value *create(Type *Ty, unsigned NumReservedValues,
                                  sandboxir::BasicBlock *InsertAtEnd,
                                  sandboxir::Context &SBCtx,
                                  const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::PHINode &SBCI) {
    return SBCI.hash();
  }
  Type *getSrcTy() const { return cast<llvm::CastInst>(Val)->getSrcTy(); }
  Type *getDestTy() const { return cast<llvm::CastInst>(Val)->getDestTy(); }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::PHINode>(Val) && "Expected PHINode!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::PHINode &SBCI) {
    SBCI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class SelectInst : public sandboxir::Instruction {
  /// Use sandboxir::Context::createSelectInst(). Don't call the
  /// constructor directly.
  SelectInst(llvm::SelectInst *CI, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::Select, Opcode::Select, CI, Ctx) {}
  friend sandboxir::Context; // for sandboxir::SelectInst()
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static sandboxir::Value *
  create(sandboxir::Value *Cond, sandboxir::Value *True,
         sandboxir::Value *False, sandboxir::Instruction *InsertBefore,
         sandboxir::Context &SBCtx, const Twine &Name = "");
  static sandboxir::Value *
  create(sandboxir::Value *Cond, sandboxir::Value *True,
         sandboxir::Value *False, sandboxir::BasicBlock *InsertAtEnd,
         sandboxir::Context &SBCtx, const Twine &Name = "");
  sandboxir::Value *getCondition() { return getOperand(0); }
  sandboxir::Value *getTrueValue() { return getOperand(1); }
  sandboxir::Value *getFalseValue() { return getOperand(2); }

  void setCondition(sandboxir::Value *New) { setOperand(0, New); }
  void setTrueValue(sandboxir::Value *New) { setOperand(1, New); }
  void setFalseValue(sandboxir::Value *New) { setOperand(2, New); }
  void swapValues() { cast<llvm::SelectInst>(Val)->swapValues(); }
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::SelectInst &SBSI) {
    return SBSI.hash();
  }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::SelectInst>(Val) && "Expected SelectInst!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::SelectInst &SBSI) {
    SBSI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class BinaryOperator : public sandboxir::Instruction {
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
  /// Use sandboxir::Context::createBinaryOperator(). Don't call the
  /// constructor directly.
  BinaryOperator(llvm::BinaryOperator *BO, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::BinOp, getBinOpOpcode(BO->getOpcode()),
                               BO, Ctx) {}
  friend sandboxir::Context; // for sandboxir::SelectInst()
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static sandboxir::Value *create(sandboxir::Instruction::Opcode Op,
                                  sandboxir::Value *LHS, sandboxir::Value *RHS,
                                  sandboxir::Instruction *InsertBefore,
                                  sandboxir::Context &SBCtx,
                                  const Twine &Name = "");
  static sandboxir::Value *create(sandboxir::Instruction::Opcode Op,
                                  sandboxir::Value *LHS, sandboxir::Value *RHS,
                                  sandboxir::BasicBlock *InsertAtEnd,
                                  sandboxir::Context &SBCtx,
                                  const Twine &Name = "");
  static sandboxir::Value *
  createWithCopiedFlags(sandboxir::Instruction::Opcode Op,
                        sandboxir::Value *LHS, sandboxir::Value *RHS,
                        sandboxir::Value *CopyFrom,
                        sandboxir::Instruction *InsertBefore,
                        sandboxir::Context &SBCtx, const Twine &Name = "");
  static sandboxir::Value *
  createWithCopiedFlags(sandboxir::Instruction::Opcode Op,
                        sandboxir::Value *LHS, sandboxir::Value *RHS,
                        sandboxir::Value *CopyFrom,
                        sandboxir::BasicBlock *InsertAtEnd,
                        sandboxir::Context &SBCtx, const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From);
  void swapOperands() { swapOperandsInternal(0, 1); }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::BinaryOperator &SBBO) {
    return SBBO.hash();
  }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::BinaryOperator>(Val) && "Expected BinaryOperator!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::BinaryOperator &SBBO) {
    SBBO.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class UnaryOperator : public sandboxir::Instruction {
  static Opcode getUnaryOpcode(llvm::Instruction::UnaryOps UnOp) {
    switch (UnOp) {
    case llvm::Instruction::FNeg:
      return Opcode::FNeg;
    case llvm::Instruction::UnaryOpsEnd:
      llvm_unreachable("Bad UnOp!");
    }
    llvm_unreachable("Unhandled UnOp!");
  }
  /// Use sandboxir::Context::createUnaryOperator(). Don't call the
  /// constructor directly.
  UnaryOperator(llvm::UnaryOperator *UO, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::UnOp, getUnaryOpcode(UO->getOpcode()),
                               UO, Ctx) {}
  friend sandboxir::Context; // for sandboxir::SelectInst()
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  static sandboxir::Value *
  createWithCopiedFlags(sandboxir::Instruction::Opcode Op,
                        sandboxir::Value *OpV, sandboxir::Value *CopyFrom,
                        sandboxir::Instruction *InsertBefore,
                        sandboxir::Context &SBCtx, const Twine &Name = "");
  static sandboxir::Value *
  createWithCopiedFlags(sandboxir::Instruction::Opcode Op,
                        sandboxir::Value *OpV, sandboxir::Value *CopyFrom,
                        sandboxir::BasicBlock *InsertAtEnd,
                        sandboxir::Context &SBCtx, const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From);
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::UnaryOperator &SBUO) {
    return SBUO.hash();
  }
#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::UnaryOperator>(Val) && "Expected UnaryOperator!");
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::UnaryOperator &SBUO) {
    SBUO.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class InsertElementInst : public sandboxir::Instruction {
  /// Use sandboxir::Context::createInsertElementInst(). Don't call
  /// the constructor directly.
  InsertElementInst(llvm::Instruction *I, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::Insert, Opcode::Insert, I, Ctx) {}
  InsertElementInst(ClassID SubclassID, llvm::Instruction *I,
                    sandboxir::Context &Ctx)
      : sandboxir::Instruction(SubclassID, Opcode::Insert, I, Ctx) {}
  friend class sandboxir::Context; // For accessing the constructor in
                                   // create*()
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  static sandboxir::Value *
  create(sandboxir::Value *Vec, sandboxir::Value *NewElt, sandboxir::Value *Idx,
         sandboxir::Instruction *InsertBefore, sandboxir::Context &SBCtx,
         const Twine &Name = "");
  static sandboxir::Value *
  create(sandboxir::Value *Vec, sandboxir::Value *NewElt, sandboxir::Value *Idx,
         sandboxir::BasicBlock *InsertAtEnd, sandboxir::Context &SBCtx,
         const Twine &Name = "");
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Insert;
  }
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::InsertElementInst &SBGI) {
    return SBGI.hash();
  }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::InsertElementInst &SBGI) {
    SBGI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class ExtractElementInst : public sandboxir::Instruction {
  /// Use sandboxir::Context::createExtractElementInst(). Don't call
  /// the constructor directly.
  ExtractElementInst(llvm::Instruction *I, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::Extract, Opcode::Extract, I, Ctx) {}
  ExtractElementInst(ClassID SubclassID, llvm::Instruction *I,
                     sandboxir::Context &Ctx)
      : sandboxir::Instruction(SubclassID, Opcode::Extract, I, Ctx) {}
  friend class sandboxir::Context; // For accessing the constructor in
                                   // create*()
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  static sandboxir::Value *create(sandboxir::Value *Vec, sandboxir::Value *Idx,
                                  sandboxir::Instruction *InsertBefore,
                                  sandboxir::Context &SBCtx,
                                  const Twine &Name = "");
  static sandboxir::Value *create(sandboxir::Value *Vec, sandboxir::Value *Idx,
                                  sandboxir::BasicBlock *InsertAtEnd,
                                  sandboxir::Context &SBCtx,
                                  const Twine &Name = "");
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Extract;
  }
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::ExtractElementInst &SBGI) {
    return SBGI.hash();
  }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::ExtractElementInst &SBGI) {
    SBGI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class ShuffleVectorInst : public sandboxir::Instruction {
  /// Use sandboxir::Context::createShuffleVectorInst(). Don't call
  /// the constructor directly.
  ShuffleVectorInst(llvm::Instruction *I, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::ShuffleVec, Opcode::ShuffleVec, I,
                               Ctx) {}
  ShuffleVectorInst(ClassID SubclassID, llvm::Instruction *I,
                    sandboxir::Context &Ctx)
      : sandboxir::Instruction(SubclassID, Opcode::ShuffleVec, I, Ctx) {}
  friend class sandboxir::Context; // For accessing the constructor in
                                   // create*()
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  static sandboxir::Value *create(sandboxir::Value *V1, sandboxir::Value *V2,
                                  sandboxir::Value *Mask,
                                  sandboxir::Instruction *InsertBefore,
                                  sandboxir::Context &SBCtx,
                                  const Twine &Name = "");
  static sandboxir::Value *create(sandboxir::Value *V1, sandboxir::Value *V2,
                                  sandboxir::Value *Mask,
                                  sandboxir::BasicBlock *InsertAtEnd,
                                  sandboxir::Context &SBCtx,
                                  const Twine &Name = "");
  static sandboxir::Value *create(sandboxir::Value *V1, sandboxir::Value *V2,
                                  ArrayRef<int> Mask,
                                  sandboxir::Instruction *InsertBefore,
                                  sandboxir::Context &SBCtx,
                                  const Twine &Name = "");
  static sandboxir::Value *create(sandboxir::Value *V1, sandboxir::Value *V2,
                                  ArrayRef<int> Mask,
                                  sandboxir::BasicBlock *InsertAtEnd,
                                  sandboxir::Context &SBCtx,
                                  const Twine &Name = "");
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::ShuffleVec;
  }
  SmallVector<int> getShuffleMask() const {
    SmallVector<int> Mask;
    cast<llvm::ShuffleVectorInst>(Val)->getShuffleMask(Mask);
    return Mask;
  }
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::ShuffleVectorInst &SBGI) {
    return SBGI.hash();
  }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::ShuffleVectorInst &SBGI) {
    SBGI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class RetInst : public sandboxir::Instruction {
  /// Use sandboxir::Context::createRetInst(). Don't call the
  /// constructor directly.
  RetInst(llvm::Instruction *I, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::Ret, Opcode::Ret, I, Ctx) {}
  RetInst(ClassID SubclassID, llvm::Instruction *I, sandboxir::Context &Ctx)
      : sandboxir::Instruction(SubclassID, Opcode::Ret, I, Ctx) {}
  friend class sandboxir::Context; // For accessing the constructor in
                                   // create*()
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  static sandboxir::Value *create(sandboxir::Value *RetVal,
                                  sandboxir::Instruction *InsertBefore,
                                  sandboxir::Context &SBCtx);
  static sandboxir::Value *create(sandboxir::Value *RetVal,
                                  sandboxir::BasicBlock *InsertAtEnd,
                                  sandboxir::Context &SBCtx);
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Ret;
  }
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  /// \Returns null if there is no return value.
  sandboxir::Value *getReturnValue() const;
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::RetInst &I) { return I.hash(); }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS, const sandboxir::RetInst &I) {
    I.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class CallInst : public sandboxir::Instruction {
  /// Use sandboxir::Context::createCallInst(). Don't call the
  /// constructor directly.
  CallInst(llvm::Instruction *I, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::Call, Opcode::Call, I, Ctx) {}
  CallInst(ClassID SubclassID, llvm::Instruction *I, sandboxir::Context &Ctx)
      : sandboxir::Instruction(SubclassID, Opcode::Call, I, Ctx) {}
  friend class sandboxir::Context; // For accessing the constructor in
                                   // create*()
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  static sandboxir::CallInst *create(FunctionType *FTy, sandboxir::Value *Func,
                                     ArrayRef<sandboxir::Value *> Args,
                                     sandboxir::BBIterator WhereIt,
                                     sandboxir::BasicBlock *WhereBB,
                                     sandboxir::Context &SBCtx,
                                     const Twine &NameStr = "");
  static sandboxir::CallInst *create(FunctionType *FTy, sandboxir::Value *Func,
                                     ArrayRef<sandboxir::Value *> Args,
                                     sandboxir::Instruction *InsertBefore,
                                     sandboxir::Context &SBCtx,
                                     const Twine &NameStr = "");
  static sandboxir::CallInst *create(FunctionType *FTy, sandboxir::Value *Func,
                                     ArrayRef<sandboxir::Value *> Args,
                                     sandboxir::BasicBlock *InsertAtEnd,
                                     sandboxir::Context &SBCtx,
                                     const Twine &NameStr = "");

  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Call;
  }
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::CallInst &I) { return I.hash(); }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::CallInst &I) {
    I.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class GetElementPtrInst : public sandboxir::Instruction {
  /// Use sandboxir::Context::createGetElementPtrInst(). Don't call
  /// the constructor directly.
  GetElementPtrInst(llvm::Instruction *I, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::GetElementPtr, Opcode::GetElementPtr, I,
                               Ctx) {}
  GetElementPtrInst(ClassID SubclassID, llvm::Instruction *I,
                    sandboxir::Context &Ctx)
      : sandboxir::Instruction(SubclassID, Opcode::GetElementPtr, I, Ctx) {}
  friend class sandboxir::Context; // For accessing the constructor in
                                   // create*()
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  static sandboxir::Value *
  create(Type *Ty, sandboxir::Value *Ptr, ArrayRef<sandboxir::Value *> IdxList,
         sandboxir::BBIterator WhereIt, sandboxir::BasicBlock *WhereBB,
         sandboxir::Context &SBCtx, const Twine &NameStr = "");
  static sandboxir::Value *create(Type *Ty, sandboxir::Value *Ptr,
                                  ArrayRef<sandboxir::Value *> IdxList,
                                  sandboxir::Instruction *InsertBefore,
                                  sandboxir::Context &SBCtx,
                                  const Twine &NameStr = "");
  static sandboxir::Value *create(Type *Ty, sandboxir::Value *Ptr,
                                  ArrayRef<sandboxir::Value *> IdxList,
                                  sandboxir::BasicBlock *InsertAtEnd,
                                  sandboxir::Context &SBCtx,
                                  const Twine &NameStr = "");

  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::GetElementPtr;
  }
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::GetElementPtrInst &I) {
    return I.hash();
  }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::GetElementPtrInst &I) {
    I.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class OpaqueInst : public sandboxir::Instruction {
  /// Use sandboxir::Context::createOpaqueInstruction(). Don't call the
  /// constructor directly.
  OpaqueInst(llvm::Instruction *I, sandboxir::Context &Ctx)
      : sandboxir::Instruction(ClassID::Opaque, Opcode::Opaque, I, Ctx) {}
  OpaqueInst(ClassID SubclassID, llvm::Instruction *I, sandboxir::Context &Ctx)
      : sandboxir::Instruction(SubclassID, Opcode::Opaque, I, Ctx) {}
  friend class sandboxir::BasicBlock;
  friend class sandboxir::Context; // For creating SB constants.
  void detachExtras() final {}
  sandboxir::Use getOperandUseInternal(unsigned OpIdx,
                                       bool Verify) const final {
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
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Opaque;
  }
  unsigned getUseOperandNo(const sandboxir::Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
  hash_code hash() const override { return hashCommon(); }
  friend hash_code hash_value(const sandboxir::OpaqueInst &SBGI) {
    return SBGI.hash();
  }
#ifndef NDEBUG
  void verify() const final {}
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::OpaqueInst &SBGI) {
    SBGI.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const override;
  void dumpVerbose(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dumpVerbose() const override;
#endif
};

class BasicBlock : public sandboxir::Value {
  /// Assigns an ordering number to instructions in the block. This is used for
  /// quick comesBefore() lookups or for a rough estimate of distance.
  DenseMap<sandboxir::Instruction *, int64_t> InstrNumberMap;
  /// When we first assign numbers to instructions we use this step. This allows
  /// us to insert new instructions in between without renumbering the whole
  /// block.
public:
  static constexpr const int64_t InstrNumberingStep = 64;

private:
  void renumberInstructions();
  /// This is called after \p I has been inserted into its parent block.
  void assignInstrNumber(sandboxir::Instruction *I);
  void removeInstrNumber(sandboxir::Instruction *I);
  friend void sandboxir::Instruction::moveBefore(sandboxir::BasicBlock &,
                                                 const sandboxir::BBIterator &);
  friend void sandboxir::Instruction::insertBefore(sandboxir::Instruction *);
  friend void sandboxir::Instruction::insertInto(sandboxir::BasicBlock *,
                                                 const sandboxir::BBIterator &);
  friend void sandboxir::Instruction::eraseFromParent();
  friend void sandboxir::Instruction::removeFromParent();

public:
  int64_t getInstrNumber(const sandboxir::Instruction *I) const {
    auto It = InstrNumberMap.find(I);
    assert(It != InstrNumberMap.end() && "Missing InstrNumber!");
    return It->second;
  }
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From);
  /// Builds a graph that contains all values in \p BB in their original form
  /// i.e., no vectorization is taking place here.
  void buildSBBasicBlockFromIR(llvm::BasicBlock *BB);
  /// \Returns the iterator to the first non-PHI instruction.
  sandboxir::BBIterator getFirstNonPHIIt();

private:
  friend void sandboxir::Value::replaceUsesWithIf(
      sandboxir::Value *,
      llvm::function_ref<bool(sandboxir::User *,
                              unsigned)>); // for ChangeTracker.
  friend void sandboxir::Value::replaceAllUsesWith(
      sandboxir::Value *); // for ChangeTracker.
  friend void
  sandboxir::User::setOperand(unsigned,
                              sandboxir::Value *); // for ChangeTracker

  /// Detach sandboxir::BasicBlock from the underlying BB. This is called by
  /// the destructor.
  void detach();
  /// Use sandboxir::Context::createBasicBlock().
  BasicBlock(llvm::BasicBlock *BB, sandboxir::Context &SBCtx);
  friend class sandboxir::Context; // For createBasicBlock().
  friend class sandboxir::BasicBlockAttorney;

public:
  ~BasicBlock();
  sandboxir::Function *getParent() const;
  /// Detaches the block and its instructions from LLVM IR.
  void detachFromLLVMIR();
  using iterator = sandboxir::BBIterator;
  iterator begin() const;
  iterator end() const {
    auto *BB = cast<llvm::BasicBlock>(Val);
    return iterator(BB, BB->end(), &Ctx);
  }
  sandboxir::Context &getContext() const { return Ctx; }
  SandboxIRTracker &getTracker();
  sandboxir::Instruction *getTerminator() const;
  auto LLVMSize() const { return cast<llvm::BasicBlock>(Val)->size(); }

  hash_code hash() const final {
    return hash_combine(sandboxir::Value::hashCommon(),
                        hash_combine_range(begin(), end()));
  }
  friend hash_code hash_value(const sandboxir::BasicBlock &SBBB) {
    return SBBB.hash();
  }

  bool empty() const { return begin() == end(); }
  sandboxir::Instruction &front() const;
  sandboxir::Instruction &back() const;

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
  // void verifyIR(const DmpVector<sandboxir::Value *> &Instrs) const;
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const sandboxir::BasicBlock &SBBB) {
    SBBB.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
  void dumpVerbose(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dumpVerbose() const final;
  /// Dump a range of instructions near \p SBV.
  LLVM_DUMP_METHOD void dumpInstrs(sandboxir::Value *SBV, int Num) const;
#endif
};

/// A client-attorney class for sandboxir::BasicBlock that allows access to
/// selected private members.
class BasicBlockAttorney {
  static llvm::BasicBlock *getBB(sandboxir::BasicBlock *SBBB) {
    return cast<llvm::BasicBlock>(SBBB->Val);
  }
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS;
#include "llvm/SandboxIR/SandboxIRValues.def"
};

class Context {
public:
  using RemoveCBTy = std::function<void(sandboxir::Instruction *)>;
  using InsertCBTy = std::function<void(sandboxir::Instruction *)>;
  using MoveCBTy =
      std::function<void(sandboxir::Instruction *, sandboxir::BasicBlock &,
                         const sandboxir::BBIterator &)>;

  friend class sandboxir::PackInst; // For detachValue()

protected:
  LLVMContext &LLVMCtx;
  SandboxIRTracker ChangeTracker;
  IRBuilder<ConstantFolder> LLVMIRBuilder;

  /// Vector of callbacks called when an IR Instruction is about to get erased.
  SmallVector<std::unique_ptr<RemoveCBTy>> RemoveInstrCallbacks;
  DenseMap<sandboxir::BasicBlock *, SmallVector<std::unique_ptr<RemoveCBTy>>>
      RemoveInstrCallbacksBB;
  SmallVector<std::unique_ptr<InsertCBTy>> InsertInstrCallbacks;
  DenseMap<sandboxir::BasicBlock *, SmallVector<std::unique_ptr<InsertCBTy>>>
      InsertInstrCallbacksBB;
  SmallVector<std::unique_ptr<MoveCBTy>> MoveInstrCallbacks;
  DenseMap<sandboxir::BasicBlock *, SmallVector<std::unique_ptr<MoveCBTy>>>
      MoveInstrCallbacksBB;

  /// Maps LLVM Value to the corresponding sandboxir::Value. Owns all
  /// SandboxIR objects.
  DenseMap<llvm::Value *, std::unique_ptr<sandboxir::Value>>
      LLVMValueToSBValueMap;
  /// In SandboxIR some instructions correspond to multiple IR Instructions,
  /// like Packs. For such cases we map the IR instructions to the key used in
  /// LLVMValueToSBValueMap.
  DenseMap<llvm::Value *, llvm::Value *> MultiInstrMap;

  friend sandboxir::BasicBlock::~BasicBlock(); // For removing the
                                               // scheduler.
  /// This is true during quickFlush(). It helps with some assertions that would
  /// otherwise trigger.
  bool InQuickFlush = false;

  /// This is true during the initial creation of SandboxIR. This helps select
  /// different code paths during/after creation of SandboxIR.
  bool DontNumberInstrs = false;

  friend class sandboxir::ContextAttorney; // for setScheduler(),
                                           // clearScheduler()
  /// Removes \p V from the maps and returns the unique_ptr.
  std::unique_ptr<sandboxir::Value> detachValue(llvm::Value *V);

  friend void sandboxir::Instruction::eraseFromParent();
  friend void sandboxir::Instruction::removeFromParent();
  friend void sandboxir::Instruction::moveBefore(sandboxir::BasicBlock &,
                                                 const sandboxir::BBIterator &);

  void runRemoveInstrCallbacks(sandboxir::Instruction *I);
  void runInsertInstrCallbacks(sandboxir::Instruction *I);
  void runMoveInstrCallbacks(sandboxir::Instruction *I,
                             sandboxir::BasicBlock &SBBB,
                             const sandboxir::BBIterator &WhereIt);

  virtual sandboxir::Value *
  createValueFromExtractElement(llvm::ExtractElementInst *ExtractI, int Depth) {
    return getOrCreateExtractElementInst(ExtractI);
  }
  sandboxir::Value *
  getValueFromExtractElement(llvm::ExtractElementInst *ExtractI) const;
  sandboxir::Value *
  getOrCreateValueFromExtractElement(llvm::ExtractElementInst *ExtractI,
                                     int Depth);

  virtual sandboxir::Value *
  createValueFromInsertElement(llvm::InsertElementInst *InsertI, int Depth) {
    return getOrCreateInsertElementInst(InsertI);
  }
  sandboxir::Value *
  getValueFromInsertElement(llvm::InsertElementInst *InsertI) const;
  sandboxir::Value *
  getOrCreateValueFromInsertElement(llvm::InsertElementInst *InsertI,
                                    int Depth);

  virtual sandboxir::Value *
  createValueFromShuffleVector(llvm::ShuffleVectorInst *ShuffleI, int Depth) {
    return getOrCreateShuffleVectorInst(ShuffleI);
  }

  sandboxir::Value *
  getValueFromShuffleVector(llvm::ShuffleVectorInst *ShuffleI) const;
  sandboxir::Value *
  getOrCreateValueFromShuffleVector(llvm::ShuffleVectorInst *ShuffleI,
                                    int Depth);

  /// This runs right after \p SBB has been created.
  virtual void createdBasicBlock(sandboxir::BasicBlock &BB) {}

#if !defined NDEBUG && defined SBVEC_EXPENSIVE_CHECKS
  /// Runs right after an instruction has moved in \p BB. This is used for
  /// testing the DAG and Scheduler by sandboxir::SBVecContext.
  virtual void afterMoveInstrHook(sandboxir::BasicBlock &BB) {}
#endif
  /// This is called by the sandboxir::BasicBlock's destructor.
  virtual void destroyingBB(sandboxir::BasicBlock &BB) {}

  /// Helper for avoiding recursion loop when creating SBConstants.
  SmallDenseSet<llvm::Constant *, 8> VisitedConstants;
  sandboxir::Value *getOrCreateValueInternal(llvm::Value *V, int Depth,
                                             llvm::User *U = nullptr);

public:
  Context(LLVMContext &LLVMCtx);
  virtual ~Context() {}
  SandboxIRTracker &getTracker() { return ChangeTracker; }
  auto &getLLVMIRBuilder() { return LLVMIRBuilder; }
  size_t getNumValues() const {
    return LLVMValueToSBValueMap.size() + MultiInstrMap.size();
  }

  /// Remove \p SBV from all SandboxIR maps and stop owning it. This effectively
  /// detaches \p SBV from the underlying IR.
  std::unique_ptr<sandboxir::Value> detach(sandboxir::Value *SBV);
  sandboxir::Value *registerValue(std::unique_ptr<sandboxir::Value> &&SBVPtr);

  sandboxir::Value *getValue(llvm::Value *V) const;
  const sandboxir::Value *getValue(const llvm::Value *V) const {
    return getValue(const_cast<llvm::Value *>(V));
  }

  sandboxir::Constant *getConstant(llvm::Constant *C) const;
  sandboxir::Constant *getOrCreateConstant(llvm::Constant *C);

  sandboxir::Value *getOrCreateValue(llvm::Value *V);

  /// Helper function called when we create sandboxir::Instructions that
  /// create new constant operands. It goes through V's operands and creates
  /// sandboxir::Constants.
  void createMissingConstantOperands(llvm::Value *V);

  // Arguments
  sandboxir::Argument *getArgument(llvm::Argument *Arg) const;
  sandboxir::Argument *createArgument(llvm::Argument *Arg);
  sandboxir::Argument *getOrCreateArgument(llvm::Argument *Arg);

  // InsertElementInstruction
  sandboxir::InsertElementInst *
  getInsertElementInst(llvm::InsertElementInst *I) const;
  sandboxir::InsertElementInst *
  createInsertElementInst(llvm::InsertElementInst *I);
  sandboxir::InsertElementInst *
  getOrCreateInsertElementInst(llvm::InsertElementInst *I);

  // InsertElementInstruction
  sandboxir::ExtractElementInst *
  getExtractElementInst(llvm::ExtractElementInst *I) const;
  sandboxir::ExtractElementInst *
  createExtractElementInst(llvm::ExtractElementInst *I);
  sandboxir::ExtractElementInst *
  getOrCreateExtractElementInst(llvm::ExtractElementInst *I);

  // ShuffleVectorInstruction
  sandboxir::ShuffleVectorInst *
  getShuffleVectorInst(llvm::ShuffleVectorInst *I) const;
  sandboxir::ShuffleVectorInst *
  createShuffleVectorInst(llvm::ShuffleVectorInst *I);
  sandboxir::ShuffleVectorInst *
  getOrCreateShuffleVectorInst(llvm::ShuffleVectorInst *I);

  // Return
  sandboxir::RetInst *getRetInst(llvm::ReturnInst *I) const;
  sandboxir::RetInst *createRetInst(llvm::ReturnInst *I);
  sandboxir::RetInst *getOrCreateRetInst(llvm::ReturnInst *I);

  // Call
  sandboxir::CallInst *getCallInst(llvm::CallInst *I) const;
  sandboxir::CallInst *createCallInst(llvm::CallInst *I);
  sandboxir::CallInst *getOrCreateCallInst(llvm::CallInst *I);

  // GEP
  sandboxir::GetElementPtrInst *
  getGetElementPtrInst(llvm::GetElementPtrInst *I) const;
  sandboxir::GetElementPtrInst *
  createGetElementPtrInst(llvm::GetElementPtrInst *I);
  sandboxir::GetElementPtrInst *
  getOrCreateGetElementPtrInst(llvm::GetElementPtrInst *I);

  // OpaqueInstr
  sandboxir::OpaqueInst *getOpaqueInstruction(llvm::Instruction *I) const;
  sandboxir::OpaqueInst *createOpaqueInstruction(llvm::Instruction *I);
  sandboxir::OpaqueInst *getOrCreateOpaqueInstruction(llvm::Instruction *I);

  // BranchInst
  sandboxir::BranchInst *getBranchInst(llvm::BranchInst *BI) const;
  sandboxir::BranchInst *createBranchInst(llvm::BranchInst *BI);
  sandboxir::BranchInst *getOrCreateBranchInst(llvm::BranchInst *BI);

  // Store
  sandboxir::StoreInst *getStoreInst(llvm::StoreInst *SI) const;
  sandboxir::StoreInst *createStoreInst(llvm::StoreInst *SI);
  sandboxir::StoreInst *getOrCreateStoreInst(llvm::StoreInst *SI);

  // Load
  sandboxir::LoadInst *getLoadInst(llvm::LoadInst *LI) const;
  sandboxir::LoadInst *createLoadInst(llvm::LoadInst *LI);
  sandboxir::LoadInst *getOrCreateLoadInst(llvm::LoadInst *LI);

  // Cast
  sandboxir::CastInst *getCastInst(llvm::CastInst *CI) const;
  sandboxir::CastInst *createCastInst(llvm::CastInst *CI);
  sandboxir::CastInst *getOrCreateCastInst(llvm::CastInst *CI);

  // PHI
  sandboxir::PHINode *getPHINode(llvm::PHINode *PHI) const;
  sandboxir::PHINode *createPHINode(llvm::PHINode *PHI);
  sandboxir::PHINode *getOrCreatePHINode(llvm::PHINode *PHI);

  // Select
  sandboxir::SelectInst *getSelectInst(llvm::SelectInst *SI) const;
  sandboxir::SelectInst *createSelectInst(llvm::SelectInst *SI);
  sandboxir::SelectInst *getOrCreateSelectInst(llvm::SelectInst *SI);

  // BinaryOperator
  sandboxir::BinaryOperator *getBinaryOperator(llvm::BinaryOperator *BO) const;
  sandboxir::BinaryOperator *createBinaryOperator(llvm::BinaryOperator *BO);
  sandboxir::BinaryOperator *
  getOrCreateBinaryOperator(llvm::BinaryOperator *BO);

  // UnaryOperator
  sandboxir::UnaryOperator *getUnaryOperator(llvm::UnaryOperator *UO) const;
  sandboxir::UnaryOperator *createUnaryOperator(llvm::UnaryOperator *UO);
  sandboxir::UnaryOperator *getOrCreateUnaryOperator(llvm::UnaryOperator *UO);

  // Cmp
  sandboxir::CmpInst *getCmpInst(llvm::CmpInst *CI) const;
  sandboxir::CmpInst *createCmpInst(llvm::CmpInst *CI);
  sandboxir::CmpInst *getOrCreateCmpInst(llvm::CmpInst *CI);

  // Block
  sandboxir::BasicBlock *getBasicBlock(llvm::BasicBlock *BB) const;
  sandboxir::BasicBlock *createBasicBlock(llvm::BasicBlock *BB);

  // Function
  sandboxir::Function *getFunction(llvm::Function *F) const;
  sandboxir::Function *createFunction(llvm::Function *F, bool CreateBBs = true);

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
  RemoveCBTy *registerRemoveInstrCallbackBB(sandboxir::BasicBlock &BB,
                                            RemoveCBTy CB);
  void unregisterRemoveInstrCallbackBB(sandboxir::BasicBlock &BB,
                                       RemoveCBTy *CB);

  InsertCBTy *registerInsertInstrCallbackBB(sandboxir::BasicBlock &BB,
                                            InsertCBTy CB);
  void unregisterInsertInstrCallbackBB(sandboxir::BasicBlock &BB,
                                       InsertCBTy *CB);

  MoveCBTy *registerMoveInstrCallbackBB(sandboxir::BasicBlock &BB, MoveCBTy CB);
  void unregisterMoveInstrCallbackBB(sandboxir::BasicBlock &BB, MoveCBTy *CB);

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
  friend class sandboxir::ContextAttorney;
};

/// A client-attorney class for sandboxir::Context.
class ContextAttorney {
  friend class sandboxir::Region;
  friend class sandboxir::RegionBuilderFromMD;

public:
  static LLVMContext &getLLVMContext(sandboxir::Context &Ctx) {
    return Ctx.LLVMCtx;
  }
};

class Function : public sandboxir::Value {
  llvm::Function *getFunction() const { return cast<llvm::Function>(Val); }

public:
  Function(llvm::Function *F, sandboxir::Context &Ctx)
      : sandboxir::Value(ClassID::Function, F, Ctx) {}
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Function;
  }

  /// Iterates over sandboxir::BasicBlocks
  class iterator {
    llvm::Function::iterator It;
#ifndef NDEBUG
    llvm::Function *F;
#endif
    sandboxir::Context *Ctx;

  public:
    using difference_type = std::ptrdiff_t;
    using value_type = sandboxir::BasicBlock;
    using pointer = sandboxir::BasicBlock *;
    using reference = value_type &;
    using iterator_category = std::bidirectional_iterator_tag;

#ifndef NDEBUG
    iterator() : F(nullptr), Ctx(nullptr) {}
    iterator(llvm::Function::iterator It, llvm::Function *F,
             sandboxir::Context &Ctx)
        : It(It), F(F), Ctx(&Ctx) {}
#else
    iterator() : Ctx(nullptr) {}
    iterator(llvm::Function::iterator It, sandboxir::Context &Ctx)
        : It(It), Ctx(&Ctx) {}
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
      return *cast<sandboxir::BasicBlock>(Ctx->getValue(&*It));
    }
    const sandboxir::BasicBlock &operator*() const {
      assert(It != F->end() && "Dereferencing end()!");
      return *cast<sandboxir::BasicBlock>(Ctx->getValue(&*It));
    }
  };

  sandboxir::Argument *getArg(unsigned Idx) const {
    llvm::Argument *Arg = getFunction()->getArg(Idx);
    return cast<sandboxir::Argument>(Ctx.getValue(Arg));
  }

  size_t arg_size() const { return getFunction()->arg_size(); }
  bool arg_empty() const { return getFunction()->arg_empty(); }

  struct LLVMArgToSBArgConst {
    sandboxir::Context &Ctx;
    LLVMArgToSBArgConst(sandboxir::Context &Ctx) : Ctx(Ctx) {}
    const sandboxir::Argument &operator()(const llvm::Argument &Arg) const {
      return *cast<sandboxir::Argument>(
          Ctx.getValue(const_cast<llvm::Argument *>(&Arg)));
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
    sandboxir::Context &Ctx;
    LLVMArgToSBArg(sandboxir::Context &Ctx) : Ctx(Ctx) {}
    sandboxir::Argument &operator()(llvm::Argument &Arg) const {
      return *cast<sandboxir::Argument>(Ctx.getValue(&Arg));
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

  sandboxir::BasicBlock &getEntryBlock() const {
    llvm::BasicBlock &EntryBB = getFunction()->getEntryBlock();
    return *cast<sandboxir::BasicBlock>(Ctx.getValue(&EntryBB));
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
    auto Hash = hash_combine(sandboxir::Value::hashCommon(),
                             hash_combine_range(begin(), end()));
    for (auto ArgIdx : seq<unsigned>(0, arg_size()))
      Hash = hash_combine(Hash, getArg(ArgIdx));
    return Hash;
  }
  friend hash_code hash_value(const sandboxir::Function &SBF) {
    return SBF.hash();
  }

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

using sb_succ_iterator =
    SuccIterator<sandboxir::Instruction, sandboxir::BasicBlock>;
using const_sb_succ_iterator =
    SuccIterator<const sandboxir::Instruction, const sandboxir::BasicBlock>;
using sb_succ_range = iterator_range<sb_succ_iterator>;
using const_sb_succ_range = iterator_range<const_sb_succ_iterator>;

inline sb_succ_iterator succ_begin(sandboxir::Instruction *I) {
  return sb_succ_iterator(I);
}
inline const_sb_succ_iterator succ_begin(const sandboxir::Instruction *I) {
  return const_sb_succ_iterator(I);
}
inline sb_succ_iterator succ_end(sandboxir::Instruction *I) {
  return sb_succ_iterator(I, true);
}
inline const_sb_succ_iterator succ_end(const sandboxir::Instruction *I) {
  return const_sb_succ_iterator(I, true);
}
inline bool succ_empty(const sandboxir::Instruction *I) {
  return succ_begin(I) == succ_end(I);
}
inline unsigned succ_size(const sandboxir::Instruction *I) {
  return std::distance(succ_begin(I), succ_end(I));
}
inline sb_succ_range successors(sandboxir::Instruction *I) {
  return sb_succ_range(succ_begin(I), succ_end(I));
}
inline const_sb_succ_range successors(const sandboxir::Instruction *I) {
  return const_sb_succ_range(succ_begin(I), succ_end(I));
}

inline sb_succ_iterator succ_begin(sandboxir::BasicBlock *BB) {
  return sb_succ_iterator(BB->getTerminator());
}
inline const_sb_succ_iterator succ_begin(const sandboxir::BasicBlock *BB) {
  return const_sb_succ_iterator(BB->getTerminator());
}
inline sb_succ_iterator succ_end(sandboxir::BasicBlock *BB) {
  return sb_succ_iterator(BB->getTerminator(), true);
}
inline const_sb_succ_iterator succ_end(const sandboxir::BasicBlock *BB) {
  return const_sb_succ_iterator(BB->getTerminator(), true);
}
inline bool succ_empty(const sandboxir::BasicBlock *BB) {
  return succ_begin(BB) == succ_end(BB);
}
inline unsigned succ_size(const sandboxir::BasicBlock *BB) {
  return std::distance(succ_begin(BB), succ_end(BB));
}
inline sb_succ_range successors(sandboxir::BasicBlock *BB) {
  return sb_succ_range(succ_begin(BB), succ_end(BB));
}
inline const_sb_succ_range successors(const sandboxir::BasicBlock *BB) {
  return const_sb_succ_range(succ_begin(BB), succ_end(BB));
}
} // namespace sandboxir

// GraphTraits for sandboxir::BasicBlock.
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
DmpVector<sandboxir::Value *> getOperandBundle(const RangeT &Bndl,
                                               unsigned OpIdx) {
  DmpVector<sandboxir::Value *> OpVec;
  OpVec.reserve(Bndl.size());
  for (auto *SBV : Bndl) {
    auto *SBI = cast<sandboxir::Instruction>(SBV);
    assert(OpIdx < SBI->getNumOperands() && "Out of bounds!");
    OpVec.push_back(SBI->getOperand(OpIdx));
  }
  return OpVec;
}

template <typename RangeT>
SmallVector<DmpVector<sandboxir::Value *>, 2>
getOperandBundles(const RangeT &Bndl) {
  SmallVector<DmpVector<sandboxir::Value *>, 2> OpVecs;
#ifndef NDEBUG
  unsigned NumOps = cast<sandboxir::Instruction>(Bndl[0])->getNumOperands();
  assert(all_of(drop_begin(Bndl),
                [NumOps](auto *V) {
                  return cast<sandboxir::Instruction>(V)->getNumOperands() ==
                         NumOps;
                }) &&
         "Expected same number of operands!");
#endif
  for (unsigned OpIdx :
       seq<unsigned>(cast<sandboxir::Instruction>(Bndl[0])->getNumOperands()))
    OpVecs.push_back(getOperandBundle(Bndl, OpIdx));
  return OpVecs;
}

} // namespace sandboxir
} // namespace llvm
#endif // LLVM_TRANSFORMS_SANDBOXIR_SANDBOXIR_H
