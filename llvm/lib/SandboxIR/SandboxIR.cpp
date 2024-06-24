//===- SandboxIR.cpp - A transactional overlay IR on top of LLVM IR -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include <sstream>

using namespace llvm::sandboxir;

#define DEBUG_TYPE "SBVec"

Value *Use::get() const { return Ctx->getValue(LLVMUse->get()); }
unsigned Use::getOperandNo() const { return Usr->getUseOperandNo(*this); }
void Use::set(Value *Val) {
  auto &Tracker = Ctx->getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<UseSet>(*LLVMUse, Tracker));
  llvm::Value *LLVMVal = ValueAttorney::getValue(Val);
  LLVMUse->set(LLVMVal);
}
void Use::swap(Use &Other) {
  auto &Tracker = Ctx->getTracker();
  llvm::Use &OtherUse = *Other.LLVMUse;
  if (Tracker.tracking())
    Tracker.track(std::make_unique<UseSwap>(*LLVMUse, *Other.LLVMUse, Tracker));
  LLVMUse->swap(OtherUse);
}

#ifndef NDEBUG
void Use::dump(raw_ostream &OS) const {
  Value *Def = nullptr;
  if (LLVMUse == nullptr)
    OS << "<null> LLVM Use! ";
  else
    Def = Ctx->getValue(LLVMUse->get());
  OS << "Def:  ";
  if (Def == nullptr)
    OS << "NULL";
  else
    OS << *Def;
  OS << "\n";

  OS << "User: ";
  if (Usr == nullptr)
    OS << "NULL";
  else
    OS << *Usr;
  OS << "\n";

  OS << "OperandNo: ";
  if (Usr == nullptr)
    OS << "N/A";
  else
    OS << getOperandNo();
  OS << "\n";
}

void Use::dump() const { dump(dbgs()); }
#endif // NDEBUG

Use OperandUseIterator::operator*() const { return Use; }

OperandUseIterator &OperandUseIterator::operator++() {
  assert(Use.LLVMUse != nullptr && "Already at end!");
  User *User = Use.getUser();
  Use = User->getOperandUseInternal(Use.getOperandNo() + 1, /*Verify=*/false);
  return *this;
}

Use UserUseIterator::operator*() const { return Use; }

UserUseIterator &UserUseIterator::operator++() {
  llvm::Use *&LLVMUse = Use.LLVMUse;
  assert(LLVMUse != nullptr && "Already at end!");
  LLVMUse = LLVMUse->getNext();
  if (LLVMUse == nullptr) {
    Use.Usr = nullptr;
    return *this;
  }
  auto *Ctx = Use.Ctx;
  auto *LLVMUser = LLVMUse->getUser();
  User *User = cast_or_null<sandboxir::User>(Ctx->getValue(LLVMUser));
  // This is for uses into Packs that should be skipped.
  // For example:
  //   %Op = add <2 x i8> %v, %v
  //   %Extr0 = extractelement <2 x i8> %Op, i64 0
  //   %Pack0 = insertelement <2 x i8> poison, i8 %Extr0, i64 0
  //   %Extr1 = extractelement <2 x i8> %Op, i64 1
  //   %Pack1 = insertelement <2 x i8> %Pack0, i8 %Extr1, i64 1
  // There should be only 1 Use edge from Op to Pack.
  if (User != nullptr && !User->isRealOperandUse(*LLVMUse))
    return ++(*this);
  Use.Usr = User;
  return *this;
}

Value::Value(ClassID SubclassID, llvm::Value *Val, Context &Ctx)
    : SubclassID(SubclassID), Val(Val), Ctx(Ctx) {
#ifndef NDEBUG
  UID = Ctx.getNumValues();
#endif
}

Value::use_iterator Value::use_begin() {
  llvm::Use *LLVMUse = nullptr;
  if (Val->use_begin() != Val->use_end())
    LLVMUse = &*Val->use_begin();
  User *User = LLVMUse != nullptr ? cast_or_null<sandboxir::User>(Ctx.getValue(
                                        Val->use_begin()->getUser()))
                                  : nullptr;
  return use_iterator(Use(LLVMUse, User, Ctx));
}

Value::user_iterator Value::user_begin() {
  auto UseBegin = Val->use_begin();
  auto UseEnd = Val->use_end();
  bool AtEnd = UseBegin == UseEnd;
  llvm::Use *LLVMUse = AtEnd ? nullptr : &*UseBegin;
  User *User =
      AtEnd ? nullptr
            : cast_or_null<sandboxir::User>(Ctx.getValue(&*LLVMUse->getUser()));
  return user_iterator(Use(LLVMUse, User, Ctx));
}

unsigned Value::getNumUsers() const {
  // Look for unique users.
  SmallPtrSet<Value *, 4> UserNs;
  for (llvm::User *U : Val->users())
    UserNs.insert(getContext().getValue(U));
  return UserNs.size();
}

unsigned Value::getNumUses() const {
  unsigned Cnt = 0;
  for (llvm::User *U : Val->users()) {
    (void)U;
    ++Cnt;
  }
  return Cnt;
}

bool Value::hasNUsersOrMore(unsigned Num) const {
  SmallPtrSet<Value *, 4> UserNs;
  for (llvm::User *U : Val->users()) {
    UserNs.insert(getContext().getValue(U));
    if (UserNs.size() >= Num)
      return true;
  }
  return false;
}

Value *Value::getSingleUser() const {
  assert(Val->hasOneUser() && "Expected single user");
  return *users().begin();
}

Context &Value::getContext() const { return Ctx; }

SandboxIRTracker &Value::getTracker() { return getContext().getTracker(); }

void Value::replaceUsesWithIf(Value *OtherV,
                              llvm::function_ref<bool(Use Use)> ShouldReplace) {
  assert(getType() == OtherV->getType() && "Can't replace with different type");
  llvm::Value *OtherVal = OtherV->Val;
  auto &Tracker = getTracker();
  Val->replaceUsesWithIf(
      OtherVal, [&ShouldReplace, &Tracker, this](llvm::Use &LLVMUse) -> bool {
        User *DstU = cast_or_null<User>(Ctx.getValue(LLVMUse.getUser()));
        if (DstU == nullptr)
          return false;
        unsigned OpIdx = DstU->getOperandUseIdx(LLVMUse);
        if (!ShouldReplace(Use(&LLVMUse, DstU, Ctx)))
          return false;
        if (Tracker.tracking())
          // Tracking like so should be cheaper than replaceAllUsesWith()
          Tracker.track(std::make_unique<SetOperand>(DstU, OpIdx, Tracker));
        return true;
      });
}

void Value::replaceAllUsesWith(Value *Other) {
  assert(getType() == Other->getType() &&
         "Replacing with Value of different type!");
  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<ReplaceAllUsesWith>(this, Tracker));
  Val->replaceAllUsesWith(Other->Val);
}

#ifndef NDEBUG
std::string Value::getName() const {
  std::stringstream SS;
  SS << "SB" << UID << ".";
  return SS.str();
}

void Value::dumpCommonHeader(raw_ostream &OS) const {
  OS << getName() << " " << getSubclassIDStr(SubclassID) << " ";
}

void Value::dumpCommonFooter(raw_ostream &OS) const {
  OS.indent(2) << "Val: ";
  if (Val)
    OS << *Val;
  else
    OS << "NULL";
  OS << "\n";

  // TODO: For now also dump users, but should be removed.
  if (!isa<llvm::Constant>(Val)) {
    OS << "Users: ";
    for (auto *SBU : users()) {
      if (SBU != nullptr)
        OS << SBU->getName();
      else
        OS << "NULL";
      OS << ", ";
    }
  }
}

void Value::dumpCommonPrefix(raw_ostream &OS) const {
  if (Val)
    OS << *Val;
  else
    OS << "NULL ";
}

void Value::dumpCommonSuffix(raw_ostream &OS) const {
  OS << " ; " << getName() << " (" << getSubclassIDStr(SubclassID) << ") "
     << this;
}

void Value::printAsOperandCommon(raw_ostream &OS) const {
  if (Val)
    Val->printAsOperand(OS);
  else
    OS << "NULL ";
}

void Value::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Argument::Argument(llvm::Argument *Arg, Context &Ctx)
    : Value(ClassID::Argument, Arg, Ctx) {}

User::User(ClassID ID, llvm::Value *V, Context &Ctx) : Value(ID, V, Ctx) {}

Use User::getOperandUseDefault(unsigned OpIdx, bool Verify) const {
  assert((!Verify || OpIdx < getNumOperands()) && "Out of bounds!");
  assert(isa<llvm::User>(Val) && "Non-users have no operands!");
  llvm::Use *LLVMUse;
  if (OpIdx != getNumOperands())
    LLVMUse = &cast<llvm::User>(Val)->getOperandUse(OpIdx);
  else
    LLVMUse = cast<llvm::User>(Val)->op_end();
  return Use(LLVMUse, const_cast<User *>(this), Ctx);
}

#ifndef NDEBUG
void User::verifyUserOfLLVMUse(const llvm::Use &Use) const {
  assert(Ctx.getValue(Use.getUser()) == this &&
         "Use not found in this SBUser's operands!");
}
#endif

bool User::classof(const Value *From) {
  switch (From->getSubclassID()) {
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)                                                    \
  case ClassID::ID:                                                            \
    return true;
#define DEF_INSTR(ID, OPC, CLASS)                                              \
  case ClassID::ID:                                                            \
    return true;
#include "llvm/SandboxIR/SandboxIRValues.def"
  default:
    return false;
  }
  return false;
}

User::op_iterator User::op_begin() {
  assert(isa<llvm::User>(Val) && "Expect User value!");
  return op_iterator(getOperandUseInternal(0, /*Verify=*/false));
}

User::op_iterator User::op_end() {
  assert(isa<llvm::User>(Val) && "Expect User value!");
  return op_iterator(getOperandUseInternal(getNumOperands(), /*Verify=*/false));
}

User::const_op_iterator User::op_begin() const {
  return const_cast<User *>(this)->op_begin();
}

User::const_op_iterator User::op_end() const {
  return const_cast<User *>(this)->op_end();
}

Value *User::getSingleOperand() const {
  assert(getNumOperands() == 1 && "Expected exactly 1 operand");
  return getOperand(0);
}

void User::setOperand(unsigned OperandIdx, Value *Operand) {
  if (!isa<llvm::User>(Val))
    llvm_unreachable("No operands!");
  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<SetOperand>(this, OperandIdx, Tracker));
  cast<llvm::User>(Val)->setOperand(OperandIdx,
                                    ValueAttorney::getValue(Operand));
}

bool User::replaceUsesOfWith(Value *FromV, Value *ToV) {
  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(
        std::make_unique<ReplaceUsesOfWith>(this, FromV, ToV, Tracker));

  bool Change = false;
  llvm::Value *FromVIR = ValueAttorney::getValue(FromV);
  llvm::Value *ToVIR = ValueAttorney::getValue(ToV);
  if (auto *SBI = dyn_cast<Instruction>(Ctx.getValue(Val))) {
    for (llvm::Instruction *I : SBI->getLLVMInstrs())
      Change |= I->replaceUsesOfWith(FromVIR, ToVIR);
    return Change;
  }
  return cast<llvm::User>(Val)->replaceUsesOfWith(FromVIR, ToVIR);
}

#ifndef NDEBUG
void User::dumpCommonHeader(raw_ostream &OS) const {
  Value::dumpCommonHeader(OS);
  OS << "(";
  for (auto [OpIdx, Use] : enumerate(operands())) {
    Value *Op = Use;
    if (OpIdx != 0)
      OS << ", ";
    if (Op != nullptr)
      OS << Op->getName();
    else
      OS << "<NULL OpN>";
  }
  OS << ")";
}
#endif

BBIterator &BBIterator::operator++() {
  auto ItE = BB->end();
  assert(It != ItE && "Already at end!");
  ++It;
  if (It == ItE)
    return *this;
  Instruction &NextI = *cast<Instruction>(Ctx->getValue(&*It));
  unsigned Num = NextI.getNumOfIRInstrs();
  assert(Num > 0 && "Bad getNumOfIRInstrs()");
  It = std::next(It, Num - 1);
  return *this;
}

BBIterator &BBIterator::operator--() {
  assert(It != BB->begin() && "Already at begin!");
  if (It == BB->end()) {
    --It;
    return *this;
  }
  Instruction &CurrI = **this;
  unsigned Num = CurrI.getNumOfIRInstrs();
  assert(Num > 0 && "Bad getNumOfIRInstrs()");
  assert(std::prev(It, Num - 1) != BB->begin() && "Already at begin!");
  It = std::prev(It, Num);
  return *this;
}

Instruction::Instruction(ClassID ID, Opcode Opc, llvm::Instruction *I,
                         Context &Ctx)
    : User(ID, I, Ctx), Opc(Opc) {
  assert((!isa<llvm::StoreInst>(I) || SubclassID == ClassID::Store) &&
         "Create a StoreInstruction!");
  assert((!isa<llvm::LoadInst>(I) || SubclassID == ClassID::Load) &&
         "Create a LoadInstruction!");
  assert((!isa<llvm::CastInst>(I) ||
          I->getOpcode() == llvm::Instruction::AddrSpaceCast ||
          SubclassID == ClassID::Cast) &&
         "Create a CastInstruction!");
  assert((!isa<llvm::PHINode>(I) || SubclassID == ClassID::PHI) &&
         "Create a PHINode!");
  assert((!isa<llvm::CmpInst>(I) || SubclassID == ClassID::Cmp) &&
         "Create a CmpInstruction!");
  assert((!isa<llvm::SelectInst>(I) || SubclassID == ClassID::Select) &&
         "Create a SelectInstruction!");
  assert((!isa<llvm::BinaryOperator>(I) || SubclassID == ClassID::BinOp) &&
         "Create a BinaryOperator!");
  assert((!isa<llvm::UnaryOperator>(I) || SubclassID == ClassID::UnOp) &&
         "Create a UnaryOperator!");
}

llvm::Instruction *Instruction::getTopmostLLVMInstruction() const {
  Instruction *Prev = getPrevNode();
  if (Prev == nullptr) {
    // If at top of the BB, return the first BB instruction.
    return &*cast<llvm::BasicBlock>(ValueAttorney::getValue(getParent()))
                 ->begin();
  }
  // Else get the Previous sandbox IR instruction's bottom IR instruction and
  // return its successor.
  llvm::Instruction *PrevBotI =
      cast<llvm::Instruction>(ValueAttorney::getValue(Prev));
  return PrevBotI->getNextNode();
}

BBIterator Instruction::getIterator() const {
  auto *I = cast<llvm::Instruction>(Val);
  return BasicBlock::iterator(I->getParent(), I->getIterator(), &Ctx);
}

bool BBIterator::atBegin() const {
  // Fast path: if the internal iterator is at begin().
  if (It == BB->begin())
    return true;
  // We may still be at begin if this is a multi-IR sandbox::Instruction and it
  // is pointing to its bottom LLVM IR Instr.
  unsigned NumInstrs = getI(It)->getNumOfIRInstrs();
  if (NumInstrs == 1)
    // This is a single-IR SBI. Since It != BB->begin() we are not at begin.
    return false;
  return std::prev(It, NumInstrs - 1) == BB->begin();
}

Instruction *Instruction::getNextNode() const {
  assert(getParent() != nullptr && "Detached!");
  assert(getIterator() != getParent()->end() && "Already at end!");
  auto *CurrI = cast<llvm::Instruction>(Val);
  assert(CurrI->getParent() != nullptr && "LLVM IR instr is detached!");
  auto *NextI = CurrI->getNextNode();
  auto *NextSBI = cast_or_null<Instruction>(Ctx.getValue(NextI));
  if (NextSBI == nullptr)
    return nullptr;
  return NextSBI;
}

Instruction *Instruction::getPrevNode() const {
  assert(getParent() != nullptr && "Detached!");
  auto It = getIterator();
  if (!It.atBegin())
    return std::prev(getIterator()).get();
  return nullptr;
}

llvm::Instruction::UnaryOps Instruction::getIRUnaryOp(Opcode Opc) {
  switch (Opc) {
  case Opcode::FNeg:
    return static_cast<llvm::Instruction::UnaryOps>(llvm::Instruction::FNeg);
  default:
    llvm_unreachable("Not a unary op!");
  }
}

llvm::Instruction::BinaryOps Instruction::getIRBinaryOp(Opcode Opc) {
  switch (Opc) {
  case Opcode::Add:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::Add);
  case Opcode::FAdd:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::FAdd);
  case Opcode::Sub:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::Sub);
  case Opcode::FSub:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::FSub);
  case Opcode::Mul:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::Mul);
  case Opcode::FMul:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::FMul);
  case Opcode::UDiv:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::UDiv);
  case Opcode::SDiv:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::SDiv);
  case Opcode::FDiv:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::FDiv);
  case Opcode::URem:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::URem);
  case Opcode::SRem:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::SRem);
  case Opcode::FRem:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::FRem);
  case Opcode::Shl:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::Shl);
  case Opcode::LShr:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::LShr);
  case Opcode::AShr:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::AShr);
  case Opcode::And:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::And);
  case Opcode::Or:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::Or);
  case Opcode::Xor:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::Xor);
  default:
    llvm_unreachable("Not a unary op!");
  }
}

llvm::Instruction::CastOps Instruction::getIRCastOp(Opcode Opc) {
  switch (Opc) {
  case Opcode::ZExt:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::ZExt);
  case Opcode::SExt:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::SExt);
  case Opcode::FPToUI:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::FPToUI);
  case Opcode::FPToSI:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::FPToSI);
  case Opcode::FPExt:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::FPExt);
  case Opcode::PtrToInt:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::PtrToInt);
  case Opcode::IntToPtr:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::IntToPtr);
  case Opcode::SIToFP:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::SIToFP);
  case Opcode::UIToFP:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::UIToFP);
  case Opcode::Trunc:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::Trunc);
  case Opcode::FPTrunc:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::FPTrunc);
  case Opcode::BitCast:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::BitCast);
  default:
    llvm_unreachable("Not a unary op!");
  }
}

const char *Instruction::getOpcodeName(Opcode Opc) {
  switch (Opc) {
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define OP(OPC)                                                                \
  case Opcode::OPC:                                                            \
    return #OPC;
#define OPCODES(...) __VA_ARGS__
#define DEF_INSTR(ID, OPC, CLASS) OPC
#include "llvm/SandboxIR/SandboxIRValues.def"
  }
}

bool Instruction::classof(const Value *From) {
  switch (From->getSubclassID()) {
#define DEF_INSTR(ID, OPC, CLASS)                                              \
  case ClassID::ID:                                                            \
    return true;
#include "llvm/SandboxIR/SandboxIRValues.def"
  default:
    return false;
  }
}

int64_t Instruction::getInstrNumber() const {
  auto *Parent = getParent();
  assert(Parent != nullptr && "Can't get number of a detached instruction!");
  return Parent->getInstrNumber(this);
}

void Instruction::removeFromParent() {
  // Update InstrNumberMap
  getParent()->removeInstrNumber(this);
  // Run the callbacks before we unregister it.
  Ctx.runRemoveInstrCallbacks(this);

  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<InstrRemoveFromParent>(this, Tracker));

  // Detach all the IR instructions from their parent BB.
  for (llvm::Instruction *I : getLLVMInstrs()) {
    I->removeFromParent();
  }
}

void Instruction::eraseFromParent() {
  assert(users().empty() && "Still connected to users, can't erase!");
  auto IRInstrs = getLLVMInstrs();
  // Run the callbacks before we unregister it.
  Ctx.runRemoveInstrCallbacks(this);
  auto &Tracker = getTracker();

  // Update InstrNumbers
  getParent()->removeInstrNumber(this);

  // Detach from instruction-specific maps.
  auto SBIPtr = getContext().detach(this);

  if (Tracker.tracking()) {
    // Track deletion from IR to SandboxIR maps.
    Tracker.track(
        std::make_unique<EraseFromParent>(std::move(SBIPtr), Ctx, Tracker));
  } else if (!Tracker.inRevert()) {
    // 1. Regardless of whether we are tracking or not, we should not leak
    //    memory.
    //    So track instrs that got "deleted" such that we actually delete them.
    // 2. This also helps with avoid dangling uses of internal InsertElements of
    //    Packs because we only detach the external facing edges.
    // Note: reverting requires the tables be populated so tracking of the
    // erasing action should happen in this order.
    Tracker.track(std::make_unique<DeleteOnAccept>(this, Tracker));
  }

  if (Tracker.inRevert()) {
    // If this is called by CreateAndInsertInstr::revert() then we should just
    // erase all instructions.
    // Erase bottom-up to avoid erasing instructions with attached users.
    for (llvm::Instruction *I : reverse(IRInstrs))
      I->eraseFromParent();
  } else {
    // We don't actually delete the IR instruction, because then it would be
    // impossible to bring it back from the dead at the same memory location.
    // Instead we remove it from its BB and track its current location.
    for (llvm::Instruction *I : IRInstrs) {
      I->removeFromParent();
    }
    for (llvm::Instruction *I : getLLVMInstrsWithExternalOperands()) {
      I->dropAllReferences();
    }
  }
}

BasicBlock *Instruction::getParent() const {
  auto *BB = cast<llvm::Instruction>(Val)->getParent();
  if (BB == nullptr)
    return nullptr;
  return Ctx.getBasicBlock(BB);
}

uint64_t Instruction::getApproximateDistanceTo(Instruction *ToI) const {
  auto FromNum = getInstrNumber();
  auto ToNum = ToI->getInstrNumber();
  return std::abs(ToNum - FromNum) / BasicBlock::InstrNumberingStep;
}

void Instruction::moveBefore(BasicBlock &SBBB, const BBIterator &WhereIt) {
  if (std::next(getIterator()) == WhereIt)
    // Destination is same as origin, nothing to do.
    return;
  auto &Tracker = getTracker();
  if (Tracker.tracking()) {
    Tracker.track(std::make_unique<MoveInstr>(this, Tracker));
  }
  Ctx.runMoveInstrCallbacks(this, SBBB, WhereIt);

  auto *BB = cast<llvm::BasicBlock>(ValueAttorney::getValue(&SBBB));
  llvm::BasicBlock::iterator It;
  if (WhereIt == SBBB.end())
    It = BB->end();
  else {
    Instruction *WhereI = &*WhereIt;
    It = WhereI->getTopmostLLVMInstruction()->getIterator();
  }
  // Update instruction numbering (part 1)
  if (auto *Parent = getParent())
    Parent->removeInstrNumber(this);
  // Do the actual move in LLVM IR.
  for (auto *I : getLLVMInstrs())
    I->moveBefore(*BB, It);
  // Update instruction numbering (part 2)
  SBBB.assignInstrNumber(this);
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  Ctx.afterMoveInstrHook(SBBB);
#endif
}

void Instruction::insertBefore(Instruction *BeforeI) {
  // Maintain instruction numbering
  if (auto *Parent = getParent())
    Parent->removeInstrNumber(this);
  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<InsertToBB>(BeforeI, getParent(), Tracker));
  llvm::Instruction *BeforeTopI = BeforeI->getTopmostLLVMInstruction();
  auto IRInstrs = getLLVMInstrs();
  for (llvm::Instruction *I : IRInstrs)
    I->insertBefore(BeforeTopI);
  // Update instruction numbering.
  BeforeI->getParent()->assignInstrNumber(this);
}

void Instruction::insertAfter(Instruction *AfterI) {
  insertInto(AfterI->getParent(), std::next(AfterI->getIterator()));
}

void Instruction::insertInto(BasicBlock *SBBB, const BBIterator &WhereIt) {
  // Maintain instruction numbering
  if (auto *Parent = getParent())
    Parent->removeInstrNumber(this);
  llvm::BasicBlock *BB = cast<llvm::BasicBlock>(ValueAttorney::getValue(SBBB));
  llvm::Instruction *BeforeI;
  Instruction *SBBeforeI;
  llvm::BasicBlock::iterator BeforeIt;
  if (WhereIt != SBBB->end()) {
    SBBeforeI = &*WhereIt;
    BeforeI = SBBeforeI->getTopmostLLVMInstruction();
    BeforeIt = BeforeI->getIterator();
  } else {
    SBBeforeI = nullptr;
    BeforeI = nullptr;
    BeforeIt = BB->end();
  }
  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<InsertToBB>(SBBeforeI, SBBB, Tracker));
  for (llvm::Instruction *I : getLLVMInstrs())
    I->insertInto(BB, BeforeIt);
  // Update instruction numbering.
  SBBB->assignInstrNumber(this);
}

BasicBlock *Instruction::getSuccessor(unsigned Idx) const {
  return cast<BasicBlock>(
      Ctx.getValue(cast<llvm::Instruction>(Val)->getSuccessor(Idx)));
}

#ifndef NDEBUG
void Instruction::dump(raw_ostream &OS) const {
  OS << "Unimplemented! Please override dump().";
}
void Instruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
void Instruction::dumpVerbose(raw_ostream &OS) const {
  OS << "Unimplemented! Please override dumpVerbose().";
}
void Instruction::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
#endif

Constant::Constant(llvm::Constant *C, Context &Ctx)
    : User(ClassID::Constant, C, Ctx) {}

bool Constant::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Constant ||
         From->getSubclassID() == ClassID::Function;
}
#ifndef NDEBUG
void Constant::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void Constant::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void Constant::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void Constant::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

bool Argument::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Argument;
}

#ifndef NDEBUG
void Argument::printAsOperand(raw_ostream &OS) const {
  printAsOperandCommon(OS);
}
void Argument::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}
void Argument::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
void Argument::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void Argument::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
#endif

Value *CmpInst::create(llvm::CmpInst::Predicate Pred, Value *LHS, Value *RHS,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name, MDNode *FPMathTag) {
  llvm::Value *LHSIR = ValueAttorney::getValue(LHS);
  llvm::Value *RHSIR = ValueAttorney::getValue(RHS);
  llvm::Instruction *BeforeIR = InsertBefore->getTopmostLLVMInstruction();
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  auto *NewV = Builder.CreateCmp(Pred, LHSIR, RHSIR, Name, FPMathTag);
  if (auto *NewCI = dyn_cast<llvm::CmpInst>(NewV))
    return Ctx.createCmpInst(NewCI);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *CmpInst::create(llvm::CmpInst::Predicate Pred, Value *LHS, Value *RHS,
                       BasicBlock *InsertAtEnd, Context &Ctx, const Twine &Name,
                       MDNode *FPMathTag) {
  llvm::Value *LHSIR = ValueAttorney::getValue(LHS);
  llvm::Value *RHSIR = ValueAttorney::getValue(RHS);
  llvm::BasicBlock *InsertAtEndIR = BasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertAtEndIR);
  auto *NewV = Builder.CreateCmp(Pred, LHSIR, RHSIR, Name, FPMathTag);
  if (auto *NewCI = dyn_cast<llvm::CmpInst>(NewV))
    return Ctx.createCmpInst(NewCI);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

bool CmpInst::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Cmp;
}

#ifndef NDEBUG
void CmpInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void CmpInst::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void CmpInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void CmpInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

BranchInst *BranchInst::create(BasicBlock *IfTrue, Instruction *InsertBefore,
                               Context &Ctx) {
  llvm::BasicBlock *IfTrueLLVM = BasicBlockAttorney::getBB(IfTrue);
  llvm::Instruction *InsertBeforeLLVM =
      cast<llvm::Instruction>(ValueAttorney::getValue(InsertBefore));
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertBeforeLLVM);
  llvm::BranchInst *NewBr = Builder.CreateBr(IfTrueLLVM);
  return Ctx.createBranchInst(NewBr);
}

BranchInst *BranchInst::create(BasicBlock *IfTrue, BasicBlock *InsertAtEnd,
                               Context &Ctx) {
  llvm::BasicBlock *IfTrueLLVM = BasicBlockAttorney::getBB(IfTrue);
  llvm::BasicBlock *InsertAtEndLLVM = BasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertAtEndLLVM);
  llvm::BranchInst *NewBr = Builder.CreateBr(IfTrueLLVM);
  return Ctx.createBranchInst(NewBr);
}

BranchInst *BranchInst::create(BasicBlock *IfTrue, BasicBlock *IfFalse,
                               Value *Cond, Instruction *InsertBefore,
                               Context &Ctx) {
  llvm::BasicBlock *IfTrueLLVM = BasicBlockAttorney::getBB(IfTrue);
  llvm::BasicBlock *IfFalseLLVM = BasicBlockAttorney::getBB(IfFalse);
  llvm::Value *CondLLVM = BasicBlockAttorney::getBB(IfFalse);
  llvm::Instruction *InsertBeforeLLVM =
      cast<llvm::Instruction>(ValueAttorney::getValue(InsertBefore));
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertBeforeLLVM);
  llvm::BranchInst *NewBr =
      Builder.CreateCondBr(CondLLVM, IfTrueLLVM, IfFalseLLVM);
  return Ctx.createBranchInst(NewBr);
}

BranchInst *BranchInst::create(BasicBlock *IfTrue, BasicBlock *IfFalse,
                               Value *Cond, BasicBlock *InsertAtEnd,
                               Context &Ctx) {
  llvm::BasicBlock *IfTrueLLVM = BasicBlockAttorney::getBB(IfTrue);
  llvm::BasicBlock *IfFalseLLVM = BasicBlockAttorney::getBB(IfFalse);
  llvm::Value *CondLLVM = BasicBlockAttorney::getBB(IfFalse);
  llvm::BasicBlock *InsertAtEndLLVM = BasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertAtEndLLVM);
  llvm::BranchInst *NewBr =
      Builder.CreateCondBr(CondLLVM, IfTrueLLVM, IfFalseLLVM);
  return Ctx.createBranchInst(NewBr);
}

bool BranchInst::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Br;
}

Value *BranchInst::getCondition() const {
  assert(isConditional() && "Cannot get condition of an uncond branch!");
  return Ctx.getValue(cast<llvm::BranchInst>(Val)->getCondition());
}

BasicBlock *BranchInst::getSuccessor(unsigned i) const {
  assert(i < getNumSuccessors() && "Successor # out of range for Branch!");
  return cast_or_null<BasicBlock>(
      Ctx.getValue(cast<llvm::BranchInst>(Val)->getSuccessor(i)));
}

void BranchInst::setSuccessor(unsigned Idx, BasicBlock *NewSucc) {
  assert((Idx == 0 || Idx == 1) && "Out of bounds!");
  setOperand(2u - Idx, NewSucc);
}

BasicBlock *BranchInst::LLVMBBToSBBB::operator()(llvm::BasicBlock *BB) const {
  return cast<BasicBlock>(Ctx.getValue(BB));
}
const BasicBlock *
BranchInst::ConstLLVMBBToSBBB::operator()(const llvm::BasicBlock *BB) const {
  return cast<BasicBlock>(Ctx.getValue(BB));
}
#ifndef NDEBUG
void BranchInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}
void BranchInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
void BranchInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void BranchInst::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}

#endif // NDEBUG

StoreInst *StoreInst::create(Value *V, Value *Ptr, MaybeAlign Align,
                             Instruction *InsertBefore, Context &Ctx) {
  llvm::Value *ValIR = ValueAttorney::getValue(V);
  llvm::Value *PtrIR = ValueAttorney::getValue(Ptr);
  llvm::Instruction *BeforeIR = InsertBefore->getTopmostLLVMInstruction();
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  auto *NewSI =
      Builder.CreateAlignedStore(ValIR, PtrIR, Align, /*isVolatile=*/false);
  auto *NewSBI = Ctx.createStoreInst(NewSI);
  return NewSBI;
}
StoreInst *StoreInst::create(Value *V, Value *Ptr, MaybeAlign Align,
                             BasicBlock *InsertAtEnd, Context &Ctx) {
  llvm::Value *ValIR = ValueAttorney::getValue(V);
  llvm::Value *PtrIR = ValueAttorney::getValue(Ptr);
  llvm::BasicBlock *InsertAtEndIR = BasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertAtEndIR);
  auto *NewSI =
      Builder.CreateAlignedStore(ValIR, PtrIR, Align, /*isVolatile=*/false);
  auto *NewSBI = Ctx.createStoreInst(NewSI);
  return NewSBI;
}

bool StoreInst::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Store;
}

Value *StoreInst::getValueOperand() const {
  return Ctx.getValue(cast<llvm::StoreInst>(Val)->getValueOperand());
}

Value *StoreInst::getPointerOperand() const {
  return Ctx.getValue(cast<llvm::StoreInst>(Val)->getPointerOperand());
}

#ifndef NDEBUG
void StoreInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void StoreInst::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void StoreInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void StoreInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

LoadInst *LoadInst::create(Type *Ty, Value *Ptr, MaybeAlign Align,
                           Instruction *InsertBefore, Context &Ctx,
                           const Twine &Name) {
  llvm::Value *PtrIR = ValueAttorney::getValue(Ptr);
  llvm::Instruction *BeforeIR = InsertBefore->getTopmostLLVMInstruction();
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  auto *NewLI =
      Builder.CreateAlignedLoad(Ty, PtrIR, Align, /*isVolatile=*/false, Name);
  auto *NewSBI = Ctx.createLoadInst(NewLI);
  return NewSBI;
}

LoadInst *LoadInst::create(Type *Ty, Value *Ptr, MaybeAlign Align,
                           BasicBlock *InsertAtEnd, Context &Ctx,
                           const Twine &Name) {
  llvm::Value *PtrIR = ValueAttorney::getValue(Ptr);
  llvm::BasicBlock *InsertAtEndIR = BasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertAtEndIR);
  auto *NewLI =
      Builder.CreateAlignedLoad(Ty, PtrIR, Align, /*isVolatile=*/false, Name);
  auto *NewSBI = Ctx.createLoadInst(NewLI);
  return NewSBI;
}

bool LoadInst::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Load;
}

Value *LoadInst::getPointerOperand() const {
  return Ctx.getValue(cast<llvm::LoadInst>(Val)->getPointerOperand());
}

#ifndef NDEBUG
void LoadInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void LoadInst::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void LoadInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void LoadInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Value *CastInst::create(Type *Ty, Opcode Op, Value *Operand,
                        Instruction *InsertBefore, Context &Ctx,
                        const Twine &Name) {
  llvm::Value *IROperand = ValueAttorney::getValue(Operand);
  llvm::Instruction *BeforeIR = InsertBefore->getTopmostLLVMInstruction();
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  auto *NewV = Builder.CreateCast(getIRCastOp(Op), IROperand, Ty, Name);
  if (auto *NewCI = dyn_cast<llvm::CastInst>(NewV))
    return Ctx.createCastInst(NewCI);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *CastInst::create(Type *Ty, Opcode Op, Value *Operand,
                        BasicBlock *InsertAtEnd, Context &Ctx,
                        const Twine &Name) {
  llvm::Value *IROperand = ValueAttorney::getValue(Operand);
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(BasicBlockAttorney::getBB(InsertAtEnd));
  auto *NewV = Builder.CreateCast(getIRCastOp(Op), IROperand, Ty, Name);
  if (auto *NewCI = dyn_cast<llvm::CastInst>(NewV))
    return Ctx.createCastInst(NewCI);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

bool CastInst::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Cast;
}

#ifndef NDEBUG
void CastInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void CastInst::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void CastInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void CastInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Value *PHINode::create(Type *Ty, unsigned NumReservedValues,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name) {
  llvm::Instruction *InsertBeforeIR = InsertBefore->getTopmostLLVMInstruction();
  llvm::PHINode *NewPHI =
      llvm::PHINode::Create(Ty, NumReservedValues, Name, InsertBeforeIR);
  return Ctx.createPHINode(NewPHI);
}

Value *PHINode::create(Type *Ty, unsigned NumReservedValues,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name) {
  llvm::BasicBlock *InsertAtEndIR = BasicBlockAttorney::getBB(InsertAtEnd);
  llvm::PHINode *NewPHI =
      llvm::PHINode::Create(Ty, NumReservedValues, Name, InsertAtEndIR);
  return Ctx.createPHINode(NewPHI);
}

bool PHINode::classof(const Value *From) {
  return From->getSubclassID() == ClassID::PHI;
}

#ifndef NDEBUG
void PHINode::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void PHINode::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void PHINode::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void PHINode::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Value *SelectInst::create(Value *Cond, Value *True, Value *False,
                          Instruction *InsertBefore, Context &Ctx,
                          const Twine &Name) {
  llvm::Value *IRCond = ValueAttorney::getValue(Cond);
  llvm::Value *IRTrue = ValueAttorney::getValue(True);
  llvm::Value *IRFalse = ValueAttorney::getValue(False);
  llvm::Instruction *BeforeIR = InsertBefore->getTopmostLLVMInstruction();
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  llvm::Value *NewV = Builder.CreateSelect(IRCond, IRTrue, IRFalse, Name);
  if (auto *NewSI = dyn_cast<llvm::SelectInst>(NewV))
    return Ctx.createSelectInst(NewSI);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *SelectInst::create(Value *Cond, Value *True, Value *False,
                          BasicBlock *InsertAtEnd, Context &Ctx,
                          const Twine &Name) {
  llvm::Value *IRCond = ValueAttorney::getValue(Cond);
  llvm::Value *IRTrue = ValueAttorney::getValue(True);
  llvm::Value *IRFalse = ValueAttorney::getValue(False);
  llvm::BasicBlock *IRInsertAtEnd = BasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(IRInsertAtEnd);
  llvm::Value *NewV = Builder.CreateSelect(IRCond, IRTrue, IRFalse, Name);
  if (auto *NewSI = dyn_cast<llvm::SelectInst>(NewV))
    return Ctx.createSelectInst(NewSI);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

bool SelectInst::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Select;
}

#ifndef NDEBUG
void SelectInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SelectInst::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SelectInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SelectInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Value *BinaryOperator::create(Instruction::Opcode Op, Value *LHS, Value *RHS,
                              Instruction *InsertBefore, Context &Ctx,
                              const Twine &Name) {
  llvm::Value *IRLHS = ValueAttorney::getValue(LHS);
  llvm::Value *IRRHS = ValueAttorney::getValue(RHS);
  llvm::Instruction *BeforeIR = InsertBefore->getTopmostLLVMInstruction();
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  llvm::Value *NewV =
      Builder.CreateBinOp(getIRBinaryOp(Op), IRLHS, IRRHS, Name);
  if (auto *NewBinOp = dyn_cast<llvm::BinaryOperator>(NewV)) {
    return Ctx.createBinaryOperator(NewBinOp);
  }
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *BinaryOperator::create(Instruction::Opcode Op, Value *LHS, Value *RHS,
                              BasicBlock *InsertAtEnd, Context &Ctx,
                              const Twine &Name) {
  llvm::Value *IRLHS = ValueAttorney::getValue(LHS);
  llvm::Value *IRRHS = ValueAttorney::getValue(RHS);
  llvm::BasicBlock *IRInsertAtEnd = BasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(IRInsertAtEnd);
  llvm::Value *NewV =
      Builder.CreateBinOp(getIRBinaryOp(Op), IRLHS, IRRHS, Name);
  if (auto *NewBinOp = dyn_cast<llvm::BinaryOperator>(NewV)) {
    return Ctx.createBinaryOperator(NewBinOp);
  }
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *BinaryOperator::createWithCopiedFlags(Instruction::Opcode Op, Value *LHS,
                                             Value *RHS, Value *CopyFrom,
                                             Instruction *InsertBefore,
                                             Context &Ctx, const Twine &Name) {
  Value *NewV = create(Op, LHS, RHS, InsertBefore, Ctx, Name);
  llvm::Value *IRCopyFrom = ValueAttorney::getValue(CopyFrom);
  if (isa<BinaryOperator>(NewV))
    cast<llvm::BinaryOperator>(ValueAttorney::getValue(NewV))
        ->copyIRFlags(IRCopyFrom);
  return NewV;
}

Value *BinaryOperator::createWithCopiedFlags(Instruction::Opcode Op, Value *LHS,
                                             Value *RHS, Value *CopyFrom,
                                             BasicBlock *InsertAtEnd,
                                             Context &Ctx, const Twine &Name) {
  Value *NewV = create(Op, LHS, RHS, InsertAtEnd, Ctx, Name);
  llvm::Value *IRCopyFrom = ValueAttorney::getValue(CopyFrom);
  if (isa<BinaryOperator>(NewV))
    cast<llvm::BinaryOperator>(ValueAttorney::getValue(NewV))
        ->copyIRFlags(IRCopyFrom);
  return NewV;
}

bool BinaryOperator::classof(const Value *From) {
  return From->getSubclassID() == ClassID::BinOp;
}

#ifndef NDEBUG
void BinaryOperator::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void BinaryOperator::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void BinaryOperator::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void BinaryOperator::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Value *UnaryOperator::createWithCopiedFlags(Instruction::Opcode Op, Value *OpV,
                                            Value *CopyFrom,
                                            Instruction *InsertBefore,
                                            Context &Ctx, const Twine &Name) {
  llvm::Value *IROpV = ValueAttorney::getValue(OpV);
  llvm::Value *IRCopyFrom = ValueAttorney::getValue(CopyFrom);
  llvm::Instruction *BeforeIR = InsertBefore->getTopmostLLVMInstruction();
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  llvm::Value *NewV = Builder.CreateUnOp(getIRUnaryOp(Op), IROpV, Name);
  if (auto *NewBinOp = dyn_cast<llvm::UnaryOperator>(NewV)) {
    NewBinOp->copyIRFlags(IRCopyFrom);
    return Ctx.createUnaryOperator(NewBinOp);
  }
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *UnaryOperator::createWithCopiedFlags(Instruction::Opcode Op, Value *OpV,
                                            Value *CopyFrom,
                                            BasicBlock *InsertAtEnd,
                                            Context &Ctx, const Twine &Name) {
  llvm::Value *IROpV = ValueAttorney::getValue(OpV);
  llvm::Value *IRCopyFrom = ValueAttorney::getValue(CopyFrom);
  llvm::BasicBlock *IRInsertAtEnd = BasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(IRInsertAtEnd);
  llvm::Value *NewV = Builder.CreateUnOp(getIRUnaryOp(Op), IROpV, Name);
  if (auto *NewBinOp = dyn_cast<llvm::UnaryOperator>(NewV)) {
    NewBinOp->copyIRFlags(IRCopyFrom);
    return Ctx.createUnaryOperator(NewBinOp);
  }
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

bool UnaryOperator::classof(const Value *From) {
  return From->getSubclassID() == ClassID::UnOp;
}

#ifndef NDEBUG
void UnaryOperator::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void UnaryOperator::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void UnaryOperator::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void UnaryOperator::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Value *RetInst::create(Value *RetVal, Instruction *InsertBefore, Context &Ctx) {
  llvm::Value *LLVMRet = ValueAttorney::getValue(RetVal);
  llvm::Instruction *BeforeIR = InsertBefore->getTopmostLLVMInstruction();
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  llvm::ReturnInst *NewV;
  if (RetVal != nullptr)
    NewV = Builder.CreateRet(LLVMRet);
  else
    NewV = Builder.CreateRetVoid();
  if (auto *NewRI = dyn_cast<ReturnInst>(NewV))
    return Ctx.createRetInst(NewRI);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *RetInst::create(Value *RetVal, BasicBlock *InsertAtEnd, Context &Ctx) {
  llvm::Value *LLVMRet =
      RetVal != nullptr ? ValueAttorney::getValue(RetVal) : nullptr;
  llvm::BasicBlock *LLVMInsertAtEnd = BasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertAtEnd);
  llvm::ReturnInst *NewV;
  if (RetVal != nullptr)
    NewV = Builder.CreateRet(LLVMRet);
  else
    NewV = Builder.CreateRetVoid();
  if (auto *NewRI = dyn_cast<ReturnInst>(NewV))
    return Ctx.createRetInst(NewRI);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *RetInst::getReturnValue() const {
  auto *LLVMRetVal = cast<ReturnInst>(Val)->getReturnValue();
  return LLVMRetVal != nullptr ? Ctx.getValue(LLVMRetVal) : nullptr;
}

#ifndef NDEBUG
void RetInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void RetInst::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void RetInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void RetInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

CallInst *CallInst::create(FunctionType *FTy, Value *Func,
                           ArrayRef<Value *> Args, BasicBlock::iterator WhereIt,
                           BasicBlock *WhereBB, Context &Ctx,
                           const Twine &NameStr) {
  llvm::Value *LLVMFunc = ValueAttorney::getValue(Func);
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(BasicBlockAttorney::getBB(WhereBB));
  SmallVector<llvm::Value *> LLVMArgs;
  LLVMArgs.reserve(Args.size());
  for (Value *Arg : Args)
    LLVMArgs.push_back(ValueAttorney::getValue(Arg));
  llvm::CallInst *NewCI = Builder.CreateCall(FTy, LLVMFunc, LLVMArgs, NameStr);
  return Ctx.createCallInst(NewCI);
}

CallInst *CallInst::create(FunctionType *FTy, Value *Func,
                           ArrayRef<Value *> Args, Instruction *InsertBefore,
                           Context &Ctx, const Twine &NameStr) {
  return CallInst::create(FTy, Func, Args, InsertBefore->getIterator(),
                          InsertBefore->getParent(), Ctx, NameStr);
}

CallInst *CallInst::create(FunctionType *FTy, Value *Func,
                           ArrayRef<Value *> Args, BasicBlock *InsertAtEnd,
                           Context &Ctx, const Twine &NameStr) {
  return CallInst::create(FTy, Func, Args, InsertAtEnd->end(), InsertAtEnd, Ctx,
                          NameStr);
}

#ifndef NDEBUG
void CallInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void CallInst::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void CallInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void CallInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Value *GetElementPtrInst::create(Type *Ty, Value *Ptr,
                                 ArrayRef<Value *> IdxList,
                                 BasicBlock::iterator WhereIt,
                                 BasicBlock *WhereBB, Context &Ctx,
                                 const Twine &NameStr) {
  llvm::Value *LLVMPtr = ValueAttorney::getValue(Ptr);
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(BasicBlockAttorney::getBB(WhereBB));
  SmallVector<llvm::Value *> LLVMIdxList;
  LLVMIdxList.reserve(IdxList.size());
  for (Value *Idx : IdxList)
    LLVMIdxList.push_back(ValueAttorney::getValue(Idx));
  llvm::Value *NewV = Builder.CreateGEP(Ty, LLVMPtr, LLVMIdxList, NameStr);
  if (auto *NewGEP = dyn_cast<llvm::GetElementPtrInst>(NewV))
    return Ctx.createGetElementPtrInst(NewGEP);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *GetElementPtrInst::create(Type *Ty, Value *Ptr,
                                 ArrayRef<Value *> IdxList,
                                 Instruction *InsertBefore, Context &Ctx,
                                 const Twine &NameStr) {
  return GetElementPtrInst::create(Ty, Ptr, IdxList,
                                   InsertBefore->getIterator(),
                                   InsertBefore->getParent(), Ctx, NameStr);
}

Value *GetElementPtrInst::create(Type *Ty, Value *Ptr,
                                 ArrayRef<Value *> IdxList,
                                 BasicBlock *InsertAtEnd, Context &Ctx,
                                 const Twine &NameStr) {
  return GetElementPtrInst::create(Ty, Ptr, IdxList, InsertAtEnd->end(),
                                   InsertAtEnd, Ctx, NameStr);
}

#ifndef NDEBUG
void GetElementPtrInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void GetElementPtrInst::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void GetElementPtrInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void GetElementPtrInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

#ifndef NDEBUG
void OpaqueInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void OpaqueInst::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void OpaqueInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void OpaqueInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Value *InsertElementInst::create(Value *Vec, Value *NewElt, Value *Idx,
                                 Instruction *InsertBefore, Context &Ctx,
                                 const Twine &Name) {
  llvm::Value *LLVMVec = ValueAttorney::getValue(Vec);
  llvm::Value *LLVMNewElt = ValueAttorney::getValue(NewElt);
  llvm::Value *LLVMIdx = ValueAttorney::getValue(Idx);
  llvm::Instruction *LLVMInsertBefore =
      cast<llvm::Instruction>(ValueAttorney::getValue(InsertBefore));
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertBefore);
  llvm::Value *NewV =
      Builder.CreateInsertElement(LLVMVec, LLVMNewElt, LLVMIdx, Name);
  if (auto *NewInsert = dyn_cast<llvm::InsertElementInst>(NewV))
    return Ctx.createInsertElementInst(NewInsert);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *InsertElementInst::create(Value *Vec, Value *NewElt, Value *Idx,
                                 BasicBlock *InsertAtEnd, Context &Ctx,
                                 const Twine &Name) {
  llvm::Value *LLVMVec = ValueAttorney::getValue(Vec);
  llvm::Value *LLVMNewElt = ValueAttorney::getValue(NewElt);
  llvm::Value *LLVMIdx = ValueAttorney::getValue(Idx);
  llvm::BasicBlock *LLVMInsertAtEnd =
      cast<llvm::BasicBlock>(BasicBlockAttorney::getBB(InsertAtEnd));
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertAtEnd);
  llvm::Value *NewV =
      Builder.CreateInsertElement(LLVMVec, LLVMNewElt, LLVMIdx, Name);
  if (auto *NewInsert = dyn_cast<llvm::InsertElementInst>(NewV))
    return Ctx.createInsertElementInst(NewInsert);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

#ifndef NDEBUG
void InsertElementInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void InsertElementInst::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void InsertElementInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void InsertElementInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Value *ExtractElementInst::create(Value *Vec, Value *Idx,
                                  Instruction *InsertBefore, Context &Ctx,
                                  const Twine &Name) {
  llvm::Value *LLVMVec = ValueAttorney::getValue(Vec);
  llvm::Value *LLVMIdx = ValueAttorney::getValue(Idx);
  llvm::Instruction *LLVMInsertBefore =
      cast<llvm::Instruction>(ValueAttorney::getValue(InsertBefore));
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertBefore);
  llvm::Value *NewV = Builder.CreateExtractElement(LLVMVec, LLVMIdx, Name);
  if (auto *NewExtract = dyn_cast<llvm::ExtractElementInst>(NewV))
    return Ctx.createExtractElementInst(NewExtract);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *ExtractElementInst::create(Value *Vec, Value *Idx,
                                  BasicBlock *InsertAtEnd, Context &Ctx,
                                  const Twine &Name) {
  llvm::Value *LLVMVec = ValueAttorney::getValue(Vec);
  llvm::Value *LLVMIdx = ValueAttorney::getValue(Idx);
  llvm::BasicBlock *LLVMInsertAtEnd =
      cast<llvm::BasicBlock>(BasicBlockAttorney::getBB(InsertAtEnd));
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertAtEnd);
  llvm::Value *NewV = Builder.CreateExtractElement(LLVMVec, LLVMIdx, Name);
  if (auto *NewExtract = dyn_cast<llvm::ExtractElementInst>(NewV))
    return Ctx.createExtractElementInst(NewExtract);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

#ifndef NDEBUG
void ExtractElementInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void ExtractElementInst::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void ExtractElementInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void ExtractElementInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Value *ShuffleVectorInst::create(Value *V1, Value *V2, Value *Mask,
                                 Instruction *InsertBefore, Context &Ctx,
                                 const Twine &Name) {
  llvm::Value *LLVMV1 = ValueAttorney::getValue(V1);
  llvm::Value *LLVMV2 = ValueAttorney::getValue(V2);
  llvm::Value *LLVMMask = ValueAttorney::getValue(Mask);
  llvm::Instruction *LLVMInsertBefore =
      cast<llvm::Instruction>(ValueAttorney::getValue(InsertBefore));
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertBefore);
  llvm::Value *NewV =
      Builder.CreateShuffleVector(LLVMV1, LLVMV2, LLVMMask, Name);
  if (auto *NewShuffleVec = dyn_cast<llvm::ShuffleVectorInst>(NewV))
    return Ctx.createShuffleVectorInst(NewShuffleVec);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *ShuffleVectorInst::create(Value *V1, Value *V2, Value *Mask,
                                 BasicBlock *InsertAtEnd, Context &Ctx,
                                 const Twine &Name) {
  llvm::Value *LLVMV1 = ValueAttorney::getValue(V1);
  llvm::Value *LLVMV2 = ValueAttorney::getValue(V2);
  llvm::Value *LLVMMask = ValueAttorney::getValue(Mask);
  llvm::BasicBlock *LLVMInsertAtEnd =
      cast<llvm::BasicBlock>(BasicBlockAttorney::getBB(InsertAtEnd));
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertAtEnd);
  llvm::Value *NewV =
      Builder.CreateShuffleVector(LLVMV1, LLVMV2, LLVMMask, Name);
  if (auto *NewShuffleVec = dyn_cast<llvm::ShuffleVectorInst>(NewV))
    return Ctx.createShuffleVectorInst(NewShuffleVec);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *ShuffleVectorInst::create(Value *V1, Value *V2, ArrayRef<int> Mask,
                                 Instruction *InsertBefore, Context &Ctx,
                                 const Twine &Name) {
  llvm::Value *LLVMV1 = ValueAttorney::getValue(V1);
  llvm::Value *LLVMV2 = ValueAttorney::getValue(V2);
  llvm::Instruction *LLVMInsertBefore =
      cast<llvm::Instruction>(ValueAttorney::getValue(InsertBefore));
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertBefore);
  llvm::Value *NewV = Builder.CreateShuffleVector(LLVMV1, LLVMV2, Mask, Name);
  if (auto *NewShuffleVec = dyn_cast<llvm::ShuffleVectorInst>(NewV))
    return Ctx.createShuffleVectorInst(NewShuffleVec);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *ShuffleVectorInst::create(Value *V1, Value *V2, ArrayRef<int> Mask,
                                 BasicBlock *InsertAtEnd, Context &Ctx,
                                 const Twine &Name) {
  llvm::Value *LLVMV1 = ValueAttorney::getValue(V1);
  llvm::Value *LLVMV2 = ValueAttorney::getValue(V2);
  llvm::BasicBlock *LLVMInsertAtEnd =
      cast<llvm::BasicBlock>(BasicBlockAttorney::getBB(InsertAtEnd));
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(LLVMInsertAtEnd);
  llvm::Value *NewV = Builder.CreateShuffleVector(LLVMV1, LLVMV2, Mask, Name);
  if (auto *NewShuffleVec = dyn_cast<llvm::ShuffleVectorInst>(NewV))
    return Ctx.createShuffleVectorInst(NewShuffleVec);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

#ifndef NDEBUG
void ShuffleVectorInst::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void ShuffleVectorInst::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void ShuffleVectorInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void ShuffleVectorInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

void Function::detachFromLLVMIR() {
  for (BasicBlock &SBBB : *this)
    SBBB.detachFromLLVMIR();
  // Detach the actual SBFunction.
  Ctx.detach(this);
}

#ifndef NDEBUG
void Function::dumpNameAndArgs(raw_ostream &OS) const {
  llvm::Function *F = getFunction();
  OS << *getType() << " @" << F->getName() << "(";
  auto NumArgs = F->arg_size();
  for (auto [Idx, Arg] : enumerate(F->args())) {
    auto *SBArg = cast_or_null<Argument>(Ctx.getValue(&Arg));
    if (SBArg == nullptr)
      OS << "NULL";
    else
      SBArg->printAsOperand(OS);
    if (Idx + 1 < NumArgs)
      OS << ", ";
  }
  OS << ")";
}
void Function::dump(raw_ostream &OS) const {
  dumpNameAndArgs(OS);
  OS << " {\n";
  llvm::Function *F = getFunction();
  llvm::BasicBlock &LastBB = F->back();
  for (llvm::BasicBlock &BB : *F) {
    auto *SBBB = cast_or_null<BasicBlock>(Ctx.getValue(&BB));
    if (SBBB == nullptr)
      OS << "NULL";
    else
      OS << *SBBB;
    if (&BB != &LastBB)
      OS << "\n";
  }
  OS << "}\n";
}
void Function::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
void Function::dumpVerbose(raw_ostream &OS) const {
  dumpNameAndArgs(OS);
  OS << " {\n";
  for (llvm::BasicBlock &BB : *getFunction()) {
    auto *SBBB = cast_or_null<BasicBlock>(Ctx.getValue(&BB));
    if (SBBB == nullptr)
      OS << "NULL";
    else
      SBBB->dumpVerbose(OS);
    OS << "\n";
  }
  OS << "}\n";
}
void Function::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
#endif

Context::Context(LLVMContext &LLVMCtx)
    : LLVMCtx(LLVMCtx), LLVMIRBuilder(LLVMCtx, ConstantFolder()) {}

std::unique_ptr<Value> Context::detachValue(llvm::Value *V) {
  std::unique_ptr<Value> Erased;
  auto It = LLVMValueToValueMap.find(V);
  if (It != LLVMValueToValueMap.end()) {
    auto *Val = It->second.release();
    Erased = std::unique_ptr<Value>(Val);
    LLVMValueToValueMap.erase(It);
  }
  MultiInstrMap.erase(V);
  return Erased;
}

Value *Context::getValue(llvm::Value *V) const {
  // In the common case we should find the value in LLVMValueToValueMap.
  auto It = LLVMValueToValueMap.find(V);
  if (It != LLVMValueToValueMap.end())
    return It->second.get();
  // Instrs that map to multiple IR Instrs (like Packs) use a second map.
  auto It2 = MultiInstrMap.find(V);
  if (It2 != MultiInstrMap.end()) {
    llvm::Value *Key = It2->second;
    assert(Key != V && "Bad entry in MultiInstrMap!");
    return getValue(Key);
  }
  return nullptr;
}

Constant *Context::getConstant(llvm::Constant *C) const {
  return cast_or_null<Constant>(getValue(C));
}

Constant *Context::getOrCreateConstant(llvm::Constant *C) {
  auto Pair = LLVMValueToValueMap.insert({C, nullptr});
  auto It = Pair.first;
  if (Pair.second) {
    It->second = std::unique_ptr<Constant>(new Constant(C, *this));
    return cast<Constant>(It->second.get());
  }
  return cast<Constant>(It->second.get());
}

std::unique_ptr<Value> Context::detach(Value *SBV) {
  if (auto *SBI = dyn_cast<Instruction>(SBV))
    SBI->detachExtras();
#ifndef NDEBUG
  switch (SBV->getSubclassID()) {
  case Value::ClassID::Constant:
    llvm_unreachable("Can't detach a constant!");
    break;
  case Value::ClassID::User:
    llvm_unreachable("Can't detach a user!");
    break;
  default:
    break;
  }
#endif
  llvm::Value *V = ValueAttorney::getValue(SBV);
  return detachValue(V);
}

Value *Context::registerValue(std::unique_ptr<Value> &&SBVPtr) {
  auto &Tracker = getTracker();
  if (Tracker.tracking() && isa<Instruction>(SBVPtr.get()))
    Tracker.track(std::make_unique<CreateAndInsertInstr>(
        cast<Instruction>(SBVPtr.get()), Tracker));

  assert(SBVPtr->getSubclassID() != Value::ClassID::User &&
         "Can't register a user!");
  Value *V = SBVPtr.get();
  llvm::Value *Key = ValueAttorney::getValue(V);
  LLVMValueToValueMap[Key] = std::move(SBVPtr);
  // For multi-LLVM-Instruction SBInstructrions we also need to map the rest.
  if (auto *I = dyn_cast<Instruction>(V)) {
    auto LLVMInstrs = I->getLLVMInstrs();
    for (auto *InternalI : drop_begin(reverse(LLVMInstrs)))
      MultiInstrMap[InternalI] = Key;

    if (!DontNumberInstrs)
      // Number the instruction
      I->getParent()->assignInstrNumber(I);
    runInsertInstrCallbacks(I);
  }
  return V;
}

void Context::createMissingConstantOperands(llvm::Value *V) {
  // Create SandboxIR for all new constant operands.
  if (llvm::User *U = dyn_cast<llvm::User>(V)) {
    for (llvm::Value *Op : U->operands()) {
      if (auto *ConstOp = dyn_cast<llvm::Constant>(Op))
        getOrCreateConstant(ConstOp);
    }
  }
}

// BranchInst
BranchInst *Context::getBranchInst(llvm::BranchInst *BI) const {
  return cast_or_null<BranchInst>(getValue(BI));
}

BranchInst *Context::createBranchInst(llvm::BranchInst *BI) {
  assert(BI->getParent() != nullptr && "Detached!");
  assert(getBranchInst(BI) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<BranchInst>(new BranchInst(BI, *this));
  return cast<BranchInst>(registerValue(std::move(NewPtr)));
}

BranchInst *Context::getOrCreateBranchInst(llvm::BranchInst *BI) {
  if (auto *SBBI = getBranchInst(BI))
    return SBBI;
  return createBranchInst(BI);
}

// Store
StoreInst *Context::getStoreInst(llvm::StoreInst *SI) const {
  return cast_or_null<StoreInst>(getValue(SI));
}

StoreInst *Context::createStoreInst(llvm::StoreInst *SI) {
  assert(SI->getParent() != nullptr && "Detached!");
  assert(getStoreInst(SI) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<StoreInst>(new StoreInst(SI, *this));
  return cast<StoreInst>(registerValue(std::move(NewPtr)));
}

StoreInst *Context::getOrCreateStoreInst(llvm::StoreInst *SI) {
  if (auto *SBSI = getStoreInst(SI))
    return SBSI;
  return createStoreInst(SI);
}

// Load
LoadInst *Context::getLoadInst(llvm::LoadInst *LI) const {
  return cast_or_null<LoadInst>(getValue(LI));
}

LoadInst *Context::createLoadInst(llvm::LoadInst *LI) {
  assert(getLoadInst(LI) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<LoadInst>(new LoadInst(LI, *this));
  return cast<LoadInst>(registerValue(std::move(NewPtr)));
}

LoadInst *Context::getOrCreateLoadInst(llvm::LoadInst *LI) {
  if (auto *SBLI = getLoadInst(LI))
    return SBLI;
  return createLoadInst(LI);
}

// Cast
CastInst *Context::getCastInst(llvm::CastInst *CI) const {
  return cast_or_null<CastInst>(getValue(CI));
}

CastInst *Context::createCastInst(llvm::CastInst *CI) {
  assert(getCastInst(CI) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<CastInst>(new CastInst(CI, *this));
  return cast<CastInst>(registerValue(std::move(NewPtr)));
}

CastInst *Context::getOrCreateCastInst(llvm::CastInst *CI) {
  if (auto *SBCI = getCastInst(CI))
    return SBCI;
  return createCastInst(CI);
}

// PHI
PHINode *Context::getPHINode(llvm::PHINode *PHI) const {
  return cast_or_null<PHINode>(getValue(PHI));
}

PHINode *Context::createPHINode(llvm::PHINode *PHI) {
  assert(getPHINode(PHI) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<PHINode>(new PHINode(PHI, *this));
  return cast<PHINode>(registerValue(std::move(NewPtr)));
}

PHINode *Context::getOrCreatePHINode(llvm::PHINode *PHI) {
  if (auto *SBPHI = getPHINode(PHI))
    return SBPHI;
  return createPHINode(PHI);
}

// Select
SelectInst *Context::getSelectInst(llvm::SelectInst *SI) const {
  return cast_or_null<SelectInst>(getValue(SI));
}

SelectInst *Context::createSelectInst(llvm::SelectInst *SI) {
  assert(getSelectInst(SI) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<SelectInst>(new SelectInst(SI, *this));
  return cast<SelectInst>(registerValue(std::move(NewPtr)));
}

SelectInst *Context::getOrCreateSelectInst(llvm::SelectInst *SI) {
  if (auto *SBSI = getSelectInst(SI))
    return SBSI;
  return createSelectInst(SI);
}

// BinaryOperator
BinaryOperator *Context::getBinaryOperator(llvm::BinaryOperator *BO) const {
  return cast_or_null<BinaryOperator>(getValue(BO));
}

BinaryOperator *Context::createBinaryOperator(llvm::BinaryOperator *BO) {
  assert(getBinaryOperator(BO) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<BinaryOperator>(new BinaryOperator(BO, *this));
  return cast<BinaryOperator>(registerValue(std::move(NewPtr)));
}

BinaryOperator *Context::getOrCreateBinaryOperator(llvm::BinaryOperator *BO) {
  if (auto *SBBO = getBinaryOperator(BO))
    return SBBO;
  return createBinaryOperator(BO);
}

// UnaryOperator
UnaryOperator *Context::getUnaryOperator(llvm::UnaryOperator *UO) const {
  return cast_or_null<UnaryOperator>(getValue(UO));
}

UnaryOperator *Context::createUnaryOperator(llvm::UnaryOperator *UO) {
  assert(getUnaryOperator(UO) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<UnaryOperator>(new UnaryOperator(UO, *this));
  return cast<UnaryOperator>(registerValue(std::move(NewPtr)));
}

UnaryOperator *Context::getOrCreateUnaryOperator(llvm::UnaryOperator *UO) {
  if (auto *SBUO = getUnaryOperator(UO))
    return SBUO;
  return createUnaryOperator(UO);
}

// Cmp
CmpInst *Context::getCmpInst(llvm::CmpInst *CI) const {
  return cast_or_null<CmpInst>(getValue(CI));
}

CmpInst *Context::createCmpInst(llvm::CmpInst *CI) {
  assert(getCmpInst(CI) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<CmpInst>(new CmpInst(CI, *this));
  return cast<CmpInst>(registerValue(std::move(NewPtr)));
}

CmpInst *Context::getOrCreateCmpInst(llvm::CmpInst *CI) {
  if (auto *SBCI = getCmpInst(CI))
    return SBCI;
  return createCmpInst(CI);
}

Value *Context::getOrCreateValue(llvm::Value *V) {
  return getOrCreateValueInternal(V, 0);
}

Value *Context::getOrCreateValueInternal(llvm::Value *V, int Depth,
                                         llvm::User *U) {
  assert(Depth < 666 && "Infinite recursion?");
  // TODO: Use switch-case with subclass IDs instead of `if`.
  if (auto *C = dyn_cast<llvm::Constant>(V)) {
    // Globals may be self-referencing, like @bar = global [1 x ptr] [ptr @bar].
    // Avoid infinite loops by early returning once we detect a loop.
    if (isa<GlobalValue>(C)) {
      if (Depth == 0)
        VisitedConstants.clear();
      if (!VisitedConstants.insert(C).second)
        return nullptr; //  recursion loop!
    }
    for (llvm::Value *COp : C->operands())
      getOrCreateValueInternal(COp, Depth + 1, C);
    return getOrCreateConstant(C);
  }
  if (auto *Arg = dyn_cast<llvm::Argument>(V)) {
    return getOrCreateArgument(Arg);
  }
  if (auto *BB = dyn_cast<llvm::BasicBlock>(V)) {
    assert(isa<BlockAddress>(U) &&
           "This won't create a SBBB, don't call this function directly!");
    if (auto *SBBB = getValue(BB))
      return SBBB;
    // TODO: return a SBOpaqueValue
    return nullptr;
  }
  assert(isa<llvm::Instruction>(V) && "Expected Instruction");
  switch (cast<llvm::Instruction>(V)->getOpcode()) {
  case llvm::Instruction::PHI:
    return getOrCreatePHINode(cast<llvm::PHINode>(V));
  case llvm::Instruction::ExtractElement:
    return getOrCreateValueFromExtractElement(cast<llvm::ExtractElementInst>(V),
                                              Depth);
  case llvm::Instruction::ExtractValue:
    goto opaque_label;
  case llvm::Instruction::InsertElement:
    return getOrCreateValueFromInsertElement(cast<llvm::InsertElementInst>(V),
                                             Depth);
  case llvm::Instruction::ZExt:
  case llvm::Instruction::SExt:
  case llvm::Instruction::FPToUI:
  case llvm::Instruction::FPToSI:
  case llvm::Instruction::FPExt:
  case llvm::Instruction::PtrToInt:
  case llvm::Instruction::IntToPtr:
  case llvm::Instruction::SIToFP:
  case llvm::Instruction::UIToFP:
  case llvm::Instruction::Trunc:
  case llvm::Instruction::FPTrunc:
  case llvm::Instruction::BitCast:
    return getOrCreateCastInst(cast<llvm::CastInst>(V));
  case llvm::Instruction::FCmp:
  case llvm::Instruction::ICmp:
    return getOrCreateCmpInst(cast<llvm::CmpInst>(V));
  case llvm::Instruction::Select:
    return getOrCreateSelectInst(cast<llvm::SelectInst>(V));
  case llvm::Instruction::FNeg:
    return getOrCreateUnaryOperator(cast<llvm::UnaryOperator>(V));
  case llvm::Instruction::Add:
  case llvm::Instruction::FAdd:
  case llvm::Instruction::Sub:
  case llvm::Instruction::FSub:
  case llvm::Instruction::Mul:
  case llvm::Instruction::FMul:
  case llvm::Instruction::UDiv:
  case llvm::Instruction::SDiv:
  case llvm::Instruction::FDiv:
  case llvm::Instruction::URem:
  case llvm::Instruction::SRem:
  case llvm::Instruction::FRem:
  case llvm::Instruction::Shl:
  case llvm::Instruction::LShr:
  case llvm::Instruction::AShr:
  case llvm::Instruction::And:
  case llvm::Instruction::Or:
  case llvm::Instruction::Xor:
    return getOrCreateBinaryOperator(cast<llvm::BinaryOperator>(V));
  case llvm::Instruction::Load:
    return getOrCreateLoadInst(cast<llvm::LoadInst>(V));
  case llvm::Instruction::Store:
    return getOrCreateStoreInst(cast<llvm::StoreInst>(V));
  case llvm::Instruction::GetElementPtr:
    return getOrCreateGetElementPtrInst(cast<llvm::GetElementPtrInst>(V));
  case llvm::Instruction::Call:
    return getOrCreateCallInst(cast<llvm::CallInst>(V));
  case llvm::Instruction::ShuffleVector:
    return getOrCreateValueFromShuffleVector(cast<llvm::ShuffleVectorInst>(V),
                                             Depth);
  case llvm::Instruction::Ret:
    return getOrCreateRetInst(cast<llvm::ReturnInst>(V));
  case llvm::Instruction::Br:
    return getOrCreateBranchInst(cast<llvm::BranchInst>(V));
  default:
  opaque_label:
    return getOrCreateOpaqueInstruction(cast<llvm::Instruction>(V));
  }
}

void BasicBlock::renumberInstructions() {
  int64_t Num = 0;
  for (Instruction &IRef : *this) {
    InstrNumberMap[&IRef] = Num;
    Num += InstrNumberingStep;
  }
}

void BasicBlock::assignInstrNumber(Instruction *I) {
  int64_t Num;
  assert(I->getParent() && "Expected a parent block!");
  if (I->getNextNode() == nullptr) {
    // Inserting at the end of the block.
    if (empty())
      Num = 0;
    else {
      assert(I->getPrevNode() != nullptr && "Handle by empty()");
      auto PrevNum = getInstrNumber(I->getPrevNode());
      assert(PrevNum < std::numeric_limits<decltype(PrevNum)>::max() -
                           InstrNumberingStep &&
             "You're gonna need a bigger boat!");
      Num = PrevNum + InstrNumberingStep;
    }
  } else if (I->getPrevNode() == nullptr) {
    // Inserting at the beginning of the block.
    Instruction *NextI = I->getNextNode();
    assert(NextI != nullptr &&
           "Should've been handled by `if (I->getNextNode() == null)`");
    auto NextNum = getInstrNumber(NextI);
    assert(NextNum > std::numeric_limits<decltype(NextNum)>::min() +
                         InstrNumberingStep &&
           "You're gonna need a bigger boat!");
    Num = NextNum - InstrNumberingStep;
  } else {
    // Inserting between two instructions.
    auto GetNum = [this](Instruction *I) -> std::optional<int64_t> {
      auto *NextI = I->getNextNode();
      auto *PrevI = I->getPrevNode();
      assert(NextI != nullptr && PrevI != nullptr && "Expected next and prev");
      int64_t NextNum = getInstrNumber(NextI);
      int64_t PrevNum = getInstrNumber(PrevI);
      int64_t NewNum = (PrevNum + NextNum) / 2;
      bool LargeEnoughGap = NewNum != PrevNum && NewNum != NextNum;
      if (!LargeEnoughGap)
        return std::nullopt;
      return NewNum;
    };
    auto NumOpt = GetNum(I);
    if (!NumOpt) {
      renumberInstructions();
      NumOpt = GetNum(I);
      assert(NumOpt && "Expected a large enough gap after renumbering");
    }
    Num = *NumOpt;
  }
  InstrNumberMap[I] = Num;
}

void BasicBlock::removeInstrNumber(Instruction *I) { InstrNumberMap.erase(I); }

bool BasicBlock::classof(const Value *From) {
  return From->getSubclassID() == Value::ClassID::Block;
}

void BasicBlock::buildBasicBlockFromIR(llvm::BasicBlock *BB) {
  for (llvm::Instruction &IRef : reverse(*BB)) {
    llvm::Instruction *I = &IRef;
    Value *SBV = Ctx.getOrCreateValue(I);
    for (auto [OpIdx, Op] : enumerate(I->operands())) {
      // For now Unpacks only have a single operand.
      if (OpIdx > 0 && isa<Instruction>(SBV) &&
          cast<Instruction>(SBV)->getNumOperands() == 1)
        continue;
      // Skip instruction's label operands
      if (isa<llvm::BasicBlock>(Op))
        continue;
      // Skip metadata for now
      if (isa<llvm::MetadataAsValue>(Op))
        continue;
      // Skip asm
      if (isa<llvm::InlineAsm>(Op))
        continue;
      Ctx.getOrCreateValue(Op);
    }
  }
  // Instruction numbering
  renumberInstructions();
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  verify();
#endif
}

BasicBlock::iterator BasicBlock::getFirstNonPHIIt() {
  llvm::Instruction *FirstI = cast<llvm::BasicBlock>(Val)->getFirstNonPHI();
  return FirstI == nullptr
             ? begin()
             : cast<Instruction>(Ctx.getValue(FirstI))->getIterator();
}

BasicBlock::BasicBlock(llvm::BasicBlock *BB, Context &Ctx)
    : Value(ClassID::Block, BB, Ctx) {}

BasicBlock::~BasicBlock() {
  Ctx.destroyingBB(*this);
  // This BB is now gone, so there is no need for the BB-specific callbacks.
  Ctx.RemoveInstrCallbacksBB.erase(this);
  Ctx.InsertInstrCallbacksBB.erase(this);
  Ctx.MoveInstrCallbacksBB.erase(this);
}

Function *BasicBlock::getParent() const {
  auto *BB = cast<llvm::BasicBlock>(Val);
  auto *F = BB->getParent();
  if (F == nullptr)
    // Detached
    return nullptr;
  return Ctx.getFunction(F);
}

BasicBlock::iterator BasicBlock::begin() const {
  llvm::BasicBlock *BB = cast<llvm::BasicBlock>(Val);
  llvm::BasicBlock::iterator It = BB->begin();
  if (!BB->empty()) {
    auto *SBV = Ctx.getValue(&*BB->begin());
    assert(SBV != nullptr && "No SandboxIR for BB->begin()!");
    auto *SBI = cast<Instruction>(SBV);
    unsigned Num = SBI->getNumOfIRInstrs();
    assert(Num >= 1u && "Bad getNumOfIRInstrs()");
    It = std::next(It, Num - 1);
  }
  return iterator(BB, It, &Ctx);
}

void BasicBlock::detach() {
  // We are detaching bottom-up because detaching some SandboxIR
  // Instructions require non-detached operands.
  // Note: we are in the process of detaching from the underlying BB, so we
  //       can't rely on 1-1 mapping between IR and SandboxIR.
  for (llvm::Instruction &I : reverse(*cast<llvm::BasicBlock>(Val))) {
    if (auto *SI = Ctx.getValue(&I))
      Ctx.detach(SI);
  }
}

void BasicBlock::detachFromLLVMIR() {
  // Detach instructions
  detach();
  // Detach the actual BB
  Ctx.detach(this);
}

Argument *Context::getArgument(llvm::Argument *Arg) const {
  return cast_or_null<Argument>(getValue(Arg));
}

Argument *Context::createArgument(llvm::Argument *Arg) {
  assert(getArgument(Arg) == nullptr && "Already exists!");
  auto NewArg = std::unique_ptr<Argument>(new Argument(Arg, *this));
  return cast<Argument>(registerValue(std::move(NewArg)));
}

Argument *Context::getOrCreateArgument(llvm::Argument *Arg) {
  // TODO: Try to avoid two lookups in getOrCreate functions.
  if (auto *TArg = getArgument(Arg))
    return TArg;
  return createArgument(Arg);
}

Value *
Context::getValueFromExtractElement(llvm::ExtractElementInst *ExtractI) const {
  return getValue(ExtractI);
}
Value *
Context::getOrCreateValueFromExtractElement(llvm::ExtractElementInst *ExtractI,
                                            int Depth) {
  if (auto *SBV = getValueFromExtractElement(ExtractI))
    return SBV;
  return createValueFromExtractElement(ExtractI, Depth);
}

Value *
Context::getValueFromInsertElement(llvm::InsertElementInst *InsertI) const {
  return getValue(InsertI);
}
Value *
Context::getOrCreateValueFromInsertElement(llvm::InsertElementInst *InsertI,
                                           int Depth) {
  if (auto *SBV = getValueFromInsertElement(InsertI))
    return SBV;
  return createValueFromInsertElement(InsertI, Depth);
}

Value *
Context::getValueFromShuffleVector(llvm::ShuffleVectorInst *ShuffleI) const {
  return getValue(ShuffleI);
}

Value *
Context::getOrCreateValueFromShuffleVector(llvm::ShuffleVectorInst *ShuffleI,
                                           int Depth) {
  if (auto *SBV = getValueFromShuffleVector(ShuffleI))
    return SBV;
  return createValueFromShuffleVector(ShuffleI, Depth);
}

OpaqueInst *Context::getOpaqueInstruction(llvm::Instruction *I) const {
  return cast_or_null<OpaqueInst>(getValue(I));
}

OpaqueInst *Context::createOpaqueInstruction(llvm::Instruction *I) {
  assert(getOpaqueInstruction(I) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<OpaqueInst>(new OpaqueInst(I, *this));
  return cast<OpaqueInst>(registerValue(std::move(NewPtr)));
}

OpaqueInst *Context::getOrCreateOpaqueInstruction(llvm::Instruction *I) {
  assert(!isa<llvm::Constant>(I) && "Please use getOrCreateConstant()");
  assert(!isa<llvm::Argument>(I) && "Please use getOrCreateArgument()");
  auto *SBV = getOpaqueInstruction(I);
  if (SBV != nullptr)
    return SBV;
  return createOpaqueInstruction(I);
}

InsertElementInst *
Context::getInsertElementInst(llvm::InsertElementInst *I) const {
  return cast_or_null<InsertElementInst>(getValue(I));
}

InsertElementInst *
Context::createInsertElementInst(llvm::InsertElementInst *I) {
  assert(getInsertElementInst(I) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<InsertElementInst>(new InsertElementInst(I, *this));
  return cast<InsertElementInst>(registerValue(std::move(NewPtr)));
}

InsertElementInst *
Context::getOrCreateInsertElementInst(llvm::InsertElementInst *I) {
  assert(!isa<llvm::Constant>(I) && "Please use getOrCreateConstant()");
  assert(!isa<llvm::Argument>(I) && "Please use getOrCreateArgument()");
  auto *SBV = getInsertElementInst(I);
  if (SBV != nullptr)
    return SBV;
  return createInsertElementInst(I);
}

ExtractElementInst *
Context::getExtractElementInst(llvm::ExtractElementInst *I) const {
  return cast_or_null<ExtractElementInst>(getValue(I));
}

ExtractElementInst *
Context::createExtractElementInst(llvm::ExtractElementInst *I) {
  assert(getExtractElementInst(I) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<ExtractElementInst>(new ExtractElementInst(I, *this));
  return cast<ExtractElementInst>(registerValue(std::move(NewPtr)));
}

ExtractElementInst *
Context::getOrCreateExtractElementInst(llvm::ExtractElementInst *I) {
  assert(!isa<llvm::Constant>(I) && "Please use getOrCreateConstant()");
  assert(!isa<llvm::Argument>(I) && "Please use getOrCreateArgument()");
  auto *SBV = getExtractElementInst(I);
  if (SBV != nullptr)
    return SBV;
  return createExtractElementInst(I);
}

ShuffleVectorInst *
Context::getShuffleVectorInst(llvm::ShuffleVectorInst *I) const {
  return cast_or_null<ShuffleVectorInst>(getValue(I));
}

ShuffleVectorInst *
Context::createShuffleVectorInst(llvm::ShuffleVectorInst *I) {
  assert(getShuffleVectorInst(I) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<ShuffleVectorInst>(new ShuffleVectorInst(I, *this));
  return cast<ShuffleVectorInst>(registerValue(std::move(NewPtr)));
}

ShuffleVectorInst *
Context::getOrCreateShuffleVectorInst(llvm::ShuffleVectorInst *I) {
  assert(!isa<llvm::Constant>(I) && "Please use getOrCreateConstant()");
  assert(!isa<llvm::Argument>(I) && "Please use getOrCreateArgument()");
  auto *SBV = getShuffleVectorInst(I);
  if (SBV != nullptr)
    return SBV;
  return createShuffleVectorInst(I);
}

RetInst *Context::getRetInst(llvm::ReturnInst *I) const {
  return cast_or_null<RetInst>(getValue(I));
}

RetInst *Context::createRetInst(llvm::ReturnInst *I) {
  assert(getRetInst(I) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<RetInst>(new RetInst(I, *this));
  return cast<RetInst>(registerValue(std::move(NewPtr)));
}

RetInst *Context::getOrCreateRetInst(llvm::ReturnInst *I) {
  assert(!isa<llvm::Constant>(I) && "Please use getOrCreateConstant()");
  assert(!isa<llvm::Argument>(I) && "Please use getOrCreateArgument()");
  auto *SBV = getRetInst(I);
  if (SBV != nullptr)
    return SBV;
  return createRetInst(I);
}

CallInst *Context::getCallInst(llvm::CallInst *I) const {
  return cast_or_null<CallInst>(getValue(I));
}

CallInst *Context::createCallInst(llvm::CallInst *I) {
  assert(getCallInst(I) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<CallInst>(new CallInst(I, *this));
  return cast<CallInst>(registerValue(std::move(NewPtr)));
}

CallInst *Context::getOrCreateCallInst(llvm::CallInst *I) {
  assert(!isa<llvm::Constant>(I) && "Please use getOrCreateConstant()");
  assert(!isa<llvm::Argument>(I) && "Please use getOrCreateArgument()");
  auto *SBV = getCallInst(I);
  if (SBV != nullptr)
    return SBV;
  return createCallInst(I);
}

GetElementPtrInst *
Context::getGetElementPtrInst(llvm::GetElementPtrInst *I) const {
  return cast_or_null<GetElementPtrInst>(getValue(I));
}

GetElementPtrInst *
Context::createGetElementPtrInst(llvm::GetElementPtrInst *I) {
  assert(getGetElementPtrInst(I) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<GetElementPtrInst>(new GetElementPtrInst(I, *this));
  return cast<GetElementPtrInst>(registerValue(std::move(NewPtr)));
}

GetElementPtrInst *
Context::getOrCreateGetElementPtrInst(llvm::GetElementPtrInst *I) {
  assert(!isa<llvm::Constant>(I) && "Please use getOrCreateConstant()");
  assert(!isa<llvm::Argument>(I) && "Please use getOrCreateArgument()");
  auto *SBV = getGetElementPtrInst(I);
  if (SBV != nullptr)
    return SBV;
  return createGetElementPtrInst(I);
}

BasicBlock *Context::getBasicBlock(llvm::BasicBlock *BB) const {
  return cast_or_null<BasicBlock>(getValue(BB));
}

BasicBlock *Context::createBasicBlock(llvm::BasicBlock *BB) {
  DontNumberInstrs = true;
  assert(getBasicBlock(BB) == nullptr && "Already exists!");
  auto NewBBPtr = std::unique_ptr<BasicBlock>(new BasicBlock(BB, *this));
  auto *SBBB = cast<BasicBlock>(registerValue(std::move(NewBBPtr)));
  // Create SandboxIR for BB's body.
  SBBB->buildBasicBlockFromIR(BB);

  // Run hook.
  createdBasicBlock(*SBBB);

  DontNumberInstrs = false;
  return SBBB;
}

Function *Context::getFunction(llvm::Function *F) const {
  return cast_or_null<Function>(getValue(F));
}

Function *Context::createFunction(llvm::Function *F, bool CreateBBs) {
  assert(getFunction(F) == nullptr && "Already exists!");
  auto NewFPtr = std::unique_ptr<Function>(new Function(F, *this));
  // Create arguments.
  for (auto &Arg : F->args())
    getOrCreateArgument(&Arg);
  // Create BBs.
  if (CreateBBs) {
    for (auto &BB : *F)
      createBasicBlock(&BB);
  }
  auto *SBF = cast<Function>(registerValue(std::move(NewFPtr)));
  return SBF;
}

#ifndef NDEBUG
void BasicBlock::verify() {
  auto *BB = cast<llvm::BasicBlock>(Val);
  for (llvm::Instruction &IRef : *BB) {
    // Check that all llvm instructions in BB have a corresponding SBValue.
    assert(getContext().getValue(&IRef) != nullptr && "No SBValue for IRef!");
  }

  Instruction *LastI = nullptr;
  unsigned CntInstrs = 0;
  for (Instruction &IRef : *this) {
    ++CntInstrs;
    // Check instrunction numbering.
    if (LastI != nullptr)
      assert(LastI->comesBefore(&IRef) && "Broken instruction numbering!");
    LastI = &IRef;
  }
  assert(InstrNumberMap.size() == CntInstrs &&
         "Forgot to add/remove instrs from map?");

  // Note: we are not simply doing bool HaveSandboxIRForWholeFn = getParent()
  // because there is the corner case of @function operands of constants.
  Value *SBF = Ctx.getValue(BB->getParent());
  bool HaveSandboxIRForWholeFn = SBF != nullptr && isa<Function>(SBF);
  // Check operand/user consistency.
  for (const Instruction &SBI : *this) {
    llvm::Value *V = ValueAttorney::getValue(&SBI);
    assert(!isa<llvm::BasicBlock>(V) && "Broken SBBasicBlock construction!");

    // Note: This is expensive for packs, so skip based on num of LLVM instrs.
    if (SBI.getNumOfIRInstrs() < 16) {
      for (auto [OpIdx, Use] : enumerate(SBI.operands())) {
        Value *Op = Use;
        if (HaveSandboxIRForWholeFn) {
          llvm::Value *LLVMOp = UseAttorney::getLLVMUse(Use)->get();
          if (isa<llvm::Instruction>(LLVMOp) || isa<llvm::Constant>(LLVMOp))
            assert(Op != nullptr && "Null instruction/constant operands are "
                                    "not allowed when we have "
                                    "SandboxIR for the whole function");
        }
        if (Op == nullptr)
          continue;
        // Op could be an operand of a ConstantVector. We don't model this.
        assert((isa<llvm::Constant>(ValueAttorney::getValue(Op)) ||
                find(Op->users(), &SBI) != Op->users().end()) &&
               "If Op is SBI's operand, then SBI should be in Op's users.");
        // Count how many times Op is found in operands.
        unsigned CntOpEdges = 0;
        for_each(SBI.operands(), [&CntOpEdges, Op](Value *TmpOp) {
          if (TmpOp == Op)
            ++CntOpEdges;
        });
        if (CntOpEdges > 1 && !isa<Constant>(Op)) {
          // Check that Op has `CntOp` users matching `SBI`.
          unsigned CntUserEdges = 0;
          for_each(Op->users(), [&CntUserEdges, &SBI](User *User) {
            if (User == &SBI)
              ++CntUserEdges;
          });
          assert(
              CntOpEdges == CntUserEdges &&
              "Broken IR! User edges count doesn't match operand edges count!");
        }
      }
    }
    for (auto *User : SBI.users()) {
      if (User == nullptr)
        continue;
      assert(find(User->operands(), &SBI) != User->operands().end() &&
             "If User is in SBI's users, then SBI should be in User's "
             "operands.");
    }
  }

  Instruction *LastNonPHI = nullptr;
  // Checks opcodes and other.
  for (Instruction &SBI : *this) {
    if (LLVM_UNLIKELY(SBI.isPad())) {
      assert(&SBI == &*getFirstNonPHIIt() &&
             "{Landing,Catch,Cleanup}Pad Instructions must be the non-PHI!");
    }
    if (isa<PHINode>(&SBI)) {
      if (LastNonPHI != nullptr) {
        errs() << "SBPHIs not grouped at top of BB!\n";
        errs() << SBI << "\n";
        errs() << *LastNonPHI << "\n";
        llvm_unreachable("Broken SandboxIR");
      }
    } else {
      LastNonPHI = &SBI;
    }
  }

  // Check that we only have a single SBValue for every constant.
  DenseMap<llvm::Value *, const Value *> Map;
  for (const Value &SBV : *this) {
    llvm::Value *V = ValueAttorney::getValue(&SBV);
    if (isa<llvm::Constant>(V)) {
      auto Pair = Map.insert({V, &SBV});
      if (!Pair.second) {
        auto It = Pair.first;
        assert(&SBV == It->second &&
               "Expected a unique SBValue for each LLVM IR constant!");
      }
    }
  }
}

void BasicBlock::verifyLLVMIR() const {
  // Check that all llvm instructions in BB have a corresponding SBValue.
  auto *BB = cast<llvm::BasicBlock>(Val);
  llvm::Instruction *LastI = nullptr;
  for (llvm::Instruction &IRef : *BB) {
    llvm::Instruction *I = &IRef;
    for (llvm::Value *Op : I->operands()) {
      auto *OpI = dyn_cast<llvm::Instruction>(Op);
      if (OpI == nullptr)
        continue;
      if (OpI->getParent() != BB)
        continue;
      if (!isa<llvm::PHINode>(I) && !OpI->comesBefore(I)) {
        errs() << "Instruction does not dominate uses!\n";
        errs() << *Op << " " << Op << "\n";
        errs() << *I << " " << I << "\n";
        errs() << "\n";

        errs() << "SBValues:\n";
        auto *SBOp = Ctx.getValue(Op);
        if (SBOp != nullptr)
          errs() << *SBOp << " " << SBOp << "\n";
        else
          errs() << "No SBValue for Op\n";
        auto *SBI = Ctx.getValue(I);
        if (SBI != nullptr)
          errs() << *SBI << " " << SBI << "\n";
        else
          errs() << "No SBValue for I\n";
        llvm_unreachable("Instruction does not dominate uses!");
      }
    }

    if (LastI != nullptr && isa<llvm::PHINode>(I) &&
        !isa<llvm::PHINode>(LastI)) {
      errs() << "PHIs not grouped at top of BB!\n";
      errs() << *LastI << " " << LastI << "\n";
      errs() << *I << " " << I << "\n";
      llvm_unreachable("PHIs not grouped at top of BB!\n");
    }
    LastI = I;
  }
}

void BasicBlock::dumpVerbose(raw_ostream &OS) const {
  for (const auto &SBI : reverse(*this)) {
    SBI.dumpVerbose(OS);
    OS << "\n";
  }
}
void BasicBlock::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
void BasicBlock::dump(raw_ostream &OS) const {
  llvm::BasicBlock *BB = cast<llvm::BasicBlock>(Val);
  const auto &Name = BB->getName();
  OS << Name;
  if (!Name.empty())
    OS << ":\n";
  // If there are Instructions in the BB that are not mapped to SandboxIR, then
  // use a crash-proof dump.
  if (any_of(*BB, [this](llvm::Instruction &I) {
        return Ctx.getValue(&I) == nullptr;
      })) {
    OS << "<Crash-proof mode!>\n";
    DenseSet<Instruction *> Visited;
    for (llvm::Instruction &IRef : *BB) {
      Value *SBV = Ctx.getValue(&IRef);
      if (SBV == nullptr)
        OS << IRef << " *** No SandboxIR ***\n";
      else {
        auto *SBI = dyn_cast<Instruction>(SBV);
        if (SBI == nullptr)
          OS << IRef << " *** Not a SBInstruction!!! ***\n";
        else {
          if (Visited.insert(SBI).second)
            OS << *SBI << "\n";
        }
      }
    }
  } else {
    for (auto &SBI : *this) {
      SBI.dump(OS);
      OS << "\n";
    }
  }
}
void BasicBlock::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

void BasicBlock::dumpInstrs(Value *SBV, int Num) const {
  auto *SBI = dyn_cast<Instruction>(SBV);
  if (SBI == nullptr) {
    dbgs() << "SBV == null!\n";
    return;
  }
  auto *FromI = SBI;
  for (auto Cnt : seq<int>(0, Num)) {
    (void)Cnt;
    auto *PrevI = FromI->getPrevNode();
    if (PrevI == nullptr)
      break;
    FromI = PrevI;
  }
  auto *ToI = SBI;
  for (auto Cnt : seq<int>(0, Num)) {
    (void)Cnt;
    auto *NextI = ToI->getNextNode();
    if (NextI == nullptr)
      break;
    ToI = NextI;
  }
  for (Instruction *I = FromI, *E = ToI->getNextNode(); I != E;
       I = I->getNextNode())
    dbgs() << *I << "\n";
}
#endif

SandboxIRTracker &BasicBlock::getTracker() { return Ctx.getTracker(); }

Instruction *BasicBlock::getTerminator() const {
  auto *TerminatorV =
      Ctx.getValue(cast<llvm::BasicBlock>(Val)->getTerminator());
  return cast_or_null<Instruction>(TerminatorV);
}

BasicBlock::iterator::pointer
BasicBlock::iterator::getI(llvm::BasicBlock::iterator It) const {
  Instruction *SBI = cast_or_null<Instruction>(Ctx->getValue(&*It));
  assert(
      (!SBI || cast<llvm::Instruction>(ValueAttorney::getValue(SBI)) == &*It) &&
      "It should always point at the bottom IR instruction of a "
      "SBInstruction!");
  return SBI;
}

Context::RemoveCBTy *Context::registerRemoveInstrCallback(RemoveCBTy CB) {
  std::unique_ptr<RemoveCBTy> CBPtr(new RemoveCBTy(CB));
  RemoveCBTy *CBRaw = CBPtr.get();
  RemoveInstrCallbacks.push_back(std::move(CBPtr));
  return CBRaw;
}

void Context::unregisterRemoveInstrCallback(RemoveCBTy *CB) {
  auto It = find_if(RemoveInstrCallbacks,
                    [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert((InQuickFlush || It != RemoveInstrCallbacks.end()) &&
         "Callback not registered!");
  if (It != RemoveInstrCallbacks.end())
    RemoveInstrCallbacks.erase(It);
}

Context::InsertCBTy *Context::registerInsertInstrCallback(InsertCBTy CB) {
  std::unique_ptr<InsertCBTy> CBPtr(new InsertCBTy(CB));
  InsertCBTy *CBRaw = CBPtr.get();
  InsertInstrCallbacks.push_back(std::move(CBPtr));
  return CBRaw;
}

void Context::unregisterInsertInstrCallback(InsertCBTy *CB) {
  auto It = find_if(InsertInstrCallbacks,
                    [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert(It != InsertInstrCallbacks.end() && "Callback not registered!");
  InsertInstrCallbacks.erase(It);
}

Context::MoveCBTy *Context::registerMoveInstrCallback(MoveCBTy CB) {
  std::unique_ptr<MoveCBTy> CBPtr(new MoveCBTy(CB));
  MoveCBTy *CBRaw = CBPtr.get();
  MoveInstrCallbacks.push_back(std::move(CBPtr));
  return CBRaw;
}

void Context::unregisterMoveInstrCallback(MoveCBTy *CB) {
  auto It = find_if(MoveInstrCallbacks,
                    [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert(It != MoveInstrCallbacks.end() && "Callback not registered!");
  MoveInstrCallbacks.erase(It);
}

Context::RemoveCBTy *Context::registerRemoveInstrCallbackBB(BasicBlock &BB,
                                                            RemoveCBTy CB) {
  std::unique_ptr<RemoveCBTy> CBPtr(new RemoveCBTy(CB));
  RemoveCBTy *CBRaw = CBPtr.get();
  RemoveInstrCallbacksBB[&BB].push_back(std::move(CBPtr));
  return CBRaw;
}

void Context::unregisterRemoveInstrCallbackBB(BasicBlock &BB, RemoveCBTy *CB) {
  auto MapIt = RemoveInstrCallbacksBB.find(&BB);
  assert((InQuickFlush || MapIt != RemoveInstrCallbacksBB.end()) &&
         "Callback not registered at BB!");
  auto &Vec = MapIt->second;
  auto It = find_if(Vec, [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert((InQuickFlush || It != Vec.end()) && "Callback not registered!");
  if (It != Vec.end())
    Vec.erase(It);
}

Context::InsertCBTy *Context::registerInsertInstrCallbackBB(BasicBlock &BB,
                                                            InsertCBTy CB) {
  std::unique_ptr<InsertCBTy> CBPtr(new InsertCBTy(CB));
  InsertCBTy *CBRaw = CBPtr.get();
  InsertInstrCallbacksBB[&BB].push_back(std::move(CBPtr));
  return CBRaw;
}

void Context::unregisterInsertInstrCallbackBB(BasicBlock &BB, InsertCBTy *CB) {
  auto MapIt = InsertInstrCallbacksBB.find(&BB);
  assert((InQuickFlush || MapIt != InsertInstrCallbacksBB.end()) &&
         "Callback not registered at BB!");
  auto &Vec = MapIt->second;
  auto It = find_if(Vec, [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert(It != Vec.end() && "Callback not registered!");
  Vec.erase(It);
}

Context::MoveCBTy *Context::registerMoveInstrCallbackBB(BasicBlock &BB,
                                                        MoveCBTy CB) {
  std::unique_ptr<MoveCBTy> CBPtr(new MoveCBTy(CB));
  MoveCBTy *CBRaw = CBPtr.get();
  MoveInstrCallbacksBB[&BB].push_back(std::move(CBPtr));
  return CBRaw;
}

void Context::unregisterMoveInstrCallbackBB(BasicBlock &BB, MoveCBTy *CB) {
  auto MapIt = MoveInstrCallbacksBB.find(&BB);
  assert((InQuickFlush || MapIt != MoveInstrCallbacksBB.end()) &&
         "Callback not registered at BB!");
  auto &Vec = MapIt->second;
  auto It = find_if(Vec, [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert(It != Vec.end() && "Callback not registered!");
  Vec.erase(It);
}

void Context::quickFlush() {
  InQuickFlush = true;

  RemoveInstrCallbacks.clear();
  InsertInstrCallbacks.clear();
  MoveInstrCallbacks.clear();

  RemoveInstrCallbacksBB.clear();
  InsertInstrCallbacksBB.clear();
  MoveInstrCallbacksBB.clear();

  LLVMValueToValueMap.clear();
  MultiInstrMap.clear();
  InQuickFlush = false;
}

void Context::runRemoveInstrCallbacks(Instruction *SBI) {
#ifndef NDEBUG
  if (CallbacksDisabled)
    return;
#endif
  for (auto &CBPtr : RemoveInstrCallbacks)
    (*CBPtr)(SBI);

  auto *BB = SBI->getParent();
  auto It = RemoveInstrCallbacksBB.find(BB);
  if (It != RemoveInstrCallbacksBB.end()) {
    for (auto &CBPtr : It->second)
      (*CBPtr)(SBI);
  }
}

void Context::runInsertInstrCallbacks(Instruction *SBI) {
#ifndef NDEBUG
  if (CallbacksDisabled)
    return;
#endif
  for (auto &CBPtr : InsertInstrCallbacks)
    (*CBPtr)(SBI);

  auto *BB = SBI->getParent();
  auto It = InsertInstrCallbacksBB.find(BB);
  if (It != InsertInstrCallbacksBB.end()) {
    for (auto &CBPtr : It->second)
      (*CBPtr)(SBI);
  }
}

void Context::runMoveInstrCallbacks(Instruction *SBI, BasicBlock &SBBB,
                                    const BBIterator &WhereIt) {
#ifndef NDEBUG
  if (CallbacksDisabled)
    return;
#endif
  for (auto &CBPtr : MoveInstrCallbacks)
    (*CBPtr)(SBI, SBBB, WhereIt);

  auto It = MoveInstrCallbacksBB.find(&SBBB);
  if (It != MoveInstrCallbacksBB.end()) {
    for (auto &CBPtr : It->second)
      (*CBPtr)(SBI, SBBB, WhereIt);
  }
}
