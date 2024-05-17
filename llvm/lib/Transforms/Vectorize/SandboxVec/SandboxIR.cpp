//===- SandboxIR.cpp - A transactional IR overlay of LLVM IR------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Utils.h"
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "SBVec"

SBValue *SBUse::get() const { return Ctxt->getSBValue(LLVMUse->get()); }
unsigned SBUse::getOperandNo() const {
  unsigned OpNo = User->getUseOperandNo(*this);
#ifndef NDEBUG
  if (!(isa<SBPackInstruction>(User) || isa<SBShuffleInstruction>(User) ||
        isa<SBUnpackInstruction>(User)))
    assert(LLVMUse->getOperandNo() == User->getUseOperandNo(*this) &&
           "OpIdx should match LLVM's OperandNo for simple instrs.");
#endif
  return OpNo;
}
#ifndef NDEBUG
void SBUse::dump(raw_ostream &OS) const {
  SBValue *Def = nullptr;
  if (LLVMUse == nullptr)
    OS << "<null> LLVM Use! ";
  else
    Def = Ctxt->getSBValue(LLVMUse->get());
  OS << "Def:  ";
  if (Def == nullptr)
    OS << "NULL";
  else
    OS << *Def;
  OS << "\n";

  OS << "User: ";
  if (User == nullptr)
    OS << "NULL";
  else
    OS << *User;
  OS << "\n";

  OS << "OperandNo: ";
  if (User == nullptr)
    OS << "N/A";
  else
    OS << getOperandNo();
  OS << "\n";
}

void SBUse::dump() const { dump(dbgs()); }
#endif // NDEBUG

SBUse SBOperandUseIterator::operator*() const { return Use; }

SBOperandUseIterator &SBOperandUseIterator::operator++() {
  assert(Use.LLVMUse != nullptr && "Already at end!");
  SBUser *User = Use.getUser();
  Use = User->getOperandUseInternal(Use.getOperandNo() + 1, /*Verify=*/false);
  return *this;
}

SBUse SBUserUseIterator::operator*() const { return Use; }

SBUserUseIterator &SBUserUseIterator::operator++() {
  llvm::Use *&LLVMUse = Use.LLVMUse;
  assert(LLVMUse != nullptr && "Already at end!");
  LLVMUse = LLVMUse->getNext();
  if (LLVMUse == nullptr) {
    Use.User = nullptr;
    return *this;
  }
  auto *Ctxt = Use.Ctxt;
  auto *LLVMUser = LLVMUse->getUser();
  SBUser *User = cast_or_null<SBUser>(Ctxt->getSBValue(LLVMUser));
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
  Use.User = User;
  return *this;
}

SBValue::SBValue(ClassID SubclassID, Value *Val, SBContext &Ctxt)
    : SubclassID(SubclassID), Val(Val), Ctxt(Ctxt) {
#ifndef NDEBUG
  UID = Ctxt.getNumValues();
#endif
}

SBValue::use_iterator SBValue::use_begin() {
  llvm::Use *LLVMUse = nullptr;
  if (Val->use_begin() != Val->use_end())
    LLVMUse = &*Val->use_begin();
  SBUser *User = LLVMUse != nullptr
                       ? cast_or_null<SBUser>(
                             Ctxt.getSBValue(Val->use_begin()->getUser()))
                       : nullptr;
  return use_iterator(SBUse(LLVMUse, User, Ctxt));
}

SBValue::user_iterator SBValue::user_begin() {
  auto UseBegin = Val->use_begin();
  auto UseEnd = Val->use_end();
  bool AtEnd = UseBegin == UseEnd;
  llvm::Use *LLVMUse = AtEnd ? nullptr : &*UseBegin;
  SBUser *User =
      AtEnd ? nullptr
            : cast_or_null<SBUser>(Ctxt.getSBValue(&*LLVMUse->getUser()));
  return user_iterator(SBUse(LLVMUse, User, Ctxt));
}

unsigned SBValue::getNumUsers() const {
  // Look for unique users.
  SmallPtrSet<SBValue *, 4> UserNs;
  for (User *U : Val->users())
    UserNs.insert(getContext().getSBValue(U));
  return UserNs.size();
}

unsigned SBValue::getNumUses() const {
  unsigned Cnt = 0;
  for (User *U : Val->users()) {
    (void)U;
    ++Cnt;
  }
  return Cnt;
}

bool SBValue::hasNUsersOrMore(unsigned Num) const {
  SmallPtrSet<SBValue *, 4> UserNs;
  for (User *U : Val->users()) {
    UserNs.insert(getContext().getSBValue(U));
    if (UserNs.size() >= Num)
      return true;
  }
  return false;
}

SBValue *SBValue::getSingleUser() const {
  assert(Val->hasOneUser() && "Expected single user");
  return *users().begin();
}

SBContext &SBValue::getContext() const { return Ctxt; }

SandboxIRTracker &SBValue::getTracker() { return getContext().getTracker(); }

void SBValue::replaceUsesWithIf(
    SBValue *OtherV,
    llvm::function_ref<bool(SBUser *DstU, unsigned OpIdx)> ShouldReplace) {
  assert(getType() == OtherV->getType() && "Can't replace with different type");
  Value *OtherVal = OtherV->Val;
  auto &Tracker = getTracker();
  Val->replaceUsesWithIf(
      OtherVal, [&ShouldReplace, &Tracker, this](Use &U) -> bool {
        SBUser *DstU = cast_or_null<SBUser>(Ctxt.getSBValue(U.getUser()));
        if (DstU == nullptr)
          return false;
        unsigned OpIdx = DstU->getOperandUseIdx(U);
        if (!ShouldReplace(DstU, OpIdx))
          return false;
        if (Tracker.tracking())
          // Tracking like so should be cheaper than replaceAllUsesWith()
          Tracker.track(std::make_unique<SetOperand>(DstU, OpIdx, Tracker));
        return true;
      });
}

void SBValue::replaceAllUsesWith(SBValue *Other) {
  assert(getType() == Other->getType() &&
         "Replacing with SBValue of different type!");
  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<ReplaceAllUsesWith>(this, Tracker));
  Val->replaceAllUsesWith(Other->Val);
}

#ifndef NDEBUG
std::string SBValue::getName() const {
  std::stringstream SS;
  SS << "T" << UID << ".";
  return SS.str();
}

void SBValue::dumpCommonHeader(raw_ostream &OS) const {
  OS << getName() << " " << getSubclassIDStr(SubclassID) << " ";
}

void SBValue::dumpCommonFooter(raw_ostream &OS) const {
  OS.indent(2) << "Val: ";
  if (Val)
    OS << *Val;
  else
    OS << "NULL";
  OS << "\n";

  // TODO: For now also dump users, but should be removed.
  if (!isa<Constant>(Val)) {
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

void SBValue::dumpCommonPrefix(raw_ostream &OS) const {
  if (Val)
    OS << *Val;
  else
    OS << "NULL ";
}

void SBValue::dumpCommonSuffix(raw_ostream &OS) const {
  OS << " ; " << getName() << " (" << getSubclassIDStr(SubclassID) << ") "
     << this;
}

void SBValue::printAsOperandCommon(raw_ostream &OS) const {
  if (Val)
    Val->printAsOperand(OS);
  else
    OS << "NULL ";
}

void SBValue::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBArgument::SBArgument(Argument *Arg, SBContext &SBCtxt)
    : SBValue(ClassID::Argument, Arg, SBCtxt) {}

SBUser::SBUser(ClassID ID, Value *V, SBContext &SBCtxt)
    : SBValue(ID, V, SBCtxt) {}

SBUse SBUser::getOperandUseDefault(unsigned OpIdx, bool Verify) const {
  assert((!Verify || OpIdx < getNumOperands()) && "Out of bounds!");
  assert(isa<User>(Val) && "Non-users have no operands!");
  llvm::Use *LLVMUse;
  if (OpIdx != getNumOperands())
    LLVMUse = &cast<User>(Val)->getOperandUse(OpIdx);
  else
    LLVMUse = cast<User>(Val)->op_end();
  return SBUse(LLVMUse, const_cast<SBUser *>(this), Ctxt);
}

bool SBUser::classof(const SBValue *From) {
  switch (From->getSubclassID()) {
  case ClassID::Pack:
  case ClassID::Unpack:
  case ClassID::Shuffle:
  case ClassID::OpaqueInstr:
  case ClassID::User:
  case ClassID::Constant:
  case ClassID::Store:
  case ClassID::Load:
  case ClassID::Cast:
  case ClassID::PHI:
  case ClassID::Select:
  case ClassID::BinOp:
  case ClassID::UnOp:
  case ClassID::Cmp:
    return true;
  case ClassID::Argument:
  case ClassID::Block:
  case ClassID::Function:
    return false;
  }
  return false;
}

SBUser::op_iterator SBUser::op_begin() {
  assert(isa<User>(Val) && "Expect User value!");
  return op_iterator(getOperandUseInternal(0, /*Verify=*/false));
}

SBUser::op_iterator SBUser::op_end() {
  assert(isa<User>(Val) && "Expect User value!");
  return op_iterator(getOperandUseInternal(getNumOperands(), /*Verify=*/false));
}

SBUser::const_op_iterator SBUser::op_begin() const {
  return const_cast<SBUser *>(this)->op_begin();
}

SBUser::const_op_iterator SBUser::op_end() const {
  return const_cast<SBUser *>(this)->op_end();
}

SBValue *SBUser::getSingleOperand() const {
  assert(getNumOperands() == 1 && "Expected exactly 1 operand");
  return getOperand(0);
}

void SBUser::setOperand(unsigned OperandIdx, SBValue *Operand) {
  if (!isa<User>(Val))
    llvm_unreachable("No operands!");
  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<SetOperand>(this, OperandIdx, Tracker));
  cast<User>(Val)->setOperand(OperandIdx, ValueAttorney::getValue(Operand));
}

bool SBUser::replaceUsesOfWith(SBValue *FromV, SBValue *ToV) {
  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(
        std::make_unique<ReplaceUsesOfWith>(this, FromV, ToV, Tracker));

  bool Change = false;
  Value *FromVIR = ValueAttorney::getValue(FromV);
  Value *ToVIR = ValueAttorney::getValue(ToV);
  if (auto *SBI = dyn_cast<SBInstruction>(Ctxt.getSBValue(Val))) {
    for (Instruction *I : SBI->getIRInstrs())
      Change |= I->replaceUsesOfWith(FromVIR, ToVIR);
    return Change;
  }
  return cast<User>(Val)->replaceUsesOfWith(FromVIR, ToVIR);
}

#ifndef NDEBUG
void SBUser::dumpCommonHeader(raw_ostream &OS) const {
  SBValue::dumpCommonHeader(OS);
  OS << "(";
  for (auto [OpIdx, Use] : enumerate(operands())) {
    SBValue *Op = Use;
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

unsigned SBUser::getOperandUseIdx(const Use &UseToMatch) const {
  // For all single-IR SandboxIR instrs the Use index matches the operand index.
  // This is not true for Multi-IR instructions, like Pack where multiple
  // Use edges correspond to a single operand index. Those cases need
  // SBInstruction-specific implementations.
  switch (SubclassID) {
  case ClassID::Pack:
    llvm_unreachable("Needs its own implementation!");
  case ClassID::Shuffle:
  case ClassID::Unpack:
  case ClassID::OpaqueInstr:
  case ClassID::Argument:
  case ClassID::User:
  case ClassID::Constant:
  case ClassID::Block:
  case ClassID::Store:
  case ClassID::Load:
  case ClassID::Cast:
  case ClassID::PHI:
  case ClassID::Select:
  case ClassID::BinOp:
  case ClassID::UnOp:
  case ClassID::Cmp:
  case ClassID::Function:
    assert(Ctxt.getSBValue(UseToMatch.getUser()) == this &&
           "Use not found in this SBUser's operands!");
    return UseToMatch.getOperandNo();
  }
}

SBBBIterator &SBBBIterator::operator++() {
  auto ItE = BB->end();
  assert(It != ItE && "Already at end!");
  ++It;
  if (It == ItE)
    return *this;
  SBInstruction &NextI = *cast<SBInstruction>(SBCtxt->getSBValue(&*It));
  unsigned Num = NextI.getNumOfIRInstrs();
  assert(Num > 0 && "Bad getNumOfIRInstrs()");
  It = std::next(It, Num - 1);
  return *this;
}

SBBBIterator &SBBBIterator::operator--() {
  assert(It != BB->begin() && "Already at begin!");
  if (It == BB->end()) {
    --It;
    return *this;
  }
  SBInstruction &CurrI = **this;
  unsigned Num = CurrI.getNumOfIRInstrs();
  assert(Num > 0 && "Bad getNumOfIRInstrs()");
  assert(std::prev(It, Num - 1) != BB->begin() && "Already at begin!");
  It = std::prev(It, Num);
  return *this;
}

SBInstruction::SBInstruction(ClassID ID, Instruction *I,
                                 SBContext &SBCtxt)
    : SBUser(ID, I, SBCtxt) {
  assert((!isa<StoreInst>(I) || SubclassID == ClassID::Store) &&
         "Create a SBStoreInstruction!");
  assert((!isa<LoadInst>(I) || SubclassID == ClassID::Load) &&
         "Create a SBLoadInstruction!");
  assert((!isa<CastInst>(I) || I->getOpcode() == Instruction::AddrSpaceCast ||
          SubclassID == ClassID::Cast) &&
         "Create a SBCastInstruction!");
  assert((!isa<PHINode>(I) || SubclassID == ClassID::PHI) &&
         "Create a SBPHINode!");
  assert((!isa<CmpInst>(I) || SubclassID == ClassID::Cmp) &&
         "Create a SBCmpInstruction!");
  assert((!isa<SelectInst>(I) || SubclassID == ClassID::Select) &&
         "Create a SBSelectInstruction!");
  assert((!isa<BinaryOperator>(I) || SubclassID == ClassID::BinOp) &&
         "Create a SBBinaryOperator!");
  assert((!isa<UnaryOperator>(I) || SubclassID == ClassID::UnOp) &&
         "Create a SBUnaryOperator!");
}

Instruction *SBInstruction::getTopmostIRInstruction() const {
  SBInstruction *Prev = getPrevNode();
  if (Prev == nullptr) {
    // If at top of the BB, return the first BB instruction.
    return &*cast<BasicBlock>(ValueAttorney::getValue(getParent()))->begin();
  }
  // Else get the Previous SB IR instruction's bottom IR instruction and
  // return its successor.
  Instruction *PrevBotI = cast<Instruction>(ValueAttorney::getValue(Prev));
  return PrevBotI->getNextNode();
}

SBBBIterator SBInstruction::getIterator() const {
  auto *I = cast<Instruction>(Val);
  return SBBasicBlock::iterator(I->getParent(), I->getIterator(), &Ctxt);
}

bool SBBBIterator::atBegin() const {
  // Fast path: if the internal iterator is at begin().
  if (It == BB->begin())
    return true;
  // We may still be at begin if this is a multi-IR SBInstruction and It is
  // pointing to its bottom IR Instr.
  unsigned NumInstrs = getSBI(It)->getNumOfIRInstrs();
  if (NumInstrs == 1)
    // This is a single-IR SBI. Since It != BB->begin() we are not at begin.
    return false;
  return std::prev(It, NumInstrs - 1) == BB->begin();
}

SBInstruction *SBInstruction::getNextNode() const {
  assert(getParent() != nullptr && "Detached!");
  assert(getIterator() != getParent()->end() && "Already at end!");
  auto *CurrI = cast<Instruction>(Val);
  assert(CurrI->getParent() != nullptr && "LLVM IR instr is detached!");
  auto *NextI = CurrI->getNextNode();
  auto *NextSBI = cast_or_null<SBInstruction>(Ctxt.getSBValue(NextI));
  if (NextSBI == nullptr)
    return nullptr;
  return NextSBI;
}

SBInstruction *SBInstruction::getPrevNode() const {
  assert(getParent() != nullptr && "Detached!");
  auto It = getIterator();
  if (!It.atBegin())
    return std::prev(getIterator()).get();
  return nullptr;
}

SBInstruction::Opcode SBInstruction::getOpcode() const {
  switch (cast<Instruction>(Val)->getOpcode()) {
  case Instruction::InsertElement:
    return SubclassID == ClassID::Pack ? Opcode::Pack : Opcode::Opaque;
  case Instruction::ExtractElement:
    return SubclassID == ClassID::Unpack ? Opcode::Unpack : Opcode::Opaque;
  case Instruction::ZExt:
    return Opcode::ZExt;
  case Instruction::SExt:
    return Opcode::SExt;
  case Instruction::FPToUI:
    return Opcode::FPToUI;
  case Instruction::FPToSI:
    return Opcode::FPToSI;
  case Instruction::FPExt:
    return Opcode::FPExt;
  case Instruction::PtrToInt:
    return Opcode::PtrToInt;
  case Instruction::IntToPtr:
    return Opcode::IntToPtr;
  case Instruction::SIToFP:
    return Opcode::SIToFP;
  case Instruction::UIToFP:
    return Opcode::UIToFP;
  case Instruction::Trunc:
    return Opcode::Trunc;
  case Instruction::FPTrunc:
    return Opcode::FPTrunc;
  case Instruction::BitCast:
    return Opcode::BitCast;
  case Instruction::FCmp:
    return Opcode::FCmp;
  case Instruction::ICmp:
    return Opcode::ICmp;
  case Instruction::Select:
    return Opcode::Select;
  case Instruction::FNeg:
    return Opcode::FNeg;
  case Instruction::Add:
    return Opcode::Add;
  case Instruction::FAdd:
    return Opcode::FAdd;
  case Instruction::Sub:
    return Opcode::Sub;
  case Instruction::FSub:
    return Opcode::FSub;
  case Instruction::Mul:
    return Opcode::Mul;
  case Instruction::FMul:
    return Opcode::FMul;
  case Instruction::UDiv:
    return Opcode::UDiv;
  case Instruction::SDiv:
    return Opcode::SDiv;
  case Instruction::FDiv:
    return Opcode::FDiv;
  case Instruction::URem:
    return Opcode::URem;
  case Instruction::SRem:
    return Opcode::SRem;
  case Instruction::FRem:
    return Opcode::FRem;
  case Instruction::Shl:
    return Opcode::Shl;
  case Instruction::LShr:
    return Opcode::LShr;
  case Instruction::AShr:
    return Opcode::AShr;
  case Instruction::And:
    return Opcode::And;
  case Instruction::Or:
    return Opcode::Or;
  case Instruction::Xor:
    return Opcode::Xor;
  case Instruction::Load:
    return Opcode::Load;
  case Instruction::Store:
    return Opcode::Store;

  case Instruction::GetElementPtr:
  case Instruction::Call:
  case Instruction::PHI:
  case Instruction::InsertValue:
  case Instruction::ExtractValue:
    return Opcode::Opaque;
  case Instruction::ShuffleVector: {
    if (SubclassID == ClassID::Shuffle)
      return Opcode::Shuffle;
    return Opcode::Opaque;
  }
  default:
    return Opcode::Opaque;
  }
}

Instruction::UnaryOps SBInstruction::getIRUnaryOp(Opcode Opc) {
  switch (Opc) {
  // Vector-related
  case Opcode::Shuffle:
  case Opcode::Pack:
  case Opcode::Unpack:
  // Casts
  case Opcode::ZExt:
  case Opcode::SExt:
  case Opcode::FPToUI:
  case Opcode::FPToSI:
  case Opcode::FPExt:
  case Opcode::PtrToInt:
  case Opcode::IntToPtr:
  case Opcode::SIToFP:
  case Opcode::UIToFP:
  case Opcode::Trunc:
  case Opcode::FPTrunc:
  case Opcode::BitCast:
  // Cmp
  case Opcode::FCmp:
  case Opcode::ICmp:
  // Select
  case Opcode::Select:
    llvm_unreachable("Not a unary op!");
  // Unary
  case Opcode::FNeg:
    return static_cast<Instruction::UnaryOps>(Instruction::FNeg);
  // BinOp
  case Opcode::Add:
  case Opcode::FAdd:
  case Opcode::Sub:
  case Opcode::FSub:
  case Opcode::Mul:
  case Opcode::FMul:
  case Opcode::UDiv:
  case Opcode::SDiv:
  case Opcode::FDiv:
  case Opcode::URem:
  case Opcode::SRem:
  case Opcode::FRem:
  case Opcode::Shl:
  case Opcode::LShr:
  case Opcode::AShr:
  case Opcode::And:
  case Opcode::Or:
  case Opcode::Xor:
  // Mem
  case Opcode::Load:
  case Opcode::Store:
  // Opaque for everything else
  case Opcode::Opaque:
    llvm_unreachable("Not a unary op!");
  }
}

Instruction::BinaryOps SBInstruction::getIRBinaryOp(Opcode Opc) {
  switch (Opc) {
  // Vector-related
  case Opcode::Shuffle:
  case Opcode::Pack:
  case Opcode::Unpack:
  // Casts
  case Opcode::ZExt:
  case Opcode::SExt:
  case Opcode::FPToUI:
  case Opcode::FPToSI:
  case Opcode::FPExt:
  case Opcode::PtrToInt:
  case Opcode::IntToPtr:
  case Opcode::SIToFP:
  case Opcode::UIToFP:
  case Opcode::Trunc:
  case Opcode::FPTrunc:
  case Opcode::BitCast:
  // Cmp
  case Opcode::FCmp:
  case Opcode::ICmp:
  // Select
  case Opcode::Select:
  // Unary
  case Opcode::FNeg:
    llvm_unreachable("Not a unary op!");
  // BinOp
  case Opcode::Add:
    return static_cast<Instruction::BinaryOps>(Instruction::Add);
  case Opcode::FAdd:
    return static_cast<Instruction::BinaryOps>(Instruction::FAdd);
  case Opcode::Sub:
    return static_cast<Instruction::BinaryOps>(Instruction::Sub);
  case Opcode::FSub:
    return static_cast<Instruction::BinaryOps>(Instruction::FSub);
  case Opcode::Mul:
    return static_cast<Instruction::BinaryOps>(Instruction::Mul);
  case Opcode::FMul:
    return static_cast<Instruction::BinaryOps>(Instruction::FMul);
  case Opcode::UDiv:
    return static_cast<Instruction::BinaryOps>(Instruction::UDiv);
  case Opcode::SDiv:
    return static_cast<Instruction::BinaryOps>(Instruction::SDiv);
  case Opcode::FDiv:
    return static_cast<Instruction::BinaryOps>(Instruction::FDiv);
  case Opcode::URem:
    return static_cast<Instruction::BinaryOps>(Instruction::URem);
  case Opcode::SRem:
    return static_cast<Instruction::BinaryOps>(Instruction::SRem);
  case Opcode::FRem:
    return static_cast<Instruction::BinaryOps>(Instruction::FRem);
  case Opcode::Shl:
    return static_cast<Instruction::BinaryOps>(Instruction::Shl);
  case Opcode::LShr:
    return static_cast<Instruction::BinaryOps>(Instruction::LShr);
  case Opcode::AShr:
    return static_cast<Instruction::BinaryOps>(Instruction::AShr);
  case Opcode::And:
    return static_cast<Instruction::BinaryOps>(Instruction::And);
  case Opcode::Or:
    return static_cast<Instruction::BinaryOps>(Instruction::Or);
  case Opcode::Xor:
    return static_cast<Instruction::BinaryOps>(Instruction::Xor);
  // Mem
  case Opcode::Load:
  case Opcode::Store:
  // Opaque for everything else
  case Opcode::Opaque:
    llvm_unreachable("Not a unary op!");
  }
}

Instruction::CastOps SBInstruction::getIRCastOp(Opcode Opc) {
  switch (Opc) {
  // Vector-related
  case Opcode::Shuffle:
  case Opcode::Pack:
  case Opcode::Unpack:
    llvm_unreachable("Not a unary op!");
  // Casts
  case Opcode::ZExt:
    return static_cast<Instruction::CastOps>(Instruction::ZExt);
  case Opcode::SExt:
    return static_cast<Instruction::CastOps>(Instruction::SExt);
  case Opcode::FPToUI:
    return static_cast<Instruction::CastOps>(Instruction::FPToUI);
  case Opcode::FPToSI:
    return static_cast<Instruction::CastOps>(Instruction::FPToSI);
  case Opcode::FPExt:
    return static_cast<Instruction::CastOps>(Instruction::FPExt);
  case Opcode::PtrToInt:
    return static_cast<Instruction::CastOps>(Instruction::PtrToInt);
  case Opcode::IntToPtr:
    return static_cast<Instruction::CastOps>(Instruction::IntToPtr);
  case Opcode::SIToFP:
    return static_cast<Instruction::CastOps>(Instruction::SIToFP);
  case Opcode::UIToFP:
    return static_cast<Instruction::CastOps>(Instruction::UIToFP);
  case Opcode::Trunc:
    return static_cast<Instruction::CastOps>(Instruction::Trunc);
  case Opcode::FPTrunc:
    return static_cast<Instruction::CastOps>(Instruction::FPTrunc);
  case Opcode::BitCast:
    return static_cast<Instruction::CastOps>(Instruction::BitCast);
  // Cmp
  case Opcode::FCmp:
  case Opcode::ICmp:
  // Select
  case Opcode::Select:
  // Unary
  case Opcode::FNeg:
  // BinOp
  case Opcode::Add:
  case Opcode::FAdd:
  case Opcode::Sub:
  case Opcode::FSub:
  case Opcode::Mul:
  case Opcode::FMul:
  case Opcode::UDiv:
  case Opcode::SDiv:
  case Opcode::FDiv:
  case Opcode::URem:
  case Opcode::SRem:
  case Opcode::FRem:
  case Opcode::Shl:
  case Opcode::LShr:
  case Opcode::AShr:
  case Opcode::And:
  case Opcode::Or:
  case Opcode::Xor:
  // Mem
  case Opcode::Load:
  case Opcode::Store:
  // Opaque for everything else
  case Opcode::Opaque:
    llvm_unreachable("Not a unary op!");
  }
}

const char *SBInstruction::getOpcodeName(Opcode Opc) {
  switch (Opc) {
  // Vector-related
  case Opcode::Shuffle:
    return "Shuffle";
  case Opcode::Pack:
    return "Pack";
  case Opcode::Unpack:
    return "Unpack";
  // Casts
  case Opcode::ZExt:
    return "ZExt";
  case Opcode::SExt:
    return "SExt";
  case Opcode::FPToUI:
    return "FPToUI";
  case Opcode::FPToSI:
    return "FPToSI";
  case Opcode::FPExt:
    return "FPExt";
  case Opcode::PtrToInt:
    return "PtrToInt";
  case Opcode::IntToPtr:
    return "IntToPtr";
  case Opcode::SIToFP:
    return "SIToFP";
  case Opcode::UIToFP:
    return "UIToFP";
  case Opcode::Trunc:
    return "Trunc";
  case Opcode::FPTrunc:
    return "FPTrunc";
  case Opcode::BitCast:
    return "BitCast";
  // Cmp
  case Opcode::FCmp:
    return "FCmp";
  case Opcode::ICmp:
    return "ICmp";
  // Select
  case Opcode::Select:
    return "Select";
  // Unary
  case Opcode::FNeg:
    return "FNeg";
  // BinOp
  case Opcode::Add:
    return "Add";
  case Opcode::FAdd:
    return "FAdd";
  case Opcode::Sub:
    return "Sub";
  case Opcode::FSub:
    return "FSub";
  case Opcode::Mul:
    return "Mul";
  case Opcode::FMul:
    return "FMul";
  case Opcode::UDiv:
    return "UDiv";
  case Opcode::SDiv:
    return "SDiv";
  case Opcode::FDiv:
    return "FDiv";
  case Opcode::URem:
    return "URem";
  case Opcode::SRem:
    return "SRem";
  case Opcode::FRem:
    return "FRem";
  case Opcode::Shl:
    return "Shl";
  case Opcode::LShr:
    return "LShr";
  case Opcode::AShr:
    return "AShr";
  case Opcode::And:
    return "And";
  case Opcode::Or:
    return "Or";
  case Opcode::Xor:
    return "Xor";
  // Mem
  case Opcode::Load:
    return "Load";
  case Opcode::Store:
    return "Store";
  // Opaque for everything else
  case Opcode::Opaque:
    return "Opaque";
  }
}

bool SBInstruction::classof(const SBValue *From) {
  switch (From->getSubclassID()) {
  case ClassID::Pack:
  case ClassID::Unpack:
  case ClassID::Shuffle:
  case ClassID::OpaqueInstr:
  case ClassID::Store:
  case ClassID::Load:
  case ClassID::Cast:
  case ClassID::PHI:
  case ClassID::Select:
  case ClassID::BinOp:
  case ClassID::UnOp:
  case ClassID::Cmp:
    return true;
  case ClassID::Argument:
  case ClassID::User:
  case ClassID::Constant:
  case ClassID::Block:
  case ClassID::Function:
    return false;
  }
  return false;
}

Bundle<Instruction *> SBInstruction::getIRInstrs() const {
  // We may need to erase additional instructions.
  Bundle<Instruction *> IRInstrs;
  switch (getSubclassID()) {
  case ClassID::Pack:
    IRInstrs = cast<SBPackInstruction>(this)->getIRInstrsInternal();
    break;
  case ClassID::Shuffle:
  case ClassID::Unpack:
  case ClassID::OpaqueInstr:
  case ClassID::Store:
  case ClassID::Load:
  case ClassID::Cast:
  case ClassID::PHI:
  case ClassID::Select:
  case ClassID::BinOp:
  case ClassID::UnOp:
  case ClassID::Cmp:
    IRInstrs.push_back(cast<Instruction>(Val));
    break;
  case ClassID::Argument:
  case ClassID::User:
  case ClassID::Function:
    llvm_unreachable("Has no parent!");
  case ClassID::Block:
    llvm_unreachable("Unimplemented!");
  case ClassID::Constant:
    llvm_unreachable("Constants can't be erased from parent");
  }
#ifndef NDEBUG
  for (auto Idx : seq<unsigned>(1, IRInstrs.size())) {
    auto *I1 = IRInstrs[Idx - 1];
    auto *I2 = IRInstrs[Idx];
    assert(((!I1->getParent() || I2->getParent()) || I2->comesBefore(I1)) &&
           "Expected reverse program order!");
  }
#endif
  return IRInstrs;
}

Bundle<Instruction *> SBInstruction::getExternalFacingIRInstrs() const {
  // We may need to erase additional instructions.
  Bundle<Instruction *> IRInstrs;
  switch (getSubclassID()) {
  case ClassID::Pack:
    llvm_unreachable("Should be handled by SBPackInstruction::getIRInstrs()");
    break;
  case ClassID::Shuffle:
  case ClassID::Unpack:
  case ClassID::OpaqueInstr:
  case ClassID::Store:
  case ClassID::Load:
  case ClassID::Cast:
  case ClassID::PHI:
  case ClassID::Select:
  case ClassID::BinOp:
  case ClassID::UnOp:
  case ClassID::Cmp:
    return getIRInstrs();
  case ClassID::Argument:
  case ClassID::User:
  case ClassID::Function:
  case ClassID::Block:
  case ClassID::Constant:
    llvm_unreachable("N/A");
  }
}

void SBInstruction::removeFromParent() {
  // Run the callbacks before we unregister it.
  Ctxt.runRemoveInstrCallbacks(this);

  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<InstrRemoveFromParent>(this, Tracker));

  // Detach all the IR instructions from their parent BB.
  for (Instruction *I : getIRInstrs()) {
    I->removeFromParent();
  }
}

void SBInstruction::eraseFromParent() {
  assert(users().empty() && "Still connected to users, can't erase!");
  auto IRInstrs = getIRInstrs();
  // Run the callbacks before we unregister it.
  Ctxt.runRemoveInstrCallbacks(this);
  auto &Tracker = getTracker();

  // Detach from instruction-specific maps.
  auto SBIPtr = getContext().detach(this);

  if (Tracker.tracking()) {
    // Track deletion from IR to SandboxIR maps.
    Tracker.track(
        std::make_unique<EraseFromParent>(std::move(SBIPtr), Ctxt, Tracker));
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
    for (Instruction *I : getIRInstrs())
      I->eraseFromParent();
  } else {
    // We don't actually delete the IR instruction, because then it would be
    // impossible to bring it back from the dead at the same memory location.
    // Instead we remove it from its BB and track its current location.
    for (Instruction *I : getIRInstrs()) {
      I->removeFromParent();
    }
    for (Instruction *I : getExternalFacingIRInstrs()) {
      I->dropAllReferences();
    }
  }
}

SBBasicBlock *SBInstruction::getParent() const {
  auto *BB = cast<Instruction>(Val)->getParent();
  if (BB == nullptr)
    return nullptr;
  return Ctxt.getSBBasicBlock(BB);
}

void SBInstruction::moveBefore(SBBasicBlock &SBBB,
                                 const SBBBIterator &WhereIt) {
  if (std::next(getIterator()) == WhereIt)
    // Destination is same as origin, nothing to do.
    return;
  auto &Tracker = getTracker();
  if (Tracker.tracking()) {
    Tracker.track(std::make_unique<MoveInstr>(this, Tracker));
  }
  Ctxt.runMoveInstrCallbacks(this, SBBB, WhereIt);

  auto *BB = cast<BasicBlock>(ValueAttorney::getValue(&SBBB));
  BasicBlock::iterator It;
  if (WhereIt == SBBB.end())
    It = BB->end();
  else {
    SBInstruction *WhereI = &*WhereIt;
    It = WhereI->getTopmostIRInstruction()->getIterator();
  }
  auto IRInstrsInProgramOrder(getIRInstrs());
  sort(IRInstrsInProgramOrder,
       [](auto *I1, auto *I2) { return I1->comesBefore(I2); });
  for (auto *I : IRInstrsInProgramOrder)
    I->moveBefore(*BB, It);
}

void SBInstruction::insertBefore(SBInstruction *BeforeI) {
  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<InsertToBB>(BeforeI, getParent(), Tracker));
  Instruction *BeforeTopI = BeforeI->getTopmostIRInstruction();
  auto IRInstrs = getIRInstrs();
  for (Instruction *I : reverse(IRInstrs))
    I->insertBefore(BeforeTopI);
}

void SBInstruction::insertInto(SBBasicBlock *SBBB,
                                 const SBBBIterator &WhereIt) {
  BasicBlock *BB = cast<BasicBlock>(ValueAttorney::getValue(SBBB));
  Instruction *BeforeI;
  SBInstruction *SBBeforeI;
  BasicBlock::iterator BeforeIt;
  if (WhereIt != SBBB->end()) {
    SBBeforeI = &*WhereIt;
    BeforeI = SBBeforeI->getTopmostIRInstruction();
    BeforeIt = BeforeI->getIterator();
  } else {
    SBBeforeI = nullptr;
    BeforeI = nullptr;
    BeforeIt = BB->end();
  }
  auto &Tracker = getTracker();
  if (Tracker.tracking())
    Tracker.track(std::make_unique<InsertToBB>(SBBeforeI, SBBB, Tracker));
  cast<Instruction>(Val)->insertInto(BB, BeforeIt);
}

SBBasicBlock *SBInstruction::getSuccessor(unsigned Idx) const {
  return cast<SBBasicBlock>(
      Ctxt.getSBValue(cast<Instruction>(Val)->getSuccessor(Idx)));
}

#ifndef NDEBUG
void SBInstruction::dump(raw_ostream &OS) const {
  OS << "Unimplemented! Please override dump().";
}
void SBInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
void SBInstruction::dumpVerbose(raw_ostream &OS) const {
  OS << "Unimplemented! Please override dumpVerbose().";
}
void SBInstruction::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
#endif

PackInstrBundle::PackInstrBundle(const ValueBundle &PackInstrsBndl) {
  PackInstrs.reserve(PackInstrsBndl.size());
  copy(PackInstrsBndl.instrRange(), std::back_inserter(PackInstrs));
  // Sort in reverse program order.
  sort(PackInstrs, [](auto *I1, auto *I2) { return I2->comesBefore(I1); });
}

Bundle<Instruction *> SBPackInstruction::getIRInstrsInternal() const {
  return PackInstrs;
}

Bundle<Instruction *> SBPackInstruction::getExternalFacingIRInstrs() const {
  SmallVector<Instruction *> IRInstrs;
  for (Instruction *I : PackInstrs) {
    if (auto *InsertI = dyn_cast<InsertElementInst>(I)) {
      // If this is an internal insert, it must have an Extract operand, which
      // is the external facing IR instruction.
      if (auto *ExtractOp =
              dyn_cast<ExtractElementInst>(InsertI->getOperand(1))) {
        if (find(PackInstrs, ExtractOp) != PackInstrs.end())
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

Use &PackInstrBundle::getExternalFacingOperandUse(
    InsertElementInst *InsertI) const {
  // Get the Insert's edge and check if its source is a Pack Extract. If it is,
  // then don't use the Insert's edge, but rather the Extract's edge.
  Use &OpUse = InsertI->getOperandUse(1);
  Value *Op = OpUse.get();
  if (!isa<ExtractElementInst>(Op) ||
      find(PackInstrs, cast<ExtractElementInst>(Op)) == PackInstrs.end())
    return OpUse;
  // This is an extract used in the pack-from-vector pattern.
  return cast<ExtractElementInst>(Op)->getOperandUse(0);
}

static bool isSingleUseEdge(Use &ExtFacingUse) {
  return !isa<ExtractElementInst>(ExtFacingUse.getUser());
}
static bool isLastOfMultiUseEdge(Use &ExtFacingUse) {
  User *U = ExtFacingUse.getUser();
  assert(isa<ExtractElementInst>(U) &&
         "A multi-Use edge must have an Extract operand!");
  // If the user is not an extract, then this is a single-Use edge.
  auto *ExtractI = cast<ExtractElementInst>(U);
  auto ExtrIdx = *SBUtils::getExtractLane(ExtractI);
  int Lanes = cast<FixedVectorType>(ExtractI->getVectorOperand()->getType())
                  ->getNumElements();
  return ExtrIdx == Lanes - 1;
}

InsertElementInst *PackInstrBundle::getInsertAtLane(int Lane) const {
  auto It = find_if(PackInstrs, [Lane](Value *V) {
    return isa<InsertElementInst>(V) &&
           *SBUtils::getInsertLane(cast<InsertElementInst>(V)) == Lane;
  });
  return It != PackInstrs.end() ? cast<InsertElementInst>(*It) : nullptr;
}

InsertElementInst *PackInstrBundle::getTopInsert() const {
  auto Range = reverse(PackInstrs);
  auto It = find_if(Range, [](auto *I) { return isa<InsertElementInst>(I); });
  assert(It != Range.end() && "Not found!");
  return cast<InsertElementInst>(*It);
}

InsertElementInst *PackInstrBundle::getBotInsert() const {
  auto It =
      find_if(PackInstrs, [](auto *I) { return isa<InsertElementInst>(I); });
  assert(It != PackInstrs.end() && "Not found!");
  return cast<InsertElementInst>(*It);
}

void PackInstrBundle::doOnOperands(
    function_ref<bool(Use &, bool)> DoOnOpFn) const {
  // Constant operands may be folded into the poison vector, or poison operands
  // can also be folded into a single vector poison value.
  auto *TopInsertI = getTopInsert();
  auto *PoisonVal = cast<Constant>(TopInsertI->getOperand(0));
  auto *PoisonConstantVec = dyn_cast<ConstantVector>(PoisonVal);
  auto Lanes = cast<FixedVectorType>(PoisonVal->getType())->getNumElements();
  assert((PoisonConstantVec == nullptr ||
          PoisonConstantVec->getNumOperands() == Lanes) &&
         "Bad Lanes or PoisonConstantVec!");
  for (auto Lane : seq<unsigned>(0, Lanes)) {
    InsertElementInst *InsertI = getInsertAtLane(Lane);
    // A missing insert means that the operand was folded into the poison vector
    Use *OpUsePtr = nullptr;
    if (InsertI != nullptr) {
      OpUsePtr = &getExternalFacingOperandUse(InsertI);
    } else if (PoisonConstantVec != nullptr) {
      OpUsePtr = &PoisonConstantVec->getOperandUse(Lane);
    } else {
      auto *TopInsertI = cast<InsertElementInst>(PackInstrs.front());
      OpUsePtr = &TopInsertI->getOperandUse(0);
    }
    Use &OpUse = *OpUsePtr;
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

unsigned SBPackInstruction::getOperandUseIdx(const Use &UseToMatch) const {
  unsigned RealOpIdx = 0;
  bool Found = false;
  doOnOperands([&UseToMatch, &RealOpIdx, &Found](Use &Use, bool IsRealOp) {
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

Use &PackInstrBundle::getBndlOperandUse(unsigned OperandIdx) const {
  unsigned RealOpIdx = 0;
  // Special case for op_end().
  if (OperandIdx == getNumOperands())
    return *getBotInsert()->op_end();

  Use *ReturnUse = nullptr;
  doOnOperands([&RealOpIdx, &ReturnUse, OperandIdx](Use &Use, bool IsRealOp) {
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

unsigned PackInstrBundle::getNumOperands() const {
  // Not breaking for any operand will give us the total number of operands.
  unsigned RealOpCnt = 0;
  doOnOperands([&RealOpCnt](Use &Use, bool IsRealOp) {
    if (IsRealOp)
      ++RealOpCnt;
    return false; // Don't break
  });
  return RealOpCnt;
}

#ifndef NDEBUG
void PackInstrBundle::verify() const {
  // Make sure that the consecutive Extracts that make up the
  // pack-from-vector pattern have the same operand. This could break during
  // a SBPackInstruction::setOperand() operation.
  ExtractElementInst *LastExtractI = nullptr;
  doOnOperands([&LastExtractI](Use &Use, bool IsRealOp) -> bool {
    if (IsRealOp && LastExtractI != nullptr) {
      // We expect an extract that extracts from the same vector as
      // LastExtractI, but the next lane.
      assert(isa<ExtractElementInst>(Use.getUser()) && "Expect Extract");
      auto *ExtractI = cast<ExtractElementInst>(Use.getUser());
      assert(Use.get() == ExtractI->getVectorOperand() && "Sanity check");
      assert(ExtractI->getVectorOperand() == LastExtractI->getVectorOperand() &&
             "Most likely setOperand() did not update all Extracts!");
      assert(ExtractI->getIndexOperand() != LastExtractI->getIndexOperand() &&
             "Expected different indices");
      LastExtractI = nullptr;
    } else {
      LastExtractI = dyn_cast<ExtractElementInst>(Use.getUser());
    }
    return false;
  });
}
#endif

SBPackInstruction::SBPackInstruction(const ValueBundle &Instrs,
                                         SBContext &SBCtxt)
    : PackInstrBundle(Instrs),
      SBInstruction(ClassID::Pack, getBottomInsert(Instrs), SBCtxt) {
  assert(all_of(PackInstrs,
                [](Value *V) {
                  return isa<InsertElementInst>(V) ||
                         isa<ExtractElementInst>(V);
                }) &&
         "Expected inserts or extracts!");
#ifndef NDEBUG
  for (auto Idx : seq<unsigned>(1, PackInstrs.size()))
    assert(this->PackInstrs[Idx]->comesBefore(this->PackInstrs[Idx - 1]) &&
           "Expecte reverse program order!");
  assert(all_of(drop_begin(this->PackInstrs),
                [this](auto *I) {
                  return I->comesBefore(cast<Instruction>(Val));
                }) &&
         "Val should be the bottom instruction!");
#endif
}

SBUser::op_iterator SBPackInstruction::op_begin() {
  return op_iterator(getOperandUseInternal(0, /*Verify=*/false));
}

SBUser::op_iterator SBPackInstruction::op_end() {
  return op_iterator(getOperandUseInternal(getNumOperands(), /*Verify=*/false));
}

SBUser::const_op_iterator SBPackInstruction::op_begin() const {
  return const_cast<SBPackInstruction *>(this)->op_begin();
}

SBUser::const_op_iterator SBPackInstruction::op_end() const {
  return const_cast<SBPackInstruction *>(this)->op_end();
}

SBConstant::SBConstant(Constant *C, SBContext &SBCtxt)
    : SBUser(ClassID::Constant, C, SBCtxt) {}

bool SBConstant::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Constant ||
         From->getSubclassID() == ClassID::Function;
}
#ifndef NDEBUG
void SBConstant::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBConstant::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBConstant::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBConstant::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBUse SBPackInstruction::getOperandUseInternal(unsigned OperandIdx,
                                                   bool Verify) const {
  assert((!Verify || OperandIdx < getNumOperands()) && "Out of bounds!");
  Use &LLVMUse = PackInstrBundle::getBndlOperandUse(OperandIdx);
  return SBUse(&LLVMUse, const_cast<SBPackInstruction *>(this), Ctxt);
}

bool SBPackInstruction::isRealOperandUse(Use &OpUse) const {
  bool IsReal = false;
  bool Found = true;
  doOnOperands([&OpUse, &IsReal, &Found](Use &Use, bool IsRealOp) {
    if (&Use != &OpUse)
      return false; // Don't break
    IsReal = IsRealOp;
    Found = true;
    return true; // Break
  });
  assert(Found && "OpUse not found!");
  return IsReal;
}

void SBPackInstruction::setOperand(unsigned OperandIdx, SBValue *Operand) {
  assert(OperandIdx < getNumOperands() && "Out of bounds!");
  assert(Operand->getType() == SBUser::getOperand(OperandIdx)->getType() &&
         "Operand of wrong type!");
  Value *NewOp = ValueAttorney::getValue(Operand);
  unsigned RealOpIdx = 0;
  doOnOperands([NewOp, OperandIdx, &RealOpIdx, this](Use &Use, bool IsRealOp) {
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

InsertElementInst *
SBPackInstruction::getBottomInsert(const ValueBundle &Instrs) const {
  // Get the bottom insert by removing the vector operands from the set until we
  // have only the bottom left.
  DenseSet<Value *> AllPackInstrs(Instrs.begin(), Instrs.end());
  for (auto *PackI : Instrs) {
    assert((isa<InsertElementInst>(PackI) || isa<ExtractElementInst>(PackI)) &&
           "Expected Insert or Extract");
    AllPackInstrs.erase(cast<Instruction>(PackI)->getOperand(0));
    AllPackInstrs.erase(cast<Instruction>(PackI)->getOperand(1));
  }
  assert(AllPackInstrs.size() == 1 && "Unexpected pack instruction structure");
  return cast<InsertElementInst>(*AllPackInstrs.begin());
}

bool SBPackInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Pack;
}

std::variant<ValueBundle, Constant *>
SBPackInstruction::createIR(const SBValBundle &ToPack,
                              SBBasicBlock *Parent,
                              SBInstruction *BeforeI) {
  // A Pack should be placed after the latest packed value.
  auto *BB = SBBasicBlockAttorney::getBB(Parent);
  auto &LLVMIRBuilder = Parent->getContext().getLLVMIRBuilder();
  if (BeforeI != nullptr)
    LLVMIRBuilder.SetInsertPoint(BeforeI->getTopmostIRInstruction());
  else
    SBUtils::setInsertPointAfter(ToPack.getValueBundle(), BB, LLVMIRBuilder,
                                  /*SkipPHIs=*/true);

  Type *ScalarTy = SBUtils::getCommonScalarTypeFast(ToPack);
  unsigned Lanes = SBUtils::getNumLanes(ToPack);
  auto *VecTy = SBUtils::getWideType(ScalarTy, Lanes);

  // Create a series of pack instructions.
  ValueBundle AllPackInstrs;
  Value *LastInsert = PoisonValue::get(VecTy);

  auto Collect = [&AllPackInstrs](Value *NewV) {
    assert(isa<Instruction>(NewV) && "Expected instruction!");
    auto *I = cast<Instruction>(NewV);
    AllPackInstrs.push_back(I);
  };

  unsigned InsertIdx = 0;
  for (SBValue *SBV : ToPack) {
    Value *Elm = ValueAttorney::getValue(SBV);
    if (Elm->getType()->isVectorTy()) {
      unsigned NumElms =
          cast<FixedVectorType>(Elm->getType())->getNumElements();
      for (auto ExtrLane : seq<int>(0, NumElms)) {
        // This may return a Constant if Elm is a Constant.
        auto *ExtrI =
            LLVMIRBuilder.CreateExtractElement(Elm, ExtrLane, "XPack");
        if (auto *ExtrC = dyn_cast<Constant>(ExtrI))
          Parent->getContext().getOrCreateSBConstant(ExtrC);
        else
          Collect(ExtrI);
        // This may also return a Constant if ExtrI is a Constant.
        LastInsert = LLVMIRBuilder.CreateInsertElement(LastInsert, ExtrI,
                                                       InsertIdx++, "Pack");
        if (auto *C = dyn_cast<Constant>(LastInsert)) {
          if (InsertIdx == Lanes)
            return C;
          Parent->getContext().getOrCreateSBValue(C);
        } else
          Collect(LastInsert);
      }
    } else {
      // This may be folded into a Constant if LastInsert is a Constant. In that
      // case we only collect the last constant.
      LastInsert = LLVMIRBuilder.CreateInsertElement(LastInsert, Elm,
                                                     InsertIdx++, "Pack");
      if (auto *C = dyn_cast<Constant>(LastInsert)) {
        if (InsertIdx == Lanes)
          return C;
        Parent->getContext().getOrCreateSBValue(C);
      } else
        Collect(LastInsert);
    }
  }
#ifndef NDEBUG
  Parent->verifyIR();
#endif
  return AllPackInstrs;
}

bool SBArgument::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Argument;
}

#ifndef NDEBUG
void SBArgument::printAsOperand(raw_ostream &OS) const {
  printAsOperandCommon(OS);
}
void SBArgument::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}
void SBArgument::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
void SBArgument::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBArgument::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
#endif

#ifndef NDEBUG
void SBPackInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  OS.indent(2) << "PackInstrs:\n";
  for (auto *I : PackInstrs)
    OS.indent(2) << *I << "\n";
  dumpCommonFooter(OS);
}
void SBPackInstruction::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}

void SBPackInstruction::dump(raw_ostream &OS) const {
  // Sort pack instructions in program order.
  auto SortedInstrs = getIRInstrs();
  if (all_of(SortedInstrs, [](Instruction *I) { return I->getParent(); }))
    sort(SortedInstrs, [](auto *I1, auto *I2) { return I1->comesBefore(I2); });
  else
    OS << "** Error: Not all IR Instrs have a parent! **";

  auto ExtFacing = getExternalFacingIRInstrs();
  unsigned NumOperands = getNumOperands();

  for (auto [Idx, I] : enumerate(SortedInstrs)) {
    OS << *I;
    dumpCommonSuffix(OS);
    // Print the lane.
    bool IsExt = find(ExtFacing, I) != ExtFacing.end();
    if (IsExt) {
      Use *OpUse = nullptr;
      if (auto *InsertI = dyn_cast<InsertElementInst>(I)) {
        OpUse = &InsertI->getOperandUse(1);
      } else if (auto *ExtractI = dyn_cast<ExtractElementInst>(I)) {
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
void SBPackInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

SBShuffleInstruction::SBShuffleInstruction(const ShuffleMask &Mask,
                                               SBValue *Op,
                                               SBBasicBlock *Parent)
    : SBShuffleInstruction(createIR(Mask, Op, Parent), Parent->getContext()) {
  Ctxt.createMissingConstantOperands(Val);
  assert(Val != nullptr && "Shuffle was folded!");
}

SBUser::op_iterator SBShuffleInstruction::op_begin() {
  return op_iterator(getOperandUseInternal(0, /*Verify=*/false));
}
SBUser::op_iterator SBShuffleInstruction::op_end() {
  return op_iterator(getOperandUseInternal(getNumOperands(), /*Verify=*/false));
}
SBUser::const_op_iterator SBShuffleInstruction::op_begin() const {
  return const_cast<SBShuffleInstruction *>(this)->op_begin();
}
SBUser::const_op_iterator SBShuffleInstruction::op_end() const {
  return const_cast<SBShuffleInstruction *>(this)->op_end();
}

void SBShuffleInstruction::setOperand(unsigned OperandIdx,
                                        SBValue *Operand) {
  assert(OperandIdx == 0 && "A SBShuffleInstruction has exactly 1 operand!");
  SBUser::setOperand(OperandIdx, Operand);
}

bool SBShuffleInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Shuffle;
}

ShuffleVectorInst *SBShuffleInstruction::createIR(const ShuffleMask &Mask,
                                                    SBValue *Op,
                                                    SBBasicBlock *Parent) {
  Value *Vec = ValueAttorney::getValue(Op);
  auto &LLVMIRBuilder = Parent->getContext().getLLVMIRBuilder();
  SBUtils::setInsertPointAfter(
      ValueBundle(Vec), SBBasicBlockAttorney::getBB(Parent), LLVMIRBuilder);
  auto *Shuffle = cast_or_null<ShuffleVectorInst>(
      LLVMIRBuilder.CreateShuffleVector(Vec, Mask, "Shuf"));
#ifndef NDEBUG
  Parent->verifyIR();
#endif
  return Shuffle;
}

#ifndef NDEBUG
void SBShuffleInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "  " << getMask() << "\n";
  dumpCommonFooter(OS);
}
void SBShuffleInstruction::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
void SBShuffleInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
  OS << " ; " << getMask();
}
void SBShuffleInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBUnpackInstruction::SBUnpackInstruction(ExtractElementInst *ExtractI,
                                             SBValue *UnpackOp,
                                             unsigned UnpackLane,
                                             SBContext &SBCtxt)
    : SBInstruction(ClassID::Unpack, ExtractI, SBCtxt) {
  assert(getOperand(0) == UnpackOp && "Bad operand!");
}

SBUnpackInstruction::SBUnpackInstruction(ShuffleVectorInst *ShuffleI,
                                             SBValue *UnpackOp,
                                             unsigned UnpackLane,
                                             SBContext &SBCtxt)
    : SBInstruction(ClassID::Unpack, ShuffleI, SBCtxt) {
  assert(getOperand(0) == UnpackOp && "Bad operand!");
}

Value *SBUnpackInstruction::createIR(SBValue *UnpackOp,
                                       SBBasicBlock *Parent, unsigned Lane,
                                       unsigned Lanes) {
  auto &LLVMIRBuilder = Parent->getContext().getLLVMIRBuilder();
  Value *OpVec = ValueAttorney::getValue(UnpackOp);
  SBUtils::setInsertPointAfter(
      ValueBundle{OpVec}, SBBasicBlockAttorney::getBB(Parent), LLVMIRBuilder);
  Value *Unpack;
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
#ifndef NDEBUG
  Parent->verifyIR();
#endif
  return Unpack;
}

bool SBUnpackInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Unpack;
}

#ifndef NDEBUG
void SBUnpackInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << " " << "lane:" << getUnpackLane() << "\n";
  dumpCommonFooter(OS);
}
void SBUnpackInstruction::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
void SBUnpackInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
  OS << " lane:" << getUnpackLane();
}
void SBUnpackInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif
SBValue *SBCmpInstruction::create(CmpInst::Predicate Pred, SBValue *LHS,
                                      SBValue *RHS,
                                      SBInstruction *InsertBefore,
                                      SBContext &SBCtxt, const Twine &Name,
                                      MDNode *FPMathTag) {
  Value *LHSIR = ValueAttorney::getValue(LHS);
  Value *RHSIR = ValueAttorney::getValue(RHS);
  Instruction *BeforeIR = InsertBefore->getTopmostIRInstruction();
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  auto *NewV = Builder.CreateCmp(Pred, LHSIR, RHSIR, Name, FPMathTag);
  if (auto *NewCI = dyn_cast<CmpInst>(NewV))
    return SBCtxt.createSBCmpInstruction(NewCI);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBCmpInstruction::create(CmpInst::Predicate Pred, SBValue *LHS,
                                      SBValue *RHS,
                                      SBBasicBlock *InsertAtEnd,
                                      SBContext &SBCtxt, const Twine &Name,
                                      MDNode *FPMathTag) {
  Value *LHSIR = ValueAttorney::getValue(LHS);
  Value *RHSIR = ValueAttorney::getValue(RHS);
  BasicBlock *InsertAtEndIR = SBBasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertAtEndIR);
  auto *NewV = Builder.CreateCmp(Pred, LHSIR, RHSIR, Name, FPMathTag);
  if (auto *NewCI = dyn_cast<CmpInst>(NewV))
    return SBCtxt.createSBCmpInstruction(NewCI);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

bool SBCmpInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Cmp;
}

#ifndef NDEBUG
void SBCmpInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBCmpInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBCmpInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBCmpInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBStoreInstruction *
SBStoreInstruction::create(SBValue *V, SBValue *Ptr, MaybeAlign Align,
                             SBInstruction *InsertBefore,
                             SBContext &SBCtxt) {
  Value *ValIR = ValueAttorney::getValue(V);
  Value *PtrIR = ValueAttorney::getValue(Ptr);
  Instruction *BeforeIR = InsertBefore->getTopmostIRInstruction();
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  auto *NewSI =
      Builder.CreateAlignedStore(ValIR, PtrIR, Align, /*isVolatile=*/false);
  auto *NewSBI = SBCtxt.createSBStoreInstruction(NewSI);
  return NewSBI;
}
SBStoreInstruction *SBStoreInstruction::create(SBValue *V, SBValue *Ptr,
                                                   MaybeAlign Align,
                                                   SBBasicBlock *InsertAtEnd,
                                                   SBContext &SBCtxt) {
  Value *ValIR = ValueAttorney::getValue(V);
  Value *PtrIR = ValueAttorney::getValue(Ptr);
  BasicBlock *InsertAtEndIR = SBBasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertAtEndIR);
  auto *NewSI =
      Builder.CreateAlignedStore(ValIR, PtrIR, Align, /*isVolatile=*/false);
  auto *NewSBI = SBCtxt.createSBStoreInstruction(NewSI);
  return NewSBI;
}

bool SBStoreInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Store;
}

SBValue *SBStoreInstruction::getValueOperand() const {
  return Ctxt.getSBValue(cast<StoreInst>(Val)->getValueOperand());
}

SBValue *SBStoreInstruction::getPointerOperand() const {
  return Ctxt.getSBValue(cast<StoreInst>(Val)->getPointerOperand());
}

#ifndef NDEBUG
void SBStoreInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBStoreInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBStoreInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBStoreInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBLoadInstruction *SBLoadInstruction::create(Type *Ty, SBValue *Ptr,
                                                 MaybeAlign Align,
                                                 SBInstruction *InsertBefore,
                                                 SBContext &SBCtxt,
                                                 const Twine &Name) {
  Value *PtrIR = ValueAttorney::getValue(Ptr);
  Instruction *BeforeIR = InsertBefore->getTopmostIRInstruction();
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  auto *NewLI =
      Builder.CreateAlignedLoad(Ty, PtrIR, Align, /*isVolatile=*/false, Name);
  auto *NewSBI = SBCtxt.createSBLoadInstruction(NewLI);
  return NewSBI;
}

SBLoadInstruction *SBLoadInstruction::create(Type *Ty, SBValue *Ptr,
                                                 MaybeAlign Align,
                                                 SBBasicBlock *InsertAtEnd,
                                                 SBContext &SBCtxt,
                                                 const Twine &Name) {
  Value *PtrIR = ValueAttorney::getValue(Ptr);
  BasicBlock *InsertAtEndIR = SBBasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertAtEndIR);
  auto *NewLI =
      Builder.CreateAlignedLoad(Ty, PtrIR, Align, /*isVolatile=*/false, Name);
  auto *NewSBI = SBCtxt.createSBLoadInstruction(NewLI);
  return NewSBI;
}

bool SBLoadInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Load;
}

SBValue *SBLoadInstruction::getPointerOperand() const {
  return Ctxt.getSBValue(cast<LoadInst>(Val)->getPointerOperand());
}

#ifndef NDEBUG
void SBLoadInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBLoadInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBLoadInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBLoadInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBValue *SBCastInstruction::create(Type *Ty, Opcode Op, SBValue *Operand,
                                       SBInstruction *InsertBefore,
                                       SBContext &SBCtxt,
                                       const Twine &Name) {
  Value *IROperand = ValueAttorney::getValue(Operand);
  Instruction *BeforeIR = InsertBefore->getTopmostIRInstruction();
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  auto *NewV = Builder.CreateCast(getIRCastOp(Op), IROperand, Ty, Name);
  if (auto *NewCI = dyn_cast<CastInst>(NewV))
    return SBCtxt.createSBCastInstruction(NewCI);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBCastInstruction::create(Type *Ty, Opcode Op, SBValue *Operand,
                                       SBBasicBlock *InsertAtEnd,
                                       SBContext &SBCtxt,
                                       const Twine &Name) {
  Value *IROperand = ValueAttorney::getValue(Operand);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(SBBasicBlockAttorney::getBB(InsertAtEnd));
  auto *NewV = Builder.CreateCast(getIRCastOp(Op), IROperand, Ty, Name);
  if (auto *NewCI = dyn_cast<CastInst>(NewV))
    return SBCtxt.createSBCastInstruction(NewCI);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

bool SBCastInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Cast;
}

#ifndef NDEBUG
void SBCastInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBCastInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBCastInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBCastInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBValue *SBPHINode::create(Type *Ty, unsigned NumReservedValues,
                               SBInstruction *InsertBefore,
                               SBContext &SBCtxt, const Twine &Name) {
  Instruction *InsertBeforeIR = InsertBefore->getTopmostIRInstruction();
  PHINode *NewPHI =
      PHINode::Create(Ty, NumReservedValues, Name, InsertBeforeIR);
  return SBCtxt.createSBPHINode(NewPHI);
}

SBValue *SBPHINode::create(Type *Ty, unsigned NumReservedValues,
                               SBBasicBlock *InsertAtEnd,
                               SBContext &SBCtxt, const Twine &Name) {
  BasicBlock *InsertAtEndIR = SBBasicBlockAttorney::getBB(InsertAtEnd);
  PHINode *NewPHI = PHINode::Create(Ty, NumReservedValues, Name, InsertAtEndIR);
  return SBCtxt.createSBPHINode(NewPHI);
}

bool SBPHINode::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::PHI;
}

#ifndef NDEBUG
void SBPHINode::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBPHINode::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBPHINode::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBPHINode::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBValue *SBSelectInstruction::create(SBValue *Cond, SBValue *True,
                                         SBValue *False,
                                         SBInstruction *InsertBefore,
                                         SBContext &SBCtxt,
                                         const Twine &Name) {
  Value *IRCond = ValueAttorney::getValue(Cond);
  Value *IRTrue = ValueAttorney::getValue(True);
  Value *IRFalse = ValueAttorney::getValue(False);
  Instruction *BeforeIR = InsertBefore->getTopmostIRInstruction();
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  Value *NewV = Builder.CreateSelect(IRCond, IRTrue, IRFalse, Name);
  if (auto *NewSI = dyn_cast<SelectInst>(NewV))
    return SBCtxt.createSBSelectInstruction(NewSI);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBSelectInstruction::create(SBValue *Cond, SBValue *True,
                                         SBValue *False,
                                         SBBasicBlock *InsertAtEnd,
                                         SBContext &SBCtxt,
                                         const Twine &Name) {
  Value *IRCond = ValueAttorney::getValue(Cond);
  Value *IRTrue = ValueAttorney::getValue(True);
  Value *IRFalse = ValueAttorney::getValue(False);
  BasicBlock *IRInsertAtEnd = SBBasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(IRInsertAtEnd);
  Value *NewV = Builder.CreateSelect(IRCond, IRTrue, IRFalse, Name);
  if (auto *NewSI = dyn_cast<SelectInst>(NewV))
    return SBCtxt.createSBSelectInstruction(NewSI);
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

bool SBSelectInstruction::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::Select;
}

#ifndef NDEBUG
void SBSelectInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBSelectInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBSelectInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBSelectInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBValue *SBBinaryOperator::createWithCopiedFlags(
    SBInstruction::Opcode Op, SBValue *LHS, SBValue *RHS,
    SBValue *CopyFrom, SBInstruction *InsertBefore, SBContext &SBCtxt,
    const Twine &Name) {
  Value *IRLHS = ValueAttorney::getValue(LHS);
  Value *IRRHS = ValueAttorney::getValue(RHS);
  Value *IRCopyFrom = ValueAttorney::getValue(CopyFrom);
  Instruction *BeforeIR = InsertBefore->getTopmostIRInstruction();
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  Value *NewV = Builder.CreateBinOp(getIRBinaryOp(Op), IRLHS, IRRHS, Name);
  if (auto *NewBinOp = dyn_cast<BinaryOperator>(NewV)) {
    NewBinOp->copyIRFlags(IRCopyFrom);
    return SBCtxt.createSBBinaryOperator(NewBinOp);
  }
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBBinaryOperator::createWithCopiedFlags(
    SBInstruction::Opcode Op, SBValue *LHS, SBValue *RHS,
    SBValue *CopyFrom, SBBasicBlock *InsertAtEnd, SBContext &SBCtxt,
    const Twine &Name) {
  Value *IRLHS = ValueAttorney::getValue(LHS);
  Value *IRRHS = ValueAttorney::getValue(RHS);
  Value *IRCopyFrom = ValueAttorney::getValue(CopyFrom);
  BasicBlock *IRInsertAtEnd = SBBasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(IRInsertAtEnd);
  Value *NewV = Builder.CreateBinOp(getIRBinaryOp(Op), IRLHS, IRRHS, Name);
  if (auto *NewBinOp = dyn_cast<BinaryOperator>(NewV)) {
    NewBinOp->copyIRFlags(IRCopyFrom);
    return SBCtxt.createSBBinaryOperator(NewBinOp);
  }
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

bool SBBinaryOperator::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::BinOp;
}

#ifndef NDEBUG
void SBBinaryOperator::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBBinaryOperator::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBBinaryOperator::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBBinaryOperator::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SBValue *SBUnaryOperator::createWithCopiedFlags(
    SBInstruction::Opcode Op, SBValue *OpV, SBValue *CopyFrom,
    SBInstruction *InsertBefore, SBContext &SBCtxt, const Twine &Name) {
  Value *IROpV = ValueAttorney::getValue(OpV);
  Value *IRCopyFrom = ValueAttorney::getValue(CopyFrom);
  Instruction *BeforeIR = InsertBefore->getTopmostIRInstruction();
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  Value *NewV = Builder.CreateUnOp(getIRUnaryOp(Op), IROpV, Name);
  if (auto *NewBinOp = dyn_cast<UnaryOperator>(NewV)) {
    NewBinOp->copyIRFlags(IRCopyFrom);
    return SBCtxt.createSBUnaryOperator(NewBinOp);
  }
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

SBValue *SBUnaryOperator::createWithCopiedFlags(
    SBInstruction::Opcode Op, SBValue *OpV, SBValue *CopyFrom,
    SBBasicBlock *InsertAtEnd, SBContext &SBCtxt, const Twine &Name) {
  Value *IROpV = ValueAttorney::getValue(OpV);
  Value *IRCopyFrom = ValueAttorney::getValue(CopyFrom);
  BasicBlock *IRInsertAtEnd = SBBasicBlockAttorney::getBB(InsertAtEnd);
  auto &Builder = SBCtxt.getLLVMIRBuilder();
  Builder.SetInsertPoint(IRInsertAtEnd);
  Value *NewV = Builder.CreateUnOp(getIRUnaryOp(Op), IROpV, Name);
  if (auto *NewBinOp = dyn_cast<UnaryOperator>(NewV)) {
    NewBinOp->copyIRFlags(IRCopyFrom);
    return SBCtxt.createSBUnaryOperator(NewBinOp);
  }
  assert(isa<Constant>(NewV) && "Expected constant");
  return SBCtxt.getOrCreateSBConstant(cast<Constant>(NewV));
}

bool SBUnaryOperator::classof(const SBValue *From) {
  return From->getSubclassID() == ClassID::UnOp;
}

#ifndef NDEBUG
void SBUnaryOperator::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBUnaryOperator::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBUnaryOperator::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBUnaryOperator::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

#ifndef NDEBUG
void SBOpaqueInstruction::dumpVerbose(raw_ostream &OS) const {
  dumpCommonHeader(OS);
  OS << "\n";
  dumpCommonFooter(OS);
}
void SBOpaqueInstruction::dumpVerbose() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBOpaqueInstruction::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SBOpaqueInstruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

void SBFunction::detachFromLLVMIR() {
  for (SBBasicBlock &SBBB : *this)
    SBBB.detachFromLLVMIR();
  // Detach the actual SBFunction.
  Ctxt.detach(this);
}

#ifndef NDEBUG
void SBFunction::dumpNameAndArgs(raw_ostream &OS) const {
  Function *F = getFunction();
  OS << *getType() << " @" << F->getName() << "(";
  auto NumArgs = F->arg_size();
  for (auto [Idx, Arg] : enumerate(F->args())) {
    auto *SBArg = cast_or_null<SBArgument>(Ctxt.getSBValue(&Arg));
    if (SBArg == nullptr)
      OS << "NULL";
    else
      SBArg->printAsOperand(OS);
    if (Idx + 1 < NumArgs)
      OS << ", ";
  }
  OS << ")";
}
void SBFunction::dump(raw_ostream &OS) const {
  dumpNameAndArgs(OS);
  OS << " {\n";
  Function *F = getFunction();
  BasicBlock &LastBB = F->back();
  for (BasicBlock &BB : *F) {
    auto *SBBB = cast_or_null<SBBasicBlock>(Ctxt.getSBValue(&BB));
    if (SBBB == nullptr)
      OS << "NULL";
    else
      OS << *SBBB;
    if (&BB != &LastBB)
      OS << "\n";
  }
  OS << "}\n";
}
void SBFunction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
void SBFunction::dumpVerbose(raw_ostream &OS) const {
  dumpNameAndArgs(OS);
  OS << " {\n";
  for (BasicBlock &BB : *getFunction()) {
    auto *SBBB = cast_or_null<SBBasicBlock>(Ctxt.getSBValue(&BB));
    if (SBBB == nullptr)
      OS << "NULL";
    else
      SBBB->dumpVerbose(OS);
    OS << "\n";
  }
  OS << "}\n";
}
void SBFunction::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
#endif

SBContext::SBContext(LLVMContext &LLVMCtxt, AliasAnalysis &AA)
    : LLVMCtxt(LLVMCtxt), AA(AA), LLVMIRBuilder(LLVMCtxt, ConstantFolder()) {}

Scheduler *SBContext::getScheduler(SBBasicBlock *SBBB) const {
  auto It = SchedForSBBB.find(SBBB);
  assert(It != SchedForSBBB.end() &&
         "Scheduler for SBBB should have been created by now!");
  return It->second.get();
}

const DependencyGraph &SBContext::getDAG(SBBasicBlock *SBBB) const {
  return getScheduler(SBBB)->getDAG();
}

void SBContext::SchedulerDeleter::operator()(
    Scheduler *Ptr) const {
  delete Ptr;
}

std::unique_ptr<SBValue> SBContext::detachValue(Value *V) {
  std::unique_ptr<SBValue> Erased;
  auto It = LLVMValueToSBValueMap.find(V);
  if (It != LLVMValueToSBValueMap.end()) {
    auto *Val = It->second.release();
    Erased = std::unique_ptr<SBValue>(Val);
    LLVMValueToSBValueMap.erase(It);
  }
  MultiInstrMap.erase(V);
  return Erased;
}

SBValue *SBContext::getSBValue(Value *V) const {
  // In the common case we should find the value in LLVMValueToSBValueMap.
  auto It = LLVMValueToSBValueMap.find(V);
  if (It != LLVMValueToSBValueMap.end())
    return It->second.get();
  // Instrs that map to multiple IR Instrs (like Packs) use a second map.
  auto It2 = MultiInstrMap.find(V);
  if (It2 != MultiInstrMap.end()) {
    Value *Key = It2->second;
    assert(Key != V && "Bad entry in MultiInstrMap!");
    return getSBValue(Key);
  }
  return nullptr;
}

SBConstant *SBContext::getSBConstant(Constant *C) const {
  return cast_or_null<SBConstant>(getSBValue(C));
}

SBConstant *SBContext::getOrCreateSBConstant(Constant *C) {
  auto Pair = LLVMValueToSBValueMap.insert({C, nullptr});
  auto It = Pair.first;
  if (Pair.second) {
    It->second = std::unique_ptr<SBConstant>(new SBConstant(C, *this));
    return cast<SBConstant>(It->second.get());
  }
  return cast<SBConstant>(It->second.get());
}

std::unique_ptr<SBValue> SBContext::detach(SBValue *SBV) {
  switch (SBV->getSubclassID()) {
  case SBValue::ClassID::Pack: {
    auto *Pack = cast<SBPackInstruction>(SBV);
    auto *PackV = ValueAttorney::getValue(Pack);
    for (auto *PI : Pack->getPackInstrs())
      if (PI != PackV) // Skip the bottom value, gets detached later
        detachValue(PI);
    break;
  }
  case SBValue::ClassID::Unpack:
  case SBValue::ClassID::Shuffle:
  case SBValue::ClassID::OpaqueInstr:
  case SBValue::ClassID::Argument:
  case SBValue::ClassID::Block:
  case SBValue::ClassID::Function:
  case SBValue::ClassID::Store:
  case SBValue::ClassID::Load:
  case SBValue::ClassID::Cast:
  case SBValue::ClassID::PHI:
  case SBValue::ClassID::Select:
  case SBValue::ClassID::BinOp:
  case SBValue::ClassID::UnOp:
  case SBValue::ClassID::Cmp:
    break;
  case SBValue::ClassID::Constant:
    llvm_unreachable("Can't detach a constant!");
  case SBValue::ClassID::User:
    llvm_unreachable("Can't detach a user!");
  }
  Value *V = ValueAttorney::getValue(SBV);
  return detachValue(V);
}

SBValue *
SBContext::registerSBValue(std::unique_ptr<SBValue> &&SBVPtr) {
  auto &Tracker = getTracker();
  if (Tracker.tracking() && isa<SBInstruction>(SBVPtr.get()))
    Tracker.track(std::make_unique<CreateAndInsertInstr>(
        cast<SBInstruction>(SBVPtr.get()), Tracker));

  auto RegisterSBValue =
      [this](std::unique_ptr<SBValue> &&SBVPtr) -> SBValue * {
    switch (SBVPtr->getSubclassID()) {
    case SBValue::ClassID::Unpack: {
      auto *SBV = cast<SBUnpackInstruction>(SBVPtr.get());
      Value *ExtractI = ValueAttorney::getValue(SBV);
      LLVMValueToSBValueMap[ExtractI] = std::move(SBVPtr);
      return SBV;
    }
    case SBValue::ClassID::Pack: {
      auto *SBV = cast<SBPackInstruction>(SBVPtr.get());
      Value *Key = SBV->Val;
      LLVMValueToSBValueMap[Key] = std::move(SBVPtr);
      for (auto *PackI : SBV->getPackInstrs())
        MultiInstrMap[PackI] = Key;
      return SBV;
    }
    case SBValue::ClassID::OpaqueInstr: {
      auto *SBI = cast<SBOpaqueInstruction>(SBVPtr.get());
      Value *V = ValueAttorney::getValue(SBI);
      LLVMValueToSBValueMap[V] = std::move(SBVPtr);
      return SBI;
    }
    case SBValue::ClassID::Argument:
    case SBValue::ClassID::Shuffle:
    case SBValue::ClassID::Store:
    case SBValue::ClassID::Load:
    case SBValue::ClassID::Cast:
    case SBValue::ClassID::PHI:
    case SBValue::ClassID::Select:
    case SBValue::ClassID::BinOp:
    case SBValue::ClassID::UnOp:
    case SBValue::ClassID::Cmp:
    case SBValue::ClassID::Constant:
    case SBValue::ClassID::Block:
    case SBValue::ClassID::Function: {
      auto *SBV = SBVPtr.get();
      auto *V = ValueAttorney::getValue(SBV);
      LLVMValueToSBValueMap[V] = std::move(SBVPtr);
      return SBV;
    }
    case SBValue::ClassID::User:
      llvm_unreachable("Can't register a user!");
    }
  };
  SBValue *SBV = RegisterSBValue(std::move(SBVPtr));
  if (auto *SBI = dyn_cast<SBInstruction>(SBV))
    runInsertInstrCallbacks(SBI);
  return SBV;
}

void SBContext::createMissingConstantOperands(Value *V) {
  // Create SandboxIR for all new constant operands.
  if (User *U = dyn_cast<User>(V)) {
    for (Value *Op : U->operands()) {
      if (auto *ConstOp = dyn_cast<Constant>(Op))
        getOrCreateSBConstant(ConstOp);
    }
  }
}

// Pack
SBValue *SBContext::createSBPackInstruction(const SBValBundle &PackOps,
                                                  SBBasicBlock *SBBB,
                                                  SBInstruction *BeforeI) {
  std::variant<ValueBundle, Constant *> BorC =
      SBPackInstruction::createIR(PackOps, SBBB, BeforeI);
  // CreateIR packed constants which resulted in a single folded Constant.
  if (Constant **CPtr = std::get_if<Constant *>(&BorC))
    return getOrCreateSBValue(*CPtr);

  for (Value *V : std::get<ValueBundle>(BorC))
    createMissingConstantOperands(V);
  // If we created Instructions then create and return a Pack.
  SBValue *NewSBV = createSBPackInstruction(std::get<ValueBundle>(BorC));
  return NewSBV;
}

SBPackInstruction *
SBContext::createSBPackInstruction(const ValueBundle &PackInstrs) {
  assert(all_of(PackInstrs,
                [](Value *V) {
                  return isa<InsertElementInst>(V) ||
                         isa<ExtractElementInst>(V);
                }) &&
         "Expected inserts or extracts!");
  auto NewPtr = std::unique_ptr<SBPackInstruction>(
      new SBPackInstruction(PackInstrs, *this));
  return cast<SBPackInstruction>(registerSBValue(std::move(NewPtr)));
}

// Shuffle
SBShuffleInstruction *
SBContext::createSBShuffleInstruction(ShuffleMask &Mask, SBValue *Op,
                                          SBBasicBlock *SBBB) {
  auto NewPtr = std::unique_ptr<SBShuffleInstruction>(
      new SBShuffleInstruction(Mask, Op, SBBB));
  return cast<SBShuffleInstruction>(registerSBValue(std::move(NewPtr)));
}

SBShuffleInstruction *
SBContext::getSBShuffleInstruction(ShuffleVectorInst *ShuffleI) const {
  return cast_or_null<SBShuffleInstruction>(getSBValue(ShuffleI));
}

SBShuffleInstruction *
SBContext::createSBShuffleInstruction(ShuffleVectorInst *ShuffleI) {
  assert(getSBShuffleInstruction(ShuffleI) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<SBShuffleInstruction>(
      new SBShuffleInstruction(ShuffleI, *this));
  return cast<SBShuffleInstruction>(registerSBValue(std::move(NewPtr)));
}

SBShuffleInstruction *
SBContext::getOrCreateSBShuffleInstruction(ShuffleVectorInst *ShuffleI) {
  if (auto *Shuffle = getSBShuffleInstruction(ShuffleI))
    return Shuffle;
  return createSBShuffleInstruction(ShuffleI);
}

// Store
SBStoreInstruction *
SBContext::getSBStoreInstruction(StoreInst *SI) const {
  return cast_or_null<SBStoreInstruction>(getSBValue(SI));
}

SBStoreInstruction *SBContext::createSBStoreInstruction(StoreInst *SI) {
  assert(SI->getParent() != nullptr && "Detached!");
  assert(getSBStoreInstruction(SI) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<SBStoreInstruction>(
      new SBStoreInstruction(SI, *this));
  return cast<SBStoreInstruction>(registerSBValue(std::move(NewPtr)));
}

SBStoreInstruction *
SBContext::getOrCreateSBStoreInstruction(StoreInst *SI) {
  if (auto *SBSI = getSBStoreInstruction(SI))
    return SBSI;
  return createSBStoreInstruction(SI);
}

// Load
SBLoadInstruction *SBContext::getSBLoadInstruction(LoadInst *LI) const {
  return cast_or_null<SBLoadInstruction>(getSBValue(LI));
}

SBLoadInstruction *SBContext::createSBLoadInstruction(LoadInst *LI) {
  assert(getSBLoadInstruction(LI) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<SBLoadInstruction>(new SBLoadInstruction(LI, *this));
  return cast<SBLoadInstruction>(registerSBValue(std::move(NewPtr)));
}

SBLoadInstruction *SBContext::getOrCreateSBLoadInstruction(LoadInst *LI) {
  if (auto *SBLI = getSBLoadInstruction(LI))
    return SBLI;
  return createSBLoadInstruction(LI);
}

// Cast
SBCastInstruction *SBContext::getSBCastInstruction(CastInst *CI) const {
  return cast_or_null<SBCastInstruction>(getSBValue(CI));
}

SBCastInstruction *SBContext::createSBCastInstruction(CastInst *CI) {
  assert(getSBCastInstruction(CI) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<SBCastInstruction>(new SBCastInstruction(CI, *this));
  return cast<SBCastInstruction>(registerSBValue(std::move(NewPtr)));
}

SBCastInstruction *SBContext::getOrCreateSBCastInstruction(CastInst *CI) {
  if (auto *SBCI = getSBCastInstruction(CI))
    return SBCI;
  return createSBCastInstruction(CI);
}

// PHI
SBPHINode *SBContext::getSBPHINode(PHINode *PHI) const {
  return cast_or_null<SBPHINode>(getSBValue(PHI));
}

SBPHINode *SBContext::createSBPHINode(PHINode *PHI) {
  assert(getSBPHINode(PHI) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<SBPHINode>(new SBPHINode(PHI, *this));
  return cast<SBPHINode>(registerSBValue(std::move(NewPtr)));
}

SBPHINode *SBContext::getOrCreateSBPHINode(PHINode *PHI) {
  if (auto *SBPHI = getSBPHINode(PHI))
    return SBPHI;
  return createSBPHINode(PHI);
}

// Select
SBSelectInstruction *
SBContext::getSBSelectInstruction(SelectInst *SI) const {
  return cast_or_null<SBSelectInstruction>(getSBValue(SI));
}

SBSelectInstruction *
SBContext::createSBSelectInstruction(SelectInst *SI) {
  assert(getSBSelectInstruction(SI) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<SBSelectInstruction>(
      new SBSelectInstruction(SI, *this));
  return cast<SBSelectInstruction>(registerSBValue(std::move(NewPtr)));
}

SBSelectInstruction *
SBContext::getOrCreateSBSelectInstruction(SelectInst *SI) {
  if (auto *SBSI = getSBSelectInstruction(SI))
    return SBSI;
  return createSBSelectInstruction(SI);
}

// BinaryOperator
SBBinaryOperator *
SBContext::getSBBinaryOperator(BinaryOperator *BO) const {
  return cast_or_null<SBBinaryOperator>(getSBValue(BO));
}

SBBinaryOperator *SBContext::createSBBinaryOperator(BinaryOperator *BO) {
  assert(getSBBinaryOperator(BO) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<SBBinaryOperator>(new SBBinaryOperator(BO, *this));
  return cast<SBBinaryOperator>(registerSBValue(std::move(NewPtr)));
}

SBBinaryOperator *
SBContext::getOrCreateSBBinaryOperator(BinaryOperator *BO) {
  if (auto *SBBO = getSBBinaryOperator(BO))
    return SBBO;
  return createSBBinaryOperator(BO);
}

// UnaryOperator
SBUnaryOperator *SBContext::getSBUnaryOperator(UnaryOperator *UO) const {
  return cast_or_null<SBUnaryOperator>(getSBValue(UO));
}

SBUnaryOperator *SBContext::createSBUnaryOperator(UnaryOperator *UO) {
  assert(getSBUnaryOperator(UO) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<SBUnaryOperator>(new SBUnaryOperator(UO, *this));
  return cast<SBUnaryOperator>(registerSBValue(std::move(NewPtr)));
}

SBUnaryOperator *
SBContext::getOrCreateSBUnaryOperator(UnaryOperator *UO) {
  if (auto *SBUO = getSBUnaryOperator(UO))
    return SBUO;
  return createSBUnaryOperator(UO);
}

// Cmp
SBCmpInstruction *SBContext::getSBCmpInstruction(CmpInst *CI) const {
  return cast_or_null<SBCmpInstruction>(getSBValue(CI));
}

SBCmpInstruction *SBContext::createSBCmpInstruction(CmpInst *CI) {
  assert(getSBCmpInstruction(CI) == nullptr && "Already exists!");
  auto NewPtr =
      std::unique_ptr<SBCmpInstruction>(new SBCmpInstruction(CI, *this));
  return cast<SBCmpInstruction>(registerSBValue(std::move(NewPtr)));
}

SBCmpInstruction *SBContext::getOrCreateSBCmpInstruction(CmpInst *CI) {
  if (auto *SBCI = getSBCmpInstruction(CI))
    return SBCI;
  return createSBCmpInstruction(CI);
}

// Unpack
SBValue *SBContext::createSBUnpackInstruction(SBValue *Op,
                                                    unsigned UnpackLane,
                                                    SBBasicBlock *SBBB,
                                                    unsigned LanesToUnpack) {
  Value *V =
      SBUnpackInstruction::createIR(Op, SBBB, UnpackLane, LanesToUnpack);
  createMissingConstantOperands(V);
  auto *NewSBV = getOrCreateSBValue(V);
  return NewSBV;
}

SBUnpackInstruction *
SBContext::getSBUnpackInstruction(ExtractElementInst *ExtractI) const {
  auto *SBV = getSBValue(ExtractI);
  return SBV ? cast<SBUnpackInstruction>(SBV) : nullptr;
}

SBUnpackInstruction *
SBContext::createSBUnpackInstruction(ExtractElementInst *ExtractI) {
  assert(getSBUnpackInstruction(ExtractI) == nullptr && "Already exists!");
  auto *Op = getSBValue(ExtractI->getVectorOperand());
  assert(Op != nullptr &&
         "Please create the operand SBValue before calling this function!");
  Value *Idx = ExtractI->getIndexOperand();
  assert(isa<ConstantInt>(Idx) && "Can only handle constant int index!");
  auto Lane = cast<ConstantInt>(Idx)->getSExtValue();
  auto NewPtr = std::unique_ptr<SBUnpackInstruction>(
      new SBUnpackInstruction(ExtractI, Op, Lane, *this));
  assert(NewPtr->getOperand(0) == Op && "Bad operand!");
  return cast<SBUnpackInstruction>(registerSBValue(std::move(NewPtr)));
}

SBUnpackInstruction *
SBContext::getOrCreateSBUnpackInstruction(ExtractElementInst *ExtractI) {
  if (auto *Unpack = getSBUnpackInstruction(ExtractI))
    return Unpack;
  return createSBUnpackInstruction(ExtractI);
}

SBUnpackInstruction *
SBContext::getSBUnpackInstruction(ShuffleVectorInst *ShuffleI) const {
  auto *SBV = getSBValue(ShuffleI);
  return SBV ? cast<SBUnpackInstruction>(SBV) : nullptr;
}

SBUnpackInstruction *
SBContext::createSBUnpackInstruction(ShuffleVectorInst *ShuffleI) {
  assert(getSBUnpackInstruction(ShuffleI) == nullptr && "Already exists!");
  auto *Op = getSBValue(ShuffleI->getOperand(1));
  assert(Op != nullptr &&
         "Please create the operand SBValue before calling this function!");
  auto Lane = SBUnpackInstruction::getShuffleLane(ShuffleI);
  auto NewPtr = std::unique_ptr<SBUnpackInstruction>(
      new SBUnpackInstruction(ShuffleI, Op, Lane, *this));
  assert(NewPtr->getOperand(0) == Op && "Bad operand!");
  return cast<SBUnpackInstruction>(registerSBValue(std::move(NewPtr)));
}

SBUnpackInstruction *
SBContext::getOrCreateSBUnpackInstruction(ShuffleVectorInst *ShuffleI) {
  if (auto *Unpack = getSBUnpackInstruction(ShuffleI))
    return Unpack;
  return createSBUnpackInstruction(ShuffleI);
}

#ifndef NDEBUG
static std::optional<int> getPoisonVectorLanes(Value *V) {
  if (!isa<PoisonValue>(V))
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
static std::pair<ValueBundle, ValueBundle>
matchPackAndGetPackInstrs(InsertElementInst *PackBottomInsert) {
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
  ValueBundle PackInstrs;
  int TotalLanes = SBUtils::getNumLanes(PackBottomInsert);
  int ExpectedInsertLane = TotalLanes - 1;
  InsertElementInst *LastInsert = nullptr;
  ExtractElementInst *LastExtractInGroup = nullptr;
  // Walk the chain bottom-up collecting the matched instrs into `PackInstrs`
  for (Instruction *CurrI = PackBottomInsert;
       CurrI != nullptr && (ExpectedInsertLane >= 0 || ExpectedExtractLane > 0);
       ExpectedInsertLane -= (isa<InsertElementInst>(CurrI) ? 1 : 0),
                   CurrI = CurrI->getPrevNode()) {
    // Checks for both Insert and Extract:
    bool IsAtBottom = PackInstrs.empty();
    if (IsAtBottom) {
      // The bottom instr must be an Insert (Rule 1).
      if (!isa<InsertElementInst>(CurrI))
        return {};
    } else {
      // If not the last instruction and it does not have a single user then
      // discard it (Rule 6).
      if (!CurrI->hasOneUse())
        return {};
      // Discard user is not the previous instr in the pattern (Rule 6).
      User *SingleUser = *CurrI->users().begin();
      if (SingleUser != LastInsert)
        return {};
      assert(isa<InsertElementInst>(SingleUser) && "The user must be an Inset");
    }

    // We expect a constant lane that matches ExpectedInsertLane (Rules 1,2,7).
    if (auto InsertI = dyn_cast<InsertElementInst>(CurrI)) {
      auto LaneOpt = SBUtils::getConstantIndex(CurrI);
      if (!LaneOpt || *LaneOpt != ExpectedInsertLane)
        return {};
      assert((IsAtBottom ||
              cast<InsertElementInst>(*CurrI->users().begin())->getOperand(0) ==
                  CurrI) &&
             "CurrI must be the user's vector operand!");
      LastInsert = InsertI;
    } else if (auto *ExtractI = dyn_cast<ExtractElementInst>(CurrI)) {
      // The extract's lane must be constant (Rule 3a).
      auto ExtractLaneOpt = SBUtils::getExtractLane(ExtractI);
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
      assert(cast<InsertElementInst>(*CurrI->users().begin())->getOperand(1) ==
                 CurrI &&
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
  Instruction *TopI = cast<Instruction>(PackInstrs.back());
  assert((getPoisonVectorLanes(TopI->getOperand(0)) ||
          *SBUtils::getConstantIndex(TopI) == 0) &&
         "TopI is pointing to the wrong instruction!");
#endif // NDEBUG
  // If this is the top-most insert, its operand must be poison (Rule 5).
  if (!isa<PoisonValue>(LastInsert->getOperand(0)))
    return {};

  // Collect operands.
  ValueBundle Operands;
  for (unsigned Idx = 0, E = PackInstrs.size(); Idx < E; ++Idx) {
    Value *V = PackInstrs[Idx];
    if (isa<InsertElementInst>(V)) {
      Operands.push_back(cast<InsertElementInst>(V)->getOperand(1));
      continue;
    }
    assert(isa<ExtractElementInst>(V) && "Expected Extract!");
    auto *Extract = cast<ExtractElementInst>(V);
    Value *Op = Extract->getVectorOperand();
    Operands.push_back(Op);
    // Now we need to skip all Inserts and Extracts reading `Extract`.
    unsigned Skip = SBUtils::getNumLanes(Op) * 2 - 1;
    Idx += Skip;
  }
  return {PackInstrs, Operands};
}

SBValue *SBContext::getOrCreateSBValue(Value *V) {
  return getOrCreateSBValueInternal(V, 0);
}

SBValue *SBContext::getOrCreateSBValueInternal(Value *V, int Depth,
                                                     User *U) {
  assert(Depth < 666 && "Infinite recursion?");
  // TODO: Use switch-case with subclass IDs instead of `if`.
  if (auto *C = dyn_cast<Constant>(V)) {
    // Globals may be self-referencing, like @bar = global [1 x ptr] [ptr @bar].
    // Avoid infinite loops by early returning once we detect a loop.
    if (isa<GlobalValue>(C)) {
      if (Depth == 0)
        VisitedConstants.clear();
      if (!VisitedConstants.insert(C).second)
        return nullptr; //  recursion loop!
    }
    for (Value *COp : C->operands())
      getOrCreateSBValueInternal(COp, Depth + 1, C);
    return getOrCreateSBConstant(C);
  }
  if (auto *Arg = dyn_cast<Argument>(V)) {
    return getOrCreateSBArgument(Arg);
  }
  if (auto *BB = dyn_cast<BasicBlock>(V)) {
    assert(isa<BlockAddress>(U) &&
           "This won't create a SBBB, don't call this function directly!");
    if (auto *SBBB = getSBValue(BB))
      return SBBB;
    // TODO: return a SBOpaqueValue
    return nullptr;
  }
  assert(isa<Instruction>(V) && "Expected Instruction");
  switch (cast<Instruction>(V)->getOpcode()) {
  case Instruction::PHI:
    return getOrCreateSBPHINode(cast<PHINode>(V));
  case Instruction::ExtractElement: {
    // Check that all indices are ConstantInts.
    auto *ExtractI = cast<ExtractElementInst>(V);
    if (!isa<ConstantInt>(ExtractI->getIndexOperand()))
      return getOrCreateSBOpaqueInstruction(ExtractI);
    getOrCreateSBValueInternal(ExtractI->getVectorOperand(), Depth + 1);
    // ExtractI could be a member of either Unpack or Pack from vectors.
    if (SBValue *Extract = getSBValue(ExtractI))
      return Extract;
    return createSBUnpackInstruction(ExtractI);
  }
  case Instruction::ExtractValue:
    goto opaque_label;
  case Instruction::InsertElement: {
    auto *InsertI = dyn_cast<InsertElementInst>(V);
    if (auto *Insert = getSBValue(InsertI))
      return Insert;
    // Check if this is the bottom of an InsertElementInst packing pattern.
    auto [PackInstrs, PackOperands] = matchPackAndGetPackInstrs(InsertI);
    if (PackInstrs.empty())
      return getOrCreateSBOpaqueInstruction(InsertI);
    // Else create a new SBPackInstruction.
    return createSBPackInstruction(PackInstrs);
  }
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::FPExt:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
  case Instruction::Trunc:
  case Instruction::FPTrunc:
  case Instruction::BitCast:
    return getOrCreateSBCastInstruction(cast<CastInst>(V));
  case Instruction::FCmp:
  case Instruction::ICmp:
    return getOrCreateSBCmpInstruction(cast<CmpInst>(V));
  case Instruction::Select:
    return getOrCreateSBSelectInstruction(cast<SelectInst>(V));
  case Instruction::FNeg:
    return getOrCreateSBUnaryOperator(cast<UnaryOperator>(V));
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    return getOrCreateSBBinaryOperator(cast<BinaryOperator>(V));
  case Instruction::Load:
    return getOrCreateSBLoadInstruction(cast<LoadInst>(V));
  case Instruction::Store:
    return getOrCreateSBStoreInstruction(cast<StoreInst>(V));
  case Instruction::GetElementPtr:
  case Instruction::Call:
    goto opaque_label;
  case Instruction::ShuffleVector: {
    auto *ShuffleI = dyn_cast<ShuffleVectorInst>(V);
    // Check that we are only using the first operand.
    // TODO: Is a poison/undef operand always 2nd operand when canonicalized?
    if (SBUnpackInstruction::isUnpack(ShuffleI)) {
      getOrCreateSBValueInternal(ShuffleI->getOperand(0), Depth + 1);
      getOrCreateSBValueInternal(ShuffleI->getOperand(1), Depth + 1);
      return getOrCreateSBUnpackInstruction(ShuffleI);
    }
    if (ShuffleI->isSingleSource())
      return getOrCreateSBShuffleInstruction(ShuffleI);
    return getOrCreateSBOpaqueInstruction(ShuffleI);
  }
  default:
  opaque_label:
    return getOrCreateSBOpaqueInstruction(cast<Instruction>(V));
  }
}

bool SBBasicBlock::classof(const SBValue *From) {
  return From->getSubclassID() == SBValue::ClassID::Block;
}

void SBBasicBlock::buildSBBasicBlockFromIR(BasicBlock *BB) {
  for (Instruction &IRef : reverse(*BB)) {
    Instruction *I = &IRef;
    SBValue *SBV = Ctxt.getOrCreateSBValue(I);
    for (auto [OpIdx, Op] : enumerate(I->operands())) {
      // For now Unpacks only have a single operand.
      if (isa<SBUnpackInstruction>(SBV) && OpIdx > 0)
        continue;
      // For now Shuffles only have a single operand.
      if (isa<SBShuffleInstruction>(SBV) && OpIdx > 0)
        continue;
      // Skip instruction's label operands
      if (isa<BasicBlock>(Op))
        continue;
      // Skip metadata for now
      if (isa<MetadataAsValue>(Op))
        continue;
      // Skip asm
      if (isa<InlineAsm>(Op))
        continue;
      Ctxt.getOrCreateSBValue(Op);
    }
  }
}

SBBasicBlock::iterator SBBasicBlock::getFirstNonPHIIt() {
  Instruction *FirstI = cast<BasicBlock>(Val)->getFirstNonPHI();
  return FirstI == nullptr
             ? begin()
             : cast<SBInstruction>(Ctxt.getSBValue(FirstI))->getIterator();
}

SBBasicBlock::SBBasicBlock(BasicBlock *BB, SBContext &SBCtxt)
    : SBValue(ClassID::Block, BB, SBCtxt) {}

SBBasicBlock::~SBBasicBlock() {
  Ctxt.SchedForSBBB.erase(this);
  // This BB is now gone, so there is no need for the BB-specific callbacks.
  Ctxt.RemoveInstrCallbacksBB.erase(this);
  Ctxt.InsertInstrCallbacksBB.erase(this);
  Ctxt.MoveInstrCallbacksBB.erase(this);
}

SBFunction *SBBasicBlock::getParent() const {
  auto *BB = cast<BasicBlock>(Val);
  auto *F = BB->getParent();
  if (F == nullptr)
    // Detached
    return nullptr;
  return Ctxt.getSBFunction(F);
}

SBBasicBlock::iterator SBBasicBlock::begin() const {
  BasicBlock *BB = cast<BasicBlock>(Val);
  BasicBlock::iterator It = BB->begin();
  if (!BB->empty()) {
    auto *SBV = Ctxt.getSBValue(&*BB->begin());
    assert(SBV != nullptr && "No SandboxIR for BB->begin()!");
    auto *SBI = cast<SBInstruction>(SBV);
    unsigned Num = SBI->getNumOfIRInstrs();
    assert(Num >= 1u && "Bad getNumOfIRInstrs()");
    It = std::next(It, Num - 1);
  }
  return iterator(BB, It, &Ctxt);
}

void SBBasicBlock::detach() {
  // We are detaching bottom-up because detaching some SandboxIR
  // Instructions require non-detached operands.
  // Note: we are in the process of detaching from the underlying BB, so we
  //       can't rely on 1-1 mapping between IR and SandboxIR.
  for (Instruction &I : reverse(*cast<BasicBlock>(Val))) {
    if (auto *SI = Ctxt.getSBValue(&I))
      Ctxt.detach(SI);
  }
}

void SBBasicBlock::detachFromLLVMIR() {
  // Detach instructions
  detach();
  // Detach the actual BB
  Ctxt.detach(this);
}

SBArgument *SBContext::getSBArgument(Argument *Arg) const {
  return cast_or_null<SBArgument>(getSBValue(Arg));
}

SBArgument *SBContext::createSBArgument(Argument *Arg) {
  assert(getSBArgument(Arg) == nullptr && "Already exists!");
  auto NewArg = std::unique_ptr<SBArgument>(new SBArgument(Arg, *this));
  return cast<SBArgument>(registerSBValue(std::move(NewArg)));
}

SBArgument *SBContext::getOrCreateSBArgument(Argument *Arg) {
  // TODO: Try to avoid two lookups in getOrCreate functions.
  if (auto *TArg = getSBArgument(Arg))
    return TArg;
  return createSBArgument(Arg);
}

SBOpaqueInstruction *
SBContext::getSBOpaqueInstruction(Instruction *I) const {
  return cast_or_null<SBOpaqueInstruction>(getSBValue(I));
}

SBOpaqueInstruction *
SBContext::createSBOpaqueInstruction(Instruction *I) {
  assert(getSBOpaqueInstruction(I) == nullptr && "Already exists!");
  auto NewPtr = std::unique_ptr<SBOpaqueInstruction>(
      new SBOpaqueInstruction(I, *this));
  return cast<SBOpaqueInstruction>(registerSBValue(std::move(NewPtr)));
}

SBOpaqueInstruction *
SBContext::getOrCreateSBOpaqueInstruction(Instruction *I) {
  assert(!isa<Constant>(I) && "Please use getOrCreateSBConstant()");
  assert(!isa<Argument>(I) && "Please use getOrCreateSBArgument()");
  auto *SBV = getSBOpaqueInstruction(I);
  if (SBV != nullptr)
    return SBV;
  return createSBOpaqueInstruction(I);
}

SBBasicBlock *SBContext::getSBBasicBlock(BasicBlock *BB) const {
  return cast_or_null<SBBasicBlock>(getSBValue(BB));
}

SBBasicBlock *SBContext::createSBBasicBlock(BasicBlock *BB) {
  assert(getSBBasicBlock(BB) == nullptr && "Already exists!");
  auto NewBBPtr =
      std::unique_ptr<SBBasicBlock>(new SBBasicBlock(BB, *this));
  auto *SBBB = cast<SBBasicBlock>(registerSBValue(std::move(NewBBPtr)));
  // Create SandboxIR for BB's body.
  SBBB->buildSBBasicBlockFromIR(BB);

  // Create a scheduler object for this particular SBBB.
  // Note: This should be done *after* we populate SBBB.
  auto Pair = SchedForSBBB.try_emplace(
      SBBB, std::unique_ptr<Scheduler, SchedulerDeleter>(
                  new Scheduler(*SBBB, AA, *this)));
  (void)Pair;
  assert(Pair.second && "Creating a scheduler for SBBB for the second time!");

  return SBBB;
}

SBFunction *SBContext::getSBFunction(Function *F) const {
  return cast_or_null<SBFunction>(getSBValue(F));
}

SBFunction *SBContext::createSBFunction(Function *F, bool CreateBBs) {
  assert(getSBFunction(F) == nullptr && "Already exists!");
  auto NewFPtr = std::unique_ptr<SBFunction>(new SBFunction(F, *this));
  // Create arguments.
  for (auto &Arg : F->args())
    getOrCreateSBArgument(&Arg);
  // Create BBs.
  if (CreateBBs) {
    for (auto &BB : *F)
      createSBBasicBlock(&BB);
  }
  return cast<SBFunction>(registerSBValue(std::move(NewFPtr)));
}

SBInstruction &SBBasicBlock::front() const {
  auto *BB = cast<BasicBlock>(Val);
  assert(!BB->empty() && "Empty block!");
  auto *SBI = cast<SBInstruction>(getContext().getSBValue(&*BB->begin()));
  assert(SBI != nullptr && "Expected Instr!");
  return *SBI;
}

SBInstruction &SBBasicBlock::back() const {
  auto *BB = cast<BasicBlock>(Val);
  assert(!BB->empty() && "Empty block!");
  auto *SBI =
      cast<SBInstruction>(getContext().getSBValue(&*BB->rbegin()));
  assert(SBI != nullptr && "Expected Instr!");
  return *SBI;
}

#ifndef NDEBUG
void SBBasicBlock::verify() {
  // Check that all llvm instructions in BB have a corresponding SBValue.
  auto *BB = cast<BasicBlock>(Val);
  for (Instruction &IRef : *BB)
    assert(getContext().getSBValue(&IRef) != nullptr &&
           "No SBValue for IRef!");

  // Note: we are not simply doing bool HaveSandboxIRForWholeFn = getParent()
  // because there is the corner case of @function operands of constants.
  SBValue *SBF = Ctxt.getSBValue(BB->getParent());
  bool HaveSandboxIRForWholeFn = SBF != nullptr && isa<SBFunction>(SBF);
  // Check operand/user consistency.
  for (const SBInstruction &SBI : *this) {
    Value *V = ValueAttorney::getValue(&SBI);
    assert(!isa<BasicBlock>(V) && "Broken SBBasicBlock construction!");
    for (auto [OpIdx, Use] : enumerate(SBI.operands())) {
      SBValue *Op = Use;
      if (HaveSandboxIRForWholeFn)
        assert(Op != nullptr && "Null operands are not allowed when we have "
                                "SandboxIR for the whole function");
      if (Op == nullptr)
        continue;
      // Op could be an operand of a ConstantVector. We don't model this.
      assert((isa<Constant>(ValueAttorney::getValue(Op)) ||
              find(Op->users(), &SBI) != Op->users().end()) &&
             "If Op is SBI's operand, then SBI should be in Op's users.");
      // Count how many times Op is found in operands:
      unsigned CntOpEdges = 0;
      for_each(SBI.operands(), [&CntOpEdges, Op](SBValue *TmpOp) {
        if (TmpOp == Op)
          ++CntOpEdges;
      });
      if (CntOpEdges > 1) {
        // Check that Op has `CntOp` users matching `SBI`.
        unsigned CntUserEdges = 0;
        for_each(Op->users(), [&CntUserEdges, &SBI](SBUser *User) {
          if (User == &SBI)
            ++CntUserEdges;
        });
        assert(
            CntOpEdges == CntUserEdges &&
            "Broken IR! User edges count doesn't match operand edges count!");
      }
    }
    for (auto *User : SBI.users()) {
      if (User == nullptr)
        continue;
      assert(find(User->operands(), &SBI) != User->operands().end() &&
             "If SBU is in SBI's users, then SBI should be in SBU's "
             "operands.");
    }
  }

  SBInstruction *LastNonPHI = nullptr;
  // Checks opcodes and other.
  for (SBInstruction &SBI : *this) {
    if (LLVM_UNLIKELY(SBI.isPad())) {
      assert(&SBI == &*getFirstNonPHIIt() &&
             "{Landing,Catch,Cleanup}Pad Instructions must be the non-PHI!");
    }
    switch (SBI.getSubclassID()) {
    case SBValue::ClassID::Pack: {
      const auto *Pack = cast<SBPackInstruction>(&SBI);
      if (any_of(Pack->operands(),
                 [](SBValue *Op) { return Op->isVector(); }))
        assert((isa<FixedVectorType>(Pack->getOperand(0)->getType()) ||
                Pack->getNumOperands() <
                    cast<FixedVectorType>(Pack->getExpectedType())
                        ->getNumElements()) &&
               "This has vector operands. We expect fewer operands than lanes");
      Pack->verify();
      break;
    }
    case SBValue::ClassID::Shuffle: {
      const auto *Shuffle = cast<SBShuffleInstruction>(&SBI);
      assert(Shuffle->getMask().size() == Shuffle->lanes() &&
             "Expected same number of indices as lanes.");
      assert((int)Shuffle->lanes() ==
                 SBUtils::getNumLanes(Shuffle->getOperand(0)->getType()) &&
             "A SBShuffle should not unpack, it should only reorder lanes!");
      Shuffle->getMask().verify();
      assert(Shuffle->getNumOperands() == 1 && "Expected a single operand");
      break;
    }
    case SBValue::ClassID::Unpack: {
      const auto *Unpack = cast<SBUnpackInstruction>(&SBI);
      (void)Unpack;
      break;
    }
    case SBValue::ClassID::OpaqueInstr: {
      Value *V = ValueAttorney::getValue(&SBI);
      (void)V;
      break;
    }
    case SBValue::ClassID::Argument:
      assert(isa<Argument>(ValueAttorney::getValue(&SBI)) &&
             "Expected Argument!");
      break;
    case SBValue ::ClassID::User:
      assert(isa<User>(ValueAttorney::getValue(&SBI)) && "Expected User!");
      break;
    case SBValue::ClassID::Constant:
      assert(isa<Constant>(ValueAttorney::getValue(&SBI)) &&
             "Expected Constant!");
      break;
    case SBValue::ClassID::Block:
      assert(isa<BasicBlock>(ValueAttorney::getValue(&SBI)) &&
             "Expected BasicBlock!");
      break;
    case SBValue::ClassID::Function:
      assert(isa<Function>(ValueAttorney::getValue(&SBI)) &&
             "Expected Function!");
      break;
    case SBValue::ClassID::Store:
      assert(isa<StoreInst>(ValueAttorney::getValue(&SBI)) &&
             "Expected StoreInst!");
      break;
    case SBValue::ClassID::Cmp:
      assert(isa<CmpInst>(ValueAttorney::getValue(&SBI)) &&
             "Expected CmpInst!");
      break;
    case SBValue::ClassID::Load:
      assert(isa<LoadInst>(ValueAttorney::getValue(&SBI)) &&
             "Expected LoadInst!");
      break;
    case SBValue::ClassID::Cast:
      assert(isa<CastInst>(ValueAttorney::getValue(&SBI)) &&
             "Expected CastInst!");
      break;
    case SBValue::ClassID::PHI:
      assert(isa<PHINode>(ValueAttorney::getValue(&SBI)) &&
             "Expected PHINode!");
      if (LastNonPHI != nullptr) {
        errs() << "SBPHIs not grouped at top of BB!\n";
        errs() << SBI << "\n";
        errs() << *LastNonPHI << "\n";
        llvm_unreachable("Broken SandboxIR");
      }
      break;
    case SBValue::ClassID::Select:
      assert(isa<SelectInst>(ValueAttorney::getValue(&SBI)) &&
             "Expected SelectInst!");
      break;
    case SBValue::ClassID::BinOp:
      assert(isa<BinaryOperator>(ValueAttorney::getValue(&SBI)) &&
             "Expected BinaryOperator!");
      break;
    case SBValue::ClassID::UnOp:
      assert(isa<UnaryOperator>(ValueAttorney::getValue(&SBI)) &&
             "Expected UnaryOperator!");
      break;
    }

    if (!isa<SBPHINode>(&SBI))
      LastNonPHI = &SBI;
  }

  // Check that we only have a single SBValue for every constant.
  DenseMap<Value *, const SBValue *> Map;
  for (const SBValue &SBV : *this) {
    Value *V = ValueAttorney::getValue(&SBV);
    if (isa<Constant>(V)) {
      auto Pair = Map.insert({V, &SBV});
      if (!Pair.second) {
        auto It = Pair.first;
        assert(&SBV == It->second &&
               "Expected a unique SBValue for each LLVM IR constant!");
      }
    }
  }
}

void SBBasicBlock::verifyIR() const {
#ifdef SBVEC_EXPENSIVE_CHECKS
  // Check that all llvm instructions in BB have a corresponding SBValue.
  auto *BB = cast<BasicBlock>(Val);
  Instruction *LastI = nullptr;
  for (Instruction &IRef : *BB) {
    Instruction *I = &IRef;
    for (Value *Op : I->operands()) {
      auto *OpI = dyn_cast<Instruction>(Op);
      if (OpI == nullptr)
        continue;
      if (OpI->getParent() != BB)
        continue;
      if (!isa<PHINode>(I) && !OpI->comesBefore(I)) {
        errs() << "Instruction does not dominate uses!\n";
        errs() << *Op << " " << Op << "\n";
        errs() << *I << " " << I << "\n";
        errs() << "\n";

        errs() << "SBValues:\n";
        auto *SBOp = Ctxt.getSBValue(Op);
        if (SBOp != nullptr)
          errs() << *SBOp << " " << SBOp << "\n";
        else
          errs() << "No SBValue for Op\n";
        auto *SBI = Ctxt.getSBValue(I);
        if (SBI != nullptr)
          errs() << *SBI << " " << SBI << "\n";
        else
          errs() << "No SBValue for I\n";
        llvm_unreachable("Instruction does not dominate uses!");
      }
    }

    if (LastI != nullptr && isa<PHINode>(I) && !isa<PHINode>(LastI)) {
      errs() << "PHIs not grouped at top of BB!\n";
      errs() << *LastI << " " << LastI << "\n";
      errs() << *I << " " << I << "\n";
      llvm_unreachable("PHIs not grouped at top of BB!\n");
    }
    LastI = I;
  }
#endif
}

void SBBasicBlock::dumpVerbose(raw_ostream &OS) const {
  for (const auto &SBI : reverse(*this)) {
    SBI.dumpVerbose(OS);
    OS << "\n";
  }
}
void SBBasicBlock::dumpVerbose() const {
  dumpVerbose(dbgs());
  dbgs() << "\n";
}
void SBBasicBlock::dump(raw_ostream &OS) const {
  BasicBlock *BB = cast<BasicBlock>(Val);
  const auto &Name = BB->getName();
  OS << Name;
  if (!Name.empty())
    OS << ":\n";
  // If there are Instructions in the BB that are not mapped to SandboxIR, then use
  // a crash-proof dump.
  if (any_of(*BB, [this](Instruction &I) {
        return Ctxt.getSBValue(&I) == nullptr;
      })) {
    OS << "<Crash-proof mode!>\n";
    DenseSet<SBInstruction *> Visited;
    for (Instruction &IRef : *BB) {
      SBValue *SBV = Ctxt.getSBValue(&IRef);
      if (SBV == nullptr)
        OS << IRef << " *** No SandboxIR ***\n";
      else {
        auto *SBI = dyn_cast<SBInstruction>(SBV);
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
void SBBasicBlock::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBBasicBlock::dumpInstrs(SBValue *SBV, int Num) const {
  auto *SBI = dyn_cast<SBInstruction>(SBV);
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
  for (SBInstruction *I = FromI, *E = ToI->getNextNode(); I != E;
       I = I->getNextNode())
    dbgs() << *I << "\n";
}
#endif

SandboxIRTracker &SBBasicBlock::getTracker() { return Ctxt.getTracker(); }

SBInstruction *SBBasicBlock::getTerminator() const {
  auto *TerminatorV = Ctxt.getSBValue(cast<BasicBlock>(Val)->getTerminator());
  return cast_or_null<SBInstruction>(TerminatorV);
}

SBBasicBlock::iterator::pointer
SBBasicBlock::iterator::getSBI(BasicBlock::iterator It) const {
  SBInstruction *SBI =
      cast_or_null<SBInstruction>(SBCtxt->getSBValue(&*It));
  assert(
      (!SBI || cast<Instruction>(ValueAttorney::getValue(SBI)) == &*It) &&
      "It should always point at the bottom IR instruction of a "
      "SBInstruction!");
  return SBI;
}

SBContext::RemoveCBTy *
SBContext::registerRemoveInstrCallback(RemoveCBTy CB) {
  std::unique_ptr<RemoveCBTy> CBPtr(new RemoveCBTy(CB));
  RemoveCBTy *CBRaw = CBPtr.get();
  RemoveInstrCallbacks.push_back(std::move(CBPtr));
  return CBRaw;
}

void SBContext::unregisterRemoveInstrCallback(RemoveCBTy *CB) {
  auto It = find_if(RemoveInstrCallbacks,
                    [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert((InQuickFlush || It != RemoveInstrCallbacks.end()) &&
         "Callback not registered!");
  if (It != RemoveInstrCallbacks.end())
    RemoveInstrCallbacks.erase(It);
}

SBContext::InsertCBTy *
SBContext::registerInsertInstrCallback(InsertCBTy CB) {
  std::unique_ptr<InsertCBTy> CBPtr(new InsertCBTy(CB));
  InsertCBTy *CBRaw = CBPtr.get();
  InsertInstrCallbacks.push_back(std::move(CBPtr));
  return CBRaw;
}

void SBContext::unregisterInsertInstrCallback(InsertCBTy *CB) {
  auto It = find_if(InsertInstrCallbacks,
                    [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert(It != InsertInstrCallbacks.end() && "Callback not registered!");
  InsertInstrCallbacks.erase(It);
}

SBContext::MoveCBTy *SBContext::registerMoveInstrCallback(MoveCBTy CB) {
  std::unique_ptr<MoveCBTy> CBPtr(new MoveCBTy(CB));
  MoveCBTy *CBRaw = CBPtr.get();
  MoveInstrCallbacks.push_back(std::move(CBPtr));
  return CBRaw;
}

void SBContext::unregisterMoveInstrCallback(MoveCBTy *CB) {
  auto It = find_if(MoveInstrCallbacks,
                    [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert(It != MoveInstrCallbacks.end() && "Callback not registered!");
  MoveInstrCallbacks.erase(It);
}

SBContext::RemoveCBTy *
SBContext::registerRemoveInstrCallbackBB(SBBasicBlock &BB, RemoveCBTy CB) {
  std::unique_ptr<RemoveCBTy> CBPtr(new RemoveCBTy(CB));
  RemoveCBTy *CBRaw = CBPtr.get();
  RemoveInstrCallbacksBB[&BB].push_back(std::move(CBPtr));
  return CBRaw;
}

void SBContext::unregisterRemoveInstrCallbackBB(SBBasicBlock &BB,
                                                  RemoveCBTy *CB) {
  auto MapIt = RemoveInstrCallbacksBB.find(&BB);
  assert((InQuickFlush || MapIt != RemoveInstrCallbacksBB.end()) &&
         "Callback not registered at BB!");
  auto &Vec = MapIt->second;
  auto It = find_if(Vec, [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert((InQuickFlush || It != Vec.end()) && "Callback not registered!");
  if (It != Vec.end())
    Vec.erase(It);
}

SBContext::InsertCBTy *
SBContext::registerInsertInstrCallbackBB(SBBasicBlock &BB, InsertCBTy CB) {
  std::unique_ptr<InsertCBTy> CBPtr(new InsertCBTy(CB));
  InsertCBTy *CBRaw = CBPtr.get();
  InsertInstrCallbacksBB[&BB].push_back(std::move(CBPtr));
  return CBRaw;
}

void SBContext::unregisterInsertInstrCallbackBB(SBBasicBlock &BB,
                                                  InsertCBTy *CB) {
  auto MapIt = InsertInstrCallbacksBB.find(&BB);
  assert((InQuickFlush || MapIt != InsertInstrCallbacksBB.end()) &&
         "Callback not registered at BB!");
  auto &Vec = MapIt->second;
  auto It = find_if(Vec, [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert(It != Vec.end() && "Callback not registered!");
  Vec.erase(It);
}

SBContext::MoveCBTy *
SBContext::registerMoveInstrCallbackBB(SBBasicBlock &BB, MoveCBTy CB) {
  std::unique_ptr<MoveCBTy> CBPtr(new MoveCBTy(CB));
  MoveCBTy *CBRaw = CBPtr.get();
  MoveInstrCallbacksBB[&BB].push_back(std::move(CBPtr));
  return CBRaw;
}

void SBContext::unregisterMoveInstrCallbackBB(SBBasicBlock &BB,
                                                MoveCBTy *CB) {
  auto MapIt = MoveInstrCallbacksBB.find(&BB);
  assert((InQuickFlush || MapIt != MoveInstrCallbacksBB.end()) &&
         "Callback not registered at BB!");
  auto &Vec = MapIt->second;
  auto It = find_if(Vec, [CB](const auto &FPtr) { return FPtr.get() == CB; });
  assert(It != Vec.end() && "Callback not registered!");
  Vec.erase(It);
}

void SBContext::quickFlush() {
  InQuickFlush = true;
  SchedForSBBB.clear();

  RemoveInstrCallbacks.clear();
  InsertInstrCallbacks.clear();
  MoveInstrCallbacks.clear();

  RemoveInstrCallbacksBB.clear();
  InsertInstrCallbacksBB.clear();
  MoveInstrCallbacksBB.clear();

  Sched = nullptr;
  LLVMValueToSBValueMap.clear();
  MultiInstrMap.clear();
  InQuickFlush = false;
}

void SBContext::runRemoveInstrCallbacks(SBInstruction *SBI) {
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

void SBContext::runInsertInstrCallbacks(SBInstruction *SBI) {
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

void SBContext::runMoveInstrCallbacks(SBInstruction *SBI,
                                        SBBasicBlock &SBBB,
                                        const SBBBIterator &WhereIt) {
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
