//===- Analysis.cpp - Legality analysis -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Analysis.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"

using namespace llvm;

#define DEBUG_TYPE "SBVec"

namespace llvm {

const char *resultIDToStr(ResultID ID) {
  switch (ID) {
  case ResultID::PerfectReuseVector:
    return "PerfectReuseVector";
  case ResultID::UnpackAndPack:
    return "UnpackAndPack";
  case ResultID::Unpack:
    return "Unpack";
  case ResultID::UnpackTopDown:
    return "UnpackTopDown";
  case ResultID::Pack:
    return "Pack";
  case ResultID::SimpleWiden:
    return "SimpleWiden";
  case ResultID::KeepLane0:
    return "KeepLane0";
  }
  llvm_unreachable("Unimplemented ID");
}

raw_ostream &operator<<(raw_ostream &OS, ResultID ID) {
  OS << resultIDToStr(ID);
  return OS;
}

const char *noVecReasonToStr(NoVecReason R) {
  switch (R) {
  case NoVecReason::NonInstructions:
    return "NonInstructions";
  case NoVecReason::DiffOpcodes:
    return "DiffOpcodes";
  case NoVecReason::DiffTypes:
    return "DiffTypes";
  case NoVecReason::DiffMathFlags:
    return "DiffMathFlags";
  case NoVecReason::DiffWrapFlags:
    return "DiffWrapFlags";
  case NoVecReason::UnpackAndShuffle:
    return "UnpackAndShuffle";
  case NoVecReason::CantSchedule:
    return "CantSchedule";
  case NoVecReason::NonConsecutive:
    return "NonConsecutive";
  case NoVecReason::Other:
    return "Other";
  case NoVecReason::Opaque:
    return "Opaque";
  case NoVecReason::OrigSource:
    return "OrigSource";
  case NoVecReason::UnsupportedOpcode:
    return "UnsupportedOpcode";
  case NoVecReason::CrossBBs:
    return "CrossBBs";
  case NoVecReason::RecursionLimit:
    return "RecursionLimit";
  case NoVecReason::Unimplemented:
    return "Unimplemented";
  }
  llvm_unreachable("Unimplemented Reason");
}
} // namespace llvm

AnalysisResult::AnalysisResult(ResultID ID, const SBValBundle &Vals,
                               SBAnalysis &Analysis)
    : SubclassID(ID), Analysis(Analysis) {
  Analysis.updateMap(Vals, this);
}

void SBAnalysis::updateMap(const SBValBundle &Vals, AnalysisResult *Result) {
  for (auto *SBV : Vals)
    ValueToAnalysisMap[SBV] = Result;
#ifndef NDEBUG
  BndlAnalysisVec.emplace_back(Vals, Result);
#endif
}

SBAnalysis::SBAnalysis(ScalarEvolution &SE, const DataLayout &DL)
    : SE(SE), DL(DL) {}

void SBAnalysis::init(SBBasicBlock &SBBB) {
  assert(ResultPool.empty() && ValueToAnalysisMap.empty() &&
         "Forgot to call clean()?");
  this->SBBB = &SBBB;
}

#ifndef NDEBUG
void AnalysisResult::dump(raw_ostream &OS) const { dumpCommon(OS); }
LLVM_DUMP_METHOD void AnalysisResult::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

PerfectReuseVector::PerfectReuseVector(SBValue *CommonOp,
                                       const SBValBundle &Vals,
                                       SBAnalysis &A)
    : AnalysisResult(ResultID::PerfectReuseVector, Vals, A), CommonOp(CommonOp),
      ShuffMask(ShuffleMask::getIdentity(CommonOp->lanes())) {}

#ifndef NDEBUG
void PerfectReuseVector::dump(raw_ostream &OS) const {
  dumpCommon(OS);
  OS << "\n";
  OS.indent(2) << "CommonOp: ";
  if (CommonOp != nullptr)
    OS << *CommonOp;
  else
    OS << "NULL\n";
  OS.indent(2) << "ShuffleMask: " << ShuffMask;
}
LLVM_DUMP_METHOD void PerfectReuseVector::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
LLVM_DUMP_METHOD void UnpackAndPack::dump() const {
  UnpackAndPack::dump(dbgs());
  dbgs() << "\n";
}
LLVM_DUMP_METHOD void AnalysisResultWithReason::dump() const {
  AnalysisResultWithReason::dump(dbgs());
  dbgs() << "\n";
}
LLVM_DUMP_METHOD void UnpackTopDown::dump() const {
  UnpackTopDown::dump(dbgs());
  dbgs() << "\n";
}
LLVM_DUMP_METHOD void Unpack::dump() const {
  Unpack::dump(dbgs());
  dbgs() << "\n";
}
LLVM_DUMP_METHOD void Pack::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
void Pack::dump(raw_ostream &OS) const { AnalysisResultWithReason::dump(OS); }
#endif // NDEBUG

std::optional<NoVecReason> SBAnalysis::cantVectorizeBasedOnOpcodesAndTypes(
    const SBValBundle &SBBndl) {
  auto *TI0 = cast<SBInstruction>(SBBndl[0]);
  auto Opcode = TI0->getOpcode();
  // If they have different opcodes, then we cannot form a vector (for now).
  if (any_of(drop_begin(SBBndl), [Opcode](SBValue *TV) {
        return cast<SBInstruction>(TV)->getOpcode() != Opcode;
      }))
    return NoVecReason::DiffOpcodes;

  // If not the same scalar type, Pack.
  if (SBUtils::getCommonScalarType(SBBndl) == nullptr)
    return NoVecReason::DiffTypes;

  // TODO: Allow vectorization of instrs with different flags as long as we
  // change them to the least common one.
  // For now pack if differnt FastMathFlags.
  if (Opcode != SBInstruction::Opcode::Opaque && isa<SBInstruction>(TI0) &&
      cast<SBInstruction>(TI0)->isFPMath() &&
      SBUtils::differentMathFlags(SBBndl))
    return NoVecReason::DiffMathFlags;

  if (Opcode != SBInstruction::Opcode::Opaque && isa<SBInstruction>(TI0) &&
      SBUtils::differentWrapFlags(SBBndl))
    return NoVecReason::DiffWrapFlags;

  // Now we need to do further checks for specific opocdes.
  switch (Opcode) {
  case SBInstruction::Opcode::Shuffle:
    return NoVecReason::Other;
  case SBInstruction::Opcode::Pack:
    // TODO: Implement this
    return NoVecReason::Unimplemented;
  case SBInstruction::Opcode::Unpack: {
    // We can combine them if they extract from the same vector and if the
    // indices are all constant. In code generation we will emit a shuffle.
    auto *OpN0 = TI0->getOperand(0);
    if (all_of(SBBndl, [OpN0](SBValue *SBV) {
          return cast<SBUser>(SBV)->getOperand(0) == OpN0;
        }))
      return std::nullopt;
    // TODO: I think this needs a new pack reason, like 'DiffInternals'?
    return NoVecReason::Other;
  }
  case SBInstruction::Opcode::ZExt:
  case SBInstruction::Opcode::SExt:
  case SBInstruction::Opcode::FPToUI:
  case SBInstruction::Opcode::FPToSI:
  case SBInstruction::Opcode::FPExt:
  case SBInstruction::Opcode::PtrToInt:
  case SBInstruction::Opcode::IntToPtr:
  case SBInstruction::Opcode::SIToFP:
  case SBInstruction::Opcode::UIToFP:
  case SBInstruction::Opcode::Trunc:
  case SBInstruction::Opcode::FPTrunc:
  case SBInstruction::Opcode::BitCast: {
    // We have already checked that they are of the same opcode.
    assert(all_of(SBBndl,
                  [Opcode](SBValue *SBV) {
                    return cast<SBInstruction>(SBV)->getOpcode() == Opcode;
                  }) &&
           "Different opcodes, should have early returned!");
    Type *FromTy0 = TI0->getOperand(0)->getExpectedType();
    if (any_of(drop_begin(SBBndl), [FromTy0](SBValue *SBV) {
          return cast<SBUser>(SBV)->getOperand(0)->getExpectedType() !=
                 FromTy0;
        }))
      return NoVecReason::DiffTypes;
    return std::nullopt;
  }
  case SBInstruction::Opcode::FCmp:
  case SBInstruction::Opcode::ICmp: {
    // We need the same predicate..
    auto Pred0 = cast<SBCmpInstruction>(TI0)->getPredicate();
    bool Same = all_of(SBBndl, [Pred0](SBValue *SBV) {
      return cast<SBCmpInstruction>(SBV)->getPredicate() == Pred0;
    });
    if (Same)
      return std::nullopt;
    return NoVecReason::DiffOpcodes;
  }
  case SBInstruction::Opcode::Select:
    return std::nullopt;
  case SBInstruction::Opcode::FNeg:
    return std::nullopt;
  case SBInstruction::Opcode::Add:
  case SBInstruction::Opcode::FAdd:
  case SBInstruction::Opcode::Sub:
  case SBInstruction::Opcode::FSub:
  case SBInstruction::Opcode::Mul:
  case SBInstruction::Opcode::FMul:
  case SBInstruction::Opcode::FRem:
  case SBInstruction::Opcode::UDiv:
  case SBInstruction::Opcode::SDiv:
  case SBInstruction::Opcode::FDiv:
  case SBInstruction::Opcode::URem:
  case SBInstruction::Opcode::SRem:
  case SBInstruction::Opcode::Shl:
  case SBInstruction::Opcode::LShr:
  case SBInstruction::Opcode::AShr:
  case SBInstruction::Opcode::And:
  case SBInstruction::Opcode::Or:
  case SBInstruction::Opcode::Xor:
    return std::nullopt;
  case SBInstruction::Opcode::Load:
    if (SBUtils::areConsecutive<SBLoadInstruction>(SBBndl, SE, DL))
      return std::nullopt;
    return NoVecReason::NonConsecutive;
  case SBInstruction::Opcode::Store:
    if (SBUtils::areConsecutive<SBStoreInstruction>(SBBndl, SE, DL))
      return std::nullopt;
    return NoVecReason::NonConsecutive;
  case SBInstruction::Opcode::Opaque:
    return NoVecReason::Opaque;
  }
}

static std::optional<ShuffleMask>
getShuffleMaskForUnpacks(const SBValBundle &Bndl) {
  ShuffleMask::IndicesVecT Indices;
  Indices.reserve(Bndl.size());
  for (auto *SBV : Bndl) {
    auto *Unpack = cast<SBUnpackInstruction>(SBV);
    if (Unpack == nullptr)
      return std::nullopt;
    Indices.push_back(Unpack->getUnpackLane());
  }
  return ShuffleMask(std::move(Indices));
}

AnalysisResult *SBAnalysis::getBndlAnalysis(
    const SBValBundle &SBBndl, const InstructionMaps &InstrMaps,
    SmallPtrSet<SBInstruction *, 4> &ProbablyDead, bool TopDown) {
  // Find all unique nodes that contain SBBndl values.
  auto [Matches, SetIsIncomplete] =
      InstrMaps.getVectorsThatCombinedScalars(SBBndl);
  if (Matches.size() == 1 && !SetIsIncomplete) {
    auto *ExistingVec = *Matches.begin();
    auto NeedsShuffleMaskOpt = InstrMaps.getShuffleMask(ExistingVec, SBBndl);
    if (TopDown)
      NeedsShuffleMaskOpt = NeedsShuffleMaskOpt->getInverse();
    // If all values in CombineScalars are in the same order as in SBBndl.
    assert(NeedsShuffleMaskOpt &&
           "If we have a single matching node, then we don't expect values "
           "in SBBndl that don't belong to that node");
    assert(NeedsShuffleMaskOpt->size() == SBUtils::getNumLanes(ExistingVec) &&
           "Expected same number of lanes. Bad mask!");
    return createAnalysisResult<PerfectReuseVector>(
        ExistingVec, *NeedsShuffleMaskOpt, SBBndl);
  }
  if (Matches.size() > 1 || (Matches.size() == 1 && SetIsIncomplete)) {
    // We are collecting values from (topdown: extracting values to) 2 or more
    // vectors, or from (topdown: to) one vector and external values, so we need
    // both Unpacking and Pakcing.
    return createAnalysisResult<UnpackAndPack>(SBBndl);
  }

  // If ValueBundle contains values other than instructions, we need to Pack.
  if (any_of(SBBndl,
             [](auto *SBV) { return !isa<SBInstruction>(SBV); })) {
    LLVM_DEBUG(
        dbgs() << "Not vectorizing: ValueBundle contains non-instructions\n"
               << SBBndl);
    assert(!TopDown && "Expected bottom-up only!");
    return createAnalysisResult<Pack>(NoVecReason::NonInstructions, SBBndl);
  }

  // Disable crossing BBs.
  if (any_of(SBBndl, [this](SBValue *SBV) {
        return cast<SBInstruction>(SBV)->getParent() != SBBB;
      })) {
    if (!TopDown)
      return createAnalysisResult<Pack>(NoVecReason::CrossBBs, SBBndl);
    return createAnalysisResult<UnpackTopDown>(NoVecReason::CrossBBs, SBBndl);
  }

  // Check opcodes and other instruction-specific data.
  if (auto NoVecReasonOpt = cantVectorizeBasedOnOpcodesAndTypes(SBBndl)) {
    LLVM_DEBUG(dbgs() << "Not vectorizing reason: "
                      << noVecReasonToStr(*NoVecReasonOpt) << ":\n"
                      << SBBndl);
    if (TopDown)
      return createAnalysisResult<UnpackTopDown>(*NoVecReasonOpt, SBBndl);
    return createAnalysisResult<Pack>(*NoVecReasonOpt, SBBndl);
  }

  // Check if it is legal to schedule `SBBndl`.
#ifndef NDEBUG
  SBBB->verifyIR();
#endif
  auto *Scheduler = SBBB->getContext().getScheduler(SBBB);
  bool LegalToSchedule = Scheduler->trySchedule(SBBndl, TopDown);
  if (!LegalToSchedule) {
#ifndef NDEBUG
    SBBB->verifyIR();
#endif
    LLVM_DEBUG(dbgs() << "Not vectorizing: Cannot schedule bundle\n"
                      << SBBndl);
    if (TopDown)
      return createAnalysisResult<UnpackTopDown>(NoVecReason::CantSchedule,
                                                 SBBndl);
    return createAnalysisResult<Pack>(NoVecReason::CantSchedule, SBBndl);
  }
#ifndef NDEBUG
  SBBB->verifyIR();
#endif
  assert(Matches.empty() && SetIsIncomplete && "Handled above");

  // Some opcodes need special treatment.
  auto *I0 = cast<SBInstruction>(SBBndl[0]);
  switch (I0->getOpcode()) {
  case SBInstruction::Opcode::Unpack: {
    if (TopDown) {
      // TODO: Unimplemented.
      return createAnalysisResult<UnpackTopDown>(NoVecReason::Unimplemented,
                                                 SBBndl);
    }
    // May need a shuffle.
    auto *VecOp = I0->getOperand(0);
    assert(all_of(SBBndl,
                  [VecOp](SBValue *SBV) {
                    return cast<SBUser>(SBV)->getOperand(0) == VecOp;
                  }) &&
           "We have already checked that they read from same operand 0");
    auto ShuffMaskOpt = getShuffleMaskForUnpacks(SBBndl);
    assert(ShuffMaskOpt && "Shouldn't fail!");

    auto CollectProbablyDead = [&SBBndl, &ProbablyDead](int Offset = 0) {
      for (auto *SBV : drop_begin(SBBndl, Offset))
        ProbablyDead.insert(cast<SBInstruction>(SBV));
    };
    // This could be: (i) a full Shuffle, (ii) a vector Unpack, or (iii) a Pack
    // of the extracted values.
    if ((int)ShuffMaskOpt->size() == SBUtils::getNumLanes(VecOp->getType())) {
      CollectProbablyDead();
      return createAnalysisResult<PerfectReuseVector>(VecOp, *ShuffMaskOpt,
                                                      SBBndl);
    }
    CollectProbablyDead(ShuffMaskOpt->size());
    if (ShuffMaskOpt->isInOrder())
      return createAnalysisResult<Unpack>(VecOp, (*ShuffMaskOpt)[0],
                                          ShuffMaskOpt->size(), SBBndl);
    return createAnalysisResult<Pack>(NoVecReason::UnpackAndShuffle, SBBndl);
  }
  case SBInstruction::Opcode::Select: {
    if (TopDown) {
      // TODO: Unimplemented.
      return createAnalysisResult<UnpackTopDown>(NoVecReason::Unimplemented,
                                                 SBBndl);
    }
    // To vectorize vector selects with scalar conditions we need to broadcast
    // the conditions:
    //
    // From:
    //  %s0 = select i1 %c0, <2 x i64> %lhs0, <2 x i64> %rhs0
    //  %s1 = select i1 %c1, <2 x i64> %lhs1, <2 x i64> %rhs1
    // To:
    //  %vec_c0 = insertelement <4 x i1> poison,  i1 %c0, 0
    //  %vec_c1 = insertelement <4 x i1> %vec_c0, i1 %c0, 1
    //  %vec_c2 = insertelement <4 x i1> %vec_c1, i1 %c1, 2
    //  %vec_c =  insertelement <4 x i1> %vec_c2, i1 %c1, 4
    //  %vec_s = select <4 x i1> %vec_c, <4 x i64> %vec_lhs, <4 x i64> %vec_rhs
    //
    // TODO: We can't handle this currently. Implement this to try it out.
    //
    // We are currently vectorizing selects with vector conditions like this:
    //  %s0 = select <2 x i1> %c0, <2 x i64> %lhs0, <2 x i64> %rhs0
    //  %s1 = select <2 x i1> %c1, <2 x i64> %lhs1, <2 x i64> %rhs1
    // Into:
    //  %XPack = extractelement <2 x i1> %c0, i64 0
    //  %Pack = insertelement <4 x i1> poison, i1 %XPack, i64 0
    //  %XPack1 = extractelement <2 x i1> %c0, i64 1
    //  %Pack2 = insertelement <4 x i1> %Pack, i1 %XPack1, i64 1
    //  %XPack3 = extractelement <2 x i1> %c1, i64 0
    //  %Pack4 = insertelement <4 x i1> %Pack2, i1 %XPack3, i64 2
    //  %XPack5 = extractelement <2 x i1> %c1, i64 1
    //  %vec_c = insertelement <4 x i1> %Pack4, i1 %XPack5, i64 3
    //  %vec_s = select <4 x i1> %vec_c, <2 x i64> %vec_lhs, <2 x i64> %vec_rhs
    //
    auto *Sel = cast<SBInstruction>(SBBndl[0]);
    if (Sel->getOperand(0)->lanes() != Sel->getOperand(1)->lanes())
      return createAnalysisResult<Pack>(NoVecReason::UnsupportedOpcode,
                                        SBBndl);
    break;
  }
  case SBInstruction::Opcode::Opaque: {
    if (TopDown)
      return createAnalysisResult<UnpackTopDown>(NoVecReason::UnsupportedOpcode,
                                                 SBBndl);
    return createAnalysisResult<Pack>(NoVecReason::UnsupportedOpcode, SBBndl);
  }
  case SBInstruction::Opcode::FCmp:
  case SBInstruction::Opcode::ICmp: {
    if (TopDown) {
      // TODO: Unimplemented.
      return createAnalysisResult<UnpackTopDown>(NoVecReason::Unimplemented,
                                                 SBBndl);
    }
    // The operands of Cmps also need to be of the same type.
    Type *Op0Ty = I0->getOperand(0)->getType();
    if (any_of(drop_begin(SBBndl), [Op0Ty](SBValue *SBV) {
          return cast<SBInstruction>(SBV)->getOperand(0)->getType() !=
                 Op0Ty;
        }))
      return createAnalysisResult<Pack>(NoVecReason::DiffTypes, SBBndl);
    break;
  }
  default:
    break;
  }

  // Siple vectorization of values.
  return createAnalysisResult<SimpleWiden>(SBBndl);
}

void SBAnalysis::clear() {
  ResultPool.clear();
  ValueToAnalysisMap.clear();
#ifndef NDEBUG
  BndlAnalysisVec.clear();
#endif
}

#ifndef NDEBUG
void SBAnalysis::dump(raw_ostream &OS) const {
  for (const auto &Pair : reverse(BndlAnalysisVec))
    OS << *Pair.second << "\n" << Pair.first << "\n";
}
void SBAnalysis::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
void SBAnalysis::dumpInstrs(raw_ostream &OS) const {
  SBInstruction *TopI = nullptr;
  SBInstruction *BotI = nullptr;
  for (auto &Pair : BndlAnalysisVec)
    for (auto *SBV : Pair.first)
      if (auto *SBI = cast<SBInstruction>(SBV)) {
        if (TopI == nullptr) {
          TopI = SBI;
        } else {
          TopI = SBI->comesBefore(TopI) ? SBI : TopI;
        }
        if (BotI == nullptr) {
          BotI = SBI;
        } else {
          BotI = BotI->comesBefore(SBI) ? SBI : BotI;
        }
      }
  if (TopI == nullptr || BotI == nullptr) {
    OS << "NULL\n";
    return;
  }
  for (auto *I = TopI, *E = BotI->getNextNode(); I != E; I = I->getNextNode())
    OS << *I << "\n";
}
void SBAnalysis::dumpInstrs() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif
