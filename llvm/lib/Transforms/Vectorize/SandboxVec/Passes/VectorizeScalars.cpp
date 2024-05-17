//===- VectorizeScalars.cpp - SB Pass that combines scalars into vectors --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Passes/VectorizeScalars.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SeedCollector.h"

#define DEBUG_TYPE "SBVec"

using namespace llvm;

static cl::opt<unsigned>
    OverrideVecRegBits("sbvec-vec-reg-bits", cl::init(0), cl::Hidden,
                       cl::desc("Override the vector register size in bits, "
                                "which is otherwise found by querying TTI."));

static cl::opt<bool>
    AllowNonPow2("sbvec-allow-non-pow2", cl::init(false), cl::Hidden,
                 cl::desc("Allow non-power-of-2 vectorization."));

static cl::opt<unsigned long>
    StopAt("sbvec-stop-at",
           cl::init(std::numeric_limits<unsigned long>::max()), cl::Hidden,
           cl::desc("Vectorize if the invocation count is < than this. 0 "
                    "disables vectorization."));

static cl::opt<unsigned> MaxRecursionLimit(
    "sbvec-max-rec-limit", cl::init(std::numeric_limits<unsigned>::max()),
    cl::Hidden,
    cl::desc("While recursively vectorizing bottom-up, stop after this many "
             "calls to the recursive function. Used for debugging. NOTE: this "
             "is not the max recursion depth!"));

static constexpr const unsigned long MaxRecLimAtAlwaysEnabled =
    std::numeric_limits<unsigned long>::max();
static cl::opt<unsigned long> MaxRecursionLimitAt(
    "sbvec-max-rec-limit-at", cl::init(MaxRecLimAtAlwaysEnabled), cl::Hidden,
    cl::desc("Apply the max recursion limit only at this invocation."));

static cl::opt<unsigned>
    SliceOffsetLimit("sbvec-slice-offset-limit", cl::init(4u), cl::Hidden,
                     cl::desc("Limits the slice offsets that we are attempting "
                              "to vectorize, to save compilation time."));

static cl::opt<unsigned> TopDownUseLimit(
    "sbvec-top-down-use-limit", cl::init(2u), cl::Hidden,
    cl::desc(
        "Limits the number of users considered in top-down vectorization."));

/// \Returns {SBValBundle, ShouldKeepLane0} pairs for all operands.
static SmallVector<SBValBundle>
getOperandBundlesSafe(const SBValBundle &Bndl) {
  SmallVector<SBValBundle> OpBndls;
  switch (cast<SBInstruction>(Bndl[0])->getOpcode()) {
  case SBInstruction::Opcode::Shuffle:
  case SBInstruction::Opcode::Pack:
  case SBInstruction::Opcode::Unpack:
    llvm_unreachable("Unimplemented");
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
  case SBInstruction::Opcode::BitCast:
    return {Bndl.getOperandBundle(0)};
  case SBInstruction::Opcode::FCmp:
  case SBInstruction::Opcode::ICmp:
    return {Bndl.getOperandBundle(0), Bndl.getOperandBundle(1)};
  case SBInstruction::Opcode::Select:
    return {Bndl.getOperandBundle(0), Bndl.getOperandBundle(1),
            Bndl.getOperandBundle(2)};
  case SBInstruction::Opcode::FNeg:
    return {Bndl.getOperandBundle(0)};
  case SBInstruction::Opcode::Add:
  case SBInstruction::Opcode::FAdd:
  case SBInstruction::Opcode::Sub:
  case SBInstruction::Opcode::FSub:
  case SBInstruction::Opcode::Mul:
  case SBInstruction::Opcode::FMul:
  case SBInstruction::Opcode::UDiv:
  case SBInstruction::Opcode::SDiv:
  case SBInstruction::Opcode::FDiv:
  case SBInstruction::Opcode::URem:
  case SBInstruction::Opcode::SRem:
  case SBInstruction::Opcode::FRem:
  case SBInstruction::Opcode::Shl:
  case SBInstruction::Opcode::LShr:
  case SBInstruction::Opcode::AShr:
  case SBInstruction::Opcode::And:
  case SBInstruction::Opcode::Or:
  case SBInstruction::Opcode::Xor:
    return {Bndl.getOperandBundle(0), Bndl.getOperandBundle(1)};
  case SBInstruction::Opcode::Load:
    return {Bndl.getOperandBundle(0)};
  case SBInstruction::Opcode::Store:
    return {Bndl.getOperandBundle(0), Bndl.getOperandBundle(1)};
  case SBInstruction::Opcode::Opaque: {
    llvm_unreachable("Unimplemented");
  }
  }
}
namespace llvm {

std::pair<SmallVector<SmallVector<SBUse, 2>, 2>, bool>
getUsesBundlesPicky(const SBValBundle &Bndl) {
  // For now we accept bundles where each element has the same number of users.
  if (Bndl[0]->hasNUsesOrMore(TopDownUseLimit + 1))
    return {{}, /*Fail=*/true};
  unsigned NumUses = Bndl[0]->getNumUses();
  if (any_of(drop_begin(Bndl),
             [NumUses](auto *SBV) { return !SBV->hasNUses(NumUses); }))
    return {{}, /*Fail=*/true};

  // Helper data structure for sorting uses based on OperandNo and opcode.
  SmallVector<SmallVector<SBUse>> UsesPerLane;
  unsigned Lanes = Bndl.size();
  UsesPerLane.resize(Lanes);
  for (auto [Idx, SBV] : enumerate(Bndl)) {
    auto &Uses = UsesPerLane[Idx];
    for (SBUse Use : SBV->uses())
      Uses.push_back(Use);
    // Sort uses to try to match opcode and operand index, this should
    // increase the chances of vectorizing the users.
    stable_sort(Uses, [](const SBUse &Use1, const SBUse &Use2) {
      unsigned OpNo1 = Use1.getOperandNo();
      unsigned OpNo2 = Use2.getOperandNo();
      if (OpNo1 != OpNo2)
        return OpNo1 < OpNo2;
      auto Opc1 = cast<SBInstruction>(Use1.getUser())->getOpcode();
      auto Opc2 = cast<SBInstruction>(Use2.getUser())->getOpcode();
      if (Opc1 != Opc2)
        return Opc1 < Opc2;
      return false;
    });
  }

  // Now do the final filtering and return the the uses.
  SmallVector<SmallVector<SBUse, 2>, 2> FinalUses;
  FinalUses.reserve(NumUses);
  for (auto UseCnt : seq<unsigned>(NumUses)) {
    SmallVector<SBUse, 2> FinalUsePerLane;
    auto &Lane0Uses = UsesPerLane[/*Lane=*/0];
    const SBUse &Lane0Use = Lane0Uses[UseCnt];
    FinalUsePerLane.push_back(Lane0Use);
    unsigned Lane0OpNo = Lane0Use.getOperandNo();
    for (auto &LaneUses : drop_begin(UsesPerLane)) {
      const SBUse &LaneUse = LaneUses[UseCnt];
      unsigned LaneOpNo = LaneUse.getOperandNo();
      // If we can't match the operand number, return FAIL.
      if (LaneOpNo != Lane0OpNo)
        return {{}, /*Fail=*/true};
      FinalUsePerLane.push_back(LaneUse);
    }
    FinalUses.push_back(std::move(FinalUsePerLane));
  }
  return {FinalUses, /*Fail=*/false};
}
} // namespace llvm

VectorizeFromSeeds::VectorizeFromSeeds(SBBasicBlock *SBBB,
                                       SBContext &Ctxt, ScalarEvolution &SE,
                                       const DataLayout &DL,
                                       TargetTransformInfo &TTI)
    : SBBB(SBBB), Ctxt(Ctxt), SE(SE), DL(DL), TTI(TTI), Analysis(SE, DL) {
  if (SBBB != nullptr)
    Analysis.init(*SBBB);
}

#ifndef NDEBUG
void VectorizeFromSeeds::dump(raw_ostream &OS) const { Analysis.dump(); }
void VectorizeFromSeeds::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
void VectorizeFromSeeds::dumpInstrs(raw_ostream &OS) const {
  Analysis.dumpInstrs(OS);
}
void VectorizeFromSeeds::dumpInstrs() const {
  dumpInstrs(dbgs());
  dbgs() << "\n";
}
#endif

SBValue *
VectorizeFromSeeds::createVecInstruction(const SBValBundle &Bndl,
                                         const SBValBundle &Operands,
                                         SBContext &Ctxt) {
  assert(
      all_of(Bndl, [](auto *SBV) { return isa<SBInstruction>(SBV); }) &&
      "Expect SBInstructions");

  // This lambda creates the instruction but won't add DebugInfo.
  auto CreateVecInstructionImpl = [&Ctxt](const SBValBundle &Bndl,
                                          const SBValBundle &Operands) {
    Type *ScalarTy = SBUtils::getElementType(Bndl[0]->getExpectedType());
    auto *VecTy = SBUtils::getWideType(ScalarTy, SBUtils::getNumLanes(Bndl));
    // TODO: Use a SBBuilder to avoid calling different ::create() functions.
    auto [WhereBB, WhereIt] = SBUtils::getInsertPointAfterInstrs(Bndl);

    auto Opcode = cast<SBInstruction>(Bndl[0])->getOpcode();
    switch (Opcode) {
    case SBInstruction::Opcode::Shuffle:
    case SBInstruction::Opcode::Pack:
    case SBInstruction::Opcode::Unpack:
      llvm_unreachable("Unimplemented");
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
      assert(Operands.size() == 1u && "Casts are unary!");

      SBValue *SBV;
      if (WhereIt != WhereBB->end())
        SBV = SBCastInstruction::create(VecTy, Opcode, Operands[0],
                                            &*WhereIt, Ctxt, "VCast");
      else
        SBV = SBCastInstruction::create(VecTy, Opcode, Operands[0], WhereBB,
                                            Ctxt, "VCast");

      return SBV;
    }
    case SBInstruction::Opcode::FCmp:
    case SBInstruction::Opcode::ICmp: {
      auto Pred = cast<SBCmpInstruction>(Bndl[0])->getPredicate();
      assert(all_of(drop_begin(Bndl),
                    [Pred](auto *SBV) {
                      return cast<SBCmpInstruction>(SBV)->getPredicate() ==
                             Pred;
                    }) &&
             "Expected same predicate across bundle.");
      if (WhereIt != WhereBB->end())
        return SBCmpInstruction::create(Pred, Operands[0], Operands[1],
                                          &*WhereIt, Ctxt, "VCmp");
      return SBCmpInstruction::create(Pred, Operands[0], Operands[1], WhereBB,
                                        Ctxt, "Vcmp");
    }
    case SBInstruction::Opcode::Select: {
      SBValue *SBV;
      if (WhereIt != WhereBB->end())
        SBV = SBSelectInstruction::create(
            Operands[0], Operands[1], Operands[2], &*WhereIt, Ctxt, "Vec");
      else
        SBV = SBSelectInstruction::create(
            Operands[0], Operands[1], Operands[2], WhereBB, Ctxt, "Vec");
      return SBV;
    }
    case SBInstruction::Opcode::FNeg: {
      auto *UOp0 = cast<SBUnaryOperator>(Bndl[0]);
      auto OpC = UOp0->getOpcode();
      SBValue *SBV;
      if (WhereIt != WhereBB->end())
        SBV = SBUnaryOperator::createWithCopiedFlags(
            OpC, Operands[0], UOp0, &*WhereIt, Ctxt, "Vec");
      else
        SBV = SBUnaryOperator::createWithCopiedFlags(OpC, Operands[0], UOp0,
                                                         WhereBB, Ctxt, "Vec");

      if (isa<SBUnaryOperator>(SBV)) {
        if (UOp0->isFPMath()) {
          assert(!SBUtils::differentMathFlags(Bndl) &&
                 "For now should have packed!");
          // TODO: Use least common math flags if are not the same across Bndl.
        }
      }
      return SBV;
    }
    case SBInstruction::Opcode::Add:
    case SBInstruction::Opcode::FAdd:
    case SBInstruction::Opcode::Sub:
    case SBInstruction::Opcode::FSub:
    case SBInstruction::Opcode::Mul:
    case SBInstruction::Opcode::FMul:
    case SBInstruction::Opcode::UDiv:
    case SBInstruction::Opcode::SDiv:
    case SBInstruction::Opcode::FDiv:
    case SBInstruction::Opcode::URem:
    case SBInstruction::Opcode::SRem:
    case SBInstruction::Opcode::FRem:
    case SBInstruction::Opcode::Shl:
    case SBInstruction::Opcode::LShr:
    case SBInstruction::Opcode::AShr:
    case SBInstruction::Opcode::And:
    case SBInstruction::Opcode::Or:
    case SBInstruction::Opcode::Xor: {
      auto *BinOp0 = cast<SBBinaryOperator>(Bndl[0]);
      auto *LHS = Operands[0];
      auto *RHS = Operands[1];
      SBValue *SBV;
      if (WhereIt != WhereBB->end())
        SBV = SBBinaryOperator::createWithCopiedFlags(
            BinOp0->getOpcode(), LHS, RHS, BinOp0, &*WhereIt, Ctxt, "Vec");
      else
        SBV = SBBinaryOperator::createWithCopiedFlags(
            BinOp0->getOpcode(), LHS, RHS, WhereBB, &*WhereIt, Ctxt, "Vec");
      return SBV;
    }
    case SBInstruction::Opcode::Load: {
      SBValue *SBV;
      auto *Ld0 = cast<SBLoadInstruction>(Bndl[0]);
      SBValue *Ptr = Ld0->getPointerOperand();
      if (WhereIt != WhereBB->end())
        SBV = SBLoadInstruction::create(VecTy, Ptr, Ld0->getAlign(),
                                            &*WhereIt, Ctxt, "VecL");
      else
        SBV = SBLoadInstruction::create(VecTy, Ptr, Ld0->getAlign(),
                                            WhereBB, Ctxt, "VecL");
      return SBV;
    }
    case SBInstruction::Opcode::Store: {
      SBValue *SBV;
      auto Align = cast<SBStoreInstruction>(Bndl[0])->getAlign();
      SBValue *Val = Operands[0];
      SBValue *Ptr = Operands[1];
      if (WhereIt != WhereBB->end())
        SBV = SBStoreInstruction::create(Val, Ptr, Align, &*WhereIt, Ctxt);
      else
        SBV = SBStoreInstruction::create(Val, Ptr, Align, WhereBB, Ctxt);
      return SBV;
    }
    case SBInstruction::Opcode::Opaque:
      llvm_unreachable("Unimplemented");
      break;
    }
  };

  auto *NewV = CreateVecInstructionImpl(Bndl, Operands);
  if (auto *NewI = dyn_cast<SBInstruction>(NewV))
    SBUtils::propagateMetadata(NewI, Bndl);
  return NewV;
}

#ifndef NDEBUG
void VectorizeFromSeeds::verifyDAGAndSchedule(const SBValBundle &Bndl,
                                              SBValue *NewSBV) const {
  // Make sure that the DAG and Scheduler have been notified about the
  // creation of the new IR and have created nodes for them.
  auto It = find_if(
      Bndl, [](SBValue *SBV) { return isa<SBInstruction>(SBV); });
  if (It != Bndl.end()) {
    auto *SBBB = cast<SBInstruction>(*It)->getParent();
    if (auto *Sched = Ctxt.getScheduler(SBBB)) {
      auto &DAG = Sched->getDAG();
      if (auto *NewSBI = dyn_cast<SBInstruction>(NewSBV)) {
        assert(DAG.getNode(NewSBI) != nullptr && "Expected DAG node!");
        assert(Sched->getBundle(NewSBI) != nullptr && "Expected Bundle!");
      }
    }
  }
}
#endif

bool VectorizeFromSeeds::recOverDebugLimitAtInvocation() const {
  bool MaxRecLimitEnabled = MaxRecursionLimitAt == MaxRecLimAtAlwaysEnabled ||
                            MaxRecursionLimitAt == InvocationCnt - 1;
  if (LLVM_UNLIKELY(MaxRecLimitEnabled && RecCnt > MaxRecursionLimit))
    return true;
  return false;
}

bool VectorizeFromSeeds::maybeStopRecursionForDebugging(
    const SBValBundle &Bndl, AnalysisResult *&AnalysisRes) {
  ++RecCnt;
  if (LLVM_UNLIKELY(recOverDebugLimitAtInvocation())) {
    // Stores (at root) can't be packed, so we need a special-case for them.
    if (RecCnt == 1)
      return true;
    if (!isa<KeepLane0>(AnalysisRes))
      AnalysisRes = Analysis.createAnalysisResult<Pack>(
          NoVecReason::RecursionLimit, Bndl);
  }
  return false;
}

SBValue *VectorizeFromSeeds::vectorizeRec(
    SBValBundle &Bndl, AnalysisResult *AnalysisRes,
    SmallPtrSet<SBInstruction *, 4> &EraseCandidates, bool TopDown,
    SBValue *VecOp, unsigned OperandNo) {
  LLVM_DEBUG(dbgs() << "Bndl:" << Bndl << " Analysis: " << *AnalysisRes
                    << "\n";);
  // Early return or modify `AnalysisRes` for debugging
  if (maybeStopRecursionForDebugging(Bndl, AnalysisRes))
    return nullptr;

  switch (AnalysisRes->getSubclassID()) {
  case ResultID::PerfectReuseVector: {
    auto *Res = cast<PerfectReuseVector>(AnalysisRes);
    SBValue *CommonVal = Res->getCommonVal();
    SBValue *New;
    if (Res->needsShuffle()) {
      ShuffleMask ShuffMask = Res->getShuffleMask();
      if (TopDown) {
        New = Ctxt.createSBShuffleInstruction(ShuffMask, VecOp, SBBB);
        auto *ShuffleI = cast<SBInstruction>(New);
        auto *VecUserI = cast<SBInstruction>(CommonVal);
        VecUserI->setOperand(OperandNo, ShuffleI);
      } else {
        New = Ctxt.createSBShuffleInstruction(ShuffMask, CommonVal, SBBB);
      }
    } else {
      New = CommonVal;
    }
    return New;
  }
  case ResultID::UnpackAndPack: {
    // We found multiple vector instrs (result of combining scalars) that
    // match the values in bundle. So we need to create several Unpacks and a
    // Pack.
    // For example if `Bndl` contains elements {A,B} but both have already been
    // vectorized as part of vectors %Vec0 {X,A} and %Vec1 {Y,Z,B,W} then we
    // need two Unpacks, one for each vector and one pack:
    //   %unpkA = Unpack %Vec0,  lane 1
    //   %unpkB = Unpack %Vec1,  lane 2
    //   %pakAB = Pack   %unpkA, %unpkB
    SBValBundle PackVals(Bndl.size());
    for (auto [Lane, SBV] : enumerate(Bndl)) {
      if (auto *Vec = InstrMaps.getVectorForScalar(SBV)) {
        // // TODO: Reuse existing Unpack if it already exists.
        // SBValue *Unpack = Ctxt.getSBUnpackInstruction(Vec, Lane);
        auto OrigLane = InstrMaps.getScalarLane(Vec, SBV);
        auto *Unpack = Ctxt.createSBUnpackInstruction(
            Vec, OrigLane, SBBB, SBUtils::getNumLanes(SBV));
        PackVals.push_back(Unpack);
      } else {
        PackVals.push_back(SBV);
      }
    }
    auto *Pack = Ctxt.createSBPackInstruction(PackVals, SBBB);
    return Pack;
  }
  case ResultID::Unpack: {
    auto *Res = cast<Unpack>(AnalysisRes);
    auto *Unpack = Ctxt.createSBUnpackInstruction(
        Res->getVecOp(), Res->getFirstLaneToUnpack(), SBBB,
        Res->getNumLanesToUnpack());
    return Unpack;
  }
  case ResultID::UnpackTopDown: {
    assert(TopDown && "This is used only in TopDown vectorization!");
    emitUnpacks(VecOp, InstrMaps.getScalarsFor(VecOp));
    return nullptr;
  }
  case ResultID::Pack: {
    auto *New = Ctxt.createSBPackInstruction(Bndl, SBBB);
    return New;
  }
  case ResultID::SimpleWiden: {
    SBValue *NewVec = nullptr;
    SBValue *Bndl0 = Bndl[0];
    bool IsLoadOrStore =
        isa<SBLoadInstruction>(Bndl0) || isa<SBStoreInstruction>(Bndl0);
    if (!TopDown) {
      SmallVector<SBValBundle> OperandBndls = getOperandBundlesSafe(Bndl);
      SBValBundle Operands(2);
      for (auto [OpIdx, OpBndl] : enumerate(OperandBndls)) {
        bool IsAddrOperand =
            IsLoadOrStore &&
            OpIdx == SBUtils::getAddrOperandIdx(cast<SBInstruction>(Bndl0));
        auto *AnalysisRes =
            IsAddrOperand ? Analysis.createAnalysisResult<KeepLane0>(OpBndl)
                          : Analysis.getBndlAnalysis(OpBndl, InstrMaps,
                                                     EraseCandidates, TopDown);
        auto *New = vectorizeRec(OpBndl, AnalysisRes, EraseCandidates, TopDown,
                                 /*VecOp=*/nullptr);
        Operands.push_back(New);
      }
      // Post-order
      NewVec = createVecInstruction(Bndl, Operands, Ctxt);
      InstrMaps.registerVector(NewVec, Bndl);
      // There may be scalar users of values in Bndl so emit unpacks from
      // NewSBV to them.
      emitUnpacks(NewVec, Bndl);
#ifndef NDEBUG
      verifyDAGAndSchedule(Bndl, NewVec);
#endif
    } else {
      SBValBundle Operands(2);
      for (auto [OpIdx, OpBndl] : enumerate(Bndl.getOperandBundles())) {
        bool IsAddrOperand =
            IsLoadOrStore &&
            OpIdx == SBUtils::getAddrOperandIdx(cast<SBInstruction>(Bndl0));
        if (IsAddrOperand) {
          Operands.push_back(OpBndl[0]);
          continue;
        }
        auto [Set, Incomplete] =
            InstrMaps.getVectorsThatCombinedScalars(OpBndl);
        if (!Incomplete && Set.size() == 1) {
          Operands.push_back(*Set.begin());
        } else {
          // Generate packs to get the vector value from the operand.
          auto *Pack = Ctxt.createSBPackInstruction(
              OpBndl, SBBB, /*BeforeI=*/SBUtils::getHighest(Bndl));
          Operands.push_back(Pack);
        }
      }
      // Pre-order
      NewVec = createVecInstruction(Bndl, Operands, Ctxt);
      InstrMaps.registerVector(NewVec, Bndl);
#ifndef NDEBUG
      verifyDAGAndSchedule(Bndl, NewVec);
#endif

      auto [UsesBndls, Fail] = getUsesBundlesPicky(Bndl);
      if (Fail) {
        emitUnpacks(NewVec, Bndl);
      } else {
        for (auto &Uses : UsesBndls) {
          SBValBundle Users(Uses.size());
          for (SBUse &Use : Uses)
            Users.push_back(Use.getUser());
          unsigned OperandNo = Uses[0].getOperandNo();
          assert(all_of(drop_begin(Uses),
                        [OperandNo](const SBUse &Use) {
                          return Use.getOperandNo() == OperandNo;
                        }) &&
                 "Expected all edges in same direction!");

          auto *AnalysisRes = Analysis.getBndlAnalysis(
              Users, InstrMaps, EraseCandidates, TopDown);

          vectorizeRec(Users, AnalysisRes, EraseCandidates, TopDown,
                       /*VecOp=*/NewVec, OperandNo);
        }
      }
    }

    for (auto *SBV : Bndl)
      EraseCandidates.insert(cast<SBInstruction>(SBV));
    return NewVec;
  }
  case ResultID::KeepLane0: {
    // Since we are still building the graph, we can't know for sure if a
    // value is unused, so add it to the EraseCandidates list.
    for (SBValue *SBV : drop_begin(Bndl)) {
      if (!isa<SBInstruction>(SBV))
        continue; // Skip constants like inline-geps, args etc.
      EraseCandidates.insert(cast<SBInstruction>(SBV));
    }
    return Bndl[0];
  }
  }
  llvm_unreachable("Missing switch cases!");
}

void VectorizeFromSeeds::tryEraseDeadInstrs(
    SmallPtrSet<SBInstruction *, 4> &EraseCandidates) {
  bool Change = true;
  while (Change) {
    Change = false;
    for (auto *SBI : make_early_inc_range(EraseCandidates))
      if (SBI->getNumUsers() == 0) {
        SBI->eraseFromParent();
        Change = true;
        EraseCandidates.erase(SBI);
      }
  }
}

SBInstruction *VectorizeFromSeeds::tryVectorize(
    SBValBundle &Seeds, SBRegion &Rgn,
    SmallPtrSet<SBInstruction *, 4> &EraseCandidates, bool TopDown) {
  InstrMaps.setRegion(Rgn);
  SBBB = cast<SBInstruction>(Seeds[0])->getParent();
  // Used for debugging.
  if (LLVM_UNLIKELY(InvocationCnt++ >= StopAt))
    return nullptr;
  // Make sure the Scheduler and DAG state are reset.
  SBBB->getContext().getScheduler(SBBB)->startFresh(SBBB);
  auto *AnalysisRes =
      Analysis.getBndlAnalysis(Seeds, InstrMaps, EraseCandidates, TopDown);
  // The analysis runs the scheduler which will most likely modify the IR.
  Changed = true;
  // Don't bother vectorizing futher if seeds cannot form a vector.
  if (!isa<SimpleWiden>(AnalysisRes)) {
    Analysis.clear();
    return nullptr;
  }
  RecCnt = 0;
  SBValue *Root = vectorizeRec(Seeds, AnalysisRes, EraseCandidates, TopDown,
                                 /*VecOp=*/nullptr);
  Analysis.clear();
#ifndef NDEBUG
  SBBB->verifyIR();
#endif
  return cast_or_null<SBInstruction>(Root);
}

void VectorizeFromSeeds::emitUnpacks(SBValue *Vec,
                                     const SBValBundle &Scalars) {
  assert(SBUtils::isVector(Vec) && "Expect vector");
  SmallVector<std::pair<unsigned, SBValue *>> UnpackData;
  // If SBI has non-bundled users, then we need to create an Unpack.
  unsigned Lane = 0;
  for (SBValue *OrigElem : Scalars) {
    assert(OrigElem != nullptr && "Expected non-null");
    if (any_of(OrigElem->users(), [this](SBUser *SBU) {
          // Skip if not an instruction.
          if (!isa<SBInstruction>(SBU))
            return false;
          // Skip users that will be widened.
          auto *Res = Analysis.getAnalysisResult(SBU);
          if (Res != nullptr && isa<SimpleWiden>(Res))
            return false;
          return InstrMaps.getVectorForScalar(cast<SBInstruction>(SBU)) ==
                 nullptr;
        }))
      // Create the Unpack after loop to avoid adding values while
      // iterating
      UnpackData.push_back({Lane, OrigElem});
    Lane += OrigElem->lanes();
  }

  // Now create the unpacks.
  for (auto [Lane, OrigElem] : UnpackData) {
    auto *Unpack =
        Ctxt.createSBUnpackInstruction(Vec, Lane, SBBB, OrigElem->lanes());

    // There is a rare case when we may need a bitcast:
    // If we are vectorizing values <1 x TYPE> and we have an external user of
    // <1 x TYPE> then our unpack will be of type TYPE, not <1 x TYPE>
    Type *OrigTy = OrigElem->getType();
    bool RequiresBitcast = Unpack->getType() != OrigTy;
    if (RequiresBitcast) {
      assert(cast<FixedVectorType>(OrigTy)->getNumElements() == 1 &&
             "Expected something like <1 x TYPE>");
      SBInstruction *BeforeI =
          isa<SBInstruction>(Unpack)
              ? cast<SBInstruction>(Unpack)->getNextNode()
              : &*SBBB->getFirstNonPHIIt();
      Unpack = SBCastInstruction::create(
          OrigTy, SBInstruction::Opcode::BitCast, Unpack, BeforeI, Ctxt);
    }
    // Now connect UnpackN to those users that were not combined into a vector.
    OrigElem->replaceUsesWithIf(Unpack, [this](auto *DstU, unsigned OpIdx) {
      return DstU == nullptr || InstrMaps.getVectorForScalar(
                                    cast<SBInstruction>(DstU)) == nullptr;
    });
  }
}

bool VectorizeScalars::runOnSBBasicBlock(SBBasicBlock &BB) {
  assert(BB.getContext().getTracker().empty() &&
         "Expected empty tracker before running the pass!");
  Changed = false;
  Analysis.init(BB);

  unsigned VecRegBits =
      OverrideVecRegBits != 0
          ? OverrideVecRegBits
          : TTI.getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector)
                .getFixedValue();
  assert(InstrMaps.empty() &&
         "Scalar-Vector maps should be cleared after each vec attempt!");
  SeedCollector SC(&BB, DL, SE);
  auto RunOnSeed = [this, VecRegBits, &BB](SeedBundle &Seed,
                                           bool TopDown = false) {
    assert(!Seed.allUsed() &&
           "Should have been skipped by iterator::operator++ !");
    unsigned TyBits = DL.getTypeSizeInBits(SBUtils::getElementType(
        SBUtils::getExpectedType(Seed[Seed.getFirstUnusedElementIdx()])));
    // Try to create the largest vector supported by the target. If it fails
    // reduce the vector size by half.
    for (unsigned SliceBits = std::min(VecRegBits, Seed.getNumUnusedBits(DL)),
                  E = 2 * TyBits;
         SliceBits >= E;
         SliceBits = SBUtils::getFloorPowerOf2(SliceBits / 2)) {
      if (Seed.allUsed())
        break;

      unsigned OffsetLimit = SliceOffsetLimit;
      // Keep trying offsets after FirstUnusedElementIdx, until we vectorize the
      // slice. This could be quite expensive, so we enforce a limit.
      for (unsigned Offset = Seed.getFirstUnusedElementIdx(), OE = Seed.size();
           Offset + 1 < OE; Offset += 1) {
        // Seeds are getting used as we vectorize, so skip them.
        if (Seed.isUsed(Offset))
          continue;
        if (OffsetLimit == 0)
          break;
        if (Seed.allUsed())
          break;
        SBValBundle SeedSlice =
            Seed.getSlice(Offset, SliceBits, !AllowNonPow2, DL);
        if (SeedSlice.empty())
          continue;

        assert(SeedSlice.size() >= 2 && "Should have been rejected!");
        SmallPtrSet<SBInstruction *, 4> EraseCandidates;
        assert(BB.getContext().getTracker().empty() &&
               "Expected empty tracker!");
        SBRegion Rgn(BB, Ctxt, TTI);
        // Do the actual vectorization.
        auto *Root = tryVectorize(SeedSlice, Rgn, EraseCandidates, TopDown);
        if (Root != nullptr) {
          // TODO: We may need heuristics like depth or a cost threshold.
          // Mark all lanes as "used" so that we skip them in the future.
          Seed.setUsed(Offset, SeedSlice.size());
        }
        // Maintain the data structures.
        for (auto *I : EraseCandidates)
          InstrMaps.eraseScalar(I);
        tryEraseDeadInstrs(EraseCandidates);

        // Do our region-pass-manager duties and run all region passes on `Rgn`.
        runAllPassesOnRgn(Rgn);
      }
    }
  };
  for (SeedBundle &Seed : SC.getStoreSeeds())
    RunOnSeed(Seed);
  for (SeedBundle &Seed : SC.getLoadSeeds())
    RunOnSeed(Seed, /*TopDown=*/true);
  for (SeedBundle &Seed : SC.getNonMemSeeds())
    RunOnSeed(Seed);
  InstrMaps.clear();
  return Changed;
}

unsigned long VectorizeFromSeeds::InvocationCnt;
