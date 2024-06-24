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
#include "llvm/Transforms/Vectorize/SandboxVec/VecUtils.h"

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
    StopAt("sbvec-stop-at", cl::init(std::numeric_limits<unsigned long>::max()),
           cl::Hidden,
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

/// \Returns {DmpVector<SBValue *>, ShouldKeepLane0} pairs for all operands.
template <typename RangeT>
static SmallVector<DmpVector<sandboxir::Value *>>
getOperandBundlesSafe(const RangeT Bndl) {
  SmallVector<DmpVector<sandboxir::Value *>> OpBndls;
  switch (cast<sandboxir::Instruction>(Bndl[0])->getOpcode()) {
  case sandboxir::Instruction::Opcode::Shuffle:
  case sandboxir::Instruction::Opcode::Pack:
  case sandboxir::Instruction::Opcode::Unpack:
    llvm_unreachable("Unimplemented");
  case sandboxir::Instruction::Opcode::ZExt:
  case sandboxir::Instruction::Opcode::SExt:
  case sandboxir::Instruction::Opcode::FPToUI:
  case sandboxir::Instruction::Opcode::FPToSI:
  case sandboxir::Instruction::Opcode::FPExt:
  case sandboxir::Instruction::Opcode::PtrToInt:
  case sandboxir::Instruction::Opcode::IntToPtr:
  case sandboxir::Instruction::Opcode::SIToFP:
  case sandboxir::Instruction::Opcode::UIToFP:
  case sandboxir::Instruction::Opcode::Trunc:
  case sandboxir::Instruction::Opcode::FPTrunc:
  case sandboxir::Instruction::Opcode::BitCast:
    return {getOperandBundle(Bndl, 0)};
  case sandboxir::Instruction::Opcode::FCmp:
  case sandboxir::Instruction::Opcode::ICmp:
    return {getOperandBundle(Bndl, 0), getOperandBundle(Bndl, 1)};
  case sandboxir::Instruction::Opcode::Select:
    return {getOperandBundle(Bndl, 0), getOperandBundle(Bndl, 1),
            getOperandBundle(Bndl, 2)};
  case sandboxir::Instruction::Opcode::FNeg:
    return {getOperandBundle(Bndl, 0)};
  case sandboxir::Instruction::Opcode::Add:
  case sandboxir::Instruction::Opcode::FAdd:
  case sandboxir::Instruction::Opcode::Sub:
  case sandboxir::Instruction::Opcode::FSub:
  case sandboxir::Instruction::Opcode::Mul:
  case sandboxir::Instruction::Opcode::FMul:
  case sandboxir::Instruction::Opcode::UDiv:
  case sandboxir::Instruction::Opcode::SDiv:
  case sandboxir::Instruction::Opcode::FDiv:
  case sandboxir::Instruction::Opcode::URem:
  case sandboxir::Instruction::Opcode::SRem:
  case sandboxir::Instruction::Opcode::FRem:
  case sandboxir::Instruction::Opcode::Shl:
  case sandboxir::Instruction::Opcode::LShr:
  case sandboxir::Instruction::Opcode::AShr:
  case sandboxir::Instruction::Opcode::And:
  case sandboxir::Instruction::Opcode::Or:
  case sandboxir::Instruction::Opcode::Xor:
    return {getOperandBundle(Bndl, 0), getOperandBundle(Bndl, 1)};
  case sandboxir::Instruction::Opcode::Load:
    return {getOperandBundle(Bndl, 0)};
  case sandboxir::Instruction::Opcode::Store:
    return {getOperandBundle(Bndl, 0), getOperandBundle(Bndl, 1)};
  case sandboxir::Instruction::Opcode::PHI:
  case sandboxir::Instruction::Opcode::Opaque:
  case sandboxir::Instruction::Opcode::Ret:
  case sandboxir::Instruction::Opcode::Br:
  case sandboxir::Instruction::Opcode::AddrSpaceCast:
  case sandboxir::Instruction::Opcode::Call:
  case sandboxir::Instruction::Opcode::GetElementPtr:
    llvm_unreachable("Unimplemented");
  case sandboxir::Instruction::Opcode::Insert:
  case sandboxir::Instruction::Opcode::Extract:
  case sandboxir::Instruction::Opcode::ShuffleVec:
    llvm_unreachable("Shouldn't be generated");
  }
  llvm_unreachable("Missing switch case!");
}
namespace llvm {
namespace sandboxir {

std::pair<SmallVector<SmallVector<sandboxir::Use, 2>, 2>, bool>
getUsesBundlesPicky(const DmpVector<sandboxir::Value *> &Bndl) {
  // For now we accept bundles where each element has the same number of users.
  if (Bndl[0]->hasNUsesOrMore(TopDownUseLimit + 1))
    return {{}, /*Fail=*/true};
  unsigned NumUses = Bndl[0]->getNumUses();
  if (any_of(drop_begin(Bndl),
             [NumUses](auto *SBV) { return !SBV->hasNUses(NumUses); }))
    return {{}, /*Fail=*/true};

  // Helper data structure for sorting uses based on OperandNo and opcode.
  SmallVector<SmallVector<sandboxir::Use>> UsesPerLane;
  unsigned Lanes = Bndl.size();
  UsesPerLane.resize(Lanes);
  for (auto [Idx, SBV] : enumerate(Bndl)) {
    auto &Uses = UsesPerLane[Idx];
    for (sandboxir::Use Use : SBV->uses())
      Uses.push_back(Use);
    // Sort uses to try to match opcode and operand index, this should
    // increase the chances of vectorizing the users.
    stable_sort(
        Uses, [](const sandboxir::Use &Use1, const sandboxir::Use &Use2) {
          unsigned OpNo1 = Use1.getOperandNo();
          unsigned OpNo2 = Use2.getOperandNo();
          if (OpNo1 != OpNo2)
            return OpNo1 < OpNo2;
          auto Opc1 = cast<sandboxir::Instruction>(Use1.getUser())->getOpcode();
          auto Opc2 = cast<sandboxir::Instruction>(Use2.getUser())->getOpcode();
          if (Opc1 != Opc2)
            return Opc1 < Opc2;
          return false;
        });
  }

  // Now do the final filtering and return the the uses.
  SmallVector<SmallVector<sandboxir::Use, 2>, 2> FinalUses;
  FinalUses.reserve(NumUses);
  for (auto UseCnt : seq<unsigned>(NumUses)) {
    SmallVector<sandboxir::Use, 2> FinalUsePerLane;
    auto &Lane0Uses = UsesPerLane[/*Lane=*/0];
    const sandboxir::Use &Lane0Use = Lane0Uses[UseCnt];
    FinalUsePerLane.push_back(Lane0Use);
    unsigned Lane0OpNo = Lane0Use.getOperandNo();
    for (auto &LaneUses : drop_begin(UsesPerLane)) {
      const sandboxir::Use &LaneUse = LaneUses[UseCnt];
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
} // namespace sandboxir
} // namespace llvm

sandboxir::VectorizeFromSeeds::VectorizeFromSeeds(sandboxir::BasicBlock *SBBB,
                                                  sandboxir::SBVecContext &Ctx,
                                                  ScalarEvolution &SE,
                                                  const DataLayout &DL,
                                                  TargetTransformInfo &TTI)
    : SBBB(SBBB), Ctx(Ctx), SE(SE), DL(DL), TTI(TTI), Analysis(SE, DL) {
  if (SBBB != nullptr)
    Analysis.init(*SBBB);
}

#ifndef NDEBUG
void sandboxir::VectorizeFromSeeds::dump(raw_ostream &OS) const {
  Analysis.dump();
}
void sandboxir::VectorizeFromSeeds::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
void sandboxir::VectorizeFromSeeds::dumpInstrs(raw_ostream &OS) const {
  Analysis.dumpInstrs(OS);
}
void sandboxir::VectorizeFromSeeds::dumpInstrs() const {
  dumpInstrs(dbgs());
  dbgs() << "\n";
}
#endif

sandboxir::Value *sandboxir::VectorizeFromSeeds::createVecInstruction(
    const DmpVector<sandboxir::Value *> &Bndl,
    const DmpVector<sandboxir::Value *> &Operands, sandboxir::Context &Ctx) {
  assert(all_of(Bndl,
                [](auto *SBV) { return isa<sandboxir::Instruction>(SBV); }) &&
         "Expect sandboxir::Instructions");

  // This lambda creates the instruction but won't add DebugInfo.
  auto CreateVecInstructionImpl =
      [&Ctx](const DmpVector<sandboxir::Value *> &Bndl,
              const DmpVector<sandboxir::Value *> &Operands) {
        Type *ScalarTy = sandboxir::VecUtils::getElementType(
            sandboxir::VecUtils::getExpectedType(Bndl[0]));
        auto *VecTy = sandboxir::VecUtils::getWideType(
            ScalarTy, sandboxir::VecUtils::getNumLanes(Bndl));
        // TODO: Use a sandboxir::SBBuilder to avoid calling different
        // ::create() functions.
        auto [WhereBB, WhereIt] =
            sandboxir::VecUtils::getInsertPointAfterInstrs(Bndl);

        auto Opcode = cast<sandboxir::Instruction>(Bndl[0])->getOpcode();
        switch (Opcode) {
        case sandboxir::Instruction::Opcode::Shuffle:
        case sandboxir::Instruction::Opcode::Pack:
        case sandboxir::Instruction::Opcode::Unpack:
          llvm_unreachable("Unimplemented");
        case sandboxir::Instruction::Opcode::ZExt:
        case sandboxir::Instruction::Opcode::SExt:
        case sandboxir::Instruction::Opcode::FPToUI:
        case sandboxir::Instruction::Opcode::FPToSI:
        case sandboxir::Instruction::Opcode::FPExt:
        case sandboxir::Instruction::Opcode::PtrToInt:
        case sandboxir::Instruction::Opcode::IntToPtr:
        case sandboxir::Instruction::Opcode::SIToFP:
        case sandboxir::Instruction::Opcode::UIToFP:
        case sandboxir::Instruction::Opcode::Trunc:
        case sandboxir::Instruction::Opcode::FPTrunc:
        case sandboxir::Instruction::Opcode::BitCast: {
          assert(Operands.size() == 1u && "Casts are unary!");

          sandboxir::Value *SBV;
          if (WhereIt != WhereBB->end())
            SBV = sandboxir::CastInst::create(VecTy, Opcode, Operands[0],
                                              &*WhereIt, Ctx, "VCast");
          else
            SBV = sandboxir::CastInst::create(VecTy, Opcode, Operands[0],
                                              WhereBB, Ctx, "VCast");

          return SBV;
        }
        case sandboxir::Instruction::Opcode::FCmp:
        case sandboxir::Instruction::Opcode::ICmp: {
          auto Pred = cast<sandboxir::CmpInst>(Bndl[0])->getPredicate();
          assert(
              all_of(drop_begin(Bndl),
                     [Pred](auto *SBV) {
                       return cast<sandboxir::CmpInst>(SBV)->getPredicate() ==
                              Pred;
                     }) &&
              "Expected same predicate across bundle.");
          if (WhereIt != WhereBB->end())
            return sandboxir::CmpInst::create(Pred, Operands[0], Operands[1],
                                              &*WhereIt, Ctx, "VCmp");
          return sandboxir::CmpInst::create(Pred, Operands[0], Operands[1],
                                            WhereBB, Ctx, "Vcmp");
        }
        case sandboxir::Instruction::Opcode::Select: {
          sandboxir::Value *SBV;
          if (WhereIt != WhereBB->end())
            SBV = sandboxir::SelectInst::create(
                Operands[0], Operands[1], Operands[2], &*WhereIt, Ctx, "Vec");
          else
            SBV = sandboxir::SelectInst::create(
                Operands[0], Operands[1], Operands[2], WhereBB, Ctx, "Vec");
          return SBV;
        }
        case sandboxir::Instruction::Opcode::FNeg: {
          auto *UOp0 = cast<sandboxir::UnaryOperator>(Bndl[0]);
          auto OpC = UOp0->getOpcode();
          sandboxir::Value *SBV;
          if (WhereIt != WhereBB->end())
            SBV = sandboxir::UnaryOperator::createWithCopiedFlags(
                OpC, Operands[0], UOp0, &*WhereIt, Ctx, "Vec");
          else
            SBV = sandboxir::UnaryOperator::createWithCopiedFlags(
                OpC, Operands[0], UOp0, WhereBB, Ctx, "Vec");

          if (isa<sandboxir::UnaryOperator>(SBV)) {
            if (UOp0->isFPMath()) {
              assert(!sandboxir::VecUtils::differentMathFlags(Bndl) &&
                     "For now should have packed!");
              // TODO: Use least common math flags if are not the same across
              // Bndl.
            }
          }
          return SBV;
        }
        case sandboxir::Instruction::Opcode::Add:
        case sandboxir::Instruction::Opcode::FAdd:
        case sandboxir::Instruction::Opcode::Sub:
        case sandboxir::Instruction::Opcode::FSub:
        case sandboxir::Instruction::Opcode::Mul:
        case sandboxir::Instruction::Opcode::FMul:
        case sandboxir::Instruction::Opcode::UDiv:
        case sandboxir::Instruction::Opcode::SDiv:
        case sandboxir::Instruction::Opcode::FDiv:
        case sandboxir::Instruction::Opcode::URem:
        case sandboxir::Instruction::Opcode::SRem:
        case sandboxir::Instruction::Opcode::FRem:
        case sandboxir::Instruction::Opcode::Shl:
        case sandboxir::Instruction::Opcode::LShr:
        case sandboxir::Instruction::Opcode::AShr:
        case sandboxir::Instruction::Opcode::And:
        case sandboxir::Instruction::Opcode::Or:
        case sandboxir::Instruction::Opcode::Xor: {
          auto *BinOp0 = cast<sandboxir::BinaryOperator>(Bndl[0]);
          auto *LHS = Operands[0];
          auto *RHS = Operands[1];
          sandboxir::Value *SBV;
          if (WhereIt != WhereBB->end())
            SBV = sandboxir::BinaryOperator::createWithCopiedFlags(
                BinOp0->getOpcode(), LHS, RHS, BinOp0, &*WhereIt, Ctx, "Vec");
          else
            SBV = sandboxir::BinaryOperator::createWithCopiedFlags(
                BinOp0->getOpcode(), LHS, RHS, WhereBB, &*WhereIt, Ctx, "Vec");
          return SBV;
        }
        case sandboxir::Instruction::Opcode::Load: {
          sandboxir::Value *SBV;
          auto *Ld0 = cast<sandboxir::LoadInst>(Bndl[0]);
          sandboxir::Value *Ptr = Ld0->getPointerOperand();
          if (WhereIt != WhereBB->end())
            SBV = sandboxir::LoadInst::create(VecTy, Ptr, Ld0->getAlign(),
                                              &*WhereIt, Ctx, "VecL");
          else
            SBV = sandboxir::LoadInst::create(VecTy, Ptr, Ld0->getAlign(),
                                              WhereBB, Ctx, "VecL");
          return SBV;
        }
        case sandboxir::Instruction::Opcode::Store: {
          sandboxir::Value *SBV;
          auto Align = cast<sandboxir::StoreInst>(Bndl[0])->getAlign();
          sandboxir::Value *Val = Operands[0];
          sandboxir::Value *Ptr = Operands[1];
          if (WhereIt != WhereBB->end())
            SBV =
                sandboxir::StoreInst::create(Val, Ptr, Align, &*WhereIt, Ctx);
          else
            SBV = sandboxir::StoreInst::create(Val, Ptr, Align, WhereBB, Ctx);
          return SBV;
        }
        case sandboxir::Instruction::Opcode::Br:
        case sandboxir::Instruction::Opcode::Ret:
        case sandboxir::Instruction::Opcode::PHI:
        case sandboxir::Instruction::Opcode::Opaque:
        case sandboxir::Instruction::Opcode::AddrSpaceCast:
        case sandboxir::Instruction::Opcode::Call:
        case sandboxir::Instruction::Opcode::GetElementPtr:
          llvm_unreachable("Unimplemented");
          break;
        case sandboxir::Instruction::Opcode::Insert:
        case sandboxir::Instruction::Opcode::Extract:
        case sandboxir::Instruction::Opcode::ShuffleVec:
          llvm_unreachable("Shouldn't be generated");
          break;
        }
        llvm_unreachable("Missing switch case!");
      };

  auto *NewV = CreateVecInstructionImpl(Bndl, Operands);
  if (auto *NewI = dyn_cast<sandboxir::Instruction>(NewV))
    sandboxir::VecUtilsPrivileged::propagateMetadata(NewI, Bndl);
  return NewV;
}

#ifndef NDEBUG
void sandboxir::VectorizeFromSeeds::verifyDAGAndSchedule(
    const DmpVector<sandboxir::Value *> &Bndl, sandboxir::Value *NewSBV) const {
  // Make sure that the DAG and Scheduler have been notified about the
  // creation of the new IR and have created nodes for them.
  auto It = find_if(Bndl, [](sandboxir::Value *SBV) {
    return isa<sandboxir::Instruction>(SBV);
  });
  if (It != Bndl.end()) {
    auto *SBBB = cast<sandboxir::Instruction>(*It)->getParent();
    if (auto *Sched = Ctx.getScheduler(SBBB)) {
      auto &DAG = Sched->getDAG();
      if (auto *NewSBI = dyn_cast<sandboxir::Instruction>(NewSBV)) {
        assert(DAG.getNode(NewSBI) != nullptr && "Expected DAG node!");
        assert(Sched->getBundle(NewSBI) != nullptr && "Expected Bundle!");
      }
    }
  }
}
#endif

bool sandboxir::VectorizeFromSeeds::recOverDebugLimitAtInvocation() const {
  bool MaxRecLimitEnabled = MaxRecursionLimitAt == MaxRecLimAtAlwaysEnabled ||
                            MaxRecursionLimitAt == InvocationCnt - 1;
  if (LLVM_UNLIKELY(MaxRecLimitEnabled && RecCnt > MaxRecursionLimit))
    return true;
  return false;
}

bool sandboxir::VectorizeFromSeeds::maybeStopRecursionForDebugging(
    const DmpVector<sandboxir::Value *> &Bndl, AnalysisResult *&AnalysisRes) {
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

sandboxir::Value *sandboxir::VectorizeFromSeeds::vectorizeRec(
    const DmpVectorView<sandboxir::Value *> &Bndl, AnalysisResult *AnalysisRes,
    SmallPtrSet<sandboxir::Instruction *, 4> &EraseCandidates, bool TopDown,
    sandboxir::Value *VecOp, unsigned OperandNo) {
  LLVM_DEBUG(dbgs() << "Bndl:" << Bndl << " Analysis: " << *AnalysisRes
                    << "\n";);
  // Early return or modify `AnalysisRes` for debugging
  if (maybeStopRecursionForDebugging(Bndl, AnalysisRes))
    return nullptr;

  switch (AnalysisRes->getSubclassID()) {
  case ResultID::PerfectReuseVector: {
    auto *Res = cast<PerfectReuseVector>(AnalysisRes);
    sandboxir::Value *CommonVal = Res->getCommonVal();
    sandboxir::Value *New;
    if (Res->needsShuffle()) {
      ShuffleMask ShuffMask = Res->getShuffleMask();
      if (TopDown) {
        auto WhereIt = sandboxir::VecUtils::getInsertPointAfter({VecOp}, SBBB);
        New = sandboxir::ShuffleInst::create(VecOp, ShuffMask, WhereIt, SBBB,
                                             Ctx);
        auto *ShuffleI = cast<sandboxir::Instruction>(New);
        auto *VecUserI = cast<sandboxir::Instruction>(CommonVal);
        VecUserI->setOperand(OperandNo, ShuffleI);
      } else {
        auto WhereIt =
            sandboxir::VecUtils::getInsertPointAfter({CommonVal}, SBBB);
        New = sandboxir::ShuffleInst::create(CommonVal, ShuffMask, WhereIt,
                                             SBBB, Ctx);
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
    DmpVector<sandboxir::Value *> PackVals(Bndl.size());
    for (auto [Lane, SBV] : enumerate(Bndl)) {
      if (auto *Vec = InstrMaps.getVectorForScalar(SBV)) {
        // // TODO: Reuse existing Unpack if it already exists.
        // sandboxir::Value *Unpack = Ctx.getUnpackInst(Vec, Lane);
        auto OrigLane = InstrMaps.getScalarLane(Vec, SBV);
        auto WhereIt = sandboxir::VecUtils::getInsertPointAfter({Vec}, SBBB);
        auto *Unpack = sandboxir::UnpackInst::create(
            Vec, OrigLane, sandboxir::VecUtils::getNumLanes(SBV), WhereIt, SBBB,
            Ctx);
        PackVals.push_back(Unpack);
      } else {
        PackVals.push_back(SBV);
      }
    }
    auto WhereIt = sandboxir::VecUtils::getInsertPointAfter(PackVals, SBBB);
    auto *Pack = sandboxir::PackInst::create(PackVals, WhereIt, SBBB, Ctx);
    return Pack;
  }
  case ResultID::Unpack: {
    auto *Res = cast<Unpack>(AnalysisRes);
    auto WhereIt =
        sandboxir::VecUtils::getInsertPointAfter({Res->getVecOp()}, SBBB);
    auto *Unpack = sandboxir::UnpackInst::create(
        Res->getVecOp(), Res->getFirstLaneToUnpack(),
        Res->getNumLanesToUnpack(), WhereIt, SBBB, Ctx);
    return Unpack;
  }
  case ResultID::UnpackTopDown: {
    assert(TopDown && "This is used only in TopDown vectorization!");
    emitUnpacks(VecOp, InstrMaps.getScalarsFor(VecOp));
    return nullptr;
  }
  case ResultID::Pack: {
    auto WhereIt = sandboxir::VecUtils::getInsertPointAfter(Bndl, SBBB);
    auto *New = sandboxir::PackInst::create(Bndl, WhereIt, SBBB, Ctx);
    return New;
  }
  case ResultID::SimpleWiden: {
    sandboxir::Value *NewVec = nullptr;
    sandboxir::Value *Bndl0 = Bndl[0];
    bool IsLoadOrStore =
        isa<sandboxir::LoadInst>(Bndl0) || isa<sandboxir::StoreInst>(Bndl0);
    if (!TopDown) {
      SmallVector<DmpVector<sandboxir::Value *>> OperandBndls =
          getOperandBundlesSafe(Bndl);
      DmpVector<sandboxir::Value *> Operands(2);
      for (auto [OpIdx, OpBndl] : enumerate(OperandBndls)) {
        bool IsAddrOperand =
            IsLoadOrStore && OpIdx == sandboxir::VecUtils::getAddrOperandIdx(
                                          cast<sandboxir::Instruction>(Bndl0));
        auto *AnalysisRes =
            IsAddrOperand ? Analysis.createAnalysisResult<KeepLane0>(OpBndl)
                          : Analysis.getBndlAnalysis(OpBndl, InstrMaps,
                                                     EraseCandidates, TopDown);
        auto *New = vectorizeRec(OpBndl, AnalysisRes, EraseCandidates, TopDown,
                                 /*VecOp=*/nullptr);
        Operands.push_back(New);
      }
      // Post-order
      NewVec = createVecInstruction(Bndl, Operands, Ctx);
      InstrMaps.registerVector(NewVec, Bndl);
      // There may be scalar users of values in Bndl so emit unpacks from
      // NewSBV to them.
      emitUnpacks(NewVec, Bndl);
#ifndef NDEBUG
      verifyDAGAndSchedule(Bndl, NewVec);
#endif
    } else {
      DmpVector<sandboxir::Value *> Operands(2);
      for (auto [OpIdx, OpBndl] : enumerate(getOperandBundles(Bndl))) {
        bool IsAddrOperand =
            IsLoadOrStore && OpIdx == sandboxir::VecUtils::getAddrOperandIdx(
                                          cast<sandboxir::Instruction>(Bndl0));
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
          auto *Pack = sandboxir::PackInst::create(
              OpBndl, /*BeforeI=*/sandboxir::VecUtils::getHighest(Bndl), Ctx);
          Operands.push_back(Pack);
        }
      }
      // Pre-order
      NewVec = createVecInstruction(Bndl, Operands, Ctx);
      InstrMaps.registerVector(NewVec, Bndl);
#ifndef NDEBUG
      verifyDAGAndSchedule(Bndl, NewVec);
#endif

      auto [UsesBndls, Fail] = getUsesBundlesPicky(Bndl);
      if (Fail) {
        emitUnpacks(NewVec, Bndl);
      } else {
        for (auto &Uses : UsesBndls) {
          DmpVector<sandboxir::Value *> Users(Uses.size());
          for (sandboxir::Use &Use : Uses)
            Users.push_back(Use.getUser());
          unsigned OperandNo = Uses[0].getOperandNo();
          assert(all_of(drop_begin(Uses),
                        [OperandNo](const sandboxir::Use &Use) {
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
      EraseCandidates.insert(cast<sandboxir::Instruction>(SBV));
    return NewVec;
  }
  case ResultID::KeepLane0: {
    // Since we are still building the graph, we can't know for sure if a
    // value is unused, so add it to the EraseCandidates list.
    for (sandboxir::Value *SBV : drop_begin(Bndl)) {
      if (!isa<sandboxir::Instruction>(SBV))
        continue; // Skip constants like inline-geps, args etc.
      EraseCandidates.insert(cast<sandboxir::Instruction>(SBV));
    }
    return Bndl[0];
  }
  }
  llvm_unreachable("Missing switch cases!");
}

void sandboxir::VectorizeFromSeeds::tryEraseDeadInstrs(
    SmallPtrSet<sandboxir::Instruction *, 4> &EraseCandidates) {
  bool Change = true;
  while (Change) {
    Change = false;
    for (auto *SBI : make_early_inc_range(EraseCandidates))
      if (SBI->getNumUses() == 0) {
        SBI->eraseFromParent();
        Change = true;
        EraseCandidates.erase(SBI);
      }
  }
}

sandboxir::Instruction *sandboxir::VectorizeFromSeeds::tryVectorize(
    const DmpVectorView<sandboxir::Value *> &Seeds, sandboxir::Region &Rgn,
    SmallPtrSet<sandboxir::Instruction *, 4> &EraseCandidates, bool TopDown) {
  InstrMaps.setRegion(Rgn);
  SBBB = cast<sandboxir::Instruction>(Seeds[0])->getParent();
  // Used for debugging.
  if (LLVM_UNLIKELY(InvocationCnt++ >= StopAt))
    return nullptr;
  // Make sure the Scheduler and DAG state are reset.
  Ctx.getScheduler(SBBB)->startFresh(SBBB);
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  Ctx.getScheduler(SBBB)->getDAG().verify();
#endif
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
  sandboxir::Value *Root =
      vectorizeRec(Seeds, AnalysisRes, EraseCandidates, TopDown,
                   /*VecOp=*/nullptr);
  Analysis.clear();
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  SBBB->verifyLLVMIR();
#endif
  // Maintain the data structures.
  for (auto *I : EraseCandidates)
    InstrMaps.eraseScalar(I);
  tryEraseDeadInstrs(EraseCandidates);

  return cast_or_null<sandboxir::Instruction>(Root);
}

void sandboxir::VectorizeFromSeeds::emitUnpacks(
    sandboxir::Value *Vec, const DmpVector<sandboxir::Value *> &Scalars) {
  assert(sandboxir::VecUtils::isVector(Vec) && "Expect vector");
  SmallVector<std::pair<unsigned, sandboxir::Value *>> UnpackData;
  // If SBI has non-bundled users, then we need to create an Unpack.
  unsigned Lane = 0;
  for (sandboxir::Value *OrigElem : Scalars) {
    assert(OrigElem != nullptr && "Expected non-null");
    if (any_of(OrigElem->users(), [this](sandboxir::User *SBU) {
          // Skip if not an instruction.
          if (!isa<sandboxir::Instruction>(SBU))
            return false;
          // Skip users that will be widened.
          auto *Res = Analysis.getAnalysisResult(SBU);
          if (Res != nullptr && isa<SimpleWiden>(Res))
            return false;
          return InstrMaps.getVectorForScalar(
                     cast<sandboxir::Instruction>(SBU)) == nullptr;
        }))
      // Create the Unpack after loop to avoid adding values while
      // iterating
      UnpackData.push_back({Lane, OrigElem});
    Lane += sandboxir::VecUtils::getNumLanes(OrigElem);
  }

  // Now create the unpacks.
  for (auto [Lane, OrigElem] : UnpackData) {
    auto WhereIt = sandboxir::VecUtils::getInsertPointAfter({Vec}, SBBB);
    auto *Unpack = sandboxir::UnpackInst::create(
        Vec, Lane, sandboxir::VecUtils::getNumLanes(OrigElem), WhereIt, SBBB,
        Ctx);

    // There is a rare case when we may need a bitcast:
    // If we are vectorizing values <1 x TYPE> and we have an external user of
    // <1 x TYPE> then our unpack will be of type TYPE, not <1 x TYPE>
    Type *OrigTy = OrigElem->getType();
    bool RequiresBitcast = Unpack->getType() != OrigTy;
    if (RequiresBitcast) {
      assert(cast<FixedVectorType>(OrigTy)->getNumElements() == 1 &&
             "Expected something like <1 x TYPE>");
      sandboxir::Instruction *BeforeI =
          isa<sandboxir::Instruction>(Unpack)
              ? cast<sandboxir::Instruction>(Unpack)->getNextNode()
              : &*SBBB->getFirstNonPHIIt();
      Unpack = sandboxir::CastInst::create(
          OrigTy, sandboxir::Instruction::Opcode::BitCast, Unpack, BeforeI,
          Ctx);
    }
    // Now connect UnpackN to those users that were not combined into a vector.
    OrigElem->replaceUsesWithIf(Unpack, [this](auto Use) {
      auto *DstU = Use.getUser();
      return DstU == nullptr ||
             InstrMaps.getVectorForScalar(cast<sandboxir::Instruction>(DstU)) ==
                 nullptr;
    });
  }
}

bool sandboxir::VectorizeScalars::runOnSBBasicBlock(sandboxir::BasicBlock &BB) {
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
    unsigned TyBits = DL.getTypeSizeInBits(sandboxir::VecUtils::getElementType(
        sandboxir::VecUtils::getExpectedType(
            Seed[Seed.getFirstUnusedElementIdx()])));
    // Try to create the largest vector supported by the target. If it fails
    // reduce the vector size by half.
    for (unsigned SliceBits = std::min(VecRegBits, Seed.getNumUnusedBits(DL)),
                  E = 2 * TyBits;
         SliceBits >= E;
         SliceBits = sandboxir::VecUtils::getFloorPowerOf2(SliceBits / 2)) {
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
        DmpVectorView<sandboxir::Value *> SeedSlice =
            Seed.getSlice(Offset, SliceBits, !AllowNonPow2, DL);
        if (SeedSlice.empty())
          continue;

        assert(SeedSlice.size() >= 2 && "Should have been rejected!");
        SmallPtrSet<sandboxir::Instruction *, 4> EraseCandidates;
        assert(BB.getContext().getTracker().empty() &&
               "Expected empty tracker!");
        sandboxir::Region Rgn(BB, Ctx, TTI);
        // Do the actual vectorization.
        auto *Root = tryVectorize(SeedSlice, Rgn, EraseCandidates, TopDown);
        (void)Root;
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

unsigned long sandboxir::VectorizeFromSeeds::InvocationCnt;
