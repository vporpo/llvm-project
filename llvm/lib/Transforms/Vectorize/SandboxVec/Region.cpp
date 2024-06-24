//===- Region.cpp - The region used by Region passes ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Region.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/SandboxIR/SandboxIR.h"

using namespace llvm;

/// Absolute threshold values greater than this disable cost modeling. This is a
/// workaround for cost model crashes.
static constexpr const int DisableCostModelThreshold = 999;
cl::opt<int> CostThreshold(
    "sbvec-cost-threshold", cl::init(0), cl::Hidden,
    cl::desc("We accept vectorization only if the cost of the IR after "
             "vectorization is at least this much better than the cost before. "
             "NOTE: Absolute threshold values larger than "
             "`DisableCostModelThreshold` automatically reject or accept "
             "without running the cost model."));

void sandboxir::Region::add(sandboxir::Value *VectorI) {
  Vectors.insert(VectorI);

  bool EnableCostModel = std::abs(CostThreshold) < DisableCostModelThreshold;
  auto Cost =
      EnableCostModel ? CM.getCost(cast<sandboxir::Instruction>(VectorI)) : 0;
  VectorCost += Cost;

  // Create and attach metadata.
  MDNode *RgnMDN =
      getOrCreateRegionMDN(sandboxir::ContextAttorney::getLLVMContext(Ctx));
  cast<llvm::Instruction>(ValueAttorney::getValue(VectorI))
      ->setMetadata(MDKind, RgnMDN);
}

MDNode *sandboxir::Region::getOrCreateRegionMDN(LLVMContext &LLVMCtx) {
  if (RegionMDN != nullptr)
    return RegionMDN;
  MDString *RegionMDStr = MDString::get(LLVMCtx, MDStrRegion);
  auto *RegionIDMD = ConstantAsMetadata::get(
      ConstantInt::get(LLVMCtx, APInt(sizeof(RegionID) * 8, RegionID)));
  RegionMDN = MDNode::get(LLVMCtx, {RegionMDStr, RegionIDMD});
#ifndef NDEBUG
  verifyRegionMDN(RegionMDN);
#endif
  return RegionMDN;
}

void sandboxir::Region::remove(sandboxir::Instruction *RemI) {
  bool EnableCostModel = std::abs(CostThreshold) < DisableCostModelThreshold;
  auto Cost = EnableCostModel ? CM.getCost(RemI) : 0;
  if (Vectors.remove(RemI)) {
    VectorCost -= Cost;
  } else {
    // `RemI` is not in `Vectors` so it's a scalar. Add its cost to ScalarCost.
    ScalarCost += Cost;
  }
}

void sandboxir::Region::dropMetadata() {
  for (sandboxir::Value *V : Vectors)
    cast<llvm::Instruction>(ValueAttorney::getValue(V))
        ->setMetadata(MDKind, nullptr);
}

sandboxir::Region::Region(sandboxir::BasicBlock &SBBB, sandboxir::Context &Ctx,
                          TargetTransformInfo &TTI)
    : SBBB(SBBB), Ctx(Ctx), CM(TTI) {
  LLVMContext &LLVMCtx = sandboxir::ContextAttorney::getLLVMContext(Ctx);
  static unsigned StaticRegionID;
  RegionID = StaticRegionID++;
  RegionMDN = getOrCreateRegionMDN(LLVMCtx);

  // Register a callback so that we get notified about new instructions.
  NewInstrCB = Ctx.registerInsertInstrCallback(
      [this](sandboxir::Instruction *VectorI) { add(VectorI); });
  // Register a callback so that we update the region when instrs get removed.
  RemInstrCB = Ctx.registerRemoveInstrCallback(
      [this](sandboxir::Instruction *RemI) { remove(RemI); });
}

sandboxir::Region::~Region() {
  Ctx.unregisterInsertInstrCallback(NewInstrCB);
  NewInstrCB = nullptr;
  Ctx.unregisterRemoveInstrCallback(RemInstrCB);
  RemInstrCB = nullptr;
}

bool sandboxir::Region::empty() const { return Vectors.empty(); }

InstructionCost sandboxir::Region::getVectorMinusScalarCost() const {
  bool EnableCostModel = std::abs(CostThreshold) < DisableCostModelThreshold;
  if (!EnableCostModel)
    return CostThreshold < 0 ? -DisableCostModelThreshold
                             : DisableCostModelThreshold;
  return VectorCost - ScalarCost;
}

#ifndef NDEBUG
bool sandboxir::Region::operator==(const sandboxir::Region &Other) const {
  if (Vectors.size() != Other.Vectors.size())
    return false;
  if (!std::is_permutation(Vectors.begin(), Vectors.end(),
                           Other.Vectors.begin()))
    return false;
  return true;
}

void sandboxir::Region::dump(raw_ostream &OS) const {
  OS << "RegionID: " << getID() << " ScalarCost=" << ScalarCost
     << " VectorCost=" << VectorCost << "\n";
  for (auto *Vector : Vectors)
    OS << *Vector << "\n";
}

void sandboxir::Region::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

void sandboxir::Region::unreachable(MDNode *MDN) {
  errs() << "Bad node: " << *MDN << "\n";
  llvm_unreachable("Bad region metadata!");
}

void sandboxir::Region::verifyRegionMDN(MDNode *RgnMDN) {
  if (RgnMDN->getNumOperands() != 2) {
    errs() << "A " << sandboxir::Region::MDStrRegion
           << " MDNode should have exactly 2 operands, but got "
           << RgnMDN->getNumOperands() << "! \n";
    unreachable(RgnMDN);
  }
  auto *RgnMDStr = dyn_cast<MDString>(
      RgnMDN->getOperand(sandboxir::Region::TLRegionStrOpIdx));
  if (RgnMDStr == nullptr) {
    errs() << "A " << sandboxir::Region::MDStrRegion
           << " MDNode should have an MDString as its operand "
           << sandboxir::Region::TLRegionStrOpIdx << "!\n";
    unreachable(RgnMDN);
  }
  if (RgnMDStr->getString() != sandboxir::Region::MDStrRegion) {
    errs() << "A " << sandboxir::Region::MDStrRegion
           << " MDNode should have the MDString \""
           << sandboxir::Region::MDStrRegion << "\" as its operand "
           << sandboxir::Region::TLRegionStrOpIdx << "!\n";
    unreachable(RgnMDN);
  }
  auto *RgnIDVAM = dyn_cast<ValueAsMetadata>(
      RgnMDN->getOperand(sandboxir::Region::TLRegionIDOpIdx));
  if (RgnIDVAM == nullptr || RgnIDVAM->getValue() == nullptr ||
      !isa<ConstantInt>(RgnIDVAM->getValue())) {
    errs() << "Operand " << sandboxir::Region::TLRegionIDOpIdx << " of a "
           << sandboxir::Region::MDStrRegion
           << " should be the ID (constant int)!\n";
    unreachable(RgnMDN);
  }
}
#endif // NDEBUG

sandboxir::RegionBuilderFromMD::RegionBuilderFromMD(sandboxir::Context &Ctx,
                                                    TargetTransformInfo &TTI)
    : LLVMCtx(sandboxir::ContextAttorney::getLLVMContext(Ctx)), TTI(TTI) {}

MDNode *sandboxir::RegionBuilderFromMD::getRegionMDN(
    sandboxir::Instruction *SBI) const {
  MDNode *MDN = sandboxir::InstructionAttorney::getMetadata(
      SBI, sandboxir::Region::MDKind);
#ifndef NDEBUG
  if (MDN != nullptr)
    sandboxir::Region::verifyRegionMDN(MDN);
#endif
  return MDN;
}

SmallVector<std::unique_ptr<sandboxir::Region>>
sandboxir::RegionBuilderFromMD::createRegionsFromMD(
    sandboxir::BasicBlock &SBBB) {
  SmallVector<std::unique_ptr<sandboxir::Region>> Regions;
  auto &Ctx = SBBB.getContext();
  DenseMap<MDNode *, sandboxir::Region *> MDNToRegion;
  for (sandboxir::Instruction &SBI : reverse(SBBB)) {
    if (auto *RgnMDN = getRegionMDN(&SBI)) {
      sandboxir::Region *Rgn = nullptr;
      auto It = MDNToRegion.find(RgnMDN);
      if (It == MDNToRegion.end()) {
        Regions.push_back(
            std::make_unique<sandboxir::Region>(SBBB, Ctx, TTI));
        Rgn = Regions.back().get();
        MDNToRegion[RgnMDN] = Rgn;
      } else {
        Rgn = It->second;
      }
      Rgn->add(&SBI);
    }
  }
  return Regions;
}
