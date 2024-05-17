//===- SBRegion.cpp - The region used by SBRegion passes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/SBRegion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"

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

void SBRegion::add(SBValue *VectorI) {
  Vectors.insert(VectorI);

  bool EnableCostModel = std::abs(CostThreshold) < DisableCostModelThreshold;
  auto Cost = EnableCostModel ? CM.getCost(cast<SBInstruction>(VectorI)) : 0;
  VectorCost += Cost;

  // Create and attach metadata.
  MDNode *RgnMDN =
      getOrCreateRegionMDN(SBContextAttorney::getLLVMContext(Ctxt));
  cast<Instruction>(ValueAttorney::getValue(VectorI))
      ->setMetadata(MDKind, RgnMDN);
}

MDNode *SBRegion::getOrCreateRegionMDN(LLVMContext &LLVMCtxt) {
  if (RegionMDN != nullptr)
    return RegionMDN;
  MDString *RegionMDStr = MDString::get(LLVMCtxt, MDStrRegion);
  auto *RegionIDMD = ConstantAsMetadata::get(
      ConstantInt::get(LLVMCtxt, APInt(sizeof(RegionID) * 8, RegionID)));
  RegionMDN = MDNode::get(LLVMCtxt, {RegionMDStr, RegionIDMD});
#ifndef NDEBUG
  verifyRegionMDN(RegionMDN);
#endif
  return RegionMDN;
}

void SBRegion::remove(SBInstruction *RemI) {
  bool EnableCostModel = std::abs(CostThreshold) < DisableCostModelThreshold;
  auto Cost = EnableCostModel ? CM.getCost(RemI) : 0;
  if (Vectors.remove(RemI)) {
    VectorCost -= Cost;
  } else {
    // `RemI` is not in `Vectors` so it's a scalar. Add its cost to ScalarCost.
    ScalarCost += Cost;
  }
}

void SBRegion::dropMetadata() {
  for (SBValue *V : Vectors)
    cast<Instruction>(ValueAttorney::getValue(V))->setMetadata(MDKind, nullptr);
}

SBRegion::SBRegion(SBBasicBlock &SBBB, SBContext &Ctxt,
                     TargetTransformInfo &TTI)
    : SBBB(SBBB), Ctxt(Ctxt), CM(TTI) {
  LLVMContext &LLVMCtxt = SBContextAttorney::getLLVMContext(Ctxt);
  static unsigned StaticRegionID;
  RegionID = StaticRegionID++;
  RegionMDN = getOrCreateRegionMDN(LLVMCtxt);

  // Register a callback so that we get notified about new instructions.
  NewInstrCB = Ctxt.registerInsertInstrCallback(
      [this](SBInstruction *VectorI) { add(VectorI); });
  // Register a callback so that we update the region when instrs get removed.
  RemInstrCB = Ctxt.registerRemoveInstrCallback(
      [this](SBInstruction *RemI) { remove(RemI); });
}

SBRegion::~SBRegion() {
  Ctxt.unregisterInsertInstrCallback(NewInstrCB);
  NewInstrCB = nullptr;
  Ctxt.unregisterRemoveInstrCallback(RemInstrCB);
  RemInstrCB = nullptr;
}

bool SBRegion::empty() const { return Vectors.empty(); }

InstructionCost SBRegion::getVectorMinusScalarCost() const {
  bool EnableCostModel = std::abs(CostThreshold) < DisableCostModelThreshold;
  if (!EnableCostModel)
    return CostThreshold < 0 ? -DisableCostModelThreshold
                             : DisableCostModelThreshold;
  return VectorCost - ScalarCost;
}

#ifndef NDEBUG
bool SBRegion::operator==(const SBRegion &Other) const {
  if (Vectors.size() != Other.Vectors.size())
    return false;
  if (!std::is_permutation(Vectors.begin(), Vectors.end(),
                           Other.Vectors.begin()))
    return false;
  return true;
}

void SBRegion::dump(raw_ostream &OS) const {
  OS << "RegionID: " << getID() << " ScalarCost=" << ScalarCost
     << " VectorCost=" << VectorCost << "\n";
  for (auto *Vector : Vectors)
    OS << *Vector << "\n";
}

void SBRegion::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

void SBRegion::unreachable(MDNode *MDN) {
  errs() << "Bad node: " << *MDN << "\n";
  llvm_unreachable("Bad region metadata!");
}

void SBRegion::verifyRegionMDN(MDNode *RgnMDN) {
  if (RgnMDN->getNumOperands() != 2) {
    errs() << "A " << SBRegion::MDStrRegion
           << " MDNode should have exactly 2 operands, but got "
           << RgnMDN->getNumOperands() << "! \n";
    unreachable(RgnMDN);
  }
  auto *RgnMDStr =
      dyn_cast<MDString>(RgnMDN->getOperand(SBRegion::TLRegionStrOpIdx));
  if (RgnMDStr == nullptr) {
    errs() << "A " << SBRegion::MDStrRegion
           << " MDNode should have an MDString as its operand "
           << SBRegion::TLRegionStrOpIdx << "!\n";
    unreachable(RgnMDN);
  }
  if (RgnMDStr->getString() != SBRegion::MDStrRegion) {
    errs() << "A " << SBRegion::MDStrRegion
           << " MDNode should have the MDString \"" << SBRegion::MDStrRegion
           << "\" as its operand " << SBRegion::TLRegionStrOpIdx << "!\n";
    unreachable(RgnMDN);
  }
  auto *RgnIDVAM =
      dyn_cast<ValueAsMetadata>(RgnMDN->getOperand(SBRegion::TLRegionIDOpIdx));
  if (RgnIDVAM == nullptr || RgnIDVAM->getValue() == nullptr ||
      !isa<ConstantInt>(RgnIDVAM->getValue())) {
    errs() << "Operand " << SBRegion::TLRegionIDOpIdx << " of a "
           << SBRegion::MDStrRegion << " should be the ID (constant int)!\n";
    unreachable(RgnMDN);
  }
}
#endif // NDEBUG

SBRegionBuilderFromMD::SBRegionBuilderFromMD(SBContext &Ctxt,
                                               TargetTransformInfo &TTI)
    : LLVMCtxt(SBContextAttorney::getLLVMContext(Ctxt)), TTI(TTI) {}

MDNode *SBRegionBuilderFromMD::getRegionMDN(SBInstruction *SBI) const {
  MDNode *MDN = SBInstructionAttorney::getMetadata(SBI, SBRegion::MDKind);
#ifndef NDEBUG
  if (MDN != nullptr)
    SBRegion::verifyRegionMDN(MDN);
#endif
  return MDN;
}

SmallVector<std::unique_ptr<SBRegion>>
SBRegionBuilderFromMD::createRegionsFromMD(SBBasicBlock &SBBB) {
  SmallVector<std::unique_ptr<SBRegion>> Regions;
  auto &Ctxt = SBBB.getContext();
  DenseMap<MDNode *, SBRegion *> MDNToRegion;
  for (SBInstruction &SBI : reverse(SBBB)) {
    if (auto *RgnMDN = getRegionMDN(&SBI)) {
      SBRegion *Rgn = nullptr;
      auto It = MDNToRegion.find(RgnMDN);
      if (It == MDNToRegion.end()) {
        Regions.push_back(std::make_unique<SBRegion>(SBBB, Ctxt, TTI));
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
