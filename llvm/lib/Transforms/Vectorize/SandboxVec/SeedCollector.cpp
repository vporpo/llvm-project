//===- SeedCollection.cpp  -0000000----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/SeedCollector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Utils.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"

using namespace llvm;

cl::opt<bool>
    DisableStoreSeeds("sbvec-disable-store-seeds", cl::init(false),
                      cl::Hidden,
                      cl::desc("Don't collect store seed instructions."));
cl::opt<bool>
    DisableLoadSeeds("sbvec-disable-load-seeds", cl::init(true), cl::Hidden,
                     cl::desc("Don't collect load seed instructions."));

cl::opt<bool> DisableInstrOpSeeds(
    "sbvec-disable-instr-op-seeds", cl::init(false), cl::Hidden,
    cl::desc("Don't collect operands of single instructions as seeds."));

#define LoadSeedsDef "loads"
#define StoreSeedsDef "stores"
#define InstrOpSeedsDef "instrops"
cl::opt<std::string>
    ForceSeed("sbvec-force-seeds", cl::init(""), cl::Hidden,
              cl::desc("Enable only this type of seeds. This can be one "
                       "of: '" LoadSeedsDef "','" StoreSeedsDef
                       "','" InstrOpSeedsDef "'."));

cl::opt<unsigned> SeedGroupsLimit(
    "sbvec-seed-groups-limit", cl::init(256), cl::Hidden,
    cl::desc("Limit the number of collected seeds groups in a BB to "
             "cap compilation time."));
cl::opt<unsigned> SeedBundleSizeLimit(
    "sbvec-seed-bundle-size-limit", cl::init(32), cl::Hidden,
    cl::desc("Limit the size of the seed bundle to cap compilation time."));

SeedBundle::SeedBundle(SBInstruction *SBI) : SBValBundle(SBI) {}

SBValBundle SeedBundle::getSlice(unsigned Offset, unsigned MaxVecRegBits,
                                   bool ForcePowOf2,
                                   const DataLayout &DL) const {
  // We count both the bits and the elements of the slice we are about to build.
  // The bits tell us whether this is a legal slice (that is <= MaxVecRegBits),
  // and the num of elements help us do the actual slicing.
  unsigned BitsSum = 0;
  // As we are collecting slice elements we may go over the limit, so we need to
  // remember the last legal one. This is used for the creation of the slice.
  unsigned LastGoodBitsSum = 0;
  unsigned LastGoodNumSliceElements = 0;
  // We are skipping any used elements and all below `Offset`.
  assert(Offset >= getFirstUnusedElementIdx() && "Expected offset at unused!");
  unsigned FirstGoodElementIdx = Offset;
  // Go through elements starting at FirstGoodElementIdx.
  for (auto [ElementCnt, SBV] :
       enumerate(make_range(std::next(begin(), FirstGoodElementIdx), end()))) {
    // Stop if we found a used element.
    if (isUsed(FirstGoodElementIdx + ElementCnt))
      break;
    BitsSum += SBUtils::getNumBits(SBV, DL);
    // Stop if the bits sum is over the limit.
    if (BitsSum > MaxVecRegBits)
      break;
    // If forcing a power-of-2 bit-size we check if this bit size is accepted.
    if (ForcePowOf2 && !SBUtils::isPowerOf2(BitsSum))
      continue;
    LastGoodBitsSum = BitsSum;
    LastGoodNumSliceElements = ElementCnt + 1;
  }
  if (LastGoodNumSliceElements < 2)
    return {};
  if (LastGoodBitsSum == 0)
    return {};
  SBValBundle Slice(LastGoodNumSliceElements);
  for (unsigned Idx : seq<unsigned>(
           FirstGoodElementIdx, FirstGoodElementIdx + LastGoodNumSliceElements))
    Slice.push_back(*(begin() + Idx));
  assert(
      (!ForcePowOf2 || SBUtils::isPowerOf2(SBUtils::getNumBits(Slice, DL))) &&
      "Expected power of 2!");
  assert(Slice.size() >= 2 && "Bad size!");
  return Slice;
}

unsigned SeedBundle::getNumUnusedBits(const DataLayout &DL) const {
  unsigned Bits = 0;
  for (auto [Elm, SBV] : enumerate(*this)) {
    if (isUsed(Elm))
      continue;
    Bits += SBUtils::getNumBits(SBV, DL);
  }
  return Bits;
}

#ifndef NDEBUG
void SeedBundle::dump(raw_ostream &OS) const {
  for (auto [ElmIdx, SBV] : enumerate(Vals)) {
    OS.indent(2) << ElmIdx << ". ";
    if (isUsed(ElmIdx))
      OS << "[USED] <perhaps deleted>";
    else
      OS << *SBV;
    OS << "\n";
  }
}
void SeedBundle::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

/// \Returns false if \p ElemTy cannot be vectorized on the target. This is to
/// save compilation time by skipping these typee early on.. Otherwise the cost
/// model would have to turn them down.
static bool cannotVectorizeOnTarget(Type *ElemTy) {
  return ElemTy->isX86_FP80Ty() || ElemTy->isPPC_FP128Ty();
}

#ifndef NDEBUG
void SeedContainer::dump(raw_ostream &OS) const {
  for (const SeedBundle &SB : *this) {
    SB.dump(OS);
    OS << "\n";
  }
}
void SeedContainer::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

template <typename LoadOrStoreT>
MemSeedContainer::KeyT MemSeedContainer::getKey(LoadOrStoreT *LSI) const {
  assert((isa<SBLoadInstruction>(LSI) || isa<SBStoreInstruction>(LSI)) &&
         "Expected Load or Store!");
  Value *Ptr =
      getUnderlyingObject(ValueAttorney::getValue(LSI->getPointerOperand()));
  Type *Ty = SBUtils::getElementType(SBUtils::getExpectedType(LSI));
  SBInstruction::Opcode Op = LSI->getOpcode();
  return {Ptr, Ty, Op};
}
// Explicit instantiations
template MemSeedContainer::KeyT
MemSeedContainer::getKey<SBLoadInstruction>(SBLoadInstruction *LSI) const;
template MemSeedContainer::KeyT
MemSeedContainer::getKey<SBStoreInstruction>(SBStoreInstruction *LSI) const;

bool MemSeedContainer::erase(SBValue *SBV) {
  assert(
      (isa<SBLoadInstruction>(SBV) || isa<SBStoreInstruction>(SBV)) &&
      "Expected Load or Store!");
  auto It = LookupMap.find(SBV);
  if (It == LookupMap.end())
    return false;
  SeedBundle *Bndl = It->second;
  Bndl->setUsed(SBV);
  return true;
}

template <typename LoadOrStoreT>
void MemSeedContainer::insert(LoadOrStoreT *LSI) {
  // Get all seeds that correspond to the key.
  auto &SeedsVec = Seeds[getKey(LSI)];
  // Try to append `LSI` into an existing seedbundle, and return on success.
  bool Inserted = false;
  for (auto &SeedBundlePtr : SeedsVec) {
    // Cap compilation time by keeping SeedBundles small because attempting to
    // vectorize them does not scale. So start a new SeedBundle instead of
    // appending to an existing one.
    if (SeedBundlePtr->size() >= SeedBundleSizeLimit)
      continue;
    if ((Inserted = SeedBundlePtr->tryInsert(LSI, &DL, &SE))) {
      assert(!LookupMap.contains(LSI) && "Expected unique key->value!");
      LookupMap[LSI] = SeedBundlePtr.get();
      return;
    }
  }
  // If we didn't find a suitable bundle, create a new one.
  if (!Inserted) {
    auto BndlPtr = std::make_unique<MemSeedBundle<LoadOrStoreT>>(LSI);
    assert(!LookupMap.contains(LSI) && "Expected unique key->value!");
    LookupMap[LSI] = BndlPtr.get();
    SeedsVec.push_back(std::move(BndlPtr));
  }
}

// Explicit instantiations
template void
MemSeedContainer::insert<SBLoadInstruction>(SBLoadInstruction *);
template void
MemSeedContainer::insert<SBStoreInstruction>(SBStoreInstruction *);

void MemSeedContainer::purge() {
  SmallVector<KeyT> KeysToErase;
  for (auto &Pair : Seeds) {
    auto &SeedsVec = Pair.second;
    assert(!SeedsVec.empty() && "We don't expect any empty vectors!");
    DenseSet<SBValue *> RemovedVals;
    SeedsVec.erase(remove_if(SeedsVec,
                             [&RemovedVals](const auto &SeedPtr) {
                               bool Remove = SeedPtr->size() == 1;
                               if (Remove)
                                 RemovedVals.insert(*SeedPtr->begin());
                               return Remove;
                             }),
                   SeedsVec.end());
    if (SeedsVec.empty())
      KeysToErase.push_back(Pair.first);
    for (SBValue *SBV : RemovedVals)
      LookupMap.erase(SBV);
  }
  for (const auto &Key : KeysToErase)
    Seeds.erase(Key);
}

#ifndef NDEBUG
void MemSeedContainer::dump(raw_ostream &OS) const {
  for (const auto &Pair : Seeds) {
    auto [Val, Ty, Opc] = Pair.first;
    const auto &SeedsVec = Pair.second;
    OS << "[Val=" << *Val << " Ty=" << *Ty << " Opc=" << Opc << "]\n";
    for (const auto &SeedPtr : SeedsVec) {
      SeedPtr->dump(OS);
      OS << "\n";
    }
  }
}

void MemSeedContainer::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

template <typename LoadOrStoreT> static bool isBadMemSeed(LoadOrStoreT *LSI) {
  if (!LSI->isSimple())
    return true;
  Type *ElemTy = SBUtils::getElementType(SBUtils::getExpectedType(LSI));
  if (!VectorType::isValidElementType(ElemTy))
    return true;
  if (cannotVectorizeOnTarget(ElemTy))
    return true;
  return false;
}

SeedCollector::SeedCollector(SBBasicBlock *SBBB, const DataLayout &DL,
                             ScalarEvolution &SE)
    : StoreSeeds(DL, SE), LoadSeeds(DL, SE), Ctxt(SBBB->getContext()) {
  // Register a callback for updating the seed datastructures upon instr removal
  RemoveInstrCB =
      Ctxt.registerRemoveInstrCallback([this](SBInstruction *ErasedI) {
        NonStoreSeeds.erase(ErasedI);
        if (isa<SBStoreInstruction>(ErasedI))
          StoreSeeds.erase(ErasedI);
        else if (isa<SBLoadInstruction>(ErasedI))
          LoadSeeds.erase(ErasedI);
      });

  bool CollectStores = !DisableStoreSeeds;
  bool CollectLoads = !DisableLoadSeeds;
  bool CollectInstrOps = !DisableInstrOpSeeds;
  if (LLVM_UNLIKELY(!ForceSeed.empty())) {
    // Disable all.
    CollectStores = false;
    CollectLoads = false;
    CollectInstrOps = false;
    // Enable only the selected one.
    if (ForceSeed == StoreSeedsDef)
      CollectStores = true;
    else if (ForceSeed == LoadSeedsDef)
      CollectLoads = true;
    else if (ForceSeed == InstrOpSeedsDef)
      CollectInstrOps = true;
    else {
      errs() << "Bad argument '" << ForceSeed << "' in -" << ForceSeed.ArgStr
             << "='" << ForceSeed << "'.\n";
      errs() << "Description: " << ForceSeed.HelpStr << "\n";
      exit(1);
    }
  }
  // Collect seeds.
  for (SBInstruction &SBI : *SBBB) {
    if (LLVM_LIKELY(CollectStores)) {
      if (auto *TSI = dyn_cast<SBStoreInstruction>(&SBI)) {
        if (isBadMemSeed<SBStoreInstruction>(TSI))
          continue;
        // Cap compilation time.
        if (totalNumSeedGroups() > SeedGroupsLimit)
          break;
        StoreSeeds.insert(TSI);
      }
    }

    if (LLVM_LIKELY(CollectLoads)) {
      if (auto *TLI = dyn_cast<SBLoadInstruction>(&SBI)) {
        if (isBadMemSeed<SBLoadInstruction>(TLI))
          continue;
        // Cap compilation time.
        if (totalNumSeedGroups() > SeedGroupsLimit)
          break;
        LoadSeeds.insert(TLI);
      }
    }

    if (LLVM_LIKELY(CollectInstrOps)) {
      if (SBI.getNumOperands() == 2) {
        // Cap compilation time.
        if (totalNumSeedGroups() > SeedGroupsLimit)
          break;
        auto *OpLHS = dyn_cast_or_null<SBInstruction>(SBI.getOperand(0));
        auto *OpRHS = dyn_cast_or_null<SBInstruction>(SBI.getOperand(1));
        if (OpLHS && OpRHS && OpLHS != OpRHS &&
            OpLHS->getOpcode() == OpRHS->getOpcode() &&
            OpLHS->getParent() == SBBB && OpRHS->getParent() == SBBB)
          NonStoreSeeds.insert(SBValBundle{OpLHS, OpRHS});
      }
    }
  }

  // TODO: Is this really needed?
  StoreSeeds.purge();
  LoadSeeds.purge();
}

SeedCollector::~SeedCollector() {
  Ctxt.unregisterRemoveInstrCallback(RemoveInstrCB);
}

#ifndef NDEBUG
void SeedCollector::dump(raw_ostream &OS) const {
  OS << "=== StoreSeeds ===\n";
  StoreSeeds.dump(OS);
  OS << "=== LoadSeeds ===\n";
  LoadSeeds.dump(OS);
  OS << "=== NonMemSeeds ===\n";
  NonStoreSeeds.dump(OS);
}

void SeedCollector::dump() const { dump(dbgs()); }
#endif
