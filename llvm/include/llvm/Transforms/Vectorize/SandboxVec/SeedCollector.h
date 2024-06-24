//===- SeedCollector.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This class collects the seed instructions that are used as starting points
// for forming the vectorization graph.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SEEDCOLLECTOR_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SEEDCOLLECTOR_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/SandboxIR/DmpVector.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Transforms/Vectorize/SandboxVec/VecUtils.h"
#include <iterator>
#include <memory>

namespace llvm {
namespace sandboxir {
class Instruction;
class StoreInst;
class BasicBlock;

/// An ordered set of Instructions that can be vectorized.
class SeedBundle : public DmpVector<sandboxir::Value *> {
  /// The lanes that we have already vectorized.
  BitVector UsedLanes;

public:
  explicit SeedBundle(DmpVector<sandboxir::Value *> &&Bndl)
      : DmpVector<sandboxir::Value *>(std::move(Bndl)) {}
  explicit SeedBundle(const DmpVector<sandboxir::Value *> &Bndl)
      : DmpVector<sandboxir::Value *>(Bndl) {}
  /// Initialize a seed with \p SBI.
  explicit SeedBundle(sandboxir::Instruction *SBI);
  /// No need to allow copies.
  SeedBundle(const SeedBundle &) = delete;
  SeedBundle &operator=(const SeedBundle &) = delete;
  virtual ~SeedBundle() {}
  using iterator = DmpVector<sandboxir::Value *>::iterator;
  using const_iterator = DmpVector<sandboxir::Value *>::const_iterator;
  sandboxir::Instruction *operator[](unsigned Idx) const {
    return cast<sandboxir::Instruction>(
        DmpVector<sandboxir::Value *>::operator[](Idx));
  }
  /// This will insert \p SBI into its sorting position.
  virtual bool tryInsert(sandboxir::Instruction *SBI, const DataLayout *DL,
                         ScalarEvolution *SE) {
    llvm_unreachable("Unimplemented!");
  }
  unsigned getFirstUnusedElementIdx() const {
    for (unsigned ElmIdx : seq<unsigned>(0, size()))
      if (!isUsed(ElmIdx))
        return ElmIdx;
    return size();
  }
  /// Marks elements as 'used' so that we skip them in `getSlice()`.
  void setUsed(unsigned ElementIdx, unsigned Sz = 1, bool VerifyUnused = true) {
    if (ElementIdx + Sz >= UsedLanes.size())
      UsedLanes.resize(ElementIdx + Sz);
    for (unsigned Idx : seq<unsigned>(ElementIdx, ElementIdx + Sz)) {
      assert((!VerifyUnused || !UsedLanes.test(Idx)) &&
             "Already marked as used!");
      UsedLanes.set(Idx);
    }
  }
  void setUsed(sandboxir::Value *SBV) {
    auto It = find(*this, SBV);
    assert(It != end() && "SBV not in the bundle!");
    auto Idx = It - begin();
    setUsed(Idx, 1, /*VerifyUnused=*/false);
  }
  bool isUsed(unsigned Element) const {
    return Element >= UsedLanes.size() ? false : UsedLanes.test(Element);
  }
  // TODO: Make this a constant-time operation.
  bool allUsed() const { return UsedLanes.count() == size(); }
  /// \Returns a slice of seed elements, starting at \p Offset, with a total
  /// size <= \p MaxVecRegBits. If \p ForcePowOf2 is true, then the returned
  /// slice should have a total number of bits that is a power of 2.
  DmpVectorView<sandboxir::Value *> getSlice(unsigned Offset,
                                               unsigned MaxVecRegBits,
                                               bool ForcePowOf2,
                                               const DataLayout &DL) const;
  /// \Returns the number of unused bits in the seed by skipping used elements.
  unsigned getNumUnusedBits(const DataLayout &DL) const;
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif // NDEBUG
};

template <typename LoadOrStoreT> class MemSeedBundle : public SeedBundle {
public:
  explicit MemSeedBundle(DmpVector<sandboxir::Value *> &&Bndl)
      : SeedBundle(std::move(Bndl)) {
    assert(all_of(*this, [](auto *SBV) { return isa<LoadOrStoreT>(SBV); }) &&
           "Expected Load or Store instructions!");
  }
  explicit MemSeedBundle(LoadOrStoreT *MemI) : SeedBundle(MemI) {
    assert(isa<LoadOrStoreT>(MemI) && "Expected Load or Store!");
  }
  bool tryInsert(sandboxir::Instruction *SBI, const DataLayout *DL,
                 ScalarEvolution *SE) final {
    assert(isa<LoadOrStoreT>(SBI) && "Expected a Store or a Load!");
    auto Cmp = [DL, SE](sandboxir::Value *SBV1, sandboxir::Value *SBV2) {
      return sandboxir::VecUtils::comesBeforeInMem(
          cast<LoadOrStoreT>(SBV1), cast<LoadOrStoreT>(SBV2), *SE, *DL);
    };
    // Find the first element after TSI in mem. Then insert TSI before it.
    auto WhereIt = std::upper_bound(begin(), end(), SBI, Cmp);
    insert(WhereIt, SBI);
    return true;
  }
};
using StoreSeedBundle = MemSeedBundle<sandboxir::StoreInst>;
using LoadSeedBundle = MemSeedBundle<sandboxir::LoadInst>;

class SeedContainerBase {
public:
  SeedContainerBase() = default;
  virtual ~SeedContainerBase() {}
  // No need to allow copies.
  SeedContainerBase(const SeedContainerBase &) = delete;
  SeedContainerBase &operator=(const SeedContainerBase &) = delete;
  virtual bool erase(sandboxir::Value *SBV) = 0;
#ifndef NDEBUG
  virtual void dump(raw_ostream &OS) const = 0;
  virtual void dump() const = 0;
#endif // NDEBUG
};

/// A datastructure that holds seed bundles, but also has a constant-time erase
/// which is needed to maintain this data structure upon instruction erase.
class SeedContainer : public SeedContainerBase {
  /// Maps each instruction in the seed bundle to the seed bundles that contain
  /// them.
  DenseMap<sandboxir::Value *, DenseSet<SeedBundle *>> SeedToBundlesMap;
  /// An ordered collection of the bundles for iterating over them
  /// deterministically.
  using MapTy = MapVector<SeedBundle *, std::unique_ptr<SeedBundle>>;
  MapTy OrderedPool;

public:
  class iterator {
    MapTy::const_iterator It;
    const MapTy *Map = nullptr;

  public:
    using difference_type = std::ptrdiff_t;
    using value_type = SeedBundle;
    using pointer = value_type *;
    using reference = value_type &;
    using iterator_category = std::input_iterator_tag;

    explicit iterator(MapTy::const_iterator It, const MapTy &Map)
        : It(It), Map(&Map) {}
    // Skip used bundles by repeatedly calling operator++().
    void skipUsed() {
      auto AtEnd = [this]() { return It == Map->end(); };
      while (!AtEnd() && this->operator*().allUsed())
        ++(*this);
    }
    value_type &operator*() const { return *It->second; }
    bool operator==(const iterator &Other) const { return It == Other.It; }
    bool operator!=(const iterator &Other) const { return !(*this == Other); }
    iterator &operator++() {
      ++It;
      skipUsed();
      return *this;
    }
    iterator operator++(int) {
      auto Copy = *this;
      ++(*this);
      return Copy;
    }
  };
  SeedContainer() : SeedContainerBase() {}
  void insert(DmpVector<sandboxir::Value *> &&Seeds) {
    auto SBPtr = std::make_unique<SeedBundle>(std::move(Seeds));
    auto *SB = SBPtr.get();
#ifndef NDEBUG
    DenseSet<sandboxir::Value *> Visited;
    Visited.reserve(SBPtr->size());
#endif
    for (sandboxir::Value *Seed : *SBPtr) {
      SeedToBundlesMap[Seed].insert(SB);
#ifndef NDEBUG
      assert(Visited.insert(Seed).second &&
             "Value appears more than once in bundle!");
#endif
    }
    OrderedPool.insert({SB, std::move(SBPtr)});
  }
  /// Despite its name, this won't actually erase \p SBV from the container.
  /// Instead it marks it as used. \Returns true if \p SBV was found in the
  /// bundles.
  bool erase(sandboxir::Value *SBV) final {
    auto It = SeedToBundlesMap.find(SBV);
    if (It == SeedToBundlesMap.end())
      return false;
    for (auto *SB : It->second)
      SB->setUsed(SBV);
    return true;
  }
  iterator begin() const {
    if (OrderedPool.empty())
      return end();
    auto BeginIt = iterator(OrderedPool.begin(), OrderedPool);
    BeginIt.skipUsed();
    return BeginIt;
  }
  iterator end() const { return iterator(OrderedPool.end(), OrderedPool); }
  unsigned size() const { return SeedToBundlesMap.size(); }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
#endif // NDEBUG
};

class MemSeedContainer : public SeedContainerBase {
  using KeyT =
      std::tuple<llvm::Value *, Type *, sandboxir::Instruction::Opcode>;
  using ValT = SmallVector<std::unique_ptr<SeedBundle>>;
  using SeedMapT = MapVector<KeyT, ValT>;
  /// Map from {pointer llvm::value *, Type, Opcode} to a vector of bundles
  SeedMapT Seeds;
  // TODO: This is not strictly needed, we could be using a key lookup into
  // `Seeds`, but there can be multiple values with the same key, so the value
  // may not actually be in the bundles in `Seeds`.
  DenseMap<sandboxir::Value *, SeedBundle *> LookupMap;

  const DataLayout &DL;
  ScalarEvolution &SE;

  template <typename LoadOrStoreT> KeyT getKey(LoadOrStoreT *LSI) const;

public:
  MemSeedContainer(const DataLayout &DL, ScalarEvolution &SE)
      : SeedContainerBase(), DL(DL), SE(SE) {}
  class iterator {
    SeedMapT::iterator MapIt;
    int VecIdx;
    SeedMapT *Map = nullptr;
    ValT *Vec = nullptr;

  public:
    static constexpr const int EndIdx = -1;
    using difference_type = std::ptrdiff_t;
    using value_type = SeedBundle;
    using pointer = value_type *;
    using reference = value_type &;
    using iterator_category = std::input_iterator_tag;

    iterator(SeedMapT::iterator MapIt, int VecIdx, SeedMapT &Map, ValT *Vec)
        : MapIt(MapIt), VecIdx(VecIdx), Map(&Map), Vec(Vec) {}
    value_type &operator*() {
      assert(Vec != nullptr && "Already at end!");
      return *(*Vec)[VecIdx];
    }
    // Skip used bundles by repeatedly calling operator++().
    void skipUsed() {
      auto AtEnd = [this]() { return VecIdx == EndIdx; };
      while (!AtEnd() && this->operator*().allUsed())
        ++(*this);
    }
    iterator &operator++() {
      assert(VecIdx >= 0 && "Already at end!");
      int VecSz = Vec->size();
      auto NextVecIdx = VecIdx + 1;
      if (NextVecIdx < VecSz) {
        VecIdx = NextVecIdx;
      } else {
        assert(MapIt != Map->end() && "Already at end!");
        VecIdx = 0;
        ++MapIt;
        if (MapIt != Map->end())
          Vec = &MapIt->second;
        else {
          Vec = nullptr;
          VecIdx = EndIdx;
        }
      }
      skipUsed();
      return *this;
    }
    iterator operator++(int) {
      auto Copy = *this;
      ++(*this);
      return Copy;
    }
    bool operator==(const iterator &Other) const {
      assert(Map == Other.Map && "Iterator of different objects!");
      return MapIt == Other.MapIt && VecIdx == Other.VecIdx;
    }
    bool operator!=(const iterator &Other) const { return !(*this == Other); }
  };
  using const_iterator = SeedMapT::const_iterator;
  template <typename LoadOrStoreT> void insert(LoadOrStoreT *LSI);
  bool erase(sandboxir::Value *SBV) final;
  bool erase(const KeyT &Key) { return Seeds.erase(Key); }
  iterator begin() {
    if (Seeds.empty())
      return end();
    auto BeginIt = iterator(Seeds.begin(), 0, Seeds, &Seeds.begin()->second);
    BeginIt.skipUsed();
    return BeginIt;
  }
  iterator end() {
    return iterator(Seeds.end(), iterator::EndIdx, Seeds, nullptr);
  }
  unsigned size() const { return Seeds.size(); }
  /// Erases all single-element bundles.
  void purge();
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
#endif // NDEBUG
};

class SeedCollector {
  MemSeedContainer StoreSeeds;
  MemSeedContainer LoadSeeds;

  /// This holds all other seed bundles other than stores.
  SeedContainer NonStoreSeeds;

  /// Callback called upon instruction removal.
  sandboxir::Context::RemoveCBTy *RemoveInstrCB;
  sandboxir::Context &Ctx;

  /// Helper for collecting sandboxir::SB{Store,Load}Instruction seeds.
  template <typename LoadOrStoreT>
  void insertMemSeed(LoadOrStoreT *LSI, const DataLayout &DL,
                     ScalarEvolution &SE);
  /// \Returns the number of SeedBundle groups for all seed types.
  /// This is to be used for limiting compilation time.
  unsigned totalNumSeedGroups() const {
    return StoreSeeds.size() + LoadSeeds.size() + NonStoreSeeds.size();
  }

public:
  SeedCollector(sandboxir::BasicBlock *SBBB, const DataLayout &DL,
                ScalarEvolution &SE);
  ~SeedCollector();

  iterator_range<MemSeedContainer::iterator> getStoreSeeds() {
    return {StoreSeeds.begin(), StoreSeeds.end()};
  }
  iterator_range<MemSeedContainer::iterator> getLoadSeeds() {
    return {LoadSeeds.begin(), LoadSeeds.end()};
  }
  iterator_range<SeedContainer::iterator> getNonMemSeeds() {
    return {NonStoreSeeds.begin(), NonStoreSeeds.end()};
  }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  void dump() const;
#endif
};

} // namespace sandboxir
} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SEEDCOLLECTOR_H
