//===--- AMDGPUResourceTracker.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Resource-based dependency tracking for AMDGPU wait insertion.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPURESOURCETRACKER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPURESOURCETRACKER_H

#include "AMDGPUCounters.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

class AMDGPUTestBase_ResourceTracker_TrackRegisterDependencies_Test;
class AMDGPUTestBase_ResourceTracker_StoreRegDefNoRawHazard_Test;
class AMDGPUTestBase_ResourceTracker_LoadImplicitTupleDefNoHazard_Test;
class AMDGPUTestBase_ResourceTracker_TrackMultipleLoadsToSameCounter_Test;
class AMDGPUTestBase_ResourceTracker_IsNonZeroWaitLegal_Test;
class AMDGPUTestBase_ResourceTracker_IsNonZeroWaitLegal_DsCnt_Test;
class AMDGPUTestBase_ResourceTracker_GetWaitFor_MixedDsTypes_Test;
class AMDGPUTestBase_ResourceTracker_NeedsFlatEarlyCompletionWorkaround_Test;
class AMDGPUResourceTrackerTest_InstrBuffer_Test;
class AMDGPUResourceTrackerTest_InstrBuffer_Merge_Test;
class AMDGPUResourceTrackerTest_AsyncBuffer_Test;
class AMDGPUResourceTrackerTest_AsyncBuffer_Merge_Test;

namespace llvm {

class MachineInstr;
class SIInstrInfo;
class AAResults;

namespace AMDGPU {

/// Wraps a MachineInstr tracked by a hardware counter. TrackedInstr objects
/// populate the Counter. The default constructor creates an "unknown" entry
/// representing outstanding operations from outside this function (e.g. a
/// caller's stores at the entry of a non-entry function).
class TrackedInstr {
  PointerIntPair<MachineInstr *, 1, bool> Value;

  // Helper for DenseMapInfo Empty/Tombstone.
  static TrackedInstr fromOpaqueValue(void *V) {
    TrackedInstr TI(nullptr);
    TI.Value = PointerIntPair<MachineInstr *, 1, bool>::getFromOpaqueValue(V);
    return TI;
  }
  friend struct llvm::DenseMapInfo<TrackedInstr>; // For fromOpaqueValue()

public:
  /// Creates a tracked instruction wrapping \p MI. A nullptr \p MI creates an
  /// unknown entry representing outstanding operations from outside this
  /// function.
  TrackedInstr(MachineInstr *MI) : Value(MI, MI == nullptr) {}

  MachineInstr *getMI() const { return Value.getPointer(); }
  bool isUnknown() const { return Value.getInt(); }

  bool operator==(const TrackedInstr &O) const {
    return Value.getOpaqueValue() == O.Value.getOpaqueValue();
  }
  bool operator!=(const TrackedInstr &O) const { return !(*this == O); }

  uintptr_t getOpaqueValue() const {
    return reinterpret_cast<uintptr_t>(Value.getOpaqueValue());
  }

#ifndef NDEBUG
  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  friend raw_ostream &operator<<(raw_ostream &OS, const TrackedInstr &TI) {
    TI.print(OS);
    return OS;
  }
#endif
};

} // namespace AMDGPU

} // namespace llvm

template <> struct llvm::DenseMapInfo<llvm::AMDGPU::TrackedInstr> {
  static inline llvm::AMDGPU::TrackedInstr getEmptyKey() {
    return llvm::AMDGPU::TrackedInstr::fromOpaqueValue(
        reinterpret_cast<void *>(~0lu));
  }
  static inline llvm::AMDGPU::TrackedInstr getTombstoneKey() {
    return llvm::AMDGPU::TrackedInstr::fromOpaqueValue(
        reinterpret_cast<void *>(~0lu - 1));
  }
  static unsigned getHashValue(const llvm::AMDGPU::TrackedInstr &V) {
    return llvm::DenseMapInfo<uintptr_t>::getHashValue(V.getOpaqueValue());
  }
  static bool isEqual(const llvm::AMDGPU::TrackedInstr &LHS,
                      const llvm::AMDGPU::TrackedInstr &RHS) {
    return LHS == RHS;
  }
};


namespace llvm {
namespace AMDGPU {

class InstrBufferBase {
protected:
  /// Hash for constant-time inequality check.
  /// The format of this hash: first 32-bits contain the instruction count and
  /// the rest contain the sum of the 32-bit truncated instruction pointers.
  /// This format allows us to quickly update the hash when we update the
  /// buffer.
  uint64_t Hash = 0;

  enum class Action {
    Add,
    Remove,
  };
  template <Action ActionT>
  uint64_t getHash(TrackedInstr TI) const {
    uint32_t OldMIHashSum = Hash & 0xffffffff;
    uint32_t MIHash = static_cast<uint32_t>(TI.getOpaqueValue());
    uint32_t MIHashSum = ActionT == Action::Add ? OldMIHashSum + MIHash
                                                : OldMIHashSum - MIHash;
    uint32_t OldCount = Hash >> 32;
    uint32_t NewCount =
        ActionT == Action::Add ? OldCount + 1 : OldCount - 1;
    return (static_cast<uint64_t>(NewCount) << 32) | MIHashSum;
  }

public:
  enum SubclassID {
    InstrBufferID,
    AsyncBufferID,
  };
  virtual ~InstrBufferBase() = default;

protected:
  SubclassID ID;

public:
  explicit InstrBufferBase(SubclassID ID) : ID(ID) {}
  SubclassID getID() const { return ID; }
  /// Insert \p MIs at the end of the buffer (all at same index).
  virtual void pushBack(ArrayRef<MachineInstr *> MIs) = 0;
  /// The largest index in the buffer.
  virtual unsigned getTopIndex() const = 0;
  /// Returns the index of \p MI in the buffer in constant time.
  virtual unsigned getIndex(MachineInstr *MI) const = 0;
  /// Removes the instructions that correspond to the top \p NumIndices.
  virtual void popFront(unsigned NumIndices = 1) = 0;
  /// Removes every instruction satisfying \p Pred, leaving the positions of the
  /// surviving instructions unchanged (so their wait values are preserved).
  /// Removing an instruction in the middle leaves a gap, which is an already-
  /// supported state. Leading and trailing gaps are trimmed so that empty() and
  /// getTopIndex() remain accurate.
  virtual void removeIf(function_ref<bool(const TrackedInstr &)> Pred) = 0;

  /// Two buffers are equal if they contain the same instructions at the same
  /// user-visible indices. Internal indices may differ due to different
  /// push/pop histories.
  virtual bool operator==(const InstrBufferBase &Other) const = 0;
  bool operator!=(const InstrBufferBase &Other) const {
    return !(*this == Other);
  }
  /// Merge \p Other into this buffer. Instructions are placed at positions
  /// that preserve their wait values (distance from end). This ensures that
  /// after merging counters from different paths, wait computations remain
  /// correct. If the same instruction exists in both buffers, the position
  /// giving the lower wait value (more conservative) is kept.
  ///
  /// Example: one path issues 3 instructions, other issues 2. Positions are
  /// remapped to preserve wait values:
  /// Merging [A, B, C] (TopIndex=3) with [D, E] (TopIndex=2):
  ///   - Before: D at pos 0, wait = 2-1-0 = 1
  ///   - After:  D at pos 1, wait = 3-1-1 = 1 (preserved)
  ///   - Before: E at pos 1, wait = 2-1-1 = 0
  ///   - After:  E at pos 2, wait = 3-1-2 = 0 (preserved)
  ///   - Result: [A, {B,D}, {C,E}]
  ///
  /// Complexity: O(N) where N is the number of instructions in Other.
  virtual void merge(const InstrBufferBase &Other) = 0;

  bool empty() const { return getTopIndex() == 0; }

  /// Returns the total number of instructions in the buffer.
  virtual unsigned numInstrs() const = 0;

  /// Returns true if \p MI is in the buffer.
  virtual bool contains(MachineInstr *MI) const = 0;

  /// Returns true if the buffer contains an unknown entry.
  virtual bool hasUnknown() const = 0;

  /// Returns all entries in the buffer (iteration order is unspecified).
  // TODO: Ideally this should return an iterator range to avoid copies.
  virtual SmallVector<TrackedInstr> instrsUnordered() const = 0;

  /// Returns the most recently inserted entries, or an empty set if empty.
  virtual SmallDenseSet<TrackedInstr, 2> back() const = 0;

  /// Returns the Nth set of entries from the end, or an empty set if N >=
  /// size() or if the index is empty (gap from duplicate pushBack).
  /// N=0 returns the most recent index.
  virtual const SmallDenseSet<TrackedInstr, 2> &getNthFromEnd(unsigned N) const = 0;

  virtual void clear() = 0;

  /// The number of entries/slots in the buffer. This could be lower than the
  /// number of instructions because there could be more than one instruction
  /// per entry.
  virtual unsigned getDepth() const = 0;

  /// Iterate over instructions in increasing index order.
  /// De-referencing the iterator returns the set of instructions at that index,
  /// or an empty set if the index is a gap (from duplicate pushBack).
  class iterator {
  public:
    /// This is the actual implementation of the iterator and should be
    /// overriden by InstrBufferBase subclasses.
    class Base {
    public:
      virtual ~Base() {}
      virtual SmallDenseSet<TrackedInstr, 2> deref() const = 0;
      virtual void advance() = 0;
      virtual bool equal(const Base &Other) const = 0;
    };
  private:
    std::unique_ptr<Base> BaseIt;

  public:
    iterator(std::unique_ptr<Base> BaseIt) : BaseIt(std::move(BaseIt)) {}

    using iterator_category = std::forward_iterator_tag;
    using value_type = TrackedInstr;
    using difference_type = std::ptrdiff_t;
    using pointer = TrackedInstr *;
    using reference = TrackedInstr &;

    SmallDenseSet<TrackedInstr, 2> operator*() const { return BaseIt->deref(); }
    iterator &operator++() {
      BaseIt->advance();
      return *this;
    }
    bool operator==(const iterator &Other) const {
      return BaseIt->equal(*Other.BaseIt);
    }
    bool operator!=(const iterator &Other) const { return !(*this == Other); }
  };
  virtual iterator begin() const = 0;
  virtual iterator end() const = 0;
#ifndef NDEBUG
  virtual void print(raw_ostream &OS, bool Verbose = true) const = 0;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

/// Common members across both InstrBuffer and AsyncBuffer.
class InstrBufferCommon {
protected:
  /// Maps each instruction to the set of internal indices. Unlike InstrBuffer
  /// instructions are not de-duplicated, we can have more than one index per
  /// instruction.
  DenseMap<unsigned, SmallDenseSet<TrackedInstr, 2>> IdxToInstrMap;
  /// To avoid an O(N) update of indices on pop() we use immutable indices that
  /// keep increasing. So BottomIdx and TopIdx refer to these always-increasing
  /// indices. These are internal
  unsigned BottomIdxInternal = 0;
  unsigned TopIdxInternal = 0;

  unsigned getInternalIdxFromExternal(unsigned Idx) const {
    assert(Idx <= TopIdxInternal - BottomIdxInternal && "Out of bounds!");
    return TopIdxInternal - Idx - 1;
  }

  /// The external index is counting in the opposite way compared to the
  /// internal one, i.e., the latest entry is 0.
  unsigned getExternalIdxFromInternal(unsigned InternalIdx) const {
    return TopIdxInternal - InternalIdx - 1;
  }
};

/// A double ended queue style data structure that holds an ordered set of
/// instructions with:
/// - insertion:        O(1)
/// - deletion:         O(1)
/// - MI->index lookup: O(1)
/// - index->MI lookup: O(1)
/// - removeIf:         O(N) where N = total instructions in the buffer
/// - inequality check: O(N) worst case, O(1) usual
/// - equality check:   O(N)
/// - merge:            O(N)
/// - contains:         O(1)
/// - clear:            O(1)
///
/// To achieve fast insertion/deletion/lookup, the buffer is implemented using a
/// map from internal (internal) indices to instructions. New instructions are
/// mapped to higher index values.
class InstrBuffer final : public InstrBufferBase, public InstrBufferCommon {
  friend class ::AMDGPUResourceTrackerTest_InstrBuffer_Test;
  friend class ::AMDGPUResourceTrackerTest_InstrBuffer_Merge_Test;

  /// Maps internal index to the set of instructions at that index.
  DenseMap<TrackedInstr, unsigned> InstrToIdxMap;

public:
  InstrBuffer() : InstrBufferBase(SubclassID::InstrBufferID) {}
  /// For isa/dyn_cast.
  static bool classof(const InstrBufferBase *From) {
    return From->getID() == SubclassID::InstrBufferID;
  }
  /// Insert \p MIs at the end of the buffer (all at same index), or move them
  /// to the end if already present. A nullptr MIs inserts an unknown entry
  /// representing outstanding operations from outside this function. Examples:
  ///   [A, B] -> pushBack({C})    -> [A, B, C]
  ///   [A, B] -> pushBack({C, D}) -> [A, B, {C, D}]
  ///   [A, B] -> pushBack({B})    -> [A, B] (no-op)
  ///   [A, B] -> pushBack({A})    -> [B, A] (swap)
  ///   [A, B, C] -> pushBack({B}) -> [A, C, B] (swap)
  ///   [A, B, C] -> pushBack({A}) -> [C, B, A] (swap)
  void pushBack(ArrayRef<MachineInstr *> MIs) override;
  unsigned getTopIndex() const override {
    return TopIdxInternal - BottomIdxInternal;
  }
  /// Returns the index of \p MI in the buffer in constant time.
  unsigned getIndex(MachineInstr *MI) const override {
    return InstrToIdxMap.at(MI) - BottomIdxInternal;
  }
  /// Removes the instructions that correspond to the top \p NumIndices.
  void popFront(unsigned NumIndices = 1) override;
  /// Removes every instruction satisfying \p Pred, leaving the positions of the
  /// surviving instructions unchanged (so their wait values are preserved).
  /// Removing an instruction in the middle leaves a gap, which is an already-
  /// supported state. Leading and trailing gaps are trimmed so that empty() and
  /// getTopIndex() remain accurate.
  void removeIf(function_ref<bool(const TrackedInstr &)> Pred) override;

  /// Two buffers are equal if they contain the same instructions at the same
  /// user-visible indices. Internal indices may differ due to different
  /// push/pop histories.
  bool operator==(const InstrBufferBase &Other) const override;
  /// Merge \p Other into this buffer. Instructions are placed at positions
  /// that preserve their wait values (distance from end). This ensures that
  /// after merging counters from different paths, wait computations remain
  /// correct. If the same instruction exists in both buffers, the position
  /// giving the lower wait value (more conservative) is kept.
  ///
  /// Example: one path issues 3 instructions, other issues 2. Positions are
  /// remapped to preserve wait values:
  /// Merging [A, B, C] (TopIndex=3) with [D, E] (TopIndex=2):
  ///   - Before: D at pos 0, wait = 2-1-0 = 1
  ///   - After:  D at pos 1, wait = 3-1-1 = 1 (preserved)
  ///   - Before: E at pos 1, wait = 2-1-1 = 0
  ///   - After:  E at pos 2, wait = 3-1-2 = 0 (preserved)
  ///   - Result: [A, {B,D}, {C,E}]
  ///
  /// Complexity: O(N) where N is the number of instructions in Other.
  void merge(const InstrBufferBase &Other) override;

  /// Returns the total number of instructions in the buffer.
  unsigned numInstrs() const override { return InstrToIdxMap.size(); }

  /// Returns true if \p MI is in the buffer.
  bool contains(MachineInstr *MI) const override {
    assert(MI && "Use hasIncomingUnknown() instead");
    return InstrToIdxMap.count(MI);
  }

  /// Returns true if the buffer contains an unknown entry.
  bool hasUnknown() const override { return InstrToIdxMap.count(nullptr); }

  /// Returns all entries in the buffer (iteration order is unspecified).
  SmallVector<TrackedInstr> instrsUnordered() const override {
    return SmallVector<TrackedInstr>(make_first_range(InstrToIdxMap));
  }

  /// Returns the most recently inserted entries, or an empty set if empty.
  SmallDenseSet<TrackedInstr, 2> back() const override;

  /// Returns the Nth set of entries from the end, or an empty set if N >=
  /// size() or if the index is empty (gap from duplicate pushBack).
  /// N=0 returns the most recent index.
  const SmallDenseSet<TrackedInstr, 2> &
  getNthFromEnd(unsigned N) const override;

  void clear() override;

  unsigned getDepth() const override {
    return TopIdxInternal - BottomIdxInternal;
  }

private:
  /// The implementation of iterator::Base for InstrBuffer.
  class IteratorImpl : public iterator::Base {
    const InstrBuffer *Buf;
    unsigned Idx;

  public:
    IteratorImpl(const InstrBuffer *Buf, unsigned Idx) : Buf(Buf), Idx(Idx) {}
    SmallDenseSet<TrackedInstr, 2> deref() const override {
      auto It = Buf->IdxToInstrMap.find(Idx);
      if (It == Buf->IdxToInstrMap.end())
        return {};
      return It->second;
    }
    void advance() override { ++Idx; }
    bool equal(const Base &OtherBase) const override {
      const auto &Other = static_cast<const IteratorImpl &>(OtherBase);
      return Idx == Other.Idx;
    }
  };

public:
  iterator begin() const override {
    return iterator(std::make_unique<IteratorImpl>(this, BottomIdxInternal));
  }
  iterator end() const override {
    return iterator(std::make_unique<IteratorImpl>(this, TopIdxInternal));
  }
#ifndef NDEBUG
  void print(raw_ostream &OS, bool Verbose = true) const override;
#endif
};

/// A buffer that allows the same instruction to appear at multiple positions
/// (no deduplication). Used for async DMA counters (AsyncCnt, TensorCnt) where
/// the same instruction can be tracked multiple times (once per loop iteration).
///
/// Unlike InstrBuffer, pushBack() always creates a new entry at the top even
/// if the instruction already exists at an earlier position. Uses the same
/// immutable-index scheme as InstrBuffer for O(1) popFront.
///
/// - insertion:        O(1)
/// - deletion:         O(1)
/// - MI->index lookup: O(1)
/// - index->MI lookup: O(1)
/// - removeIf:         O(N)
/// - inequality check: O(N) worst case, O(1) usual
/// - equality check:   O(N)
/// - merge:            O(N)
/// - contains:         O(1)
/// - hasUnknown:       O(1)
/// - clear:            O(1)
class AsyncBuffer final : public InstrBufferBase, public InstrBufferCommon {
  friend class ::AMDGPUResourceTrackerTest_InstrBuffer_Test;
  friend class ::AMDGPUResourceTrackerTest_AsyncBuffer_Test;
  friend class ::AMDGPUResourceTrackerTest_AsyncBuffer_Merge_Test;

  /// Maps each instruction to the set of internal indices. Unlike InstrBuffer
  /// instructions are not de-duplicated, we can have more than one index per
  /// instruction.
  DenseMap<TrackedInstr, SmallDenseSet<unsigned, 2>> InstrToIdxsMap;

  /// Reference to the global sequence number counter owned by AllCounters.
  /// Incremented on each pushBack to provide cross-counter ordering.
  uint64_t &GlobalSeqNum;

  /// Maps the sequence number of the marked entries to all internal indices
  /// that correspond to this sequence number. Multiple indices are the result
  /// of merges.
  DenseMap<uint64_t, SmallSet<unsigned, 1>> SeqNumToInternalIdxs;
  /// Inverse of SeqNumToInternalIdx: internal index to seqnums.
  DenseMap<unsigned, SmallSet<uint64_t, 1>> InternalIdxToSeqNums;
  /// Inclusive range of mark-event sequence numbers currently in this buffer.
  /// Maintained by pushBack(), merge(), popFront(), and clear().
  uint64_t MinMarkedSeqNum = std::numeric_limits<uint64_t>::max();
  uint64_t MaxMarkedSeqNum = 0;
  /// Set by merge() to indicate a CFG merge occurred. Cleared by pushBack().
  /// Used by getWaitForNthMarked() to clamp N to MaxAsyncMarks-1, replicating
  /// the old pass's behavior of capping the AsyncMarks array at merge points.
  // TODO: This is a workaround, remove it once we migrate to the new pass.
  bool MergedAndCapped = false;

  static const SmallDenseSet<TrackedInstr, 2> EmptySet;

public:
  explicit AsyncBuffer(uint64_t &GlobalSeqNum)
      : InstrBufferBase(SubclassID::AsyncBufferID),
        GlobalSeqNum(GlobalSeqNum) {}

  static bool classof(const InstrBufferBase *From) {
    return From->getID() == SubclassID::AsyncBufferID;
  }

  void pushBack(ArrayRef<MachineInstr *> MIs) override;
  unsigned getTopIndex() const override;
  // Returns the maximum (i.e., latest) index of MI.
  // Note: This is currently linear to the number of duplicate MIs.
  unsigned getIndex(MachineInstr *MI) const override;
  void popFront(unsigned NumIndices = 1) override;
  void removeIf(function_ref<bool(const TrackedInstr &)> Pred) override;
  bool operator==(const InstrBufferBase &OtherBase) const override;
  void merge(const InstrBufferBase &OtherBase) override;
  unsigned numInstrs() const override;
  bool contains(MachineInstr *MI) const override;
  bool hasUnknown() const override;
  SmallVector<TrackedInstr> instrsUnordered() const override;
  SmallDenseSet<TrackedInstr, 2> back() const override;
  const SmallDenseSet<TrackedInstr, 2> &getNthFromEnd(unsigned N) const override;
  void clear() override;
  unsigned getDepth() const override {
    return TopIdxInternal - BottomIdxInternal;
  }
  iterator begin() const override;
  iterator end() const override;

  /// Returns the mark-event sequence number for slot \p UserIdx (0=oldest),
  /// or nullopt if that slot is not marked.
  /// WARNING: For debug prints/unit tests only!
  const SmallSet<uint64_t, 1> &getSeqNumsForIndex(unsigned UserIdx) const {
    static const SmallSet<uint64_t, 1> Empty;
    auto It = InternalIdxToSeqNums.find(BottomIdxInternal + UserIdx);
    if (It == InternalIdxToSeqNums.end())
      return Empty;
    return It->second;
  }

  /// Returns true if \p MI is immediately followed by an ASYNCMARK instruction.
  static bool isAsyncMarked(const MachineInstr *MI);

  /// Returns the most recently pushed marked instruction (the one whose slot
  /// has the highest mark-event sequence number), or nullptr if none.
  MachineInstr *getLastMarkedInstr() const;

  /// Returns the wait value for the Nth marked entry counting from the top
  /// (0 = most recently marked). Unmarked entries are skipped. N is clamped
  /// to MaxAsyncMarks-1 unless MaxAsyncMarks is 0 (unlimited). Returns nullopt
  /// if no marked entries exist.
  std::optional<unsigned> getWaitForNthMarked(unsigned N) const;

  /// Returns the ASYNCMARK instruction that marks \p MI — the next ASYNCMARK
  /// reachable from \p MI in program order, skipping async DMA instructions of
  /// a different counter type. \p MI must be in MarkedInstrs (i.e. previously
  /// confirmed by isAsyncMarked), so the ASYNCMARK is guaranteed to exist.
  static MachineInstr *getAsyncMarkFor(MachineInstr *MI);

  /// Finds the Nth most recently pushed marked entry across multiple buffers,
  /// ordered by slot sequence number descending. Returns a vector of
  /// {buffer_index, index_in_buffer} entries (may include co-marked entries
  /// from other buffers belonging to the same ASYNCMARK event), or an empty
  /// vector if fewer than N+1 marked entries exist across all buffers.
  static SmallVector<std::pair<unsigned, unsigned>>
  getNthMostRecentMarkedAmong(ArrayRef<AsyncBuffer *> Bufs, unsigned N);

#ifndef NDEBUG
  void print(raw_ostream &OS, bool Verbose = true) const override;
#endif

private:
  class IteratorImpl : public iterator::Base {
    const AsyncBuffer *Buf;
    unsigned Idx;

  public:
    IteratorImpl(const AsyncBuffer *Buf, unsigned Idx) : Buf(Buf), Idx(Idx) {}
    SmallDenseSet<TrackedInstr, 2> deref() const override;
    void advance() override { ++Idx; }
    bool equal(const Base &OtherBase) const override {
      return Idx == static_cast<const IteratorImpl &>(OtherBase).Idx;
    }
  };
};

class AllCounters;

/// Corresponds to a hardware counter and tracks instructions that are in the
/// process of writing to a resource (register or memory) and no wait
/// instruction for this resource has been issued.
class Counter {
  friend class AllCounters; // for default construction

public:
  enum class SubclassID { DedupCounter, NonDedupCounter };

  static bool classof(const Counter *C) {
    return C->getSubclassID() == SubclassID::DedupCounter;
  }
  virtual ~Counter() = default;

protected:
  SubclassID ID;
  CounterType T = LoadCnt();
  std::unique_ptr<InstrBufferBase> Instrs;
  // Maximum number of instructions to track. 0 means unlimited.
  const unsigned MaxSize = 0;
  // If true, drop the oldest entries once the counter overflows past MaxSize,
  // instead of keeping them and clamping the wait value. This is correct for a
  // counter that orders a producer's *source reads* (ExpCnt): an overflowed
  // entry's reads are complete, so it needs no wait. It is NOT correct for a
  // counter that orders a producer's *result* (the load counters): an overflowed
  // load's result may still be in flight, so those clamp instead (see
  // getWaitFor).
  const bool DropOnOverflow = false;

  Counter()
      : ID(SubclassID::DedupCounter), Instrs(std::make_unique<InstrBuffer>()) {}

  /// Constructor for subclasses — takes the SubclassID and defers buffer
  /// creation to the subclass.
  Counter(SubclassID ID, const CounterType &T, unsigned MaxSize,
          bool DropOnOverflow)
      : ID(ID), T(T), MaxSize(MaxSize), DropOnOverflow(DropOnOverflow) {}

public:
  SubclassID getSubclassID() const { return ID; }

  /// Constructor for non-async counters using InstrBuffer.
  Counter(const CounterType &T, unsigned MaxSize, bool DropOnOverflow)
      : Counter(SubclassID::DedupCounter, T, MaxSize, DropOnOverflow) {
    Instrs = std::make_unique<InstrBuffer>();
  }

  const CounterType &getType() const { return T; }
  StringLiteral getName() const { return T.getName(); }

  /// Insert all instructions in MIs into the hardware at a new entry, all using
  /// the same index.
  void insert(ArrayRef<MachineInstr *> MIs) {
    Instrs->pushBack(MIs);
    // See DropOnOverflow: trim the oldest entries so an overflowed producer
    // (e.g. an export past the ExpCnt depth) produces no wait.
    if (DropOnOverflow && MaxSize > 0 && Instrs->getTopIndex() > MaxSize)
      Instrs->popFront(Instrs->getTopIndex() - MaxSize);
  }

  unsigned size() const { return Instrs->numInstrs(); }
  bool empty() const { return Instrs->empty(); }

  /// Mark that the counter may have outstanding operations from outside this
  /// function (e.g. a caller's stores at the entry of a non-entry function).
  /// Inserts an unknown entry that behaves like a dummy instruction at the
  /// oldest position. A wait-for-zero clears it.
  void setIncomingUnknown() { Instrs->pushBack(nullptr); }
  /// Whether the counter has an unknown entry from outside this function.
  bool hasIncomingUnknown() const { return Instrs->hasUnknown(); }
  bool contains(MachineInstr *MI) const { return Instrs->contains(MI); }
  /// Returns all known MachineInstrs in the counter (excludes unknown entries).
  SmallVector<MachineInstr *> instrsUnordered() const {
    SmallVector<MachineInstr *> Result;
    for (TrackedInstr TI : Instrs->instrsUnordered())
      if (!TI.isUnknown())
        Result.push_back(TI.getMI());
    return Result;
  }
  /// Returns any instruction from the most recent index satisfying \p Pred,
  /// or nullptr if no instruction satisfies the predicate.
  template <typename PredT>
  MachineInstr *getLastInstrWhere(PredT Pred) const {
    for (unsigned N = 0; N < size(); ++N) {
      for (TrackedInstr TI : Instrs->getNthFromEnd(N)) {
        if (!TI.isUnknown() && Pred(*TI.getMI()))
          return TI.getMI();
      }
    }
    return nullptr;
  }

  /// Returns the set of instructions at the Nth index from the end.
  /// N=0 returns from the most recent index.
  ///
  /// With \p Saturate false, an out-of-range N returns the empty set.
  ///
  /// With \p Saturate true, an out-of-range N is clamped to the oldest position
  /// only when the counter is full (at MaxSize) - i.e. when older entries may
  /// have been dropped at the cap, so a request beyond the survivors can no
  /// longer be satisfied precisely and should fall back to the oldest. When the
  /// counter is not full no entries were dropped, so an out-of-range N is still
  /// empty. (Used by WAIT_ASYNCMARK to force a wait after a merge truncated the
  /// marks, mirroring the old pass's determineAsyncWait.) The position
  /// arithmetic stays inside the buffer.
  const SmallDenseSet<TrackedInstr, 2> &getNthFromEnd(unsigned N,
                                                      bool Saturate = false) const {
    unsigned NumPositions = Instrs->getTopIndex();
    if (Saturate && MaxSize > 0 && NumPositions == MaxSize && N >= NumPositions)
      N = NumPositions - 1;
    return Instrs->getNthFromEnd(N);
  }

  /// Apply a wait for this counter. Removes instructions that have completed
  /// (i.e., the oldest instructions up to the wait value).
  /// For example, if we have [I1, I2, I3] and wait for 1, I1 and I2 complete,
  /// leaving [I3].
  void applyWait(unsigned WaitValue) {
    unsigned Size = Instrs->getTopIndex();
    unsigned ToRemove = Size > WaitValue ? Size - WaitValue : 0;
    Instrs->popFront(ToRemove);
  }

  /// \returns the counter value we should wait for \p MI to have its results
  /// ready.
  std::optional<unsigned> getWaitFor(const MachineInstr &MI) const {
    if (!Instrs->contains(const_cast<MachineInstr *>(&MI)))
      return std::nullopt;
    // The counter that we need to wait for corresponds to the distance from
    // the end of the buffer.
    //
    //         Oldest   Newest Wait for: I1  I2  I3
    // Buffer:   I1            Counter:  0   -   -
    // Buffer:   I1,  I2,      Counter:  1   0   -
    // Buffer:   I1,  I2,  I3  Counter:  2   1   0
    //
    unsigned Idx = Instrs->getIndex(const_cast<MachineInstr *>(&MI));
    unsigned Wait = Instrs->getTopIndex() - 1 - Idx;
    // Clamp wait to MaxSize - 1 when the counter has overflowed. This matches
    // the old pass behavior where we wait for the oldest trackable instruction.
    if (MaxSize > 0 && Wait >= MaxSize)
      Wait = MaxSize - 1;
    return Wait;
  }

  unsigned getDepth() const { return Instrs->getDepth(); }

  /// Returns the depth capped at MaxSize for convergence checking.
  /// Non-deduplicating counters can grow with each iteration but are bounded by
  /// MaxSize. Capping lets the dataflow detect the fixed point once all slots
  /// are filled in.
  unsigned getConvergenceDepth() const {
    unsigned D = getDepth();
    return (MaxSize > 0 && D > MaxSize) ? MaxSize : D;
  }

  /// Merge another counter's state into this one.
  void merge(const Counter &Other) {
    assert(T == Other.T && "Merging counters of different types");
    Instrs->merge(*Other.Instrs);
  }

  /// Check if two counters have equivalent state.
  bool operator==(const Counter &Other) const {
    assert(T == Other.T && "Comparing counters of different types");
    return *Instrs == *Other.Instrs;
  }
  bool operator!=(const Counter &Other) const { return !(*this == Other); }

  void clear() { Instrs->clear(); }

  /// Remove every pending instruction satisfying \p Pred. Surviving
  /// instructions keep their wait values. Unknown entries are never removed.
  template <typename PredT> void removeIf(PredT Pred) {
    Instrs->removeIf([&](TrackedInstr TI) {
      return !TI.isUnknown() && Pred(TI.getMI());
    });
  }

  /// Check if the counter has mixed event types pending (e.g., both LDS and
  /// GDS operations). When mixed types are pending, the hardware may decrement
  /// the counter out-of-order, so wait values must be 0 to ensure correctness.
  bool hasMixedEventTypes(const SIInstrInfo &TII) const;

  /// Check if the counter has any pending FLAT instructions that can access
  /// both VMEM and LDS. On pre-GFX10 targets, such FLAT operations can report
  /// early completion (the counter decrements before the operation actually
  /// completes), so we must wait for 0 to ensure correctness.
  bool hasPendingFlat(const SIInstrInfo &TII) const;

  /// Check if this counter requires the FLAT early completion workaround.
  /// Pre-GFX10 FLAT can report early completion, so if any FLAT is pending
  /// on vmcnt/lgkmcnt, position-based waits are unsafe.
  bool needsFlatEarlyCompletionWorkaround(const GCNSubtarget &ST) const;

  /// Check if the counter has any pending VMEM instructions with a VmemType
  /// other than VMEM_NOSAMPLER. Used for Point Sample Acceleration check.
  bool hasNonNosamplerVmemType() const;

  /// Returns the counters that are affected by the given instruction.
  /// This is a static method that determines which hardware counters an
  /// instruction will increment when it is issued.
  static SmallVector<CounterType, 4>
  getCountersForInstr(const MachineInstr &MI, const GCNSubtarget &ST,
                      SchedulingMode SchedMode = SchedulingMode::NoExpert);

  /// Returns true if we can rely on a non-zero wait for the dependency between
  /// \p SrcMI and \p DstMI. This is legal only if all instructions tracked by
  /// this Counter are guaranteed to update the hardware counter in execution
  /// order, which matches the order they are listed in the Counter data
  /// structure.
  /// This applies to VMEM instructions of the same VmemType (NOSAMPLER,
  /// SAMPLER, BVH) when the subtarget has the VmemWriteVgprInOrder feature.
  /// FLAT instructions (which can access either global memory or LDS) don't
  /// participate in VmemType tracking, but their global memory accesses are
  /// in-order with other VMEM instructions for vmcnt purposes.
  bool isNonZeroWaitLegal(const MachineInstr &SrcMI, const MachineInstr &DstMI,
                          const GCNSubtarget &ST, SchedulingMode SchedMode) const;

#ifndef NDEBUG
  virtual void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  friend raw_ostream &operator<<(raw_ostream &OS, const Counter &Ctr) {
    Ctr.print(OS);
    return OS;
  }
#endif
};

/// Specialization of Counter for async DMA counters (AsyncCnt, TensorCnt).
/// Uses AsyncBuffer to allow the same instruction at multiple positions
/// (one per loop iteration) and supports getWaitForNthMarked() for
/// WAIT_ASYNCMARK lowering.
class AsyncCounter : public Counter {
public:
  AsyncCounter(const CounterType &T, unsigned MaxSize, uint64_t &GlobalSeqNum)
      : Counter(SubclassID::NonDedupCounter, T, MaxSize,
                /*DropOnOverflow=*/false) {
    Instrs = std::make_unique<AsyncBuffer>(GlobalSeqNum);
  }
  virtual ~AsyncCounter() = default;


  static bool classof(const Counter *C) {
    return C->getSubclassID() == SubclassID::NonDedupCounter;
  }

  /// Returns the most recently pushed marked instruction, or nullptr if none.
  MachineInstr *getLastMarkedInstr() const {
    return cast<AsyncBuffer>(Instrs.get())->getLastMarkedInstr();
  }

  /// Returns the wait value for the Nth marked entry (0=most recent).
  std::optional<unsigned> getWaitForNthMarked(unsigned N) const {
    return cast<AsyncBuffer>(Instrs.get())->getWaitForNthMarked(N);
  }

  /// Finds the Nth most recently pushed marked entry across \p Counters,
  /// ordered by slot sequence number. Returns a vector of {AsyncCounter*,
  /// index_in_counter (FromEnd)} entries, or an empty vector if not found.
  static SmallVector<std::pair<AsyncCounter *, unsigned>>
  getNthMostRecentMarkedAmong(ArrayRef<AsyncCounter *> Counters, unsigned N);

#ifndef NDEBUG
  void print(raw_ostream &OS) const override;
 #endif
};

/// Associates a pending MachineInstr with the Counter tracking it and the
/// wait value needed for it to complete.
struct WaitDescriptor {
  MachineInstr *MI;
  CounterType Cntr;
  unsigned Wait;
  WaitDescriptor(MachineInstr *MI, CounterType Cntr, unsigned Wait)
      : MI(MI), Cntr(Cntr), Wait(Wait) {}
  WaitDescriptor(CounterType Cntr, unsigned Wait)
      : MI(nullptr), Cntr(Cntr), Wait(Wait) {}
  bool operator==(const WaitDescriptor &Other) const {
    return MI == Other.MI && Cntr == Other.Cntr && Wait == Other.Wait;
  }
  bool operator!=(const WaitDescriptor &Other) const {
    return !(*this == Other);
  }
#ifndef NDEBUG
  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  friend raw_ostream &operator<<(raw_ostream &OS, const WaitDescriptor &CAW) {
    CAW.print(OS);
    return OS;
  }
#endif
};

class WaitDescriptors {
  SmallVector<WaitDescriptor, 4> Vec;

  /// Insert a new entry or update an existing one with the same counter ID.
  /// If an entry for the same counter already exists, keep the minimum wait
  /// value. Otherwise, insert at the sorted position to maintain counter ID
  /// order. This eliminates the need for a separate pruning step.
  void insertOrUpdate(MachineInstr *MI, CounterType Cntr, unsigned Wait);

public:
  WaitDescriptors() = default;
  WaitDescriptors(std::initializer_list<WaitDescriptor> IL) {
    for (const auto &CAW : IL)
      insert(CAW);
  }

  /// \Returns the wait associated to \p Cntr or nullptr if not found.
  WaitDescriptor *get(CounterType Cntr);

  WaitDescriptor *get(CounterType Cntr) const {
    return const_cast<WaitDescriptors *>(this)->get(Cntr);
  }

  void insert(const WaitDescriptor &V) {
    insertOrUpdate(V.MI, V.Cntr, V.Wait);
  }
  void emplace(MachineInstr *MI, CounterType Cntr, unsigned Wait) {
    insertOrUpdate(MI, Cntr, Wait);
  }
  void emplace(CounterType Cntr, unsigned Wait) {
    insertOrUpdate(nullptr, Cntr, Wait);
  }
  void reserve(size_t N) { Vec.reserve(N); }
  bool empty() const { return Vec.empty(); }
  size_t size() const { return Vec.size(); }

  bool operator==(const WaitDescriptors &Other) const {
    return Vec == Other.Vec;
  }
  bool operator!=(const WaitDescriptors &Other) const {
    return !(*this == Other);
  }

  auto begin() { return Vec.begin(); }
  auto end() { return Vec.end(); }
  auto begin() const { return Vec.begin(); }
  auto end() const { return Vec.end(); }

  template <typename PredT> void erase_if(PredT P) { llvm::erase_if(Vec, P); }

#ifndef NDEBUG
  void print(raw_ostream &OS) const {
    for (const auto &Entry : Vec)
      OS << Entry;
  }
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const WaitDescriptors &CAW) {
    CAW.print(OS);
    return OS;
  }
  LLVM_DUMP_METHOD void dump() const;
#endif
};

/// A thin wrapper over the counters.
class AllCounters {
  const InstCounters *ICounters = nullptr;
  DenseMap<CounterType, std::unique_ptr<Counter>> Map;
  /// A shared global sequence number, shared by the counters. Note that this is
  /// zero in each new BB because each BB gets its own AllCounters object.
  uint64_t GlobalSeqNum = 0;

public:
  AllCounters(const GCNSubtarget &ST, SchedulingMode SchedMode,
              const InstCounters *ICounters);

  const Counter &operator[](const CounterType &T) const {
    auto It = Map.find(T);
    assert(It != Map.end() && "Counter type not available on target!");
    return *It->second;
  }
  Counter &operator[](const CounterType &T) {
    auto It = Map.find(T);
    assert(It != Map.end() && "Counter type not available on target!");
    return *It->second;
  }

  /// Iterate over all counters.
  auto begin() const { return Map.begin(); }
  auto end() const { return Map.end(); }

  WaitDescriptors get(MachineInstr &MI) const;

  /// Returns the waits for all counters.
  WaitDescriptors get() const;

  /// Merge another AllCounters's state into this one. Used for dataflow
  /// analysis to combine counter states from multiple predecessor blocks.
  void merge(const AllCounters &Other);

  /// Clear all counters. Used to reset state at the start of dataflow
  /// iteration for a block.
  void clear();

#ifndef NDEBUG
  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

/// Types of register dependencies.
enum class RegDepType {
  RAW, ///> Read-after-write: DstMI reads a register written by SrcMI.
  WAR, ///> Write-after-read: DstMI writes a register read by SrcMI.
  WAW, ///> Write-after-write: DstMI writes a register also written by SrcMI.
  RAR, ///> Read-after-read: DstMI reads a register read by SrcMI.
};

#ifndef NDEBUG
StringLiteral regDepTypeToStr(RegDepType Dep);
#endif

/// Tracks the resources modified by each instruction.
/// The API allows you to query which Instruction last wrote to the resource and
/// also the corresponding hardware counter and counter value that we need to
/// wait for.
class ResourceTracker {
public:
  /// How an instruction accesses a register unit.
  enum class RegAccessType {
    Def, // Instruction defines (writes) the register.
    Use  // Instruction uses (reads) the register.
  };

  /// Determine dependency type from access types:
  ///   DstAccess=Def, SrcAccess=Def → WAW (both write)
  ///   DstAccess=Def, SrcAccess=Use → WAR (src reads, dst writes)
  ///   DstAccess=Use, SrcAccess=Def → RAW (src writes, dst reads)
  ///   DstAccess=Use, SrcAccess=Use → no hazard (both read)
  static RegDepType getDepType(RegAccessType SrcAccess,
                               RegAccessType DstAccess) {
    if (DstAccess == RegAccessType::Use)
      return SrcAccess == RegAccessType::Def ? RegDepType::RAW
                                             : RegDepType::RAR;
    assert(DstAccess == RegAccessType::Def && "Expected DEF");
    return SrcAccess == RegAccessType::Def ? RegDepType::WAW : RegDepType::WAR;
  }

private:
  InstCounters ICounters;
  const GCNSubtarget *ST = nullptr;
  const SIInstrInfo *TII = nullptr;
  AAResults *AA = nullptr;
  SchedulingMode SchedMode = SchedulingMode::NoExpert;
  AllCounters Counters;

  /// Tracks an instruction, how it accesses a register unit, and the hardware
  /// counter that orders that access.
  ///
  /// For a DEF, Counter is a result counter (used for RAW/WAW hazards). For a
  /// USE, Counter is the WAR counter that orders the read: XCnt for an address
  /// read, ExpCnt for an EXP/LDSDIR VGPR source or a gfx6 store/atomic data
  /// operand. A read that no counter orders is not a hazard and is not recorded.
  ///
  /// A register unit keeps the latest accessor for each distinct (Access,
  /// Counter), so accesses ordered by different counters do not overwrite each
  /// other (e.g. a return atomic's result DEF on VmCnt and its data-source USE
  /// on ExpCnt, both on the same register).
  struct RegUnitInfo {
    MachineInstr *MI;
    RegAccessType Access;
    CounterType Counter;
    bool operator==(const RegUnitInfo &Other) const {
      return MI == Other.MI && Access == Other.Access && Counter == Other.Counter;
    }
  };

  /// Map register units to instructions that access them. We use MCRegUnit
  /// instead of Register because a super-register access (e.g.,
  /// vgpr0_vgpr1_vgpr2_vgpr3) should be found when querying for any of its
  /// sub-registers (e.g., vgpr0). After merging from multiple predecessors,
  /// a register unit may have multiple potential accessors from different
  /// blocks.
  DenseMap<MCRegUnit, SmallVector<RegUnitInfo, 2>> RegUnitToInstrsMap;

  friend class InsertWaitcnt;

  /// Returns the waits needed for the given register without any in-order
  /// completion optimizations. This is only for unit testing.
  WaitDescriptors getWaitForReg(Register Reg) const;
  friend class ::AMDGPUTestBase_ResourceTracker_TrackRegisterDependencies_Test;
  friend class ::AMDGPUTestBase_ResourceTracker_StoreRegDefNoRawHazard_Test;
  friend class ::AMDGPUTestBase_ResourceTracker_LoadImplicitTupleDefNoHazard_Test;
  friend class ::AMDGPUTestBase_ResourceTracker_TrackMultipleLoadsToSameCounter_Test;
  friend class ::AMDGPUTestBase_ResourceTracker_IsNonZeroWaitLegal_Test;
  friend class ::AMDGPUTestBase_ResourceTracker_IsNonZeroWaitLegal_DsCnt_Test;
  friend class ::AMDGPUTestBase_ResourceTracker_GetWaitFor_MixedDsTypes_Test;
  friend class ::AMDGPUTestBase_ResourceTracker_NeedsFlatEarlyCompletionWorkaround_Test;

  /// For some instruction the hardware already applies waits automatically.
  void applyHWImpliedWaits(MachineInstr &MI);

  /// There is a clear distinction in the responsibilites of caller and callee
  /// regarding waits.
  void drainCountersPerCallingConvention(MachineInstr &MI);

public:
  ResourceTracker(const GCNSubtarget *ST, AAResults *AA,
                  SchedulingMode SchedMode)
      : ST(ST), TII(ST->getInstrInfo()), AA(AA), SchedMode(SchedMode),
        Counters(*ST, SchedMode, &ICounters) {}

  /// \Returns the async counters for the current target.
  SmallVector<AsyncCounter *, 2> getAsyncCounters();

  /// Go over all resources written by \p MI and populate the data-structures.
  void track(MachineInstr &MI);

  /// Detect if MI triggers hardware implicit XCnt sync between SMEM and VMEM.
  /// Returns true if MI is SMEM with pending VMEM, or VMEM with pending SMEM.
  bool impliesXcntSync(const MachineInstr &MI) const;

  /// Returns the instructions that may have accessed the given register unit.
  /// We track DEF operands for all memory instructions (RAW/WAW hazards), and
  /// USE operands for XCnt instructions on gfx1250+ (WAR hazards for address
  /// translation). Each entry includes whether the instruction DEFs or USEs
  /// the register. Within a single block we only track the latest accessor
  /// (earlier accesses are overwritten), but after merging from multiple
  /// predecessors there may be multiple entries.
  ArrayRef<RegUnitInfo> getInstrsFor(MCRegUnit RU) const;

  /// Returns the register whose units should be scanned when looking for
  /// dependencies of \p DstMI on \p Reg. The 16-bit halves of a VGPR are
  /// separate register units and normally tracked independently. On targets
  /// where a D16 VALU instruction writes the whole 32-bit VGPR
  /// (hasD16Writes32BitVgpr), a 16-bit VGPR accessed by a VALU \p DstMI is
  /// widened to its enclosing 32-bit register (when the other half has a pending
  /// op) so the cross-half dependency is found; otherwise \p Reg is returned
  /// unchanged.
  MCRegister getEffectiveDepReg(Register Reg, const MachineInstr &DstMI) const;

  WaitDescriptors getWaitFor(Register Reg, MachineInstr &DstMI,
                                     RegAccessType DstAccess) const;

  /// Returns the waits needed for cross-unit LDS memory dependencies.
  /// Specifically, when VMEM-to-LDS operations (BUFFER_LOAD_*_LDS,
  /// GLOBAL_LOAD_LDS_*) are pending and a DS operation accesses LDS,
  /// we need a vmcnt wait to ensure the VMEM-to-LDS writes complete.
  WaitDescriptors getWaitForMemory(const MachineInstr &MI) const;

  /// Update the counters by applying CAWs waits.
  void drainCounters(const WaitDescriptors &CAWs);

  /// Applies waits upon destruction.
  class ApplyWaitsGuard {
    ResourceTracker &RT;
    const WaitDescriptors &CAWs;
  public:
    ApplyWaitsGuard(ResourceTracker &RT, const WaitDescriptors &CAWs) : RT(RT), CAWs(CAWs) {}
    ~ApplyWaitsGuard() {
      RT.drainCounters(CAWs);
    }
  };

  /// Defer applying waits until the guard goes out of scope. This can be very
  /// helpful in functions with early exits, preventing returning without
  /// applying the waits.
  [[nodiscard]] ApplyWaitsGuard getApplyWaitsGuard(const WaitDescriptors &CAWs) {
    return ApplyWaitsGuard(*this, CAWs);
  }

  const Counter &getCounter(const CounterType &Cntr) const {
    return Counters[Cntr];
  }
  Counter &getCounter(const CounterType &Cntr) { return Counters[Cntr]; }

  /// Seed \p Cntr with unknown incoming state, modeling operations from outside
  /// this function (e.g. a caller's stores at the entry of a non-entry
  /// function). See Counter::IncomingUnknown.
  void setCounterIncomingUnknown(const CounterType &Cntr) {
    Counters[Cntr].setIncomingUnknown();
  }

  /// Merge another ResourceTracker's state into this one. Used for dataflow
  /// analysis to combine states from multiple predecessor blocks.
  void merge(const ResourceTracker &Other);

  /// This is the state used to check for convergence.
  /// Just like in the original pass we are using the counter wait values.
  class ConvergenceState {
    WaitDescriptors Waits;
    /// Can only be constructed by the ResourceTracker.
    explicit ConvergenceState(const WaitDescriptors &Waits) : Waits(Waits) {}
    friend class ResourceTracker;

  public:
    bool operator==(const ConvergenceState &Other) const {
      return Waits == Other.Waits;
    }
    bool operator!=(const ConvergenceState &Other) const {
      return !(*this == Other);
    }

#ifndef NDEBUG
    raw_ostream &print(raw_ostream &OS) const {
      OS << Waits;
      return OS;
    }
    LLVM_DUMP_METHOD void dump() const;
#endif
  };

  ConvergenceState getConvergenceState() const {
    return ConvergenceState(Counters.get());
  }

  /// Clear all tracked state. Used to reset state at the start of dataflow
  /// iteration for a block.
  void clear();

#ifndef NDEBUG
  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

} // namespace AMDGPU

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPURESOURCETRACKER_H
