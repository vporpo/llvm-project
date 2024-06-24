//===- DependencyGraph.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_DEPENDENCYGRAPH_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_DEPENDENCYGRAPH_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/User.h"
#include "llvm/SandboxIR/DmpVector.h"
#include "llvm/Transforms/Vectorize/SandboxVec/InstrInterval.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include <iterator>

namespace llvm {
namespace sandboxir {

class Instruction;
class SchedBundle;
class Scheduler;

// The incremental dependency DAG
// ------------------------------
// This DAG provides iterators that iterate over both def-use/use-def
// dependencies and memory/other dependencies. We only actually store the
// memory/other dependencies, the def-use/use-def dependencies come directly
// from LLVM IR.
//
// This DAG does not include ordering dependencies for:
//  - PHIs: There is no edge between PHI->NonPHI instructions.
//  - Terminator: No edge between Instr->Terminator.
// These would require a lot of edges which would hurt compile-time.
// NOTE: Forcing ordering should be taken care of by the scheduler!
//
// The DAG is built incrementally in sections. This helps save compilation time
// if sections of the BB contain no vectorization opportunities and as such
// require no DAG.
//
// Building the DAG involves creating DAG `Node` objects, one for each
// sandboxir::Instruction. Nodes provide predecessor/successor iterators.
// Internally, memory/other dependencies are implemented with a vector of
// pointers to the dependent dependent Node.
//
// Extending the DAG to include a new instruction region requires a few more
// steps than just building a new DAG:
// - The first step is to create new Nodes, one for each instruction in the new
//   region. While doing so, we are also creating the intra-region memory
//   dependencies. This is exactly what we would do if we created a new DAG.
// - Now we need to create any intra-region dependencies, which includes:
//   - Going over all memory/other Nodes of the new region and connecting them
//     to all dependent memory/other Nodes of the original region.
//   - Going over all instructions of the new region, looking for def/use
//     users that belong to the original region and adjusting UnscheduledSuccs.
//
// When we build a DAG for a region we will also create Node's for sources of
// memory dependencies even if the corresponding instruction is outside the
// current region. We build memory deps by walking up the instruction chain,
// looking for instructions that may alias.
//

class DependencyGraph {
public:
  class Node;
  /// Iterate over use-def/def-use and memory predecessors/successors.
  template <typename OpItT> class PredSuccIteratorTemplate {
    /// Operand or User Iterator.
    OpItT OpIt;
    /// Iterator for memory or other dependences for Node::PredDeps or SuccDeps.
    SetVector<Node *>::iterator OtherIt;
    Node *N;
    DependencyGraph *Parent;
    /// Skip non-Instruction OpIt iterators and values not mapped to Nodes yet.
    /// If \p PreInc is true then increment the iterator before trying to skip.
    void SkipOpIt(bool PreInc);

  public:
    using difference_type = std::ptrdiff_t;
    using value_type = Node *;
    using pointer = value_type *;
    using reference = value_type &;
    using iterator_category = std::input_iterator_tag;
    PredSuccIteratorTemplate(OpItT OpIt, SetVector<Node *>::iterator OtherIt,
                             Node *N, DependencyGraph &Parent);
    value_type operator*();
    PredSuccIteratorTemplate &operator++();
    PredSuccIteratorTemplate operator++(int);
    bool operator==(const PredSuccIteratorTemplate &Other) const;
    bool operator!=(const PredSuccIteratorTemplate &Other) const;
  };

  using PredIterator = PredSuccIteratorTemplate<sandboxir::User::op_iterator>;
  using SuccIterator =
      PredSuccIteratorTemplate<sandboxir::Value::user_iterator>;

  /// A DAG Node that points to a sandboxir::Instruction. It also holds the
  /// memory dependencies.
  class Node {
    friend PredIterator;
    friend SuccIterator;
    friend class DependencyGraph;
    sandboxir::Instruction *I;
    // TODO: Use bit-fields
    bool Scheduled = false;
    bool IsMem = false;
    bool InReadyList = false;
    DependencyGraph &Parent;
    /// During scheduling this node can be part of a SchedBundle.
    SchedBundle *ParentBundle = nullptr;
    /// Predecessor memory/other dependencies.
    SetVector<Node *> Preds;
    /// Successor memory/other dependencies.
    SetVector<Node *> Succs;
    /// This is used by the scheduler to determine when this node is ready.
    uint32_t UnscheduledSuccs = 0;
    /// Adds (memory/other) dependency N->this.
    void addMemPred(Node *N);
    /// Removes (memory/other) dependency N->this.
    void eraseMemPred(Node *N);
    /// \Returns the DAG Node that corresponds to the previous instruction in
    /// the instruction chain, or null if at top or if no node found.
    Node *getPrevNode() const;
    /// \Returns the DAG Node that corresponds to the next instruction in
    /// the instruction chain, or null if at top or if no node found.
    Node *getNextNode() const;
    /// Walks up the instruction chain looking for the next memory instruction.
    /// \Returns the corresponding DAG Node, or null if no instruction found.
    Node *getPrevMem() const;
    /// Walks down the instr chain looking for the next memory instruction.
    /// \Returns the corresponding DAG Node, or null if no instruction found.
    Node *getNextMem() const;

  public:
    Node(sandboxir::Instruction *SBI, DependencyGraph &Parent);
    bool isMem() const { return IsMem; }
    bool isInReadyList() const { return InReadyList; }
    void setIsInReadyList() { InReadyList = true; }
    void resetIsInReadyList() { InReadyList = false; }
    sandboxir::Instruction *getInstruction() const { return I; }
    bool isScheduled() const { return Scheduled; }
    void setScheduled(bool Val) {
      // Mark as scheduled.
      Scheduled = Val;
      assert((!Val || UnscheduledSuccs == 0) && "Not ready?");
    }
    PredIterator pred_begin() const;
    PredIterator pred_end() const;
    /// \Returns a range of both use-def and memory predecessors.
    iterator_range<PredIterator> preds() const {
      return iterator_range<PredIterator>(pred_begin(), pred_end());
    }
    /// \Returns only the memory predecessors.
    const auto &mem_preds() const { return Preds; }

    SuccIterator succ_begin() const;
    SuccIterator succ_end() const;
    /// \Returns a range of both def-use and memory successors.
    iterator_range<SuccIterator> succs() const {
      return iterator_range<SuccIterator>(succ_begin(), succ_end());
    }
    bool hasNoSuccs() const { return succ_begin() == succ_end(); }
    /// \Returns only the memory succsessors.
    const auto &mem_succs() const { return Succs; }
    /// \Returns true if \p N is an immediate predecessor. This is linear to the
    /// use-def operands + constant time to the memory dependencies.
    bool hasImmPred(Node *N) const;
    /// \Returns true if \p N is an immediate memory-dep predececessor. This is
    /// a constant-time operation.
    bool hasMemPred(Node *N) const;
    /// \Returns true if this node's instruction comes before N's in the BB.
    bool comesBefore(Node *N) const { return I->comesBefore(N->I); }
    bool comesBeforeOrEqual(Node *N) const {
      return I == N->I || I->comesBefore(N->I);
    }
    /// \Returns this if this node's instruction is earlier in the BB than \p
    /// N's, or N otherwise.
    Node *getEarliest(Node *N) const {
      return comesBefore(N) ? const_cast<Node *>(this) : N;
    }
    /// \Returns this if this node's instruction is later in the BB than \p
    /// N's, or N otherwise.
    Node *getLatest(Node *N) const {
      return comesBefore(N) ? N : const_cast<Node *>(this);
    }
    /// \Returns true if there is a dependency path from \p N to this.
    /// WARNING: this is a *very* expensive operation as it walks all
    /// predecesor paths looking for \p N. It is meant for testing.
    bool dependsOn(Node *N) const;
    /// \Returns true if all of its in-region successors have been scheduled.
    bool allSuccsReady() const { return UnscheduledSuccs == 0; }
    // bool allSuccsReady() const { return UnscheduledSuccs == 0; }
    uint32_t getNumUnscheduledSuccs() const { return UnscheduledSuccs; }
    void decrementUnscheduledSuccs() {
      // Note: UnscheduledSuccs could be 0 after Scheduler clearState() and
      // before rescheduling.
      assert(!Scheduled && "Expected not scheduled node!");
      assert(UnscheduledSuccs > 0 && "Broken counter!");
      --UnscheduledSuccs;
    }
    void incrementUnscheduledSuccs() {
      assert(
          Parent.inView(this) &&
          "Don't waste time updating the counter for nodes outside the view. "
          "They will be calculated from scratch once we extend the view.");
      assert(!InReadyList && "In ready list!");
      assert(!Scheduled && "Already scheduled!");
      assert(UnscheduledSuccs !=
                 std::numeric_limits<decltype(UnscheduledSuccs)>::max() &&
             "Bad UnscheduledSuccs!");
      ++UnscheduledSuccs;
    }
    void resetUnscheduledSuccs() { UnscheduledSuccs = 0; }
    SchedBundle *getBundle() const { return ParentBundle; }
    // TODO: Make private
    /// Should only be used by `Scheduler::createBundle() and reset()`.
    void setBundle(SchedBundle *NewBundle) {
      assert(NewBundle != nullptr);
      assert(ParentBundle == nullptr && "Belongs to other bundle!");
      ParentBundle = NewBundle;
    }

    void removeFromBundle();
#ifndef NDEBUG
    /// If \p InstrIntervalOnly is true we only print nodes within the current
    /// region.
    void dump(raw_ostream &OS, bool InstrIntervalOnly = false,
              bool PrintDeps = true) const;
    LLVM_DUMP_METHOD void dump() const {
      dump(dbgs());
      dbgs() << "\n";
    }
    friend raw_ostream &operator<<(raw_ostream &OS, const Node &N) {
      N.dump(OS);
      return OS;
    }
    /// Checks that the node is well constructed. Crashes on error.
    void verify() const;
#endif
  };

  /// Helps walk Nodes in an order that follows the instruction chain.
  /// This is used for updating the DAG.
  class NodeIterator {
    Node *N;
    bool IsEnd;

  public:
    using difference_type = std::ptrdiff_t;
    using value_type = Node;
    using pointer = value_type *;
    using reference = Node &;
    using iterator_category = std::bidirectional_iterator_tag;
    NodeIterator(Node *N, bool IsEnd) : N(N), IsEnd(IsEnd) {}
    NodeIterator &operator++() {
      assert(!IsEnd && "Already at end!");
      Node *NextN = N->getNextNode();
      if (NextN == nullptr)
        IsEnd = true;
      else
        N = NextN;
      return *this;
    }
    NodeIterator operator++(int) {
      auto ItCopy = *this;
      ++*this;
      return ItCopy;
    }
    NodeIterator &operator--() {
      if (IsEnd)
        IsEnd = false;
      else
        N = N->getPrevNode();
      return *this;
    }
    NodeIterator operator--(int) {
      auto ItCopy = *this;
      --*this;
      return ItCopy;
    }
    reference operator*() { return *N; }
    const Node &operator*() const { return *N; }
    bool operator==(const NodeIterator &Other) const {
      return N == Other.N && IsEnd == Other.IsEnd;
    }
    bool operator!=(const NodeIterator &Other) const {
      return !(*this == Other);
    }
  };

  /// Walks in the order of the instruction chain but skips non-mem Nodes.
  /// This is used for updating the DAG.
  class NodeMemIterator {
    Node *N;
    bool IsEnd;

  public:
    using difference_type = std::ptrdiff_t;
    using value_type = Node;
    using pointer = value_type *;
    using reference = value_type &;
    using iterator_category = std::bidirectional_iterator_tag;
    NodeMemIterator(Node *N, bool IsEnd) : N(N), IsEnd(IsEnd) {
      assert((N == nullptr || N->isMem()) && "Expects mem node!");
    }
    NodeMemIterator &operator++() {
      assert(!IsEnd && "Already at end!");
      Node *NextN = N->getNextMem();
      if (NextN == nullptr)
        IsEnd = true;
      else
        N = NextN;
      return *this;
    }
    NodeMemIterator operator++(int) {
      auto ItCopy = *this;
      ++*this;
      return ItCopy;
    }
    NodeMemIterator &operator--() {
      if (IsEnd)
        IsEnd = false;
      else
        N = N->getPrevMem();
      return *this;
    }
    NodeMemIterator operator--(int) {
      auto ItCopy = *this;
      --*this;
      return ItCopy;
    }
    reference operator*() { return *N; }
    const Node &operator*() const { return *N; }
    bool operator==(const NodeMemIterator &Other) const {
      return N == Other.N && IsEnd == Other.IsEnd;
    }
    bool operator!=(const NodeMemIterator &Other) const {
      return !(*this == Other);
    }
  };

  class NodeRange : public iterator_range<NodeIterator> {
  public:
    NodeRange(NodeIterator Begin, NodeIterator End)
        : iterator_range(Begin, End) {}
#ifndef NDEBUG
    LLVM_DUMP_METHOD void dump();
#endif
  };

  // All-node ranges.
  static NodeRange makeRange(Node *Top, Node *Bot) {
    assert(Top->comesBeforeOrEqual(Bot) && "Expect Top before Bot");
    return NodeRange(NodeIterator(Top, false),
                     std::next(NodeIterator(Bot, false)));
  }
  NodeRange makeRange(sandboxir::Instruction *Top,
                      sandboxir::Instruction *Bot) {
    assert((Top == Bot || Top->comesBefore(Bot)) && "Expect Top before Bot");
    return NodeRange(NodeIterator(getNode(Top), false),
                     std::next(NodeIterator(getNode(Bot), false)));
  }
  static NodeRange makeEmptyRange() {
    return NodeRange(NodeIterator(nullptr, true), NodeIterator(nullptr, true));
  }

  // Mem-node ranges.
  class NodeMemRange : public iterator_range<NodeMemIterator> {
  public:
    NodeMemRange(NodeMemIterator Begin, NodeMemIterator End)
        : iterator_range(Begin, End) {}
#ifndef NDEBUG
    LLVM_DUMP_METHOD void dump();
#endif
  };

  static NodeMemRange makeMemRange(Node *Top, Node *Bot) {
    assert(Top->comesBeforeOrEqual(Bot) && "Expect Top before Bot");
    return NodeMemRange(NodeMemIterator(Top, false),
                        std::next(NodeMemIterator(Bot, false)));
  }
  NodeMemRange makeMemRange(sandboxir::Instruction *Top,
                            sandboxir::Instruction *Bot) {
    assert((Top == Bot || Top->comesBefore(Bot)) && "Expect Top before Bot");
    return NodeMemRange(NodeMemIterator(getNode(Top), false),
                        std::next(NodeMemIterator(getNode(Bot), false)));
  }
  static NodeMemRange makeEmptyMemRange() {
    return NodeMemRange(NodeMemIterator(nullptr, true),
                        NodeMemIterator(nullptr, true));
  }

  NodeMemRange makeMemRangeFromNonMem(Node *TopN, Node *BotN) const {
    assert(TopN->comesBeforeOrEqual(BotN) && "Expected Top before Bot");
    if (!TopN->isMem())
      TopN = TopN->getNextMem();
    if (TopN == nullptr)
      return makeEmptyMemRange();
    if (!BotN->isMem())
      BotN = BotN->getPrevMem();
    if (BotN == nullptr)
      return makeEmptyMemRange();
    if (!TopN->comesBeforeOrEqual(BotN))
      return makeEmptyMemRange();
    return makeMemRange(TopN, BotN);
  }
  NodeMemRange makeMemRangeFromNonMem(sandboxir::Instruction *Top,
                                      sandboxir::Instruction *BotN) {
    return makeMemRangeFromNonMem(getNode(Top), getNode(BotN));
  }

private:
  DenseMap<sandboxir::Instruction *, std::unique_ptr<Node>> InstrToNodeMap;
  /// The DAG spans this instruction interval, which means that all instructions
  /// in the interval have a corresponding DAG node and all their dependencies
  /// within this region have been set.
  InstrInterval DAGInterval;
  /// A view over the DAG. This helps when the DAG has already been built and we
  /// want a narrower view of it. The ViewRange is always contained
  /// (inclusively) in DAGInterval.
  InstrInterval ViewRange;
  sandboxir::Context &Ctx;
  AliasAnalysis &AA;
  /// Limits the number of AA queries we do while building the DAG.
  int AAQueryBudget;
  Scheduler *Sched = nullptr;
  std::unique_ptr<BatchAAResults> BatchAA;

  /// Nodes newly added to the View. Used to ask the scheduler to update
  /// UnscheduledSuccs.
  SmallPtrSet<Node *, 16> NewViewNodes;

  /// If the user has not initialized the DAG don't bother updating it.
  bool TrackingEnabled = false;

  /// This cache is used by hasDep() and makes sure that the DAG is reversible
  /// during IR revert().
  DenseMap<std::pair<sandboxir::Instruction *, sandboxir::Instruction *>,
           bool>
      DepCache;
  std::pair<sandboxir::Instruction *, sandboxir::Instruction *>
  getDepCacheKey(sandboxir::Instruction *I1, sandboxir::Instruction *I2);
  void setDepCache(sandboxir::Instruction *I1, sandboxir::Instruction *I2,
                   bool HasDep);
  bool getDepCache(sandboxir::Instruction *I1, sandboxir::Instruction *I2);

  /// Helpers for notifying the scheduler about new view nodes such that it
  /// updates the DAG with correct ready counters.
  void collectNewViewNodesInit();
  void collectNewViewNodesNotify();
  friend class DependencyGraphAttorney; // For testing

  enum class DependencyType {
    RAW,   ///> Read After Write
    WAW,   ///> Write After Write
    RAR,   ///> Read After Read
    WAR,   ///> Write After Read
    CTRL,  ///> Dependencies related to PHIs or Terminators
    OTHER, ///> Currently used for stack related instrs
    NONE,  ///> No memory/other dependency
  };
  /// \Returns the dependency type depending on whether instructions may
  /// read/write memory or whether they are some specific opcode-related
  /// restrictions.
  /// Note: It does not check whether a memory dependency is actually correct,
  /// as it won't call AA. Therefore it returns the worst-case dep type.
  static DependencyType getRoughDepType(sandboxir::Instruction *FromI,
                                        sandboxir::Instruction *ToI);
  enum class DepResult {
    HasDep,
    NoDep,
    ReachedAnalysisLimit,
  };
  /// \Returns true if there is a memory/other dependency \p SrcI->DstI and
  /// updates \p AABudget.
  bool alias(llvm::Instruction *SrcIR, llvm::Instruction *DstIR,
             DependencyType DepType, int &AABudget);

public:
  /// If not explicitly enabled it won't track IR changes.
  void enableTracking() { TrackingEnabled = true; }
  bool trackingEnabled() const { return TrackingEnabled; }
  /// \Returns the aproximate size of the DAG View if we extend it to include \p
  /// Instrs.
  size_t getProjectedApproxSize(const DmpVector<sandboxir::Value *> &Instrs);
  void setAAQueryBudget(int Budget) { AAQueryBudget = Budget; }
  bool hasDep(sandboxir::Instruction *SrcI, sandboxir::Instruction *DstI,
              int &AABudget);
  /// Add mem deps to \p DstN by scanning bottom-up \p ScanRange.
  void scanAndAddDeps(Node *DstN, NodeMemRange ScanRange);
  /// Creates DAG nodes for all non-debug instructions in \p Rgn.
  void createNodesFor(const InstrInterval &Rgn);
  /// Helper function that returns a memory node range from \p ScanTopN to the
  /// earlier of \p ScanBotN and \p AboveN.
  NodeMemRange getScanRange(Node *ScanTopN, Node *ScanBotN, Node *AboveN);
  /// Extends the dependency graph by scanning \p NewInstrInterval and adding
  /// any necessary edges to \p OldInstrInterval.
  void extendDAG(const InstrInterval &OldInstrInterval,
                 const InstrInterval &NewInstrInterval);
  /// \Returns the Node that corresponds to \p I or if not found creates one
  /// and returns it. Set \p IsMem if this node is a candidate source or sink
  /// of a mem/other dependency.
  Node *getOrCreateNode(sandboxir::Instruction *I);

  InstrInterval extendView(const InstrInterval &NewViewRange);
  /// Extend the DAG and View to include \p Instrs. \Returns the new region
  /// section that corresponds to the extension of the DAG's view region. This
  /// will crash if we are attempting to extend in both directions, above and
  /// below.
  InstrInterval extendInternal(const DmpVector<sandboxir::Value *> &Instrs);

public:
  DependencyGraph(sandboxir::Context &Ctx, AliasAnalysis &AA,
                  int AAQueryBudget = 10, Scheduler *Sched = nullptr);
  DependencyGraph(const DependencyGraph &) = delete;
  DependencyGraph &operator=(DependencyGraph &) = delete;

  /// Scans the DAG for root nodes and \returns them. This is mainly meant for
  /// testing or debug.
  /// WARNING: This is an expensive O(N^2) operation.
  SmallVector<Node *> getRoots() const;
  /// \Returns the Node that corresponds to \p I or null if not found.
  /// Please note that debug info intrinsics are skipped.
  Node *getNode(sandboxir::Instruction *SBI) const;
  /// Same as \p getNode() but returns null if \p SBI is null.
  Node *getNodeOrNull(sandboxir::Instruction *SBI) const;
  /// Extend the DAG and View to include \p Instrs. \Returns the new region
  /// section that corresponds to the extension of the DAG's view region. This
  /// will crash if we are attempting to extend in both directions, above and
  /// below.
  InstrInterval extend(const DmpVector<sandboxir::Value *> &Instrs);
  /// Safely clears the view starting from \p FromI all the way to the
  /// top/bottom depending on \p Above.
  void trimView(sandboxir::Instruction *FromI, bool Above);
  /// Clears the DAG's ViewRange. This means that the DAG now looks like it
  /// is empty, so we have to use `extend()` to access it. This is used to
  /// avoid rebuilding the whole DAG when trying to re-schedule.
  void resetView();
  const InstrInterval &getView() const { return ViewRange; }
  /// Used for testing.
  const InstrInterval &getDAGInterval() const { return DAGInterval; }
  sandboxir::Instruction *getTop() const { return ViewRange.from(); }
  sandboxir::Instruction *getBottom() const { return ViewRange.to(); }
  bool inView(sandboxir::Instruction *I) const {
    return ViewRange.contains(I);
  }
  bool inView(Node *N) const { return inView(N->getInstruction()); }
  /// Clears all state.
  void clear();
  /// \Returns a range of all nodes in the graph.
  auto nodes() {
    return map_range(InstrToNodeMap, [](const auto &Pair) -> Node * {
      return Pair.second.get();
    });
  }
  /// Go over \p N's successors and check for all dependencies to \p N's
  /// predecessors, assuming that \p N has been removed from the DAG.
  /// \Returns a vector of {PredN, SuccN} pairs.
  SmallVector<std::pair<Node *, Node *>> getDepsFromPredsToSuccs(Node *N);
  /// Erases \p N from the DAG.
  void erase(Node *N, bool CalledByScheduler = false);
  /// This should be called *before* \p I gets erased from its parent.
  void erase(sandboxir::Instruction *I);
  /// Inserts \p I to the DAG and adds any depepndencies needed.
  void notifyInsert(sandboxir::Instruction *I);
  /// Insert \p I to the DAG, but don't insert it to any region and don't
  /// analyze its deps. These steps should be done by the scheduler.
  Node *insert(sandboxir::Instruction *I);
  /// Notify the DAG's regions about instruction movement.
  void notifyMoveInstr(sandboxir::Instruction *I,
                       sandboxir::BasicBlock::iterator BeforeIt,
                       sandboxir::BasicBlock *BB);
#ifndef NDEBUG
  /// If \p InstrIntervalOnly is true we only print nodes within the current
  /// region.
  void dump(raw_ostream &OS, bool InstrIntervalOnly = false,
            bool InViewOnly = false) const;
  LLVM_DUMP_METHOD void dump() const {
    dump(dbgs());
    dbgs() << "\n";
  }
  // Dump only the nodes that are in the DAG view.
  LLVM_DUMP_METHOD void dumpView() const {
    dump(dbgs(), /*InstrIntervalOnly=*/false, /*InViewOnly=*/true);
    dbgs() << "\n";
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const DependencyGraph &G) {
    G.dump(OS);
    return OS;
  }
  /// Verifies that the graph is well-constructed. Crashes on error.
  void verify(bool CheckReadyCnt = true);
#endif
};

} // namespace sandboxir
} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_DEPENDENCYGRAPH_H
