//===- Scheduler.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SCHEDULER_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SCHEDULER_H

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/SandboxIR/DmpVector.h"
#include "llvm/SandboxIR/SandboxIRTracker.h"
#include "llvm/Transforms/Vectorize/SandboxVec/DependencyGraph.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIRVec.h"
#include <iterator>

namespace llvm {
namespace sandboxir {

class Instruction;
class Scheduler;

/// This is a group of Nodes that are to be scheduled on the same cycle.
class SchedBundle {
  using ContainerTy = SmallVector<DependencyGraph::Node *, 4>;
  ContainerTy Nodes;
  /// The FinalSchedule list that contains the bundles.
  Scheduler *Sched = nullptr;
  friend class Scheduler; // for SchedList and ScheduleIdx.

  friend class SchedBundleAttorney;

public:
  SchedBundle() = default;
  SchedBundle(std::initializer_list<sandboxir::Instruction *> SBInstrs,
              Scheduler &Sched)
      : SchedBundle(
            DmpVector<sandboxir::Value *>{SBInstrs.begin(), SBInstrs.end()},
            Sched) {}
  SchedBundle(const DmpVector<sandboxir::Value *> &SBInstrs,
              Scheduler &Sched);
  SchedBundle(SmallVector<DependencyGraph::Node *, 4> &&Nodes,
              Scheduler &Sched);
  SchedBundle(const SmallVector<DependencyGraph::Node *, 4> &Nodes,
              Scheduler &Sched);
  ~SchedBundle();

  /// Move all instructions of the bundle on top of the lowest instruction in
  /// the bundle.
  void cluster();
  /// Move the first bundle instruction before \p BeforeIt in \p sandboxir::SBBB
  /// and the rest of the instructions on top of it.
  void cluster(sandboxir::BasicBlock::iterator BeforeIt,
               sandboxir::BasicBlock *SBBB);

  sandboxir::Instruction *getTopI() const;

  sandboxir::Instruction *getBotI() const;

  bool empty() const { return Nodes.empty(); }

  explicit SchedBundle(DependencyGraph::Node *N) : Nodes({N}) {}
  bool operator==(const SchedBundle &Other) const {
    return size() == Other.size() && equal(Nodes, Other.Nodes);
  }
  DependencyGraph::Node *operator[](unsigned Idx) const { return Nodes[Idx]; }
  Scheduler &getScheduler() const;
  /// \Returns the previous bundle in the schedule, or nullptr if at the top.
  SchedBundle *getPrev();
  /// \Returns the next bundle in the schedule, or nullptr if at the bottom.
  SchedBundle *getNext();
  /// \Returns true if this bundle comes before \p B in the FinalSchedule.
  bool comesBefore(SchedBundle *B);
  /// \Returns the bundle that shows up in the FinalSchedule the earliest.
  /// WARNING: This is a linear-time operation!
  static SchedBundle *getEarliest(SchedBundle *B1, SchedBundle *B2) {
    return B1->comesBefore(B2) ? B1 : B2;
  }
  static SchedBundle *getLatest(SchedBundle *B1, SchedBundle *B2) {
    return B1->comesBefore(B2) ? B2 : B1;
  }

  DependencyGraph::Node *top() const;
  DependencyGraph::Node *bottom() const;
  bool allSuccsReady() const;
  bool isScheduled() const;
  void setScheduled(bool Val);
  void reserve(uint32_t Sz) { Nodes.reserve(Sz); }
  void eraseFromParent();
  void remove(DependencyGraph::Node *N);
  bool contains(const DependencyGraph::Node *N) const;
  DependencyGraph::Node *back() const { return Nodes.back(); }
  size_t size() const { return Nodes.size(); }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  friend raw_ostream &operator<<(raw_ostream &OS, const SchedBundle &B) {
    B.dump(OS);
    return OS;
  }
#endif
  using const_iterator = ContainerTy::const_iterator;
  using iterator = ContainerTy::iterator;
  const_iterator begin() const { return Nodes.begin(); }
  const_iterator end() const { return Nodes.end(); }
  iterator begin() { return Nodes.begin(); }
  iterator end() { return Nodes.end(); }
  auto nodes() const { return make_range(begin(), end()); }
  auto nodes() { return make_range(begin(), end()); }
};

class SchedBundleAttorney {
public:
  static SchedBundle::ContainerTy &getNodes(SchedBundle &SB) {
    return SB.Nodes;
  }
  friend class Node;
};

class ReadyListContainer {
protected:
  // TODO: Use a PriorityQueue instead once we introduce a sorting heuristic.

  /// Since the DAG does not include edges for PHIs-before others we maintain a
  /// separate list for PHI nodes.
  std::list<DependencyGraph::Node *> PHIList;
  /// Landingpads/CleanupPads/CatchPads should be the first non-PHI.
  DependencyGraph::Node *PadN = nullptr;
  /// The list for non-PHI non-Terminator instructions.
  std::list<DependencyGraph::Node *> List;
  /// Since the DAG does not include edges between the terminator and all other
  /// instructions in the block, we keep the terminator separately.
  DependencyGraph::Node *Terminator = nullptr;

public:
  /// WARNING: This is meant to be used for testing only!
  bool contains(DependencyGraph::Node *N) const;
  /// WARNING: This is meant to be used for testing only!
  SmallVector<DependencyGraph::Node *> getContents() const;

  uint64_t size() const;
  bool empty() const;
  void insert(DependencyGraph::Node *N);
  void remove(DependencyGraph::Node *N);
  /// Removes the front element of the list and returns it.
  DependencyGraph::Node *pop();
  void clear();
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  void dump() const;
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const ReadyListContainer &RList) {
    RList.dump(OS);
    return OS;
  }
#endif
};

class Scheduler {
  friend class SchedulerAttorney; // For testing
  DependencyGraph DAG;
  friend SchedBundle::SchedBundle(const DmpVector<sandboxir::Value *> &,
                                  Scheduler &); // For DAG.
  ReadyListContainer ReadyList;

  // These make up the schedule list.
  SmallVector<std::unique_ptr<SchedBundle>> BundlePool;
  /// The scheduler's top bundle. When instructions get scheduled they get
  /// placed on top of \p TopSB.
  SchedBundle *TopSB = nullptr;
  size_t NumScheduledBndls = 0;
  sandboxir::BasicBlock &SBBB;
  sandboxir::SBVecContext &Ctx;
  /// Set by trySchedule().
  bool TopDown = false;
  /// Don't allow changing TopDown direction. This bool is used to check it.
  bool DirectionSet = false;
  /// This is set true when we start scheduling and false when we clear state.
  bool Scheduling = false;

  /// Callback handle only needed for unregistering it.
  sandboxir::Context::RemoveCBTy *RemoveInstrCB = nullptr;
  /// Callback handle only needed for unregistering it.
  sandboxir::Context::InsertCBTy *InsertInstrCB = nullptr;
  /// Callback handle only needed for unregistering it.
  sandboxir::Context::MoveCBTy *MoveInstrCB = nullptr;

  // Friends for `Tracker`.
  friend void SchedBundle::cluster();
  friend void SchedBundle::cluster(sandboxir::BasicBlock::iterator BeforeIt,
                                   sandboxir::BasicBlock *BB);

  SchedBundle *createBundle(const DmpVector<sandboxir::Value *> &SBInstrs);
  SchedBundle *createBundle(DependencyGraph::Node *N) {
    return createBundle({N->getInstruction()});
  }
  void decrDepsReadyCounter(DependencyGraph::Node *N);

public:
  /// Used by unittests.
  void startScheduling() { Scheduling = true; } // TODO: make it private.
  bool scheduling() const { return Scheduling; }

  bool isTopDown() const { return TopDown; }
  // These are used for unit testing.
  const auto &getReadyList() const { return ReadyList; }
  const DependencyGraph &getDAG() const { return DAG; }
  /// WARNING: This is only meant to be used by tests!
  DependencyGraph &getDAG() { return DAG; }

private:
  /// Schedules \p B at the top of the schedule list and appends any ready
  /// predecessors to the ready list.
  void scheduleAndUpdateReadyList(SchedBundle *B);
  /// Schedules all instructions until (including) \p Instrs. The in-between
  /// instructions are scheduled as standalone bundles, while all instrs in \p
  /// Instrs are scheduled together in a bundle on the same cycle. \Returns true
  /// if it managed to succesfully schedule all \p Instrs in one bundle.
  bool tryScheduleUntil(const DmpVector<sandboxir::Value *> &Instrs);
#ifndef NDEBUG
public:
  /// Verifies that we are scheduling bottom-up.
  void verifyDirection(const DmpVector<sandboxir::Value *> &SBInstrs,
                       bool TopDown);
  /// Verifies bundle positions.
  void verifySchedule();

private:
#endif
  void eraseBundle(SchedBundle *SB);
  friend void SchedBundle::eraseFromParent(); // for eraseBundle()
  /// Helper function that performs some common internal state cleanup.
  void clearState();
  friend class AcceptOrRevert; // for clearState()

  /// The scheduler is owned by sandboxir::Context and is created
  /// automatically by it for each sandboxir::BasicBlock.
  Scheduler(sandboxir::BasicBlock &SBBB, AliasAnalysis &AA,
            sandboxir::SBVecContext &Ctx);
  friend class sandboxir::SBVecContext; // for createdBasicBlock() calling the
                                        // constructor

  /// Prepares the DAG and Scheduler for a fresh schedule for \p Instrs. This
  /// assumes that none of the instrs have been scheduled.
  bool
  extendRegionAndUpdateReadyList(const DmpVector<sandboxir::Value *> &Instrs);

public: // public for the tests.
  /// This should be called before \p I gets erased. It helps maintain the
  /// scheduler's state when instructions get removed from parent.
  /// NOTE: Should only be called by sandboxir::Context!
  void notifyRemove(sandboxir::Instruction *SBI, bool CalledByDAG = false);
  void notifyInsert(sandboxir::Instruction *SBI);
  void notifyMove(sandboxir::Instruction *SBI, sandboxir::BasicBlock &SBBB,
                  const sandboxir::BBIterator &WhereIt);

  /// The scheduling state of the instructions in the bundle.
  enum class BndlSchedState {
    NoneScheduled,
    PartiallyOrDifferentlyScheduled,
    FullyScheduled,
  };
  /// \Returns whether none/some/all of \p Instrs have been scheduled.
  BndlSchedState
  getBndlSchedState(const DmpVector<sandboxir::Value *> &Instrs) const;

  /// Destroy the top-most part of the schedule that includes \p Instrs.
  void trimSchedule(const DmpVector<sandboxir::Value *> &Instrs);

  /// Updates the DAG nodes' ready counters.
  void notifyNewViewNodes(
      const SmallPtrSet<DependencyGraph::Node *, 16> &NewViewNodes);

public:
  ~Scheduler();
  Scheduler(const Scheduler &) = delete;
  void enableTracking() { DAG.enableTracking(); }
  Scheduler &operator=(Scheduler &) = delete;

  /// A simple list scheduler used for stress-testing.
  void listSchedule(sandboxir::BasicBlock *BB);
  /// Tries to schedule \p Instrs on the same scheduling cycle.
  bool trySchedule(const DmpVector<sandboxir::Value *> &Instrs,
                   bool TopDown = false);
  bool scheduleEmpty() const;
  size_t scheduleSize() const { return NumScheduledBndls; }
  SchedBundle *getTop() const { return TopSB; }
  /// \Returns the SchedBundle that \p I's node is associated with. \Returns
  /// null if either there is no Node for it or no SchedBundle for that Node.
  SchedBundle *getBundle(sandboxir::Instruction *SBI) const;
  // TODO: Remove once we migrate to SandboxIR.
  void startTracking(sandboxir::BasicBlock *SBBB) {
    Ctx.getTracker().start(SBBB);
  }
  void stopTracking() { Ctx.getTracker().stop(); }
  SandboxIRTracker &getTracker() { return Ctx.getTracker(); }
  /// Just accept the current IR changes.
  void accept();
  /// Reset the scheduler state as if no nodes have been scheduled. This is
  /// linear-time to the nodes in DAG. Please note that this won't clear the
  /// DAG, it just resets the DAG view.
  /// This gets called in between vectorization attempts of a \p SBBB.
  void startFresh(sandboxir::BasicBlock *SBBB);
  /// Reverts all scheduler state, resetting the DAG view and undoing any IR
  /// changes.
  /// This is typically called when we want to completely undo all the scheduler
  /// state, like in a failed scheduling attempt.
  void revert();
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const { dump(dbgs()); }
#endif
};

/// A client-attorney class used for testing private funcs of Scheduler.
class SchedulerAttorney {
public:
  using BndlSchedState = Scheduler::BndlSchedState;
  static BndlSchedState
  getBndlSchedState(Scheduler &Sched,
                    const DmpVector<sandboxir::Value *> &Instrs) {
    return Sched.getBndlSchedState(Instrs);
  }
  static void trimSchedule(Scheduler &Sched,
                           const DmpVector<sandboxir::Value *> &Instrs) {
    return Sched.trimSchedule(Instrs);
  }
  static bool extendRegionAndUpdateReadyList(
      Scheduler &Sched, const DmpVector<sandboxir::Value *> &Instrs) {
    return Sched.extendRegionAndUpdateReadyList(Instrs);
  }
};
} // namespace sandboxir
} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SCHEDULER_H
