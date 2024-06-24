//===- Scheduler.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Transforms/Vectorize/SandboxVec/VecUtils.h"
#include <algorithm>

using namespace llvm;

static cl::opt<int>
    AAQueriesBudget("sbvec-dependency-graph-aa-budget", cl::init(10),
                    cl::Hidden,
                    cl::desc("Limits the number of AA quries performed per "
                             "region to reduce compilation time."));
static cl::opt<size_t>
    DAGSizeLimit("sbvec-dag-size-limit", cl::init(256), cl::Hidden,
                 cl::desc("Limit compilation time by setting an (approximate) "
                          "limit on the DAG size."));

sandboxir::SchedBundle::SchedBundle(
    const DmpVector<sandboxir::Value *> &SBInstrs, Scheduler &Sched)
    : Sched(&Sched) {
  auto &DAG = Sched.getDAG();
  Nodes.reserve(SBInstrs.size());
  for (auto *SBV : SBInstrs) {
    auto *N = DAG.getNode(cast<sandboxir::Instruction>(SBV));
    assert(N != nullptr && "No node found for `I`!");
    N->setBundle(this);
    Nodes.push_back(N);
  }
}

sandboxir::SchedBundle::SchedBundle(
    SmallVector<DependencyGraph::Node *, 4> &&Nodes, Scheduler &Sched)
    : Nodes(std::move(Nodes)), Sched(&Sched) {
  for (auto *N : this->Nodes)
    N->setBundle(this);
}

sandboxir::SchedBundle::SchedBundle(
    const SmallVector<DependencyGraph::Node *, 4> &Nodes, Scheduler &Sched)
    : Nodes(Nodes), Sched(&Sched) {
  for (auto *N : this->Nodes)
    N->setBundle(this);
}

sandboxir::SchedBundle::~SchedBundle() {
  while (!Nodes.empty()) {
    auto *N = Nodes.back();
    N->removeFromBundle();
  }
}

void sandboxir::SchedBundle::cluster() {
  // Cluster them at the bottom instruction.
  cluster(std::next(bottom()->getInstruction()->getIterator()),
          bottom()->getInstruction()->getParent());
}

void sandboxir::SchedBundle::cluster(sandboxir::BasicBlock::iterator BeforeIt,
                                     sandboxir::BasicBlock *SBBB) {
  assert((BeforeIt == SBBB->end() || (*BeforeIt).getParent() == SBBB) &&
         "Iterator not in BB!");
  // Make sure that `BeforeIt` does not point to an instr in the bundle.
  while (any_of(Nodes, [BeforeIt](auto *N) {
    return N->getInstruction()->getIterator() == BeforeIt;
  }))
    ++BeforeIt;
  assert(none_of(Nodes,
                 [BeforeIt](auto *N) {
                   return N->getInstruction()->getIterator() == BeforeIt;
                 }) &&
         "BeforeIt should not point to any node in the bundle, otherwise we "
         "won't get program order.");
  // Try to maintain program order.
  auto SortedNodes = Nodes;
  sort(SortedNodes, [](const auto *N1, const auto *N2) {
    return N1->getInstruction()->comesBefore(N2->getInstruction());
  });

  for (auto *N : SortedNodes) {
    sandboxir::Instruction *SBI = N->getInstruction();
    if (SBI->getIterator() == BeforeIt)
      continue;
    SBI->moveBefore(*SBI->getParent(), BeforeIt);
  }
}

sandboxir::Instruction *sandboxir::SchedBundle::getTopI() const {
  auto *TopI = Nodes.front()->getInstruction();
  for (const auto *N : drop_begin(Nodes)) {
    auto *I = N->getInstruction();
    if (I->comesBefore(TopI))
      TopI = I;
  }
  return TopI;
}

sandboxir::Instruction *sandboxir::SchedBundle::getBotI() const {
  auto *BotI = Nodes.front()->getInstruction();
  for (const auto *N : drop_begin(Nodes)) {
    auto *I = N->getInstruction();
    if (BotI->comesBefore(I))
      BotI = I;
  }
  return BotI;
}

sandboxir::Scheduler &sandboxir::SchedBundle::getScheduler() const {
  return *Sched;
}

sandboxir::SchedBundle *sandboxir::SchedBundle::getPrev() {
  if (!Sched->isTopDown() && this == Sched->getTop())
    return nullptr;
  for (sandboxir::Instruction *I = getTopI()->getPrevNode(); I != nullptr;
       I = I->getPrevNode())
    if (auto *SB = Sched->getBundle(I))
      return SB;
  return nullptr;
}

sandboxir::SchedBundle *sandboxir::SchedBundle::getNext() {
  if (Sched->isTopDown() && this == Sched->getTop())
    return nullptr;
  for (sandboxir::Instruction *I = getBotI()->getNextNode(); I != nullptr;
       I = I->getNextNode())
    if (auto *SB = Sched->getBundle(I))
      return SB;
  return nullptr;
}

bool sandboxir::SchedBundle::comesBefore(SchedBundle *B) {
  return (*begin())->getInstruction()->comesBefore(
      (*B->begin())->getInstruction());
}

sandboxir::DependencyGraph::Node *sandboxir::SchedBundle::top() const {
  // TODO: Perhaps cache this?
  return *std::min_element(Nodes.begin(), Nodes.end(), [](auto *N1, auto *N2) {
    return N1->getInstruction()->comesBefore(N2->getInstruction());
  });
}

sandboxir::DependencyGraph::Node *sandboxir::SchedBundle::bottom() const {
  // TODO: Perhaps cache this?
  return *std::max_element(Nodes.begin(), Nodes.end(), [](auto *N1, auto *N2) {
    return N1->getInstruction()->comesBefore(N2->getInstruction());
  });
}

bool sandboxir::SchedBundle::allSuccsReady() const {
  return all_of(Nodes, [](auto *N) { return N->allSuccsReady(); });
}

bool sandboxir::SchedBundle::isScheduled() const {
#ifndef NDEBUG
  assert(!Nodes.empty() && "Empty Bundle!");
  auto IsScheduled = [](auto *N) { return N->isScheduled(); };
  assert((all_of(Nodes, IsScheduled) || none_of(Nodes, IsScheduled)) &&
         "Expected either all or none");
#endif // NDEBUg
  return Nodes.back()->isScheduled();
}

void sandboxir::SchedBundle::setScheduled(bool Val) {
  assert(!Nodes.empty() && "Empty Bundle!");
  for (auto *N : Nodes)
    N->setScheduled(Val);
}

void sandboxir::SchedBundle::eraseFromParent() { Sched->eraseBundle(this); }

void sandboxir::SchedBundle::remove(sandboxir::DependencyGraph::Node *N) {
  if (size() == 1) {
    // If bundle contains only \p N, then don't remove the nodes first, because
    // erasing relies on them.
    assert(*begin() == N && "N is expected to be the only node");
    // The bundle is no longer needed, so erase it fro the scheduler.
    eraseFromParent();
  } else {
    N->removeFromBundle();
  }
}

bool sandboxir::SchedBundle::contains(
    const sandboxir::DependencyGraph::Node *N) const {
  // TODO: Make this constant time.
  return find(Nodes, N) != Nodes.end();
}

#ifndef NDEBUG
void sandboxir::SchedBundle::dump(raw_ostream &OS) const {
  OS << "[";
  for (auto [Idx, N] : enumerate(Nodes)) {
    N->dump(OS, /*RegionOnly=*/true, /*PrintDeps=*/false);
    if (Idx + 1 < Nodes.size())
      OS << " ; ";
  }
  OS << " ]";
}

void sandboxir::SchedBundle::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

sandboxir::SchedBundle *sandboxir::Scheduler::createBundle(
    const DmpVector<sandboxir::Value *> &SBInstrs) {
  assert(none_of(SBInstrs,
                 [this](auto *SBV) {
                   return ReadyList.contains(
                       DAG.getNode(cast<sandboxir::Instruction>(SBV)));
                 }) &&
         "Creating a Bundle for an instr contained in ReadyList!");
  assert(none_of(SBInstrs,
                 [this](sandboxir::Value *SBV) {
                   return DAG.getNode(cast<sandboxir::Instruction>(SBV))
                              ->getBundle() != nullptr;
                 }) &&
         "SBV already in a bundle!");
  assert(all_of(SBInstrs,
                [this](sandboxir::Value *SBV) {
                  return DAG.inView(
                      DAG.getNode(cast<sandboxir::Instruction>(SBV)));
                }) &&
         "Expected all bundle instrs in view!");
  auto SBPtr = std::unique_ptr<SchedBundle>(new SchedBundle(SBInstrs, *this));
  auto *SB = SBPtr.get();
  BundlePool.push_back(std::move(SBPtr));
#ifndef NDEBUG
  for (sandboxir::Value *SBV : SBInstrs)
    assert(DAG.getNode(cast<sandboxir::Instruction>(SBV))->getBundle() ==
               SB &&
           "Expected SchedBundle!");
#endif // NDEBUG
  return SB;
}

bool sandboxir::ReadyListContainer::contains(
    sandboxir::DependencyGraph::Node *N) const {
  auto *I = N->getInstruction();
  if (I->isTerminator())
    return Terminator == N;
  if (isa<sandboxir::PHINode>(I))
    return find(PHIList, N) != PHIList.end();
  return find(List, N) != List.end();
}

SmallVector<sandboxir::DependencyGraph::Node *>
sandboxir::ReadyListContainer::getContents() const {
  SmallVector<sandboxir::DependencyGraph::Node *> Nodes;
  if (Terminator != nullptr)
    Nodes.push_back(Terminator);
  for (const auto &L : {List, PHIList})
    for (auto *N : L)
      Nodes.push_back(N);
  return Nodes;
}

uint64_t sandboxir::ReadyListContainer::size() const {
  return PHIList.size() + List.size() + (int)(Terminator != nullptr);
}

bool sandboxir::ReadyListContainer::empty() const {
  return PHIList.empty() && List.empty() && Terminator == nullptr;
}

void sandboxir::ReadyListContainer::insert(
    sandboxir::DependencyGraph::Node *N) {
  assert(!N->isScheduled() && "Scheduled nodes should not be re-scheduled!");
  assert(N->allSuccsReady() && "Expected ready node!");
  assert(!contains(N) && "Already in ready list!");
  sandboxir::Instruction *I = N->getInstruction();
  N->setIsInReadyList();
  // It is the scheduler's responsibility to schedule BB Terminators after any
  // other instruction and PHIs before any other instruction.
  if (I->isTerminator()) {
    assert(Terminator == nullptr && "Terminator already in ready list!");
    Terminator = N;
    return;
  }
  // PHIs should be scheduled last.
  if (isa<sandboxir::PHINode>(I)) {
    PHIList.push_back(N);
    return;
  }
  // *pad nodes are kept separately as they need to be the first non-PHI.
  if (LLVM_UNLIKELY(I->isPad())) {
    assert(PadN == nullptr && "Expected a single pad in a BB!");
    PadN = N;
    return;
  }
  List.push_back(N);
}

void sandboxir::ReadyListContainer::remove(
    sandboxir::DependencyGraph::Node *N) {
  N->resetIsInReadyList();
  sandboxir::Instruction *I = N->getInstruction();
  if (I->isTerminator()) {
    Terminator = nullptr;
    return;
  }
  if (isa<sandboxir::PHINode>(I)) {
    PHIList.remove(N);
    return;
  }
  List.remove(N);
}

sandboxir::DependencyGraph::Node *sandboxir::ReadyListContainer::pop() {
  auto PopInternal = [this]() {
    if (Terminator != nullptr) {
      auto *Copy = Terminator;
      Terminator = nullptr;
      return Copy;
    }
    if (!List.empty()) {
      auto *FrontN = List.front();
      List.pop_front();
      return FrontN;
    }
    if (PadN != nullptr) {
      auto *PadNCopy = PadN;
      PadN = nullptr;
      return PadNCopy;
    }
    if (!PHIList.empty()) {
      auto *FrontN = PHIList.front();
      PHIList.pop_front();
      return FrontN;
    }
    llvm_unreachable("Empty ready list!");
  };

  auto *N = PopInternal();
  assert(N->allSuccsReady() && "Not ready!");
  N->resetIsInReadyList();
  return N;
}

void sandboxir::ReadyListContainer::clear() {
  for (auto &Lst : {PHIList, List})
    for_each(Lst, [](auto *N) { return N->resetIsInReadyList(); });
  PHIList.clear();
  List.clear();
  Terminator = nullptr;
}

#ifndef NDEBUG
void sandboxir::ReadyListContainer::dump(raw_ostream &OS) const {
  for (const auto &L : {PHIList, List}) {
    for (auto *ReadyN : L) {
      ReadyN->dump(OS, /*RegionOnly=*/true, /*PrintDeps=*/false);
      OS << "\n";
    }
  }
  if (PadN != nullptr) {
    PadN->dump(OS, /*RegionOnly=*/true, /*PrintDeps=*/false);
    OS << "\n";
  }
  if (Terminator != nullptr) {
    Terminator->dump(OS, /*RegionOnly=*/true, /*PrintDeps=*/false);
    OS << "\n";
  }
  OS << "\n";
}
void sandboxir::ReadyListContainer::dump() const { dump(dbgs()); }
#endif

void sandboxir::Scheduler::notifyNewViewNodes(
    const SmallPtrSet<DependencyGraph::Node *, 16> &NewViewNodes) {
  // Don't update if we are not scheduling.
  if (!Scheduling)
    return;
  // Nodes
  // +---+ -
  // |   | | Need update UnscheduledSuccs for deps -> New
  // |   | |
  // |---| -
  // |New| | Need to create UnscheduledSuccs for the first time
  // |---| -
  // |   | | Nothing to do
  // +---+ -
  //
  if (!TopDown) {
    // Go over all new nodes and: (i) set their counters, and (ii) increment the
    // counters of their predecessors in the old section.
    for (auto *N : NewViewNodes) {
      N->resetUnscheduledSuccs();
      // For each unscheduled successor increment N's counter.
      for (auto *SuccN : N->succs()) {
        if (!DAG.inView(SuccN))
          continue;
        if (SuccN->isScheduled())
          continue;
        N->incrementUnscheduledSuccs();
      }
      assert(!N->isScheduled());
      // For each predecessor not in NewViewNodes, increment its counter
      for (auto *PredN : N->preds()) {
        // Don't double count.
        if (NewViewNodes.contains(PredN))
          continue;
        if (!DAG.inView(PredN))
          continue;
        if (PredN->isScheduled())
          continue;
        if (ReadyList.contains(PredN))
          ReadyList.remove(PredN);
        PredN->incrementUnscheduledSuccs();
      }
    }
  } else {
    // Go over all new nodes and: (i) set their counters, and (ii) increment the
    // counters of their predecessors in the old section.
    for (auto *N : NewViewNodes) {
      N->resetUnscheduledSuccs();
      // For each unscheduled successor increment N's counter.
      for (auto *PredN : N->preds()) {
        if (!DAG.inView(PredN))
          continue;
        if (PredN->isScheduled())
          continue;
        N->incrementUnscheduledSuccs();
      }
      assert(!N->isScheduled());
      // For each predecessor not in NewViewNodes, increment its counter
      for (auto *SuccN : N->succs()) {
        // Don't double count.
        if (NewViewNodes.contains(SuccN))
          continue;
        if (!DAG.inView(SuccN))
          continue;
        if (SuccN->isScheduled())
          continue;
        if (ReadyList.contains(SuccN))
          ReadyList.remove(SuccN);
        SuccN->incrementUnscheduledSuccs();
      }
    }
  }
}

sandboxir::Scheduler::Scheduler(sandboxir::BasicBlock &SBBB,
                                AliasAnalysis &AA,
                                sandboxir::SBVecContext &Ctx)
    : DAG(Ctx, AA, AAQueriesBudget, this), SBBB(SBBB), Ctx(Ctx) {
  // Make sure all datastructures are reset.
  clearState();

  // Get notified about IR Instruction deletion.
  RemoveInstrCB = Ctx.registerRemoveInstrCallbackBB(
      SBBB, [this](sandboxir::Instruction *SBI) {
        if (SBI->getParent() == &this->SBBB)
          notifyRemove(SBI);
      });

  InsertInstrCB = Ctx.registerInsertInstrCallbackBB(
      SBBB, [this](sandboxir::Instruction *SBI) {
        if (SBI->getParent() == &this->SBBB)
          notifyInsert(SBI);
      });

  MoveInstrCB = Ctx.registerMoveInstrCallbackBB(
      SBBB, [this](sandboxir::Instruction *SBI, sandboxir::BasicBlock &SBBB,
                   const sandboxir::BBIterator &WhereIt) {
        if (SBI->getParent() == &this->SBBB)
          notifyMove(SBI, SBBB, WhereIt);
      });
  // NOTE: Don't forget to unregister them in the destructor!
}

sandboxir::Scheduler::~Scheduler() {
  Ctx.getTracker().accept();
  Ctx.unregisterRemoveInstrCallbackBB(SBBB, RemoveInstrCB);
  Ctx.unregisterInsertInstrCallbackBB(SBBB, InsertInstrCB);
  Ctx.unregisterMoveInstrCallbackBB(SBBB, MoveInstrCB);
}

void sandboxir::Scheduler::decrDepsReadyCounter(
    sandboxir::DependencyGraph::Node *N) {
  if (!TopDown) {
    for (auto *PredN : N->preds()) {
      if (!DAG.inView(PredN))
        continue;
      if (PredN->isScheduled())
        continue;
      PredN->decrementUnscheduledSuccs();
      if (PredN->allSuccsReady())
        ReadyList.insert(PredN);
    }
  } else {
    for (auto *SuccN : N->succs()) {
      if (!DAG.inView(SuccN))
        continue;
      if (SuccN->isScheduled())
        continue;
      SuccN->decrementUnscheduledSuccs();
      if (SuccN->allSuccsReady())
        ReadyList.insert(SuccN);
    }
  }
}

void sandboxir::Scheduler::scheduleAndUpdateReadyList(
    sandboxir::SchedBundle *B) {
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  DAG.verify();
#endif
  assert(!B->isScheduled() && "Expected unscheduled node");

  // Append `B` to the final schedule.
  sandboxir::BasicBlock *BB =
      cast<sandboxir::BasicBlock>(DAG.getView().to()->getParent());
  sandboxir::BasicBlock::iterator Where;
  if (!TopDown)
    Where =
        TopSB != nullptr
            ? TopSB->getTopI()->getIterator()
            : cast<sandboxir::Instruction>(DAG.getView().to())->getIterator();
  else
    Where = TopSB != nullptr
                ? std::next(TopSB->getBotI()->getIterator())
                : std::next(cast<sandboxir::Instruction>(DAG.getView().from())
                                ->getIterator());
  assert(all_of(B->nodes(), [this](auto *N) { return DAG.inView(N); }) &&
         "Expected all bundle nodes in view!");
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  if (!Ctx.getTracker().inRevert())
    Ctx.getScheduler(&SBBB)->getDAG().verify();
#endif
  B->cluster(Where, BB);
  if (!TopDown)
    TopSB = TopSB ? sandboxir::SchedBundle::getEarliest(B, TopSB) : B;
  else
    TopSB = TopSB ? sandboxir::SchedBundle::getLatest(B, TopSB) : B;

  // Decrement UnscheduledSuccs counter.
  // Also inserts to ready list if they become ready.
  for (auto *N : *B)
    decrDepsReadyCounter(N);

  // Finally mark all nodes in the bundle as "scheduled".
  B->setScheduled(true);

  ++NumScheduledBndls;
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  DAG.verify();
#endif
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  // verifySchedule();
  BB->verifyLLVMIR();
#endif
}

bool sandboxir::Scheduler::tryScheduleUntil(
    const DmpVector<sandboxir::Value *> &Instrs) {
  // Collect all DAG Nodes that correspond to Instrs. These are not being
  // scheduled right-away, instead they are deferred until all of them are
  // ready.
  DmpVector<DependencyGraph::Node *> Deferred;
  Deferred.reserve(Instrs.size());

  // Keep scheduling ready nodes and those corresponding to Instrs.
  while (!ReadyList.empty()) {
    auto *ReadyN = ReadyList.pop();
    assert(ReadyN->allSuccsReady() && "Not ready!");
    // We defer scheduling of any instuction in `Instrs` until we can schedule
    // all of them at the same time in a single scheduling bundle.
    bool Defer = any_of(Instrs, [ReadyN](sandboxir::Value *SBV) {
      return ReadyN->getInstruction() == SBV;
    });
    if (Defer) {
      // This is a node for DeferredNodes, so add it in.
      assert(find(Deferred, ReadyN) == Deferred.end() && "Duplicate!");
      assert(find(Instrs, ReadyN->getInstruction()) != Instrs.end() &&
             "A deferred instr must bin in `Instrs`!");
      Deferred.push_back(ReadyN);
      bool ReadyToScheduleDeferred = Deferred.size() == Instrs.size();
      if (ReadyToScheduleDeferred) {
        scheduleAndUpdateReadyList(createBundle(Instrs));
        return true;
      }
      continue;
    }
    // Schedule `ReadyN` as a standalone bundle.
    scheduleAndUpdateReadyList(createBundle(ReadyN));
  }
  assert(Deferred.size() != Instrs.size() &&
         "We should have succesfully scheduled and early-returned!");
  return false;
}

#ifndef NDEBUG
void sandboxir::Scheduler::verifyDirection(
    const DmpVector<sandboxir::Value *> &SBInstrs, bool TopDown) {
  // Check that we are scheduling according to `TopDown`.
  for (sandboxir::Value *SBV : SBInstrs) {
    auto *N = DAG.getNode(cast<sandboxir::Instruction>(SBV));
    assert(N != nullptr && "DAG not extended?");
    if (!TopDown) {
      // There should be no Bundle->Instrs dependencies.
      // Exception: when we are re-scheduling Instrs due to triangular deps.
      for (auto *PredN : N->preds()) {
        if (SchedBundle *PredBundle = PredN->getBundle()) {
          // If we are re-scheduling, ignore any bundles not in view.
          if (none_of(*PredBundle, [this](auto *BN) {
                return DAG.getView().contains(BN->getInstruction());
              }))
            continue;
          llvm_unreachable("B->Instrs dependency! Not bottom-up!");
        }
      }
    } else {
      // There should be no Instrs->Bundle dependencies.
      // Exception: when we are re-scheduling Instrs due to triangular deps.
      for (auto *SuccN : N->succs()) {
        if (SchedBundle *SuccBundle = SuccN->getBundle()) {
          // If we are re-scheduling, ignore any bundles not in view.
          if (none_of(*SuccBundle, [this](auto *BN) {
                return DAG.getView().contains(BN->getInstruction());
              }))
            continue;
          llvm_unreachable("Instrs->B dependency! Not top-down!");
        }
      }
    }
  }
}

void sandboxir::Scheduler::verifySchedule() {
  // Check that there are no gaps in the schedule.
  SmallVector<sandboxir::Instruction *> Instrs;
  Instrs.reserve(2 * BundlePool.size());
  for (auto &SBPtr : BundlePool) {
    for (auto *N : *SBPtr)
      Instrs.push_back(N->getInstruction());
  }
  sort(Instrs, [](auto *I1, auto *I2) { return I1->comesBefore(I2); });
  sandboxir::Instruction *LastI = Instrs.front();
  for (auto *I : drop_begin(Instrs)) {
    // When we erase instructions we erase the users before the definition, so
    // when reverting, we are adding the use before the def, which can create
    // gaps in the schedule.
    bool DisableCheck = Ctx.getTracker().inRevert();
    if (!DisableCheck && I->getPrevNode() != LastI) {
      errs() << "Gap in the schedule!\n";
      errs() << *LastI << " " << LastI << "\n";
      errs() << *I << " " << I << "\n";
      llvm_unreachable("Gap in the schedule!\n");
    }
    LastI = I;
  }

  // Limit overhead
  if (NumScheduledBndls < 50) {
    auto HasDep = [](SchedBundle &B1, SchedBundle &B2) -> bool {
      for (auto *N1 : B1)
        for (auto *N2 : B2)
          if (any_of(N1->succs(), [N2](auto *N) { return N == N2; }))
            return true;
      return false;
    };
    if (scheduleEmpty())
      return;
    for (SchedBundle *SB1 = TopSB; SB1 != nullptr; SB1 = SB1->getNext()) {
      for (SchedBundle *SB2 = SB1->getNext(); SB2 != nullptr;
           SB2 = SB2->getNext()) {
        // Check that there is no dependency B2->B1, except if B1 contains PHI
        // nodes.
        assert((isa<sandboxir::PHINode>((*SB1->begin())->getInstruction()) ||
                !HasDep(*SB2, *SB1)) &&
               "B2->B1 dependency!");
      }
    }
  }
}
#endif

void sandboxir::Scheduler::listSchedule(sandboxir::BasicBlock *BB) {
  clearState();
  DAG.extend(
      DmpVector<sandboxir::Value *>{&*BB->begin(), BB->getTerminator()});
  SmallVector<DependencyGraph::Node *> ReadyList;
  for (auto *RootN : DAG.getRoots())
    ReadyList.push_back(RootN);

  while (!ReadyList.empty()) {
    auto *N = ReadyList.back();
    ReadyList.pop_back();
    SchedBundle *SB = createBundle(N);
    scheduleAndUpdateReadyList(SB);
  }
}

sandboxir::Scheduler::BndlSchedState sandboxir::Scheduler::getBndlSchedState(
    const DmpVector<sandboxir::Value *> &Instrs) const {
  assert(!Instrs.empty() && "Expected non-empty bundle");
  bool PartiallyScheduled = false;
  bool FullyScheduled = true;
  for (auto *SBV : Instrs) {
    auto *N = DAG.getNode(cast<sandboxir::Instruction>(SBV));
    if (N != nullptr && N->isScheduled())
      PartiallyScheduled = true;
    else
      FullyScheduled = false;
  }
  if (FullyScheduled) {
    // If not all instrs in the bundle are in the same SchedBundle then this
    // should be considered as partially-scheduled, because we will need to
    // re-schedule.
    SchedBundle *SB =
        DAG.getNode(cast<sandboxir::Instruction>(Instrs[0]))->getBundle();
    assert(SB != nullptr && "FullyScheduled assumes that there is an SB!");
    if (any_of(drop_begin(Instrs), [this, SB](sandboxir::Value *SBV) {
          return DAG.getNode(cast<sandboxir::Instruction>(SBV))
                     ->getBundle() != SB;
        }))
      FullyScheduled = false;
  }
  return FullyScheduled       ? BndlSchedState::FullyScheduled
         : PartiallyScheduled ? BndlSchedState::PartiallyOrDifferentlyScheduled
                              : BndlSchedState::NoneScheduled;
}

void sandboxir::Scheduler::trimSchedule(
    const DmpVector<sandboxir::Value *> &Instrs) {
  sandboxir::Instruction *TopI =
      !TopDown ? TopSB->getTopI() : TopSB->getBotI();
  sandboxir::Instruction *LowestI =
      !TopDown ? sandboxir::VecUtils::getLowest(Instrs)
               : sandboxir::VecUtils::getHighest(Instrs);
#ifndef NDEBUG
  for (auto *I = LowestI, *E = TopI->getPrevNode(); I != E;
       I = !TopDown ? I->getPrevNode() : I->getNextNode())
    assert(DAG.getNode(I)->getBundle() && "Expect a valid Node and Bundle!");
#endif
  // We need to destroy the schedule bundles from LowestI all the way to the
  // top.
  for (auto *I = LowestI,
            *E = !TopDown ? TopI->getPrevNode() : TopI->getNextNode();
       I != E; I = !TopDown ? I->getPrevNode() : I->getNextNode()) {
    auto *N = DAG.getNode(I);
    // Skip SchedBundles that got erased in a previous iteration.
    if (auto *SB = N->getBundle())
      eraseBundle(SB);
  }
  // Reset the top section of the view, to create a fresh DAG for scheduling.
  DAG.trimView(LowestI, !TopDown);
  // Since we are scheduling NewRegion from scratch, we clear the ready lists.
  // The nodes currently in the list may not be ready after clearing the View.
  ReadyList.clear();
}

bool sandboxir::Scheduler::extendRegionAndUpdateReadyList(
    const DmpVector<sandboxir::Value *> &Instrs) {
#ifndef NDEBUG
  for (auto *SBV : Instrs) {
    auto *SBI = cast<sandboxir::Instruction>(SBV);
    auto *N = DAG.getNode(SBI);
    assert((N == nullptr || !N->isScheduled()) && getBundle(SBI) == nullptr &&
           "Expected non-scheduled instrs, perhaps trimSchedule() broken?");
  }
#endif
  auto ExtensionRegion = DAG.extend(Instrs);
#ifndef NDEBUG
  verifyDirection(Instrs, TopDown);
#endif
  // Add all ready instrs of ExtensionRegion to the ready list.
  auto AddToReadyList = [this,
                         &ExtensionRegion](sandboxir::Instruction &IRef) {
    sandboxir::Instruction *I = &IRef;
    (void)ExtensionRegion;
    assert((NumScheduledBndls == 0 || ExtensionRegion.contains(I) ||
            all_of(*TopSB,
                   [I, this](DependencyGraph::Node *N) {
                     if (TopDown)
                       return N->getInstruction()->comesBefore(I);
                     return I->comesBefore(N->getInstruction());
                   })) &&
           "Bad direction!");
    auto *N = DAG.getNode(I);
    if (N->allSuccsReady()) {
      assert(!N->isInReadyList() && "This is a fresh region, so none of these "
                                    "nodes should be in the ready list.");
      ReadyList.insert(N);
    }
  };
  if (!TopDown)
    for (sandboxir::Instruction &IRef : reverse(ExtensionRegion))
      AddToReadyList(IRef);
  else
    for (sandboxir::Instruction &IRef : ExtensionRegion)
      AddToReadyList(IRef);
  return true;
}

bool sandboxir::Scheduler::trySchedule(
    const DmpVector<sandboxir::Value *> &Instrs, bool TopDown) {
  Scheduling = true;
  if (!DirectionSet) {
    Scheduler::TopDown = TopDown;
    DirectionSet = true;
  } else {
    assert(TopDown == Scheduler::TopDown && "Wrong direction!");
  }
  assert(all_of(Instrs,
                [this](auto *SBV) {
                  return cast<sandboxir::Instruction>(SBV)->getParent() ==
                         &SBBB;
                }) &&
         "This scheduler is for a different BB!");
  if (!sandboxir::VecUtils::areInSameBB(Instrs))
    return false;

  auto SchedState = getBndlSchedState(Instrs);
  switch (SchedState) {
  case BndlSchedState::FullyScheduled:
    // Nothing to do.
    return true;
  case BndlSchedState::PartiallyOrDifferentlyScheduled:
    // If one or more instrs are already scheduled we need to destroy the
    // top-most part of the schedule that includes the instrs in the bundle and
    // re-schedule.
    trimSchedule(Instrs);
    [[fallthrough]];
  case BndlSchedState::NoneScheduled:
    if (DAG.getProjectedApproxSize(Instrs) > DAGSizeLimit)
      return false;
    extendRegionAndUpdateReadyList(Instrs);
    // Do the actual scheduling.
    bool Ret = tryScheduleUntil(Instrs);
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
    DAG.verify();
#endif
    return Ret;
  }
}

bool sandboxir::Scheduler::scheduleEmpty() const {
  assert(((NumScheduledBndls == 0 && TopSB == nullptr) ||
          (NumScheduledBndls != 0 && TopSB != nullptr)) &&
         "NumScheduledBndls == 0 if and only if TopSB == nullptr");
  return NumScheduledBndls == 0;
}

sandboxir::SchedBundle *
sandboxir::Scheduler::getBundle(sandboxir::Instruction *SBI) const {
  auto *N = DAG.getNode(SBI);
  return N != nullptr ? N->getBundle() : nullptr;
}

void sandboxir::Scheduler::eraseBundle(sandboxir::SchedBundle *SB) {
  assert(!scheduleEmpty() && "Expected non-empty!");
  assert(TopSB != nullptr && "Expected non-empty!");
  if (TopSB == SB)
    TopSB = NumScheduledBndls == 1 ? nullptr : TopSB->getNext();
  while (!SB->empty()) {
    auto *N = SB->back();
    N->removeFromBundle();
  }
  assert(SB->empty() && "All nodes should have been removed by now");
  --NumScheduledBndls;
}

void sandboxir::Scheduler::clearState() {
  TopSB = nullptr;
  DirectionSet = false;
  NumScheduledBndls = 0;
  for (sandboxir::DependencyGraph::Node *N : DAG.nodes())
    N->removeFromBundle();
  ReadyList.clear();
  DAG.resetView();
  Scheduling = false;
}

void sandboxir::Scheduler::notifyRemove(sandboxir::Instruction *SBI,
                                        bool CalledByDAG) {
  if (!DAG.trackingEnabled())
    return;
  auto *N = DAG.getNode(SBI);
  if (N == nullptr) // TODO: Check why this happens.
    return;
  // We don't maintain the scheduler when reverting since the vectorizer always
  // starts with a fresh instance.
  if (Scheduling && !Ctx.getTracker().inRevert()) {
    if (!N->isScheduled()) {
      decrDepsReadyCounter(N);
    }
    // WARNING: ReadyList.remove(N) must be called before DAG.erase(N) because
    // we need to access N's contents!
    ReadyList.remove(N);
  }
  if (!CalledByDAG)
    DAG.erase(N, /*CalledByScheduler=*/true);
}

void sandboxir::Scheduler::notifyInsert(sandboxir::Instruction *I) {
  if (!DAG.trackingEnabled())
    return;
  // Create a new node for `SBI` and update dependencies.
  DAG.notifyInsert(I);
  // We don't maintain the scheduler when reverting since the vectorizer always
  // starts with a fresh instance.
  if (Ctx.getTracker().inRevert())
    return;
  // Nothing to do if we haven't started scheduling.
  if (!Scheduling)
    return;
  auto *N = DAG.getNode(I);

  // `SBI` could be above TopSB or below. We trust the user-specified location
  // and update the scheduler to reflect this. However, the scheduler requires a
  // valid top-of-schedule (TopSB).
  //            1.Above   2.InSchedule  3.Below
  //            -------   ------------  -------
  //              `I`
  //  TopSB: -
  //         |
  //         |                `I`
  //         |
  //         -
  // Schedule^^                           `I`
  //
  if (TopSB == nullptr || (!TopDown ? I->comesBefore(TopSB->getTopI())
                                    : I->comesAfter(TopSB->getBotI()))) {
    // 1. Above TopSB in the direction of the schedule.
    // Now schedule all instructions in program order until `I`.
    sandboxir::Instruction *ToI;
    sandboxir::Instruction *FromI;
    if (!TopDown) {
      ToI = I->getPrevNode();
      FromI = TopSB ? TopSB->getTopI()->getPrevNode() : I;
    } else {
      ToI = I->getNextNode();
      FromI = TopSB ? TopSB->getBotI()->getNextNode() : I;
    }
    for (sandboxir::Instruction *RunnerI = FromI, *E = ToI; RunnerI != E;
         RunnerI = !TopDown ? RunnerI->getPrevNode() : RunnerI->getNextNode()) {
      auto *RunnerN = DAG.getNode(RunnerI);
      assert(RunnerN != nullptr && "Forgot to call DAG.extend() ?");
      // Make sure it is removed from the ready list to avoid trying to schedule
      // it twice.
      ReadyList.remove(RunnerN);
      scheduleAndUpdateReadyList(createBundle(RunnerN));
    }
  } else {
    auto IsInSchedule = [this](sandboxir::Instruction *I) {
      auto *PrevI = !TopDown ? I->getPrevNode() : I->getNextNode();
      if (PrevI == nullptr)
        return false;
      auto *PrevN = DAG.getNode(PrevI);
      if (!DAG.inView(PrevN))
        return false;
      return PrevN->isScheduled();
    };
    assert(DAG.inView(N) && "DAG view not updated!");
    if (IsInSchedule(I)) {
      // 2. Between TopSB and the first bundle in the schedule.
      // Create a bundle for it and act as if it was scheduled by the scheduler,
      // updating the UnscheduledSuccs of dependent nodes.
      auto *NewSB = createBundle(N);
      NewSB->setScheduled(true);
      decrDepsReadyCounter(N);
      ++NumScheduledBndls;
    } else {
      // 3. Below the schedule, in the opposite direction of the schedule.
      // Reschedule from SBI to TopSB to avoid gaps in the schedule.
      // TODO: is this really needed?
      auto *TopI = TopSB->getTopI();
      startFresh(&SBBB);
      bool Success = trySchedule({I}, TopDown);
      (void)Success;
      assert(Success && "Failed to schedule initial instruction!");
      Success = tryScheduleUntil({TopI});
      assert(Success && "Failed to schedule topmost instruction!");
    }
  }

#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  DAG.verify();
  verifySchedule();
#endif
}

void sandboxir::Scheduler::notifyMove(sandboxir::Instruction *SBI,
                                      sandboxir::BasicBlock &SBBB,
                                      const sandboxir::BBIterator &WhereIt) {
  if (!DAG.trackingEnabled())
    return;
  DAG.notifyMoveInstr(SBI, WhereIt, &SBBB);
}

void sandboxir::Scheduler::accept() { Ctx.getTracker().accept(); }

void sandboxir::Scheduler::startFresh(sandboxir::BasicBlock *SBBB) {
  DAG.enableTracking();
  clearState();
  if (!Ctx.getTracker().tracking())
    Ctx.getTracker().start(SBBB);
}

void sandboxir::Scheduler::revert() {
  DAG.resetView();
  Ctx.getTracker().revert();
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  DAG.verify();
#endif
  clearState();
}

#ifndef NDEBUG
void sandboxir::Scheduler::dump(raw_ostream &OS) const {
  OS << "ReadyList:\n";
  ReadyList.dump(OS);

  OS << "\nSchedule:\n";
  if (TopSB == nullptr) {
    OS << "Empty? TopSB == null\n";
    return;
  }
  auto GetTopmostSB = [this]() {
    if (!TopDown)
      return TopSB;
    auto *TopmostSB = TopSB;
    for (auto *SB = TopSB; SB != nullptr; SB = SB->getPrev())
      TopmostSB = SB;
    return TopmostSB;
  };
  auto *SB = GetTopmostSB();
  for (auto Cnt : seq<size_t>(0, NumScheduledBndls)) {
    (void)Cnt;
    if (SB == nullptr) {
      OS << "NULL SB, Error!\n";
      return;
    }
    OS << *SB << "\n";
    SB = SB->getNext();
  }
}
#endif
