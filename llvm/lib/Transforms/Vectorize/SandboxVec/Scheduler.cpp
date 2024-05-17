//===- Scheduler.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"
#include <algorithm>

using namespace llvm;

static cl::opt<int>
    AAQueriesBudget("sbvec-dependency-graph-aa-budget", cl::init(10),
                    cl::Hidden,
                    cl::desc("Limits the number of AA quries performed per "
                             "region to reduce compilation time."));

SchedBundle::SchedBundle(const SBValBundle &SBInstrs, Scheduler &Sched)
    : Sched(&Sched) {
  auto &DAG = Sched.getDAG();
  Nodes.reserve(SBInstrs.size());
  for (auto *SBV : SBInstrs) {
    auto *N = DAG.getNode(cast<SBInstruction>(SBV));
    assert(N != nullptr && "No node found for `I`!");
    N->setBundle(this);
    Nodes.push_back(N);
  }
}

SchedBundle::SchedBundle(SmallVector<DependencyGraph::Node *, 4> &&Nodes,
                         Scheduler &Sched)
    : Nodes(std::move(Nodes)), Sched(&Sched) {}

SchedBundle::SchedBundle(const SmallVector<DependencyGraph::Node *, 4> &Nodes,
                         Scheduler &Sched)
    : Nodes(Nodes), Sched(&Sched) {}

SchedBundle::~SchedBundle() {
  for (auto *N : Nodes)
    N->removeFromBundle();
}

void SchedBundle::cluster() {
  // Cluster them at the bottom instruction.
  cluster(std::next(bottom()->getInstruction()->getIterator()),
          bottom()->getInstruction()->getParent());
}

void SchedBundle::cluster(SBBasicBlock::iterator BeforeIt,
                          SBBasicBlock *SBBB) {
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
    SBInstruction *SBI = N->getInstruction();
    if (SBI->getIterator() == BeforeIt)
      continue;
    SBI->moveBefore(*SBI->getParent(), BeforeIt);
  }
}

SBInstruction *SchedBundle::getTopI() const {
  auto *TopI = Nodes.front()->getInstruction();
  for (const auto *N : drop_begin(Nodes)) {
    auto *I = N->getInstruction();
    if (I->comesBefore(TopI))
      TopI = I;
  }
  return TopI;
}

SBInstruction *SchedBundle::getBotI() const {
  auto *BotI = Nodes.front()->getInstruction();
  for (const auto *N : drop_begin(Nodes)) {
    auto *I = N->getInstruction();
    if (BotI->comesBefore(I))
      BotI = I;
  }
  return BotI;
}

Scheduler &SchedBundle::getScheduler() const { return *Sched; }

SchedBundle *SchedBundle::getPrev() {
  if (!Sched->isTopDown() && this == Sched->getTop())
    return nullptr;
  for (SBInstruction *I = getTopI()->getPrevNode(); I != nullptr;
       I = I->getPrevNode())
    if (auto *SB = Sched->getBundle(I))
      return SB;
  return nullptr;
}

SchedBundle *SchedBundle::getNext() {
  if (Sched->isTopDown() && this == Sched->getTop())
    return nullptr;
  for (SBInstruction *I = getBotI()->getNextNode(); I != nullptr;
       I = I->getNextNode())
    if (auto *SB = Sched->getBundle(I))
      return SB;
  return nullptr;
}

bool SchedBundle::comesBefore(SchedBundle *B) {
  return (*begin())->getInstruction()->comesBefore(
      (*B->begin())->getInstruction());
}

DependencyGraph::Node *SchedBundle::top() const {
  // TODO: Perhaps cache this?
  return *std::min_element(Nodes.begin(), Nodes.end(), [](auto *N1, auto *N2) {
    return N1->getInstruction()->comesBefore(N2->getInstruction());
  });
}

DependencyGraph::Node *SchedBundle::bottom() const {
  // TODO: Perhaps cache this?
  return *std::max_element(Nodes.begin(), Nodes.end(), [](auto *N1, auto *N2) {
    return N1->getInstruction()->comesBefore(N2->getInstruction());
  });
}

bool SchedBundle::allSuccsReady() const {
  return all_of(Nodes, [](auto *N) { return N->allSuccsReady(); });
}

bool SchedBundle::isScheduled() const {
#ifndef NDEBUG
  assert(!Nodes.empty() && "Empty Bundle!");
  auto IsScheduled = [](auto *N) { return N->isScheduled(); };
  assert((all_of(Nodes, IsScheduled) || none_of(Nodes, IsScheduled)) &&
         "Expected either all or none");
#endif // NDEBUg
  return Nodes.back()->isScheduled();
}

void SchedBundle::setScheduled(bool Val) {
  assert(!Nodes.empty() && "Empty Bundle!");
  for (auto *N : Nodes)
    N->setScheduled(Val);
}

void SchedBundle::eraseFromParent() { Sched->eraseBundle(this); }

void SchedBundle::remove(DependencyGraph::Node *N) {
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

bool SchedBundle::contains(const DependencyGraph::Node *N) const {
  // TODO: Make this constant time.
  return find(Nodes, N) != Nodes.end();
}

#ifndef NDEBUG
void SchedBundle::dump(raw_ostream &OS) const {
  OS << "[";
  for (auto [Idx, N] : enumerate(Nodes)) {
    N->dump(OS, /*RegionOnly=*/true, /*PrintDeps=*/false);
    if (Idx + 1 < Nodes.size())
      OS << " ; ";
  }
  OS << " ]";
}

void SchedBundle::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SchedBundle *Scheduler::createBundle(const SBValBundle &SBInstrs) {
  assert(none_of(SBInstrs,
                 [this](auto *SBV) {
                   return ReadyList.contains(
                       DAG.getNode(cast<SBInstruction>(SBV)));
                 }) &&
         "Creating a Bundle for an instr contained in ReadyList!");
  assert(
      none_of(SBInstrs,
              [this](SBValue *SBV) {
                return DAG.getNode(cast<SBInstruction>(SBV))->getBundle() !=
                       nullptr;
              }) &&
      "SBV already in a bundle!");
  auto SBPtr = std::unique_ptr<SchedBundle>(new SchedBundle(SBInstrs, *this));
  auto *SB = SBPtr.get();
  BundlePool.push_back(std::move(SBPtr));
#ifndef NDEBUG
  for (SBValue *SBV : SBInstrs)
    assert(DAG.getNode(cast<SBInstruction>(SBV))->getBundle() == SB &&
           "Expected SchedBundle!");
#endif // NDEBUG
  return SB;
}

bool ReadyListContainer::contains(DependencyGraph::Node *N) const {
  auto *I = N->getInstruction();
  if (I->isTerminator())
    return Terminator == N;
  if (isa<SBPHINode>(I))
    return find(PHIList, N) != PHIList.end();
  return find(List, N) != List.end();
}

SmallVector<DependencyGraph::Node *>
ReadyListContainer::getContents() const {
  SmallVector<DependencyGraph::Node *> Nodes;
  if (Terminator != nullptr)
    Nodes.push_back(Terminator);
  for (const auto &L : {List, PHIList})
    for (auto *N : L)
      Nodes.push_back(N);
  return Nodes;
}

uint64_t ReadyListContainer::size() const {
  return PHIList.size() + List.size() + (int)(Terminator != nullptr);
}

bool ReadyListContainer::empty() const {
  return PHIList.empty() && List.empty() && Terminator == nullptr;
}

void ReadyListContainer::insert(DependencyGraph::Node *N) {
  assert(!N->isScheduled() && "Scheduled nodes should not be re-scheduled!");
  assert(N->allSuccsReady() && "Expected ready node!");
  assert(!contains(N) && "Already in ready list!");
  SBInstruction *I = N->getInstruction();
  N->setIsInReadyList();
  // It is the scheduler's responsibility to schedule BB Terminators after any
  // other instruction and PHIs before any other instruction.
  if (I->isTerminator()) {
    assert(Terminator == nullptr && "Terminator already in ready list!");
    Terminator = N;
    return;
  }
  // PHIs should be scheduled last.
  if (isa<SBPHINode>(I)) {
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

void ReadyListContainer::remove(DependencyGraph::Node *N) {
  N->resetIsInReadyList();
  SBInstruction *I = N->getInstruction();
  if (I->isTerminator()) {
    Terminator = nullptr;
    return;
  }
  if (isa<SBPHINode>(I)) {
    PHIList.remove(N);
    return;
  }
  List.remove(N);
}

DependencyGraph::Node *ReadyListContainer::pop() {
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
  N->resetIsInReadyList();
  return N;
}

void ReadyListContainer::clear() {
  for (auto &Lst : {PHIList, List})
    for_each(Lst, [](auto *N) { return N->resetIsInReadyList(); });
  PHIList.clear();
  List.clear();
  Terminator = nullptr;
}

#ifndef NDEBUG
void ReadyListContainer::dump(raw_ostream &OS) const {
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
void ReadyListContainer::dump() const { dump(dbgs()); }
#endif

void Scheduler::notifyNewDep(DependencyGraph::Node *FromN,
                                   DependencyGraph::Node *ToN) {
  if (FromN->isScheduled() || ToN->isScheduled())
    return;
  // Update the UnscheduledSuccs for either this or N, depending on direction.
  auto *WhichN = TopDown ? ToN : FromN;
  WhichN->incrementUnscheduledSuccs();
}

void Scheduler::notifyNewNode(DependencyGraph::Node *N,
                                    bool OnlyDefUse, bool UpdateDepNodes) {
  // If N's dependency DepN is in the ViewRange, then increment the counter.
  auto IsScheduled = [this](auto *DepN) {
    return !DAG.getView().contains(DepN->getInstruction()) ||
           DepN->isScheduled();
  };
  if (OnlyDefUse) {
    // Update N's UnscheduledSuccs counter by checking the schedduling status
    // of operands/users (this excludes mem deps).
    if (!TopDown) {
      for (auto *SuccU : N->getInstruction()->users()) {
        if (auto *SuccI = dyn_cast<SBInstruction>(SuccU)) {
          auto *SuccN = DAG.getNode(SuccI);
          if (SuccN != nullptr && !IsScheduled(SuccN))
            N->incrementUnscheduledSuccs();
        }
      }
      if (UpdateDepNodes) {
        assert(!N->isScheduled());
        for (SBValue *OpV : N->getInstruction()->operands()) {
          // OpV may be null for newly created constant/poison operands.
          if (auto *OpI = dyn_cast_or_null<SBInstruction>(OpV)) {
            auto *OpN = DAG.getNode(OpI);
            if (OpN != nullptr && !IsScheduled(OpN))
              OpN->incrementUnscheduledSuccs();
          }
        }
      }
    } else {
      for (SBValue *OpV : N->getInstruction()->operands()) {
        // OpV may be null for newly created constant/poison operands.
        if (auto *OpI = dyn_cast_or_null<SBInstruction>(OpV)) {
          auto *SuccN = DAG.getNode(OpI);
          if (SuccN != nullptr && !IsScheduled(SuccN))
            N->incrementUnscheduledSuccs();
        }
      }
      if (UpdateDepNodes) {
        assert(!N->isScheduled());
        for (SBUser *U : N->getInstruction()->users()) {
          if (auto *UI = dyn_cast<SBInstruction>(U)) {
            auto *UN = DAG.getNode(UI);
            if (UN != nullptr && !IsScheduled(UN))
              UN->incrementUnscheduledSuccs();
          }
        }
      }
    }
  } else {
    // Update N's UnscheduledSuccs counter by checking the schedduling status
    // of deps (this includes def-use/use-def deps).
    if (!TopDown) {
      for (auto *SuccN : N->succs())
        if (!IsScheduled(SuccN))
          N->incrementUnscheduledSuccs();
      if (UpdateDepNodes) {
        assert(!N->isScheduled());
        for (auto *PredN : N->preds())
          if (!IsScheduled(PredN))
            PredN->incrementUnscheduledSuccs();
      }
    } else {
      for (auto *PredN : N->preds())
        if (!IsScheduled(PredN))
          N->incrementUnscheduledSuccs();
      if (UpdateDepNodes) {
        assert(!N->isScheduled());
        for (auto *SuccN : N->succs())
          if (!IsScheduled(SuccN))
            SuccN->incrementUnscheduledSuccs();
      }
    }
  }
}

void Scheduler::notifyEraseNode(DependencyGraph::Node *N) {
  if (!N->isScheduled())
    N->decrementUnscheduledSuccs();
}

Scheduler::Scheduler(SBBasicBlock &SBBB, AliasAnalysis &AA,
                                 SBContext &Ctxt)
    : DAG(Ctxt, AA, this, AAQueriesBudget), SBBB(SBBB), Ctxt(Ctxt) {
  // Make sure all datastructures are reset.
  clearState();

  // Get notified about IR Instruction deletion.
  RemoveInstrCB = Ctxt.registerRemoveInstrCallbackBB(
      SBBB, [this](SBInstruction *SBI) {
        if (SBI->getParent() == &this->SBBB)
          notifyRemove(SBI);
      });

  InsertInstrCB = Ctxt.registerInsertInstrCallbackBB(
      SBBB, [this](SBInstruction *SBI) {
        if (SBI->getParent() == &this->SBBB)
          notifyInsert(SBI);
      });

  MoveInstrCB = Ctxt.registerMoveInstrCallbackBB(
      SBBB, [this](SBInstruction *SBI, SBBasicBlock &SBBB,
                     const SBBBIterator &WhereIt) {
        if (SBI->getParent() == &this->SBBB)
          notifyMove(SBI, SBBB, WhereIt);
      });
  // NOTE: Don't forget to unregister them in the destructor!
}

Scheduler::~Scheduler() {
  Ctxt.getTracker().accept();
  Ctxt.unregisterRemoveInstrCallbackBB(SBBB, RemoveInstrCB);
  Ctxt.unregisterInsertInstrCallbackBB(SBBB, InsertInstrCB);
  Ctxt.unregisterMoveInstrCallbackBB(SBBB, MoveInstrCB);
}

void Scheduler::tryAddPredsToReadyList(DependencyGraph::Node *N) {
  SmallPtrSet<DependencyGraph::Node *, 2> Visited;
  auto TryAddToReadyList = [this, &Visited](DependencyGraph::Node *DepN) {
    // Skip already scheduled.
    if (DepN->getBundle())
      return;
    if (!DAG.inView(DepN))
      return;
    // Skip duplicates.
    if (!Visited.insert(DepN).second)
      return;
    // This can happen when a newly created instruction is inserted into the
    // BB while the scheduler already has instructions in the ready list.
    if (DepN->isInReadyList())
      return;
    ReadyList.insert(DepN);
  };
  if (!TopDown) {
    for (auto *PredN : N->preds()) {
      if (!PredN->allSuccsReady()) // TODO: Is this correct?
        break;
      TryAddToReadyList(PredN);
    }
  } else {
    for (auto *SuccN : N->succs()) {
      if (!SuccN->allSuccsReady()) // TODO: Is this correct?
        break;
      TryAddToReadyList(SuccN);
    }
  }
}

void Scheduler::scheduleAndUpdateReadyList(SchedBundle *B) {
  assert(!B->isScheduled() && "Expected unscheduled node");
  B->setScheduled(true);

  // Append `B` to the final schedule.
  SBBasicBlock *BB = cast<SBBasicBlock>(DAG.getView().to()->getParent());
  SBBasicBlock::iterator Where;
  if (!TopDown)
    Where = TopSB != nullptr
                ? TopSB->getTopI()->getIterator()
                : cast<SBInstruction>(DAG.getView().to())->getIterator();
  else
    Where =
        TopSB != nullptr
            ? std::next(TopSB->getBotI()->getIterator())
            : std::next(
                  cast<SBInstruction>(DAG.getView().from())->getIterator());
  B->cluster(Where, BB);
  if (!TopDown)
    TopSB = TopSB ? SchedBundle::getEarliest(B, TopSB) : B;
  else
    TopSB = TopSB ? SchedBundle::getLatest(B, TopSB) : B;

  // Update UnscheduledSuccs counter.
  for (auto *N : *B) {
    if (!TopDown)
      for (auto *PredN : N->preds())
        PredN->decrementUnscheduledSuccs();
    else
      for (auto *SuccN : N->succs())
        SuccN->decrementUnscheduledSuccs();
  }

  ++NumScheduledBndls;

  for (auto *N : *B)
    tryAddPredsToReadyList(N);
#ifndef NDEBUG
  // verifySchedule();
  SBValBundle Instrs;
  for (auto *N : *B)
    Instrs.push_back(N->getInstruction());
  BB->verifyIR();
#endif
}

bool Scheduler::tryScheduleUntil(const SBValBundle &Instrs) {
  // Collect all DAG Nodes that correspond to Instrs. These are not being
  // scheduled right-away, instead they are deferred until all of them are
  // ready.
  Bundle<DependencyGraph::Node *> Deferred;
  Deferred.reserve(Instrs.size());

  // Keep scheduling ready nodes and those corresponding to Instrs.
  while (!ReadyList.empty()) {
    auto *ReadyN = ReadyList.pop();
    // We defer scheduling of any instuction in `Instrs` until we can schedule
    // all of them at the same time in a single scheduling bundle.
    bool Defer = any_of(Instrs, [ReadyN](SBValue *SBV) {
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
void Scheduler::verifyDirection(const SBValBundle &SBInstrs,
                                      bool TopDown) {
  // Check that we are scheduling according to `TopDown`.
  for (SBValue *SBV : SBInstrs) {
    auto *N = DAG.getNode(cast<SBInstruction>(SBV));
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

void Scheduler::verifySchedule() {
  // Check that there are no gaps in the schedule.
  SmallVector<SBInstruction *> Instrs;
  Instrs.reserve(2 * BundlePool.size());
  for (auto &SBPtr : BundlePool) {
    for (auto *N : *SBPtr)
      Instrs.push_back(N->getInstruction());
  }
  sort(Instrs, [](auto *I1, auto *I2) { return I1->comesBefore(I2); });
  SBInstruction *LastI = Instrs.front();
  for (auto *I : drop_begin(Instrs)) {
    // When we erase instructions we erase the users before the definition, so
    // when reverting, we are adding the use before the def, which can create
    // gaps in the schedule.
    bool DisableCheck = Ctxt.getTracker().inRevert();
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
        assert((isa<SBPHINode>((*SB1->begin())->getInstruction()) ||
                !HasDep(*SB2, *SB1)) &&
               "B2->B1 dependency!");
      }
    }
  }
}
#endif

void Scheduler::listSchedule(SBBasicBlock *BB) {
  clearState();
  DAG.extend(SBValBundle{&*BB->begin(), BB->getTerminator()});
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

Scheduler::BndlSchedState
Scheduler::getBndlSchedState(const SBValBundle &Instrs) const {
  assert(!Instrs.empty() && "Expected non-empty bundle");
  bool PartiallyScheduled = false;
  bool FullyScheduled = true;
  for (auto *SBV : Instrs) {
    auto *N = DAG.getNode(cast<SBInstruction>(SBV));
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
        DAG.getNode(cast<SBInstruction>(Instrs[0]))->getBundle();
    assert(SB != nullptr && "FullyScheduled assumes that there is an SB!");
    if (any_of(drop_begin(Instrs), [this, SB](SBValue *SBV) {
          return DAG.getNode(cast<SBInstruction>(SBV))->getBundle() != SB;
        }))
      FullyScheduled = false;
  }
  return FullyScheduled       ? BndlSchedState::FullyScheduled
         : PartiallyScheduled ? BndlSchedState::PartiallyOrDifferentlyScheduled
                              : BndlSchedState::NoneScheduled;
}

void Scheduler::trimSchedule(const SBValBundle &Instrs) {
  SBInstruction *TopI = !TopDown ? TopSB->getTopI() : TopSB->getBotI();
  SBInstruction *LowestI =
      !TopDown ? SBUtils::getLowest(Instrs) : SBUtils::getHighest(Instrs);
#ifndef NDEBUG
  for (auto *I = LowestI, *E = TopI->getPrevNode(); I != E;
       I = !TopDown ? I->getPrevNode() : I->getNextNode())
    assert(DAG.getNode(I)->getBundle() && "Expect a valid Node and Bundle!");
#endif
  // We need to destroy the schedule from LowestI all the way to the
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
  // Update TopSB to reflect the changes.
  if (SBInstruction *NewTopI =
          !TopDown ? LowestI->getNextNode() : LowestI->getPrevNode()) {
    auto *NewTopN = DAG.getNode(NewTopI);
    TopSB = NewTopN ? NewTopN->getBundle() : nullptr;
  } else
    TopSB = nullptr;
}

bool Scheduler::extendRegionAndUpdateReadyList(
    const SBValBundle &Instrs) {
#ifndef NDEBUG
  for (auto *SBV : Instrs) {
    auto *SBI = cast<SBInstruction>(SBV);
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
  auto AddToReadyList = [this, &ExtensionRegion](SBInstruction &IRef) {
    SBInstruction *I = &IRef;
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
    for (SBInstruction &IRef : reverse(ExtensionRegion))
      AddToReadyList(IRef);
  else
    for (SBInstruction &IRef : ExtensionRegion)
      AddToReadyList(IRef);
  return true;
}

bool Scheduler::trySchedule(const SBValBundle &Instrs, bool TopDown) {
  if (!DirectionSet) {
    Scheduler::TopDown = TopDown;
    DirectionSet = true;
  } else {
    assert(TopDown == Scheduler::TopDown && "Wrong direction!");
  }
  assert(all_of(Instrs,
                [this](auto *SBV) {
                  return cast<SBInstruction>(SBV)->getParent() == &SBBB;
                }) &&
         "This scheduler is for a different BB!");
  if (!SBUtils::areInSameBB(Instrs))
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
    extendRegionAndUpdateReadyList(Instrs);
    // Do the actual scheduling.
    return tryScheduleUntil(Instrs);
  }
}

bool Scheduler::scheduleEmpty() const {
  assert(((NumScheduledBndls == 0 && TopSB == nullptr) ||
          (NumScheduledBndls != 0 && TopSB != nullptr)) &&
         "NumScheduledBndls == 0 if and only if TopSB == nullptr");
  return NumScheduledBndls == 0;
}

SchedBundle *Scheduler::getBundle(SBInstruction *SBI) const {
  auto *N = DAG.getNode(SBI);
  return N != nullptr ? N->getBundle() : nullptr;
}

void Scheduler::eraseBundle(SchedBundle *SB) {
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

void Scheduler::clearState() {
  TopSB = nullptr;
  DirectionSet = false;
  NumScheduledBndls = 0;
  for (DependencyGraph::Node *N : DAG.nodes())
    N->removeFromBundle();
  ReadyList.clear();
  DAG.resetView();
}

void Scheduler::notifyRemove(SBInstruction *SBI) {
  // WARNING: ReadyList.remove(N) must be called before DAG.erase(N) because we
  // need to access N's contents!
  auto *N = DAG.getNode(SBI);
  if (N != nullptr) {
    ReadyList.remove(N);
  }
  DAG.notifyRemove(SBI);
}

void Scheduler::notifyInsert(SBInstruction *SBI) {
  // Create a new node for `SBI` and update dependencies.
  DAG.notifyInsert(SBI);

  // `SBI` could be above TopSB or below. We trust the user-specified location
  // and update the scheduler to reflect this. However, the scheduler requires a
  // valid top-of-schedule (TopSB).
  if (TopSB == nullptr || (!TopDown ? SBI->comesBefore(TopSB->getTopI())
                                    : SBI->comesAfter(TopSB->getBotI()))) {
    // If `SBI` is above TopSB, then schedule all instrs in (TopSB, SBI] and
    // update TopSB.
    SBInstruction *ToI;
    SBInstruction *FromI;
    if (!TopDown) {
      ToI = SBI->getPrevNode();
      FromI = TopSB ? TopSB->getTopI()->getPrevNode() : SBI;
    } else {
      ToI = SBI->getNextNode();
      FromI = TopSB ? TopSB->getBotI()->getNextNode() : SBI;
    }
    for (SBInstruction *RunnerI = FromI, *E = ToI; RunnerI != E;
         RunnerI = !TopDown ? RunnerI->getPrevNode() : RunnerI->getNextNode()) {
      auto *RunnerN = DAG.getNode(RunnerI);
      assert(RunnerN != nullptr && "Forgot to call DAG.extend() ?");
      // Make sure it is removed from the ready list to avoid trying to schedule
      // it twice.
      ReadyList.remove(RunnerN);
      scheduleAndUpdateReadyList(createBundle(RunnerN));
    }
  } else {
    // If `I` is before TopSB, then create a bundle for `I`, don't change TopSB.
    auto *NewSB = createBundle(DAG.getNode(SBI));
    NewSB->setScheduled(true);
    ++NumScheduledBndls;
  }

#ifndef NDEBUG
  verifySchedule();
#endif
}

void Scheduler::notifyMove(SBInstruction *SBI, SBBasicBlock &SBBB,
                                 const SBBBIterator &WhereIt) {
  DAG.notifyMoveInstr(SBI, WhereIt, &SBBB);
}

void Scheduler::accept() { Ctxt.getTracker().accept(); }

void Scheduler::startFresh(SBBasicBlock *SBBB) {
  clearState();
  if (!Ctxt.getTracker().tracking())
    Ctxt.getTracker().start(SBBB);
}

void Scheduler::revert() {
  DAG.resetView();
  Ctxt.getTracker().revert();
#ifdef SBVEC_EXPENSIVE_CHECKS
  DAG.verify();
#endif
  clearState();
}

#ifndef NDEBUG
void Scheduler::dump(raw_ostream &OS) const {
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
