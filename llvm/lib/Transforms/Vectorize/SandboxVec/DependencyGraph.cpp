//===- DependencyGraph.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/DependencyGraph.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"
#include <cstdint>

using namespace llvm;
using namespace llvm::PatternMatch;

template <typename OpItT>
DependencyGraph::PredSuccIteratorTemplate<
    OpItT>::PredSuccIteratorTemplate(OpItT OpIt,
                                     SetVector<Node *>::iterator OtherIt,
                                     Node *N, DependencyGraph &Parent)
    : OpIt(OpIt), OtherIt(OtherIt), N(N), Parent(&Parent) {
  SkipOpIt(/*PreInc=*/false);
}
template <typename OpItT>
void DependencyGraph::PredSuccIteratorTemplate<OpItT>::SkipOpIt(
    bool PreInc) {
  OpItT EndIt;
  if constexpr (std::is_same<OpItT, SBUser::op_iterator>::value)
    // PredIterator
    EndIt = N->I->op_end();
  else
    // SuccIterator
    EndIt = N->I->user_end();
  if (PreInc) {
    if (OpIt != EndIt) {
      ++OpIt;
    } else {
#ifndef NDEBUG
      SetVector<Node *>::iterator End;
      if constexpr (std::is_same<OpItT, SBUser::op_iterator>::value)
        // PredIterator
        End = N->Preds.end();
      else
        // SuccIterator
        End = N->Succs.end();
      assert(OtherIt != End && "Already at end!");
#endif
      ++OtherIt;
    }
  }
  // Skip non-instructions and instructions that have not been mapped to a
  // Node. Note: There is no need to skip OtherIt, because that points to
  // Node.
  auto ShouldSkip = [this](auto OpIt) {
    SBValue *SBV = *OpIt;
    if (SBV == nullptr)
      return true;
    if (!isa<SBInstruction>(SBV))
      return true;
    SBValue *OpV = *OpIt;
    if (!Parent->getNode(cast<SBInstruction>(OpV)))
      return true;
    return false;
  };
  while (OpIt != EndIt && ShouldSkip(OpIt))
    ++OpIt;
}

template <typename OpItT>
typename DependencyGraph::PredSuccIteratorTemplate<OpItT>::value_type
DependencyGraph::PredSuccIteratorTemplate<OpItT>::operator*() {
  SBValue *V = nullptr;
  OpItT OpEnd;
  if constexpr (std::is_same<OpItT, SBUser::op_iterator>::value)
    // PredIterator
    OpEnd = N->I->op_end();
  else {
    // SuccIterator
    OpEnd = N->I->user_end();
  }
  if (OpIt != OpEnd) {
    if constexpr (std::is_same<OpItT, SBUser::op_iterator>::value) {
      // PredIterator
      V = *OpIt;
    } else {
      // SuccIterator
      V = *OpIt;
    }
  } else {
    SetVector<Node *>::iterator End;
    if constexpr (std::is_same<OpItT, SBUser::op_iterator>::value)
      // PredIterator
      End = N->Preds.end();
    else
      // SuccIterator
      End = N->Succs.end();
    if (OtherIt != End)
      V = (*OtherIt)->I;
  }
  assert(V != nullptr && "Dereferencing end() ?");
  auto *N = Parent->getNode(cast<SBInstruction>(V));
  assert(N != nullptr && "`V` not mapped to a `Node` yet!");
  return N;
}

template <typename OpItT>
typename DependencyGraph::PredSuccIteratorTemplate<OpItT> &
DependencyGraph::PredSuccIteratorTemplate<OpItT>::operator++() {
  SkipOpIt(/*PreInc=*/true);
  return *this;
}

template <typename OpItT>
typename DependencyGraph::PredSuccIteratorTemplate<OpItT>
DependencyGraph::PredSuccIteratorTemplate<OpItT>::operator++(int) {
  auto Copy = *this;
  SkipOpIt(/*PreInc=*/true);
  return Copy;
}

template <typename OpItT>
bool DependencyGraph::PredSuccIteratorTemplate<OpItT>::operator==(
    const PredSuccIteratorTemplate<OpItT> &Other) const {
  assert(Other.N == N && "Iterators of different nodes!");
  assert(Other.Parent == Parent && "Iterators of different graphs!");
  return Other.OpIt == OpIt && Other.OtherIt == OtherIt;
}

template <typename OpItT>
bool DependencyGraph::PredSuccIteratorTemplate<OpItT>::operator!=(
    const PredSuccIteratorTemplate<OpItT> &Other) const {
  return !(*this == Other);
}

#ifndef NDEBUG
void DependencyGraph::NodeMemRange::dump() {
  for (Node &N : *this)
    N.dump();
}
void DependencyGraph::NodeRange::dump() {
  for (Node &N : *this)
    N.dump();
}
#endif // NDEBUG

DependencyGraph::Node::Node(SBInstruction *SBI,
                                  DependencyGraph &Parent)
    : I(SBI), IsMem(SBI->isMemInst()), Parent(Parent) {}

void DependencyGraph::Node::addMemPred(Node *N) {
  // Don't add a memory dep if there is already a use-def edge to `N`.
  auto OpRange = I->operands();
  if (find(OpRange, N->getInstruction()) != OpRange.end())
    return;
  // Add the dependency N->this
  bool Inserted = Preds.insert(N);
  N->Succs.insert(this);
  // Notify the scheduler so that it maintains UnscheduledSuccs counters.
  if (Inserted && Parent.ViewRange.contains(I) && Parent.Sched != nullptr)
    Parent.Sched->notifyNewDep(N, this);
}

void DependencyGraph::Node::eraseMemPred(Node *N) {
  N->Succs.remove(this);
  if (Parent.Sched)
    Parent.Sched->notifyEraseNode(N);
  Preds.remove(N);
}

static SBInstruction *getPrevMemInst(SBInstruction *I) {
  for (I = I->getPrevNode(); I != nullptr; I = I->getPrevNode()) {
    if (I->isDbgInfo())
      continue;
    if (I->isMemInst())
      return I;
  }
  return nullptr;
}

static SBInstruction *getNextMemInst(SBInstruction *I) {
  for (I = I->getNextNode(); I != nullptr; I = I->getNextNode()) {
    if (I->isDbgInfo())
      continue;
    if (I->isMemInst())
      return I;
  }
  return nullptr;
}

DependencyGraph::Node *DependencyGraph::Node::getPrevNode() const {
  return Parent.getNodeOrNull(getInstruction()->getPrevNode());
}

DependencyGraph::Node *DependencyGraph::Node::getNextNode() const {
  return Parent.getNodeOrNull(getInstruction()->getNextNode());
}

DependencyGraph::Node *DependencyGraph::Node::getPrevMem() const {
  return Parent.getNodeOrNull(getPrevMemInst(getInstruction()));
}

DependencyGraph::Node *DependencyGraph::Node::getNextMem() const {
  return Parent.getNodeOrNull(getNextMemInst(getInstruction()));
}

DependencyGraph::PredIterator
DependencyGraph::Node::pred_begin() const {
  auto *This = const_cast<Node *>(this);
  return PredIterator(I->op_begin(), This->Preds.begin(), This, Parent);
}

DependencyGraph::PredIterator
DependencyGraph::Node::pred_end() const {
  auto *This = const_cast<Node *>(this);
  return PredIterator(I->op_end(), This->Preds.end(), This, Parent);
}

DependencyGraph::SuccIterator
DependencyGraph::Node::succ_begin() const {
  auto *This = const_cast<Node *>(this);
  return SuccIterator(I->user_begin(), This->Succs.begin(), This, Parent);
}

DependencyGraph::SuccIterator
DependencyGraph::Node::succ_end() const {
  auto *This = const_cast<Node *>(this);
  return SuccIterator(I->user_end(), This->Succs.end(), This, Parent);
}

bool DependencyGraph::Node::hasImmPred(Node *N) const {
  auto Operands = I->operands();
  return Preds.contains(N) ||
         find(Operands, N->getInstruction()) != Operands.end();
}

bool DependencyGraph::Node::hasMemPred(Node *N) const {
  return Preds.contains(N);
}

bool DependencyGraph::Node::dependsOn(Node *N) const {
  // BFS search up the DAG starting from this node, looking for `N`.
  SmallDenseSet<Node *, 16> Visited;
  SmallVector<Node *, 16> Worklist;
  for (Node *PredN : preds())
    Worklist.push_back(PredN);
  while (!Worklist.empty()) {
    Node *CurrN = Worklist.front();
    if (CurrN == N)
      return true;
    Worklist.erase(Worklist.begin());
    if (!Visited.insert(CurrN).second)
      continue;

    for (Node *PredN : CurrN->preds())
      Worklist.push_back(PredN);
  }
  return false;
}

void DependencyGraph::Node::removeFromBundle() {
  if (ParentBundle != nullptr) {
    // Performance optimization: We are iterating in reverse because this is the
    // order followed by Scheduler::eraseBundle().
    auto RevIt =
        find(reverse(SchedBundleAttorney::getNodes(*ParentBundle)), this);
    assert(RevIt != SchedBundleAttorney::getNodes(*ParentBundle).rend() &&
           "Node not in bundle!");
    auto It = std::next(RevIt).base();
    SchedBundleAttorney::getNodes(*ParentBundle).erase(It);
    ParentBundle = nullptr;
  }
  Scheduled = false;
  InReadyList = false;
}

#ifndef NDEBUG
void DependencyGraph::Node::dump(raw_ostream &OS, bool InstrRangeOnly,
                                       bool PrintDeps) const {
  I->dump(OS);
  if (!InstrRangeOnly) {
    OS << "; ";
    if (isScheduled())
      OS << "Scheduled";
    OS << " UnschedSuccs=" << UnscheduledSuccs;
  }

  if (PrintDeps) {
    // Collect the predecessors and sort them based on which comes first in BB.
    SmallVector<Node *> PredsVec;
    for (Node *PredN : preds()) {
      if (InstrRangeOnly && !Parent.DAGRange.contains(PredN->getInstruction()))
        continue;
      PredsVec.push_back(PredN);
    }
    stable_sort(PredsVec, [](Node *N1, Node *N2) {
      return N1->getInstruction()->comesBefore(N2->getInstruction());
    });
    for (Node *PredN : PredsVec) {
      OS << "\n";
      const char *DepType = hasMemPred(PredN) ? "M-" : "UD";
      OS.indent(6) << "<-" << DepType << "-";
      PredN->I->dump(OS);
    }

    // Same for successors.
    SmallVector<Node *> SuccsVec;
    for (Node *SuccN : succs()) {
      if (InstrRangeOnly && !Parent.DAGRange.contains(SuccN->getInstruction()))
        continue;
      SuccsVec.push_back(SuccN);
    }
    stable_sort(SuccsVec, [](Node *N1, Node *N2) {
      return N1->getInstruction()->comesBefore(N2->getInstruction());
    });
    for (Node *SuccN : SuccsVec) {
      OS << "\n";
      const char *DepType =
          SuccN->hasMemPred(const_cast<Node *>(this)) ? "M-" : "DU";
      OS.indent(6) << "-" << DepType << "->";
      SuccN->I->dump(OS);
    }
  }
}

void DependencyGraph::Node::verify() const {
  // Check that it is pointing to an instruction.
  assert(I != nullptr && "Expected instruction");
  // Check the preds/succs.
  for (Node *PredN : Preds) {
    assert(PredN->Succs.contains(const_cast<Node *>(this)) &&
           "This node not found in predecessor's successors.");
    assert(PredN != this && "Edge to self!");
    assert(PredN->getInstruction()->comesBefore(I) &&
           "Predecessor should come before this!");
  }
  for (Node *SuccN : Succs) {
    assert(SuccN->Preds.contains(const_cast<Node *>(this)) &&
           "This node not found in successor's predecessors.");
    assert(SuccN != this && "Edge to self!");
    assert(I->comesBefore(SuccN->getInstruction()) &&
           "This should come before successors!");
  }
  if (ParentBundle != nullptr)
    assert(find(*ParentBundle, this) != ParentBundle->end() &&
           "ParentBundle does not contain this Node!");
}

void DependencyGraph::dump(raw_ostream &OS, bool InstrRangeOnly,
                                 bool InViewOnly) const {
  OS << "\n";
  // InstrToNodeMap is unoredred so we need to create an ordered vector.
  SmallVector<Node *> Nodes;
  Nodes.reserve(InstrToNodeMap.size());
  for (const auto &Pair : InstrToNodeMap)
    Nodes.push_back(Pair.second.get());
  // Sort them based on which one comes first in the BB.
  stable_sort(Nodes, [](Node *N1, Node *N2) {
    return N1->getInstruction()->comesBefore(N2->getInstruction());
  });

  // Go over all nodes in the vector and print them.
  SmallVector<Node *> NodesInInstrRange;
  for (Node *N : Nodes) {
    // If we are printing only within region, skip any instrs outside the region
    if (InstrRangeOnly && !DAGRange.contains(N->getInstruction()))
      continue;
    NodesInInstrRange.push_back(N);
  }
  stable_sort(NodesInInstrRange, [](Node *N1, Node *N2) {
    return N1->getInstruction()->comesBefore(N2->getInstruction());
  });

  if (!InstrRangeOnly && !NodesInInstrRange.empty()) {
    OS << "In ViewRange\n";
    OS << "vv\n";
  }
  for (Node *N : NodesInInstrRange) {
    if (InViewOnly && !ViewRange.contains(N->getInstruction()))
      continue;
    const char *Prefix = "";
    if (!InstrRangeOnly) {
      Prefix = " ";
      if (ViewRange.contains(N->getInstruction()))
        Prefix = "*";
      OS << Prefix << " ";
    }
    N->dump(OS, InstrRangeOnly);
    OS << "\n";
  }

  if (!InstrRangeOnly) {
    OS << "\n";
    OS << "\nRoots:\n";
    auto Roots = getRoots();
    stable_sort(Roots, [](Node *N1, Node *N2) {
      return N1->getInstruction()->comesBefore(N2->getInstruction());
    });
    for (Node *RootN : Roots)
      OS << *RootN << "\n";
    OS << "---\n";
  }
}
#endif // NDEBUG

SmallVector<DependencyGraph::Node *>
DependencyGraph::getRoots() const {
  SmallVector<Node *> Roots;
  Roots.reserve(32);
  for (const auto &[I, NPtr] : InstrToNodeMap) {
    auto *N = NPtr.get();
    // If it is not visible it can't be a root node.
    if (!ViewRange.contains(N->getInstruction()))
      continue;
    // If any of its successors are visible, then it's not a root.
    if (!N->hasNoSuccs() && any_of(N->succs(), [this](auto *SuccN) {
          return ViewRange.contains(SuccN->getInstruction());
        }))
      continue;
    Roots.push_back(N);
  }
  return Roots;
}

DependencyGraph::Node *
DependencyGraph::getNode(SBInstruction *SBI) const {
  assert(SBI != nullptr && "Expected non-null instruction");
  auto It = InstrToNodeMap.find(SBI);
  return It != InstrToNodeMap.end() ? It->second.get() : nullptr;
}
DependencyGraph::Node *
DependencyGraph::getNodeOrNull(SBInstruction *SBI) const {
  if (SBI == nullptr)
    return nullptr;
  return getNode(SBI);
}

DependencyGraph::Node *
DependencyGraph::getOrCreateNode(SBInstruction *SBI) {
  assert(!SBI->isDbgInfo() && "No debug info intrinsics allowed!");
  auto [It, NotInMap] = InstrToNodeMap.try_emplace(SBI);
  if (NotInMap)
    It->second = std::make_unique<Node>(SBI, *this);
  return It->second.get();
}

static bool isOrdered(Instruction *I) {
  if (auto *LI = dyn_cast<LoadInst>(I))
    return !LI->isUnordered();
  if (auto *SI = dyn_cast<StoreInst>(I))
    return !SI->isUnordered();
  if (I->isFenceLike())
    return true;
  return false;
}

DependencyGraph::DependencyType
DependencyGraph::getRoughDepType(SBInstruction *FromI,
                                       SBInstruction *ToI) {
  if (FromI->mayWriteToMemory()) {
    if (ToI->mayReadFromMemory())
      return DependencyType::RAW;
    if (ToI->mayWriteToMemory())
      return DependencyType::WAW;
  }
  if (FromI->mayReadFromMemory()) {
    if (ToI->mayWriteToMemory())
      return DependencyType::WAR;
    if (ToI->mayReadFromMemory())
      return DependencyType::RAR;
  }
  if (isa<SBPHINode>(FromI) || isa<SBPHINode>(ToI))
    return DependencyType::CTRL;
  if (ToI->isTerminator())
    return DependencyType::CTRL;
  if (FromI->isStackRelated() || ToI->isStackRelated())
    return DependencyType::OTHER;
  return DependencyType::NONE;
}

bool DependencyGraph::alias(Instruction *SrcIR, Instruction *DstIR,
                                  DependencyType DepType, int &AABudget) const {
  std::optional<MemoryLocation> DstLocOpt = MemoryLocation::getOrNone(DstIR);
  if (!DstLocOpt)
    return true;
  // Check aliasing.
  assert((SrcIR->mayReadFromMemory() || SrcIR->mayWriteToMemory()) &&
         "Expected a mem instr");
  // We limit the AA checks to reduce compilation time.
  if (--AABudget < 0)
    return true;
  ModRefInfo SrcModRef = isOrdered(SrcIR)
                             ? ModRefInfo::Mod
                             : BatchAA->getModRefInfo(SrcIR, *DstLocOpt);
  switch (DepType) {
  case DependencyType::RAW:
  case DependencyType::WAW:
    return isModSet(SrcModRef);
  case DependencyType::WAR:
    return isRefSet(SrcModRef);
  default:
    llvm_unreachable("Expected only RAW, WAW and WAR!");
  }
}

bool DependencyGraph::hasDep(SBInstruction *SrcI, SBInstruction *DstI,
                                   int &AABudget) const {
  Instruction *SrcIR = cast<Instruction>(ValueAttorney::getValue(SrcI));
  Instruction *DstIR = cast<Instruction>(ValueAttorney::getValue(DstI));

  DependencyType RoughDepType = getRoughDepType(SrcI, DstI);
  switch (RoughDepType) {
  case DependencyType::RAR:
    return false;
  case DependencyType::RAW:
  case DependencyType::WAW:
  case DependencyType::WAR:
    return alias(SrcIR, DstIR, RoughDepType, AABudget);
  case DependencyType::CTRL:
    // Adding actual dep edges from PHIs/to terminator would just create too
    // many edges, which would be bad for compile-time.
    // So we ignore them in the DAG formation but handle them in the
    // scheduler, while sorting the ready list.
    return false;
  case DependencyType::OTHER:
    return true;
  case DependencyType::NONE:
    return false;
  }
}

void DependencyGraph::scanAndAddDeps(Node *DstN, NodeMemRange ScanRange) {
  assert(DstN->isMem() && "DstN is the mem dep destination, so it must be mem");
  SBInstruction *DstI = DstN->getInstruction();
  int AABudget = AAQueryBudget;
  // Walk up the instruction chain from ScanRange bottom to top, looking for
  // memory instrs that may alias.
  for (Node &SrcN : reverse(ScanRange)) {
    SBInstruction *SrcI = SrcN.getInstruction();
    if (hasDep(SrcI, DstI, AABudget))
      DstN->addMemPred(&SrcN);
  }
}

void DependencyGraph::createNodesFor(const InstrRange &Rgn) {
  for (SBInstruction &I : Rgn) {
    if (I.isDbgInfo())
      continue;
    getOrCreateNode(&I);
  }
}

DependencyGraph::NodeMemRange
DependencyGraph::getScanRange(Node *ScanTopN, Node *ScanBotN,
                                    Node *AboveN) {
  if (ScanBotN->comesBefore(AboveN))
    return makeMemRangeFromNonMem(ScanTopN, ScanBotN);
  // Range is [ScanTopN - AboveN)
  Node *DstMemN = AboveN->getPrevMem();
  if (DstMemN == nullptr || DstMemN->comesBefore(ScanTopN))
    return makeEmptyMemRange();
  return makeMemRangeFromNonMem(ScanTopN, DstMemN);
}

void DependencyGraph::extendDAG(const InstrRange &OldInstrRange,
                                      const InstrRange &NewInstrRange) {
  assert(!NewInstrRange.empty() && "Expected non-empty NewInstrRange!");
  // Create DAG nodes for the new region.
  createNodesFor(NewInstrRange);

  // 1. OldInstrRange empty       2. New is below Old      3. New is above old
  // ------------------       -------------------      -------------------
  //                                        Scan:           DstN:    Scan:
  //                          +---+         -ScanTopN  +---+DstFromN -ScanTopN
  //                          |   |         |          |New|         |
  //                          |Old|         |          +---+         -ScanBotN
  //                          |   |         |          +---+
  //      DstN:    Scan:      +---+DstN:    |          |   |
  // +---+DstFromN -ScanTopN  +---+DstFromN |          |Old|
  // |New|         |          |New|         |          |   |
  // +---+DstToN   -ScanBotN  +---+DstToN   -ScanBotN  +---+DstToN

  Node *NewTopN = getNode(NewInstrRange.from());
  Node *NewBotN = getNode(NewInstrRange.to());

  // 1. OldInstrRange is empty.
  if (OldInstrRange.empty()) {
    NodeMemRange DstNRange = makeMemRangeFromNonMem(NewTopN, NewBotN);
    for (Node &DstN : DstNRange)
      scanAndAddDeps(&DstN, getScanRange(NewTopN, NewBotN, &DstN));
    return;
  }

  Node *OldTopN = getNode(OldInstrRange.from());
  Node *OldBotN = getNode(OldInstrRange.to());
  bool NewInstrRangeIsBelowOld =
      OldInstrRange.to()->comesBefore(NewInstrRange.from());
  // 2. NewInstrRange is below OldInstrRange.
  if (NewInstrRangeIsBelowOld) {
    NodeMemRange DstNRange = makeMemRangeFromNonMem(NewTopN, NewBotN);
    for (Node &DstN : DstNRange)
      scanAndAddDeps(&DstN, getScanRange(OldTopN, NewBotN, &DstN));
    return;
  }

  // 3. NewInstrRange is above OldInstrRange.
  NodeMemRange DstNRange = makeMemRangeFromNonMem(NewTopN, OldBotN);
  for (Node &DstN : DstNRange)
    scanAndAddDeps(&DstN, getScanRange(NewTopN, NewBotN, &DstN));
}

InstrRange DependencyGraph::extendView(const InstrRange &NewViewRange) {
#ifndef NDEBUG
  auto OrigViewRange = ViewRange;
#endif
  // Extend ViewRange and recompute the UnscheduledSuccs for all new instrs
  // in the view.
  auto NewViewSections = NewViewRange - ViewRange;
  assert(NewViewSections.size() == 1 && "Expected a single region!");
  auto NewViewSection = NewViewSections.back();

  for (auto &I : NewViewSection) {
    auto *N = getNode(&I);
    // Skip Debug intrinsics.
    if (I.isDbgInfo())
      continue;
    ViewRange = ViewRange.getUnionSingleSpan(NewViewSection);

    N->resetUnscheduledSuccs();
    // For each in-region unscheduled dependent node increment UnscheduledDeps.
    if (Sched != nullptr)
      Sched->notifyNewNode(N, /*OnlyDefUse=*/false, /*UpdateDepNodes=*/false);
  }
  assert(ViewRange == NewViewRange &&
         "ViewRange should have been updated by now!");
#ifndef NDEBUG
  auto Diff = NewViewSection - OrigViewRange;
  assert(Diff.size() != 2 && "Extending View region in both directions!");
#ifdef SBVEC_EXPENSIVE_CHECKS
  verify();
#endif
#endif
  return NewViewSection;
}

InstrRange DependencyGraph::extend(const SBValBundle &Instrs) {
  assert(SBUtils::areInSameBB(Instrs) && "Instrs expected to be in same BB!");
  assert(none_of(Instrs,
                 [](SBValue *SBV) {
                   return cast<SBInstruction>(SBV)->isDbgInfo();
                 }) &&
         "Expected no debug info intrinsics!");
  // 1. Extend the DAGRange and create new deps.
  InstrRange FinalViewRange = ViewRange.getUnionSingleSpan(InstrRange(Instrs));
  // Now if needed, extend the DAG to include nodes that are in
  // RequestedInstrRange but not in the existing DAG.
  auto FinalDAGRange = DAGRange.getUnionSingleSpan(FinalViewRange);
  auto NewDAGSections = FinalDAGRange - DAGRange;
  for (const InstrRange &NewSection : NewDAGSections) {
    if (NewSection.empty())
      continue;
    // Extend the DAG to include the new instructions.
    extendDAG(DAGRange, NewSection);
    // Update the region to include the new section
    DAGRange = DAGRange.getUnionSingleSpan(NewSection);

    assert(DAGRange.contains(ViewRange) && "View should be contained in DAG");
  }
  assert(DAGRange == FinalDAGRange && "DAGRange should have been updated!");

  // 2. Extend the View
  return extendView(FinalViewRange);
}

void DependencyGraph::trimView(SBInstruction *FromI, bool Above) {
  assert(ViewRange.contains(FromI) && "Expect it to be in region");
  InstrRange ToTrim = Above ? InstrRange{ViewRange.from(), FromI}
                            : InstrRange{FromI, ViewRange.to()};
  for (SBInstruction &I : ToTrim) {
    auto *N = getNode(&I);
    N->resetUnscheduledSuccs();
  }
  ViewRange = ViewRange.getSingleDifference(ToTrim);
}

void DependencyGraph::resetView() {
  if (ViewRange.empty())
    return;
  ViewRange.clear();
}

void DependencyGraph::clear() {
  resetView();
  InstrToNodeMap.clear();
  DAGRange.clear();
}

SmallVector<
    std::pair<DependencyGraph::Node *, DependencyGraph::Node *>>
DependencyGraph::getDepsFromPredsToSuccs(Node *N) const {
  SmallVector<std::pair<Node *, Node *>> Deps;
  Deps.reserve(4);
  // Add SuccN->PredN edges for all dependent successors.
  // NOTE: We are iterating over mem_succs() and not succs(), because it is
  // illegal to remove `N` while its def-use successors are still connected.
  for (Node *SuccN : N->mem_succs()) {
    for (Node *PredN : N->preds()) {
      int AABudget = 1;
      if (hasDep(PredN->getInstruction(), SuccN->getInstruction(), AABudget))
        Deps.push_back({PredN, SuccN});
    }
  }
  return Deps;
}

void DependencyGraph::erase(Node *N) {
  // Add dependencies from N's successors to all its dependent predecessors.
  // NOTE: We need to remove `N` before adding the dependencies because the
  // presense of `N` in the DAG may block the creation of the new dependencies
  // since we are skipping transient edges.
  auto Deps = getDepsFromPredsToSuccs(N);
  // Erase N->SuccN for all successors.
  while (!N->mem_succs().empty()) {
    auto *SuccN = *N->succs().begin();
    SuccN->eraseMemPred(N);
  }
  // Erase PredN->N for all predecessors.
  while (!N->mem_preds().empty()) {
    auto *PredN = *N->mem_preds().begin();
    N->eraseMemPred(PredN);
  }
  // Remove it from scheduling bundle.
  if (auto *SchedBundle = N->getBundle())
    SchedBundle->remove(N);
  // Add new dependencies.
  for (auto [SrcN, DstN] : Deps)
    DstN->addMemPred(SrcN);
  // Notify regions about instruction deletion.
  SBInstruction *I = N->getInstruction();
  ViewRange.erase(I, /*CheckContained=*/false);
  DAGRange.erase(I, /*CheckContained=*/false);
  // This frees memory, so should be done last.
  InstrToNodeMap.erase(I);
}

void DependencyGraph::erase(SBInstruction *I) {
  auto *N = getNode(I);
  erase(N);
  // Invalidate BatchAA on erase to be on the safe side.
  BatchAA = std::make_unique<BatchAAResults>(AA);
}

void DependencyGraph::notifyRemove(SBInstruction *I) {
  auto *N = getNode(I);
  if (N != nullptr) {
    erase(N);
  }
  ViewRange.erase(I, /*CheckContained=*/false);
  DAGRange.erase(I, /*CheckContained=*/false);
}

void DependencyGraph::notifyInsert(SBInstruction *I) {
  insertAndAddDeps(I);
}

DependencyGraph::Node *
DependencyGraph::insert(SBInstruction *NewI) {

  return getOrCreateNode(NewI);
}

DependencyGraph::Node *
DependencyGraph::insertAndAddDeps(SBInstruction *NewI) {
  // Return existing node if we found one.
  if (auto *ExistingN = getNode(NewI))
    return ExistingN;
  Node *NewN = getOrCreateNode(NewI);
  if (DAGRange.contains(NewI)) {
    // Create the new dependencies.
    if (NewN->isMem()) {
      // If NewN touches memory we need to do a proper scan for dependencies.
      // Scan mem instrs above NewN and add edges to AboveN->NewN
      scanAndAddDeps(NewN, getScanRange(getNode(DAGRange.from()),
                                        getNode(DAGRange.to()), NewN));
      // Go over mem instrs under NewN and add edges NewN->UnderN
      Node *FirstUnderN = NewN->getNextMem();
      Node *ToUnderN = getNode(DAGRange.to());
      if (FirstUnderN != nullptr &&
          (ToUnderN == FirstUnderN || FirstUnderN->comesBefore(ToUnderN)))
        for (Node &UnderN : makeMemRangeFromNonMem(FirstUnderN, ToUnderN))
          scanAndAddDeps(&UnderN, makeMemRange(NewN, NewN));
    }
    // Regardless of whether NewN touches memory or not, it may have use-def
    // dependencies. We need to go over the operands and update their
    // UnscheduledSuccs counter, but only if there is no memory dependency edge.
    // Because in that case we would have already incremented the edge in
    // scanAndAddDeps().
    if (Sched != nullptr && ViewRange.contains(NewI))
      Sched->notifyNewNode(NewN, /*OnlyDefUse=*/true, /*UpdateDepNodes=*/true);
  }
  // Extend the DAG to make sure NewI is included in both DAG and View.
  extend({NewI});

  assert(DAGRange.contains(NewI) &&
         "NewI is meant to replace instructions from within the DAG so it "
         "should be in the DAGRange!");
#ifndef NDEBUG
#ifdef SBVEC_EXPENSIVE_CHECKS
  verify();
#endif
#endif
  return NewN;
}

void DependencyGraph::notifyMoveInstr(SBInstruction *I,
                                            SBBasicBlock::iterator BeforeIt,
                                            SBBasicBlock *BB) {
  // If `I` doesn't move, nothing to do.
  if (BeforeIt == I->getIterator() || BeforeIt == std::next(I->getIterator()))
    return;
  if (DAGRange.empty())
    return;
  // If this is a instruction motion is done by the scheduler, then the movement
  // is confined to within the regions, so just need to maintain the border
  // instructions if they move.
  DAGRange.notifyMoveInstr(I, BeforeIt, BB);
  if (!Ctxt.getTracker().inRevert()) {
    // Don't update the ViewRange when reverting, because this cannot
    // always be reverted.
    ViewRange.notifyMoveInstr(I, BeforeIt, BB);
  }
}

#ifndef NDEBUG
void DependencyGraph::verify() {
  assert(DAGRange.contains(ViewRange) && "DAGRange should contain ViewRange");
  // Check the node edges.
  for (const auto &Pair : InstrToNodeMap) {
    Node *N = Pair.second.get();
    N->verify();
  }
}
#endif

DependencyGraph::DependencyGraph(SBContext &Ctxt,
                                             AliasAnalysis &AA,
                                             Scheduler *Sched,
                                             int AAQueryBudget)
    : Ctxt(Ctxt), AA(AA), Sched(Sched),
      BatchAA(std::make_unique<BatchAAResults>(AA)),
      AAQueryBudget(AAQueryBudget) {}

template class DependencyGraph::PredSuccIteratorTemplate<
    SBUser::op_iterator>;
template class DependencyGraph::PredSuccIteratorTemplate<
    SBValue::user_iterator>;
