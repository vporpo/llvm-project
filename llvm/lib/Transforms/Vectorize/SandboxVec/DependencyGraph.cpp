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
#include "llvm/Transforms/Vectorize/SandboxVec/VecUtils.h"
#include <cstdint>

using namespace llvm;
using namespace llvm::PatternMatch;

template <typename OpItT>
sandboxir::DependencyGraph::PredSuccIteratorTemplate<
    OpItT>::PredSuccIteratorTemplate(OpItT OpIt,
                                     SetVector<Node *>::iterator OtherIt,
                                     Node *N, DependencyGraph &Parent)
    : OpIt(OpIt), OtherIt(OtherIt), N(N), Parent(&Parent) {
  SkipOpIt(/*PreInc=*/false);
}
template <typename OpItT>
void sandboxir::DependencyGraph::PredSuccIteratorTemplate<OpItT>::SkipOpIt(
    bool PreInc) {
  OpItT EndIt;
  if constexpr (std::is_same<OpItT, sandboxir::User::op_iterator>::value)
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
      if constexpr (std::is_same<OpItT, sandboxir::User::op_iterator>::value)
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
    sandboxir::Value *SBV = *OpIt;
    if (SBV == nullptr)
      return true;
    if (!isa<sandboxir::Instruction>(SBV))
      return true;
    sandboxir::Value *OpV = *OpIt;
    if (!Parent->getNode(cast<sandboxir::Instruction>(OpV)))
      return true;
    return false;
  };
  while (OpIt != EndIt && ShouldSkip(OpIt))
    ++OpIt;
}

template <typename OpItT>
typename sandboxir::DependencyGraph::PredSuccIteratorTemplate<OpItT>::value_type
sandboxir::DependencyGraph::PredSuccIteratorTemplate<OpItT>::operator*() {
  sandboxir::Value *V = nullptr;
  OpItT OpEnd;
  if constexpr (std::is_same<OpItT, sandboxir::User::op_iterator>::value)
    // PredIterator
    OpEnd = N->I->op_end();
  else {
    // SuccIterator
    OpEnd = N->I->user_end();
  }
  if (OpIt != OpEnd) {
    if constexpr (std::is_same<OpItT, sandboxir::User::op_iterator>::value) {
      // PredIterator
      V = *OpIt;
    } else {
      // SuccIterator
      V = *OpIt;
    }
  } else {
    SetVector<Node *>::iterator End;
    if constexpr (std::is_same<OpItT, sandboxir::User::op_iterator>::value)
      // PredIterator
      End = N->Preds.end();
    else
      // SuccIterator
      End = N->Succs.end();
    if (OtherIt != End)
      V = (*OtherIt)->I;
  }
  assert(V != nullptr && "Dereferencing end() ?");
  auto *N = Parent->getNode(cast<sandboxir::Instruction>(V));
  assert(N != nullptr && "`V` not mapped to a `Node` yet!");
  return N;
}

template <typename OpItT>
typename sandboxir::DependencyGraph::PredSuccIteratorTemplate<OpItT> &
sandboxir::DependencyGraph::PredSuccIteratorTemplate<OpItT>::operator++() {
  SkipOpIt(/*PreInc=*/true);
  return *this;
}

template <typename OpItT>
typename sandboxir::DependencyGraph::PredSuccIteratorTemplate<OpItT>
sandboxir::DependencyGraph::PredSuccIteratorTemplate<OpItT>::operator++(int) {
  auto Copy = *this;
  SkipOpIt(/*PreInc=*/true);
  return Copy;
}

template <typename OpItT>
bool sandboxir::DependencyGraph::PredSuccIteratorTemplate<OpItT>::operator==(
    const PredSuccIteratorTemplate<OpItT> &Other) const {
  assert(Other.N == N && "Iterators of different nodes!");
  assert(Other.Parent == Parent && "Iterators of different graphs!");
  return Other.OpIt == OpIt && Other.OtherIt == OtherIt;
}

template <typename OpItT>
bool sandboxir::DependencyGraph::PredSuccIteratorTemplate<OpItT>::operator!=(
    const PredSuccIteratorTemplate<OpItT> &Other) const {
  return !(*this == Other);
}

#ifndef NDEBUG
void sandboxir::DependencyGraph::NodeMemRange::dump() {
  for (Node &N : *this)
    N.dump();
}
void sandboxir::DependencyGraph::NodeRange::dump() {
  for (Node &N : *this)
    N.dump();
}
#endif // NDEBUG

sandboxir::DependencyGraph::Node::Node(sandboxir::Instruction *SBI,
                                       DependencyGraph &Parent)
    : I(SBI), IsMem(SBI->isMemInst()), Parent(Parent) {}

void sandboxir::DependencyGraph::Node::addMemPred(Node *N) {
  // Don't add a memory dep if there is already a use-def edge to `N`.
  auto OpRange = I->operands();
  if (find(OpRange, N->getInstruction()) != OpRange.end())
    return;
  // Add the dependency N->this
  Preds.insert(N);
  N->Succs.insert(this);
}

void sandboxir::DependencyGraph::Node::eraseMemPred(Node *N) {
  N->Succs.remove(this);
  Preds.remove(N);
}

static sandboxir::Instruction *getPrevMemInst(sandboxir::Instruction *I) {
  for (I = I->getPrevNode(); I != nullptr; I = I->getPrevNode()) {
    if (I->isDbgInfo())
      continue;
    if (I->isMemInst())
      return I;
  }
  return nullptr;
}

static sandboxir::Instruction *getNextMemInst(sandboxir::Instruction *I) {
  for (I = I->getNextNode(); I != nullptr; I = I->getNextNode()) {
    if (I->isDbgInfo())
      continue;
    if (I->isMemInst())
      return I;
  }
  return nullptr;
}

sandboxir::DependencyGraph::Node *
sandboxir::DependencyGraph::Node::getPrevNode() const {
  return Parent.getNodeOrNull(getInstruction()->getPrevNode());
}

sandboxir::DependencyGraph::Node *
sandboxir::DependencyGraph::Node::getNextNode() const {
  return Parent.getNodeOrNull(getInstruction()->getNextNode());
}

sandboxir::DependencyGraph::Node *
sandboxir::DependencyGraph::Node::getPrevMem() const {
  return Parent.getNodeOrNull(getPrevMemInst(getInstruction()));
}

sandboxir::DependencyGraph::Node *
sandboxir::DependencyGraph::Node::getNextMem() const {
  return Parent.getNodeOrNull(getNextMemInst(getInstruction()));
}

sandboxir::DependencyGraph::PredIterator
sandboxir::DependencyGraph::Node::pred_begin() const {
  auto *This = const_cast<Node *>(this);
  return PredIterator(I->op_begin(), This->Preds.begin(), This, Parent);
}

sandboxir::DependencyGraph::PredIterator
sandboxir::DependencyGraph::Node::pred_end() const {
  auto *This = const_cast<Node *>(this);
  return PredIterator(I->op_end(), This->Preds.end(), This, Parent);
}

sandboxir::DependencyGraph::SuccIterator
sandboxir::DependencyGraph::Node::succ_begin() const {
  auto *This = const_cast<Node *>(this);
  return SuccIterator(I->user_begin(), This->Succs.begin(), This, Parent);
}

sandboxir::DependencyGraph::SuccIterator
sandboxir::DependencyGraph::Node::succ_end() const {
  auto *This = const_cast<Node *>(this);
  return SuccIterator(I->user_end(), This->Succs.end(), This, Parent);
}

bool sandboxir::DependencyGraph::Node::hasImmPred(Node *N) const {
  auto Operands = I->operands();
  return Preds.contains(N) ||
         find(Operands, N->getInstruction()) != Operands.end();
}

bool sandboxir::DependencyGraph::Node::hasMemPred(Node *N) const {
  return Preds.contains(N);
}

bool sandboxir::DependencyGraph::Node::dependsOn(Node *N) const {
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

void sandboxir::DependencyGraph::Node::removeFromBundle() {
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

std::pair<sandboxir::Instruction *, sandboxir::Instruction *>
sandboxir::DependencyGraph::getDepCacheKey(sandboxir::Instruction *I1,
                                           sandboxir::Instruction *I2) {
  if (reinterpret_cast<uintptr_t>(I1) < reinterpret_cast<uintptr_t>(I2))
    return {I1, I2};
  return {I2, I1};
}
void sandboxir::DependencyGraph::setDepCache(sandboxir::Instruction *I1,
                                             sandboxir::Instruction *I2,
                                             bool HasDep) {
  DepCache[getDepCacheKey(I1, I2)] = HasDep;
}
bool sandboxir::DependencyGraph::getDepCache(sandboxir::Instruction *I1,
                                             sandboxir::Instruction *I2) {
  auto It = DepCache.find(getDepCacheKey(I1, I2));
  if (It == DepCache.end())
    // If not in the cache it must mean that I2 was lower than I1, so it was no
    // teven scanned for dependencies. So it is safe to return false.
    return false;
  return It->second;
}

#ifndef NDEBUG
void sandboxir::DependencyGraph::Node::dump(raw_ostream &OS,
                                            bool InstrIntervalOnly,
                                            bool PrintDeps) const {
  I->dump(OS);
  if (!InstrIntervalOnly) {
    OS << "; ";
    if (isScheduled())
      OS << "Scheduled";
    OS << " UnschedSuccs=" << UnscheduledSuccs;
  }

  if (PrintDeps) {
    // Collect the predecessors and sort them based on which comes first in BB.
    SmallVector<Node *> PredsVec;
    for (Node *PredN : preds()) {
      if (InstrIntervalOnly &&
          !Parent.DAGInterval.contains(PredN->getInstruction()))
        continue;
      PredsVec.push_back(PredN);
    }
    stable_sort(PredsVec, [](Node *N1, Node *N2) {
      return N1->getInstruction()->comesBefore(N2->getInstruction());
    });
    for (Node *PredN : PredsVec) {
      OS << "\n";
      if (Parent.inView(PredN))
        OS << "*";
      else
        OS << " ";
      const char *DepType = hasMemPred(PredN) ? "M-" : "UD";
      OS.indent(6) << "<-" << DepType << "-";
      PredN->I->dump(OS);
      if (PredN->isScheduled())
        OS << " Scheduled";
    }

    // Same for successors.
    SmallVector<Node *> SuccsVec;
    for (Node *SuccN : succs()) {
      if (InstrIntervalOnly &&
          !Parent.DAGInterval.contains(SuccN->getInstruction()))
        continue;
      SuccsVec.push_back(SuccN);
    }
    stable_sort(SuccsVec, [](Node *N1, Node *N2) {
      return N1->getInstruction()->comesBefore(N2->getInstruction());
    });
    for (Node *SuccN : SuccsVec) {
      OS << "\n";
      if (Parent.inView(SuccN))
        OS << "*";
      else
        OS << " ";
      const char *DepType =
          SuccN->hasMemPred(const_cast<Node *>(this)) ? "M-" : "DU";
      OS.indent(6) << "-" << DepType << "->";
      SuccN->I->dump(OS);
      if (SuccN->isScheduled())
        OS << " Scheduled";
    }
  }
}

void sandboxir::DependencyGraph::Node::verify() const {
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

void sandboxir::DependencyGraph::dump(raw_ostream &OS, bool InstrIntervalOnly,
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
  SmallVector<Node *> NodesInInstrInterval;
  for (Node *N : Nodes) {
    // If we are printing only within region, skip any instrs outside the region
    if (InstrIntervalOnly && !DAGInterval.contains(N->getInstruction()))
      continue;
    NodesInInstrInterval.push_back(N);
  }
  stable_sort(NodesInInstrInterval, [](Node *N1, Node *N2) {
    return N1->getInstruction()->comesBefore(N2->getInstruction());
  });

  if (!InstrIntervalOnly && !NodesInInstrInterval.empty()) {
    OS << "In ViewRange\n";
    OS << "vv\n";
  }
  for (Node *N : NodesInInstrInterval) {
    if (InViewOnly && !ViewRange.contains(N->getInstruction()))
      continue;
    const char *Prefix = "";
    if (!InstrIntervalOnly) {
      Prefix = " ";
      if (ViewRange.contains(N->getInstruction()))
        Prefix = "*";
      OS << Prefix << " ";
    }
    N->dump(OS, InstrIntervalOnly);
    OS << "\n";
  }

  if (!InstrIntervalOnly) {
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

SmallVector<sandboxir::DependencyGraph::Node *>
sandboxir::DependencyGraph::getRoots() const {
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

sandboxir::DependencyGraph::Node *
sandboxir::DependencyGraph::getNode(sandboxir::Instruction *SBI) const {
  assert(SBI != nullptr && "Expected non-null instruction");
  auto It = InstrToNodeMap.find(SBI);
  return It != InstrToNodeMap.end() ? It->second.get() : nullptr;
}
sandboxir::DependencyGraph::Node *
sandboxir::DependencyGraph::getNodeOrNull(sandboxir::Instruction *SBI) const {
  if (SBI == nullptr)
    return nullptr;
  return getNode(SBI);
}

sandboxir::DependencyGraph::Node *
sandboxir::DependencyGraph::getOrCreateNode(sandboxir::Instruction *SBI) {
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

sandboxir::DependencyGraph::DependencyType
sandboxir::DependencyGraph::getRoughDepType(sandboxir::Instruction *FromI,
                                            sandboxir::Instruction *ToI) {
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
  if (isa<sandboxir::PHINode>(FromI) || isa<sandboxir::PHINode>(ToI))
    return DependencyType::CTRL;
  if (ToI->isTerminator())
    return DependencyType::CTRL;
  if (FromI->isStackRelated() || ToI->isStackRelated())
    return DependencyType::OTHER;
  return DependencyType::NONE;
}

bool sandboxir::DependencyGraph::alias(llvm::Instruction *SrcIR,
                                       llvm::Instruction *DstIR,
                                       DependencyType DepType, int &AABudget) {
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

size_t sandboxir::DependencyGraph::getProjectedApproxSize(
    const DmpVector<sandboxir::Value *> &Instrs) {
  InstrInterval Range(Instrs);
  InstrInterval UnionRange = Range.getUnionSingleSpan(ViewRange);
  assert(!UnionRange.empty() && "Expected non-empty range!");
  return UnionRange.from()->getApproximateDistanceTo(UnionRange.to());
}

bool sandboxir::DependencyGraph::hasDep(sandboxir::Instruction *SrcI,
                                        sandboxir::Instruction *DstI,
                                        int &AABudget) {
  // This cache is used for correctness. Without it reverting the IR can lead to
  // a broken DAG, as hasDep() may return a defferent result for the same inputs
  bool InRevert = Ctx.getTracker().inRevert();
  if (InRevert) {
    // TODO: Is this correct? What if the instruction operands have changed?
    return getDepCache(SrcI, DstI);
  }

  llvm::Instruction *SrcIR =
      cast<llvm::Instruction>(ValueAttorney::getValue(SrcI));
  llvm::Instruction *DstIR =
      cast<llvm::Instruction>(ValueAttorney::getValue(DstI));

  DependencyType RoughDepType = getRoughDepType(SrcI, DstI);

  auto HasDep = [&]() {
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
  };
  bool Has = HasDep();
  if (!InRevert)
    setDepCache(SrcI, DstI, Has);
  return Has;
}

void sandboxir::DependencyGraph::scanAndAddDeps(Node *DstN,
                                                NodeMemRange ScanRange) {
  assert(DstN->isMem() && "DstN is the mem dep destination, so it must be mem");
  sandboxir::Instruction *DstI = DstN->getInstruction();
  int AABudget = AAQueryBudget;
  // Walk up the instruction chain from ScanRange bottom to top, looking for
  // memory instrs that may alias.
  for (Node &SrcN : reverse(ScanRange)) {
    sandboxir::Instruction *SrcI = SrcN.getInstruction();
    if (hasDep(SrcI, DstI, AABudget))
      DstN->addMemPred(&SrcN);
  }
}

void sandboxir::DependencyGraph::createNodesFor(const InstrInterval &Rgn) {
  for (sandboxir::Instruction &I : Rgn) {
    if (I.isDbgInfo())
      continue;
    getOrCreateNode(&I);
  }
}

sandboxir::DependencyGraph::NodeMemRange
sandboxir::DependencyGraph::getScanRange(Node *ScanTopN, Node *ScanBotN,
                                         Node *AboveN) {
  if (ScanBotN->comesBefore(AboveN))
    return makeMemRangeFromNonMem(ScanTopN, ScanBotN);
  // Range is [ScanTopN - AboveN)
  Node *DstMemN = AboveN->getPrevMem();
  if (DstMemN == nullptr || DstMemN->comesBefore(ScanTopN))
    return makeEmptyMemRange();
  return makeMemRangeFromNonMem(ScanTopN, DstMemN);
}

void sandboxir::DependencyGraph::extendDAG(
    const InstrInterval &OldInstrInterval,
    const InstrInterval &NewInstrInterval) {
  assert(!NewInstrInterval.empty() && "Expected non-empty NewInstrInterval!");
  // Create DAG nodes for the new region.
  createNodesFor(NewInstrInterval);

  // 1. OldInstrInterval empty 2. New is below Old     3. New is above old
  // ------------------------  -------------------      -------------------
  //                                         Scan:           DstN:    Scan:
  //                           +---+         -ScanTopN  +---+DstFromN -ScanTopN
  //                           |   |         |          |New|         |
  //                           |Old|         |          +---+         -ScanBotN
  //                           |   |         |          +---+
  //      DstN:    Scan:       +---+DstN:    |          |   |
  // +---+DstFromN -ScanTopN   +---+DstFromN |          |Old|
  // |New|         |           |New|         |          |   |
  // +---+DstToN   -ScanBotN   +---+DstToN   -ScanBotN  +---+DstToN

  Node *NewTopN = getNode(NewInstrInterval.from());
  Node *NewBotN = getNode(NewInstrInterval.to());

  // 1. OldInstrInterval is empty.
  if (OldInstrInterval.empty()) {
    NodeMemRange DstNRange = makeMemRangeFromNonMem(NewTopN, NewBotN);
    for (Node &DstN : DstNRange)
      scanAndAddDeps(&DstN, getScanRange(NewTopN, NewBotN, &DstN));
    return;
  }

  Node *OldTopN = getNode(OldInstrInterval.from());
  Node *OldBotN = getNode(OldInstrInterval.to());
  bool NewInstrIntervalIsBelowOld =
      OldInstrInterval.to()->comesBefore(NewInstrInterval.from());
  // 2. NewInstrInterval is below OldInstrInterval.
  if (NewInstrIntervalIsBelowOld) {
    NodeMemRange DstNRange = makeMemRangeFromNonMem(NewTopN, NewBotN);
    for (Node &DstN : DstNRange)
      scanAndAddDeps(&DstN, getScanRange(OldTopN, NewBotN, &DstN));
    return;
  }

  // 3. NewInstrInterval is above OldInstrInterval.
  NodeMemRange DstNRange = makeMemRangeFromNonMem(NewTopN, OldBotN);
  for (Node &DstN : DstNRange)
    scanAndAddDeps(&DstN, getScanRange(NewTopN, NewBotN, &DstN));
}

sandboxir::InstrInterval sandboxir::DependencyGraph::extendView(
    const sandboxir::InstrInterval &NewViewRange) {
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
      NewViewNodes.insert(N);
  }
  assert(ViewRange == NewViewRange &&
         "ViewRange should have been updated by now!");
#ifndef NDEBUG
  auto Diff = NewViewSection - OrigViewRange;
  assert(Diff.size() != 2 && "Extending View region in both directions!");
#endif
  return NewViewSection;
}

void sandboxir::DependencyGraph::collectNewViewNodesInit() {
  NewViewNodes.clear();
}
void sandboxir::DependencyGraph::collectNewViewNodesNotify() {
  if (!TrackingEnabled)
    return;
  if (Sched != nullptr)
    Sched->notifyNewViewNodes(NewViewNodes);
}

sandboxir::InstrInterval sandboxir::DependencyGraph::extendInternal(
    const DmpVector<sandboxir::Value *> &Instrs) {
  assert(sandboxir::VecUtils::areInSameBB(Instrs) &&
         "Instrs expected to be in same BB!");
  assert(none_of(Instrs,
                 [](sandboxir::Value *SBV) {
                   return cast<sandboxir::Instruction>(SBV)->isDbgInfo();
                 }) &&
         "Expected no debug info intrinsics!");
  // 1. Extend the DAGInterval and create new deps.
  InstrInterval FinalViewRange =
      ViewRange.getUnionSingleSpan(InstrInterval(Instrs));
  // Now if needed, extend the DAG to include nodes that are in
  // RequestedInstrInterval but not in the existing DAG.
  auto FinalDAGInterval = DAGInterval.getUnionSingleSpan(FinalViewRange);
  auto NewDAGSections = FinalDAGInterval - DAGInterval;
  for (const InstrInterval &NewSection : NewDAGSections) {
    if (NewSection.empty())
      continue;
    // Extend the DAG to include the new instructions.
    extendDAG(DAGInterval, NewSection);
    // Update the region to include the new section
    DAGInterval = DAGInterval.getUnionSingleSpan(NewSection);

    assert(DAGInterval.contains(ViewRange) && "View should be contained in DAG");
  }
  assert(DAGInterval == FinalDAGInterval && "DAGInterval should have been updated!");

  // 2. Extend the View
  auto NewView = extendView(FinalViewRange);
  return NewView;
}

sandboxir::InstrInterval sandboxir::DependencyGraph::extend(
    const DmpVector<sandboxir::Value *> &Instrs) {
  TrackingEnabled = true; // TODO: Replace with assertion?
  collectNewViewNodesInit();
  auto Range = extendInternal(Instrs);
  collectNewViewNodesNotify();
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  verify();
#endif
  return Range;
}

void sandboxir::DependencyGraph::trimView(sandboxir::Instruction *FromI,
                                          bool Above) {
  assert(ViewRange.contains(FromI) && "Expect it to be in region");
  InstrInterval ToTrim = Above ? InstrInterval{ViewRange.from(), FromI}
                               : InstrInterval{FromI, ViewRange.to()};
  for (sandboxir::Instruction &I : ToTrim) {
    auto *N = getNode(&I);
    N->resetUnscheduledSuccs();
  }
  ViewRange = ViewRange.getSingleDifference(ToTrim);
}

void sandboxir::DependencyGraph::resetView() {
  if (ViewRange.empty())
    return;
  ViewRange.clear();
}

void sandboxir::DependencyGraph::clear() {
  resetView();
  InstrToNodeMap.clear();
  DAGInterval.clear();
  DepCache.clear();
}

SmallVector<std::pair<sandboxir::DependencyGraph::Node *,
                      sandboxir::DependencyGraph::Node *>>
sandboxir::DependencyGraph::getDepsFromPredsToSuccs(Node *N) {
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

void sandboxir::DependencyGraph::erase(Node *N, bool CalledByScheduler) {
  if (Sched != nullptr && !CalledByScheduler)
    Sched->notifyRemove(N->getInstruction(), /*CalledByDAG=*/true);
  // Add dependencies from N's successors to all its dependent predecessors.
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
  sandboxir::Instruction *I = N->getInstruction();
  ViewRange.erase(I, /*CheckContained=*/false);
  DAGInterval.erase(I, /*CheckContained=*/false);
  // This frees memory, so should be done last.
  InstrToNodeMap.erase(I);
}

void sandboxir::DependencyGraph::erase(sandboxir::Instruction *I) {
  auto *N = getNode(I);
  erase(N);
  // Invalidate BatchAA on erase to be on the safe side.
  BatchAA = std::make_unique<BatchAAResults>(AA);
}

sandboxir::DependencyGraph::Node *
sandboxir::DependencyGraph::insert(sandboxir::Instruction *NewI) {
  return getOrCreateNode(NewI);
}

void sandboxir::DependencyGraph::notifyInsert(sandboxir::Instruction *I) {
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  verify();
#endif
  if (!TrackingEnabled)
    return;

  collectNewViewNodesInit();
  // If we already have a Node for `I`, then the dependencies have already been
  // set. So just extend the view.
  if (auto *ExistingN = getNode(I)) {
    ;
  } else {
    Node *NewN = getOrCreateNode(I);
    if (DAGInterval.contains(I)) {
      // Create the new dependencies.
      if (NewN->isMem()) {
        // TODO: Replace this with a call to extendDAG()

        // If NewN touches memory we need to do a proper scan for dependencies.
        // Scan mem instrs above NewN and add edges to AboveN->NewN
        scanAndAddDeps(NewN, getScanRange(getNode(DAGInterval.from()),
                                          getNode(DAGInterval.to()), NewN));
        // Go over mem instrs under NewN and add edges NewN->UnderN
        Node *FirstUnderN = NewN->getNextMem();
        Node *ToUnderN = getNode(DAGInterval.to());
        if (FirstUnderN != nullptr &&
            (ToUnderN == FirstUnderN || FirstUnderN->comesBefore(ToUnderN)))
          for (Node &UnderN : makeMemRangeFromNonMem(FirstUnderN, ToUnderN))
            scanAndAddDeps(&UnderN, makeMemRange(NewN, NewN));
      }
    }
    NewViewNodes.insert(NewN);
  }
  // Extend the DAG to make sure I is included in both DAG and View.
  extendInternal({I});

  assert(DAGInterval.contains(I) &&
         "I is meant to replace instructions from within the DAG so it "
         "should be in the DAGInterval!");
  collectNewViewNodesNotify();
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  verify();
#endif
}

void sandboxir::DependencyGraph::notifyMoveInstr(
    sandboxir::Instruction *I, sandboxir::BasicBlock::iterator BeforeIt,
    sandboxir::BasicBlock *BB) {
  if (!TrackingEnabled)
    return;
  // If `I` doesn't move, nothing to do.
  if (BeforeIt == I->getIterator() || BeforeIt == std::next(I->getIterator()))
    return;
  if (DAGInterval.empty())
    return;
  // If this is a instruction motion is done by the scheduler, then the movement
  // is confined to within the regions, so just need to maintain the border
  // instructions if they move.
  DAGInterval.notifyMoveInstr(I, BeforeIt, BB);
  if (!Ctx.getTracker().inRevert()) {
    // Don't update the ViewRange when reverting, because this cannot
    // always be reverted.
    ViewRange.notifyMoveInstr(I, BeforeIt, BB);
  }
}

#ifndef NDEBUG
void sandboxir::DependencyGraph::verify(bool CheckReadyCnt) {
  assert(DAGInterval.contains(ViewRange) && "DAGInterval should contain ViewRange");
  // Check the node edges.
  for (const auto &Pair : InstrToNodeMap) {
    Node *N = Pair.second.get();
    N->verify();

    if (Sched != nullptr && Sched->scheduling() && CheckReadyCnt) {
      if (inView(N) && !N->isScheduled()) {
        unsigned Unsched = N->getNumUnscheduledSuccs();
        unsigned Cnt = 0;
        auto CountUnscheduled = [this, &Cnt](Node *N) {
          if (!inView(N))
            return;
          if (N->isScheduled())
            return;
          Cnt += 1;
        };
        if (!Sched->isTopDown()) {
          for (auto *SuccN : N->succs())
            CountUnscheduled(SuccN);
        } else {
          for (auto *PredN : N->preds())
            CountUnscheduled(PredN);
        }
        assert(Unsched == Cnt && "Bad UnscheduledSuccs!");
      }
    }
  }
}
#endif

sandboxir::DependencyGraph::DependencyGraph(sandboxir::Context &Ctx,
                                            AliasAnalysis &AA,
                                            int AAQueryBudget, Scheduler *Sched)
    : Ctx(Ctx), AA(AA), AAQueryBudget(AAQueryBudget), Sched(Sched),
      BatchAA(std::make_unique<BatchAAResults>(AA)) {}

template class sandboxir::DependencyGraph::PredSuccIteratorTemplate<
    sandboxir::User::op_iterator>;
template class sandboxir::DependencyGraph::PredSuccIteratorTemplate<
    sandboxir::Value::user_iterator>;
