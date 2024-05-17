//===- SBPassManager.cpp - Registers and executes SB passes ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/SBPassManager.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"

using namespace llvm;

/// A magic string for the default pass pipeline.
const char *DefaultPipelineMagicStr = "*";

// Pass pipeline syntax:
//   <pass1>[,<pass2>[,...<passN>]]
//   For example: -sbvec-passes=BBpass1,BBpass2
// This implicitly adds a pass-manager for each consecutive group of passes.
// For example:
//   BBpass1,RGNpass1,RGNpass2,BBpass2
// is equivalent to long syntax:
//   bb-pass-manager(BBpass1,rgn-pass-manager(RGNpass1,RGNpass2),BBpass2)
//
cl::opt<std::string> UserDefinedPassPipeline(
    "sbvec-passes", cl::init(DefaultPipelineMagicStr), cl::Hidden,
    cl::desc("Comma-separated list of SBVec sub-passes. If not set "
             "we run the predefined pipeline."));

SBPass *SBPassRegistry::getPassByName(StringRef PassName) const {
  auto It = find_if(PassesPool, [PassName](const auto &PassPtr) {
    return PassPtr->getName() == PassName;
  });
  return It != PassesPool.end() ? It->get() : nullptr;
}

SBPass *SBPassRegistry::getPassByFlag(StringRef PassFlag) const {
  auto It = find_if(PassesPool, [PassFlag](const auto &PassPtr) {
    return PassPtr->getFlag() == PassFlag;
  });
  return It != PassesPool.end() ? It->get() : nullptr;
}

SBPass *SBPassRegistry::registerPass(std::unique_ptr<SBPass> &&PassPtr) {
  auto *Pass = PassPtr.get();
  StringRef PassName = Pass->getName();
  // Passes should have unique names/flags except for the pass managers.
  if (!Pass->asPM()) {
    if (auto *ExistingPass = getPassByName(PassName)) {
      errs() << "Pass with name '" << PassName << "' already exists!\n";
      errs() << "Existing pass flag: '" << ExistingPass->getFlag() << "'\n";
      exit(1);
    }
    const auto &PassFlag = Pass->getFlag();
    if (auto *ExistingPass = getPassByFlag(PassFlag)) {
      errs() << "Pass with flag '" << PassFlag << "' arleady exists!\n";
      errs() << "Existing pass name: '" << ExistingPass->getName() << "'\n";
      exit(1);
    }
  }
  PassesPool.push_back(std::move(PassPtr));
  return Pass;
}

SBPassManager *SBPassRegistry::parseAndCreateUserDefinedPassPipeline(
    const std::string &UserPipelineStr) {
  // Add EndToken to the end to ease parsing.
  std::string PipelineStr = UserPipelineStr + EndToken;
  int FlagBeginIdx = 0;
  SmallVector<SBPassManager *, 2> PassManagerStack;
  // Start with a BBPM
  auto *InitialPM = cast<SBBBPassManager>(
      registerPass(std::make_unique<SBBBPassManager>()));
  PassManagerStack.push_back(InitialPM);

  // Make sure there is an AcceptOrRevert pass at the end of each RegionPM.
  // This is for convenience when writing lit tests.
  auto TryAddAcceptOrRevertPass = [this, &PassManagerStack] {
    auto *CurrPM = PassManagerStack.back();
    if (isa<SBRegionPassManager>(CurrPM)) {
      auto *AcceptOrRevertPass = getPassByName("AcceptOrRevert");
      assert(AcceptOrRevertPass != nullptr && "Pass not registered!");
      auto *LastPass = CurrPM->getLastPass();
      if (LastPass != nullptr && LastPass != AcceptOrRevertPass)
        CurrPM->addPass(AcceptOrRevertPass);
    }
  };

  for (auto [Idx, C] : enumerate(PipelineStr)) {
    bool FoundDelim = C == EndToken || C == PassDelimToken;
    if (!FoundDelim)
      continue;
    // Create a new PassDescr node.
    unsigned Sz = Idx - FlagBeginIdx;
    std::string PassFlag(&PipelineStr[FlagBeginIdx], Sz);
    FlagBeginIdx = Idx + 1;
    if (PassFlag.empty())
      continue;

    SBPass *Pass = getPassByFlag(PassFlag);
    if (Pass == nullptr) {
      errs() << "Pass '" << PassFlag << "' not registered!\n";
      exit(1);
    }
    auto *CurrPM = PassManagerStack.back();
    // Add the missing pass manager for this type of pass.
    switch (Pass->getSubclassID()) {
    case SBPass::ClassID::BBPass: {
      TryAddAcceptOrRevertPass();
      // If we are going from a Region pass to a BB Pass, update the current PM
      // in the PM stack.
      if (!isa<SBBBPassManager>(CurrPM)) {
        assert(isa<SBRegionPassManager>(CurrPM) && "New PM type!");
        PassManagerStack.pop_back();
        CurrPM = PassManagerStack.back();
        assert(isa<SBBBPassManager>(CurrPM) && "Expected BBPM!");
      }
      // Add the pass to the pass manager.
      CurrPM->addPass(Pass);
      break;
    }
    case SBPass::ClassID::RegionPass: {
      // Create the missing pass manager if needed.
      if (!isa<SBRegionPassManager>(CurrPM)) {
        assert(isa<SBBBPassManager>(CurrPM) && "New PM type!");
        auto *NewPM = cast<SBRegionPassManager>(
            registerPass(std::make_unique<DefaultRegionPassManager>(TTI)));
        CurrPM->addPass(NewPM);
        PassManagerStack.push_back(NewPM);
        CurrPM = NewPM;
      }
      // Add the pass to the pass manager.
      CurrPM->addPass(Pass);
      break;
    }
    case SBPass::ClassID::BBPassManager:
      errs() << "The initial BB Pass manager is added automatically!";
      exit(1);
    case SBPass::ClassID::FnPass:
      llvm_unreachable("Unimplemented!");
    case SBPass::ClassID::RegionPassManager: {
      // Add the new pass manager.
      CurrPM->addPass(Pass);
      auto *PM = Pass->asPM();
      PassManagerStack.push_back(PM);
      break;
    }
    }
  }
  TryAddAcceptOrRevertPass();
  return PassManagerStack.front();
}

#ifndef NDEBUG
void SBPassRegistry::dump(raw_ostream &OS) const {
  SmallVector<SBPass *> SortedPasses;
  for (const auto &PassPtr : PassesPool)
    SortedPasses.push_back(PassPtr.get());
  sort(SortedPasses, [](const SBPass *Pass1, const SBPass *Pass2) {
    return Pass1->getName() < Pass2->getName();
  });
  for (const auto *Pass : SortedPasses)
    OS << Pass->getName() << " " << Pass->getFlag() << "\n";
  OS << "\n";
}

void SBPassRegistry::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

SBPass *SBPassManager::asPass() {
  switch (SubclassID) {
  case SBPass::ClassID::BBPassManager:
    return cast<SBBBPassManager>(this);
  case SBPass::ClassID::RegionPassManager:
    return cast<SBRegionPassManager>(this);
  case SBPass::ClassID::FnPass:
  case SBPass::ClassID::BBPass:
  case SBPass::ClassID::RegionPass:
    llvm_unreachable("Bad ID!");
  }
}

void SBPassManager::addPass(SBPass *Pass) {
#ifndef NDEBUG
  switch (SubclassID) {
  case SBPass::ClassID::BBPassManager:
    assert(isa<SBBBPass>(Pass) && "Expected BB Pass!");
    break;
  case SBPass::ClassID::RegionPassManager:
    assert(isa<SBRegionPass>(Pass) && "Expected Region Pass!");
    break;
  case SBPass::ClassID::FnPass:
  case SBPass::ClassID::BBPass:
  case SBPass::ClassID::RegionPass:
    llvm_unreachable("Bad ID!");
  }
#endif
  Passes.push_back(Pass);
}

void SBPassManager::dumpPassPipeline(raw_ostream &OS) const {
  OS << const_cast<SBPassManager *>(this)->asPass()->getFlag() << "(";
  for (auto [Idx, Pass] : enumerate(Passes)) {
    if (auto *PassPM = Pass->asPM()) {
      PassPM->dumpPassPipeline(OS);
    } else {
      OS << Pass->getFlag();
    }
    if (Idx + 1 != Passes.size())
      OS << SBPassRegistry::PassDelimToken;
  }
  OS << ")";
}

#ifndef NDEBUG
void SBPassManager::dumpPassPipeline() const {
  dumpPassPipeline(dbgs());
  dbgs() << "\n";
}

void SBPassManager::dump(raw_ostream &OS) const {
  OS << const_cast<SBPassManager *>(this)->asPass()->getFlag();
  OS << "(";
  for (auto [Idx, Pass] : enumerate(Passes)) {
    OS << Pass->getFlag();
    if (Idx + 1 != Passes.size())
      OS << ",";
  }
  OS << ")";
}

void SBPassManager::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

bool SBBBPassManager::runAllPassesOnBB(SBBasicBlock &SBBB) {
  bool ChangeAll = false;
#ifndef NDEBUG
  SBContext &Ctxt = SBBB.getContext();
#endif
  for (SBPass *GenericPass : Passes) {
    SBBBPass *Pass = dyn_cast<SBBBPass>(GenericPass);
    if (Pass == nullptr)
      continue;
#ifndef NDEBUG
    auto HashBefore = SBBB.hash();
#endif
    assert(Ctxt.getTracker().empty() &&
           "Expected empty tracker before running the pass!");
    bool Change = Pass->runOnSBBasicBlock(SBBB);
    assert(Ctxt.getTracker().empty() &&
           "It is the pass's responsibility to clean the tracker!");
#ifndef NDEBUG
#ifdef SBVEC_EXPENSIVE_CHECKS
    SBBB.verify();
#endif
    auto HashAfter = SBBB.hash();
    if (HashAfter != HashBefore && !Change) {
      errs() << "Bad return value of pass: " << Pass->getName() << "\n";
      llvm_unreachable("SBBB changed but did pass not return true!");
    }
#endif // NDEBUG
    if (!Change)
      continue;
#ifndef NDEBUG
    // TODO: Disable the verifier.
    // SBBB.verifyFunctionIR();
    SBBB.verifyIR();
#endif
    // SBBB.getContext().getScheduler()->acceptAndRestart();
    ChangeAll = true;
  }
  return ChangeAll;
}

bool SBBBPassManager::runAllPasses(SBValue &Container) {
  auto *SBF = cast<SBFunction>(&Container);
  bool Change = false;
  auto &SBCtxt = SBF->getContext();
  // Visit BBs in post-order. Note: This skips unreachable blocks.
  // for (BasicBlock *BB : post_order(&SBF.getEntryBlock())) {
  // Visit BBs in post-order. Note: This skips unreachable blocks.
  for (SBBasicBlock *SBBB : post_order(&SBF->getEntryBlock())) {
    assert(!SBCtxt.getTracker().tracking() &&
           "Don't track the creation of SBBB");
    // Start tracking.
    SBCtxt.getTracker().start(SBBB);
    assert(SBCtxt.getTracker().empty() && "Expected empty tracker.");
    // Now run all the SBVec passes on it.
    Change |= runAllPassesOnBB(*SBBB);
    assert(SBCtxt.getTracker().empty() && "Expected empty tracker.");
    SBCtxt.getTracker().stop();
  }
  return Change;
}

bool SBBBPassManager::runOnSBFunction(SBFunction &SBF) {
  return runAllPasses(SBF);
}

bool SBRegionPassManager::runAllPassesOnRgn(SBRegion &Rgn) {
  bool Change = false;
#ifndef NDEBUG
  auto *SBBB = Rgn.getParent();
#endif
  for (SBPass *Pass : Passes) {
    auto *RgnPass = cast<SBRegionPass>(Pass);
#ifndef NDEBUG
    auto HashBefore = SBBB->hash();
#endif
    bool LocalChange = RgnPass->runOnRegion(Rgn);
    Change |= LocalChange;
#ifndef NDEBUG
#ifdef SBVEC_EXPENSIVE_CHECKS
    SBBB->verify();
#endif
    auto HashAfter = SBBB->hash();
    assert((HashAfter == HashBefore || LocalChange) &&
           "SBBB changed but did pass not return true!");
#endif // NDEBUG
    if (!LocalChange)
      continue;
#ifndef NDEBUG
    // TODO: Disable the verifier.
    // SBBB.verifyFunctionIR();
    SBBB->verifyIR();
#endif
  }
  assert(Rgn.getContext().getTracker().empty() &&
         "Should have been accepted/rejected by now!");
  return Change;
}

bool DefaultRegionPassManager::runAllPasses(SBValue &Container) {
  auto *SBBB = cast<SBBasicBlock>(&Container);
  bool ChangeAll = false;
  SBRegionBuilderFromMD RgnBuilder(SBBB->getContext(), TTI);
  for (auto &RgnPtr : RgnBuilder.createRegionsFromMD(*SBBB))
    ChangeAll |= runAllPassesOnRgn(*RgnPtr);
  return ChangeAll;
}

bool DefaultRegionPassManager::runOnSBBasicBlock(SBBasicBlock &SBBB) {
  return runAllPasses(SBBB);
}
