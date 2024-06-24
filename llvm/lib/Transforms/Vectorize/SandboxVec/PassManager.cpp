//===- PassManager.cpp - Registers and executes SB passes -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVec/PassManager.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/SandboxIR/SandboxIR.h"

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

sandboxir::SBPass *
sandboxir::SBPassRegistry::getPassByName(StringRef PassName) const {
  auto It = find_if(PassesPool, [PassName](const auto &PassPtr) {
    return PassPtr->getName() == PassName;
  });
  return It != PassesPool.end() ? It->get() : nullptr;
}

sandboxir::SBPass *
sandboxir::SBPassRegistry::getPassByFlag(StringRef PassFlag) const {
  auto It = find_if(PassesPool, [PassFlag](const auto &PassPtr) {
    return PassPtr->getFlag() == PassFlag;
  });
  return It != PassesPool.end() ? It->get() : nullptr;
}

sandboxir::SBPass *sandboxir::SBPassRegistry::registerPass(
    std::unique_ptr<sandboxir::SBPass> &&PassPtr) {
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

sandboxir::PassManager *
sandboxir::SBPassRegistry::parseAndCreateUserDefinedPassPipeline(
    const std::string &UserPipelineStr) {
  // Add EndToken to the end to ease parsing.
  std::string PipelineStr = UserPipelineStr + EndToken;
  int FlagBeginIdx = 0;
  SmallVector<sandboxir::PassManager *, 2> PassManagerStack;
  // Start with a BBPM
  auto *InitialPM = cast<sandboxir::SBBBPassManager>(
      registerPass(std::make_unique<sandboxir::SBBBPassManager>()));
  PassManagerStack.push_back(InitialPM);

  // Make sure there is an AcceptOrRevert pass at the end of each RegionPM.
  // This is for convenience when writing lit tests.
  auto TryAddAcceptOrRevertPass = [this, &PassManagerStack] {
    auto *CurrPM = PassManagerStack.back();
    if (isa<sandboxir::RegionPassManager>(CurrPM)) {
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

    sandboxir::SBPass *Pass = getPassByFlag(PassFlag);
    if (Pass == nullptr) {
      errs() << "Pass '" << PassFlag << "' not registered!\n";
      exit(1);
    }
    auto *CurrPM = PassManagerStack.back();
    // Add the missing pass manager for this type of pass.
    switch (Pass->getSubclassID()) {
    case sandboxir::SBPass::ClassID::BBPass: {
      TryAddAcceptOrRevertPass();
      // If we are going from a Region pass to a BB Pass, update the current PM
      // in the PM stack.
      if (!isa<sandboxir::SBBBPassManager>(CurrPM)) {
        assert(isa<sandboxir::RegionPassManager>(CurrPM) && "New PM type!");
        PassManagerStack.pop_back();
        CurrPM = PassManagerStack.back();
        assert(isa<sandboxir::SBBBPassManager>(CurrPM) && "Expected BBPM!");
      }
      // Add the pass to the pass manager.
      CurrPM->addPass(Pass);
      break;
    }
    case sandboxir::SBPass::ClassID::RegionPass: {
      // Create the missing pass manager if needed.
      if (!isa<sandboxir::RegionPassManager>(CurrPM)) {
        assert(isa<sandboxir::SBBBPassManager>(CurrPM) && "New PM type!");
        auto *NewPM = cast<sandboxir::RegionPassManager>(
            registerPass(std::make_unique<DefaultRegionPassManager>(TTI)));
        CurrPM->addPass(NewPM);
        PassManagerStack.push_back(NewPM);
        CurrPM = NewPM;
      }
      // Add the pass to the pass manager.
      CurrPM->addPass(Pass);
      break;
    }
    case sandboxir::SBPass::ClassID::BBPassManager:
      errs() << "The initial BB Pass manager is added automatically!";
      exit(1);
    case sandboxir::SBPass::ClassID::FnPass:
      llvm_unreachable("Unimplemented!");
    case sandboxir::SBPass::ClassID::RegionPassManager: {
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
void sandboxir::SBPassRegistry::dump(raw_ostream &OS) const {
  SmallVector<sandboxir::SBPass *> SortedPasses;
  for (const auto &PassPtr : PassesPool)
    SortedPasses.push_back(PassPtr.get());
  sort(SortedPasses,
       [](const sandboxir::SBPass *Pass1, const sandboxir::SBPass *Pass2) {
         return Pass1->getName() < Pass2->getName();
       });
  for (const auto *Pass : SortedPasses)
    OS << Pass->getName() << " " << Pass->getFlag() << "\n";
  OS << "\n";
}

void sandboxir::SBPassRegistry::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

sandboxir::SBPass *sandboxir::PassManager::asPass() {
  switch (SubclassID) {
  case sandboxir::SBPass::ClassID::BBPassManager:
    return cast<sandboxir::SBBBPassManager>(this);
  case sandboxir::SBPass::ClassID::RegionPassManager:
    return cast<sandboxir::RegionPassManager>(this);
  case sandboxir::SBPass::ClassID::FnPass:
  case sandboxir::SBPass::ClassID::BBPass:
  case sandboxir::SBPass::ClassID::RegionPass:
    llvm_unreachable("Bad ID!");
  }
}

void sandboxir::PassManager::addPass(sandboxir::SBPass *Pass) {
#ifndef NDEBUG
  switch (SubclassID) {
  case sandboxir::SBPass::ClassID::BBPassManager:
    assert(isa<sandboxir::SBBBPass>(Pass) && "Expected BB Pass!");
    break;
  case sandboxir::SBPass::ClassID::RegionPassManager:
    assert(isa<sandboxir::RegionPass>(Pass) && "Expected Region Pass!");
    break;
  case sandboxir::SBPass::ClassID::FnPass:
  case sandboxir::SBPass::ClassID::BBPass:
  case sandboxir::SBPass::ClassID::RegionPass:
    llvm_unreachable("Bad ID!");
  }
#endif
  Passes.push_back(Pass);
}

void sandboxir::PassManager::dumpPassPipeline(raw_ostream &OS) const {
  OS << const_cast<sandboxir::PassManager *>(this)->asPass()->getFlag()
     << "(";
  for (auto [Idx, Pass] : enumerate(Passes)) {
    if (auto *PassPM = Pass->asPM()) {
      PassPM->dumpPassPipeline(OS);
    } else {
      OS << Pass->getFlag();
    }
    if (Idx + 1 != Passes.size())
      OS << sandboxir::SBPassRegistry::PassDelimToken;
  }
  OS << ")";
}

#ifndef NDEBUG
void sandboxir::PassManager::dumpPassPipeline() const {
  dumpPassPipeline(dbgs());
  dbgs() << "\n";
}

void sandboxir::PassManager::dump(raw_ostream &OS) const {
  OS << const_cast<sandboxir::PassManager *>(this)->asPass()->getFlag();
  OS << "(";
  for (auto [Idx, Pass] : enumerate(Passes)) {
    OS << Pass->getFlag();
    if (Idx + 1 != Passes.size())
      OS << ",";
  }
  OS << ")";
}

void sandboxir::PassManager::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif

bool sandboxir::SBBBPassManager::runAllPassesOnBB(
    sandboxir::BasicBlock &SBBB) {
  bool ChangeAll = false;
#ifndef NDEBUG
  sandboxir::Context &Ctx = SBBB.getContext();
#endif
  for (sandboxir::SBPass *GenericPass : Passes) {
    sandboxir::SBBBPass *Pass = dyn_cast<sandboxir::SBBBPass>(GenericPass);
    if (Pass == nullptr)
      continue;
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
    auto HashBefore = SBBB.hash();
#endif
    assert(Ctx.getTracker().empty() &&
           "Expected empty tracker before running the pass!");
    bool Change = Pass->runOnSBBasicBlock(SBBB);
    assert(Ctx.getTracker().empty() &&
           "It is the pass's responsibility to clean the tracker!");
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
    SBBB.verify();
    auto HashAfter = SBBB.hash();
    if (HashAfter != HashBefore && !Change) {
      errs() << "Bad return value of pass: " << Pass->getName() << "\n";
      llvm_unreachable("SBBB changed but did pass not return true!");
    }
#endif
    if (!Change)
      continue;
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
    // TODO: Disable the verifier.
    // SBBB.verifyFunctionIR();
    SBBB.verifyLLVMIR();
#endif
    // SBBB.getContext().getScheduler()->acceptAndRestart();
    ChangeAll = true;
  }
  return ChangeAll;
}

bool sandboxir::SBBBPassManager::runAllPasses(sandboxir::Value &Container) {
  auto *SBF = cast<sandboxir::Function>(&Container);
  bool Change = false;
  auto &SBCtx = SBF->getContext();
  // Visit BBs in post-order. Note: This skips unreachable blocks.
  // for (BasicBlock *BB : post_order(&SBF.getEntryBlock())) {
  // Visit BBs in post-order. Note: This skips unreachable blocks.
  for (sandboxir::BasicBlock *SBBB : post_order(&SBF->getEntryBlock())) {
    assert(!SBCtx.getTracker().tracking() &&
           "Don't track the creation of SBBB");
    // Start tracking.
    SBCtx.getTracker().start(SBBB);
    assert(SBCtx.getTracker().empty() && "Expected empty tracker.");
    // Now run all the SBVec passes on it.
    Change |= runAllPassesOnBB(*SBBB);
    assert(SBCtx.getTracker().empty() && "Expected empty tracker.");
    SBCtx.getTracker().stop();
  }
  return Change;
}

bool sandboxir::SBBBPassManager::runOnSBFunction(sandboxir::Function &SBF) {
  return runAllPasses(SBF);
}

bool sandboxir::RegionPassManager::runAllPassesOnRgn(
    sandboxir::Region &Rgn) {
  bool Change = false;
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  auto *SBBB = Rgn.getParent();
#endif
  for (sandboxir::SBPass *Pass : Passes) {
    auto *RgnPass = cast<sandboxir::RegionPass>(Pass);
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
    auto HashBefore = SBBB->hash();
#endif
    bool LocalChange = RgnPass->runOnRegion(Rgn);
    Change |= LocalChange;
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
    SBBB->verify();
    auto HashAfter = SBBB->hash();
    assert((HashAfter == HashBefore || LocalChange) &&
           "SBBB changed but did pass not return true!");
#endif
    if (!LocalChange)
      continue;
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
    // TODO: Disable the verifier.
    // SBBB.verifyFunctionIR();
    SBBB->verifyLLVMIR();
#endif
  }
  assert(Rgn.getContext().getTracker().empty() &&
         "Should have been accepted/rejected by now!");
  return Change;
}

bool sandboxir::DefaultRegionPassManager::runAllPasses(
    sandboxir::Value &Container) {
  auto *SBBB = cast<sandboxir::BasicBlock>(&Container);
  bool ChangeAll = false;
  sandboxir::RegionBuilderFromMD RgnBuilder(SBBB->getContext(), TTI);
  for (auto &RgnPtr : RgnBuilder.createRegionsFromMD(*SBBB))
    ChangeAll |= runAllPassesOnRgn(*RgnPtr);
  return ChangeAll;
}

bool sandboxir::DefaultRegionPassManager::runOnSBBasicBlock(
    sandboxir::BasicBlock &SBBB) {
  return runAllPasses(SBBB);
}
