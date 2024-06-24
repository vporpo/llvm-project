//===- SandboxVectorizer.cpp - The Sanbox Vectorizer ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#include "llvm/Transforms/Vectorize/SandboxVec/SandboxVectorizer.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Passes/AcceptOrRevert.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Passes/CollectAndDumpSeeds.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Passes/DumpRegion.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Passes/PackReuse.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Passes/VectorizePackOperands.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Passes/VectorizeScalars.h"
#include "llvm/Transforms/Vectorize/SandboxVec/PassManager.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Scheduler.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SeedCollector.h"
#include <cstring>
#include <list>

using namespace llvm;

#define SV_NAME "sandbox-vectorizer"
#define DEBUG_TYPE "SBVEC"

cl::opt<bool>
    SBVecDisable("sbvec-disable", cl::init(false), cl::Hidden,
                 cl::desc("Disable the Sandbox Vectorization passes"));

cl::opt<bool> PrintPassPipeline("sbvec-print-pass-pipeline", cl::init(false),
                                cl::Hidden,
                                cl::desc("Prints the pass pipeline."));

extern const char *DefaultPipelineMagicStr;
extern cl::opt<std::string> UserDefinedPassPipeline;

// This option is useful for bisection debugging.
// For example you may use it to figure out which filename is the one causing a
// miscompile. You can specify a regex for the filename like: "/[a-m][^/]*"
// which will enable any file name starting with 'a' to 'm' and disable the
// rest. If the miscompile goes away, then we try "/[n-z][^/]*" for the other
// half of the range, from 'n' to 'z'. If we can reproduce the miscompile then
// we can keep looking in [n-r] and [s-z] and so on, in a binary-search fashion.
//
// Please note that we are using [^/]* and not .* to make sure that we are
// matching the actual filename and not some other directory in the path.
cl::opt<std::string> AllowFiles(
    "sbvec-allow-files", cl::init(".*"), cl::Hidden,
    cl::desc("Run the vectorizer only on file paths that match any in the "
             "list of comma-separated regex's."));
static constexpr const char AllowFilesDelim = ',';

PreservedAnalyses SandboxVectorizerPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  TTI = &AM.getResult<TargetIRAnalysis>(F);
  SE = &AM.getResult<ScalarEvolutionAnalysis>(F);
  DL = &F.getParent()->getDataLayout();
  AA = &AM.getResult<AAManager>(F);

  bool Changed = runImpl(F);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

bool SandboxVectorizerPass::allowFile(const std::string &SrcFilePath) {
  // Iterate over all files in AllowFiles separated by `AllowFilesDelim`.
  size_t DelimPos = 0;
  do {
    size_t LastPos = DelimPos != 0 ? DelimPos + 1 : DelimPos;
    DelimPos = AllowFiles.find(AllowFilesDelim, LastPos);
    auto FileNameToMatch = AllowFiles.substr(LastPos, DelimPos - LastPos);
    if (FileNameToMatch.empty())
      return false;
    // Note: This only runs when debugging so its OK not to reuse the regex.
    std::regex FileNameRegex(std::string(".*") + FileNameToMatch);
    if (std::regex_match(SrcFilePath, FileNameRegex))
      return true;
  } while (DelimPos != std::string::npos);
  return false;
}

bool SandboxVectorizerPass::runImpl(Function &F) {
  if (SBVecDisable)
    return false;

  // This is used for debugging.
  if (LLVM_UNLIKELY(AllowFiles != ".*")) {
    const auto &SrcFilePath = F.getParent()->getSourceFileName();
    if (!allowFile(SrcFilePath)) {
      // dbgs() << "SBVec: Skipping " << SrcFilePath
      //        << " as requested by the user (-" << AllowFiles.ArgStr << "="
      //        << AllowFiles << ")\n";
      return false;
    }
  }

  // If the target claims to have no vector registers don't attempt
  // vectorization.
  if (!TTI->getNumberOfRegisters(TTI->getRegisterClassForType(true))) {
    LLVM_DEBUG(
        dbgs()
        << "SBVec: Didn't find any vector registers for target, abort.\n");
    return false;
  }

  // Don't vectorize when the attribute NoImplicitFloat is used.
  if (F.hasFnAttribute(Attribute::NoImplicitFloat))
    return false;

  sandboxir::SBVecContext Ctx(F.getContext(), *AA);

  LLVM_DEBUG(dbgs() << "SBVec: Analyzing blocks in " << F.getName() << ".\n");

  // Create SandboxIR for `F`.
  sandboxir::Function &SBF = *Ctx.createFunction(&F);

  // The registry owns the passes.
  sandboxir::SBPassRegistry PassRegistry(*TTI);

  // === SBVec Pass registration ===
  auto *VectorizerPM =
      cast<sandboxir::RegionPassManager>(PassRegistry.registerPass(
          std::make_unique<sandboxir::VectorizeScalars>(Ctx, *SE, *DL, *TTI)));
  auto *VectorizePackOperandsPass = PassRegistry.registerPass(
      std::make_unique<sandboxir::VectorizePackOperands>(Ctx, *SE, *DL, *TTI));
  (void)VectorizePackOperandsPass;
  PassRegistry.registerPass(
      std::make_unique<sandboxir::CollectAndDumpSeeds>(Ctx, *SE, *DL, *TTI));
  PassRegistry.registerPass(std::make_unique<sandboxir::DumpRegion>());
  auto *AcceptOrRevertPass = PassRegistry.registerPass(
      std::make_unique<sandboxir::AcceptOrRevert>(Ctx));
  auto *PackReusePass =
      PassRegistry.registerPass(std::make_unique<sandboxir::PackReuse>());
  (void)PackReusePass;
  // === End of SBVec Pass registration ===

  sandboxir::PassManager *InitPM = nullptr;
  if (UserDefinedPassPipeline != DefaultPipelineMagicStr) {
    InitPM = PassRegistry.parseAndCreateUserDefinedPassPipeline(
        UserDefinedPassPipeline);
  } else {
    // Now define the default pass pipeline.
    auto *BBPM = cast<sandboxir::SBBBPassManager>(PassRegistry.registerPass(
        std::make_unique<sandboxir::SBBBPassManager>()));
    BBPM->addPass(VectorizerPM);
    // VectorizerPM->addPass(VectorizePackOperandsPass);
    VectorizerPM->addPass(PackReusePass);
    VectorizerPM->addPass(AcceptOrRevertPass);
    InitPM = BBPM;
  }
  if (PrintPassPipeline) {
    InitPM->dumpPassPipeline(outs());
    outs() << "\n";
  }
  bool Change = InitPM->runAllPasses(SBF);
  Ctx.quickFlush();
  return Change;
}
