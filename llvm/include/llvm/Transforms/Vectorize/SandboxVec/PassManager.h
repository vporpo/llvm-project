//===- PassManager.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registers and executes the Sandbox Vectorizer passes.
//
// The pass manager contains an ordered sequence of passes that it runs.
// Note that in this design a pass manager is also a pass. So a pass manager
// runs when it is time for it to run in its parent pass-manager pass sequence.
//
// We define these types pass managers:
//
//                   | PM Input    | Pass In | Pass Out
// ------------------|-------------|---------|---------
// BBPassManager     | Function    | BB      | BB
// RegionPassManager | BB          | Region  | Region
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSMANAGER_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSMANAGER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Pass.h"
#include <memory>

namespace llvm {
namespace sandboxir {

class Function;
class Value;

class PassManager;
class SBBBPassManager;
class RegionPassManager;

/// Holds the pass objects and their names.
class SBPassRegistry {
  /// Owns the passes.
  SmallVector<std::unique_ptr<sandboxir::SBPass>> PassesPool;
  TargetTransformInfo &TTI;

  /// \Returns the pass with name \p PassName, or null if not found.
  sandboxir::SBPass *getPassByName(StringRef PassName) const;
  /// \Returns the pass with the given \p PassFlag, or null if not found.
  sandboxir::SBPass *getPassByFlag(StringRef PassFlag) const;

public:
  static constexpr const char PassDelimToken = ',';
  static constexpr const char EndToken = '\0';
  SBPassRegistry(TargetTransformInfo &TTI) : TTI(TTI) {}
  /// Registers and takes ownership of \p PassPtr.
  sandboxir::SBPass *registerPass(std::unique_ptr<sandboxir::SBPass> &&PassPtr);
  /// \Returns the first pass manager in the pipeline.
  sandboxir::PassManager *
  parseAndCreateUserDefinedPassPipeline(const std::string &UserPipelineStr);
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  void dump() const;
#endif // NDEBUG
};

class PassManager {
protected:
  /// The list of passes that this pass manager will run.
  SmallVector<sandboxir::SBPass *> Passes;
  sandboxir::SBPass::ClassID SubclassID;

public:
  explicit PassManager(sandboxir::SBPass::ClassID SubclassID)
      : SubclassID(SubclassID) {}
  PassManager(const sandboxir::PassManager &) = delete;
  virtual ~PassManager() = default;
  sandboxir::PassManager &
  operator=(const sandboxir::PassManager &) = delete;
  auto getPMSubclassID() const { return SubclassID; }
  sandboxir::SBPass *asPass();
  /// Adds \p Pass to the pass pipeline.
  void addPass(sandboxir::SBPass *Pass);
  /// \Returns the last pass in the pipeline or null if empty.
  sandboxir::SBPass *getLastPass() {
    return !Passes.empty() ? Passes.back() : nullptr;
  }
  /// Runs all passes in the `Passes` list.
  virtual bool runAllPasses(sandboxir::Value &Container) = 0;
  void dumpPassPipeline(raw_ostream &OS) const;
#ifndef NDEBUG
  LLVM_DUMP_METHOD void dumpPassPipeline() const;
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

class SBBBPassManager : public sandboxir::PassManager,
                        public sandboxir::SBFnPass {
  bool runAllPassesOnBB(sandboxir::BasicBlock &SBBB);

public:
  SBBBPassManager()
      : PassManager(sandboxir::SBPass::ClassID::BBPassManager),
        sandboxir::SBFnPass("BBPassManager", "bb-pass-manager",
                            ClassID::BBPassManager) {}
  /// For isa/dyn_cast etc.
  static bool classof(const sandboxir::SBPass *From) {
    return From->getSubclassID() == ClassID::BBPassManager;
  }
  static bool classof(const sandboxir::PassManager *From) {
    return From->getPMSubclassID() == ClassID::BBPassManager;
  }
  bool runAllPasses(sandboxir::Value &Container) final;
  bool runOnSBFunction(sandboxir::Function &SBF) final;
};

class RegionPassManager : public sandboxir::PassManager,
                            public sandboxir::SBBBPass {
protected:
  bool runAllPassesOnRgn(sandboxir::Region &Rgn);

public:
  RegionPassManager(const std::string &Name, const std::string &Flag)
      : sandboxir::PassManager(sandboxir::SBPass::ClassID::RegionPassManager),
        sandboxir::SBBBPass(Name, Flag, ClassID::RegionPassManager) {}
  /// For isa/dyn_cast etc.
  static bool classof(const sandboxir::SBPass *From) {
    return From->getSubclassID() == ClassID::RegionPassManager;
  }
  static bool classof(const sandboxir::PassManager *From) {
    return From->getPMSubclassID() == ClassID::RegionPassManager;
  }
};

/// The default region pass manager forms regions by parsing the metadata.
class DefaultRegionPassManager : public sandboxir::RegionPassManager {
  TargetTransformInfo &TTI;

public:
  DefaultRegionPassManager(TargetTransformInfo &TTI)
      : sandboxir::RegionPassManager("DefaultRegionPM", "default-region-pm"),
        TTI(TTI) {}
  bool runAllPasses(sandboxir::Value &Container) final;
  bool runOnSBBasicBlock(sandboxir::BasicBlock &SBBB) final;
};

} // namespace sandboxir
} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSMANAGER_H
