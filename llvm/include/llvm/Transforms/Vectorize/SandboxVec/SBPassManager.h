//===- SBPassManager.h --------------------------------------*- C++ -*-===//
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
//                     | PM Input      | Pass In | Pass Out
// --------------------|---------------|---------|---------
// SBBBPassManager     | SBFunction    | SBBB    | SBBB
// SBRegionPassManager | SBBB          | Region  | Region
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SBPASSMANAGER_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SBPASSMANAGER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SBPass.h"
#include <memory>

namespace llvm {

class TargetTransformInfo;
class SBFunction;
class SBValue;

class SBPassManager;
class SBBBPassManager;
class SBRegionPassManager;

/// Holds the pass objects and their names.
class SBPassRegistry {
  /// Owns the passes.
  SmallVector<std::unique_ptr<SBPass>> PassesPool;
  TargetTransformInfo &TTI;

  /// \Returns the pass with name \p PassName, or null if not found.
  SBPass *getPassByName(StringRef PassName) const;
  /// \Returns the pass with the given \p PassFlag, or null if not found.
  SBPass *getPassByFlag(StringRef PassFlag) const;

public:
  static constexpr const char PassDelimToken = ',';
  static constexpr const char EndToken = '\0';
  SBPassRegistry(TargetTransformInfo &TTI) : TTI(TTI) {}
  /// Registers and takes ownership of \p PassPtr.
  SBPass *registerPass(std::unique_ptr<SBPass> &&PassPtr);
  /// \Returns the first pass manager in the pipeline.
  SBPassManager *
  parseAndCreateUserDefinedPassPipeline(const std::string &UserPipelineStr);
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  void dump() const;
#endif // NDEBUG
};

class SBPassManager {
protected:
  /// The list of passes that this pass manager will run.
  SmallVector<SBPass *> Passes;
  SBPass::ClassID SubclassID;

public:
  explicit SBPassManager(SBPass::ClassID SubclassID)
      : SubclassID(SubclassID) {}
  SBPassManager(const SBPassManager &) = delete;
  virtual ~SBPassManager() = default;
  SBPassManager &operator=(const SBPassManager &) = delete;
  auto getPMSubclassID() const { return SubclassID; }
  SBPass *asPass();
  /// Adds \p Pass to the pass pipeline.
  void addPass(SBPass *Pass);
  /// \Returns the last pass in the pipeline or null if empty.
  SBPass *getLastPass() { return !Passes.empty() ? Passes.back() : nullptr; }
  /// Runs all passes in the `Passes` list.
  virtual bool runAllPasses(SBValue &Container) = 0;
  void dumpPassPipeline(raw_ostream &OS) const;
#ifndef NDEBUG
  LLVM_DUMP_METHOD void dumpPassPipeline() const;
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

class SBBBPassManager : public SBPassManager, public SBFnPass {
  bool runAllPassesOnBB(SBBasicBlock &SBBB);

public:
  SBBBPassManager()
      : SBPassManager(SBPass::ClassID::BBPassManager),
        SBFnPass("BBPassManager", "bb-pass-manager", ClassID::BBPassManager) {}
  /// For isa/dyn_cast etc.
  static bool classof(const SBPass *From) {
    return From->getSubclassID() == ClassID::BBPassManager;
  }
  static bool classof(const SBPassManager *From) {
    return From->getPMSubclassID() == ClassID::BBPassManager;
  }
  bool runAllPasses(SBValue &Container) final;
  bool runOnSBFunction(SBFunction &SBF) final;
};

class SBRegionPassManager : public SBPassManager, public SBBBPass {
protected:
  bool runAllPassesOnRgn(SBRegion &Rgn);

public:
  SBRegionPassManager(const std::string &Name, const std::string &Flag)
      : SBPassManager(SBPass::ClassID::RegionPassManager),
        SBBBPass(Name, Flag, ClassID::RegionPassManager) {}
  /// For isa/dyn_cast etc.
  static bool classof(const SBPass *From) {
    return From->getSubclassID() == ClassID::RegionPassManager;
  }
  static bool classof(const SBPassManager *From) {
    return From->getPMSubclassID() == ClassID::RegionPassManager;
  }
};

/// The default region pass manager forms regions by parsing the metadata.
class DefaultRegionPassManager : public SBRegionPassManager {
  TargetTransformInfo &TTI;

public:
  DefaultRegionPassManager(TargetTransformInfo &TTI)
      : SBRegionPassManager("DefaultRegionPM", "default-region-pm"), TTI(TTI) {
  }
  bool runAllPasses(SBValue &Container) final;
  bool runOnSBBasicBlock(SBBasicBlock &SBBB) final;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SBPASSMANAGER_H
