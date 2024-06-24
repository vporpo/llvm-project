//===- Pass.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An SBPass implements transformations that operate on SandboxIR.
//
// We define several types of passes:
//
//                   | Input        | Output
// ------------------|--------------|------
// SBFnPass          | SBFunction   | SBFunction
// SBBBPass          | SBBasicBlock | SBBasicBlock
// SBRegionPass      | SBRegion     | SBRegion
//
//
#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASS_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASS_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Region.h"

namespace llvm {
namespace sandboxir {

class Function;
class BasicBlock;
class PassManager;

/// The base class of an SB Pass.
class SBPass {
public:
  enum class ClassID : unsigned {
    FnPass,
    BBPassManager,
    BBPass,
    RegionPassManager,
    RegionPass,
  };
  static const char *getSubclassIDStr(ClassID ID) {
    switch (ID) {
    case ClassID::FnPass:
      return "FnPass";
    case ClassID::BBPassManager:
      return "BBPassManager";
    case ClassID::BBPass:
      return "BBPass";
    case ClassID::RegionPassManager:
      return "RegionPassManager";
    case ClassID::RegionPass:
      return "RegionPass";
    }
    llvm_unreachable("Unimplemented ID");
  }

protected:
  /// The pass name.
  const std::string Name;
  /// The command-line flag used to specify that this pass should run.
  const std::string Flag;
  /// Used for isa/cast/dyn_cast.
  ClassID SubclassID;

public:
  SBPass(const std::string &Name, const std::string &Flag, ClassID SubclassID)
      : Name(Name), Flag(Flag), SubclassID(SubclassID) {}
  virtual ~SBPass() {}
  StringRef getName() const { return Name; }
  StringRef getFlag() const { return Flag; }
  ClassID getSubclassID() const { return SubclassID; }
  /// Cast to an SBPassManager object or nullptr if not a pass manager.
  sandboxir::PassManager *asPM();
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, const SBPass &Pass) {
    Pass.dump(OS);
    return OS;
  }
  virtual void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD virtual void dump() const;
#endif
};

/// A pass that runs on a SBFunction
class SBFnPass : public SBPass {
protected:
  SBFnPass(const std::string &Name, const std::string &Flag, ClassID PassID)
      : SBPass(Name, Flag, PassID) {}

public:
  SBFnPass(const std::string &Name, const std::string &Flag)
      : SBPass(Name, Flag, ClassID::FnPass) {}
  /// For isa/dyn_cast etc.
  static bool classof(const SBPass *From) {
    switch (From->getSubclassID()) {
    case ClassID::FnPass:
    case ClassID::BBPassManager:
      return true;
    case ClassID::BBPass:
    case ClassID::RegionPassManager:
    case ClassID::RegionPass:
      return false;
    }
  }
  /// \Returns true if it made changes to the SBF.
  virtual bool runOnSBFunction(sandboxir::Function &SBF) = 0;
};

/// A pass that runs on a SBBB.
class SBBBPass : public SBPass {
protected:
  SBBBPass(const std::string &Name, const std::string &Flag, ClassID PassID)
      : SBPass(Name, Flag, PassID) {
    assert(!Name.empty() && "Empty name!");
    assert(!Flag.empty() && "Empty flag!");
  }

public:
  SBBBPass(const std::string &Name, const std::string &Flag)
      : SBPass(Name, Flag, ClassID::BBPass) {}
  /// For isa/dyn_cast etc.
  static bool classof(const SBPass *From) {
    switch (From->getSubclassID()) {
    case ClassID::FnPass:
    case ClassID::BBPassManager:
      return false;
    case ClassID::BBPass:
    case ClassID::RegionPassManager:
      return true;
    case ClassID::RegionPass:
      return false;
    }
  }
  /// \Returns true if it made changes to the SBBB.
  virtual bool runOnSBBasicBlock(sandboxir::BasicBlock &SBBB) = 0;
};

/// A pass that runs on a specified SB region, starting from a root node.
class RegionPass : public SBPass {
protected:
  RegionPass(const std::string &Name, const std::string &Flag, ClassID PassID)
      : SBPass(Name, Flag, PassID) {}

public:
  RegionPass(const std::string &Name, const std::string &Flag)
      : RegionPass(Name, Flag, ClassID::RegionPass) {}
  /// For isa/dyn_cast etc.
  static bool classof(const SBPass *From) {
    switch (From->getSubclassID()) {
    case ClassID::FnPass:
    case ClassID::BBPassManager:
    case ClassID::BBPass:
    case ClassID::RegionPassManager:
      return false;
    case ClassID::RegionPass:
      return true;
    }
  }
  virtual bool runOnRegion(sandboxir::Region &Rgn) = 0;
};
} // namespace sandboxir
} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASS_H
