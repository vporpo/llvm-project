//===- VectorizeScalars.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_VECTORIZESCALARS_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_VECTORIZESCALARS_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Analysis.h"
#include "llvm/SandboxIR/DmpVector.h"
#include "llvm/Transforms/Vectorize/SandboxVec/InstrInterval.h"
#include "llvm/Transforms/Vectorize/SandboxVec/InstrMaps.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Pass.h"
#include "llvm/Transforms/Vectorize/SandboxVec/PassManager.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIRVec.h"

extern llvm::cl::opt<int> CostThreshold;

namespace llvm {
namespace sandboxir {

/// \Returns bundles of users for all elements in \p Bndl and whether we failed.
std::pair<SmallVector<SmallVector<sandboxir::Use, 2>, 2>, bool>
getUsesBundlesPicky(const DmpVector<sandboxir::Value *> &Bndl);

/// This is the actual implementation of the bottom-up vectorizer.
class VectorizeFromSeeds {
protected:
  /// This is set to true if we made any IR changes during this pass.
  bool Changed = false;

#ifndef NDEBUG
  void verifyDAGAndSchedule(const DmpVector<sandboxir::Value *> &Bndl,
                            sandboxir::Value *NewSBV) const;
#endif
  /// Counts the recursion calls to vectorizeRec(). Used for debugging.
  unsigned RecCnt = 0;
  /// \Returns true if we are over the recursion limit (for debugging).
  inline bool recOverDebugLimitAtInvocation() const;
  /// Stops recursion by checking `RecCnt` against the limits set by the user.
  /// It will early return or modify \p AnalysisRes. This is used for debugging.
  bool
  maybeStopRecursionForDebugging(const DmpVector<sandboxir::Value *> &Bndl,
                                 AnalysisResult *&AnalysisRes);
  sandboxir::Value *
  vectorizeRec(const DmpVectorView<sandboxir::Value *> &Bndl,
               AnalysisResult *AnalysisRes,
               SmallPtrSet<sandboxir::Instruction *, 4> &EraseCandidates,
               bool TopDown, sandboxir::Value *VecOp = nullptr,
               unsigned OperandNo = std::numeric_limits<unsigned>::max());
  friend class VectorizeFromSeedsAttorney;
  sandboxir::BasicBlock *SBBB = nullptr;
  sandboxir::SBVecContext &Ctx;
  ScalarEvolution &SE;
  const DataLayout &DL;
  TargetTransformInfo &TTI;
  sandboxir::Analysis Analysis;
  /// Create unpacks hanging off \p Vec, which is the result of widening \p
  /// Scalars.
  void emitUnpacks(sandboxir::Value *Vec,
                   const DmpVector<sandboxir::Value *> &Scalars);
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  void dumpInstrs(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dumpInstrs() const;
#endif
  static sandboxir::Value *
  createVecInstruction(const DmpVector<sandboxir::Value *> &Bndl,
                       const DmpVector<sandboxir::Value *> &Operands,
                       sandboxir::Context &Ctx);

  InstructionMaps InstrMaps;
  /// For tests.
  InstructionMaps &getInstrMaps() { return InstrMaps; }
  /// This is only used for bisection debugging. It is static so that we can
  /// count across functions.
  static unsigned long InvocationCnt;

public:
  VectorizeFromSeeds(sandboxir::BasicBlock *SBBB,
                     sandboxir::SBVecContext &Ctx, ScalarEvolution &SE,
                     const DataLayout &DL, TargetTransformInfo &TTI);
  sandboxir::Instruction *
  tryVectorize(const DmpVectorView<sandboxir::Value *> &Seeds,
               sandboxir::Region &Rgn,
               SmallPtrSet<sandboxir::Instruction *, 4> &EraseCandidates,
               bool TopDown = false);
  static void tryEraseDeadInstrs(
      SmallPtrSet<sandboxir::Instruction *, 4> &EraseCandidates);
  /// Used by tests.
  sandboxir::Analysis &getAnalysis() { return Analysis; }
};

/// Used by tests.
class VectorizeFromSeedsAttorney {
public:
  static sandboxir::Value *
  vectorizeRec(VectorizeFromSeeds &Vec, DmpVector<sandboxir::Value *> &Bndl,
               AnalysisResult *AnalysisRes,
               SmallPtrSet<sandboxir::Instruction *, 4> &EraseCandidates,
               bool TopDown, sandboxir::Value *VecOp = nullptr) {
    return Vec.vectorizeRec(Bndl, std::move(AnalysisRes), EraseCandidates,
                            TopDown, VecOp);
  }
  static void setBB(VectorizeFromSeeds &Vec, sandboxir::BasicBlock &BB) {
    Vec.SBBB = &BB;
  }
  static InstructionMaps &getInstrMaps(VectorizeFromSeeds &Vec) {
    return Vec.getInstrMaps();
  }
};

/// A pass wrapper for the bottom-up vectorizer.
class VectorizeScalars : public sandboxir::RegionPassManager,
                         public VectorizeFromSeeds {
public:
  VectorizeScalars(sandboxir::SBVecContext &Ctx, ScalarEvolution &SE,
                   const DataLayout &DL, TargetTransformInfo &TTI)
      : sandboxir::RegionPassManager("VectorizeScalars", "boup-vectorize"),
        VectorizeFromSeeds(nullptr, Ctx, SE, DL, TTI) {}
  bool runOnSBBasicBlock(sandboxir::BasicBlock &BB) final;
  bool runAllPasses(sandboxir::Value &Container) final {
    return runOnSBBasicBlock(cast<sandboxir::BasicBlock>(Container));
  }
};

} // namespace sandboxir
} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_VECTORIZESCALARS_H
