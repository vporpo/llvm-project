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
#include "llvm/Transforms/Vectorize/SandboxVec/Bundle.h"
#include "llvm/Transforms/Vectorize/SandboxVec/InstrMaps.h"
#include "llvm/Transforms/Vectorize/SandboxVec/InstrRange.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SBPass.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SBPassManager.h"
#include "llvm/Transforms/Vectorize/SandboxVec/SandboxIR.h"

extern llvm::cl::opt<int> CostThreshold;

namespace llvm {

/// \Returns bundles of users for all elements in \p Bndl and whether we failed.
std::pair<SmallVector<SmallVector<SBUse, 2>, 2>, bool>
getUsesBundlesPicky(const SBValBundle &Bndl);

/// This is the actual implementation of the bottom-up vectorizer.
class VectorizeFromSeeds {
protected:
  /// This is set to true if we made any IR changes during this pass.
  bool Changed = false;

#ifndef NDEBUG
  void verifyDAGAndSchedule(const SBValBundle &Bndl,
                            SBValue *NewSBV) const;
#endif
  /// Counts the recursion calls to vectorizeRec(). Used for debugging.
  unsigned RecCnt = 0;
  /// \Returns true if we are over the recursion limit (for debugging).
  inline bool recOverDebugLimitAtInvocation() const;
  /// Stops recursion by checking `RecCnt` against the limits set by the user.
  /// It will early return or modify \p AnalysisRes. This is used for debugging.
  bool maybeStopRecursionForDebugging(const SBValBundle &Bndl,
                                      AnalysisResult *&AnalysisRes);
  SBValue *
  vectorizeRec(SBValBundle &Bndl, AnalysisResult *AnalysisRes,
               SmallPtrSet<SBInstruction *, 4> &EraseCandidates, bool TopDown,
               SBValue *VecOp = nullptr,
               unsigned OperandNo = std::numeric_limits<unsigned>::max());
  friend class VectorizeFromSeedsAttorney;
  SBBasicBlock *SBBB = nullptr;
  SBContext &Ctxt;
  ScalarEvolution &SE;
  const DataLayout &DL;
  TargetTransformInfo &TTI;
  SBAnalysis Analysis;
  /// Create unpacks hanging off \p Vec, which is the result of widening \p
  /// Scalars.
  void emitUnpacks(SBValue *Vec, const SBValBundle &Scalars);
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  void dumpInstrs(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dumpInstrs() const;
#endif
  static SBValue *createVecInstruction(const SBValBundle &Bndl,
                                         const SBValBundle &Operands,
                                         SBContext &Ctxt);

  InstructionMaps InstrMaps;
  /// For tests.
  InstructionMaps &getInstrMaps() { return InstrMaps; }
  /// This is only used for bisection debugging. It is static so that we can
  /// count across functions.
  static unsigned long InvocationCnt;

public:
  VectorizeFromSeeds(SBBasicBlock *SBBB, SBContext &Ctxt,
                     ScalarEvolution &SE, const DataLayout &DL,
                     TargetTransformInfo &TTI);
  SBInstruction *
  tryVectorize(SBValBundle &Seeds, SBRegion &Rgn,
               SmallPtrSet<SBInstruction *, 4> &EraseCandidates,
               bool TopDown = false);
  static void
  tryEraseDeadInstrs(SmallPtrSet<SBInstruction *, 4> &EraseCandidates);
  /// Used by tests.
  SBAnalysis &getAnalysis() { return Analysis; }
};

/// Used by tests.
class VectorizeFromSeedsAttorney {
public:
  static SBValue *
  vectorizeRec(VectorizeFromSeeds &Vec, SBValBundle &Bndl,
               AnalysisResult *AnalysisRes,
               SmallPtrSet<SBInstruction *, 4> &EraseCandidates, bool TopDown,
               SBValue *VecOp = nullptr) {
    return Vec.vectorizeRec(Bndl, std::move(AnalysisRes), EraseCandidates,
                            TopDown, VecOp);
  }
  static void setBB(VectorizeFromSeeds &Vec, SBBasicBlock &BB) {
    Vec.SBBB = &BB;
  }
  static InstructionMaps &getInstrMaps(VectorizeFromSeeds &Vec) {
    return Vec.getInstrMaps();
  }
};

/// A pass wrapper for the bottom-up vectorizer.
class VectorizeScalars : public SBRegionPassManager,
                         public VectorizeFromSeeds {
public:
  VectorizeScalars(SBContext &Ctxt, ScalarEvolution &SE, const DataLayout &DL,
                   TargetTransformInfo &TTI)
      : SBRegionPassManager("VectorizeScalars", "boup-vectorize"),
        VectorizeFromSeeds(nullptr, Ctxt, SE, DL, TTI) {}
  bool runOnSBBasicBlock(SBBasicBlock &BB) final;
  bool runAllPasses(SBValue &Container) final {
    return runOnSBBasicBlock(cast<SBBasicBlock>(Container));
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_PASSES_VECTORIZESCALARS_H
