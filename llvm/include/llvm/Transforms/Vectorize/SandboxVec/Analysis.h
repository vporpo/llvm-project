//===- Analysis.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Analyze bundles and check if/how we can vectorize them.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_ANALYSIS_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_ANALYSIS_H

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Bundle.h"
#include "llvm/Transforms/Vectorize/SandboxVec/InstrMaps.h"
#include "llvm/Transforms/Vectorize/SandboxVec/Utils.h"
#include <memory>

namespace llvm {

class SBInstruction;
class SBBasicBlock;

enum class ResultID {
  PerfectReuseVector, ///> No shuffle needed
  UnpackAndPack,      ///> Collect values from 2 or more vectors
  Unpack,             ///> Collect values from 1 vector
  UnpackTopDown,      ///> Create unpacks for all values in the bundle.
  Pack,               ///> Collect scalar values
  SimpleWiden,
  KeepLane0,
};

const char *resultIDToStr(ResultID ID);

enum class NoVecReason {
  NonInstructions,
  DiffOpcodes,
  DiffTypes,
  DiffMathFlags,
  DiffWrapFlags,
  UnpackAndShuffle,
  CantSchedule,
  NonConsecutive,
  Other,
  Opaque,
  OrigSource,
  UnsupportedOpcode,
  CrossBBs,
  RecursionLimit,
  Unimplemented,
};

const char *noVecReasonToStr(NoVecReason R);

class SBAnalysis;

/// Abstract class describing the analysis result.
class AnalysisResult {
protected:
  /// For isa/dyn_cast.
  ResultID SubclassID;
  SBAnalysis &Analysis;

public:
  AnalysisResult(ResultID ID, const SBValBundle &Vals, SBAnalysis &Analysis);
  virtual ~AnalysisResult() {}
  ResultID getSubclassID() const { return SubclassID; }
  hash_code hashCommon() const { return hash_value(SubclassID); }
  virtual hash_code hash() const = 0;
  friend hash_code hash_value(const AnalysisResult &R) { return R.hash(); }
#ifndef NDEBUG
  void dumpCommon(raw_ostream &OS) const { OS << resultIDToStr(SubclassID); }
  virtual void dump(raw_ostream &OS) const = 0;
  LLVM_DUMP_METHOD virtual void dump() const = 0;
  friend raw_ostream &operator<<(raw_ostream &OS, const AnalysisResult &R) {
    R.dump(OS);
    return OS;
  }
#endif // NDEBUG
};

class PerfectReuseVector : public AnalysisResult {
  SBValue *CommonOp = nullptr;
  ShuffleMask ShuffMask;

public:
  /// Creates a perfect diamond (no shuffling).
  PerfectReuseVector(SBValue *CommonOp, const SBValBundle &Vals,
                     SBAnalysis &A);
  PerfectReuseVector(SBValue *CommonOp, const ShuffleMask &ShuffMask,
                     const SBValBundle &Vals, SBAnalysis &A)
      : AnalysisResult(ResultID::PerfectReuseVector, Vals, A),
        CommonOp(CommonOp), ShuffMask(ShuffMask) {}
  PerfectReuseVector(SBValue *CommonOp, ShuffleMask &&ShuffMask,
                     const SBValBundle &Vals, SBAnalysis &Analysis)
      : AnalysisResult(ResultID::PerfectReuseVector, Vals, Analysis),
        CommonOp(CommonOp), ShuffMask(std::move(ShuffMask)) {}
  SBValue *getCommonVal() const { return CommonOp; }
  const ShuffleMask &getShuffleMask() const { return ShuffMask; }
  bool needsShuffle() const { return !ShuffMask.isIdentity(); }
  /// For isa/dyn_cast.
  static bool classof(const AnalysisResult *R) {
    return R->getSubclassID() == ResultID::PerfectReuseVector;
  }
  hash_code hash() const final {
    return hash_combine(hashCommon(), CommonOp, ShuffMask);
  }
  friend hash_code hash_value(const PerfectReuseVector &R) { return R.hash(); }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const PerfectReuseVector &R) {
    R.dump(OS);
    return OS;
  }
#endif // NDEBUG
};

class AnalysisResultWithReason : public AnalysisResult {
  NoVecReason Reason;

public:
  AnalysisResultWithReason(ResultID ID, NoVecReason Reason,
                           const SBValBundle &Vals, SBAnalysis &Analysis)
      : AnalysisResult(ID, Vals, Analysis), Reason(Reason) {}
  NoVecReason getReason() const { return Reason; }
  hash_code hash() const override { return hash_combine(hashCommon(), Reason); }
  friend hash_code hash_value(const AnalysisResultWithReason &R) {
    return R.hash();
  }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const override {
    dumpCommon(OS);
    OS << " " << noVecReasonToStr(Reason);
  }
  LLVM_DUMP_METHOD void dump() const override;
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const AnalysisResultWithReason &R) {
    static_cast<const AnalysisResult *>(&R)->dump(OS);
    return OS;
  }
#endif // NDEBUG
};

class UnpackTopDown : public AnalysisResultWithReason {
public:
  UnpackTopDown(NoVecReason Reason, const SBValBundle &Vals,
                SBAnalysis &Analysis)
      : AnalysisResultWithReason(ResultID::UnpackTopDown, Reason, Vals,
                                 Analysis) {}
  /// For isa/dyn_cast.
  static bool classof(const AnalysisResult *R) {
    return R->getSubclassID() == ResultID::UnpackTopDown;
  }
  hash_code hash() const final { return hashCommon(); }
  friend hash_code hash_value(const UnpackTopDown &R) { return R.hash(); }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { AnalysisResultWithReason::dump(OS); }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const UnpackTopDown &R) {
    static_cast<const AnalysisResult *>(&R)->dump(OS);
    return OS;
  }
#endif // NDEBUG
};

class Unpack : public AnalysisResult {
  SBValue *VecOp;
  int FirstLaneToUnpack;
  int NumLanesToUnpack;

public:
  Unpack(SBValue *VecOp, int FirstLaneToUnpack, int NumLanesToUnpack,
         const SBValBundle &Vals, SBAnalysis &Analysis)
      : AnalysisResult(ResultID::Unpack, Vals, Analysis), VecOp(VecOp),
        FirstLaneToUnpack(FirstLaneToUnpack),
        NumLanesToUnpack(NumLanesToUnpack) {}
  /// For isa/dyn_cast.
  static bool classof(const AnalysisResult *R) {
    return R->getSubclassID() == ResultID::Unpack;
  }
  SBValue *getVecOp() const { return VecOp; }
  int getFirstLaneToUnpack() const { return FirstLaneToUnpack; }
  int getNumLanesToUnpack() const { return NumLanesToUnpack; }
  hash_code hash() const final {
    return hash_combine(hashCommon(), VecOp, FirstLaneToUnpack,
                        NumLanesToUnpack);
  }
  friend hash_code hash_value(const Unpack &R) { return R.hash(); }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const Unpack &R) {
    static_cast<const AnalysisResult *>(&R)->dump(OS);
    return OS;
  }
#endif // NDEBUG
};

class UnpackAndPack : public AnalysisResult {
public:
  UnpackAndPack(const SBValBundle &Vals, SBAnalysis &Analysis)
      : AnalysisResult(ResultID::UnpackAndPack, Vals, Analysis) {}
  /// For isa/dyn_cast.
  static bool classof(const AnalysisResult *R) {
    return R->getSubclassID() == ResultID::UnpackAndPack;
  }
  hash_code hash() const final { return hash_combine(hashCommon()); }
  friend hash_code hash_value(const UnpackAndPack &R) { return R.hash(); }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const UnpackAndPack &R) {
    static_cast<const AnalysisResult *>(&R)->dump(OS);
    return OS;
  }
#endif // NDEBUG
};

class Pack : public AnalysisResultWithReason {
public:
  Pack(NoVecReason Reason, const SBValBundle &Vals, SBAnalysis &Analysis)
      : AnalysisResultWithReason(ResultID::Pack, Reason, Vals, Analysis) {}
  /// For isa/dyn_cast.
  static bool classof(const AnalysisResult *R) {
    return R->getSubclassID() == ResultID::Pack;
  }
  hash_code hash() const final { return AnalysisResultWithReason::hash(); }
  friend hash_code hash_value(const Pack &R) { return R.hash(); }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final;
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const Pack &R) {
    R.dump(OS);
    return OS;
  }
#endif // NDEBUG;
};

class SchedBundle;

class SimpleWiden : public AnalysisResult {
public:
  SimpleWiden(const SBValBundle &Vals, SBAnalysis &Analysis)
      : AnalysisResult(ResultID::SimpleWiden, Vals, Analysis) {}
  /// For isa/dyn_cast.
  static bool classof(const AnalysisResult *R) {
    return R->getSubclassID() == ResultID::SimpleWiden;
  }
  hash_code hash() const final { return hash_combine(hashCommon()); }
  friend hash_code hash_value(const SimpleWiden &R) { return R.hash(); }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final {
    dump(dbgs());
    dbgs() << "\n";
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const SimpleWiden &R) {
    R.dump(OS);
    return OS;
  }
#endif // NDEBUG;
};

class KeepLane0 : public AnalysisResult {
public:
  KeepLane0(const SBValBundle &Vals, SBAnalysis &Analysis)
      : AnalysisResult(ResultID::KeepLane0, Vals, Analysis) {}
  /// For isa/dyn_cast.
  static bool classof(const AnalysisResult *R) {
    return R->getSubclassID() == ResultID::KeepLane0;
  }
  hash_code hash() const final { return hash_combine(hashCommon()); }
  friend hash_code hash_value(const KeepLane0 &R) { return R.hash(); }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final {
    dump(dbgs());
    dbgs() << "\n";
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const KeepLane0 &R) {
    R.dump(OS);
    return OS;
  }
#endif // NDEBUG;
};

class SBAnalysis {
  SBBasicBlock *SBBB = nullptr;
  ScalarEvolution &SE;
  const DataLayout &DL;
  SmallVector<std::unique_ptr<AnalysisResult>> ResultPool;
  /// The analysis result for each individual value.
  DenseMap<SBValue *, AnalysisResult *> ValueToAnalysisMap;
#ifndef NDEBUG
  /// Used for debugging.
  SmallVector<std::pair<SBValBundle, AnalysisResult *>> BndlAnalysisVec;
#endif

  void updateMap(const SBValBundle &Vals, AnalysisResult *Result);
  // AnalysisResult constructor calls updateMap().
  friend AnalysisResult::AnalysisResult(ResultID ID, const SBValBundle &Vals,
                                        SBAnalysis &Analysis);

public:
  SBAnalysis(ScalarEvolution &SE, const DataLayout &DL);
  SBAnalysis(const SBAnalysis &) = delete;
  void init(SBBasicBlock &SBBB);
  /// An AnalysisResult factory.
  template <typename ResultT, typename... ArgsT>
  ResultT *createAnalysisResult(ArgsT... Args) {
    ResultPool.push_back(std::make_unique<ResultT>(Args..., *this));
    return cast<ResultT>(ResultPool.back().get());
  }
  /// \Returns a cached analysis result for \p SBV or null if not found.
  AnalysisResult *getAnalysisResult(SBValue *SBV) const {
    auto It = ValueToAnalysisMap.find(SBV);
    return It != ValueToAnalysisMap.end() ? It->second : nullptr;
  }
  /// Checks opcodes and other instruction-specific data and returns a
  /// BndlAnalysis result describing if and how we can widen \p SBBndl.
  std::optional<NoVecReason>
  cantVectorizeBasedOnOpcodesAndTypes(const SBValBundle &SBBndl);

  /// \Returns instructions that are probably dead in \p ProbablyDead.
  AnalysisResult *getBndlAnalysis(
      const SBValBundle &SBBndl, const InstructionMaps &InstrMaps,
      SmallPtrSet<SBInstruction *, 4> &ProbablyDead, bool TopDown);
  /// Clears the maps.
  void clear();
#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  void dumpInstrs(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dumpInstrs() const;
#endif
};

raw_ostream &operator<<(raw_ostream &Os, ResultID ID);

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_ANALYSIS_H
