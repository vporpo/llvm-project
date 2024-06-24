//===- SandboxVectorizer.h ---------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SANDBOXVECTORIZER_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SANDBOXVECTORIZER_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include <regex>

namespace llvm {

Pass *createSandboxVectorizerPass();

class TargetTransformInfo;
class BasicBlock;
class Value;

class SandboxVectorizerPass : public PassInfoMixin<SandboxVectorizerPass> {
  DenseSet<sandboxir::Value *> Visited;
  ScalarEvolution *SE = nullptr;
  const DataLayout *DL = nullptr;
  TargetTransformInfo *TTI = nullptr;
  AliasAnalysis *AA = nullptr;

  /// \Returns true if we should attempt to vectorize \p SrcFilePath based on
  /// `AllowFiles` option.
  bool allowFile(const std::string &SrcFilePath);

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  bool runImpl(Function &F);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SANDBOXVECTORIZER_H
