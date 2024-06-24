add_llvm_component_library(LLVMSandboxVec
  Analysis.cpp
  CostModel.cpp
  DependencyGraph.cpp
  InstrMaps.cpp
  SandboxIRVec.cpp
  SandboxVectorizer.cpp
  Pass.cpp
  PassManager.cpp
  Region.cpp
  Scheduler.cpp
  SeedCollector.cpp
  InstrInterval.cpp
  VecUtils.cpp
  Passes/AcceptOrRevert.cpp
  Passes/CollectAndDumpSeeds.cpp
  Passes/DumpRegion.cpp
  Passes/PackReuse.cpp
  Passes/VectorizePackOperands.cpp
  Passes/VectorizeScalars.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Vectorize/SandboxVec
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Vectorize/SandboxVec/Passes

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  SandboxIR
  Support
  TransformUtils
  )
