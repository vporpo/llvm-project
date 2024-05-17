add_subdirectory(Passes)

add_llvm_component_library(LLVMSandboxVec
  Analysis.cpp
  Bundle.cpp
  CostModel.cpp
  DependencyGraph.cpp
  InstrMaps.cpp
  SandboxIR.cpp
  SandboxIRTracker.cpp
  SandboxVectorizer.cpp
  SBPass.cpp
  SBPassManager.cpp
  SBRegion.cpp
  Scheduler.cpp
  SeedCollector.cpp
  InstrRange.cpp
  Utils.cpp 

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Vectorize
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Vectorize/SandboxVec
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Vectorize/SandboxVec/Passes

  DEPENDS
  intrinsics_gen
  LLVMVectorize

  LINK_COMPONENTS
  Analysis
  Core
  Support
  TransformUtils
  Vectorize
  SBPasses
  )
