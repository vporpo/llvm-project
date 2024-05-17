add_llvm_component_library(LLVMSBPasses
  AcceptOrRevert.cpp
  CollectAndDumpSeeds.cpp
  DumpRegion.cpp
  PackReuse.cpp
  VectorizePackOperands.cpp
  VectorizeScalars.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Vectorize/SandboxVec/Passes

  DEPENDS
  intrinsics_gen
  LLVMSandboxVec

  LINK_COMPONENTS
  Vectorize
  SandboxVec
)
