add_llvm_component_library(LLVMSandboxIR
  SandboxIR.cpp
  SandboxIRTracker.cpp
  DmpVector.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SandboxIR

  LINK_COMPONENTS
  Core
  Support
  )

