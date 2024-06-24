set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Vectorize
  AsmParser
  TargetParser
  SandboxIR
  SandboxVec
  )

add_llvm_unittest(SandboxVectorizeTests
  AnalysisTest.cpp
  DependencyGraphTest.cpp
  InstrIntervalTest.cpp
  RegionTest.cpp
  SchedulerTest.cpp
  SeedCollectorTest.cpp
  SandboxIRVecTest.cpp
  SandboxIRVecTrackerTest.cpp
  VecUtilsTest.cpp
  VectorizeScalarsTest.cpp
  )

set_property(TARGET SandboxVectorizeTests PROPERTY FOLDER "Tests/UnitTests/TransformsTests")
