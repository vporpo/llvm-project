set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Vectorize
  AsmParser
  TargetParser
  )

add_llvm_unittest(SandboxVectorizeTests
  AnalysisTest.cpp
  BundleTest.cpp
  DependencyGraphTest.cpp
  InstrRangeTest.cpp
  SBRegionTest.cpp
  SchedulerTest.cpp
  SeedCollectorTest.cpp
  SandboxIRTest.cpp
  SandboxIRTrackerTest.cpp
  UtilsTest.cpp
  VectorizeScalarsTest.cpp
  )

set_property(TARGET SandboxVectorizeTests PROPERTY FOLDER "Tests/UnitTests/TransformsTests")
