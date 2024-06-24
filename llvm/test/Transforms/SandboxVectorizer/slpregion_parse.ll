; RUN: opt -disable-output -passes=sandbox-vectorizer -sbvec-passes=dump-region %s 2>&1 | FileCheck %s
; REQUIRES: asserts
; !!!WARNING!!! This won't update automatically with update_test_checks.py !


; CHECK: RegionID: 0 ScalarCost=[[SC:[0-9]+]] VectorCost=[[VC:[0-9]+]]
; CHECK-NEXT:   store <2 x double> %ld, ptr %ptr, align 16, !sb !0
; CHECK-NEXT:   %ld = load <2 x double>, ptr %ptr, align 16, !sb !0

define void @slpregion_parse(ptr %ptr) {
bb0:
  %ptr0 = getelementptr float, ptr %ptr, i32 0
  %ptr1 = getelementptr float, ptr %ptr, i32 1
  %ld0 = load float, ptr %ptr0
  %ld1 = load float, ptr %ptr1
  store float %ld0, ptr %ptr0
  store float %ld1, ptr %ptr1

  %ld = load <2 x double>, ptr %ptr, !sb !0
  store <2 x double> %ld, ptr %ptr, !sb !0
  ret void
}

!0 = !{!"region", i32 0}        ; Region 0 top-level MDNode
