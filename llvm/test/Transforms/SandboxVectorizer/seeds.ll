; RUN: opt -disable-output -passes=sandbox-vectorizer -sbvec-passes=collect-and-dump-seeds -sbvec-vec-reg-bits=512 %s 2>&1 | FileCheck %s
; REQUIRES: asserts
; !!!WARNING!!! This won't update automatically with update_test_checks.py !

define void @scalar_store_seeds(ptr %ptr, float %val0, float %val1, float %val2, float %val3) {
; CHECK: [Val=ptr %ptr Ty=float Opc=Store]
; CHECK-NEXT:  0.   store float %val0, ptr %gep0, align 8
; CHECK-NEXT:  1.   store float %val1, ptr %gep1, align 8
; CHECK-NEXT:  2.   store float %val2, ptr %gep2, align 8
; CHECK-NEXT:  3.   store float %val3, ptr %gep3, align 8

  %gep0 = getelementptr inbounds float, ptr %ptr, i64 0
  %gep1 = getelementptr inbounds float, ptr %ptr, i64 1
  %gep2 = getelementptr inbounds float, ptr %ptr, i64 2
  %gep3 = getelementptr inbounds float, ptr %ptr, i64 3
  store float %val0, ptr %gep0, align 8
  store float %val3, ptr %gep3, align 8
  store float %val2, ptr %gep2, align 8
  store float %val1, ptr %gep1, align 8
  ret void
}

define void @scalar_store_seeds_with_gaps(ptr %ptr, float %val0, float %val1, float %val2, float %val3) {
; CHECK: [Val=ptr %ptr Ty=float Opc=Store]
; CHECK-NEXT:  0.   store float %val0, ptr %gep0, align 8
; CHECK-NEXT:  1.   store float %val1, ptr %gep1, align 8
; CHECK-NEXT:  2.   store float %val2, ptr %gep2, align 8
; CHECK-NEXT:  3.   store float %val3, ptr %gep3, align 8
  %gep0 = getelementptr inbounds float, ptr %ptr, i64 0
  %gep1 = getelementptr inbounds float, ptr %ptr, i64 3
  %gep2 = getelementptr inbounds float, ptr %ptr, i64 4
  %gep3 = getelementptr inbounds float, ptr %ptr, i64 6
  store float %val0, ptr %gep0, align 8
  store float %val3, ptr %gep3, align 8
  store float %val2, ptr %gep2, align 8
  store float %val1, ptr %gep1, align 8
  ret void
}

define void @mixed_scalar_vector_store_seeds(ptr %ptr, float %val0, <2 x float> %val1, float %val3) {
; CHECK: [Val=ptr %ptr Ty=float Opc=Store]
; CHECK-NEXT:  0.   store float %val0, ptr %gep0, align 8
; CHECK-NEXT:  1.   store <2 x float> %val1, ptr %gep1, align 8
; CHECK-NEXT:  2.   store float %val3, ptr %gep3, align 8
  %gep0 = getelementptr inbounds float, ptr %ptr, i64 0
  %gep1 = getelementptr inbounds float, ptr %ptr, i64 1
  %gep3 = getelementptr inbounds float, ptr %ptr, i64 3
  store float %val0, ptr %gep0, align 8
  store <2 x float> %val1, ptr %gep1, align 8
  store float %val3, ptr %gep3, align 8
  ret void
}
