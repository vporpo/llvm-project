; RUN: opt -disable-output -passes=sandbox-vectorizer -sbvec-passes=collect-and-dump-seeds -sbvec-vec-reg-bits=512 -sbvec-seed-bundle-size-limit=2 %s 2>&1 | FileCheck %s --check-prefix=BUNDLE2
; RUN: opt -disable-output -passes=sandbox-vectorizer -sbvec-passes=collect-and-dump-seeds -sbvec-vec-reg-bits=512 -sbvec-seed-bundle-size-limit=3 %s 2>&1 | FileCheck %s --check-prefix=BUNDLE3
; RUN: opt -disable-output -passes=sandbox-vectorizer -sbvec-passes=collect-and-dump-seeds -sbvec-vec-reg-bits=512 -sbvec-seed-groups-limit=1 %s 2>&1 | FileCheck %s --check-prefix=GROUPS1
; REQUIRES: asserts
; !!!WARNING!!! This won't update automatically with update_test_checks.py !

; Checks the limits placed on seed collection for limiting compilation time.
define void @seeds(ptr %ptrA, ptr %ptrB, float %val0, float %val1, float %val2, float %val3) {
; BUNDLE2: === StoreSeeds ===
; BUNDLE2-NEXT: [Val=ptr %ptrA Ty=float Opc=Store]
; BUNDLE2-NEXT:   0.   store float %val0, ptr %gepA0, align 8
; BUNDLE2-NEXT:   1.   store float %val1, ptr %gepA1, align 8
; BUNDLE2-EMPTY: 
; BUNDLE2-NEXT:   0.   store float %val2, ptr %gepA2, align 8
; BUNDLE2-NEXT:   1.   store float %val3, ptr %gepA3, align 8
; BUNDLE2-EMPTY: 
; BUNDLE2-NEXT: [Val=ptr %ptrB Ty=float Opc=Store]
; BUNDLE2-NEXT:   0.   store float %val0, ptr %gepB0, align 8
; BUNDLE2-NEXT:   1.   store float %val1, ptr %gepB1, align 8
; BUNDLE2-EMPTY: 
; BUNDLE2-NEXT:   0.   store float %val2, ptr %gepB2, align 8
; BUNDLE2-NEXT:   1.   store float %val3, ptr %gepB3, align 8
; BUNDLE2-EMPTY: 
; BUNDLE2-NEXT: === LoadSeeds ===


; BUNDLE3: === StoreSeeds ===
; BUNDLE3-NEXT: [Val=ptr %ptrA Ty=float Opc=Store]
; BUNDLE3-NEXT:   0.   store float %val0, ptr %gepA0, align 8
; BUNDLE3-NEXT:   1.   store float %val1, ptr %gepA1, align 8
; BUNDLE3-NEXT:   2.   store float %val2, ptr %gepA2, align 8
; BUNDLE3-EMPTY: 
; BUNDLE3-NEXT: [Val=ptr %ptrB Ty=float Opc=Store]
; BUNDLE3-NEXT:   0.   store float %val0, ptr %gepB0, align 8
; BUNDLE3-NEXT:   1.   store float %val1, ptr %gepB1, align 8
; BUNDLE3-NEXT:   2.   store float %val2, ptr %gepB2, align 8
; BUNDLE3-EMPTY: 
; BUNDLE3-NEXT: === LoadSeeds ===

; GROUPS1: === StoreSeeds ===
; GROUPS1-NEXT: [Val=ptr %ptrA Ty=float Opc=Store]
; GROUPS1-NEXT:  0.   store float %val0, ptr %gepA0, align 8
; GROUPS1-NEXT:  1.   store float %val1, ptr %gepA1, align 8
; GROUPS1-NEXT:  2.   store float %val2, ptr %gepA2, align 8
; GROUPS1-NEXT:  3.   store float %val3, ptr %gepA3, align 8
; GROUPS1-EMPTY:
; GROUPS1-NEXT: === LoadSeeds ===

  %gepA0 = getelementptr inbounds float, ptr %ptrA, i64 0
  %gepA1 = getelementptr inbounds float, ptr %ptrA, i64 1
  %gepA2 = getelementptr inbounds float, ptr %ptrA, i64 2
  %gepA3 = getelementptr inbounds float, ptr %ptrA, i64 3
  store float %val0, ptr %gepA0, align 8
  store float %val1, ptr %gepA1, align 8
  store float %val2, ptr %gepA2, align 8
  store float %val3, ptr %gepA3, align 8

  %gepB0 = getelementptr inbounds float, ptr %ptrB, i64 0
  %gepB1 = getelementptr inbounds float, ptr %ptrB, i64 1
  %gepB2 = getelementptr inbounds float, ptr %ptrB, i64 2
  %gepB3 = getelementptr inbounds float, ptr %ptrB, i64 3
  store float %val0, ptr %gepB0, align 8
  store float %val1, ptr %gepB1, align 8
  store float %val2, ptr %gepB2, align 8
  store float %val3, ptr %gepB3, align 8

  ret void
}
