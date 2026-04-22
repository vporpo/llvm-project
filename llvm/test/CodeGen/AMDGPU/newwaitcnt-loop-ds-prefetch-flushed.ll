; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -o - %s | FileCheck %s
; XFAIL: *

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -o - %s -amdgpu-enable-new-insert-waitcnts | FileCheck %s

; Test for DS prefetch with flush points: preheader has DS loads and SMEM loads
; for kernel arguments. Loop uses DS results and one SMEM result (loop bound).
;
; The new pass has BETTER KmCnt scheduling than the old pass:
;
; Old pass emits s_wait_kmcnt 0 early, blocking until BOTH SMEM loads complete:
;   s_load_b32 s1, ...          ; KmCnt = 1 (LDS base address)
;   s_load_b32 s0, ...          ; KmCnt = 2 (loop bound)
;   s_wait_kmcnt 0x0            ; STALL: wait for both s0 and s1
;   v_lshl_add_u32 v11, ..., s1 ; use s1
;   ...
;   ds_load_b64 ...             ; DS loads blocked by KmCnt stall
;   ds_load_b64 ...
;   s_wait_dscnt 0x0
;   ; loop starts
;
; New pass waits incrementally, allowing more overlap:
;   s_load_b32 s1, ...          ; KmCnt = 1 (LDS base address)
;   s_load_b32 s0, ...          ; KmCnt = 2 (loop bound)
;   s_wait_kmcnt 0x1            ; wait for s1 only (s0 still loading)
;   v_lshl_add_u32 v11, ..., s1 ; use s1
;   ...                         ; ALU work overlaps with s0 load
;   ds_load_b64 ...             ; DS loads issue while s0 still loading
;   ds_load_b64 ...
;   s_wait_dscnt 0x0
;   s_wait_kmcnt 0x0            ; wait for s0 right before loop needs it
;   ; loop starts
;
; The new pass allows DS loads and ALU instructions to execute in parallel
; with the remaining s0 SMEM latency, reducing total stall time.
;
; This test is XFAIL because the old pass cannot match the new pass's output.

; CHECK-LABEL: ds_prefetch_flushed:
; CHECK:       ; %bb.0: ; %entry
; CHECK:         s_load_b32 s1,
; CHECK-NEXT:    s_load_b32 s0,
; CHECK:         s_wait_kmcnt 0x1
; CHECK:         v_lshl_add_u32 v11, v10, 6, s1
; CHECK:         ds_load_b64
; CHECK-NEXT:    ds_load_b64
; CHECK-NEXT:    s_wait_dscnt 0x0
; CHECK-NEXT:    s_wait_kmcnt 0x0
; CHECK-NEXT:  .LBB0_1: ; %loop

define amdgpu_kernel void @ds_prefetch_flushed(ptr addrspace(3) %lds, ptr addrspace(1) %out, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %base1 = shl i32 %tid, 2
  %base2 = shl i32 %tid, 3
  %base3 = shl i32 %tid, 4
  %base4 = shl i32 %tid, 5

  ; Preheader: single 64-bit DS load each (ds_load_b64 / 2 x float)
  %ptr.pre2 = getelementptr <2 x float>, ptr addrspace(3) %lds, i32 %base2, i32 1
  %init.v2 = load <2 x float>, ptr addrspace(3) %ptr.pre2, align 8
  %ptr.pre1 = getelementptr <2 x float>, ptr addrspace(3) %lds, i32 %base1
  %init.v1 = load <2 x float>, ptr addrspace(3) %ptr.pre1, align 8

  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi <2 x float> [ zeroinitializer, %entry ], [ %acc.next, %loop ]
  %prefetch1 = phi <2 x float> [ zeroinitializer, %entry ], [ %load3, %loop ]
  %prefetch2 = phi <2 x float> [ zeroinitializer, %entry ], [ %load4, %loop ]

  %use.pre1 = fadd <2 x float> %acc, %init.v1
  %use.pre2 = fadd <2 x float> %use.pre1, %init.v2
  %use.pf1 = fadd <2 x float> %use.pre2, %prefetch1
  %use.pf2 = fadd <2 x float> %use.pf1, %prefetch2

  call void @llvm.amdgcn.s.barrier()

  %off1 = add i32 %base1, %i
  %ptr1 = getelementptr <2 x float>, ptr addrspace(3) %lds, i32 %off1
  %load1 = load <2 x float>, ptr addrspace(3) %ptr1, align 8

  %off2 = add i32 %base2, %i
  %ptr2 = getelementptr <2 x float>, ptr addrspace(3) %lds, i32 %off2
  %load2 = load <2 x float>, ptr addrspace(3) %ptr2, align 8

  %off3 = add i32 %base3, %i
  %ptr3 = getelementptr <2 x float>, ptr addrspace(3) %lds, i32 %off3
  %load3 = load <2 x float>, ptr addrspace(3) %ptr3, align 8

  %off4 = add i32 %base4, %i
  %ptr4 = getelementptr <2 x float>, ptr addrspace(3) %lds, i32 %off4
  %load4 = load <2 x float>, ptr addrspace(3) %ptr4, align 8

  %sum = fadd <2 x float> %load1, %load2
  %acc.next = fadd <2 x float> %use.pf2, %sum

  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit, !llvm.loop !0

exit:
  %out.ptr = getelementptr <2 x float>, ptr addrspace(1) %out, i32 %tid
  store <2 x float> %acc.next, ptr addrspace(1) %out.ptr, align 8
  ret void
}

!0 = !{!1}
!1 = !{!"llvm.loop.unroll.disable"}

declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
