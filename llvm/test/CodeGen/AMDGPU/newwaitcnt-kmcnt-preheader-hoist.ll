; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -amdgpu-enable-new-insert-waitcnts=false < %s | FileCheck %s --check-prefixes=NOHOIST
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck %s --check-prefixes=NOHOIST
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -amdgpu-new-insert-waitcnts-kmcnt-preheader-flush < %s | FileCheck %s --check-prefixes=HOIST

; A scalar (SMEM) load in the preheader produces a loop-invariant value that is
; read every iteration. The kmcnt wait for it can be hoisted into the preheader:
; the preheader dominates the loop, the loop contains no SMEM load to re-arm
; kmcnt, and nothing on the backedge reads the value without going through the
; preheader wait first. So a single preheader s_wait_kmcnt suffices and no wait
; is needed inside the loop.
;
; The old pass never flushes KmCnt into a preheader (it only flushes VmCnt/DsCnt),
; so the new pass matches it by default - the kmcnt wait stays inside the loop.
; The hoist is available under -amdgpu-new-insert-waitcnts-kmcnt-preheader-flush.

; Default (old pass, and new pass without the flag): wait stays inside the loop.
; NOHOIST-LABEL: kmcnt_hoist:
; NOHOIST:        s_load_b32 s0, s[0:1], 0x0
; NOHOIST:      .LBB{{[0-9]+}}_1: ; %loop
; NOHOIST:        s_wait_kmcnt 0x0
; NOHOIST-NEXT:   s_add_co_i32 s4, s0, s1

; With the flag: the wait is hoisted before the loop, none inside it.
; HOIST-LABEL: kmcnt_hoist:
; HOIST:        s_load_b32 s0, s[0:1], 0x0
; HOIST:        s_wait_kmcnt 0x0
; HOIST:      .LBB{{[0-9]+}}_1: ; %loop
; HOIST-NOT:    s_wait_kmcnt
; HOIST:        s_add_co_i32 s4, s0, s1
; HOIST:        s_cbranch_scc1 .LBB{{[0-9]+}}_1

define amdgpu_kernel void @kmcnt_hoist(ptr addrspace(4) %p, ptr addrspace(1) %out, i32 %n) {
entry:
  %v = load i32, ptr addrspace(4) %p, align 4
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %gep = getelementptr i32, ptr addrspace(1) %out, i32 %i
  %iv = add i32 %v, %i
  store i32 %iv, ptr addrspace(1) %gep
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
