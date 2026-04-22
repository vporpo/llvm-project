; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -amdgpu-enable-new-insert-waitcnts=false < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck %s

; DROP CASE: a soft s_wait_xcnt fence sitting before a store can be implied by a
; KmCnt wait that the pass inserts during dataflow for the store's SMEM address
; operand. The pre-dataflow xcnt simplification cannot see that inserted wait, so
; the redundant xcnt must also be removed in a post-dataflow pass. The store's
; KmCnt wait (s_wait_kmcnt 0) drains the address translation, so no separate
; s_wait_xcnt is needed.
; CHECK-LABEL: smem_store_after_branch:
; CHECK:        v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s0
; CHECK-NEXT:   s_wait_kmcnt 0x0
; CHECK-NOT:    s_wait_xcnt
; CHECK-NEXT:   global_store_b32 v0, v1, s[2:3] scope:SCOPE_SYS
define amdgpu_kernel void @smem_store_after_branch(ptr addrspace(1) %arg, i32 %cnd) {
entry:
  %c = icmp eq i32 %cnd, 0
  br i1 %c, label %bb3, label %bb2
bb2:
  call void asm sideeffect "v_nop_e64", ""()
  br label %bb3
bb3:
  store volatile i32 %cnd, ptr addrspace(1) %arg
  ret void
}

; KEEP CASE: the s_wait_xcnt 0 here is a WAR dependency the pass inserts because
; v_cvt_pk_bf16_f32 overwrites v0, which holds the load's address. It happens to
; land next to s_wait_loadcnt 0, but it is not a soft fence and must NOT be
; removed by the post-dataflow simplification.
; CHECK-LABEL: load_war_xcnt:
; CHECK:        global_load_b32 v0, v[0:1], off
; CHECK-NEXT:   s_wait_loadcnt 0x0
; CHECK-NEXT:   s_wait_xcnt 0x0
; CHECK-NEXT:   v_cvt_pk_bf16_f32 v0, v0, s0
; CHECK-NEXT:   global_store_b16 v[2:3], v0, off
define void @load_war_xcnt(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %val = load float, ptr addrspace(1) %in
  %val.bf16 = fptrunc float %val to bfloat
  store bfloat %val.bf16, ptr addrspace(1) %out
  ret void
}
