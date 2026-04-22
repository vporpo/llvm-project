; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -amdgpu-enable-new-insert-waitcnts=false < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck %s

; S_WAIT_ASYNCCNT is a standalone counter with its own instruction. When it sits
; adjacent to another wait (here a wait_asyncmark(0) lowers to s_wait_asynccnt 0
; right next to an s_wait_dscnt 0), the wait-combiner must not fold it into the
; packed S_WAITCNT immediate - that immediate carries no AsyncCnt field, so the
; async wait would be silently dropped. Both the s_wait_dscnt and the
; s_wait_asynccnt must survive.

declare void @llvm.amdgcn.global.load.async.to.lds.b32(ptr addrspace(1), ptr addrspace(3), i32, i32)
declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16)

; CHECK-LABEL: repro:
; CHECK: s_wait_asynccnt 0x1
; CHECK: ds_load_b32
; The dscnt and asynccnt waits are independent; order is not significant.
; CHECK-DAG: s_wait_dscnt 0x0
; CHECK-DAG: s_wait_asynccnt 0x0
define amdgpu_kernel void @repro(ptr addrspace(1) %foo, ptr addrspace(3) %lds, ptr addrspace(1) %out) {
entry:
  call void @llvm.amdgcn.global.load.async.to.lds.b32(ptr addrspace(1) %foo, ptr addrspace(3) %lds, i32 4, i32 u0x20)
  call void @llvm.amdgcn.asyncmark()
  %lds1 = getelementptr i32, ptr addrspace(3) %lds, i32 1
  %foo1 = getelementptr i32, ptr addrspace(1) %foo, i32 1
  call void @llvm.amdgcn.global.load.async.to.lds.b32(ptr addrspace(1) %foo1, ptr addrspace(3) %lds1, i32 4, i32 u0x20)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.wait.asyncmark(i16 1)
  %v0 = load i32, ptr addrspace(3) %lds
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  %sum = add i32 %v0, 1
  store i32 %sum, ptr addrspace(1) %out
  ret void
}
