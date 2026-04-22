; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -amdgpu-enable-new-insert-waitcnts=false < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck %s

; A software-pipelined loop issues an async LDS DMA in the preheader whose result
; is consumed inside the loop. The preheader-flush optimization considers every
; counter whose pending op is live-in to the loop, which includes the async marks
; tracked on the AsyncMarkPseudoCnt pseudo counter. That pseudo has no hardware
; wait instruction, so it must be skipped during the flush - otherwise the pass
; asks for its (nonexistent) wait opcode and crashes. Async completion is handled
; by WAIT_ASYNCMARK instead.

declare void @llvm.amdgcn.global.load.async.to.lds.b32(ptr addrspace(1), ptr addrspace(3), i32, i32)
declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16)

; CHECK-LABEL: repro:
; CHECK: global_load_async_to_lds_b32
; CHECK: s_wait_asynccnt 0x1
; CHECK: ds_load_b32
define amdgpu_kernel void @repro(ptr addrspace(1) %foo, ptr addrspace(3) %lds, i32 %n) {
prolog:
  call void @llvm.amdgcn.global.load.async.to.lds.b32(ptr addrspace(1) %foo, ptr addrspace(3) %lds, i32 4, i32 u0x20)
  call void @llvm.amdgcn.asyncmark()
  br label %loop

loop:
  %i = phi i32 [ 1, %prolog ], [ %i.next, %loop ]
  %lds_gep = getelementptr i32, ptr addrspace(3) %lds, i32 %i
  call void @llvm.amdgcn.global.load.async.to.lds.b32(ptr addrspace(1) %foo, ptr addrspace(3) %lds_gep, i32 4, i32 u0x20)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.wait.asyncmark(i16 1)
  %v = load i32, ptr addrspace(3) %lds
  store i32 %v, ptr addrspace(1) %foo
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
