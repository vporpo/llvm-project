; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck %s

; Async LDS DMA marks are capped at 16 (the old pass's MaxAsyncMarks). When two
; paths join and their combined mark count exceeds the cap, the marks are
; truncated at the merge, so a wait_asyncmark(N) with N at/beyond the cap must be
; clamped: the wait can be no larger than MaxAsyncMarks-1 (15). Here the else
; path contributes 17 marks (> 16), so wait_asyncmark(16) clamps to vmcnt(15),
; not vmcnt(16). (Pre-gfx12 async LDS DMA completes on vmcnt.)

declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1), ptr addrspace(3), i32, i32, i32)
declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16)

; CHECK-LABEL: asyncmark_merge_clamp:
; CHECK: ; wait_asyncmark(16)
; CHECK-NEXT: s_waitcnt vmcnt(15)
define void @asyncmark_merge_clamp(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 %n, ptr addrspace(1) %out) {
entry:
  %c = icmp slt i32 0, %n
  br i1 %c, label %then, label %else

then:
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  br label %endif

else:
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  br label %endif

endif:
  call void @llvm.amdgcn.wait.asyncmark(i16 16)
  %v = load i32, ptr addrspace(3) %lds
  store i32 %v, ptr addrspace(1) %out
  ret void
}
