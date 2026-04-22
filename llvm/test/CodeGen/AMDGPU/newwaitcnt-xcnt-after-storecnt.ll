; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -amdgpu-enable-new-insert-waitcnts=false < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck %s

; XCnt tracks address translation, which completes when the memory op does. So a
; soft s_wait_xcnt that sits next to an s_wait_storecnt 0 is implied by it - but
; only when the next memory op is not a store. Two stores' address translations
; are ordered on XCnt, so an xcnt before a store must be kept.

; store-then-load: the xcnt before the load is redundant (a load's address
; translation is not ordered after the store's), so it must be dropped.
; CHECK-LABEL: store_then_load:
; CHECK:        scratch_store_b16 off, v0, s32 scope:SCOPE_SYS
; CHECK-NEXT:   s_wait_storecnt 0x0
; CHECK-NOT:    s_wait_xcnt
; CHECK-NEXT:   scratch_load_u16 v0, off, s32 scope:SCOPE_SYS
define bfloat @store_then_load(bfloat %in) {
entry:
  %in.addr = alloca bfloat, align 2, addrspace(5)
  store volatile bfloat %in, ptr addrspace(5) %in.addr, align 2
  %loaded = load volatile bfloat, ptr addrspace(5) %in.addr, align 2
  ret bfloat %loaded
}

; store-then-store: the xcnt before the second store orders the two stores'
; address translations and must be kept.
; CHECK-LABEL: store_then_store:
; CHECK:        scratch_store_b16 v0, v2, off scope:SCOPE_SYS
; CHECK-NEXT:   s_wait_storecnt 0x0
; CHECK-NEXT:   s_wait_xcnt 0x0
; CHECK-NEXT:   scratch_store_b32 v0, v1, off scope:SCOPE_SYS
define void @store_then_store(ptr addrspace(5) %p, i32 %a, i16 %b) {
  store volatile i16 %b, ptr addrspace(5) %p
  store volatile i32 %a, ptr addrspace(5) %p
  ret void
}
