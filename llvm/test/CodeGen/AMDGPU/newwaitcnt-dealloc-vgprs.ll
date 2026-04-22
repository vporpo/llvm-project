; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=+real-true16 -amdgpu-enable-new-insert-waitcnts=false < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=+real-true16 < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=+real-true16 -amdgpu-new-insert-waitcnts-dealloc-vgprs=false < %s | FileCheck -check-prefix=NODEALLOC %s

; On GFX11+, when a kernel reaches S_ENDPGM with outstanding non-scratch VMEM
; stores (and no pending scratch store), the waitcnt pass sends a DEALLOC_VGPRS
; message to release the VGPRs early instead of waiting for the stores to finish.
; A preceding s_nop is required on subtargets that need it (gfx11, not gfx1250).
; The new pass must emit this message, matching the old pass. The optimization can
; be disabled with -amdgpu-new-insert-waitcnts-dealloc-vgprs=false.

declare void @ext()

; CHECK-LABEL: kernel_call_then_two_volatile_stores:
; CHECK:        s_swappc_b64
; CHECK:        global_store_b8 v[0:1], v0, off dlc
; CHECK:        global_store_b32 v[0:1], v1, off dlc
; CHECK:        s_waitcnt_vscnt null, 0x0
; CHECK-NEXT:   s_nop 0
; CHECK-NEXT:   s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
; CHECK-NEXT:   s_endpgm

; With the optimization disabled, no DEALLOC_VGPRS message is emitted.
; NODEALLOC-LABEL: kernel_call_then_two_volatile_stores:
; NODEALLOC:        global_store_b32 v[0:1], v1, off dlc
; NODEALLOC-NEXT:   s_waitcnt_vscnt null, 0x0
; NODEALLOC-NOT:    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
; NODEALLOC:        s_endpgm
define amdgpu_kernel void @kernel_call_then_two_volatile_stores(i8 %a, i32 %b) {
  call void @ext()
  store volatile i8 %a, ptr addrspace(1) poison
  store volatile i32 %b, ptr addrspace(1) poison
  ret void
}
