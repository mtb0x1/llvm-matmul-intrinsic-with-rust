; NVVM intrinsics for PTX thread/block info
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y() nounwind readnone

; Kernel: C = A*B
define void @ll_matmul_gpu(float addrspace(1)* %A, float addrspace(1)* %B, float addrspace(1)* %C, i32 %M, i32 %N, i32 %K) {
entry:
  ; Read thread and block indices
  %tidx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %tidy = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %bidx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %bidy = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %bdimx = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %bdimy = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()

  ; compute row = blockIdx.y*blockDim.y + threadIdx.y
  %bidx_times_bdimy = mul i32 %bidy, %bdimy
  %row = add i32 %bidx_times_bdimy, %tidy

  ; compute col = blockIdx.x*blockDim.x + threadIdx.x
  %bidx_times_bdimx = mul i32 %bidx, %bdimx
  %col = add i32 %bidx_times_bdimx, %tidx

  ; Bounds check
  %row_ok = icmp ult i32 %row, %M
  %col_ok = icmp ult i32 %col, %N
  %in_bounds = and i1 %row_ok, %col_ok
  br i1 %in_bounds, label %compute, label %exit

compute:
  ; Compute row base for A (row * K)
  %row_mul_K = mul i32 %row, %K
  %row_base_A_i64 = sext i32 %row_mul_K to i64

  ; Compute row base for C (row * N)
  %row_mul_N = mul i32 %row, %N
  %row_base_C_i64 = sext i32 %row_mul_N to i64

  %col_i64 = sext i32 %col to i64
  %c_index = add i64 %row_base_C_i64, %col_i64
  %cptr = getelementptr float, float addrspace(1)* %C, i64 %c_index

  ; Initialize accumulator
  %acc_init = fadd float 0.0, 0.0
  br label %loop_cond

loop_cond:
  %k_phi = phi i32 [0, %compute], [%k_next, %loop_body]
  %acc_phi = phi float [%acc_init, %compute], [%acc_next, %loop_body]
  %k_ok = icmp slt i32 %k_phi, %K
  br i1 %k_ok, label %loop_body, label %loop_end

loop_body:
  ; Load A[row*K + k]
  %k_i64 = sext i32 %k_phi to i64
  %a_index = add i64 %row_base_A_i64, %k_i64
  %aptr = getelementptr float, float addrspace(1)* %A, i64 %a_index
  %aval = load float, float addrspace(1)* %aptr, align 16

  ; Load B[k*N + col]
  %kN = mul i32 %k_phi, %N
  %kN_i64 = sext i32 %kN to i64
  %b_index = add i64 %kN_i64, %col_i64
  %bptr = getelementptr float, float addrspace(1)* %B, i64 %b_index
  %bval = load float, float addrspace(1)* %bptr, align 16

  ; Multiply-add
  %mul = fmul float %aval, %bval
  %acc_next = fadd float %acc_phi, %mul

  ; Increment k
  %k_next = add i32 %k_phi, 1
  br label %loop_cond

loop_end:
  store float %acc_phi, float addrspace(1)* %cptr, align 16
  br label %exit

exit:
  ret void
}

; NVVM kernel annotation
!nvvm.annotations = !{!0}
!0 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32)* @ll_matmul_gpu, !"kernel", i32 1}
