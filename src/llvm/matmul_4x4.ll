define void @ll_matmul_4x4(float*  %a, float*  %b, float*  %result) {
entry:
  ; load matrix
  %a_col = call <16 x float> @llvm.matrix.column.major.load.v16f32.p0f32(float* %a, i64 4, i1 false, i32 4, i32 4)
  %b_col = call <16 x float> @llvm.matrix.column.major.load.v16f32.p0f32(float* %b, i64 4, i1 false, i32 4, i32 4)

  ; load matrix
  %res_transposed = call <16 x float> @llvm.matrix.multiply.v16f32.v16f32.v16f32(<16 x float> %b_col, <16 x float> %a_col, i32 4, i32 4, i32 4)

  ; save resukt
  call void @llvm.matrix.column.major.store.v16f32.p0f32(<16 x float> %res_transposed, float* %result, i64 4, i1 false, i32 4, i32 4)

  ret void
}


; https://llvm.org/doxygen/classllvm_1_1ShuffleVectorInst.html
; https://www.llvm.org/docs/LangRef.html#shufflevector-instruction
; odd wording choice in llvm, but i guess it works so ... poison it is
; https://www.llvm.org/docs/LangRef.html#poison-values

define void @ll_matmul_4x4_unrolled(float* %a, float* %b, float* %result) {
entry:

; load b matrix and hold it in registers
  ; b0
  %ptr_b0 = getelementptr float, float* %b, i64 0
  %b0 = load <4 x float>, float* %ptr_b0, align 4
  
  ; b1
  %ptr_b1 = getelementptr float, float* %b, i64 4
  %b1 = load <4 x float>, float* %ptr_b1, align 4
  
  ; b2
  %ptr_b2 = getelementptr float, float* %b, i64 8
  %b2 = load <4 x float>, float* %ptr_b2, align 4
  
  ; b3
  %ptr_b3 = getelementptr float, float* %b, i64 12
  %b3 = load <4 x float>, float* %ptr_b3, align 4

; do row 0
  ; res[0] = (A[0][0] * B0) + (A[0][1] * B1) + (A[0][2] * B2) + (A[0][3] * B3)
  %ptr_a0 = getelementptr float, float* %a, i64 0
  %a0_vec = load <4 x float>, float* %ptr_a0, align 4

  ; step1 : broadcast new value  A[0][0] * B0
  %a0_0 = shufflevector <4 x float> %a0_vec, <4 x float> poison, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %row0_step1 = fmul <4 x float> %a0_0, %b0

  ; step2 : broadcast new value A[0][1] * B1 + Accumulate
  %a0_1 = shufflevector <4 x float> %a0_vec, <4 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %row0_step2 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a0_1, <4 x float> %b1, <4 x float> %row0_step1)

  ; step3 : broadcast new value A[0][2] * B2 + Accumulate
  %a0_2 = shufflevector <4 x float> %a0_vec, <4 x float> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %row0_step3 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a0_2, <4 x float> %b2, <4 x float> %row0_step2)

  ; step4 : broadcast new value A[0][3] * B3 + Accumulate
  %a0_3 = shufflevector <4 x float> %a0_vec, <4 x float> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %res_row0 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a0_3, <4 x float> %b3, <4 x float> %row0_step3)

  ; store result row 0
  %ptr_res0 = getelementptr float, float* %result, i64 0
  store <4 x float> %res_row0, float* %ptr_res0, align 4

; do row 1
  ; res[1] = (A[1][0] * B0) + (A[1][1] * B1) + (A[1][2] * B2) + (A[1][3] * B3)
  %ptr_a1 = getelementptr float, float* %a, i64 4
  %a1_vec = load <4 x float>, float* %ptr_a1, align 4

  ; res[1] = (A[1][0] * B0) + (A[1][1] * B1) + (A[1][2] * B2) + (A[1][3] * B3)
  %a1_0 = shufflevector <4 x float> %a1_vec, <4 x float> poison, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %row1_step1 = fmul <4 x float> %a1_0, %b0

  ; step2 : broadcast new value A[1][1] * B1 + Accumulate
  %a1_1 = shufflevector <4 x float> %a1_vec, <4 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %row1_step2 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a1_1, <4 x float> %b1, <4 x float> %row1_step1)

  ; step3 : broadcast new value A[1][2] * B2 + Accumulate
  %a1_2 = shufflevector <4 x float> %a1_vec, <4 x float> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %row1_step3 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a1_2, <4 x float> %b2, <4 x float> %row1_step2)

  ; step4 : broadcast new value A[1][3] * B3 + Accumulate
  %a1_3 = shufflevector <4 x float> %a1_vec, <4 x float> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %res_row1 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a1_3, <4 x float> %b3, <4 x float> %row1_step3)

  ; store result row 1
  %ptr_res1 = getelementptr float, float* %result, i64 4
  store <4 x float> %res_row1, float* %ptr_res1, align 4

; do row 2
  ; res[2] = (A[2][0] * B0) + (A[2][1] * B1) + (A[2][2] * B2) + (A[2][3] * B3)
  %ptr_a2 = getelementptr float, float* %a, i64 8
  %a2_vec = load <4 x float>, float* %ptr_a2, align 4

  ; step1 : broadcast new value A[2][0] * B0
  %a2_0 = shufflevector <4 x float> %a2_vec, <4 x float> poison, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %row2_step1 = fmul <4 x float> %a2_0, %b0

  ; step2 : broadcast new value A[2][1] * B1 + Accumulate
  %a2_1 = shufflevector <4 x float> %a2_vec, <4 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %row2_step2 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a2_1, <4 x float> %b1, <4 x float> %row2_step1)

  ; step3 : broadcast new value A[2][2] * B2 + Accumulate
  %a2_2 = shufflevector <4 x float> %a2_vec, <4 x float> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %row2_step3 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a2_2, <4 x float> %b2, <4 x float> %row2_step2)

  ; step4 : broadcast new value A[2][3] * B3 + Accumulate
  %a2_3 = shufflevector <4 x float> %a2_vec, <4 x float> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %res_row2 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a2_3, <4 x float> %b3, <4 x float> %row2_step3)

  ; store result row 2
  %ptr_res2 = getelementptr float, float* %result, i64 8
  store <4 x float> %res_row2, float* %ptr_res2, align 4

; do row 3
  ; res[3] = (A[3][0] * B0) + (A[3][1] * B1) + (A[3][2] * B2) + (A[3][3] * B3)
  %ptr_a3 = getelementptr float, float* %a, i64 12
  %a3_vec = load <4 x float>, float* %ptr_a3, align 4

  ; step1 : broadcast new value A[3][0] * B0
  %a3_0 = shufflevector <4 x float> %a3_vec, <4 x float> poison, <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %row3_step1 = fmul <4 x float> %a3_0, %b0

  ; step2 : broadcast new value A[3][1] * B1 + Accumulate
  %a3_1 = shufflevector <4 x float> %a3_vec, <4 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %row3_step2 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a3_1, <4 x float> %b1, <4 x float> %row3_step1)

  ; step3 : broadcast new value A[3][2] * B2 + Accumulate
  %a3_2 = shufflevector <4 x float> %a3_vec, <4 x float> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %row3_step3 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a3_2, <4 x float> %b2, <4 x float> %row3_step2)

  ; step4 : broadcast new value A[3][3] * B3 + Accumulate
  %a3_3 = shufflevector <4 x float> %a3_vec, <4 x float> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %res_row3 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a3_3, <4 x float> %b3, <4 x float> %row3_step3)

  ; store result row 3
  %ptr_res3 = getelementptr float, float* %result, i64 12
  store <4 x float> %res_row3, float* %ptr_res3, align 4

  ret void
}
