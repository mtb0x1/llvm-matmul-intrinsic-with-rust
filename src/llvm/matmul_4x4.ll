define void @ll_matmul_4x4_using_transpose(float* %a, float* %b, float* %result) {
entry:
  ; load matrix
  %a_mat = call <16 x float> @llvm.matrix.column.major.load.v16f32.p0f32(float* %a, i64 4, i1 false, i32 4, i32 4)
  %a_t = call <16 x float> @llvm.matrix.transpose.v16f32.p0f32(<16 x float> %a_mat, i32 4, i32 4)

  ; load matrix
  %b_mat = call <16 x float> @llvm.matrix.column.major.load.v16f32.p0f32(float* %b, i64 4, i1 false, i32 4, i32 4)
  %b_t = call <16 x float> @llvm.matrix.transpose.v16f32.p0f32(<16 x float> %b_mat, i32 4, i32 4)

  ; mult matrixs
  %res_t = call <16 x float> @llvm.matrix.multiply.v16f32.v16f32.v16f32(<16 x float> %a_t, <16 x float> %b_t, i32 4, i32 4, i32 4)
  
  ; transpose result
  %res_mat = call <16 x float> @llvm.matrix.transpose.v16f32.p0f32(<16 x float> %res_t, i32 4, i32 4)

  ; save resukt
  call void @llvm.matrix.column.major.store.v16f32.p0f32(<16 x float> %res_mat, float* %result, i64 4, i1 false, i32 4, i32 4)

  ret void
}

define void @ll_matmul_4x4_unrolled(float* %a, float* %b, float* %result) {
entry:
  ; zero result matrix
  %result_0 = getelementptr float, float* %result, i32 0
  %result_1 = getelementptr float, float* %result, i32 1
  %result_2 = getelementptr float, float* %result, i32 2
  %result_3 = getelementptr float, float* %result, i32 3
  %result_4 = getelementptr float, float* %result, i32 4
  %result_5 = getelementptr float, float* %result, i32 5
  %result_6 = getelementptr float, float* %result, i32 6
  %result_7 = getelementptr float, float* %result, i32 7
  %result_8 = getelementptr float, float* %result, i32 8
  %result_9 = getelementptr float, float* %result, i32 9
  %result_10 = getelementptr float, float* %result, i32 10
  %result_11 = getelementptr float, float* %result, i32 11
  %result_12 = getelementptr float, float* %result, i32 12
  %result_13 = getelementptr float, float* %result, i32 13
  %result_14 = getelementptr float, float* %result, i32 14
  %result_15 = getelementptr float, float* %result, i32 15
  
  store float 0.0, float* %result_0
  store float 0.0, float* %result_1
  store float 0.0, float* %result_2
  store float 0.0, float* %result_3
  store float 0.0, float* %result_4
  store float 0.0, float* %result_5
  store float 0.0, float* %result_6
  store float 0.0, float* %result_7
  store float 0.0, float* %result_8
  store float 0.0, float* %result_9
  store float 0.0, float* %result_10
  store float 0.0, float* %result_11
  store float 0.0, float* %result_12
  store float 0.0, float* %result_13
  store float 0.0, float* %result_14
  store float 0.0, float* %result_15

  ; load a
  %a0 = load float, float* %a
  %a1 = getelementptr float, float* %a, i32 1
  %a2 = load float, float* %a1
  %a3 = getelementptr float, float* %a, i32 2
  %a4 = load float, float* %a3
  %a5 = getelementptr float, float* %a, i32 3
  %a6 = load float, float* %a5
  %a7 = getelementptr float, float* %a, i32 4
  %a8 = load float, float* %a7
  %a9 = getelementptr float, float* %a, i32 5
  %a10 = load float, float* %a9
  %a11 = getelementptr float, float* %a, i32 6
  %a12 = load float, float* %a11
  %a13 = getelementptr float, float* %a, i32 7
  %a14 = load float, float* %a13
  %a15 = getelementptr float, float* %a, i32 8
  %a16 = load float, float* %a15
  %a17 = getelementptr float, float* %a, i32 9
  %a18 = load float, float* %a17
  %a19 = getelementptr float, float* %a, i32 10
  %a20 = load float, float* %a19
  %a21 = getelementptr float, float* %a, i32 11
  %a22 = load float, float* %a21
  %a23 = getelementptr float, float* %a, i32 12
  %a24 = load float, float* %a23
  %a25 = getelementptr float, float* %a, i32 13
  %a26 = load float, float* %a25
  %a27 = getelementptr float, float* %a, i32 14
  %a28 = load float, float* %a27
  %a29 = getelementptr float, float* %a, i32 15
  %a30 = load float, float* %a29

  ; load b
  %b0 = load float, float* %b
  %b1 = getelementptr float, float* %b, i32 1
  %b2 = load float, float* %b1
  %b3 = getelementptr float, float* %b, i32 2
  %b4 = load float, float* %b3
  %b5 = getelementptr float, float* %b, i32 3
  %b6 = load float, float* %b5
  %b7 = getelementptr float, float* %b, i32 4
  %b8 = load float, float* %b7
  %b9 = getelementptr float, float* %b, i32 5
  %b10 = load float, float* %b9
  %b11 = getelementptr float, float* %b, i32 6
  %b12 = load float, float* %b11
  %b13 = getelementptr float, float* %b, i32 7
  %b14 = load float, float* %b13
  %b15 = getelementptr float, float* %b, i32 8
  %b16 = load float, float* %b15
  %b17 = getelementptr float, float* %b, i32 9
  %b18 = load float, float* %b17
  %b19 = getelementptr float, float* %b, i32 10
  %b20 = load float, float* %b19
  %b21 = getelementptr float, float* %b, i32 11
  %b22 = load float, float* %b21
  %b23 = getelementptr float, float* %b, i32 12
  %b24 = load float, float* %b23
  %b25 = getelementptr float, float* %b, i32 13
  %b26 = load float, float* %b25
  %b27 = getelementptr float, float* %b, i32 14
  %b28 = load float, float* %b27
  %b29 = getelementptr float, float* %b, i32 15
  %b30 = load float, float* %b29

  ; c = a * b
  ; row 0 of a * columns of b
  %c0 = fmul float %a0, %b0
  %c1 = fmul float %a2, %b8
  %c2 = fadd float %c0, %c1
  %c3 = fmul float %a4, %b16
  %c4 = fadd float %c2, %c3
  %c5 = fmul float %a6, %b24
  %c6 = fadd float %c4, %c5
  store float %c6, float* %result_0

  %c7 = fmul float %a0, %b2
  %c8 = fmul float %a2, %b10
  %c9 = fadd float %c7, %c8
  %c10 = fmul float %a4, %b18
  %c11 = fadd float %c9, %c10
  %c12 = fmul float %a6, %b26
  %c13 = fadd float %c11, %c12
  store float %c13, float* %result_1

  %c14 = fmul float %a0, %b4
  %c15 = fmul float %a2, %b12
  %c16 = fadd float %c14, %c15
  %c17 = fmul float %a4, %b20
  %c18 = fadd float %c16, %c17
  %c19 = fmul float %a6, %b28
  %c20 = fadd float %c18, %c19
  store float %c20, float* %result_2

  %c21 = fmul float %a0, %b6
  %c22 = fmul float %a2, %b14
  %c23 = fadd float %c21, %c22
  %c24 = fmul float %a4, %b22
  %c25 = fadd float %c23, %c24
  %c26 = fmul float %a6, %b30
  %c27 = fadd float %c25, %c26
  store float %c27, float* %result_3

  ; row 1 of a * columns of b
  %c28 = fmul float %a8, %b0
  %c29 = fmul float %a10, %b8
  %c30 = fadd float %c28, %c29
  %c31 = fmul float %a12, %b16
  %c32 = fadd float %c30, %c31
  %c33 = fmul float %a14, %b24
  %c34 = fadd float %c32, %c33
  store float %c34, float* %result_4

  %c35 = fmul float %a8, %b2
  %c36 = fmul float %a10, %b10
  %c37 = fadd float %c35, %c36
  %c38 = fmul float %a12, %b18
  %c39 = fadd float %c37, %c38
  %c40 = fmul float %a14, %b26
  %c41 = fadd float %c39, %c40
  store float %c41, float* %result_5

  %c42 = fmul float %a8, %b4
  %c43 = fmul float %a10, %b12
  %c44 = fadd float %c42, %c43
  %c45 = fmul float %a12, %b20
  %c46 = fadd float %c44, %c45
  %c47 = fmul float %a14, %b28
  %c48 = fadd float %c46, %c47
  store float %c48, float* %result_6

  %c49 = fmul float %a8, %b6
  %c50 = fmul float %a10, %b14
  %c51 = fadd float %c49, %c50
  %c52 = fmul float %a12, %b22
  %c53 = fadd float %c51, %c52
  %c54 = fmul float %a14, %b30
  %c55 = fadd float %c53, %c54
  store float %c55, float* %result_7

  ; row 2 of a * columns of b
  %c56 = fmul float %a16, %b0
  %c57 = fmul float %a18, %b8
  %c58 = fadd float %c56, %c57
  %c59 = fmul float %a20, %b16
  %c60 = fadd float %c58, %c59
  %c61 = fmul float %a22, %b24
  %c62 = fadd float %c60, %c61
  store float %c62, float* %result_8

  %c63 = fmul float %a16, %b2
  %c64 = fmul float %a18, %b10
  %c65 = fadd float %c63, %c64
  %c66 = fmul float %a20, %b18
  %c67 = fadd float %c65, %c66
  %c68 = fmul float %a22, %b26
  %c69 = fadd float %c67, %c68
  store float %c69, float* %result_9

  %c70 = fmul float %a16, %b4
  %c71 = fmul float %a18, %b12
  %c72 = fadd float %c70, %c71
  %c73 = fmul float %a20, %b20
  %c74 = fadd float %c72, %c73
  %c75 = fmul float %a22, %b28
  %c76 = fadd float %c74, %c75
  store float %c76, float* %result_10

  %c77 = fmul float %a16, %b6
  %c78 = fmul float %a18, %b14
  %c79 = fadd float %c77, %c78
  %c80 = fmul float %a20, %b22
  %c81 = fadd float %c79, %c80
  %c82 = fmul float %a22, %b30
  %c83 = fadd float %c81, %c82
  store float %c83, float* %result_11

  ; row 3 of a * columns of b
  %c84 = fmul float %a24, %b0
  %c85 = fmul float %a26, %b8
  %c86 = fadd float %c84, %c85
  %c87 = fmul float %a28, %b16
  %c88 = fadd float %c86, %c87
  %c89 = fmul float %a30, %b24
  %c90 = fadd float %c88, %c89
  store float %c90, float* %result_12

  %c91 = fmul float %a24, %b2
  %c92 = fmul float %a26, %b10
  %c93 = fadd float %c91, %c92
  %c94 = fmul float %a28, %b18
  %c95 = fadd float %c93, %c94
  %c96 = fmul float %a30, %b26
  %c97 = fadd float %c95, %c96
  store float %c97, float* %result_13

  %c98 = fmul float %a24, %b4
  %c99 = fmul float %a26, %b12
  %c100 = fadd float %c98, %c99
  %c101 = fmul float %a28, %b20
  %c102 = fadd float %c100, %c101
  %c103 = fmul float %a30, %b28
  %c104 = fadd float %c102, %c103
  store float %c104, float* %result_14

  %c105 = fmul float %a24, %b6
  %c106 = fmul float %a26, %b14
  %c107 = fadd float %c105, %c106
  %c108 = fmul float %a28, %b22
  %c109 = fadd float %c107, %c108
  %c110 = fmul float %a30, %b30
  %c111 = fadd float %c109, %c110
  store float %c111, float* %result_15

  ret void
}