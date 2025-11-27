define void @ll_matmul_jit(float* %a, float* %b, float* %result) {
entry:
  ; load matrix
  %a_mat = call <262144 x float> @llvm.matrix.column.major.load.v262144f32.p0f32(float* %a, i64 512, i1 false, i32 512, i32 512)
  
  ; load matrix
  %b_mat = call <262144 x float> @llvm.matrix.column.major.load.v262144f32.p0f32(float* %b, i64 512, i1 false, i32 512, i32 512)
  
  ; mult matrixs
  %c_mat = call <262144 x float> @llvm.matrix.multiply.v262144f32.v262144f32.v262144f32(<262144 x float> %a_mat, <262144 x float> %b_mat, i32 512, i32 512, i32 512)
  
  ; save resukt
  ; not sure about the i64 262144
  call void @llvm.matrix.column.major.store.v262144f32.p0f32(<262144 x float> %c_mat, float* %result, i64 512, i1 false, i32 512, i32 512)

  ret void
}
