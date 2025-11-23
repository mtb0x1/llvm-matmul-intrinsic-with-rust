pub mod llvm;
pub use llvm::ll_matmul_builtin;
pub use llvm::ll_matmul_jit_with_template;
unsafe extern "C" {
    #[link_name = "ll_matmul_4x4_using_transpose"]
    pub unsafe fn ll_matmul_4x4_using_transpose(a: *const f32, b: *const f32, result: *mut f32);
    #[link_name = "ll_matmul_4x4_unrolled"]
    pub unsafe fn ll_matmul_4x4_unrolled(a: *const f32, b: *const f32, result: *mut f32);
}
