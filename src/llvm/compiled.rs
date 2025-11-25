// check build.rs for the build process
#[allow(unused)]
unsafe extern "C" {
    #[link_name = "ll_matmul_4x4"]
    pub unsafe fn ll_matmul_4x4(a: *const f32, b: *const f32, result: *mut f32);
    #[link_name = "ll_matmul_4x4_unrolled"]
    pub unsafe fn ll_matmul_4x4_unrolled(a: *const f32, b: *const f32, result: *mut f32);
}
