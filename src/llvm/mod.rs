pub mod jit;
pub use jit::col_major_to_row_major;
pub use jit::compile_matmul_jit_with_template;
pub use jit::ll_matmul_jit_with_template;
pub use jit::row_major_to_col_major;

#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub use gpu::ll_matmul_gpu_compiled;
#[cfg(feature = "gpu")]
pub use gpu::ll_matmul_gpu_jit;

mod compiled;
#[allow(unused)]
pub use compiled::ll_matmul_4x4;
#[allow(unused)]
pub use compiled::ll_matmul_4x4_unrolled;
