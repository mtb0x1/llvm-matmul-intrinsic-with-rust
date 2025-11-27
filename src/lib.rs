pub mod common;
pub use common::DEFAULT_IR_TEMPLATE_JIT_CPU;
pub use common::TEMPLATE_JIT_CPU_ENV;
pub use common::TEMPLATE_JIT_CPU_ENV_FUNCTION_NAME;
pub mod llvm;

#[cfg(feature = "gpu")]
pub use llvm::gpu::cuda_driver::*;
#[cfg(feature = "gpu")]
pub use llvm::gpu::ll_matmul_gpu_compiled;
#[cfg(feature = "gpu")]
pub use llvm::gpu::ll_matmul_gpu_jit;

pub use llvm::col_major_to_row_major;
pub use llvm::compile_matmul_jit_with_template;
pub use llvm::ll_matmul_4x4;
pub use llvm::ll_matmul_4x4_unrolled;
pub use llvm::ll_matmul_jit_with_template;
pub use llvm::row_major_to_col_major;
