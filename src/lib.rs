pub mod common;
pub use common::DEFAULT_IR_TEMPLATE;
pub use common::TEMPLATE_ENV;
pub mod llvm;
pub use llvm::ll_matmul_4x4;
pub use llvm::ll_matmul_4x4_unrolled;
pub use llvm::ll_matmul_jit_with_template;
