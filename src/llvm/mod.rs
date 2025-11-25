mod jit;
pub use jit::ll_matmul_jit_with_template;

mod compiled;
#[allow(unused)]
pub use compiled::ll_matmul_4x4;
#[allow(unused)]
pub use compiled::ll_matmul_4x4_unrolled;
