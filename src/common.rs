use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub const DEFAULT_IR_4X4_CPU: &str = include_str!("llvm/matmul_4x4.ll");
pub const TEMPLATE_JIT_CPU_ENV: &str = "LL_MATMUL_TEMPLATE";
pub const TEMPLATE_JIT_CPU_ENV_FUNCTION_NAME: &str = "LL_MATMUL_TEMPLATE_FUNCTION_NAME";
pub const DEFAULT_IR_TEMPLATE_JIT_CPU: &str = include_str!("llvm/matmul_intrinsic_naive.tmpl");
pub const DEFAULT_FUNCTION_NAME_JIT_CPU: &str = "ll_matmul_cpu_jit";
pub const DEFAULT_IR_TEMPLATE_GPU: &str = include_str!("llvm/gpu/matmul_for_gpu.ll");
pub const DEFAULT_FUNCTION_NAME_GPU: &str = "ll_matmul_gpu";
#[cfg(feature = "gpu")]
pub static GPU_FATBIN_PAYLOAD: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_for_gpu.fatbin"));
#[cfg(feature = "gpu")]
const _: () = assert!(GPU_FATBIN_PAYLOAD.len() > 0);
#[cfg(feature = "gpu")]
pub static GPU_PTX_PAYLOAD: &str = include_str!(concat!(env!("OUT_DIR"), "/matmul_for_gpu.ptx"));
#[cfg(feature = "gpu")]
const _: () = assert!(GPU_PTX_PAYLOAD.len() > 0);

pub fn generate_random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..rows * cols)
        .map(|_| rng.random_range(1f32..255f32))
        .collect()
}

// TODO : make less naive
pub fn native_matmul(
    a: &[f32],
    a_dims: (usize, usize),
    b: &[f32],
    b_dims: (usize, usize),
) -> Vec<f32> {
    let (m, k) = a_dims;
    let (k2, n) = b_dims;
    assert_eq!(k, k2, "Matrix dimensions must agree");

    let mut result = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    result
}

pub fn assert_vec_eq(result: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(
        result.len(),
        expected.len(),
        "result and expected lengths don't match"
    );
    let mut error = false;
    let mut buff = String::new();
    //dbg!(expected, result);
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        error = error || (r - e).abs() > epsilon;
        buff.push_str(&format!("diff at index {}: got {}, expected {}\n", i, r, e));
    }
    if error {
        panic!("{}", buff);
    }
}
