use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub const TEMPLATE_ENV: &str = "LL_MATMUL_TEMPLATE";
pub const DEFAULT_IR_TEMPLATE: &str = include_str!("llvm/matmul_intrinsic_naive.tmpl");

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
