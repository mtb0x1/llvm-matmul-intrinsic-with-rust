use llvm_matmul_intrinsic_with_rust::ll_matmul_jit_with_template;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

fn generate_random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..rows * cols)
        .map(|_| rng.random_range(1f32..255f32))
        .collect()
}

fn assert_vec_eq(result: &[f32], expected: &[f32], epsilon: f32) {
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

fn run_matmul(m: usize, n: usize, k: usize) {
    println!("Running {}x{} * {}x{} matmul...", m, k, k, n);
    let seed = 42;
    let a_vec = generate_random_matrix(m, k, seed);
    let b_vec = generate_random_matrix(k, n, seed);

    let start = Instant::now();
    let result = unsafe {
        ll_matmul_jit_with_template(
            &a_vec,
            (m, k),
            &b_vec,
            (k, n),
            None, /* N.B: using default ir template, size matters, see comments*/
        )
    };
    let duration = start.elapsed();
    assert!(result.len() == m * n);
    assert_vec_eq(
        &result,
        &native_matmul(&a_vec, (m, k), &b_vec, (k, n)),
        0.0001,
    );
    println!("Completed in {:?}", duration);
    println!(
        "Result sample (first 10 elements): {:?}",
        &result[..10.min(result.len())]
    );
}

fn native_matmul(a: &[f32], a_dims: (usize, usize), b: &[f32], b_dims: (usize, usize)) -> Vec<f32> {
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

// so, long story short
// when targeting non specilized hardware,
// opt will go crazy and try to lower intrinsics to the lowest level possible
// and this will make the code explode in size
// and lead to oom
// so we need to have a threshold
// up to you to check which value is best for you
const THRESHOLD: usize = 32;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let size: usize = if args.len() > 1 {
        let size = args[1].parse().expect("Invalid size argument");
        if size < THRESHOLD {
            eprintln!(
                "Danger zone: size might cause OOM (check comments above) {}",
                THRESHOLD
            );
        }
        size
    } else {
        println!("Using default size {}", THRESHOLD);
        THRESHOLD
    };

    println!("Starting debug_large_matmul example with size {}", size);

    run_matmul(size, size, size);

    println!("Finished all matrix multiplications");
}
