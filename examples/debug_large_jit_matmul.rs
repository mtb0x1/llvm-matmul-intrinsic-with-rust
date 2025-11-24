use llvm_matmul_intrinsic_with_rust::{
    common::{assert_vec_eq, generate_random_matrix, native_matmul},
    ll_matmul_jit_with_template,
};
use std::time::Instant;

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
        if size > THRESHOLD {
            eprintln!(
                "\n ===> Danger zone: size might cause OOM (check comments above) {} <=== \n",
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
