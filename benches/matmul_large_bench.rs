use criterion::{Criterion, criterion_group, criterion_main};
use faer::prelude::*;
use llvm_matmul_intrinsic_with_rust::ll_matmul_jit;
use matrixmultiply::sgemm;
use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::hint::black_box;

fn generate_random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..rows * cols)
        .map(|_| rng.random_range(1f32..255f32))
        .collect()
}
const SEED: u64 = 42;
fn bench_matmul_mid(c: &mut Criterion) {
    let m = 512;
    let n = 512;
    let k = 512;

    let a_vec = generate_random_matrix(m, k, SEED);
    let b_vec = generate_random_matrix(k, n, SEED);

    let a_ndarray = Array2::from_shape_vec((m, k), a_vec.clone()).unwrap();
    let b_ndarray = Array2::from_shape_vec((k, n), b_vec.clone()).unwrap();

    let a_faer = Mat::from_fn(m, k, |i, j| a_vec[i * k + j]);
    let b_faer = Mat::from_fn(k, n, |i, j| b_vec[i * n + j]);

    let mut group = c.benchmark_group("matmul_mid_512x512");

    group.bench_function("ndarray_dot", |bencher| {
        bencher.iter(|| {
            let c_ndarray = black_box(&a_ndarray).dot(black_box(&b_ndarray));
            black_box(c_ndarray)
        })
    });

    group.bench_function("matrixmultiply_sgemm", |bencher| {
        bencher.iter(|| {
            let mut c = vec![0.0f32; m * n];
            unsafe {
                sgemm(
                    m,
                    k,
                    n,
                    1.0,
                    black_box(a_vec.as_ptr()),
                    k as isize,
                    1,
                    black_box(b_vec.as_ptr()),
                    n as isize,
                    1,
                    0.0,
                    c.as_mut_ptr(),
                    n as isize,
                    1,
                )
            };
            black_box(c)
        })
    });

    group.bench_function("faer_matmul", |bencher| {
        bencher.iter(|| {
            let c_faer = black_box(&a_faer) * black_box(&b_faer);
            black_box(c_faer)
        })
    });

    group.bench_function("ll_matmul_jit_generic", |bencher| {
        bencher.iter(|| {
            let result =
                unsafe { ll_matmul_jit(black_box(&a_vec), (m, k), black_box(&b_vec), (k, n)) };
            black_box(result)
        })
    });
    group.finish();
}

fn bench_matmul_big(c: &mut Criterion) {
    let m = 1024;
    let n = 1024;
    let k = 1024;

    let a_vec = generate_random_matrix(m, k, SEED);
    let b_vec = generate_random_matrix(k, n, SEED);

    let a_ndarray = Array2::from_shape_vec((m, k), a_vec.clone()).unwrap();
    let b_ndarray = Array2::from_shape_vec((k, n), b_vec.clone()).unwrap();

    let a_faer = Mat::from_fn(m, k, |i, j| a_vec[i * k + j]);
    let b_faer = Mat::from_fn(k, n, |i, j| b_vec[i * n + j]);

    let mut group = c.benchmark_group("matmul_big_1024x1024");
    group.sample_size(10); // Reduce sample size for large matrices

    group.bench_function("ndarray_dot", |bencher| {
        bencher.iter(|| {
            let c_ndarray = black_box(&a_ndarray).dot(black_box(&b_ndarray));
            black_box(c_ndarray)
        })
    });

    group.bench_function("matrixmultiply_sgemm", |bencher| {
        bencher.iter(|| {
            let mut c = vec![0.0f32; m * n];
            unsafe {
                sgemm(
                    m,
                    k,
                    n,
                    1.0,
                    black_box(a_vec.as_ptr()),
                    k as isize,
                    1,
                    black_box(b_vec.as_ptr()),
                    n as isize,
                    1,
                    0.0,
                    c.as_mut_ptr(),
                    n as isize,
                    1,
                )
            };
            black_box(c)
        })
    });

    group.bench_function("faer_matmul", |bencher| {
        bencher.iter(|| {
            let c_faer = black_box(&a_faer) * black_box(&b_faer);
            black_box(c_faer)
        })
    });

    group.bench_function("ll_matmul_jit_generic", |bencher| {
        bencher.iter(|| {
            let result =
                unsafe { ll_matmul_jit(black_box(&a_vec), (m, k), black_box(&b_vec), (k, n)) };
            black_box(result)
        })
    });
    group.finish();
}

criterion_group!(benches, bench_matmul_mid, bench_matmul_big);
criterion_main!(benches);
