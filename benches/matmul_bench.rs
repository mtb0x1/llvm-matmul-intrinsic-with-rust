use criterion::{Criterion, criterion_group, criterion_main};
use faer::prelude::*;
use llvm_intrinsic_with_rust::{
    col_major_to_row_major, common::generate_random_matrix, compile_matmul_jit_with_template,
    row_major_to_col_major,
};
use matrixmultiply::sgemm;
use ndarray::Array2;
use std::hint::black_box;

#[cfg(feature = "gpu")]
use llvm_intrinsic_with_rust::ll_matmul_gpu_compiled;
#[cfg(feature = "gpu")]
use llvm_intrinsic_with_rust::ll_matmul_gpu_jit;

const SEED: u64 = 42;

fn bench_matmul_small(c: &mut Criterion) {
    let m = 32;
    let n = 32;
    let k = 32;

    let a_vec = generate_random_matrix(m, k, SEED);
    let b_vec = generate_random_matrix(k, n, SEED);

    let a_ndarray = Array2::from_shape_vec((m, k), a_vec.clone()).unwrap();
    let b_ndarray = Array2::from_shape_vec((k, n), b_vec.clone()).unwrap();

    let a_faer = Mat::from_fn(m, k, |i, j| a_vec[i * k + j]);
    let b_faer = Mat::from_fn(k, n, |i, j| b_vec[i * n + j]);

    let mut group = c.benchmark_group("matmul_small_32x32");
    let mut result = black_box(vec![0.0f32; m * n]);

    group.bench_function("ndarray_dot", |bencher| {
        bencher.iter(|| {
            let _ = black_box(black_box(&a_ndarray).dot(black_box(&b_ndarray)));
        })
    });

    group.bench_function("matrixmultiply_sgemm", |bencher| {
        bencher.iter(|| {
            unsafe {
                black_box(sgemm(
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
                    black_box(result.as_mut_ptr()),
                    n as isize,
                    1,
                ))
            };
        })
    });

    group.bench_function("faer_matmul", |bencher| {
        bencher.iter(|| {
            let _ = black_box(black_box(&a_faer) * black_box(&b_faer));
        })
    });

    group.bench_function("ll_matmul_jit_with_template", |bencher| {
        let ll_matmul_jit_with_template_entry =
            match unsafe { compile_matmul_jit_with_template(m, n, k, None) } {
                Ok(func) => func,
                Err(e) => panic!("Failed to compile JIT function: {}", e),
            };
        let a_col_major = row_major_to_col_major(&a_vec, m, k);
        let b_col_major = row_major_to_col_major(&b_vec, k, n);
        let mut result = vec![0.0; m * n];
        bencher.iter(|| {
            let _ = unsafe {
                black_box(ll_matmul_jit_with_template_entry.func.call(
                    black_box(a_col_major.as_ptr()),
                    black_box(b_col_major.as_ptr()),
                    black_box(result.as_mut_ptr()),
                ))
            };
        })
    });

    #[cfg(feature = "gpu")]
    group.bench_function("ll_matmul_gpu_jit", |bencher| {
        bencher.iter(|| {
            let _ = unsafe {
                black_box(ll_matmul_gpu_jit(
                    black_box(&a_vec),
                    (m, k),
                    black_box(&b_vec),
                    (k, n),
                ))
            };
        })
    });

    #[cfg(feature = "gpu")]
    group.bench_function("ll_matmul_gpu_compiled", |bencher| {
        bencher.iter(|| {
            let _ = unsafe {
                black_box(ll_matmul_gpu_compiled(
                    black_box(&a_vec),
                    (m, k),
                    black_box(&b_vec),
                    (k, n),
                ))
            };
        })
    });

    group.finish();
}

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
            let _ = black_box(black_box(&a_ndarray).dot(black_box(&b_ndarray)));
        })
    });
    let mut result = black_box(vec![0.0f32; m * n]);
    group.bench_function("matrixmultiply_sgemm", |bencher| {
        bencher.iter(|| {
            unsafe {
                black_box(sgemm(
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
                    black_box(result.as_mut_ptr()),
                    n as isize,
                    1,
                ))
            };
        })
    });

    group.bench_function("faer_matmul", |bencher| {
        bencher.iter(|| {
            let _ = black_box(black_box(&a_faer) * black_box(&b_faer));
        })
    });

    #[cfg(feature = "gpu")]
    group.bench_function("ll_matmul_gpu_jit", |bencher| {
        bencher.iter(|| {
            let _ = unsafe {
                black_box(ll_matmul_gpu_jit(
                    black_box(&a_vec),
                    (m, k),
                    black_box(&b_vec),
                    (k, n),
                ))
            };
        })
    });

    #[cfg(feature = "gpu")]
    group.bench_function("ll_matmul_gpu_compiled", |bencher| {
        bencher.iter(|| {
            let _ = unsafe {
                black_box(ll_matmul_gpu_compiled(
                    black_box(&a_vec),
                    (m, k),
                    black_box(&b_vec),
                    (k, n),
                ))
            };
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
    group.sample_size(10);
    let mut result = black_box(vec![0.0f32; m * n]);
    group.bench_function("ndarray_dot", |bencher| {
        bencher.iter(|| {
            let _ = black_box(black_box(&a_ndarray).dot(black_box(&b_ndarray)));
        })
    });

    group.bench_function("matrixmultiply_sgemm", |bencher| {
        bencher.iter(|| {
            unsafe {
                black_box(sgemm(
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
                    black_box(result.as_mut_ptr()),
                    n as isize,
                    1,
                ))
            };
        })
    });

    group.bench_function("faer_matmul", |bencher| {
        bencher.iter(|| {
            let _ = black_box(black_box(&a_faer) * black_box(&b_faer));
        })
    });

    #[cfg(feature = "gpu")]
    group.bench_function("ll_matmul_gpu_jit", |bencher| {
        bencher.iter(|| {
            let _ = unsafe {
                black_box(ll_matmul_gpu_jit(
                    black_box(&a_vec),
                    (m, k),
                    black_box(&b_vec),
                    (k, n),
                ))
            };
        })
    });

    #[cfg(feature = "gpu")]
    group.bench_function("ll_matmul_gpu_compiled", |bencher| {
        bencher.iter(|| {
            let _ = unsafe {
                black_box(ll_matmul_gpu_compiled(
                    black_box(&a_vec),
                    (m, k),
                    black_box(&b_vec),
                    (k, n),
                ))
            };
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_small,
    bench_matmul_mid,
    bench_matmul_big
);
criterion_main!(benches);
