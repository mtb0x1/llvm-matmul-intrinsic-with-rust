use criterion::{Criterion, criterion_group, criterion_main};
use faer::prelude::*;
use llvm_matmul_intrinsic_with_rust::{
    common::generate_random_matrix, ll_matmul_jit_with_template,
};
use matrixmultiply::sgemm;
use ndarray::Array2;
use std::hint::black_box;
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

    group.bench_function("ndarray_dot", |bencher| {
        bencher.iter(|| {
            let c_ndarray = black_box(black_box(&a_ndarray).dot(black_box(&b_ndarray)));
            black_box(c_ndarray)
        })
    });

    group.bench_function("matrixmultiply_sgemm", |bencher| {
        let mut c = black_box(vec![0.0f32; m * n]);
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
                    c.as_mut_ptr(),
                    n as isize,
                    1,
                ))
            };
            black_box(c.clone())
        })
    });

    group.bench_function("faer_matmul", |bencher| {
        bencher.iter(|| {
            let c_faer = black_box(black_box(&a_faer) * black_box(&b_faer));
            black_box(c_faer)
        })
    });

    group.bench_function("ll_matmul_jit_with_template", |bencher| {
        bencher.iter(|| {
            let result = unsafe {
                black_box(ll_matmul_jit_with_template(
                    black_box(&a_vec),
                    (m, k),
                    black_box(&b_vec),
                    (k, n),
                    None,
                ))
            };
            black_box(result)
        })
    });

    // group.bench_function("ll_matmul_builtin", |bencher| {
    //     bencher.iter(|| {
    //         let result =
    //             unsafe { ll_matmul_builtin(black_box(&a_vec), (m, k), black_box(&b_vec), (k, n)) };
    //         black_box(result)
    //     })
    // });
    // group.finish();
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
            let c_ndarray = black_box(black_box(&a_ndarray).dot(black_box(&b_ndarray)));
            black_box(c_ndarray)
        })
    });

    group.bench_function("matrixmultiply_sgemm", |bencher| {
        let mut c = black_box(vec![0.0f32; m * n]);
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
                    c.as_mut_ptr(),
                    n as isize,
                    1,
                ))
            };
            black_box(c.clone())
        })
    });

    group.bench_function("faer_matmul", |bencher| {
        bencher.iter(|| {
            let c_faer = black_box(black_box(&a_faer) * black_box(&b_faer));
            black_box(c_faer)
        })
    });

    //  no jit
    //  TODO : create a not so naive template that
    //  can be used with jit without oom on opt phase

    // group.bench_function("ll_matmul_builtin", |bencher| {
    //     bencher.iter(|| {
    //         let result =
    //             unsafe { ll_matmul_builtin(black_box(&a_vec), (m, k), black_box(&b_vec), (k, n)) };
    //         black_box(result)
    //     })
    // });
    // group.finish();
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

    group.bench_function("ndarray_dot", |bencher| {
        bencher.iter(|| {
            let c_ndarray = black_box(black_box(&a_ndarray).dot(black_box(&b_ndarray)));
            black_box(c_ndarray)
        })
    });

    group.bench_function("matrixmultiply_sgemm", |bencher| {
        let mut c = black_box(vec![0.0f32; m * n]);
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
                    c.as_mut_ptr(),
                    n as isize,
                    1,
                ))
            };
            black_box(c.clone())
        })
    });

    group.bench_function("faer_matmul", |bencher| {
        bencher.iter(|| {
            let c_faer = black_box(black_box(&a_faer) * black_box(&b_faer));
            black_box(c_faer)
        })
    });

    //  no jit
    //  TODO : create a not so naive template that
    //  can be used with jit without oom on opt phase

    // group.bench_function("ll_matmul_builtin", |bencher| {
    //     bencher.iter(|| {
    //         let result =
    //             unsafe { ll_matmul_builtin(black_box(&a_vec), (m, k), black_box(&b_vec), (k, n)) };
    //         black_box(result)
    //     })
    // });
    // group.finish();
}

criterion_group!(
    benches,
    bench_matmul_small,
    bench_matmul_mid,
    bench_matmul_big
);
criterion_main!(benches);
