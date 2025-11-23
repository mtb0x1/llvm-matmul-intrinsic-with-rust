use criterion::{Criterion, criterion_group, criterion_main};
use faer::prelude::*;
use llvm_matmul_intrinsic_with_rust::ll_matmul_jit;
use matrixmultiply::sgemm;
use ndarray::Array2;
use std::hint::black_box;

// pointers are valid and aligned ?
// a and b are 4x4 matrices
unsafe extern "C" {
    #[link_name = "ll_matmul_4x4_using_transpose"]
    unsafe fn ll_matmul_4x4_using_transpose(a: *const f32, b: *const f32, result: *mut f32);
    #[link_name = "ll_matmul_4x4_unrolled"]
    unsafe fn ll_matmul_4x4_unrolled(a: *const f32, b: *const f32, result: *mut f32);
}

fn bench_matmul_4x4(c: &mut Criterion) {
    // Create random 4x4 matrices
    let a: [f32; 16] = black_box([
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ]);
    let b: [f32; 16] = black_box([
        16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
    ]);

    let a_ndarray = Array2::from_shape_vec((4, 4), a.to_vec())
        .expect("A shape must match the number of elements");
    let b_ndarray = Array2::from_shape_vec((4, 4), b.to_vec())
        .expect("B shape must match the number of elements");

    let a_faer = Mat::from_fn(4, 4, |i, j| a[i * 4 + j]);
    let b_faer = Mat::from_fn(4, 4, |i, j| b[i * 4 + j]);

    c.bench_function("ll_matmul_4x4_unrolled", |bencher| {
        bencher.iter(|| {
            let mut result = black_box([0.0f32; 16]);
            unsafe {
                ll_matmul_4x4_unrolled(
                    black_box(a.as_ptr()),
                    black_box(b.as_ptr()),
                    black_box(result.as_mut_ptr()),
                )
            };
            black_box(result)
        })
    });

    c.bench_function("ll_matmul_4x4_using_transpose", |bencher| {
        bencher.iter(|| {
            let mut result = black_box([0.0f32; 16]);
            unsafe {
                ll_matmul_4x4_using_transpose(
                    black_box(a.as_ptr()),
                    black_box(b.as_ptr()),
                    black_box(result.as_mut_ptr()),
                )
            };
            black_box(result)
        })
    });

    c.bench_function("ndarray_dot", |bencher| {
        bencher.iter(|| {
            let c_ndarray = black_box(&a_ndarray).dot(black_box(&b_ndarray));
            black_box(c_ndarray)
        })
    });

    c.bench_function("matrixmultiply_sgemm", |bencher| {
        bencher.iter(|| {
            let mut c = black_box([0.0f32; 16]);
            unsafe {
                sgemm(
                    4,
                    4,
                    4,
                    1.0,
                    black_box(a.as_ptr()),
                    4,
                    1,
                    black_box(b.as_ptr()),
                    4,
                    1,
                    0.0,
                    c.as_mut_ptr(),
                    4,
                    1,
                )
            };
            black_box(c)
        })
    });

    c.bench_function("faer_matmul", |bencher| {
        bencher.iter(|| {
            let c_faer = black_box(&a_faer) * black_box(&b_faer);
            black_box(c_faer)
        })
    });

    c.bench_function("ll_matmul_jit_generic", |bencher| {
        bencher.iter(|| {
            let result = unsafe { ll_matmul_jit(black_box(&a), (4, 4), black_box(&b), (4, 4)) };
            black_box(result)
        })
    });
}

criterion_group!(benches, bench_matmul_4x4);
criterion_main!(benches);
