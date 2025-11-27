use criterion::{Criterion, criterion_group, criterion_main};
use faer::prelude::*;
use llvm_intrinsic_with_rust::col_major_to_row_major;
use llvm_intrinsic_with_rust::compile_matmul_jit_with_template;
use llvm_intrinsic_with_rust::ll_matmul_4x4;
use llvm_intrinsic_with_rust::ll_matmul_4x4_unrolled;
use llvm_intrinsic_with_rust::ll_matmul_jit_with_template;
use llvm_intrinsic_with_rust::row_major_to_col_major;
use matrixmultiply::sgemm;
use ndarray::Array2;
use std::hint::black_box;

#[cfg(feature = "gpu")]
use llvm_intrinsic_with_rust::ll_matmul_gpu_compiled;
#[cfg(feature = "gpu")]
use llvm_intrinsic_with_rust::ll_matmul_gpu_jit;

fn bench_matmul_4x4(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("matmul_4x4");
    let mut result = black_box(vec![0.0f32; 16]);
    group.bench_function("ll_matmul_4x4_unrolled", |bencher| {
        bencher.iter(|| {
            unsafe {
                black_box(ll_matmul_4x4_unrolled(
                    black_box(a.as_ptr()),
                    black_box(b.as_ptr()),
                    black_box(result.as_mut_ptr()),
                ))
            };
        })
    });

    group.bench_function("ll_matmul_4x4", |bencher| {
        bencher.iter(|| {
            unsafe {
                black_box(ll_matmul_4x4(
                    black_box(a.as_ptr()),
                    black_box(b.as_ptr()),
                    black_box(result.as_mut_ptr()),
                ))
            };
        })
    });

    group.bench_function("ndarray_dot", |bencher| {
        bencher.iter(|| {
            let _ = black_box(black_box(&a_ndarray).dot(black_box(&b_ndarray)));
        })
    });

    group.bench_function("matrixmultiply_sgemm", |bencher| {
        bencher.iter(|| {
            unsafe {
                black_box(sgemm(
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
                    black_box(result.as_mut_ptr()),
                    4,
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

    group.bench_function("ll_matmul_cpu_jit_with_template", |bencher| {
        let ll_matmul_jit_with_template_entry =
            match unsafe { compile_matmul_jit_with_template(4, 4, 4, None) } {
                Ok(func) => func,
                Err(e) => panic!("Failed to compile JIT function: {}", e),
            };
        let a_col_major = row_major_to_col_major(&a, 4, 4);
        let b_col_major = row_major_to_col_major(&b, 4, 4);
        let mut result = vec![0.0; 4 * 4];
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
                    black_box(&a),
                    (4, 4),
                    black_box(&b),
                    (4, 4),
                ))
            };
        })
    });

    #[cfg(feature = "gpu")]
    group.bench_function("ll_matmul_gpu_compiled", |bencher| {
        bencher.iter(|| {
            let _ = unsafe {
                black_box(ll_matmul_gpu_compiled(
                    black_box(&a),
                    (4, 4),
                    black_box(&b),
                    (4, 4),
                ))
            };
        })
    });

    group.finish();
}

criterion_group!(benches, bench_matmul_4x4);
criterion_main!(benches);
