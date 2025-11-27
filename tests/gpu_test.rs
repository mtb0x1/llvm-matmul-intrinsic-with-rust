use llvm_intrinsic_with_rust::common::native_matmul;
#[cfg(feature = "gpu")]
use llvm_intrinsic_with_rust::llvm::gpu::ll_matmul_gpu_compiled;
#[cfg(feature = "gpu")]
use llvm_intrinsic_with_rust::llvm::gpu::ll_matmul_gpu_jit;

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_compiled_matmul() {
    // 4x4 * 4x4
    let a = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ];
    let b = [
        16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
    ];
    let a_shape = (4, 4);
    let b_shape = (4, 4);

    let expected = native_matmul(&a, a_shape, &b, b_shape);

    // We need to ensure CUDA is available too, but ll_matmul_nvptx handles init.
    // If CUDA init fails, it will panic, which is fine for a test (it fails).

    eprintln!("[test_gpu_compiled_matmul] Launching kernel with:");
    eprintln!("a ptr: {:p}", &a as *const _);
    eprintln!("b ptr: {:p}", &b as *const _);
    eprintln!("a_shape: {:?}, b_shape: {:?}", a_shape, b_shape);
    let result = unsafe { ll_matmul_gpu_compiled(&a, a_shape, &b, b_shape) };

    assert_eq!(result.len(), expected.len());
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-4,
            "Mismatch at index {}: {} != {}",
            i,
            r,
            e
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_jit_matmul() {
    // 4x4 * 4x4
    let a = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ];
    let b = [
        16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
    ];
    let a_shape = (4, 4);
    let b_shape = (4, 4);

    let expected = native_matmul(&a, a_shape, &b, b_shape);

    // We need to ensure CUDA is available too, but ll_matmul_nvptx handles init.
    // If CUDA init fails, it will panic, which is fine for a test (it fails).

    eprintln!("[test_gpu_jit_matmul] Launching kernel with:");
    eprintln!("a ptr: {:p}", &a as *const _);
    eprintln!("b ptr: {:p}", &b as *const _);
    eprintln!("a_shape: {:?}, b_shape: {:?}", a_shape, b_shape);
    let result = unsafe { ll_matmul_gpu_jit(&a, a_shape, &b, b_shape) };

    assert_eq!(result.len(), expected.len());
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-4,
            "Mismatch at index {}: {} != {}",
            i,
            r,
            e
        );
    }
}
