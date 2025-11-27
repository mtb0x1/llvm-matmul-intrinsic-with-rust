use llvm_intrinsic_with_rust::common::native_matmul;
use llvm_intrinsic_with_rust::ll_matmul_4x4;
use llvm_intrinsic_with_rust::ll_matmul_4x4_unrolled;
#[cfg(feature = "gpu")]
use llvm_intrinsic_with_rust::llvm::gpu::ll_matmul_gpu_compiled;
#[cfg(feature = "gpu")]
use llvm_intrinsic_with_rust::llvm::gpu::ll_matmul_gpu_jit;
use llvm_intrinsic_with_rust::llvm::ll_matmul_jit_with_template;

fn main() {
    // 2x3 * 3x4 = (2x4)
    let a = [1., 2., 3., 4., 5., 6.];
    let a_shape = (2, 3);
    let b = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.];
    let b_shape = (3, 4);
    let _result_shape = (2, 4);

    let result = unsafe { ll_matmul_jit_with_template(&a, a_shape, &b, b_shape, None) };
    println!("LLVM CPU JIT WITH TEMPLATE (2x3 * 3x2)  : {:?}", result);

    let result = native_matmul(&a, a_shape, &b, b_shape);
    println!("Native GENERIC (2x3 * 3x2)              : {:?}", result);

    // 4x4 * 4x4 = (4x4)
    let a: [f32; 16] = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ];
    let a_shape = (4, 4);
    let b: [f32; 16] = [
        16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
    ];
    let b_shape = (4, 4);
    let result = native_matmul(&a, a_shape, &b, b_shape);
    println!("Native 4x4                              : {:?}", result);

    let result = unsafe { ll_matmul_jit_with_template(&a, a_shape, &b, b_shape, None) };
    println!("LLVM CPU JIT WITH TEMPLATE (4x4 * 4x4)  : {:?}", result);

    let mut result = [0.0f32; 16];
    assert!(result.iter().all(|f| *f == 0.0f32));
    unsafe { ll_matmul_4x4_unrolled(a.as_ptr(), b.as_ptr(), result.as_mut_ptr()) };
    println!("LLVM unrolled 4x4                       : {:?}", result);

    let mut result = [0.0f32; 16];
    assert!(result.iter().all(|f| *f == 0.0f32));
    unsafe { ll_matmul_4x4(a.as_ptr(), b.as_ptr(), result.as_mut_ptr()) };
    println!("LLVM transpose 4x4                      : {:?}", result);

    #[cfg(feature = "gpu")]
    {
        let result = unsafe { ll_matmul_gpu_compiled(&a, a_shape, &b, b_shape) };
        println!("LLVM GPU compiled                       : {:?}", result);

        let result = unsafe { ll_matmul_gpu_jit(&a, a_shape, &b, b_shape) };
        println!("LLVM GPU JIT                            : {:?}", result);
    }
}
