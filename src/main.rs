mod llvm;
use llvm::ll_matmul_jit_with_template;

use llvm_matmul_intrinsic_with_rust::common::native_matmul;
use llvm_matmul_intrinsic_with_rust::ll_matmul_4x4;
use llvm_matmul_intrinsic_with_rust::ll_matmul_4x4_unrolled;

fn main() {
    // 2x3 * 3x4 = (2x4)
    let a = [1., 2., 3., 4., 5., 6.];
    let a_shape = (2, 3);
    let b = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.];
    let b_shape = (3, 4);
    let _result_shape = (2, 4);

    let result = unsafe { ll_matmul_jit_with_template(&a, a_shape, &b, b_shape, None) };
    println!("LLVM JIT WITH TEMPLATE (2x3 * 3x2)      : {:?}", result);

    let result = native_matmul(&a, a_shape, &b, b_shape);
    println!("Native GENERIC         (2x3 * 3x2)      : {:?}", result);

    // 4x4 * 4x4 = (4x4)
    let a: [f32; 16] = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ];
    let b: [f32; 16] = [
        16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
    ];
    let result = native_matmul(&a, (4, 4), &b, (4, 4));
    println!("Native 4x4                              : {:?}", result);

    let mut result = [0.0f32; 16];
    assert!(result.iter().all(|f| *f == 0.0f32));
    unsafe { ll_matmul_4x4_unrolled(a.as_ptr(), b.as_ptr(), result.as_mut_ptr()) };
    println!("LLVM unrolled 4x4                       : {:?}", result);

    let mut result = [0.0f32; 16];
    assert!(result.iter().all(|f| *f == 0.0f32));
    unsafe { ll_matmul_4x4(a.as_ptr(), b.as_ptr(), result.as_mut_ptr()) };
    println!("LLVM transpose 4x4                      : {:?}", result);
}
