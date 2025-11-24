use llvm_matmul_intrinsic_with_rust::common::assert_vec_eq;
use llvm_matmul_intrinsic_with_rust::llvm::ll_matmul_4x4_unrolled;
use llvm_matmul_intrinsic_with_rust::llvm::ll_matmul_4x4_using_transpose;
use llvm_matmul_intrinsic_with_rust::llvm::ll_matmul_jit_with_template;
use ndarray::Array2;

#[test]
fn test_matmul_ndarray_vs_generic() {
    // 2x3 * 3x4 = 2x4
    let a = [1., 2., 3., 4., 5., 6.];
    let a_shape = (2, 3);
    let b = [16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5.];
    let b_shape = (3, 4);
    let c_generic = unsafe { ll_matmul_jit_with_template(&a, a_shape, &b, b_shape, None) };

    use ndarray::Array2;
    let a_ndarray = Array2::from_shape_vec((2, 3), a.to_vec())
        .expect("A shape must match the number of elements");
    let b_ndarray = Array2::from_shape_vec((3, 4), b.to_vec())
        .expect("B shape must match the number of elements");
    let c_ndarray = a_ndarray.dot(&b_ndarray);
    let c_ndarray = c_ndarray
        .as_slice()
        .expect("Failed to get slice from C ndarray");

    assert_vec_eq(&c_ndarray, &c_generic, 1e-4);
}

#[test]
fn test_matmul_ndarray_vs_llvm_unrolled() {
    // 4x4 * 4x4 = 4x4
    let a: [f32; 16] = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ];
    let b: [f32; 16] = [
        16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
    ];
    let mut c = [0.0f32; 16];

    //llvm result in C
    unsafe { ll_matmul_4x4_unrolled(a.as_ptr(), b.as_ptr(), c.as_mut_ptr()) };

    // matmul using ndarray apis
    let a_ndarray = Array2::from_shape_vec((4, 4), a.to_vec())
        .expect("A shape must match the number of elements");
    let b_ndarray = Array2::from_shape_vec((4, 4), b.to_vec())
        .expect("B shape must match the number of elements");
    let c_ndarray = a_ndarray.dot(&b_ndarray);
    let c_ndarray = c_ndarray
        .as_slice()
        .expect("failed to get slice from C ndarray");

    assert_vec_eq(&c, &c_ndarray, 1e-4);
}

#[test]
fn test_matmul_llvm_transpose_vs_llvm_unrolled() {
    // 4x4 * 4x4 = 4x4
    let a: [f32; 16] = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ];
    let b: [f32; 16] = [
        16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
    ];

    let mut c_transposed = [0.0f32; 16];
    unsafe { ll_matmul_4x4_using_transpose(a.as_ptr(), b.as_ptr(), c_transposed.as_mut_ptr()) };

    let mut c_unrolled = [0.0f32; 16];
    unsafe { ll_matmul_4x4_unrolled(a.as_ptr(), b.as_ptr(), c_unrolled.as_mut_ptr()) };

    assert_vec_eq(&c_transposed, &c_unrolled, 1e-4);
}

//FIXME:
// add test for built-in
// empty arrays ?
// overflow ?
// underflow ?
