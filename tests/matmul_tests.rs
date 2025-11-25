use llvm_matmul_intrinsic_with_rust::common::assert_vec_eq;
use llvm_matmul_intrinsic_with_rust::llvm::ll_matmul_4x4;
use llvm_matmul_intrinsic_with_rust::llvm::ll_matmul_4x4_unrolled;
use llvm_matmul_intrinsic_with_rust::llvm::ll_matmul_jit_with_template;
use ndarray::Array2;

fn test_4x4_vs_ndarray(matmul_fn: unsafe extern "C" fn(*const f32, *const f32, *mut f32)) {
    let a: [f32; 16] = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ];
    let b: [f32; 16] = [
        16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
    ];
    let mut result = [0.0f32; 16];

    unsafe { matmul_fn(a.as_ptr(), b.as_ptr(), result.as_mut_ptr()) };

    let a_ndarray = Array2::from_shape_vec((4, 4), a.to_vec()).unwrap();
    let b_ndarray = Array2::from_shape_vec((4, 4), b.to_vec()).unwrap();
    let expected = a_ndarray.dot(&b_ndarray);
    let expected = expected.as_slice().unwrap();

    assert_vec_eq(&result, expected, 1e-4);
}

fn test_4x4_identity(matmul_fn: unsafe extern "C" fn(*const f32, *const f32, *mut f32)) {
    let a: [f32; 16] = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ];
    let identity: [f32; 16] = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    ];
    let mut result = [0.0f32; 16];

    unsafe { matmul_fn(a.as_ptr(), identity.as_ptr(), result.as_mut_ptr()) };

    assert_vec_eq(&result, &a, 1e-4);
}

fn test_4x4_zero(matmul_fn: unsafe extern "C" fn(*const f32, *const f32, *mut f32)) {
    let a: [f32; 16] = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ];
    let zero: [f32; 16] = [0.; 16];
    let mut result = [999.0f32; 16];

    unsafe { matmul_fn(a.as_ptr(), zero.as_ptr(), result.as_mut_ptr()) };

    assert_vec_eq(&result, &zero, 1e-4);
}

fn test_4x4_negatives(matmul_fn: unsafe extern "C" fn(*const f32, *const f32, *mut f32)) {
    let a: [f32; 16] = [
        1., -2., 3., -4., -5., 6., -7., 8., 9., -10., 11., -12., -13., 14., -15., 16.,
    ];
    let b: [f32; 16] = [
        -1., 2., -3., 4., 5., -6., 7., -8., -9., 10., -11., 12., 13., -14., 15., -16.,
    ];
    let mut result = [0.0f32; 16];

    unsafe { matmul_fn(a.as_ptr(), b.as_ptr(), result.as_mut_ptr()) };

    let a_ndarray = Array2::from_shape_vec((4, 4), a.to_vec()).unwrap();
    let b_ndarray = Array2::from_shape_vec((4, 4), b.to_vec()).unwrap();
    let expected = a_ndarray.dot(&b_ndarray);
    let expected = expected.as_slice().unwrap();

    assert_vec_eq(&result, expected, 1e-4);
}

fn test_4x4_precision(matmul_fn: unsafe extern "C" fn(*const f32, *const f32, *mut f32)) {
    let a: [f32; 16] = [
        1e-10, 2e-10, 3e-10, 4e-10, 5e-10, 6e-10, 7e-10, 8e-10, 9e-10, 1e-9, 1.1e-9, 1.2e-9,
        1.3e-9, 1.4e-9, 1.5e-9, 1.6e-9,
    ];
    let b: [f32; 16] = [
        1.6e-9, 1.5e-9, 1.4e-9, 1.3e-9, 1.2e-9, 1.1e-9, 1e-9, 9e-10, 8e-10, 7e-10, 6e-10, 5e-10,
        4e-10, 3e-10, 2e-10, 1e-10,
    ];
    let mut result = [0.0f32; 16];

    unsafe { matmul_fn(a.as_ptr(), b.as_ptr(), result.as_mut_ptr()) };

    let a_ndarray = Array2::from_shape_vec((4, 4), a.to_vec()).unwrap();
    let b_ndarray = Array2::from_shape_vec((4, 4), b.to_vec()).unwrap();
    let expected = a_ndarray.dot(&b_ndarray);
    let expected = expected.as_slice().unwrap();

    assert_vec_eq(&result, expected, 1e-18);
}

fn test_4x4_empty(matmul_fn: unsafe extern "C" fn(*const f32, *const f32, *mut f32)) {
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let mut c: [f32; 0] = [];
    unsafe { matmul_fn(a.as_ptr(), b.as_ptr(), c.as_mut_ptr()) };
}

// Tests for ll_matmul_4x4_unrolled
#[test]
fn test_ll_matmul_4x4_unrolled_vs_ndarray() {
    test_4x4_vs_ndarray(ll_matmul_4x4_unrolled);
}

#[test]
fn test_ll_matmul_4x4_unrolled_identity_matrix() {
    test_4x4_identity(ll_matmul_4x4_unrolled);
}

#[test]
fn test_ll_matmul_4x4_unrolled_zero_matrix() {
    test_4x4_zero(ll_matmul_4x4_unrolled);
}

#[test]
fn test_ll_matmul_4x4_unrolled_negative_values() {
    test_4x4_negatives(ll_matmul_4x4_unrolled);
}

#[test]
fn test_ll_matmul_4x4_unrolled_small_values_precision() {
    test_4x4_precision(ll_matmul_4x4_unrolled);
}

#[test]
fn test_ll_matmul_4x4_unrolled_empty_array() {
    test_4x4_empty(ll_matmul_4x4_unrolled);
}

// Tests for ll_matmul_4x4
#[test]
fn test_ll_matmul_4x4_vs_ndarray() {
    test_4x4_vs_ndarray(ll_matmul_4x4);
}

#[test]
fn test_ll_matmul_4x4_identity_matrix() {
    test_4x4_identity(ll_matmul_4x4);
}

#[test]
fn test_ll_matmul_4x4_zero_matrix() {
    test_4x4_zero(ll_matmul_4x4);
}

#[test]
fn test_ll_matmul_4x4_negative_values() {
    test_4x4_negatives(ll_matmul_4x4);
}

#[test]
fn test_ll_matmul_4x4_small_values_precision() {
    test_4x4_precision(ll_matmul_4x4);
}

#[test]
fn test_ll_matmul_4x4_empty_array() {
    test_4x4_empty(ll_matmul_4x4);
}

// cross val between both 4x4 implementations
#[test]
fn test_4x4_unrolled_vs_transpose_consistency() {
    let a: [f32; 16] = [
        2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22., 24., 26., 28., 30., 32.,
    ];
    let b: [f32; 16] = [
        1., 3., 5., 7., 9., 11., 13., 15., 17., 19., 21., 23., 25., 27., 29., 31.,
    ];

    let mut result_unrolled = [0.0f32; 16];
    let mut result_transpose = [0.0f32; 16];

    unsafe {
        ll_matmul_4x4_unrolled(a.as_ptr(), b.as_ptr(), result_unrolled.as_mut_ptr());
        ll_matmul_4x4(a.as_ptr(), b.as_ptr(), result_transpose.as_mut_ptr());
    };

    assert_vec_eq(&result_unrolled, &result_transpose, 1e-4);
}

fn test_generic_2x3_times_3x4(
    matmul_fn: unsafe fn(&[f32], (usize, usize), &[f32], (usize, usize), Option<&str>) -> Vec<f32>,
) {
    let a = [1., 2., 3., 4., 5., 6.];
    let b = [16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5.];
    let result = unsafe { matmul_fn(&a, (2, 3), &b, (3, 4), None) };

    let a_ndarray = Array2::from_shape_vec((2, 3), a.to_vec()).unwrap();
    let b_ndarray = Array2::from_shape_vec((3, 4), b.to_vec()).unwrap();
    let expected = a_ndarray.dot(&b_ndarray);
    let expected = expected.as_slice().unwrap();

    assert_vec_eq(&result, expected, 1e-4);
}

fn test_generic_1x1(
    matmul_fn: unsafe fn(&[f32], (usize, usize), &[f32], (usize, usize), Option<&str>) -> Vec<f32>,
) {
    let a = [5.0];
    let b = [3.0];
    let result = unsafe { matmul_fn(&a, (1, 1), &b, (1, 1), None) };
    let expected = [15.0];
    assert_vec_eq(&result, &expected, 1e-4);
}

fn test_generic_zero(
    matmul_fn: unsafe fn(&[f32], (usize, usize), &[f32], (usize, usize), Option<&str>) -> Vec<f32>,
) {
    let a = [1., 2., 3., 4., 5., 6.];
    let b = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.];
    let result = unsafe { matmul_fn(&a, (2, 3), &b, (3, 4), None) };
    let expected = [0., 0., 0., 0., 0., 0., 0., 0.];
    assert_vec_eq(&result, &expected, 1e-4);
}

fn test_generic_identity(
    matmul_fn: unsafe fn(&[f32], (usize, usize), &[f32], (usize, usize), Option<&str>) -> Vec<f32>,
) {
    let a: [f32; 16] = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ];
    let identity: [f32; 16] = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    ];
    let result = unsafe { matmul_fn(&a, (4, 4), &identity, (4, 4), None) };
    assert_vec_eq(&result, &a, 1e-4);
}

fn test_generic_negatives(
    matmul_fn: unsafe fn(&[f32], (usize, usize), &[f32], (usize, usize), Option<&str>) -> Vec<f32>,
) {
    let a = [1., -2., 3., -4., 5., -6.];
    let b = [-1., 2., -3., 4., -5., 6., -7., 8., -9., 10., -11., 12.];
    let result = unsafe { matmul_fn(&a, (2, 3), &b, (3, 4), None) };

    let a_ndarray = Array2::from_shape_vec((2, 3), a.to_vec()).unwrap();
    let b_ndarray = Array2::from_shape_vec((3, 4), b.to_vec()).unwrap();
    let expected = a_ndarray.dot(&b_ndarray);
    let expected = expected.as_slice().unwrap();

    assert_vec_eq(&result, expected, 1e-4);
}

fn test_generic_rectangular(
    matmul_fn: unsafe fn(&[f32], (usize, usize), &[f32], (usize, usize), Option<&str>) -> Vec<f32>,
) {
    let a = [1., 2., 3., 4., 5., 6.];
    let b = [7., 8., 9., 10., 11., 12., 13., 14., 15., 16.];
    let result = unsafe { matmul_fn(&a, (3, 2), &b, (2, 5), None) };

    let a_ndarray = Array2::from_shape_vec((3, 2), a.to_vec()).unwrap();
    let b_ndarray = Array2::from_shape_vec((2, 5), b.to_vec()).unwrap();
    let expected = a_ndarray.dot(&b_ndarray);
    let expected = expected.as_slice().unwrap();

    assert_vec_eq(&result, expected, 1e-4);
}

fn test_generic_precision(
    matmul_fn: unsafe fn(&[f32], (usize, usize), &[f32], (usize, usize), Option<&str>) -> Vec<f32>,
) {
    let a = [1e-10, 2e-10, 3e-10, 4e-10];
    let b = [5e-10, 6e-10, 7e-10, 8e-10];
    let result = unsafe { matmul_fn(&a, (2, 2), &b, (2, 2), None) };

    let a_ndarray = Array2::from_shape_vec((2, 2), a.to_vec()).unwrap();
    let b_ndarray = Array2::from_shape_vec((2, 2), b.to_vec()).unwrap();
    let expected = a_ndarray.dot(&b_ndarray);
    let expected = expected.as_slice().unwrap();

    assert_vec_eq(&result, expected, 1e-18);
}

fn test_generic_dot_product(
    matmul_fn: unsafe fn(&[f32], (usize, usize), &[f32], (usize, usize), Option<&str>) -> Vec<f32>,
) {
    let a = [1., 2., 3., 4.];
    let b = [5., 6., 7., 8.];
    let result = unsafe { matmul_fn(&a, (1, 4), &b, (4, 1), None) };
    let expected = [1. * 5. + 2. * 6. + 3. * 7. + 4. * 8.];
    assert_vec_eq(&result, &expected, 1e-4);
}

fn test_generic_outer_product(
    matmul_fn: unsafe fn(&[f32], (usize, usize), &[f32], (usize, usize), Option<&str>) -> Vec<f32>,
) {
    let a = [1., 2., 3., 4.];
    let b = [5., 6., 7., 8.];
    let result = unsafe { matmul_fn(&a, (4, 1), &b, (1, 4), None) };

    #[rustfmt::skip]
    let expected: [f32; 16] = [
        5., 6., 7., 8.,
        10., 12., 14., 16.,
        15., 18., 21., 24.,
        20., 24., 28., 32.,
    ];
    assert_vec_eq(&result, &expected, 1e-4);
}

fn test_generic_non_commutative(
    matmul_fn: unsafe fn(&[f32], (usize, usize), &[f32], (usize, usize), Option<&str>) -> Vec<f32>,
) {
    let a = [1., 2., 3., 4.];
    let b = [5., 6., 7., 8.];

    let ab = unsafe { matmul_fn(&a, (2, 2), &b, (2, 2), None) };
    let ba = unsafe { matmul_fn(&b, (2, 2), &a, (2, 2), None) };

    assert!(ab != ba, "Matrix multiplication should not be commutative");
}

// Tests for ll_matmul_jit_with_template
#[test]
fn test_ll_matmul_jit_with_template_vs_ndarray_2x3_times_3x4() {
    test_generic_2x3_times_3x4(ll_matmul_jit_with_template);
}

#[test]
fn test_ll_matmul_jit_with_template_1x1_matrix() {
    test_generic_1x1(ll_matmul_jit_with_template);
}

#[test]
fn test_ll_matmul_jit_with_template_zero_matrix() {
    test_generic_zero(ll_matmul_jit_with_template);
}

#[test]
fn test_ll_matmul_jit_with_template_identity_matrix_4x4() {
    test_generic_identity(ll_matmul_jit_with_template);
}

#[test]
fn test_ll_matmul_jit_with_template_negative_values() {
    test_generic_negatives(ll_matmul_jit_with_template);
}

#[test]
fn test_ll_matmul_jit_with_template_rectangular_3x2_times_2x5() {
    test_generic_rectangular(ll_matmul_jit_with_template);
}

#[test]
fn test_ll_matmul_jit_with_template_small_values_precision() {
    test_generic_precision(ll_matmul_jit_with_template);
}

#[test]
fn test_ll_matmul_jit_with_template_single_row_times_column() {
    test_generic_dot_product(ll_matmul_jit_with_template);
}

#[test]
fn test_ll_matmul_jit_with_template_column_times_row() {
    test_generic_outer_product(ll_matmul_jit_with_template);
}

#[test]
fn test_ll_matmul_jit_with_template_matrix_not_commutative() {
    test_generic_non_commutative(ll_matmul_jit_with_template);
}

#[test]
#[should_panic(expected = "shapes dosn't match")]
fn test_ll_matmul_jit_with_template_invalid_dimension_mismatch() {
    let a = [1., 2., 3., 4., 5., 6.];
    let b = [1., 2., 3., 4., 5., 6.];
    let _result = unsafe { ll_matmul_jit_with_template(&a, (2, 3), &b, (2, 3), None) };
}

#[test]
#[should_panic(expected = "empty arrays are not supported")]
fn test_ll_matmul_jit_with_template_empty_array() {
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let _result = unsafe { ll_matmul_jit_with_template(&a, (0, 0), &b, (0, 0), None) };
}
