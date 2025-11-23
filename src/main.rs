mod llvm;
use llvm::ll_matmul_builtin;
use llvm::ll_matmul_jit_with_template;

// pointers are valid and aligned ?
// a and b are 4x4 matrices
// check llvm/matmul_4x4.ll for the implementation.
unsafe extern "C" {
    #[link_name = "ll_matmul_4x4_using_transpose"]
    unsafe fn ll_matmul_4x4_using_transpose(a: *const f32, b: *const f32, result: *mut f32);
    #[link_name = "ll_matmul_4x4_unrolled"]
    unsafe fn ll_matmul_4x4_unrolled(a: *const f32, b: *const f32, result: *mut f32);
}

fn native_matmul(a: &[f32], a_dims: (usize, usize), b: &[f32], b_dims: (usize, usize)) -> Vec<f32> {
    let (m, k) = a_dims;
    let (k2, n) = b_dims;
    assert_eq!(k, k2, "Matrix dimensions must agree");

    let mut result = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    result
}

fn main() {
    // 2x3 * 3x4 = (2x4)
    let a = [1., 2., 3., 4., 5., 6.];
    let a_shape = (2, 3);
    let b = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.];
    let b_shape = (3, 4);

    let _result_shape = (2, 4);
    let result = unsafe { ll_matmul_jit_with_template(&a, a_shape, &b, b_shape, None) };
    println!("LLVM JIT WITH TEMPLATE (2x3 * 3x2)      : {:?}", result);

    // let result = unsafe { ll_matmul_builtin(&a, a_shape, &b, b_shape) };
    // println!("LLVM BUILTIN\t(2x3 * 3x2)      : {:?}", result);

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
    unsafe { ll_matmul_4x4_using_transpose(a.as_ptr(), b.as_ptr(), result.as_mut_ptr()) };
    println!("LLVM transpose 4x4                      : {:?}", result);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn assert_vec_eq(result: &[f32], expected: &[f32], epsilon: f32) {
        assert_eq!(
            result.len(),
            expected.len(),
            "result and expected lengths don't match"
        );
        let mut error = false;
        let mut buff = String::new();
        //dbg!(expected, result);
        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            error = error || (r - e).abs() > epsilon;
            buff.push_str(&format!("diff at index {}: got {}, expected {}\n", i, r, e));
        }
        if error {
            panic!("{}", buff);
        }
    }

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
}
