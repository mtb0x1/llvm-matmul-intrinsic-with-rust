use core::panic;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CStr;

use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::llvm_sys;
use inkwell::llvm_sys::core::LLVMDisposeMessage;
use inkwell::llvm_sys::error::LLVMGetErrorMessage;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, RelocMode, Target, TargetMachine};
use std::collections::hash_map::Entry;

type LlMatmulJitSig = unsafe extern "C" fn(*const f32, *const f32, *mut f32);

thread_local! {
    static JIT_FUNCTIONS: RefCell<HashMap<(usize, usize, usize), JitFunction<'static,LlMatmulJitSig>>> = RefCell::new(HashMap::new());
}

// template to be udated at runtime
// it seems that m, n, k needs to be consts in llvm api
// TODO : check how backend compiler does it ? ( if this really needs to be const)

// Matrix multiplication (row major in rust, column major in llvm):
// I guess this needs a FIXME, but for now it doesn't worth the hassel
//  C(m×n) = A(m×k) * B(k×n)
pub unsafe fn ll_matmul_jit(
    a: &[f32],
    a_shape: (usize, usize),
    b: &[f32],
    b_shape: (usize, usize),
) -> Vec<f32> {
    let m = a_shape.0;
    let n = b_shape.1;
    let k = a_shape.1; // or b_shape.0
    assert!(a_shape.1 == b_shape.0, "shapes dosn't match");

    let shape_key = (m, n, k);

    // FIXME: can we avoid the cloning in here ?
    let ll_matmul_jit = JIT_FUNCTIONS.with(|cell| {
        let mut map = cell.borrow_mut();
        match map.entry(shape_key) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                // We compile and insert into the cache
                let compiled_func =
                    unsafe { compile_matmul_jit(m, n, k).expect("JIT Compilation failed") };
                entry.insert(compiled_func.clone());
                compiled_func
            }
        }
    });

    let a_col_major = row_major_to_col_major(a, a_shape.0, a_shape.1);
    let b_col_major = row_major_to_col_major(b, b_shape.0, b_shape.1);
    let mut result = vec![0.0; m * n];

    unsafe {
        ll_matmul_jit.call(
            a_col_major.as_ptr(),
            b_col_major.as_ptr(),
            result.as_mut_ptr(),
        );
    }

    col_major_to_row_major(&result, m, n)
}

unsafe fn compile_matmul_jit(
    m: usize,
    n: usize,
    k: usize,
) -> Option<JitFunction<'static, LlMatmulJitSig>> {
    //println!("compiling matmul_jit for m={}, n={}, k={}", m, n, k);

    let ir_template: &str = include_str!("matmul_ll_template.tmpl");

    let ir_runtime = ir_template
        .replace("{M}", &m.to_string())
        .replace("{N}", &n.to_string())
        .replace("{K}", &k.to_string())
        .replace("{VEC_A_SIZE}", &((m * k).to_string()))
        .replace("{VEC_B_SIZE}", &((k * n).to_string()))
        .replace("{VEC_C_SIZE}", &((m * n).to_string()))
        .replace("{A_STRIDE}", &m.to_string())
        .replace("{B_STRIDE}", &k.to_string())
        .replace("{C_STRIDE}", &m.to_string());

    // FIXME: memory to watch for I guess ?
    // probably a struct where a context (or list of those) lives along side with jit functions cache.
    let context = Box::leak(Box::new(Context::create()));

    let buffer = MemoryBuffer::create_from_memory_range_copy(ir_runtime.as_bytes(), "matmul_ir");
    let module: Module<'static> = context
        .create_module_from_ir(buffer)
        .expect("Failed to parse LLVM IR");

    // lowering so we can jit,
    // cause we use too high level matrix intrinsics
    // we need to reproduce the build.rs logic
    // opt to lower the ll code
    // llc the lowered ll code

    // the lowering pass code is a rip off of https://github.com/TheDan64/inkwell
    Target::initialize_all(&Default::default());
    let triple = TargetMachine::get_default_triple();
    let target = match Target::from_triple(&triple) {
        Ok(target) => target,
        Err(e) => panic!(
            "{} {}",
            format!("target from triplet failed : {}", triple),
            e
        ),
    };
    let machine = match target.create_target_machine(
        &triple,
        //TODO : Add cpu features as optionals
        "generic", //TargetMachine::get_host_cpu_name().to_string().as_str(),
        "",        //TargetMachine::get_host_cpu_features().to_string().as_str(),
        OptimizationLevel::Default,
        RelocMode::Default,
        CodeModel::Default,
    ) {
        Some(tm) => tm,
        None => panic!("couldn't create target machine"),
    };

    let pass_options = PassBuilderOptions::create();
    pass_options.set_verify_each(false);
    #[cfg(debug_assertions)]
    pass_options.set_debug_logging(true);
    pass_options.set_loop_interleaving(true);
    pass_options.set_loop_vectorization(true);
    pass_options.set_loop_slp_vectorization(true);
    pass_options.set_loop_unrolling(true);
    pass_options.set_forget_all_scev_in_loop_unroll(true);
    pass_options.set_licm_mssa_opt_cap(1);
    pass_options.set_licm_mssa_no_acc_for_promotion_cap(10);
    pass_options.set_call_graph_profile(true);
    pass_options.set_merge_functions(true);

    unsafe {
        let error = llvm_sys::transforms::pass_builder::LLVMRunPasses(
            module.as_mut_ptr(),
            c"lower-matrix-intrinsics".as_ptr(),
            machine.as_mut_ptr(),
            pass_options.as_mut_ptr(),
        );
        if !error.is_null() {
            let message = LLVMGetErrorMessage(error);
            if !message.is_null() {
                let message_str = CStr::from_ptr(message).to_string_lossy().into_owned();
                println!("The error message: {}", message_str);
                LLVMDisposeMessage(message);
            } else {
                panic!("shit happend can't recover any diag! good luck");
            }
        }
    };

    let execution_engine = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .expect("Failed to create JIT execution engine");

    let ll_matmul_jit: JitFunction<LlMatmulJitSig> = unsafe {
        execution_engine
            .get_function("ll_matmul_jit")
            .expect("Failed to find JIT function")
    };
    Some(ll_matmul_jit)
}

/// Converts a matrix from row-major to column-major order.
/// Input: src - flat row-major matrix (m x n)
/// Output: flat column-major matrix (m x n)
#[inline(always)]
pub(crate) fn row_major_to_col_major(src: &[f32], m: usize, n: usize) -> Vec<f32> {
    assert!(
        !src.is_empty(),
        "row_major_to_col_major :: `src` can't be empty"
    );
    let mut dst = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            dst[col * m + row] = src[row * n + col];
        }
    }
    dst
}

/// Converts a matrix from column-major to row-major order.
/// Input: src - flat column-major matrix (m x n)
/// Output: flat row-major matrix (m x n)
#[inline(always)]
pub(crate) fn col_major_to_row_major(src: &[f32], m: usize, n: usize) -> Vec<f32> {
    assert!(
        !src.is_empty(),
        "col_major_to_row_major :: `src` can't be empty"
    );
    let mut dst = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            dst[row * n + col] = src[col * m + row];
        }
    }
    dst
}
