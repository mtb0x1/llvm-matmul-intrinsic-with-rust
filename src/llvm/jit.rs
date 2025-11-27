use core::panic;
use std::collections::{HashMap, hash_map::Entry};
use std::env;
use std::ffi::CStr;
use std::fs;
use std::sync::{Arc, Mutex, OnceLock};

use crate::common::{DEFAULT_FUNCTION_NAME_JIT_CPU, TEMPLATE_JIT_CPU_ENV};
use crate::common::{DEFAULT_IR_TEMPLATE_JIT_CPU, TEMPLATE_JIT_CPU_ENV_FUNCTION_NAME};

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

type LlMatmulJitSig = unsafe extern "C" fn(*const f32, *const f32, *mut f32);
type ShapeKey = (usize, usize, usize);

pub static JIT_CACHE: OnceLock<JitCache> = OnceLock::new();

#[allow(dead_code)]
pub struct JitEntry {
    // ExecutionEngine and JitFunction hold references to LLVM objects
    // that must not be dropped. We use Box::leak to convert to 'static references.
    pub func: JitFunction<'static, LlMatmulJitSig>,
}

// Context and ExecutionEngine are not modified after creation.
// every JitEntry has its own Context and ExecutionEngine, so there is no risk of data race.
unsafe impl Send for JitEntry {}
unsafe impl Sync for JitEntry {}

pub struct JitCache {
    map: Mutex<HashMap<ShapeKey, Arc<JitEntry>>>,
}

#[derive(Debug)]
enum JitError {
    CompilationFailed(String),
}

impl JitCache {
    fn new() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
        }
    }

    fn get_or_compile(
        &self,
        shape: ShapeKey,
        ir_template: Option<&str>,
    ) -> Result<Arc<JitEntry>, JitError> {
        // First check with read lock (if we had RwLock, but Mutex is fine for now)
        // Optimization: check if exists before compiling
        {
            let map = self.map.lock().unwrap();
            if let Some(e) = map.get(&shape).cloned() {
                return Ok(e);
            }
        }

        // compile, create a Box<Context>, create module with that context,
        // create execution_engine, get function, wrap in Arc<JitEntry>
        let entry = unsafe {
            compile_matmul_jit_with_template(shape.0, shape.1, shape.2, ir_template)
                .map_err(JitError::CompilationFailed)?
        };

        let entry = Arc::new(entry);

        let mut map = self.map.lock().unwrap();

        // in case another thread compiled it already while we were jit-compiling
        match map.entry(shape) {
            Entry::Occupied(e) => Ok(e.get().clone()),
            Entry::Vacant(e) => {
                e.insert(entry.clone());
                Ok(entry)
            }
        }
    }
}

// template to be udated at runtime
// Matrix multiplication (row major in rust, column major in llvm):
// I guess this needs a FIXME, but for now it doesn't worth the hassel
//  C(m×n) = A(m×k) * B(k×n)
pub unsafe fn ll_matmul_jit_with_template(
    a: &[f32],
    a_shape: (usize, usize),
    b: &[f32],
    b_shape: (usize, usize),
    ir_template: Option<&str>,
) -> Vec<f32> {
    assert!(
        a_shape.0 > 0 && a_shape.1 > 0 && b_shape.0 > 0 && b_shape.1 > 0,
        "empty arrays are not supported"
    );
    assert!(a_shape.1 == b_shape.0, "shapes dosn't match");

    let m = a_shape.0;
    let n = b_shape.1;
    let k = a_shape.1; // or b_shape.0

    let shape_key: ShapeKey = (m, n, k);

    let cache = JIT_CACHE.get_or_init(JitCache::new);
    let entry = cache
        .get_or_compile(shape_key, ir_template)
        .unwrap_or_else(|e| match e {
            JitError::CompilationFailed(msg) => panic!("JIT Compilation failed: {}", msg),
        });

    let a_col_major = row_major_to_col_major(a, a_shape.0, a_shape.1);
    let b_col_major = row_major_to_col_major(b, b_shape.0, b_shape.1);
    let mut result = vec![0.0; m * n];

    unsafe {
        //println!("calling ll_matmul_jit");
        entry.func.call(
            a_col_major.as_ptr(),
            b_col_major.as_ptr(),
            result.as_mut_ptr(),
        );
    }

    col_major_to_row_major(&result, m, n)
}

pub unsafe fn compile_matmul_jit_with_template(
    m: usize,
    n: usize,
    k: usize,
    ir_template: Option<&str>,
) -> Result<JitEntry, String> {
    //println!("compiling matmul_jit for m={}, n={}, k={}", m, n, k);

    let template_content = if let Some(t) = ir_template {
        t.to_string()
    } else if let Ok(path) = env::var(TEMPLATE_JIT_CPU_ENV) {
        fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("Failed to read template from {}: {}", path, e);
        })
    } else {
        eprintln!(
            r#" 
// You are using `DEFAULT_IR_TEMPLATE` (naive)
// with size of ({m}x{n} * {k}x{n})
// when targeting non specilized hardware (CPU),
// opt will go crazy and try to lower intrinsics to the lowest level possible
// and this will make the code explode in size
// and lead to oom
// so we need to have a threshold (32 in my case)
// up to you to check which value is best for you
// you can play with examples/debug_large_matmul.rs to check which value is best for you
"#
        );
        DEFAULT_IR_TEMPLATE_JIT_CPU.to_string()
    };

    // Check if the template contains placeholders (it should for JIT instantiation)
    if !template_content.contains("{M}")
        && !template_content.contains("{N}")
        && !template_content.contains("{K}")
    {
        return Err(
            "Template must contain placeholders like {M}, {N}, {K} for matrix dimensions. \
             If using a hardcoded IR file (like matmul_4x4.ll), do not use it as a template for different sizes.".to_string()
        );
    }

    let ir_runtime = template_content
        .replace("{M}", &m.to_string())
        .replace("{N}", &n.to_string())
        .replace("{K}", &k.to_string())
        .replace("{VEC_A_SIZE}", &((m * k).to_string()))
        .replace("{VEC_B_SIZE}", &((k * n).to_string()))
        .replace("{VEC_C_SIZE}", &((m * n).to_string()))
        .replace("{A_STRIDE}", &m.to_string())
        .replace("{B_STRIDE}", &k.to_string())
        .replace("{C_STRIDE}", &m.to_string());

    //println!("IR instantiated:\n{}", ir_runtime);

    // each JIT compilation gets its own context leaked to 'static
    // this is okay(?) because llvm-ontext needs to live for the entire program
    let context = Box::leak(Box::new(Context::create()));

    let buffer = MemoryBuffer::create_from_memory_range_copy(ir_runtime.as_bytes(), "matmul_ir");
    let module: Module<'static> = match context.create_module_from_ir(buffer) {
        Ok(module) => module,
        Err(e) => {
            return Err(format!("Failed to parse LLVM IR: {}", e));
        }
    };

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
        Err(e) => {
            return Err(format!(
                "{} {}",
                format!("target from triplet failed : {}", triple),
                e
            ));
        }
    };
    let machine = match target.create_target_machine(
        &triple,
        TargetMachine::get_host_cpu_name()
            .to_str()
            .expect("host cpu name"), // FIXME: is this same as native ?
        "+avx2,+fma",
        OptimizationLevel::Aggressive,
        RelocMode::PIC,
        CodeModel::JITDefault,
    ) {
        Some(tm) => tm,
        None => return Err("couldn't create target machine".to_string()),
    };

    let pass_options = PassBuilderOptions::create();
    // TODO : this pass fails set to true
    // needs FIXME ?
    pass_options.set_verify_each(false);
    //FIXME: should be true in debug mode, but not in test or bench mode
    pass_options.set_debug_logging(false);

    // https://llvm.org/docs/NewPassManager.html#invoking-opt
    // opt --help
    // the 4 next passes are most have for matrix multiplication
    pass_options.set_loop_interleaving(true);
    pass_options.set_loop_vectorization(true);
    pass_options.set_loop_slp_vectorization(true);
    pass_options.set_loop_unrolling(true);

    // don't know about these
    // we need to align with build.rs
    pass_options.set_forget_all_scev_in_loop_unroll(true);
    pass_options.set_licm_mssa_opt_cap(1);
    pass_options.set_licm_mssa_no_acc_for_promotion_cap(10);
    pass_options.set_call_graph_profile(true);
    pass_options.set_merge_functions(true);

    unsafe {
        //println!("running passes");
        let error = llvm_sys::transforms::pass_builder::LLVMRunPasses(
            module.as_mut_ptr(),
            c"lower-matrix-intrinsics".as_ptr(),
            machine.as_mut_ptr(),
            pass_options.as_mut_ptr(),
        );
        //println!("passes run done");
        if !error.is_null() {
            let message = LLVMGetErrorMessage(error);
            if !message.is_null() {
                let message_str = CStr::from_ptr(message).to_string_lossy().into_owned();
                LLVMDisposeMessage(message);
                return Err(format!(
                    "shit happend can't recover any diag! good luck: {}",
                    message_str
                ));
            } else {
                return Err("shit happend can't recover any diag! good luck".to_string());
            }
        }
    };

    //println!("IR lowered:\n{}", module.print_to_string());

    let execution_engine = Box::leak(Box::new(
        match module.create_jit_execution_engine(OptimizationLevel::Aggressive) {
            Ok(execution_engine) => execution_engine,
            Err(e) => {
                return Err(format!("Failed to create JIT execution engine: {}", e));
            }
        },
    ));
    //println!("execution_engine created");

    let ll_matmul_jit: JitFunction<LlMatmulJitSig> = unsafe {
        let function_name = env::var(TEMPLATE_JIT_CPU_ENV_FUNCTION_NAME)
            .unwrap_or(DEFAULT_FUNCTION_NAME_JIT_CPU.to_string());
        match execution_engine.get_function(function_name.as_str()) {
            Ok(ll_matmul_jit) => ll_matmul_jit,
            Err(e) => {
                return Err(format!(
                    "Failed to find JIT function {} : {}",
                    function_name, e
                ));
            }
        }
    };
    //println!("ll_matmul_jit found");
    Ok(JitEntry {
        func: ll_matmul_jit,
    })
}

/// Converts a matrix from row-major to column-major order.
/// Input: src - flat row-major matrix (m x n)
/// Output: flat column-major matrix (m x n)
#[inline(always)]
pub fn row_major_to_col_major(src: &[f32], m: usize, n: usize) -> Vec<f32> {
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
pub fn col_major_to_row_major(src: &[f32], m: usize, n: usize) -> Vec<f32> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_caching() {
        let cache = JIT_CACHE.get_or_init(JitCache::new);

        let shape1: ShapeKey = (2, 2, 2);
        let shape2: ShapeKey = (3, 3, 3);

        let entry1_a = cache
            .get_or_compile(shape1, None)
            .expect("Failed to compile shape1");

        let entry1_b = cache
            .get_or_compile(shape1, None)
            .expect("Failed to compile shape1 again");
        assert!(
            Arc::ptr_eq(&entry1_a, &entry1_b),
            "Cache should return the same Arc for the same shape"
        );

        let entry2 = cache
            .get_or_compile(shape2, None)
            .expect("Failed to compile shape2");
        assert!(
            !Arc::ptr_eq(&entry1_a, &entry2),
            "Cache should return different Arcs for different shapes"
        );

        let entry1_c = cache
            .get_or_compile(shape1, None)
            .expect("Failed to retrieve shape1");
        assert!(
            Arc::ptr_eq(&entry1_a, &entry1_c),
            "Original entry should still be in cache"
        );
    }
}
