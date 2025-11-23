use std::env;
use std::path::PathBuf;
use std::process::Command;

fn compile_llvm_ir(ll_file: &PathBuf, obj_file: &PathBuf, is_debug: bool) {
    println!("cargo:rerun-if-changed=src/llvm/{}", ll_file.display());

    // .ll needs to be lowered (vectorized ?) by opt
    // cause guess what, llc is a piece of junk !
    let lowered_ll_file = obj_file.with_extension("lowered.ll");
    let mut opt_command = Command::new("opt");
    opt_command
        .arg("-passes=lower-matrix-intrinsics")
        .arg("-S")
        .arg("-o")
        .arg(&lowered_ll_file)
        .arg(ll_file);
    if is_debug {
        opt_command.arg("--debug-entry-values");
        opt_command.arg("-print-after=lower-matrix-intrinsics");
    } else {
        opt_command.arg("--thinlto-bc");
    }

    let status = opt_command
        .status()
        .expect("Failed to execute opt. Make sure LLVM opt is installed and in PATH.");

    if !status.success() {
        panic!(
            "opt failed to lower matrix intrinsics in LLVM IR file: {:?}",
            ll_file
        );
    }

    // fine, there, you have a IR that
    // even my grandma can execute.
    let mut llc_command = Command::new("llc");
    llc_command
        .arg("-mattr=+avx2,+fma")
        .arg("-mcpu=native")
        .arg("--relocation-model=pic")
        .arg("-filetype=obj")
        .arg("-o")
        .arg(obj_file)
        .arg(&lowered_ll_file);
    if is_debug {
        llc_command.arg("--asm-verbose");
        llc_command.arg("--debug-entry-values");
        llc_command.arg("--debugger-tune=gdb");
        // TODO : switch all to row major
        //llc_command.arg("--matrix-default-layout=column-major"); // row-major

        // we can add bunch of fp flags if need to
        //llc_command.arg("-print-asm-code");
        //llc_command.arg("-time-passes");
    } else {
        llc_command.arg("-O3");
    }

    let status = llc_command
        .status()
        .expect("Failed to execute llc. Make sure LLVM llc is installed and in PATH.");

    if !status.success() {
        panic!("llc failed to compile LLVM IR file: {:?}", ll_file);
    }
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let is_debug = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string()) == "debug";

    // matmul_4x4.ll file contains :
    // - the llvm ir for the matmul with transpose operation
    // - the llvm ir for the matmul with unrolled operation

    // the transpose one need lowering with opt
    // LLVM matrix intrinsics (like llvm.matrix.transpose, llvm.matrix.multiply, llvm.matrix.column.major.load/store)
    // require lowering (i.e., transformation from high-level matrix intrinsic IR to something the backend
    // can actually JIT/compile))

    // the unrolled one I don't think it needs lowering
    // FIXME: move unrolled one to a different file
    // and avoid opting it (later for benchmark might have an impact)

    let matmul_4x4_ll = manifest_dir.join("src/llvm/matmul_4x4.ll");
    let matmul_4x4_obj = out_dir.join("matmul_4x4.o");
    compile_llvm_ir(&matmul_4x4_ll, &matmul_4x4_obj, is_debug);

    // link with all .o files
    println!("cargo:rustc-link-arg={}", matmul_4x4_obj.display());

    // generic one is jit
}
