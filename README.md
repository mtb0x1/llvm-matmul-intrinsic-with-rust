# LLVM Intrinsic with Rust - POC

This is a **Proof of Concept** project demonstrating the use of LLVM's matrix multiplication intrinsics with Rust.

## Features

- **CPU Implementations**:
  - **4x4 Matmul Routines**: Optimized implementations including unrolled loops and transposed matrix support (compiled)
  - **Generic JIT Compilation**: Runtime LLVM IR compilation for matrix multiplication of arbitrary sizes with customizable templates
  
- **GPU Implementations**:
  - **CUDA-based GPU Kernels**: Compiled and JIT-compiled GPU matrix multiplication routines
  - **Automatic Context Management**: CUDA context initialization and memory management
  
- **Customizable Templates**: Swap LLVM IR implementation strategies at runtime using environment variables

## Requirements

- **LLVM**: v21+ (Ensure `llvm-config` is in your PATH or `LLVM_SYS_211_PREFIX` is set)
- **CUDA**: v13.0+ (Required for GPU computation and benchmarks)
- **Rust**: Nightly toolchain recommended for latest features.
- **Linker**: `mold` is recommended for faster linking times.

## Usage

### Basic Run

```bash
cargo run
```

This runs CPU and GPU implementations (if CUDA 13.0+ is available) with various matrix sizes:
- 2x3 * 3x4 multiplication
- 4x4 matrix multiplications using different implementations

### CPU-Only Execution

```bash
cargo run --no-default-features
```

### GPU-Enabled Execution

```bash
cargo run --features gpu
```

Requires CUDA 13.0+ installed and configured.

### Using Custom JIT Templates

Control the LLVM IR template for CPU JIT compilation via the `LL_MATMUL_TEMPLATE` environment variable.

**Use the unrolled loop implementation:**
```bash
LL_MATMUL_TEMPLATE=src/llvm/matmul_unrolled.tmpl cargo run
```

**Use the naive intrinsic implementation (default):**
```bash
cargo run
```

### Customizing Function Name

Specify custom function names in LLVM IR templates via `LL_MATMUL_TEMPLATE_FUNCTION_NAME`:

```bash
LL_MATMUL_TEMPLATE=src/llvm/matmul_unrolled.tmpl LL_MATMUL_TEMPLATE_FUNCTION_NAME=ll_matmul_cpu_jit cargo run
```

### Running Tests

```bash
cargo test
```

## Benchmarks

Run comprehensive benchmarks comparing CPU and GPU implementations across different matrix sizes:

```bash
cargo bench --features gpu
```

### Benchmark Suites

- **matmul_4x4**: Small 4x4 matrix operations comparing:
  - LLVM intrinsics (unrolled and transposed)
  - CPU JIT compilation
  - GPU implementations (compiled and JIT)
  - Reference libraries (ndarray, matrixmultiply, faer)

- **matmul_small_32x32**: 32x32 matrix operations
- **matmul_mid_512x512**: 512x512 matrix operations
- **matmul_big_1024x1024**: 1024x1024 matrix operations

### Key Observations

- **JIT vs Compiled**: JIT implementations are slower than pre-compiled code due to compilation overhead
- **CPU vs GPU**: GPU excels with larger matrices (512x512+), compensating for context creation and data transfer costs
- **Similar but Different Code**: Benchmarking different but equivalent LLVM code affects performance comparisons
- **Power Management**: GPU and CPU power settings impact results
- **Known Limitations**: 
  - GPU JIT and compiled implementations have redundant code (refactoring pending)
  - GPU benchmarks include context creation and memory transfer overhead

*Note: Performance varies significantly based on matrix dimensions, template strategy, and hardware configuration.*

## Important Note

This project uses unstable LLVM features and intrinsics. It is provided as-is for educational purposes.
