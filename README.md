# LLVM Matmul Intrinsic with Rust - POC

This is a **Proof of Concept** project demonstrating the use of LLVM's matrix multiplication intrinsics with Rust.

## Features

- **4x4 Matmul Routine**: optimized unrolled implementation with transposed support.
- **Generic JIT Implementation**: Runtime LLVM IR compilation for matrix multiplication of arbitrary sizes.
- **Customizable Templates**: Ability to swap LLVM IR implementation strategies at runtime using environment variables.

## Requirements

- **LLVM**: v21+ (Ensure `llvm-config` is in your PATH or `LLVM_SYS_211_PREFIX` is set)
- **Rust**: Nightly toolchain recommended for latest features.
- **Linker**: `mold` is recommended for faster linking times.

## Usage

### Basic Run

```bash
cargo run
```

### Using Custom JIT Templates

You can switch the LLVM IR template used by the JIT compiler by setting the `LL_MATMUL_TEMPLATE` environment variable.

**Use the unrolled loop implementation:**
```bash
LL_MATMUL_TEMPLATE=src/llvm/matmul_unrolled.tmpl cargo run
```

**Use the naive intrinsic implementation (default):**
```bash
# This is the default behavior if no env var is set
cargo run
```

### Running Tests

```bash
cargo test
```

## Benchmarks

To run benchmarks:

```bash
cargo bench
```

*Note: Performance varies significantly based on the template used and the matrix dimensions.*

## Important Note

This project uses unstable LLVM features and intrinsics. It is provided as-is for educational purposes.
