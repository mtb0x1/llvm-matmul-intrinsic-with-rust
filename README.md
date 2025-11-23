# LLVM Matmul Intrinsic with Rust - POC

This is a **Proof of Concept** project demonstrating the use of LLVM's matrix multiplication intrinsics with Rust. 

- provides a 4x4 matmul routine (unrolled, with transposed support)
- includes a generic LLVM-based matrix multiplication implementation that is compiled and run as JIT at runtime

## Important Note 

This project was built with and using  :
- unstable llvm version ([b00c62](https://github.com/llvm/llvm-project/commit/b00c620b3504565d9769a434bc7d4e97854cd788))

- unstable linker version : mold ([5b7102](https://github.com/rui314/mold/commit/5b710298fcae04bfcb00d3e3dd324c5638f02696)

## Requirements

- LLVM (with matrix intrinsics support, 21+ ?)
- Rust toolchain (make sure it has a compatible llvm version : rustc -vV | grep "LLVM")

## Usage

```bash
cargo build
cargo run
cargo test
```

## Benchmark

```console
ll_matmul_4x4_using_transpose  time:   [19.854 ns 19.940 ns 20.121 ns]
ll_matmul_4x4_unrolled         time:   [27.314 ns 27.393 ns 27.520 ns]
matrixmultiply_sgemm  (4x4)    time:   [81.371 ns 81.404 ns 81.440 ns]
ndarray_dot           (4x4)    time:   [99.155 ns 99.513 ns 100.23 ns]
faer_matmul           (4x4)    time:   [100.80 ns 100.90 ns 101.01 ns]
ll_matmul_jit_generic (4x4)    time:   [176.33 ns 177.01 ns 178.00 ns]
```

## Note
This is a minimal POC and is provided as-is for educational purposes.
