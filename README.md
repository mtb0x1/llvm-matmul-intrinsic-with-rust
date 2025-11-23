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

maybe ?


## Note
This is a minimal POC and is provided as-is for educational purposes.
