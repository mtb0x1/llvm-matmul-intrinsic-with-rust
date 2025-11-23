# LLVM Matmul Intrinsic with Rust - POC

This is a **Proof of Concept** project demonstrating the use of LLVM's matrix multiplication intrinsics with Rust. 

- provides a 4x4 matmul routine (unrolled, with transposed support)
- includes a generic LLVM-based matrix multiplication implementation that is compiled and run as JIT at runtime

## Important Note 

This project was built with and using  :
- unstable llvm version ([b00c62](https://github.com/llvm/llvm-project/commit/b00c620b3504565d9769a434bc7d4e97854cd788))

- unstable linker version : mold ([5b7102](https://github.com/rui314/mold/commit/5b710298fcae04bfcb00d3e3dd324c5638f02696))

## Requirements

- LLVM (v22+ ?)
- Rust (ex: rust v1.91+) (make sure it has a compatible llvm version : rustc -vV | grep "LLVM")

## Usage

```bash
cargo build
cargo run
cargo test
```

## Benchmark

```console
4x4 :
    ll_matmul_4x4_unrolled                        time:   [35.781 ns 37.247 ns 38.872 ns]
    ll_matmul_4x4_using_transpose                 time:   [36.268 ns 37.623 ns 38.972 ns]
    faer_matmul                                   time:   [109.13 ns 110.42 ns 112.16 ns]
    matrixmultiply_sgemm                          time:   [132.35 ns 132.98 ns 133.70 ns]
    ndarray_dot                                   time:   [149.48 ns 151.14 ns 152.70 ns]
    ll_matmul_jit_with_template                   time:   [170.64 ns 171.90 ns 173.40 ns]

32x32 :
    ndarray_dot                        time:   [1.2072 µs 1.2164 µs 1.2271 µs]
    matrixmultiply_sgemm               time:   [1.4104 µs 1.4263 µs 1.4567 µs]
    ll_matmul_jit_with_template        time:   [5.2151 µs 5.2679 µs 5.3327 µs]
    faer_matmul                        time:   [63.739 µs 70.954 µs 79.596 µs] # ???
```

## Note
This is a minimal POC and is provided as-is for educational purposes.
