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
ll_matmul_4x4_unrolled  time:   [35.444 ns 35.493 ns 35.555 ns]
Found 6 outliers among 100 measurements (6.00%)
  1 (1.00%) low severe
  1 (1.00%) low mild
  1 (1.00%) high mild
  3 (3.00%) high severe

ll_matmul_4x4_using_transpose
                        time:   [23.005 ns 23.052 ns 23.103 ns]
Found 6 outliers among 100 measurements (6.00%)
  1 (1.00%) low mild
  3 (3.00%) high mild
  2 (2.00%) high severe

ndarray_dot             time:   [138.61 ns 138.87 ns 139.18 ns]
Found 9 outliers among 100 measurements (9.00%)
  2 (2.00%) low mild
  5 (5.00%) high mild
  2 (2.00%) high severe

matrixmultiply_sgemm    time:   [81.371 ns 81.581 ns 81.835 ns]
Found 8 outliers among 100 measurements (8.00%)
  3 (3.00%) low mild
  1 (1.00%) high mild
  4 (4.00%) high severe

faer_matmul             time:   [130.19 ns 131.46 ns 132.65 ns]
Found 16 outliers among 100 measurements (16.00%)
  11 (11.00%) high mild
  5 (5.00%) high severe

ll_matmul_jit_generic   time:   [2.1640 ms 2.1889 ms 2.2207 ms]
Found 10 outliers among 100 measurements (10.00%)
  4 (4.00%) high mild
  6 (6.00%) high severe

```

## Note
This is a minimal POC and is provided as-is for educational purposes.
