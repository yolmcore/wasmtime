// cranelift/codegen/src/isa/x64/inst/avx512/mod.rs
//
// AVX-512: (Zen 5) AVX-512 instruction support.
//
// This module provides native 512-bit SIMD instruction emission for AMD EPYC
// 5th Gen (Avx512) and Ryzen 9000 series (Zen 5) processors with full AVX-512
// support.
//
// ## Modules
// - `defs`: Instruction definitions (opcodes, enums, metadata)
// - `emit`: EVEX instruction emission functions
// - `encoding`: EVEX prefix encoding utilities
// - `regs`: K-register utilities
//
// ## Supported Instructions
// - ALU: VPADDD/Q, VPSUBD/Q, VPMULLD/Q, VPAND/OR/XOR, shifts, min/max
// - Compare: VPCMPD/Q with result to k-register
// - Compress/Expand: VPCOMPRESSD/Q, VPEXPANDD/Q
// - Masked Load/Store: VMOVDQU32/64 with fault suppression
// - Gather/Scatter: VPGATHERD/Q, VPSCATTERD/Q
// - K-register ops: KAND, KOR, KXOR, KNOT, KANDN, KMOV, KORTEST

pub mod defs;
pub mod emit;
pub mod encoding;
pub mod regs;
#[cfg(test)]
mod runtime_tests;

pub use defs::*;
