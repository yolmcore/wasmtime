// cranelift/codegen/src/isa/x64/inst/turin/regs.rs
//
// YOLM FORK: Turin AVX-512 k-register utilities.
//
// This module provides utilities for working with AVX-512 mask registers (k-registers).
// The actual k-register definitions are in the main regs.rs module.

// Re-export k-register constructors from main regs module
pub use super::super::regs::{k0, k1, k2, k3, k4, k5, k6, k7, k_preg};

/// Returns the hardware encoding for a k-register (0-7).
pub fn k_hw_enc(kreg: u8) -> u8 {
    debug_assert!(kreg < 8, "k-register index must be 0-7");
    kreg
}

/// Check if a k-register index is valid for masking (k1-k7).
/// k0 is special and means "no masking".
pub fn is_valid_mask_kreg(kreg: u8) -> bool {
    kreg >= 1 && kreg <= 7
}
