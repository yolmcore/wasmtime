// cranelift/codegen/src/isa/x64/inst/turin/defs.rs
//
// YOLM FORK: AVX-512 instruction definitions for Turin/Zen 5.
//
// This module defines the AVX-512 instruction opcodes, merge modes,
// and related types used for Turin-native SIMD operations.

/// Merge mode for AVX-512 masked operations.
///
/// AVX-512 supports two merge modes when using mask registers:
/// - **Merging**: Elements where the mask bit is 0 retain their original value
/// - **Zeroing**: Elements where the mask bit is 0 are set to zero
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MergeMode {
    /// Preserve destination elements where mask bit is 0.
    Merging,
    /// Zero destination elements where mask bit is 0.
    #[default]
    Zeroing,
}

/// Mask register ALU operations.
///
/// These operate on k-registers directly (k1-k7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskAluOp {
    /// KAND - Bitwise AND of mask registers
    Kand,
    /// KOR - Bitwise OR of mask registers
    Kor,
    /// KXOR - Bitwise XOR of mask registers
    Kxor,
    /// KNOT - Bitwise NOT of mask register
    Knot,
    /// KANDN - Bitwise AND NOT of mask registers
    Kandn,
}

/// AVX-512 comparison conditions (for VPCMPD, VPCMPQ, etc.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Avx512Cond {
    /// Equal
    Eq = 0,
    /// Less than (signed)
    Lt = 1,
    /// Less than or equal (signed)
    Le = 2,
    /// Not equal
    Neq = 4,
    /// Greater than or equal (signed)
    Ge = 5,
    /// Greater than (signed)
    Gt = 6,
}

/// AVX-512 ALU operations for Turin.
///
/// These operations are optimized for the AMD EPYC Turin (Zen 5) architecture.
/// All operations use 512-bit vectors (ZMM registers) with EVEX encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Avx512AluOp {
    // =========================================
    // Integer Arithmetic Operations
    // =========================================

    /// VPADDD - Packed doubleword (32-bit) integer add
    Vpaddd,
    /// VPADDQ - Packed quadword (64-bit) integer add
    Vpaddq,
    /// VPSUBD - Packed doubleword (32-bit) integer subtract
    Vpsubd,
    /// VPSUBQ - Packed quadword (64-bit) integer subtract
    Vpsubq,
    /// VPMULLD - Packed 32-bit multiply low (low 32 bits of 32x32 result)
    Vpmulld,
    /// VPMULLQ - Packed 64-bit multiply low (low 64 bits of 64x64 result)
    Vpmullq,

    // =========================================
    // Bitwise Logical Operations
    // =========================================

    /// VPANDD - Packed bitwise AND (32-bit elements)
    Vpandd,
    /// VPANDQ - Packed bitwise AND (64-bit elements)
    Vpandq,
    /// VPORD - Packed bitwise OR (32-bit elements)
    Vpord,
    /// VPORQ - Packed bitwise OR (64-bit elements)
    Vporq,
    /// VPXORD - Packed bitwise XOR (32-bit elements)
    Vpxord,
    /// VPXORQ - Packed bitwise XOR (64-bit elements)
    Vpxorq,
    /// VPANDND - Packed bitwise AND NOT (32-bit elements)
    Vpandnd,
    /// VPANDNQ - Packed bitwise AND NOT (64-bit elements)
    Vpandnq,

    // =========================================
    // Shift Operations
    // =========================================

    /// VPSLLD - Packed shift left logical (32-bit, uniform shift)
    Vpslld,
    /// VPSLLQ - Packed shift left logical (64-bit, uniform shift)
    Vpsllq,
    /// VPSRLD - Packed shift right logical (32-bit, uniform shift)
    Vpsrld,
    /// VPSRLQ - Packed shift right logical (64-bit, uniform shift)
    Vpsrlq,
    /// VPSRAD - Packed shift right arithmetic (32-bit, uniform shift)
    Vpsrad,
    /// VPSRAQ - Packed shift right arithmetic (64-bit, uniform shift)
    Vpsraq,
    /// VPSLLVD - Packed shift left logical variable (32-bit, per-element shift)
    Vpsllvd,
    /// VPSLLVQ - Packed shift left logical variable (64-bit, per-element shift)
    Vpsllvq,
    /// VPSRLVD - Packed shift right logical variable (32-bit, per-element shift)
    Vpsrlvd,
    /// VPSRLVQ - Packed shift right logical variable (64-bit, per-element shift)
    Vpsrlvq,
    /// VPSRAVD - Packed shift right arithmetic variable (32-bit, per-element shift)
    Vpsravd,
    /// VPSRAVQ - Packed shift right arithmetic variable (64-bit, per-element shift)
    Vpsravq,

    // =========================================
    // Min/Max Operations
    // =========================================

    /// VPMINSD - Packed minimum signed (32-bit)
    Vpminsd,
    /// VPMINSQ - Packed minimum signed (64-bit)
    Vpminsq,
    /// VPMAXSD - Packed maximum signed (32-bit)
    Vpmaxsd,
    /// VPMAXSQ - Packed maximum signed (64-bit)
    Vpmaxsq,
    /// VPMINUD - Packed minimum unsigned (32-bit)
    Vpminud,
    /// VPMINUQ - Packed minimum unsigned (64-bit)
    Vpminuq,
    /// VPMAXUD - Packed maximum unsigned (32-bit)
    Vpmaxud,
    /// VPMAXUQ - Packed maximum unsigned (64-bit)
    Vpmaxuq,

    // =========================================
    // Absolute Value Operations
    // =========================================

    /// VPABSD - Packed absolute value (32-bit)
    Vpabsd,
    /// VPABSQ - Packed absolute value (64-bit)
    Vpabsq,

    // =========================================
    // Broadcast Operations
    // =========================================

    /// VPBROADCASTD - Broadcast 32-bit element to all lanes
    Vpbroadcastd,
    /// VPBROADCASTQ - Broadcast 64-bit element to all lanes
    Vpbroadcastq,

    // =========================================
    // Blend Operations
    // =========================================

    /// VPBLENDMD - Blend 32-bit elements using mask
    Vpblendmd,
    /// VPBLENDMQ - Blend 64-bit elements using mask
    Vpblendmq,

    // =========================================
    // Permute Operations (critical for columnar operations)
    // =========================================

    /// VPERMD - Permute 32-bit elements
    Vpermd,
    /// VPERMQ - Permute 64-bit elements
    Vpermq,
    /// VPERMI2D - Permute from two sources (32-bit)
    Vpermi2d,
    /// VPERMI2Q - Permute from two sources (64-bit)
    Vpermi2q,
    /// VPERMT2D - Permute from two sources with table (32-bit)
    Vpermt2d,
    /// VPERMT2Q - Permute from two sources with table (64-bit)
    Vpermt2q,

    // =========================================
    // Conflict Detection (for parallel histograms)
    // =========================================

    /// VPCONFLICTD - Detect 32-bit element conflicts
    Vpconflictd,
    /// VPCONFLICTQ - Detect 64-bit element conflicts
    Vpconflictq,

    // =========================================
    // Ternary Logic (powerful for complex operations)
    // =========================================

    /// VPTERNLOGD - Ternary logic on 32-bit elements
    Vpternlogd,
    /// VPTERNLOGQ - Ternary logic on 64-bit elements
    Vpternlogq,
}

impl Avx512AluOp {
    /// Returns the primary opcode byte for this operation.
    pub fn opcode(&self) -> u8 {
        match self {
            // Integer arithmetic
            Avx512AluOp::Vpaddd => 0xFE,
            Avx512AluOp::Vpaddq => 0xD4,
            Avx512AluOp::Vpsubd => 0xFA,
            Avx512AluOp::Vpsubq => 0xFB,
            Avx512AluOp::Vpmulld => 0x40,
            Avx512AluOp::Vpmullq => 0x40,

            // Bitwise logical
            Avx512AluOp::Vpandd => 0xDB,
            Avx512AluOp::Vpandq => 0xDB,
            Avx512AluOp::Vpord => 0xEB,
            Avx512AluOp::Vporq => 0xEB,
            Avx512AluOp::Vpxord => 0xEF,
            Avx512AluOp::Vpxorq => 0xEF,
            Avx512AluOp::Vpandnd => 0xDF,
            Avx512AluOp::Vpandnq => 0xDF,

            // Shifts (uniform)
            Avx512AluOp::Vpslld => 0xF2,
            Avx512AluOp::Vpsllq => 0xF3,
            Avx512AluOp::Vpsrld => 0xD2,
            Avx512AluOp::Vpsrlq => 0xD3,
            Avx512AluOp::Vpsrad => 0xE2,
            Avx512AluOp::Vpsraq => 0xE2, // Same opcode, different W bit

            // Shifts (variable)
            Avx512AluOp::Vpsllvd => 0x47,
            Avx512AluOp::Vpsllvq => 0x47,
            Avx512AluOp::Vpsrlvd => 0x45,
            Avx512AluOp::Vpsrlvq => 0x45,
            Avx512AluOp::Vpsravd => 0x46,
            Avx512AluOp::Vpsravq => 0x46,

            // Min/Max
            Avx512AluOp::Vpminsd => 0x39,
            Avx512AluOp::Vpminsq => 0x39,
            Avx512AluOp::Vpmaxsd => 0x3D,
            Avx512AluOp::Vpmaxsq => 0x3D,
            Avx512AluOp::Vpminud => 0x3B,
            Avx512AluOp::Vpminuq => 0x3B,
            Avx512AluOp::Vpmaxud => 0x3F,
            Avx512AluOp::Vpmaxuq => 0x3F,

            // Absolute value
            Avx512AluOp::Vpabsd => 0x1E,
            Avx512AluOp::Vpabsq => 0x1F,

            // Broadcast
            Avx512AluOp::Vpbroadcastd => 0x58,
            Avx512AluOp::Vpbroadcastq => 0x59,

            // Blend
            Avx512AluOp::Vpblendmd => 0x64,
            Avx512AluOp::Vpblendmq => 0x64,

            // Permute
            Avx512AluOp::Vpermd => 0x36,
            Avx512AluOp::Vpermq => 0x36,
            Avx512AluOp::Vpermi2d => 0x76,
            Avx512AluOp::Vpermi2q => 0x76,
            Avx512AluOp::Vpermt2d => 0x7E,
            Avx512AluOp::Vpermt2q => 0x7E,

            // Conflict detection
            Avx512AluOp::Vpconflictd => 0xC4,
            Avx512AluOp::Vpconflictq => 0xC4,

            // Ternary logic
            Avx512AluOp::Vpternlogd => 0x25,
            Avx512AluOp::Vpternlogq => 0x25,
        }
    }

    /// Returns the EVEX opcode map for this operation.
    ///
    /// - 0x01 = 0F map
    /// - 0x02 = 0F38 map
    /// - 0x03 = 0F3A map
    pub fn evex_map(&self) -> u8 {
        match self {
            // 0F map (simple arithmetic/logical)
            Avx512AluOp::Vpaddd
            | Avx512AluOp::Vpaddq
            | Avx512AluOp::Vpsubd
            | Avx512AluOp::Vpsubq
            | Avx512AluOp::Vpandd
            | Avx512AluOp::Vpandq
            | Avx512AluOp::Vpord
            | Avx512AluOp::Vporq
            | Avx512AluOp::Vpxord
            | Avx512AluOp::Vpxorq
            | Avx512AluOp::Vpandnd
            | Avx512AluOp::Vpandnq
            | Avx512AluOp::Vpslld
            | Avx512AluOp::Vpsllq
            | Avx512AluOp::Vpsrld
            | Avx512AluOp::Vpsrlq
            | Avx512AluOp::Vpsrad
            | Avx512AluOp::Vpsraq => 0x01,

            // 0F38 map (more complex operations)
            Avx512AluOp::Vpmulld
            | Avx512AluOp::Vpmullq
            | Avx512AluOp::Vpsllvd
            | Avx512AluOp::Vpsllvq
            | Avx512AluOp::Vpsrlvd
            | Avx512AluOp::Vpsrlvq
            | Avx512AluOp::Vpsravd
            | Avx512AluOp::Vpsravq
            | Avx512AluOp::Vpminsd
            | Avx512AluOp::Vpminsq
            | Avx512AluOp::Vpmaxsd
            | Avx512AluOp::Vpmaxsq
            | Avx512AluOp::Vpminud
            | Avx512AluOp::Vpminuq
            | Avx512AluOp::Vpmaxud
            | Avx512AluOp::Vpmaxuq
            | Avx512AluOp::Vpabsd
            | Avx512AluOp::Vpabsq
            | Avx512AluOp::Vpbroadcastd
            | Avx512AluOp::Vpbroadcastq
            | Avx512AluOp::Vpblendmd
            | Avx512AluOp::Vpblendmq
            | Avx512AluOp::Vpermd
            | Avx512AluOp::Vpermq
            | Avx512AluOp::Vpermi2d
            | Avx512AluOp::Vpermi2q
            | Avx512AluOp::Vpermt2d
            | Avx512AluOp::Vpermt2q
            | Avx512AluOp::Vpconflictd
            | Avx512AluOp::Vpconflictq => 0x02,

            // 0F3A map (ternary logic)
            Avx512AluOp::Vpternlogd | Avx512AluOp::Vpternlogq => 0x03,
        }
    }

    /// Returns the EVEX.pp (prefix) field for this operation.
    ///
    /// - 0x00 = no prefix
    /// - 0x01 = 66 prefix
    /// - 0x02 = F3 prefix
    /// - 0x03 = F2 prefix
    pub fn evex_pp(&self) -> u8 {
        // All our integer operations use 66 prefix
        0x01
    }

    /// Returns the EVEX.W bit for this operation.
    ///
    /// - false (0) = 32-bit operation
    /// - true (1) = 64-bit operation
    pub fn evex_w(&self) -> bool {
        match self {
            // 64-bit element operations
            Avx512AluOp::Vpaddq
            | Avx512AluOp::Vpsubq
            | Avx512AluOp::Vpmullq
            | Avx512AluOp::Vpandq
            | Avx512AluOp::Vporq
            | Avx512AluOp::Vpxorq
            | Avx512AluOp::Vpandnq
            | Avx512AluOp::Vpsllq
            | Avx512AluOp::Vpsrlq
            | Avx512AluOp::Vpsraq
            | Avx512AluOp::Vpsllvq
            | Avx512AluOp::Vpsrlvq
            | Avx512AluOp::Vpsravq
            | Avx512AluOp::Vpminsq
            | Avx512AluOp::Vpmaxsq
            | Avx512AluOp::Vpminuq
            | Avx512AluOp::Vpmaxuq
            | Avx512AluOp::Vpabsq
            | Avx512AluOp::Vpbroadcastq
            | Avx512AluOp::Vpblendmq
            | Avx512AluOp::Vpermq
            | Avx512AluOp::Vpermi2q
            | Avx512AluOp::Vpermt2q
            | Avx512AluOp::Vpconflictq
            | Avx512AluOp::Vpternlogq => true,

            // 32-bit element operations
            _ => false,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            Avx512AluOp::Vpaddd => "vpaddd",
            Avx512AluOp::Vpaddq => "vpaddq",
            Avx512AluOp::Vpsubd => "vpsubd",
            Avx512AluOp::Vpsubq => "vpsubq",
            Avx512AluOp::Vpmulld => "vpmulld",
            Avx512AluOp::Vpmullq => "vpmullq",
            Avx512AluOp::Vpandd => "vpandd",
            Avx512AluOp::Vpandq => "vpandq",
            Avx512AluOp::Vpord => "vpord",
            Avx512AluOp::Vporq => "vporq",
            Avx512AluOp::Vpxord => "vpxord",
            Avx512AluOp::Vpxorq => "vpxorq",
            Avx512AluOp::Vpandnd => "vpandnd",
            Avx512AluOp::Vpandnq => "vpandnq",
            Avx512AluOp::Vpslld => "vpslld",
            Avx512AluOp::Vpsllq => "vpsllq",
            Avx512AluOp::Vpsrld => "vpsrld",
            Avx512AluOp::Vpsrlq => "vpsrlq",
            Avx512AluOp::Vpsrad => "vpsrad",
            Avx512AluOp::Vpsraq => "vpsraq",
            Avx512AluOp::Vpsllvd => "vpsllvd",
            Avx512AluOp::Vpsllvq => "vpsllvq",
            Avx512AluOp::Vpsrlvd => "vpsrlvd",
            Avx512AluOp::Vpsrlvq => "vpsrlvq",
            Avx512AluOp::Vpsravd => "vpsravd",
            Avx512AluOp::Vpsravq => "vpsravq",
            Avx512AluOp::Vpminsd => "vpminsd",
            Avx512AluOp::Vpminsq => "vpminsq",
            Avx512AluOp::Vpmaxsd => "vpmaxsd",
            Avx512AluOp::Vpmaxsq => "vpmaxsq",
            Avx512AluOp::Vpminud => "vpminud",
            Avx512AluOp::Vpminuq => "vpminuq",
            Avx512AluOp::Vpmaxud => "vpmaxud",
            Avx512AluOp::Vpmaxuq => "vpmaxuq",
            Avx512AluOp::Vpabsd => "vpabsd",
            Avx512AluOp::Vpabsq => "vpabsq",
            Avx512AluOp::Vpbroadcastd => "vpbroadcastd",
            Avx512AluOp::Vpbroadcastq => "vpbroadcastq",
            Avx512AluOp::Vpblendmd => "vpblendmd",
            Avx512AluOp::Vpblendmq => "vpblendmq",
            Avx512AluOp::Vpermd => "vpermd",
            Avx512AluOp::Vpermq => "vpermq",
            Avx512AluOp::Vpermi2d => "vpermi2d",
            Avx512AluOp::Vpermi2q => "vpermi2q",
            Avx512AluOp::Vpermt2d => "vpermt2d",
            Avx512AluOp::Vpermt2q => "vpermt2q",
            Avx512AluOp::Vpconflictd => "vpconflictd",
            Avx512AluOp::Vpconflictq => "vpconflictq",
            Avx512AluOp::Vpternlogd => "vpternlogd",
            Avx512AluOp::Vpternlogq => "vpternlogq",
        }
    }

    /// Returns true if this is a 64-bit element operation.
    pub fn is_64bit(&self) -> bool {
        self.evex_w()
    }

    /// Returns true if this operation requires a third operand (ternary logic).
    pub fn is_ternary(&self) -> bool {
        matches!(self, Avx512AluOp::Vpternlogd | Avx512AluOp::Vpternlogq)
    }

    /// Returns true if this is a unary operation (single source).
    pub fn is_unary(&self) -> bool {
        matches!(
            self,
            Avx512AluOp::Vpabsd
                | Avx512AluOp::Vpabsq
                | Avx512AluOp::Vpbroadcastd
                | Avx512AluOp::Vpbroadcastq
        )
    }
}

impl MaskAluOp {
    /// Returns true if this is a unary operation (KNOT).
    pub fn is_unary(&self) -> bool {
        matches!(self, MaskAluOp::Knot)
    }

    /// Returns the VEX opcode for this mask operation.
    pub fn vex_opcode(&self) -> u8 {
        match self {
            MaskAluOp::Kand => 0x41,
            MaskAluOp::Kor => 0x45,
            MaskAluOp::Kxor => 0x47,
            MaskAluOp::Knot => 0x44,
            MaskAluOp::Kandn => 0x42,
        }
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            MaskAluOp::Kand => "kandw",
            MaskAluOp::Kor => "korw",
            MaskAluOp::Kxor => "kxorw",
            MaskAluOp::Knot => "knotw",
            MaskAluOp::Kandn => "kandnw",
        }
    }
}

impl Avx512Cond {
    /// Returns a human-readable name for this condition.
    pub fn name(&self) -> &'static str {
        match self {
            Avx512Cond::Eq => "eq",
            Avx512Cond::Lt => "lt",
            Avx512Cond::Le => "le",
            Avx512Cond::Neq => "neq",
            Avx512Cond::Ge => "ge",
            Avx512Cond::Gt => "gt",
        }
    }

    /// Returns the immediate byte encoding for this condition.
    pub fn imm(&self) -> u8 {
        *self as u8
    }
}

// =============================================================================
// Validation Functions
// =============================================================================

/// Validates that a k-register index is valid for use as a write mask (k1-k7).
/// k0 is reserved and means "no masking".
#[inline]
pub fn validate_mask_register(kreg: u8) -> Result<(), &'static str> {
    if kreg > 7 {
        Err("k-register index must be 0-7")
    } else {
        Ok(())
    }
}

/// Validates that a k-register index is valid for use as an active mask (k1-k7).
/// Returns an error if k0 is used where a real mask is required.
#[inline]
pub fn validate_active_mask_register(kreg: u8) -> Result<(), &'static str> {
    if kreg == 0 {
        Err("k0 cannot be used as an active write mask; use k1-k7")
    } else if kreg > 7 {
        Err("k-register index must be 1-7 for active masking")
    } else {
        Ok(())
    }
}

/// Validates that OperandSize is appropriate for AVX-512 integer operations.
#[inline]
pub fn validate_avx512_operand_size(size: &crate::isa::x64::inst::args::OperandSize) -> Result<(), &'static str> {
    use crate::isa::x64::inst::args::OperandSize;
    match size {
        OperandSize::Size32 | OperandSize::Size64 => Ok(()),
        _ => Err("AVX-512 integer operations only support 32-bit or 64-bit element sizes"),
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Avx512AluOp Tests
    // =========================================================================

    #[test]
    fn test_avx512_alu_op_opcodes() {
        // Verify critical opcodes match Intel documentation
        assert_eq!(Avx512AluOp::Vpaddd.opcode(), 0xFE);
        assert_eq!(Avx512AluOp::Vpaddq.opcode(), 0xD4);
        assert_eq!(Avx512AluOp::Vpsubd.opcode(), 0xFA);
        assert_eq!(Avx512AluOp::Vpsubq.opcode(), 0xFB);
        assert_eq!(Avx512AluOp::Vpmulld.opcode(), 0x40);
        assert_eq!(Avx512AluOp::Vpandd.opcode(), 0xDB);
        assert_eq!(Avx512AluOp::Vpord.opcode(), 0xEB);
        assert_eq!(Avx512AluOp::Vpxord.opcode(), 0xEF);
    }

    #[test]
    fn test_avx512_alu_op_evex_map() {
        // 0F map operations
        assert_eq!(Avx512AluOp::Vpaddd.evex_map(), 0x01);
        assert_eq!(Avx512AluOp::Vpaddq.evex_map(), 0x01);
        assert_eq!(Avx512AluOp::Vpandd.evex_map(), 0x01);

        // 0F38 map operations
        assert_eq!(Avx512AluOp::Vpmulld.evex_map(), 0x02);
        assert_eq!(Avx512AluOp::Vpminsd.evex_map(), 0x02);
        assert_eq!(Avx512AluOp::Vpermd.evex_map(), 0x02);

        // 0F3A map operations
        assert_eq!(Avx512AluOp::Vpternlogd.evex_map(), 0x03);
        assert_eq!(Avx512AluOp::Vpternlogq.evex_map(), 0x03);
    }

    #[test]
    fn test_avx512_alu_op_evex_w() {
        // 32-bit operations should have W=0
        assert!(!Avx512AluOp::Vpaddd.evex_w());
        assert!(!Avx512AluOp::Vpsubd.evex_w());
        assert!(!Avx512AluOp::Vpandd.evex_w());

        // 64-bit operations should have W=1
        assert!(Avx512AluOp::Vpaddq.evex_w());
        assert!(Avx512AluOp::Vpsubq.evex_w());
        assert!(Avx512AluOp::Vpandq.evex_w());
    }

    #[test]
    fn test_avx512_alu_op_evex_pp() {
        // All integer ops use 66 prefix
        assert_eq!(Avx512AluOp::Vpaddd.evex_pp(), 0x01);
        assert_eq!(Avx512AluOp::Vpaddq.evex_pp(), 0x01);
        assert_eq!(Avx512AluOp::Vpermd.evex_pp(), 0x01);
    }

    #[test]
    fn test_avx512_alu_op_is_64bit() {
        assert!(!Avx512AluOp::Vpaddd.is_64bit());
        assert!(Avx512AluOp::Vpaddq.is_64bit());
        assert!(!Avx512AluOp::Vpmulld.is_64bit());
        assert!(Avx512AluOp::Vpmullq.is_64bit());
    }

    #[test]
    fn test_avx512_alu_op_is_unary() {
        assert!(Avx512AluOp::Vpabsd.is_unary());
        assert!(Avx512AluOp::Vpbroadcastd.is_unary());
        assert!(!Avx512AluOp::Vpaddd.is_unary());
        assert!(!Avx512AluOp::Vpermd.is_unary());
    }

    #[test]
    fn test_avx512_alu_op_is_ternary() {
        assert!(Avx512AluOp::Vpternlogd.is_ternary());
        assert!(Avx512AluOp::Vpternlogq.is_ternary());
        assert!(!Avx512AluOp::Vpaddd.is_ternary());
    }

    #[test]
    fn test_avx512_alu_op_names() {
        assert_eq!(Avx512AluOp::Vpaddd.name(), "vpaddd");
        assert_eq!(Avx512AluOp::Vpaddq.name(), "vpaddq");
        assert_eq!(Avx512AluOp::Vpternlogd.name(), "vpternlogd");
    }

    // =========================================================================
    // MergeMode Tests
    // =========================================================================

    #[test]
    fn test_merge_mode_default() {
        assert_eq!(MergeMode::default(), MergeMode::Zeroing);
    }

    // =========================================================================
    // MaskAluOp Tests
    // =========================================================================

    #[test]
    fn test_mask_alu_op_is_unary() {
        assert!(MaskAluOp::Knot.is_unary());
        assert!(!MaskAluOp::Kand.is_unary());
        assert!(!MaskAluOp::Kor.is_unary());
        assert!(!MaskAluOp::Kxor.is_unary());
        assert!(!MaskAluOp::Kandn.is_unary());
    }

    #[test]
    fn test_mask_alu_op_vex_opcode() {
        assert_eq!(MaskAluOp::Kand.vex_opcode(), 0x41);
        assert_eq!(MaskAluOp::Kor.vex_opcode(), 0x45);
        assert_eq!(MaskAluOp::Kxor.vex_opcode(), 0x47);
        assert_eq!(MaskAluOp::Knot.vex_opcode(), 0x44);
        assert_eq!(MaskAluOp::Kandn.vex_opcode(), 0x42);
    }

    #[test]
    fn test_mask_alu_op_names() {
        assert_eq!(MaskAluOp::Kand.name(), "kandw");
        assert_eq!(MaskAluOp::Kor.name(), "korw");
        assert_eq!(MaskAluOp::Kxor.name(), "kxorw");
        assert_eq!(MaskAluOp::Knot.name(), "knotw");
        assert_eq!(MaskAluOp::Kandn.name(), "kandnw");
    }

    // =========================================================================
    // Avx512Cond Tests
    // =========================================================================

    #[test]
    fn test_avx512_cond_imm() {
        assert_eq!(Avx512Cond::Eq.imm(), 0);
        assert_eq!(Avx512Cond::Lt.imm(), 1);
        assert_eq!(Avx512Cond::Le.imm(), 2);
        assert_eq!(Avx512Cond::Neq.imm(), 4);
        assert_eq!(Avx512Cond::Ge.imm(), 5);
        assert_eq!(Avx512Cond::Gt.imm(), 6);
    }

    #[test]
    fn test_avx512_cond_names() {
        assert_eq!(Avx512Cond::Eq.name(), "eq");
        assert_eq!(Avx512Cond::Lt.name(), "lt");
        assert_eq!(Avx512Cond::Le.name(), "le");
        assert_eq!(Avx512Cond::Neq.name(), "neq");
        assert_eq!(Avx512Cond::Ge.name(), "ge");
        assert_eq!(Avx512Cond::Gt.name(), "gt");
    }

    // =========================================================================
    // Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_mask_register() {
        // Valid k-registers
        for i in 0..=7 {
            assert!(validate_mask_register(i).is_ok());
        }
        // Invalid k-registers
        assert!(validate_mask_register(8).is_err());
        assert!(validate_mask_register(255).is_err());
    }

    #[test]
    fn test_validate_active_mask_register() {
        // k0 is not valid for active masking
        assert!(validate_active_mask_register(0).is_err());
        // k1-k7 are valid
        for i in 1..=7 {
            assert!(validate_active_mask_register(i).is_ok());
        }
        // Invalid k-registers
        assert!(validate_active_mask_register(8).is_err());
    }
}
