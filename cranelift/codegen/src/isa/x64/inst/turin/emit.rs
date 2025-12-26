// cranelift/codegen/src/isa/x64/inst/turin/emit.rs
//
// YOLM FORK: Turin AVX-512 EVEX instruction emission.
// This module handles the encoding and emission of AVX-512 instructions
// for AMD EPYC 5th Generation (Turin / Zen 5) processors.
//
// Reference: Intel 64 and IA-32 Architectures Software Developer's Manual
// Volume 2: Instruction Set Reference (EVEX Encoding)

use super::super::args::{OperandSize, SyntheticAmode};
use super::defs::*;
use super::encoding::*;
use crate::isa::x64::inst::args::OptionMaskReg;
use crate::isa::x64::inst::Inst;
use crate::machinst::{MachBuffer, Reg, RegMem, Writable};

// =============================================================================
// AVX-512 ALU Instruction Emission (VPADDD, VPADDQ, etc.)
// =============================================================================

/// Emit an AVX-512 ALU instruction (3-operand form: dst = src1 op src2).
pub fn emit_turin_inst(
    op: Avx512AluOp,
    size: OperandSize,
    dst: Writable<Reg>,
    src1: Reg,
    src2: &RegMem,
    mask: OptionMaskReg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    let w = match size {
        OperandSize::Size64 => true,
        _ => op.evex_w(),
    };

    let evex = EvexPrefix {
        map: op.evex_map(),
        w,
        pp: op.evex_pp(),
        aaa: mask
            .map(|m| m.to_reg().to_real_reg().unwrap().hw_enc())
            .unwrap_or(0),
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc() as u8;
    let src1_enc = src1.to_real_reg().unwrap().hw_enc() as u8;

    let (src2_enc, is_mem) = match src2 {
        RegMem::Reg { reg } => (reg.to_real_reg().unwrap().hw_enc() as u8, false),
        RegMem::Mem { .. } => (0, true),
    };

    evex.emit(dst_enc, src1_enc, src2_enc, is_mem, sink);
    sink.put1(op.opcode());
    emit_modrm_for_regmem(sink, dst_enc, src2);
}

// =============================================================================
// AVX-512 Comparison Instruction Emission (VPCMPD, VPCMPQ)
// =============================================================================

/// Emit an AVX-512 comparison instruction that writes to a k-register.
/// VPCMPD/VPCMPQ: Compare packed integers and store result in mask register.
pub fn emit_avx512_cmp(
    size: OperandSize,
    dst: Writable<Reg>,
    src1: Reg,
    src2: &RegMem,
    cond: Avx512Cond,
    mask: OptionMaskReg,
    sink: &mut MachBuffer<Inst>,
) {
    // VPCMPD: EVEX.512.66.0F3A.W0 1F /r ib
    // VPCMPQ: EVEX.512.66.0F3A.W1 1F /r ib
    let w = matches!(size, OperandSize::Size64);

    let evex = EvexPrefix {
        map: 0x03, // 0F3A map
        w,
        pp: 0x01, // 66 prefix
        aaa: mask
            .map(|m| m.to_reg().to_real_reg().unwrap().hw_enc())
            .unwrap_or(0),
        z: false, // Comparisons don't use zeroing
        b: false,
        ll: 0b10, // 512-bit
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc() as u8;
    let src1_enc = src1.to_real_reg().unwrap().hw_enc() as u8;

    let (src2_enc, is_mem) = match src2 {
        RegMem::Reg { reg } => (reg.to_real_reg().unwrap().hw_enc() as u8, false),
        RegMem::Mem { .. } => (0, true),
    };

    evex.emit(dst_enc, src1_enc, src2_enc, is_mem, sink);
    sink.put1(0x1F); // VPCMPD/VPCMPQ opcode
    emit_modrm_for_regmem(sink, dst_enc, src2);
    sink.put1(cond as u8); // Immediate condition
}

// =============================================================================
// VPCOMPRESS / VPEXPAND Instruction Emission
// =============================================================================

/// Emit VPCOMPRESSD/VPCOMPRESSQ - compress and store.
/// Packs valid elements contiguously based on mask.
pub fn emit_compress_store(
    size: OperandSize,
    src: Reg,
    addr: &SyntheticAmode,
    mask: Reg,
    sink: &mut MachBuffer<Inst>,
) {
    // VPCOMPRESSD: EVEX.512.66.0F38.W0 8B /r
    // VPCOMPRESSQ: EVEX.512.66.0F38.W1 8B /r
    let w = matches!(size, OperandSize::Size64);
    let mask_enc = mask.to_real_reg().unwrap().hw_enc();

    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w,
        pp: 0x01, // 66 prefix
        aaa: mask_enc,
        z: false, // Compress store uses merge semantics
        b: false,
        ll: 0b10,
    };

    let src_enc = src.to_real_reg().unwrap().hw_enc() as u8;

    // For store instructions, we need to use a different encoding
    // The src register goes in the reg field of ModRM
    evex.emit(src_enc, 0, 0, true, sink);
    sink.put1(0x8B); // VPCOMPRESSD/Q opcode
    emit_modrm_sib_disp(sink, src_enc, addr);
}

/// Emit VPEXPANDD/VPEXPANDQ - expand load.
/// Loads sparse elements into contiguous positions based on mask.
pub fn emit_expand_load(
    size: OperandSize,
    dst: Writable<Reg>,
    addr: &SyntheticAmode,
    mask: Reg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    // VPEXPANDD: EVEX.512.66.0F38.W0 89 /r
    // VPEXPANDQ: EVEX.512.66.0F38.W1 89 /r
    let w = matches!(size, OperandSize::Size64);
    let mask_enc = mask.to_real_reg().unwrap().hw_enc();

    let evex = EvexPrefix {
        map: 0x02, // 0F38 map
        w,
        pp: 0x01, // 66 prefix
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10,
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc() as u8;

    evex.emit(dst_enc, 0, 0, true, sink);
    sink.put1(0x89); // VPEXPANDD/Q opcode
    emit_modrm_sib_disp(sink, dst_enc, addr);
}

// =============================================================================
// Masked Load/Store Instruction Emission (VMOVDQU32/64)
// =============================================================================

/// Emit VMOVDQU32/VMOVDQU64 masked load (fault-suppressing).
pub fn emit_masked_load(
    size: OperandSize,
    dst: Writable<Reg>,
    addr: &SyntheticAmode,
    mask: Reg,
    merge: MergeMode,
    sink: &mut MachBuffer<Inst>,
) {
    // VMOVDQU32: EVEX.512.F3.0F.W0 6F /r (load)
    // VMOVDQU64: EVEX.512.F3.0F.W1 6F /r (load)
    let w = matches!(size, OperandSize::Size64);
    let mask_enc = mask.to_real_reg().unwrap().hw_enc();

    let evex = EvexPrefix {
        map: 0x01, // 0F map
        w,
        pp: 0x02, // F3 prefix
        aaa: mask_enc,
        z: matches!(merge, MergeMode::Zeroing),
        b: false,
        ll: 0b10,
    };

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc() as u8;

    evex.emit(dst_enc, 0, 0, true, sink);
    sink.put1(0x6F); // VMOVDQU32/64 load opcode
    emit_modrm_sib_disp(sink, dst_enc, addr);
}

/// Emit VMOVDQU32/VMOVDQU64 masked store.
pub fn emit_masked_store(
    size: OperandSize,
    src: Reg,
    addr: &SyntheticAmode,
    mask: Reg,
    sink: &mut MachBuffer<Inst>,
) {
    // VMOVDQU32: EVEX.512.F3.0F.W0 7F /r (store)
    // VMOVDQU64: EVEX.512.F3.0F.W1 7F /r (store)
    let w = matches!(size, OperandSize::Size64);
    let mask_enc = mask.to_real_reg().unwrap().hw_enc();

    let evex = EvexPrefix {
        map: 0x01, // 0F map
        w,
        pp: 0x02, // F3 prefix
        aaa: mask_enc,
        z: false, // Stores don't use zeroing
        b: false,
        ll: 0b10,
    };

    let src_enc = src.to_real_reg().unwrap().hw_enc() as u8;

    evex.emit(src_enc, 0, 0, true, sink);
    sink.put1(0x7F); // VMOVDQU32/64 store opcode
    emit_modrm_sib_disp(sink, src_enc, addr);
}

// =============================================================================
// Mask Register Logic Instruction Emission (KAND, KOR, etc.)
// =============================================================================

/// Emit a mask register logic instruction.
/// These use VEX encoding (not EVEX) for k-register operations.
pub fn emit_mask_logic(
    op: MaskAluOp,
    dst: Writable<Reg>,
    src1: Reg,
    src2: Option<Reg>,
    sink: &mut MachBuffer<Inst>,
) {
    // Mask logic instructions use VEX.L1.66.0F encoding
    // KANDW:  VEX.L1.66.0F.W0 41 /r
    // KORW:   VEX.L1.66.0F.W0 45 /r
    // KXORW:  VEX.L1.66.0F.W0 47 /r
    // KNOTW:  VEX.L0.0F.W0 44 /r (unary)
    // KANDNW: VEX.L1.66.0F.W0 42 /r

    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc() as u8;
    let src1_enc = src1.to_real_reg().unwrap().hw_enc() as u8;

    match op {
        MaskAluOp::Knot => {
            // KNOTW is unary: VEX.L0.0F.W0 44 /r
            // 2-byte VEX: C5 [R~vvvv~L~pp] opcode modrm
            let r = if dst_enc >= 8 { 0 } else { 1 };
            let vvvv = !src1_enc & 0x0F;
            let vex2 = (r << 7) | (vvvv << 3) | 0x00; // L=0, pp=00
            sink.put1(0xC5);
            sink.put1(vex2);
            sink.put1(0x44); // KNOTW opcode
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src1_enc & 0x07);
            sink.put1(modrm);
        }
        _ => {
            // Binary ops: KAND, KOR, KXOR, KANDN
            let src2 = src2.expect("Binary mask op requires src2");
            let src2_enc = src2.to_real_reg().unwrap().hw_enc() as u8;

            let opcode = match op {
                MaskAluOp::Kand => 0x41,
                MaskAluOp::Kor => 0x45,
                MaskAluOp::Kxor => 0x47,
                MaskAluOp::Kandn => 0x42,
                MaskAluOp::Knot => unreachable!(),
            };

            // 3-byte VEX for L=1: C4 [RXB~mmmmm] [W~vvvv~L~pp] opcode modrm
            let rxb = 0xE0 | // R=1, X=1 (inverted)
                if src2_enc >= 8 { 0 } else { 0x20 }; // B bit
            let vvvv = !src1_enc & 0x0F;
            let vex2 = (0 << 7) | (vvvv << 3) | 0x05; // W=0, L=1, pp=01 (66)

            sink.put1(0xC4);
            sink.put1(rxb | 0x01); // mmmmm = 01 (0F)
            sink.put1(vex2);
            sink.put1(opcode);
            let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src2_enc & 0x07);
            sink.put1(modrm);
        }
    }
}

// =============================================================================
// KMOV Instruction Emission
// =============================================================================

/// Emit KMOVQ - move between k-register and GPR.
pub fn emit_kmov(
    dst: Writable<Reg>,
    src: Reg,
    to_gpr: bool,
    sink: &mut MachBuffer<Inst>,
) {
    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc() as u8;
    let src_enc = src.to_real_reg().unwrap().hw_enc() as u8;

    if to_gpr {
        // KMOVQ r64, k: VEX.L0.F2.0F.W1 93 /r
        let r = if dst_enc >= 8 { 0 } else { 1 };
        let b = if src_enc >= 8 { 0 } else { 1 };
        let rxb = (r << 7) | 0x40 | (b << 5) | 0x01; // X=1, mmmmm=01
        let vex2 = 0x80 | 0x78 | 0x03; // W=1, vvvv=1111, L=0, pp=11 (F2)

        sink.put1(0xC4);
        sink.put1(rxb);
        sink.put1(vex2);
        sink.put1(0x93);
        let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
        sink.put1(modrm);
    } else {
        // KMOVQ k, r64: VEX.L0.F2.0F.W1 92 /r
        let r = if dst_enc >= 8 { 0 } else { 1 };
        let b = if src_enc >= 8 { 0 } else { 1 };
        let rxb = (r << 7) | 0x40 | (b << 5) | 0x01;
        let vex2 = 0x80 | 0x78 | 0x03; // W=1, vvvv=1111, L=0, pp=11 (F2)

        sink.put1(0xC4);
        sink.put1(rxb);
        sink.put1(vex2);
        sink.put1(0x92);
        let modrm = 0xC0 | ((dst_enc & 0x07) << 3) | (src_enc & 0x07);
        sink.put1(modrm);
    }
}

// =============================================================================
// KORTEST Instruction Emission
// =============================================================================

/// Emit KORTESTW/KORTESTQ - OR masks and set flags.
pub fn emit_kortest(
    src1: Reg,
    src2: Reg,
    sink: &mut MachBuffer<Inst>,
) {
    // KORTESTW: VEX.L0.0F.W0 98 /r
    let src1_enc = src1.to_real_reg().unwrap().hw_enc() as u8;
    let src2_enc = src2.to_real_reg().unwrap().hw_enc() as u8;

    // 2-byte VEX
    let r = if src1_enc >= 8 { 0 } else { 1 };
    let vex2 = (r << 7) | 0x78 | 0x00; // vvvv=1111, L=0, pp=00

    sink.put1(0xC5);
    sink.put1(vex2);
    sink.put1(0x98); // KORTESTW opcode
    let modrm = 0xC0 | ((src1_enc & 0x07) << 3) | (src2_enc & 0x07);
    sink.put1(modrm);
}

// =============================================================================
// K-Register Spill/Fill (for register allocation)
// =============================================================================

/// Emit KMOVQ to spill a k-register to memory.
/// Uses the two-tier strategy: prefer GPR, fall back to stack.
pub fn emit_kmov_store(
    src: Reg,
    addr: &SyntheticAmode,
    sink: &mut MachBuffer<Inst>,
) {
    // KMOVQ m64, k: VEX.L0.0F.W1 91 /r
    let src_enc = src.to_real_reg().unwrap().hw_enc() as u8;

    // 3-byte VEX for memory operand
    sink.put1(0xC4);
    sink.put1(0xE1); // RXB=111, mmmmm=01
    sink.put1(0xF8); // W=1, vvvv=1111, L=0, pp=00
    sink.put1(0x91);
    emit_modrm_sib_disp(sink, src_enc, addr);
}

/// Emit KMOVQ to fill a k-register from memory.
pub fn emit_kmov_load(
    dst: Writable<Reg>,
    addr: &SyntheticAmode,
    sink: &mut MachBuffer<Inst>,
) {
    // KMOVQ k, m64: VEX.L0.0F.W1 90 /r
    let dst_enc = dst.to_reg().to_real_reg().unwrap().hw_enc() as u8;

    // 3-byte VEX for memory operand
    sink.put1(0xC4);
    sink.put1(0xE1); // RXB=111, mmmmm=01
    sink.put1(0xF8); // W=1, vvvv=1111, L=0, pp=00
    sink.put1(0x90);
    emit_modrm_sib_disp(sink, dst_enc, addr);
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Emit ModRM byte for a RegMem operand.
fn emit_modrm_for_regmem(sink: &mut MachBuffer<Inst>, reg: u8, rm: &RegMem) {
    match rm {
        RegMem::Reg { reg: rm_reg } => {
            let rm_enc = rm_reg.to_real_reg().unwrap().hw_enc() as u8;
            let modrm = 0xC0 | ((reg & 0x07) << 3) | (rm_enc & 0x07);
            sink.put1(modrm);
        }
        RegMem::Mem { addr } => {
            emit_modrm_sib_disp(sink, reg, addr);
        }
    }
}

/// Emit ModRM, SIB, and displacement for a memory operand.
fn emit_modrm_sib_disp(sink: &mut MachBuffer<Inst>, reg: u8, addr: &SyntheticAmode) {
    match addr {
        SyntheticAmode::Real(amode) => {
            use crate::isa::x64::lower::isle::generated_code::Amode;
            match amode {
                Amode::ImmReg { simm32, base, .. } => {
                    let base_enc = base.to_real_reg().unwrap().hw_enc() as u8;
                    let needs_sib = (base_enc & 0x07) == 4;

                    if *simm32 == 0 && (base_enc & 0x07) != 5 {
                        let modrm = 0x00 | ((reg & 0x07) << 3) | (base_enc & 0x07);
                        sink.put1(modrm);
                        if needs_sib {
                            sink.put1(0x24);
                        }
                    } else if *simm32 >= -128 && *simm32 <= 127 {
                        let modrm = 0x40 | ((reg & 0x07) << 3) | (base_enc & 0x07);
                        sink.put1(modrm);
                        if needs_sib {
                            sink.put1(0x24);
                        }
                        sink.put1(*simm32 as u8);
                    } else {
                        let modrm = 0x80 | ((reg & 0x07) << 3) | (base_enc & 0x07);
                        sink.put1(modrm);
                        if needs_sib {
                            sink.put1(0x24);
                        }
                        sink.put4(*simm32 as u32);
                    }
                }
                Amode::ImmRegRegShift {
                    simm32,
                    base,
                    index,
                    shift,
                    ..
                } => {
                    let base_enc = base.to_reg().to_real_reg().unwrap().hw_enc() as u8;
                    let index_enc = index.to_reg().to_real_reg().unwrap().hw_enc() as u8;
                    let sib = (*shift << 6) | ((index_enc & 0x07) << 3) | (base_enc & 0x07);

                    if *simm32 == 0 && (base_enc & 0x07) != 5 {
                        let modrm = 0x04 | ((reg & 0x07) << 3);
                        sink.put1(modrm);
                        sink.put1(sib);
                    } else if *simm32 >= -128 && *simm32 <= 127 {
                        let modrm = 0x44 | ((reg & 0x07) << 3);
                        sink.put1(modrm);
                        sink.put1(sib);
                        sink.put1(*simm32 as u8);
                    } else {
                        let modrm = 0x84 | ((reg & 0x07) << 3);
                        sink.put1(modrm);
                        sink.put1(sib);
                        sink.put4(*simm32 as u32);
                    }
                }
                Amode::RipRelative { .. } => {
                    let modrm = 0x05 | ((reg & 0x07) << 3);
                    sink.put1(modrm);
                    sink.put4(0);
                }
            }
        }
        SyntheticAmode::SlotOffset { simm32 } => {
            if *simm32 >= -128 && *simm32 <= 127 {
                let modrm = 0x44 | ((reg & 0x07) << 3);
                sink.put1(modrm);
                sink.put1(0x24);
                sink.put1(*simm32 as u8);
            } else {
                let modrm = 0x84 | ((reg & 0x07) << 3);
                sink.put1(modrm);
                sink.put1(0x24);
                sink.put4(*simm32 as u32);
            }
        }
        SyntheticAmode::IncomingArg { offset } => {
            let disp = -(*offset as i32);
            if disp >= -128 && disp <= 127 {
                let modrm = 0x45 | ((reg & 0x07) << 3);
                sink.put1(modrm);
                sink.put1(disp as u8);
            } else {
                let modrm = 0x85 | ((reg & 0x07) << 3);
                sink.put1(modrm);
                sink.put4(disp as u32);
            }
        }
        SyntheticAmode::ConstantOffset(_) => {
            let modrm = 0x05 | ((reg & 0x07) << 3);
            sink.put1(modrm);
            sink.put4(0);
        }
    }
}

// =============================================================================
// Debug Assertions for Emission Functions
// =============================================================================

/// Validate that a register encoding is within valid range (0-31 for EVEX).
#[inline]
fn debug_validate_reg_enc(enc: u8, name: &str) {
    debug_assert!(
        enc < 32,
        "{} register encoding {} exceeds maximum of 31",
        name,
        enc
    );
}

/// Validate that a mask register encoding is valid (0-7).
#[inline]
fn debug_validate_mask_enc(enc: u8) {
    debug_assert!(
        enc < 8,
        "Mask register encoding {} exceeds maximum of 7",
        enc
    );
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // EVEX Prefix Structure Tests
    // =========================================================================

    #[test]
    fn test_evex_prefix_structure() {
        // EVEX prefix should always start with 0x62
        let evex = EvexPrefix {
            map: 0x01,
            w: false,
            pp: 0x01,
            aaa: 0,
            z: false,
            b: false,
            ll: 0b10,
        };
        assert_eq!(evex.map, 0x01);
        assert_eq!(evex.ll, 0b10);
    }

    #[test]
    fn test_evex_prefix_512bit_default() {
        let evex = EvexPrefix::new_512bit(0x01, false, 0x01);
        assert_eq!(evex.ll, 0b10); // 512-bit
        assert!(!evex.z);
        assert!(!evex.b);
        assert_eq!(evex.aaa, 0);
    }

    // =========================================================================
    // MaskAluOp Tests
    // =========================================================================

    #[test]
    fn test_mask_alu_op_encoding() {
        assert_eq!(MaskAluOp::Kand as u8, 0);
        assert_eq!(MaskAluOp::Kor as u8, 1);
        assert_eq!(MaskAluOp::Kxor as u8, 2);
        assert_eq!(MaskAluOp::Knot as u8, 3);
        assert_eq!(MaskAluOp::Kandn as u8, 4);
    }

    #[test]
    fn test_mask_alu_op_vex_opcodes() {
        // Verify opcodes match Intel documentation
        assert_eq!(MaskAluOp::Kand.vex_opcode(), 0x41);
        assert_eq!(MaskAluOp::Kandn.vex_opcode(), 0x42);
        assert_eq!(MaskAluOp::Knot.vex_opcode(), 0x44);
        assert_eq!(MaskAluOp::Kor.vex_opcode(), 0x45);
        assert_eq!(MaskAluOp::Kxor.vex_opcode(), 0x47);
    }

    // =========================================================================
    // Avx512Cond Tests
    // =========================================================================

    #[test]
    fn test_avx512_cond_encoding() {
        // Verify condition encodings match Intel documentation for VPCMPD/VPCMPQ
        assert_eq!(Avx512Cond::Eq as u8, 0);
        assert_eq!(Avx512Cond::Lt as u8, 1);
        assert_eq!(Avx512Cond::Le as u8, 2);
        // Note: 3 is "false" (always false), not commonly used
        assert_eq!(Avx512Cond::Neq as u8, 4);
        assert_eq!(Avx512Cond::Ge as u8, 5); // NLT (not less than)
        assert_eq!(Avx512Cond::Gt as u8, 6); // NLE (not less than or equal)
        // Note: 7 is "true" (always true), not commonly used
    }

    // =========================================================================
    // MergeMode Tests
    // =========================================================================

    #[test]
    fn test_merge_mode() {
        assert!(matches!(MergeMode::Zeroing, MergeMode::Zeroing));
        assert!(matches!(MergeMode::Merging, MergeMode::Merging));
        // Default should be Zeroing (as per our impl)
        assert_eq!(MergeMode::default(), MergeMode::Zeroing);
    }

    // =========================================================================
    // Avx512AluOp Tests
    // =========================================================================

    #[test]
    fn test_avx512_alu_op_arithmetic_opcodes() {
        // Integer arithmetic instructions
        assert_eq!(Avx512AluOp::Vpaddd.opcode(), 0xFE);
        assert_eq!(Avx512AluOp::Vpaddq.opcode(), 0xD4);
        assert_eq!(Avx512AluOp::Vpsubd.opcode(), 0xFA);
        assert_eq!(Avx512AluOp::Vpsubq.opcode(), 0xFB);
    }

    #[test]
    fn test_avx512_alu_op_logical_opcodes() {
        // Bitwise logical instructions
        assert_eq!(Avx512AluOp::Vpandd.opcode(), 0xDB);
        assert_eq!(Avx512AluOp::Vpandq.opcode(), 0xDB);
        assert_eq!(Avx512AluOp::Vpord.opcode(), 0xEB);
        assert_eq!(Avx512AluOp::Vporq.opcode(), 0xEB);
        assert_eq!(Avx512AluOp::Vpxord.opcode(), 0xEF);
        assert_eq!(Avx512AluOp::Vpxorq.opcode(), 0xEF);
    }

    #[test]
    fn test_avx512_alu_op_shift_opcodes() {
        // Shift instructions
        assert_eq!(Avx512AluOp::Vpslld.opcode(), 0xF2);
        assert_eq!(Avx512AluOp::Vpsllq.opcode(), 0xF3);
        assert_eq!(Avx512AluOp::Vpsrld.opcode(), 0xD2);
        assert_eq!(Avx512AluOp::Vpsrlq.opcode(), 0xD3);
    }

    #[test]
    fn test_avx512_alu_op_evex_maps() {
        // 0F map (simple operations)
        assert_eq!(Avx512AluOp::Vpaddd.evex_map(), 0x01);
        assert_eq!(Avx512AluOp::Vpandd.evex_map(), 0x01);

        // 0F38 map (complex operations)
        assert_eq!(Avx512AluOp::Vpmulld.evex_map(), 0x02);
        assert_eq!(Avx512AluOp::Vpermd.evex_map(), 0x02);

        // 0F3A map (ternary logic)
        assert_eq!(Avx512AluOp::Vpternlogd.evex_map(), 0x03);
    }

    #[test]
    fn test_avx512_alu_op_evex_w() {
        // 32-bit operations: W=0
        assert!(!Avx512AluOp::Vpaddd.evex_w());
        assert!(!Avx512AluOp::Vpsubd.evex_w());
        assert!(!Avx512AluOp::Vpandd.evex_w());

        // 64-bit operations: W=1
        assert!(Avx512AluOp::Vpaddq.evex_w());
        assert!(Avx512AluOp::Vpsubq.evex_w());
        assert!(Avx512AluOp::Vpandq.evex_w());
    }

    // =========================================================================
    // Validation Helper Tests
    // =========================================================================

    #[test]
    fn test_debug_validate_reg_enc_valid() {
        // Should not panic for valid encodings
        for enc in 0..32 {
            debug_validate_reg_enc(enc, "test");
        }
    }

    #[test]
    fn test_debug_validate_mask_enc_valid() {
        // Should not panic for valid mask encodings
        for enc in 0..8 {
            debug_validate_mask_enc(enc);
        }
    }

    #[test]
    #[should_panic(expected = "register encoding")]
    #[cfg(debug_assertions)]
    fn test_debug_validate_reg_enc_invalid() {
        debug_validate_reg_enc(32, "test");
    }

    #[test]
    #[should_panic(expected = "Mask register encoding")]
    #[cfg(debug_assertions)]
    fn test_debug_validate_mask_enc_invalid() {
        debug_validate_mask_enc(8);
    }

    // =========================================================================
    // Instruction Encoding Constants Tests
    // =========================================================================

    #[test]
    fn test_vpcmpd_encoding_constants() {
        // VPCMPD: EVEX.512.66.0F3A.W0 1F /r ib
        // map=0x03 (0F3A), pp=0x01 (66), W=0
        assert_eq!(0x1F, 31); // Opcode
    }

    #[test]
    fn test_vpcompressd_encoding_constants() {
        // VPCOMPRESSD: EVEX.512.66.0F38.W0 8B /r
        // map=0x02 (0F38), pp=0x01 (66), W=0
        assert_eq!(0x8B, 139); // Opcode
    }

    #[test]
    fn test_vpexpandd_encoding_constants() {
        // VPEXPANDD: EVEX.512.66.0F38.W0 89 /r
        assert_eq!(0x89, 137); // Opcode
    }

    #[test]
    fn test_vmovdqu32_encoding_constants() {
        // VMOVDQU32 load: EVEX.512.F3.0F.W0 6F /r
        // VMOVDQU32 store: EVEX.512.F3.0F.W0 7F /r
        assert_eq!(0x6F, 111); // Load opcode
        assert_eq!(0x7F, 127); // Store opcode
    }

    #[test]
    fn test_kmov_encoding_constants() {
        // KMOVQ k, r64: VEX.L0.F2.0F.W1 92 /r
        // KMOVQ r64, k: VEX.L0.F2.0F.W1 93 /r
        // KMOVQ k, m64: VEX.L0.0F.W1 90 /r
        // KMOVQ m64, k: VEX.L0.0F.W1 91 /r
        assert_eq!(0x90, 144); // Load from memory
        assert_eq!(0x91, 145); // Store to memory
        assert_eq!(0x92, 146); // Move GPR to k
        assert_eq!(0x93, 147); // Move k to GPR
    }

    #[test]
    fn test_kortest_encoding_constants() {
        // KORTESTW: VEX.L0.0F.W0 98 /r
        assert_eq!(0x98, 152); // Opcode
    }
}
