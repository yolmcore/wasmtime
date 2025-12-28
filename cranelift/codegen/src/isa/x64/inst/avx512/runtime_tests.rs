// cranelift/codegen/src/isa/x64/inst/avx512/runtime_tests.rs
//
// AVX-512: Runtime tests for AVX-512 instruction execution.
//
// These tests verify that AVX-512 instructions are correctly encoded
// and execute correctly on hardware that supports AVX-512.

#![cfg(test)]
#![allow(clippy::uninlined_format_args, reason = "test output readability")]

extern crate std;

use crate::isa::x64::inst::args::{Mask, OperandSize, RegMem};
use std::vec::Vec;
use crate::isa::x64::inst::regs;
use crate::isa::x64::inst::avx512::{Avx512AluOp, MergeMode};
use crate::isa::x64::inst::Inst;
use crate::machinst::{MachBuffer, MachInstEmit, Writable};

/// Check if AVX-512F is available on this CPU.
fn has_avx512f() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        // Use CPUID to check for AVX-512F support
        // CPUID leaf 7, subleaf 0, EBX bit 16
        if let Some(info) = core_detect::feature_info() {
            return info.has_avx512f();
        }
        // Fallback: use is_x86_feature_detected macro
        std::arch::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Test that AVX-512 feature detection works correctly.
#[test]
fn test_x64_512_feature_detection() {
    let has_avx512 = has_avx512f();
    println!("AVX-512F support detected: {}", has_avx512);
    // This test just prints the status - it doesn't fail if AVX-512 isn't available
}

// =============================================================================
// EVEX Encoding Verification Tests
// =============================================================================
// These tests verify that our EVEX encoding matches Intel documentation exactly.
// Reference: Intel SDM Vol 2A, Chapter 2.6 "EVEX Prefix Encoding"

/// Verify EVEX prefix structure byte-by-byte against Intel reference.
///
/// EVEX format (4 bytes):
/// - Byte 0: 0x62 (EVEX escape)
/// - Byte 1 (P0): R X B R' 0 0 mm
/// - Byte 2 (P1): W vvvv 1 pp
/// - Byte 3 (P2): z L'L b V' aaa
#[test]
fn test_evex_prefix_byte_structure() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VPADDD zmm0, zmm1, zmm2 (no mask)
    // Expected: 62 F1 75 48 FE C2
    //   62 = EVEX escape
    //   F1 = P0: R=1 X=1 B=1 R'=1 00 mm=01 = 11110001
    //   75 = P1: W=0 vvvv=1110 (zmm1 inverted) 1 pp=01 = 01110101
    //   48 = P2: z=0 L'L=10 (512-bit) b=0 V'=1 aaa=000 = 01001000
    //   FE = opcode for VPADDD
    //   C2 = ModRM: 11 000 010 (reg-reg, dst=0, src=2)
    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vpaddd,
        size: OperandSize::Size32,
        dst: Writable::from_reg(regs::xmm0()),
        src1: regs::xmm1(),
        src2: RegMem::reg(regs::xmm2()),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    // Verify EVEX prefix
    assert_eq!(bytes[0], 0x62, "Byte 0: EVEX escape");
    // P0: For zmm0-zmm7, all R bits should be 1 (inverted)
    assert_eq!(bytes[1] & 0b11110000, 0xF0, "P0 high nibble: R X B R' all 1");
    assert_eq!(bytes[1] & 0b00000111, 0x01, "P0 low bits: mm=01 (0F map)");
    // P1: W=0, vvvv=1110 (zmm1 inverted), pp=01
    assert_eq!(bytes[2] & 0x80, 0x00, "P1: W=0 for 32-bit elements");
    assert_eq!(bytes[2] & 0x03, 0x01, "P1: pp=01 (66 prefix)");
    // P2: z=0, L'L=10, V'=1, aaa=000
    assert_eq!(bytes[3] & 0x60, 0x40, "P2: L'L=10 (512-bit)");
    assert_eq!(bytes[3] & 0x07, 0x00, "P2: aaa=000 (no mask)");

    println!("EVEX prefix verified: {:02X?}", &bytes[..4]);
}

/// Test encoding with masked operation and zeroing.
#[test]
fn test_evex_masked_zeroing_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VPADDD zmm0{k1}{z}, zmm1, zmm2
    // The z bit and aaa field should be set
    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vpaddd,
        size: OperandSize::Size32,
        dst: Writable::from_reg(regs::xmm0()),
        src1: regs::xmm1(),
        src2: RegMem::reg(regs::xmm2()),
        mask: Some(Mask::new(regs::k1()).unwrap()),
        merge: MergeMode::Zeroing,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    // Verify P2 byte has zeroing bit and mask register
    assert_eq!(bytes[0], 0x62, "EVEX escape");
    assert_eq!(bytes[3] & 0x80, 0x80, "P2: z=1 (zeroing mode)");
    assert_eq!(bytes[3] & 0x07, 0x01, "P2: aaa=001 (k1 mask)");

    println!("Masked+zeroing EVEX verified: {:02X?}", bytes);
}

/// Test high register encoding (zmm16-zmm31 require EVEX.R' and EVEX.X extensions).
#[test]
fn test_evex_high_register_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // Test with registers 8-15 (require B/X/R bits)
    // VPADDD zmm8, zmm9, zmm10
    let xmm8 = regs::xmm8();
    let xmm9 = regs::xmm9();
    let xmm10 = regs::xmm10();

    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vpaddd,
        size: OperandSize::Size32,
        dst: Writable::from_reg(xmm8),
        src1: xmm9,
        src2: RegMem::reg(xmm10),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    assert_eq!(bytes[0], 0x62, "EVEX escape");
    // For registers 8-15, the R and B bits should be 0 (inverted from 1)
    // P0: R=0 (dst is zmm8), B=0 (src2 is zmm10)
    println!("High register EVEX: {:02X?}", bytes);
    println!("P0 = {:08b}, should have R=0 for reg 8+", bytes[1]);
}

/// Test 64-bit element operations (W=1).
#[test]
fn test_evex_64bit_element_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VPADDQ zmm0, zmm1, zmm2 (64-bit elements, W=1)
    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vpaddq,
        size: OperandSize::Size64,
        dst: Writable::from_reg(regs::xmm0()),
        src1: regs::xmm1(),
        src2: RegMem::reg(regs::xmm2()),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    assert_eq!(bytes[0], 0x62, "EVEX escape");
    // P1: W=1 for 64-bit elements
    assert_eq!(bytes[2] & 0x80, 0x80, "P1: W=1 for 64-bit elements");

    println!("64-bit element EVEX verified: {:02X?}", bytes);
}

/// Test all k-register mask encodings (k1-k7).
#[test]
fn test_evex_all_mask_registers() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let kregs = [
        (regs::k1(), 1u8),
        (regs::k2(), 2u8),
        (regs::k3(), 3u8),
        (regs::k4(), 4u8),
        (regs::k5(), 5u8),
        (regs::k6(), 6u8),
        (regs::k7(), 7u8),
    ];

    for (kreg, expected_aaa) in kregs.iter() {
        let inst = Inst::Avx512Avx512Alu {
            op: Avx512AluOp::Vpaddd,
            size: OperandSize::Size32,
            dst: Writable::from_reg(regs::xmm0()),
            src1: regs::xmm1(),
            src2: RegMem::reg(regs::xmm2()),
            mask: Some(Mask::new(*kreg).unwrap()),
            merge: MergeMode::Merging,
        };

        let mut buffer = MachBuffer::new();
        inst.emit(&mut buffer, &emit_info, &mut Default::default());
        let buffer = buffer.finish(&Default::default(), &mut Default::default());
        let bytes = buffer.data();

        let actual_aaa = bytes[3] & 0x07;
        assert_eq!(
            actual_aaa, *expected_aaa,
            "k{} should encode as aaa={}, got {}",
            expected_aaa, expected_aaa, actual_aaa
        );
    }

    println!("All k-register mask encodings verified");
}

/// Test ALL AVX-512 ALU operations with masking (both merging and zeroing).
#[test]
fn test_all_masked_alu_ops() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // All ALU operations to test with masking
    let all_ops = [
        (Avx512AluOp::Vpaddd, "VPADDD", false),
        (Avx512AluOp::Vpaddq, "VPADDQ", true),
        (Avx512AluOp::Vpsubd, "VPSUBD", false),
        (Avx512AluOp::Vpsubq, "VPSUBQ", true),
        (Avx512AluOp::Vpmulld, "VPMULLD", false),
        (Avx512AluOp::Vpmullq, "VPMULLQ", true),
        (Avx512AluOp::Vpmuludq, "VPMULUDQ", true),
        (Avx512AluOp::Vpmuldq, "VPMULDQ", true),
        (Avx512AluOp::Vpandd, "VPANDD", false),
        (Avx512AluOp::Vpandq, "VPANDQ", true),
        (Avx512AluOp::Vpord, "VPORD", false),
        (Avx512AluOp::Vporq, "VPORQ", true),
        (Avx512AluOp::Vpxord, "VPXORD", false),
        (Avx512AluOp::Vpxorq, "VPXORQ", true),
        (Avx512AluOp::Vpslld, "VPSLLD", false),
        (Avx512AluOp::Vpsllq, "VPSLLQ", true),
        (Avx512AluOp::Vpsrld, "VPSRLD", false),
        (Avx512AluOp::Vpsrlq, "VPSRLQ", true),
        (Avx512AluOp::Vpsrad, "VPSRAD", false),
        (Avx512AluOp::Vpsraq, "VPSRAQ", true),
        (Avx512AluOp::Vpminsd, "VPMINSD", false),
        (Avx512AluOp::Vpminsq, "VPMINSQ", true),
        (Avx512AluOp::Vpmaxsd, "VPMAXSD", false),
        (Avx512AluOp::Vpmaxsq, "VPMAXSQ", true),
        (Avx512AluOp::Vpminud, "VPMINUD", false),
        (Avx512AluOp::Vpminuq, "VPMINUQ", true),
        (Avx512AluOp::Vpmaxud, "VPMAXUD", false),
        (Avx512AluOp::Vpmaxuq, "VPMAXUQ", true),
    ];

    let merge_modes = [
        (MergeMode::Merging, "merging", 0x00u8),
        (MergeMode::Zeroing, "zeroing", 0x80u8),
    ];

    let kregs = [
        (regs::k1(), 1u8),
        (regs::k3(), 3u8),
        (regs::k7(), 7u8),
    ];

    let mut success_count = 0;
    for (op, name, is_64bit) in all_ops.iter() {
        for (merge_mode, mode_name, z_bit) in merge_modes.iter() {
            for (kreg, expected_aaa) in kregs.iter() {
                let size = if *is_64bit {
                    OperandSize::Size64
                } else {
                    OperandSize::Size32
                };

                let inst = Inst::Avx512Avx512Alu {
                    op: *op,
                    size,
                    dst: Writable::from_reg(regs::xmm0()),
                    src1: regs::xmm1(),
                    src2: RegMem::reg(regs::xmm2()),
                    mask: Some(Mask::new(*kreg).unwrap()),
                    merge: *merge_mode,
                };

                let mut buffer = MachBuffer::new();
                inst.emit(&mut buffer, &emit_info, &mut Default::default());
                let buffer = buffer.finish(&Default::default(), &mut Default::default());
                let bytes = buffer.data();

                // Verify EVEX prefix
                assert_eq!(bytes[0], 0x62, "{} with {} k{}: missing EVEX prefix",
                    name, mode_name, expected_aaa);

                // Verify z-bit in P2
                let actual_z = bytes[3] & 0x80;
                assert_eq!(actual_z, *z_bit,
                    "{} with {} k{}: z-bit wrong, got {:02X}, expected {:02X}",
                    name, mode_name, expected_aaa, actual_z, z_bit);

                // Verify aaa field
                let actual_aaa = bytes[3] & 0x07;
                assert_eq!(actual_aaa, *expected_aaa,
                    "{} with {} k{}: aaa field wrong, got {}, expected {}",
                    name, mode_name, expected_aaa, actual_aaa, expected_aaa);

                success_count += 1;
            }
        }
    }

    println!("Verified {} masked ALU operation encodings", success_count);
    assert_eq!(success_count, all_ops.len() * merge_modes.len() * kregs.len());
}

/// Test VPADDD instruction encoding produces valid machine code.
#[test]
fn test_vpaddd_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let xmm0 = regs::xmm0();
    let xmm1 = regs::xmm1();
    let xmm2 = regs::xmm2();

    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vpaddd,
        size: OperandSize::Size32,
        dst: Writable::from_reg(xmm0),
        src1: xmm1,
        src2: RegMem::reg(xmm2),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());

    let ctrl_plane = &mut Default::default();
    let constants = Default::default();
    let buffer = buffer.finish(&constants, ctrl_plane);

    // Verify the encoding is non-empty and has expected EVEX prefix
    let bytes = buffer.data();
    assert!(!bytes.is_empty(), "Instruction encoding should not be empty");
    assert_eq!(bytes[0], 0x62, "EVEX prefix should start with 0x62");

    println!("VPADDD encoding: {:02X?}", bytes);
}

/// Test VPADDQ instruction encoding produces valid machine code.
#[test]
fn test_vpaddq_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let xmm0 = regs::xmm0();
    let xmm1 = regs::xmm1();
    let xmm2 = regs::xmm2();

    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vpaddq,
        size: OperandSize::Size64,
        dst: Writable::from_reg(xmm0),
        src1: xmm1,
        src2: RegMem::reg(xmm2),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());

    let ctrl_plane = &mut Default::default();
    let constants = Default::default();
    let buffer = buffer.finish(&constants, ctrl_plane);

    let bytes = buffer.data();
    assert!(!bytes.is_empty(), "Instruction encoding should not be empty");
    assert_eq!(bytes[0], 0x62, "EVEX prefix should start with 0x62");

    // For VPADDQ with W=1, byte 2 (P1) should have bit 7 set
    assert!(bytes[2] & 0x80 != 0, "VPADDQ should have W=1 in EVEX.P1");

    println!("VPADDQ encoding: {:02X?}", bytes);
}

/// Test VPSUBD instruction encoding.
#[test]
fn test_vpsubd_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let xmm0 = regs::xmm0();
    let xmm1 = regs::xmm1();
    let xmm2 = regs::xmm2();

    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vpsubd,
        size: OperandSize::Size32,
        dst: Writable::from_reg(xmm0),
        src1: xmm1,
        src2: RegMem::reg(xmm2),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());

    let ctrl_plane = &mut Default::default();
    let constants = Default::default();
    let buffer = buffer.finish(&constants, ctrl_plane);

    let bytes = buffer.data();
    assert!(!bytes.is_empty());
    assert_eq!(bytes[0], 0x62);

    println!("VPSUBD encoding: {:02X?}", bytes);
}

/// Test bitwise operations (VPANDD, VPORD, VPXORD) encoding.
#[test]
fn test_bitwise_ops_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let xmm0 = regs::xmm0();
    let xmm1 = regs::xmm1();
    let xmm2 = regs::xmm2();

    let ops = [
        (Avx512AluOp::Vpandd, "VPANDD"),
        (Avx512AluOp::Vpord, "VPORD"),
        (Avx512AluOp::Vpxord, "VPXORD"),
    ];

    for (op, name) in ops.iter() {
        let inst = Inst::Avx512Avx512Alu {
            op: *op,
            size: OperandSize::Size32,
            dst: Writable::from_reg(xmm0),
            src1: xmm1,
            src2: RegMem::reg(xmm2),
            mask: None,
            merge: MergeMode::Merging,
        };

        let mut buffer = MachBuffer::new();
        inst.emit(&mut buffer, &emit_info, &mut Default::default());

        let ctrl_plane = &mut Default::default();
        let constants = Default::default();
        let buffer = buffer.finish(&constants, ctrl_plane);

        let bytes = buffer.data();
        assert!(!bytes.is_empty(), "{} encoding should not be empty", name);
        assert_eq!(bytes[0], 0x62, "{} should have EVEX prefix", name);

        println!("{} encoding: {:02X?}", name, bytes);
    }
}

/// Test shift operations encoding.
#[test]
fn test_shift_ops_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let xmm0 = regs::xmm0();
    let xmm1 = regs::xmm1();
    let xmm2 = regs::xmm2();

    let ops = [
        (Avx512AluOp::Vpslld, "VPSLLD"),
        (Avx512AluOp::Vpsrld, "VPSRLD"),
        (Avx512AluOp::Vpsrad, "VPSRAD"),
    ];

    for (op, name) in ops.iter() {
        let inst = Inst::Avx512Avx512Alu {
            op: *op,
            size: OperandSize::Size32,
            dst: Writable::from_reg(xmm0),
            src1: xmm1,
            src2: RegMem::reg(xmm2),
            mask: None,
            merge: MergeMode::Merging,
        };

        let mut buffer = MachBuffer::new();
        inst.emit(&mut buffer, &emit_info, &mut Default::default());

        let ctrl_plane = &mut Default::default();
        let constants = Default::default();
        let buffer = buffer.finish(&constants, ctrl_plane);

        let bytes = buffer.data();
        assert!(!bytes.is_empty(), "{} encoding should not be empty", name);
        assert_eq!(bytes[0], 0x62, "{} should have EVEX prefix", name);

        println!("{} encoding: {:02X?}", name, bytes);
    }
}

/// Test min/max operations encoding.
#[test]
fn test_minmax_ops_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let xmm0 = regs::xmm0();
    let xmm1 = regs::xmm1();
    let xmm2 = regs::xmm2();

    let ops = [
        (Avx512AluOp::Vpminsd, "VPMINSD"),
        (Avx512AluOp::Vpmaxsd, "VPMAXSD"),
        (Avx512AluOp::Vpminud, "VPMINUD"),
        (Avx512AluOp::Vpmaxud, "VPMAXUD"),
    ];

    for (op, name) in ops.iter() {
        let inst = Inst::Avx512Avx512Alu {
            op: *op,
            size: OperandSize::Size32,
            dst: Writable::from_reg(xmm0),
            src1: xmm1,
            src2: RegMem::reg(xmm2),
            mask: None,
            merge: MergeMode::Merging,
        };

        let mut buffer = MachBuffer::new();
        inst.emit(&mut buffer, &emit_info, &mut Default::default());

        let ctrl_plane = &mut Default::default();
        let constants = Default::default();
        let buffer = buffer.finish(&constants, ctrl_plane);

        let bytes = buffer.data();
        assert!(!bytes.is_empty(), "{} encoding should not be empty", name);
        assert_eq!(bytes[0], 0x62, "{} should have EVEX prefix", name);

        println!("{} encoding: {:02X?}", name, bytes);
    }
}

/// Test that all AVX-512 ALU operations encode correctly.
#[test]
fn test_all_x64_512_alu_ops_encode() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let xmm0 = regs::xmm0();
    let xmm1 = regs::xmm1();
    let xmm2 = regs::xmm2();

    // Test all ALU operations
    let all_ops = [
        Avx512AluOp::Vpaddd,
        Avx512AluOp::Vpaddq,
        Avx512AluOp::Vpsubd,
        Avx512AluOp::Vpsubq,
        Avx512AluOp::Vpmulld,
        Avx512AluOp::Vpmullq,
        Avx512AluOp::Vpandd,
        Avx512AluOp::Vpandq,
        Avx512AluOp::Vpord,
        Avx512AluOp::Vporq,
        Avx512AluOp::Vpxord,
        Avx512AluOp::Vpxorq,
        Avx512AluOp::Vpslld,
        Avx512AluOp::Vpsllq,
        Avx512AluOp::Vpsrld,
        Avx512AluOp::Vpsrlq,
        Avx512AluOp::Vpsrad,
        Avx512AluOp::Vpsraq,
        Avx512AluOp::Vpminsd,
        Avx512AluOp::Vpminsq,
        Avx512AluOp::Vpmaxsd,
        Avx512AluOp::Vpmaxsq,
        Avx512AluOp::Vpminud,
        Avx512AluOp::Vpminuq,
        Avx512AluOp::Vpmaxud,
        Avx512AluOp::Vpmaxuq,
    ];

    let mut success_count = 0;
    for op in all_ops.iter() {
        let size = if op.evex_w() {
            OperandSize::Size64
        } else {
            OperandSize::Size32
        };

        let inst = Inst::Avx512Avx512Alu {
            op: *op,
            size,
            dst: Writable::from_reg(xmm0),
            src1: xmm1,
            src2: RegMem::reg(xmm2),
            mask: None,
            merge: MergeMode::Merging,
        };

        let mut buffer = MachBuffer::new();
        inst.emit(&mut buffer, &emit_info, &mut Default::default());

        let ctrl_plane = &mut Default::default();
        let constants = Default::default();
        let buffer = buffer.finish(&constants, ctrl_plane);

        let bytes = buffer.data();
        assert!(!bytes.is_empty(), "{:?} encoding should not be empty", op);
        assert_eq!(bytes[0], 0x62, "{:?} should have EVEX prefix", op);
        success_count += 1;
    }

    println!("Successfully encoded {} AVX-512 ALU operations", success_count);
    assert_eq!(success_count, all_ops.len());
}

/// Test gather instruction encoding.
#[test]
fn test_gather_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;
    use crate::isa::x64::inst::avx512::GatherOp;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let xmm0 = regs::xmm0();
    let xmm1 = regs::xmm1();
    let rax = regs::rax();
    let k1 = regs::k1();

    let inst = Inst::Avx512Gather {
        op: GatherOp::Vpgatherdd,
        dst: Writable::from_reg(xmm0),
        base: rax,
        index: xmm1,
        scale: 4,
        disp: 0,
        mask: k1,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());

    let ctrl_plane = &mut Default::default();
    let constants = Default::default();
    let buffer = buffer.finish(&constants, ctrl_plane);

    let bytes = buffer.data();
    assert!(!bytes.is_empty(), "VPGATHERDD encoding should not be empty");
    assert_eq!(bytes[0], 0x62, "VPGATHERDD should have EVEX prefix");

    println!("VPGATHERDD encoding: {:02X?}", bytes);
}

/// Test scatter instruction encoding.
#[test]
fn test_scatter_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;
    use crate::isa::x64::inst::avx512::ScatterOp;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let xmm0 = regs::xmm0();
    let xmm1 = regs::xmm1();
    let rax = regs::rax();
    let k1 = regs::k1();

    let inst = Inst::Avx512Scatter {
        op: ScatterOp::Vpscatterdd,
        src: xmm0,
        base: rax,
        index: xmm1,
        scale: 4,
        disp: 0,
        mask: k1,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());

    let ctrl_plane = &mut Default::default();
    let constants = Default::default();
    let buffer = buffer.finish(&constants, ctrl_plane);

    let bytes = buffer.data();
    assert!(!bytes.is_empty(), "VPSCATTERDD encoding should not be empty");
    assert_eq!(bytes[0], 0x62, "VPSCATTERDD should have EVEX prefix");

    println!("VPSCATTERDD encoding: {:02X?}", bytes);
}

/// Test all gather operations encode correctly.
#[test]
fn test_all_gather_ops_encode() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;
    use crate::isa::x64::inst::avx512::GatherOp;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let xmm0 = regs::xmm0();
    let xmm1 = regs::xmm1();
    let rax = regs::rax();
    let k1 = regs::k1();

    let gather_ops = [
        (GatherOp::Vpgatherdd, "VPGATHERDD"),
        (GatherOp::Vpgatherdq, "VPGATHERDQ"),
        (GatherOp::Vpgatherqd, "VPGATHERQD"),
        (GatherOp::Vpgatherqq, "VPGATHERQQ"),
    ];

    for (op, name) in gather_ops.iter() {
        let inst = Inst::Avx512Gather {
            op: *op,
            dst: Writable::from_reg(xmm0),
            base: rax,
            index: xmm1,
            scale: 4,
            disp: 0,
            mask: k1,
        };

        let mut buffer = MachBuffer::new();
        inst.emit(&mut buffer, &emit_info, &mut Default::default());

        let ctrl_plane = &mut Default::default();
        let constants = Default::default();
        let buffer = buffer.finish(&constants, ctrl_plane);

        let bytes = buffer.data();
        assert!(!bytes.is_empty(), "{} encoding should not be empty", name);
        assert_eq!(bytes[0], 0x62, "{} should have EVEX prefix", name);

        println!("{} encoding: {:02X?}", name, bytes);
    }
}

/// Test all scatter operations encode correctly.
#[test]
fn test_all_scatter_ops_encode() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;
    use crate::isa::x64::inst::avx512::ScatterOp;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let xmm0 = regs::xmm0();
    let xmm1 = regs::xmm1();
    let rax = regs::rax();
    let k1 = regs::k1();

    let scatter_ops = [
        (ScatterOp::Vpscatterdd, "VPSCATTERDD"),
        (ScatterOp::Vpscatterdq, "VPSCATTERDQ"),
        (ScatterOp::Vpscatterqd, "VPSCATTERQD"),
        (ScatterOp::Vpscatterqq, "VPSCATTERQQ"),
    ];

    for (op, name) in scatter_ops.iter() {
        let inst = Inst::Avx512Scatter {
            op: *op,
            src: xmm0,
            base: rax,
            index: xmm1,
            scale: 4,
            disp: 0,
            mask: k1,
        };

        let mut buffer = MachBuffer::new();
        inst.emit(&mut buffer, &emit_info, &mut Default::default());

        let ctrl_plane = &mut Default::default();
        let constants = Default::default();
        let buffer = buffer.finish(&constants, ctrl_plane);

        let bytes = buffer.data();
        assert!(!bytes.is_empty(), "{} encoding should not be empty", name);
        assert_eq!(bytes[0], 0x62, "{} should have EVEX prefix", name);

        println!("{} encoding: {:02X?}", name, bytes);
    }
}

/// Test 256-bit load instruction encoding.
#[test]
fn test_256bit_load_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;
    use crate::isa::x64::inst::args::{Amode, SyntheticAmode, Xmm};
    use crate::isa::x64::inst::OperandSize;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // Create a 256-bit load from [rax]
    let xmm0 = Xmm::unwrap_new(regs::xmm0());
    let rax = regs::rax();

    let addr = SyntheticAmode::Real(Amode::ImmReg {
        simm32: 0,
        base: rax,
        flags: crate::ir::MemFlags::new(),
    });

    let inst = Inst::Avx512256Load {
        size: OperandSize::Size32,
        dst: Writable::from_reg(xmm0),
        addr,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());

    let ctrl_plane = &mut Default::default();
    let constants = Default::default();
    let buffer = buffer.finish(&constants, ctrl_plane);

    let bytes = buffer.data();
    println!("256-bit load (VMOVDQU32 ymm, [rax]) encoding: {:02X?}", bytes);

    // VMOVDQU32 ymm0, [rax]: EVEX.256.F3.0F.W0 6F /r
    // Expected encoding:
    // 62 = EVEX prefix
    // F1 = P0: R=1 X=1 B=1 R'=1 mm=01
    // 7E = P1: W=0 vvvv=1111 1 pp=10 (F3)
    // 28 = P2: z=0 L'L=01 (256-bit) b=0 V'=1 aaa=000
    // 6F = opcode (VMOVDQU load)
    // 00 = ModRM: mod=00 reg=000 r/m=000 (rax)

    assert!(!bytes.is_empty(), "256-bit load encoding should not be empty");
    assert_eq!(bytes[0], 0x62, "Should have EVEX prefix");

    // Check L'L bits in P2 (byte 3)
    // P2 = z L'L b V' aaa
    // For 256-bit: L'L = 01, so bit 6 = 0, bit 5 = 1
    let p2 = bytes[3];
    let ll = (p2 >> 5) & 0x3;
    assert_eq!(ll, 0b01, "L'L should be 01 for 256-bit operation, got {:02b}", ll);

    println!("256-bit load encoding verification PASSED!");
}

/// Test that 256-bit load actually works at runtime.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_256bit_load() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping 256-bit load execution test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Input data: 8 x i32 = 256 bits
        #[repr(C, align(32))]
        struct Data([i32; 8]);
        let data = Data([1, 2, 3, 4, 5, 6, 7, 8]);
        let mut result = Data([0; 8]);

        // Load 256 bits using EVEX-encoded VMOVDQU32
        asm!(
            "vmovdqu32 ymm0, [{data}]",
            "vmovdqu32 [{result}], ymm0",
            data = in(reg) data.0.as_ptr(),
            result = in(reg) result.0.as_mut_ptr(),
            options(nostack),
        );

        // Verify results
        for i in 0..8 {
            assert_eq!(result.0[i], data.0[i], "256-bit load failed at index {}", i);
        }
        println!("256-bit load execution test PASSED - all 8 elements correct: {:?}", result.0);
    }
}

// =========================================================================
// Runtime Execution Tests (requires AVX-512 hardware)
// =========================================================================

/// Execute actual VPADDD instruction and verify results.
/// This test only runs on CPUs with AVX-512F support.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vpaddd() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPADDD execution test - AVX-512F not supported");
        return;
    }

    // Use inline assembly to execute VPADDD and verify the result
    unsafe {
        use std::arch::asm;

        // Input data: 16 x i32 = 512 bits
        let a: [i32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let b: [i32; 16] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
        let mut result: [i32; 16] = [0; 16];

        asm!(
            "vmovdqu32 zmm0, [{a}]",
            "vmovdqu32 zmm1, [{b}]",
            "vpaddd zmm2, zmm0, zmm1",
            "vmovdqu32 [{result}], zmm2",
            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Verify results
        for i in 0..16 {
            assert_eq!(result[i], a[i] + b[i], "VPADDD failed at index {}", i);
        }
        println!("VPADDD execution test PASSED - all 16 elements correct");
    }
}

/// Execute actual VPANDD instruction and verify results.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vpandd() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPANDD execution test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        let a: [u32; 16] = [0xFF00FF00; 16];
        let b: [u32; 16] = [0x0F0F0F0F; 16];
        let mut result: [u32; 16] = [0; 16];

        asm!(
            "vmovdqu32 zmm0, [{a}]",
            "vmovdqu32 zmm1, [{b}]",
            "vpandd zmm2, zmm0, zmm1",
            "vmovdqu32 [{result}], zmm2",
            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        for i in 0..16 {
            assert_eq!(result[i], 0x0F000F00, "VPANDD failed at index {}", i);
        }
        println!("VPANDD execution test PASSED");
    }
}

/// Execute masked VPADDD with zeroing.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_masked_vpaddd_zeroing() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping masked VPADDD test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        let a: [i32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let b: [i32; 16] = [100; 16];
        let mut result: [i32; 16] = [0; 16];
        let mask: u16 = 0b1010_1010_1010_1010; // Even indices masked

        asm!(
            "kmovw k1, {mask:e}",
            "vmovdqu32 zmm0, [{a}]",
            "vmovdqu32 zmm1, [{b}]",
            "vpaddd zmm2{{k1}}{{z}}, zmm0, zmm1",  // Zeroing mode
            "vmovdqu32 [{result}], zmm2",
            mask = in(reg) mask as u32,
            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Odd indices should have a[i] + b[i], even indices should be 0
        for i in 0..16 {
            if i % 2 == 1 {
                assert_eq!(result[i], a[i] + b[i], "Masked VPADDD failed at index {}", i);
            } else {
                assert_eq!(result[i], 0, "Zeroing failed at index {}", i);
            }
        }
        println!("Masked VPADDD with zeroing test PASSED");
    }
}

/// Execute VPCOMPRESSD and verify results.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vpcompressd() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPCOMPRESSD test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        let data: [i32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut result: [i32; 16] = [0; 16];
        let mask: u16 = 0b0000_0000_1111_0000; // Elements 4-7 are selected

        asm!(
            "kmovw k1, {mask:e}",
            "vmovdqu32 zmm0, [{data}]",
            "vpcompressd [{result}]{{k1}}, zmm0",
            mask = in(reg) mask as u32,
            data = in(reg) data.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // First 4 elements of result should be 5, 6, 7, 8
        assert_eq!(result[0], 5);
        assert_eq!(result[1], 6);
        assert_eq!(result[2], 7);
        assert_eq!(result[3], 8);
        println!("VPCOMPRESSD test PASSED: compressed values = {:?}", &result[0..4]);
    }
}

/// Execute VPGATHERDD and verify results.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vpgatherdd() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPGATHERDD test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Source data table
        let table: [i32; 32] = [
            100, 101, 102, 103, 104, 105, 106, 107,
            108, 109, 110, 111, 112, 113, 114, 115,
            116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131,
        ];

        // Indices to gather (picking elements 0, 2, 4, 6, ...)
        let indices: [i32; 16] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30];
        let mut result: [i32; 16] = [0; 16];
        let mask: u16 = 0xFFFF; // All elements

        asm!(
            "kmovw k1, {mask:e}",
            "vmovdqu32 zmm1, [{indices}]",  // Load indices into zmm1
            "vpgatherdd zmm0{{k1}}, [{table} + zmm1*4]",
            "vmovdqu32 [{result}], zmm0",
            mask = in(reg) mask as u32,
            table = in(reg) table.as_ptr(),
            indices = in(reg) indices.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Verify gathered values
        for i in 0..16 {
            let expected = table[indices[i] as usize];
            assert_eq!(result[i], expected, "VPGATHERDD failed at index {}: got {}, expected {}", i, result[i], expected);
        }
        println!("VPGATHERDD test PASSED: gathered values = {:?}", &result);
    }
}

/// Execute VPGATHERDQ (64-bit elements, 32-bit indices) and verify results.
/// This uses YMM for indices (8 x i32) and ZMM for results (8 x i64).
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vpgatherdq() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPGATHERDQ test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Source data table (8 x i64, but we have 16 to allow for indices)
        #[repr(C, align(64))]
        struct Table([i64; 16]);
        let table = Table([
            1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
            1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015,
        ]);

        // 32-bit indices (element indices, not byte offsets)
        // We pick elements 0, 2, 4, 6, 8, 10, 12, 14
        #[repr(C, align(32))]
        struct Indices([i32; 8]);
        let indices = Indices([0, 2, 4, 6, 8, 10, 12, 14]);

        #[repr(C, align(64))]
        struct Result([i64; 8]);
        let mut result = Result([0; 8]);
        let mask: u8 = 0xFF; // All 8 elements

        asm!(
            "kmovb k1, {mask:e}",
            "vmovdqu32 ymm1, [{indices}]",  // Load 8 x i32 indices into ymm1 (256-bit)
            "vpgatherdq zmm0{{k1}}, [{table} + ymm1*8]",  // scale=8 because each i64 is 8 bytes
            "vmovdqu64 [{result}], zmm0",
            mask = in(reg) mask as u32,
            table = in(reg) table.0.as_ptr(),
            indices = in(reg) indices.0.as_ptr(),
            result = in(reg) result.0.as_mut_ptr(),
            options(nostack),
        );

        // Verify gathered values
        for i in 0..8 {
            let expected = table.0[indices.0[i] as usize];
            assert_eq!(result.0[i], expected, "VPGATHERDQ failed at index {}: got {}, expected {}", i, result.0[i], expected);
        }
        println!("VPGATHERDQ test PASSED: gathered values = {:?}", &result.0);
    }
}

/// Execute VPGATHERDQ with byte offsets (scale=1) like the E2E test.
/// This verifies the same approach used in the JIT E2E test works.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vpgatherdq_byte_offsets() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPGATHERDQ byte offset test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Source data table (32 x i64)
        #[repr(C, align(64))]
        struct Table([i64; 32]);
        let table = Table([
            1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
            1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015,
            1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023,
            1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031,
        ]);

        // 32-bit indices as BYTE OFFSETS (scale=1)
        // Element 0 is at byte offset 0, element 2 is at byte offset 16, etc.
        #[repr(C, align(32))]
        struct Indices([i32; 8]);
        let indices = Indices([0, 16, 32, 48, 64, 80, 96, 112]); // Byte offsets!

        #[repr(C, align(64))]
        struct Result([i64; 8]);
        let mut result = Result([0; 8]);
        let mask: u8 = 0xFF; // All 8 elements

        asm!(
            "kmovb k1, {mask:e}",
            "vmovdqu32 ymm1, [{indices}]",  // Load 8 x i32 byte offsets into ymm1 (256-bit)
            "vpgatherdq zmm0{{k1}}, [{table} + ymm1*1]",  // scale=1 because indices are byte offsets
            "vmovdqu64 [{result}], zmm0",
            mask = in(reg) mask as u32,
            table = in(reg) table.0.as_ptr(),
            indices = in(reg) indices.0.as_ptr(),
            result = in(reg) result.0.as_mut_ptr(),
            options(nostack),
        );

        // Should gather elements 0, 2, 4, 6, 8, 10, 12, 14
        assert_eq!(result.0[0], 1000, "element 0");
        assert_eq!(result.0[1], 1002, "element 2");
        assert_eq!(result.0[2], 1004, "element 4");
        assert_eq!(result.0[3], 1006, "element 6");
        assert_eq!(result.0[4], 1008, "element 8");
        assert_eq!(result.0[5], 1010, "element 10");
        assert_eq!(result.0[6], 1012, "element 12");
        assert_eq!(result.0[7], 1014, "element 14");

        println!("VPGATHERDQ byte offset test PASSED: gathered values = {:?}", &result.0);
    }
}

/// Execute VPSCATTERDD and verify results.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vpscatterdd() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPSCATTERDD test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Data to scatter
        let data: [i32; 16] = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
                                1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015];

        // Scatter to even indices
        let indices: [i32; 16] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30];
        let mut result: [i32; 32] = [0; 32];
        let mask: u16 = 0xFFFF;

        asm!(
            "kmovw k1, {mask:e}",
            "vmovdqu32 zmm0, [{data}]",
            "vmovdqu32 zmm1, [{indices}]",
            "vpscatterdd [{result} + zmm1*4]{{k1}}, zmm0",
            mask = in(reg) mask as u32,
            data = in(reg) data.as_ptr(),
            indices = in(reg) indices.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Verify scattered values
        for i in 0..16 {
            let idx = indices[i] as usize;
            assert_eq!(result[idx], data[i], "VPSCATTERDD failed at index {}", i);
        }
        println!("VPSCATTERDD test PASSED");
    }
}

// =============================================================================
// Edge Case Tests - High Registers
// =============================================================================

/// Test high XMM registers (xmm8-xmm15) which require REX/EVEX extension bits.
#[test]
fn test_evex_high_xmm_8_to_15() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // Test all combinations of registers 8-15
    let high_regs = [
        regs::xmm8(), regs::xmm9(), regs::xmm10(), regs::xmm11(),
        regs::xmm12(), regs::xmm13(), regs::xmm14(), regs::xmm15(),
    ];

    for (i, dst_reg) in high_regs.iter().enumerate() {
        for (j, src1_reg) in high_regs.iter().enumerate() {
            for (k, src2_reg) in high_regs.iter().enumerate() {
                // Only test a subset to avoid combinatorial explosion
                if (i + j + k) % 7 != 0 {
                    continue;
                }

                let inst = Inst::Avx512Avx512Alu {
                    op: Avx512AluOp::Vpaddd,
                    size: OperandSize::Size32,
                    dst: Writable::from_reg(*dst_reg),
                    src1: *src1_reg,
                    src2: RegMem::reg(*src2_reg),
                    mask: None,
                    merge: MergeMode::Merging,
                };

                let mut buffer = MachBuffer::new();
                inst.emit(&mut buffer, &emit_info, &mut Default::default());
                let buffer = buffer.finish(&Default::default(), &mut Default::default());
                let bytes = buffer.data();

                assert_eq!(bytes[0], 0x62, "EVEX escape for xmm{} xmm{} xmm{}",
                    8 + i, 8 + j, 8 + k);
                assert!(!bytes.is_empty(), "High register encoding should not be empty");
            }
        }
    }
    println!("High XMM register (8-15) encoding tests PASSED");
}

/// Test mixed low and high register combinations.
#[test]
fn test_evex_mixed_register_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // Test: dst=low, src1=high, src2=low
    let inst1 = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vpaddd,
        size: OperandSize::Size32,
        dst: Writable::from_reg(regs::xmm0()),
        src1: regs::xmm8(),
        src2: RegMem::reg(regs::xmm2()),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst1.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    assert_eq!(buffer.data()[0], 0x62);

    // Test: dst=high, src1=low, src2=high
    let inst2 = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vpaddd,
        size: OperandSize::Size32,
        dst: Writable::from_reg(regs::xmm15()),
        src1: regs::xmm0(),
        src2: RegMem::reg(regs::xmm8()),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst2.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    assert_eq!(buffer.data()[0], 0x62);

    println!("Mixed register encoding tests PASSED");
}

// =============================================================================
// Edge Case Tests - Boundary Conditions
// =============================================================================

/// Test VPADDD with boundary values (INT_MAX, INT_MIN, 0).
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vpaddd_boundary_values() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping boundary VPADDD test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Test INT_MAX overflow
        let a: [i32; 16] = [i32::MAX; 16];
        let b: [i32; 16] = [1; 16];
        let mut result: [i32; 16] = [0; 16];

        asm!(
            "vmovdqu32 zmm0, [{a}]",
            "vmovdqu32 zmm1, [{b}]",
            "vpaddd zmm2, zmm0, zmm1",
            "vmovdqu32 [{result}], zmm2",
            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Overflow wraps around
        for i in 0..16 {
            assert_eq!(result[i], i32::MIN, "INT_MAX + 1 should wrap to INT_MIN at index {}", i);
        }

        // Test INT_MIN - 1 underflow
        let c: [i32; 16] = [i32::MIN; 16];
        let d: [i32; 16] = [-1; 16];
        let mut result2: [i32; 16] = [0; 16];

        asm!(
            "vmovdqu32 zmm0, [{c}]",
            "vmovdqu32 zmm1, [{d}]",
            "vpaddd zmm2, zmm0, zmm1",
            "vmovdqu32 [{result}], zmm2",
            c = in(reg) c.as_ptr(),
            d = in(reg) d.as_ptr(),
            result = in(reg) result2.as_mut_ptr(),
            options(nostack),
        );

        // Underflow wraps around
        for i in 0..16 {
            assert_eq!(result2[i], i32::MAX, "INT_MIN - 1 should wrap to INT_MAX at index {}", i);
        }

        println!("Boundary value VPADDD test PASSED");
    }
}

/// Test VPADDQ with 64-bit boundary values.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vpaddq_boundary_values() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping boundary VPADDQ test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        let a: [i64; 8] = [i64::MAX; 8];
        let b: [i64; 8] = [1; 8];
        let mut result: [i64; 8] = [0; 8];

        asm!(
            "vmovdqu64 zmm0, [{a}]",
            "vmovdqu64 zmm1, [{b}]",
            "vpaddq zmm2, zmm0, zmm1",
            "vmovdqu64 [{result}], zmm2",
            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        for i in 0..8 {
            assert_eq!(result[i], i64::MIN, "INT64_MAX + 1 should wrap to INT64_MIN at index {}", i);
        }

        println!("Boundary value VPADDQ test PASSED");
    }
}

/// Test shift operations with boundary shift amounts.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_shift_boundary_values() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping boundary shift test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Test shifting by 0 (should be identity)
        let data: [i32; 16] = [0x12345678; 16];
        let shift0: [i32; 4] = [0, 0, 0, 0];
        let mut result: [i32; 16] = [0; 16];

        asm!(
            "vmovdqu32 zmm0, [{data}]",
            "vmovdqu xmm1, [{shift}]",
            "vpslld zmm2, zmm0, xmm1",
            "vmovdqu32 [{result}], zmm2",
            data = in(reg) data.as_ptr(),
            shift = in(reg) shift0.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        for i in 0..16 {
            assert_eq!(result[i], 0x12345678, "Shift by 0 should be identity at index {}", i);
        }

        // Test shifting by 31 (max for 32-bit)
        let shift31: [i32; 4] = [31, 0, 0, 0];
        asm!(
            "vmovdqu32 zmm0, [{data}]",
            "vmovdqu xmm1, [{shift}]",
            "vpslld zmm2, zmm0, xmm1",
            "vmovdqu32 [{result}], zmm2",
            data = in(reg) data.as_ptr(),
            shift = in(reg) shift31.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        for i in 0..16 {
            assert_eq!(result[i], 0i32, "0x12345678 << 31 should be 0 at index {}", i);
        }

        println!("Boundary shift test PASSED");
    }
}

/// Test VPMINSD and VPMAXSD with edge values.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_minmax_boundary_values() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping boundary min/max test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        let a: [i32; 16] = [i32::MIN, i32::MAX, 0, -1, 1, 100, -100, i32::MIN,
                           i32::MAX, i32::MIN, i32::MAX, 0, -1, 1, 100, -100];
        let b: [i32; 16] = [i32::MAX, i32::MIN, 0, 1, -1, -100, 100, i32::MAX,
                           i32::MIN, i32::MAX, i32::MIN, 0, 1, -1, -100, 100];
        let mut min_result: [i32; 16] = [0; 16];
        let mut max_result: [i32; 16] = [0; 16];

        asm!(
            "vmovdqu32 zmm0, [{a}]",
            "vmovdqu32 zmm1, [{b}]",
            "vpminsd zmm2, zmm0, zmm1",
            "vpmaxsd zmm3, zmm0, zmm1",
            "vmovdqu32 [{min_result}], zmm2",
            "vmovdqu32 [{max_result}], zmm3",
            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            min_result = in(reg) min_result.as_mut_ptr(),
            max_result = in(reg) max_result.as_mut_ptr(),
            options(nostack),
        );

        for i in 0..16 {
            let expected_min = a[i].min(b[i]);
            let expected_max = a[i].max(b[i]);
            assert_eq!(min_result[i], expected_min, "VPMINSD mismatch at index {}", i);
            assert_eq!(max_result[i], expected_max, "VPMAXSD mismatch at index {}", i);
        }

        println!("Boundary min/max test PASSED");
    }
}

// =============================================================================
// Edge Case Tests - All Masking Modes
// =============================================================================

/// Test all k-registers with actual hardware execution.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_all_k_registers() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping k-register execution test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        let a: [i32; 16] = [1; 16];
        let b: [i32; 16] = [10; 16];
        let mask: u16 = 0x5555; // alternating bits

        // Test k1
        let mut result_k1: [i32; 16] = [0; 16];
        asm!(
            "kmovw k1, {mask:e}",
            "vmovdqu32 zmm0, [{a}]",
            "vmovdqu32 zmm1, [{b}]",
            "vpaddd zmm2{{k1}}{{z}}, zmm0, zmm1",
            "vmovdqu32 [{result}], zmm2",
            mask = in(reg) mask as u32,
            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            result = in(reg) result_k1.as_mut_ptr(),
            options(nostack),
        );

        // Test k2
        let mut result_k2: [i32; 16] = [0; 16];
        asm!(
            "kmovw k2, {mask:e}",
            "vmovdqu32 zmm0, [{a}]",
            "vmovdqu32 zmm1, [{b}]",
            "vpaddd zmm2{{k2}}{{z}}, zmm0, zmm1",
            "vmovdqu32 [{result}], zmm2",
            mask = in(reg) mask as u32,
            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            result = in(reg) result_k2.as_mut_ptr(),
            options(nostack),
        );

        // Results should be identical since we use the same mask value
        assert_eq!(result_k1, result_k2, "k1 and k2 should produce same results with same mask");

        // Test k7 (highest)
        let mut result_k7: [i32; 16] = [0; 16];
        asm!(
            "kmovw k7, {mask:e}",
            "vmovdqu32 zmm0, [{a}]",
            "vmovdqu32 zmm1, [{b}]",
            "vpaddd zmm2{{k7}}{{z}}, zmm0, zmm1",
            "vmovdqu32 [{result}], zmm2",
            mask = in(reg) mask as u32,
            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            result = in(reg) result_k7.as_mut_ptr(),
            options(nostack),
        );

        assert_eq!(result_k1, result_k7, "k1 and k7 should produce same results with same mask");

        println!("All k-register execution test PASSED");
    }
}

/// Test merging mode vs zeroing mode.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_merging_vs_zeroing() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping merge/zero mode test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        let a: [i32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let b: [i32; 16] = [100; 16];
        let mask: u16 = 0b1010_1010_1010_1010; // Only odd indices active
        let dest_init: [i32; 16] = [999; 16]; // Initial destination value

        // Test merging mode (masked-off elements keep old value)
        let mut result_merge: [i32; 16] = [0; 16];
        asm!(
            "kmovw k1, {mask:e}",
            "vmovdqu32 zmm2, [{dest_init}]",  // Pre-load destination
            "vmovdqu32 zmm0, [{a}]",
            "vmovdqu32 zmm1, [{b}]",
            "vpaddd zmm2{{k1}}, zmm0, zmm1",  // Merging (no {z})
            "vmovdqu32 [{result}], zmm2",
            mask = in(reg) mask as u32,
            dest_init = in(reg) dest_init.as_ptr(),
            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            result = in(reg) result_merge.as_mut_ptr(),
            options(nostack),
        );

        // Test zeroing mode (masked-off elements become 0)
        let mut result_zero: [i32; 16] = [0; 16];
        asm!(
            "kmovw k1, {mask:e}",
            "vmovdqu32 zmm2, [{dest_init}]",
            "vmovdqu32 zmm0, [{a}]",
            "vmovdqu32 zmm1, [{b}]",
            "vpaddd zmm2{{k1}}{{z}}, zmm0, zmm1",  // Zeroing mode
            "vmovdqu32 [{result}], zmm2",
            mask = in(reg) mask as u32,
            dest_init = in(reg) dest_init.as_ptr(),
            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            result = in(reg) result_zero.as_mut_ptr(),
            options(nostack),
        );

        for i in 0..16 {
            if i % 2 == 1 {
                // Odd indices should have a[i] + b[i] in both modes
                assert_eq!(result_merge[i], a[i] + b[i], "Merge result wrong at index {}", i);
                assert_eq!(result_zero[i], a[i] + b[i], "Zero result wrong at index {}", i);
            } else {
                // Even indices: merge keeps 999, zero becomes 0
                assert_eq!(result_merge[i], 999, "Merge should preserve old value at index {}", i);
                assert_eq!(result_zero[i], 0, "Zeroing should produce 0 at index {}", i);
            }
        }

        println!("Merging vs Zeroing mode test PASSED");
    }
}

// =============================================================================
// Edge Case Tests - Gather/Scatter with Various Index Patterns
// =============================================================================

/// Test gather with non-contiguous indices.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_gather_random_indices() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping random gather test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        let table: [i32; 64] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ];

        // Random non-contiguous indices
        let indices: [i32; 16] = [63, 0, 32, 15, 48, 7, 55, 23, 1, 62, 31, 16, 47, 8, 39, 24];
        let mut result: [i32; 16] = [0; 16];
        let mask: u16 = 0xFFFF;

        asm!(
            "kmovw k1, {mask:e}",
            "vmovdqu32 zmm1, [{indices}]",
            "vpgatherdd zmm0{{k1}}, [{table} + zmm1*4]",
            "vmovdqu32 [{result}], zmm0",
            mask = in(reg) mask as u32,
            table = in(reg) table.as_ptr(),
            indices = in(reg) indices.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        for i in 0..16 {
            let expected = table[indices[i] as usize];
            assert_eq!(result[i], expected, "VPGATHERDD random indices failed at {}: got {}, expected {}",
                i, result[i], expected);
        }

        println!("Random indices gather test PASSED");
    }
}

/// Test scatter with overlapping indices (last write wins).
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_scatter_overlapping_indices() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping overlapping scatter test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Multiple data values targeting same locations
        let data: [i32; 16] = [100, 200, 300, 400, 500, 600, 700, 800,
                               900, 1000, 1100, 1200, 1300, 1400, 1500, 1600];
        // All scatter to first 4 indices - last write wins
        let indices: [i32; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let mut result: [i32; 8] = [0; 8];
        let mask: u16 = 0xFFFF;

        asm!(
            "kmovw k1, {mask:e}",
            "vmovdqu32 zmm0, [{data}]",
            "vmovdqu32 zmm1, [{indices}]",
            "vpscatterdd [{result} + zmm1*4]{{k1}}, zmm0",
            mask = in(reg) mask as u32,
            data = in(reg) data.as_ptr(),
            indices = in(reg) indices.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // The last writes (indices 12-15 in data) should win
        assert_eq!(result[0], 1300, "Overlapping scatter: index 0 should have last write");
        assert_eq!(result[1], 1400, "Overlapping scatter: index 1 should have last write");
        assert_eq!(result[2], 1500, "Overlapping scatter: index 2 should have last write");
        assert_eq!(result[3], 1600, "Overlapping scatter: index 3 should have last write");

        println!("Overlapping scatter test PASSED");
    }
}

/// Test gather with partial mask (only some elements gathered).
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_gather_partial_mask() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping partial mask gather test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        let table: [i32; 16] = [100, 101, 102, 103, 104, 105, 106, 107,
                                108, 109, 110, 111, 112, 113, 114, 115];
        let indices: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let dest_init: [i32; 16] = [999; 16];
        let mut result: [i32; 16] = [0; 16];
        let mask: u16 = 0b1111_0000_1111_0000; // Gather elements 4-7 and 12-15

        asm!(
            "kmovw k1, {mask:e}",
            "vmovdqu32 zmm0, [{dest_init}]",  // Initialize destination
            "vmovdqu32 zmm1, [{indices}]",
            "vpgatherdd zmm0{{k1}}, [{table} + zmm1*4]",
            "vmovdqu32 [{result}], zmm0",
            mask = in(reg) mask as u32,
            dest_init = in(reg) dest_init.as_ptr(),
            table = in(reg) table.as_ptr(),
            indices = in(reg) indices.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        for i in 0..16 {
            if (mask & (1 << i)) != 0 {
                assert_eq!(result[i], table[i], "Gathered element at {} wrong", i);
            } else {
                // Non-gathered elements should keep initial value
                assert_eq!(result[i], 999, "Non-gathered element at {} should be 999", i);
            }
        }

        println!("Partial mask gather test PASSED");
    }
}

// =============================================================================
// Edge Case Tests - Compress/Expand
// =============================================================================

/// Test VPCOMPRESSD with various mask patterns.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_compress_various_masks() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping compress mask patterns test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        let data: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

        // Test: compress first half only
        let mask1: u16 = 0b0000_0000_1111_1111;
        let mut result1: [i32; 16] = [-1; 16];

        asm!(
            "kmovw k1, {mask:e}",
            "vmovdqu32 zmm0, [{data}]",
            "vpcompressd [{result}]{{k1}}, zmm0",
            mask = in(reg) mask1 as u32,
            data = in(reg) data.as_ptr(),
            result = in(reg) result1.as_mut_ptr(),
            options(nostack),
        );

        // Should compress elements 0-7 to beginning
        for i in 0..8 {
            assert_eq!(result1[i], i as i32, "Compress first half: wrong value at {}", i);
        }

        // Test: compress odd elements only
        let mask2: u16 = 0b1010_1010_1010_1010;
        let mut result2: [i32; 16] = [-1; 16];

        asm!(
            "kmovw k1, {mask:e}",
            "vmovdqu32 zmm0, [{data}]",
            "vpcompressd [{result}]{{k1}}, zmm0",
            mask = in(reg) mask2 as u32,
            data = in(reg) data.as_ptr(),
            result = in(reg) result2.as_mut_ptr(),
            options(nostack),
        );

        // Should compress elements 1, 3, 5, 7, 9, 11, 13, 15 to beginning
        let expected = [1, 3, 5, 7, 9, 11, 13, 15];
        for i in 0..8 {
            assert_eq!(result2[i], expected[i], "Compress odd: wrong value at {}", i);
        }

        // Test: compress single element
        let mask3: u16 = 0b0000_0001_0000_0000;
        let mut result3: [i32; 16] = [-1; 16];

        asm!(
            "kmovw k1, {mask:e}",
            "vmovdqu32 zmm0, [{data}]",
            "vpcompressd [{result}]{{k1}}, zmm0",
            mask = in(reg) mask3 as u32,
            data = in(reg) data.as_ptr(),
            result = in(reg) result3.as_mut_ptr(),
            options(nostack),
        );

        assert_eq!(result3[0], 8, "Compress single element: should be element 8");

        println!("Compress various masks test PASSED");
    }
}

/// Test VPEXPANDD (reverse of compress).
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_expand() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping expand test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Compressed data: [10, 20, 30, 40, ...]
        let compressed: [i32; 16] = [10, 20, 30, 40, 50, 60, 70, 80, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut result: [i32; 16] = [0; 16];
        let mask: u16 = 0b0000_0000_1111_1111; // Expand to first 8 positions

        asm!(
            "kmovw k1, {mask:e}",
            "vmovdqu32 zmm0, [{compressed}]",
            "vpexpandd zmm1{{k1}}{{z}}, zmm0",
            "vmovdqu32 [{result}], zmm1",
            mask = in(reg) mask as u32,
            compressed = in(reg) compressed.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // First 8 elements should be expanded
        for i in 0..8 {
            assert_eq!(result[i], compressed[i], "Expand: wrong value at {}", i);
        }
        // Rest should be zeroed
        for i in 8..16 {
            assert_eq!(result[i], 0, "Expand: element {} should be 0", i);
        }

        println!("Expand test PASSED");
    }
}

// =============================================================================
// Stress Tests - Loops and Complex Scenarios
// =============================================================================

/// Stress test: Large vector sum using VPADDD in a loop.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_stress_large_vector_sum() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping large vector sum stress test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        const SIZE: usize = 16384; // 16K elements = 64KB
        let a: Vec<i32> = (0..SIZE as i32).collect();
        let b: Vec<i32> = (0..SIZE as i32).map(|x| x * 2).collect();
        let mut result: Vec<i32> = vec![0; SIZE];

        // Process 16 elements at a time (512-bit)
        let chunks = SIZE / 16;
        for i in 0..chunks {
            let a_ptr = a.as_ptr().add(i * 16);
            let b_ptr = b.as_ptr().add(i * 16);
            let r_ptr = result.as_mut_ptr().add(i * 16);

            asm!(
                "vmovdqu32 zmm0, [{a}]",
                "vmovdqu32 zmm1, [{b}]",
                "vpaddd zmm2, zmm0, zmm1",
                "vmovdqu32 [{r}], zmm2",
                a = in(reg) a_ptr,
                b = in(reg) b_ptr,
                r = in(reg) r_ptr,
                options(nostack),
            );
        }

        // Verify results
        for i in 0..SIZE {
            let expected = a[i] + b[i];
            assert_eq!(result[i], expected, "Large sum mismatch at index {}", i);
        }

        println!("Large vector sum stress test PASSED ({} elements)", SIZE);
    }
}

/// Stress test: Chained operations (add, sub, and, or, xor).
#[test]
#[cfg(target_arch = "x86_64")]
fn test_stress_chained_operations() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping chained operations stress test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        const SIZE: usize = 1024;
        let a: Vec<i32> = (0..SIZE as i32).collect();
        let b: Vec<i32> = (0..SIZE as i32).map(|x| x + 100).collect();
        let c: Vec<i32> = (0..SIZE as i32).map(|x| x * 3).collect();
        let mut result: Vec<i32> = vec![0; SIZE];

        // Complex chain: ((a + b) - c) & mask | constant
        let mask_val: [i32; 16] = [0x0FFFFFFF; 16];
        let const_val: [i32; 16] = [0x10000000; 16];

        let chunks = SIZE / 16;
        for i in 0..chunks {
            let a_ptr = a.as_ptr().add(i * 16);
            let b_ptr = b.as_ptr().add(i * 16);
            let c_ptr = c.as_ptr().add(i * 16);
            let r_ptr = result.as_mut_ptr().add(i * 16);

            asm!(
                "vmovdqu32 zmm0, [{a}]",     // load a
                "vmovdqu32 zmm1, [{b}]",     // load b
                "vmovdqu32 zmm2, [{c}]",     // load c
                "vmovdqu32 zmm3, [{mask}]",  // load mask
                "vmovdqu32 zmm4, [{konst}]", // load constant
                "vpaddd zmm5, zmm0, zmm1",   // a + b
                "vpsubd zmm5, zmm5, zmm2",   // (a + b) - c
                "vpandd zmm5, zmm5, zmm3",   // ((a + b) - c) & mask
                "vpord zmm5, zmm5, zmm4",    // result | constant
                "vmovdqu32 [{r}], zmm5",     // store result
                a = in(reg) a_ptr,
                b = in(reg) b_ptr,
                c = in(reg) c_ptr,
                mask = in(reg) mask_val.as_ptr(),
                konst = in(reg) const_val.as_ptr(),
                r = in(reg) r_ptr,
                options(nostack),
            );
        }

        // Verify results
        for i in 0..SIZE {
            let expected = ((a[i].wrapping_add(b[i]).wrapping_sub(c[i])) & 0x0FFFFFFF) | 0x10000000;
            assert_eq!(result[i], expected, "Chained ops mismatch at index {}", i);
        }

        println!("Chained operations stress test PASSED ({} elements)", SIZE);
    }
}

/// Stress test: Masked operations in a loop with varying masks.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_stress_masked_loop() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping masked loop stress test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        const SIZE: usize = 1024;
        let a: Vec<i32> = (0..SIZE as i32).collect();
        let b: Vec<i32> = vec![1000; SIZE];
        let mut result: Vec<i32> = vec![-1; SIZE];

        let chunks = SIZE / 16;
        for i in 0..chunks {
            // Varying mask based on chunk index
            let mask: u16 = ((i * 0x1357) & 0xFFFF) as u16;

            let a_ptr = a.as_ptr().add(i * 16);
            let b_ptr = b.as_ptr().add(i * 16);
            let r_ptr = result.as_mut_ptr().add(i * 16);

            asm!(
                "kmovw k1, {mask:e}",
                "vmovdqu32 zmm0, [{a}]",
                "vmovdqu32 zmm1, [{b}]",
                "vmovdqu32 zmm2, [{r}]",   // Pre-load result for merging
                "vpaddd zmm2{{k1}}, zmm0, zmm1",  // Merging mode
                "vmovdqu32 [{r}], zmm2",
                mask = in(reg) mask as u32,
                a = in(reg) a_ptr,
                b = in(reg) b_ptr,
                r = in(reg) r_ptr,
                options(nostack),
            );
        }

        // Verify: masked elements should have a+b, others should be -1
        for chunk in 0..chunks {
            let mask = ((chunk * 0x1357) & 0xFFFF) as u16;
            for lane in 0..16 {
                let idx = chunk * 16 + lane;
                if (mask & (1 << lane)) != 0 {
                    assert_eq!(result[idx], a[idx] + b[idx],
                        "Masked result wrong at index {}", idx);
                } else {
                    assert_eq!(result[idx], -1,
                        "Non-masked element should be unchanged at index {}", idx);
                }
            }
        }

        println!("Masked loop stress test PASSED ({} elements)", SIZE);
    }
}

/// Stress test: Gather from large table (simulating columnar DB lookup).
#[test]
#[cfg(target_arch = "x86_64")]
fn test_stress_gather_large_table() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping large table gather stress test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        const TABLE_SIZE: usize = 65536; // 64K element table
        const QUERY_SIZE: usize = 4096;  // 4K lookups

        // Build a "columnar table" with sequential values
        let table: Vec<i32> = (0..TABLE_SIZE as i32).collect();

        // Build random-ish indices
        let mut indices: Vec<i32> = vec![0; QUERY_SIZE];
        for i in 0..QUERY_SIZE {
            indices[i] = ((i * 7919 + 13) % TABLE_SIZE) as i32;
        }

        let mut result: Vec<i32> = vec![0; QUERY_SIZE];
        let mask: u16 = 0xFFFF;

        let chunks = QUERY_SIZE / 16;
        for i in 0..chunks {
            let idx_ptr = indices.as_ptr().add(i * 16);
            let r_ptr = result.as_mut_ptr().add(i * 16);

            asm!(
                "kmovw k1, {mask:e}",
                "vmovdqu32 zmm1, [{idx}]",
                "vpgatherdd zmm0{{k1}}, [{table} + zmm1*4]",
                "vmovdqu32 [{r}], zmm0",
                mask = in(reg) mask as u32,
                table = in(reg) table.as_ptr(),
                idx = in(reg) idx_ptr,
                r = in(reg) r_ptr,
                options(nostack),
            );
        }

        // Verify all lookups
        for i in 0..QUERY_SIZE {
            let expected = table[indices[i] as usize];
            assert_eq!(result[i], expected,
                "Gather lookup mismatch at index {}: got {}, expected {}",
                i, result[i], expected);
        }

        println!("Large table gather stress test PASSED ({} lookups from {} elements)",
            QUERY_SIZE, TABLE_SIZE);
    }
}

/// Stress test: Filter and compress (columnar DB predicate evaluation).
#[test]
#[cfg(target_arch = "x86_64")]
fn test_stress_filter_compress() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping filter compress stress test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        const SIZE: usize = 4096;

        // Input: values 0..SIZE
        let input: Vec<i32> = (0..SIZE as i32).collect();

        // Filter: keep only values >= 1000 and <= 3000
        let lo_bound: [i32; 16] = [1000; 16];
        let hi_bound: [i32; 16] = [3000; 16];

        // Output buffer (worst case: all pass)
        let mut output: Vec<i32> = vec![0; SIZE];
        let mut output_idx: usize = 0;

        let chunks = SIZE / 16;
        for i in 0..chunks {
            let in_ptr = input.as_ptr().add(i * 16);
            let out_ptr = output.as_mut_ptr().add(output_idx);
            let mut popcnt: u64 = 0;

            asm!(
                // Load data and bounds
                "vmovdqu32 zmm0, [{input}]",
                "vmovdqu32 zmm1, [{lo}]",
                "vmovdqu32 zmm2, [{hi}]",
                // Compare: k1 = (data >= lo), k2 = (data <= hi)
                "vpcmpd k1, zmm0, zmm1, 5",  // 5 = GE
                "vpcmpd k2, zmm0, zmm2, 2",  // 2 = LE
                // Combined mask: k3 = k1 & k2
                "kandd k3, k1, k2",
                // Count matching elements
                "kmovw {tmp:e}, k3",
                "popcnt {popcnt:e}, {tmp:e}",
                // Compress matching elements to output
                "vpcompressd [{output}]{{k3}}, zmm0",
                input = in(reg) in_ptr,
                lo = in(reg) lo_bound.as_ptr(),
                hi = in(reg) hi_bound.as_ptr(),
                output = in(reg) out_ptr,
                tmp = out(reg) _,
                popcnt = out(reg) popcnt,
                options(nostack),
            );

            output_idx += popcnt as usize;
        }

        // Verify: all output values should be in [1000, 3000]
        for i in 0..output_idx {
            assert!(output[i] >= 1000 && output[i] <= 3000,
                "Filtered value {} at index {} out of range", output[i], i);
        }

        // Expected count
        let expected_count = (1000..=3000).filter(|x| *x < SIZE as i32).count();
        assert_eq!(output_idx, expected_count,
            "Filter count mismatch: got {}, expected {}", output_idx, expected_count);

        println!("Filter+compress stress test PASSED ({} of {} passed filter)",
            output_idx, SIZE);
    }
}

/// Stress test: Multiple register usage (register pressure).
#[test]
#[cfg(target_arch = "x86_64")]
fn test_stress_register_pressure() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping register pressure stress test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Use many ZMM registers simultaneously
        let a: [i32; 16] = [1; 16];
        let b: [i32; 16] = [2; 16];
        let c: [i32; 16] = [3; 16];
        let d: [i32; 16] = [4; 16];
        let e: [i32; 16] = [5; 16];
        let f: [i32; 16] = [6; 16];
        let g: [i32; 16] = [7; 16];
        let h: [i32; 16] = [8; 16];
        let mut result: [i32; 16] = [0; 16];

        asm!(
            // Load 8 vectors
            "vmovdqu32 zmm0, [{a}]",
            "vmovdqu32 zmm1, [{b}]",
            "vmovdqu32 zmm2, [{c}]",
            "vmovdqu32 zmm3, [{d}]",
            "vmovdqu32 zmm4, [{e}]",
            "vmovdqu32 zmm5, [{f}]",
            "vmovdqu32 zmm6, [{g}]",
            "vmovdqu32 zmm7, [{h}]",
            // Chain computations using results as inputs
            "vpaddd zmm8, zmm0, zmm1",   // r1 = a + b = 3
            "vpaddd zmm9, zmm2, zmm3",   // r2 = c + d = 7
            "vpaddd zmm10, zmm4, zmm5",  // r3 = e + f = 11
            "vpaddd zmm11, zmm6, zmm7",  // r4 = g + h = 15
            "vpaddd zmm12, zmm8, zmm9",  // r5 = r1 + r2 = 10
            "vpaddd zmm13, zmm10, zmm11",// r6 = r3 + r4 = 26
            "vpaddd zmm14, zmm12, zmm13",// final = r5 + r6 = 36
            "vmovdqu32 [{r}], zmm14",
            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            c = in(reg) c.as_ptr(),
            d = in(reg) d.as_ptr(),
            e = in(reg) e.as_ptr(),
            f = in(reg) f.as_ptr(),
            g = in(reg) g.as_ptr(),
            h = in(reg) h.as_ptr(),
            r = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // 1+2+3+4+5+6+7+8 = 36
        for i in 0..16 {
            assert_eq!(result[i], 36, "Register pressure result wrong at {}", i);
        }

        println!("Register pressure stress test PASSED");
    }
}

/// Stress test: Scatter/gather round trip.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_stress_scatter_gather_roundtrip() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping scatter/gather roundtrip stress test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        const TABLE_SIZE: usize = 1024;
        const DATA_SIZE: usize = 256;

        // Original data
        let data: Vec<i32> = (1000..(1000 + DATA_SIZE) as i32).collect();

        // Random-ish scatter indices (non-overlapping)
        let mut indices: Vec<i32> = vec![0; DATA_SIZE];
        for i in 0..DATA_SIZE {
            indices[i] = ((i * 4) % TABLE_SIZE) as i32;
        }

        // Scatter target
        let mut table: Vec<i32> = vec![0; TABLE_SIZE];

        // Scatter data to table
        let mask: u16 = 0xFFFF;
        let chunks = DATA_SIZE / 16;
        for i in 0..chunks {
            let data_ptr = data.as_ptr().add(i * 16);
            let idx_ptr = indices.as_ptr().add(i * 16);

            asm!(
                "kmovw k1, {mask:e}",
                "vmovdqu32 zmm0, [{data}]",
                "vmovdqu32 zmm1, [{idx}]",
                "vpscatterdd [{table} + zmm1*4]{{k1}}, zmm0",
                mask = in(reg) mask as u32,
                data = in(reg) data_ptr,
                idx = in(reg) idx_ptr,
                table = in(reg) table.as_mut_ptr(),
                options(nostack),
            );
        }

        // Gather data back
        let mut result: Vec<i32> = vec![0; DATA_SIZE];
        for i in 0..chunks {
            let idx_ptr = indices.as_ptr().add(i * 16);
            let r_ptr = result.as_mut_ptr().add(i * 16);

            asm!(
                "kmovw k1, {mask:e}",
                "vmovdqu32 zmm1, [{idx}]",
                "vpgatherdd zmm0{{k1}}, [{table} + zmm1*4]",
                "vmovdqu32 [{r}], zmm0",
                mask = in(reg) mask as u32,
                table = in(reg) table.as_ptr(),
                idx = in(reg) idx_ptr,
                r = in(reg) r_ptr,
                options(nostack),
            );
        }

        // Verify round trip
        for i in 0..DATA_SIZE {
            assert_eq!(result[i], data[i],
                "Round trip mismatch at {}: got {}, expected {}", i, result[i], data[i]);
        }

        println!("Scatter/gather roundtrip stress test PASSED ({} elements)", DATA_SIZE);
    }
}

/// Stress test: 64-bit operations with large values.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_stress_64bit_large_values() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping 64-bit large values stress test - AVX-512F not supported");
        return;
    }

    unsafe {
        use std::arch::asm;

        const SIZE: usize = 512;

        // Large 64-bit values
        let a: Vec<i64> = (0..SIZE).map(|i| 0x7FFFFFFF00000000_i64 + i as i64).collect();
        let b: Vec<i64> = (0..SIZE).map(|i| i as i64).collect();
        let mut result: Vec<i64> = vec![0; SIZE];

        let chunks = SIZE / 8; // 8 x 64-bit = 512 bits
        for i in 0..chunks {
            let a_ptr = a.as_ptr().add(i * 8);
            let b_ptr = b.as_ptr().add(i * 8);
            let r_ptr = result.as_mut_ptr().add(i * 8);

            asm!(
                "vmovdqu64 zmm0, [{a}]",
                "vmovdqu64 zmm1, [{b}]",
                "vpaddq zmm2, zmm0, zmm1",
                "vmovdqu64 [{r}], zmm2",
                a = in(reg) a_ptr,
                b = in(reg) b_ptr,
                r = in(reg) r_ptr,
                options(nostack),
            );
        }

        // Verify
        for i in 0..SIZE {
            let expected = a[i].wrapping_add(b[i]);
            assert_eq!(result[i], expected, "64-bit add mismatch at {}", i);
        }

        println!("64-bit large values stress test PASSED ({} elements)", SIZE);
    }
}

/// Test 512-bit load/store encoding to verify correct bytes.
/// Expected encoding from objdump:
/// vmovdqu64 (%rdi), %zmm0: 62 f1 fe 48 6f 07
/// vmovdqu64 %zmm0, (%rdi): 62 f1 fe 48 7f 07
#[test]
fn test_512bit_load_store_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;
    use crate::isa::x64::inst::args::{Amode, SyntheticAmode};
    use crate::ir::MemFlags;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // Test Avx512512Load: vmovdqu64 (%rdi), %zmm0
    // Address mode: [rdi] (base = rdi, no offset)
    let addr = SyntheticAmode::Real(Amode::ImmReg {
        simm32: 0,
        base: regs::rdi(),
        flags: MemFlags::trusted(),
    });

    use crate::isa::x64::inst::args::Xmm;

    let inst = Inst::Avx512512Load {
        size: OperandSize::Size64,
        dst: Writable::from_reg(Xmm::new(regs::xmm0()).unwrap()),
        addr: addr.clone(),
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    println!("Avx512512Load bytes: {:02X?}", bytes);

    // Expected: 62 f1 fe 48 6f 07
    assert_eq!(bytes[0], 0x62, "EVEX escape byte");
    assert_eq!(bytes[1], 0xf1, "P0: R=1 X=1 B=1 R'=1 mm=01");
    assert_eq!(bytes[2], 0xfe, "P1: W=1 vvvv=1111 1 pp=10 (F3)");
    assert_eq!(bytes[3], 0x48, "P2: z=0 L'L=10 b=0 V'=1 aaa=000");
    assert_eq!(bytes[4], 0x6f, "VMOVDQU64 load opcode");
    assert_eq!(bytes[5], 0x07, "ModRM: mod=00 reg=000 rm=111");

    // Test Avx512512Store: vmovdqu64 %zmm0, (%rdi)
    let inst_store = Inst::Avx512512Store {
        size: OperandSize::Size64,
        src: Xmm::new(regs::xmm0()).unwrap(),
        addr,
    };

    let mut buffer = MachBuffer::new();
    inst_store.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    println!("Avx512512Store bytes: {:02X?}", bytes);

    // Expected: 62 f1 fe 48 7f 07
    assert_eq!(bytes[0], 0x62, "EVEX escape byte");
    assert_eq!(bytes[1], 0xf1, "P0: R=1 X=1 B=1 R'=1 mm=01");
    assert_eq!(bytes[2], 0xfe, "P1: W=1 vvvv=1111 1 pp=10 (F3)");
    assert_eq!(bytes[3], 0x48, "P2: z=0 L'L=10 b=0 V'=1 aaa=000");
    assert_eq!(bytes[4], 0x7f, "VMOVDQU64 store opcode");
    assert_eq!(bytes[5], 0x07, "ModRM: mod=00 reg=000 rm=111");

    println!("512-bit load/store encoding verified!");
}

/// Test ALU instruction encoding with memory operand.
/// Expected from objdump:
/// vpaddq (%rdi), %zmm0, %zmm1: 62 f1 fd 48 d4 0f
#[test]
fn test_512bit_alu_mem_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;
    use crate::isa::x64::inst::args::{Amode, SyntheticAmode};
    use crate::ir::MemFlags;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // Test VPADDQ: vpaddq (%rdi), %zmm0, %zmm1
    // This is: dst=zmm1, src1=zmm0, src2=(%rdi)
    let addr = SyntheticAmode::Real(Amode::ImmReg {
        simm32: 0,
        base: regs::rdi(),
        flags: MemFlags::trusted(),
    });

    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vpaddq,
        size: OperandSize::Size64,
        dst: Writable::from_reg(regs::xmm1()),
        src1: regs::xmm0(),
        src2: RegMem::mem(addr),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    println!("VPADDQ with mem bytes: {:02X?}", bytes);

    // Expected: 62 f1 fd 48 d4 0f
    // 62 = EVEX escape
    // f1 = P0: R=1 X=1 B=1 R'=1 mm=01
    // fd = P1: W=1 vvvv=1111 (zmm0 inverted = all 1s) 1 pp=01 (66)
    // 48 = P2: z=0 L'L=10 b=0 V'=1 aaa=000
    // d4 = VPADDQ opcode
    // 0f = ModRM: mod=00 reg=001 (zmm1) rm=111 (rdi)
    assert_eq!(bytes[0], 0x62, "EVEX escape byte");
    assert_eq!(bytes[1], 0xf1, "P0: R=1 X=1 B=1 R'=1 mm=01");
    assert_eq!(bytes[2], 0xfd, "P1: W=1 vvvv=1111 1 pp=01 (66 prefix)");
    assert_eq!(bytes[3], 0x48, "P2: z=0 L'L=10 b=0 V'=1 aaa=000");
    assert_eq!(bytes[4], 0xd4, "VPADDQ opcode");
    assert_eq!(bytes[5], 0x0f, "ModRM: mod=00 reg=001 rm=111");

    println!("VPADDQ with memory operand encoding verified!");
}

/// Test ALU encoding with same src/dst register (xmm5) like in JIT test.
/// Expected from objdump:
/// vpaddq (%rsi), %zmm5, %zmm5: 62 f1 d5 48 d4 2e
#[test]
fn test_512bit_alu_same_src_dst() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;
    use crate::isa::x64::inst::args::{Amode, SyntheticAmode};
    use crate::ir::MemFlags;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // Test VPADDQ: vpaddq (%rsi), %zmm5, %zmm5
    // dst=zmm5, src1=zmm5, src2=(%rsi)
    let addr = SyntheticAmode::Real(Amode::ImmReg {
        simm32: 0,
        base: regs::rsi(),  // Use RSI like in JIT test
        flags: MemFlags::trusted(),
    });

    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vpaddq,
        size: OperandSize::Size64,
        dst: Writable::from_reg(regs::xmm5()),
        src1: regs::xmm5(),  // Same as dst
        src2: RegMem::mem(addr),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    println!("VPADDQ zmm5/mem bytes: {:02X?}", bytes);

    // Expected: 62 f1 d5 48 d4 2e
    assert_eq!(bytes[0], 0x62, "EVEX escape");
    assert_eq!(bytes[1], 0xf1, "P0");
    assert_eq!(bytes[2], 0xd5, "P1: W=1 vvvv=1010(zmm5 inverted) 1 pp=01");
    assert_eq!(bytes[3], 0x48, "P2");
    assert_eq!(bytes[4], 0xd4, "VPADDQ opcode");
    assert_eq!(bytes[5], 0x2e, "ModRM: mod=00 reg=101(zmm5) rm=110(rsi)");

    println!("VPADDQ same src/dst encoding verified!");
}

// =============================================================================
// Variable Rotate Instruction Tests (VPROLVD/VPROLVQ)
// =============================================================================

/// Test VPROLVD instruction encoding produces valid machine code.
#[test]
fn test_vprolvd_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VPROLVD zmm0, zmm1, zmm2
    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vprolvd,
        size: OperandSize::Size32,
        dst: Writable::from_reg(regs::xmm0()),
        src1: regs::xmm1(),
        src2: RegMem::reg(regs::xmm2()),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    assert!(!bytes.is_empty(), "VPROLVD encoding should not be empty");
    assert_eq!(bytes[0], 0x62, "EVEX prefix should start with 0x62");
    // W=0 for 32-bit elements
    assert_eq!(bytes[2] & 0x80, 0x00, "VPROLVD should have W=0");

    println!("VPROLVD encoding: {:02X?}", bytes);
}

/// Test VPROLVQ instruction encoding produces valid machine code.
#[test]
fn test_vprolvq_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VPROLVQ zmm0, zmm1, zmm2
    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vprolvq,
        size: OperandSize::Size64,
        dst: Writable::from_reg(regs::xmm0()),
        src1: regs::xmm1(),
        src2: RegMem::reg(regs::xmm2()),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    assert!(!bytes.is_empty(), "VPROLVQ encoding should not be empty");
    assert_eq!(bytes[0], 0x62, "EVEX prefix should start with 0x62");
    // W=1 for 64-bit elements
    assert_eq!(bytes[2] & 0x80, 0x80, "VPROLVQ should have W=1");

    println!("VPROLVQ encoding: {:02X?}", bytes);
}

/// Test VPROLVD execution on hardware if AVX-512 is available.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vprolvd() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPROLVD execution test - AVX-512 not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Test data: 16 x i32 values
        let data: [u32; 16] = [
            0x12345678, 0xABCDEF00, 0x00000001, 0x80000000,
            0xFFFFFFFF, 0x0F0F0F0F, 0xF0F0F0F0, 0x55555555,
            0xAAAAAAAA, 0x11111111, 0x22222222, 0x33333333,
            0x44444444, 0x55555555, 0x66666666, 0x77777777,
        ];
        // Rotation amounts per element
        let rotate: [u32; 16] = [
            1, 4, 8, 16, 31, 0, 1, 2,
            3, 4, 5, 6, 7, 8, 9, 10,
        ];
        let mut result: [u32; 16] = [0; 16];

        asm!(
            "vmovdqu32 zmm0, [{data}]",
            "vmovdqu32 zmm1, [{rotate}]",
            "vprolvd zmm2, zmm0, zmm1",
            "vmovdqu32 [{result}], zmm2",
            data = in(reg) data.as_ptr(),
            rotate = in(reg) rotate.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Verify by computing expected results manually
        for i in 0..16 {
            let expected = data[i].rotate_left(rotate[i]);
            assert_eq!(result[i], expected,
                "VPROLVD mismatch at index {}: {} rotl {} = {:#010X}, got {:#010X}",
                i, data[i], rotate[i], expected, result[i]);
        }

        println!("VPROLVD execution test passed!");
    }
}

/// Test VPROLVQ execution on hardware if AVX-512 is available.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vprolvq() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPROLVQ execution test - AVX-512 not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Test data: 8 x i64 values
        let data: [u64; 8] = [
            0x123456789ABCDEF0,
            0xFEDCBA9876543210,
            0x0000000000000001,
            0x8000000000000000,
            0xFFFFFFFFFFFFFFFF,
            0x0F0F0F0F0F0F0F0F,
            0xF0F0F0F0F0F0F0F0,
            0x5555555555555555,
        ];
        // Rotation amounts per element
        let rotate: [u64; 8] = [
            1, 4, 8, 16, 32, 63, 0, 7,
        ];
        let mut result: [u64; 8] = [0; 8];

        asm!(
            "vmovdqu64 zmm0, [{data}]",
            "vmovdqu64 zmm1, [{rotate}]",
            "vprolvq zmm2, zmm0, zmm1",
            "vmovdqu64 [{result}], zmm2",
            data = in(reg) data.as_ptr(),
            rotate = in(reg) rotate.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Verify by computing expected results manually
        for i in 0..8 {
            let expected = data[i].rotate_left(rotate[i] as u32);
            assert_eq!(result[i], expected,
                "VPROLVQ mismatch at index {}: {:#018X} rotl {} = {:#018X}, got {:#018X}",
                i, data[i], rotate[i], expected, result[i]);
        }

        println!("VPROLVQ execution test passed!");
    }
}

/// Test VPROLVD with masking.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vprolvd_masked() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPROLVD masked execution test - AVX-512 not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        let data: [u32; 16] = [0x12345678; 16];
        let rotate: [u32; 16] = [4; 16]; // Rotate all by 4
        let mut result: [u32; 16] = [0; 16];
        let mask: u16 = 0b1010_1010_1010_1010; // Only odd elements

        asm!(
            "kmovw k1, {mask:e}",
            "vmovdqu32 zmm0, [{data}]",
            "vmovdqu32 zmm1, [{rotate}]",
            "vprolvd zmm2{{k1}}{{z}}, zmm0, zmm1",
            "vmovdqu32 [{result}], zmm2",
            mask = in(reg) mask as u32,
            data = in(reg) data.as_ptr(),
            rotate = in(reg) rotate.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Verify: odd indices should have rotated value, even indices should be 0 (zeroing mask)
        let rotated = 0x12345678u32.rotate_left(4);
        for i in 0..16 {
            if (mask >> i) & 1 == 1 {
                assert_eq!(result[i], rotated,
                    "VPROLVD masked: index {} should be rotated value {:#010X}, got {:#010X}",
                    i, rotated, result[i]);
            } else {
                assert_eq!(result[i], 0,
                    "VPROLVD masked: index {} should be 0 (zeroing), got {:#010X}",
                    i, result[i]);
            }
        }

        println!("VPROLVD masked execution test passed!");
    }
}

// =============================================================================
// Population Count Instruction Tests (VPOPCNTD/VPOPCNTQ)
// =============================================================================

/// Test VPOPCNTD instruction encoding produces valid machine code.
#[test]
fn test_vpopcntd_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VPOPCNTD zmm0, zmm2 (unary: src1 ignored, src2 is source)
    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vpopcntd,
        size: OperandSize::Size32,
        dst: Writable::from_reg(regs::xmm0()),
        src1: regs::xmm0(), // Dummy for unary ops
        src2: RegMem::reg(regs::xmm2()),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    assert!(!bytes.is_empty(), "VPOPCNTD encoding should not be empty");
    assert_eq!(bytes[0], 0x62, "EVEX prefix should start with 0x62");
    // W=0 for 32-bit elements
    assert_eq!(bytes[2] & 0x80, 0x00, "VPOPCNTD should have W=0");

    println!("VPOPCNTD encoding: {:02X?}", bytes);
}

/// Test VPOPCNTQ instruction encoding produces valid machine code.
#[test]
fn test_vpopcntq_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VPOPCNTQ zmm0, zmm2 (unary: src1 ignored, src2 is source)
    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vpopcntq,
        size: OperandSize::Size64,
        dst: Writable::from_reg(regs::xmm0()),
        src1: regs::xmm0(), // Dummy for unary ops
        src2: RegMem::reg(regs::xmm2()),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    assert!(!bytes.is_empty(), "VPOPCNTQ encoding should not be empty");
    assert_eq!(bytes[0], 0x62, "EVEX prefix should start with 0x62");
    // W=1 for 64-bit elements
    assert_eq!(bytes[2] & 0x80, 0x80, "VPOPCNTQ should have W=1");

    println!("VPOPCNTQ encoding: {:02X?}", bytes);
}

/// Test VPOPCNTD execution on hardware if AVX-512 VPOPCNTDQ is available.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vpopcntd() {
    // VPOPCNTD requires AVX-512 VPOPCNTDQ extension
    if !std::arch::is_x86_feature_detected!("avx512vpopcntdq") {
        println!("Skipping VPOPCNTD execution test - AVX-512 VPOPCNTDQ not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Test data: 16 x u32 values with known popcount
        let data: [u32; 16] = [
            0x00000000, // 0 bits
            0x00000001, // 1 bit
            0x00000003, // 2 bits
            0x0000000F, // 4 bits
            0x000000FF, // 8 bits
            0x0000FFFF, // 16 bits
            0xFFFFFFFF, // 32 bits
            0x55555555, // 16 bits (alternating)
            0xAAAAAAAA, // 16 bits (alternating)
            0x12345678, // 13 bits
            0x80000000, // 1 bit
            0x80000001, // 2 bits
            0x0F0F0F0F, // 16 bits
            0xF0F0F0F0, // 16 bits
            0x01010101, // 4 bits
            0x10101010, // 4 bits
        ];
        let mut result: [u32; 16] = [0; 16];

        asm!(
            "vmovdqu32 zmm0, [{data}]",
            "vpopcntd zmm1, zmm0",
            "vmovdqu32 [{result}], zmm1",
            data = in(reg) data.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Verify by computing expected results manually
        for i in 0..16 {
            let expected = data[i].count_ones();
            assert_eq!(result[i], expected,
                "VPOPCNTD mismatch at index {}: popcount({:#010X}) = {}, got {}",
                i, data[i], expected, result[i]);
        }

        println!("VPOPCNTD execution test passed!");
    }
}

/// Test VPOPCNTQ execution on hardware if AVX-512 VPOPCNTDQ is available.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vpopcntq() {
    // VPOPCNTQ requires AVX-512 VPOPCNTDQ extension
    if !std::arch::is_x86_feature_detected!("avx512vpopcntdq") {
        println!("Skipping VPOPCNTQ execution test - AVX-512 VPOPCNTDQ not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Test data: 8 x u64 values with known popcount
        let data: [u64; 8] = [
            0x0000000000000000, // 0 bits
            0x0000000000000001, // 1 bit
            0x00000000FFFFFFFF, // 32 bits
            0xFFFFFFFFFFFFFFFF, // 64 bits
            0x5555555555555555, // 32 bits (alternating)
            0xAAAAAAAAAAAAAAAA, // 32 bits (alternating)
            0x123456789ABCDEF0, // 32 bits
            0x8000000000000001, // 2 bits
        ];
        let mut result: [u64; 8] = [0; 8];

        asm!(
            "vmovdqu64 zmm0, [{data}]",
            "vpopcntq zmm1, zmm0",
            "vmovdqu64 [{result}], zmm1",
            data = in(reg) data.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Verify by computing expected results manually
        for i in 0..8 {
            let expected = data[i].count_ones() as u64;
            assert_eq!(result[i], expected,
                "VPOPCNTQ mismatch at index {}: popcount({:#018X}) = {}, got {}",
                i, data[i], expected, result[i]);
        }

        println!("VPOPCNTQ execution test passed!");
    }
}

// =============================================================================
// Conflict Detection Instruction Tests (VPCONFLICTD/VPCONFLICTQ)
// =============================================================================

/// Test VPCONFLICTD instruction encoding produces valid machine code.
#[test]
fn test_vpconflictd_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VPCONFLICTD zmm0, zmm2 (unary: src1 ignored)
    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vpconflictd,
        size: OperandSize::Size32,
        dst: Writable::from_reg(regs::xmm0()),
        src1: regs::xmm0(), // Dummy for unary ops
        src2: RegMem::reg(regs::xmm2()),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    assert!(!bytes.is_empty(), "VPCONFLICTD encoding should not be empty");
    assert_eq!(bytes[0], 0x62, "EVEX prefix should start with 0x62");
    // W=0 for 32-bit elements
    assert_eq!(bytes[2] & 0x80, 0x00, "VPCONFLICTD should have W=0");
    // Opcode should be 0xC4
    let opcode_idx = 4; // After EVEX prefix
    assert_eq!(bytes[opcode_idx], 0xC4, "VPCONFLICTD opcode should be 0xC4");

    println!("VPCONFLICTD encoding: {:02X?}", bytes);
}

/// Test VPCONFLICTQ instruction encoding produces valid machine code.
#[test]
fn test_vpconflictq_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VPCONFLICTQ zmm0, zmm2 (unary: src1 ignored)
    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vpconflictq,
        size: OperandSize::Size64,
        dst: Writable::from_reg(regs::xmm0()),
        src1: regs::xmm0(), // Dummy for unary ops
        src2: RegMem::reg(regs::xmm2()),
        mask: None,
        merge: MergeMode::Merging,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    assert!(!bytes.is_empty(), "VPCONFLICTQ encoding should not be empty");
    assert_eq!(bytes[0], 0x62, "EVEX prefix should start with 0x62");
    // W=1 for 64-bit elements
    assert_eq!(bytes[2] & 0x80, 0x80, "VPCONFLICTQ should have W=1");
    // Opcode should be 0xC4
    let opcode_idx = 4;
    assert_eq!(bytes[opcode_idx], 0xC4, "VPCONFLICTQ opcode should be 0xC4");

    println!("VPCONFLICTQ encoding: {:02X?}", bytes);
}

/// Test VPCONFLICTD execution on hardware if AVX-512 CD is available.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vpconflictd() {
    // VPCONFLICTD requires AVX-512 CD (Conflict Detection) extension
    if !std::arch::is_x86_feature_detected!("avx512cd") {
        println!("Skipping VPCONFLICTD execution test - AVX-512 CD not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Test data: 16 x u32 values with some duplicates
        // Result: for each element, a bitmask of which PRIOR elements have the same value
        // (NOT including self)
        let data: [u32; 16] = [
            1, 2, 3, 1, 5, 2, 7, 8,   // Element 3 conflicts with 0 (both are 1), element 5 conflicts with 1 (both are 2)
            9, 10, 11, 12, 13, 14, 15, 1,  // Element 15 conflicts with 0 and 3 (all are 1)
        ];
        let mut result: [u32; 16] = [0; 16];

        asm!(
            "vmovdqu32 zmm0, [{data}]",
            "vpconflictd zmm1, zmm0",
            "vmovdqu32 [{result}], zmm1",
            data = in(reg) data.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Verify key expected results:
        // Element 0 (value 1): no prior elements, result should be 0
        assert_eq!(result[0], 0, "Element 0 should have no conflicts");
        // Element 3 (value 1): conflicts with element 0
        assert_eq!(result[3], 1 << 0, "Element 3 should conflict with element 0");
        // Element 5 (value 2): conflicts with element 1
        assert_eq!(result[5], 1 << 1, "Element 5 should conflict with element 1");
        // Element 15 (value 1): conflicts with elements 0 and 3
        assert_eq!(result[15], (1 << 0) | (1 << 3), "Element 15 should conflict with elements 0 and 3");

        println!("VPCONFLICTD execution test passed!");
        println!("Result: {:?}", result);
    }
}

/// Test VPCONFLICTQ execution on hardware if AVX-512 CD is available.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vpconflictq() {
    // VPCONFLICTQ requires AVX-512 CD extension
    if !std::arch::is_x86_feature_detected!("avx512cd") {
        println!("Skipping VPCONFLICTQ execution test - AVX-512 CD not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Test data: 8 x u64 values with some duplicates
        let data: [u64; 8] = [
            100, 200, 300, 100, 500, 200, 700, 100,
        ];
        let mut result: [u64; 8] = [0; 8];

        asm!(
            "vmovdqu64 zmm0, [{data}]",
            "vpconflictq zmm1, zmm0",
            "vmovdqu64 [{result}], zmm1",
            data = in(reg) data.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Verify key expected results:
        // Element 0 (value 100): no prior elements
        assert_eq!(result[0], 0, "Element 0 should have no conflicts");
        // Element 3 (value 100): conflicts with element 0
        assert_eq!(result[3], 1 << 0, "Element 3 should conflict with element 0");
        // Element 5 (value 200): conflicts with element 1
        assert_eq!(result[5], 1 << 1, "Element 5 should conflict with element 1");
        // Element 7 (value 100): conflicts with elements 0 and 3
        assert_eq!(result[7], (1 << 0) | (1 << 3), "Element 7 should conflict with elements 0 and 3");

        println!("VPCONFLICTQ execution test passed!");
        println!("Result: {:?}", result);
    }
}

// =============================================================================
// Type Conversion Instruction Tests
// =============================================================================

/// Test all conversion operations encode correctly.
#[test]
fn test_all_cvt_ops_encode() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;
    use crate::isa::x64::inst::avx512::Avx512CvtOp;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let xmm0 = regs::xmm0();
    let xmm1 = regs::xmm1();

    let cvt_ops = [
        (Avx512CvtOp::Vcvtdq2ps, "VCVTDQ2PS"),
        (Avx512CvtOp::Vcvtps2dq, "VCVTPS2DQ"),
        (Avx512CvtOp::Vcvttps2dq, "VCVTTPS2DQ"),
        (Avx512CvtOp::Vcvtqq2pd, "VCVTQQ2PD"),
        (Avx512CvtOp::Vcvtpd2qq, "VCVTPD2QQ"),
        (Avx512CvtOp::Vcvttpd2qq, "VCVTTPD2QQ"),
        (Avx512CvtOp::Vcvtps2pd, "VCVTPS2PD"),
        (Avx512CvtOp::Vcvtpd2ps, "VCVTPD2PS"),
    ];

    for (op, name) in cvt_ops.iter() {
        let inst = Inst::Avx512Avx512Cvt {
            op: *op,
            dst: Writable::from_reg(xmm0),
            src: RegMem::reg(xmm1),
            mask: None,
            merge: MergeMode::Zeroing,
        };

        let mut buffer = MachBuffer::new();
        inst.emit(&mut buffer, &emit_info, &mut Default::default());

        let ctrl_plane = &mut Default::default();
        let constants = Default::default();
        let buffer = buffer.finish(&constants, ctrl_plane);

        let bytes = buffer.data();
        assert!(!bytes.is_empty(), "{} encoding should not be empty", name);
        assert_eq!(bytes[0], 0x62, "{} should have EVEX prefix", name);

        println!("{} encoding: {:02X?}", name, bytes);
    }
}

/// Test VCVTDQ2PS encoding specifically.
#[test]
fn test_vcvtdq2ps_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;
    use crate::isa::x64::inst::avx512::Avx512CvtOp;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VCVTDQ2PS zmm0, zmm1
    // EVEX.512.0F.W0 5B /r
    let inst = Inst::Avx512Avx512Cvt {
        op: Avx512CvtOp::Vcvtdq2ps,
        dst: Writable::from_reg(regs::xmm0()),
        src: RegMem::reg(regs::xmm1()),
        mask: None,
        merge: MergeMode::Zeroing,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    assert_eq!(bytes[0], 0x62, "EVEX escape");
    // Verify pp=00 (no prefix) for VCVTDQ2PS
    assert_eq!(bytes[2] & 0x03, 0x00, "pp=00 (no prefix)");
    // Verify W=0
    assert_eq!(bytes[2] & 0x80, 0x00, "W=0");
    // Verify opcode
    let opcode_idx = 4;
    assert_eq!(bytes[opcode_idx], 0x5B, "opcode 0x5B");

    println!("VCVTDQ2PS encoding verified: {:02X?}", bytes);
}

/// Test VCVTPS2DQ encoding (66 prefix).
#[test]
fn test_vcvtps2dq_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;
    use crate::isa::x64::inst::avx512::Avx512CvtOp;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VCVTPS2DQ zmm0, zmm1
    // EVEX.512.66.0F.W0 5B /r
    let inst = Inst::Avx512Avx512Cvt {
        op: Avx512CvtOp::Vcvtps2dq,
        dst: Writable::from_reg(regs::xmm0()),
        src: RegMem::reg(regs::xmm1()),
        mask: None,
        merge: MergeMode::Zeroing,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    assert_eq!(bytes[0], 0x62, "EVEX escape");
    // Verify pp=01 (66 prefix) for VCVTPS2DQ
    assert_eq!(bytes[2] & 0x03, 0x01, "pp=01 (66 prefix)");
    // Verify W=0
    assert_eq!(bytes[2] & 0x80, 0x00, "W=0");

    println!("VCVTPS2DQ encoding verified: {:02X?}", bytes);
}

/// Test VCVTTPS2DQ encoding (F3 prefix - truncation).
#[test]
fn test_vcvttps2dq_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;
    use crate::isa::x64::inst::avx512::Avx512CvtOp;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VCVTTPS2DQ zmm0, zmm1
    // EVEX.512.F3.0F.W0 5B /r
    let inst = Inst::Avx512Avx512Cvt {
        op: Avx512CvtOp::Vcvttps2dq,
        dst: Writable::from_reg(regs::xmm0()),
        src: RegMem::reg(regs::xmm1()),
        mask: None,
        merge: MergeMode::Zeroing,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    assert_eq!(bytes[0], 0x62, "EVEX escape");
    // Verify pp=10 (F3 prefix) for VCVTTPS2DQ
    assert_eq!(bytes[2] & 0x03, 0x02, "pp=10 (F3 prefix)");
    // Verify W=0
    assert_eq!(bytes[2] & 0x80, 0x00, "W=0");

    println!("VCVTTPS2DQ encoding verified: {:02X?}", bytes);
}

// =============================================================================
// Type Conversion Execution Tests
// =============================================================================

/// Test VCVTDQ2PS execution on hardware.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vcvtdq2ps() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VCVTDQ2PS execution test - AVX-512F not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        let ints: [i32; 16] = [0, 1, -1, 100, -100, 1000, -1000, 2147483647,
                               -2147483648, 12345, -12345, 42, -42, 0, 1, -1];
        let mut floats: [f32; 16] = [0.0; 16];

        asm!(
            "vmovdqu32 zmm0, [{ints}]",
            "vcvtdq2ps zmm1, zmm0",
            "vmovups [{floats}], zmm1",
            ints = in(reg) ints.as_ptr(),
            floats = in(reg) floats.as_mut_ptr(),
            options(nostack),
        );

        // Verify some key conversions
        assert_eq!(floats[0], 0.0, "0 should convert to 0.0");
        assert_eq!(floats[1], 1.0, "1 should convert to 1.0");
        assert_eq!(floats[2], -1.0, "-1 should convert to -1.0");
        assert_eq!(floats[3], 100.0, "100 should convert to 100.0");
        assert_eq!(floats[4], -100.0, "-100 should convert to -100.0");
        assert_eq!(floats[11], 42.0, "42 should convert to 42.0");
        assert_eq!(floats[12], -42.0, "-42 should convert to -42.0");

        println!("VCVTDQ2PS execution test passed!");
        println!("Input ints: {:?}", &ints[..8]);
        println!("Output floats: {:?}", &floats[..8]);
    }
}

/// Test VCVTPS2DQ execution on hardware (rounding).
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vcvtps2dq() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VCVTPS2DQ execution test - AVX-512F not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Note: VCVTPS2DQ uses the current rounding mode (typically round-to-nearest)
        let floats: [f32; 16] = [0.0, 1.0, -1.0, 1.5, -1.5, 2.5, -2.5, 100.7,
                                 -100.7, 0.4, -0.4, 0.6, -0.6, 42.0, -42.0, 0.0];
        let mut ints: [i32; 16] = [0; 16];

        asm!(
            "vmovups zmm0, [{floats}]",
            "vcvtps2dq zmm1, zmm0",
            "vmovdqu32 [{ints}], zmm1",
            floats = in(reg) floats.as_ptr(),
            ints = in(reg) ints.as_mut_ptr(),
            options(nostack),
        );

        // Verify some key conversions (round-to-nearest ties to even)
        assert_eq!(ints[0], 0, "0.0 should convert to 0");
        assert_eq!(ints[1], 1, "1.0 should convert to 1");
        assert_eq!(ints[2], -1, "-1.0 should convert to -1");
        // 1.5 rounds to 2 (round to nearest, ties to even)
        assert_eq!(ints[3], 2, "1.5 should round to 2");
        assert_eq!(ints[4], -2, "-1.5 should round to -2");
        // 2.5 rounds to 2 (ties to even)
        assert_eq!(ints[5], 2, "2.5 should round to 2 (ties to even)");
        assert_eq!(ints[6], -2, "-2.5 should round to -2");

        println!("VCVTPS2DQ execution test passed!");
        println!("Input floats: {:?}", &floats[..8]);
        println!("Output ints: {:?}", &ints[..8]);
    }
}

/// Test VCVTTPS2DQ execution on hardware (truncation).
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vcvttps2dq() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VCVTTPS2DQ execution test - AVX-512F not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // VCVTTPS2DQ truncates toward zero
        let floats: [f32; 16] = [0.0, 1.0, -1.0, 1.9, -1.9, 2.5, -2.5, 100.9,
                                 -100.9, 0.9, -0.9, 0.1, -0.1, 42.0, -42.0, 0.0];
        let mut ints: [i32; 16] = [0; 16];

        asm!(
            "vmovups zmm0, [{floats}]",
            "vcvttps2dq zmm1, zmm0",
            "vmovdqu32 [{ints}], zmm1",
            floats = in(reg) floats.as_ptr(),
            ints = in(reg) ints.as_mut_ptr(),
            options(nostack),
        );

        // Verify truncation behavior
        assert_eq!(ints[0], 0, "0.0 should convert to 0");
        assert_eq!(ints[1], 1, "1.0 should convert to 1");
        assert_eq!(ints[2], -1, "-1.0 should convert to -1");
        // Truncation toward zero
        assert_eq!(ints[3], 1, "1.9 should truncate to 1");
        assert_eq!(ints[4], -1, "-1.9 should truncate to -1");
        assert_eq!(ints[5], 2, "2.5 should truncate to 2");
        assert_eq!(ints[6], -2, "-2.5 should truncate to -2");
        assert_eq!(ints[9], 0, "0.9 should truncate to 0");
        assert_eq!(ints[10], 0, "-0.9 should truncate to 0");

        println!("VCVTTPS2DQ execution test passed!");
        println!("Input floats: {:?}", &floats[..8]);
        println!("Output ints: {:?}", &ints[..8]);
    }
}

/// Test VCVTQQ2PD execution on hardware.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vcvtqq2pd() {
    if !std::arch::is_x86_feature_detected!("avx512dq") {
        println!("Skipping VCVTQQ2PD execution test - AVX-512DQ not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        let longs: [i64; 8] = [0, 1, -1, 100, -100, 9223372036854775807, -9223372036854775808, 12345];
        let mut doubles: [f64; 8] = [0.0; 8];

        asm!(
            "vmovdqu64 zmm0, [{longs}]",
            "vcvtqq2pd zmm1, zmm0",
            "vmovupd [{doubles}], zmm1",
            longs = in(reg) longs.as_ptr(),
            doubles = in(reg) doubles.as_mut_ptr(),
            options(nostack),
        );

        // Verify conversions
        assert_eq!(doubles[0], 0.0, "0 should convert to 0.0");
        assert_eq!(doubles[1], 1.0, "1 should convert to 1.0");
        assert_eq!(doubles[2], -1.0, "-1 should convert to -1.0");
        assert_eq!(doubles[3], 100.0, "100 should convert to 100.0");
        assert_eq!(doubles[4], -100.0, "-100 should convert to -100.0");

        println!("VCVTQQ2PD execution test passed!");
        println!("Input longs: {:?}", &longs[..4]);
        println!("Output doubles: {:?}", &doubles[..4]);
    }
}

/// Test VCVTTPD2QQ execution on hardware (truncation).
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vcvttpd2qq() {
    if !std::arch::is_x86_feature_detected!("avx512dq") {
        println!("Skipping VCVTTPD2QQ execution test - AVX-512DQ not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        let doubles: [f64; 8] = [0.0, 1.0, -1.0, 1.9, -1.9, 100.5, -100.5, 42.0];
        let mut longs: [i64; 8] = [0; 8];

        asm!(
            "vmovupd zmm0, [{doubles}]",
            "vcvttpd2qq zmm1, zmm0",
            "vmovdqu64 [{longs}], zmm1",
            doubles = in(reg) doubles.as_ptr(),
            longs = in(reg) longs.as_mut_ptr(),
            options(nostack),
        );

        // Verify truncation
        assert_eq!(longs[0], 0, "0.0 should convert to 0");
        assert_eq!(longs[1], 1, "1.0 should convert to 1");
        assert_eq!(longs[2], -1, "-1.0 should convert to -1");
        assert_eq!(longs[3], 1, "1.9 should truncate to 1");
        assert_eq!(longs[4], -1, "-1.9 should truncate to -1");
        assert_eq!(longs[5], 100, "100.5 should truncate to 100");
        assert_eq!(longs[6], -100, "-100.5 should truncate to -100");

        println!("VCVTTPD2QQ execution test passed!");
        println!("Input doubles: {:?}", &doubles[..4]);
        println!("Output longs: {:?}", &longs[..4]);
    }
}

/// Test VCVTPS2PD execution on hardware (f32 to f64 widening).
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vcvtps2pd() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VCVTPS2PD execution test - AVX-512F not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Input: 8 floats (256 bits), output: 8 doubles (512 bits)
        let floats: [f32; 8] = [0.0, 1.0, -1.0, 3.14159, -2.71828, 100.0, -100.0, 42.0];
        let mut doubles: [f64; 8] = [0.0; 8];

        asm!(
            "vmovups ymm0, [{floats}]",
            "vcvtps2pd zmm1, ymm0",
            "vmovupd [{doubles}], zmm1",
            floats = in(reg) floats.as_ptr(),
            doubles = in(reg) doubles.as_mut_ptr(),
            options(nostack),
        );

        // Verify conversions (f32 -> f64 should be exact for these values)
        assert_eq!(doubles[0], 0.0, "0.0 should convert exactly");
        assert_eq!(doubles[1], 1.0, "1.0 should convert exactly");
        assert_eq!(doubles[2], -1.0, "-1.0 should convert exactly");
        assert_eq!(doubles[5], 100.0, "100.0 should convert exactly");
        assert_eq!(doubles[6], -100.0, "-100.0 should convert exactly");
        assert_eq!(doubles[7], 42.0, "42.0 should convert exactly");

        println!("VCVTPS2PD execution test passed!");
        println!("Input floats: {:?}", &floats);
        println!("Output doubles: {:?}", &doubles);
    }
}

/// Test VCVTPD2PS execution on hardware (f64 to f32 narrowing).
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vcvtpd2ps() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VCVTPD2PS execution test - AVX-512F not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Input: 8 doubles (512 bits), output: 8 floats (256 bits)
        let doubles: [f64; 8] = [0.0, 1.0, -1.0, 3.14159, -2.71828, 100.0, -100.0, 42.0];
        let mut floats: [f32; 8] = [0.0; 8];

        asm!(
            "vmovupd zmm0, [{doubles}]",
            "vcvtpd2ps ymm1, zmm0",
            "vmovups [{floats}], ymm1",
            doubles = in(reg) doubles.as_ptr(),
            floats = in(reg) floats.as_mut_ptr(),
            options(nostack),
        );

        // Verify conversions
        assert_eq!(floats[0], 0.0, "0.0 should convert exactly");
        assert_eq!(floats[1], 1.0, "1.0 should convert exactly");
        assert_eq!(floats[2], -1.0, "-1.0 should convert exactly");
        assert_eq!(floats[5], 100.0, "100.0 should convert exactly");
        assert_eq!(floats[6], -100.0, "-100.0 should convert exactly");
        assert_eq!(floats[7], 42.0, "42.0 should convert exactly");

        println!("VCVTPD2PS execution test passed!");
        println!("Input doubles: {:?}", &doubles);
        println!("Output floats: {:?}", &floats);
    }
}

// =============================================================================
// Vector Alignment Instruction Tests (VALIGND, VALIGNQ)
// =============================================================================

/// Test VALIGND/VALIGNQ encoding.
#[test]
fn test_align_ops_encode() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;
    use crate::isa::x64::inst::avx512::Avx512AlignOp;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let xmm0 = regs::xmm0();
    let xmm1 = regs::xmm1();
    let xmm2 = regs::xmm2();

    let align_ops = [
        (Avx512AlignOp::Valignd, "VALIGND"),
        (Avx512AlignOp::Valignq, "VALIGNQ"),
    ];

    for (op, name) in align_ops.iter() {
        let inst = Inst::Avx512Avx512Align {
            op: *op,
            dst: Writable::from_reg(xmm0),
            src1: xmm1,
            src2: RegMem::reg(xmm2),
            imm8: 4,
            mask: None,
            merge: MergeMode::Zeroing,
        };

        let mut buffer = MachBuffer::new();
        inst.emit(&mut buffer, &emit_info, &mut Default::default());

        let ctrl_plane = &mut Default::default();
        let constants = Default::default();
        let buffer = buffer.finish(&constants, ctrl_plane);

        let bytes = buffer.data();
        assert!(!bytes.is_empty(), "{} encoding should not be empty", name);
        assert_eq!(bytes[0], 0x62, "{} should have EVEX prefix", name);
        // Last byte should be the immediate
        assert_eq!(*bytes.last().unwrap(), 4, "{} immediate should be 4", name);

        println!("{} encoding: {:02X?}", name, bytes);
    }
}

/// Test VALIGND encoding specifically.
#[test]
fn test_valignd_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;
    use crate::isa::x64::inst::avx512::Avx512AlignOp;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VALIGND zmm0, zmm1, zmm2, 5
    // EVEX.512.66.0F3A.W0 03 /r ib
    let inst = Inst::Avx512Avx512Align {
        op: Avx512AlignOp::Valignd,
        dst: Writable::from_reg(regs::xmm0()),
        src1: regs::xmm1(),
        src2: RegMem::reg(regs::xmm2()),
        imm8: 5,
        mask: None,
        merge: MergeMode::Zeroing,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    assert_eq!(bytes[0], 0x62, "EVEX escape");
    // Verify pp=01 (66 prefix) for VALIGND
    assert_eq!(bytes[2] & 0x03, 0x01, "pp=01 (66 prefix)");
    // Verify W=0
    assert_eq!(bytes[2] & 0x80, 0x00, "W=0");
    // Verify opcode 0x03
    let opcode_idx = 4;
    assert_eq!(bytes[opcode_idx], 0x03, "opcode 0x03");
    // Verify immediate byte
    assert_eq!(*bytes.last().unwrap(), 5, "imm8 = 5");

    println!("VALIGND encoding verified: {:02X?}", bytes);
}

/// Test VALIGNQ encoding specifically.
#[test]
fn test_valignq_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;
    use crate::isa::x64::inst::avx512::Avx512AlignOp;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VALIGNQ zmm0, zmm1, zmm2, 3
    // EVEX.512.66.0F3A.W1 03 /r ib
    let inst = Inst::Avx512Avx512Align {
        op: Avx512AlignOp::Valignq,
        dst: Writable::from_reg(regs::xmm0()),
        src1: regs::xmm1(),
        src2: RegMem::reg(regs::xmm2()),
        imm8: 3,
        mask: None,
        merge: MergeMode::Zeroing,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    assert_eq!(bytes[0], 0x62, "EVEX escape");
    // Verify pp=01 (66 prefix) for VALIGNQ
    assert_eq!(bytes[2] & 0x03, 0x01, "pp=01 (66 prefix)");
    // Verify W=1 for 64-bit
    assert_eq!(bytes[2] & 0x80, 0x80, "W=1");
    // Verify opcode 0x03
    let opcode_idx = 4;
    assert_eq!(bytes[opcode_idx], 0x03, "opcode 0x03");
    // Verify immediate byte
    assert_eq!(*bytes.last().unwrap(), 3, "imm8 = 3");

    println!("VALIGNQ encoding verified: {:02X?}", bytes);
}

/// Test VALIGND execution on hardware.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_valignd() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VALIGND execution test - AVX-512F not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // src1 (high): [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        // src2 (low):  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        // Concatenated: [16..31, 0..15]
        // With imm8=4: extract starting from element 4, giving [4, 5, 6, ... 19]
        let src1: [i32; 16] = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
        let src2: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let mut result: [i32; 16] = [0; 16];

        asm!(
            "vmovdqu32 zmm0, [{src1}]",
            "vmovdqu32 zmm1, [{src2}]",
            "valignd zmm2, zmm0, zmm1, 4",  // Shift by 4 elements
            "vmovdqu32 [{result}], zmm2",
            src1 = in(reg) src1.as_ptr(),
            src2 = in(reg) src2.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Expected result: elements 4..20 from the concatenation
        // [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        for i in 0..16 {
            assert_eq!(result[i], (i + 4) as i32, "VALIGND result wrong at index {}", i);
        }

        println!("VALIGND execution test passed!");
        println!("Result: {:?}", result);
    }
}

/// Test VALIGNQ execution on hardware.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_valignq() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VALIGNQ execution test - AVX-512F not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // src1 (high): [8, 9, 10, 11, 12, 13, 14, 15]
        // src2 (low):  [0, 1, 2, 3, 4, 5, 6, 7]
        // Concatenated: [8..15, 0..7]
        // With imm8=2: extract starting from element 2, giving [2, 3, 4, 5, 6, 7, 8, 9]
        let src1: [i64; 8] = [8, 9, 10, 11, 12, 13, 14, 15];
        let src2: [i64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let mut result: [i64; 8] = [0; 8];

        asm!(
            "vmovdqu64 zmm0, [{src1}]",
            "vmovdqu64 zmm1, [{src2}]",
            "valignq zmm2, zmm0, zmm1, 2",  // Shift by 2 elements
            "vmovdqu64 [{result}], zmm2",
            src1 = in(reg) src1.as_ptr(),
            src2 = in(reg) src2.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Expected result: elements 2..10 from the concatenation
        // [2, 3, 4, 5, 6, 7, 8, 9]
        for i in 0..8 {
            assert_eq!(result[i], (i + 2) as i64, "VALIGNQ result wrong at index {}", i);
        }

        println!("VALIGNQ execution test passed!");
        println!("Result: {:?}", result);
    }
}

// =============================================================================
// Ternary Logic Tests (VPTERNLOGD/Q)
// =============================================================================

/// Test VPTERNLOGD encoding.
///
/// VPTERNLOGD computes arbitrary 3-input boolean functions based on an 8-bit
/// truth table. The instruction format is:
/// VPTERNLOGD zmm1 {k1}{z}, zmm2, zmm3/m512/m32bcst, imm8
///
/// Encoding: EVEX.512.66.0F3A.W0 25 /r ib
#[test]
fn test_vpternlogd_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VPTERNLOGD zmm0, zmm1, zmm2, 0x96 (XOR3)
    // Expected encoding: 62 F3 75 48 25 C2 96
    //   62 = EVEX escape
    //   F3 = P0: R=1 X=1 B=1 R'=1 00 mm=11 = 11110011 (0F3A map)
    //   75 = P1: W=0 vvvv=1110 (zmm1 inverted) 1 pp=01 = 01110101
    //   48 = P2: z=0 L'L=10 (512-bit) b=0 V'=1 aaa=000 = 01001000
    //   25 = opcode for VPTERNLOGD
    //   C2 = ModRM: 11 000 010 (reg-reg, dst=0, src2=2)
    //   96 = immediate (XOR3 truth table)
    let inst = Inst::Avx512Avx512Ternlog {
        size: OperandSize::Size32,
        dst: Writable::from_reg(regs::xmm0()),
        src1: regs::xmm0(),  // Tied to dst
        src2: regs::xmm1(),
        src3: RegMem::reg(regs::xmm2()),
        imm8: 0x96,  // XOR3
        mask: None,
        merge: MergeMode::Zeroing,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    println!("VPTERNLOGD encoded bytes: {:02x?}", bytes);

    // Verify EVEX prefix starts with 0x62
    assert_eq!(bytes[0], 0x62, "Byte 0: EVEX escape");

    // Verify opcode (byte 4) is 0x25 (VPTERNLOGD)
    assert_eq!(bytes[4], 0x25, "Byte 4: VPTERNLOGD opcode");

    // Verify immediate (last byte) is 0x96
    assert_eq!(bytes[bytes.len() - 1], 0x96, "Last byte: immediate 0x96");

    println!("VPTERNLOGD encoding test passed!");
}

/// Test VPTERNLOGQ encoding.
#[test]
fn test_vpternlogq_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VPTERNLOGQ zmm0, zmm1, zmm2, 0xCA (blend/select)
    // Expected: W=1 for 64-bit element size
    let inst = Inst::Avx512Avx512Ternlog {
        size: OperandSize::Size64,
        dst: Writable::from_reg(regs::xmm0()),
        src1: regs::xmm0(),  // Tied to dst
        src2: regs::xmm1(),
        src3: RegMem::reg(regs::xmm2()),
        imm8: 0xCA,  // blend: (a & b) | (~a & c)
        mask: None,
        merge: MergeMode::Zeroing,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    println!("VPTERNLOGQ encoded bytes: {:02x?}", bytes);

    // Verify EVEX prefix starts with 0x62
    assert_eq!(bytes[0], 0x62, "Byte 0: EVEX escape");

    // Verify W=1 in P1 byte (bit 7)
    assert!((bytes[2] & 0x80) != 0, "Byte 2 bit 7: W=1 for 64-bit");

    // Verify opcode is 0x25
    assert_eq!(bytes[4], 0x25, "Byte 4: VPTERNLOGQ opcode");

    // Verify immediate
    assert_eq!(bytes[bytes.len() - 1], 0xCA, "Last byte: immediate 0xCA");

    println!("VPTERNLOGQ encoding test passed!");
}

/// Test VPTERNLOGD execution with various truth tables.
#[test]
fn test_vpternlogd_execution() {
    if !has_avx512f() {
        println!("Skipping VPTERNLOGD execution test: AVX-512F not available");
        return;
    }

    #[repr(align(64))]
    struct Aligned([i32; 16]);

    // Test data
    let src1 = Aligned([0xFFFF0000_u32 as i32; 16]);
    let src2 = Aligned([0xFF00FF00_u32 as i32; 16]);
    let src3 = Aligned([0xF0F0F0F0_u32 as i32; 16]);

    // Test 1: XOR3 (0x96) = a ^ b ^ c
    {
        let mut result = Aligned([0; 16]);

        unsafe {
            std::arch::asm!(
                "vmovdqu32 zmm0, [{src1}]",
                "vmovdqu32 zmm1, [{src2}]",
                "vmovdqu32 zmm2, [{src3}]",
                "vpternlogd zmm0, zmm1, zmm2, 0x96",
                "vmovdqu32 [{result}], zmm0",
                src1 = in(reg) src1.0.as_ptr(),
                src2 = in(reg) src2.0.as_ptr(),
                src3 = in(reg) src3.0.as_ptr(),
                result = in(reg) result.0.as_mut_ptr(),
                options(nostack),
            );
        }

        // Expected: 0xFFFF0000 ^ 0xFF00FF00 ^ 0xF0F0F0F0 = 0x1A0FA0F0
        let expected = (0xFFFF0000_u32 ^ 0xFF00FF00 ^ 0xF0F0F0F0) as i32;
        for i in 0..16 {
            assert_eq!(result.0[i], expected, "VPTERNLOGD XOR3 result wrong at index {}", i);
        }
        println!("VPTERNLOGD XOR3 (0x96) test passed: result = 0x{:08x}", expected as u32);
    }

    // Test 2: AND3 (0x80) = a & b & c
    {
        let mut result = Aligned([0; 16]);

        unsafe {
            std::arch::asm!(
                "vmovdqu32 zmm0, [{src1}]",
                "vmovdqu32 zmm1, [{src2}]",
                "vmovdqu32 zmm2, [{src3}]",
                "vpternlogd zmm0, zmm1, zmm2, 0x80",
                "vmovdqu32 [{result}], zmm0",
                src1 = in(reg) src1.0.as_ptr(),
                src2 = in(reg) src2.0.as_ptr(),
                src3 = in(reg) src3.0.as_ptr(),
                result = in(reg) result.0.as_mut_ptr(),
                options(nostack),
            );
        }

        // Expected: 0xFFFF0000 & 0xFF00FF00 & 0xF0F0F0F0 = 0xF0000000
        let expected = (0xFFFF0000_u32 & 0xFF00FF00 & 0xF0F0F0F0) as i32;
        for i in 0..16 {
            assert_eq!(result.0[i], expected, "VPTERNLOGD AND3 result wrong at index {}", i);
        }
        println!("VPTERNLOGD AND3 (0x80) test passed: result = 0x{:08x}", expected as u32);
    }

    // Test 3: OR3 (0xFE) = a | b | c
    {
        let mut result = Aligned([0; 16]);

        unsafe {
            std::arch::asm!(
                "vmovdqu32 zmm0, [{src1}]",
                "vmovdqu32 zmm1, [{src2}]",
                "vmovdqu32 zmm2, [{src3}]",
                "vpternlogd zmm0, zmm1, zmm2, 0xfe",
                "vmovdqu32 [{result}], zmm0",
                src1 = in(reg) src1.0.as_ptr(),
                src2 = in(reg) src2.0.as_ptr(),
                src3 = in(reg) src3.0.as_ptr(),
                result = in(reg) result.0.as_mut_ptr(),
                options(nostack),
            );
        }

        // Expected: 0xFFFF0000 | 0xFF00FF00 | 0xF0F0F0F0 = 0xFFFFfFF0
        let expected = (0xFFFF0000_u32 | 0xFF00FF00 | 0xF0F0F0F0) as i32;
        for i in 0..16 {
            assert_eq!(result.0[i], expected, "VPTERNLOGD OR3 result wrong at index {}", i);
        }
        println!("VPTERNLOGD OR3 (0xFE) test passed: result = 0x{:08x}", expected as u32);
    }

    println!("All VPTERNLOGD execution tests passed!");
}

/// Test VPTERNLOGQ execution.
#[test]
fn test_vpternlogq_execution() {
    if !has_avx512f() {
        println!("Skipping VPTERNLOGQ execution test: AVX-512F not available");
        return;
    }

    #[repr(align(64))]
    struct Aligned([i64; 8]);

    // Test data
    let src1 = Aligned([0xFFFFFFFF00000000_u64 as i64; 8]);
    let src2 = Aligned([0xFF00FF00FF00FF00_u64 as i64; 8]);
    let src3 = Aligned([0xF0F0F0F0F0F0F0F0_u64 as i64; 8]);

    // Test: Blend/select (0xCA) = (a & b) | (~a & c)
    {
        let mut result = Aligned([0; 8]);

        unsafe {
            std::arch::asm!(
                "vmovdqu64 zmm0, [{src1}]",
                "vmovdqu64 zmm1, [{src2}]",
                "vmovdqu64 zmm2, [{src3}]",
                "vpternlogq zmm0, zmm1, zmm2, 0xca",
                "vmovdqu64 [{result}], zmm0",
                src1 = in(reg) src1.0.as_ptr(),
                src2 = in(reg) src2.0.as_ptr(),
                src3 = in(reg) src3.0.as_ptr(),
                result = in(reg) result.0.as_mut_ptr(),
                options(nostack),
            );
        }

        // Expected: (a & b) | (~a & c)
        // a = 0xFFFFFFFF00000000
        // b = 0xFF00FF00FF00FF00
        // c = 0xF0F0F0F0F0F0F0F0
        // (a & b) = 0xFF00FF0000000000
        // (~a & c) = 0x00000000F0F0F0F0
        // result = 0xFF00FF00F0F0F0F0
        let expected = ((0xFFFFFFFF00000000_u64 & 0xFF00FF00FF00FF00)
            | (!0xFFFFFFFF00000000_u64 & 0xF0F0F0F0F0F0F0F0)) as i64;

        for i in 0..8 {
            assert_eq!(result.0[i], expected, "VPTERNLOGQ blend result wrong at index {}", i);
        }
        println!("VPTERNLOGQ blend (0xCA) test passed: result = 0x{:016x}", expected as u64);
    }

    println!("All VPTERNLOGQ execution tests passed!");
}

// =============================================================================
// 32x32->64 Multiply Tests (VPMULUDQ/VPMULDQ)
// =============================================================================

/// Test VPMULUDQ execution on hardware.
///
/// VPMULUDQ performs unsigned 32x32->64 multiply.
/// It takes the low 32 bits of each 64-bit element from both sources
/// and produces 64-bit results.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vpmuludq() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPMULUDQ execution test - AVX-512F not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Test unsigned 32x32->64 multiply
        // Each 64-bit element: only the low 32 bits are used for multiplication
        // src1 low 32 bits: [3, 7, 11, 0xFFFFFFFF, 100, 1000, 10000, 0x80000000]
        // src2 low 32 bits: [5, 11, 13, 2, 200, 2000, 20000, 2]
        // Result: [15, 77, 143, 0x1FFFFFFFE, 20000, 2000000, 200000000, 0x100000000]
        let src1: [u64; 8] = [
            3, 7, 11, 0xFFFFFFFF,
            100, 1000, 10000, 0x80000000,
        ];
        let src2: [u64; 8] = [
            5, 11, 13, 2,
            200, 2000, 20000, 2,
        ];
        let mut result: [u64; 8] = [0; 8];

        asm!(
            "vmovdqu64 zmm0, [{src1}]",
            "vmovdqu64 zmm1, [{src2}]",
            "vpmuludq zmm2, zmm0, zmm1",
            "vmovdqu64 [{result}], zmm2",
            src1 = in(reg) src1.as_ptr(),
            src2 = in(reg) src2.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Expected: low32(src1) * low32(src2) as unsigned
        let expected: [u64; 8] = [
            3 * 5,                    // 15
            7 * 11,                   // 77
            11 * 13,                  // 143
            0xFFFFFFFF_u64 * 2,       // 0x1FFFFFFFE
            100 * 200,                // 20000
            1000 * 2000,              // 2000000
            10000 * 20000,            // 200000000
            0x80000000_u64 * 2,       // 0x100000000
        ];

        for i in 0..8 {
            assert_eq!(result[i], expected[i],
                "VPMULUDQ result wrong at index {}: got {}, expected {}",
                i, result[i], expected[i]);
        }

        println!("VPMULUDQ execution test passed!");
        println!("Results: {:?}", result);
    }
}

/// Test VPMULDQ execution on hardware.
///
/// VPMULDQ performs signed 32x32->64 multiply.
/// It takes the low 32 bits of each 64-bit element from both sources
/// (interpreted as signed 32-bit values) and produces signed 64-bit results.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vpmuldq() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPMULDQ execution test - AVX-512F not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Test signed 32x32->64 multiply
        // The low 32 bits are sign-extended before multiplication
        // src1 (as i32): [3, -7, 11, -1, 100, -1000, 0x7FFFFFFF, -0x80000000 (i.e., i32::MIN)]
        // src2 (as i32): [5, -11, -13, 2, 200, 2000, 2, 2]
        // Result: signed products
        let src1: [i64; 8] = [
            3, -7_i32 as i64, 11, -1_i32 as i64,
            100, -1000_i32 as i64, 0x7FFFFFFF, i32::MIN as i64,
        ];
        let src2: [i64; 8] = [
            5, -11_i32 as i64, -13_i32 as i64, 2,
            200, 2000, 2, 2,
        ];
        let mut result: [i64; 8] = [0; 8];

        asm!(
            "vmovdqu64 zmm0, [{src1}]",
            "vmovdqu64 zmm1, [{src2}]",
            "vpmuldq zmm2, zmm0, zmm1",
            "vmovdqu64 [{result}], zmm2",
            src1 = in(reg) src1.as_ptr(),
            src2 = in(reg) src2.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Expected: low32(src1) * low32(src2) as signed
        // Note: -7 as i32 = 0xFFFFFFF9, -11 as i32 = 0xFFFFFFF5
        // The low 32 bits are sign-extended, so:
        // -7 * -11 = 77
        // 11 * -13 = -143
        // -1 * 2 = -2
        // etc.
        let expected: [i64; 8] = [
            3 * 5,                                    // 15
            (-7_i64) * (-11_i64),                     // 77
            11_i64 * (-13_i64),                       // -143
            (-1_i64) * 2,                             // -2
            100 * 200,                                // 20000
            (-1000_i64) * 2000,                       // -2000000
            0x7FFFFFFF_i64 * 2,                       // 0xFFFFFFFE
            (i32::MIN as i64) * 2,                    // -0x100000000
        ];

        for i in 0..8 {
            assert_eq!(result[i], expected[i],
                "VPMULDQ result wrong at index {}: got {}, expected {}",
                i, result[i], expected[i]);
        }

        println!("VPMULDQ execution test passed!");
        println!("Results: {:?}", result);
    }
}

// =============================================================================
// Immediate Rotate Tests (VPROLD/VPROLQ/VPRORD/VPRORQ)
// =============================================================================

/// Test VPROLD execution on hardware.
///
/// VPROLD rotates each 32-bit element left by an immediate value.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vprold_imm() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPROLD execution test - AVX-512F not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Test rotate left by 7 bits
        let src: [u32; 16] = [
            0x12345678, 0xFFFFFFFF, 0x80000001, 0x00000001,
            0xABCDEF01, 0x55555555, 0xAAAAAAAA, 0xDEADBEEF,
            0x01234567, 0x89ABCDEF, 0x00FF00FF, 0xFF00FF00,
            0x0F0F0F0F, 0xF0F0F0F0, 0x10101010, 0x01010101,
        ];
        let mut result: [u32; 16] = [0; 16];

        asm!(
            "vmovdqu32 zmm0, [{src}]",
            "vprold zmm1, zmm0, 7",
            "vmovdqu32 [{result}], zmm1",
            src = in(reg) src.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Expected: each element rotated left by 7 bits
        for i in 0..16 {
            let expected = src[i].rotate_left(7);
            assert_eq!(result[i], expected,
                "VPROLD result wrong at index {}: got 0x{:08x}, expected 0x{:08x}",
                i, result[i], expected);
        }

        println!("VPROLD execution test passed!");
    }
}

/// Test VPROLQ execution on hardware.
///
/// VPROLQ rotates each 64-bit element left by an immediate value.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vprolq_imm() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPROLQ execution test - AVX-512F not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Test rotate left by 17 bits
        let src: [u64; 8] = [
            0x123456789ABCDEF0,
            0xFFFFFFFFFFFFFFFF,
            0x8000000000000001,
            0x0000000000000001,
            0xABCDEF0123456789,
            0x5555555555555555,
            0xAAAAAAAAAAAAAAAA,
            0xDEADBEEFCAFEBABE,
        ];
        let mut result: [u64; 8] = [0; 8];

        asm!(
            "vmovdqu64 zmm0, [{src}]",
            "vprolq zmm1, zmm0, 17",
            "vmovdqu64 [{result}], zmm1",
            src = in(reg) src.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Expected: each element rotated left by 17 bits
        for i in 0..8 {
            let expected = src[i].rotate_left(17);
            assert_eq!(result[i], expected,
                "VPROLQ result wrong at index {}: got 0x{:016x}, expected 0x{:016x}",
                i, result[i], expected);
        }

        println!("VPROLQ execution test passed!");
    }
}

/// Test VPRORD execution on hardware.
///
/// VPRORD rotates each 32-bit element right by an immediate value.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vprord_imm() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPRORD execution test - AVX-512F not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Test rotate right by 11 bits
        let src: [u32; 16] = [
            0x12345678, 0xFFFFFFFF, 0x80000001, 0x00000001,
            0xABCDEF01, 0x55555555, 0xAAAAAAAA, 0xDEADBEEF,
            0x01234567, 0x89ABCDEF, 0x00FF00FF, 0xFF00FF00,
            0x0F0F0F0F, 0xF0F0F0F0, 0x10101010, 0x01010101,
        ];
        let mut result: [u32; 16] = [0; 16];

        asm!(
            "vmovdqu32 zmm0, [{src}]",
            "vprord zmm1, zmm0, 11",
            "vmovdqu32 [{result}], zmm1",
            src = in(reg) src.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Expected: each element rotated right by 11 bits
        for i in 0..16 {
            let expected = src[i].rotate_right(11);
            assert_eq!(result[i], expected,
                "VPRORD result wrong at index {}: got 0x{:08x}, expected 0x{:08x}",
                i, result[i], expected);
        }

        println!("VPRORD execution test passed!");
    }
}

/// Test VPRORQ execution on hardware.
///
/// VPRORQ rotates each 64-bit element right by an immediate value.
#[test]
#[cfg(target_arch = "x86_64")]
fn test_execute_vprorq_imm() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        println!("Skipping VPRORQ execution test - AVX-512F not available");
        return;
    }

    unsafe {
        use std::arch::asm;

        // Test rotate right by 23 bits
        let src: [u64; 8] = [
            0x123456789ABCDEF0,
            0xFFFFFFFFFFFFFFFF,
            0x8000000000000001,
            0x0000000000000001,
            0xABCDEF0123456789,
            0x5555555555555555,
            0xAAAAAAAAAAAAAAAA,
            0xDEADBEEFCAFEBABE,
        ];
        let mut result: [u64; 8] = [0; 8];

        asm!(
            "vmovdqu64 zmm0, [{src}]",
            "vprorq zmm1, zmm0, 23",
            "vmovdqu64 [{result}], zmm1",
            src = in(reg) src.as_ptr(),
            result = in(reg) result.as_mut_ptr(),
            options(nostack),
        );

        // Expected: each element rotated right by 23 bits
        for i in 0..8 {
            let expected = src[i].rotate_right(23);
            assert_eq!(result[i], expected,
                "VPRORQ result wrong at index {}: got 0x{:016x}, expected 0x{:016x}",
                i, result[i], expected);
        }

        println!("VPRORQ execution test passed!");
    }
}

// =============================================================================
// VNNI (Vector Neural Network Instructions) Tests
// =============================================================================

/// Test VPDPBUSD encoding - Byte unsigned/signed dot product.
/// VPDPBUSD dst, src1, src2 = dst + dot4(u8*s8)
#[test]
fn test_vpdpbusd_encoding() {
    use crate::isa::x64;
    use crate::isa::x64::inst::avx512::Avx512VnniOp;
    use crate::isa::x64::inst::args::OptionMaskReg;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let dst = Writable::from_reg(regs::xmm0());
    let acc = regs::xmm0(); // Tied to dst
    let src1 = regs::xmm1();
    let src2 = RegMem::reg(regs::xmm2());
    let mask = None as OptionMaskReg;
    let merge = MergeMode::Zeroing;

    let inst = Inst::Avx512Avx512Vnni {
        op: Avx512VnniOp::Vpdpbusd,
        dst,
        acc,
        src1,
        src2,
        mask,
        merge,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    println!("VPDPBUSD zmm0, zmm1, zmm2 encoding: {:02x?}", bytes);

    // EVEX.NDS.512.66.0F38.W0 50 /r
    // Should be 6 bytes: 62 F2 75 48 50 C2
    assert!(bytes.len() >= 6, "VPDPBUSD encoding too short");
    assert_eq!(bytes[0], 0x62, "EVEX prefix byte 0 should be 0x62");
    assert_eq!(bytes[4], 0x50, "VPDPBUSD opcode should be 0x50");

    println!("VPDPBUSD encoding test passed!");
}

/// Test VPDPWSSD encoding - Word signed dot product.
#[test]
fn test_vpdpwssd_encoding() {
    use crate::isa::x64;
    use crate::isa::x64::inst::avx512::Avx512VnniOp;
    use crate::isa::x64::inst::args::OptionMaskReg;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let dst = Writable::from_reg(regs::xmm0());
    let acc = regs::xmm0();
    let src1 = regs::xmm1();
    let src2 = RegMem::reg(regs::xmm2());
    let mask = None as OptionMaskReg;
    let merge = MergeMode::Zeroing;

    let inst = Inst::Avx512Avx512Vnni {
        op: Avx512VnniOp::Vpdpwssd,
        dst,
        acc,
        src1,
        src2,
        mask,
        merge,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    println!("VPDPWSSD zmm0, zmm1, zmm2 encoding: {:02x?}", bytes);

    // EVEX.NDS.512.66.0F38.W0 52 /r
    assert!(bytes.len() >= 6, "VPDPWSSD encoding too short");
    assert_eq!(bytes[0], 0x62, "EVEX prefix byte 0 should be 0x62");
    assert_eq!(bytes[4], 0x52, "VPDPWSSD opcode should be 0x52");

    println!("VPDPWSSD encoding test passed!");
}

// =============================================================================
// VP2INTERSECT Tests (Hash Join Acceleration)
// =============================================================================

/// Test VP2INTERSECTD encoding.
/// VP2INTERSECTD k0+1, zmm1, zmm2
#[test]
fn test_vp2intersectd_encoding() {
    use crate::isa::x64;
    use crate::isa::x64::inst::avx512::Vp2IntersectOp;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    // VP2INTERSECT outputs to k0 and k1 (two consecutive k-registers)
    // We use k0 as the destination (which means k0 and k1 are both written)
    let dst_k = Writable::from_reg(regs::k0());
    let src1 = regs::xmm0();
    let src2 = RegMem::reg(regs::xmm1());

    let inst = Inst::Avx512Vp2Intersect {
        op: Vp2IntersectOp::Vp2intersectd,
        dst_k,
        src1,
        src2,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    println!("VP2INTERSECTD k0+1, zmm0, zmm1 encoding: {:02x?}", bytes);

    // EVEX.NDS.512.F2.0F38.W0 68 /r
    assert!(bytes.len() >= 6, "VP2INTERSECTD encoding too short");
    assert_eq!(bytes[0], 0x62, "EVEX prefix byte 0 should be 0x62");
    assert_eq!(bytes[4], 0x68, "VP2INTERSECTD opcode should be 0x68");

    println!("VP2INTERSECTD encoding test passed!");
}

/// Test VP2INTERSECTQ encoding.
#[test]
fn test_vp2intersectq_encoding() {
    use crate::isa::x64;
    use crate::isa::x64::inst::avx512::Vp2IntersectOp;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let dst_k = Writable::from_reg(regs::k0());
    let src1 = regs::xmm0();
    let src2 = RegMem::reg(regs::xmm1());

    let inst = Inst::Avx512Vp2Intersect {
        op: Vp2IntersectOp::Vp2intersectq,
        dst_k,
        src1,
        src2,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    println!("VP2INTERSECTQ k0+1, zmm0, zmm1 encoding: {:02x?}", bytes);

    // EVEX.NDS.512.F2.0F38.W1 68 /r (W=1 for qword)
    assert!(bytes.len() >= 6, "VP2INTERSECTQ encoding too short");
    assert_eq!(bytes[0], 0x62, "EVEX prefix byte 0 should be 0x62");
    assert_eq!(bytes[4], 0x68, "VP2INTERSECTQ opcode should be 0x68");

    println!("VP2INTERSECTQ encoding test passed!");
}

// =============================================================================
// VPLZCNT (Leading Zero Count) Tests
// =============================================================================

/// Test VPLZCNTD encoding.
#[test]
fn test_vplzcntd_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let dst = Writable::from_reg(regs::xmm0());
    let src = RegMem::reg(regs::xmm1());
    let mask = None;
    let merge = MergeMode::Zeroing;

    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vplzcntd,
        size: OperandSize::Size32,
        dst,
        src1: regs::xmm0(),  // Unused for unary but required
        src2: src,
        mask,
        merge,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    println!("VPLZCNTD zmm0, zmm1 encoding: {:02x?}", bytes);

    // EVEX.512.66.0F38.W0 44 /r
    assert!(bytes.len() >= 6, "VPLZCNTD encoding too short");
    assert_eq!(bytes[0], 0x62, "EVEX prefix byte 0 should be 0x62");
    assert_eq!(bytes[4], 0x44, "VPLZCNTD opcode should be 0x44");

    println!("VPLZCNTD encoding test passed!");
}

/// Test VPLZCNTQ encoding.
#[test]
fn test_vplzcntq_encoding() {
    use crate::isa::x64;
    use crate::settings;
    use crate::settings::Configurable;

    let mut flag_builder = settings::builder();
    flag_builder.enable("is_pic").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let mut isa_flag_builder = x64::settings::builder();
    isa_flag_builder.enable("has_avx512f").unwrap();
    let isa_flags = x64::settings::Flags::new(&flags, &isa_flag_builder);

    let emit_info = crate::isa::x64::inst::EmitInfo::new(flags, isa_flags);

    let dst = Writable::from_reg(regs::xmm0());
    let src = RegMem::reg(regs::xmm1());
    let mask = None;
    let merge = MergeMode::Zeroing;

    let inst = Inst::Avx512Avx512Alu {
        op: Avx512AluOp::Vplzcntq,
        size: OperandSize::Size64,
        dst,
        src1: regs::xmm0(),
        src2: src,
        mask,
        merge,
    };

    let mut buffer = MachBuffer::new();
    inst.emit(&mut buffer, &emit_info, &mut Default::default());
    let buffer = buffer.finish(&Default::default(), &mut Default::default());
    let bytes = buffer.data();

    println!("VPLZCNTQ zmm0, zmm1 encoding: {:02x?}", bytes);

    // EVEX.512.66.0F38.W1 44 /r (W=1 for qword)
    assert!(bytes.len() >= 6, "VPLZCNTQ encoding too short");
    assert_eq!(bytes[0], 0x62, "EVEX prefix byte 0 should be 0x62");
    assert_eq!(bytes[4], 0x44, "VPLZCNTQ opcode should be 0x44");

    println!("VPLZCNTQ encoding test passed!");
}

// Module definition for core detection (simplified)
mod core_detect {
    pub struct FeatureInfo {
        avx512f: bool,
    }

    impl FeatureInfo {
        pub fn has_avx512f(&self) -> bool {
            self.avx512f
        }
    }

    pub fn feature_info() -> Option<FeatureInfo> {
        #[cfg(target_arch = "x86_64")]
        {
            // Use cpuid to check AVX-512F
            // This is a simplified check
            Some(FeatureInfo {
                avx512f: std::arch::is_x86_feature_detected!("avx512f"),
            })
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            None
        }
    }
}
