# AVX-512 Upstream Readiness Progress

## Status: IN PROGRESS

Last Updated: 2025-12-28

---

## Critical Issues

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| C1 | Missing ZMM16-31 registers | DONE | Fully implemented! 32 registers now available |
| C2 | Vendor-specific branding | DONE | Removed Turin/Zen 5/AMD EPYC from avx512.isle + tests |
| C3 | Missing CLIF filetests | DONE | Added avx512.clif with 17 compile tests |
| C4 | Platform gating missing | DONE | Added #![cfg(target_arch)] to all 3 test files |
| C5 | Operand corrections | DONE | Fixed VCVTPD2PS/VCVTPD2DQ/VCVTTPD2DQ to use zmm_m512 |

## High Priority Issues

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| H1 | Missing unsigned comparisons | DONE | Added VPCMPUD/VPCMPUQ support for I32X16/I64X8 |
| H2 | Missing AVX-512DQ guard | DONE | Added has_avx512dq to vextracti64x2/vinserti64x2 rules |
| H3 | Rule formatting | DONE | Fixed 246 rules (rule 17/18 spacing) |
| H4 | VCVTPS2PD tuple type | DONE | Fixed to Half, also fixed VCVTDQ2PD |
| H5 | Move benchmarks | DONE | Already in correct location (cranelift/jit/tests/) |
| H6 | Missing edge case tests | DEFER | Future work - tests cover happy paths |

## Medium Priority Issues

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| M1 | Run cargo fmt | DONE | Formatting fixed |
| M2 | Add #[must_use] | DONE | Added to Kmask new/enc/to_string |
| M3 | Document priority scheme | DONE | Already in avx512.isle header |
| M4 | K-register allocation | DEFER | Complex - future work |
| M5 | Missing instructions | DONE | Added VMOVAPS/VMOVAPD/VMOVUPS/VMOVUPD (512-bit) |
| M6 | Remove println in tests | DONE | Removed redundant PASS/PASSED messages |

## Low Priority Issues

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| L1 | Format name consolidation | DEFER | Style preference - not blocking |
| L2 | VL variants | DEFER | 128/256-bit EVEX - future work |
| L3 | Gather/scatter support | DEFER | Future work |

---

## Test Results

| Test Suite | Status | Count |
|------------|--------|-------|
| cranelift-assembler-x64 | ✅ PASS | 16 tests |
| cranelift-codegen | ✅ PASS | 218 tests |
| avx512_sql_ops | ✅ PASS | 22 tests |
| avx512_e2e | ✅ PASS | 17 tests |
| avx512_perf | ✅ PASS | 8 tests |

---

## Change Log

### 2025-12-28
- Created tracking document
- C1: **FULLY IMPLEMENTED ZMM16-31 register support!**
  - Added XMM16-31 encoding constants to xmm.rs
  - Added xmm16()-xmm31() register constructors to regs.rs
  - Added `enable_simd32` shared setting to conditionally expose registers
  - Updated abi.rs to add xmm16-31 to allocatable pool when enabled
  - This doubles the available SIMD registers for AVX-512 code!
- C2: Removed vendor branding (Turin/Zen5/AMD EPYC) from avx512.isle and tests
- C4: Added platform gating (#![cfg(target_arch)]) to all 3 test files
- C5: Fixed VCVTPD2PS/VCVTPD2DQ/VCVTTPD2DQ to use zmm_m512 for memory operands
- H2: Added has_avx512dq guard to vextracti64x2/vinserti64x2 ISLE rules
- H3: Fixed 246 ISLE rules with proper spacing (rule 17 vs rule 17)
- H4: Fixed VCVTPS2PD and VCVTDQ2PD tuple type from Full to Half
- M1: Ran cargo fmt
- M2: Added #[must_use] to Kmask methods
- Fixed fuzz test vcmpps/vcmpngeps/vpcmp mnemonic comparison issues
- All tests passing (265+ total)

### 2025-12-28 (cont'd)
- Fixed OnceLock static naming in abi.rs (PINNED_SIMD32_ENV, PINNED_ENV, SIMD32_ENV, DEFAULT_ENV)
- C3: Added CLIF filetests at cranelift/filetests/filetests/isa/x64/avx512.clif
  - 17 functions testing: iadd, isub, imul, band, bor, bxor, fadd, fsub, fmul, fdiv, sqrt, ineg
  - Covers i64x8, i32x16, f64x8, f32x16 types

### 2025-12-29
- H1: Added unsigned integer comparisons (VPCMPUD/VPCMPUQ)
  - Added x64_512_vpcmpud/vpcmpuq wrappers in inst.isle
  - Added specific helpers: vpcmpultd, vpcmpuled, vpcmpugtd, vpcmpuged (32-bit)
  - Added specific helpers: vpcmpultq, vpcmpuleq, vpcmpugtq, vpcmpugeq (64-bit)
  - Added lowering rules for UnsignedGreaterThan, UnsignedLessThan, etc.
- H5: Confirmed benchmarks already in correct location (cranelift/jit/tests/)
- M5: Added floating-point move instructions
  - VMOVAPS/VMOVAPD (aligned, 512-bit)
  - VMOVUPS/VMOVUPD (unaligned, 512-bit)
- M6: Cleaned up test output
  - Removed 109 redundant "PASSED" println statements from avx512_e2e.rs
  - Removed "PASS" statements from avx512_sql_ops.rs
  - Kept informational "Skipping" and benchmark output messages
