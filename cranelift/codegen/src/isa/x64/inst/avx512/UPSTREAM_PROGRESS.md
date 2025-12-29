# AVX-512 Upstream Readiness Progress

## Status: IN PROGRESS

Last Updated: 2025-12-28

---

## Critical Issues

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| C1 | Missing ZMM16-31 registers | DONE | Fully implemented! 32 registers now available |
| C2 | Vendor-specific branding | DONE | Removed Turin/Zen 5/AMD EPYC from avx512.isle + tests |
| C3 | Missing CLIF filetests | PENDING | Add compile-only tests |
| C4 | Platform gating missing | DONE | Added #![cfg(target_arch)] to all 3 test files |
| C5 | Operand corrections | DONE | Fixed VCVTPD2PS/VCVTPD2DQ/VCVTTPD2DQ to use zmm_m512 |

## High Priority Issues

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| H1 | Missing unsigned comparisons | DEFER | Future work - not needed for initial merge |
| H2 | Missing AVX-512DQ guard | DONE | Added has_avx512dq to vextracti64x2/vinserti64x2 rules |
| H3 | Rule formatting | DONE | Fixed 246 rules (rule 17/18 spacing) |
| H4 | VCVTPS2PD tuple type | DONE | Fixed to Half, also fixed VCVTDQ2PD |
| H5 | Move benchmarks | DEFER | Organizational - not blocking |
| H6 | Missing edge case tests | DEFER | Future work - tests cover happy paths |

## Medium Priority Issues

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| M1 | Run cargo fmt | DONE | Formatting fixed |
| M2 | Add #[must_use] | DONE | Added to Kmask new/enc/to_string |
| M3 | Document priority scheme | DONE | Already in avx512.isle header |
| M4 | K-register allocation | DEFER | Complex - future work |
| M5 | Missing instructions | DEFER | VMOVAPS/VMOVAPD - future work |
| M6 | Remove println in tests | DEFER | Benchmarks need output |

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
