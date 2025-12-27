# AVX-512 Cranelift IR Extension Plan

## Overview

This document describes the Cranelift IR extensions needed for full AVX-512 support,
optimized for columnar database workloads on AMD EPYC Turin (Zen 5) processors.

## Design Principles

1. **Blend Pattern**: Use a single `x86_avx512_blend` op that fuses with arithmetic ops
2. **Minimal New Opcodes**: Reuse existing IR ops where possible (iadd, isub, etc.)
3. **Mask as Integer**: Represent k-register masks as I8 (8 lanes) or I16 (16 lanes)
4. **Target-Specific**: New ops are prefixed with `x86_avx512_` to indicate x86-only

## Vector Types (Already Exist in Cranelift)

```
512-bit types:
  I64X8   - 8 x i64 (primary for columnar i64/timestamp/decimal)
  I32X16  - 16 x i32 (primary for columnar i32)
  F64X8   - 8 x f64 (for columnar double)
  F32X16  - 16 x f32 (for columnar float)
  I8X64   - 64 x i8 (for string/byte operations)
  I16X32  - 32 x i16 (for short integers)

Mask types (use integers):
  I8      - mask for 8-lane vectors (I64X8, F64X8)
  I16     - mask for 16-lane vectors (I32X16, F32X16)
  I32     - mask for 32-lane vectors (I16X32)
  I64     - mask for 64-lane vectors (I8X64)
```

---

## New IR Opcodes

### 1. Blend (Masked Select) - THE KEY OPERATION

```
x86_avx512_blend(mask: Ix, if_true: VEC, if_false: VEC) -> VEC
```

**Semantics**: For each lane i, select if_true[i] if mask bit i is 1, else if_false[i].

**Fusion Rules** (in ISLE):
```lisp
;; blend(mask, iadd(a,b), passthrough) -> VPADDD{k}
;; blend(mask, isub(a,b), passthrough) -> VPSUBD{k}
;; blend(mask, band(a,b), passthrough) -> VPANDD{k}
;; ... etc for all arithmetic ops
```

**Zeroing Mode**: `blend(mask, op(a,b), zero_vector)` -> op{k}{z}

**Usage**:
```rust
let sum = builder.ins().iadd(a, b);
let result = builder.ins().x86_avx512_blend(mask, sum, passthrough);
// Fuses to: VPADDQ zmm{k}, zmm, zmm
```

---

### 2. Vector Compare to Mask

```
x86_avx512_icmp(cc: IntCC, a: VEC, b: VEC) -> Ix (mask)
x86_avx512_fcmp(cc: FloatCC, a: VEC, b: VEC) -> Ix (mask)
```

**Semantics**: Compare lanes, produce mask with 1 for true, 0 for false.

**Maps to**: VPCMPD, VPCMPQ, VCMPPS, VCMPPD

**Usage**:
```rust
let mask = builder.ins().x86_avx512_icmp(IntCC::SignedGreaterThan, col, threshold);
// VPCMPQ k1, zmm0, zmm1, 6  (6 = NLE = GT)
```

---

### 3. Compress (Filter Operation)

```
x86_avx512_vcompressd(src: I32X16, mask: I16) -> I32X16
x86_avx512_vcompressq(src: I64X8, mask: I8) -> I64X8
```

**Semantics**: Pack elements where mask bit is 1 to the low lanes. High lanes undefined.

**Popcount for count**: Use `popcnt` on mask to get number of valid elements.

**Maps to**: VPCOMPRESSD, VPCOMPRESSQ

**Usage** (columnar filter):
```rust
let mask = builder.ins().x86_avx512_icmp(IntCC::SignedGreaterThan, col, threshold);
let compressed = builder.ins().x86_avx512_vcompressq(col, mask);
let count = builder.ins().popcnt(mask);  // existing IR op
builder.ins().store(flags, compressed, out_ptr, 0);
```

---

### 4. Expand (Inverse of Compress)

```
x86_avx512_vexpandd(src: I32X16, mask: I16) -> I32X16
x86_avx512_vexpandq(src: I64X8, mask: I8) -> I64X8
```

**Semantics**: Read consecutive elements from src, scatter to lanes where mask bit is 1.

**Maps to**: VPEXPANDD, VPEXPANDQ

---

### 5. Gather (Indexed Load)

```
x86_avx512_vpgatherdd(base: I64, indices: I32X16, mask: I16) -> I32X16
x86_avx512_vpgatherdq(base: I64, indices: I32X8, mask: I8) -> I64X8
x86_avx512_vpgatherqd(base: I64, indices: I64X8, mask: I8) -> I32X8
x86_avx512_vpgatherqq(base: I64, indices: I64X8, mask: I8) -> I64X8
```

**Semantics**: Load values from base + indices[i] * scale for each lane where mask is 1.

**Maps to**: VPGATHERDD, VPGATHERDQ, VPGATHERQD, VPGATHERQQ

**Usage** (columnar lookup):
```rust
let indices = builder.ins().load(I32X16, ...);  // row IDs
let mask = /* all ones or computed */;
let values = builder.ins().x86_avx512_vpgatherdd(table_ptr, indices, mask);
```

---

### 6. Scatter (Indexed Store)

```
x86_avx512_vpscatterdd(base: I64, indices: I32X16, src: I32X16, mask: I16)
x86_avx512_vpscatterdq(base: I64, indices: I32X8, src: I64X8, mask: I8)
x86_avx512_vpscatterqd(base: I64, indices: I64X8, src: I32X8, mask: I8)
x86_avx512_vpscatterqq(base: I64, indices: I64X8, src: I64X8, mask: I8)
```

**Semantics**: Store src[i] to base + indices[i] * scale for each lane where mask is 1.

**Maps to**: VPSCATTERDD, VPSCATTERDQ, VPSCATTERQD, VPSCATTERQQ

---

### 7. Mask Manipulation (K-register Operations)

```
x86_avx512_kand(a: Ix, b: Ix) -> Ix     ; AND masks
x86_avx512_kandn(a: Ix, b: Ix) -> Ix    ; AND-NOT masks (a & ~b)
x86_avx512_kor(a: Ix, b: Ix) -> Ix      ; OR masks
x86_avx512_kxor(a: Ix, b: Ix) -> Ix     ; XOR masks
x86_avx512_knot(a: Ix) -> Ix            ; NOT mask
x86_avx512_ktest(a: Ix) -> flags        ; Test mask (for branching)
```

**Note**: These could also just use regular `band`, `bor`, `bxor`, `bnot` on integer types
and let the lowering figure it out. But explicit k-ops may generate better code.

**Alternative**: Just use existing integer ops on mask values:
```rust
let combined_mask = builder.ins().band(mask1, mask2);  // lowers to KAND or AND
```

---

### 8. Broadcast/Splat (Already Exists)

```
splat(scalar) -> VEC   ; existing Cranelift op
```

**Usage**:
```rust
let threshold_vec = builder.ins().splat(I64X8, threshold_scalar);
```

---

### 9. Load/Store 512-bit (Use Existing Ops)

```
load(I64X8, MemFlags, ptr, offset) -> I64X8    ; existing
store(MemFlags, value: I64X8, ptr, offset)     ; existing
```

**Lowering**: These should lower to VMOVDQU64 / VMOVDQU32 for 512-bit types.

---

## Fusion Rules Summary (ISLE)

```lisp
;;;; Blend + Arithmetic -> Masked Arithmetic

;; I64X8 operations
(rule (lower (x86_avx512_blend mask (iadd a b) passthrough))
      (if-let $I64X8 (value_type a))
      (turin_alu_masked Vpaddq a b mask passthrough Merging))

(rule (lower (x86_avx512_blend mask (iadd a b) (splat _ (iconst 0))))
      (if-let $I64X8 (value_type a))
      (turin_alu_masked Vpaddq a b mask _ Zeroing))

;; I32X16 operations
(rule (lower (x86_avx512_blend mask (iadd a b) passthrough))
      (if-let $I32X16 (value_type a))
      (turin_alu_masked Vpaddd a b mask passthrough Merging))

;; ... similar rules for isub, imul, band, bor, bxor, ishl, ushr, sshr, etc.

;;;; Unmasked 512-bit arithmetic (no blend)
(rule (lower (iadd a b))
      (if-let $I64X8 (value_type a))
      (if-let true (has_avx512f))
      (turin_alu Vpaddq a b))

(rule (lower (iadd a b))
      (if-let $I32X16 (value_type a))
      (if-let true (has_avx512f))
      (turin_alu Vpaddd a b))
```

---

## Complete Opcode List

### New IR Opcodes to Add

| Opcode | Operands | Result | Description |
|--------|----------|--------|-------------|
| `x86_avx512_blend` | mask, if_true, if_false | VEC | Masked select (fuses with arithmetic) |
| `x86_avx512_icmp` | cc, a, b | Ix | Integer vector compare to mask |
| `x86_avx512_fcmp` | cc, a, b | Ix | Float vector compare to mask |
| `x86_avx512_vcompressd` | src, mask | I32X16 | Compress 32-bit elements |
| `x86_avx512_vcompressq` | src, mask | I64X8 | Compress 64-bit elements |
| `x86_avx512_vexpandd` | src, mask | I32X16 | Expand 32-bit elements |
| `x86_avx512_vexpandq` | src, mask | I64X8 | Expand 64-bit elements |
| `x86_avx512_vpgatherdd` | base, idx, mask | I32X16 | Gather 32-bit with 32-bit indices |
| `x86_avx512_vpgatherdq` | base, idx, mask | I64X8 | Gather 64-bit with 32-bit indices |
| `x86_avx512_vpgatherqq` | base, idx, mask | I64X8 | Gather 64-bit with 64-bit indices |
| `x86_avx512_vpscatterdd` | base, idx, src, mask | void | Scatter 32-bit with 32-bit indices |
| `x86_avx512_vpscatterdq` | base, idx, src, mask | void | Scatter 64-bit with 32-bit indices |
| `x86_avx512_vpscatterqq` | base, idx, src, mask | void | Scatter 64-bit with 64-bit indices |

### Existing IR Ops That Need 512-bit Lowering

| Existing Op | 512-bit Lowering |
|-------------|------------------|
| `iadd` | VPADDD (I32X16), VPADDQ (I64X8) |
| `isub` | VPSUBD (I32X16), VPSUBQ (I64X8) |
| `imul` | VPMULLD (I32X16), VPMULLQ (I64X8) |
| `band` | VPANDD (I32X16), VPANDQ (I64X8) |
| `bor` | VPORD (I32X16), VPORQ (I64X8) |
| `bxor` | VPXORD (I32X16), VPXORQ (I64X8) |
| `ishl` | VPSLLD (I32X16), VPSLLQ (I64X8) |
| `ushr` | VPSRLD (I32X16), VPSRLQ (I64X8) |
| `sshr` | VPSRAD (I32X16), VPSRAQ (I64X8) |
| `smin` | VPMINSD (I32X16), VPMINSQ (I64X8) |
| `smax` | VPMAXSD (I32X16), VPMAXSQ (I64X8) |
| `umin` | VPMINUD (I32X16), VPMINUQ (I64X8) |
| `umax` | VPMAXUD (I32X16), VPMAXUQ (I64X8) |
| `load` | VMOVDQU32/64 |
| `store` | VMOVDQU32/64 |
| `splat` | VPBROADCASTD/Q |

---

## Usage Examples for Columnar DB

### Example 1: Filtered Aggregation (SUM WHERE col > 100)

```rust
fn emit_filtered_sum(builder: &mut FunctionBuilder, col_ptr: Value, len: Value) -> Value {
    let zero = builder.ins().iconst(I64, 0);
    let zero_vec = builder.ins().splat(I64X8, zero);
    let threshold = builder.ins().iconst(I64, 100);
    let threshold_vec = builder.ins().splat(I64X8, threshold);
    let eight = builder.ins().iconst(I64, 8);

    let mut accumulator = zero_vec;
    let mut i = zero;

    // Loop over 8 elements at a time
    loop {
        let chunk = builder.ins().load(I64X8, MemFlags::trusted(), col_ptr, i);

        // Generate mask: col > 100
        let mask = builder.ins().x86_avx512_icmp(IntCC::SignedGreaterThan, chunk, threshold_vec);

        // Masked add: accumulator += chunk WHERE mask
        let sum = builder.ins().iadd(accumulator, chunk);
        accumulator = builder.ins().x86_avx512_blend(mask, sum, accumulator);

        i = builder.ins().iadd(i, eight);
        // ... loop condition
    }

    // Horizontal sum of accumulator lanes
    // ... reduce I64X8 to I64
}
```

### Example 2: Hash Table Probe (Gather)

```rust
fn emit_hash_probe(builder: &mut FunctionBuilder,
                   keys: Value,        // I64X8 - 8 keys to probe
                   table_ptr: Value,   // base of hash table
                   ) -> Value {
    // Compute hash indices (simplified)
    let indices = builder.ins().band(keys, /* table_mask */);
    let indices_i32 = builder.ins().ireduce(I32X8, indices);

    // Gather values from hash table
    let all_ones = builder.ins().iconst(I8, 0xFF);
    let values = builder.ins().x86_avx512_vpgatherdq(table_ptr, indices_i32, all_ones);

    values
}
```

### Example 3: Filter + Compress (SELECT WHERE)

```rust
fn emit_filter_compress(builder: &mut FunctionBuilder,
                        input_ptr: Value,
                        output_ptr: Value,
                        predicate_val: Value,  // threshold
                        ) -> Value /* count */ {
    let chunk = builder.ins().load(I64X8, MemFlags::trusted(), input_ptr, 0);
    let threshold_vec = builder.ins().splat(I64X8, predicate_val);

    // Generate predicate mask
    let mask = builder.ins().x86_avx512_icmp(IntCC::SignedGreaterThan, chunk, threshold_vec);

    // Compress matching elements
    let compressed = builder.ins().x86_avx512_vcompressq(chunk, mask);

    // Store compressed result
    builder.ins().store(MemFlags::trusted(), compressed, output_ptr, 0);

    // Return count of matching elements
    let mask_i64 = builder.ins().uextend(I64, mask);
    builder.ins().popcnt(mask_i64)
}
```

---

## Implementation Order

1. **Phase 1: Basic 512-bit Arithmetic**
   - Add lowering rules for iadd, isub, imul, band, bor, bxor on I64X8/I32X16
   - Add load/store lowering for 512-bit types
   - Test with simple vector operations

2. **Phase 2: Blend + Fusion**
   - Add `x86_avx512_blend` IR opcode
   - Add ISLE fusion rules for blend + arithmetic
   - Test masked operations

3. **Phase 3: Comparison**
   - Add `x86_avx512_icmp` and `x86_avx512_fcmp`
   - Test predicate generation

4. **Phase 4: Compress/Expand**
   - Add compress/expand opcodes
   - Test filter patterns

5. **Phase 5: Gather/Scatter**
   - Add gather/scatter opcodes
   - Test hash table patterns

---

## Files to Modify

1. `cranelift/codegen/meta/src/shared/instructions.rs` - Add new IR opcodes
2. `cranelift/codegen/meta/src/shared/formats.rs` - Add instruction formats if needed
3. `cranelift/codegen/src/isa/x64/lower.isle` - Add unmasked 512-bit lowering rules
4. `cranelift/codegen/src/isa/x64/lower/isle/turin.isle` - Add masked/AVX-512 specific lowering
5. `cranelift/codegen/src/isa/x64/inst/turin/emit.rs` - Already has emission (may need updates)
6. Tests in `cranelift/codegen/src/isa/x64/inst/turin/runtime_tests.rs`
