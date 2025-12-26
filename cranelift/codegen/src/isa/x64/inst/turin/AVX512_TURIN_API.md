# AVX-512 Turin API for Cranelift

This document describes the AVX-512 Turin (Zen 5) instruction extensions added to Cranelift
for high-performance SIMD operations targeting AMD EPYC 5th Generation processors.

## Overview

The Turin module provides native 512-bit SIMD operations using AVX-512 with EVEX encoding.
These instructions are optimized for database workloads (HTAP) and columnar data processing.

## Architecture

```
cranelift/codegen/src/isa/x64/inst/turin/
├── mod.rs       # Module exports
├── defs.rs      # Instruction and enum definitions
├── emit.rs      # Instruction emission (EVEX encoding)
├── encoding.rs  # EVEX prefix encoding
└── regs.rs      # K-register utilities
```

## Instruction Types

### 1. TurinAvx512Alu - ALU Operations

General-purpose 512-bit integer ALU operations.

```rust
Inst::TurinAvx512Alu {
    op: Avx512AluOp,      // Operation type (VPADDD, VPADDQ, etc.)
    size: OperandSize,     // Size32 or Size64
    dst: WritableReg,      // Destination ZMM register
    src1: Reg,             // First source ZMM register
    src2: RegMem,          // Second source (register or memory)
    mask: OptionMaskReg,   // Optional write mask (k1-k7)
    merge: MergeMode,      // Zeroing or Merging mode
}
```

**Available Operations:**

| Category | 32-bit | 64-bit |
|----------|--------|--------|
| Add | `Vpaddd` | `Vpaddq` |
| Subtract | `Vpsubd` | `Vpsubq` |
| Multiply | `Vpmulld` | `Vpmullq` |
| AND | `Vpandd` | `Vpandq` |
| OR | `Vpord` | `Vporq` |
| XOR | `Vpxord` | `Vpxorq` |
| AND NOT | `Vpandnd` | `Vpandnq` |
| Shift Left | `Vpslld` | `Vpsllq` |
| Shift Right Logical | `Vpsrld` | `Vpsrlq` |
| Shift Right Arithmetic | `Vpsrad` | `Vpsraq` |
| Variable Shift Left | `Vpsllvd` | `Vpsllvq` |
| Variable Shift Right Logical | `Vpsrlvd` | `Vpsrlvq` |
| Variable Shift Right Arithmetic | `Vpsravd` | `Vpsravq` |
| Min Signed | `Vpminsd` | `Vpminsq` |
| Max Signed | `Vpmaxsd` | `Vpmaxsq` |
| Min Unsigned | `Vpminud` | `Vpminuq` |
| Max Unsigned | `Vpmaxud` | `Vpmaxuq` |
| Absolute Value | `Vpabsd` | `Vpabsq` |
| Broadcast | `Vpbroadcastd` | `Vpbroadcastq` |
| Blend | `Vpblendmd` | `Vpblendmq` |
| Permute | `Vpermd` | `Vpermq` |
| Permute 2-source | `Vpermi2d` | `Vpermi2q` |
| Permute 2-source (table) | `Vpermt2d` | `Vpermt2q` |
| Conflict Detection | `Vpconflictd` | `Vpconflictq` |
| Ternary Logic | `Vpternlogd` | `Vpternlogq` |

### 2. TurinAvx512Cmp - Comparison Operations

Vector comparisons that produce a mask result in a k-register.

```rust
Inst::TurinAvx512Cmp {
    size: OperandSize,     // Size32 or Size64
    dst: WritableReg,      // Destination k-register
    src1: Reg,             // First source ZMM register
    src2: RegMem,          // Second source (register or memory)
    cond: Avx512Cond,      // Comparison condition
    mask: OptionMaskReg,   // Optional additional mask
}
```

**Comparison Conditions:**

| Condition | Encoding | Description |
|-----------|----------|-------------|
| `Eq` | 0 | Equal |
| `Lt` | 1 | Less than (signed) |
| `Le` | 2 | Less than or equal (signed) |
| `Neq` | 4 | Not equal |
| `Ge` | 5 | Greater than or equal (signed) |
| `Gt` | 6 | Greater than (signed) |

### 3. TurinCompressStore - Compress Store

Packs valid elements contiguously based on mask and stores to memory.
Critical for columnar output in database operations.

```rust
Inst::TurinCompressStore {
    size: OperandSize,     // Size32 or Size64
    src: Reg,              // Source ZMM register
    addr: SyntheticAmode,  // Destination memory address
    mask: Reg,             // k-register controlling which elements to store
}
```

**Use Case:** Writing filtered results to output buffers.

### 4. TurinExpandLoad - Expand Load

Loads sparse elements into contiguous positions based on mask.

```rust
Inst::TurinExpandLoad {
    size: OperandSize,     // Size32 or Size64
    dst: WritableReg,      // Destination ZMM register
    addr: SyntheticAmode,  // Source memory address
    mask: Reg,             // k-register controlling which positions to fill
    merge: MergeMode,      // Zeroing or Merging mode
}
```

**Use Case:** Gathering filtered data from sparse storage.

### 5. TurinMaskedLoad - Masked Load (Fault-Suppressing)

Loads data with fault suppression for invalid mask positions.

```rust
Inst::TurinMaskedLoad {
    size: OperandSize,     // Size32 or Size64
    dst: WritableReg,      // Destination ZMM register
    addr: SyntheticAmode,  // Source memory address
    mask: Reg,             // k-register controlling valid loads
    merge: MergeMode,      // Zeroing or Merging mode
}
```

**Key Feature:** Memory faults are suppressed for lanes where the mask bit is 0.
This enables safe speculative loading.

### 6. TurinMaskedStore - Masked Store

Stores data only for lanes where the mask bit is 1.

```rust
Inst::TurinMaskedStore {
    size: OperandSize,     // Size32 or Size64
    src: Reg,              // Source ZMM register
    addr: SyntheticAmode,  // Destination memory address
    mask: Reg,             // k-register controlling valid stores
}
```

### 7. TurinMaskLogic - Mask Register Logic

Operations on k-registers (mask registers).

```rust
Inst::TurinMaskLogic {
    op: MaskAluOp,         // KAND, KOR, KXOR, KNOT, KANDN
    dst: WritableReg,      // Destination k-register
    src1: Reg,             // First source k-register
    src2: OptionReg,       // Second source (None for KNOT)
}
```

**Operations:**

| Operation | Description |
|-----------|-------------|
| `Kand` | Bitwise AND |
| `Kor` | Bitwise OR |
| `Kxor` | Bitwise XOR |
| `Knot` | Bitwise NOT (unary) |
| `Kandn` | Bitwise AND NOT |

### 8. TurinKmov - Mask Register Move

Move between k-registers and GPRs.

```rust
Inst::TurinKmov {
    dst: WritableReg,      // Destination register
    src: Reg,              // Source register
    to_gpr: bool,          // true = k -> GPR, false = GPR -> k
}
```

### 9. TurinKortest - Mask Test

OR two k-registers and set CPU flags (for branching).

```rust
Inst::TurinKortest {
    src1: Reg,             // First k-register
    src2: Reg,             // Second k-register
}
```

**Use Case:** Testing if any elements match a condition.

## Merge Modes

AVX-512 supports two merge modes for masked operations:

```rust
pub enum MergeMode {
    /// Zeroing: Elements where mask bit is 0 are set to zero
    Zeroing,
    /// Merging: Elements where mask bit is 0 retain their original value
    Merging,
}
```

**Default:** `Zeroing` (recommended for most operations)

## K-Registers (Mask Registers)

AVX-512 provides 8 mask registers (k0-k7):

- **k0**: Special - means "no masking" (all elements active)
- **k1-k7**: Available for write masking

```rust
// Access k-registers via the regs module
use cranelift_codegen::isa::x64::inst::regs::{k0, k1, k2, k3, k4, k5, k6, k7};
```

## Usage Examples

### Example 1: Masked Vector Addition

```rust
// Add two vectors with masking (only process elements where mask is 1)
// Result elements where mask is 0 will be zeroed
let inst = Inst::TurinAvx512Alu {
    op: Avx512AluOp::Vpaddd,
    size: OperandSize::Size32,
    dst: writable_zmm0,
    src1: zmm1,
    src2: RegMem::reg(zmm2),
    mask: Some(k1),  // Only add where k1 bit is set
    merge: MergeMode::Zeroing,
};
```

### Example 2: Filtered Comparison (Database WHERE clause)

```rust
// Compare column values: result in k-register
let cmp_inst = Inst::TurinAvx512Cmp {
    size: OperandSize::Size32,
    dst: writable_k1,
    src1: zmm_column_data,
    src2: RegMem::reg(zmm_threshold),
    cond: Avx512Cond::Gt,  // column > threshold
    mask: None,
};

// Use mask to selectively process matching rows
let process_inst = Inst::TurinAvx512Alu {
    op: Avx512AluOp::Vpaddd,
    size: OperandSize::Size32,
    dst: writable_zmm_result,
    src1: zmm_data,
    src2: RegMem::reg(zmm_increment),
    mask: Some(k1),  // Only process matching rows
    merge: MergeMode::Zeroing,
};
```

### Example 3: Compress Store for Columnar Output

```rust
// After filtering, compress results to output buffer
let compress = Inst::TurinCompressStore {
    size: OperandSize::Size32,
    src: zmm_results,
    addr: output_buffer_addr,
    mask: k1,  // Mask from previous comparison
};
```

### Example 4: Fault-Tolerant Load

```rust
// Load with fault suppression (safe for speculative access)
let load = Inst::TurinMaskedLoad {
    size: OperandSize::Size64,
    dst: writable_zmm0,
    addr: memory_addr,
    mask: k1,  // Only load where k1 is set
    merge: MergeMode::Zeroing,
};
```

### Example 5: Combine Masks

```rust
// AND two masks together
let and_masks = Inst::TurinMaskLogic {
    op: MaskAluOp::Kand,
    dst: writable_k3,
    src1: k1,
    src2: Some(k2),
};

// Test if any bits are set
let test = Inst::TurinKortest {
    src1: k3,
    src2: k3,  // OR with itself
};
// Then branch based on ZF flag
```

## Feature Requirements

These instructions require AVX-512F (Foundation) support:

```rust
impl Inst {
    fn is_available(&self, info: &EmitInfo) -> bool {
        match self {
            Inst::TurinAvx512Alu { .. }
            | Inst::TurinAvx512Cmp { .. }
            | Inst::TurinCompressStore { .. }
            | Inst::TurinExpandLoad { .. }
            | Inst::TurinMaskedLoad { .. }
            | Inst::TurinMaskedStore { .. }
            | Inst::TurinMaskLogic { .. }
            | Inst::TurinKmov { .. }
            | Inst::TurinKortest { .. } => info.avx512f(),
            // ...
        }
    }
}
```

## Validation Functions

The module provides validation functions to catch errors early:

```rust
use cranelift_codegen::isa::x64::inst::turin::defs::{
    validate_mask_register,
    validate_active_mask_register,
    validate_avx512_operand_size,
};

// Validate k-register index (0-7)
validate_mask_register(kreg_index)?;

// Validate active mask (k1-k7, not k0)
validate_active_mask_register(kreg_index)?;

// Validate operand size for AVX-512 integer ops
validate_avx512_operand_size(&size)?;
```

## EVEX Encoding Details

All Turin instructions use EVEX encoding (4-byte prefix):

```
Byte 0: 0x62 (EVEX escape)
Byte 1 (P0): [R:1][X:1][B:1][R':1][0:1][0:1][m:2]
Byte 2 (P1): [W:1][vvvv:4][1:1][pp:2]
Byte 3 (P2): [z:1][L':1][L:1][b:1][V':1][aaa:3]
```

The `EvexPrefix` struct provides builder methods:

```rust
let prefix = EvexPrefix::new_512bit(map, w, pp)
    .with_mask(k1_index)
    .with_zeroing();

// Validate before emission
prefix.validate()?;
```

## Testing

Run the unit tests:

```bash
cargo test -p cranelift-codegen -- turin
```

Key test coverage:
- Opcode correctness vs Intel documentation
- EVEX encoding validation
- Mask register validation
- Merge mode behavior

## Performance Notes

1. **512-bit Operations**: All operations use 512-bit (ZMM) registers by default
2. **Mask-First Design**: Use k-registers for predication instead of blend instructions
3. **Compress/Expand**: Native support for columnar data operations
4. **Fault Suppression**: Masked loads safely handle speculative access

## References

- Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 2
- AMD64 Architecture Programmer's Manual, Volume 4 (128-Bit and 256-Bit Media Instructions)
- AMD EPYC 9005 Series Processor Architecture (Turin/Zen 5)
