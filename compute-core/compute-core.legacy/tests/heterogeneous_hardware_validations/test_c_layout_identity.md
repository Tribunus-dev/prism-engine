# Test C: Cross-Lane IOSurface Layout Identity

**Compiler decision**: Whether `SharedFp16ActivationContract` must include a
`semantic_axes` + `IndexMapping` proof, or whether "same IOSurface, same FP16
element type" is sufficient for cross-lane correctness.

## Null Hypothesis

A single IOSurface containing FP16 data can be read correctly by both a Metal
GPU kernel (interpreting as `device half*` row-major `[B, S, H]`) and an ANE
Core ML program (interpreting as `[1, C, 1, S]` IOSurface) without any data
reorganization, for the common case where H = C, S = spatial.

## Experimental Design

### Step 1: Fill an IOSurface with a known pattern

Write a deterministic FP16 test pattern to a shared IOSurface:
- Pattern: each element = `f16(hidden_index * 1.0 + seq_index * 0.1)`
- Shape: `[1, hidden=4096, 1, seq=64]` (ANE convention) or equivalently
  `[1, 64, 4096]` (Metal row-major [B, S, H])

This pattern has a known mathematical structure: the value at (s, h) encodes
both its sequence position and its channel position.

### Step 2: Read via Metal kernel

Write a minimal Metal compute kernel that:
1. Binds the IOSurface as a `texture_buffer<half, access::read>`
2. Also binds the same IOSurface as a `device half*` buffer (via
   `IOSurfaceGetBaseAddress`)
3. Reads each element at (s, h) and writes to a CPU-readable output buffer
4. Records the byte offset at which each logical element (s, h) was read

Output: for each logical position (s, h), the physical byte offset in the
IOSurface and the value read.

### Step 3: Read via Core ML program

Write a trivial MIL program that:
1. Takes the same IOSurface as input (`[1, hidden, 1, seq]`)
2. Passes it through identity (or a single `mul(x, 1.0)`)
3. Outputs to another IOSurface

After prediction, read the output buffer and compare element-by-element.

### Step 4: Read via CPU (Accelerate)

Map the same IOSurface via `IOSurfaceGetBaseAddress`, cast to `float16_t*`,
and read elements at expected (s, h) positions assuming both possible layouts:
- ANE-style: `offset = (s * hidden + h) * 2`
- Metal row-major: `offset = (s * hidden + h) * 2` (same for NHWC-like vs row-major
  when the logical shape is [B, S, H] with no extra dimensions)

But test with different shapes to find the divergence point:
| Logical shape | ANE physical (C×S) | Metal row-major (S×H) | Equivalent? |
|--------------|--------------------|-----------------------|-------------|
| [1, 16, 1, 64] | 16×64 | 64×16 | NO — transposed |
| [1, 64, 1, 4096] | 64×4096 | 64×4096 | YES — H=C, S=seq |
| [1, 4096, 1, 64] | 4096×64 | 64×4096 | NO — C≠S |
| [1, 32, 8, 128] (n_heads, head_dim) | 32×(8×128) | 8×128×32 | NO — head layout differs |

### Step 5: Compare index mappings

For each shape pair, compute:
- Does Metal's read address (texture_buffer or device pointer) match ANE's
  read address for the same logical element?
- If not: what is the discrepancy pattern? (transpose, strided, different axis
  ordering?)

## Acceptance Criteria

| Result | Interpretation | Compiler Action |
|--------|----------------|-----------------|
| All three lanes read identical values at identical byte offsets for all tested shapes | No IndexMapping needed — IOSurface is a universal interchange format | `SharedFp16ActivationContract` can skip `IndexMapping` for common shapes |
| Shapes where H=C, S=seq match, but head×head_dim shapes differ | Index equivalence holds for simple shapes but breaks for multi-dimensional tensors | `IndexMapping` required when `semantic_axes` has >3 logical axes |
| No shape produces identical byte-level readings across all three lanes | IOSurface is NOT a zero-cost interchange format | `boundary_kind` must be `RepackRequired` unless compiler inserts explicit conversion |

## Required Evidence

For each (logical_shape, physical_layout) pair, a table showing:
- `(s, h)` logical element → byte offset in IOSurface for each lane
- Whether the value read at that offset matches the written pattern
- The maximum absolute error between any two lanes' readings

## Hardware Requirements

- Apple Silicon Mac (M1+)
- Metal-capable (all Apple Silicon)
- Core ML with ANE support (macOS 14+ recommended)
- IOSurface framework (included in macOS)

## Expected Outcomes (Hypothesis)

For the simplest case where ANE's `[1, C, 1, S]` happens to have C = hidden
and S = seq, the byte layout for a FP16 element matches Metal's row-major
`[S, H]`. For shapes involving attention heads (Q, K, V projections where
the logical shape has head_dim as a separate axis), the ANE's 4D tensor is
flattened differently than Metal's 2D/3D view, and the index mapping diverges.
This means `IndexMapping` is required for attention and multi-head projections
but may be elided for simple FFN layers where the activation is `[1, H, 1, S]`
and S corresponds to the batch/sequence dimension.
