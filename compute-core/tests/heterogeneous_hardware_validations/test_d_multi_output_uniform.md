# Test D: Multi-Output Uniform Buffer Requirement

**Compiler decision**: Whether the compiler must pad multi-output ANE program
slots to uniform allocation size, or whether Core ML handles this internally.

## Null Hypothesis

Core ML's `scaled_dot_product_attention` and multi-output `matmul` programs
tolerate output IOSurfaces with different allocation sizes when executed on
ANE with `cpuAndNeuralEngine` policy.

## Experimental Design

### Step 1: Build multi-output MIL programs

Three program families:

| Program | Outputs | Output shapes | Allocation strategy |
|---------|---------|---------------|---------------------|
| D_qkv_natural | Q, K, V | Q=[n_heads×hd], K=[n_kv×hd], V=[n_kv×hd] | Natural sizes (K, V smaller than Q) |
| D_qkv_padded | Q, K, V | Same logical shapes | All padded to Q size |
| D_attn_natural | O, K, V (stateful) | O=[seq,hd], K=[seq,hd], V=[seq,hd] | Natural sizes |
| D_attn_padded | O, K, V | Same logical shapes | All padded to max |

### Step 2: Run with natural allocation

Load and run D_qkv_natural and D_attn_natural with both compute policies:
- `cpuAndNeuralEngine`
- `cpuOnly` (reference)

Check for:
- `ANEProgramProcessRequestDirect() Failed with status=0x1d` (Orion's observed failure)
- Or: Core ML returns successful results

### Step 3: Compare padded vs natural output

For any program that runs successfully with natural allocation:
- Compare padded output (only first `natural_bytes` per output) vs natural output
- `max_abs_error` and `cosine_similarity`

### Step 4: Vary the size disparity

Test with increasing output size ratios:
- 1:1 (equal — should always work)
- 1:2 (K is half of Q)
- 1:4
- 1:8
- 1:16

## Acceptance Criteria

| Result | Interpretation | Compiler Action |
|--------|----------------|-----------------|
| All natural-allocation programs fail with status=0x1d on ANE | Core ML does NOT pad automatically | Slots MUST be padded to uniform size in cimage resource plan |
| All natural-allocation programs succeed with correct outputs | Core ML handles size matching internally | No compiler action needed |
| Some succeed, some fail depending on ratio | Unstable — must pad to be safe | Emit padded slots and verify on each SoC generation |

## Required Evidence

For each (program, compute_policy, ratio): success/failure, output
max_abs_error vs cpuOnly reference, Core ML error code string on failure.

## Hardware Requirements

- Apple Silicon Mac (M1+)
- macOS 14.0+

## Expected Outcomes (Hypothesis)

Core ML's constexpr ops and runtime handle buffer size matching internally
for the public API path. Orion's failure was specific to its direct `_ANEProgram`
path that bypasses Core ML's buffer management. D_qkv_natural will succeed on
Core ML but fail on Orion's direct path. This means the compiler does NOT need
to pad for Core ML, but SHOULD pad if supporting direct ANE dispatch.
