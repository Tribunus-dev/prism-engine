# Test A: Core ML masked SDPA Fidelity vs Orion Direct Path

**Compiler decision**: Whether the compiler must decompose masked SDPA into
Q@K^T → additive mask → softmax → @V for ANE-resident attention, or whether
Core ML's opaque `scaled_dot_product_attention` with causal mask is safe.

## Null Hypothesis

Causal masks passed to Core ML's `scaled_dot_product_attention` MIL op with
`cpuAndNeuralEngine` compute policy produce bit-identical results to the
same model run on CPU-only, for all tested sequence lengths and batch sizes.

## Experimental Design

Three program variants, each fed identical Q, K, V, and causal mask data:

| Variant | Route | Mask mechanism |
|---------|-------|----------------|
| A_cpu_only | CPU-only (`computeUnits = .cpuOnly`) | MIL SDPA with `causal_mask` arg |
| A_ane_opaque | Same .mlmodelc, `computeUnits = .cpuAndNeuralEngine` | Same MIL SDPA with `causal_mask` arg |
| A_decomposed | ANE manually decomposed: Q@K^T → add mask → softmax → @V, `cpuAndNeuralEngine` | Explicit MIL `add(mask)` before softmax |

### Step 1: Build three .mlmodelc files

Each .mlmodelc contains a single attention block:
- Input: Q [1, n_heads, seq, head_dim], K [1, n_kv_heads, seq, head_dim], V [1, n_kv_heads, seq, head_dim]
- Causal mask: [1, 1, seq, seq] with 0 in lower triangle, -inf in upper
- Output: O [1, n_heads, seq, head_dim]

Shapes to test:
- seq = 16, 64, 256, 1024, 4096
- head_dim = 64, 128
- n_heads = 8, 32

Variants A_cpu_only and A_ane_opaque share the same .mlmodelc (same MIL SDPA).
Variant A_decomposed has MIL-level explicit Q@K^T → add → softmax → @V.

### Step 2: Run inference on each

For each variant and shape:
1. Load the .mlmodelc with the specified compute policy
2. Run 100 predictions (discard first as warmup)
3. Record per-token output [1, n_heads, seq, head_dim]

### Step 3: Compare

Compute `max_abs_error` and `cosine_similarity` between pairs:

```
error(A_ane_opaque, A_cpu_only)   # Does Core ML SDPA with mask survive ANE dispatch?
error(A_decomposed, A_cpu_only)    # Does manual decomposition match CPU reference?
error(A_decomposed, A_ane_opaque)  # Do the two ANE routes agree with each other?
```

Also run Orion's direct private-API path (if available) on the same Q/K/V/mask:
```
error(A_orion_direct, A_cpu_only) # Does Orion's direct path show mask failure?
```

### Step 4: Profile placement

Use `MLComputePlan.loadContentsOfURL` + `deviceUsage(for:)` on each MIL operation
to determine which ops land on which device for each variant and compute policy.

## Acceptance Criteria

| Condition | Pass | Fail | Compiler Action |
|-----------|------|------|-----------------|
| `error(A_ane_opaque, A_cpu_only) < 1e-3` for all seq | Core ML handles mask correctly through its path | Core ML SDPA mask is incorrect | Classify as `CorrectnessQualifiedButPlacementUnknown` |
| `error(A_decomposed, A_cpu_only) < 1e-3` for all seq | Manual decomposition is correct | Decomposition has numerical issue | Qualify decomposition as ANE-performance variant |
| `A_orion_direct` diverges but `A_ane_opaque` matches | Orion's direct path is not Core ML's | — | Document: private-API path has separate constraints from Core ML |
| `A_ane_opaque` SDPA op shows `deviceUsage = neuralEngine` | Masked SDPA executes on ANE | SDPA op lands on CPU | Measure throughput to decide if manual decomposition wins |

## Required Evidence

A single JSON output `test_a_results.json` containing:
- For each (model_path, compute_policy, seq, head_dim): output tensor path, mean latency (100 runs)
- For each comparison pair: max_abs_error, cosine_similarity, matching_token_count (top-1 argmax match rate)
- Device usage breakdown per operation from MLComputePlan

## Hardware Requirements

- Apple Silicon Mac (M1 or later)
- macOS 14.0+ (for MLComputePlan API)
- Xcode Command Line Tools (for coremlcompiler)
- Orion repo available locally for comparison run (optional)

## Expected Outcomes (Hypothesis)

Based on ANEMLL's Gemma3 results, we expect:
- A_ane_opaque ≈ A_cpu_only for all seq (Core ML path handles mask correctly)
- A_orion_direct ≠ A_cpu_only for seq > 1 (Orion's direct path ignores causal mask)
- The SDPA op lands on CPU in A_ane_opaque (Core ML routes masked SDPA to CPU for correctness)

If A_ane_opaque SDPA lands on neuralEngine AND matches CPU, then Core ML has
fixed the mask handling internally and Orion's finding does not apply to Core ML.
If SDPA lands on CPU, the throughput benefit of manual decomposition for ANE
becomes a real optimization target.
