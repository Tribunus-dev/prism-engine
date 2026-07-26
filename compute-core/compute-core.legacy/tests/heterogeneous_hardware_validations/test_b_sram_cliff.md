# Test B: ANE SRAM Cliff Onset and Spill Penalty

**Compiler decision**: Whether `VariantSteadyStateCost` needs a nonlinear
`spill_penalty_ns` term beyond linear `memory_ns`, and whether the planner
must know ANE SRAM capacity per SoC family.

## Null Hypothesis

ANE inference latency scales linearly with model memory footprint (weights + activations). There is no step-function throughput collapse at any specific memory threshold.

## Experimental Design

Build a set of `.mlmodelc` programs whose total weight + activation footprint
ranges from 4 MB to 256 MB in controlled increments. Measure per-token latency
and identify the onset of any nonlinear slowdown.

### Step 1: Construct programs with graduated SRAM pressure

Create a parameterized MIL program family:
- Input: `[1, hidden, 1, seq]` FP16
- Weight: `[hidden, hidden]` Dense FP16 matmul
- Output: `[1, hidden, 1, seq]` FP16

Vary `hidden` dimension to sweep memory footprint:

| hidden | Weight MB (FP16) | Activation MB (seq=64) | Total MB | Expected SRAM status |
|--------|------------------|------------------------|----------|---------------------|
| 512    | 0.5              | 0.03                   | ~0.5     | Well within SRAM |
| 1024   | 2.0              | 0.13                   | ~2.1     | Within SRAM |
| 2048   | 8.0              | 0.5                    | ~8.5     | Within SRAM |
| 4096   | 32.0             | 2.1                    | ~34      | At SRAM boundary |
| 6144   | 72.0             | 4.7                    | ~77      | Exceeds SRAM |
| 8192   | 128.0            | 8.4                    | ~136     | Far exceeds SRAM |

Run two sub-tests per hidden size:

**Sub-test B1: Single matmul latency**
- Load a single `matmul(hidden, hidden)` program
- Measure per-invocation latency (100 runs, discard warmup)

**Sub-test B2: Sequential pipeline pressure**
- Chain 3 identical matmul programs (A→B→C) sharing IO surfaces
- Measure end-to-end latency for the chain
- Compare to 3× single-matmul latency
- If spill occurs, the chain cost will be sub-linear (spill amortized) or
  super-linear (inter-program SRAM conflict)

### Step 2: Profile with varying sequence length

Repeat B1 and B2 with seq = 1, 16, 64, 256, 512 to see if activation
footprint combines with weight footprint to trigger spill at different points.

### Step 3: Palettized vs dense weight comparison

For hidden=4096 and hidden=8192, repeat B1 with:
- Direct FP16 weights (baseline)
- LUT4-palettized weights (constexpr_lut_to_dense, decompressed at load)

Is the decompressed weight FP16 memory footprint the same as direct FP16?
Does palettization change the spill onset point?

## Acceptance Criteria

| Observation | Interpretation | Compiler Action |
|-------------|----------------|-----------------|
| Latency increases linearly with hidden size across all tested sizes | No SRAM cliff; linear memory model is sufficient | Cost model can use linear `memory_ns` |
| Latency jumps ≥30% at a specific hidden×seq combination | SRAM cliff confirmed | Add `spill_penalty_ns` = measured jump to `VariantSteadyStateCost` |
| Palettized weights spill at same point as direct FP16 | Decompressed weight has same footprint — palettization saves artifact size, not runtime memory | `resident_bytes` in `VariantPrepareCost` = dense weight size, not palette size |
| Palettized weights spill LATER than direct FP16 | ANE may keep palette in SRAM and decompress on-the-fly | `resident_bytes` = palette size only (update cost model) |

## Required Evidence

CSV output with columns:
`hidden, seq, weight_format, program_count, mean_latency_ns, p50_ns, p95_ns, memory_footprint_bytes`

Plus a scatter plot annotation identifying any observed discontinuity.

## Hardware Requirements

- Apple Silicon Mac (M1 Pro or better recommended for wider SRAM range)
- M4 Max results preferred (largest SRAM — 32MB) but M1 also informative (~24MB)
- macOS 14.0+ for Core ML 7.x

## Expected Outcomes (Hypothesis)

Based on maderix's measurement of ~32MB SRAM on M4 Max:
- Sub-32MB: linear scaling, ~0.5-1 TFLOPS
- Above 32MB: ~30% throughput drop, visible as a latency discontinuity
- The cliff onset depends on WEIGHT + activation, not either alone
- Palettized decompression at load time means the dense weight occupies SRAM,
  so palettization does NOT delay the cliff — it only reduces artifact size
