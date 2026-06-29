# Test F: Prepare vs Steady-State Cost Separation

**Compiler decision**: Whether `VariantPrepareCost` and `VariantSteadyStateCost`
must be separate types in the compiler cost model, and what representative
values are for each component.

## Null Hypothesis

The cost of weight decompression via `constexpr_lut_to_dense` is negligible
compared to per-token compute cost, and does not need to be modeled separately.

## Experimental Design

### Step 1: Build three .mlmodelc variants of the same MLP block

Program: `x → gate_proj → SiLU → up_proj → mul → down_proj → output`
hidden=4096, intermediate=11008 (standard LLM MLP).

| Variant | Weight format | Expected artifact size |
|---------|--------------|----------------------|
| F_direct_f16 | Direct FP16 weights | ~180 MB |
| F_lut4 | LUT4-palettized (constexpr_lut_to_dense) | ~45 MB |
| F_lut6 | LUT6-palettized | ~68 MB |

### Step 2: Measure load time

For each variant:
```
let t0 = Instant::now();
let model = try_load_mlmodelc(path, computePolicy);
let t_load = t0.elapsed();
```

Repeat 10 times (cold start — clear Core ML cache between runs).

### Step 3: Measure memory footprint

Before and after loading:
```
let rss_before = get_resident_bytes();
let model = load(...);
let rss_after = get_resident_bytes();
let footprint = rss_after - rss_before;
```

### Step 4: Measure per-token inference latency

For each variant, after warmup (10 predictions):
```
let latencies: Vec<Duration> = (0..100).map(|_| {
    let t0 = Instant::now();
    model.predict(...);
    t0.elapsed()
}).collect();
```

### Step 5: Sweep artifact size vs load time

Compute `disk_size = path.metadata().len()` and compare to load time:
- Does LUT4 load faster (less disk I/O) or slower (decompression cost)?
- Is the tradeoff worth it for a server that loads once and serves millions of tokens?

## Acceptance Criteria

| Metric | Interpretation | Compiler Action |
|--------|----------------|-----------------|
| F_lut4 load_time >> F_direct_f16 load_time | Decompression at load time is significant | `prepare_cost.dense_materialization_ns` = measured delta |
| F_lut4 footprint ≈ F_direct_f16 footprint | Decompressed weights occupy same memory as direct | `resident_bytes` = dense weight size, NOT palette size |
| F_lut4 footprint << F_direct_f16 footprint | ANE keeps palette compressed in SRAM | `resident_bytes` = palette size; update SRAM cliff model |
| F_lut4 per-token latency < F_direct_f16 per-token latency | Palettization saves compute bandwidth | Favor palettized variants in cost model |
| F_lut4 per-token latency ≈ F_direct_f16 latency | No compute benefit; palette is just storage optimization | Palettization affects prepare cost only |

## Required Evidence

Table:
```
variant | disk_mb | load_time_ms | rss_delta_mb | p50_latency_us | p95_latency_us | tflops_est
F_direct_f16 | ... | ... | ... | ... | ... | ...
F_lut4       | ... | ... | ... | ... | ... | ...
F_lut6       | ... | ... | ... | ... | ... | ...
```

Plus `prepare_cost` and `steady_state_cost` breakdown for each.

## Hardware Requirements

- Apple Silicon Mac (M1+)
- macOS 14.0+
- ~2GB free disk space for .mlmodelc artifacts

## Expected Outcomes (Hypothesis)

- `F_lut4` artifact is ~4× smaller than `F_direct_f16` (16 palette entries per
  output channel = 2 bytes per weight value vs 2 bytes for direct FP16, but
  palette codebook adds overhead; expected compression ratio ~3-4×).
- `F_lut4` load time is ~2× longer (decompression at load).
- `F_lut4` RSS ≈ `F_direct_f16` RSS (decompressed weights occupy same memory).
  This contradicts the intuition that palettization saves runtime memory —
  it only saves artifact size on disk. The 32MB SRAM cliff applies equally
  to both formats.
- Per-token latency is roughly equal (the matmul dominates, and both paths
  produce the same FP16 weight matrix).
- Therefore: palettization is a DISK/LOAD-TIME optimization (faster download,
  smaller artifact storage), NOT a runtime memory or compute optimization
  (unless ANE can keep the palette compressed in SRAM, which this test disproves
  or confirms).
