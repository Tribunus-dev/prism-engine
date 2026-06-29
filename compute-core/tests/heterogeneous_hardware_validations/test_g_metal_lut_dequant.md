# Test G: Metal LUT Dequantization Shader Viability

**Compiler decision**: Whether `WeightEncoding::PalettizedLut4` with
`MetalShaderLutDequant` can be marked as `Planned` vs `Supported`, and
what the performance profile looks like when implemented.

## Null Hypothesis

A Metal compute shader that decompresses LUT4-palettized weights (16-entry
per-channel codebook + 4-bit indices) and performs FP16 matrix multiplication
can match or exceed the throughput of loading pre-decompressed FP16 weights
from device memory.

## Experimental Design

### Step 1: Implement the Metal kernel

Write a Metal shader with two variants:

**Variant G_dequant_separate:**
```
kernel void dequant_lut4(
    device const uint8_t* indices   [[buffer(0)]],
    device const half* codebook     [[buffer(1)]],
    device half* output             [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint out_channel = gid.y;   // output channel (row)
    uint in_idx = gid.x;        // input column

    uint byte = indices[out_channel * in_dim/2 + in_idx/2];
    uint index = (in_idx & 1) ? (byte >> 4) : (byte & 0xF);
    output[out_channel * in_dim + in_idx] = codebook[out_channel * 16 + index];
}
```

**Variant G_fused_matmul:**
A single kernel that reads palette and indices, dequantizes on-the-fly
in the inner loop, and accumulates the matmul output — avoiding the
intermediate decompressed weight write to device memory:

```
kernel void lut4_matmul(
    device const half* input         [[buffer(0)]],
    device const uint8_t* indices    [[buffer(1)]],
    device const half* codebook      [[buffer(2)]],
    device half* output              [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;  // output channel
    half acc = 0;
    for (uint k = 0; k < in_dim; k++) {
        uint byte = indices[row * in_dim/2 + k/2];
        uint idx = (k & 1) ? (byte >> 4) : (byte & 0xF);
        half w = codebook[row * 16 + idx];
        acc += input[k] * w;
    }
    output[row] = acc;
}
```

### Step 2: Benchmark against baseline

Compare three approaches:

| Variant | Description |
|---------|-------------|
| G_direct_f16 | Standard matmul with FP16 weight buffer (no dequant) |
| G_dequant_separate | Dequantize to FP16 buffer, then matmul (two kernels) |
| G_fused_matmul | Single kernel: dequant+matmul fused |

Metrics per variant:
- Kernel dispatch latency (GPU time, not including CPU submission)
- Peak memory bandwidth utilization (estimated from bytes read / duration)
- Total CPU→GPU submission overhead

### Step 3: Vary matrix dimensions

Test at:
- hidden=4096, intermediate=11008 (standard LLM MLP)
- hidden=2048, intermediate=8192
- hidden=512, intermediate=2048

At each size:
- Does the fused kernel memory access pattern (random-ish palette lookups vs
  sequential weight read) cause cache thrashing?
- Does the separate dequant+matmul win due to better cache behavior?

### Step 4: Profile with Xcode GPU counters

If available, capture:
- L1/L2 cache hit rate
- ALU utilization
- Memory bandwidth utilization
- Thread occupancy

## Acceptance Criteria

| Result | Interpretation | Compiler Action |
|--------|----------------|-----------------|
| G_fused_matmul within 20% of G_direct_f16 throughput | Metal LUT dequant is viable | Mark `MetalShaderLutDequant` as `Supported` with measured cost |
| G_fused_matmul 2×+ slower than G_direct_f16 | Fused dequant has prohibitive overhead | Keep as `Planned`; require fusing optimization (e.g., tile-sized dequant) |
| G_dequant_separate matches G_direct_f16 | Dequant+matmul is viable but uses intermediate memory | Add intermediate buffer cost to `VariantSteadyStateCost::memory_ns` |
| All variants fail or produce wrong output | LUT format indices do not match Metal's expected layout | Investigate: byte ordering, channel alignment |

## Required Evidence

For each (variant, hidden, intermediate): kernel_duration_us, p50_ns, p95_ns,
bytes_read, estimated_bandwidth_gbps, output max_abs_error vs FP16 reference matmul.

## Hardware Requirements

- Apple Silicon Mac (any)
- Metal 3.0+ (macOS 13.0+)
- Xcode GPU frame capture for profiling (optional but recommended)

## Expected Outcomes (Hypothesis)

- G_direct_f16 will be fastest (no indirection, sequential memory access).
- G_fused_matmul will be 1.3-1.8× slower due to irregular palette lookup
  pattern causing L1 cache misses (the codebook is small per-channel but
  the index→codebook indirection prevents prefetch).
- G_dequant_separate will be ~1.5-2× slower overall (two kernel dispatches +
  intermediate buffer write + read).
- The gap narrows at larger matrix sizes where compute dominates memory.
- Therefore: `MetalShaderLutDequant` should be `Planned`, not `Supported`,
  until a tiled approach (dequant a tile, matmul tile, repeat) is implemented
  and benchmarked.
