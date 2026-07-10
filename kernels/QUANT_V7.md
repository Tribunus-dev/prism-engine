# Prism v7 quantization: layered tile640 ternary + two-level micro-scales

Raises ternary fidelity toward **activation/output parity** while keeping the
tile640 640-weight fetch. Reference implementation + A/B harness: `tools/quant_lab.rs`
(runs anywhere). Production pipeline: `compute-core/src/compute_image/compile/ternary_pipeline.rs`
(std-only; unit-tested). Kernel: `ternary_tile640_gemv.metal` (v7 two-level scales
+ sparse outlier add-back).

## Why (the honest evidence)

From `quant_lab` (`rel_L2` = per-weight; **`act_err` = ‖Wx−W′x‖/‖Wx‖**, the metric
that predicts model quality; density = fraction of weights kept):

**Realistic weights (Gaussian + 0.3% outliers):**

| config | rel_L2 | act_err | density | bpw |
|---|---|---|---|---|
| absmax global-640 (old baseline) | 0.481 | 0.489 | 1.6% | 1.625 |
| absmean global-640 | 0.918 | 0.918 | 65% | 1.625 |
| absmean per-lane (micro) | 0.879 | 0.880 | 67% | 2.400 |
| **+ outlier 0.5% (sparse bf16)** | **0.322** | **0.324** | 87% | 2.560 |
| **FULL two-level-int8 + all** | **0.322** | **0.324** | 87% | **2.185** |

**Takeaways that shaped the defaults:**
- **Outlier extraction is the dominant lever** on real (heavy-tailed) weights: act_err **0.88 → 0.32** (2.7×). absmax "looks" ok on L2 only because it zeros ~98% of weights and keeps the high-energy outliers — useless for the bulk.
- **Two-level int8 scales are free**: identical fidelity at **2.19 bpw** vs 2.56 for flat-bf16, and they kill the fp16 overflow/underflow bug.
- **Deadzone τ and error diffusion did NOT help** on synthetic data (error diffusion *hurt*: forced ±1s add dot-product noise). They are implemented but **gated off by default** — validate on real Gemma 4 activations (Mac A/B) before enabling.
- Ternary is irreducibly lossy (best act_err ~0.32). Closing the last gap needs AWQ/GPTQ + outlier-fraction tuning on real weights.

**Evidence-backed default `QuantConfig`:** absmean + per-lane micro-scale + outlier extraction (0.5%) + two-level int8 + least-squares scale recompute; deadzone-τ<0.5 / error-diffusion / stochastic **off**.

## v7 `.cimage` format

Bump `CimageHeader.version` 6 → 7. The dense ternary segment (`TernaryWeights`,
tile640) is unchanged. Add/redefine:

| segment / field | dtype | shape | notes |
|---|---|---|---|
| `PageScales` (new) | bf16 (u16) | `[out_dim * nt]` | one page-max per 640-page |
| `LaneScales` (new) | int8 (u8) | `[out_dim * nt * 32]` | relative scale per 20-weight lane |
| `OutlierIndex` (new) | u32×2 | `[n_outliers]` | (row, col) of each extracted weight |
| `OutlierValues` (new) | bf16 (u16) | `[n_outliers]` | full-precision outlier magnitudes |

`ternary_pipeline::quantize_tensor()` returns exactly these (`packed`,
`page_scales`, `lane_scales`, `outliers`); the v7 writer serializes each to its
segment. Retire the old fp16 per-256 `BlockScales` for ternary tensors.

### Wiring into the compiler
1. Replace the `ternary_quantize_block` call site in `ternary.rs` with
   `ternary_pipeline::quantize_tensor(weights, out_dim, in_dim, &cfg)`.
2. Write the four segments above; bump the version and `segment_count`.
3. `load_v2` (cimage_loader.rs): create Metal buffers for `PageScales`,
   `LaneScales`, and the outlier arrays; assert all v7 tags present (fail loud).
4. Metal dispatch: bind buffers per `ternary_tile640_gemv` (page bf16 + lane
   int8), then a second `ternary_outlier_addback` pass over the sparse arrays
   (needs an atomic-float output accumulator, Metal 3 / Apple GPU family 7+).

## AWQ/GPTQ (feature-gated — Mac only)

Guard behind a Cargo feature `quant-awq` (off by default) so non-macOS / no-calib
builds stay green. It hooks in **before** step 3 of the pipeline: instead of
plain absmean, profile activations on a calibration set and scale/compensate the
lanes to minimize *activation* error (not per-weight L2).

- **Calibration source:** local Gemma 4 bf16 weights + a small text corpus. Run
  a forward pass, collect per-channel activation magnitudes, derive per-lane
  importance, and iteratively quantize least-important lanes first, compensating
  with the not-yet-quantized ones (GPTQ) or scaling by activation salience (AWQ).
- Cannot be validated in the Linux sandbox (needs a real forward pass) → authored
  + gated, validated by you on Mac.

## Mac A/B runbook

```bash
# 1. Baseline vs candidate compile (feature flags select the pipeline)
cd compute-core && cargo run --release --features prism-backend --bin tribunus-ecs-compile -- \
    --local-dir ~/models/gemma4-12b-it --output ~/models/g4-v7.cimage
# (add --features quant-awq once calibration is wired)

# 2. Quantizer A/B on real weight slices (dump a tensor to f32 .bin, then:)
rustc -O tools/quant_lab.rs -o /tmp/quant_lab && /tmp/quant_lab   # synthetic tables
#   extend quant_lab main() to load your .bin slice for real-weight numbers

# 3. Output parity: run both .cimage builds through the server on a fixed
#    prompt set and compare perplexity + logit MSE. The metric to trust is
#    act_err / perplexity, NOT per-weight L2.
```

Config knobs to sweep on real weights: `outlier_frac` (0.1–1%), `tau` (0.25–0.5),
`error_diffusion` on/off, `Rounding::Stochastic` vs `DeadzoneAbsmean`, and AWQ on/off.
