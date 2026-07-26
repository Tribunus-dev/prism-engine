# Supported Configurations

## Supported Models

|Model Family|Status|Notes|
|---|---|---|
|Gemma 4 12B Unified|Production (megakernel)|Full 48-layer transformer via persistent GPU kernel. Enabled by `prism-backend` + `metal-dispatch`. Runtime-compiled from `megakernel/shaders/gemma4_full.metal` (2972 lines) or precompiled metallib.|
|Gemma 4 12B QAT|Experimental|INT4 variant via `megakernel/shaders/gemma4_full_int4.metal`. NV24-packed ternary weights.|
|Qwen2.5 variants|Experimental|GGUF ingestion works (`compile/source.rs`). Metal dispatch wrappers exist for per-layer inference but full pipeline not validated end-to-end.|
|Qwen3-TTS|Experimental|Audio codec + talker pipeline. Requires TTS-specific megakernel support (`tts/talker.rs`).|
|Llama-family|Experimental (v0)|CImage v0 format supports synthetic shards. No end-to-end Llama compilation verified.|
|BitNet b1.58|Experimental|Ternary ECS pipeline exists (`bitnet/`). Reference implementation verified in isolation.|

## Supported Backends

|Backend|Status|Feature Gate|Notes|
|---|---|---|---|
|Metal (Apple GPU)|Production|`metal-dispatch`|Megakernel (persistent transformer) and per-layer dispatch via `CImageMetalRegionRunner`. RawF32, NF4 tile640, Ternary tile640, INT8 tile640 kernels exist as `.metal` templates.|
|ANE (Apple Neural Engine)|Experimental|`ane`|ANE eligibility checking, MIL builder, IOSurface pipeline exist. No end-to-end ANE compilation validated for a full model.|
|MLX|Experimental|`mlx-backend`|MLX executor exists but is feature-gated and research-surface. Used primarily for teacher-forcing in distillation.|
|CPU (reference)|Production (reference only)|`backend-cpu`|CPU reference kernels exist for all quantization formats. Used for differential testing. Not intended for production inference.|
|ROCm (AMD)|Not yet|`amd-rocm`|Module is empty stub.|
|Intel Level Zero|Not yet|`level-zero-probe`|Probe-only, no execution.|

## Supported Quantization Formats

|Format|Status|Policy|Notes|
|---|---|---|---|
|RawF32|Production|Default fallback|All tensors stored as IEEE 754 single-precision. Guaranteed correct.|
|NF4 Tile640 (g128)|Experimental|Policy-disabled|Full codec exists (`nf4tile640/`). Metal GEMV kernel exists. Calibration suite exists. Disabled in `compiler_policy.json` — operator gate fails for some tensors.|
|Ternary Tile640|Experimental|Policy-disabled (`enabled: false`)|Full codec exists (`ternary/`). Metal GEMV kernel exists. ECS admission pipeline has NRMSE screening. Disabled because substitution.enabled=false.|
|INT8 Tile640|Experimental|Policy-disabled|Kernel exists in templates. Codec partially implemented. Policy shows `rawf32_required` for most tensors.|
|oQ (Observed Quantization)|Not yet|N/A|todo!() stub.|

## Known Limitations (from PR A assessment)

1. **Metal RawF32 kernel has a correctness bug**: The `cimage_linear_rawf32` kernel produces anti-correlated output (cosine -0.18) vs CPU reference when exercised through the staged-kernel runner (`CImageMetalRegionRunner`). Root cause unknown — likely a buffer layout or dimension mismatch. Test is quarantined with `#[ignore]`.
2. **11 todo!() sites**: Core paths like speculative decode, KV runtime, SSD cache, and oQ quantization are stubs that now return errors instead of panicking.
3. **Multiple Metal compilation paths**: Five distinct paths exist (megakernel, template codegen, fusion lowering, dispatch wrappers, AOT stub) — no single canonical Metal compiler.
4. **ECS pipe not wired**: The ECS compilation pipeline is structurally clean but does not produce cimage artifacts.

## Compilation Pipeline Status

```text
Source → canonical model IR → rep planning → exec graph → fusion → backend lowering → kernel compile → cimage packaging → verify → seal
  GGUF    ModelIr              ─ not yet ─     ─ not yet ─   existing   5 paths exist   compile/pipeline    test gap    not yet
  HF      (PR B)              (PR B)           (PR B)        (PR C)     (PR D)          cimage_packer       v0          (PR H)
```

The current working path is: GGUF → compile/pipeline.rs + cimage_packer/pipeline.rs → RawF32 cimage → manual deployment.
