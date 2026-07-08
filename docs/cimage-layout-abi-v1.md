# CImage Layout ABI v1

> Logical tensor ↔ physical tile layout ↔ lane-specific execution view.
> Quantization decides how many bits. Layout decides whether those bits
> arrive in the right lane in the right order.

## 1. Three-Layer Model

The cimage separates three concerns that are currently conflated in a single
tile640 block descriptor.

### 1.1 LogicalTensor

The model-level contract: what this tensor means, not how it is stored.

```
LogicalTensor:
  key:         "model.layers.0.self_attn.q_proj.weight"
  class:       Gemma4.DecoderAttentionProjection
  shape:       [4096, 3840]            # [out_features, in_features]
  logical_op:  Matmul { contract: InputAxis, produce: OutputAxis }
  data_type:   Weight
  orientation: ColumnMajor             # W logically [out_f, in_f]
  boundary:    DecoderInternal
```

- `class` maps to compiler_policy.json rules.
- `logical_op` records the operation contract so the layout resolver
  knows which axis is the reduction axis (input) and which is the
  accumulation axis (output).
- `boundary` classifies the tensor's role in the computation graph.

### 1.2 PhysicalTileLayout

How the tensor is actually laid out in the cimage payload segment.

```
PhysicalTileLayout:
  format:              NF4
  tile_family:         Tile640
  tile_shape:          [640, 640]       # elements per tile
  storage_order:       RowMajor          # output-major within tile
  group_size:          32
  group_axis:          PackedContiguous  # ← critical: resolves axis ambiguity
  metadata_layout:     AdjacentTile     # scales/biases inline after codes
  padding_policy:      ZeroPadToTile
  alignment_bytes:     256               # segment alignment for mmap
  interleave:          None
```

- `tile_family` declares the kernel contract (Tile640, Tile256, Tile1024).
  The number is NOT a global constant — it is a parameter.
- `tile_shape` gives the exact logical dimensions within each tile.
  Currently always [640, 640] for the Tile640 family, but permits
  rectangular tiles if profiling shows a win.
- `group_axis` replaces the implicit axis assumption. Options:

  ```
  group_axis:
    PackedContiguous   # contiguous in storage order (current behavior)
    OutputAxis         # groups span output-index space
    InputAxis          # groups span input-index space
    TileLocal          # groups do not cross tile boundaries
  ```

  For decoder projections the default is `PackedContiguous` (which happens
  to be output-major). For patch_dense the cimage should record
  `InputAxis` so the kernel knows to dequant in input-contiguous groups.

### 1.3 ExecutionView

A lane-specific materialization of the tensor for a specific execution backend.

```
ExecutionView:
  lane:                MetalFusedDecode
  derived_from:        PhysicalTileLayout
  data_offset:         1048576           # offset into cimage segment
  data_length:         524288            # bytes
  metadata_offset:     0
  metadata_length:     16384
  codec_overrides:     { }               # lane-specific param tweaks
  repacking_required:  false
  residency:           AlwaysMapped
```

A single tensor may have multiple ExecutionViews. The cimage manifest
selects the active view set per hardware target class.

## 2. Hardware Target Classes

Across Apple Silicon the variables that matter for cimage layout are
memory bandwidth, SLC/cache, GPU generation, ANE path, memory capacity,
thermal envelope, and die topology.

### Target class definitions

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Target Class    │ Memory │ Bandwidth │ GPU Cores │ Chip Type       │
  ├─────────────────┼────────┼───────────┼───────────┼─────────────────┤
  │ A18Neo          │  8 GB  │   ~60 GB/s│ 5         │ Single die      │
  │ MBase           │ 16 GB  │  ~120 GB/s│ 8-10      │ Single die      │
  │ MPro            │ 24 GB  │  ~273 GB/s│ 16-20     │ Single die      │
  │ MMax            │ 48 GB  │  ~546 GB/s│ 32-40     │ Single die      │
  │ MUltra          │ 96 GB+ │  ~800 GB/s│ 64-80     │ Multi-die       │
  └─────────────────────────────────────────────────────────────────────┘
```

### Layout preference per class

**A18Neo** (MacBook Neo / iPhone)
- Priority: smallest resident set, FP16/INT8 fallback, minimal KV
- Max one execution view per tensor
- No RawF32 except where absolutely validated
- Cap context at 2048
- TTS optionally offloaded (not resident)

**MBase** (MacBook Air, base Mac mini)
- Priority: byte reduction, fused GPU decode, avoid scratch spikes
- Scratch budget: 512 MB
- Prefer fused Metal fused-decode over general matmul
- Max one execution view per tensor

**MPro** (MacBook Pro base, Mac mini Pro)
- Priority: balanced GPU/ANE split, larger tile batches
- Scratch budget: 1 GB
- Allow up to 2 views per tensor if useful
- Can overlap GPU and ANE lane

**MMax** (MacBook Pro high-end, Mac Studio)
- Priority: wide streaming tiles, aggressive fused kernels
- Scratch budget: 2 GB
- Allow Metal Tensor API view alongside custom fused kernels
- Profiler chooses winner per tensor

**MUltra** (Mac Pro, Mac Studio Ultra)
- Priority: shard/stripe large tensors, avoid cross-die churn
- Scratch budget: 4 GB
- Prefer layer locality — keep a layer's working set coherent
- Allow layer-range sharding across dies

## 3. Tensor-Class Layout Rules

The cimage config carries per-family layout rules rather than per-tensor
heuristics. These rules are in `compiler_policy.json` alongside the codec
evidence.

```json
{
  "rule": "Gemma4.DecoderMlpProjection",
  "codec": "NF4",
  "tile_family": "Tile640",
  "group_size": 32,
  "group_axis": "PackedContiguous",
  "execution_views": ["metal_fused_decode"],
  "validation_axes": ["OutputRow"],
  "boundary": "DecoderInternal"
}
```

```json
{
  "rule": "Gemma4.VisionPatchProjection",
  "match_keys": ["patch_dense"],
  "codec": "RawF32",
  "group_axis": "InputAxis",
  "execution_views": ["metal_prefill_fallback"],
  "validation_axes": ["InputColumn", "OutputRow"],
  "boundary": "ModalityBridge"
}
```

```json
{
  "rule": "Qwen3TTS.Projection",
  "codec": "INT8",
  "tile_family": "Tile640",
  "group_size": 128,
  "group_axis": "PackedContiguous",
  "execution_views": ["coreml_ane", "metal_fallback"],
  "boundary": "TtsInternal"
}
```

The `boundary` field controls handoff behavior. Cross-boundary tensors
(modality bridges, codec heads, LM head) should prefer ANE-compatible
layout or simple contiguous FP16. Decoder-internal tensors should
prefer the GPU fused kernel layout.

The `validation_axes` field tells the sweep which axis attribution to
collect. If `validation_axes = ["InputColumn"]` and the sweep finds
input-column accumulation exceeding operator max-abs, it flags a
hard failure instead of a soft weight-NRMSE rejection.

## 4. Hardware Layout Profiles

Each hardware target class has a profile that constrains the cimage.

```json
{
  "apple_a18_neo_tiny": {
    "max_resident_views_per_tensor": 1,
    "prefer_fp16_over_rawf32": true,
    "max_context_default": 2048,
    "allow_tts_resident": false,
    "scratch_budget_mb": 256
  },
  "apple_m_base_memory_bound": {
    "max_resident_views_per_tensor": 1,
    "prefer_fused_gpu_decode": true,
    "scratch_budget_mb": 512
  },
  "apple_m_pro_balanced": {
    "max_resident_views_per_tensor": 2,
    "allow_ane_gpu_overlap": true,
    "scratch_budget_mb": 1024
  },
  "apple_m_max_bandwidth": {
    "larger_tile_experiments": true,
    "allow_tensor_api_view": true,
    "scratch_budget_mb": 2048
  },
  "apple_m_ultra_sharded": {
    "allow_layer_sharding": true,
    "prefer_layer_locality": true,
    "scratch_budget_mb": 4096
  }
}
```

## 5. Residency Modes

```pub enum ResidencyMode {
    /// Always mapped into GPU-addressable memory.
    AlwaysMapped,
    /// Mapped on first access, kept resident.
    LazyMapped,
    /// Materialized on first use (may involve repacking).
    MaterializeOnFirstUse,
    /// Scratch buffer, not persisted across sessions.
    EphemeralScratch,
    /// Mutually exclusive with another view — only one resident at a time.
    /// Useful for memory-bound targets where GPU and ANE views coexist
    /// in the payload but only one is executed per session.
    MutuallyExclusiveViewGroup(String),
}
```

For A18Neo, the constraint `max_resident_views_per_tensor = 1`
means all views beyond the first for a given tensor must be
`LazyMapped` or `MutuallyExclusiveViewGroup`.

## 6. Compiler Pipeline

The three-stage pipeline replaces the current monolithic codec decision:

```
Tensor source
    │
    ▼
PolicyResolver:
  ── selects admissible codec from evidence
  ── consults compiler_policy.json per tensor_class
  ── outputs CodecCandidate { family, group_size, codebook }
    │
    ▼
LayoutResolver:
  ── selects tile_family based on hardware target + tensor_class
  ── assigns group_axis, storage_order
  ── picks metadata layout and alignment
  ── may produce multiple candidate layouts
  ── outputs PhysicalTileLayout
    │
    ▼
ExecutionPlanner:
  ── selects execution lanes (GPU/ANE/CPU)
  ── assigns ResidencyMode per view
  ── may produce ExecutionView for each lane
  ── manages mutually exclusive view groups
    │
    ▼
CimageSegment
```

The sweep becomes a cross-product: PolicyResolver candidates ×
LayoutResolver candidates × ExecutionPlanner candidates. The
profile runner measures the result and feeds back to the
evidence system.

## 7. Remaining tile640 naming

The codebase currently hardcodes `TILE_ELEMENTS = 640` in
`nf4tile640/mod.rs`. This should become a parameter:

```rust
pub struct TileFamily {
    pub name: &'static str,       // "Tile640", "Tile256", "Tile1024"
    pub tile_elements: u32,       // 640
    pub tile_rows: u32,           // 640 (may differ from cols for rectangular)
    pub tile_cols: u32,           // 640
    pub default_group_sizes: Vec<u32>, // [32, 64, 128]
}
```

For the current release the tile parameterization can default to
`TileFamily::tile640()` everywhere. The type exists so M5 Max can
later request `TileFamily::tile1024()` without invasive refactors.

## 8. Validation Requirements

A cimage is valid when:

1. Every tensor has a LogicalTensor entry matching the source model.
2. Every tensor has at least one PhysicalTileLayout.
3. Every PhysicalTileLayout has a matching ExecutionView for at least
   one lane on the target hardware.
4. No tensor's total resident bytes exceed the target's scratch budget.
5. No two ExecutionViews for the same tensor are simultaneously resident
   if they belong to the same MutuallyExclusiveViewGroup.
6. The `group_axis` field is non-ambiguous — no two code paths interpret
   it differently.
7. Every tensor archived as RawF32 has a one-line explanation in the
   cimage manifest citing the rejection reason (e.g. "operator max-abs
   tail exceeds INT8 gate — see evidence receipt abc123").
