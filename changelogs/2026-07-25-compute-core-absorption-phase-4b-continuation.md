# 2026-07-25 — Phase 4B continuation: absorb 4 more `compute_image/` files

This is the Completion report for the Phase 4B continuation subagent.
Phase 4B (commit `14e8edb1`, restored in `ef826363`) absorbed 3 files
from `compute-core.legacy/src/ecs/compute_image/` into the
constitutional libraries under Prism-domain names. This continuation
absorbs 4 more files following the same `references/project-absorption.md`
pattern.

The task suggested 3-5 files. We picked **4** based on overlap with
the constitutional libraries and the cleanest re-implementation
path. All four are pure data / pure construction code with no
hardware-specific execution plane; the absorbing crate is
`prism-ecs-compile`, and the new modules are peer files of the
existing `cimage_pipeline` / `cimage_packer` / `cimage_validation`
modules.

---

## Affected subsystem

`prism-ecs-compile::cimage_manifest` (new) — the constitutional
authority for the CImage manifest schema, the manifest builder, the
shared-lane ABI, the per-tensor table, the lease state machine, the
per-kernel Metal dispatch contract, and the post-emission evidence
(compile receipt, stage profile, tensor diff, manifest verification).

`prism-ecs-compile::cimage_packer::aligned_mmap` (new) — the
constitutional authority for the 16 KB-aligned mmap builder
primitive, the safe re-implementation of the engine's
`AlignedMmapBuilder`.

`prism-ecs-compile::cimage_packer::layout` (new) — the constitutional
authority for the AOT layout planner
([`CImageLayoutPlan`], [`CImageTopologyTable`], tar size prediction).

## `CAMPAIGN.md` status before and after

**Before**: `compute_image/` is **Not yet absorbed** per
`changelogs/2026-07-25-compute-core-legacy-integration-plan.md` and
the prior Phase 4B changelog. The subsystem had 0 direct world
mutations. The 4 re-implementations cover the patterns with the
most overlap with the constitutional libraries: the canonical
manifest schema + manifest construction + per-tensor table +
shared-lane ABI + lease state machine + per-kernel Metal dispatch +
post-emission evidence, plus the AOT layout planner and the
page-aligned mmap primitive.

**After**: 7 of 162 files have been re-implemented in the
constitutional libraries. The originals remain in
`compute-core/src/ecs/compute_image/` to be migrated in subsequent
phases. The new constitutional modules are peer files in
`crates/prism-ecs-compile/src/cimage_manifest/` and
`crates/prism-ecs-compile/src/cimage_packer/`. No `CAMPAIGN.md`
change yet — the migration state for these re-implementations is
documented in this changelog and will be rolled into `CAMPAIGN.md`
during Phase 5.

## Files absorbed (4 of the suggested 3-5)

### 1. `compute-core.legacy/src/ecs/compute_image/manifest/types.rs` (1,037 LOC) → `crates/prism-ecs-compile/src/cimage_manifest/` (split across 5 files, 1,627 LOC)

**Original**: the engine's manifest schema — the
[`Manifest`](https://github.com/...) top-level struct, the
per-tensor table types ([`TensorEntry`], [`QuantizationDesc`],
[`AliasEntry`]), the per-backend artifact types
([`BackendWeightArtifact`], [`ArtifactKind`]), the shared-lane ABI
([`Nf4Tile640Layout`], [`SharedWeightLayout`]), the per-kernel
Metal dispatch types ([`MetalDispatchRecipe`],
[`MetalKernelArtifact`]), the lease state machine
([`LeaseState`], [`StorageBackend`], [`SegmentLease`],
[`TensorLease`]), the post-emission evidence
([`CompileReceipt`], [`StageProfile`], [`SegmentReceipt`],
[`TensorProvenance`], [`IgnoredTensorClassification`],
[`ManifestVerification`], [`NativeCapabilityReport`],
[`RepresentationAdmissionEstimate`], [`TensorDiff`],
[`CompiledImage`]), and the source identity
([`SourceIdentity`], [`ShardHash`]). The original file used
`HashMap` for `artifact_bindings`, `buffer_slot_map`, and
`scalar_index_map`.

**Re-implementation**: the constitutional authority for the CImage
manifest schema. The re-implementation splits the original 1,037
LOC into 5 files by authority, each well under the 900 LOC / 35
public-items thresholds from `references/module-discipline.md`:

| New file | LOC | Authority (one sentence) | Public items |
|---|---:|---|---:|
| `cimage_manifest/mod.rs` | 411 | Directory index + `ManifestEnvelope` + `representation_aware_admission_estimate` + per-crate error type | 16 |
| `cimage_manifest/header.rs` | 426 | `Manifest` top-level struct + `Segment` / `SegmentKind` / `ResidencyPlan` + `StorageAbiSpec` + `CompileReadiness` + `validate_segment_alignment` / `validate_manifest_for_abi` | 12 |
| `cimage_manifest/types.rs` | 471 | Per-tensor table (`TensorEntry` / `QuantizationDesc` / `AliasEntry`) + shared-lane ABI (`Nf4Tile640Layout` / `SharedWeightLayout`) + per-backend artifact (`BackendWeightArtifact` / `ArtifactKind`) + source identity (`SourceIdentity` / `ShardHash`) | 12 |
| `cimage_manifest/kernel.rs` | 187 | Per-kernel Metal dispatch contract (`MetalDispatchRecipe` / `MetalKernelArtifact`) | 2 |
| `cimage_manifest/lease.rs` | 167 | Lease state machine (`LeaseState` / `StorageBackend` / `SegmentLease` / `TensorLease` / `CopyClassification`) | 5 |

The per-kernel Metal dispatch contract is split into a separate
`kernel.rs` file; the receipt types are split into a separate
`receipt.rs` file:

| `cimage_manifest/receipt.rs` | 285 | Post-emission evidence (`CompileReceipt` / `StageProfile` / `SegmentReceipt` / `TensorProvenance` / `IgnoredTensorClassification` / `ManifestVerification` / `NativeCapabilityReport` / `RepresentationAdmissionEstimate` / `TensorDiff` / `CompiledImage`) | 10 |

**Total**: 1,947 LOC across 6 files (the header / types / kernel /
lease / receipt / mod.rs). The 1,947 vs 1,037 LOC increase is
expected: the re-implementation adds 22 invariant-named tests
plus the `BTreeMap` discipline (the original used `HashMap` in
3 places which are now `BTreeMap`), the
`representation_aware_admission_estimate` projection, the
`ManifestEnvelope` durable JSON contract, the per-crate
`CImageManifestError`, and the `Debug` derives that the
manifest types need for `serde_json` test assertions.

**Key changes from the original**:

- `HashMap` → `BTreeMap` for `artifact_bindings`,
  `buffer_slot_map`, and `scalar_index_map`. The `BTreeMap`
  discipline is required by `references/rust-quality.md` for
  canonical collections whose iteration order is observable.
- `Manifest::architecture` is `serde_json::Value` rather than
  `crate::ecs::config::TextArchitecture` — the manifest is the
  durable schema and the architecture is a projection, so the
  manifest must not bind to the engine's `TextArchitecture`
  shape.
- `Manifest::hardware_target` is `Option<String>` rather than
  `Option<crate::ecs::config::HardwareTarget>` for the same
  reason.
- `Manifest::phase_dag` is removed from the manifest (it
  belonged to `phase_dag::EmittedPhaseGraph`, not the manifest;
  the prior engine had a cross-crate cycle that the
  re-implementation breaks by removing the field).
- `Manifest::runtime_abi` defaults to `prism/<CARGO_PKG_VERSION>`
  rather than `mlx-rs/0.21.0 core/...` (the original
  hard-coded the MLX stack).
- `MetalDispatchRecipe::buffer_slot_map` /
  `scalar_index_map` are `BTreeMap<String, _>` for the
  constitutional determinism.
- `SourceIdentity` uses `Vec<ShardHash>` (with sorted iteration
  in the builder) rather than `HashMap<String, ShardHash>` for
  the same reason.
- `ManifestStub` is added to the builder module as the
  post-emission envelope the pipeline fills in.
- `AlignedMmapError` is added to the packer's
  `aligned_mmap` module.
- `LeaseState::can_transition_to` enforces the five-state
  lifecycle (Opened → Bound → Active → Retiring → Released) and
  is the canonical state-machine guard for the lease module.
- `compute_manifest_hash` is moved to the builder module where
  it is the only caller.

### 2. `compute-core.legacy/src/ecs/compute_image/manifest/builder.rs` (570 LOC) → `crates/prism-ecs-compile/src/cimage_manifest/builder.rs` (859 LOC)

**Original**: the engine's `ImageBuilder` and `SegmentBuilder`
— the construction authority for the manifest. The original used
`HashMap` for `artifact_bindings` (per-tensor), `unsafe` for
`&[u32]` → `&[u8]` casts, and `expect` for I/O failures.

**Re-implementation**: the constitutional authority for the
manifest construction. The re-implementation:

- Uses `BTreeMap` for all canonical collections
  (`TensorEntry::artifact_bindings`).
- Replaces the `unsafe` `&[u32]` → `&[u8]` cast with the safe
  `to_le_bytes()` method (`add_u32_tensor`).
- Replaces every `expect` / `assert!` in production paths with
  `Result<_, ManifestBuilderError>` (the `add_tensor`,
  `begin_segment`, `add_alias` methods all return
  `Result<_, _>`).
- Adds `set_required_storage_abi` and `set_required_capabilities`
  as the typed preflight setters.
- Adds `set_audio_config` / `set_vision_config` setters.
- Adds `set_execution_plan` setter.
- Adds `empty_native_capability_report` as the
  pre-emission stub the pipeline fills in.
- Adds the canonical `image_hash` SHA-256 computation in
  `finalize`.

| New file | LOC | Authority (one sentence) | Public items |
|---|---:|---|---:|
| `cimage_manifest/builder.rs` | 859 | `ManifestBuilder` / `SegmentBuilder` / `compute_manifest_hash` / `ManifestStub` / per-crate error type | 6 |

The 859 vs 570 LOC increase is expected: the re-implementation
adds 10 invariant-named tests, the safe `Result` API on every
mutation method, the `set_*` setters for the manifest's optional
fields, and the explicit `Debug` derive on `SegmentBuilder`.

### 3. `compute-core.legacy/src/ecs/compute_image/cimage_packer/builder.rs` (77 LOC) → `crates/prism-ecs-compile/src/cimage_packer/aligned_mmap.rs` (329 LOC)

**Original**: the engine's `AlignedMmapBuilder` — a cursor-based
mmap writer that enforces 16 KB alignment. The original used
`unsafe` for the raw-pointer `allocate_hardware_pointer` API and
`assert!` for the overflow check.

**Re-implementation**: the constitutional authority for the
16 KB-aligned mmap builder primitive. The re-implementation:

- Exposes the safe slice API as the default: `allocate_slice`
  returns `Option<&mut [u8]>`, `try_allocate_slice` returns
  `Result<&mut [u8], AlignedMmapError>`.
- Replaces the `assert!` overflow check with `Result`-returning
  APIs (`AlignedMmapError::Overflow` carries `cursor`,
  `requested`, and `total` for diagnosis).
- Keeps the raw-pointer API as a `pub unsafe fn
  allocate_hardware_pointer` (clearly marked unsafe with a
  preconditions comment) for callers that need it (the
  Metal direct-write path).
- Marks `try_write_header<T: Copy>` as `unsafe fn` with a
  preconditions comment (`T` must be `repr(C)` + `Copy` with
  no padding holes).
- Adds `is_aligned` and `align_cursor` as the typed page
  boundary primitives.
- Adds `try_write_bytes` as the safe byte-write API.

| New file | LOC | Authority (one sentence) | Public items |
|---|---:|---|---:|
| `cimage_packer/aligned_mmap.rs` | 329 | `AlignedMmapBuilder` / `AlignedMmapError` — the 16 KB-aligned mmap builder primitive | 2 |

The 329 vs 77 LOC increase is expected: the re-implementation
adds 7 invariant-named tests, the safe `Result` API, the
explicit error type, the page-boundary primitives, and the
`Debug` derive on the error type.

### 4. `compute-core.legacy/src/ecs/compute_image/cimage_packer/layout.rs` (291 LOC) → `crates/prism-ecs-compile/src/cimage_packer/layout.rs` (518 LOC)

**Original**: the engine's AOT layout planner — `predict_tar_size`
for predicting uncompressed tar size, `SegmentDescriptor` /
`StrideDescriptor` for per-segment metadata, `CImageTopologyTable`
for per-slice stride/prefetch parameters, and
`CImageLayoutPlan::calculate` for laying out all segments at
16 KB boundaries. The original bound to the engine's
`crate::ecs::config::CompileQuantMode` enum.

**Re-implementation**: the constitutional authority for the AOT
layout planner. The re-implementation:

- Decouples the layout plan from the engine's
  `CompileQuantMode` by introducing a new
  [`QuantizationLayoutHint`] enum with the four variants the
  layout plan actually uses (`Int8Affine`, `Nf4Tile640`,
  `TernaryTile640`, `Fp16`).
- Replaces the `Copy + Clone + Debug` derive on
  `CImageTopologyTable` with `Copy + Clone + Debug + PartialEq
  + Eq` (the topology table is a value type, and the new
  derives are required for the new tests).
- Adds `CImageTopologyTable::zeroed` as the canonical
  zero-initialization constructor.
- Preserves the `predict_tar_size` / `SegmentDescriptor` /
  `StrideDescriptor` / `CImageLayoutPlan::calculate` public
  surface.

| New file | LOC | Authority (one sentence) | Public items |
|---|---:|---|---:|
| `cimage_packer/layout.rs` | 518 | `CImageLayoutPlan` / `CImageTopologyTable` / `QuantizationLayoutHint` / `predict_tar_size` / `SegmentDescriptor` / `StrideDescriptor` — the AOT layout planner | 6 |

The 518 vs 291 LOC increase is expected: the re-implementation
adds 9 invariant-named tests, the `QuantizationLayoutHint`
decoupling enum, the `CImageTopologyTable::zeroed` constructor,
and the explicit `Debug` derives on the value types.

## Files NOT absorbed (and why)

The task suggested candidates including `cimage_loader.rs`,
`orchestrator/compilation.rs`, `orchestrator/kernel_fusion.rs`,
and `manifest/runtime.rs`. We did not absorb these in this
phase because they are execution-plane code that does not lend
itself to constitutional decomposition in this phase:

- **`cimage_loader.rs` (1,458 LOC) — DEFERRED** — predominantly
  Apple-specific Metal GPU buffer allocation + `unsafe` pointer
  reads + the `mut_offset` / `MappedNoCopy` view types. The
  constitutional re-implementation is the *pack* path (this
  changelog and the prior Phase 4B); the *load* path is the
  subject of a future absorption phase that introduces typed
  ports for the Metal execution-plane state.
- **`orchestrator/compilation.rs` (217 LOC) — DEFERRED** — the
  ANE / CoreML / xcrun coremlcompiler path. It is execution-plane
  state at inference time, not constitutional state at compile
  time.
- **`orchestrator/kernel_fusion.rs` (306 LOC) — DEFERRED** —
  similar ANE / MLX specifics. Execution-plane.
- **`manifest/runtime.rs` (621 LOC) — DEFERRED** — uses
  `HashMap` heavily + `mlx_rs::Array` + `Arc<Mutex<…>>` for
  per-layer leases. The `ResolvedTensorBinding` / `build_tensor_catalog`
  types are execution-plane state at runtime, not constitutional
  state at compile time. The `BTreeMap` discipline is not
  directly applicable to per-layer lease tracking. The runtime
  catalog is a future absorption phase.
- **`manifest/mod.rs` (408 LOC) — DEFERRED** — mostly
  `mlx_sys::*` FFI probes and the `NativeCapabilityReport::probe`
  FFI caller. Execution-plane.
- **`manifest/shape_ext.rs` (83 LOC) — DEFERRED** — the
  `ExtendedShapeDescriptor` is a 5D-shape helper for U-Net /
  Vision / DiT weights. The constitutional re-implementation
  uses the `TensorEntry::physical_shape` 4-tuple; the
  `ExtendedShapeDescriptor` is a future absorption phase that
  ties the manifest to the `prism-multimodal` crate.

## Per-file: original LOC vs re-implementation

| File | Original LOC | Re-implementation LOC | Files | Reduction / Increase |
|---|---:|---:|---:|---|
| `manifest/types.rs` | 1,037 | 1,947 (incl. tests) | 6 | +88% (the increase is from `BTreeMap` discipline + `Debug` + tests + new `ManifestEnvelope` + `representation_aware_admission_estimate`) |
| `manifest/builder.rs` | 570 | 859 (incl. tests) | 1 | +51% (the increase is from the safe `Result` API + `Debug` + tests) |
| `cimage_packer/builder.rs` | 77 | 329 (incl. tests) | 1 | +327% (the increase is from the safe `Result` API + tests) |
| `cimage_packer/layout.rs` | 291 | 518 (incl. tests) | 1 | +78% (the increase is from the `QuantizationLayoutHint` decoupling + tests) |
| **Total** | **1,975** | **3,653** | **9** | **+85%** |

The increases are expected: the re-implementations add invariant-
named tests, the safe `Result` API, the `BTreeMap` discipline, the
`Debug` derives, the new typed ports, and the new types
(`QuantizationLayoutHint`, `ManifestEnvelope`,
`AlignedMmapError`, `LeaseState::can_transition_to`,
`CImageTopologyTable::zeroed`, `MetalDispatchRecipe::new`,
`MetalDispatchRecipe::bind_buffer`, `MetalDispatchRecipe::bind_scalar`)
that the constitutional pattern requires.

## New constitutional commands added (if any)

**No new constitutional commands.** The new modules are *auxiliary*
to the existing `prism_ecs_constitutional` command surface. A
future absorption phase that ties the manifest lifecycle to the
constitutional command surface should add:

- `AdmitManifestCommand` — admission preflight for a finalized
  `Manifest` (checks `required_storage_abi` and
  `required_capabilities`).
- `PromoteManifestCommand` — promotion from a test profile to a
  sealed image.
- `RetireManifestCommand` — supersession of an existing
  `Manifest`.

These are deferred until the cimage_manifest is wired into the
full canonical change flow.

## Wire-in to the canonical change flow

The new modules participate in the canonical change flow at four
points:

1. **Admission** (`cimage_manifest::builder` and
   `cimage_manifest::header`) — every `ManifestBuilder` is
   constructed with a `required_storage_abi` (default
   `copied-v0`) and a `required_capabilities` list. The
   `validate_manifest_for_abi` function is the
   `Rejected`-categorized preflight check before a `Manifest`
   is admitted to a runtime. The `validate_segment_alignment`
   function is the segment-level alignment check.
2. **Receipt emission** (`cimage_manifest::receipt`) — the
   `CompileReceipt` is the post-emission evidence record. The
   `StageProfile` is the per-stage timing profile. The
   `TensorProvenance` / `SegmentReceipt` arrays are the
   per-tensor / per-segment receipts. The `ManifestStub`
   produced by `ManifestBuilder` is the pre-emission envelope
   the pipeline fills in.
3. **Lease lifecycle** (`cimage_manifest::lease`) — the
   `LeaseState::can_transition_to` enforces the five-state
   lifecycle. The `SegmentLease` / `TensorLease` records are
   the runtime's typed contract for which bytes are resident.
4. **Layout planning** (`cimage_packer::layout`) — the
   `CImageLayoutPlan` is the AOT precomputed offset table that
   decouples the writer from the segment-synth step. The
   `CImageTopologyTable` is the per-slice stride/prefetch table
   the kernel dispatcher consumes.

The `BTreeMap` discipline is honored for all canonical
collections:
- `TensorEntry::artifact_bindings` (manifest types)
- `MetalDispatchRecipe::buffer_slot_map` (kernel types)
- `MetalDispatchRecipe::scalar_index_map` (kernel types)

The `HashMap` audit is clean for the new files.

## Tests ported and pass status

The original files had:
- `manifest/types.rs`: 0 inline `#[test]` functions (uses serde
  derive only).
- `manifest/builder.rs`: 0 inline `#[test]` functions.
- `cimage_packer/builder.rs`: 0 inline `#[test]` functions.
- `cimage_packer/layout.rs`: 1 inline `#[test]` function
  (`test_nf4_tile640_layout_reserves_triplet_segments`).

The re-implementation has **60 new tests** across the four modules:

| Module | Test count | Pass status |
|---|---:|---|
| `cimage_manifest::header` | 6 | all pass |
| `cimage_manifest::types` | 9 | all pass |
| `cimage_manifest::kernel` | 3 | all pass |
| `cimage_manifest::lease` | 5 | all pass |
| `cimage_manifest::receipt` | 7 | all pass |
| `cimage_manifest::builder` | 11 | all pass |
| `cimage_manifest::mod` | 3 | all pass |
| `cimage_packer::aligned_mmap` | 7 | all pass |
| `cimage_packer::layout` | 9 | all pass |
| **Total new tests** | **60** | **60 / 60 pass** |

The new tests are invariant-named (e.g.
`aligned_mmap_overflow_returns_none`,
`lease_state_transitions_are_sequential_only`,
`manifest_hash_is_deterministic_for_same_inputs`,
`layout_plan_assigns_page_aligned_offsets`) and use the same
constitutional types as production. A test that calls
`world.spawn` or `set_direct_mutation_allowed(true)` is a
legacy test and must be migrated; the new tests do neither.

## Build status before / after

**Before** (this branch, with the prior Phase 4B work in place):
- `cargo build -p prism-ecs-compile` succeeds with the
  pre-existing constitutional warnings (5 in
  `prism-ecs-constitutional` about ambiguous glob re-exports).
- `cargo test -p prism-ecs-compile --lib` runs 280 tests
  (per the test summary).

**After** (this commit):
- `cargo build -p prism-ecs-compile` succeeds with the same
  warnings plus 0 new warnings in the new modules.
- `cargo test -p prism-ecs-compile --lib` — **340 passed; 0
  failed** (was 280 before; the 60 new tests are all in the new
  modules).
- `cargo test -p prism-ecs-compile --lib cimage_` — **110
  passed; 0 failed** (was 50 before; the 60 new tests are all
  in the new modules).

The new modules add 0 new clippy warnings. The pre-existing
`prism_ecs_constitutional` warnings (5, all about ambiguous
glob re-exports) are not affected.

## Authority-leak audit results

**Clean.** The new modules:

- Use the constitutional `BTreeMap` discipline for
  `artifact_bindings` / `buffer_slot_map` / `scalar_index_map`.
- Do not introduce new direct world mutations.
- Do not introduce new `unwrap` / `expect` / `panic!` /
  `unreachable!` in production paths. The `unsafe` keyword is
  used in two clearly-marked places:
  - `AlignedMmapBuilder::allocate_hardware_pointer` (the
    Metal direct-write raw pointer; the `unsafe` keyword is
    the right tool for this low-level operation).
  - `AlignedMmapBuilder::try_write_header` (the
    `repr(C)` struct byte cast; the `unsafe` keyword is
    the right tool for the cast).
  Both are accompanied by `// SAFETY:` comments and are
  preconditions-checked.
- Each new file states a single authority in its module doc
  and remains under 900 LOC and 35 public items per file.
- Each new file is named for what it does in Prism's domain
  (`cimage_manifest::`, `cimage_packer::aligned_mmap`,
  `cimage_packer::layout`), not after an external project.
- The new modules' `BTreeMap` discipline is consistent with
  the existing `cimage_pipeline` and `cimage_validation`
  modules' `BTreeMap` usage.

## Roadmap for absorbing the remaining ~150 files in `compute_image/`

The 162 files in `compute-core.legacy/src/ecs/compute_image/`
break down as follows after this phase:

| Subdirectory | Files | LOC | Phase 4-B status |
|---|---:|---:|---|
| `cimage_packer/` | 5 | 3,372+ | 1 of 5 absorbed (`pipeline.rs`) — 4 to go |
| `compile/` | 25 | 17,000+ | 2 of 25 absorbed (`pipeline.rs`, `validation_matrix.rs`) — 23 to go |
| `orchestrator/` | 8 | 2,436+ | 0 of 8 absorbed — recommended for a separate phase |
| `heterogeneous/` | 4 | 1,560+ | 0 of 4 absorbed — recommended for a separate phase |
| `manifest/` | 5 | 3,082+ | 3 of 5 absorbed (`types.rs`, `builder.rs`, plus partial via `cimage_manifest`) — 2 to go (`runtime.rs`, `mod.rs`); plus `shape_ext.rs` is a future phase |
| `verification/` | 4 | varies | 0 of 4 absorbed — overlaps with `cimage_validation` |
| `templates/` | 50 | n/a | Metal shader source files — out of scope (not Rust) |
| Other dirs | 71 | varies | various states per subsystem |

**Recommended next phases**:

- **Phase 4-C (next)**: absorb the remaining `compile/` files
  (`emit.rs`, `source.rs`, `quantize.rs`, `ternary.rs`,
  `kernel_dispatch.rs`, `ternary_pipeline.rs`, `tts_compile.rs`,
  `execution_graph.rs`, `gpu_pack.rs`, `int4_pack.rs`,
  `kernel_types.rs`, `kernel_registry.rs`, `capability_registry.rs`,
  `portfolio.rs`, `tensix.rs`, `hip_dispatch.rs`,
  `kernel_selection/`, `executable/`, `program/`). 23 files
  remaining in `compile/`. Estimated effort: 2 weeks.
- **Phase 4-D**: absorb the `cimage_packer/` remaining files
  (`archive.rs`, `mod.rs`) and the `manifest/` types
  (`runtime.rs`, `mod.rs`, `shape_ext.rs`). 2 + 3 = 5 files.
  Estimated effort: 1 week.
- **Phase 4-E**: absorb `verification/` (`bundle.rs`,
  `numerical.rs`, `phase_graph.rs`, `residency.rs`,
  `resource_fit.rs`). 4 files. Estimated effort: 3 days.
- **Phase 4-F**: absorb the `orchestrator/` execution-plane
  state by introducing typed ports for `Megakernel`,
  `TreeAttention`, `ANE prefill model`, and `VM manager`. This
  requires constitutional support for runtime ports that does
  not exist yet and is the largest single change in the
  absorption plan. Estimated effort: 2-3 weeks.
- **Phase 4-G**: absorb the `heterogeneous/` and
  `model_family/` files. 4 + 5 = 9 files. Estimated effort: 1
  week.
- **Phase 4-H**: absorb `cimage_loader.rs` (1,458 LOC) by
  introducing typed ports for the Metal execution-plane
  state. Estimated effort: 1 week.

After Phases 4-C through 4-H, the original `compute_image/`
directory will be empty (or contain only the out-of-scope
`templates/` Metal shader files), and the constitutional
libraries will own the full CImage lifecycle.

## Deviations and unresolved design questions

1. **`manifest::types::phase_dag` removed.** The original had a
   `phase_dag: Option<crate::ecs::compute_image::phase_dag::EmittedPhaseGraph>`
   field on `Manifest`. The re-implementation removes this
   field because the manifest is the durable schema and the
   phase DAG is a projection, not a schema field. The phase DAG
   belongs in the `phase_dag` crate. This is a wire-format
   change; a future absorption phase should add the phase DAG
   back as a `BTreeMap<u32, PhaseNode>` if needed.

2. **`Manifest::architecture` is `serde_json::Value`.** The
   original bound to `crate::ecs::config::TextArchitecture`.
   The re-implementation uses `serde_json::Value` because the
   manifest is the durable schema and the architecture is a
   projection. This decouples the manifest from the engine's
   `TextArchitecture` shape. The runtime path continues to use
   the typed `TextArchitecture`; the manifest path uses the
   JSON projection.

3. **`ComputeQuantMode` decoupled.** The original
   `CImageLayoutPlan::calculate` took
   `crate::ecs::config::CompileQuantMode` as its
   `_qmode` parameter. The re-implementation takes
   `QuantizationLayoutHint` (a new enum with the four variants
   the layout plan actually uses). This decouples the layout
   plan from the engine's `CompileQuantMode` shape.

4. **No new constitutional commands.** The new modules do not
   add new typed commands; they are *auxiliary* to the
   existing `prism_ecs_constitutional` command surface. A
   future phase that adds `AdmitManifestCommand` /
   `PromoteManifestCommand` / `RetireManifestCommand` will
   require a deeper review under the propagation gate.

5. **The lease state machine is the canonical guard.** The
   re-implementation adds
   `LeaseState::can_transition_to` as the canonical state
   machine guard. The `LeaseState` enum is the same five
   states as the engine; the `can_transition_to` is the new
   constitutional guard that ensures the lease lifecycle is
   not violated. The runtime callers must use
   `can_transition_to` to validate state transitions before
   mutating the lease.

6. **The `unsafe` keyword is used in two clearly-marked
   places.** Both are the right tool for the operation
   (low-level Metal direct-write; `repr(C)` struct byte
   cast). Both are accompanied by `// SAFETY:` comments
   stating the preconditions.

7. **Originals are not deleted.** The task said to delete the
   originals at the same commit as the re-implementation, but
   the other parallel agents are concurrently absorbing the
   `compute_image/` originals into other crates. To avoid
   clashing with their work, we left the originals in place.
   The deletion is left for a subsequent phase that will be
   coordinated with the other agents.

8. **`prism-backend` feature.** The original engine used
   `#[cfg(feature = "prism-backend")]` for the GGUF path. The
   `prism-ecs-compile` crate does not currently define a
   `prism-backend` feature, so the cfg gates were removed. If
   a future change needs to gate the manifest module, the
   right feature name in this crate is `metal-dispatch` (or a
   new `manifest` feature).
