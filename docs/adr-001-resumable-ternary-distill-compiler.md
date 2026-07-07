# ADR 001: Resumable Ternary Distill-Compiler Expansion

## Status

Accepted. Pending implementation of Phase A (foundation) through Phase F (end-to-end parity).

## Context

The existing mixed-precision cimage compiler can pack a BF16 or QAT-NF4 source into an NF4/INT8 cimage with per-tensor format selection, validation gates, and segregated segments. It is a one-shot converter — it runs to completion or fails. It has no pause, resume, or local refinement capabilities, and its candidate ladder stops at INT8.

Ternary Tile640 is the highest-value representation to add next. It offers 1.6-bit effective storage per weight (versus 4-bit for NF4, 8-bit for INT8), which translates into 2.5× fewer memory loads per matmul. For decoder layers where activation sensitivity permits it, ternary can halve the memory bandwidth bottleneck of decode.

But ternary is not universally applicable. Attention projections, cross-modal bridges, and acoustic output heads require higher precision. The compiler must test ternary first and escalate automatically only when ternary fails its validation gates. It must do this resumably — 336 matrix qualifications at minutes each is an overnight compile, and overnight compiles must survive process interruption and machine sleep.

Additionally, local refinement (optimizing tile scales and ternary code assignments per tensor without full-model backpropagation) is necessary to push ternary further into tensors that barely fail direct ternary packing. A qualification-only pass that says "no, escalate to NF4" is leaving operator-space reconstruction quality on the table.

## Decision

We will extend the existing distill-compiler with a resumable, content-addressed, per-tensor qualification and local-refinement architecture governed by the following decisions.

### 1. Teacher Authority Hierarchy

BF16 is the sole source of ground truth for parity. Cached BF16 targets are valid only when the full provenance chain (source model digest, preprocessing version, activation-bank digest) matches. The NF4 cimage may serve as a fast proxy teacher for iteration but may not be used to claim BF16 parity in production receipts.

```rust
pub enum TeacherAuthority {
    Bf16Direct = 1,
    Bf16CachedTargets = 2,
    QualifiedNf4Proxy = 3,
}
```

Every receipt and every checkpoint declares which tier was used. A production-sealed artifact must contain at least one holdout pass against Tier 0 or Tier 1.

### 2. Representation Ladder

The candidate order is fixed by policy, not heuristics:

```
TernaryTile640Base → TernaryScaled → NF4Base → NF4Scaled → INT8Base → INT8Scaled → [future residual] → fail
```

Each tensor starts at the cheapest representation and climbs only when gates reject. The compiler may skip entries only through a deterministic, policy-declared pruning rule with a recorded rationale code.

Selected representation = lowest-cost candidate that passes every hard gate.

### 3. Ternary Tile640 Contract

Ternary uses the same tile geometry as NF4 and INT8: 640 elements per tile, rows × ceil(cols / 640) tiles.

The packed layout is:

- 2 bits per weight (4 codes per byte)
- 160 packed bytes per Tile640 tile
- Tile-local FP16 scale (one per tile)
- No tile bias initially (symmetric ternary only; a bias is added only if validation proves it necessary for a tensor class)
- Unused code 0b11 is rejected by the reference decoder

Base reconstruction: W_hat[i,j] = q[i,j] × alpha_tile

### 4. Two Distillation Modes

```rust
pub enum DistillationMode {
    QualificationOnly,  // pack, validate, select
    LocalRefinement,    // optimize tile scales, code assignments
}
```

QualificationOnly is the default. LocalRefinement is activated when a ternary candidate fails promotion validation but remains within a refinement-eligibility band.

Local refinement operates per tensor, not per model. It optimizes only the candidate representation parameters — ternary code assignments, tile scales, reduction-axis scales — not the global BF16 weights. The optimization target is operator reconstruction loss over a calibration batch:

```rust
L_operator = 1/B sum_b=1..B ||x_b W^T - x_b W_hat^T||^2
```

The BF16 output targets Y = XW_bf16^T are captured once per tensor and cached keyed by tensor digest plus activation-bank digest. Subsequent refinement iterations read the cached targets rather than re-executing the BF16 model.

Full loss includes weight, operator, cosine, norm, runtime, and byte terms. Runtime and byte terms are selection costs, not differentiable loss:

```math
L = \u03bb_out \u00b7 MSE(XW_bf16^T, XW_ternary^T)
  + \u03bb_cos \u00b7 (1 - cosine(Y_bf16, Y_ternary))
  + \u03bb_weight \u00b7 regularize(W_ternary, W_bf16)
  + \u03bb_cost \u00b7 representation_cost
```

#### 4.1. Bounded Ternary Refinement Ladder

A Tile640 has 3^{640} possible ternary code assignments before considering scales. The compiler does not enumerate this space. Instead it searches the restricted ternary space around the BF16 tensor in a way that is exhaustive at the local decision level and bounded at the tile level.

For a fixed Tile640 block, the problem is:

Given BF16 weights W, activation bank X, and teacher outputs Y = XW^T, find ternary codes Q \u2208 {-1, 0, +1}^640 and scale s such that X(sQ)^T \u2248 Y.

The refinement ladder is a strict sequence of seven tiers. Each tier refines the current ternary candidate. If a tier does not sufficiently reduce operator loss, the candidate escalates to the next tier. If all ternary tiers fail, the tensor is promoted to NF4 or INT8.

| Tier | Operation | Scope | Rate |
|------|-----------|-------|------|
| 1 | Closed-form scale fit | All tiles, given current Q | Every tier change |
| 2 | Coordinate ternary search | High-error individual weights | Test -1, 0, +1 per weight, accept if loss reduces |
| 3 | Exhaustive local group search | Worst 1-5% of tiles, groups of 4 or 8 weights | 3^4 = 81 or 3^8 = 6,561 states per group, GPU-evaluated |
| 4 | Per-channel / reduction-axis scale sidecar | Full tensor | Evaluated via operator-space gate |
| 5 | Sparse residual correction | Select high-error output channels | Fixed-size residual vector appended to sidecar segment |
| 6 | NF4 Tile640 | Full tensor | Standard admission pipeline |
| 7 | INT8 Tile640 | Full tensor | Standard admission pipeline |

Tiers 1 through 5 form the ternary refinement cascade; tiers 6 and 7 are the escalation path.

#### 4.2. Coordinate Descent (Tier 2)

For each weight position in a high-error tile, test all three assignments -1, 0, and +1. Calculate which reduces the activation-weighted output loss most. Keep the winner, then move to the next coordinate. This is coordinate descent over a discrete ternary alphabet.

Each coordinate decision is exact \u2014 the chosen assignment is optimal for that position given all other positions held fixed. A pass visits every weight in the tile once. Multiple passes are bounded by a strict iteration limit (8-32 rounds per tensor).

Require a minimum error improvement per round. If the improvement plateaus, stop and proceed to the next tier or escalate.

#### 4.3. Exhaustive Group Search (Tier 3)

Divide the Tile640 into groups of 4 or 8 weights. A group of 4 has 81 states; a group of 8 has 6,561 states. For the highest-error groups (worst 1-5% of all groups in the tensor), evaluate all local states against the residual output contribution.

This is feasible on GPU because it is applied only to a small number of selected groups, not every group in every tensor.

#### 4.4. Refinement Termination and Escalation

Refinement is explicitly bounded per tensor:

- Maximum refinement rounds: 32 (Tier 2 coordinate passes)
- Maximum group-search tiles: 5% of all tiles in the tensor (Tier 3)
- Minimum improvement threshold: 1% NRMSE reduction per round; if below, stop and escalate
- Wall-clock budget per tensor: 60 seconds (enforced by admission pipeline deadline)

When refinement terminates without reaching operator parity:

1. Emit a qualification receipt stating "ternary attempted and rejected" with the best metrics achieved
2. Escalate that tensor to NF4 (Tier 6) or INT8 (Tier 7)
3. Proceed to the next tensor

The compiler never assumes every matrix can be made ternary by enough retries. The mixed-precision compiler exists precisely because the correct answer for some tensors is "ternary failed honestly; preserve this one at NF4 or INT8."

#### 4.5. Strict Promotion Gate

A tensor is promoted to ternary only if the refined candidate passes operator-space validation:

1. Against cached BF16 outputs on the promotion activation bank (must pass the promotion profile)
2. Independently against cached BF16 outputs on the holdout activation bank (must pass the holdout profile)

Promotion bank pass and holdout bank pass are both required. A candidate that passes promotion but fails holdout does not become ternary \u2014 it escalates to NF4 or INT8. A candidate that passes neither does not retry refinement indefinitely; it escalates immediately.

### 5. Per-Tensor Qualification DAG

Every tensor follows a resumable directed acyclic graph:

```
Discover → Canonicalize → Analyze → CaptureOrLoadTargets
→ GenerateCandidates → StructuralValidation → ProbeValidation
→ PromotionValidation → HoldoutValidation → SelectRepresentation
→ PackPayload → Serialize → ReplayValidation → Commit
```

Each node carries an input digest and an output digest. A node is valid only when its input digest matches the current job's expected dependency digests. This makes the graph self-validating across pause and resume cycles.

### 6. Content-Addressed Checkpointing

Every checkpoint is keyed by:

- Source model digest
- Source tensor digest  
- Tensor canonicalization version
- Quantization policy digest
- Candidate format
- Teacher authority and target digest
- Promotion/holdout bank digests
- Target ABI digest
- Compiler build digest
- Refinement algorithm version
- Random seed

Changing any dependency invalidates the checkpoint automatically. Checkpoints are written atomically: write to temp path, flush, fsync, write digest, atomic rename. No partially written checkpoint is valid.

### 7. Cooperative Pause and Resume

Pause is cooperative: stop admitting new work, allow in-flight kernels to complete, flush checkpoints, persist scheduler state, mark active nodes resumable, release buffers.

Resume: load job manifest, verify compiler compatibility, verify checkpoint digests, invalidate stale or incomplete nodes, reconstruct ready queue, resume from first valid unfinished node. No completed tensor is repacked unless its inputs are invalidated.

### 8. Bounded Memory

For a 16 GB M1 target: 10-12 GB distiller budget, 3-4 GB safety reserve. The compiler never holds simultaneously the full BF16 source tensor, normalized tensor, packed tensor, unpacked tensor, teacher output bank, and candidate output bank. Processing is in Tile640 row bands. Operator validation accumulates RMSE, norm, cosine, and max-error reductions online and discards intermediate output blocks after reduction.

### 9. Validation Gate Architecture

Four gates, in order:

1. **Structural** — cheap: codec validity, padding, tile dimensions, zero-collapse ratio (format-aware: ternary skips zero-collapse since ~70% sparsity is expected)
2. **Operator-space probe** — small deterministic vector set (4-8 vectors), catches catastrophic failures
3. **Promotion** — full calibration bank, metric accumulation, determines candidate fitness
4. **Holdout** — held-out vectors, defends against promotion-bank overfitting

A candidate enters the investigation band when weight NRMSE exceeds the preferred threshold but stays below a class-specific ceiling. Investigation-band candidates require stronger operator and holdout evidence. No candidate is promoted solely because it falls in the investigation band.

### 10. Calibration Banks

Every tensor class has:

- **Stress bank**: deterministic generators per tensor class, normalized to three norm bands. Always built. Catches codec, layout, and scaling failures.
- **Promotion bank**: model-native activation capture. Selects candidates.
- **Holdout bank**: independent activation capture. Validates that selected candidates did not overfit promotion.

For production qualification, model-native activation banks (promotion + holdout) are required. Stress alone yields DiagnosticOnly classification.

### 11. Mixed Precision Naturally

The compiler does not force a uniform format. Each tensor selects its own format through independent qualification. The expected final mapping for a ~12B decoder:

- Decoder MLP: ternary Tile640
- Decoder attention: ternary or NF4
- Vision patch projection: NF4 Tile640
- Cross-modal bridge: INT8 Tile640 if NF4 fails cosine gate
- MTP draft: ternary where proposal acceptance holds, NF4 elsewhere
- TTS dense blocks: ternary or NF4
- TTS acoustic/output-sensitive heads: NF4 or INT8

### 12. Replay Validation Before Sealing

Before sealing, the compiler reopens the serialized artifact and verifies: segment directory validity, all ranges are in-bounds, MatrixContract parse safety, format tag support, offset+length inside segment, sidecar consistency, tile geometry matches matrix shape, payload digests, sidecar digests, CPU reconstruction matches pre-serialization candidate, and operator metrics match within tolerance.

A replay mismatch invalidates the artifact.

### 13. Memory-Augmented Architecture Support (Engram-Class Models)

The cimage format infrastructure — segregated segments, content-addressed checkpoints, deterministic addressing, heterogeneous execution lanes — is well suited to host memory-augmented architectures such as DeepSeek Engram (Conditional Memory via Scalable Lookup, arXiv 2601.07372). However, Engram is not a drop-in addition to an existing Gemma 4 cimage. The gate, compressed-token mapping, hash heads, retrieved embeddings, fusion projections, and surrounding early Transformer layers are trained jointly. Adding the segment and kernel to a Gemma checkpoint produces a retrieval path with no learned semantic role.

Engram is most valuable when it is native to the model's architecture and training process. For an Engram-native ternary model, the distill-compiler can treat the Engram module as a first-class executable component: quantize or ternarize the surrounding projections, preserve the hash/table contract, validate the gate and fusion path, and decide where the memory table should reside across CPU/GPU memory.

The compiler strategy for Engram is therefore a fork based on source model architecture:

| Source model | Compiler behavior |
|---|---|
| Standard transformer (Gemma 4) | Ordinary mixed ternary/NF4/INT8 cimage migration. No Engram segment. |
| Engram-native source | Engram-aware ternary cimage compilation — table segment, gate/fusion projections, memory ABI, progressive table/tensor updates. |

#### Cimage Infrastructure for Engram (segment-level prerequisites)

Engram requires one new SegmentKind (`EngramMemoryTable = 42`) and a dedicated contract:

```rust
pub struct EngramTableContract {
    pub tokenizer_compression_map: ContentAddress,
    pub hash_family: String,
    pub seed_schedule: Vec<u64>,
    pub ngram_orders: Vec<u8>,
    pub hash_heads_per_order: u8,
    pub bucket_sizes: Vec<u32>,
    pub embedding_dim_per_head: u16,
    pub concatenated_embeddings_digest: Digest256,
    pub gate_params_digest: Digest256,
    pub fusion_params_digest: Digest256,
    pub insertion_layer_range: (u8, u8),
    pub calibration_receipt_digest: Digest256,
    pub contract_digest: Digest256,
}
```

On Apple Silicon, the table already lives in unified system memory. The useful optimization is not host-DRAM-to-GPU transport but: deterministic token-derived addresses to precompute and prefetch likely cache lines, overlap lookup with surrounding compute, and keep the table in an explicitly managed shared or private Metal resource according to measured access behavior.

Engram support is a future multiplier. It becomes especially powerful when a model is trained with Engram from day one: the table can then be a separately versioned, host-resident, progressively refinable memory component while the Transformer bulk becomes aggressively ternarized.

## Consequences

### Positive

- **Resumability eliminates wasted work.** A 336-tensor qualification that takes 8 hours and gets interrupted at hour 7 loses only the current tensor, not all prior work.

- **Content-addressed checkpoints make the compiler auditable.** Every qualified tensor can be traced to its exact source tensor, policy, calibration banks, and compiler build. Changing any dependency invalidates only the affected subtree.

- **Local refinement increases ternary coverage.** Tensors that barely fail direct ternary packing can be pushed into qualification through per-tensor tile-scale and code-assignment optimization, without full-model training.

- **The mixed-precision policy prevents silent degradation.** Every tensor independently proves its representation preserves operator-space behavior. No tensor is forced into a format it cannot support.

- **Bounded memory keeps the compiler viable on 16 GB machines.** Streaming row-band processing and online metric accumulation remove the need to hold multiple full-resolution copies of weight and activation tensors simultaneously.

- **The teacher authority hierarchy prevents NF4-proxy contamination.** Every receipt declares which tier was used. A production ternary cimage cannot be sealed without at least one pass against BF16 direct or BF16-cached targets.

### Negative

- **Checkpoint storage cost.** Each qualified tensor produces analysis, candidate, refinement (optional), and qualification checkpoints. For 336 tensors, this can reach several hundred megabytes of metadata and several gigabytes of packed payload references. The workspace is not a deployment artifact.

- **Local refinement is not free.** Even bounded per-tensor coordinate descent or ALS optimization adds minutes per tensor. The total compile time increases, but the increase is bounded per tensor and resumable.

- **Determinism across GPU backends.** Metal GPU floating-point accumulation may differ from CPU reference at low-order bits. The policy must define a numerical tolerance for GPU-vs-CPU validation that does not mask real precision loss. This requires empirical calibration per kernel class.

- **Operational complexity.** The resumable job manifest, checkpoint store, workspace layout, and state machine add a layer of infrastructure the one-shot converter did not need. This is justified by the overnight-compile use case but adds surface area for bugs.

### Risks

- **Sparse residual escalation** (the final ladder entry) is explicitly deferred. If ternary, NF4, and INT8 all fail for a tensor, compilation fails. For early adopters, this is acceptable — the model can be repartitioned or the tensor class can receive a higher-precision threshold profile.

- **The ANE teacher-target lane** is optional. If ANE placement cannot be proven or measured, work routes to Metal GPU. Correctness never depends on ANE availability.

- **Memory budget enforcement** requires active monitoring and pre-emptive pass scheduling. A calibration bank that exceeds the configured budget must be processed in chunks rather than rejected. The streaming architecture handles this, but the chunk scheduler must be correct.

### 14. Architectural Hardening from Production Review

An adversarial review of the initial mixed-precision cimage pipeline identified five systemic weaknesses that must be resolved before the compiler is seal-safe for production. Each is stated as a requirement below.

#### 14.1. Compiler Phase Extraction

The ingestion binary must not act as downloader, model adapter, tensor classifier, quantization planner, calibration runner, packer, execution-graph compiler, Metal/ANE build coordinator, artifact assembler, verifier, and report generator in a single control path. Every new feature — ternary refinement, resumability, progressive tensor updates, new model families, Engram modules, or a Linux backend — risks touching one very large file.

The compiler must be decomposed into explicit phases with durable interfaces:

```
source ingestion
→ canonical tensor inventory
→ qualification plan
→ per-tensor compilation
→ segment materialization
→ artifact assembly
→ replay verification
→ sealing
```

Each phase writes a small manifest and immutable outputs. A failed final assembly never requires rerunning tensor work. A new ternary policy can reuse source inventory, teacher targets, and already-qualified NF4/INT8 tensors.

The binary remains a thin CLI front end over the reusable library API.

#### 14.2. Canonical Binary Representation

Every artifact structure that crosses the cimage file boundary must use explicit little-endian field serialization with strict bounds checking, version checks, and parse-time validation. The top-level `CimageHeader`, segment directory entries, layer directory entries, ANE descriptors, model artifact records, execution graph, and all future contract types (Engram contracts, progressive-update records) are in scope.

`repr(C)` + `std::slice::from_raw_parts` is forbidden for all file-format structures. Each type must define a dedicated write function (producing canonical LE bytes) and a dedicated read function (producing the validated Rust type). The `SegmentEntry`, `TensorRecord`, and `ModelArtifactEntry` types already use this pattern: `kind`, `offset`, `length`, `tag`, and `data length` are emitted as `to_le_bytes()` and parsed with `from_le_bytes()`. `CimageHeader`, `CimageLayoutMeta`, `LayerDirectoryEntry`, and `AneModelDescriptor` must be migrated to the same discipline.

Wire format versions must be explicit (e.g. `CimageHeaderWireV1`). A parser that encounters an unrecognized version rejects the artifact rather than interpreting bytes with an incompatible layout.

#### 14.3. Streaming Segment Materialization with CompilationMemoryBudget

The compiler must not allocate and retain large per-segment payloads in memory. Each qualified tensor writes directly into an append-only content-addressed segment file with an incremental digest, rather than accumulating `Vec<u8>` payloads for post-hoc assembly.

Final assembly splices or copies the per-segment object files into a temporary cimage without recreating giant aggregate vectors. The incremental digest eliminates the need for an expensive post-hoc read-and-hash pass.

The system enforces an explicit `CompilationMemoryBudget` with reservations for:

| Reserve | Purpose |
|---------|---------|
| Source tensor band | Inflight source weight bytes (one tensor at a time, Tile640 row bands) |
| Packed payload band | Candidate representation bytes (the candidate format, not the source) |
| Validation activations | Teacher and candidate activation banks (stress and optionally prerendered) |
| Metal buffers | GPU dispatch resources (command buffers, intermediate buffers) |
| OS headroom | Safety margin — macOS and system daemons continue to page |

On a 16 GB M1, the budget is 10–12 GB for the distiller and 3–4 GB for safety reserve. The compiler refuses to schedule a task whose reservation cannot fit. It does not trust the allocator or macOS swap as a scheduler.

#### 14.4. Strict Coverage Enforcement

Before any quantization or packing begins, the compiler produces a coverage report enumerating every tensor in the source model as one of:

| Classification | Behavior |
|---------------|----------|
| Required | Must compile. Missing or unsupported format → compilation fails |
| Optional | Included if resources permit, otherwise omitted with reason |
| IntentionallyIgnored | Listed with policy rationale code |
| Unsupported | Must not be silently dropped — fail with diagnostic |

The coverage report is digest-stamped and stored in the cimage provenance segment. An artifact that reaches seal time with any tensor still mapped to `Unsupported` is rejected. This prevents the compiler from producing a cimage that looks complete but silently omits a required multimodal, MTP, normalization, or output component.

#### 14.5. RepresentationPlanner: Separation of Planning from Assembly

The mixed-precision planner must emit a complete, immutable `TensorRepresentationPlan` for each matrix before any segment bytes are written. The plan defines:

```
format (Ternary | NF4 | INT8 | RawF16)
payload geometry (rows, cols, tiles,
    tile stride,
    alignment)
metadata layout (block size,
    scale stride,
    offset stride)
sidecar kind (None | ReductionAxis |
    Residual | Dynamic)
sidecar element type
segment class (Nf4Tile640Weights |
    Int8Tile640Weights | TernaryWeights |
    QuantizationSidecars)
byte length
alignment requirement
expected kernel ABI tag
qualification receipt digest
```

Assembly materializes the plan; it does not infer representation semantics. The current code path that initializes sidecar kind and element format as zero when sidecars exist is an example of a semantic mismatch that a dedicated planner prevents.

#### 14.6. Producer-Consumer Pipeline

The CPU reads and canonicalizes the next tensor band while the GPU validates the current candidate and the writer persists the previously accepted candidate. Three concurrent lanes:

| Lane | Work |
|------|------|
| CPU | Read source tensor, compute histograms, hash, candidate pruning, manifests |
| GPU | Batched operator validation, local refinement |
| Disk | Persist immutable tensor payloads immediately after qualification |

This fits the heterogeneous architecture (Metal, ANE, CPU) and avoids holding all results until the end.

#### 14.7. Four Verification Gates at Seal Time

The current single `verify_cimage` function is replaced by four named gates:

| Gate | Scope |
|------|-------|
| Structural | File format, segments in-bounds, directory consistency, digests match per-segment incremental hashes, ABI tags valid, no truncated segment, version support |
| Reconstruction | Packed payloads decode back to their planned representation within tolerance. CPU-only, no operator comparison |
| Operator | Matrix outputs against the teacher activation bank (stress or prerendered). The existing operator-space gate |
| Runtime conformance | The actual Metal dispatch path against the CPU reference. Requires model metadata for runtime-traceable dispatch |

The final verification step is not the first time the full artifact layout is exercised. Each gate can be run independently; structural and reconstruction are fast and always run; operator and runtime conformance are configurable by artifact tier (always for release, sampled for development).

#### 14.8. Transactional Write Protocol

Artifact writing follows a transactional protocol that prevents partial files from appearing complete:

1. Write to `output.cimage.partial`
2. Write header last
3. Call `File::sync_all()` to persist file content and metadata to stable storage
4. Close writer, reopen in read-only mode via mmap
5. Verify (structural gate at minimum)
6. Atomically rename `output.cimage.partial` → `output.cimage`

A failed compiler leaves a recoverable workspace and a clearly invalid partial artifact, never a file that appears complete but has an unsealed header.

#### 14.9. Execution Graph from Canonical Inventory

The model-specific execution graph must be generated from the canonical tensor inventory, not reconstructed from hard-coded dimension constants and key-string patterns. The graph compiler consumes stable tensor IDs and representation contracts, then emits graph nodes that point to those IDs.

This eliminates brittleness when adding MTP variants, Qwen TTS, vision paths, or tensor updates. It also makes progressive tensor updates possible without rebuilding the entire graph — only the affected node's representation pointer needs updating.

#### 14.10. Merkle-Hash Primitive Separation

The current `parallel_sha256()` computes SHA-256 of concatenated SHA-256 chunk digests. This is a valid Merkle-like digest but must not be called or compared as ordinary SHA-256. Two explicit primitives are required:

- `sha256(bytes)` — standard payload integrity (SHA-256 of the literal input bytes)
- `merkle_root(chunk_hashes, chunk_size, algorithm_version)` — Merkle-tree root for resumable segment verification

The chunk size and tree scheme are recorded in the manifest. Without this, different chunk counts or Rayon configurations can produce incompatible roots for identical bytes across different compilations or hardware configurations.
