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

Full loss includes weight, operator, cosine, norm, runtime, and byte terms. Runtime and byte terms are selection costs, not differentiable loss.

Initial refinement algorithm: alternating least-squares scale update followed by threshold-based code reassignment. No full backpropagation through the model. Each refinement epoch produces a new candidate version; only the best is retained.

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
