# ADR 002: Progressive Tensor Update and Distributed Ternarization

## Status

Accepted. Pending implementation of the initial vertical slice (Phase 1-2).

## Context

A sealed cimage is immutable. If a tensor's representation is later found to be suboptimal - an NF4 tensor that could have been ternary, or an INT8 tensor that could have been NF4 - there is currently no way to improve it without recompiling the entire model and redistributing a multi-gigabyte artifact.

This is wasteful in two ways. First, compiler improvements or better calibration data may produce better representations for individual tensors over time, but the one-shot compiler model forces a full rebuild. Second, Prism Agent users have idle local compute resources (CPU, GPU, ANE) that could contribute to tensor qualification work, but there is no protocol for distributing bounded deterministic jobs or consuming their output.

The specification defines a protocol for progressive model evolution through signed, content-addressed tensor-generation updates. A base cimage begins as a qualified NF4, INT8, or mixed-precision artifact. Over time, individual tensors may be replaced with better-qualified representations, including ternary Tile640 payloads, scaled ternary payloads, or bounded residual overlays. User devices contribute idle local compute toward deterministic tensor qualification and refinement jobs, subject to strict privacy and safety constraints.

## Decision

We will implement a progressive tensor update and distributed ternarization system governed by the following architectural decisions.

### 1. Immutable Generations, Bounded History

Every tensor has a stable identity independent of physical segment offsets. Each accepted representation is an immutable generation. A tensor accumulates at most one active generation, one optional bounded residual overlay, and one pending rollback generation. No unbounded overlay chain is allowed. The base cimage remains runnable if no updates are installed.

```rust
pub struct TensorId {
    pub model_namespace: String,
    pub module_path: String,
    pub logical_name: String,
    pub ordinal: u32,
}
```

### 2. Content-Addressed Everything

Every object in the protocol is content-addressed by digest: job manifests, calibration bundles, teacher targets, candidate proposals, payload chunks, verification receipts, signed updates, revocation records, compaction manifests. All use SHA-256 or BLAKE3 with Ed25519 signatures.

```rust
pub struct ContentAddress {
    pub algorithm: HashAlgorithm,
    pub digest: Digest256,
    pub byte_len: u64,
}
```

### 3. Base Artifact Lineage

Every progressive update targets exactly one base artifact lineage. A tensor update is rejected if the base cimage root, source model digest, target ABI digest, tensor schema version, codebook version, compiler policy digest, or teacher-target corpus digest differ from the active lineage.

```rust
pub struct BaseArtifactIdentity {
    pub model_family: String,
    pub model_revision: String,
    pub base_cimage_root: Digest256,
    pub source_model_digest: Digest256,
    pub target_abi_digest: Digest256,
    pub compiler_policy_digest: Digest256,
}
```

### 4. Update Classes

Six update classes are defined:

- **FullTensorReplacement**: replaces the active tensor payload with a new payload of the same or different representation.
- **PrecisionPromotion**: changes a tensor from ternary to NF4 or INT8 when validation proves the prior insufficient.
- **TernaryMigration**: replaces an NF4 or INT8 tensor with a qualified ternary representation.
- **ResidualOverlay**: adds one bounded correction layer to an existing tensor, allowed only when the runtime ABI explicitly supports it.
- **KernelScheduleUpdate**: changes only compile-time specialization metadata, never payload bytes.
- **MetadataOnlyReceiptUpgrade**: adds stronger validation evidence without changing runtime-visible bytes.

### 4a. Memory-Augmented Architecture Updates (Engram-Class Models)

For Engram-native models, memory-table updates differ from weight-matrix updates in essential ways. The Engram module is not an auxiliary lookup attached to an already-trained model — the gate, compressed-token mapping, hash heads, retrieved embeddings, fusion projections, and surrounding early Transformer layers are trained jointly. An update that changes only table buckets or prunes low-density entries can invalidate the learned distribution unless that operation is specifically trained and re-qualified.

The atomic unit for an Engram update is therefore the full Engram module generation, not a partial table replacement:

```rust
pub struct EngramModuleGeneration {
    pub module_id: String,
    pub base_cimage_root: Digest256,
    pub tokenizer_compression_map: ContentAddress,
    pub hash_family: String,
    pub seed_schedule: Vec<u64>,
    pub ngram_orders: Vec<u8>,
    pub hash_heads_per_order: u8,
    pub bucket_sizes: Vec<u32>,
    pub embedding_table_digest: Digest256,
    pub gate_parameters_digest: Digest256,
    pub fusion_parameters_digest: Digest256,
    pub insertion_layer_range: (u8, u8),
    pub calibration_receipt_digest: Digest256,
    pub qualification_receipt_digest: Digest256,
    pub target_abi_digest: Digest256,
    pub contract_digest: Digest256,
}
```

An `EngramModuleUpdate` must atomically version the tokenizer projection, hash specification, bucket geometry, table payload, gate weights, fusion weights, and compatibility range for the consuming layers. Partial-table operations (bucket pruning, reordering) require separate qualification receipts proving the operation preserves the learned distribution.

This is not a blocker for existing Gemma 4 models, which have no Engram module. It is a future update class for Engram-native architectures.

### 5. Tensor Generation Contract

Each accepted tensor generation is self-describing:

```rust
pub struct TensorGenerationContract {
    pub tensor_id: TensorId,
    pub generation_id: Digest256,
    pub parent_generation_id: Option<Digest256>,
    pub source_tensor_digest: Digest256,
    pub base_cimage_root: Digest256,
    pub selected_format: QuantizedWeightFormat,
    pub codebook_version: u8,
    pub rows: u32, pub cols: u32,
    pub tile_elements: u16, pub tiles_per_row: u32,
    pub payload_digest: Digest256,
    pub tile_metadata_digest: Digest256,
    pub sidecar_digest: Option<Digest256>,
    pub residual_digest: Option<Digest256>,
    pub sidecar_kind: SidecarKind,
    pub sidecar_element_format: SidecarElementFormat,
    pub qualification_receipt_digest: Digest256,
    pub target_abi_digest: Digest256,
    pub kernel_schedule_digest: Digest256,
    pub created_at_unix_ms: u64,
}
```

A generation is immutable. Any improved candidate produces a new generation ID.

### 6. Distributed Work Separation of Powers

A user device can propose a tensor update but cannot authorize it. The protocol separates three roles:

- **Contributor** (user device): runs bounded deterministic qualification jobs, produces candidates.
- **Verifier** (trusted Prism service or owner-controlled machine): independently validates candidates before signing.
- **Resolver** (runtime): selects the highest trusted compatible generation per tensor ID at startup.

Contributors cannot sign official updates, alter calibration corpus identity, choose teacher targets, choose thresholds, or alter parent generations.

### 7. Deterministic Qualification Jobs

Every distributed job is signed and content-addressed. The job contract fixes every parameter that affects the candidate: tensor source digest, shape, candidate format, codebook version, tile geometry, teacher target digest, calibration bank digest, quantization profile, seed, refinement iteration budget, candidate traversal order, numeric tolerance policy, compiler version, and kernel ABI version.

```rust
pub struct TensorQualificationJob {
    pub job_id: Digest256,
    pub protocol_version: u32,
    pub base_identity: BaseArtifactIdentity,
    pub tensor_id: TensorId,
    pub required_parent_generation: Digest256,
    pub permitted_candidate_formats: Vec<QuantizedWeightFormat>,
    pub teacher_authority: TeacherAuthority,
    pub teacher_target_digest: Digest256,
    pub stress_bank_digest: Option<Digest256>,
    pub promotion_bank_digest: Option<Digest256>,
    pub holdout_bank_digest: Option<Digest256>,
    pub qualification_profile_digest: Digest256,
    pub deterministic_seed: u64,
    pub expires_at_unix_ms: u64,
    pub data_scope: DataScopeDeclaration,
    pub signer_key_id: KeyId,
    pub signature: Signature,
}
```

The contributor must reject the job if any signed dependency is missing, expired, incompatible, or violates local policy.

### 8. Privacy Boundary

The distributed ternarization service uses only one of these data sources: bundled public calibration corpus, publicly downloadable calibration corpus, signed synthetic stress bank, or signed BF16 teacher target cache. The initial product supports only `PublicCalibrationOnly`, `BundledCalibrationOnly`, and `SyntheticOnly`.

```rust
pub enum DataScopeDeclaration {
    PublicCalibrationOnly,
    BundledCalibrationOnly,
    SyntheticOnly,
    ExplicitUserOptInData,
}
```

`ExplicitUserOptInData` must not exist in production until a separate privacy design, consent flow, storage policy, deletion policy, and poisoning defense model exist. A job must not ask a device to upload personal prompts, conversation history, browser content, screenshots, microphone audio, local files, user embeddings, or personal-context graph data.

### 9. Contributor Privacy and Scheduling

Contribution is opt-in with aggressive defaults: disabled by default, require external power, require unmetered network, pause during active inference, pause during battery power. The local scheduler must never interfere with foreground Prism Agent inference. Foreground inference has absolute priority.

```rust
pub struct ContributorPolicy {
    pub enabled: bool,
    pub require_external_power: bool,
    pub require_unmetered_network: bool,
    pub max_cpu_percent: u8,
    pub max_gpu_percent: u8,
    pub max_temperature_celsius: f32,
    pub pause_when_inference_active: bool,
    pub pause_when_user_active: bool,
}
```

The scheduler state machine distinguishes 13 states from Disabled through Failed, including paused states for thermal, battery, inference, and user activity.

### 10. Trusted Verification Pipeline

Every candidate must pass independent verification before acceptance:

```
candidate proposal received
→ signature verification
→ base lineage verification
→ payload digest verification
→ tensor schema verification
→ contract parsing
→ segment bounds validation
→ CPU reference reconstruction
→ promotion-bank replay
→ holdout-bank replay
→ runtime kernel conformance replay
→ artifact resolution simulation
→ benchmark / cost validation
→ acceptance or rejection
```

The verifier runs against authoritative teacher data. For high-value updates, verification occurs on at least CPU reference and Metal backend. Contributor-generated metrics are advisory. The verifier's metrics are authoritative.

### 11. Monotonic Upgrade Rule

A candidate supersedes an existing generation only when all hard quality thresholds pass AND quality is not materially worse AND at least one cost or quality objective improves. A policy defines a small tolerance band for hardware numerical variation. For example: mean cosine may not decrease by more than 0.0001, operator NRMSE may not increase by more than 0.002, and payload bytes or estimated runtime cost must improve by a minimum percentage.

### 12. Poisoning Resistance

The system assumes some contributor devices are malicious, faulty, or compromised. Defenses include: replay validation of every candidate, immutable payload digests, expiring jobs, fail-closed parsing, bounded upload sizes, contributor rate-limiting, and quarantine of suspicious contributors. Contributor reputation may inform scheduling priority but must never replace independent verification.

### 13. Runtime Resolution

At startup, the runtime resolves the active model view: load base cimage, verify base seal, load trusted update index, select compatible signed updates, resolve the newest accepted generation per tensor ID, reject conflicting or unsupported updates, construct runtime tensor map, select kernel schedule, begin inference. Resolution is deterministic.

### 14. Rollback

Rollback is immediate and local. Every applied update has a parent generation. The runtime preserves at least one known-good prior generation until the new generation survives configurable local health checks. The user may disable all progressive updates and run only the sealed base cimage.

### 15. Local Health Checks Before Activation

Before activating an update, the runtime verifies the package signature, payload digests, MatrixContract bounds, required kernel availability, runs a compact CPU reconstruction probe and compact target-backend inference probe, and verifies output metrics against shipped receipt tolerances.

### 16. Compaction

After many accepted tensor updates, a new base cimage may be produced by resolving the current active generation for every tensor, rebuilding segment layout and MatrixContract, recomputing kernel specialization, running artifact replay validation and end-to-end parity validation, and sealing the next-generation cimage. Compaction triggers include update count exceeding policy limits, segment fragmentation, or enough tensors having migrated to ternary.

### 17. Revocation

An accepted update may be revoked with a signed revocation record specifying the update ID, reason code, affected tensor IDs, and optional replacement update ID. The runtime disables the revoked generation, restores the previous valid generation, and records a local rollback receipt.

### 18. Initial Rollout Scope

The first production version is intentionally narrow: one model family, one base cimage lineage, one tensor replacement update at a time, NF4 baseline targeting ternary migration for decoder MLP tensors, CPU and Metal verifier replay, public calibration corpus only, opt-in idle local compute only, no personal data, no economic rewards, no arbitrary residual overlays. The first target is a small set of decoder MLP tensors known to be likely ternary-friendly.

### 19. Suggested Initial Vertical Slice

The minimum viable progressive ternarization path is: qualified NF4 base cimage → one eligible decoder MLP tensor → signed public activation bank → local ternary Tile640 candidate generation → local checkpoint and resume → candidate upload → authoritative CPU + Metal replay → signed official tensor replacement → client downloads update → runtime resolves new ternary tensor → local health check → optional rollback.

## Consequences

### Positive

- **The model improves incrementally without redistribution.** A 300 MB update replacing one tensor is downloadable over any connection; a 12 GB full cimage is not.

- **Idle user compute becomes useful without privacy risk.** Jobs are deterministic, bounded, and operate only on public calibration data. No personal data leaves the device.

- **The separation of powers prevents abuse.** Contributors cannot authorize their own updates. Verifier replay is the single source of truth for acceptance.

- **Rollback is built in, not bolted on.** Every update has a parent, preservation of prior generations, and a user-facing disable switch.

- **The content-addressed protocol is network-resilient.** Objects are identified by digest, cached locally, and transferable over any transport (HTTPS, P2P, sneaker net).

### Negative

- **Operational complexity of the verifier.** A trusted verifier service must be maintained, monitored, and secured. Key rotation, revocation distribution, and replay infrastructure add surface area.

- **Determinism is hard across heterogeneous hardware.** A candidate that passes on an M1 Metal backend may produce different low-bit results on an Intel GPU or CPU. The policy tolerance band must be empirically calibrated per kernel class.

- **Storage for generation history.** Preserving prior generations for rollback doubles or triples per-tensor storage on device. This is bounded (one prior generation) but still significant for large models.

### Risks

- **Poisoning through calibration bank compromise.** If an attacker gains control of the verifier's signing key or the calibration corpus distribution channel, they can authorize malicious updates. Hardware-backed key storage and signed calibration artifacts mitigate but do not eliminate this.

- **Contributor incentive without financial rewards is uncertain.** Users may not opt in without visible benefit. The initial rollout should treat contribution as a power-user or developer feature until the quality impact is demonstrated.

- **Schedule evolution complexity.** If tensor updates change the optimal fused-kernel schedule, the runtime must handle schedule entries that reference specific generation digests. A schedule update without matching payload generations must be rejected.

- **The first vertical slice is deliberately narrow.** If decoder MLP tensors are the only target, the system works but the benefit is limited. Expanding to attention projections, vision bridges, and TTS heads requires broader validation pipelines and more calibration data.
