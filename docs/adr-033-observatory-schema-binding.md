# ADR-033: Schema Binding for Observatory v1 Evidence Artifacts

## Status

Proposed — research complete; schemas are partially bound. The gaps require follow-on ADRs.

## Context

`OBSERVATORY_V1_SPEC.md` v1.0 §10 names eight evidence-artifact schemas the site projects
onto its surfaces. Six of the names in §10 are placeholders ("the engine's `cimage-manifest`
schema (current version, named in the ADR)"). The constitutional rule is that the
schema name in §10 must bind to an actual schema identifier in the engine, or to a
schema explicitly created for the site by an ADR. This ADR performs that binding for
the artifacts that exist in the engine, names the artifacts that do not yet have a
canonical schema, and proposes the ADRs that close the gaps.

The spec also declares (§4.3) that JSON Schema is the canonical schema language,
checked into `schemas/` at the repository root. The current repository has no
`schemas/` directory. The JSON Schema files are a deliverable of this ADR and its
follow-ons; the engine types are the source of truth from which the JSON Schemas are
generated (`typify` direction per §4.3) or by hand for the artifacts whose Rust types
are awkward to project.

This ADR is the schema-binding deliverable promised by the spec's §14.1. It is signed
in the release log before Phase 3 (the manuscript) begins.

## Survey of existing engine types

The engine has many typed records that are candidates for the §10 artifacts. The
following table names the relevant types, the crate that owns them, and the §10
placeholder they map to (or the gap they expose).

| Engine type | Crate / file | Maps to §10 | Notes |
|---|---|---|---|
| `CImageHeader` | `prism-ecs-compile/src/cimage/mod.rs` | `cimage-manifest` (the public projection, not the binary header) | The binary `.cimage` file is a JSON header followed by 16 KB-aligned tensor payloads. The `CImageManifest` summary in the same file is a small, separately published projection. |
| `CImageManifest` | `prism-ecs-compile/src/cimage/mod.rs` | `cimage-manifest` (the summary record) | Already a small struct with `schema_version`, `source_digest`, `tensor_count`, `kernel_count`. The site projects this, not the full header. |
| `TensorRecord` | `prism-ecs-compile/src/cimage/mod.rs` | (part of `cimage-manifest`) | One entry per tensor in the cimage header. |
| `TensorPayloadEntry` | `prism-ecs-compile/src/cimage/mod.rs` | (runtime-only; not published) | The runtime's loaded tensor payload, not a schema-bound record. |
| `ValidationReceipt` | `prism-ecs-constitutional/src/compilation.rs` | `qualify-record` (closest match) | The generic validation result with `job_id`, `validator_type`, `passed`, `evidence_digest`, `validated_at`. Maps to qualification, not to compilation or execution. |
| `SearchSelectionReceipt` | `prism-ecs-compile/src/search.rs` | `compile-receipt` (closest match) | The representation-search selection record. `schema_version`, `search_id`, `evaluator`, `evidence_source`, `production_evidence`, `candidates_evaluated`, `selected_candidate_digest`, `receipt_digest`. This is the closest the engine has to a "compilation receipt." |
| `CommitReceipt` | `prism-ecs-constitutional/src/world_txn.rs` | (transactional commit; not an evidence artifact) | `committed_epoch`, `journal_length`, `event_count`, `advisory_event_count`. Used internally; not currently published as an evidence artifact. |
| `ReceiptCandidate` | `prism-ecs-constitutional/src/command.rs` | (generic envelope) | A wrapper for any receipt with `id`, `kind`, `payload`, `payload_hash`. Could be the basis for the site's receipt envelope. |
| `CimagePromotion` | `prism-ecs-constitutional/src/compilation.rs` | (promotion record; not an evidence artifact) | Records the promotion of a CImage to Sealed state. Not currently an evidence artifact, but a candidate for the "promotion" half of stage 12. |
| `HeterogeneousScheduleEvidence` | `prism-ecs-compile/src/search.rs` | (search evidence) | Records per-tile schedule decisions. Not currently an evidence artifact. |
| `BackendMeasurement` | `prism-ecs-compile/src/search.rs` | (search evidence) | Records measured candidate performance. Not currently an evidence artifact. |
| `ImageGenerationCapabilityManifest` | `crates/prism-image/src/manifest.rs` | (out of scope — image-generation, not the main engine) | The image-generation capability manifest. Exists for a different product. |
| `ImageQualificationRecord` | `crates/prism-image/src/manifest.rs` | (out of scope — image-generation) | The image-generation qualification record. Same. |
| `MaterializationReceipt` | `crates/prism-image/src/types.rs` | (out of scope — image-generation) | The image-generation materialization receipt. Same. |
| `ImageGenerationReceipt` | `crates/prism-image/src/types.rs` | (out of scope — image-generation) | The image-generation execution receipt. Same. |
| Legacy receipt files `evidence_*.json` | `compute-core.legacy/receipts/` | (legacy, archived under `docs/.legacy/`) | All are of kind `QualityGateResult`. Archaeology. Treated as `LegacyRemoved` per `CAMPAIGN.md`. Not part of the v1 evidence corpus. |

## Bindings

### Binding 1. `cimage-manifest` → `CImageManifest` projection

The §10 placeholder `cimage-manifest` binds to a new public-projection schema derived
from `CImageManifest` in `prism-ecs-compile/src/cimage/mod.rs`. The public projection
is intentionally smaller than the full `CImageHeader`; it is the summary the site
publishes on the ComputeImage and Specimen pages, not the full binary header.

```text
schema name:    cimage-manifest-v1
source type:    CImageManifest (prism-ecs-compile::cimage)
projection:     schema_version, source_digest, tensor_count, kernel_count
```

The full `CImageHeader` is the binary header inside the `.cimage` file; it is not
serialized as a standalone JSON document for publication. The site's Specimen page
links to the binary artifact and renders a summary; a visitor who wants the full
header opens the artifact.

A JSON Schema for the public projection is created at
`schemas/cimage-manifest.schema.json` as a Phase 2 deliverable. The schema is hand-
authored, not generated from the Rust type, because the public projection is a
deliberate subset.

### Binding 2. `compile-receipt` → `SearchSelectionReceipt`

The §10 placeholder `compile-receipt` binds to `SearchSelectionReceipt` in
`prism-ecs-compile/src/search.rs`. This is the closest match: it records the
representation-search selection, the search space, the chosen candidate, the
fallback reason, and the receipt digest. It does not currently record `commit`,
`target`, or `build_features` explicitly, but `production_evidence` and
`receipt_digest` are the provenance anchors.

```text
schema name:    compile-receipt-v1
source type:    SearchSelectionReceipt (prism_ecs_compile::search)
fields:         schema_version, search_id, evaluator, evidence_source,
                production_evidence, candidates_evaluated, measured_candidates,
                selected_candidate_digest, fallback_reason, receipt_digest
```

A JSON Schema is created at `schemas/compile-receipt.schema.json` as a Phase 2
deliverable. The schema is hand-authored; the Rust type is the source of truth.

### Binding 3. `qualify-record` → `ValidationReceipt`

The §10 placeholder `qualify-record` binds to `ValidationReceipt` in
`prism-ecs-constitutional/src/compilation.rs`. The mapping is not 1:1:
`ValidationReceipt` is a generic validation result; the spec's `qualify-record` is
specifically a target-specific qualification record with `commit`, `target`,
`build_features`, and `validation_scope`.

```text
schema name:    qualify-record-v1
source type:    ValidationReceipt (prism_ecs_constitutional::compilation) +
                qualification harness output (not yet a Rust type)
fields:         job_id, validator_type, passed, evidence_digest, validated_at,
                plus the harness-specific: commit, target, build_features,
                validation_scope
```

The engine does not currently have a `qualify-record` type. The qualification harness
produces a JSON file that is consumed by the SSG. A follow-on ADR (proposed below)
creates a `QualificationRecord` Rust type in `prism-ecs-constitutional` (or a new
`prism-ecs-qualification` crate) and the corresponding JSON Schema.

### Binding 4. `execute-receipt` → gap

The §10 placeholder `execute-receipt` does not currently bind to a Rust type in the
engine. The runtime emits receipts through `ReceiptCandidate` envelopes in
`prism-ecs-constitutional/src/command.rs`, but the typed execution receipt — with
identity, fencing generation, deadline, artifact digest, numerical policy, route, and
outcome — does not exist as a named type.

A follow-on ADR (proposed below) creates an `ExecutionReceipt` Rust type in
`prism-ecs-constitutional` (or a new `prism-ecs-runtime-evidence` crate). The type
is the canonical execution receipt. The JSON Schema is hand-authored at
`schemas/execute-receipt.schema.json`.

### Binding 5. `replay-result` → gap

The §10 placeholder `replay-result` does not currently bind to a Rust type. The
replay applier produces a result, but the result is currently logged, not typed and
not published.

A follow-on ADR creates a `ReplayResult` Rust type and the corresponding JSON Schema
at `schemas/replay-result.schema.json`. The type records the event log digest, the
replayed world digest, the equality of the two, and the time taken.

### Binding 6. `regression-finding` → gap

The §10 placeholder `regression-finding` does not currently bind to a Rust type. The
regression harness produces output (visible in
`crates/prism-ecs-quantization/src/ternarization/gates.rs` and related files), but
the output is not currently a typed evidence artifact.

A follow-on ADR creates a `RegressionFinding` Rust type and the corresponding JSON
Schema at `schemas/regression-finding.schema.json`.

### Binding 7. `provenance-chain` → gap

The §10 placeholder `provenance-chain` does not currently bind to a Rust type. The
provenance builder (mentioned in §10 as the producer) is a future component.

A follow-on ADR creates a `ProvenanceChain` Rust type and the corresponding JSON
Schema at `schemas/provenance-chain.schema.json`. The chain records the source
package digest, the semantic graph identity, the plan identity, the artifact digest,
and the digest of the events that produced each. A visitor can verify the chain
against the published evidence corpus.

### Binding 8. Failure variants

The §10 placeholder `execute-receipt` (failure variant) and the constitutional
principle that failure is a first-class outcome (§2.2 stage 11) require that the
`ExecutionReceipt` type support a `failure_class` field. The follow-on ADR for
Binding 4 includes this field. The `ValidationReceipt` already supports `passed:
bool`; the `QualificationRecord` follow-on records `failure_reason`.

## Decision

The bindings in this ADR are the canonical schema bindings for the v1 evidence
artifacts. The site projects the schemas named here. A reviewer who finds a
different schema name in the rendered site or in `evidence-index.json` rejects the
build as a schema-binding violation.

The JSON Schema files named in this ADR are created in Phase 2. The follow-on ADRs
are scheduled for Phase 2 as well; their order is below.

## Consequences

- The site can render the Specimen, Evidence, and Status pages against a known set
  of schemas, with `evidence-index.json` referencing the schema name in each record's
  `schema` field. The validator (§4.4) confirms the field is present and the schema
  exists.
- The engine types are the source of truth. The JSON Schemas are derived projections;
  the §4.3 generation direction (JSON Schema → Rust types via `typify`) does not
  apply to the engine, which is the source. The §4.3 direction applies to the
  data-layer schemas of §4.1 (`capabilities.json`, `evidence-index.json`, etc.), not
  to the engine's evidence artifacts.
- The gaps (Bindings 4–7) are scheduled as follow-on ADRs in Phase 2. If a follow-on
  ADR is not signed before Phase 3 begins, the corresponding §10 row is removed from
  the manuscript and the corresponding corpus slot is left empty with a one-line
  statement that the schema is not yet canonical.
- The legacy `QualityGateResult` receipts in `compute-core.legacy/receipts/` are not
  part of the v1 corpus. They are archaeology. They are visible in the repository
  history and may be referenced in the Specimen page's provenance chain if and only
  if a sanitized specimen is derived from them; otherwise they are not rendered.

## Follow-on ADRs (Phase 2)

- **ADR-034: `QualificationRecord` engine type and JSON Schema.** Creates the
  `QualificationRecord` Rust type in `prism-ecs-constitutional::qualification` (or a
  new `prism-ecs-qualification` crate). Hand-authors the JSON Schema at
  `schemas/qualify-record.schema.json`. The schema includes `commit`, `target`,
  `build_features`, `validation_scope`, `passed`, `failure_reason`. The
  qualification harness emits the type.
- **ADR-035: `ExecutionReceipt` engine type and JSON Schema.** Creates the
  `ExecutionReceipt` Rust type in `prism-ecs-constitutional::execution_evidence` (or a
  new `prism-ecs-runtime-evidence` crate). Hand-authors the JSON Schema at
  `schemas/execute-receipt.schema.json`. The schema includes `identity`, `fencing_generation`,
  `deadline`, `artifact_digest`, `numerical_policy`, `route`, `outcome`,
  `failure_class`. The runtime emits the type. The constitutional principle
  (failure is first-class) is enforced by the type's required `failure_class` field.
- **ADR-036: `ReplayResult` engine type and JSON Schema.** Creates the
  `ReplayResult` Rust type and the JSON Schema at `schemas/replay-result.schema.json`.
  The schema includes `event_log_digest`, `replayed_world_digest`, `equality`,
  `duration_ms`. The replay applier emits the type.
- **ADR-037: `RegressionFinding` engine type and JSON Schema.** Creates the
  `RegressionFinding` Rust type and the JSON Schema at
  `schemas/regression-finding.schema.json`. The schema includes `baseline_evidence_id`,
  `candidate_evidence_id`, `metric`, `value`, `threshold`, `verdict`. The
  regression harness emits the type.
- **ADR-038: `ProvenanceChain` engine type and JSON Schema.** Creates the
  `ProvenanceChain` Rust type and the JSON Schema at
  `schemas/provenance-chain.schema.json`. The chain records the source package
  digest, the semantic graph identity, the plan identity, the artifact digest, and
  the digest of the events that produced each. The provenance builder emits the
  type.

## Operational record

The bindings adopted by this ADR are recorded in `evidence-index.json` as the
`schema` field of each artifact record. The schema field is required by the
§4.4 validator. A record whose `schema` field is not in the set bound by this ADR or
by one of the follow-on ADRs is rejected at build time.
