# ADR-034: Evidence Freeze for Observatory v1

## Status

Proposed — survey complete. Specimen selection pending architect signoff (H7).

## Context

`OBSERVATORY_V1_SPEC.md` v1.0 §10 names eight evidence artifacts the site must
publish, and §10 explicitly says: "Specimen selection happens in Phase 2. The
architect selects, with the corpus authors, which artifacts become the public
specimens. The selection is recorded in `evidence-index.json` and frozen before
Phase 3 begins; the manuscript is authored around the selected specimens."

This ADR is the evidence-freeze deliverable. It surveys what evidence actually
exists in the repository, names the gaps, selects specimens from what is available
with explicit sanitization where the source is synthetic or sensitive, and records
the decisions in `evidence-index.json` (Phase 2 build artifact).

The survey is honest. The engine is pre-1.0; it does not currently produce a
publishable evidence corpus end-to-end. The site cannot pretend otherwise. The
freeze is a snapshot of what is real today, with a clear path to the corpus the
spec imagines.

## Survey of existing evidence

The repository contains the following candidate evidence, in two categories:
artifacts produced by the engine (the live codebase) and artifacts in
`compute-core.legacy/` (the archived archaeology).

### Engine-produced artifacts

| Path | Size | Real or test? | Notes |
|---|---|---|---|
| `crates/prism-ecs-compile/output.cimage` | 4,194,304 bytes | test | Synthetic test fixture. `source_digest: "test-digest"`. Not a real model artifact. |
| `crates/prism-ecs-compile/test-digest.cimage` | 4,194,312 bytes | test | Same synthetic family. `source_digest: "test-digest"`. |
| `crates/prism-ecs-compile/test_digest.cimage` | 4,325,691 bytes | test | Same synthetic family, larger. `source_digest: "test-digest"`. |
| `crates/prism-ecs-compile/test-digest-build-plan.cimage` | 4,194,336 bytes | test | Same synthetic family. `source_digest: "test-digest"`. |
| `crates/prism-ecs-compile/test-digest-certify-accepts.cimage` | 4,194,312 bytes | test | Same synthetic family. `source_digest: "test-digest"`. |
| `crates/prism-ecs-compile/test-digest-certify-rejects.cimage` | 4,194,312 bytes | test | Same synthetic family. `source_digest: "test-digest"`. |
| `crates/prism-ecs-compile/test-digest-write-plan-bytes.cimage` | 4,194,312 bytes | test | Same synthetic family. `source_digest: "test-digest"`. |

All seven engine cimage files are test fixtures with `source_digest: "test-digest"`
and synthetic tensor shapes. They are not produced from a real model; they exercise
the encode/decode paths of the cimage module.

The engine does not currently produce any other evidence artifacts. The receipt
types in `prism-ecs-constitutional` and `prism-ecs-compile::search` are defined as
Rust types (per ADR-033) but are not yet emitted to stable, publishable storage by
any pipeline the repository exposes. There are no `execute-receipt` files, no
`qualify-record` files, no `replay-result` files, no `regression-finding` files,
and no `provenance-chain` files anywhere in the repository.

### Legacy artifacts

| Path | Notes |
|---|---|
| `compute-core.legacy/output.cimage` | A cimage produced by the previous codebase. Not produced by the current engine. `source_digest: "77899b405c509717c4493b73cacd9796fc18b4635ab0098fc2d1630a447cbcdc"`. |
| `compute-core.legacy/receipts/evidence_manifest.json` | The manifest of 64 quality-gate receipts. |
| `compute-core.legacy/receipts/evidence_1.json` through `evidence_64.json` | All of `receipt_kind: "QualityGateResult"`. All reference the same `manifest_digest`. These are the gate results from a previous (pre-constitutional) pipeline. |

The legacy artifacts are archaeology. `CAMPAIGN.md` marks the relevant subsystems
as `LegacyRemoved`. The site does not project them on the v1 surfaces; the
Specimen page's provenance chain may reference them only if and when a sanitized
specimen is derived from one of them with explicit declarations.

## The v1 corpus, as it can be frozen today

Given the survey, the v1 evidence corpus is constrained. The site cannot honestly
project a measurement that does not exist. The spec's §4.8 sanitization model
permits publishing artifacts whose source is a test fixture, provided the site
declares both digests and the redaction manifest. The v1 freeze does the
following:

### Selected specimens

The site publishes **one sanitized ComputeImage specimen**, derived from
`crates/prism-ecs-compile/output.cimage`. The specimen is sanitized because its
source is a test fixture, not a real model.

- **`computeimage_artifact_digest`:** the content hash of the published `.cimage`
  bytes. (A real cimage's bytes are content-addressed; the published bytes are the
  bytes the visitor downloads.)
- **`original_artifact_digest`:** the content hash of the test fixture's bytes
  (identical to the published digest in this case, because the sanitization is
  declarative, not byte-modifying).
- **`redaction_manifest`:** a structured list stating that the tensor payload bytes
  are synthetic test data; that `source_digest` is the placeholder string
  `"test-digest"`; that `model_family` and `model_config_json` are placeholder
  values; that no real model identity, real source package, or real measurement is
  implied. The site displays the manifest verbatim.
- **`remaining_verifiable`:** `digest_of_source_package` (the test fixture's
  digest is the digest of source; this is verifiable). The other members of the
  closed set (`semantic_graph_identity`, `execution_target_identity`,
  `route_identity`, `evidence_chain_integrity`) are not verifiable because the
  source is not a real model and the chain does not extend to a real execution.
  They are explicitly named as not-verifiable in the manifest.

The specimen is rendered at the Specimen page (`/computeimage/specimen/`) with the
six strata exposed and the redaction manifest displayed in plain prose. The home
page's Central Object section references the specimen at technical density.

### Gaps that prevent full §10 coverage

The v1 corpus cannot satisfy the full §10 inventory. The site displays the gaps
honestly rather than papering over them. The following artifacts are not in the v1
corpus and the corresponding Specimen / Evidence pages render a one-line statement
to that effect:

- **Compilation receipt:** not in the corpus. The `SearchSelectionReceipt` type
  exists (per ADR-033) but is not currently emitted to stable storage by a
  pipeline. The Evidence page's "Receipt Chain" section shows the type and a
  schema placeholder, and states the gap.
- **Execution receipt (success):** not in the corpus. The `ExecutionReceipt` type
  does not yet exist; the follow-on ADR-035 will create it. Until then, the
  Evidence page's "Receipt Chain" section does not show a success execution
  receipt.
- **Execution receipt (failure):** not in the corpus. Same reason. The
  constitutional principle (failure is first-class) is preserved in the manuscript;
  the rendered surface shows the gap.
- **Qualification record:** not in the corpus. The `QualificationRecord` type
  does not yet exist; ADR-034 will create it. Until then, the Status page's
  capabilities are noted as "implemented" or "qualifying" without a qualification
  record attached.
- **Replay result:** not in the corpus. The `ReplayResult` type does not yet
  exist; ADR-036 will create it.
- **Regression finding:** not in the corpus. The `RegressionFinding` type does
  not yet exist; ADR-037 will create it.
- **Provenance chain:** not in the corpus. The `ProvenanceChain` type does not
  yet exist; ADR-038 will create it. The Specimen page's provenance chain
  displays the gap with a one-line statement.

### What the Status page can honestly say in v1

The Status page is the constitutional tension point. §3 forbids status words
without evidence; §6.7 (Status brief) requires "a one-line limit" on every row.
The v1 Status page is honest by being small.

A possible v1 Status page, to be reviewed by the architect (H7):

```text
Targets
  Apple Silicon (M-series)        implemented   src: prism-*-runtime crates
                                                    commit: <current>
                                                    evidence: 1 sanitized cimage
  Linux x86_64                    implemented   src: prism-*-runtime crates
                                                    commit: <current>
                                                    evidence: none published
  Other accelerators              planned       ADR: roadmap/target-qualification.ane

Backends
  In-memory CPU                   implemented   no measurement published
  ANE (Apple Neural Engine)       planned       ADR: ADR-031 (ROCm / AITER-ATOM)
  ROCm / XDNA / NPU               planned       same

Models
  <no models admitted>            planned       no real model identity in v1

Routes
  <no routes admitted>            planned       no real route in v1
```

The "1 sanitized cimage" row is the published specimen. Every other row is a
state from §3 with a source path and a named limit. The page does not say
"validated" anywhere, because no `ValidationReceipt` has been emitted for a real
target; the ValidationReceipts emitted by tests are not published.

The architect can adjust the rows during the Phase 2 review. The point is that
the v1 Status page is a small, honest list, not a marketing surface.

## The full v1 corpus over time

The v1 freeze is the starting point, not the destination. The full §10 corpus is
built incrementally as the engine produces the artifact types named in ADR-033 and
its follow-ons. The freeze is reopened (a new ADR is signed) whenever:

- A new artifact type is added to the §10 inventory.
- A published specimen is updated, replaced, or sanitized.
- A previously gap-flagged artifact becomes available.

The release log records every freeze and every reopen. The current freeze is
ADR-034; the next freeze is named in §14.13 (conditional route evaluation) and
the §10 evidence types as they become canonical.

## Decision

The v1 evidence corpus is:

1. **One sanitized ComputeImage specimen** derived from
   `crates/prism-ecs-compile/output.cimage`, with the redaction manifest
   declared per §4.8.
2. **No other artifacts** in v1. The other §10 slots are gap-flagged.

The selection is recorded in `evidence-index.json` as the Phase 2 build
artifact. The Specimen page exposes the specimen. The home page's Central
Object references it at technical density. The Evidence page's "Receipt Chain"
section shows the receipt type schema (where it exists) and a one-line gap
statement (where it does not). The Status page is the small, honest list
above (or its architect-approved variant).

The architect's signoff (H7) is required before the freeze is binding. The
reviewer confirms:

- The sanitization of the published specimen is honest and complete.
- The gap-flagged artifacts are correctly named in the manuscript.
- The Status page's "1 sanitized cimage" row (or its variant) is acceptable.
- No §10 artifact is omitted from the gap statement; the gap statement is
  the explicit omission list, and the spec requires every §10 row to either
  have a published specimen or a named gap.

## Consequences

- The v1 site is honest about the size of its evidence corpus. The visitor
  sees a Specimen page with a clear sanitization declaration, an Evidence page
  with gap statements, and a Status page that is a small list of source
  paths and limits, not a marketing surface.
- The v1 manuscript is authored around this honest freeze. The Editorial
  Constitution rule 3 (every technical claim has an evidence boundary) is
  satisfied because every status word on the surface has a named limit and
  a source path.
- The follow-on ADRs (ADR-035 through ADR-038) close the gaps incrementally.
  Each closure is a freeze reopen, not a constitutional change.

## Follow-on work

- The architect reviews the freeze (H7) and signs or amends it.
- A new freeze is opened after ADR-035, after ADR-036, after ADR-037, and
  after ADR-038 land, naming the new artifacts that enter the corpus.
- The release log records the freeze and every reopen.
