# Prism Observatory v1 — Page Manuscript

**Status:** Draft for H1–H13 review. Binding artifact for Phase 3.
**Authority:** The complete prose of every canonical page. Component authors populate renderers from this document; the briefs in `OBSERVATORY_V1_SPEC.md` §6 are its outline, not its substitute.
**Companion documents:**
- `OBSERVATORY_V1_SPEC.md` v1.0 — governance.
- `docs/adr-032-observatory-deployment-platform.md` — Cloudflare Pages contract.
- `docs/adr-033-observatory-schema-binding.md` — schema bindings.
- `docs/adr-034-observatory-evidence-freeze.md` — v1 evidence corpus.

---

## Preamble

This is the prose the visitor reads. It is authored around the v1 evidence corpus
(ADR-034): one sanitized ComputeImage specimen, no execution receipts, no
qualification records, no replay results. The manuscript names what is real,
declares what is synthetic, and states what is missing. A page that hides a limit
fails H12; a page that uses a status word without a definition fails H1; a
paragraph that exists only to sound important fails H3.

The voice is technical, honest, and quiet. The site is not marketing. It is a
projection of a system that exists, with the limits of the projection named in
the projection.

**Conventions in this manuscript:**

- `code` marks identifiers, paths, file names, schemas, and commands. Prose does
  not use `code`; identifiers do not use proportional.
- *Italics* mark status words from the §3 vocabulary and other defined terms on
  first use. Subsequent uses are unmarked.
- The character `→` marks a transition from one stage or one page to the next.
- Square brackets `[select: id]` mark a selection that a runtime component
  surfaces. They are not rendered as text; they are markers for the renderer.

---

## Page 1 — Home (`/`)

**Brief:** §6.1.

### Hero

# Compile intelligence into something you can inspect.

Most inference runtimes make the most consequential decisions late, implicitly,
and ephemerally. The optimizer chooses a layout after the model is loaded. The
scheduler picks a lane after the request arrives. The numerics policy is decided
inside a kernel the operator cannot see. Each decision disappears the moment the
run ends.

Prism does the opposite. The decisions happen during compilation, named, signed,
and stored in a single artifact. The runtime executes the plan. The plan is
inspectable. The receipt is durable.

### The Central Object

A ComputeImage is one file. It is the boundary between compilation and execution.
It contains a model's logical identity, its physical layout, the execution views
it admits, the plan it was compiled under, the receipts that admit it, and the
payloads it carries.

[Specimen at technical density: the six strata are visible, with one selected
tensor illuminated. The Specimen page (`/computeimage/specimen/`) is the
authoritative surface for this object; the home page references it.]

The object on the home page is a sanitized specimen. The redaction manifest is
visible on the Specimen page. It says, in plain prose, what is synthetic and
what is verifiable.

### Current Reality

The capability map below is the small, honest list. It names what is implemented
in the source today, what qualifies against a named target, and what is still
planned. Every row carries a source path and a limit. No row claims measurement
the evidence does not support.

[Status table with four rows, drawn from `capabilities.json`:]

| Target / backend | State | Source | Evidence |
|---|---|---|---|
| Apple Silicon (M-series), in-memory CPU | *Implemented* | `prism-ecs-runtime/` | one sanitized cimage, no executed receipt |
| Linux x86_64, in-memory CPU | *Implemented* | `prism-ecs-runtime/` | source reviewable, no published receipt |
| ANE (Apple Neural Engine) | *Planned* | ADR-031 | no code path |
| ROCm / XDNA / NPU | *Planned* | ADR-031, roadmap | no code path |

A row marked *Planned* is a maturity claim, not a validation claim. Validation is
reserved for *Validated* records. The transition history for every capability is
preserved in `capability-history.json`; the current state is the most recent
transition's destination.

### The Journey at a Glance

A model becomes a ComputeImage, is admitted into constitutional execution, runs
across explicit hardware paths, and produces evidence. The sequence is twelve
stages. Each page of this site is one lens over the sequence.

[Horizontal strip of twelve stages, from "Source package ingest" to "Replay,
comparison, and promotion." Each stage is a one-sentence description. The whole
strip links to the Observatory's interactive view at `/observatory/life/`.]

### Compiler Lab (Recorded)

A recorded compilation run, bound to a specific build, target, model, and
feature set. The numbers below are the values the run produced. They are not
streaming. They are not live. They are the receipt's evidence.

[Recorded values from the v1 corpus, when one exists. For the v1 manuscript,
this section names its absence: "No recorded run is published in v1. The
section will populate when the corpus admits a real compilation receipt."]

### The Evidence Contract

An output is not authoritative because a backend returned it. A result that does
not survive a restart is not authoritative. A result for an expired or
superseded lease is rejected at the boundary, not silently accepted.

The receipt carries identity, fencing generation, deadline, artifact digest,
numerical policy, route, and outcome. The receipt is durable. The receipt is the
evidence. The Evidence page (`/evidence/`) names what the corpus currently
contains and what it does not.

### Where to begin

Three links, not a menu.

- *Start* (`/start/`) — if this is your first visit.
- *Architecture* (`/architecture/`) — if you read systems for a living.
- *Status* (`/status/`) — if you are evaluating whether Prism does what it says.

The Observatory at `/observatory/life/` is the deeper experience. It is reached
from inside the relevant primary page, not from the home page directly. The home
page is the entrance. The Observatory is the room.

---

## Page 2 — Start (`/start/`)

**Brief:** §6.2.

### The Short Version

Prism is a compiler and runtime for machine-learning models on local hardware.
Its single product is the ComputeImage: a sealed artifact that names the
decisions made at compile time and binds them to the plan the runtime executes.

The compiler decides representation, layout, target legality, and numerical
policy. The decisions are typed operations on the constitutional ECS. They are
recorded in a plan. The plan has an identity; re-planning produces a new plan
identity and a new admission record.

The runtime is the executor of the plan, not its author. It may not alter
semantic, representation, legality, or numerical-policy decisions in an admitted
plan. It consults the plan for those decisions and the world for everything
else: admission, ownership, leases, deadlines, transaction validity, terminal
failure.

A request enters the world through a typed command, is admitted through a
transaction, and is recorded as a durable event. The transaction is the only
path to canonical state change. The receipt is the only path to authority. The
output is not authoritative without the receipt.

A failure is a first-class outcome. It is emitted through the same receipt path,
with the same provenance, with a defined failure class. Failure is not a
regrettable variant of success; it is part of what the system records.

The vocabulary is closed. There are four capability states — *Planned*,
*Implemented*, *Qualifying*, *Validated* — and two distribution states —
*Unreleased*, *Released*. The site does not use *supported*, *ready*, *active*,
*live*, or *available* as status words. Every status word on this site has a
definition.

### The Reading Path

The path starts with motivation, ends with the source. The vocabulary is the
same at every step.

- **Motivation** — the home page's argument, restated. *Why Prism exists.*
- **Architecture** — *How Prism is organized.* The three primary contracts, the
  compiler as search, the runtime as authority, the plan as the boundary.
- **ComputeImage** — *The artifact the compiler produces.* Six inspectable
  strata, three identities, the receipt, the search history.
- **Evidence** — *What the runtime records.* The receipt chain, replay,
  failure, the corpus.

Each step links to its route. The path does not branch. The path is read in
order.

### What Prism Is Not

Prism is not an inference runtime with a compiler bolted on. It is a compiler
with a runtime built to execute its plans. The distinction is not marketing. The
runtime cannot rewrite the plan. The plan is the boundary.

Prism is not a multi-cloud orchestrator. It does not claim universal backend
support. The capability map names which targets are implemented, which qualify,
and which are still planned. Every row is a source path and a limit. Nothing
is inferred.

Prism is not a research toy. The architecture is concrete; the artifacts are
inspectable; the receipts are durable. The plan/runtime split is enforced by
types, not by convention. The Constitution is read.

Prism is not a finished product. The roadmap contains future work. The status
page contains present work. The lab contains exploratory work. The three are
not the same.

### The Shape of the Site

The home page is the entrance. The primary navigation is *Start, Architecture,
Evidence, Status, GitHub*. Everything else orbits.

The Observatory at `/observatory/life/` is the deeper experience. It follows
one specimen through the canonical journey. The selection state persists in the
URL. The JavaScript-free fallback is the same authored work without
synchronization.

The Colophon at `/colophon/` is the close. It names the author and the
collaboration model. It is the only place on the site that explains who is
building Prism and how engagement works.

---

## Page 3 — Architecture (`/architecture/`)

**Brief:** §6.3.

### One Subject, Three Contracts

A model arrives. It becomes a ComputeImage. The ComputeImage is admitted to
execution. The execution produces a receipt. The receipt is the only authority.

The journey depends on three contracts. Each contract names a decision. The
contracts do not overlap.

- **`LogicalTensor`** is the meaning. It is the tensor as the model defines it:
  shape, dtype, semantic family, role in the graph. Changing a layout does not
  change a logical tensor. Changing a representation does not change a logical
  tensor. The logical tensor is the model's claim about what the data is.
- **`PhysicalTileLayout`** is the storage. It is the tensor as the hardware
  stores it: tile sizes, padding, alignment, memory tier. The same logical
  tensor can have many physical layouts. A physical layout is a candidate
  transformation, evaluated against the model, the target, and the constraints.
- **`ExecutionView`** is the consumption. It is the contract by which a backend
  reads the data at execution time. An execution view names the lane, the
  provider, and the dispatch path. A model may have many execution views, one
  per lane, per provider, per plan.

A change to one contract does not silently rewrite another. A logical identity
is preserved across representations because the representation is bound to the
plan, not to the tensor. A physical layout is preserved across lanes because
the layout is the artifact, not the kernel. An execution view is preserved
across runs because the view is durable state, not runtime cache.

### The Compiler as Search

Compilation is a search. The search has a state space: every legal combination
of representation, layout, target constraint, and numerical policy. The search
has a cost model: the time, the memory, the precision, the numerical error. The
search has a result: a plan, admitted as durable state.

The candidate transformations are typed. A precision change is a typed
operation. A sparsification is a typed operation. A ternarization is a typed
operation. A layout change is a typed operation. Search policy may be
heuristic, evolutionary, learned, sampled, or cost-model driven. The admitted
candidate, its constraints, and its evidence are explicit. The search process
is not required to be.

The visitor sees the search space. The visitor sees the chosen path. The visitor
sees the rejected alternatives. The site is not a summary of the choice; it is
the evidence of the choice.

### The Runtime as Authority

The runtime executes the plan. It does not invent decisions. It may instantiate
only the runtime choices explicitly permitted by the plan: lane selection
within a permitted set, scheduling within declared residency policy, queue
ordering, retry within declared bounds. Anything outside those bounds is a
different plan, with a new identity, and a new admission record.

Canonical world state is the authority for everything the plan does not cover:
admission, ownership, leases, deadlines, fencing generations, transaction
validity, terminal failure. The runtime consults both. The plan does not replace
the world. The world does not replace the plan.

A result for an expired or superseded lease is rejected at the boundary, not
silently accepted. A result that does not survive a restart is not authoritative.
A failure is recorded. A recovery is recorded. The recording is the receipt.

### The Plan as the Boundary

The plan is the boundary between compilation and execution. The plan is sealed
when it is admitted. The seal is durable. The seal is the admission record.

A re-plan is a new admission. A different representation, a different layout, a
different target — each produces a new plan identity. The new plan may reuse
parts of the old plan's evidence, but it is a new plan. The artifact digest
changes if the materialization changes.

The boundary is not a convention. It is a type. The runtime cannot construct a
plan. The compiler cannot construct a request. The two meet at the artifact
and at the transaction.

### What the Evidence Supports Today

The architecture is implemented. The contracts exist. The compiler exists. The
runtime exists. The receipt path exists.

What is *Validated* on a real target: nothing in the v1 corpus. The receipt
types exist as Rust definitions; the engine has not yet emitted one to
publishable storage. The `ValidationReceipt` and `SearchSelectionReceipt` types
are bound to the v1 schema set (per ADR-033) and are the canonical types. A
follow-on ADR emits them to a stable, content-addressed path. Until then, the
site names the gap and shows the schema.

What is *Implemented*: the source code paths for compilation, the cimage
encoder/decoder, the constitutional command set, the receipt envelope
(`ReceiptCandidate`), the world transaction layer. Each row above carries a
source path; a reviewer can read the entry point and reason about its control
flow.

What is *Planned*: heterogeneous target qualification, ANE/CUDA/NPU providers
(ADR-031), end-to-end receipt emission to publishable storage, a real
qualification harness output. Each is a milestone with an exit criterion in the
roadmap.

---

## Page 4 — ComputeImage (`/computeimage/`)

**Brief:** §6.4.

### The Six Strata

A ComputeImage is one file. It is divided into six inspectable strata. Each
stratum has a purpose. Each is named.

1. **Metadata** — the model's identity, the source digest, the schema version,
   the producer, the producer's commit. The metadata is what makes the artifact
   a content-addressed object, not a renamed checkpoint.
2. **Logical tensors** — the model's logical view. Shapes, dtypes, semantic
   families, roles. The logical tensor is the model's claim.
3. **Physical layouts** — how the tensors are stored. Tile sizes, alignment,
   padding, memory tier. The physical layout is what the runtime reads.
4. **Execution views** — how a backend consumes the layouts. Lane, provider,
   dispatch path. The execution view is the contract by which the runtime
   executes.
5. **Plan and receipts** — the admitted plan and the receipts that admit it.
   The plan is durable. The receipts are durable. Together they are the
   evidence the runtime has the authority to execute.
6. **Payloads** — the tensor bytes, aligned to a 16 KB page boundary so the
   runtime can mmap them without parsing. The payloads are the data.

The six strata are not layers in a stack. They are aspects of one artifact. A
cimage has all six or it is not a cimage.

### The Three Identities

A model is not a single identity. Three identities are preserved across the
journey, and the site names all three because conflating them is a category
error.

- **`source_artifact_digest`** is the content hash of the source package as it
  arrived. The bytes. The same bytes, the same digest.
- **`semantic_graph_identity`** is the content hash of the canonical
  computational graph, recovered under a declared schema, declared operator
  semantics, and a declared normalization policy. The declaration is part of
  the identity.
- **`computeimage_artifact_digest`** is the content hash of the produced
  `.cimage` bytes. The artifact. The same plan, the same payload bytes, the
  same digest.

The site does not use the phrase "same effective computation." The site names
the declaration that produced each identity. A reviewer who wants to verify an
identity can do so against the declaration, the source, and the artifact
bytes.

### Legality

A plan is legal for a target if every operator it references admits a legal
lowering for that target, every tile size is admitted, every memory tier is
admitted as a target property, and every KV-cache policy is permitted.
Legality is decided at compile time. The artifact carries the legalization
report.

The capability map names which targets are implemented in the source, which
qualify against fixture tests, and which have been validated end-to-end with a
receipt. A cimage is *legal for a target* in the sense of "this cimage was
admitted with a legalization report for that target." It is not legal in the
sense of "this cimage will run correctly" until execution has produced a
receipt.

The v1 corpus does not contain a legalization report. The site names the gap.
The cimage on the Specimen page has no execution target — it is a test
fixture, sanitized.

### The Receipt

A cimage is admitted by a receipt. The receipt is durable. The receipt carries
identity, fencing generation, deadline, artifact digest, numerical policy,
route, and outcome.

The v1 corpus does not contain an execution receipt. The `ExecutionReceipt`
type is bound to a follow-on ADR (per ADR-033, ADR-035). The site does not
display a receipt it does not have. The Evidence page names the gap.

A failure receipt is the same shape. The `failure_class` field is required by
the type, not optional. The constitutional principle — failure is first-class
— is enforced by the type, not by a convention.

### Inspecting the Artifact

The cimage file format is described in the engine's cimage module. A developer
who wants to inspect a cimage opens it. The header is JSON. The payloads are
16 KB-aligned. The file is mmap-able.

A reviewer reads the header. The header names the source, the model family, the
tensor records, the legalization report (if any), the compilation events, the
search trace, the model manifest (if multi-model), the execution plan (if
heterogeneous). Each named field is auditable without reopening the source
model.

The Specimen page (`/computeimage/specimen/`) is the artifact browser. The
ComputeImage page is the chapter. The chapter teaches the concept. The
Specimen exposes the bytes.

---

## Page 5 — ComputeImage Specimen (`/computeimage/specimen/`)

**Brief:** §6.5.

### Orientation

This is an evidence record, not a chapter. The artifact is a sanitized
ComputeImage. Its public projection is the bytes served at this URL. Its
original source is the test fixture
`crates/prism-ecs-compile/output.cimage`.

- **`computeimage_artifact_digest`:** `<digest of the served bytes>`.
- **`source_artifact_digest`:** `<digest of the test fixture bytes>`.
  (Identical to the public projection in this case, because the sanitization
  is declarative, not byte-modifying.)
- **`plan_identity`:** not produced for a test fixture. The plan field is
  absent from the published header.
- **`receipt`:** not produced. The corpus has no execution receipt for this
  artifact. The v1 site names the gap.
- **`validation_scope`:** none. The artifact is a test fixture. No real
  target, no real run, no real measurement.

### Redaction manifest

The published bytes are sanitized. The redaction is declarative. The manifest
is:

- The tensor payload bytes are synthetic test data. The single tensor in this
  fixture is `weight`, shape `[4, 4]`, dtype `f16`, 32 bytes of synthetic
  weight values.
- The `source_digest` in the header is the placeholder string
  `"test-digest"`. It does not identify a real model package.
- The `model_family` and `model_config_json` are placeholder values.
- No real model identity, no real source package, no real measurement, and no
  real execution target is implied.

### Remaining verifiable

The closed set of verifiable relationships is declared in the spec (§4.8). For
this specimen:

- **`digest_of_source_package`:** *verifiable.* The source bytes are the test
  fixture bytes; their digest is the public digest. A reviewer can re-hash
  the file and confirm.
- **`semantic_graph_identity`:** *not verifiable.* The source is not a real
  model, so no canonical graph is recovered.
- **`execution_target_identity`:** *not verifiable.* The artifact has no
  execution target.
- **`route_identity`:** *not verifiable.* No route was admitted.
- **`evidence_chain_integrity`:** *not verifiable.* The chain does not extend
  to a real execution; the receipts it would link to are not published.

### The data

The six strata are exposed as labeled blocks. Each block contains the relevant
header field, rendered as data, with a short orientation. There is no
explanatory prose. The Specimen page is to the ComputeImage chapter what a
database record is to a chapter that references it: same fact, different
surface.

- **Metadata block:** schema version, source digest (placeholder), producer
  (`prism-ecs-compile`).
- **Logical tensors block:** the single `weight` tensor; shape, dtype,
  semantic family.
- **Physical layouts block:** the `weight` tensor's physical layout; offset,
  size, alignment.
- **Execution views block:** absent. The artifact has no execution views.
- **Plan and receipts block:** absent. The artifact has no plan or receipts.
- **Payloads block:** the synthetic weight bytes, displayed as 32 hex
  characters. The bytes are not informative; the page is honest that they are
  synthetic test data.

### Provenance chain

The chain is one link: the test fixture, with its commit. The fixture is
checked into the engine crate. The commit is the build identity. The link is
verifiable. The chain does not extend further. The site names the limit.

---

## Page 6 — Evidence (`/evidence/`)

**Brief:** §6.6.

### The Epistemic Contract

An output is not authoritative because a backend returned it. The contract is:

- Durable state change is recorded before acknowledgement. A result that does
  not survive a restart is not authoritative.
- A result for an expired or superseded lease is rejected at the boundary, not
  silently accepted.
- Failure is recorded. A failure is a first-class outcome. It is emitted
  through the same receipt path, with the same provenance, with a defined
  failure class.

The receipt carries identity, fencing generation, deadline, artifact digest,
numerical policy, route, and outcome. The receipt is durable. The output is
not authoritative without the receipt. A visitor who wants to know what
happened reads the receipt. A reviewer who wants to know why reads the
provenance.

### The Receipt Chain

A receipt links a model identity, an artifact digest, a request, a route, and
an outcome. The chain is the link from the source model to the result. Every
link is preserved. Every link is verifiable.

The v1 corpus has no execution receipt. The `ExecutionReceipt` type is bound
(ADR-033); the engine has not yet emitted one to publishable storage. A
follow-on ADR (ADR-035) creates the emission pipeline. Until then, the
Receipt Chain section shows the type's fields, names the gap, and links to
ADR-035 for the close.

A failure receipt has the same shape as a success receipt, with a populated
`failure_class` field. The constitutional principle — failure is
first-class — is enforced by the type, not by a convention. The site does not
display success without showing that failure is part of the same surface.

### Replay

Replay reads the durable event log, re-derives the canonical world, and
projects to the surface. Replay does not re-run compilation, inference,
network requests, file writes, or device allocation. It re-derives state from
durable facts.

The v1 corpus has no replay result. The `ReplayResult` type is bound
(ADR-033); the engine has not yet emitted one. A follow-on ADR (ADR-036)
creates the emission pipeline. The site names the gap.

Replay is not a debugging tool. Replay is the constitutional guarantee that
state is reconstructable from durable events. The Replay section explains the
guarantee; the corpus shows its operation when one exists.

### Failure as a First-Class Object

A failure is a receipt. The receipt has a `failure_class` field. The field
names the kind: `stale_outcome`, `provider_failure`, `restart_recovery`,
`projection_loss`, `validation_failed`. The class is a closed set. The class
is part of the receipt's type. A receipt without a `failure_class` is not a
failure receipt.

The site presents failure with the same visual dignity as success. A
rejected plan, a stale outcome, a failed qualification, and a recovered
transaction are not exceptions to the visual language. They use the same
Receipt component as successful artifacts.

The v1 corpus has no failure receipt. The type is bound; the gap is named.

### The Evidence Corpus

The corpus is the published set of evidence artifacts. Every artifact has a
stable URL. Every artifact's identity is content-addressed. The corpus is the
only authority from which measurements may be projected.

The v1 corpus is one artifact: the sanitized ComputeImage on the Specimen
page. The other §10 slots are gap-flagged. The corpus page lists every slot
and its current state.

A reader who comes to the corpus looking for a measurement finds a small
honest list. The page is not a marketing surface. It is the source of truth
for what the project has published.

---

## Page 7 — Status (`/status/`)

**Brief:** §6.7.

### The status surface

The Status page is the authoritative answer to *what exists, what qualifies,
and what has been validated* — on which hardware, against which commit, with
which evidence. Every row is a record in `capabilities.json`. Every cell is a
state from the closed vocabulary. Every cell has a source path and an
evidence class.

A row marked *Planned* is a maturity claim, not a validation claim.
Validation is reserved for *Validated* records. Capability transitions are
visible in the history.

The v1 status page is small. Every row below is source-reviewable. No row
claims measurement the evidence does not support.

### Targets

| Target | State | Source | Evidence |
|---|---|---|---|
| Apple Silicon (M-series), in-memory CPU | *Implemented* | `prism-ecs-runtime/`, `prism-image/` | one sanitized cimage, no executed receipt |
| Linux x86_64, in-memory CPU | *Implemented* | `prism-ecs-runtime/`, `prism-image/` | source reviewable, no published receipt |
| ANE (Apple Neural Engine) | *Planned* | ADR-031; roadmap | no code path |
| ROCm (MI300X-class) | *Planned* | ADR-031 | no code path |
| XDNA / XDNA2 (AMD NPU) | *Planned* | roadmap | no code path |
| Other accelerators | *Planned* | roadmap | no code path |

### Backends

| Backend | State | Source | Evidence |
|---|---|---|---|
| In-memory CPU | *Implemented* | `prism-ecs-runtime/` | source reviewable |
| Disk-backed mmap | *Implemented* | `prism-ecs-compile::cimage` | source reviewable |
| Provider-delegated kernels | *Planned* | ADR-031, roadmap | no published receipt |

### Models

| Model identity | State | Source | Evidence |
|---|---|---|---|
| (no models admitted in v1) | *Planned* | — | the corpus has no real model identity |

The v1 site does not admit a real model identity. A sanitized cimage in the
corpus identifies its test fixture as the source; the source digest is
`test-digest`, a placeholder. A real model identity is a milestone in the
roadmap.

### Routes

| Route | State | Source | Evidence |
|---|---|---|---|
| (no routes admitted in v1) | *Planned* | — | the corpus has no real route |

A route is admitted by an execution receipt on a target. The v1 corpus has no
execution receipt. A real route is a milestone in the roadmap.

### Honest limits

The list above is the list. A row is not on the list because the
implementation does not exist for it. A row is on the list as *Implemented*
because a reviewer can find the entry point in the source. A row is on the
list as *Planned* because the architecture or contract exists and the
end-to-end path does not.

No row is *Qualifying* in v1. A qualifying record requires a test or fixture
path and a named limit. The qualification harness has not yet emitted a
publishable record. A follow-on ADR (ADR-034) creates the type; the engine's
test suite produces records that the SSG can project.

No row is *Validated* in v1. A validation record requires a commit, a target,
a build, an evidence class, and a receipt. The receipt types exist; the
emission pipeline does not. A follow-on sequence of ADRs (034–038 of the
schema-binding series) closes the gap.

The transition history for every capability is preserved in
`capability-history.json`. The v1 history is short: each row above has one
genesis record. As the engine emits receipts, the history grows.

The status page does not claim that Prism "works on Apple Silicon." It says
that the source path exists, that a sanitized cimage is published, and that no
real execution receipt has been emitted. The visitor who needs the latter
knows where to look: the receipts are not in the corpus; the gaps are named.

---

## Page 8 — The Life of a ComputeImage (`/observatory/life/`)

**Brief:** §6.8.

### The thesis

The architecture is one experience, not a sequence of pages. The visitor lands
on a specimen, held in a stage that reveals the underlying data as the
visitor interacts.

### The stage

Twelve instruments, one per stage of the canonical journey. Each instrument
is a window into the data. Selecting an instrument deepens the view. The
selection persists in the URL. The selection is visible in every other
instrument that references the same identity.

The full sequence is present in HTML without JavaScript. The default density
is rendered. The selection, the depth, and the cross-instrument highlighting
are layered on top by the `SelectionController`. The JavaScript-free
fallback is the same authored work without synchronization.

### The twelve instruments

For each stage, the instrument exposes:

- **Overview density** — the major transformation. One paragraph. The
  artifact dominates the visual.
- **Technical density** — the contracts, the constraints, the layouts, the
  search results, the rejected alternatives.
- **Raw-artifact density** — the manifest, the receipt, the digest, the
  hardware identity, the schema version, the provenance.

The visitor can step through the stages in order. The visitor can select a
non-adjacent stage. The visitor can return to the previous selection by URL.

The twelve stages, with their default one-sentence description at overview
density:

1. **Source package ingest.** A model package arrives with its metadata.
2. **Graph recovery and declaration.** Prism recovers the canonical graph
   under a declared schema, operator semantics, and normalization policy.
3. **Region decomposition.** The graph is decomposed into logical tensors,
   semantic regions, and structural priors.
4. **Representation search.** Prism evaluates candidate representations.
5. **Target constraints.** Prism applies legality for the named hardware.
6. **Admitted compilation plan.** A plan is committed as durable state.
7. **ComputeImage realization.** The plan is materialized into a `.cimage`.
8. **Constitutional transaction.** A request enters through a typed command.
9. **Residency and placement.** The runtime assigns the request to lanes.
10. **Backend execution.** Provider kernels run.
11. **Output and receipt.** A typed receipt is emitted.
12. **Replay, comparison, and promotion.** The receipt and the events around
    it are replayable.

### What the v1 Observatory can show

The v1 Observatory exposes the published specimen. Selecting stage 1 opens
the source orientation. Selecting stage 7 opens the six strata. Selecting
stage 8 opens a one-line statement: *"No constitutional transaction has been
emitted to publishable storage in the v1 corpus."* Stages 9–12 are similarly
gap-stated. Stages 4, 5, 6 expose the search trace, the legalization report
field (absent), and the plan field (absent) of the published specimen.

The Observatory is honest about what it can show. The instrument panel is
fully present; the data behind it is whatever the corpus has.

### Selection

The selection state is addressable. A URL of the form
`/observatory/life/?stage=7&density=technical&tensor=weight` restores on
reload. The `SelectionController` parses, validates, normalizes, writes, and
broadcasts. The instruments subscribe.

The selection is the URL. The animation is a visual consequence. The visitor
can link to a particular stage, tensor, capability, or receipt. A reviewer
who wants to point at a particular view points at a URL.

---

## Page 9 — Run (`/run/`)

**Brief:** §6.9.

### The Developer Path

The system that exists today can be built and run on Apple Silicon from the
repository. The path is the source path: checkout, build, run, call.

```text
# 1. Clone the repository
git clone https://github.com/Tribunus-dev/prism-engine.git
cd prism-engine

# 2. Install the pinned toolchain (rust-toolchain.toml)
rustup show

# 3. Build the engine
cargo build --release

# 4. Pull a model identity and compile to a ComputeImage
cargo run --release -p prism --features full-apple -- pull --model <id>

# 5. Run the OpenAI-compatible local server
cargo run --release -p prism --features full-apple -- run

# 6. Call the server
curl localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"<id>","messages":[{"role":"user","content":"hello"}]}'
```

The commands above are the canonical developer path. The build features,
target triple, and runtime flags are named in the engine's own manifests; the
Run page does not duplicate them. The Run page names what to type, and where
to read the details.

The v1 corpus has no recorded terminal session for this path. A recorded
session is bound to a specific commit, a specific machine, a specific model
identity, and a specific feature set. The follow-on ADR (closing ADR-034) will
emit one. Until then, the Run page renders the commands without fabricated
output.

### The Packaged Path

A packaged release is a versioned, signed, distributable artifact with a
declared support boundary. Prism does not currently publish a packaged
release. The Run page names its absence rather than showing screenshots of a
product that does not exist.

When a release record exists in `releases.json`, the Packaged Path section
appears with: the version, the signature verification path, the support
boundary, the qualified targets, and the installation instructions.

### Troubleshooting

Known failure modes are named in the engine's documentation. The Receipt
component of the evidence page names the failure classes. The Run page links
to both. The visitor who hits a failure finds the class in the receipt, the
cause in the engine's documentation, and the recovery path in the
architecture.

A reviewer who hits a failure that is not named: file a report. The
Diagnostics component captures the route, the selection, the build identity,
and the failure. The report is not a bug tracker entry; it is a piece of
evidence for the next freeze.

---

## Page 10 — Roadmap (`/roadmap/`)

**Brief:** §6.10.

### The thesis

The roadmap contains only future milestones. Each milestone has an explicit
exit criterion. A milestone is not complete until the criterion is satisfied
and the corresponding capability record's state has advanced.

When a milestone's exit criterion is satisfied, the milestone leaves the
Roadmap. Its capability record remains on the Status page. The Status page
is the source of truth for what is. The Roadmap is the source of truth for
what would close the gap between is and could be.

### Milestones

| Milestone | Work | Exit criterion | Current state |
|---|---|---|---|
| **Emit `ExecutionReceipt` to publishable storage** | implement the emission path from runtime to content-addressed log | one `ExecutionReceipt` in the v1 corpus, with identity, fencing, deadline, artifact digest, route, outcome | *Planned*; the type is bound; ADR-035 closes the gap |
| **Emit `ReplayResult` to publishable storage** | implement the replay applier's output path | one `ReplayResult` in the corpus, with event log digest, replayed world digest, equality | *Planned*; the type is bound; ADR-036 closes the gap |
| **Qualify Apple Silicon (M-series) CPU execution** | run the qualification harness end-to-end, publish the record | one `QualificationRecord` in the corpus, with commit, target, build features, validation scope, *Validated* state on the corresponding capability | *Planned*; the type is bound; ADR-034 closes the gap |
| **Qualify ANE provider** | implement ANE dispatch, qualify, validate | one `QualificationRecord` for ANE, with the ANE provider admitted | *Planned*; depends on ADR-031 |
| **Qualify ROCm provider** | implement ROCm dispatch, qualify, validate | one `QualificationRecord` for ROCm, with the ROCm provider admitted | *Planned*; depends on ADR-031 |
| **Admit a real model identity** | ingest a real model, produce a real cimage, publish it | one cimage in the corpus whose `source_digest` is not a placeholder | *Planned* |
| **Admit a real route** | run a real request through the constitutional transaction, publish the receipt | one receipt in the corpus for a real route on a real target | *Planned* |
| **Make Prism ML page conditional route resolvable** | publish the relationship, artifact handoff, license boundary, evidence | the conditional route resolves to the Prism ML page rather than `/lab/` | *Planned*; H12 enforces the gate |
| **Make General Compute page conditional route resolvable** | publish the relationship, the recorded plan and run, the evidence | the conditional route resolves to the General Compute page rather than `/lab/` | *Planned*; H12 enforces the gate |
| **Ship Prism Observatory v1.0** | every gate in `OBSERVATORY_V1_SPEC.md` §12 is green | release log entry `prism-observatory-v1.0` signed by the architect | *Planned* |

### What the Roadmap is not

The Roadmap is not a status page. A capability that is on the Status page
is not on the Roadmap unless its exit criterion is unmet. A milestone whose
exit criterion is satisfied leaves the Roadmap. The Status page is updated
to reflect the new state; the Roadmap does not.

The Roadmap is not a marketing surface. It does not contain "vision" or
"direction" prose. It contains milestones, work, and exit criteria. The
work is the path; the exit criterion is the test; the current state is
where the milestone is.

The Roadmap is not a commitment to dates. A milestone does not carry an
estimated completion. The exit criterion is the commitment. The current
state is the truth.

---

## Page 11 — PrismAgent (`/prismagent/`)

**Brief:** §6.11.

### The thesis

PrismAgent is a private, local, conversational screen companion that
combines structured accessibility information with visual interpretation, and
can expose the evidence behind its own local execution.

### The Surface

PrismAgent runs locally. The user's data does not leave the device. The
agent observes the screen through the accessibility tree and authorized
visual context. The screen is the surface. The engine is the authority.

A user asks a question about what is happening on screen. The agent reads
the accessibility tree, looks at the authorized visual context, and answers.
The answer is local. The receipt is local. The evidence is local.

### The Role

PrismAgent is one product. Its role is to expose Prism's evidence to a user
who is looking at a screen, not at a CLI or a configuration file. The
diagnostics surface is a capability. The Compiler Lab access is a capability.
The multimodal assistance is a capability. The product's reason for existing
is the conversational screen companion.

### The Boundary

PrismAgent observes. PrismAgent does not make compilation decisions.
PrismAgent does not rewrite receipts. PrismAgent does not represent itself
as the engine. PrismAgent is an instrument that reads the engine and answers
the user.

When the agent exposes a Prism receipt behind an answer, the receipt is the
real receipt from the engine. The agent does not synthesize a receipt. The
agent surfaces what the engine produced.

### The Status

PrismAgent is *Unreleased* in v1. The page surfaces this state explicitly:
the product is being developed, the build is available from the source path,
no packaged release is published. When a release record appears in
`releases.json`, the page updates: the version, the signature, the support
boundary, the installation path.

The page is not a marketing surface. The page names the product, its
boundary, and its current state. A visitor who comes to the page looking for
"is it ready to ship" finds the answer: the distribution state is
*Unreleased*; the source path is reviewable; the release is a milestone in
the roadmap.

---

## Page 12 — Prism ML (`/prism-ml/`) — not in v1 manuscript

**Brief:** §6.12. Conditional route.

The v1 manuscript does not include the Prism ML page. The route resolves to
`/lab/` per §6.12. The conditional-route gate (H12) is not satisfied: the
relationship, the artifact handoff, the license boundary, and the evidence
are not yet publishable.

The redirect from `/prism-ml/` to `/lab/` is emitted by the SSG and served
by the Cloudflare Pages `_redirects` file as a real HTTP 301. A visitor who
arrives at the route lands on `/lab/`, where any Lab Note that touches Prism
ML work can be found.

When the gate is satisfied, the page manuscript is added in a new revision of
this document. The new revision is signed by the architect before the
redirect is removed.

---

## Page 13 — General Compute (`/general-compute/`) — not in v1 manuscript

**Brief:** §6.13. Conditional route.

The v1 manuscript does not include the General Compute page. The route
resolves to `/lab/` per §6.13. The conditional-route gate (H12) is not
satisfied: the relationship, the recorded plan, and the recorded run are not
yet publishable, and the relationship cannot be described without implying
endorsement or partnership.

The redirect from `/general-compute/` to `/lab/` is emitted by the SSG and
served by the Cloudflare Pages `_redirects` file as a real HTTP 301. The
visitor lands on `/lab/`.

When the gate is satisfied, the page manuscript is added. The new revision
is signed by the architect before the redirect is removed.

---

## Page 14 — Lab Notes (`/lab/`)

**Brief:** §6.14.

### The thesis

Some directions are research. They do not ship until they ship. A Lab Note
records a hypothesis, an observation, and a next experiment. A Lab Note is
not a public capability claim. A note that becomes a claim moves to the
Architecture or Status page; it leaves Lab Notes.

### What Lab Notes carry

A Lab Note carries:

- a **hypothesis** — the claim being explored;
- an **observation** — what has been seen so far, with whatever evidence
  supports it (a source path, an experiment record, a runtime trace, a
  citation, or an explicit statement that the observation is a hypothesis);
- a **next experiment** — what would change the observation, in either
  direction.

A Lab Note does not require a public capability claim's evidence. What it
lacks, compared to a Status page row, is the qualifying evidence that would
admit the work as a public capability. The observation is honest about its
own limits.

### Current notes (illustrative; the published Lab Notes are read from
`/lab/index.json` at build time)

- **Engram-like sidecars.** Hypothesis: small learned modules attached to
  the canonical state carry conditional memory that survives plan boundaries.
  Observation: the engine's `engram_learning` module implements the
  scaffolding; the sidecar's behavior under repeated admissions is not yet
  measured. Next experiment: a controlled fixture that exercises a sidecar
  across three plan identities, with the receipts compared.
- **KV-cache compaction policies.** Hypothesis: a learned eviction policy
  can match a hand-tuned one within a small loss bound, on representative
  workloads. Observation: the engine's `kv_cache_compaction` module
  implements the policy; the comparison run is not yet recorded. Next
  experiment: a side-by-side comparison on a fixed workload, with the
  receipts published.
- **Shadow calibration.** Hypothesis: shadow calibration on a held-out
  calibration set produces a quantization plan that preserves the chosen
  metric within a stated bound. Observation: the engine's
  `shadow_calibration` module produces a plan; the bound is not yet
  stated. Next experiment: a calibration run that produces a plan with a
  named bound, with the plan and the bound published.
- **Learned adapter training.** Hypothesis: a small adapter training loop
  on a fixed task produces a delta that admits to the same plan identity.
  Observation: the engine's `adapter_training` module implements the loop;
  the delta's plan-identity implication is not yet tested. Next experiment:
  a controlled adapter run, with the delta's plan identity compared.

Each note above is illustrative. The published notes are read from the data
layer; the manuscript names the kind of note the layer holds.

---

## Page 15 — Colophon (`/colophon/`)

**Brief:** §6.15.

### The thesis

Prism Engine is independently developed by Julian Torres. The system is open
source. Collaboration takes five shapes. The evidence class is recorded
either way.

### The author

Prism Engine is built by Julian Torres. The architecture is a sustained
argument that the inference stack's abstractions are inadequate, and that
the right response is a different one. The argument is not a vision
document; it is a compiler, a runtime, an artifact format, and a
constitutional model for canonical state.

### The collaboration model

The collaboration surface is named. The contracts are open. The
collaboration takes five shapes:

- **Hardware validation.** Bring a target. Run the qualification harness.
  Publish the record. The target's evidence class moves with the run.
- **Datacenter deployment.** Bring a workload and a machine. Compile it.
  Run it. Compare the plans. Publish the receipts.
- **Engineering.** Bring a system worth understanding. Build the bridge
  between Prism and the system. The bridge is open source.
- **Research.** Bring a hypothesis the engine can test. The test is a
  receipt. The result is a milestone, a Lab Note, or both.
- **Edge and robotics.** Bring a constraint that the cloud assumes away.
  The plan is the constraint. The artifact is the handoff.

### The license and contribution path

The system is open source under the project's license. The contribution
path is the repository. The collaboration surface is the issue tracker and
the discussion channels. The proprietary models remain confidential where an
engagement requires it. The evidence class is recorded either way.

### The site

This site is *Prism Observatory v1: the evidence-bound public projection of
Prism Engine.* It mirrors Prism's subject, projection, and receipt
concepts. It is not a transaction system. A browser selection is not a
constitutional event. The vocabulary is preserved because it is meaningful,
not because every action it describes happens in the browser.

The site is the first public client of the runtime used to observe the
runtime. The author wrote the engine; the engine produced the projection.
The projection is the site.

---

## Reviewer checklist (H1–H13 self-check)

The manuscript is reviewed against the Editorial Constitution. The author
self-check below is the starting point for the named human reviewers.

**H1. Status-vocabulary purity (semantic).** Every status word in the
manuscript is one of the §3 states. Forbidden status words (`supported`,
`ready`, `active`, `available`, `live` as a status word) are absent. The
manuscript's voice uses *Implemented*, *Planned*, *Qualifying*, *Validated*,
*Unreleased*, *Released* in the §3 sense, and uses them only with a source
path and a limit. **A reviewer should read each occurrence in context and
confirm.**

**H2. Repetition audit.** No paragraph says the same thing as a paragraph on
a linked page. Cross-references are explicit (e.g., "the Specimen page
(`/computeimage/specimen/`)"). The Start page restates the home page's
motivation in one paragraph by §6.2's design, and is the single allowed
restatement.

**H3. Paragraph function.** Every paragraph is justifiable in one sentence.
The reviewer should remove any paragraph that cannot be justified. The
manuscript is tight; the reviewer should reject any further compression
that loses information.

**H4. Page purpose match.** Every page makes the argument its brief in §6
mandates. A reviewer who finds a page that drifts (even by paragraph-level
dilution) flags it for rewrite.

**H5. Diagram usefulness.** Diagrams are not present in the manuscript; the
manuscript is the prose layer. Diagrams are added in the component layer,
and each carries a caption and a textual equivalent (per A8 and H5). The
manuscript does not need to produce them.

**H6. Voice consistency.** The voice is consistent across pages. The reviewer
should read the manuscript aloud. Passages that break the voice are
rewritten.

**H7. Evidence binding quality.** The Specimen page's sanitization
declaration and the Status page's small honest list are the right surface
for the v1 corpus. The reviewer (architect) confirms the sanitization is
honest, the gap statements are complete, and the Status page is the right
shape.

**H8. Component responsibility review.** The manuscript does not produce
components; it produces prose. The component layer is reviewed separately.

**H9. Visual hierarchy.** The manuscript does not produce visual
hierarchy; the visual layer is reviewed separately. The reviewer should
read the prose for visual hierarchy cues: which sections are emphasized,
which transitions are abrupt, which densities are missing.

**H10. Interaction design.** The manuscript's selection state is described
in the Observatory page (§8). The URL is the source of truth; the
interaction is a visual consequence. The reviewer confirms this language is
consistent across the manuscript.

**H11. Truth architecture drift.** The manuscript references
`capabilities.json`, `evidence-index.json`, `releases.json`,
`capability-history.json`, and the §4.1 data layer. The reviewer confirms
the references are consistent with §4.1 and that no field is asserted in
prose that is not in the data layer.

**H12. Honest limits.** The Status page names the limits. The Evidence
page names the gaps. The Run page names the absence of a packaged release.
The PrismAgent page names its *Unreleased* state. The conditional routes
(Prism ML, General Compute) are not in the manuscript; they resolve to
`/lab/`. The reviewer confirms the limits are honest and complete.

**H13. Release readiness.** The reviewer (architect) confirms the manuscript
is, in their judgement, ready to populate the component layer. The first
release is named *Prism Observatory v1* only after this gate is signed.

---

*End of manuscript. The next deliverable is the component layer populated
from this prose, or — if the architect identifies manuscript-level issues —
a revised manuscript.*
