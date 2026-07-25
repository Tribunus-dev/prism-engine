# Prism Observatory v1 — Master Specification

**Status:** APPROVED v1.0 — GOVERNANCE FROZEN.
**Authority:** Canonical for the v1 public projection of Prism Engine. Code, content, and visual work conform to this document. Departures require an ADR.
**Owner:** Julian Torres.
**Successor to:** the current dual-generation deployment (new homepage at `/`, Observatory-style interior at `/index.html` and directory routes, broken legacy `.html` links).

---

## Preamble

This document is the **governance** of the authored work. It is not the authored work. The complete page manuscript is a separate, binding artifact produced in Phase 3. Downstream agents and reviewers may not treat the editorial briefs in §6 as a license to write the final prose themselves. The briefs state the job of each page; the manuscript states what each page says.

This document also distinguishes the parts it decides from the parts that remain open. §14 lists every item the document does not yet decide. Each open item blocks the phase that depends on it. §15 names the deployment platform on which the redirects, headers, previews, and rollback the document requires actually work.

This document is frozen at v1.0. Future changes require an ADR that names the affected section, the reason for the change, and the resulting state.

---

## 0. Reading guide

Section 1 is the law. Section 2 is the spine. Section 3 is the vocabulary. Section 4 is the data layer, including history and sanitization. Section 5 is the experience. Section 6 is the **editorial briefs** the manuscript will follow. Sections 7 through 9 are the route, component, and visual structure. Section 10 is the evidence corpus. Sections 11 and 12 are the production pipeline and the release gates. Section 13 is who built it. Section 14 is what is still open. Section 15 is the deployment platform contract.

If a downstream agent cannot answer a question from this document, the question is open in §14 and the work is not ready to leave Phase 1. If a downstream agent can answer a question from this document but disagrees with the answer, the agent escalates rather than improvises.

The document is intentionally compact. Length is not authority. The constitution is enforced by reviewers reading the file, not by the file being long.

---

## 1. The Editorial Constitution

Twelve rules. None aspirational. Each is enforceable by a reviewer reading the file.

**1. One argument per page.** A page exists to make a single primary claim. Secondary observations support it or they are deleted. The home page is not a compressed version of every other page; it is the entrance to the experience.

**2. Every status word has a definition.** The four capability states (*Planned, Implemented, Qualifying, Validated*) and the two distribution states (*Unreleased, Released*) are the only states allowed on the surface. A reviewer who finds "supported," "ready," "active," "live," or "available" used as a status word marks the line for removal. The single permitted use of *live* outside the Editorial Constitution is a real-time data source that is in fact live at the moment of render; such a source is named, instrumented, and falls under the telemetry decision in §14.

**3. Every technical claim has an evidence boundary.** A sentence that names a hardware path, a backend, a model, or a measurement must point to the schema, commit, target, and evidence class that justifies it. If the provenance is missing, the sentence becomes a question and stays in §14.

**4. Every diagram has a caption and a textual equivalent.** A diagram that cannot be replaced by three sentences of prose is performing for itself. A diagram that is only reachable through the diagram has not been authored.

**5. Every introduced term is used again or it is deleted.** A glossary entry that exists only on the page that defines it is a draft, not a term. "Observe," "intent," "representation," "canonical subject," and "projection" are vocabulary under audit; they survive only where they continue to mean something specific.

**6. No section repeats an explanation because it is on a different route.** Cross-reference the source page. Do not paraphrase. The reader trusts you to have written it once.

**7. No page begins by explaining the website's own architecture.** The site does not introduce itself by describing its own projection model. That information is for the Colophon and the lab, not the front door.

**8. No future capability is presented with the same visual treatment as a measured capability.** Color, motion, density, shape, and text express state. A *Planned* element is a spectral outline. A *Validated* element is materially present. Same shape, different truth.

**9. No benchmark appears without target, build, model, method, and provenance.** A number without provenance is marketing. The site is not marketing.

**10. No call to action appears unless there is a real destination.** A button that does nothing is a promise broken at the moment of click. The site makes very few promises. Each one resolves.

**11. No paragraph exists solely to sound important.** A reviewer who cannot state the function of a paragraph in one sentence removes it. The internet endures.

**12. The site is semantically complete without JavaScript.** CSS and HTML carry the entire authored work on every canonical route, including the Observatory. JavaScript coordinates cross-view selection, restores navigation state, deepens density, and supports the keyboard and accessibility tree. It does not create the meaning of a page after load, and it does not replace an authored surface with a less complete one.

---

## 2. The Canonical Journey — "The Life of a ComputeImage"

The site has one subject and one story. The subject is a model becoming a ComputeImage, being admitted into constitutional execution, running across explicit hardware paths, and producing evidence. The story is the twelve stages below. Every page is one lens over this sequence. Every page must be able to point to the stages it observes.

### 2.1 The three identities

The journey depends on three distinct identity types. Conflating them is a category error that propagates through the data layer.

- **`source_artifact_digest`** is the content hash of the admitted source package as it arrived. It names the bytes. Two packages with the same bytes share a digest; two packages that differ in any byte do not.
- **`semantic_graph_identity`** is the content hash of the canonical computational graph recovered under a declared schema, declared operator semantics, and a declared normalization policy. Two inputs that normalize to the same canonical graph under the same declarations share this identity. It is deterministic, but it is also conditional: the same input under different declarations can yield different identities. The declaration is part of the identity, not metadata about it.
- **`computeimage_artifact_digest`** is the content hash of the produced `.cimage` bytes. It names the artifact. Two artifacts produced from the same source and the same plan, with the same payload bytes, share this digest. Changing a representation, a layout, a payload, or a plan field changes it.

The phrase "same effective computation" does not appear in the manuscript. The site uses the three identities above and names the declaration that produced each.

When a specimen is sanitized before publication, the relationship between the public artifact and the original is governed by §4.8.

### 2.2 The twelve stages

1. **Source package ingest.** A model package or other admissible source artifact arrives with whatever metadata it carries: weights, graph definition, tokenizer, configuration, license, provenance. The site does not yet claim a computation; it claims a package and its digest. Graph recovery may be partial or require additional declarations.

2. **Graph recovery and declaration.** Prism reads the package, recovers the computational graph where possible, and binds the recovery to a declared schema, operator semantics, and normalization policy. Where the package is insufficient, the site names the missing declarations rather than guessing. The semantic graph identity is assigned only when the declaration is sufficient.

3. **Region decomposition.** The graph is decomposed into logical tensors, semantic regions, control flow, and structural priors. The visitor sees what was found, what was inferred, and what the inference cost.

4. **Representation search.** Prism evaluates candidate representations: precision, sparsity, ternarization, layout. **Each candidate transformation is a typed operation. Search policy may be heuristic, evolutionary, learned, sampled, or cost-model driven. The admitted candidate, its constraints, and its evidence are explicit; the search process is not required to be.** The visitor sees the search space, the chosen path, and the rejected alternatives.

5. **Target constraints.** Prism applies legality for the named hardware: legal tile sizes, supported operators, memory tiers, KV policy, residency rules. The site names the target by identity, not by marketing.

6. **Admitted compilation plan.** A plan is committed to the world as durable state. The plan binds logical identity, representation, target legality, and numerical policy to a specific realization. The plan has its own identity; re-planning produces a new plan identity and a new admission record.

7. **ComputeImage realization.** The plan is materialized into a `.cimage` artifact. The artifact is inspectable. The visitor can open it and see the six strata the spec defines elsewhere. The artifact's digest is recorded.

8. **Constitutional transaction.** A request enters the world through a typed command, is admitted through a transaction, and is recorded as a durable event. The transaction is the only path to canonical state change.

9. **Residency and placement.** The runtime assigns the request to lanes, providers, and memory tiers according to the plan. Provider-specific capability remains visible. The plan is the authority for representation and execution legality; canonical world state remains the authority for admission, ownership, deadlines, leases, and transaction validity.

10. **Backend execution.** Provider kernels run. Hardware handles live only in this stage. The site does not show handles, queues, or device pointers. It shows the request, the plan identity, the lane, the provider, the model.

11. **Output and receipt.** A typed receipt is emitted. The receipt carries identity, fencing generation, deadline, artifact digest, numerical policy, route, and outcome. The receipt is durable. The output is not authoritative without it. **A failure is emitted through the same receipt path, with the same provenance, the same visual treatment, and a defined failure class.** Failure is not a regrettable variant of success; it is a first-class outcome of the system.

12. **Replay, comparison, and promotion.** The receipt and the events around it are replayable. State is rebuilt from the durable log without re-running effects. Successful and failed runs are comparable. Promotion is a typed operation with its own evidence class and its own admission record.

### 2.3 The plan/runtime boundary, precisely

The runtime cannot alter **semantic, representation, legality, or numerical-policy** decisions contained in an admitted plan. It may instantiate only the runtime choices **explicitly permitted by that plan** (lane selection within a permitted set, scheduling within declared residency policy, queue ordering, retry within declared bounds). Any change outside those bounds is a different plan: it produces a new plan identity, a new admission record, and a new artifact digest if materialization follows.

Canonical world state remains the sole authority for admission, ownership, leases, deadlines, fencing generations, transaction validity, and terminal failure recording. The runtime consults both the plan and the world. The plan does not replace the world; the world does not replace the plan.

The phrase "the plan is the only thing the runtime is allowed to consult" is incorrect and does not appear in the manuscript. The correct phrase is: **the plan is the sole authority for representation and execution legality; the world is the authority for everything else.**

---

## 3. The Public Status Vocabulary

Six states, organized in two dimensions. Each is a typed entity the data layer carries. The two dimensions do not collapse.

### 3.1 Capability maturity

Four states. A capability has exactly one capability state. The states form a partial order, not necessarily a total one: a capability may regress in maturity when its evidence is invalidated, and the data layer records the transition rather than overwriting history. The transition history lives in `capability-history.json` (see §4.7).

- **Planned.** The architecture, surface, or contract is described in an ADR or design note. No end-to-end code path exists. The end-to-end path cannot be executed by any combination of supported inputs.
  *Evidence requirement:* ADR or design note; absence of the code path.
  *Visual treatment:* spectral outline; no measurement, no badge color, no claim of motion.

- **Implemented.** The code path or provider boundary exists. A reviewer can find the entry point, read it, and reason about its control flow. The path may not be exercised end-to-end against a target.
  *Evidence requirement:* source path; build features; commit.
  *Visual treatment:* material outline; no claim of measurement.

- **Qualifying.** Tests, fixtures, or compile-time validation exist. Target-specific or end-to-end evidence is incomplete. The path may produce correct output for some inputs and unknown output for others.
  *Evidence requirement:* test or fixture path; defined validation scope; named limit of evidence.
  *Visual treatment:* material outline with a qualifier badge; measurement not permitted.

- **Validated.** A named build, against a named target, with a named evidence class, passed explicit conformance and receipt gates. The build is reproducible from a stated commit and feature set.
  *Evidence requirement:* commit; target; build features; evidence class; receipt or qualification record.
  *Visual treatment:* material outline; measurement permitted with provenance attached.

### 3.2 Distribution

Two states, orthogonal to maturity. A capability in any maturity state may be Unreleased or Released. Maturity and distribution compose, but neither overrides the other.

- **Unreleased.** The capability is not distributed as a versioned artifact. The source path may be available; the binary is not.
- **Released.** A versioned, signed artifact is published. The artifact has declared support boundaries, an installation path, and a known set of qualified targets.
  *Evidence requirement:* release manifest; signature; support boundary document.
  *Visual treatment:* material outline with a release tag. Only this state implies availability in the public sense.

### 3.3 Allowed maturity × distribution pairs and the prose each permits

Not every combination is meaningful. The validator enforces the allowed pairs. Distribution does not determine whether a technical observation is true; it determines whether the observation is exposed to a user as available. Maturity gates technical verbs; distribution gates availability verbs.

| Maturity | Unreleased | Released |
|---|---|---|
| **Planned** | valid. Prose may say: *planned*, *designed*, *not yet implemented*. | **invalid** |
| **Implemented** | valid. Prose may say: *implemented in the source*, *the code path exists*. | valid. Prose may say: *included in the named release*, *distributed as part of version X* (no validation claim). |
| **Qualifying** | valid. Prose may say: *qualifying*, *passing fixture tests*, *evidence incomplete*. | valid. Prose may say: *qualifying release*, *shipped with the explicit note that evidence is incomplete* (no validation claim). |
| **Validated** | valid. Prose may say: *validated on the named target*, *passing receipt-class E on commit C* (a true technical statement; not gated by distribution). | valid. Prose may say: *validated on the named target* and *released for the named target* or *included in the named release*. |

**Released + Planned is invalid.** A release may contain documentation about future work, but the future capability itself is not distributed. A planner that suggests the pair is wrong; an editor who records it has misread the data.

**Maturity governs what is true; distribution governs what is available.** A `Validated` capability, `Unreleased`, is correctly described as *validated on the named target*; that statement does not require a release. The release tag is a separate fact that, when present, permits an additional availability verb.

### 3.4 Composed capabilities

A capability that composes other capabilities has its own capability record. Its maturity is not inferred from the lowest or highest component. A multi-provider route involving three *Validated* components is not itself *Validated* until the composed route has been validated end-to-end and the result is recorded as evidence of the composition. **Prism, of all systems, does not discover transitive confidence by wishful thinking.** Composed records reference their component records by stable ID; the validator checks that every reference resolves.

### 3.5 Forbidden status language

"Live," "supported," "ready," "active," and "available" are not states. They are adjectives that smuggle claims. A reviewer who finds one used as a status word marks the line for removal. The single permitted use of "live" on the surface is a real-time data source that is in fact live at the moment of render; such a source is named, instrumented, and falls under the telemetry decision in §14.

---

## 4. Truth Architecture

The site cannot continue carrying facts inside hand-written page copy. The prose remains authored. The facts are projected from a validated data layer.

### 4.1 The canonical inputs

The data layer is the union of every file the SSG and runtime read. The set is closed: every input is named here, every output is sourced from this set, and the validator rejects any file the rest of the spec names that is not in this table.

| File | Kind | Authority | Generator | Publication form | Consumers |
|---|---|---|---|---|---|
| `site.json` | data | Project identity, canonical origin, brand tokens, author identity | Editorial | Checked-in | Shell, all surfaces |
| `navigation.json` | data | Primary nav, secondary nav, footer, breadcrumbs | Editorial | Checked-in | Shell |
| `capabilities.json` | data | Every status-bearing surface: maturity, distribution, evidence class, limit | Editorial + generated fields | Checked-in; specific fields are derived from evidence | Status, home, evidence, run |
| `architecture.json` | data | Architecture facts: contracts, identities, plan states, schema references | Editorial + derived | Checked-in; derived fields regenerate from evidence | Architecture, computeimage, runtime |
| `evidence-index.json` | data | Every published evidence artifact: schema, producer, commit, target, hardware, validation scope, digest, redactions | Editorial + CI | Checked-in; produced during a release run | Evidence, computeimage, status, lab |
| `models.json` | data | Every model identity admitted by the project: source, license, ingest path, evidence class | Editorial + CI | Checked-in | Models surface, evidence, run |
| `roadmap.json` | data | Only future milestones and their explicit exit criteria | Editorial | Checked-in | Roadmap |
| `releases.json` | data | Every published release: version, signature, support boundary, qualified targets | Editorial + CI | Checked-in | Run, status, home, prismagent |
| `observatory.json` | data | The Life of a ComputeImage specimens: stage assignments, identity references, and links to the artifact inventory | Editorial | Checked-in | The Life experience |
| `capability-history.json` | history | Append-only transition log for `capabilities.json` | Editorial + CI | Checked-in; transitions are append-only | Status, capability pages |
| `docs-publication.json` | allowlist | Names the files emitted under `/docs/` | Editorial | Checked-in | SSG (file emission) |
| `search-index.json` | generated | Static search index over the data layer | Generated at build time | Checked-in (or rebuilt by the SSG) | Site search surface, 404 |

A field's provenance is recorded. If `capabilities.json.capability.x.state` is editorial, the renderer knows it can drift; if it is generated from `evidence-index.json`, the renderer knows it cannot. A field whose provenance is unknown is treated as editorial and flagged for review.

### 4.2 Stable IDs

Every record carries a stable ID. Cross-references use IDs, not file positions, display names, or source paths. The ID format is namespaced and human-readable.

```text
capability_id       capability.backend.apple.metal.decode
evidence_id         evidence.execute.m1.decode.001
release_id          release.0.4.2
model_identity_id   model.bonsai-ternary-1.5b
roadmap_entry_id    roadmap.target-qualification.ane
```

A reference to a record that does not exist is a build failure. A reference to a record whose type does not match the expected type is a build failure.

### 4.3 Schema home and generation direction

The schemas live in a named directory: `schemas/`, at the repository root. Each schema file is named after the data file it describes (`schemas/capabilities.schema.json` and so on). The schemas are the canonical form. **JSON Schema is the canonical schema language.**

Rust types are generated from the canonical JSON Schema using `typify` (or an equivalent JSON-Schema-to-Rust generator) at build time, with the generation committed as a reproducible step. The generated types are convenience, not authority: when a generated type disagrees with the schema, the schema wins and the generation is re-run. Where manual types are unavoidable — because `typify` cannot express a discriminator, a refinement, or a custom validator — the manual type is checked in alongside the generated one, and a test asserts that the manual type validates the canonical schema. **`schemars` is not used.** `schemars` derives JSON Schema from Rust types; that would reverse the authority this document has declared.

TypeScript types in the runtime are generated via `json-schema-to-typescript` (or equivalent) at build time. The same authority applies: schema wins, generation follows.

A schema change is a constitutional change. It requires an ADR, a version bump, and a migration plan. Renderers do not silently accept a new shape.

### 4.4 The validation gate

A build does not emit unless the data layer validates. The validator is a discriminated union over record types, not a universal checklist.

```text
PlannedRecord       { id, adr, target_class, declared_limit }
ImplementedRecord   { id, source_path, build_features, commit, declared_limit }
QualifyingRecord    { id, source_path, build_features, commit, test_or_fixture_path, validation_scope, named_limit }
ValidatedRecord     { id, source_path, build_features, commit, target, evidence_id, validation_scope, named_limit }
ReleaseRecord       { id, version, signature, signature_verification_path, support_boundary_document, qualified_targets }
```

A `QualifyingRecord` is not required to name a target; a `ValidatedRecord` is. A `ReleaseRecord` carries a signature, a verification path, and a support boundary document. The validator rejects records that are missing fields appropriate to their type, and rejects fields that are inappropriate to their type. **The release gate's universal source-path / commit / target / scope requirement is replaced by these type-specific requirements.**

### 4.5 Cross-reference integrity

Every reference is checked at build time. A capability referencing a non-existent evidence record is rejected. A roadmap entry referencing a non-existent capability is rejected. A release referencing a capability it claims to qualify is rejected. A model referencing an ingest path that does not exist in the repository is rejected.

### 4.6 One source of truth per fact

A status word in page prose that disagrees with the data layer is a bug. A claim on the home page that disagrees with the architecture file is a bug. The renderer is the only writer of derived prose. Editorial fields in the data layer are reviewed like code; they are not freeform prose.

### 4.7 Capability history

Capability maturity transitions are recorded, not overwritten. The history lives in `capability-history.json`, an append-only log. Each record carries a stable `transition_id`, a `capability_id`, a monotonically increasing `sequence` per capability, a `from` state (which is `null` for the genesis record), a `to` state, an `as_of_commit`, an `evidence_id` (when the transition is supported by a specific evidence record), a `recorded_at` timestamp, and a short editorial `reason`.

```json
{
  "transition_id": "transition.capability.backend.apple.metal.decode.001",
  "capability_id": "capability.backend.apple.metal.decode",
  "sequence": 1,
  "from": null,
  "to": "implemented",
  "as_of_commit": "...",
  "evidence_id": null,
  "recorded_at": "2026-01-15T10:00:00Z",
  "reason": "Source path introduced; first reviewable code path."
}
```

```json
{
  "transition_id": "transition.capability.backend.apple.metal.decode.004",
  "capability_id": "capability.backend.apple.metal.decode",
  "sequence": 4,
  "from": "qualifying",
  "to": "validated",
  "as_of_commit": "...",
  "evidence_id": "evidence.execute.m1.decode.001",
  "recorded_at": "2026-02-20T14:32:11Z",
  "reason": "Receipt-class E passed on commit C; validation scope V."
}
```

The current state in `capabilities.json` is the transition with the highest `sequence` for that `capability_id`, where `to` is its `state`. The log is ordered by `(capability_id, sequence)`. **`recorded_at` is metadata, not ordering authority** — two transitions recorded in the same second on the same capability still receive distinct `sequence` values. A regression (e.g., from `validated` to `qualifying` after invalidating evidence) is recorded as a forward transition with a clear `reason`. The log is append-only; a transition is never removed.

### 4.8 The sanitization model

When a published artifact (manifest, receipt, plan, run output) is derived from an original but modified for publication, the original and the publication are distinct artifacts with distinct identities. The data layer records both.

A sanitized publication carries four fields, in addition to whatever its base schema requires:

```text
original_artifact_digest     the digest of the unmodified source artifact
public_projection_digest     the digest of the bytes the public site serves
redaction_manifest           a structured list of fields redacted, replaced, or removed
remaining_verifiable         an enumeration of which relationships to the original are still checkable
```

`remaining_verifiable` is a closed set: `digest_of_source_package`, `semantic_graph_identity`, `execution_target_identity`, `route_identity`, `evidence_chain_integrity`. The manifest names which of these are still computable from the public bytes and which have been broken by redaction. A public projection that claims `evidence_chain_integrity` while the receipts it links are redacted is self-contradictory and the validator rejects it.

A specimen that has been sanitized displays both digests and a one-line statement of what has been redacted, in plain prose next to the data. The visitor is never shown a digest beside bytes that do not produce it.

The choice of "real or sanitized" in §5 is decided per specimen in Phase 2. The default is sanitized unless the corpus contains an unsanitized artifact the author is willing to publish.

---

## 5. The Authored Experience — The Life of a ComputeImage

The center of the site is a single addressable narrative that contains the architecture in miniature. The visitor lands on a real or sanitized specimen ComputeImage, held in a stage that reveals the underlying data as the visitor interacts with it.

### 5.1 What the experience honestly is

**The Observatory interaction model mirrors Prism's subject, projection, and receipt concepts. It does not pretend that browser state is engine authority.** The site is a static SSG-rendered experience plus a small client-side reducer — the `SelectionController` from §8 — that parses, validates, normalizes, writes, and broadcasts the addressable selection. It is not a transaction system, and a UI event is not a constitutional event. The vocabulary is preserved because it is meaningful, not because every action it describes happens in the browser.

If a future version ships a real Observatory service backed by the Prism state model — receiving typed interactions, returning authentic receipts, persisting to a durable event log — that is a separate, larger product, scoped by its own ADR. The v1 site does not make that claim.

### 5.2 The three densities

- **Overview density** shows the major transformation. One artifact moving through twelve stages. The narrative is short; the visual is dominated by the artifact.
- **Technical density** reveals contracts, constraints, layouts, target decisions, and rejected alternatives. The visitor sees the search space, not only the chosen path.
- **Raw-artifact density** reveals the manifest, the receipt, the digest, the hardware identity, the schema version, and the provenance. The visitor sees what was actually produced and can verify it against the published evidence corpus.

### 5.3 Addressable selection

The experience is addressable through the URL. A selection in one instrument persists in the URL, restores on reload, and is visible in every other instrument that has the same identity reference. The `SelectionController` (§8) is the sole owner of selection state. `Specimen`, `Journey`, and the Observatory instruments consume the controller's broadcast. `Navigation` preserves compatible URL state when linking; it does not own the selection.

A reviewer can link directly to a particular stage, tensor, capability, or receipt. The URL is the source of selection truth; the animation is a visual consequence.

### 5.4 No-JavaScript parity

`/observatory/life/` renders a static twelve-stage document at its own URL, with the full sequence present in HTML, the specimen at its default density, and the evidence corpus linked from each stage. JavaScript coordinates highlight, density switching, and selection persistence. It does not create the page. **The no-JavaScript fallback is not "go look at a simpler thing elsewhere"; it is the same authored work without synchronization.**

---

## 6. Page Editorial Briefs

Each canonical route receives the same brief: thesis, question, tension, object, evidence, transition, sections. The complete prose of each page is produced in Phase 3 from these briefs. The briefs are binding on the manuscript: the manuscript may elaborate, but it may not contradict them, add a second argument, or remove a section. A brief that cannot be written in one sentence per slot is a brief that has not yet decided what its page is for.

### 6.1 Home — `/`

**Thesis.** Inference runtimes make critical decisions late, implicitly, and ephemerally. Prism moves those decisions into an inspectable compiled artifact and executes them through an authority model that can preserve what happened.
**Question.** What is Prism, and what does it change about how machine learning is deployed?
**Tension.** Most AI infrastructure communicates by promising every possible path. Prism communicates by naming which paths exist, which qualify, and which are still plans.
**Object.** A ComputeImage specimen, opened to its six strata, with one selected tensor illuminated.
**Evidence.** A compact current-reality surface sourced from `capabilities.json` and `releases.json`. Three to four rows, each with maturity state, distribution state, target, evidence class, and a one-line limit.
**Transition.** "Open the architecture. Or follow one computation through its life."
**Sections.** Hero. The Central Object. Current Reality. The Journey at a Glance. Compiler Lab (Recorded). The Evidence Contract. Where to begin.

### 6.2 Start — `/start/`

**Thesis.** A new visitor can understand Prism in five minutes without reading the README.
**Question.** Where do I begin?
**Tension.** Most documentation makes the visitor perform the author's mental model before answering a question. Start does the opposite: it answers the question first, then offers depth.
**Object.** A single artifact, the same ComputeImage as the home page, held still while the visitor reads.
**Evidence.** None on this page beyond the artifact itself.
**Transition.** "If you want to see the architecture, follow the artifact. If you want to run something, go to Run. If you want to evaluate, go to Status."
**Sections.** The Short Version. The Reading Path. What Prism Is Not. The Shape of the Site.
**Note on the Reading Path.** The first step, *Motivation*, is not a separate route. It is the home page's argument, restated. The Start page links the *Motivation* step to `/` and names the home page's hero as the surface. The path is then *Motivation* (Home) → *Architecture* → *ComputeImage* → *Evidence*. Each step links to its route.

### 6.3 Architecture — `/architecture/`

**Thesis.** Prism separates the decisions that model formats and runtimes usually blend together, then names each contract.
**Question.** How is Prism organized, and why?
**Tension.** Most architecture pages either drown the visitor in a layer diagram or narrate without naming the contracts. This page names the contracts and then names what each one is for.
**Object.** The three primary contracts: *LogicalTensor*, *PhysicalTileLayout*, *ExecutionView*.
**Evidence.** The architecture file. Each contract opens to its definition, its source path, and its current maturity state.
**Transition.** "The contracts bind into a deployment artifact. That artifact is the ComputeImage."
**Sections.** One Subject, Three Contracts. The Compiler as Search. The Runtime as Authority. The Plan as the Boundary. What the Evidence Supports Today.

### 6.4 ComputeImage — `/computeimage/`

**Thesis.** A ComputeImage is the boundary between compilation and execution, and it is inspectable.
**Question.** What is a `.cimage`?
**Tension.** Most model formats are renamed checkpoints. The ComputeImage is not. This page makes the difference tangible by opening one specimen to its six strata.
**Object.** One named specimen at technical density.
**Evidence.** The specimen's manifest, a real receipt, and the provenance chain from the source package to the artifact.
**Transition.** "The artifact is admitted into the runtime as a constitutional transaction. The full journey is at `/observatory/life/`; the artifact's own record is at `/computeimage/specimen/`."
**Sections.** The Six Strata. The Three Identities. Legality. The Receipt. Inspecting the Artifact.

### 6.5 ComputeImage Specimen — `/computeimage/specimen/`

**Thesis.** This route is an evidence record, not a chapter. Its job is to expose the specimen's raw material without re-teaching the concept.
**Question.** What is this specific artifact?
**Object.** One specimen, named by `computeimage_artifact_digest`, with its manifest, its receipt, its plan identity, its search history, and its provenance, rendered as data. If the specimen has been sanitized, both the original digest and the public projection digest are displayed, alongside the redaction manifest.
**Tension.** The ComputeImage page and the Specimen page must not duplicate the explanatory prose. The Specimen is to the ComputeImage chapter what a database record is to a chapter that references it: same fact, different surface.
**Evidence.** The specimen itself. The Specimen page exposes it; it does not describe it again.
**Transition.** "Cross-references return to the ComputeImage page; the specimen is the leaf, the page is the explanation."
**Sections.** A one-paragraph orientation naming the digest(s), the source identity, the plan identity, the receipt, and the validation scope. Then the data: the six strata as labeled blocks, the receipt as a labeled block, the rejected search candidates as a labeled block, the provenance chain as a labeled chain.

### 6.6 Evidence — `/evidence/`

**Thesis.** An output is not authoritative because a backend returned it. Prism preserves the chain from accepted work to execution outcome.
**Question.** Why should I trust a Prism result?
**Tension.** Most infrastructure talks about evidence without showing it. This page shows it.
**Object.** One successful run, one failed run, one replay, one comparison. Each drawn from the corpus.
**Evidence.** The corpus itself, in full.
**Transition.** "The status of every target is sourced from the same evidence. The next page is the authoritative status surface."
**Sections.** The Epistemic Contract. The Receipt Chain. Replay. Failure as a First-Class Object. The Evidence Corpus.

### 6.7 Status — `/status/`

**Thesis.** The status surface is the authoritative answer to *what exists, what qualifies, and what has been validated* — on which hardware, against which commit, with which evidence.
**Question.** What can Prism do today, and what is the evidence for that claim?
**Tension.** Most status pages inflate. This page refuses to.
**Object.** A status table. Every row is a record in `capabilities.json`. Every cell is a state from §3. Every cell has a source path and an evidence class.
**Evidence.** The capabilities file, with the visitor able to open every record to its source, build, target, and the receipt or qualification that supports it.
**Transition.** "A row marked *Planned* or *Qualifying* is a maturity claim, not a validation claim. Validation is reserved for *Validated* records. Capability transitions are visible in the history. The next page is what is next, not what is now."
**Sections.** Targets. Backends. Models. Routes. Honest Limits.

### 6.8 The Life of a ComputeImage — `/observatory/life/`

**Thesis.** The architecture is one experience, not a sequence of pages.
**Question.** What does Prism do, in order, to one artifact?
**Tension.** Most architecture presentations split the journey into topics and ask the visitor to reassemble it. This page does not split it.
**Object.** The same specimen from the ComputeImage page, held in a stage that reveals the underlying data as the visitor interacts.
**Evidence.** The specimen, the receipts, the search results, the rejected alternatives.
**Transition.** The experience itself is the transition. Selection persists in the URL and is visible in every other instrument that has the same identity reference.
**Sections.** Stage (the twelve instruments). Overview density. Technical density. Raw-artifact density. Selection. **The full sequence is present in HTML without JavaScript.**

### 6.9 Run — `/run/`

**Thesis.** The Run page presents the currently validated build-and-execution path from the repository. The thesis names the path, not the platform, until the evidence corpus names the platform.
**Question.** How do I run Prism today?
**Tension.** Most "Run" pages mix the developer CLI path with a packaged product. This page separates them clearly.
**Object.** A working terminal session from the evidence corpus. Real commands, real output, with build, commit, target, and feature set named.
**Evidence.** The recorded run, with provenance.
**Transition.** "If you want to see the evidence, follow the receipt. If you want to see the status, look at the table."
**Sections.** The Developer Path. The Packaged Path (present only when a release record exists in `releases.json`; otherwise names its absence). Troubleshooting.

### 6.10 Roadmap — `/roadmap/`

**Thesis.** The roadmap contains only future milestones and the evidence that would close each one.
**Question.** What is next?
**Tension.** Most roadmaps mix current state with future work. This one does not.
**Object.** A short list of milestones, each with an explicit exit criterion that names a capability record in `capabilities.json`.
**Evidence.** The exit criteria reference the data layer. A milestone is not complete until its criterion is satisfied and the corresponding record's state has advanced.
**Transition.** "When a milestone's exit criterion is satisfied, the milestone leaves the Roadmap. Its capability record remains on Status. The Status page is the source of truth for what is; the Roadmap is the source of truth for what would close the gap between is and could be."
**Sections.** Milestones. Exit Criteria (each entry references a capability ID).

### 6.11 PrismAgent — `/prismagent/`

**Thesis.** PrismAgent is a private, local, conversational screen companion that combines structured accessibility information with visual interpretation, and can expose the evidence behind its own local execution.
**Question.** What is PrismAgent for?
**Tension.** PrismAgent is one product, not five. The thesis is one sentence. Diagnostics, Compiler Lab access, and multimodal assistance are capabilities of the product, not the product's reason for existing.
**Object.** A walkthrough of one human task: a user asks what is happening on screen, why an interface changed, or how to proceed. PrismAgent uses the accessibility tree and authorized visual context, responds locally, and exposes the Prism receipt behind the result when relevant.
**Evidence.** A recorded session from the corpus.
**Transition.** "PrismAgent is built on Prism Engine. To see what it observes, look at the Engine."
**Sections.** The Surface. The Role. The Boundary. The Status.
**The route is unconditional.** The page exists at v1. The product's distribution state is *Unreleased* until a release record appears in `releases.json`; the page surfaces that state explicitly. Conditional routing does not apply.

### 6.12 Prism ML — `/prism-ml/` (conditional)

**Thesis.** Prism ML is the representation and calibration research that produces the model identities Prism Engine accepts. **This page exists only when the relationship, artifact handoff, license boundary, and evidence are real and publishable.** Until then, the route resolves to the relevant Lab Note, and the page is not in the manuscript.
**Object.** One example model identity and the evidence chain from research to deployment, if the corpus supports it.
**Evidence.** The model identity, the ingest path, the admission record, the ComputeImage digest.
**Transition.** "The representation work lives in Prism ML. The deployment work lives in Prism Engine. The handoff is the artifact."

### 6.13 General Compute — `/general-compute/` (conditional)

**Thesis.** General Compute is one of the named acceleration providers Prism plans and routes against. **This page exists only when the relationship can be described without implying endorsement or partnership, and when the recorded plan and recorded run are publishable.** Until then, the route resolves to the relevant Lab Note.
**Object.** A case study, clearly labeled, showing one serving request routed across the capabilities the named provider exposes.
**Evidence.** A recorded plan and a recorded run, with the provider boundary visible.
**Transition.** "The provider is the authority on its hardware. The planner is the authority on deployment. The artifact is the handoff."

### 6.14 Lab Notes — `/lab/`

**Thesis.** Some directions are research. They do not ship until they ship.
**Question.** What is being explored that is not on the public surface?
**Object.** A small set of notes, each with a hypothesis, a current observation, and a next experiment.
**Evidence.** A note's evidence is what supports the observation: a source path, an experiment record, a link to a runtime trace, a citation, or an explicit statement that the observation is a hypothesis. **A Lab Note may carry observational evidence; what it lacks is the qualifying evidence required for a public capability claim.** A note that becomes a claim moves to the Architecture or Status page; it leaves Lab Notes.
**Transition.** "If a note graduates, it leaves the lab."

### 6.15 Colophon — `/colophon/`

**Thesis.** Prism Engine is independently developed by Julian Torres. The system is open source. Collaboration takes five shapes.
**Question.** Who is building this, and how does collaboration work?
**Object.** A short statement. The authorship is named. The collaboration model is described: boundary, contracts, openness, confidentiality, evidence class.
**Evidence.** The license, the contribution path, the collaboration surface.
**Transition.** None. The page is a close, not a door.

---

## 7. Route Map

The site has one canonical URL per concept. No two URLs serve different content. No URL serves the wrong generation.

### 7.1 Canonical routes

| Path | Purpose | Owner |
|---|---|---|
| `/` | Home | `page:home` |
| `/start/` | Start | `page:start` |
| `/architecture/` | Architecture | `page:architecture` |
| `/computeimage/` | ComputeImage overview | `page:computeimage` |
| `/computeimage/specimen/` | The published specimen (evidence record, not a chapter) | `page:computeimage-specimen` |
| `/evidence/` | Evidence | `page:evidence` |
| `/status/` | Status | `page:status` |
| `/run/` | Run | `page:run` |
| `/roadmap/` | Roadmap | `page:roadmap` |
| `/observatory/life/` | The Life of a ComputeImage | `page:observatory-life` |
| `/prismagent/` | PrismAgent (unconditional route; product is Unreleased until a release record exists) | `page:prismagent` |
| `/prism-ml/` | Prism ML (conditional) | `page:prism-ml` |
| `/general-compute/` | General Compute (conditional) | `page:general-compute` |
| `/lab/` | Lab Notes | `page:lab` |
| `/colophon/` | Author's Note | `page:colophon` |
| `/docs/` | Source documentation, allowlisted | `page:docs` |

The primary nav is fixed: *Start, Architecture, Evidence, Status, GitHub*. The conditional routes are reachable from inside the relevant primary page when they exist; they are not in the primary nav. The GitHub link is external.

The `/docs/` route is **not** the repository's `docs/` directory served wholesale. It is a publication allowlist (`docs-publication.json` from §4.1): only files explicitly named there are emitted under `/docs/`. ADRs older than the current migration cutoff, internal notes, and implementation debris do not reach the live site.

### 7.2 Legacy redirects

A canonical path is a route, not a redirect. The redirect table below contains only **legacy source paths** and their **canonical destinations**. Self-redirects are not emitted; canonical paths are tested as canonical-route assertions, separately. The platform actually serves these as real HTTP 301s (see §15.3).

| Legacy path | Canonical destination |
|---|---|
| `/index.html` | `/` |
| `/architecture.html` | `/architecture/` |
| `/capabilities.html` | `/status/` |
| `/capabilities/` | `/status/` |
| `/computeimage.html` | `/computeimage/` |
| `/heterogeneous.html` | `/architecture/` |
| `/heterogeneous/` | `/architecture/` |
| `/roadmap.html` | `/roadmap/` |
| `/prism-ml.html` | `/prism-ml/` (if conditional route exists) or `/lab/` (otherwise) |
| `/general-compute.html` | `/general-compute/` (if conditional route exists) or `/lab/` (otherwise) |
| `/work-with-prism.html` | `/prismagent/` |
| `/work-with-prism/` | `/prismagent/` |
| `/demo.html` | `/observatory/life/` |
| `/demo/` | `/observatory/life/` |
| `/projection-repro.html` | `/observatory/life/` |
| `/projection-repro/` | `/observatory/life/` |
| `/field-guide.html` | `/start/` |
| `/field-guide/` | `/start/` |
| `/start-here.html` | `/start/` |
| `/start-here/` | `/start/` |
| `/run.html` | `/run/` |
| `/evidence.html` | `/evidence/` |

`/index.html` resolves to `/`. It does not inspect its former body to choose between `/` and `/observatory/life/`. The Observatory's older content is folded into `/observatory/life/` by content migration in Phase 5; that is a content rule, not a redirect rule.

### 7.3 Legacy asset retirement

The wildcard 410 rules for legacy `/js/*` and `/data/*` are deferred until a defined compatibility window has elapsed. The rules are:

- During the window, the SSG emits fingerprinted canonical assets under `/assets/`.
- During the window, old asset paths either continue to serve the old files (if a downgrade contract applies) or return 410 with a `Link` header to the new path.
- The window length and the 410 cutoff are named in `site.json` and recorded in an ADR. A gate (§12) verifies that the 410 is not emitted before the window elapses.
- The SSG refuses to emit a 410 wildcard for any path that the canonical assets would otherwise claim, and refuses to emit a 410 for any path that the data layer references.

### 7.4 404 policy

A 404 is an authored response. It links to the home page, the Start page, the Status page, and the **site search surface** (served as a static asset from the data layer's search index, per §4.1 and §14.8). It does not apologize. It does not suggest the visitor check the URL. A separate link to the repository's search field may be present; the site search is the primary suggestion.

### 7.5 Route authority

The SSG is the only writer of routes. The redirect table is part of the SSG input. The validator checks that every legacy path resolves to a canonical path and that the destination exists. Canonical paths are emitted as pages; legacy paths are emitted as redirects in the `_redirects` file. A self-redirect is a build failure.

The redirect and header mechanisms are not merely SSG output. The deployment platform named in §15 serves the `_redirects` and `_headers` files at the edge, producing real HTTP responses. The SSG produces the files; the platform serves them.

---

## 8. Component Contracts

One owned component per responsibility. Each component has a single author, a single render path, a single CSS contract, and a single acceptance test.

**SiteShell.** Composes the site frame from `Navigation`, `Footer`, `ThemeProvider`, and `ReducedMotion`. Owns the page-level focus order, the route announcement, and the skip link. Does not own the content or behavior of any composed component.

**Navigation.** Owns the primary and secondary nav. Reads `navigation.json`. Announces the current route. **Preserves compatible URL state when linking; it does not own selection state.** If a link is to a route that accepts selection, the link may carry selection query parameters; the link does not validate or transform them — that is the `SelectionController`'s job.

**Footer.** Owns the footer. Reads `site.json`. Renders canonical origin, license, authorship, and repository link.

**ThemeProvider.** Owns theme application: the active palette, the reduced-motion cascade, the forced-colors response, and the system-preference query (`prefers-color-scheme`, `prefers-reduced-motion`, `forced-colors`). Reads tokens from `site.json`. **Does not define tokens**; it consumes them. The single v1 theme is dark; `ThemeProvider` is structured to accept additional themes in a follow-on without component changes elsewhere.

**SelectionController.** Non-rendering. The sole owner of the addressable selection. Parses selection from the URL, validates against the schema, normalizes, writes back to the URL, and broadcasts changes to subscribers. `Specimen`, `Journey`, the Observatory instruments, and any other component that displays selection state consume the broadcast. `Navigation` and `SiteShell` do not. The controller's data model is named in an ADR before it ships. The controller is a reducer; it does not render DOM, does not perform network requests, and does not persist selection beyond the URL.

**Hero.** The home page entrance. Owns the central object affordance, the role selector, and the home hero copy. No other page renders a Hero.

**Journey.** The compressed canonical journey, used on the home page and Start. Owns the twelve-stage strip, the link to `/observatory/life/`, and the keyboard-navigable stage list. Subscribes to `SelectionController`.

**Specimen.** A ComputeImage specimen at one of three densities. Owns the six-stratum layout, the density selector, and the link to the published receipt. Subscribes to `SelectionController`; does not own selection state.

**Chapter.** A page-local chapter. Owns a single thesis, a single body, a single set of evidence references, and a single transition to the next chapter. Never global. Never repeats across pages.

**ChapterList.** A page-local table of contents for the current page. Renders only the chapters on the current page. The 60-entry global chapter dump is deleted from the codebase and the data layer.

**Claim.** A single claim with its maturity state, distribution state, and source path. Renders the public vocabulary, not invented language. The state is drawn from the claim's record in the data layer. **Status-bearing language is emitted only through `Claim`, `StatusTable`, and `Release`; the validator inspects those structured surfaces (see A2).**

**StatusTable.** The status surface. Renders rows from `capabilities.json`. Filters by domain and by state. Opens a row to its source, build, target, and evidence record. **Status-bearing language is emitted only through this component and through `Claim` and `Release`.**

**Release.** The release surface. Renders a single release record: version, signature, support boundary, qualified targets, and the maturity state of each capability it distributes. Reads `releases.json` and resolves the maturity of each listed capability from `capabilities.json`. **Status-bearing language — including the release tag and the availability verbs of §3.3 — is emitted only through `Release`, `Claim`, and `StatusTable`.** A `Release` that lists a capability does not change that capability's maturity; the record's `maturity` is read, not asserted.

**Receipt.** A single receipt. Renders identity, fencing generation, deadline, artifact digest, numerical policy, route, outcome. Renders the failure variant with the same visual dignity as the success variant.

**EvidenceCard.** A single evidence artifact. Renders schema, producer, commit, target, hardware, validation scope, digest, redactions. Reads `evidence-index.json`.

**RoadmapEntry.** A single milestone. Renders the description, the work, the exit criterion, and the current state of the criterion (read from the referenced capability record).

**AuthorNote.** The colophon block. Renders authorship, collaboration model, license, contribution path.

**ReducedMotion.** A global component that respects `prefers-reduced-motion`. When reduced motion is on, the prism effect is removed, ambient motion is removed, and selection state is preserved through layout, not motion. Composed by `SiteShell` via `ThemeProvider`; may also be composed independently by components that own their own motion.

**Diagnostics.** A session-local, non-persistent component. Records the current route, the current selection, and the build identity, for the visitor's own use (e.g., filing a report). Never transmits. Never persists across sessions. Does not record a "receipt of the last interaction." The component's data model is named in an ADR before it ships.

A component that does not own one of these responsibilities is a draft and does not ship. A responsibility that does not have an owner above is also a draft and does not ship.

---

## 9. Visual Direction

The visual language is already strong. The work that remains is composition.

**Type.** Ubuntu is the primary typeface across the site. Monospaced typography is reserved for identities, hashes, digests, commands, schemas, values, machine-state labels, and code. Prose never uses monospace. Identities never use proportional.

**Color and tokens.** The tokens are defined in `site.json` and consumed by every component. The themes are authored, not generated. A color is added to the palette when a responsibility needs it. No component invents a color.

**Motion.** Motion expresses state, not decoration. The prism effect refracts when the subject crosses a representation or hardware boundary. A *Planned* element is spectral geometry. A *Validated* element is materially present. A *Released* element carries a release tag. Ambient motion is permitted but is subordinated to the canonical subject.

**Status is not communicated by color or motion alone.** **No status distinction depends solely on hue, opacity, blur, glow, depth, or motion.** Every state must remain visible in text, shape, and semantics. Color and motion are reinforcement. A status that is invisible to a screen reader, a high-contrast mode, a printed page, or a color-blind visitor is not a status the site has communicated.

**Density.** The site breathes. There is rhythm between darkness and illumination, dense technical panels and open explanatory space, motion and stillness. When every section is an instrument, nothing feels important.

**No gradient soup.** A gradient is allowed when it expresses a state. A gradient on every panel is noise. The reduced-motion theme flattens gradients that are not state-bearing.

**Iconography.** Icons are functional. An icon that has no text equivalent is a draft. The site is keyboard-navigable; every interactive element is reachable by tab; the focus ring is visible; the focus order matches the visual order.

**Theme count.** **v1 ships dark only.** Light theme is a follow-on. The reasoning is recorded in the Phase 1 ADR: a deliberately authored dark-only Observatory with proper contrast, forced-colors support, and print stylesheet is preferable to a half-authored light theme that doubles visual and testing work without serving the visitor. A visitor who needs light can use a browser or OS theme override; the site's forced-colors behavior is verified regardless of base theme.

**No seventeen fonts.** The system uses Ubuntu, one monospace stack, and the system font for the OS-native affordances. That is the entire typeface budget.

---

## 10. Artifact Inventory

The site is only as credible as the evidence it shows. The first production pass curates the corpus below. Each artifact is named by its identity, its schema, its producer, its commit, its target, its hardware configuration, its validation scope, its digest, and any redactions.

| Artifact | Schema | Producer | Commit | Target | Hardware | Validation scope | Digest | Redactions |
|---|---|---|---|---|---|---|---|---|
| One ComputeImage manifest | the engine's `cimage-manifest` schema (current version, named in the ADR) | the compiler or ComputeImage builder, **not** the SSG | named | named target | named | name and limit | `computeimage_artifact_digest`; if sanitized, also `original_artifact_digest` per §4.8 | none, or named in redaction manifest |
| One compilation receipt | the engine's `compile-receipt` schema (named) | the compiler | named | named target | named | name and limit | content-addressed | none, or named in redaction manifest |
| One execution receipt | the engine's `execute-receipt` schema (named) | the runtime | named | named target | named | name and limit | content-addressed | none, or named in redaction manifest |
| One qualification record | the engine's `qualify-record` schema (named) | the qualification harness | named | named target | named | name and limit | content-addressed | none, or named in redaction manifest |
| One failure receipt | the engine's `execute-receipt` schema (failure variant, named) | the runtime | named | named target | named | name and limit | content-addressed | none, or named in redaction manifest |
| One replay result | the engine's `replay-result` schema (named) | the replay applier | named | named target | named | name and limit | content-addressed | none, or named in redaction manifest |
| One comparison or regression artifact | the engine's `regression-finding` schema (named) | the regression harness | named | named target | named | name and limit | content-addressed | none, or named in redaction manifest |
| One complete provenance chain | the engine's `provenance-chain` schema (named) | the provenance builder | named | n/a | n/a | end-to-end | content-addressed | none, or named in redaction manifest |

**The schema names above are placeholders.** Phase 1 includes an ADR that binds each schema name to the actual schema identifier used in the engine, or creates the schema if it does not exist. The site does not invent schemas to make the editorial symmetry work.

**The producer column names the component that produces the artifact, not the component that publishes it.** The compiler produces manifests. The runtime produces execution receipts. The SSG publishes public projections of the artifacts; it is not the producer. The provenance chain records the producer.

**Specimen selection happens in Phase 2.** The architect selects, with the corpus authors, which artifacts become the public specimens. The selection is recorded in `evidence-index.json` and frozen before Phase 3 begins; the manuscript is authored around the selected specimens.

**Where measurements may appear.** The evidence corpus is the sole authority from which measurements may be projected. The corpus is not the only page where measurements appear; the home page, the Status page, the Run page, and the Specimen page all project measurements from the corpus. What the corpus buys is provenance: a measurement that appears on the home page can be traced through the corpus to the receipt that supports it.

**Visual treatment of artifacts.** The site distinguishes three things:

- **Illustrative diagrams** may glow and move. They are labeled as illustrative.
- **Compile-verified artifacts** must say exactly what they prove and what they do not.
- **Measured artifacts** must identify the machine and the execution fingerprint.

**Failure is presented with the same visual dignity as success.** A rejected plan, a stale outcome, a failed qualification, and a recovered transaction are not exceptions to the visual language. They use the same Receipt and EvidenceCard components as successful artifacts.

The corpus is published under `/evidence/corpus/`. Each artifact has a stable URL. Each artifact's identity is content-addressed. The corpus is the only authority from which measurements may originate.

---

## 11. Implementation Sequence

The site is produced in nine phases. Each phase has a definition of done. The phases unlock in order. A phase that does not satisfy its definition of done blocks the next.

**Phase 1 — Editorial and truth freeze.** This document is approved. The current repository baseline is committed. The capability vocabulary is frozen. The canonical journey is frozen. The page purposes are frozen. The terminology audit is complete. The legacy pages are classified as *deleted*, *redirected*, *demoted*, or *rewritten*. **An ADR binds the schema names in §10 to the actual engine schemas, or creates them.** The light-theme decision (dark only for v1) is recorded.

**Phase 2 — Data schemas, evidence freeze, specimen selection.** The schemas in `schemas/` are authored. The data layer in §4.1 exists in the repository and validates. The validation gate is implemented with discriminated-union record types per §4.4. **The artifact inventory is curated. Specific specimens are selected, sanitized or not, and frozen in `evidence-index.json` before Phase 3 begins.** The redirect table is part of the SSG input and is validated against the legacy paths in §7.2. The status migration of the existing `capabilities.json` is complete. `capability-history.json` is initialized for the records in migration. The deployment platform ADR (§15) is signed and the Cloudflare Pages project is provisioned.

**Phase 3 — Complete manuscript.** Every page in §6 is written completely, in the prose it will use, before any component receives the prose. The manuscript is authored around the specimens frozen in Phase 2. The manuscript is reviewed against the Editorial Constitution. **The manuscript is a separate, binding artifact. The briefs in §6 are its outline, not its substitute.**

**Phase 4 — Shell, routes, tokens, and semantic component consolidation.** One shell. One navigation system. One route model (canonical paths emitted as pages, legacy paths emitted as redirects, no self-redirects). One footer. **One design-token source with a working palette — Phase 4 establishes the palette values; Phase 8 tunes composition.** The component contracts in §8 are honored in skeleton: each component renders with stub data. The `SelectionController` is implemented and tested in isolation. The status, evidence, and receipt renderers exist in semantic form. **This phase precedes the Life experience so that the central interaction is built on top of the canonical components, not before them.**

**Phase 5 — Static Life experience.** The static twelve-stage document at `/observatory/life/` is implemented. The full sequence is present in HTML. The default density is rendered. JavaScript deepens density and restores selection but does not create the page. The no-JavaScript fallback is the authored work, not a downgrade to another route.

**Phase 6 — Addressable interaction and cross-view coordination.** The selection state becomes URL-addressable. The `SelectionController` is wired to the URL. The `SelectionController` parses, validates, normalizes, writes, and broadcasts selection changes; instruments subscribe to its broadcast. The selection state restores on reload. The three densities are reachable through URL parameters. The specimen identity is the same identity the ComputeImage and Specimen pages use.

**Phase 7 — Evidence binding and rendering.** The already-selected specimens are wired into the ComputeImage, Specimen, Evidence, Run, and Observatory pages. Illustrative telemetry is replaced wherever the presentation implies a live or measured state. The Compiler Lab surface is bound to a **recorded** run from the corpus. The word *LIVE* does not appear on the surface at v1; the word *Recorded* appears where a value is from the corpus. **A live daemon surface is a follow-on, scoped by its own ADR; it is not v1.**

**Phase 8 — Editorial and visual polish.** Repetition is cut. Prose is tightened. Transitions are written. Captions are added. Density is tuned. Effects are subordinated to meaning. Every diagram has a caption and a textual equivalent. Every claim has a state and an evidence class. Every page passes the Editorial Constitution. The visual dramaturgy pass tunes rhythm, silence, and palette composition.

**Phase 9 — Release hardening.** Every gate in §12 is green. Link integrity, accessibility, keyboard behavior, reduced motion, responsive layouts, schema validation, content-claim validation, evidence applicability, canonical URLs, sitemap, robots, structured metadata, build identity, source-commit link, asset budgets, security, privacy, performance, deployment smoke test: all are release gates. Each automated gate is automated; each human gate is signed. The first release is named *Prism Observatory v1*.

---

## 12. Release Gates

The release does not ship unless every gate below is green. The gates are organized in three classes. **A gate that requires taste is not automated; a gate that does not require taste is not human-reviewed.** The mix is the rigor.

### 12.1 Automated release gates

These are enforced by tools. They return pass, fail, or a specific error. A fail blocks the release.

**A1. Route integrity.** Every canonical path serves the canonical content. Every legacy path in §7.2 serves a real HTTP 301 to its destination, served by the deployment platform from the `_redirects` file. No path serves the wrong generation. No path serves a 404 where a redirect would resolve. No self-redirects. No `/index.html` resolving to two destinations. The 410 wildcards are not emitted before the §7.3 compatibility window elapses. The A22 smoke test confirms the platform actually serves the redirects.

**A2. Status-vocabulary purity (structured).** Status-bearing language is emitted only through `Claim`, `StatusTable`, and `Release` components. The validator inspects those structured surfaces and rejects records whose `state`, `distribution_state`, or `maturity` fields are not members of the §3 vocabulary, or whose maturity × distribution pair is not in the §3.3 allowed set. A prose linter (a regex set) flags suspicious uses of the forbidden words in page prose; flagged lines are returned for human review (H1), not auto-rejected, because a plain grep cannot distinguish "available memory" from a forbidden status claim. The unconditional zero-match grep is removed.

**A3. Data-layer validation.** Every record in every data file validates against the discriminated union for its record type in §4.4. Required fields are present. Inappropriate fields are absent. References resolve. Cross-references are typed. **Allowed maturity × distribution pairs are enforced.** `capability-history.json` genesis records are validated: `from: null` only for `sequence: 1`; `sequence` is a positive integer, monotonically increasing per `capability_id`; `transition_id` is unique.

**A4. Evidence-boundary completeness.** For each `ValidatedRecord`, a `target` and an `evidence_id` are present and resolve. For each `ReleaseRecord`, a signature, a verification path, and a support boundary document are present. For each `QualifyingRecord`, a test or fixture path and a named limit are present. The universal checklist has been replaced by type-specific requirements.

**A5. Chapter list locality.** No page-local chapter list contains a chapter from another page. The 60-entry global chapter dump is absent from the codebase and the data layer.

**A6. Component registration.** Every component in the output is registered with exactly one declared responsibility identifier from §8. A component registered with zero identifiers is rejected. A component registered with more than one is rejected. **Whether the declaration is honest is judged by H8; whether the registration is correct is judged by this gate.**

**A7. Manuscript-to-page structural match.** Every page in the output references the manuscript for its brief. The page's section IDs match the brief's section list. The page does not introduce section IDs the brief does not mandate, and does not omit ones it does. **Whether the prose contradicts the brief is judged by H4; whether the structure matches is judged by this gate.**

**A8. Diagram caption and description.** Every `<figure>` in the output has a `<figcaption>` and an `aria-describedby` reference to a textual equivalent. **Whether the textual equivalent is meaningful and equivalent is judged by H5; whether the markup is present is judged by this gate.**

**A9. Reduced motion compliance.** With `prefers-reduced-motion: reduce`, the prism effect is absent, ambient motion is absent, and selection state is preserved through layout. Verified by a render of the route with the media feature forced.

**A10. Keyboard parity.** Every interactive element is reachable by tab. The focus order matches the visual order at the canonical viewport. The focus ring is visible against every background it overlaps. The skip link is functional. Verified by an automated tab-order traversal at the canonical viewport. **Whether the tab order remains correct across all responsive layouts is judged by H9.**

**A11. Screen-reader parity.** The semantic structure of every page is complete without CSS. Headings are nested correctly. Landmarks are present. `aria-live` regions are used for state changes. Alt text is present. Verified by axe-core or an equivalent automated audit, plus a manual check on the Observatory. **Whether alt text is specific rather than merely present is judged by H1.**

**A12. No-JS rendering.** Every canonical route, including `/observatory/life/`, renders meaningfully without JavaScript. The full sequence at the Observatory is present in HTML. Verified by a build that strips JS and renders each route to a static asset for comparison.

**A13. Schema and cross-reference integrity.** Every JSON file validates against the schema in `schemas/`. Every cross-reference resolves to a record of the expected type. A dangling reference is a build failure.

**A14. Evidence applicability.** A claim referencing an evidence record is rejected if the evidence record's applicability fields do not match the claim's reference: source commit or build identity, schema version, target identity, feature set, model identity, and validation scope. **Age alone is not invalidity.** A historical evidence record remains published; if it no longer supports any current claim, it may be marked `superseded` in `evidence-index.json` with a date and a reason. The mark is editorial; the superseded record remains available for history and replay.

**A15. Canonical URLs, sitemap, robots.** Every page sets its canonical URL in the head. A sitemap is generated and submitted. A robots policy is set. OG and Twitter card metadata are present.

**A16. Build identity and source commit.** Every rendered page carries a build identity and a source commit, visible in the page source and in the meta. The site knows what version of itself it is.

**A17. Status-not-by-color-alone.** Status is communicated in text, shape, and semantics in addition to color. Verified by rendering the page in forced-colors mode and by checking that every status badge has a textual label and (where applicable) an icon or shape variant.

**A18. Performance budget.** HTML per route ≤ 60 KB gzipped. Critical CSS per route ≤ 18 KB gzipped. JavaScript per route ≤ 80 KB gzipped, with the Observatory permitted up to 120 KB. LCP ≤ 2.5 s, CLS ≤ 0.1, INP ≤ 200 ms. Image budget per page named. Font-loading strategy: preloaded, subsetted, with `font-display: swap`. **Reproducible test contract:** the test runs at viewport 1366×768, on the named CI machine class (recorded in the test manifest), against the Cloudflare Pages preview URL for the candidate build, with a cold cache, repeated 5 times, with the **median** of the runs reported. Slow-4G is simulated via the named network profile (also recorded). Lighthouse and route-level asset reports are the verification.

**A19. Security and privacy.** A strict Content Security Policy is set, served as a real HTTP response header by the platform (see §15.4). **No third-party requests of any kind.** No analytics, telemetry, or third-party fonts, scripts, or images. Dependencies audited. **Local assets are filename-fingerprinted for cache identity. SRI is not required for same-origin assets (their integrity is bounded by the platform's deployment guarantees) and is moot for external assets (the site forbids them). Same-origin is an authority boundary, not a cryptographic proof of integrity: trust in the actual bytes still depends on the repository, the build process, the deployment credentials, and the hosting platform. The CSP is the primary defense; fingerprinting provides cache-busting and content identity; the platform's own integrity model is layered on top.** No `eval`, no inline scripts, no third-party iframes. No interaction data leaves the browser while telemetry remains disabled; if telemetry is enabled later, it is named in an ADR and reviewed in §14. Verified by CSP report, dependency audit, and a network capture that shows no third-party traffic.

**A20. Accessibility extras.** Contrast ratio meets WCAG 2.2 AA at the chosen theme. **Behavior at 200% and 400% zoom is verified on every canonical route** (not only the seven major ones). For intrinsically wide content — raw artifact manifests, status tables, and the Specimen page data blocks — one-dimensional contained horizontal scrolling is permitted; the gate does not require zero horizontal scrolling when the content's nature is wide tabular data. Forced-colors behavior is verified. Touch targets ≥ 44×44 CSS pixels. Status communication does not depend on color or animation. Verified by axe, by a 400%-zoom render, and by a forced-colors render.

**A21. /docs/ allowlist.** The `/docs/` route serves only files explicitly named in `docs-publication.json`. Internal notes, outdated ADRs, and implementation debris do not reach the live site. Verified by a build that diffs the served `/docs/` tree against the allowlist.

**A22. Deployment smoke test.** The smoke test runs at two points, against two distinct surfaces. **Preview smoke** runs the candidate build at the Cloudflare Pages preview URL produced for the PR or release branch. **Post-production smoke** runs against the production URL after Cloudflare builds the production deployment from the merged commit. Both follow the same path: a synthetic visitor from the home page to a Status row, opens the row to its evidence, follows the evidence to the published receipt, and reaches the receipt's stable URL. The post-production smoke additionally verifies the response headers in §15.4 (CSP, HSTS, Referrer-Policy, COOP/CORP) and the cache directives in §15.5 (`/assets/*` and `/pkg/*` immutable, HTML revalidated). The path is recorded as a deployment log with the build identity and the source commit. **A failure of the preview smoke blocks merge to `main`. A failure of the post-production smoke blocks the deployment from being declared the current production; the previous production deployment remains live, and the failed production deployment is preserved (with build identity, source commit, failure point, captured page) but not promoted. The site does not page on-call; it does not assume an on-call system exists. Alerting is added when there is a real destination for the alert.**

**A23. Schema-naming audit.** Every schema referenced in §10 and §4.3 binds to an actual schema identifier in the engine, or to a schema explicitly created for the site by an ADR. The audit is a list; the list is checked into the repository.

### 12.2 Human editorial gates

These are enforced by a named human reviewer. The reviewer's name, the date, and the review's outcome are recorded in the release log.

**H1. Status-vocabulary purity (semantic).** Beyond the A2 structured check, the reviewer confirms that the *use* of every status word in prose is correct in context. A word that passes the structured check but slips into prose as decoration is marked for removal. Alt text specificity is reviewed here.

**H2. Repetition audit.** The reviewer reads the manuscript and the rendered pages together. A paragraph that says the same thing as a paragraph on a linked page is marked. Exceptions are limited to one-sentence restatements that exist for transition.

**H3. Paragraph function.** Every paragraph must be justifiable in one sentence by the reviewer. A paragraph that cannot be justified is removed.

**H4. Page purpose match.** Every page in the rendered site makes the argument its brief in §6 mandates. A page that drifts, even by paragraph-level dilution, fails the gate. Contradictions between manuscript and brief are caught here, not by A7.

**H5. Diagram usefulness.** Every diagram adds information the surrounding prose does not. A diagram that is purely decorative is removed or replaced by its textual equivalent. The textual equivalent is judged for actual equivalence here, not by A8.

**H6. Voice consistency.** The prose sounds like one author. Vocabulary, register, and rhythm are consistent across pages. The reviewer reads the manuscript aloud; passages that break the voice are rewritten.

**H7. Evidence binding quality.** The specimens on the ComputeImage, Specimen, Evidence, Run, and Observatory pages are the right specimens for the claims they support. A specimen that is interesting but irrelevant to the page's argument is replaced.

### 12.3 Human architectural gates

These are enforced by the architect (Julian Torres) or a named delegate. The architect's signoff is recorded in the release log.

**H8. Component responsibility review.** The architect confirms that every component in §8 owns exactly the responsibility it declares and no more. A component whose declared responsibility has grown a second behavior is decomposed. The declaration's honesty is judged here, not by A6.

**H9. Visual hierarchy and responsive flow.** The architect confirms that the visual rhythm of the site serves the editorial rhythm. Sections that compete at the same intensity are reordered or quieted. Transitions that read as abrupt are softened. Density decisions are intentional. The tab order and reading order remain correct across all responsive layouts.

**H10. Interaction design.** The architect confirms that the Observatory's interaction model serves the canonical journey. An interaction that is clever but obscures the underlying data is removed. The URL is the source of selection truth; the interaction is the visual consequence.

**H11. Truth architecture drift.** The architect confirms that editorial fields in the data layer are still editorial, generated fields are still generated, and no field has drifted in a way the validator would not catch. The data layer's publication form is the form the architect approved.

**H12. Honest limits.** The architect confirms that the Status page renders the validation gaps, the Run page names its absence where a packaged release does not exist, the PrismAgent page surfaces its Unreleased state, and the conditional pages (Prism ML, General Compute) are either present with evidence or absent from the site. A page that hides a limit fails this gate.

**H13. Release readiness.** The architect confirms that the release is, in their judgement, ready. This gate is not a veto-by-default; it is a final architectural pass. The release is named *Prism Observatory v1* only after this gate is signed.

---

## 13. Colophon

Prism Engine is independently developed by Julian Torres. The system is open source under the project's license. Collaboration takes five shapes: hardware validation, datacenter deployments, engineering, research, and edge. Each shape has a deliverable. The collaboration surface is named; the contracts are open; the proprietary models remain confidential where an engagement requires it. The evidence class is recorded either way.

The site is *Prism Observatory v1: the evidence-bound public projection of Prism Engine.* It mirrors Prism's subject, projection, and receipt concepts. It is not a transaction system; a browser selection is not a constitutional event. The vocabulary is preserved because it is meaningful, not because every action it describes happens in the browser.

---

## 14. Open Decisions

The work in this document is settled where it is settled. The work below is not. Each item blocks the phase that depends on it.

1. **Schema binding.** The schema names in §10 are placeholders. Phase 1 includes an ADR that binds each name to the actual engine schema, or creates the schema. Until then, the artifact inventory is not yet a buildable inventory.
2. **Specimen selection.** The artifact inventory names the required corpus. Specific specimens are selected and frozen in Phase 2. The selection is gated by the evidence-applicability gate (A14), by the sanitization model (§4.8), and by the architect's signoff (H7).
3. **Status migration of the existing capabilities file.** The current `capabilities.json` uses a vocabulary that does not match §3. Migration is a Phase 2 deliverable. Records that cannot be migrated cleanly are removed rather than preserved in a half-migrated form. The migration initializes `capability-history.json` for each migrated record.
4. **Author's name and pronouncement in the Colophon.** §13 names Julian Torres. The manuscript must confirm the wording with the author before Phase 3 freezes it.
5. **Brand color palette values.** Tokens are defined; the palette structure is established in Phase 4; the specific values are tuned in Phase 8. The Phase 4 work commits working values; the Phase 8 work adjusts them for composition.
6. **Compiler Lab: Recorded, not Live (decided).** The home page references Compiler Lab. **The v1 surface renders *Recorded* (a value from the corpus), not *Live*.** A live daemon surface is a follow-on, scoped by its own ADR, with §12 A19 updated to permit the necessary network call when the ADR is signed.
7. **Observatory URL shape (decided for v1).** §7.1 places the Life experience at `/observatory/life/`. This is the v1 path. A case for `/life/` at the canonical root is open for a future version; it does not block the v1 manuscript.
8. **Search (decided for v1: minimal).** The 404 page's search field is satisfied by a minimal search surface over the data layer (manifest, capabilities, evidence, models), built as `search-index.json` per §4.1 and consumed by a small client-side filter on the 404 page. The 404 page's primary suggestion is the site search; a separate repository search link is permitted but secondary. Richer search is a follow-on.
9. **Telemetry.** The site is silent. A future version may collect minimal, privacy-respecting analytics. The decision is open, is a constitutional change when made, and is currently blocked by A19.
10. **Internationalization.** The site is authored in English. Internationalization is a follow-on, not part of v1.
11. **Light theme.** v1 is dark only. A light theme is a follow-on, scoped by its own ADR. The Phase 1 ADR records the reasoning.
12. **Diagnostics component data model.** §8 defines Diagnostics as session-local, non-persistent, and non-transmitting. The exact data model is open and is named in an ADR before the component ships.
13. **Conditional route evaluation.** §6.12 and §6.13 make Prism ML and General Compute conditional on evidence. The decision about which conditional routes are present at v1 release is made by the architect after Phase 7, against the evidence corpus. H12 enforces that conditional routes are either present with evidence or absent from the site.

---

## 15. Deployment Platform Contract

The redirects, headers, cache, preview, and rollback this document requires are real platform capabilities. The SSG produces the redirect table, the `_headers` file, the asset manifest, the `_redirects` file, and the build identity. The platform serves them.

### 15.1 Production host and DNS

The production host is **Cloudflare Pages**. The project is connected to this GitHub repository; the production branch is `main`. The custom domain is `prism-engine.tribunus.dev`. The domain's DNS is on Cloudflare; the Pages project attaches the custom domain via a CNAME. The existing GitHub Pages deployment is decommissioned as part of the cutover; the cutover is operational, not spec-affecting, and is recorded in the Phase 2 deployment platform ADR.

The repository stays on GitHub. The publication layer is Cloudflare. The build is invoked by Cloudflare on push to the configured branch and on pull request; the build output is the `docs/` directory produced by the SSG (see §15.2). The custom domain does not change; only the platform that serves it.

A future v2 may move to another platform if Cloudflare Pages ceases to meet the contract. The move is a constitutional change requiring an ADR; the spec is platform-agnostic except where this section names a Cloudflare-specific mechanism (e.g., `_redirects`, `_headers`).

### 15.2 Build integration

Cloudflare Pages is configured to:

- **Build command:** `bash scripts/build-site.sh` (or its documented equivalent — a single entry point that orchestrates the Rust toolchain, the WASM bundle, the SSG, and any pre-publication validations).
- **Output directory:** `docs/` (the directory the SSG already writes to).
- **Environment variables:** the rustup toolchain path, any secrets the build needs, and the build identity variables (commit SHA, build number).
- **Compatibility flags:** Node and Python are not required; the build is a Rust + shell pipeline.

The build entry point is a checked-in script. The Cloudflare Pages dashboard is not the source of build configuration; the script is.

**Rust and WASM toolchain are not preinstalled in the Cloudflare Pages build image.** The documented Cloudflare Pages build image provides Go, Node.js, Python, and Ruby; it does not list Rust. The build script therefore performs one of two equivalent strategies, recorded in the deployment platform ADR:

- **Strategy A (Pages builds):** the build script installs a pinned Rust toolchain via `rustup` (channel and components named in the ADR), installs the WASM build target (`wasm32-unknown-unknown`), installs `wasm-bindgen-cli` and `wasm-opt` at pinned versions, then runs the SSG. The toolchain install is reproducible (pinned versions, no floating network expectations beyond the rustup dist server) and is recorded in the build log so a divergent toolchain is detectable.
- **Strategy B (GitHub Actions builds, Pages serves):** a GitHub Actions workflow runs the same build script on a self-hosted or larger runner with Rust preinstalled, and uploads the resulting `docs/` directory as a Pages deployment via `wrangler pages deploy`. Pages does not run a build of its own; the dashboard's "build command" is set to a no-op or omitted. The workflow is the source of build truth.

Strategy A is preferred for v1 because it keeps the build artifact a function of the source commit on the same platform that serves it. Strategy B is the fallback if the Pages build image constraints prove incompatible with the build time. The deployment platform ADR records the choice and the rationale.

### 15.3 Redirect mechanism

Cloudflare Pages serves a `_redirects` file from the site root. The SSG emits this file from the validated §7.2 table at build time. The file's format is the standard Cloudflare Pages redirects format:

```text
/index.html                                /                                    301
/architecture.html                         /architecture/                        301
/capabilities.html                         /status/                              301
/capabilities/                             /status/                              301
/computeimage.html                         /computeimage/                        301
/heterogeneous.html                        /architecture/                        301
/heterogeneous/                            /architecture/                        301
/roadmap.html                              /roadmap/                             301
/prism-ml.html                             /prism-ml/                            301
/general-compute.html                      /general-compute/                     301
/work-with-prism.html                      /prismagent/                          301
/work-with-prism/                          /prismagent/                          301
/demo.html                                 /observatory/life/                    301
/demo/                                     /observatory/life/                    301
/projection-repro.html                     /observatory/life/                    301
/projection-repro/                         /observatory/life/                    301
/field-guide.html                          /start/                               301
/field-guide/                              /start/                               301
/start-here.html                           /start/                               301
/start-here/                               /start/                               301
/run.html                                  /run/                                 301
/evidence.html                             /evidence/                            301
```

These are real HTTP 301 responses served at the edge. Crawlers honor them. Browsers honor them. The redirect latency is the latency of an edge redirect, not a browser-side meta refresh.

**`_redirects` supports status codes 200, 301, 302, 303, 307, and 308. It does not support emitting arbitrary responses such as 410.** The legacy asset retirement described in §7.3 is therefore not expressed in `_redirects`. The deployment platform ADR names the actual mechanism: a Pages Function (preferred) or a Cloudflare Worker route that matches the retired asset path prefixes, returns a real HTTP `410 Gone` with a `Link` header to the canonical asset path, and is deployed alongside the static site. Until that mechanism exists, expired legacy asset paths are simply **absent from the build output**; they resolve through the authored 404 (§7.4) rather than through a 410. A1 requires that the platform serve the redirect as specified; A22 verifies a sample.

The conditional routes (`/prism-ml/`, `/general-compute/`) are emitted with a destination of `/lab/` if the conditional route does not exist in the manifest at build time. The validator rejects a build where the redirect table and the manifest disagree about which routes exist.

### 15.4 Header mechanism

Cloudflare Pages serves a `_headers` file from the site root. The SSG emits this file at build time. The file applies headers per path or path-prefix.

```text
/*
  Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
  Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'
  Permissions-Policy: camera=(), microphone=(), geolocation=()
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Resource-Policy: same-origin
```

The CSP is authored as one of the platform's full-feature HTTP headers, not via a `<meta>` tag. Meta-delivered CSP does not support every CSP feature (frame-ancestors, sandbox, report-uri, and others are restricted or unsupported in some browsers when delivered via meta); the spec uses real response headers.

A1 requires that the platform actually serves these headers; the A22 smoke test verifies them. A reviewer inspecting the live site with browser devtools sees the CSP, Referrer-Policy, and the rest as real response headers, not as meta tags.

The site does not use `<meta http-equiv="Content-Security-Policy">` or `<meta name="referrer">`. The meta fallback is not used because the platform supports real headers.

### 15.5 Cache policy

The `_headers` file expresses cache directives per path.

```text
/*
  Cache-Control: public, max-age=0, must-revalidate

/assets/*
  Cache-Control: public, max-age=31536000, immutable

/pkg/*
  Cache-Control: public, max-age=31536000, immutable
```

- HTML pages: revalidated every request. The site is content-driven and small; the cache benefit is minimal and the cost of staleness is real.
- Fingerprinted assets under `/assets/` and `/pkg/`: long-cache immutable. The filename hash changes when the content changes; long cache is safe.
- Service worker: not used at v1.

The cache directives are real HTTP response headers served by the platform. CDN intermediaries and browsers honor them.

### 15.6 Preview environment

Every push to a non-production branch and every pull request receives a Cloudflare Pages **preview URL**. The preview is the same build pipeline as production; the only differences are the URL and the absence of the custom domain binding. The preview URL is the surface the A22 smoke test runs against. The preview is also the surface the H1–H13 reviewers use. A reviewer approves a candidate build by approving its preview URL, not by inspecting the production site.

A preview deployment is a distinct deployment artifact. It is identified by a Cloudflare-assigned deployment ID, has its own URL, and is **not** the same artifact as any production deployment even if it was built from the same source commit. Preview deployments are valid for reviewing, gating, and human signoff; they are **not** valid rollback targets. Rollback targets are previous production deployments only.

### 15.7 Rollback and promotion

Promotion follows the documented Cloudflare Pages workflow, not a "flip the preview into production" gesture:

1. A pull request (or a release branch) is opened. Cloudflare Pages builds the preview deployment.
2. All automated gates (A1–A23) run against the preview. The smoke test (A22) runs against the preview URL. Human gates (H1–H13) are signed against the preview URL.
3. A passing review permits merging the exact reviewed commit into `main`. The source commit becomes the commit Pages will build from for production.
4. Cloudflare Pages detects the new commit on `main` and produces a **production deployment**. This is a new build, a new deployment ID, a new artifact. The source commit is identical to the preview; the deployment identities are distinct.
5. A **post-production smoke check** runs against the production URL. The check is the same path-following flow as A22, plus a verification of the response headers in §15.4 and the cache directives in §15.5. The check is automated. A failure of the check blocks the deployment from being declared the current production; the previous production deployment remains live.
6. The previous production deployment is recorded as the rollback target. The release log records the promotion with build identity, source commit, gates passed, reviewer, and timestamp.

Rollback to a previous production deployment is a separate operation. The Cloudflare Pages dashboard redeploys the previous deployment's artifact; the live site is the current production until the redeploy completes, then becomes the previous-previous production after the redeploy. There is no window during which the failed build is live. Previous production deployments are the only valid rollback targets; preview deployments are not.

When a bug is discovered in production (one that passed the gates), rollback is one click in the Cloudflare Pages dashboard: redeploy the last known-good production deployment. The release log records the rollback.

The deployment platform ADR defines the promotion and rollback workflow precisely, including the post-production smoke check command and its exit codes. The ADR is the operational source of truth for promotion; this section is the constitutional contract.

### 15.8 Cutover from GitHub Pages

The cutover from GitHub Pages to Cloudflare Pages is operational, not spec-affecting. The steps are:

1. The Cloudflare Pages project is provisioned and connected to the GitHub repository.
2. A staging deployment is verified at the Cloudflare Pages default domain.
3. The custom domain `prism-engine.tribunus.dev` is moved to Cloudflare DNS and attached to the Pages project. The TLS certificate is provisioned by Cloudflare.
4. The GitHub Pages deployment is decommissioned (the GitHub Pages source branch is set to `none`, the custom domain is removed from the GitHub Pages settings).
5. The first production build on Cloudflare Pages is promoted.
6. A new `CNAME` is not committed to the repository (Cloudflare Pages does not require it; the custom domain is configured in the dashboard). If a `CNAME` file is later required for some reason, the Phase 2 deployment platform ADR records the change.

The cutover is recorded in the Phase 2 deployment platform ADR. The cutover does not change the spec; the spec's platform contract was always Cloudflare Pages, and the GitHub Pages deployment was a pre-spec interim state.

---

*End of master specification, APPROVED v1.0. Governance is frozen. The next deliverables, in order, are: the schema-binding ADR; the deployment-platform ADR; the evidence selection and sanitization freeze; and the complete page manuscript.*
