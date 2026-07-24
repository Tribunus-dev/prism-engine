# Prism Engine Website — Full Materialization Implementation Plan

**Repository:** `Tribunus-dev/prism-engine`  
**Website root:** `/docs`  
**Audit baseline:** `main` at commit `43b83e252bc365165063f11c0c2e5facbe264116`  
**Prepared:** 2026-07-24  
**Purpose:** Finalize the static website as the authoritative, evidence-bounded public interface to the current Prism Engine codebase.

---

## 1. Executive diagnosis

The website is no longer a weak prototype. It already possesses a distinctive visual language, a coherent “Observatory” metaphor, an explicit runtime composition boundary, reusable renderers, reduced-motion handling, and a deliberate refusal to invent benchmark data.

The core problem is now **misalignment between the site’s public narrative and the repository’s actual center of gravity**.

The repository currently presents Prism as:

- an ECS-native compiler and runtime;
- a versioned ComputeImage (`.cimage`) system;
- a constitutional authority model with transactions, replay, durable-before-ack storage, stale-outcome rejection, and rebuildable projections;
- an evidence and operations surface spanning inspection, experiments, replay, comparison, regression detection, and baseline promotion;
- a heterogeneous execution framework with CPU, Metal, ROCm/HIP, XDNA/XDNA2 planning, CUDA integration boundaries, Core ML/ANE surfaces, multimodal pipelines, and audio pipelines;
- a pre-1.0 system whose support claims must be attached to build, hardware, conformance, and receipt gates.

The website still behaves as though its primary story is:

> Apple Silicon demo first, then a broad architecture promise.

That is now too narrow and, in places, stale. The correct fully materialized vision is:

> **Prism Observatory is the live public projection of one canonical deployment graph: model identity, compilation, representation search, ComputeImage realization, heterogeneous execution, constitutional state, and evidence.**

The work should therefore not be treated as another cosmetic pass. It is an **information-architecture, evidence-model, content-governance, and runtime-consolidation release**.

---

## 2. Audit baseline and factual source of truth

### 2.1 Repository state

The current README describes Prism as a compiler and runtime that emits target-aware `.cimage` artifacts and executes them through an ECS-native runtime with explicit placement, residency, scheduling, validation, and evidence.

The current repository claims include:

- CLI pull, compile, list, run, interactive chat, and OpenAI-compatible HTTP serving;
- ComputeImages carrying weights, layouts, execution plans, and validation evidence;
- CPU, Metal, ROCm/HIP, and XDNA planning;
- quantization, ternarization, mixed-precision fallback, calibration, and evolutionary search;
- constitutional ECS domains for ingestion, discovery, model/session/work, compilation, multimodal, distributed, and ingress;
- durable event storage, receipts, replay, restart recovery, stale-outcome rejection, and rebuildable projections;
- C ABI, Swift bridge, Node-API bindings, CLI binaries, and HTTP server surfaces.

The latest audited commit completes heterogeneous workload evaluation propagation and graph-canonical runtime wiring. Recent commits also remove fabricated inference and Metal receipts and reject non-authoritative legacy receipts. Those changes should materially affect the website’s evidence story.

### 2.2 Website state

The `/docs` site currently contains:

- a visually ambitious homepage;
- architecture, demo, heterogeneous execution, roadmap, field guide, Prism ML, General Compute, and partnership surfaces;
- a reusable visual vocabulary around computation, ComputeImages, receipts, and observation;
- a modular ES-module runtime with a composition root in `docs/js/app.js`;
- systems for observation graph, accessibility, state projection, persistent object state, observatory shell, canonical stages, effect budgets, scroll observation, ComputeImage rendering, receipts, navigation, and GPU effects;
- runtime diagnostics and a Playwright startup matrix.

The architecture is substantially better than a conventional static marketing site. The remaining work is to make it **authoritative, coherent, maintainable, and truthfully synchronized with the codebase**.

---

## 3. Ruthless diagnosis

## 3.1 The primary navigation encodes an outdated ontology

The global navigation is currently:

- Graph
- Artifact
- Execution
- Evidence

This is elegant but incomplete. It hides the two repository domains that now distinguish Prism most strongly:

1. **Constitutional ECS / authority**
2. **Operations / replay / experiments / proof**

“Evidence” is currently routed to `roadmap.html`, which is conceptually incorrect. A roadmap is future-oriented; evidence is present or historical proof. This conflation weakens the site’s epistemic contract.

### Required correction

The top-level information architecture should become:

- **Overview**
- **Compiler**
- **ComputeImage**
- **Runtime**
- **Evidence**
- **Status**
- **Docs / Source**

On desktop, this can remain visually compressed through chapter labels and an expanded menu. The core requirement is semantic separation, not necessarily seven persistent horizontal links.

---

## 3.2 The homepage’s first-screen claim is poetic but not sufficiently operational

“Computation was never meant to target hardware” is distinctive, but the first viewport does not immediately tell a technical visitor:

- what Prism is;
- what can be run today;
- what artifact it produces;
- what is measured versus implemented versus planned;
- why Prism is materially different from an inference runtime.

The current qualifier says the primary validated path is Apple Silicon + Metal. The repository now documents active Apple Silicon and MI300X validation paths, plus a wider set of implemented integration boundaries.

### Required correction

Preserve the poetic line as the emotional thesis, but make the operational thesis visible in the same viewport:

> Prism compiles model graphs and weights into inspectable ComputeImages, then executes them through a constitutional ECS runtime with explicit placement, residency, handoffs, and receipts.

Directly beneath it, show a **live status strip** generated from structured data:

- Apple Silicon / Metal — validating
- CPU — reference and hardening
- MI300X / ROCm-HIP — validating
- XDNA/XDNA2 — compile-verified
- ANE/Core ML — implemented boundary, qualifying
- CUDA — implemented boundary, qualifying
- Multimodal / audio — implemented pipeline, qualifying

Do not bury this in Markdown-only documentation.

---

## 3.3 The site contains a strong capability map, but it is not part of the website

`docs/current-capabilities.md` is the most accurate public description of the repository’s present state. It distinguishes:

- implemented;
- qualifying;
- planned or developing.

It describes replay, experiments, runtime boundaries, CUDA, ANE, multimodal, audio, search, MoE, and unsafe-to-claim surfaces.

Yet it is only linked from `field-guide.html` as a raw Markdown document. This is a major information-design failure: the site’s best truth model is visually and structurally disconnected from the site.

### Required correction

Create `capabilities.html` as a first-class, data-driven status surface.

The Markdown file may remain as a generated or source document, but the HTML website must expose:

- capability domains;
- evidence level;
- build/features;
- tested hardware;
- receipt or fixture references;
- known limitations;
- source paths;
- last verified commit/date.

---

## 3.4 The roadmap is stale and narratively confused

`roadmap.html` still presents “Prism Agent for Mac” as the next release and labels Apple/MI300X/XDNA work in broad terms. It also includes an ADR-031 card titled “AITer Atom ROCm provider,” while the repository’s architectural evolution and current commit history indicate a wider canonical runtime and heterogeneous evaluation story.

The roadmap currently mixes:

- present validation;
- next product release;
- architecture decision records;
- aspirational product messaging.

### Required correction

Split the responsibilities:

- `status.html`: current support and validation matrix.
- `roadmap.html`: sequenced future milestones only.
- `decisions.html` or a living ADR index: architectural decisions and their implementation state.
- `demo.html`: one reproducible release artifact, not the entire release strategy.

The roadmap should be derived from explicitly maintained milestone data, not hand-authored prose embedded across multiple pages.

---

## 3.5 The Apple demo page understates what is executable today

The README documents a real CLI and local server workflow. The demo page still says the demo is “in preparation,” “not released,” and primarily a milestone definition.

This creates a credibility conflict:

- the repository says a supported model can be pulled, compiled, run, and served;
- the demo page says the executable path is not available.

Both can be true only if the page clearly distinguishes:

1. **developer CLI path available now**
2. **packaged, signed, self-contained demonstration application not yet released**

### Required correction

Rename and restructure the page:

- “Run Prism Today” — developer workflow;
- “Packaged Demo” — future release gate.

Include the exact current CLI commands from the README, generated or validated against the CLI help surface. Show prerequisites and define what evidence the run emits.

---

## 3.6 The website’s evidence model is mostly visual, not yet bound to repository evidence

The site repeatedly references receipts, proof, replay, and evidence. This is conceptually correct. But the static experience does not currently bind those concepts to:

- evidence schemas;
- real receipt examples;
- replay bundle examples;
- benchmark comparison outputs;
- baseline promotion;
- regression detection;
- commit/build identities;
- target qualification records.

The current site is therefore describing an evidence system without adequately demonstrating it.

### Required correction

Build an **Evidence Observatory** page that renders sanitized, versioned repository fixtures.

Minimum artifact set:

- compilation receipt;
- execution receipt;
- replay result;
- benchmark comparison;
- regression finding;
- hardware qualification record;
- failure receipt;
- provenance chain.

Every displayed artifact must state:

- real fixture or illustrative;
- schema version;
- producing command/test;
- commit;
- hardware context;
- validation scope;
- redactions, if any.

---

## 3.7 The “canonical object” idea is correct, but the implementation is still page-local and duplicated

The site repeatedly renders:

- living atlas markup;
- shared ComputeImage observation markup;
- header and footer markup;
- navigation markup;
- chapter metadata.

This produces drift and makes the conceptual claim of one canonical object less convincing at the implementation level.

### Required correction

Move repeated structures into declarative templates or custom elements:

- `<prism-header>`
- `<prism-living-atlas>`
- `<prism-computeimage-observation>`
- `<prism-status-strip>`
- `<prism-footer>`

For GitHub Pages compatibility, these can be hydrated by ES modules from minimal semantic fallback markup. Avoid introducing a heavy framework.

---

## 3.8 The CSS architecture is only partially consolidated

The site loads:

- `foundation.css`
- `components.css`
- `instruments.css`
- `pages.css`

But `pages.css` imports the historical site stylesheet and many page-specific stylesheets. This is a transitional layering system, not yet a clean design system.

Risks include:

- specificity conflicts;
- hidden import ordering;
- duplicated tokens;
- page-level components leaking globally;
- difficulty auditing unused CSS;
- runtime visual failures that appear to be DOM failures.

### Required correction

Create a real layered CSS contract:

```css
@layer reset, tokens, base, layout, components, instruments, pages, utilities;
```

Then:

- migrate all tokens into `tokens.css`;
- prohibit token declarations in page files;
- separate reusable component styles from page composition;
- remove `@import` chains in production and use explicit `<link>` ordering or a build-time concatenation script;
- introduce visual regression screenshots for every page and viewport.

---

## 3.9 The JS runtime is sophisticated relative to the site, but the product value is not yet visible

`app.js` now has a composition root and many systems. This is architecturally promising, but the user-facing experience does not clearly justify the amount of machinery.

The runtime currently risks becoming an internally elegant projection layer whose visible output remains ornamental.

### Required correction

Each runtime system must own an observable product behavior and a testable contract.

Examples:

- observation graph → coordinated highlighting across compiler, artifact, runtime, evidence;
- state projection → URL-addressable state and durable visitor context;
- canonical object → one subject identity across pages;
- repository service → commit/status data loaded from generated local JSON;
- receipt renderer → real fixture rendering;
- continuity service → continue reading and resume state;
- diagnostics → development overlay and CI export.

Delete or defer any system that cannot be tied to an observable behavior, accessibility requirement, or test.

---

## 3.10 The test matrix currently validates startup, not the website

The Playwright matrix checks portal creation, projection flags, shell count, navigation count, and runtime startup scenarios. This is useful, but insufficient.

It does not prove:

- content correctness;
- navigation integrity;
- no broken links;
- responsive layout;
- keyboard operation;
- reduced motion;
- contrast;
- visual consistency;
- fixture/schema validity;
- status freshness;
- claim/evidence consistency;
- no console errors across all pages.

### Required correction

Expand the test suite into five layers:

1. structural tests;
2. interaction tests;
3. accessibility tests;
4. visual regression tests;
5. content/evidence contract tests.

---

## 3.11 The site lacks a content authority pipeline

Current capability and status statements are manually repeated across:

- README;
- homepage;
- architecture;
- heterogeneous;
- roadmap;
- demo;
- current capabilities;
- partnership pages.

This guarantees drift.

### Required correction

Introduce one canonical dataset:

`docs/data/capabilities.json`

Suggested shape:

```json
{
  "generatedFromCommit": "...",
  "verifiedAt": "2026-07-24",
  "levels": ["implemented", "qualifying", "validated", "planned"],
  "capabilities": [
    {
      "id": "metal-runtime",
      "domain": "runtime",
      "label": "Apple Silicon / Metal",
      "level": "validated",
      "summary": "...",
      "sourcePaths": ["kernels/", "src/", "compute-core/"],
      "evidence": ["docs/evidence/metal-smoke.json"],
      "buildFeatures": ["full-apple", "prism-backend"],
      "limitations": ["..."]
    }
  ]
}
```

All status displays should render from this file.

---

## 4. Fully materialized product vision

The final site should feel like a **public systems observatory**, not a marketing microsite and not a raw documentation portal.

A visitor should be able to answer, within five minutes:

1. What is Prism?
2. What problem does it solve?
3. What can I run today?
4. What is a ComputeImage?
5. How does the constitutional ECS differ from a conventional runtime?
6. What hardware paths exist?
7. Which paths are implemented, qualifying, validated, or planned?
8. What evidence does Prism preserve?
9. Where is the source that implements each claim?
10. How can I reproduce a run or collaborate?

The canonical journey should be:

```text
Model source
→ semantic graph
→ representation candidates
→ target and capability constraints
→ admitted plan
→ ComputeImage
→ constitutional work transaction
→ residency and dispatch
→ output
→ receipt
→ replay / comparison / promotion
```

Every main page should be a different instrument observing this same journey.

---

## 5. Target information architecture

## 5.1 Core pages

### `index.html` — Observatory overview

Purpose:

- identify Prism in one sentence;
- show the canonical journey;
- show current support state;
- route visitors by intent;
- introduce the central object and evidence contract.

Required modules:

- operational hero;
- status strip;
- canonical deployment journey;
- “why not a runtime?” comparison;
- ComputeImage specimen;
- constitutional ECS panel;
- evidence loop;
- capability highlights;
- run-now CTA;
- source CTA.

### `compiler.html`

Purpose:

- explain ECS-native compilation;
- representation search;
- progressive quantization and ternarization;
- mixed precision;
- calibration and admission;
- spatial planning;
- model-specific paths, including MoE.

### `computeimage.html`

Purpose:

- make the artifact contract concrete;
- render a real or sanitized manifest;
- show logical tensors, physical layouts, execution views, plans, payloads, and receipts;
- link to the ABI.

### `runtime.html`

Purpose:

- explain canonical state, work transactions, leases, residency, KV ownership, dispatch, provider boundaries, and failure handling;
- separate canonical state from ephemeral backend handles.

### `evidence.html`

Purpose:

- show receipts, provenance, replay, experiments, comparisons, regressions, baselines, and recovery;
- make Prism’s strongest differentiator executable in the reader’s mind.

### `capabilities.html`

Purpose:

- authoritative support matrix;
- implemented / qualifying / validated / planned;
- current commit and verification date;
- source and evidence links.

### `run.html`

Purpose:

- current CLI and server path;
- prerequisites;
- commands;
- expected artifact locations;
- how to inspect generated outputs;
- packaged demo status as a separate section.

### `roadmap.html`

Purpose:

- future milestones only;
- milestone dependencies;
- exit criteria;
- no current-state claims duplicated here.

## 5.2 Secondary pages

- `architecture.html` may remain as a long-form integrated deep dive, but it should be generated from or aligned with the new core pages.
- `field-guide.html` becomes documentation routing.
- `prism-ml.html` remains a partnership/research boundary page.
- `general-compute.html` should be labeled clearly as a collaboration hypothesis or provider case study.
- `work-with-prism.html` remains the commercial/collaboration surface.
- `decisions.html` indexes ADRs and their implementation status.
- `models.html` describes supported/qualifying model families.
- `multimodal.html` describes image/audio pipeline status and evidence boundaries.

---

## 6. Data and content architecture

## 6.1 Canonical data files

Create:

```text
docs/data/
  site.json
  navigation.json
  capabilities.json
  roadmap.json
  architecture.json
  evidence-index.json
  models.json
  releases.json
```

### `site.json`

- project name;
- current version;
- repository URL;
- default branch;
- license;
- contact;
- verification commit;
- verification date.

### `capabilities.json`

- domain;
- status level;
- summary;
- source paths;
- feature flags;
- hardware;
- evidence;
- limitations;
- last verified commit.

### `architecture.json`

Defines canonical nodes and edges:

- model;
- compiler;
- representation;
- spatial plan;
- ComputeImage;
- scheduler;
- provider;
- residency;
- execution;
- evidence;
- replay.

The living atlas and all architecture instruments should consume this one graph.

### `evidence-index.json`

Index of fixtures:

- type;
- schema;
- path;
- producer;
- commit;
- target;
- classification;
- explanatory copy.

---

## 7. Evidence classification contract

Adopt one vocabulary everywhere.

### Planned

Architecture, ADR, or intended surface exists. End-to-end implementation is incomplete.

### Implemented

Code path, data structure, command, or provider boundary exists.

### Qualifying

Implementation has tests, fixtures, or compile validation, but target-specific or end-to-end evidence is incomplete.

### Validated

A defined build and hardware/software configuration has passed the documented conformance and receipt gates.

### Released

A versioned artifact is published with reproducible installation and support boundaries.

No page may use “supported,” “ready,” “active,” or “available” without mapping the term to one of these levels.

---

## 8. Visual system completion

## 8.1 Preserve

- dark technical atmosphere;
- optical / prism metaphor;
- monospaced evidence labels;
- restrained gradients;
- instrument-panel composition;
- canonical ComputeImage specimen;
- chapter orientation and reading progress;
- reduced-motion behavior.

## 8.2 Correct

### Reduce ornamental duplication

The living atlas currently appears on many pages with nearly identical markup. It should be one reusable instrument whose active node and explanation change by page context.

### Establish a visual hierarchy of truth

Use visual states consistently:

- validated — solid, high-contrast;
- qualifying — striped or partially illuminated;
- implemented — outlined;
- planned — ghosted;
- failed / blocked — explicit warning state.

Never use the same glowing “active” treatment for planned and measured claims.

### Add evidence texture

Real artifacts should look materially different from illustrations:

- schema/version;
- digest;
- commit;
- hardware;
- timestamp;
- producer;
- classification badge.

### Improve density control

Provide three content densities:

- overview;
- technical detail;
- raw artifact.

Use disclosure controls rather than putting every explanation into the primary flow.

---

## 9. Runtime architecture completion

## 9.1 Keep the composition boundary

Retain `docs/js/app.js` as the composition root.

## 9.2 Introduce explicit runtime phases

```text
bootstrap
→ config
→ repository snapshot
→ canonical graph
→ DOM ownership
→ instruments
→ interactions
→ diagnostics
```

Each phase should produce a diagnostic event and have a failure policy.

## 9.3 Separate production and diagnostics bundles

Production should not expose unnecessary global diagnostics or capture large mutation timelines by default.

Suggested query modes:

- `?prismDiagnostics=on`
- `?prismDebugOverlay=on`
- `?prismRuntime=off`

## 9.4 Replace selector ownership with component ownership where possible

Instead of systems claiming broad selector strings, mount explicit custom elements or registered component roots. This reduces ownership ambiguity and DOM projection failures.

## 9.5 Add a static fallback contract

Every page must remain readable and navigable with:

- JS disabled;
- WebGL/GPU effects disabled;
- reduced motion;
- high contrast;
- narrow viewport.

The runtime should enhance, not create, core meaning.

---

## 10. Build and synchronization pipeline

Create a repository-local script:

```text
scripts/build-site-data.mjs
```

Responsibilities:

1. read README and selected machine-readable repository metadata;
2. read capability source configuration;
3. validate source paths;
4. validate evidence fixture paths;
5. stamp current commit;
6. emit `docs/data/*.json`;
7. fail on stale or contradictory status entries.

Optional later phase:

- parse Cargo metadata for crates/features;
- inspect CLI help output;
- validate ADR status;
- validate schema versions.

The site remains deployable as static files on GitHub Pages.

---

## 11. Testing strategy

## 11.1 Structural

- all HTML pages load;
- required landmarks exist;
- one H1 per page;
- navigation and footer present;
- no duplicate IDs;
- canonical subject IDs are valid;
- all internal links resolve.

## 11.2 Runtime

- all systems start in expected modes;
- no ownership conflicts;
- runtime-off mode remains functional;
- optional-system failures degrade gracefully;
- hard-abort behaves deterministically;
- no detached roots;
- no duplicate portal/effect shells.

## 11.3 Accessibility

Use Playwright plus axe:

- keyboard navigation;
- focus visibility;
- menu semantics;
- disclosure controls;
- live-region restraint;
- reduced motion;
- color contrast;
- 200% zoom;
- screen-reader labels on diagrams;
- no meaning conveyed only by animation or color.

## 11.4 Visual regression

Capture every core page at:

- 390×844;
- 768×1024;
- 1440×1000;
- reduced motion;
- light/high-contrast fallback if supported.

## 11.5 Content and evidence

- every capability has a valid level;
- every “validated” capability has evidence;
- every evidence file has schema, commit, target, producer;
- every source path exists;
- no page contains forbidden unsupported status language;
- displayed commit equals generated site-data commit;
- roadmap entries cannot be marked validated.

---

## 12. Implementation milestones

## Milestone 0 — Freeze the truth model

**Goal:** Establish the audit baseline and stop content drift.

Tasks:

- record audited commit;
- define status vocabulary;
- create `capabilities.json`;
- map every current claim to source paths and evidence;
- classify every hardware and product surface;
- identify stale pages and contradictory language.

Exit criteria:

- one capability matrix exists;
- no unresolved disagreement between README, current-capabilities, roadmap, and core pages;
- every public claim has an owner and classification.

---

## Milestone 1 — Rebuild information architecture

**Goal:** Make the site’s structure match the current repository.

Tasks:

- define final navigation;
- create page-purpose contracts;
- split status, roadmap, evidence, and run-now content;
- decide which legacy pages redirect, remain, or become case studies;
- update continue-reading sequence.

Exit criteria:

- every major repository domain has a public home;
- roadmap no longer acts as evidence;
- developer runnable path is not confused with packaged demo status.

---

## Milestone 2 — Build the canonical data layer

**Goal:** Remove hand-maintained status duplication.

Tasks:

- add `/docs/data`;
- implement site, capabilities, architecture, roadmap, evidence, and release schemas;
- implement build-time validators;
- stamp commit and verification date;
- create fixture index.

Exit criteria:

- status strips and capability views render from data;
- source/evidence references are validated;
- stale verification causes CI failure.

---

## Milestone 3 — Materialize the core pages

**Goal:** Deliver the complete product narrative.

Order:

1. `index.html`
2. `capabilities.html`
3. `evidence.html`
4. `run.html`
5. `compiler.html`
6. `computeimage.html`
7. `runtime.html`
8. revised `roadmap.html`

Exit criteria:

- a new visitor can understand and reproduce the current path;
- all capability levels are visible;
- real evidence fixtures appear;
- constitutional ECS is a first-class differentiator.

---

## Milestone 4 — Consolidate shared components

**Goal:** Eliminate repeated site shell and instrument markup.

Tasks:

- implement header, footer, atlas, ComputeImage observation, status strip, evidence badge;
- migrate duplicated markup;
- ensure semantic static fallback;
- define component ownership.

Exit criteria:

- no manually duplicated living atlas;
- no manually duplicated primary navigation;
- no manually duplicated shared ComputeImage observation;
- page context is passed declaratively.

---

## Milestone 5 — Complete the design system

**Goal:** Convert the transitional CSS stack into a stable system.

Tasks:

- centralize tokens;
- formalize CSS layers;
- split layout/components/instruments/pages;
- remove hidden import-order dependencies;
- audit unused selectors;
- normalize status colors and evidence treatments;
- implement responsive density behavior.

Exit criteria:

- visual snapshots stable;
- no page-specific selector leaks;
- all status levels use one visual grammar;
- reduced-motion and no-JS modes remain legible.

---

## Milestone 6 — Bind the Observatory to real evidence

**Goal:** Make proof more than a metaphor.

Tasks:

- select sanitized repository fixtures;
- render compilation and execution receipts;
- render replay and comparison artifacts;
- add provenance traces;
- add real failure examples;
- link each artifact to source schema and producing test/command.

Exit criteria:

- at least one real artifact for each evidence class;
- every artifact states scope and provenance;
- no illustrative artifact can be mistaken for measured evidence.

---

## Milestone 7 — Harden the runtime

**Goal:** Make the enhancement layer deterministic and maintainable.

Tasks:

- define startup phases;
- reduce selector-based ownership;
- separate production diagnostics;
- enforce mount/unmount contracts;
- eliminate non-visible systems;
- add error overlay in diagnostics mode;
- verify navigation and history behavior.

Exit criteria:

- startup matrix passes;
- all pages produce zero console errors;
- runtime-off mode passes;
- optional features fail soft;
- no DOM ownership conflict exists.

---

## Milestone 8 — Expand CI and release gates

**Goal:** Make website correctness enforceable.

Tasks:

- link checker;
- schema validator;
- content claim linter;
- axe accessibility suite;
- visual regression suite;
- performance budgets;
- GitHub Pages deployment smoke test;
- generated commit freshness check.

Exit criteria:

- one command validates the complete site;
- release cannot proceed with stale capability data;
- accessibility and visual baselines are versioned.

---

## Milestone 9 — Final editorial and launch pass

**Goal:** Remove residual ambiguity and polish the public release.

Tasks:

- technical copy edit;
- remove repeated claims;
- standardize terminology;
- validate command examples;
- verify email and source links;
- add social metadata and canonical URLs;
- add sitemap, robots, favicon, and share image;
- run external-user comprehension test.

Exit criteria:

- no contradictory status language;
- no broken link or raw orphan page;
- first-time reader can explain Prism accurately;
- technical reader can locate code and evidence;
- collaboration visitor can reach a concrete engagement path.

---

## 13. Recommended issue breakdown

### Epic A — Truth and data

- Define capability classification schema
- Audit all public claims
- Add generated verification metadata
- Add source/evidence validators
- Add content-status linter

### Epic B — Core information architecture

- Replace global navigation
- Add capabilities page
- Add evidence page
- Add run-now page
- Split roadmap and status
- Add ADR index

### Epic C — Canonical components

- Header component
- Footer component
- Living atlas component
- ComputeImage observation component
- Evidence artifact renderer
- Capability status component

### Epic D — Runtime hardening

- Startup phase model
- Explicit ownership roots
- Production/diagnostic mode split
- no-JS fallback validation
- deterministic teardown
- error-state instrumentation

### Epic E — Quality gates

- Link integrity
- Accessibility suite
- Visual regression
- Status freshness
- Evidence fixture validation
- Performance budget

---

## 14. Priority sequence for the next implementation session

The highest-value order is:

1. create the capability/status schema;
2. build `capabilities.html`;
3. correct the homepage hero and status strip;
4. replace the roadmap-as-evidence navigation error;
5. create `run.html` from the real README workflow;
6. create `evidence.html` with real fixtures;
7. consolidate shared atlas and ComputeImage markup;
8. expand Playwright from startup checks to whole-site checks.

Do not begin with another visual-effects pass. The visual language is already sufficiently distinctive. The next unit of value is **truthful materialization**.

---

## 15. Definition of done

The website is fully materialized when:

- it reflects the audited repository commit;
- every major codebase capability has a visible and classified public representation;
- every validated claim has linked evidence;
- every qualifying claim states its missing boundary;
- the runnable developer path is documented and reproducible;
- packaged demo status is distinct from CLI availability;
- the constitutional ECS and evidence/replay model are first-class;
- all instruments observe one canonical architecture graph;
- shared components are not duplicated across pages;
- the site works without JavaScript;
- the enhancement runtime starts deterministically;
- accessibility, visual, link, schema, and claim tests pass;
- the site cannot silently drift from repository reality.

---

## 16. Final recommendation

Treat the next website release as:

> **Prism Observatory v1 — the evidence-bound public projection of Prism Engine.**

That framing unifies the site’s strongest existing visual idea with the codebase’s strongest current technical reality. It also creates a durable standard: as the engine changes, the observatory updates its canonical graph, capability state, artifacts, and evidence instead of requiring another narrative rewrite.
