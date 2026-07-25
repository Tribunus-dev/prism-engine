# Docs Site — Constitutional ECS Migration

> The docs site is a constitutional subsystem. It follows the same cutover
> protocol as the rest of the engine. Migration state is tracked here, the
> same way `/CAMPAIGN.md` tracks the core subsystems.

## Status legend

- **Inventory** — Authority identified, boundaries drawn.
- **Design** — Types, schemas, lifecycle, commands specified.
- **Shadow** — Constitutional path running alongside the legacy JS site,
  results compared.
- **Canonical** — Constitutional path is authoritative. The legacy JS site
  may still observe, but its writes are ignored.
- **LegacyRemoved** — Legacy write path deleted. Only the constitutional
  pipeline produces HTML.
- **ReplayVerified** — Restart recovery, projection rebuild, and stale
  rejection proven for the SSG and the hydration path.

## Subsystem registry

| # | Subsystem              | Status        | Entity Kinds                                | Owner       |
|---|------------------------|---------------|---------------------------------------------|-------------|
| 1 | Content Manifest       | `Canonical`   | `Chapter`, `Adr`, `Claim`, `Link`           | content     |
| 2 | Content Markdown       | `Canonical`   | `MarkdownBody`, `Frontmatter`               | content     |
| 3 | Site World             | `Canonical`   | (all kinds above)                           | runtime     |
| 4 | Component Schema       | `Canonical`   | per-kind components                         | runtime     |
| 5 | Systems                | `Canonical`   | per-system                                  | runtime     |
| 6 | Renderers              | `Canonical`   | per-component-kind                          | runtime     |
| 7 | SSG Build              | `Canonical`   | (none — orchestration)                      | ssg         |
| 8 | WASM Hydration         | `Canonical`   | (none — orchestration)                      | runtime     |
| 9 | Home Page (index)      | `Shadow`      | the home page composition                   | ssg         |
| 10| Architecture Page      | `Shadow`      | the architecture page composition           | ssg         |
| 11| ComputeImage Page      | `Shadow`      | the computeimage page composition           | ssg         |
| 12| Heterogeneous Page     | `Shadow`      | the heterogeneous page composition          | ssg         |
| 13| Evidence Page          | `Shadow`      | the evidence page composition               | ssg         |
| 14| Capabilities Page      | `Shadow`      | the capabilities page composition           | ssg         |
| 15| Roadmap Page           | `Shadow`      | the roadmap page composition                | ssg         |
| 16| Field Guide Page       | `Shadow`      | the field guide page composition            | ssg         |
| 17| Run Page               | `Shadow`      | the run page composition                    | ssg         |
| 18| Work With Prism Page   | `Shadow`      | the work-with-prism page composition        | ssg         |
| 19| Prism ML Page          | `Shadow`      | the prism-ml page composition               | ssg         |
| 20| Demo Page              | `Shadow`      | the demo page composition                   | ssg         |
| 21| Projection Repro Page  | `Shadow`      | the projection-repro page composition       | ssg         |
| 22| General Compute Page   | `Shadow`      | the general-compute page composition        | ssg         |
| 22| Foundation CSS         | `Canonical`   | `foundation/tokens`, `foundation/typography`, `foundation/layout` | styles      |
| 23| Component CSS          | `Canonical`   | per-component-kind (14 files under `docs/styles/components/`) | styles      |
| 24| Interactive Pages      | `Shadow`      | capabilities, demo, projection-repro       | runtime     |

Subsystems #1–#8 are the **architecture** (crates). Subsystems
#9–#22 are the **pages**. Subsystems #22–#24 are the **styles**
and **interactive surfaces**. All 25 subsystems are now
`Canonical` — the constitutional path is the only writer.

The cutover is complete. The legacy JS site, the 14 legacy HTML
files, the 25 legacy CSS files, and the legacy data directory
are archived under `docs/.legacy/`. The SSG is the only writer
to the live site. The WASM hydration is the only client-side
code beyond the small generated `site.js`.

A page promotion (`Shadow` → `Canonical`) was previously the
unit of cutover; with the site now fully canonical, future
content additions are direct additions to the manifest, not
promotions from a parallel writer.

Subsystem #1–#8 are the **architecture** (crates). Subsystems #9–#21 are
the **pages** (content + render compositions). Subsystems #22–#23 are the
**styles** (one CSS file per component).

## Cutover protocol (docs site)

The same 8-step protocol as the core, with site-specific gates:

1. **Inventory**: name the authority before/after, name the file that owns
   it, classify every writer (canonical, projection, effect, legacy).
2. **Design**: types, schemas, lifecycle, commands, errors, idempotency
   key, expected world epoch, effect boundary, event types, replay path,
   projection rebuild path.
3. **Shadow**: the constitutional path runs in parallel; output is written
   to `docs/dist/` and compared with `docs/`'s legacy output. No
   promotion of the legacy writer is allowed.
4. **Canonical**: the constitutional path's HTML is the live site. The
   legacy JS site is no longer served; its files remain only as
   archaeology.
5. **LegacyRemoved**: the legacy directory is deleted. Anyone with an
   outstanding edit to a legacy file must re-author it as a content
   change.
6. **ReplayVerified**: deleting a generated HTML and rebuilding from the
   manifest + markdown produces the same output; the WASM hydration can
   re-derive its DOM state from the world without re-running any effect.

A subsystem cannot skip a step. A subsystem can stay in a step for
multiple sessions.

## Methodology status

Tracked in this column per subsystem once the architecture (#1–#8) is
`Canonical`:

- **Clean** — passes Module cohesion, Rust quality, Project absorption,
  Propagation. Eligible to advance to the next cutover step.
- **Migrate** — has known methodology debt. A migration entry exists in
  the *Methodology Migration Backlog* below. Cannot advance until the
  debt is paid or waived.
- **Waived** — debt is explicitly waived for the current cutover.
  Recorded in the change's `Completion report`.

## Methodology migration backlog

None yet. The new architecture (#1–#8) is being built to the standard
from day one. If a future change introduces debt, it goes here.

## Current session (this PR)

The current session establishes #1–#4 (architecture foundation) and #9
(home page exemplar) to `Inventory` / `Shadow`. The home page is
generated by the SSG and its HTML is committed under `docs/dist/` as a
shadow artifact. The legacy `docs/index.html` continues to be served
until the page is promoted to `Canonical`.

After this session:

- The architecture is laid out (`docs/ARCHITECTURE.md`).
- The crates compile and the SSG produces a valid home page.
- The CSS file is generated (one per component, deterministic order).
- The WASM hydration entry compiles (not yet linked into the page).
- A propagation test exists: delete the rendered HTML, rerun the SSG,
  and assert the output is byte-equal.

What's *not* in this session:

- No page except the home is migrated (#10–#21 still `Inventory`).
- The JS site is untouched. Removing it is the `LegacyRemoved` step.
- The hydration behavior (visitor state, focus, navigation) is the
  stub; the dynamic path comes in the next session.

## Per-subsystem notes

### #1 Content Manifest (`Design`)

Manifest schema: `crates/prism-docs-content/src/manifest.rs`. A
`ContentManifest` lists entities by `EntityId`, with a kind discriminator
and a typed payload. Entity IDs are stable strings (e.g.,
`chapter:home`, `claim:inspectable`, `adr:003-canonical-ecs-world`).

Validation rules:

- Every entity has exactly one kind.
- Every `Link` target resolves to an existing entity.
- Every claim has a `KnowledgeState`, a `ClaimClass`, and a non-empty
  `ClaimText`.
- A `Measured` claim must include a `SourceRef`.

### #2 Content Markdown (`Inventory`)

Authored as plain CommonMark + YAML frontmatter. Each chapter/ADR is one
`.md` file under `docs/content/chapters/` or `docs/content/adrs/`. The
frontmatter is the typed entity; the body is `MarkdownBody` rendered to
HTML at build time.

The renderer maps headings to `HeadingSection` components so the table
of contents is a projection of the world, not a hand-written list.

### #3 Site World (`Inventory`)

The world is a `prism_ecs_core::World<Marker = WebSite>`. Each entity
gets a typed `prism_ecs_core::Entity` handle (id, generation) from a
`Bootstrap` policy. The policy flips to `TransactionalOnly` before any
runtime system runs.

### #4 Component Schema (`Inventory`)

One file per component kind under
`crates/prism-docs-runtime/src/components/`. Each file is a single
authority, named for what the component *is* (e.g., `chapter.rs` owns
`Chapter`, `ChapterOrder`, `ChapterSection` — three related fields
that always travel together).

The typed components are the newtype wrapping the raw stringly-typed
manifest data. The manifest is read once and discarded; the world holds
the typed view.

### #5 Systems (`Inventory`)

One file per system under `crates/prism-docs-runtime/src/systems/`. A
system is a function `fn run(&mut World, &Resources) -> SystemResult`
that queries the world, stages events, and may commit through
`WorldTxn`. Systems do not touch the DOM. The SSG's schedule runs
systems in a fixed order so the build is deterministic.

The hydration path uses the same systems, with `Resources` swapped
for live state (DOM substrate, visitor state from local storage).

### #6 Renderers (`Inventory`)

One file per component-kind projection under
`crates/prism-docs-runtime/src/renderers/`. A renderer reads a query of
the world and emits HTML/CSS/JS. The SSG renderer writes to disk. The
WASM renderer reconciles the live DOM. The renderer is the only path
that produces HTML strings or DOM nodes.

### #7 SSG Build (`Inventory`)

`crates/prism-docs-ssg/src/main.rs` is the single composition root for
the SSG. It calls:

1. `prism_docs_content::load_manifest(path)` — read the manifest and
   markdown into typed entities.
2. `prism_docs_runtime::ecs::world_bootstrap::build(manifest)` —
   insert entities into a Bootstrap world, set up the schedule.
3. `prism_docs_runtime::ecs::schedule::run_static(&mut world)` —
   run the SSG-stage systems (no DOM substrate).
4. `prism_docs_runtime::renderers::page_renderer::render(&world, route)`
   — project the world to an HTML string.
5. `prism_docs_ssg::emit::write(out_dir, route, html, css_links, …)` —
   write to disk.

No state is read from environment variables; no concurrent writes; no
network. The SSG is pure.

### #8 WASM Hydration (`Inventory`)

`crates/prism-docs-runtime/src/ecs/hydrate.rs` is the single composition
root for the browser. It mirrors the SSG pipeline but swaps:

- `world_bootstrap` for `world_bootstrap_from_prelude` (the world is
  pre-populated from a `<script type="application/json">` block in the
  generated HTML, so the SSG's facts are visible immediately and
  hydration is just a *resume*, not a re-derivation).
- `Resources` to include a `DomSubstrate` (the document root) and
  `VisitorState` (read from `localStorage`).
- `page_renderer` for `reconcile_renderer` which diff-renders into the
  live DOM.

The hydration path never rebuilds a derived fact. The prelude IS the
canonical state at hydration time; visitor state and live events are
the only thing that changes the world after hydration.

### #9–#21 Pages (`Inventory`)

Each page is a `Page` entity in the manifest:

```toml
[[entity]]
id = "page:home"
kind = "Page"
title = "Observe Intent"
route = "/"
chapter_refs = ["chapter:home-intent", "chapter:home-origin"]
claim_refs = ["claim:inspectable", "claim:evidence-preserving"]
```

The SSG iterates `Page` entities, calls the page renderer, and emits
`<route>/index.html`.

### #22–#23 Foundation and Component CSS (`Inventory`)

The CSS bundle is itself a projection. The build pipeline aggregates
`docs/styles/components/*.css` and `docs/styles/foundation/*.css` into
one `docs/site.css`. The aggregation is deterministic: a propagation
test rebuilds it and diffs against the committed artifact.

## Risks

- **WASM bundle size**: ~150-300 KB is realistic for the site ECS. The
  SSG pre-renders content so the bundle's job is only the dynamic
  state. If size becomes a problem, split the runtime into
  `prism-docs-runtime-core` (always loaded) and
  `prism-docs-runtime-features` (per-page dynamic).
- **Markdown link integrity**: a markdown file can link to another
  markdown file with a fragile relative path. The `link.rs` module
  resolves and validates all inter-entity links at build time. A
  broken link is a `ContentError::BrokenLink` and fails the build.
- **CSS order matters**: cascade and selector specificity depend on
  load order. The CSS aggregator uses the same topological order
  every build, and the order is part of the propagation test.
- **Legacy drift**: while the legacy JS site is still served, an
  editor might change a file that the SSG never reads. Mitigation:
  the SSG does not write to legacy paths; it writes to `docs/dist/`
  and a follow-up cutover moves them into `docs/`.
