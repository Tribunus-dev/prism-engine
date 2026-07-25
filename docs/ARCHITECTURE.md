# Prism Engine Docs — Constitutional ECS Architecture

> **One canonical reality.** Every piece of documentation is a typed entity.
> Every visible change is a transactional mutation through `WorldTxn`. Every
> pixel on the page is a projection of the world. The docs site is not "built
> on the engine" — it *is* an engine, with the same authority model as the
> core.

This document replaces the previous ad-hoc `js/core`, `js/systems`, `js/runtime`,
`js/renderers` JS directories. The old vocabulary (`CLAIM_CLASSES`,
`KNOWLEDGE_STATES`, `OBSERVATION_KINDS`, `canonicalSubject`, `projection`, etc.)
is preserved — but it now lives as **typed Rust components** on a real
`prism-ecs-core::World`, with the same constitutional discipline the core
enforces on the rest of the engine.

## Prime directive (docs site)

Before changing content, identify which entity currently has authority over the
affected fact. After the change, exactly one entity may remain authoritative.

A duplicate manifest entry, an inline-styled `<div>`, a hand-typed HTML
section, a JS file that mutates the DOM directly, a CSS rule that hides
content the world did not suppress — any of these is a parallel authority and
fails review. The site has one canonical world, and the world has one
canonical source: `docs/content/`.

## The hard rules (docs site)

These are the same rules as the core, restated for the site. A change that
violates any of them fails review.

- **All page content is declared as entities.** Every chapter, ADR, claim,
  section, link, navigation item, and component is an entity in the world.
  HTML is rendered from the world. There is no content that exists only as
  HTML.

- **No direct DOM mutation outside renderers.** A renderer is a projection
  that reads the world and reconciles a region of the DOM. A system may
  stage events; it may not call `document.createElement` or
  `element.set_attribute` directly. The WASM hydration is a renderer; the
  SSG is a renderer; nothing else may touch the DOM.

- **All mutations go through `WorldTxn`.** During SSG bootstrap, the world
  uses `MutationPolicy::Bootstrap` so the content loader can populate the
  world atomically. After bootstrap, the policy flips to
  `TransactionalOnly` and the only writers are the hydration-time systems
  (e.g., visitor-state-system). There is no direct `world.add_component` on
  the hot path.

- **No inline styles. No `<style>` blocks per page.** Every visual
  difference is a component class in `docs/styles/components/`. Every
  component has a corresponding CSS file. The HTML and the CSS are both
  projections of the same component definition.

- **No JS file mutates the world or the DOM outside the composition root.**
  `crates/prism-docs-ssg/src/main.rs` and
  `crates/prism-docs-runtime/src/hydrate.rs` are the only composition roots.
  No `app.js` stands in for them.

- **No content duplicates.** If the same fact appears in two places (e.g.,
  the same claim in a chapter and on the home page), it is one entity, two
  renderers. Renderers are derived projections; they may be rebuilt at any
  time without touching the world.

- **Every new file states a single authority in its module doc.** Same
  one-sentence test as the core. The file owns exactly one thing.

## Crate map (docs site)

The docs site is a Cargo workspace of three crates. None of them depends on
the product or backend crates. `prism-docs-runtime` depends on
`prism-ecs-core`; the others depend on `prism-docs-runtime` and each other.

```
crates/
  prism-docs-content/    Content schema, manifest, markdown parser.
                         Depends on: serde, serde_json, toml, thiserror,
                         pulldown-cmark. No ECS dependency.
  prism-docs-runtime/    Site ECS: typed components, systems, renderers,
                         hydration entry. Depends on: prism-ecs-core,
                         web-sys, wasm-bindgen (under wasm feature).
  prism-docs-ssg/        SSG binary. Loads content, builds the world,
                         runs the schedule, emits static HTML. Depends
                         on: prism-docs-content, prism-docs-runtime.
```

The directory under `docs/` is the *output* of the build, not the source of
truth. The source of truth is `crates/prism-docs-content` and the
`docs/content/` declarative registry. If you find yourself editing
`docs/index.html` directly, you are creating a parallel authority. Edit the
content instead and rebuild.

## The canonical change flow (docs site)

`Content change -> update docs/content/* or crates/prism-docs-content types
-> cargo run -p prism-docs-ssg -> SSG loads content into a Bootstrap world
-> flip policy to TransactionalOnly -> run schedule (systems) -> renderers
project to disk as HTML/CSS/JS (WASM bundle) -> git diff docs/*.html shows
exactly the rendered change -> cargo test verifies projection + replay
rebuilds deterministically`

For a *runtime* change (a hydration behavior, a system that reacts to
visitor state, a claim validator), the flow is:

`Hydration entry -> build world from manifest (Bootstrap) -> flip to
TransactionalOnly -> systems query the world and stage WorldTxn commands
-> WorldTxn commits -> renderers reconcile the live DOM region by reading
the world -> the DOM is never written by anything but the renderer`

External work — fetching visitor state, persisting preferences, talking to
a backend — is an effect that returns a value, not a world mutation. The
result lands back in the world through a typed command. There is no other
writer.

## Module discipline (docs site)

Same rules as the core. One authority per file. 600 LOC soft limit, 900
LOC hard limit, 20 public items soft, 35 hard. Module doc states the
single authority. No `common.rs`, `utils.rs`, `helpers.rs`, `misc.rs`,
`shared.rs`, `manager.rs`, `coordinator.rs`, `controller.rs`, `service.rs`,
`facade.rs`. No `mod.rs` over 200 LOC.

For the docs site specifically, the following names are also prohibited
because they describe presentation concerns, not authority:

- `page.rs` — pages are not an authority; they are a renderer composition.
- `template.rs` — templates describe rendering, not state.
- `view.rs` — views are projections; the file should be a renderer named
  for what it projects (`chapter_renderer`, `claim_renderer`).

## File layout

```
docs/
  ARCHITECTURE.md          this file
  CAMPAIGN.md              migration state for the docs site
  content/                 canonical content source (declarative)
    manifest.toml          entity registry, kinds, links
    chapters/              one markdown file per chapter
    adrs/                  one markdown file per ADR
    claims.toml            claim entities
  styles/                  static CSS, one file per component
    foundation/            base tokens, typography, layout primitives
    components/            one file per component kind (title, claim, chapter, …)
    system/                system-driven dynamic CSS
  scripts/                 build orchestration (just calls cargo, currently)
  index.html               GENERATED — do not edit
  *.html                   GENERATED — do not edit
  pkg/                     GENERATED wasm-bindgen output

crates/
  prism-docs-content/src/
    lib.rs                 module index, error type, public exports
    manifest.rs            ContentManifest, EntityRef, KindRegistry
    chapter.rs             Chapter entity data, parse, validate
    adr.rs                 Adr entity data, parse, validate
    claim.rs               Claim entity data, validate (claim rules)
    source_ref.rs          SourceRef (file + section + line)
    markdown.rs            frontmatter + body parser (pulldown-cmark)
    link.rs                typed links: chapter->adr, claim->chapter, etc.
    error.rs               typed errors (one enum per authority)
  prism-docs-runtime/src/
    lib.rs                 module index, public exports
    components/            one file per component kind
      chapter.rs           Chapter, ChapterOrder, ChapterSection
      adr.rs               Adr, AdrNumber, AdrStatus
      claim.rs             Claim, ClaimClass, ClaimText
      link.rs              ChapterLink, AdrLink, ClaimRef
      ontology.rs          KnowledgeState, ExistenceState, ClaimClass
      observer.rs          ObserverMode, OpticalState
      body.rs              MarkdownBody (the rendered HTML body)
    resources/             one file per singleton
      visitor_state.rs     current ObserverMode, OpticalState
      site_config.rs       build metadata, site title
      dom_substrate.rs      live DOM handle (wasm only)
    systems/               one file per system
      chapter_presentation_system.rs  orders chapters for the route
      claim_validation_system.rs      validates claim refs
      nav_projection_system.rs        projects world -> nav structure
      visitor_state_system.rs         hydrates visitor state (wasm)
      canonical_focus_system.rs       resolves the canonical focus
    renderers/             one file per projection
      chapter_renderer.rs   projects chapter entities -> HTML
      claim_renderer.rs     projects claim entities -> HTML
      nav_renderer.rs       projects nav -> HTML
      hero_renderer.rs      projects hero -> HTML
      page_renderer.rs      composes a full page from renderers
    ecs/                    ECS glue over prism-ecs-core
      world_bootstrap.rs    manifest -> world, policy transitions
      schedule.rs           ordered system execution
      render_coordinator.rs renderer scheduling
      reconcile.rs          projection -> DOM diff
      hydrate.rs            wasm hydration entry
    error.rs                runtime errors (one enum per authority)
  prism-docs-ssg/src/
    main.rs                 SSG composition root (one file, one authority)
    page.rs                 one page composition (calls renderers)
    emit.rs                 write to disk
    templates/              HTML skeleton, not "templates" in the bad sense
      shell.html            page shell
      head.html             <head> section
```

The new directories replace `docs/js/`. The new files replace `docs/css/` and
`docs/*.html`. There is no migration shim — old files are removed when the
new pipeline emits their replacement.

## The hard rules restated for CSS and HTML

- One CSS file per component kind. The component file lives in
  `docs/styles/components/<kind>.css` and the renderer emits the matching
  `class="<kind> ..."`. Two component kinds in one CSS file is two
  authorities and fails review.
- HTML is generated. The only HTML files you may edit by hand are
  `crates/prism-docs-ssg/src/templates/*.html`, and even those are
  static skeletons, not content.
- The site never serves inline styles. If you need a one-off visual
  treatment, it's a new component, not a `style="..."` attribute.

## The build

```
cargo run -p prism-docs-ssg --release -- --content docs/content --out docs
```

Reads the manifest and markdown, builds the world, runs the schedule,
projects to `docs/`. The WASM bundle is built separately for hydration:

```
cargo build -p prism-docs-runtime --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/prism_docs_runtime.wasm \
  --out-dir docs/pkg --target web
```

The shell page loads `docs/pkg/prism_docs_runtime.js`, which rehydrates the
dynamic state (visitor mode, navigation focus, claim selection).

For now, while we migrate, both pipelines can coexist: the old JS site
serves visitors, the new pipeline builds into `docs/dist/` until the
migration is complete. See `docs/CAMPAIGN.md` for the current state.

## Why Rust + WASM, not a JS port

The previous JS architecture borrowed the vocabulary of the constitutional
ECS without the discipline. A `CLAIM_CLASSES` const in JS is not the same
thing as a typed `ClaimClass` component whose `KnowledgeState` is enforced
at the type level. The site was "ECS-shaped" but not ECS-constitutional.

Rust + WASM is the only way to have the site be a true consumer of the
core. `prism-ecs-core` ships a real `World` with transactional mutation,
generation-safe entity handles, queryable components, and a typed error
enumeration. Reusing that means the docs site *cannot* mutate state outside
`WorldTxn` without a compile error or a runtime `WorldError`. The
architecture is enforced by the language, not by reviewer discipline.

The cost is real: build pipeline, WASM bundle, markdown→HTML in Rust. The
payoff is also real: the docs site is constitutional code. The same
authority model that prevents the kernel from being silently wrong prevents
the docs from drifting from the architecture they describe.

## Migration from the previous JS site

See `docs/CAMPAIGN.md`. Phase order: SSG foundation (this session),
component port, system port, renderer port, page-by-page migration, JS
teardown.

## Completion report (for any docs change)

Every PR that touches the docs site must include:

- Affected page(s) and the entity kinds touched.
- The new world state (component schemas, system changes, renderer
  changes).
- The propagation chain: content source → manifest → world → renderer
  → HTML file on disk.
- A propagation test: delete the rendered HTML, rebuild from the world,
  and assert the resulting HTML is byte-equal (or near-equal, modulo
  timestamps).
- A replay test (when the runtime hydration is involved): apply the
  same visitor-state events to a fresh world, assert the same DOM
  state.
- Authority-leak audit: grep for direct `world.add_component`,
  `document.createElement`, `innerHTML`, and `setAttribute` outside
  the renderer boundary.

Treat "the page renders correctly" as a non-result. A change is incomplete
until the propagation test shows the new fact is observable in the
generated HTML *and* the world can replay the build deterministically.
