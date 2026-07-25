# Migrating the Prism Engine docs site

This document is the playbook for migrating the existing JS site
(`docs/js/`, `docs/css/`, `docs/*.html`) to the constitutional ECS
pipeline (`crates/prism-docs-*` + `docs/content/`).

The home page is already migrated as the end-to-end exemplar. This
playbook covers the remaining pages.

## TL;DR

```text
1. Author content in docs/content/ (manifest.toml + markdown)
2. cargo run -p prism-docs-ssg
3. The new HTML appears in docs/dist/
4. Diff docs/dist/*.html against the legacy docs/*.html
5. When the page is faithful, promote it from Shadow to Canonical
6. When all pages are Canonical, delete docs/js/ and the legacy HTML
```

The migration is page-by-page. Each page goes through these states
in `docs/CAMPAIGN.md`: `Inventory` → `Design` → `Shadow` →
`Canonical` → `LegacyRemoved`. A page cannot skip a state.

## Authoring a page

Every page in the new pipeline is composed from typed entities:

- One `Page` entity that declares the route, title, blurb, and
  the chapter/claim/ADR refs it composes.
- N `Chapter` entities (each with a markdown body).
- M `Claim` entities.
- K `Adr` entities (for ADR-heavy pages).
- L `Link` entities that connect the above (e.g., a chapter
  `frames` a claim).

### Example: the `architecture` page

```toml
# docs/content/manifest.toml — appended to the existing entities

[[entity]]
id = "page:architecture"
kind = "page"
route = "/architecture/"
title = "Observe Representation"
blurb = "The canonical subject, its representations, and the constitutional ECS that owns them."
chapter_refs = ["chapter:arch-subject", "chapter:arch-representations"]
claim_refs = ["claim:canonical-subject", "claim:typed-identity", "claim:soa-storage"]
adr_refs = ["adr:003-canonical-ecs-world"]
prev = "page:home"
next = "page:computeimage"

[[entity]]
id = "chapter:arch-subject"
kind = "chapter"
slug = "arch-subject"
title = "The canonical subject"
order = 1
intent = "There is one and only one subject in flight on this page."
blurb = "The single source of truth that every projection reads from."
reading_minutes = 4
body_path = "chapters/arch-subject.md"

[[entity]]
id = "chapter:arch-representations"
kind = "chapter"
slug = "arch-representations"
title = "Representations and projections"
order = 2
intent = "The same world, projected through every observable surface."
blurb = "How the world becomes the page."
reading_minutes = 5
body_path = "chapters/arch-representations.md"

[[entity]]
id = "claim:canonical-subject"
kind = "claim"
text = "One canonical subject, one canonical world. Every surface is a projection."
class = "architectural"
state = "verified"

[[entity]]
id = "claim:typed-identity"
kind = "claim"
text = "Authority-bearing values are newtypes (EntityId, ClaimClass, KnowledgeState). The type says what the value is."
class = "architectural"
state = "verified"

[[entity]]
id = "claim:soa-storage"
kind = "claim"
text = "Components are stored SoA. One 256-byte cache line holds 64 phase enums instead of striding across 4 fields."
class = "illustrative"
state = "observed"

[[entity]]
id = "adr:003-canonical-ecs-world"
kind = "adr"
number = 3
slug = "canonical-ecs-world"
title = "Canonical ECS World"
status = "accepted"
context = "We needed one canonical source of truth. The current code scattered state across maps, queues, and singletons."
decision = "Use prism-ecs-core as the single authority. Every state-bearing change goes through WorldTxn."
consequences = "Everything else becomes a projection. Backends execute immutable descriptors and return outcomes."
body_path = "adrs/adr-003-canonical-ecs-world.md"

[[entity]]
id = "link:arch-subject-frames-claim-canonical-subject"
kind = "link"
from = "chapter:arch-subject"
to = "claim:canonical-subject"
kind = "frames"
```

Then create the markdown files:

```text
docs/content/chapters/arch-subject.md
docs/content/chapters/arch-representations.md
docs/content/adrs/adr-003-canonical-ecs-world.md
```

Each markdown file has YAML frontmatter for the typed components
that are easier to author in markdown (e.g., `title`, `order`,
`reading_minutes`) and a body that the renderer projects into the
HTML.

## Adding a new entity kind

If the migration needs an entity kind the existing schema doesn't
cover (e.g., a new "scenario" or "benchmark" type):

1. Add the typed struct in
   `crates/prism-docs-content/src/<kind>.rs`.
2. Add a `validate()` method that enforces the kind's invariants.
3. Add a variant to `EntityKind` in
   `crates/prism-docs-content/src/manifest.rs`.
4. Add the discriminator in `build_manifest`'s second pass.
5. Add typed components in
   `crates/prism-docs-runtime/src/components/<kind>.rs`.
6. Wire the insertion in
   `crates/prism-docs-runtime/src/ecs/world_bootstrap.rs`.
7. Add a renderer in
   `crates/prism-docs-runtime/src/renderers/<kind>_renderer.rs`.
8. Wire the renderer into `page_renderer` if it composes on pages.
9. Add a propagation test in
   `crates/prism-docs-ssg/tests/propagation.rs` that asserts the
   new kind appears in the rendered output.

Each step is a small, reviewable change. None of them are
review-blocking; they exist to make the migration mechanical and
auditable.

## Adding a new page

Pages are cheap. A page is a `Page` entity in the manifest and a
list of `chapter_refs` / `claim_refs` / `adr_refs`. The SSG picks
it up automatically and emits a `<route>/index.html` file.

If the page is a *special* composition (e.g., the demo page, the
projection-repro page), you may need a dedicated renderer. The
playbook is:

1. Author the content (page + chapters + claims + ADRs + links).
2. Run the SSG; check the output.
3. If the default `page_renderer` is not expressive enough, add a
   new `<page>_renderer` module under
   `crates/prism-docs-runtime/src/renderers/`.
4. In `render_coordinator_system`, branch on the page's id to
   call the new renderer.
5. Add a propagation test that asserts the new renderer's
   content is in the output HTML.

## Adding a new component

Components are added in two ways:

- **Markdown frontmatter**: anything in the YAML at the top of a
  markdown file is available to the body parser and can be read
  by the renderer. Use this for content authoring conveniences
  (e.g., `reading_minutes` on a chapter).
- **Typed component**: a Rust struct with `impl Component for T`.
  Used for facts that are read by systems (not just rendered).

The constitution says: if the component is read by a system, it
must be a typed component. Markdown frontmatter is for projection
hints, not for state.

## Migration of an existing JS-driven page

The previous JS site has a few pages with significant client-side
behavior (the observatory, the demo, projection-repro). Migrating
these is more involved than the home page because they have a
runtime layer.

The playbook:

1. **Inventory the state.** What JS variables exist? What events
   fire? What does the user see? Write the answers in
   `docs/CAMPAIGN.md` for the page.
2. **Design the ECS shape.** Each state variable becomes a
   component. Each event becomes a typed system input. The page
   composition becomes a renderer.
3. **Author the static content** (manifest + markdown). The SSG
   produces the static HTML; this is `Shadow`.
4. **Author the hydration path.** The WASM bundle re-hydrates
   the world from the SSG's prelude, attaches a `DomSubstrate`,
   and runs the hydration schedule. The hydration schedule
   re-reads visitor state from `localStorage` and reconciles the
   DOM.
5. **Compare** the new HTML + WASM behavior with the legacy JS
   side-by-side. When they match, promote to `Canonical`.
6. **Delete** the legacy JS for the page.

The home page has only an SSG step; pages with a runtime need all
six.

## Building the WASM bundle

For pages with hydration, build the WASM:

```sh
cargo build -p prism-docs-runtime \
  --target wasm32-unknown-unknown --release \
  --features hydrate

wasm-bindgen target/wasm32-unknown-unknown/release/prism_docs_runtime.wasm \
  --out-dir docs/dist/pkg --target web
```

Then the generated HTML loads the bundle:

```html
<script type="module" src="/pkg/prism_docs_runtime.js"></script>
<script type="application/json" id="prism-prelude">
  { ... ContentManifest serialized as JSON ... }
</script>
```

The `prism-prelude` block is the SSG's view of the world. The
hydration entry reads it, rehydrates the world, and continues
from there.

## The completion report

Every PR that migrates a page must include a completion report
with:

- **Affected subsystem**: which `Page` entity, which chapters, which
  claims, which ADRs.
- **CAMPAIGN.md status change**: `Inventory` → `Design` → `Shadow`
  → `Canonical` → `LegacyRemoved`.
- **Canonical authority before/after**: the legacy file path that
  was the source of truth, and the new entity(ies) that own the
  same fact.
- **Propagation chain**: manifest → world → schedule → renderer
  → HTML file on disk.
- **Propagation test results**: the byte-equality assertion, the
  content-presence assertions, the component-class assertions.
- **Authority-leak audit**: grep for `world.add_component`,
  `document.createElement`, `innerHTML`, `setAttribute` outside
  the renderer boundary. Zero hits expected.
- **Remaining legacy writers**: any JS file or HTML attribute
  that still touches the page's content. They are tracked until
  removed.

## How to verify the build

```sh
# Content + runtime + SSG all compile cleanly
cargo check -p prism-docs-content -p prism-docs-runtime -p prism-docs-ssg

# All tests pass
cargo test -p prism-docs-content -p prism-docs-runtime -p prism-docs-ssg

# Build the site
cargo run -p prism-docs-ssg -- --content docs/content --out docs/dist

# Open the result
open docs/dist/index.html
```

For pages with hydration, also run:

```sh
cargo build -p prism-docs-runtime --target wasm32-unknown-unknown --release --features hydrate
wasm-bindgen target/wasm32-unknown-unknown/release/prism_docs_runtime.wasm \
  --out-dir docs/dist/pkg --target web
```

## What to do if something breaks

The most common failure modes during migration:

- **Missing markdown file** — `prism-docs-ssg` errors with
  `Io { path: ... }`. Either the body_path in the manifest is
  wrong, or the file is missing. Fix the manifest or the file.
- **Stale link** — `BrokenLink` or `StaleReference` at load time.
  The link's `to` no longer exists. Update the link or the target.
- **Measured claim with no source_refs** — `claim_validation_system`
  rejects it. The constitutional rule. Add a source_ref or change
  the claim class.
- **Schema drift** — the manifest uses a field the typed struct
  doesn't have. The error is `ManifestParse` with a `serde_json`
  explanation. Add the field to the typed struct.
- **HashMap vs BTreeMap** — never use HashMap for canonical
  collections. The schedule is supposed to be deterministic;
  HashMap breaks that. The lint denies HashMap in
  constitutional crates.

Each of these fails loudly. A migration that "compiles but
renders wrong" is a renderer bug, not a content bug. The
propagation tests catch renderer regressions.

## When you're done

When every page is `Canonical` and the legacy `docs/js/` and
legacy `docs/*.html` are empty of content:

1. Delete `docs/js/`.
2. Delete the legacy `docs/*.html` (the SSG output in
   `docs/dist/` becomes the source).
3. Update the `prism_engine` repository's `index.html` to point
   to the SSG output (or have a server rewrite the route).
4. Set the docs site subsystem statuses in `docs/CAMPAIGN.md` to
   `LegacyRemoved`.
5. Update the home page to include a one-line note that the site
   is built on the constitutional ECS, with a link to
   `docs/ARCHITECTURE.md`.

After that, the docs site is constitutional code. A change to a
chapter, claim, or ADR is a change to the typed content, not a
hand-edit to HTML. The architecture is enforced by the language,
not by reviewer discipline.
