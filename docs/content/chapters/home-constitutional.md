---
title: Constitutional Discipline
order: 3
---

The site you are reading is itself built on the same constitutional
ECS as the engine. Every page, every stylesheet, every line of
client-side JavaScript is a projection of the same world.

## What this means in practice

- **The manifest is the source of truth.** A chapter, claim, ADR, or
  page lives in `docs/content/manifest.toml` or under
  `docs/content/`. Nothing else.
- **HTML is generated.** The SSG runs the typed components through
  the renderers and writes `docs/<route>/index.html`. The HTML
  files are not authored; they are projected.
- **CSS is generated.** One file per component kind lives in
  `docs/styles/components/`. The bundle in `site.css` is a
  deterministic aggregate of those files, in canonical order.
- **JavaScript is generated.** The hydration in `site.js` reads
  `data-*` attributes the SSG emitted and reconciles the DOM.
  The only client-side code beyond that is the WASM bundle.
- **WASM is the runtime.** The Rust runtime at `crates/prism-docs-runtime/`
  compiles to `docs/pkg/prism_docs_runtime_bg.wasm`. On the
  browser, it reads the embedded prelude (a JSON snapshot of
  the manifest) and rebuilds the world.

## Why this matters

The site cannot drift from the architecture it describes. There is
one canonical world; the same renderers project it to HTML for
humans and to a `World` for the WASM. Editing a claim in the
manifest and running `cargo run -p prism-docs-ssg` is the entire
change workflow. There is no second source of truth to keep in sync.

## How a content change works

1. Author a markdown file under `docs/content/chapters/` (or
   `adrs/`).
2. Add the entity to `docs/content/manifest.toml` with the
   `kind`, `slug`, and a unique `id`.
3. Reference the new entity from a page's `chapter_refs` (or
   `claim_refs`, `adr_refs`).
4. Run `cargo run -p prism-docs-ssg`. The build is byte-deterministic.
5. The propagation tests verify the new content is in the
   rendered HTML, the prelude, and the per-component CSS.

The site is constitutional code. The cutover happened once, and
every change since is a content change. There is no legacy path
left to migrate.
