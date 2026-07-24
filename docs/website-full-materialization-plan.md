# Prism Observatory v1 — implementation record

**Baseline:** `43b83e252bc365165063f11c0c2e5facbe264116`  
**Branch:** `website/prism-observatory-v1`  
**Date:** 2026-07-24

## Objective

Materialize `/docs` as the evidence-bound public projection of Prism Engine: compiler, ComputeImage, constitutional runtime, heterogeneous providers, capabilities, and replayable evidence.

## Implemented in this branch

- Canonical capability registry with one status vocabulary.
- First-class capability/status page.
- Reframed homepage around the canonical deployment graph.
- Developer `Run Today` workflow separated from the future packaged demo.
- First-class ComputeImage artifact page.
- First-class evidence, authority, replay, and recovery page.
- Roadmap restricted to future milestones instead of current evidence claims.
- Global navigation on new and rewritten core surfaces aligned to Compiler / ComputeImage / Runtime / Evidence / Status / Run.

## Remaining consolidation tracked by the draft PR

- Migrate every legacy page to the new navigation component and page-purpose contract.
- Replace duplicated header, footer, living-atlas, and shared-object markup with declarative components.
- Add generated architecture, evidence, model, release, and navigation registries.
- Bind sanitized real receipt and replay fixtures to the Evidence page.
- Add claim linting, source-path validation, freshness checks, accessibility tests, link checks, and visual regression coverage.
- Consolidate CSS imports into explicit layers and remove residual page-specific leakage.

## Status vocabulary

- **Released:** versioned reproducible distribution with explicit support boundaries.
- **Validated:** measured on a defined build and hardware configuration with evidence.
- **Qualifying:** implemented and tested or compile-verified; target evidence is incomplete.
- **Implemented:** code path, data model, command, or provider boundary exists.
- **Planned:** architecture or accepted design exists; end-to-end implementation is incomplete.

## Definition of done

The observatory is complete when every public claim is generated or validated from one canonical registry, every validated claim links to evidence, all pages observe one architecture graph, the runnable path is reproducible, no-JS semantics remain complete, and CI rejects stale or contradictory status language.
