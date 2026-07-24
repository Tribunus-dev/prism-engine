# Prism Runtime

Status: frozen for this sprint (2026-07-24)

The Prism Runtime is one executable coordinator with systems for observation, knowledge, receipts, optics, interaction, rendering, accessibility, and visitor continuity. The systems are plugins; the runtime owns their ordering and shared subject state.

This document is the canonical reference for runtime layering and execution orchestration in the web experience.

Canonical mappings

- Subject lifecycle and canonical observational identity are defined in [`observatory-kernel.js`](js/observatory-kernel.js) and surfaced via [`runtime/create-runtime.js`](js/runtime/create-runtime.js).
- Canonical compute-image subject shape is managed by `ensureComputeImageSubject` and `updateComputeImageSubject` in `observatory-kernel.js`.
- Canonical observation flow and scene transitions are validated in [`runtime/repository-state.json`](repository-state.json) and enforced by [`core/observation-graph.js`](js/core/observation-graph.js).

Do not split runtime layers into new architecture names while the canonical flow is frozen. Consolidation should remain focused on implementation reduction and projection consistency.

## Runtime phases

`intent → observation → transformation → commit → execution → evidence → reflection`

The same phases can be rendered by the Observatory, documentation, compiler inspector, runtime debugger, ComputeImage explorer, Fabric console, presentation, or interactive ADR.

## Observation Graph

The Observation Graph replaces page-centric Scene Graph language. An observation is an entity over one subject, instrument, state, evidence boundary, transition, and semantic history.

## Visitor intents

Explore, Understand, Inspect, Validate, and Contribute are intent-based lenses. They change depth and emphasis without changing the subject or claims.
