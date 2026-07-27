# `session_lifecycle.rs` sub-decomposition — Phase 2 of the server godfile work

**Date:** 2026-07-27
**Status:** Done
**Commit:** (this commit)
**Source godfile:** `crates/prism-ecs-server/src/runtime/server/session_lifecycle.rs` (916 LOC, the residual from the Phase 1 server godfile split, see `2026-07-27-godfile-decomposition-server.md`)
**Resulting layout:** `crates/prism-ecs-server/src/runtime/server/session_lifecycle/` directory with 4 files (1 mod + 3 sub-modules).

## Authority split

The 916-LOC `session_lifecycle.rs` owned four distinct concerns packed into one file: the host-side `ControlSessionState` state machine, the worker-side `WorkerInferencePhase` state machine, the `SessionOutcome` envelope plus `GenerationControlSession` handle, and the HTTP handler functions for `POST /v1/sessions`, `GET /v1/sessions/{id}`, `GET /v1/sessions/{id}/receipt`, `DELETE /v1/sessions/{id}`, and `POST /v1/sessions/{id}/generate`. Single-authority split:

| Sub-module | Authority | Classification | LOC | Public items |
|---|---|---|---:|---:|
| `control_state.rs` | `ControlSessionState` enum + transitions (host-side state machine: `Created` → `Admitted` → `Submitted` → `PrefillRunning` → `Decoding` → `Completed` / `Cancelled`, plus the legacy `PrefillReady` path and `Failed` from any non-terminal state). | Canonical | 191 | 4 |
| `inference_state.rs` | `WorkerInferencePhase` enum + transitions (worker-side state machine: `Created` → `PrefillRunning` → `Decoding` → `Completed` / `Cancelled`, plus `Failed` from any non-terminal phase). | Canonical | 128 | 4 |
| `outcome.rs` | `SessionOutcome` outcome envelope (terminal-result variants) and `GenerationControlSession` (the canonical, no-MLX, no-KV-cache host-side session record that holds identity, policy state, lifecycle state, deadline tracking, and terminal outcome). | Canonical | 181 | 6 |
| `mod.rs` | Façade: sub-module declarations, re-exports of the canonical types to preserve import paths, the 5 HTTP handler functions (with prism-backend vs not branches), and module doc. | Canonical (HTTP ingress) | 447 | 8 (5 handlers + 3 re-exports) |
| **Total** | | | **947** | **22** |

LOC count grew by 31 vs. the 916-LOC source (≈3.4%). The growth is from: (a) the new per-file module docs and authority statements, (b) the `pub use` re-export block in `mod.rs` to preserve the canonical paths, and (c) the `pub mod` declarations in `mod.rs`. All four files are well under the 900-LOC threshold.

## Why HTTP handlers stayed in `mod.rs` and not a fourth sub-module

The five HTTP handlers (`create_session`, `generate`, `get_session`, `get_receipt`, `delete_session`) are pure ingress: they parse a request, call into the server's session-manager / receipt-store, and serialise a response. Each handler has two cfg-gated variants (one for `prism-backend`, one without). They are at a different layer from the state machines and the session handle: the state machines and the handle are pure canonical data, while the handlers are a thin ingress surface that consumes the canonical types. Putting them in `mod.rs` keeps the canonical authority in the three sub-modules and treats the HTTP surface as a façade — consistent with how the other decomposed godfiles (e.g. `bpe_tokenizer/`, `compilation/`) keep the entry-point handlers in the module root.

## Engine absorption (recap, no change)

The `session_lifecycle.rs` godfile (916 LOC) was already the canonical home of:

- `ControlSessionState` (absorbed from `compute-core/src/ecs/core/session.rs` in the server-godfile split)
- `SessionOutcome` (absorbed from the same engine file)
- `GenerationControlSession` (absorbed from the same engine file)
- `WorkerInferencePhase` (renamed from engine `InferenceSessionState` to avoid collision with the server-side `crate::runtime::server_types::InferenceSessionState`; absorbed from the same engine file)

The engine's `InferenceSession` (worker-side, MLX-backed, owns `Vec<KvCache>` + `AtomicBool`) stays in the engine — execution-boundary per criterion 1 (hardware handle) and criterion 4 (raw FFI to MLX). This sub-decomposition did not touch any engine file.

## Canonical-path preservation (re-exports)

`session_lifecycle/mod.rs` re-exports the four canonical types so the import surface is unchanged:

```rust
pub use control_state::ControlSessionState;
pub use inference_state::WorkerInferencePhase;
pub use outcome::{GenerationControlSession, SessionOutcome};
```

So the canonical paths stay:

- `prism_ecs_server::runtime::server::session_lifecycle::ControlSessionState`
- `prism_ecs_server::runtime::server::session_lifecycle::SessionOutcome`
- `prism_ecs_server::runtime::server::session_lifecycle::GenerationControlSession`
- `prism_ecs_server::runtime::server::session_lifecycle::WorkerInferencePhase`

The engine's `compute-core/src/ecs/core/session.rs` documents these as the canonical home; the re-exports keep that path valid.

## Tests

17 tests, distributed by which sub-module they exercise:

- `control_state::tests` — 6 tests: `control_state_initial_is_not_terminal`, `control_state_terminal_set`, `control_state_valid_transitions`, `control_state_failed_from_non_terminal`, `control_state_terminal_rejects_failed`, `control_state_invalid_transitions`.
- `inference_state::tests` — 4 tests: `inference_state_initial_is_not_terminal`, `inference_state_valid_transitions`, `inference_state_failed_from_non_terminal`, `inference_state_terminal_rejects_failed`.
- `outcome::tests` — 7 tests: `control_session_initial_state`, `control_session_happy_path`, `control_session_invalid_transition_preserves_state`, `control_session_identity_transition_is_noop`, `session_outcome_completed`, `session_outcome_cancelled`, `session_outcome_failed`.

Test count matches the source godfile: 6 (control_state) + 4 (generation_control_session) + 3 (session_outcome) + 4 (worker_inference_phase) = 17 tests. All 17 pass.

## Hard-rule compliance per file

| Rule | control_state | inference_state | outcome | mod |
|---|---|---|---|---|
| No `unsafe` | ✓ | ✓ | ✓ | ✓ |
| No `unwrap`/`expect`/`panic!` in production paths | ✓ (test path only) | ✓ (test path only) | ✓ (test path only) | ✓ (3 `unwrap()` in prism-backend HTTP handler branch — pre-existing in source godfile) |
| No `anyhow::Error` | ✓ | ✓ | ✓ | ✓ |
| `BTreeMap` for canonical collections | n/a (no maps) | n/a (no maps) | n/a (no maps) | n/a (no maps) |
| Newtypes for authority-bearing values | n/a (enum state) | n/a (enum phase) | preserved from source (pre-existing `String` for `session_id`, `error_code`, `reason`, etc.) | n/a |
| One-sentence module doc | ✓ | ✓ | ✓ | ✓ (with sub-module authority table) |
| Under 900 LOC | ✓ (191) | ✓ (128) | ✓ (181) | ✓ (447) |
| Under 35 public items | ✓ (4) | ✓ (4) | ✓ (6) | ✓ (8) |

## What did not change

- HTTP API surface: 5 routes preserved (`POST /v1/sessions`, `GET /v1/sessions/{id}`, `GET /v1/sessions/{id}/receipt`, `DELETE /v1/sessions/{id}`, `POST /v1/sessions/{id}/generate`).
- `request_handling.rs` references `super::session_lifecycle::create_session` etc. — those paths still resolve to the new `mod.rs` definitions.
- The engine re-exports: `compute-core/src/ecs/core/session.rs` re-exports the canonical types at the same paths; no engine change.
- State-machine semantics: `ControlSessionState`, `WorkerInferencePhase`, and `SessionOutcome` are byte-for-byte identical to the source godfile.
- `GenerationControlSession` field set and accessors: identical.

## Files changed

- **Deleted:** `crates/prism-ecs-server/src/runtime/server/session_lifecycle.rs` (916 LOC).
- **Created:** `crates/prism-ecs-server/src/runtime/server/session_lifecycle/{mod.rs,control_state.rs,inference_state.rs,outcome.rs}` (947 LOC total, distributed as above).
- **Not modified:** `crates/prism-ecs-server/src/runtime/server/mod.rs` (the existing `pub mod session_lifecycle;` line accepts both the `*.rs` and `*/mod.rs` directory forms).
- **Not modified:** any other file in the working tree. The pre-existing parallel-agent work in `modality_dispatch/` and `evaluator/strategy/` is untouched.
