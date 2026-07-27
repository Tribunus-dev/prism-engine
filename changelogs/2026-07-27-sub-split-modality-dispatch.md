# 2026-07-27 — Sub-split `server::modality_dispatch` (981 LOC) into 4 sub-modules

## Subsystem

`crates/prism-ecs-server/src/runtime/server/modality_dispatch` (HTTP modality routing surface).

## CAMPAIGN.md status

Pre-refactor: not yet listed as a separate sub-system under
`server.rs` decomposition; absorbed as part of the
`refactor(constitutional): decompose server.rs (2284 LOC) into 5 sub-modules by authority`
campaign entry (commit `e96bd616`).

Post-refactor: still canonical, now split by modality authority.

## Canonical authority before

A single 981-LOC file `crates/prism-ecs-server/src/runtime/server/modality_dispatch.rs`
owned the entire HTTP modality-routing surface (image, audio, video, embeddings, multimodal),
including the file-kind / manifest validators, the inline-payload validator, the
`capture_live_media` envelope, the `resolve_multimodal_request` plan resolver, and the
`make_vision_matmul_provider` typed port.

## Canonical authority after

The directory `crates/prism-ecs-server/src/runtime/server/modality_dispatch/` now owns
the same surface, decomposed into five files with one authority per file:

| Sub-module | Authority | LOC | Public items |
|---|---|---|---|
| `mod.rs` | directory module: `AppState` type alias, `make_vision_matmul_provider` typed port, re-exports of public handlers | 101 | 5 re-exports + 1 type alias + 1 fn = 7 |
| `image.rs` | image + video generation HTTP handlers and vision-encoder support | 277 | 2 (`generate_image`, `generate_video`) |
| `audio.rs` | text-to-speech HTTP handlers | 113 | 1 (`generate_audio`) |
| `embeddings.rs` | text-embedding HTTP handlers | 69 | 1 (`generate_embeddings`) |
| `multimodal.rs` | mixed-modality routing, plan resolution, capture envelope, manifest validation | 591 | 1 (`generate_multimodal`) |

Total: 1151 LOC across 5 files (vs 981 LOC in 1 file). The +170 LOC delta is
the per-file module doc and module-declaration boilerplate. The "authority per
file" split, not the LOC count, is the constitutional improvement.

### Module-doc authorities (one sentence each)

* **`mod.rs`**: this directory owns the canonical modality-routing surface of
  the HTTP API and re-exports the per-modality handlers plus the
  `make_vision_matmul_provider` typed port.
* **`image.rs`**: this sub-module owns the canonical HTTP handlers for
  text-to-image and text-to-video generation, the vision-encoder
  configuration resolver, and the surface glue needed to admit a vision
  model.
* **`audio.rs`**: this sub-module owns the canonical HTTP handlers for
  text-to-speech generation; the `AudioParams` request shape is owned by
  `crate::runtime::modality`, this module is a thin canonical adapter.
* **`embeddings.rs`**: this sub-module owns the canonical HTTP handler for
  text-embedding generation; when a live runtime is admitted it forwards
  to `PrismInferenceServer::generate_embeddings` (execution-boundary trait).
* **`multimodal.rs`**: this sub-module owns the canonical HTTP handler for
  mixed-modality generation, the multimodal-request plan resolver, the
  file-kind / inline-payload validators, the manifest-vs-media-kind
  validators, the `validate_plan_models` cross-check, the `capture_live_media`
  envelope, and the (currently noop) backend execution hook
  `execute_multimodal_backend`.

### Public surface preserved

* `modality_dispatch::generate_image` — re-exported from `image.rs`
* `modality_dispatch::generate_audio` — re-exported from `audio.rs`
* `modality_dispatch::generate_video` — re-exported from `image.rs`
* `modality_dispatch::generate_embeddings` — re-exported from `embeddings.rs`
* `modality_dispatch::generate_multimodal` — re-exported from `multimodal.rs`

The router in `super::request_handling` continues to use these as
`super::modality_dispatch::generate_image` etc. without churn.

## What moved where

| Original location | New location | Notes |
|---|---|---|
| `MultimodalMediaRequest`, `MultimodalHttpRequest` (private structs) | `multimodal.rs` | Used only by `resolve_multimodal_request` |
| `resolve_multimodal_request` | `multimodal.rs` | Used by `generate_multimodal` and 7 tests |
| `validate_file_media_kind` | `multimodal.rs` | Used by `resolve_multimodal_request` |
| `manifest_supports_media_kind` | `multimodal.rs` | Used by `validate_plan_models` |
| `manifest_supports_output_kind` | `multimodal.rs` | Used by `validate_plan_models` |
| `validate_plan_models` | `multimodal.rs` | Used by `generate_multimodal` |
| `vision_config_for_model` | `image.rs` | Was file-private `fn`; widened to `pub(crate)` so multimodal sub-module can call it (future work) |
| `capture_live_media` | `multimodal.rs` | Kept file-private (`fn`) since no current caller in this crate |
| `make_vision_matmul_provider` | `mod.rs` (directory level) | Kept at directory level so the typed port is discoverable from `modality_dispatch::*`; was widened from file-private to `pub(crate)` so `image.rs` tests can call it |
| `execute_multimodal_backend` | `multimodal.rs` | Kept file-private; noop canonical pass for multi-model fusion |
| `validate_inline_payload` | `multimodal.rs` | Used by `resolve_multimodal_request` |
| `generate_image` (3 cfg-gated versions) | `image.rs` | Public |
| `generate_audio` (3 cfg-gated versions) | `audio.rs` | Public |
| `generate_video` (3 cfg-gated versions) | `image.rs` | Public (video is visual generation, sibling to image) |
| `generate_embeddings` (2 cfg-gated versions) | `embeddings.rs` | Public |
| `generate_multimodal` (2 cfg-gated versions) | `multimodal.rs` | Public |
| Test `vision_provider_preserves_gemv_contract` | `image.rs::image_tests` | Tests `make_vision_matmul_provider` from `super::super` |
| Tests `plans_file_input_…`, `preserves_explicit_batched_capture_mode`, `rejects_unsupported_file_kind_before_dispatch`, `rejects_misaligned_inline_audio`, `rejects_wrong_inline_rgba_size`, `rejects_output_without_materialization_path`, `rejects_output_path_without_descriptor`, `preserves_specialist_model_namespaces_in_one_plan` | `multimodal.rs::multimodal_plan_tests` | All test `resolve_multimodal_request` |

## Effect and transaction boundaries (unchanged)

* All public handlers in this directory are HTTP ingress points. They
  produce JSON responses; they do not perform direct world mutation. The
  handlers call into `PrismInferenceServer` (an execution-boundary trait)
  for the actual work.
* The plan resolver `resolve_multimodal_request` is a pure function
  (`Value -> Result<Value, String>`); it does not call any backend.
* The execution hook `execute_multimodal_backend` is currently a noop
  canonical pass for multi-model fusion (`status: "noop"`); it does not
  invoke a backend.
* The capture envelope `capture_live_media` calls
  `prism_multimodal::capture::CaptureCoordinator` (an
  execution-boundary coordinator) to acquire camera/mic packets.

## Schema versions

No schema change. `MultimodalMediaRequest` and `MultimodalHttpRequest`
are private to `multimodal.rs` (previously private to
`modality_dispatch.rs`) and are not exposed to the public API.

## Replay behavior

No replay behavior change. This refactor is a code-organization change
with no semantic difference; existing canonical state, durable events,
and replay paths are unaffected.

## Tests

* 9 tests total preserved (1 in `image.rs::image_tests`, 8 in
  `multimodal.rs::multimodal_plan_tests`). Test count and content are
  identical to the original `modality_dispatch::multimodal_plan_tests`
  module except for the split across two test sub-modules.
* `cargo check -p prism-ecs-server --features server` passes with 22
  warnings, all pre-existing in the original file or in other crates.
  The 3 "never used" warnings for `make_vision_matmul_provider`,
  `vision_config_for_model`, and `capture_live_media` are pre-existing
  in the original file (the only caller of `make_vision_matmul_provider`
  is a `cfg(test)` test; `vision_config_for_model` and `capture_live_media`
  have no callers in this crate yet and are reserved for future
  capture/admission work). They moved to new file locations, but the
  count is unchanged.

## Verification gap

`cargo test -p prism-ecs-server --lib server::modality_dispatch` was
**blocked** at run time by pre-existing build errors in
`crates/prism-ecs-runtime` from parallel agents mid-decomposition:

```
error[E0761]: file for module `command_dispatch` found at both
  "crates/prism-ecs-runtime/src/kernel/command_dispatch.rs" and
  "crates/prism-ecs-runtime/src/kernel/command_dispatch/mod.rs"
error[E0761]: file for module `worker_protocol` found at both
  "crates/prism-ecs-runtime/src/worker_protocol.rs" and
  "crates/prism-ecs-runtime/src/worker_protocol/mod.rs"
```

These are not caused by this refactor. `cargo check -p
prism-ecs-server --features server` (without test compilation) succeeds
in 24.25s with the same 22 warnings as the pre-refactor baseline, and
the same 3 "never used" warnings on the moved functions. The build
failure is in a sibling crate that this refactor does not touch; it
will be cleared when the parallel agents complete their
`command_dispatch` and `worker_protocol` decompositions and delete the
old `.rs` files.

## Authority-leak audit

* No new direct world mutation in this directory.
* No `unsafe` in any of the 5 new files.
* No `unwrap` / `expect` / `panic!` / `unreachable!` / `todo!` /
  `unimplemented!` in any production path. The `unwrap_or` / `unwrap_or_default`
  fallbacks present in `validate_file_media_kind` and `vision_config_for_model`
  are the same as the original file and are documented at the call site.
* No `anyhow::Error`. The plan resolver returns `Result<Value, String>`,
  matching the original signature.
* `BTreeMap` is used for the only canonical collection
  (`std::collections::BTreeMap::<String, Vec<usize>>` in
  `execute_multimodal_backend` for per-model media-group fanout);
  the rest of the directory uses `Vec` and `serde_json::Value`
  exclusively. No `HashMap` / `HashSet` introduced.
* All authority-bearing values keep their pre-existing types (no
  newtype churn in this refactor). The keys and digests in the
  response JSON are produced by upstream layers; this directory
  only formats them.
* The `BTreeMap` usage in `execute_multimodal_backend` is
  observable (the order of media fanout affects validation results),
  so it must be ordered; `BTreeMap` is the correct canonical choice.

## Engine absorption status

* `compute-core/src/ecs/core/mlx_inventory.rs` was previously identified
  as the execution-boundary counterpart of
  `modality_dispatch::make_vision_matmul_provider` (the typed port
  interface to `crate::engine::metal::dispatch_fp16_matmul`).
* No engine file was modified. The `metal-dispatch` feature-gated
  `dispatch_fp16_matmul` call is preserved verbatim in
  `modality_dispatch::mod::make_vision_matmul_provider`.

## Remaining writers / future work

* `vision_config_for_model` and `capture_live_media` are still
  un-called from the current handlers. They are reserved for the
  next wave of capture/admission work; once a caller materializes,
  the `pub(crate)` visibility is sufficient.
* `execute_multimodal_backend` remains a noop. The multi-model fusion
  path is rejected with the existing error message
  "multi-model fusion across N specialised runtimes is not yet wired".
  When the engine exposes a true multi-model fusion entry point,
  this function is the canonical seam.

## Files changed

* Deleted: `crates/prism-ecs-server/src/runtime/server/modality_dispatch.rs` (981 LOC)
* Created:
  * `crates/prism-ecs-server/src/runtime/server/modality_dispatch/mod.rs` (101 LOC)
  * `crates/prism-ecs-server/src/runtime/server/modality_dispatch/image.rs` (277 LOC)
  * `crates/prism-ecs-server/src/runtime/server/modality_dispatch/audio.rs` (113 LOC)
  * `crates/prism-ecs-server/src/runtime/server/modality_dispatch/embeddings.rs` (69 LOC)
  * `crates/prism-ecs-server/src/runtime/server/modality_dispatch/multimodal.rs` (591 LOC)
* `crates/prism-ecs-server/src/runtime/server/mod.rs` re-export list is
  unchanged: the public handlers are still re-exported from
  `modality_dispatch` (now a directory) without any change to the
  `request_handling` router wiring.

## Checkpoint / commit log

* `57017cb1` — pre-work checkpoint: `checkpoint: pre-work on modality_dispatch`
* `91400154` — mid-work checkpoint: `checkpoint: modality_dispatch sub-modules in progress`
* final commit: `refactor(constitutional): split server/modality_dispatch.rs (981 LOC) into 4 sub-modules by authority`
