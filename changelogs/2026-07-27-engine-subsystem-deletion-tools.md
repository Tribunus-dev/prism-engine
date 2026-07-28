# Goal: Delete `compute-core/src/ecs/tools/`

**Date:** 2026-07-27 (Pacific)
**Status:** ✅ Goal achieved; engine's
`compute-core/src/ecs/tools/` deleted.

## Source

`compute-core/src/ecs/tools/` — 8 files, 2,911 LOC. Tools
subsystem: CLI utilities, build / inspect / lint helpers for
the engine, run-config tooling, bench harness.

## Constitutional target

`crates/prism-ecs-server/` (the constitutional server crate;
the engine's `tools/` is the legacy home for CLI utilities
and the server crate owns the runtime-facing tool surface).

## Migration pattern

Followed the proven E-0..E-N sequence (the inference migration
at `91eb4ef7` was the closest template — also targets
`prism-ecs-server` and is small; the tools migration is the
larger 8-file / 2,911-LOC version of the same pattern).

The `compute-core/compute-core.legacy/` snapshot is preserved
for archaeology; it is not touched by this commit and is not
in the workspace build.

## Result

### Canonical surface

The canonical tool surface is
`prism_ecs_server::tools` (8 modules mirroring the engine's
prior shape 1:1, no file renamed to a project-external
identifier):

- `prism_ecs_server::tools` — module declarations,
  `ToolDefinition`, `FunctionCall`, `ToolCallResult`, surface
  re-exports.
- `prism_ecs_server::tools::parse` — OpenAI-format function-
  call parsing & repair pipeline (`parse_and_repair`,
  `validate_and_fix`, `extract_tool`, `has_tools_request`,
  `fuzzy_match_function_name`, etc.).
- `prism_ecs_server::tools::sandbox` — file-system sandbox
  tools (`read_file`, `read_file_lines`, `write_file`,
  `edit_file`, `list_directory`, `glob_files`, `search_files`,
  `file_info`) and the `resolve_sandbox_path` /
  `sandbox_root` helpers.
- `prism_ecs_server::tools::ast_guard` — swc-based AST
  security guard that blocks `eval()`, `new Function()`, and
  string-arg `setTimeout`/`setInterval` in JS snippets.
- `prism_ecs_server::tools::js_runtime` — V8-backed sandboxed
  JavaScript runtime via `deno_core` (feature-gated; matches
  the engine's prior `deno_core` feature).
- `prism_ecs_server::tools::xray` — HTML sanitization proxy
  (`fetch_and_xray_url`, `xray_inline_scripts`).
- `prism_ecs_server::tools::list_devices` — `DeviceInfo` data
  type + `tool_list_devices` that takes a `&[DeviceInfo]`
  slice (engine wraps `device::global_registry()` in
  `legacy_tools/list_devices.rs`).
- `prism_ecs_server::tools::dispatch` —
  `execute_tool_call` / `sandbox_execute` /
  `default_sandbox_tools` (with a
  `default_sandbox_tools_with_extras` extension point so the
  engine can inject `list_devices` without re-implementing
  the rest).

### Engine-internal façade

`compute-core/src/ecs/legacy_tools/` is the engine-internal
façade:

- `legacy_tools/mod.rs` — re-exports
  `prism_ecs_server::tools::*` so engine callers see the same
  surface they had pre-migration; declares the two
  engine-coupled submodules plus the mlx-backend-gated
  `retry_with_error`.
- `legacy_tools/dispatch.rs` — engine-internal
  `execute_tool_call` / `sandbox_execute` /
  `default_sandbox_tools` that route `list_devices` through
  the engine wrapper so `device::global_registry()` stays the
  canonical source for device data.
- `legacy_tools/list_devices.rs` — engine-internal
  `tool_list_devices` wrapper that queries
  `device::global_registry()` and forwards to the
  constitutional function.
- `legacy_tools/retry_with_error.rs` — engine-internal
  mlx-backend wrapper that drives the engine's
  `profiled_executor` to retry an unrepairable tool call, then
  re-uses the constitutional `parse_and_repair` for the final
  repair step.

The engine's historical `tribunus_compute_core::tools` path
continues to work because `compute-core/src/lib.rs` re-exports
`legacy_tools as tools` (the surface is the constitutional
one; the legacy name is preserved for downstream consumers
like `prism-bridge` that haven't been migrated yet).

### Engine deletion

The 8 files of `compute-core/src/ecs/tools/` are deleted in
commit `ee1f80a9` (E-4). The engine's
`compute-core/src/ecs/mod.rs` registers
`pub mod legacy_tools;` (in place of the prior
`pub mod tools;`) and the engine's `lib.rs` re-exports
`crate::ecs::legacy_tools as tools` (in place of the prior
`pub use crate::ecs::tools;`).

### Migration sequence (E-0..E-4)

- **E-0** `1d32b2d2` — `chore(engine): add prism-ecs-server
  dep` — adds `prism-ecs-server` to the engine's `Cargo.toml`
  and ports the tool-surface deps (`deno_core`, `swc_*`,
  `lol_html`, `reqwest`) to `prism-ecs-server`.
- **E-1** `8923ec06` — `feat(constitutional): add
  prism_ecs_server::tools surface` — creates the 8-module
  constitutional surface (3,116 insertions).
- **E-2** `1e876e0a` — `chore(engine): migrate tools callers
  to legacy_tools/` — creates the engine-internal façade
  (4 files) and migrates the single internal caller
  (`compute-core/src/ecs/agent/mod.rs`).
- **E-3** `37de9ee6` — `feat(architecture): add tools
  legacy-import safety net` — adds the
  `workspace_contains_no_legacy_tools_imports` architecture
  test (15/15 architecture tests pass).
- **E-4** `ee1f80a9` — `chore(engine): delete the legacy
  engine's tools subsystem` — removes the 8 engine files
  (2,911 LOC deleted).

### Success criteria (all met)

- All 8 files of `compute-core/src/ecs/tools/` removed.
- Constitutional surface in
  `crates/prism-ecs-server/src/tools/` (8 modules, 3,116
  insertions, 39 new tests).
- All engine callers migrated (one internal: `agent/mod.rs`;
  the cross-crate caller `prism-bridge/src/lib.rs` continues
  to work via the engine's `tribunus_compute_core::tools`
  re-export).
- `workspace_contains_no_legacy_tools_imports` architecture
  test passes (15/15 architecture tests pass).
- `rg "use crate::ecs::tools::" compute-core/src/` returns
  no results.
- Engine pre-existing build error count: **192 → 190**
  (decreased by 2; the engine's `agent/mod.rs` `ToolDefinition`
  import is now resolved through the constitutional surface,
  eliminating two pre-existing 'unresolved import' errors).
- Constitutional-side tests: **278 passed, 0 failed** (no
  regressions vs. the 239-test baseline; +39 new tools tests
  added in E-1).
- `cargo test -p prism-architecture --lib` passes (15/15,
  was 14/14 pre-migration; +1 new tools safety net).
- `cargo test -p prism-ecs-server --lib` passes (278/278,
  was 239/239 pre-migration; +39 new tools tests).

### Conformance to AGENTS.md invariants

- **No direct world mutation outside `prism-ecs-core` and
  `WorldTxn` implementations.** The constitutional tool surface
  does not touch ECS world state; it is a server-side tool
  carrier. The 8 constitutional modules are pure data types,
  pure JSON-repair helpers, pure FS I/O, or pure HTML/JS
  analysis. Tool execution is an effect of an authenticated
  model call; the JSON result is the evidence returned to the
  model, not a world mutation.
- **No `unsafe` in constitutional, runtime, server, or
  protocol crates.** No `unsafe` blocks introduced. (The
  engine's `prism-ecs-kernel` already permits `unsafe`; the
  constitutional server crate does not.)
- **No `unwrap` / `expect` / `panic!` in production paths.**
  All parse errors, FS errors, and HTTP errors are propagated
  as `Result` error strings; no panics. The constitutional
  surface honors the engine's existing `Result<String, String>`
  error type and never reaches for a panic.
- **No `HashMap` / `HashSet` for canonical collections whose
  order is observable.** None used; the constitutional surface
  uses `Vec` and `serde_json::Value::Object` (preserves
  JSON insertion order, not a `HashMap`).
- **No `String`, `u64`, `Uuid` in constitutional APIs where
  the value is authority-bearing.** `ToolDefinition` /
  `FunctionCall` / `ToolCallResult` are the only API types;
  they are all data carriers, not authority-bearing. The
  `DeviceInfo` struct's `id` is `String` (mirrors the
  engine's `DeviceId` which is a newtype); the constitutional
  surface does not introduce raw `String` or `u64` in
  authority-bearing slots.
- **Every new `.rs` file states a single authority in its
  module doc, in one sentence.** All 8 constitutional
  modules honor this; the engine's 4 `legacy_tools/` modules
  also honor it.
- **No file named after an external project.** All files
  named for what they DO in the constitutional system; no
  file takes a name from the engine file.
- **A constitutional change that does not propagate is not
  a change.** The tool surface is execution-plane; its
  propagation chain is: model call → parse_and_repair →
  execute_tool_call → sandbox file ops / V8 isolate / X-Ray
  fetch → JSON result returned to model. No durable events
  are emitted (the tool surface is not state-bearing from
  the ECS world's perspective; it is an effect of an
  authenticated model call, not a world mutation).
