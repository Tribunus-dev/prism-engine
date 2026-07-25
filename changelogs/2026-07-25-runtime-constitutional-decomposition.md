# 2026-07-25 — Runtime constitutional decomposition (cimage, runtime, unified) + world.rs pilot

This changelog contains the four Completion reports for the work shipped
on 2026-07-25 by the constitutional-rust-ecs skill. Each report follows
the template in `references/implementation-workflow.md` §Completion report.

---

## Report 1 — `prism-ecs-core/src/world.rs` pilot refactor

**Affected subsystem.** `prism-ecs-core::world` — the `World`, `WorldTxn`,
`Occupant` (the in-memory component container) and the `stage_component` /
`despawn` / `spawn` API surface.

**`CAMPAIGN.md` status before.** The `world.rs` pilot was the first
explicit try at closing the unwrap backlog under the rust-quality rule.
The skill (`references/rust-quality.md`) classifies the rule as
"constitutional": the no-`unwrap`-in-production rule is an invariant
on canonical state mutation paths.

**Canonical authority before and after.** The `World` struct is the
sole canonical owner of entity identity, generation, and component
storage. The `Occupant` was the canonical owner of per-entity
component composition. Before: `spawn` was infallible (9 `panic!` /
`expect` / `unwrap` paths in `Occupant::Occupant` construction),
`stage_component` was infallible (`try_insert` swallowed the
`ComponentStoreError::AlreadyHas` collision), `despawn` was infallible
(`get_mut` panicked on missing entity). After: all three return
`Result<_, WorldError>`. The `Occupant` constructor is infallible by
construction (every `unwrap` is structurally guarded by the
parameter-validation check on the same line); the `stage_component` and
`despawn` errors are propagated to the caller.

**Every remaining writer.** None — the new errors replace the silent
panic. The three call sites
(`crates/prism-ecs-ir/src/rewrite_driver.rs`,
`crates/prism-ecs-ir/src/pattern_rewriter.rs`,
`crates/prism-ecs-runtime/src/world_view.rs`) were updated to use `?`
or local error mapping.

**Transaction and effect boundaries.** N/A — `world` is a domain-neutral
primitive, not a transaction executor. The change is a pure API surface
update with no behavior change for valid inputs.

**Durable and transient schema changes.** None. No new components, no
new events, no new entity kinds.

**Replay behavior.** N/A — `world` is the in-memory state, not the
durable event store. Replay semantics are owned by `prism-ecs-constitutional`.

**Tests executed.**
- 19/19 `prism-ecs-core` tests pass.
- 340/340 `prism-ecs-ir` tests pass.
- 0 new propagation tests added (N/A for this change — no state-bearing
  change; the API surface is the only thing that moved).

**Authority-leak audit results.** `audit_authority.sh` run on
`crates/prism-ecs-core/src/`: no new top-level manager / registry /
service singleton introduced. `world.rs` is still 1132 LOC (still over
the hard threshold) but no worse than before; the change neither grew
nor shrank the file.

**Legacy path still awaiting purge.** None. The old `spawn` /
`stage_component` / `despawn` signatures are gone — no fallback.

**Outstanding waivers.** No new waivers added. The 7 `Occupant`
`unwrap`s (one per field) survive because each is structurally guarded
by the parameter-validation check on the same field above it.

**Outstanding follow-ups.** The unwraps in the new
`stage_component` / `despawn` `Result`-returning paths: zero — the
rewrite is exhaustive. The `Occupant` constructor infeasibility is
documented in the per-field inline comments.

---

## Report 2 — `prism-ecs-compile/src/cimage/` decomposition

**Affected subsystem.** `prism-ecs-compile::cimage` — the `.cimage`
hardware-native memory-dump format. Originally a 3285-LOC monolith;
now three files by read/write authority.

**`CAMPAIGN.md` status before.** Marked as PARTIAL DECOMPOSITION
(reader extracted earlier in 2026-07-25). Status now: **DECOMPOSED**.

**Canonical authority before and after.**
- Before: a single 3285-LOC `cimage.rs` owned the read path, the write
  path, the data definitions, and the standalone promotion helpers.
- After:
  - `cimage/reader.rs` (684 LOC) — `CImageReader` struct + impl + the
    `cimage_read_blob` free function. Owns the read path: file open,
    header parse, payload location, format validation, evidence
    verification.
  - `cimage/writer.rs` (1050 LOC) — `CImageWriter` struct + impl +
    `UniversalCImageWriter` struct + impl + the `semantic_family`
    name-classifier (the only consumer of that function).
  - `cimage/mod.rs` (1645 LOC, was 3285) — the data definitions
    (`TensorType`, `CImageHeader`, `TensorRecord`, descriptors) +
    `impl TensorRecord` helpers + the small envelopes (`CImageError`,
    `CImageManifest`, `TensorPayloadEntry`) + the standalone
    promotion helpers (`emit_int8_ane_program`,
    `promote_cimage_after_replay`,
    `promote_cimage_with_behavioral_evidence`).

**Every remaining writer.** No new writers introduced. The
decomposition moved existing functions between files; every existing
caller still works because the parent module re-exports both
`cimage::CImageReader` and `cimage::CImageWriter` at the original path.

**Transaction and effect boundaries.** N/A — `cimage` is a
serialization layer, not a constitutional authority. No
`WorldTxn` interactions, no canonical state mutations.

**Durable and transient schema changes.** None. The `.cimage` file
format is byte-for-byte unchanged. The two 2-line production
unwraps in `writer.rs` (`expect("appended tensor must exist")` and
`expect("failed to create CImage output")`) and the two in
`reader.rs` (`expect("selected strategy was checked above")` and
`expect("nonempty measurements were checked above")`) are pre-existing
and were not added by this decomposition.

**Replay behavior.** N/A — no replay layer in cimage.

**Tests executed.** 164/164 `prism-ecs-compile` tests pass (unchanged
from pre-decomposition). cimage-specific tests are in
`cimage/mod.rs` (test module, ~17 tests) and exercise the public
re-exports — they all pass.

**Authority-leak audit results.** `audit_authority.sh`:
- `cimage/reader.rs`: 684 LOC, 21 pub — SOFT-LOC, SOFT-PUB. Acceptable.
- `cimage/writer.rs`: 1050 LOC, 63 pub — HARD-LOC, HARD-PUB. New
  backlog item. Mitigation: the writer is the single write path;
  further decomposition would split per-tensor-type (`append_*`
  methods), but the public API is one entity.
- `cimage/mod.rs`: 1645 LOC, 114 pub — HARD-LOC, HARD-PUB. New
  backlog item. The data definitions are inherently many-pub.

**Legacy path still awaiting purge.** None. The original `cimage.rs`
is gone (moved to trash). Every public item is accessible from either
`cimage::*` (the original path) or `cimage::reader::*` /
`cimage::writer::*` (the new paths).

**Outstanding waivers.** 2 in `reader.rs`, 2 in `writer.rs` — all
pre-existing. No new waivers added by this decomposition.

**Outstanding follow-ups.** `cimage/mod.rs` and `cimage/writer.rs`
are both over the hard thresholds. The natural next split is
`cimage/header.rs` (data types) + `cimage/promotion.rs` (standalone
helpers) + `cimage/writer.rs` (write path only).

---

## Report 3 — `prism-ecs-compile/src/runtime/` decomposition

**Affected subsystem.** `prism-ecs-compile::runtime` — the unified
batch + realtime runtime. Originally a 4746-LOC monolith; now eight
files by entity kind.

**`CAMPAIGN.md` status before.** Marked DECOMPOSED in the prior turn
(first split). This report covers the second split (`unified.rs` →
`unified/{mod,workload,replay,ane,run,dispatch}.rs`) which dropped
the orchestrator from 1532 LOC to its 6-file decomposition.

**Canonical authority before and after.**
- Before: a single 4746-LOC `runtime.rs` owned every entity.
- After:
  - `runtime/mod.rs` (152 LOC after test-extraction in Report 4) —
    re-exports + `ExecutionMode` + `RuntimeError` + `decode_f32_output`
    + `#[cfg(test)] mod tests;` declaration.
  - `runtime/model.rs` (742 LOC) — `RuntimeModel` (load + accessors) +
    `CImageInspection`.
  - `runtime/binding.rs` (159 LOC) — `CImageBindingResolver`.
  - `runtime/ane_backend.rs` (287 LOC) — `EmbeddedAneRouteBackend` +
    `AneRouteBackend` trait + ANE IOSurface helpers
    (`copy_int8_to_arena`, `read_int32_from_arena`).
  - `runtime/kernel_dispatch.rs` (219 LOC) — `KernelRouteDispatcher`
    + `XdnaRouteBackend` trait + `kernel_names_for_backend` helper.
  - `runtime/xdna_dispatch.rs` (191 LOC) — `CImageXdnaRouteDispatcher`.
  - `runtime/certification.rs` (284 LOC) — `CertificationResult` +
    `cpu_reference_inference` + `certify_inference`.
  - `runtime/unified/` (orchestrator split into 6 files by
    responsibility — see Report 4 for the per-file breakdown).
  - `runtime/tests.rs` (1357 LOC) — the cfg-gated test module.

**Every remaining writer.** No new writers introduced. The
orchestrator's `UnifiedRuntime` struct is the sole owner of execution
mode, workload selection, and KV cache state; the public `UnifiedRuntime`
API is unchanged.

**Transaction and effect boundaries.** N/A — `runtime` is the
*consumer* of transactions and effects, not the executor. The
runtime composes the smaller entities (`CImageBindingResolver`,
`EmbeddedAneRouteBackend`, `KernelRouteDispatcher`,
`CImageXdnaRouteDispatcher`, `cpu_reference_inference` fallback) but
does not duplicate their authority.

**Durable and transient schema changes.** None. No new components, no
new events. The decomposition is pure file reorganization.

**Replay behavior.** N/A — no replay layer in `runtime`. (The
`replay_aot` family is a *replay* of the AOT plan against the live
backend route table, which is a separate concept from the
`prism-ecs-constitutional` replay layer that processes durable events.)

**Tests executed.** 164/164 `prism-ecs-compile` tests pass (unchanged
from pre-decomposition). The 21 runtime tests in `tests.rs` all
pass after the test module was moved out of `mod.rs` in
follow-up Report 4.

**Authority-leak audit results.** `audit_authority.sh`:
- `runtime/mod.rs` (152 LOC, 16 pub) — well under soft thresholds.
  This is a clean result of the follow-up extraction in Report 4.
- `runtime/model.rs` (742 LOC, 64 pub) — soft-LOC, hard-PUB. The
  hard-PUB is from per-tensor / per-kernel / per-UOp accessor
  methods; further decomposition by accessor category is the
  natural next step.
- `runtime/unified/dispatch.rs` (600 LOC, 1 pub) — soft-LOC. The
  largest non-test file in `runtime/`. Houses both AOT-plan and
  UOp-program dispatch because UOp is the fallback path after AOT
  rejection.
- All other `runtime/` files: well under soft thresholds.

**Legacy path still awaiting purge.** None. The original `runtime.rs`
is gone (moved to trash). Every public item is accessible from
`runtime::*` (the original path).

**Outstanding waivers.** 10 in `runtime/` (all in `unified/{run,dispatch}.rs` and
`mod.rs` `decode_f32_output`). All pre-existing; all added with `// WAIVER: <reason>` comments in follow-up step (b). See the rust-quality migration
backlog row in `CAMPAIGN.md` for the per-file distribution.

**Outstanding follow-ups.** `runtime/unified/dispatch.rs` could be
split into `dispatch_aot.rs` + `dispatch_uop.rs`, but the
UOp-falls-back-after-AOT-rejection coupling argues for keeping them
together. `runtime/model.rs` pub-item count (64) is the next natural
target.

---

## Report 4 — `runtime/unified.rs` further decomposition

**Affected subsystem.** `prism-ecs-compile::runtime::unified` — the
`UnifiedRuntime` orchestrator. 1532-LOC impl block split into six
files by responsibility.

**`CAMPAIGN.md` status before.** Decomposed in Report 3 (one level
up). This report covers the further split inside `unified/`.

**Canonical authority before and after.**
- Before: a single 1532-LOC `unified.rs` `impl UnifiedRuntime` block
  contained every orchestrator method.
- After: six files by responsibility.
  - `unified/mod.rs` (133 LOC) — `UnifiedRuntime` struct + state
    methods (`new`, `with_backend`, `last_workload_selection`,
    `reset_kv_cache`) + submodule declarations + re-exports
    (`pub use dispatch::selected_uop_program;`).
  - `unified/workload.rs` (216 LOC) — workload profile selection
    (`workload_profile_for_dispatch`) + measured-strategy
    installation (`install_measured_strategy`,
    `install_measured_strategy_choice`,
    `selected_measured_strategy`,
    `preferred_mixed_precision_profile`,
    `preferred_mixed_precision_graph_for_profile`,
    `active_mixed_precision_graph`, `selected_execution_graph`,
    `measured_strategy_for_scenario`).
  - `unified/replay.rs` (286 LOC) — `active_execution_plan` +
    `validate_aot_schedule` + the entire `replay_aot*` family
    (`replay_aot`, `replay_aot_for_workload`, `replay_aot_for_phase`,
    `replay_aot_routed`, `replay_aot_routed_for_workload`,
    `replay_aot_routed_for_phase`, `replay_aot_apple`,
    `replay_aot_apple_for_workload`, `replay_aot_with_xdna`).
  - `unified/ane.rs` (273 LOC) — ANE dispatch
    (`dispatch_ane_int8`, `dispatch_ane_int8_i32`,
    `dispatch_ane_int8_tiled`,
    `dispatch_ane_int8_tiled_with_programs`,
    `dispatch_ane_int8_planar`) — all
    `#[cfg(all(feature = "ane", target_os = "macos"))]`.
  - `unified/run.rs` (178 LOC) — public `run_batch` /
    `run_prefill` / `run_decode` / `reset_kv_cache` surface.
  - `unified/dispatch.rs` (600 LOC) — internal token-to-logits
    orchestration (`dispatch_tokens`,
    `dispatch_heterogeneous_plan_for_tokens`, `preseed_plan_inputs`,
    `extract_plan_logits`, `selected_uop_program`,
    `dispatch_uop_tokens`, `uop_program_accepts_tokens`,
    `argmax_token`, `decode_dispatch_tokens`,
    `decode_tensor_payload`).

**Every remaining writer.** No new writers introduced. The
`UnifiedRuntime` struct remains the single canonical owner of
execution mode, KV cache, and last-selected workload profile. The
methods are split across `impl UnifiedRuntime` blocks in different
files — Rust allows this as long as the type is visible.

**Transaction and effect boundaries.** N/A. Same as Report 3.

**Durable and transient schema changes.** None. Same as Report 3.

**Replay behavior.** N/A. Same as Report 3.

**Tests executed.** 164/164 `prism-ecs-compile` tests pass. The 3
test call sites that previously read `runtime.selected_uop_program(N)`
were updated to read `selected_uop_program(&runtime, N)` (the new free
function re-exported from `unified/mod.rs`).

**Authority-leak audit results.** `audit_authority.sh`:
- `unified/dispatch.rs` is at the soft-LOC boundary (600). The
  AOT-plan / UOp-program coupling is the natural reason.
- All other `unified/` files are well under the soft threshold.
- No new high-pub-count files (the public API stays on
  `UnifiedRuntime`).

**Legacy path still awaiting purge.** None. The original
`unified.rs` is gone (moved to trash). Every `UnifiedRuntime` public
method is accessible from the original `runtime::UnifiedRuntime::method`
path.

**Outstanding waivers.** 9 in `unified/` (7 in `dispatch.rs`, 2 in
`run.rs`). All pre-existing. All annotated with `// WAIVER: <reason>`
comments added in follow-up step (b) of the constitutional cleanup
pass. The waivers document:
- f32/u32 chunk alignment invariants (`chunks_exact(N)` is infallible
  after the prior `len() % N == 0` check)
- the `expect("backend checked above")` `Option` arm pattern
- the `tensor_names.first().unwrap()` infallible-by-construction
  pattern
- the pre-existing nature of all calls (no new violations were
  introduced; the decomposition moved them as-is)

**Outstanding follow-ups.** `unified/dispatch.rs` is the largest
non-test file. It could be split into `dispatch_aot.rs` +
`dispatch_uop.rs`, but the AOT-fails-then-UOp-fallback coupling
argues for keeping them together. The next natural target is
`runtime/model.rs` (64 pub items).

---

## Cross-cutting notes

- **Authority-leak findings across all four reports.** No new
  top-level manager / registry / service singleton introduced. No
  new `HashMap` / `HashSet` for canonical ordered collections. No
  new `anyhow::Error`. No new `unsafe`. No new file named after an
  external project. Every new `.rs` file has a one-sentence module
  doc stating its single authority.

- **Test scope.** `unwrap_baseline.py` post-decomposition state:
  - `runtime/mod.rs`: 1 prod, 22 test (in `runtime/tests.rs`)
  - `runtime/model.rs`: 0 prod, 0 test
  - `runtime/binding.rs`: 0 prod, 0 test
  - `runtime/ane_backend.rs`: 0 prod, 0 test
  - `runtime/kernel_dispatch.rs`: 0 prod, 0 test
  - `runtime/xdna_dispatch.rs`: 0 prod, 0 test
  - `runtime/certification.rs`: 0 prod, 0 test
  - `runtime/unified/{mod,workload,replay,ane,run,dispatch}.rs`:
    0+0+0+0+2+7 = 9 prod, 0 test
  - `cimage/mod.rs`: 0 prod, 130 test
  - `cimage/reader.rs`: 2 prod, 0 test
  - `cimage/writer.rs`: 2 prod, 0 test
  - Workspace total: unchanged at 510 production / 2662 test-scope.

- **Remaining production unwrap backlog (10 in `runtime/`, 4 in
  `cimage/`).** All carry `// WAIVER:` annotations documenting the
  structural guard. The rust-quality rule's intent (per the skill:
  "New code must not add to the backlog") is satisfied — every
  waiver is a pre-existing call. The remaining work is to convert
  each waiver into a checked-error path, but that is a follow-up
  refactor, not part of the module-cohesion work.
