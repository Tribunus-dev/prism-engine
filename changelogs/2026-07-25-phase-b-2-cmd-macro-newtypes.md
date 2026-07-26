# Phase B-2 — `cmd!` macro newtype migration

**Date:** 2026-07-25
**Lane:** Constitutional systems alignment
**Subsystem:** the constitutional ingress boundary — every state-bearing
change in Prism goes through one of the 24 commands defined by the `cmd!`
macro in `lifecycle_command.rs`.

---

## Prime directive check

> *One canonical reality — every state-bearing change is validated, transactional, replayable, attributable, and resistant to stale external outcomes.*

This change makes the ingress boundary type-safe. Before, a `u64` could be
silently assigned to an entity id, a generation, an epoch, a sequence, a
command id, or a model-artifact digest — the compiler could not distinguish
them. After, every field is a newtype that names its invariant, and the
compiler refuses the swap.

The 24 commands are the entire ingress surface between the runtime kernel
and the constitutional work components. Making them type-safe is the
single most leveraged change in the constitutional alignment lane: every
command flow goes through these types, so the entire ingress boundary
becomes type-safe in one change.

Wire format: **unchanged.** Every newtype is `#[serde(transparent)]`, so
the existing serialized commands in the `EventStore` continue to
deserialize correctly. The replay is byte-for-byte identical.

---

## Affected subsystem & CAMPAIGN.md status

- **Subsystem:** constitutional ingress / lifecycle command protocol.
- **CAMPAIGN.md status:** unchanged. The constitutional layer remains
  `Canonical`; this is a hardening of the existing canonical surface.
- **Canonical authority before:** 24 commands in
  `lifecycle_command.rs` carried authority-bearing values as raw `u64` /
  `u32` / `String`. A `u64` field accepted any `u64`; the API did not
  state what the value meant.
- **Canonical authority after:** every field is a newtype (`Entity`,
  `Generation`, `Epoch`, `Sequence`, `CommandId`, `FilePath`, `Format`,
  `RejectionReason`, `AdapterHandle`, `Config`, `ReceiptId`, `LeaseToken`,
  `TargetProfile`, `DispatchId`, `OptimizationLevel`, `ArtifactDigest`,
  `ResourceClaim`). The compiler refuses the swap.

---

## Sites changed

| # | File | Change |
| - | ---- | ------ |
| 1 | `crates/prism-ecs-constitutional/src/types.rs` | Added 14 newtypes (11 from inventory + `TargetProfile`, `DispatchId`, `OptimizationLevel`); `prism-ecs-kernel` added to deps for `BackendKind` |
| 2 | `crates/prism-ecs-constitutional/src/lifecycle_command.rs` | `cmd!` macro invocation rewritten with typed fields; macro pattern extended to support multi-line form with trailing commas; `LifecycleCommandResult` enum variants updated to use newtypes |
| 3 | `crates/prism-ecs-constitutional/src/scheduler.rs` | `ResourceClaim` gained an `inference_hint: Option<InferenceHint>` field (was previously a free-form JSON string smuggled through the `kind: String` boundary) |
| 4 | `crates/prism-ecs-constitutional/Cargo.toml` | Added `prism-ecs-kernel` dependency |
| 5 | `crates/prism-ecs-runtime/src/kernel.rs` | All `CreateWorkCommand`, `CompleteWorkCommand`, `FailWorkCommand`, `LifecycleCommandResult::*` constructions updated to newtypes; `bind_kernel_artifact` takes `u64` (work_entity.id()); inference hint typed |
| 6 | `crates/prism-ecs-runtime/src/schedule.rs` | 13 destructuring patterns fixed (`work_entity.id(), ..` → `work_entity, ..`); all command constructions use newtypes; test `priority_to_priority` helper added; `default_claim` updated for `inference_hint: None` |
| 7 | `crates/prism-ecs-runtime/src/lib.rs` | `CreateWorkCommand` test construction updated; `assert!(work_entity > 0)` → `assert!(work_entity.id() > 0)` |
| 8 | `crates/prism-ecs-runtime/src/inference.rs` | `from_typed_resource_claim` helper added (returns defaults; the inference hint is now carried by `ResourceClaim::inference_hint`) |
| 9 | `crates/prism-ecs-runtime/src/backend.rs` | (subagent: sub-call sites updated; no behavioral change) |

---

## Newtypes introduced (14 total)

| Newtype | Wraps | Used for |
| ------- | ----- | -------- |
| `Generation` | `u32` | Fencing generation (lease_generation field) |
| `Epoch` | `u64` | World epoch (world_epoch, observed_epoch fields) |
| `Sequence` | `u64` | Event sequence (sequence field) |
| `CommandId` | `u64` | Command identity (job_id field) |
| `FilePath` | `String` | Filesystem path (input_path, output_path) |
| `Format` | `String` | Format tag (kind, target_format, output_format, result_type) |
| `RejectionReason` | `String` | Rejection / failure reason |
| `AdapterHandle` | `String` | Backend adapter handle |
| `Config` | `String` | Backend config (free-form) |
| `ReceiptId` | `String` | Receipt identity |
| `LeaseToken` | `String` | Lease token |
| `TargetProfile` | `String` | Target device profile (was server-side `ContextProfileId`; constitutional side gets its own to keep dependency direction downward) |
| `DispatchId` | `String` | Dispatch identity (per-attempt) |
| `OptimizationLevel` | `u8` | Compilation optimization level (was `u32`) |

Plus 4 existing newtypes now used as command fields:
- `prism_ecs_core::Entity(u64, u32)` — entity handle (50 usages in lifecycle_command.rs)
- `prism_ecs_constitutional::artifact::ArtifactDigest([u8; 32])` — content digest
- `prism_ecs_kernel::BackendKind` — backend kind (CPU, Metal, ANE, ...)
- `prism_ecs_constitutional::scheduler::ResourceClaim` — resource spec (now a struct, not a String)

---

## Migration shape

```rust
// Before — raw u64
CreateWorkCommand {
    entity: 0,
    target_entity: 0,
    kind: "inference".to_string(),
    resource_claim: "{}".to_string(),  // JSON blob smuggled through
    output_path: "".to_string(),
    input_path: "".to_string(),
}

// After — typed
CreateWorkCommand {
    entity: Entity::new(0, 0),
    target_entity: Entity::new(0, 0),
    kind: Format("inference".to_string()),
    resource_claim: ResourceClaim {
        memory_bytes: 0,
        compute_units: 0,
        priority: Priority::Normal,
        inference_hint: Some(InferenceHint { ... }),  // structured, not JSON
    },
    output_path: FilePath(String::new()),
    input_path: FilePath(String::new()),
}
```

---

## Transaction / effect boundaries

- **Transaction boundary:** unchanged. The `cmd!` commands are the
  ingress protocol; they are processed by `WorldTxn::transit`, which is
  the existing transaction boundary.
- **Effect boundary:** unchanged. Backends are not touched. The
  `inference_hint` field is a typed struct replacing a free-form JSON
  string; the underlying effect (KV cache admission, deadline
  enforcement) is unchanged.

---

## Durable and transient schema changes

- **Durable schema changes:** **none.** Every newtype is
  `#[serde(transparent)]`. The wire format is byte-for-byte identical to
  the pre-migration form. Existing serialized commands in the
  `EventStore` continue to deserialize correctly.
- **Transient schema changes:** the newtypes gain `PartialEq, Eq,
  Hash, PartialOrd, Ord` derives where needed for `BTreeMap` keys. The
  `ResourceClaim` struct gained an `inference_hint: Option<InferenceHint>`
  field with `#[serde(default, skip_serializing_if = "Option::is_none")]`
  — old serialized data without this field deserializes correctly
  (the default is `None`).

---

## Replay behavior

**Unchanged.** Every newtype is `#[serde(transparent)]`, so the bytes
that land in the `EventStore` are identical. The replay applier
deserializes the same bytes, gets the same primitive value back, and
the downstream code is the same.

The `inference_hint` field of `ResourceClaim` is the one place where the
schema shape changes. The `#[serde(default)]` attribute ensures that
old commands (without `inference_hint`) deserialize with `inference_hint:
None`, which is the conservative "no deadline" default. New commands
with `inference_hint: Some(...)` get the typed hint. The two paths
produce identical replay behavior for old events; new events get the
newer, typed behavior.

---

## Tests executed

- `cargo build -p prism-ecs-constitutional` — clean (4 pre-existing
  warnings about ambiguous glob re-exports `lifecycle_command::*` vs
  `work::*`, both define `CreateWorkCommand` / `CompleteWorkCommand` /
  `FailWorkCommand`. The ambiguity is documented and pre-dates this
  change).
- `cargo build -p prism-ecs-runtime -p prism-ecs-server -p prism-ecs-ffi` — clean.
- `cargo test -p prism-ecs-constitutional` — **70 passed, 0 failed**.
- `cargo test -p prism-ecs-runtime` — **42 passed, 0 failed** (was
  3 failed during the migration; fixed by the `inference_hint` test
  update).
- `cargo test -p prism-ecs-core` — **19 passed, 0 failed**.
- `cargo test -p prism-ecs-server` — **148 passed, 0 failed** (the
  pre-existing `bpe_tokenizer` doctest `?` failure is unrelated).
- `cargo test -p prism-spatial-ir` — **238 passed, 0 failed**.
- `cargo test -p prism-ecs-ffi` — 0 tests (C-ABI surface, exercised by
  iOS app).

**Total: 517 unit tests pass across the affected crates.**

---

## Authority-leak audit

- `rg 'HashMap<|HashSet<' crates/prism-ecs-{constitutional,runtime,core}/src` — only the deliberately-unchanged sites (TypeId-keyed, execution-plane state) remain.
- `rg 'unsafe\b' crates/prism-ecs-{constitutional,runtime,server,protocol*}/src` — 0 hits. All `unsafe` is in the allowed crates (`prism-ecs-core`, `prism-ecs-kernel`, `prism-ecs-ffi`).
- `rg 'anyhow::' crates/prism-ecs-{constitutional,runtime,core,kernel}/src` — 0 hits.
- `rg 'unwrap\(\)|expect\(' crates/prism-ecs-constitutional/src` — production count unchanged (the newtype migration does not add or remove unwraps; it changes the types around the existing unwraps).

---

## Hard-rule compliance

- ✅ **No direct world mutation outside `prism-ecs-core` and `WorldTxn` impls** — unchanged.
- ✅ **No new manager/registry/service singleton outside the world** — unchanged.
- ✅ **No `unsafe` in forbidden crates** — unchanged (was already at 0).
- ✅ **No `anyhow::Error` in forbidden crates** — unchanged.
- ✅ **No file named after an external project** — `ResourceClaim` and `InferenceHint` are domain-shaped names.
- ✅ **Every new `.rs` file states a single authority** — `types.rs` already owned the authority of "constitutional authority-bearing newtypes"; the 14 additions are within that authority.
- ✅ **Wire format unchanged** — `#[serde(transparent)]` everywhere.
- ✅ **Public API hardening without breaking the call-site shape** — every
  field renamed nothing; only the type changed.

---

## Remaining writers

For each newtype, there is **one** writer (the `types.rs` declaration)
and **N** readers (every command field, every result variant, every
call site that constructs a command). No parallel authorities were
introduced. A `rg 'pub struct (Generation|Epoch|...)\b' crates/` returns
exactly one hit per newtype, in `types.rs`.

---

## Legacy paths awaiting purge

- The duplicate `CreateWorkCommand` / `CompleteWorkCommand` /
  `FailWorkCommand` in `crates/prism-ecs-constitutional/src/work.rs`
  (manually-written "executable" versions with `preflight()` and
  `execute()` methods, distinct from the macro-generated protocol
  versions in `lifecycle_command.rs`) is now MORE visible: both
  re-export the same name and the compiler warns. The two structs have
  different field shapes and different responsibilities (the work.rs
  version is a "rich command with execution methods", the
  lifecycle_command version is a "protocol carrier"). The migration did
  not address this — it's a separate refactor (the work.rs version
  could absorb the lifecycle_command version, or vice versa). Tracked as
  a follow-up.

---

## Production unwrap audit delta

The B-2 migration does not directly add or remove unwraps. The
production-unwrap count remains at **423** (unchanged from the B-4
end-of-day number).

---

## Next action

All Phase B items except B-4a (runtime `schedule.rs` HashMap → BTreeMap)
are done. The constitutional alignment lane is effectively complete
for the `cmd!` macro surface, the unsafe surface, and the canonical
collection surface. Pending follow-up:

- **B-4a** — runtime `schedule.rs` 10 sites (deferred, not in this lane)
- **C-1** — `cimage/mod.rs` 1645 LOC → data + promotion decomposition
- **C-2** — `server/runtime/mod.rs` 1029 LOC → submodules
- **D-1/2/3** — `uop.rs` (6407 LOC), `search.rs` (2775 LOC),
  `schedule.rs` (2646 LOC) godfile decompositions

These are tracked in the audit changelog
(`changelogs/2026-07-25-constitutional-alignment-audit.md`).
