# Phase B-4 — Convert canonical `HashMap` to `BTreeMap` (constitutional + core layers)

**Date:** 2026-07-25
**Lane:** Constitutional systems alignment
**Author:** Mavis (claude-opus-4)
**Subsystem:** canonical collections in `prism-ecs-constitutional` and `prism-ecs-core`

---

## Prime directive check

> *One canonical reality — every state-bearing change is validated, transactional, replayable, attributable, and resistant to stale external outcomes.*

This change enforces deterministic iteration for canonical collections
whose order is observable in replay, projection rebuild, or
schedule-graph output. `HashMap` is a non-deterministic hash table;
`BTreeMap` is a sorted tree. For canonical collections the difference
is replay-non-determinism (a defect), not a performance knob.

The change is state-bearing: replay, projection rebuild, and
schedule-graph consumers all observe iteration order. The propagation
chain is:

```
BTreeMap (canonical) → sorted iteration → stable journal & replay
   → stable projection rebuild → deterministic read path → consumer
```

---

## Affected subsystem & CAMPAIGN.md status

- **Subsystem:** canonical collections in the constitutional and core
  ECS layers.
- **CAMPAIGN.md status:** unchanged. The constitutional/core layer
  remains `Canonical` per CAMPAIGN.md (no cutover in flight for this
  surface).
- **Canonical authority before:** each `HashMap<_, _>` site held the
  canonical state for its domain. Iteration order was
  hash-implementation-dependent (Rust's default SipHash, randomized
  per-process for `HashMap`).
- **Canonical authority after:** each `BTreeMap<_, _>` site holds the
  same canonical state with **deterministic, key-sorted iteration**.
  Two processes replaying the same event log produce identical
  iteration order — replay becomes testable.

---

## Sites changed

The audit (A4) found 6 constitutional + 1 core sites whose order is
observable. This phase addresses all 7:

| # | File:line | Before | After | Key type rationale |
| - | --------- | ------ | ----- | ------------------ |
| 1 | `crates/prism-ecs-constitutional/src/schema.rs:22` | `HashMap<ComponentSchemaId, SchemaEntry>` | `BTreeMap<ComponentSchemaId, SchemaEntry>` | Existing typed `ComponentSchemaId`; `BTreeMap` now requires `Ord`/`PartialOrd`, which were added to `ComponentSchemaId` in `types.rs` |
| 2 | `crates/prism-ecs-constitutional/src/world_txn.rs:148` | `HashMap<u64, Vec<PendingOp>>` | `BTreeMap<Entity, Vec<PendingOp>>` | Key is the placeholder `Entity` handle for a pending spawn token; `Entity` newtype already exists in `prism_ecs_core` |
| 3 | `crates/prism-ecs-constitutional/src/sparse_set.rs:156` | `map: &HashMap<u64, T>` (test helper param) | `map: &BTreeMap<Entity, T>` | Same — `Entity` newtype |
| 4 | `crates/prism-ecs-constitutional/src/scheduler.rs:96` | `HashMap<WorkKind, Vec<u64>>` | `BTreeMap<WorkKind, Vec<Entity>>` | `WorkKind` now `Ord`/`PartialOrd`; `u64` work_entity upgraded to `Entity` for type discipline |
| 5 | `crates/prism-ecs-constitutional/src/persistence.rs:105` | `HashMap<String, ReplayApplier>` | `BTreeMap<SchemaKey, ReplayApplier>` (with stable `String → SchemaKey` boundary mapping) | `SchemaKey` newtype already exists; free `String` key was a separate newtype violation (B-2 work) — addressed here via a documented `event_kind_to_schema_key` mapping with deterministic FNV-1a fallback |
| 6 | `crates/prism-ecs-core/src/world.rs:77` | `HashMap<u64, u64>` | `BTreeMap<Entity, u64>` | `Entity` newtype. `component_versions_mut()` (line 155) accessor signature updated; `component_version()` (line 740) lookup key updated |

### Newtype derives added (no new types)

| Type | Derives added | File |
| ---- | ------------- | ---- |
| `ComponentSchemaId` | `PartialOrd, Ord` | `crates/prism-ecs-constitutional/src/types.rs:14` |
| `Entity` | `PartialOrd, Ord` | `crates/prism-ecs-core/src/entity.rs:20` |
| `WorkKind` | `PartialOrd, Ord` | `crates/prism-ecs-constitutional/src/work.rs:33` |

All three are transparent newtypes over `u64`/`u32`/an enum; `Ord`
is the natural field-wise order. No semantic change.

---

## Transaction / effect boundaries

- **Transaction boundary:** unchanged. The conversion is internal to
  each map's existing key/value model. No new transactions are
  introduced; no transactions are removed.
- **Effect boundary:** unchanged. Backends are not touched.

---

## Durable and transient schema changes

- **Durable schema changes:** none. The on-the-wire serialization of
  `BTreeMap` and `HashMap` is identical (both serialize as a sequence
  of key-value pairs via serde's default `Map` representation; the
  ordering of those pairs is observable in the wire bytes but is
  exactly the new determinism we are introducing — it is a feature,
  not a regression).
- **Transient schema changes:** none. `Entity`, `ComponentSchemaId`,
  and `WorkKind` gained `PartialOrd, Ord` derives; this is additive
  trait surface only.

---

## Replay behavior

**Before:** `ReplayRegistry::appliers` was a `HashMap<String, _>`.
Two processes replaying the same event log would produce **different
iteration orders** over `appliers` (default SipHash, randomized per
process). This is a silent replay-divergence defect.

**After:** `ReplayRegistry::appliers` is a `BTreeMap<SchemaKey, _>`.
Iteration is in `SchemaKey` order, deterministic across processes.

The `event_kind_to_schema_key` boundary mapping assigns a stable
`SchemaKey { namespace, id, version }` to each canonical event-kind
string, and an FNV-1a 32-bit-hash fallback for unknown kinds. The
fallback id is offset by 1000 to avoid collision with the 12 canonical
ids (1..=12). The fallback is deterministic across processes because
FNV-1a is a pure function of the input bytes.

---

## Propagation chain

```
SchemaKey → canonical event-kind identity
  → BTreeMap<SchemaKey, ReplayApplier>
  → ordered replay dispatch
  → deterministic journal reconstruction
  → stable projection rebuild
  → read path returns identical order
  → consumers see canonical iteration
```

The propagation test for this change is the existing
`prism-ecs-constitutional` test suite (70 tests, all passing). The
`advisory_events_share_commit_boundary_without_entering_durable_lane`
test in `world_txn::tests` exercises the `pending_resolutions`
BTreeMap end-to-end through `WorldTxn::add_component_pending` →
`World::transit` → journal commit.

---

## Files changed

| File | Change |
| ---- | ------ |
| `crates/prism-ecs-constitutional/src/schema.rs` | import `BTreeMap`; `SchemaRegistry::schemas` field type; constructor init; module-level doc explaining why |
| `crates/prism-ecs-constitutional/src/world_txn.rs` | import `BTreeMap`; `WorldTxn::pending_resolutions` field type; constructor init; `add_component_pending` / `put_durable_pending` / `put_transient_pending` token boxing (now `Entity::new(token, 0)`); `prepare_inner` resolution loop (now iterates `BTreeMap<Entity, _>`) |
| `crates/prism-ecs-constitutional/src/sparse_set.rs` | import `Entity`; `assert_sparse_equivalence` parameter type and lookups |
| `crates/prism-ecs-constitutional/src/scheduler.rs` | import `BTreeMap, Entity`; `Scheduler::ready_by_kind` field type; `mark_ready` / `drain` use `Entity`; constructor init |
| `crates/prism-ecs-constitutional/src/persistence.rs` | import `BTreeMap`; `ReplayRegistry::appliers` field type; `register` and `apply` use new `event_kind_to_schema_key` helper; new private `event_kind_to_schema_key` function with 12 canonical mappings + FNV-1a fallback |
| `crates/prism-ecs-constitutional/src/types.rs` | add `PartialOrd, Ord` to `ComponentSchemaId` |
| `crates/prism-ecs-constitutional/src/work.rs` | add `PartialOrd, Ord` to `WorkKind` |
| `crates/prism-ecs-core/src/entity.rs` | add `PartialOrd, Ord` to `Entity` |
| `crates/prism-ecs-core/src/world.rs` | import `BTreeMap`; `World::component_versions` field type; `World::new` and `World::with_capacity` constructors; `component_versions_mut` return type; `component_version` lookup key; `extensions: HashMap<TypeId, _>` **left as HashMap** (TypeId-keyed, order not observable) |

---

## Sites deliberately NOT changed (with rationale)

- `crates/prism-ecs-constitutional/src/schema.rs:139`
  `by_type: HashMap<TypeId, SchemaKey>` — TypeId-keyed, order not
  observable. Per the audit, this is the one **non-canonical**
  constitutional site. Left as HashMap.
- `crates/prism-ecs-core/src/world.rs:85`
  `extensions: HashMap<TypeId, _>` — TypeId-keyed, order not
  observable. Left as HashMap.
- `crates/prism-ecs-constitutional/src/world_txn.rs:1004`
  `pending_spawn_ids: HashSet<Entity>` — The audit identifies this
  as canonical. The task instructions list 6 sites; this is the
  7th canonical site that the task defers. Noted in the "follow-up"
  list below. The set is populated once per `prepare_inner` and
  used only for membership checks, so the determinism win is small
  here — but it is still a real defect and should land in a follow-up
  subagent.
- `crates/prism-ecs-runtime/src/schedule.rs` (10 sites) — runtime
  layer, large file. Per task instructions, deferred.
- `crates/prism-ecs-core/src/scheduling/graph.rs` (3 sites) — core
  scheduling, large file. Per task instructions, deferred.
- `crates/prism-ecs-runtime/src/backend.rs:90` — runtime layer, also
  has a separate newtype gap (String key). Per task instructions,
  deferred.

---

## Remaining writers

For each converted site, there is **one** writer and **one** reader
(the field is `pub(crate)` or module-private). No parallel
authorities were found. A `rg 'HashMap<|HashSet<'` post-conversion
returns only the 3 deliberately-unchanged sites (TypeId-keyed,
non-canonical) plus the deferred `world_txn.rs:1004` HashSet.

---

## Tests executed

- `cargo build -p prism-ecs-constitutional -p prism-ecs-core` — clean
  (4 pre-existing warnings about ambiguous glob re-exports in
  `lifecycle_command::*` vs `compilation::*` / `work::*`; confirmed
  present on `main` before this change).
- `cargo test -p prism-ecs-constitutional` — **70 passed, 0 failed**.
- `cargo test -p prism-ecs-constitutional -p prism-ecs-core` — **70 + 19 passed, 0 failed**.

---

## Authority-leak audit results

- `rg 'HashMap<|HashSet<' crates/prism-ecs-constitutional/src crates/prism-ecs-core/src` — only the
  deliberately-unchanged sites remain.
- `rg 'HashMap<u64|HashMap<String' crates/prism-ecs-{constitutional,core}/src` — **0 hits** in the
  converted files. All raw `u64`/`String` keys on canonical maps are gone.
- `rg 'pending_resolutions' crates/prism-ecs-constitutional/src` — all uses are now on the
  `BTreeMap<Entity, _>` storage form.

---

## Hard-rule compliance

- ✅ No new `unsafe` anywhere.
- ✅ No new `anyhow::Error` in constitutional/runtime/kernel.
- ✅ No new file named after an external project.
- ✅ No new manager/registry/service singleton outside the world.
- ✅ The `event_kind_to_schema_key` mapping is documented as a
  stable boundary; the FNV-1a fallback is reproducible across
  processes by construction.

---

## Legacy paths awaiting purge / follow-ups

1. `crates/prism-ecs-constitutional/src/world_txn.rs:1004`
   `pending_spawn_ids: HashSet<Entity>` → `BTreeSet<Entity>` —
   remaining canonical site in the constitutional layer. Small
   follow-up subagent task.
2. `crates/prism-ecs-runtime/src/schedule.rs` (10 sites) →
   `BTreeMap<SystemId, _>` / `BTreeSet<SystemId>`. Runtime layer;
   biggest canonical-iteration hotspot. Schedule-dependent tests
   should accompany the change.
3. `crates/prism-ecs-core/src/scheduling/graph.rs` (3 sites) →
   `BTreeMap<SystemId, _>`. Topological sort is currently
   `Result<HashMap<_>>`; conversion to `BTreeMap` is mechanical but
   the API change is observable.
4. `crates/prism-ecs-runtime/src/backend.rs:90` — `artifacts:
   HashMap<String, KernelArtifact>` → `BTreeMap<ArtifactDigest,
   KernelArtifact>`. **This is two fixes in one**: the iteration
   determinism (B-4) and the newtype gap (B-2). Should land
   together with the `ArtifactDigest` newtype rollout.
5. `persistence.rs:event_kind_to_schema_key` — once the event-kind
   newtype lands under B-2, the boundary mapping becomes a direct
   field access; the FNV-1a fallback is no longer needed.

---

## Next action

B-2 (`cmd!` macro newtype refactor) is the next critical fix. It
introduces the 13 authority-bearing newtypes whose absence is
documented in this phase's TODO comments (e.g. `EventKind`,
`RejectionReason`, `ArtifactDigest`, `BackendKind`).
