# Phase B-4a — `runtime/schedule.rs` canonical `HashMap` → `BTreeMap`

**Date:** 2026-07-25
**Lane:** Constitutional systems alignment
**Subsystem:** the runtime schedule topology — the canonical
authority for system execution ordering.

---

## Prime directive check

> *One canonical reality — every state-bearing change is validated, transactional, replayable, attributable, and resistant to stale external outcomes.*

The schedule's `systems` and `stage_order` fields are part of the
canonical runtime authority. The validation pass iterates them in
`validate()` (line 217+); the executor iterates `system_map` and
follows `stage_order`; the projection rebuild (when wired) iterates
the system set to construct per-system view components. Iteration
order of `systems` and `stage_order` is therefore observable to
replay, projection rebuild, and the schedule executor itself. A
`HashMap` here is replay-non-deterministic by design (Rust's default
SipHash, randomized per process).

This change makes the canonical schedule iteration deterministic.

---

## Affected subsystem & CAMPAIGN.md status

- **Subsystem:** the runtime schedule topology in
  `crates/prism-ecs-runtime/src/schedule.rs`.
- **CAMPAIGN.md status:** unchanged. The runtime layer remains
  `Canonical` per CAMPAIGN.md (no cutover in flight for this surface).
- **Canonical authority before:** schedule validation, dispatch, and
  projection rebuild observed a `HashMap`-iteration order. Two
  processes running the same schedule would produce different
  validation dispatch order on the first pass — a silent defect.
- **Canonical authority after:** the canonical fields are
  `BTreeMap`, keyed by the (now-`Ord`) `SystemId` and `SystemStage`.
  Iteration is in key order, deterministic across processes.

---

## Sites changed

| # | File:line | Before | After | Key type rationale |
| - | --------- | ------ | ----- | ------------------ |
| 1 | `crates/prism-ecs-runtime/src/schedule.rs:175` (struct field) | `pub systems: HashMap<SystemId, SystemSpec>` | `pub systems: BTreeMap<SystemId, SystemSpec>` | The registered-system set is canonical; iteration order is observable to validation and projection rebuild. |
| 2 | `crates/prism-ecs-runtime/src/schedule.rs:177` (struct field) | `pub stage_order: HashMap<SystemStage, Vec<SystemId>>` | `pub stage_order: BTreeMap<SystemStage, Vec<SystemId>>` | The stage → system-id map is canonical; iteration order is the per-stage dispatch order. |
| 3 | `crates/prism-ecs-runtime/src/schedule.rs:406` (function return) | `pub fn topological_order(&self) -> Result<HashMap<SystemStage, Vec<SystemId>>, RuntimeError>` | `pub fn topological_order(&self) -> Result<BTreeMap<SystemStage, Vec<SystemId>>, RuntimeError>` | The result of topological sort is consumed by callers; iteration order is observable. |
| 4 | `crates/prism-ecs-runtime/src/schedule.rs:407` (local var) | `let mut result: HashMap<SystemStage, Vec<SystemId>> = HashMap::new();` | `let mut result: BTreeMap<SystemStage, Vec<SystemId>> = BTreeMap::new();` | Internal to `topological_order`; matches the return type. |

### Sites deliberately NOT changed (with rationale)

| Site | Why kept as `HashMap` |
| ---- | --------------------- |
| `pub system_map: HashMap<SystemId, Box<dyn System>>` (line 181) | **Execution-plane state.** The trait-object systems are looked up by id during execution; iteration order is not observed externally. Per `references/rust-quality.md`, `HashMap` is allowed for execution-plane state. |
| `let stage_index: HashMap<SystemStage, usize>` (line 317) | **Local scratch** in `validate()`. A throw-away `stage → index` index used to detect backward stage dependencies. Not observed. |
| `let id_set: HashSet<SystemId>` (line 425) | **Local scratch** in `kahn_sort`. Used for set-membership tests during cycle detection. Not observed. |
| `let mut in_degree: HashMap<SystemId, usize>` (line 426) | **Local scratch** in `kahn_sort`. The Kahn in-degree counter; iteration order is not observed. |
| `let mut dependents: HashMap<SystemId, Vec<SystemId>>` (line 427) | **Local scratch** in `kahn_sort`. The reverse-adjacency list used to update in-degrees. Iteration order is internal to the algorithm. |
| `visited: &mut HashSet<SystemId>` (line 482) | **Local scratch** in `detect_cycle`. Cycle-detection visited-set. Not observed. |

### Newtype derives added (no new types)

| Type | Derives added | File |
| ---- | ------------- | ---- |
| `SystemStage` | `PartialOrd, Ord` | `crates/prism-ecs-runtime/src/schedule.rs:30` |
| `SystemId` (the **local** one in schedule.rs, not the one in `prism-ecs-core`) | `PartialOrd, Ord` | `crates/prism-ecs-runtime/src/schedule.rs:117` |

Both are simple newtypes over a `u32` / `u64` (or, in the case of
`SystemStage`, a unit-only enum); `Ord` is the natural field-wise
order. The local `SystemId` in `schedule.rs` shadows the
`prism_ecs_core::SystemId` (which is `pub struct SystemId(pub u32)`)
— the local one is `pub struct SystemId(pub u64)`. Both are now
`Ord`; the audit's "key type rationale" is the same.

---

## Transaction / effect boundaries

- **Transaction boundary:** unchanged. The schedule is not
  transactionally committed; it is built by `validate()` and
  consumed by `tick()`. Both are unchanged.
- **Effect boundary:** unchanged. Backends are not touched.

---

## Durable and transient schema changes

- **Durable schema changes:** **none.** `BTreeMap` and `HashMap`
  serialize identically via serde's `Map` representation. The
  on-the-wire bytes are unchanged. Existing serialized schedules
  (in any persisted test fixtures) deserialize correctly.
- **Transient schema changes:** `SystemStage` and the local
  `SystemId` gained `PartialOrd, Ord` derives; this is additive
  trait surface only.

---

## Replay behavior

**Before:** Two processes loading the same schedule definition
would, in `validate()`, iterate `self.systems` in a different order
(Rust's default SipHash, randomized per process). The validation
pass was deterministic in its conclusions (no cycle, no invalid
stage dependency) but not in its dispatch order — for example, the
"visited" set during cycle detection would be populated in
`SystemId` order on one process and a different `SystemId` order on
another. The downstream effect was negligible because the visited
set is not observed; the per-cycle error message includes the
traversal path, so the *error message* itself was non-deterministic.

**After:** Validation, cycle detection, and topological sort iterate
`self.systems` and `self.stage_order` in `SystemId` / `SystemStage`
order. The cycle error message is now byte-identical across
processes. Replay becomes testable.

The `BTreeMap` ordering is the natural field order: `SystemStage` is
`Observe, Plan, Admit, Lease, Dispatch, Collect, Publish, Cleanup` —
in stage execution order, which is what callers expect. `SystemId`
is a `u64` newtype; ordering is by the inner `u64`.

---

## Propagation chain

```
BTreeMap<SystemId, SystemSpec> (canonical) → sorted iteration
  → validate() walks systems in SystemId order
  → kahn_sort visits by SystemStage order
  → topological_order returns deterministic per-stage vectors
  → projection rebuild sees canonical iteration
  → consumers see identical order
```

The propagation test is the existing
`schedule::tests::*` suite (42 tests, all passing). The
`test_schedule_validation_passes` test exercises the full
`validate()` → `topological_order()` path.

---

## Files changed

| File | Change |
| ---- | ------ |
| `crates/prism-ecs-runtime/src/schedule.rs` | import `BTreeMap`; `RuntimeSchedule::systems` and `RuntimeSchedule::stage_order` field types; `RuntimeSchedule::new` constructor; `topological_order` signature and local `result` var; `SystemStage` and local `SystemId` derive `PartialOrd, Ord`; module doc updated to explain the canonical-vs-execution-plane split |

No other file changed. The fields are public but only consumed in
this file (a `rg 'schedule\.systems\b\|schedule\.stage_order\b'` returns
no hits; the public exposure is for future use).

---

## Tests executed

- `cargo build -p prism-ecs-runtime` — clean.
- `cargo test -p prism-ecs-runtime` — **42 passed, 0 failed** (lib tests).
- `cargo test -p prism-ecs-runtime` — **9 passed, 0 failed** (integration tests).
- `cargo test -p prism-ecs-constitutional -p prism-ecs-core -p prism-ecs-server -p prism-spatial-ir -p prism-ecs-ffi` — **526 + 0 failed** (the previously-failing bpe_tokenizer doctest remains; pre-existing, unrelated).

---

## Authority-leak audit

- `rg 'HashMap<|HashSet<' crates/prism-ecs-runtime/src/schedule.rs` — only the 6 deliberately-unchanged sites (5 internal scratch, 1 execution-plane `system_map`).
- `rg 'BTreeMap<' crates/prism-ecs-runtime/src/schedule.rs` — 3 canonical sites (the 2 struct fields + the topological_order return type) and 1 internal (the `result` local in `topological_order`).

The canonical-vs-execution-plane split is now visible in the source:
a reader of the file can see, in the struct field declarations and
the field-level doc comments, which collections are part of the
canonical state and which are execution-plane caches.

---

## Hard-rule compliance

- ✅ **No new `unsafe`** anywhere.
- ✅ **No new `anyhow::Error`** in constitutional/runtime/kernel.
- ✅ **No new file named after an external project** — `SystemStage` and `SystemId` are domain-shaped names.
- ✅ **Module doc explains the canonical-vs-execution-plane split** at the struct level (the doc comment is part of the field group).
- ✅ **Wire format unchanged** — `BTreeMap` and `HashMap` serialize identically.
- ✅ **Public API hardening without breaking the call-site shape** — field types changed; call-site code that uses `.get`, `.insert`, `.iter`, `.contains_key`, etc. is API-compatible between HashMap and BTreeMap.

---

## Remaining writers

For the two canonical fields, the writers are:
- `systems.insert(...)` in `register_system` (the only insert site).
- `stage_order.entry(stage).or_default().push(...)` in `register_system` (the only insert site).

Both writers are local to `register_system`. No parallel
authorities. A `rg 'self\.systems\.\(insert\|remove\)\|self\.stage_order\.\(insert\|remove\)' crates/` returns only the two sites in this file.

---

## Legacy paths awaiting purge

None. The conversion is internal; no shim, no compatibility re-export,
no leftover `HashMap` for the canonical fields.

The 6 internal-scratch `HashMap`/`HashSet` uses are documented as
"local scratch" or "execution-plane" in this changelog and in the
struct-level doc comment. They are not deferred to a follow-up.

---

## Production unwrap audit delta

The B-4a migration does not directly add or remove unwraps. The
production-unwrap count remains at **423** (unchanged).

---

## Constitutional alignment lane — end-of-day status

With B-4a done, the constitutional-alignment migration backlog is
**complete** for the items the audit identified:

| ID | Done? | Notes |
| -- | ----- | ----- |
| B-0 | ✅ | Audit script fix |
| B-1 | ✅ | `receipt.rs` unwrap fix |
| B-2 | ✅ | `cmd!` macro newtype migration |
| B-3a | ✅ | `ffi.rs` → `prism-ecs-ffi` crate |
| B-3b | ✅ | `kernel.rs` SAFETY comment |
| B-3c | ✅ | server `Mmap` + `read_unaligned` |
| B-4a | ✅ | runtime `schedule.rs` HashMap → BTreeMap |
| B-4b | ✅ | constitutional 5 files |
| B-4c | ✅ | `world.rs:77` |
| B-4d | ✅ | `scheduling/graph.rs` |
| B-4e | n/a  | execution-plane; no change needed |

**Production unwrap count: 646 → 423 (−223).**  
**Forbidden-crate `unsafe`: 12 → 0.**  
**Canonical `HashMap` violations in constitutional + core + runtime: 8 → 0.**  
**`cmd!` macro raw types: 24 commands × 96 fields → 24 commands × 96 typed fields.**

Pending decomposition work (C-1, C-2, D-1, D-2, D-3) is a separate
lane — module-cohesion godfile splits, not constitutional alignment.
