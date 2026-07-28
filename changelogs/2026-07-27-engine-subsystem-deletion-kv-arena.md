# Goal: Delete `compute-core/src/ecs/kv_arena/`

**Date:** 2026-07-27 (Pacific)
**Status:** ✅ Goal achieved; engine's `compute-core/src/ecs/kv_arena/` deleted.
**Follow-up to:** `f2cfee80` (scheduling), `28a4ac14` (assistant-graph).
**Branch:** `migrate/kv-arena` (worktree: `prism-engine-kv-arena`).

## Source

`compute-core/src/ecs/kv_arena/` — 5 files, 933 LOC (deleted in E-4).

## Constitutional target

`crates/prism-kv-cache/src/arena/` (5 source modules + tests, 977 LOC).

## Migration sequence (kv-arena E-0..E-4)

| Step | Commit | Engine error count | Notes |
|------|--------|--------------------|-------|
| E-0: Audit engine dep — `prism-kv-cache` already present | (no-op) | 193 → 193 | engine already had the dep with `pub use prism_kv_cache;` |
| E-1: Add `prism-kv-cache::arena` constitutional surface | `d99b7083` | 193 | 10 tests pass on the constitutional side |
| E-2: Drop engine `lib.rs` re-export and `ecs/mod.rs` declaration | `db754467` | 193 | two tracked deletions; no external callers |
| E-3: Add architecture test `workspace_contains_no_legacy_kv_arena_imports` | `adad72f3` | 193 | parallel to the existing safety-net tests |
| E-4: `git rm -r compute-core/src/ecs/kv_arena/` | `6c67aa1a` | 193 | 5 files, 933 LOC removed |

## Success criteria — all met

- ✅ Every external engine caller (0 files) — the kv_arena module had no
  external callers; only the engine's own internal sibling imports
  referenced the module. `prism-ecs-server/src/runtime/kv.rs` has
  commented-out references to the engine surface that need no change
  (they are documentation stubs, not real imports).
- ✅ `git rm -r compute-core/src/ecs/kv_arena/` committed (E-4)
- ✅ Engine pre-existing build error count: 193 → 193 (unchanged)
- ✅ Architecture test: `workspace_contains_no_legacy_kv_arena_imports` green
- ✅ Constitutional surface tests: 10/10 pass (`cargo test -p prism-kv-cache --lib arena`)
- ✅ Architecture tests: 9/9 pass (`cargo test -p prism-architecture --lib`)
- ✅ Final `rg "use crate::ecs::kv_arena::" compute-core/src/` returns 0 results

## Engine callers (0)

This subsystem had **no external engine callers**. The only `use` statements
that referenced `crate::ecs::kv_arena::*` paths lived inside the kv_arena
directory itself (backend.rs, refcount.rs, and mod.rs use statements for
sibling imports). The only other workspace references are commented-out
documentation stubs in `crates/prism-ecs-server/src/runtime/kv.rs:609-610`,
which the safety-net test ignores because the `use` is on a `//` line.

The engine's re-export of `prism_kv_cache` (already present in
`compute-core/src/lib.rs:271`) continues to work: the constitutional
surface is now exposed via the same re-export path, so any future
caller can write `use prism_kv_cache;` or `use prism_kv_cache::arena;`
as appropriate.

## Constitutional surface (in `crates/prism-kv-cache/src/arena/`)

| File | Authority (one sentence per module-discipline rule) |
|---|---|
| `mod.rs` | canonical authority for the paged KV-cache arena facade: sequence identity, admission receipts, logical block tables, the arena error taxonomy, and the `KvBlockArena` aggregator that wires the subordinate authorities |
| `backend.rs` | canonical authority for backend-to-block residency mapping: which memory domain a block lives in and how much it costs in bytes, plus the table that records one entry per active block |
| `block.rs` | canonical authority for physical KV-cache block identity, capacity, atomic refcount, and last-access timestamp, plus the per-request logical block table that maps logical indices to physical block ids |
| `prefix.rs` | canonical authority for content-hash based prefix cache lookup: deterministic hashing of (model, tokenizer version, layer, token slice) tuples and the index that maps a hash to a physical block id with hit-rate accounting |
| `refcount.rs` | canonical authority for copy-on-write refcounting semantics and LRU eviction ordering of physical KV-cache blocks |
| `tests` (in mod.rs) | 5 arena-aggregator tests (admit, capacity, prefix hit, eviction, logical table) |

The 5 prefix tests live in `prefix.rs`'s own `#[cfg(test)] mod tests`:
test_prefix_hash_compute, test_from_tokens, test_prefix_cache_index,
test_remove_and_clear, test_hit_rate_zero_when_empty.

## Production-code changes vs. the engine

The constitutional port fixes two engine-side production panics in
`compute-core/src/ecs/kv_arena/mod.rs` to satisfy the no-panic
discipline:

| Engine | Constitutional |
|---|---|
| `let idx = self.free_list.pop().unwrap();` (line 192) | `let idx = self.free_list.pop().ok_or(ArenaError::EvictionInvariantBroken)?;` |
| `self.try_allocate().expect("KV arena: out of blocks even with eviction")` (line 240) | `self.try_allocate()?` — the typed `ArenaError` propagates to the caller |
| `allocate_prefixed` returns `PhysicalBlockId` (panics on OOM) | `allocate_prefixed` returns `Result<PhysicalBlockId, ArenaError>` |
| `block.rs:59` `SystemTime::duration_since(UNIX_EPOCH).unwrap()` | `.map(|d| d.as_nanos() as u64).unwrap_or(0)` — wall-clock pre-epoch is a degenerate case; saturating to 0 is safe |

A new `ArenaError::EvictionInvariantBroken` variant was added to the
error taxonomy to make the freed-slot path typed. The error is
unreachable in practice; it exists to give the no-panic rule a typed
alternative to the previous `unwrap`.

A minimal local `KvCachePlan` struct stands in for the engine's
`compute_image::kv_plan::KvCachePlan` — only the four fields the arena
reads (`block_tokens`, `max_blocks`, `eviction_policy`, `cow_policy`)
are present. A future migration may replace it with a constitutional
plan type when the engine absorbs its compute-image surface.

## Branch / worktree

This migration was completed on branch `migrate/kv-arena` in worktree
`/Users/user/Developer/GitHub/prism-engine-kv-arena` to avoid branch
contention with parallel agents on the main `/Users/user/Developer/GitHub/prism-engine`
worktree. Worktree was created from `main` at the engine-subsystem
deletion goal declaration merge (`0394c9ab`).

## Completion report

- **Affected subsystem:** `compute-core/src/ecs/kv_arena/` (engine, 5 files, 933 LOC)
- **`CAMPAIGN.md` status:** N/A (no prior status; new engine-deletion goal)
- **Canonical authority before:** engine file `compute-core/src/ecs/kv_arena/`
- **Canonical authority after:** `crates/prism-kv-cache/src/arena/`
- **Remaining writers:** none — kv_arena had no external callers
- **Transaction boundary:** unchanged; the arena is a pure-data structure (no world mutation, no event store writes)
- **Effect boundary:** unchanged
- **Durable schema changes:** none (kv_arena is an in-memory allocator; no schema key, no durable event)
- **Replay behavior:** unaffected (the engine never wrote canonical arena state to the event store)
- **Tests executed:**
  - `cargo test -p prism-kv-cache --lib arena` → 10/10 pass
  - `cargo test -p prism-architecture --lib` → 9/9 pass
  - `cargo check -p tribunus-compute-core --lib` → 193 pre-existing errors, unchanged
- **Authority-leak audit:** 0 external importers of `crate::ecs::kv_arena::` outside the migration inventory (verified by `workspace_contains_no_legacy_kv_arena_imports` after E-4)
- **Legacy path awaiting purge:** none — `compute-core/src/ecs/kv_arena/` is fully removed
- **Engine dep audit at E-0:** `prism-kv-cache` was already a workspace dep
  on the engine, with `pub use prism_kv_cache;` in
  `compute-core/src/lib.rs:271`. E-0 was therefore a no-op; the dep
  continues to be the only path by which any future engine caller
  reaches the constitutional surface.

## Pattern follow-up (parallel migrations)

- `core` — engine-deletion goal pending
- `backend` — engine-deletion goal pending
- `nf4tile640` — engine-deletion goal pending
- `memory` — engine-deletion goal pending

The `kv_arena` migration is the fourth engine-deletion in this series
after `scheduling` (commit `57081b28` E-15), `assistant_graph` (E-7),
and the various other migrations. The pattern (E-0..E-4 for kv_arena,
E-0..E-7 for ones with external callers) is now the proven recipe for
absorbing an engine subsystem into a constitutional crate.

## Safety record

- ✅ No `git reset`, `git stash`, `git checkout -- <file>`, or `mavis-trash` used.
- ✅ No edits to files outside the migration's scope.
- ✅ All 4 commits are bisectable: each touches only the files for its step.
- ✅ File-scoped recovery was available throughout (`git checkout
  migrate/kv-arena -- <file>` against any earlier commit).
- ✅ Checkpoint discipline: each of E-1..E-4 took < 5 minutes wall-clock,
  well inside the 30-minute budget.
- ✅ Constitutional-side `unwrap`/`expect` only appears in test code
  (the `mod tests` and `prefix::tests` modules) — production code
  follows the no-panic discipline and surfaces typed errors.
