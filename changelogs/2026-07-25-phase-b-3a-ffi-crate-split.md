# Phase B-3a — Move `prism-ecs-constitutional/src/ffi.rs` to a new `prism-ecs-ffi` crate

**Date:** 2026-07-25
**Lane:** Constitutional systems alignment
**Author:** Mavis (claude-opus-4)
**Subsystem:** `prism-ecs-constitutional` ↔ `prism-ecs-ffi` (C-ABI bridge)

---

## Prime directive check

> *One canonical reality — every state-bearing change is validated, transactional, replayable, attributable, and resistant to stale external outcomes.*

This change removes `unsafe` from a forbidden crate (constitutional). The
8 raw-pointer reads/writes move into a dedicated `prism-ecs-ffi` crate
whose single authority is the C-ABI bridge. The constitutional crate
stays `unsafe`-free; the FFI crate is the only place in the workspace
where `extern "C"` raw-pointer signatures live.

No state-bearing change. No replay impact. No transaction-boundary shift.
The change is layer-internal.

---

## Affected subsystem & CAMPAIGN.md status

- **Subsystem:** the C-ABI bridge from `PrismAgentiOS` (Swift) into the
  constitutional ECS.
- **CAMPAIGN.md status:** unchanged. This is a layer-internal
  decomposition, not a subsystem migration. The `PrismAgentiOS`
  subsystem remains in whatever status it is in (no entry in CAMPAIGN.md
  for this layer — tracked via the constitutional alignment audit
  backlog).
- **Canonical authority before:** the FFI module lived at
  `crates/prism-ecs-constitutional/src/ffi.rs` and was a sibling of the
  other constitutional modules.
- **Canonical authority after:** the FFI module lives at
  `crates/prism-ecs-ffi/src/c_abi.rs`, re-exported at the crate root.
  The constitutional crate has no FFI surface.

---

## Transaction / effect boundaries

- **Transaction boundary:** none. The FFI functions do not open a
  transaction directly — they delegate to `WorldTxn::new` /
  `World::transit`, which is the existing constitutional boundary. No
  change to that boundary.
- **Effect boundary:** unchanged. Backends are not touched. The FFI
  functions only mutate the in-process `World` (and, via
  `agent_state::tick`, the agent state machine).

---

## Durable and transient schema changes

- **Durable schema changes:** none.
- **Transient schema changes:** none. The function signatures are
  byte-for-byte identical to the prior C-ABI.

---

## Replay behavior

Unchanged. The FFI module does not participate in replay — replay
happens via `ReplayRegistry` in `persistence.rs`, which is independent
of the FFI surface.

---

## Files

| Action | Path | Notes |
| ------ | ---- | ----- |
| created | `crates/prism-ecs-ffi/Cargo.toml` | minimal manifest, depends on `prism-ecs-constitutional` + `prism-ecs-core` + `parking_lot`; no `unsafe_code = "deny"` (unsafe is required) |
| created | `crates/prism-ecs-ffi/src/lib.rs` | module doc stating the single-authority rule; re-exports `c_abi::*` |
| created | `crates/prism-ecs-ffi/src/c_abi.rs` | the moved `ffi.rs` contents; 8 `unsafe` sites, each with a `// SAFETY:` comment |
| deleted | `crates/prism-ecs-constitutional/src/ffi.rs` | (recovered via `mavis-trash`) |
| modified | `crates/prism-ecs-constitutional/src/lib.rs` | removed `pub mod ffi;`; removed `pub use ffi::*;`; replaced with a comment pointing callers at `prism_ecs_ffi` |
| modified | `Cargo.toml` | added `"crates/prism-ecs-ffi"` to `[workspace] members` |

The deleted file's full content is preserved in the git history for
this commit (if the orchestrator commits per phase, the recovery is
in `reflog`/`fsck` either way).

---

## SAFETY-comment audit

Every `unsafe` site is documented. The reason references the C-ABI
caller contract that the public header (`PrismAgentiOS/include/PrismAgentFFI.h`,
out-of-tree) defines:

| Site | Function | Why safe |
| ---- | -------- | -------- |
| `c_abi.rs:106` (`Box::from_raw`) | `prism_world_destroy` | Caller contract: non-null pointer from `prism_world_create`, no double-free. Null checked above. Pointee type matches `Mutex<World>`. |
| `c_abi.rs:147` (`&mut *world`) | `prism_subagent_spawn` | Same world-pointer contract as `prism_world_destroy`. |
| `c_abi.rs:155` (`CStr::from_ptr`) | `prism_subagent_spawn` | Caller contract: non-null `*const c_char` is a valid null-terminated C string whose lifetime extends past the function return. Converted to owned `String` immediately so we never dereference past the boundary. |
| `c_abi.rs:179` (`&mut *world`) | `prism_subagent_phase` | Same world-pointer contract. |
| `c_abi.rs:200` (`&mut *world`) | `prism_subagent_lifecycle` | Same world-pointer contract. |
| `c_abi.rs:222` (`&mut *world`) | `prism_subagent_cancel` | Same world-pointer contract. |
| `c_abi.rs:241` (`&mut *world`) | `prism_agent_tick` | Same world-pointer contract. |
| `c_abi.rs:264` (`CString::from_raw`) | `prism_free_string` | Caller contract: non-null pointer from `CString::into_raw` inside this module, not yet released (no double-free). Pointee type matches `CString`. |

The `Cargo.toml` deliberately does **not** set `[lints.rust] unsafe_code = "deny"`.
The crate is the only place in the workspace where `unsafe` is permitted
for the C-ABI bridge; turning the lint on would forbid the entire
purpose of the crate.

---

## Remaining writers

The FFI surface has exactly **one** writer — the C-ABI module. No
duplicated authorities remain in the workspace. A `rg 'prism_ecs_constitutional::ffi'`
search returns no hits, confirming no caller relied on the old
`prism_ecs_constitutional::ffi` path. The Swift-side iOS target
(`PrismAgentiOS/`) has its own header-based include path that links
the symbols by name, not by Rust module path, so the move is invisible
to the Swift caller.

---

## Tests executed

- `cargo build -p prism-ecs-ffi` — clean (one pre-existing-style warning
  about `WorldTxn::new(&mut w)` taking a mutable reference, inherited
  from the original `ffi.rs`).
- `cargo build -p prism-ecs-constitutional` — clean (4 pre-existing
  warnings about ambiguous glob re-exports in `lifecycle_command::*`
  vs `compilation::*` / `work::*`; confirmed present on `main` before
  this change by `git stash` + rebuild).
- `cargo build -p prism-ecs-constitutional -p prism-ecs-core -p prism-ecs-runtime -p prism-ecs-ffi` — all clean.
- `cargo test -p prism-ecs-ffi` — 0 tests in the new crate (no test
  infrastructure required for a 7-function C-ABI bridge; the FFI
  surface is exercised by the iOS app, not by unit tests).
- `cargo test -p prism-ecs-constitutional` — **70 passed, 0 failed**.
- `cargo test -p prism-ecs-constitutional -p prism-ecs-core` — **70 + 19 passed, 0 failed**.

`cargo test --workspace` was attempted but failed at the link step on
`prism-ecs-compile` (lib test) due to missing `tribunus_arena_*` C
symbols in the `prism-ane` library. This is a **pre-existing** linker
issue (C library symbol export from `prism-ane`'s Objective-C
artefacts) and is independent of this change; it is reproduced on
`main` without these edits. Documented for the next session.

---

## Authority-leak audit results

- `rg 'unsafe\b' crates/prism-ecs-constitutional/src` — **0 hits**.
  The 8 `unsafe` sites have all moved to `prism-ecs-ffi`.
- `rg 'unsafe\b' crates/prism-ecs-ffi/src` — **8 hits**, all with
  `// SAFETY:` comments.
- `rg 'prism_ecs_constitutional::ffi' .` — **0 hits** (no caller
  depended on the old module path).
- `rg 'prism_world_create|prism_subagent_spawn|prism_agent_tick' .` —
  definition hits only, no external callers in the Rust workspace.

---

## Hard-rule compliance

- ✅ No new `unsafe` anywhere except the moved `ffi.rs` (now in
  `prism-ecs-ffi`).
- ✅ No new `anyhow::Error` in constitutional/runtime/kernel.
- ✅ No new file named after an external project.
- ✅ No new manager/registry/service singleton outside the world.
- ✅ Module doc on the new `lib.rs` states the single authority in
  one sentence (the C-ABI bridge).
- ✅ Every `unsafe` site has a `// SAFETY:` comment naming the
  invariant (C-ABI caller contract per the public header).

---

## Legacy paths awaiting purge

None. The `ffi.rs` content is fully moved; no shim, no `pub use
prism_ecs_ffi::*;` re-export back into the constitutional crate was
added (callers in the constitutional crate's own test suite — none
found — would have to import directly from `prism_ecs_ffi`).

---

## Next action

Phase B-4 (canonical `HashMap` → `BTreeMap` in the constitutional and
core layers) is the next fix in the backlog. See
`changelogs/2026-07-25-phase-b-4-btreemap-canonical-collections.md`.
