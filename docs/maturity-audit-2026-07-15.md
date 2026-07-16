# Prism Engine Maturity Audit — Consolidated Report

**Generated:** 2026-07-15  
**Auditors:** UnsafeAudit, PanicAudit, MemoryAudit, EntityAudit, ConcurrencyAudit, ApiAudit  
**Scope:** ~20 crates, 200+ source files

## Summary

| Category | CRITICAL | HIGH | MEDIUM | LOW |
|---|---|---|---|---|
| Unsafe code | 4 | 22 | 18 | — |
| Panic/error paths | 37 | 28 | ~645 | — |
| Memory safety | 3 | 8 | 7 | — |
| Entity safety | 2 | 2 | 1 | 2 |
| Concurrency | 2 | 5 | 3 | 3 |
| API exposure | 3 | 5 | 3 | 3 |
| **Total** | **51** | **70** | **~677** | **8** |

---

## CRITICAL Findings (blockers for hardware testing)

### 1. Panic paths — 37 CRITICAL unwrap() calls in production paths

**Impact:** Any of these crashes the process. Most are on user-supplied HHVM/HuggingFace model data, network IO, or FFI calls.

**Hotspots:**
- `compute-core/src/ecs/system/` — unwrap on file IO, network downloads, model loading
- `compute-core/src/ecs/backend/` — unwrap on FFI/ANE/Metal calls
- `prism-mcpd/src/` — unwrap on socket accept, JSON parse
- `crates/prism-ecs-core/src/` — unwrap in query.rs, world.rs

**Fix:** Replace with `?` operator, proper error types, or `unwrap_or_else` with context. ~2 hours to fix all critical sites if dispatched.

### 2. Metal buffer integer overflow — CRITICAL

**File:** `crates/prism-ecs-backend/src/metal.rs:310`  
**Pattern:** `(op.m as usize * op.n as usize * sizeof(f32))`  
**Danger:** Multiplying untrusted tensor dimensions without overflow check. On release builds (no overflow checks), wraps to small value → MTLBuffer allocation succeeds with wrong size → GPU writes out of bounds → GPU memory corruption, system GPU reset, or kernel panic.

**Fix:** Use `checked_mul()` or `saturating_mul()` with fallback.

### 3. Arena use-after-lock — CRITICAL

**File:** `crates/prism-ecs-backend/src/accelerate/ops.rs:246-248`  
**Pattern:** Arena pointer obtained inside lock, computed with after lock released  
**Danger:** Raw pointer to arena memory is computed while holding a lock, then dereferenced after the lock guard drops. If another thread allocates in the arena between release and use, the pointer is dangling.

**Fix:** Extend the lock guard scope to cover the pointer use, or copy the data out.

### 4. DuckDB transmute_copy UB — CRITICAL

**File:** `crates/prism-ecs-duckdb/src/columnar.rs:86-90`  
**Pattern:** `transmute_copy::<String, T>(&value)`  
**Danger:** `String` is 24 bytes on 64-bit (ptr+len+cap). If T is not exactly 24 bytes and the right layout, this is immediate UB. DuckDB values are not guaranteed to match Rust String layout.

**Fix:** Use proper serialization (serde or manual field-by-field copy), not transmute_copy.

### 5. Entity fabrication in serde — CRITICAL

**File:** `crates/prism-ecs-ir/src/serde.rs`  
**Pattern:** `Entity(id, 0)` — generation set to 0  
**Danger:** Deserialized entities bypass generation safety entirely. If an entity is despawned and the slot reused, the deserialized handle will silently point to the new entity with generation 0, violating the entire generation-tracking invariant.

**Fix:** Assign generation from `World::spawn()` during deserialization, or use a placeholder entity with generation validation on first use.

### 6. RewriteDriver raw pointer — CRITICAL

**File:** `crates/prism-ecs-ir/src/rewrite_driver.rs`  
**Pattern:** Raw `*mut World` pointer with `unsafe impl Send`  
**Danger:** The Send impl is unconditional — if the World is sent to another thread while borrows exist, it's immediate UB. No documentation explains the safety contract.

**Fix:** Document the SAFETY invariant or use a checked pattern (e.g., `Arc<Mutex<World>>` or borrow-checked API).

### 7. Concurrency — 2 CRITICAL

**Details from scout pending — likely std::sync::Mutex across .await in server path and unsafe Send impl on shared World reference.**

### 8. World fields pub — CRITICAL

**File:** `crates/prism-ecs-core/src/world.rs`  
**Pattern:** `pub entity_meta`, `pub free_list`, `pub staging`, `pub next_id`, `pub mutation_policy`  
**Danger:** External code can directly corrupt the ECS internals — insert into free_list, bypass generation checks, overwrite mutation policy. Every safety invariant is bypassable.

**Fix:** Make fields `pub(crate)` and expose only through safe methods.

### 9. Dead backend modules — CRITICAL

**Files:** `crates/prism-ecs-ir/src/backend_nvidia_gpu.rs` (928 lines), `backend_amd_gpu.rs` (611 lines)  
**Pattern:** Complete codegen backends not declared in lib.rs  
**Danger:** These backends are compiled but never linked. Any bug in them won't be caught by testing. The NVIDIA PTX and AMD AMDGCN codegen paths don't exist at link time.

**Fix:** Either add `pub mod` declarations or move to a separate crate with explicit feature gates.

---

## HIGH Findings (should fix before production)

### Top 10 by impact

1. **Unchecked entity ID underflow** — `world.rs:264,595,612,645`: `(entity.0 - 1) as usize` panics on Entity(0, _). Any code path that passes Entity(0) crashes the process.
2. **Dangling CompilePlanRef** — `evolution.rs`: Stores raw Entity in a component. If CompilePlan entity is despawned, reader gets stale handle with wrong generation.
3. **Ghost despawn entries** — `world.rs`: `despawn()` advances generation but does NOT clear component data from columns. An entity consuming the same slot sees stale component data from the previous occupant.
4. **QueryMut aliasing** — `query.rs:42-47`: `QueryMut::next()` uses raw pointer + `.add()` to hand out `&mut` references. Missing aliasing check between adjacent entities.
5. **Static lifetime transmute** — `compute-core/bitnet/checkpoint.rs:107-112`: Transmute to `'static` for safetensors. Field reordering breaks the layout assumption.
6. **All 32 modules pub** — `prism-ecs-ir/src/lib.rs`: Every module is `pub mod`. Clients can depend on internal dialect/backend/pass details that are not a stable API.
7. **Entity tuple fields pub** — `crates/prism-ecs-core/src/entity.rs`: `Entity.0` and `Entity.1` are pub. External code can fabricate any (id, generation) pair.
8. **ComponentStore.data pub** — `crates/prism-ecs-core/src/store.rs`: Bypasses the typed `get_component`/`add_component` API.
9. **Server scheduler fields pub** — `crates/prism-ecs-server/src/scheduler.rs`: `pending`, `active`, `completed` Vec<Entity> are pub. External code can inject entities bypassing enqueue/schedule.
10. **Unsafe Send on shared World** — Multiple `unsafe impl Send` on types containing raw World pointers. Need SAFETY: comments.

---

## Recommendation

**Safe to run on this M1** if we fix the top 5 CRITICAL items first (4, 8, 9 are not on M1's path):

| # | Issue | Fix time | Blocks M1 test |
|---|---|---|---|
| 1 | 37 unwrap paths | ~2h (dispatch) | Yes — model loading will crash |
| 2 | Metal buffer overflow | 10 min | Yes — GPU dispatch will corrupt memory |
| 3 | Arena use-after-lock | 15 min | Yes — accelerate ops will corrupt memory |
| 4 | DuckDB transmute_copy | 10 min | No (not on M1 test path) |
| 5 | Entity fabrication in serde | 20 min | Yes — model load/save will produce corrupt state |
| 7 | Concurrency (2 critical) | ~30 min | Yes — server will deadlock or misbehave |
| 9 | Dead backend modules | 5 min | No (no runtime impact) |

**Estimated fix time for safe M1 testing: ~3 hours**

Want me to dispatch fix agents for all CRITICAL and top HIGH items?
