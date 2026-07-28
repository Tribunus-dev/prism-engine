# Phase A Audit: The Constitutional Engine

**Date:** 2026-07-28 (Pacific)
**Status:** Audit complete. Migration order established. First batch ready to dispatch.

## The reframe

The current architecture has two parallel realities:

1. **`compute-core/`** — the engine. 324,802 LOC. Pre-existing 192 build errors. Engine-coupled types, hardware adapters, and `legacy_*/` reference code live here.
2. **`crates/prism-ecs-*`** — the constitutional crates. 28 modules in `prism-ecs-constitutional` alone. Schemas, lifecycle, transactions, replay. No engine coupling by design.

The 25 absorbed engine subsystems now live as constitutional surfaces (in `prism-ecs-compile`, `prism-ecs-runtime`, etc.) but their `legacy_*/` reference code is still inside `compute-core/`. This dual-track is **permanent until we either**:
- (a) git-rm the `legacy_*/` dirs (impossible until every engine-coupled type is constitutional), or
- (b) make the constitutional crates the engine.

We choose (b). **`prism-ecs-constitutional` becomes the constitutional engine.**

## Target architecture

**Tier 1: Constitutional surface (THE engine, expressed constitutionally)**

```
crates/prism-ecs-constitutional/        ← central authority (already exists, 28 modules)
  ├ core/                              ← what is today prism-ecs-core
  ├ canonical/                         ← MOVED from engine: GenerationId, CimageGeneration, KernelAbi, identity, generation, provenance
  ├ config/                            ← MOVED from engine: TextArchitecture, LayerPlan, ModelExecutionPlan, operation_route
  ├ training_target/                   ← MOVED from engine: spec, resolve, feedback, export (product-shape)
  ├ residency/, device/, external_array/  ← MOVED from engine (hardware-adjacent state)
  └ lifecycle, transactions, events, scheduler, agent, ingress, distributed, multimodal  (existing)

crates/prism-ecs-{core,runtime,kernel,compile,server,ir,data,protocol,agent,codec,quantization,kv-cache,architecture}/  (existing)
  └ ane/, cache/, metal_backend/       ← MOVED here from engine (subsystem-level surfaces)
```

**Tier 2: Platform adapters (depend on Tier 1, not the reverse)**

```
crates/prism-platform-metal/           ← NEW: Metal-specific adapter (was engine's metal_backend/ + backend/metal_*)
crates/prism-platform-ane/             ← NEW: ANE-specific adapter
crates/prism-platform-mlx/             ← NEW: MLX-specific adapter (was engine's MlxBackend, backend/heterogeneous_executor)
crates/prism-platform-coreml/          ← NEW: Core ML / iOSurface adapter (was engine's backend/coreai_iosurface, coreai_bridge)
```

**Tier 3: `compute-core/` — what remains after migration**

After all engine subsystems and engine-coupled types are absorbed:
- The engine's `src/ecs/` tree contains ONLY Tier 2 adapter code (re-exported from `prism-platform-*`)
- The engine's binaries (CLI, server entry point) wire Tier 1 + Tier 2 together
- The engine becomes a ~5-10K LOC compatibility shim, OR is deleted entirely once `prism-platform-*` and product crates link directly against `prism-ecs-constitutional`.

## Engine-coupled types — current census

| Engine dir | LOC | Used by legacy_*/ | Target tier | Migration order |
|---|---|---|---|---|
| `canonical/` | 1,334 | 28 imports | `prism-ecs-constitutional::canonical` | **Batch 6** (this audit's first batch) |
| `config/` | 2,583 | 25 imports | `prism-ecs-constitutional::config` | **Batch 6** (product-shape, canonical) |
| `ane/` | 3,115 | 6 imports | `prism-ecs-compile::ane` | **Batch 6** (CImage compile concern) |
| `metal_backend/` | 2,208 | 8 imports | `prism-platform-metal` | **Batch 7** |
| `backend/` (remainder) | 15,169 | 18 imports | `prism-ecs-kernel` + `prism-platform-*` | **Batch 7** (already in progress) |
| `training_target/` | 3,798 | 11 imports | `prism-ecs-server` | **Batch 7** (product-y) |
| `cache/` (subset) | 1,978 | 5 imports | `prism-ecs-data` / `prism-kv-cache` | **Batch 7** (combined with data) |
| `kv_cache/` (subset) | ~2K | 4 imports | `prism-kv-cache` | **Batch 7** (already partially absorbed) |
| `device/` | 1,855 | 3 imports | `prism-ecs-constitutional::device` | **Batch 8** (resides in constitutional already) |
| `mapped_image/`, `external_array/`, etc. | small | 1-2 each | various | **Batch 8** (consolidated) |

## Stale cross-legacy imports (1 batch to fix)

After the 5 absorption batches, 329 imports across `legacy_*/` files still point at OLD engine paths:

| Stale import | Count | What it should be |
|---|---|---|
| `crate::ecs::runtime::*` (in legacy_runtime/) | 173 | `crate::ecs::legacy_runtime::*` |
| `crate::ecs::compute_image::*` (in legacy_compute_image_core/ etc.) | 156 | `crate::ecs::compute_image::legacy_*::*` |

**Why this is a blocker:** every new `legacy_*/` dir grows more stale imports. Until these are fixed, the `legacy_*/` dirs cannot be `git rm`'d cleanly. **Batch 6 includes a mechanical agent for this** (sed-style refactor).

## Migration order

### Batch 6 (next, 4 agents)
1. **Stale cross-legacy import fix** — mechanical sed across all `legacy_*/` files. One agent, one commit.
2. **`canonical/` → `prism-ecs-constitutional::canonical`** — 10 files, 1,334 LOC. 28 legacy_*/ import sites retargeted. Engine's `canonical/` becomes empty/renamed.
3. **`config/` → `prism-ecs-constitutional::config`** — 6 files, 2,583 LOC. 25 legacy_*/ import sites retargeted. Engine's `config/` becomes empty/renamed.
4. **`ane/` → `prism-ecs-compile::ane`** — 8 files, 3,115 LOC. 6 legacy_*/ import sites retargeted. Engine's `ane/` becomes empty/renamed.

**Batch 6 outcome:** 3 engine dirs become git-rm-able, 60+ legacy_*/ import sites move to constitutional surfaces, 329 stale cross-legacy imports fixed.

### Batch 7
- `metal_backend/` → `prism-platform-metal` (new crate)
- Finish `backend/` → `prism-ecs-kernel` (in progress, 12K LOC remaining)
- `training_target/` → `prism-ecs-server`
- `cache/` + `kv_cache/` subset → `prism-ecs-data` / `prism-kv-cache`

### Batch 8
- `device/`, `mapped_image/`, `external_array/` → `prism-ecs-constitutional::*` (consolidated)
- 19 small engine subsystems (agent, benchmark, bridge, etc.) — each a small migration

### Batch 9
- git-rm all `legacy_*/` dirs (~465 files, ~182K LOC). Engine `src/ecs/` tree drops from 324K → ~140K LOC.
- Engine pre-existing error sweep: 192 → 0 (no more "legacy preserves broken state" alibi).

### Batch 10+
- Product surface migration: PrismAgent, PrismAgentiOS, PrismMenuBar, deno-dashboard, examples/, docs/ — all re-plumbed to use `prism-ecs-constitutional` instead of `compute_core::ecs::*`.
- Engine becomes a 5-10K LOC CLI shim, then deleted.

## Propagation chain (constitutional test for batch 6)

For every state-bearing change in batch 6:
- **durable event**: `prism_ecs_constitutional::canonical::*` (or other) types become the authority
- **event store**: schema persisted via `prism-ecs-constitutional::event_store`
- **replay applier**: `prism-ecs-constitutional::lifecycle` reads schema and replays
- **projection rebuild**: `prism-ecs-constitutional::projection` (new module) rebuilds
- **read path**: legacy_*/ files that imported the old path now import `prism_ecs_constitutional::canonical::*` via re-export
- **consumer**: all constitutional crates (prism-ecs-runtime, prism-ecs-compile, etc.) can depend on `prism-ecs-constitutional` directly

## Review gates for batch 6

Per the AGENTS.md review-gates list, every batch-6 agent MUST pass:
- **Authority gate**: every new file in `prism-ecs-constitutional` states a single authority in its module doc.
- **Module cohesion gate**: no file > 200 LOC unless it's a re-export hub.
- **Rust quality gate**: per-crate error enums (thiserror), no `unwrap`/`expect`/`panic!` in production paths, no `anyhow::Error` in constitutional.
- **Project absorption gate**: name files for what they DO in the constitutional system, not after engine files.
- **Propagation gate**: durable event → event store → replay applier → projection rebuild → read path → consumer.

## Risks

1. **Constitutional-crate size explosion**: `prism-ecs-constitutional` will gain ~7K LOC (canonical + config) + ~3K (training_target) + ~2K (device, mapped_image, external_array). Total: ~12K LOC added, ~31K LOC final. Manageable.

2. **Name conflicts**: `prism-ecs-constitutional` already has `device.rs`. The engine's `device/` is 10 files of `DeviceRegistry`-style types. The agent must check for existing types before re-implementing.

3. **Backend in progress**: `backend/` migration is partially done (14/37 files in `prism-ecs-kernel`). The remaining 23 files include engine-coupled ones (MlxBackend, BackendInstance, etc.) that are out of scope for the constitutional-engine vision. Need a separate decision: move them to `prism-platform-mlx` as platform adapters.

4. **Prism-metal-runtime coupling**: `crates/prism-metal-runtime/src/pso_cache.rs` and `fusion_lowering.rs` import `tribunus_compute_core::ecs::canonical::*`. These need to migrate to `prism_ecs_constitutional::canonical::*` simultaneously with batch 6.

## Success criteria for the "constitutional engine" milestone

The moment `compute-core/` is git-rm-able:
- [ ] Every engine subsystem (50+) is either a `legacy_*/` dir or has been moved to a constitutional crate
- [ ] Every engine-coupled type is in a constitutional crate
- [ ] Every cross-legacy import is either fixed or has a known waiver
- [ ] No `use compute_core::ecs::` (or `use tribunus_compute_core::ecs::`) in any non-`legacy_*/` file in the engine
- [ ] No `use compute_core::ecs::` (or `use tribunus_compute_core::ecs::`) in any constitutional crate except `prism-platform-*` (Tier 2) which may depend on engine-crate for the legacy adapter glue
- [ ] `cargo test -p prism-architecture --lib` passes
- [ ] `cargo test -p prism-ecs-constitutional --lib` passes with all new modules
- [ ] `cargo check -p tribunus-compute-core --lib` shows 0 pre-existing errors (or a documented remaining set)

That's the target. Batch 6 is the first concrete step.
