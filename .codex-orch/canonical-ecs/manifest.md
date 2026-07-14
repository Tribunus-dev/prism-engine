# Canonical ECS Refactor — Session Manifest

## Baseline

| Field | Value |
|---|---|
| HEAD | c158db28467e45bddc6aa346f1469f41b33f5048 |
| Branch | main |
| Plan draft | ADR-003 (accepted, implementation begins Waves 1-3) |
| Dirty tree | 13 modified, 9 untracked |

## Dirty-tree inventory

### Modified files (ECS-related)
| File | Owner | Notes |
|---|---|---|
| compute-core/Cargo.toml | integrator | ECS deps |
| compute-core/src/ecs/mod.rs | integrator | Core ECS — canonical module re-org |
| compute-core/src/ecs/constitutional/world_txn.rs | integrator | Transaction engine changes |
| compute-core/src/ecs/compiler/deployment_compiler.rs | compiler | Compiler migration |
| compute-core/src/ecs/compiler/lifecycle_coordinator.rs | compiler | Compiler migration |
| compute-core/src/ecs/aot/mod.rs | compiler | AOT changes |
| compute-core/src/ecs/cimage/mod.rs | compiler | CImage changes |
| compute-core/src/ecs/quantization/precision_policy.rs | compiler | Quant changes |
| compute-core/src/ecs/runtime/engram/application.rs | runtime | Runtime changes |
| compute-core/tests/gemma4_production_serve_gate.rs | verifier | Test gate changes |

### Modified files (non-ECS — unrelated MCP/tool changes)
| File | Owner | Notes |
|---|---|---|
| Cargo.lock | — | Dep bump |
| src/llm/tools.rs | — | Unrelated |

### Untracked files
| File | Phase | Notes |
|---|---|---|
| compute-core/src/ecs/column.rs | 1 | Column<T> storage (already extracted) |
| compute-core/src/ecs/world/mod.rs | P | Prototype module |
| compute-core/src/ecs/world/commands.rs | P | Prototype Commands |
| compute-core/src/ecs/world/storage.rs | P | Prototype DenseVec (duplicates column.rs) |
| compute-core/src/ecs/world/types.rs | P | Prototype types |
| compute-core/src/ecs/world/world.rs | P | Prototype World |
| compute-core/src/ecs/world/tests.rs | P | Prototype tests |
| compute-core/src/ecs/cimage/sealed_v1.rs | — | CImage sealed format |
| docs/adr-003-canonical-ecs-world.md | — | Draft ADR |

## Protected ownership

| Role | Files |
|---|---|
| Integrator | ecs/mod.rs, ecs/column.rs, ecs/world/* (structural), ecs/constitutional/world_txn.rs, shared exports, Cargo features, module graph |
| Constitutional workers | ecs/constitutional/* except world_txn.rs |
| Compiler workers | ecs/compiler/*, ecs/system/* |
| Runtime workers | ecs/runtime/* |
| Test workers | New test files only |
| Verifier | Read-only |

## ADR adoption

## Phase 0 reconnaissance complete

### CompWorld inventory (40+ files across 6 categories)
| Category | Files | Pattern |
|---|---|---|
| Core | ecs/mod.rs | CompWorld def, CompEntity newtype, transit/prepare/apply |
| Constitutional (14) | agent_exec, artifact, compilation, device, distributed, execution, ingress, multimodal, residency, session, work, tests, schema, world_txn | preflight(&CompWorld) + execute(&mut CompWorld) via WorldTxn+transit |
| Legacy systems (14) | archive, backend_compile, backend_dispatch, backend_eval, backend_residency, backpressure_tick, buffer_lifetime, capability_registry, catalog_validation, compiler_systems, completion_ingest, download, draft_model, engine_systems, executor_systems | entities_of_kind + get_component + spawn/add_component — NO WorldTxn |
| Fusion systems (4) | analysis, dispatch, heuristic, scalar | entities_of_kind Layer/Tensor/Dispatch + spawn/add_component |
| Evolution (3) | foundation (component fields), decomposition, systems | CompWorld::new() + direct mutation |
| Component fields (2) | memory.rs BufferBinding.buffer, sync.rs FenceEdge semaphores | CompEntity as field type |
| Binaries (1) | bitnet_ecs_test | CompWorld::new() |

### Runtime world inventory (26 sites, 17 files)
| Category | Files | Pattern |
|---|---|---|
| World definition | runtime/world.rs | Entity(u32) gen-aware, ComponentVec<T>, full API |
| Long-lived | agent_slot (RwLock), ecore_pump, ane_multiplexer | Hardcoded IDs 0..31 |
| Ephemeral | compilation_systems (7 fns + 1 test) | iter_entities_with + get/get_mut + insert/spawn |
| Schedule | schedule, metadata, command | run(&mut World), Resources wrapper, CommandWriter |
| Inference systems | inference_step, audio/video inference | iter_entities_with + get + get_resource |
| NPU systems | submitter, observer, completion_observer | iter_entities_with + get/get_mut + get_resource |
| Worker systems | ingress, spawn, event_drain, watchdog, bridge, stream_observer | iter_entities_with + get + spawn/despawn/insert |
| Ledger | receipt (2 tests) | World::default() + spawn |

### Transaction audit — 15 gaps identified
| Gap | Location | Impact |
|---|---|---|
| SchemaCatalogue not wired into transit() | world_txn.rs | Extra safety available but unused |
| before_hash/after_hash/encoded_value all None | world_txn.rs | B6 future work — no integrity evidence |
| Update journal entries never produced | world_txn.rs | Missing change-type variant |
| Component versions per-entity not per-(entity,type) | mod.rs | Stale-dependency detection too coarse |
| Dual replay pathways uncoordinated | persistence.rs vs world_txn.rs | Replay behavior divergence risk |

### Owners for Phase 1
| Module | Owner | Forbidden |
|---|---|---|
| ecs/mod.rs (CompWorld) | integrator | Other agents |
| ecs/column.rs | integrator | Other agents |
| ecs/world/types.rs | integrator (types only) | Editing world/world.rs |
| Constitutional/* | Read-only until Phase 2 | Writing |
| Compiler/system/* | Read-only | Writing |
| Runtime/* | Read-only | Writing |

This session adopts the architectural decisions from ADR-003 and the orchestration plan from `orchestrate and ultrathink # Plan`.
