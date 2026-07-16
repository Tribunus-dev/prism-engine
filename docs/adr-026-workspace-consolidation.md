# ADR-026: Workspace Consolidation — Monolith Extraction, MCP Merge, MLX Removal

**Status:** Draft (pre-implementation)
**Dependency:** ADR-005 Waves 1-25 (ECS migration + compiler absorption complete)
**Owner:** workspace

## 1. Scope

Three independent consolidation actions:

1. **Extract server runtime** from `tribunus-compute-core` into `prism-ecs-server` — the monolith still holds server startup, HTTP handlers, evaluator dispatch, and admission logic that has nothing to do with compilation or ECS constitutional authority.

2. **Consolidate 11 MCP crates** into 3: `prism-mcp-core` (types/contracts), `prism-mcp-handlers` (tool implementations), `prism-mcpd` (daemon binary).

3. **Remove MLX fork** — 4 crates behind `mlx-backend` feature. Prism's ECS-native backends (Metal, CPU, NVIDIA) supersede it. Mark deprecated, remove in next cycle.

## 2. File map

### Stream A: Server extraction

| Action | Detail |
|---|---|
| Create `crates/prism-ecs-server/` | Cargo.toml depends on prism-ecs-core, prism-ecs-ir, tokio, axum |
| Move `compute-core/src/bin/prism_server.rs` | The server binary entry point |
| Move `compute-core/src/ecs/system/compiler_systems.rs` | CompileSchedule, BackendAssessment, GraphOptimizer — these are server-level orchestration, not compilation |
| Move evaluator dispatch code | The MetalEvaluator, Accelerate evaluator, ANE evaluator call sites |

### Stream B: MCP consolidation

| Action | Detail |
|---|---|
| Merge into `prism-mcp-core` | prism-mcp-model, prism-mcp-admission, prism-mcp-bench, prism-mcp-trace, prism-mcp-lab, prism-mcp-replay |
| Merge into `prism-mcp-handlers` | prism-mcp-build, prism-mcp-kernel, prism-mcp-browser |
| Keep standalone | prism-mcp-core (types), prism-mcpd (binary) |

### Stream C: MLX deprecation

| Action | Detail |
|---|---|
| Mark `mlx-backend` feature deprecated | Add deprecation notice to Cargo.toml |
| Remove `mlx-rs-fork/` from workspace members | Comment out, add note |

## 3. Gate

- `cargo check -p prism-ecs-server` passes with 0 errors
- `cargo check -p prism-mcp-core` + `prism-mcp-handlers` + `prism-mcpd` pass
- Server binary starts and serves requests after extraction
- `cargo check --features mlx-backend` produces deprecation warning but still compiles
- No functional changes to any user-facing behavior
