# Wave 27: Evolutionary Pass Pipeline — CompilePlan-Driven Systems

**Status:** Draft (pre-implementation)
**Dependency:** All Waves 1-25 (ECS core + IR + dialects + codegen backends + passes)
**Owner:** evolutionary

## 1. Scope

Wire every pass system to read its parameters from a `CompilePlan` entity instead of using hardcoded heuristics. The mutation table in `evolution.rs` already defines the searchable space — this wave connects it to the compiler pipeline.

## 2. Design

### CompilePlan entity

```rust
// A plan entity carries one component per pipeline stage
pub struct FusionPolicy(pub FusionStrategy);        // Aggressive | Conservative | None
pub struct LayoutStrategy(pub LayoutMode);           // Coalesced | Mma | Shared | Auto
pub struct TileSizes(pub Vec<(u32, u32, u32)>);     // per-tensor tile dims
pub struct PipelineDepth(pub u32);                   // 0 = no pipelining
pub struct FormatAssignment(pub Vec<(Entity, TensorFormat, TensorOperation)>); // per-tensor format
```

A `CompilePlan` marker component groups them on one entity. The evolution system mutates components on this entity; the pass systems read from it.

### Changes per pass system

| System | Reads from | Current → Evolvable |
|---|---|---|
| `fusion::analyze_dataflow` | `FusionPolicy` | BFS all edges → BFS limited by strategy |
| `layout_inference::assign_coalesced_layout` | `LayoutStrategy` | hardcoded 2D blocks → block size from plan |
| `layout_inference::assign_mma_layout` | `LayoutStrategy` | hardcoded MMA v2 → MMA version from plan |
| `codegen backends` | `TileSizes` + `FormatAssignment` | hardcoded 4x4 → per-tensor from plan |
| `pipelining::pipeline_program` | `PipelineDepth` | estimated depth → depth from plan |
| `low_bit_codec::ternary_dot_product` | `FormatAssignment` | manual dispatch → automatic per-tensor |

### Search loop

```
1. AlphaEvolve creates N CompilePlan entities with mutated components
2. For each plan: run the full compiler pipeline (fusion → layout → lowering → codegen)
3. Benchmark each result (latency, memory, power)
4. Select top performers, mutate, repeat
```
