# Wave 29: vLLM Absorption — ECS-Native Continuous Batching + PagedAttention

**Status:** Draft (pre-implementation)
**Dependency:** Wave 28 vendor runtime crates, Wave 27 evolutionary pipeline
**Owner:** serving

## 1. Scope

Absorb vLLM's continuous batching and PagedAttention algorithms into ECS-native systems in `prism-ecs-server`. Unlike vLLM which discovers memory layouts at runtime, these systems embed pre-computed plans into the CImage at compile time.

## 2. File map

All files in `crates/prism-ecs-server/src/`:

| File | Contents |
|---|---|
| `scheduler.rs` | `ContinuousBatchingScheduler` resource + `Batch` entity — coalesces pending requests into a batch, schedules across heterogeneous hardware |
| `paged_attention.rs` | `PagedAttentionTable` component — pre-computed KV page table layout per attention head |
| `memory_plan.rs` | `MemoryPlan` resource — sealed memory plan embedded in CImage: known max KV, page sizes, tensor lifetimes |
| `heterogeneous_dispatch.rs` | Routes work across ANE + GPU + NPU + CPU per the compile-time plan |
| `cgraph.rs` | Pre-compiled dispatch graphs (Metal/NVIDIA analog of CUDA graphs) baked into CImage |

## 3. Design

### ContinuousBatching resource

```rust
pub struct ContinuousBatchingScheduler {
    pub pending: Vec<Entity>,         // pending Request entities
    pub active: Vec<Entity>,          // currently executing Batch entities
    pub max_batch_size: u32,
    pub hardware_plan: Entity,        // CompilePlan from Wave 27
}
```

### PagedAttentionTable component (attached per attention head entity)

```rust
pub struct PagedAttentionTable {
    pub block_size: u32,
    pub num_blocks: u32,
    pub page_table: Vec<u32>,     // pre-computed page layout
}
```

## 4. Gate

- 100 concurrent requests batch correctly across ANE + GPU + NPU
- Memory plan pre-computation matches vLLM's runtime allocation within 5%
- PagedAttention table produces identical numerical output to vLLM on the same input
