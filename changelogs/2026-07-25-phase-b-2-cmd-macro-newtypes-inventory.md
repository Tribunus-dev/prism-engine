# Phase B-2 Inventory: `cmd!` Macro Newtypes Refactor

**Date:** 2026-07-25
**Lane:** Constitutional systems alignment
**Phase:** B-2 (the heaviest fix in the migration backlog)
**Source defect:** Audit A3 — 24 commands defined by the `cmd!` macro in
`crates/prism-ecs-constitutional/src/lifecycle_command.rs:143-145` use raw
`u64` / `u32` / `String` for authority-bearing values.

---

## 1. Prime directive

> *"if the type doesn't say what it is, the API is wrong"*

The 24 commands are the **entry points** for every state-bearing change
in the constitutional layer. Their field types are the public surface
between ingress and the `WorldTxn` machinery. A `u64` field at this
boundary can be silently assigned any other `u64` — a different
entity, a different epoch, a different dispatch — with no compiler
protest. The bug-class this enables: `let cmd = CreateWorkCommand { entity:
other_entity, ... }` when the developer meant `entity: target_entity`.

The newtypes exist to make such mistakes unrepresentable.

---

## 2. The `cmd!` macro today

`crates/prism-ecs-constitutional/src/lifecycle_command.rs:143`:

```rust
macro_rules! cmd {($($n:ident{$($f:ident:$t:ty),*}),* $(,)?)=>{
    $(#[derive(Debug,Clone,Serialize,Deserialize)] pub struct $n {$(pub $f:$t),*})*
}}
cmd! {
 CreateWorkCommand{entity:u64, target_entity:u64, kind:String, input_path:String, output_path:String, resource_claim:String},
 CreateCompilationJobCommand{entity:u64, model_artifact:u64, target_profile:String, job_id:u64, target_format:String, optimization_level:u32, enable_validation:bool},
 RequestCancellationCommand{entity:u64, reason:String},
 MarkObservedCommand{entity:u64, observed_epoch:u64},
 RecordExternalObservationCommand{entity:u64},
 RecordWorkPlanCommand{entity:u64, backend:String, output_format:String, resource_estimate_bytes:u64, timeout_ms:u64},
 MarkPrerequisiteBlockedCommand{entity:u64},
 AdmitWorkCommand{entity:u64},
 RejectWorkCommand{entity:u64, reason:String},
 DeferWorkCommand{entity:u64, reason:String},
 AcquireWorkLeaseCommand{work_entity:u64, lease_generation:u32, ttl_ms:u64},
 ReleaseWorkLeaseCommand{work_entity:u64},
 RenewWorkLeaseCommand{work_entity:u64, ttl_ms:u64},
 RecordDispatchIntentCommand{work_entity:u64, backend:String, config:String, deadline_ms:u64},
 RecordDispatchStartedCommand{work_entity:u64, adapter_handle:String},
 RecordProgressCommand{work_entity:u64},
 CompleteWorkCommand{work_entity:u64, lease_generation:u32, output:Vec<u8>, output_path:String},
 FailWorkCommand{work_entity:u64, error:String, lease_generation:u32, retryable:bool},
 MarkDispatchLostCommand{work_entity:u64},
 AttachArtifactCommand{entity:u64, digest:String},
 AttachDiagnosticsCommand{entity:u64},
 AttachEvidenceCommand{entity:u64, digest:String},
 PublishResultCommand{entity:u64, result_type:String, result:String},
 ExpireTransientCommand{entity:u64},
 MarkRetentionCompleteCommand{entity:u64}
}
```

24 commands. **~96 field declarations** (counted in audit A3).

---

## 3. Field-type → newtype mapping

For every field of every command, the newtype target is the authority-bearing
type. **Field renames are forbidden** in this refactor — only the type changes,
so call-site updates are minimal.

| Raw | Used for | Should be | Newtype exists? | Location to add |
| --- | -------- | --------- | --------------- | --------------- |
| `u64` (entity, work_entity, target_entity) | Entity handle | `Entity` | **yes** | `prism-ecs-core/src/entity.rs:20` |
| `u32` (lease_generation) | Fencing generation | `Generation` | **no** | `types.rs` (this PR) |
| `u64` (observed_epoch, world_epoch) | World epoch | `Epoch` | **no** | `types.rs` (this PR) |
| `u64` (sequence) | Event sequence | `Sequence` | **no** | `types.rs` (this PR) |
| `u64` (job_id) | Command identity | `CommandId` | **no** | `types.rs` (this PR) |
| `String` (digest) | Artifact digest | `ArtifactDigest` | **yes** | `prism-ecs-constitutional/src/artifact.rs:32` |
| `u64` (model_artifact) | Artifact identity | `ArtifactDigest` | yes (with conversion) | uses `ArtifactDigest::from_raw(u64)?` |
| `String` (input_path, output_path) | Filesystem path | `FilePath` | **no** | `types.rs` (this PR) |
| `String` (backend) | Backend kind | `BackendKind` | **yes** | `prism-ecs-runtime/src/backend.rs` |
| `String` (target_format, output_format, result_type) | Format tag | `Format` | **no** | `types.rs` (this PR) |
| `String` (error, reason) | Rejection reason | `RejectionReason` | **no** | `types.rs` (this PR) |
| `String` (adapter_handle) | Backend handle | `AdapterHandle` | **no** | `types.rs` (this PR) |
| `String` (result) | Result payload | keep `String` (or `Vec<u8>`) | n/a | n/a — `String` is acceptable for free-form result body |
| `String` (config) | Backend config | `Config` | **no** | `types.rs` (this PR) |
| `String` (resource_claim) | Resource spec | `ResourceClaim` | **yes** (struct) | `prism-ecs-constitutional/src/scheduler.rs:7` |
| `String` (target_profile) | Profile identity | `ContextProfileId` | **yes** | `prism-ecs-server/src/runtime/server_types.rs` (already used in receipt.rs) |
| `String` (receipt_id) | Receipt identity | `ReceiptId` | **no** | `types.rs` (this PR) |
| `u64` (resource_estimate_bytes) | Bytes | keep `u64` | n/a | n/a — bytes are bytes |
| `u64` (timeout_ms, ttl_ms, deadline_ms) | Milliseconds | keep `u64` | n/a | n/a — durations in ms are conventional |
| `u32` (optimization_level) | Numeric enum | `OptimizationLevel` (enum) | **no** | `types.rs` (this PR) — could be a `NonZeroU8` |
| `bool` (enable_validation, retryable) | Boolean flag | keep `bool` | n/a | n/a — bools are bools |
| `Vec<u8>` (output) | Raw bytes | keep `Vec<u8>` | n/a | n/a — bytes are bytes |

**Net newtypes to introduce (11):**

```rust
// In crates/prism-ecs-constitutional/src/types.rs

/// Fencing generation: monotonic per resource; replaced on lease acquire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Generation(pub u32);

/// World epoch: increments on every WorldTxn commit. Read by stale-fencing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Epoch(pub u64);

/// Event sequence: monotonic per EventStore; never reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Sequence(pub u64);

/// Command identity: assigned at ingress; never reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CommandId(pub u64);

/// Filesystem path: not a free String.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FilePath(pub String);

/// Format tag: e.g. "gguf", "cimage", "safetensors".
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Format(pub String);

/// Rejection reason: human-readable, validated, not a free String.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RejectionReason(pub String);

/// Adapter handle: backend-specific opaque token.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AdapterHandle(pub String);

/// Backend config: free-form key=value; validated by the backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Config(pub String);

/// Receipt identity: monotonic per work entity; never reused.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ReceiptId(pub String);

/// Lease token: opaque to the constitutional layer; verified by the
/// dispatcher at effect time.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LeaseToken(pub String);
```

**Already-existing newtypes to use:**

- `prism_ecs_core::Entity` — for `entity`, `work_entity`, `target_entity`
- `prism_ecs_constitutional::artifact::ArtifactDigest` — for `digest`, `model_artifact` (with `.0` or constructor)
- `prism_ecs_runtime::backend::BackendKind` — for `backend`
- `prism_ecs_constitutional::scheduler::ResourceClaim` — for `resource_claim`
- `prism_ecs_server::runtime::server_types::ContextProfileId` — for `target_profile`

---

## 4. Cmd! macro — refactored

After the refactor, the `cmd!` invocation reads:

```rust
cmd! {
    CreateWorkCommand{
        entity:Entity,
        target_entity:Entity,
        kind:Format,
        input_path:FilePath,
        output_path:FilePath,
        resource_claim:ResourceClaim,
    },
    CreateCompilationJobCommand{
        entity:Entity,
        model_artifact:ArtifactDigest,
        target_profile:ContextProfileId,
        job_id:CommandId,
        target_format:Format,
        optimization_level:OptimizationLevel,
        enable_validation:bool,
    },
    RequestCancellationCommand{
        entity:Entity,
        reason:RejectionReason,
    },
    MarkObservedCommand{
        entity:Entity,
        observed_epoch:Epoch,
    },
    RecordExternalObservationCommand{
        entity:Entity,
    },
    RecordWorkPlanCommand{
        entity:Entity,
        backend:BackendKind,
        output_format:Format,
        resource_estimate_bytes:u64,
        timeout_ms:u64,
    },
    MarkPrerequisiteBlockedCommand{
        entity:Entity,
    },
    AdmitWorkCommand{
        entity:Entity,
    },
    RejectWorkCommand{
        entity:Entity,
        reason:RejectionReason,
    },
    DeferWorkCommand{
        entity:Entity,
        reason:RejectionReason,
    },
    AcquireWorkLeaseCommand{
        work_entity:Entity,
        lease_generation:Generation,
        ttl_ms:u64,
    },
    ReleaseWorkLeaseCommand{
        work_entity:Entity,
    },
    RenewWorkLeaseCommand{
        work_entity:u64,
        ttl_ms:u64,
    },
    RecordDispatchIntentCommand{
        work_entity:Entity,
        backend:BackendKind,
        config:Config,
        deadline_ms:u64,
    },
    RecordDispatchStartedCommand{
        work_entity:Entity,
        adapter_handle:AdapterHandle,
    },
    RecordProgressCommand{
        work_entity:Entity,
    },
    CompleteWorkCommand{
        work_entity:Entity,
        lease_generation:Generation,
        output:Vec<u8>,
        output_path:FilePath,
    },
    FailWorkCommand{
        work_entity:Entity,
        error:RejectionReason,
        lease_generation:Generation,
        retryable:bool,
    },
    MarkDispatchLostCommand{
        work_entity:Entity,
    },
    AttachArtifactCommand{
        entity:Entity,
        digest:ArtifactDigest,
    },
    AttachDiagnosticsCommand{
        entity:Entity,
    },
    AttachEvidenceCommand{
        entity:Entity,
        digest:ArtifactDigest,
    },
    PublishResultCommand{
        entity:Entity,
        result_type:Format,
        result:String,  // free-form payload body; bound by validation, not type
    },
    ExpireTransientCommand{
        entity:Entity,
    },
    MarkRetentionCompleteCommand{
        entity:Entity,
    },
}
```

---

## 5. Call-site impact

The 24 commands are constructed and consumed in:

- `crates/prism-ecs-runtime/src/server.rs` (and other server files) — constructs
  commands from incoming requests, dispatches them.
- `crates/prism-ecs-server/src/runtime/server.rs` — receives commands.
- Any tests in `crates/prism-ecs-constitutional/src/lifecycle_command.rs` and
  test files in `crates/prism-ecs-server/`.

The audit has not enumerated every call site; the implementing agent must
run `rg "CreateWorkCommand\|CreateCompilationJobCommand\|..." crates/` to
find all construction sites and update them. The migration is mechanical:
each `u64` literal must be wrapped in the appropriate newtype
(`Entity::from_raw(x)?` or `Generation(x as u32)`).

---

## 6. Migration order

1. **Add the 11 newtypes to `types.rs`** (one commit). All are
   transparent newtypes so the wire format is unchanged.
2. **Re-export the new types from `prism-ecs-constitutional::lib.rs`**
   so `use prism_ecs_constitutional::*;` continues to work.
3. **Update the `cmd!` invocation** in `lifecycle_command.rs` to use
   the new types.
4. **Update each call site** (constructor) to wrap raw values in the
   newtype. This is the bulk of the work.
5. **Update `LifecycleCommandResult` enum** (lines 48-142) to use the
   newtypes too — the result enum has the same defect (`u64` for
   `work_entity`, `entity`, `sequence`, etc.).
6. **Run `cargo build` and `cargo test -p prism-ecs-constitutional
   -p prism-ecs-runtime -p prism-ecs-server`** and fix call-site errors
   iteratively.
7. **Write the `Completion report`** in
   `changelogs/2026-07-25-phase-b-2-cmd-macro-newtypes.md`.

---

## 7. Risks

- **Wire format:** `#[serde(transparent)]` ensures the bytes are unchanged
  on the wire. Existing serialized commands (in `EventStore`) continue to
  deserialize correctly.
- **Builder API:** The current commands are plain structs. Some
  constructors may use struct update syntax (`..Default::default()`).
  After the refactor, `Default` for `Generation`/`Epoch`/`CommandId`
  is a value of `0`, which may or may not be a valid identity. Consider
  adding `impl Default` that returns a sentinel only when appropriate.
- **`HashMap<Entity, V>` collision:** `Entity` is `(u64, u32)`. The
  existing `impl Hash for Entity` (in `prism-ecs-core`) hashes the pair.
  Any `HashMap<Entity, V>` already works; no changes needed.
- **`Serialize`/`Deserialize` for `Entity`:** must verify that
  `prism-ecs-core::Entity` has a `serde` impl, or add one. The cmd!
  commands use `#[derive(Serialize, Deserialize)]`.

---

## 8. The shape of the fix

The 11 newtypes are 11 lines of struct declarations plus 11 derive lists
plus the `From` impls for the few that are wrapped. The 24-command refactor
is a 24 × ~5 field edit = ~120 line diff. The call-site update is the
bulk: `rg 'CreateWorkCommand{' crates/` to find sites, then update each.

**Estimated effort: 1 subagent for 30-60 minutes, or 1 human-day for the
full refactor with care.**

This is the heaviest fix in the migration backlog. It is also the highest
value: every command flow goes through these types, so the entire ingress
boundary becomes type-safe.

---

## 9. Status

**Inventory: DONE.** This changelog is the spec for the B-2 subagent or
the implementer. The actual refactor lands as a follow-up change once
B-3a, B-3b, B-3c, B-4 are merged and the workspace builds cleanly.
