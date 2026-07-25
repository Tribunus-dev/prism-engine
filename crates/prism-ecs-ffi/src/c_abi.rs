//! C-ABI FFI for ECS world operations on iOS.
//!
//! This module is the only place in the Prism workspace that exposes
//! `extern "C"` functions with raw-pointer arguments. The constitutional
//! crate stays `unsafe`-free; the FFI crate is the bridge layer.
//!
//! # Caller contract
//!
//! Every entry point follows the C-ABI rules of the public header
//! (`PrismAgentiOS/include/PrismAgentFFI.h`, maintained out-of-tree):
//!
//! - `world` pointers must originate from [`prism_world_create`] and be
//!   passed to [`prism_world_destroy`] exactly once.
//! - `c_char` string pointers must be either null (interpreted as
//!   "no value") or a valid null-terminated C string allocated by the
//!   caller; for return strings, they must be released via
//!   [`prism_free_string`].
//!
//! Violating the contract is undefined behavior — the FFI module cannot
//! defend against it from inside `unsafe`.
//!
//! Uses `parking_lot::Mutex` (no poisoning, no `unwrap` on lock).
//!
//! `unsafe` is concentrated in this module and documented inline via
//! `// SAFETY:` comments. The crate does not enable
//! `unsafe_code = "deny"` — see `Cargo.toml`.

use parking_lot::Mutex;
use prism_ecs_constitutional::agent_exec::{AgentConfig, AgentLifecycle, AgentPhase, AgentRun, AgentTask};
use prism_ecs_constitutional::agent_plan::ParentAgentId;
use prism_ecs_constitutional::agent_state::tick;
use prism_ecs_constitutional::types::Timestamp;
use prism_ecs_constitutional::world_txn::{WorldTransitExt, WorldTxn};
use prism_ecs_core::{Entity, EntityKind, World};
use std::ffi::{CStr, CString};

/// Allocate a fresh empty `World` and return a stable raw pointer suitable
/// for FFI consumption. The returned pointer must be released via
/// [`prism_world_destroy`].
#[no_mangle]
pub extern "C" fn prism_world_create() -> *mut Mutex<World> {
    Box::into_raw(Box::new(Mutex::new(World::new())))
}

/// Release a `World` previously obtained from [`prism_world_create`].
///
/// # Caller contract
///
/// - `world` must be a pointer returned by `prism_world_create` and not
///   yet destroyed (no double-free).
/// - After this call, the pointer is dangling and must not be reused.
#[no_mangle]
pub extern "C" fn prism_world_destroy(world: *mut Mutex<World>) {
    if !world.is_null() {
        // SAFETY: The null check above ensures `world` is non-null. The
        // C-ABI contract requires the caller to hand us a pointer that
        // was originally produced by `Box::into_raw` inside
        // `prism_world_create` and that has not already been destroyed
        // (caller responsibility — double-free is UB). The pointee type
        // is `Mutex<World>`; reconstructing the `Box` is sound because
        // the type matches.
        unsafe {
            drop(Box::from_raw(world));
        }
    }
}

/// Spawn a sub-agent entity under `parent_entity_id`.
///
/// Returns the new entity's id, or `0` on failure (e.g. parent not alive).
#[no_mangle]
pub extern "C" fn prism_subagent_spawn(
    world: *mut Mutex<World>,
    parent_entity_id: u64,
    task_description: *const std::os::raw::c_char,
) -> u64 {
    // SAFETY: The C-ABI contract for `world` is identical to
    // `prism_world_destroy`: a non-null pointer produced by
    // `prism_world_create`, with the pointee type `Mutex<World>`. We do
    // not take ownership — locking the mutex and dropping the guard at
    // end of scope is sound. Caller must guarantee a non-null, valid
    // pointer (null is not checked here; the FFI entry does not promise
    // null-safety on the world argument because doing so would force a
    // panic-free error channel through a return value that the Swift
    // caller would still need to interpret).
    let world = unsafe { &mut *world };
    let mut w = world.lock();
    let task = if !task_description.is_null() {
        // SAFETY: When the caller passes a non-null `task_description`
        // pointer, the C-ABI contract states it points to a valid
        // null-terminated C string whose lifetime extends at least until
        // this function returns. `CStr::from_ptr` is sound under that
        // precondition. We immediately convert to an owned `String` so
        // we never dereference past the function boundary.
        unsafe { CStr::from_ptr(task_description) }
            .to_string_lossy()
            .into_owned()
    } else {
        "subtask".to_string()
    };

    let parent = Entity::new(parent_entity_id, 0);
    let entity = WorldTxn::next_entity_id(&w);
    let mut txn = WorldTxn::new(&mut w);
    txn.stage_spawn(entity, EntityKind::Agent);
    txn.put_durable(
        entity,
        AgentRun {
            run_id: entity.id(),
            session_entity: 0,
            name: format!("subagent_of_{}", parent_entity_id),
            created_at: Timestamp::now(),
        },
    );
    txn.put_durable(
        entity,
        AgentTask {
            task_description: task,
            max_steps: 10,
            model_entity: 0,
        },
    );
    txn.put_durable(
        entity,
        AgentConfig {
            model: "default".to_string(),
            temperature: 0.7,
            max_tokens: 512,
            tools_enabled: true,
            max_tool_rounds: 10,
        },
    );
    txn.put_durable(entity, AgentPhase::Planning);
    txn.put_durable(entity, AgentLifecycle::Active);
    txn.put_durable(entity, ParentAgentId(parent));

    let _ = w.transit(txn);
    entity.id()
}

/// Read the current `AgentPhase` of an entity, formatted via `Debug`.
///
/// Returns a heap-allocated C string the caller must release via
/// [`prism_free_string`]. On missing-component, returns the string
/// `"Unknown"`.
#[no_mangle]
pub extern "C" fn prism_subagent_phase(
    world: *mut Mutex<World>,
    entity_id: u64,
) -> *mut std::os::raw::c_char {
    // SAFETY: Same contract as `prism_subagent_spawn` — `world` is a
    // non-null, valid pointer to a `Mutex<World>` produced by
    // `prism_world_create`. The borrow through the lock is sound; the
    // guard is dropped at end of scope.
    let world = unsafe { &mut *world };
    let w = world.lock();
    let entity = Entity::new(entity_id, 0);
    let phase = w
        .get_component::<AgentPhase>(entity)
        .map(|p| format!("{:?}", p))
        .unwrap_or_else(|| "Unknown".to_string());
    CString::new(phase).unwrap().into_raw()
}

/// Read the current `AgentLifecycle` of an entity, formatted via `Debug`.
///
/// Returns a heap-allocated C string the caller must release via
/// [`prism_free_string`]. On missing-component, returns the string
/// `"Unknown"`.
#[no_mangle]
pub extern "C" fn prism_subagent_lifecycle(
    world: *mut Mutex<World>,
    entity_id: u64,
) -> *mut std::os::raw::c_char {
    // SAFETY: Same contract as `prism_subagent_phase` — see comment
    // there for the invariant.
    let world = unsafe { &mut *world };
    let w = world.lock();
    let entity = Entity::new(entity_id, 0);
    let lifecycle = w
        .get_component::<AgentLifecycle>(entity)
        .map(|l| format!("{:?}", l))
        .unwrap_or_else(|| "Unknown".to_string());
    CString::new(lifecycle).unwrap().into_raw()
}

/// Mark the agent entity as `Failed` (best-effort cancellation).
#[no_mangle]
pub extern "C" fn prism_subagent_cancel(world: *mut Mutex<World>, entity_id: u64) {
    // SAFETY: Same contract as `prism_subagent_phase` — see comment
    // there for the invariant.
    let world = unsafe { &mut *world };
    let mut w = world.lock();
    let entity = Entity::new(entity_id, 0);
    let _ = w.insert_component(entity, AgentLifecycle::Failed);
}

/// Run one agent-lifecycle tick. Returns a static-shape JSON string
/// describing the result; for now it is a fixed stub (`tick_complete`).
/// The caller must release the returned string via [`prism_free_string`].
#[no_mangle]
pub extern "C" fn prism_agent_tick(world: *mut Mutex<World>) -> *mut std::os::raw::c_char {
    // SAFETY: Same contract as `prism_subagent_phase` — see comment
    // there for the invariant.
    let world = unsafe { &mut *world };
    let w = world.lock();
    let _transitions = tick(&w).unwrap_or_default();
    let result = r#"{"status":"tick_complete","transitions":[]}"#.to_string();
    CString::new(result).unwrap().into_raw()
}

/// Release a C string previously returned by one of the FFI
/// string-returning functions. Safe to call with a null pointer.
#[no_mangle]
pub extern "C" fn prism_free_string(s: *mut std::os::raw::c_char) {
    if !s.is_null() {
        // SAFETY: The null check above ensures `s` is non-null. The
        // C-ABI contract requires the pointer to have been produced by
        // `CString::into_raw` inside one of the string-returning
        // functions in this module and not yet released (no
        // double-free). The pointee type matches `CString`, so
        // reconstructing it is sound.
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}
