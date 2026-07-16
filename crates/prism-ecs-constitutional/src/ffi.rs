//! C FFI for ECS world operations on iOS.
//! Exports `#[no_mangle] pub extern "C"` functions callable from Swift.
//!
//! Uses parking_lot::Mutex (no poisoning, no unwrap on lock).

use crate::agent_exec::{AgentConfig, AgentLifecycle, AgentPhase, AgentRun, AgentTask};
use crate::agent_plan::ParentAgentId;
use crate::agent_state::tick;
use crate::types::Timestamp;
use crate::world_txn::{WorldTransitExt, WorldTxn};
use parking_lot::Mutex;
use prism_ecs_core::{Entity, EntityKind, World};
use std::ffi::{CStr, CString};

#[no_mangle]
pub extern "C" fn prism_world_create() -> *mut Mutex<World> {
    Box::into_raw(Box::new(Mutex::new(World::new())))
}

#[no_mangle]
pub extern "C" fn prism_world_destroy(world: *mut Mutex<World>) {
    if !world.is_null() {
        unsafe {
            drop(Box::from_raw(world));
        }
    }
}

#[no_mangle]
pub extern "C" fn prism_subagent_spawn(
    world: *mut Mutex<World>,
    parent_entity_id: u64,
    task_description: *const std::os::raw::c_char,
) -> u64 {
    let world = unsafe { &mut *world };
    let mut w = world.lock();
    let task = if !task_description.is_null() {
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

#[no_mangle]
pub extern "C" fn prism_subagent_phase(
    world: *mut Mutex<World>,
    entity_id: u64,
) -> *mut std::os::raw::c_char {
    let world = unsafe { &mut *world };
    let w = world.lock();
    let entity = Entity::new(entity_id, 0);
    let phase = w
        .get_component::<AgentPhase>(entity)
        .map(|p| format!("{:?}", p))
        .unwrap_or_else(|| "Unknown".to_string());
    CString::new(phase).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn prism_subagent_lifecycle(
    world: *mut Mutex<World>,
    entity_id: u64,
) -> *mut std::os::raw::c_char {
    let world = unsafe { &mut *world };
    let w = world.lock();
    let entity = Entity::new(entity_id, 0);
    let lifecycle = w
        .get_component::<AgentLifecycle>(entity)
        .map(|l| format!("{:?}", l))
        .unwrap_or_else(|| "Unknown".to_string());
    CString::new(lifecycle).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn prism_subagent_cancel(world: *mut Mutex<World>, entity_id: u64) {
    let world = unsafe { &mut *world };
    let mut w = world.lock();
    let entity = Entity::new(entity_id, 0);
    let _ = w.insert_component(entity, AgentLifecycle::Failed);
}

#[no_mangle]
pub extern "C" fn prism_agent_tick(world: *mut Mutex<World>) -> *mut std::os::raw::c_char {
    let world = unsafe { &mut *world };
    let w = world.lock();
    let _transitions = tick(&w).unwrap_or_default();
    let result = r#"{"status":"tick_complete","transitions":[]}"#.to_string();
    CString::new(result).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn prism_free_string(s: *mut std::os::raw::c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}
