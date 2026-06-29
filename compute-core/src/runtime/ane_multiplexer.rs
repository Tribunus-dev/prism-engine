//! P-core ANE multiplexer — ECS system.
//!
//! Runs as a dedicated thread: acquires the World's RwLock read guard,
//! iterates all alive entities whose AgentSlot is in STATE_READY
//! (prefetched by the E-core pump), dispatches them to the Apple Neural
//! Engine (ANE), and transitions the state through EXECUTING → IDLE.
//!
//! Topology-mode selection (Slice4/8/16/32) adjusts the ANE dispatch
//! stride based on the number of active agent slots, mirroring the
//! existing DynamicMultiplexer logic.

use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use super::agent_slot::{MultiplexerState, STATE_READY, STATE_EXECUTING, STATE_IDLE};
use crate::runtime::world::Entity;
use crate::runtime::components::AgentSlot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyMode {
    Slice4,
    Slice8,
    Slice16,
    Slice32,
}

pub struct DynamicMultiplexer {
    pub current_mode: TopologyMode,
    pub pending_requests: Vec<usize>,
    pub active_execution_vector: Vec<Option<usize>>,
}

impl DynamicMultiplexer {
    pub fn new() -> Self {
        Self {
            current_mode: TopologyMode::Slice32,
            pending_requests: Vec::new(),
            active_execution_vector: vec![None; 32],
        }
    }

    pub fn evaluate_topology_shift(&mut self) -> Option<TopologyMode> {
        let active = self
            .active_execution_vector
            .iter()
            .filter(|s| s.is_some())
            .count();
        let load = active + self.pending_requests.len();
        let target = match load {
            0..=4 => TopologyMode::Slice4,
            5..=8 => TopologyMode::Slice8,
            9..=16 => TopologyMode::Slice16,
            _ => TopologyMode::Slice32,
        };
        if target != self.current_mode {
            Some(target)
        } else {
            None
        }
    }

    pub fn transition_topology(&mut self, new: TopologyMode) {
        self.current_mode = new;
        let stride = match new {
            TopologyMode::Slice4 => 40000,
            TopologyMode::Slice8 => 20000,
            TopologyMode::Slice16 => 10000,
            TopologyMode::Slice32 => 5000,
        };
        eprintln!(
            "[multiplexer] topology shifted to {:?}, stride={}",
            new, stride
        );
    }
}

/// Spawn the P-core ANE multiplexer on a dedicated thread.
///
/// Polls agent slots that have been prefetched (`STATE_READY`) via the
/// ECS World's RwLock read guard and dispatches them to the ANE.
/// After execution returns the slot to `STATE_IDLE` so the E-core pump
/// can prefetch the next batch of weights.
pub fn spawn_ane_multiplexer(
    state: Arc<MultiplexerState>,
    dynamic_mux: Arc<Mutex<DynamicMultiplexer>>,
) -> JoinHandle<()> {
    thread::spawn(move || loop {
        // Evaluate topology load and shift if warranted.
        if let Ok(mut mux) = dynamic_mux.lock() {
            if let Some(new_mode) = mux.evaluate_topology_shift() {
                mux.transition_topology(new_mode);
            }
        }

        // Acquire read lock — iterates under the same guard for consistency.
        let world = state.world.read();

        for i in 0..32u32 {
            let entity = Entity(i);
            if !world.is_alive(entity) {
                continue;
            }
            let Some(slot) = world.get::<AgentSlot>(entity) else {
                continue;
            };

            let prev = slot.load_state();
            if prev == STATE_READY {
                if slot.try_transition(STATE_READY, STATE_EXECUTING) {
                    // Placeholder: dispatch `slot.surface_id` to ANE.
                    // Real implementation would submit a MIL program
                    // referencing the slot's weight region.
                    slot.store_state(STATE_IDLE);
                }
            }
        }
        // Read guard dropped here.

        std::thread::yield_now();
    })
}
