//! Runtime executable loader — opens, validates, and prepares
//! SealedComputeImageExecutable images for execution.

pub mod scheduling;
pub mod ledger;
pub mod memory;

pub mod agent_slot;
pub mod ane_multiplexer;
pub mod signal_bus;
pub mod ecore_pump;
pub mod interceptors;
pub mod world;
pub mod components;
pub mod resources;
pub mod systems;
pub mod integration;

pub mod executable_bindings;
pub mod executable_lane;
pub mod executable_profile;
pub mod executable_seal;
pub mod executable_session;

pub use agent_slot::{AgentSlot, MultiplexerState, STATE_IDLE, STATE_PREFETCHING, STATE_READY, STATE_EXECUTING};
pub use components::*;
pub use world::{Entity, World};

pub use signal_bus::*;
pub use interceptors::*;
pub use executable_bindings::*;
pub use executable_lane::*;
pub use executable_profile::*;
pub use executable_seal::*;
pub use executable_session::*;
