//! Runtime executable loader — opens, validates, and prepares
//! SealedComputeImageExecutable images for execution.

pub mod ledger;
pub mod memory;
pub mod scheduling;

pub mod agent_slot;
pub mod ane_multiplexer;
pub mod components;
pub mod ecore_pump;
pub mod integration;
pub mod interceptors;
pub mod npu_pump;
pub mod pump_pool;
pub mod resources;
pub mod signal_bus;
pub mod systems;
pub mod world;

pub mod executable_bindings;
pub mod executable_lane;
pub mod executable_profile;
pub mod executable_seal;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod executable_session;

pub use agent_slot::{
    AgentSlot, MultiplexerState, STATE_EXECUTING, STATE_IDLE, STATE_PREFETCHING, STATE_READY,
};
pub use components::*;
pub use world::{Entity, World};

pub use executable_bindings::*;
pub use executable_lane::*;
pub use executable_profile::*;
pub use executable_seal::*;
pub use executable_session::*;
pub use interceptors::*;
pub use signal_bus::*;
