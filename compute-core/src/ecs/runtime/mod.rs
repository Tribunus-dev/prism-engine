//! Runtime executable loader — opens, validates, and prepares
//! SealedComputeImageExecutable images for execution.

pub mod ledger;
pub mod memory;
pub mod scheduling;
pub mod engram;

pub mod agent_slot;
pub mod ane_multiplexer;
pub mod compilation_systems;
pub mod components;
pub mod ecore_pump;
pub mod ecs_components;
pub mod integration;
pub mod interceptors;
pub mod npu_pump;
pub mod pump_pool;
pub mod resources;
pub mod signal_bus;
pub mod stage_graph;
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

#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use executable_bindings::*;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use executable_lane::*;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use executable_profile::*;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use executable_seal::*;
#[cfg(feature = "mlx-backend")]
pub use executable_session::*;
pub use interceptors::*;
pub use signal_bus::*;
