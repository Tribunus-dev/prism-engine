//! Engine-internal re-export shim for the tool surface (post-deletion).
//!
//! This module is the engine-internal home for the tool surface
//! after the constitutional deletion of `compute-core/src/ecs/tools/`.
//! It re-exports the constitutional surface from
//! `prism_ecs_server::tools` and adds the engine-coupled
//! extensions that depend on engine-internal state:
//!
//! - [`dispatch`] — engine-side `execute_tool_call` /
//!   `sandbox_execute` / `default_sandbox_tools` that route
//!   `list_devices` through the engine wrapper at
//!   [`list_devices::tool_list_devices`] so the engine's
//!   `device::global_registry()` is the canonical source for
//!   device data.
//! - [`list_devices`] — engine-side wrapper that queries
//!   `crate::ecs::device::global_registry()` and forwards to the
//!   constitutional [`prism_ecs_server::tools::list_devices::tool_list_devices`].
//! - [`retry_with_error`] — engine-side wrapper that drives the
//!   engine's `mlx-backend` `profiled_executor` to retry an
//!   unrepairable tool call, then re-uses the constitutional
//!   `parse_and_repair` for the final repair step.
//!
//! # Authority boundary
//!
//! This module is the engine-side façade for the constitutional
//! tool surface. It is the only engine file that imports
//! `prism_ecs_server::tools`. All engine callers go through
//! `crate::ecs::legacy_tools::*` (or its re-export at
//! `crate::ecs::tools::*` / `tribunus_compute_core::tools::*` for
//! downstream consumers that haven't migrated yet).

pub use prism_ecs_server::tools::*;

// Engine-coupled extensions — the two pieces of the original
// `compute-core/src/ecs/tools/` module that depend on engine-internal
// state. They live here so the constitutional surface stays pure.
pub mod dispatch;
pub mod list_devices;
#[cfg(feature = "mlx-backend")]
pub mod retry_with_error;
