//! `prism_ecs_runtime::runtime` — constitutional home for the provider-neutral
//! runtime kernel.
//!
//! This module is the migration target for the engine's legacy
//! `compute-core/src/ecs/runtime/` directory (92 files, 21,448 LOC, deleted in
//! the engine-deletion migration `changelogs/2026-07-27-engine-subsystem-deletion-runtime.md`).
//!
//! # Authority
//!
//! The constitutional `runtime` module owns the canonical authority for the
//! provider-neutral runtime kernel: schedule, command handling, admission,
//! dispatch coordination, ports, and receipts. The `runtime` namespace is a
//! thin re-export layer over the existing constitutional modules
//! ([`crate::schedule`], [`crate::scheduling`], [`crate::ports`],
//! [`crate::systems`], [`crate::backend`], [`crate::kernel`], and
//! [`crate::engine_receipts`]) so the canonical surface is grep-able from a
//! single import path.
//!
//! # Migration map (engine `runtime/*` → constitutional `runtime::*`)
//!
//! | Engine (legacy)                       | Constitutional (canonical)                 |
//! |---------------------------------------|--------------------------------------------|
//! | `ecs::runtime::world`                 | `prism_ecs_core::Entity` / `World`         |
//! | `ecs::runtime::world_txn`             | `prism_ecs_constitutional::world_txn`      |
//! | `ecs::runtime::constitutional_world_txn` | (alias of `prism_ecs_constitutional::world_txn`) |
//! | `ecs::runtime::scheduling::*`         | `runtime::scheduling` / `runtime::schedule` |
//! | `ecs::runtime::ledger::*`             | `runtime::receipts` (see [`receipts`])     |
//! | `ecs::runtime::engram::*`             | (engine-internal; see `legacy_runtime::engram`) |
//! | `ecs::runtime::components::*`         | (engine-internal; see `legacy_runtime::components`) |
//! | `ecs::runtime::resources::*`          | (engine-internal; see `legacy_runtime::resources`) |
//! | `ecs::runtime::systems::*`            | `runtime::systems`                         |
//! | `ecs::runtime::signal_bus`            | `runtime::signal_bus` (see [`signal_bus`])  |
//! | `ecs::runtime::interceptors`          | (engine-internal; see `legacy_runtime::interceptors`) |
//! | `ecs::runtime::pump_pool`             | (engine-internal; see `legacy_runtime::pump_pool`) |
//! | `ecs::runtime::ane_multiplexer`       | (engine-internal; see `legacy_runtime::ane_multiplexer`) |
//! | `ecs::runtime::ecore_pump`            | (engine-internal; see `legacy_runtime::ecore_pump`) |
//! | `ecs::runtime::npu_pump`              | (engine-internal; see `legacy_runtime::npu_pump`) |
//! | `ecs::runtime::compilation_systems`   | (engine-internal; see `legacy_runtime::compilation_systems`) |
//! | `ecs::runtime::serving::*`            | (engine-internal; see `legacy_runtime::serving`) |
//! | `ecs::runtime::integration::*`        | (engine-internal; see `legacy_runtime::integration`) |
//! | `ecs::runtime::stage_graph`           | (engine-internal; see `legacy_runtime::stage_graph`) |
//! | `ecs::runtime::executable_*`          | (engine-internal; see `legacy_runtime::executable_*`) |
//! | `ecs::runtime::memory`                | (engine-internal; see `legacy_runtime::memory`) |
//! | `ecs::runtime::agent_slot`            | (engine-internal; see `legacy_runtime::agent_slot`) |
//! | `ecs::runtime::ecs_components`        | (engine-internal; see `legacy_runtime::ecs_components`) |
//!
//! Engine-coupled code (multiplexers, pumps, interceptors, executors, etc.)
//! depends on engine-internal `World` / `Entity` / `Component` types and
//! stays engine-side under `compute-core/src/ecs/legacy_runtime/` (the
//! `legacy_runtime/` directory is the engine-internal execution-plane home,
//! mirroring the `memory_impl/` and `legacy_core/` patterns). Data types
//! and pure abstractions that are engine-independent are re-implemented
//! here in their own submodules, with a single authority per file.
//!
//! # Submodules
//!
//! - [`signal_bus`] — the engine-independent [`RuntimeSignal`] enum and
//!   [`SignalBus`] / [`SignalReceiver`] newtypes (was
//!   `compute-core/src/ecs/runtime/signal_bus.rs`).
//! - [`stages`] — the engine-independent [`Stage`] enum used by the schedule
//!   compiler (Intake → Admission → Residency → Prefill → Decode →
//!   PostDecode → ToolExecution → Maintenance → Receipt).
//! - [`pump_states`] — engine-independent pump state constants
//!   ([`STATE_IDLE`], [`STATE_PREFETCHING`], [`STATE_READY`],
//!   [`STATE_EXECUTING`]) and the [`MultiplexerState`] wrapper.
//! - [`receipts`] — re-export of the constitutional receipt types from
//!   [`crate::engine_receipts`], the constitutional home for tick receipts
//!   and dispatch outcomes.
//! - [`scheduling`] — re-export of [`crate::scheduling`] (the constitutional
//!   scheduling state / systems / evidence / metrics authority).
//! - [`schedule`] — re-export of [`crate::schedule`] (the constitutional
//!   `RuntimeSchedule`, `System`, and `SystemSpec` types).
//! - [`ports`] — re-export of [`crate::ports`] (the constitutional port
//!   surface: dispatcher, lease coordinator, snapshot store, etc.).
//! - [`systems`] — re-export of [`crate::systems`] (the constitutional
//!   engine-system surface: archive, backend, dispatch, residency, etc.).
//! - [`kernel`] — re-export of [`crate::kernel`] (the constitutional
//!   `RuntimeKernel` and command envelope authority).

#![allow(unused_imports)]

pub mod pump_states;
pub mod signal_bus;
pub mod stages;

pub mod kernel;
pub mod ports;
pub mod receipts;
pub mod schedule;
pub mod scheduling;
pub mod systems;
