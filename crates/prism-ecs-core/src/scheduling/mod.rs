//! Deterministic schedule compiler for the Prism ECS runtime.
//!
//! This subsystem is the execution constitution for Prism's multi-accelerator
//! inference runtime.  Systems declare access at compile time; the schedule
//! compiler validates causality and hazards, compiles a canonical manifest,
//! and emits a fixed execution order.  Structural mutations flow through
//! a provenance-stamped command buffer that becomes the future receipt ledger
//! seam.
//!
//! ## Module layout
//!
//! | Module | Purpose |
//! |---|---|
//! | `component_id` | Stable numeric IDs, bitwise masks, registries |
//! | `access` | `ComponentSet` / `ResourceSet` traits + tuple macros |
//! | `metadata` | `Stage`, `SystemId`, `SystemSpec`, `ErasedSystem` |
//! | `command` | `Command`, `StampedCommand`, `CommandWriter` |
//! | `error` | `MaskError`, `RegistryError`, `ScheduleError`, `CommandError` |
//! | `graph` | Dependency graph builder with cycle detection |

pub mod access;
pub mod command;
pub mod component_id;
pub mod error;
pub mod graph;
pub mod metadata;
