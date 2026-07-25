//! # prism-ecs-ffi
//!
//! This crate is the only place in the Prism workspace that exposes
//! C-ABI raw-pointer FFI; constitutional crates depend on us for the FFI
//! surface, never the other way around.
//!
//! ## Single authority
//!
//! The single authority owned by this crate is **the C-ABI bridge** for
//! the constitutional ECS. `prism-ecs-constitutional` and the rest of the
//! workspace stay `unsafe`-free; this crate is the only place where
//! `extern "C"` raw-pointer signatures and the corresponding
//! `unsafe { ... }` blocks are permitted.
//!
//! ## Why a separate crate?
//!
//! The `prism-ecs-constitutional` crate is in the "no `unsafe`" list
//! from `AGENTS.md`. The C-ABI bridge for iOS / Swift (`prism_world_*`,
//! `prism_subagent_*`, `prism_agent_tick`, `prism_free_string`) needs
//! raw pointers and the corresponding `unsafe` blocks. Splitting the
//! surface out lets the constitutional crate remain pure Rust while the
//! FFI crate concentrates all `unsafe` and `// SAFETY:` discipline in
//! one auditable location.
//!
//! ## Layering
//!
//! ```text
//! Swift caller
//!      │
//!      ▼
//! prism-ecs-ffi  ← this crate (C-ABI bridge; only `unsafe` here)
//!      │
//!      ▼
//! prism-ecs-constitutional  (typed commands, WorldTxn, no `unsafe`)
//!      │
//!      ▼
//! prism-ecs-core  (entity, world, store primitives; `unsafe` allowed in core)
//! ```
//!
//! The crate dependency direction is downward: this crate depends on
//! `prism-ecs-constitutional` and `prism-ecs-core`, never the reverse.
//!
//! ## Re-exports
//!
//! Every public C-ABI function is re-exported at the crate root so
//! callers can write `prism_ecs_ffi::prism_world_create` (or, in C,
//! link against the symbol directly).

#![allow(clippy::not_unsafe_ptr_arg_deref)] // The C-ABI is the source of truth.

pub mod c_abi;

pub use c_abi::*;
