//! `pipeline::lowering` — Core ML / ANE / MLX / Accelerate lowering surface.
//!
//! This module owns the canonical authority for the lowering
//! pipeline: per-op lowering parameter types, the lowering receipts,
//! and the F32 matmul test dataset that validates that real-backend
//! lowering preserves the already-qualified routes.
//!
//! The engine's per-backend lowering adapters
//! (Core ML / Accelerate / MLX) are hardware-gated and not in this
//! crate; the constitutional surface ships the typed contract.

pub mod dataset;
pub mod params;
pub mod receipts;
