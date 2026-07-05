//! Always-available projection data types.
//!
//! Extracted from `projection_identity` (research surface, `mlx-backend`) so
//! that production code — `readiness_gates` today — can name these plain data
//! enums without pulling the MLX projection-instrumentation stack into the
//! hermetic `prism-backend` build. `projection_identity` re-exports them, so
//! research-surface paths (`crate::projection_identity::RuntimeMode`) are
//! unchanged.

/// Current runtime mode affecting dispatch decisions.
///
/// Shared between MLX backend and Candle CPU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    /// Safe mode: no fused int4 for FFN projections, no mapped no-copy
    /// for quantized weights — use the authority (dequantize + matmul) path.
    Safe,
    /// Qualified mode: operations permitted only after a crash-free parity
    /// probe has passed for this exact shape class.
    Qualified,
    /// Experimental mode: all paths enabled.
    Experimental,
}
