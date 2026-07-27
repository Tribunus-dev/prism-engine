//! Canonical `ExecutionLane` enum — the constitutional home for backend placement identity.
//!
//! The `ExecutionLane` type identifies which hardware lane a kernel dispatch, a
//! scheduling lease, or a backend binding is talking about. It is the type-level
//! answer to "which device does this work go to?".
//!
//! # Authority
//!
//! The kernel crate owns this enum because every lane is a hardware concept
//! (Metal/GPU, ANE, CPU, etc.). The runtime crate uses the type when expressing
//! scheduling decisions (lease assignment, capacity management, dispatch
//! selection). The runtime never defines its own lane enum.
//!
//! # Migration provenance
//!
//! The previous home was `compute-core/src/ecs/backend/placement.rs::ExecutionLane`.
//! The engine file remains in place during the absorption window; the
//! constitutional home is the kernel. When the engine's `lane_capacity` and
//! related scheduling state move into `prism-ecs-runtime::scheduling::state`,
//! they will import from this module rather than the engine path.

use serde::{Deserialize, Serialize};

/// Identifies the hardware execution lane for a kernel dispatch, scheduling
/// lease, or backend binding.
///
/// Each variant corresponds to a distinct physical or logical execution
/// surface. The `Hash`, `Eq`, `Ord`, and `PartialOrd` derives are required
/// for placement maps, per-lane capacity tracking, and BTreeMap-backed
/// registries in the runtime scheduling state.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum ExecutionLane {
    /// Apple Metal GPU lane (MLX-backed GPU on Apple Silicon).
    MlxGpu,
    /// Apple Accelerate CPU lane (vectorized CPU on macOS / iOS).
    AccelerateCpu,
    /// Apple Neural Engine (Core ML ANE).
    CoreAiAne,
    /// Cross-platform CPU lane backed by the candle CPU runtime.
    CandleCpu,
    /// Tenstorrent Tensix lane.
    Tensix,
    /// Intel Level Zero CPU / GPU lane.
    IntelLevelZero,
}

impl ExecutionLane {
    /// Returns `true` if this lane is a Metal/GPU family lane
    /// (i.e. counts against the GPU command-buffer capacity).
    pub fn is_metal_family(self) -> bool {
        matches!(self, Self::MlxGpu | Self::Tensix)
    }

    /// Returns `true` if this lane is the Apple Neural Engine.
    pub fn is_ane(self) -> bool {
        matches!(self, Self::CoreAiAne)
    }

    /// Returns `true` if this lane is a CPU family lane
    /// (i.e. counts against the CPU worker capacity).
    pub fn is_cpu_family(self) -> bool {
        matches!(
            self,
            Self::AccelerateCpu | Self::CandleCpu | Self::IntelLevelZero
        )
    }

    /// Short canonical name (used in receipts and metrics).
    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::MlxGpu => "mlx_gpu",
            Self::AccelerateCpu => "accelerate_cpu",
            Self::CoreAiAne => "core_ai_ane",
            Self::CandleCpu => "candle_cpu",
            Self::Tensix => "tensix",
            Self::IntelLevelZero => "intel_level_zero",
        }
    }
}
