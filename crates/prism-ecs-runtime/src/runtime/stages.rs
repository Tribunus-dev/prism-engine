//! Engine-independent `Stage` enum used by the runtime schedule compiler.
//!
//! This module owns the canonical authority for the temporal execution band
//! vocabulary that the schedule compiler uses to partition systems. The
//! enum is engine-independent (no `World` or `Entity` references); the
//! engine-coupled schedule compiler that consumes these stages lives in the
//! engine's `legacy_runtime::scheduling` and depends on this module for the
//! data type.
//!
//! Migration map: `compute-core/src/ecs/runtime/scheduling/metadata::Stage`
//! (engine legacy) → `prism_ecs_runtime::runtime::stages::Stage`
//! (constitutional).
//!
//! # Ordering
//!
//! Stages are ordered by discriminant; `Stage::ALL` is the canonical
//! declaration order used by the schedule compiler's barrier-group step.

use serde::{Deserialize, Serialize};

/// Temporal execution band imposed by the schedule compiler.
///
/// All systems in one stage complete (including command-buffer drain) before
/// any system in the next stage runs. Stages are ordered by discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Stage {
    /// Request intake – enqueue commands, validate inputs.
    Intake = 0,
    /// Admission control – check budgets, policy, concurrency limits.
    Admission = 1,
    /// Weight and cache residency – migrate data to compute hardware.
    Residency = 2,
    /// ANE-assisted prefill – compute full KV cache for a prompt.
    Prefill = 3,
    /// GPU decode loop – autoregressive token generation.
    Decode = 4,
    /// Post-decode processing – MTP speculation, grammar masking, tool calls.
    PostDecode = 5,
    /// Tool execution – run external functions, collect results.
    ToolExecution = 6,
    /// Periodic maintenance – watchdog, budget reaper, migration tick.
    Maintenance = 7,
    /// Terminal receipt emission – finalize and publish state.
    Receipt = 8,
}

impl Stage {
    /// All stages in declaration order.
    pub const ALL: &'static [Stage] = &[
        Stage::Intake,
        Stage::Admission,
        Stage::Residency,
        Stage::Prefill,
        Stage::Decode,
        Stage::PostDecode,
        Stage::ToolExecution,
        Stage::Maintenance,
        Stage::Receipt,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_stages_in_declaration_order() {
        assert_eq!(Stage::ALL.len(), 9);
        assert_eq!(Stage::ALL[0], Stage::Intake);
        assert_eq!(Stage::ALL[8], Stage::Receipt);
    }

    #[test]
    fn stages_are_ordered_by_discriminant() {
        // Stage ordering is the schedule compiler's barrier-group key.
        assert!(Stage::Intake < Stage::Admission);
        assert!(Stage::Admission < Stage::Residency);
        assert!(Stage::Residency < Stage::Prefill);
        assert!(Stage::Prefill < Stage::Decode);
        assert!(Stage::Decode < Stage::PostDecode);
        assert!(Stage::PostDecode < Stage::ToolExecution);
        assert!(Stage::ToolExecution < Stage::Maintenance);
        assert!(Stage::Maintenance < Stage::Receipt);
    }

    #[test]
    fn stage_serializes_with_discriminant_name() {
        let s = Stage::Decode;
        let json = serde_json::to_string(&s).expect("serialize");
        assert_eq!(json, "\"Decode\"");
    }
}
