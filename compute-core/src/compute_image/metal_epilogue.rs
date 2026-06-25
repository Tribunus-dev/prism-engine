//! PRISM-METAL-ANE-HANDOFF-EPILOGUES-0001: Producer-aware Metal epilogue variants.
//!
//! Defines epilogue selection logic for Metal kernels that need to hand off
//! to an ANE consumer, emit Metal-native output, or use a canonical fallback.
//! The decision depends on the consumer lane and its qualification status.

use serde::{Deserialize, Serialize};

use crate::backend::placement::ExecutionLane;
use crate::compilation::activation_abi::{ActivationAbi, SlotLeaseId};

// ── Epilogue variant ─────────────────────────────────────────────────────

/// The chosen epilogue style for a Metal kernel's output.
///
/// * `MetalNative` — output stays in Metal's native layout; no ABI wrapping.
/// * `AneHandoff` — output arranged for direct ANE consumption via the
///   provided [`ActivationAbi`].
/// * `Canonical` — output arranged in a lane-neutral canonical layout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetalKernelEpilogue {
    MetalNative,
    AneHandoff(ActivationAbi),
    Canonical(ActivationAbi),
}

// ── Codegen context ──────────────────────────────────────────────────────

/// Full context for Metal codegen, including the resolved epilogue variant
/// and the slot / lane topology.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetalCodegenContext {
    pub chosen_epilogue: MetalKernelEpilogue,
    pub input_slots: Vec<SlotLeaseId>,
    pub output_slot: SlotLeaseId,
    pub output_abi: ActivationAbi,
    pub producer_lane: ExecutionLane,
    pub consumer_lane: ExecutionLane,
}

// ── Selection rules ──────────────────────────────────────────────────────

/// High-level heuristic that maps lane + qualification to an epilogue kind.
///
/// * `AlwaysMetalNative` — always use Metal-native layout.
/// * `AneWhenQualified` — use ANE handoff layout when the next consumer is
///   ANE-qualified.
/// * `CanonicalWhenUncertain` — fall back to canonical layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpilogueSelectionRule {
    AlwaysMetalNative,
    AneWhenQualified,
    CanonicalWhenUncertain,
}

// ── Selector ─────────────────────────────────────────────────────────────

/// Produce a selection rule based on the consumer lane and its qualification
/// status.
///
/// Returns:
/// - `AneWhenQualified` when `consumer_lane` is `CoreMlAne` and `is_qualified`
///   is true.
/// - `AlwaysMetalNative` when the consumer runs on the Metal GPU
///   (`MlxGpu`).
/// - `CanonicalWhenUncertain` in all other cases.
pub fn select_epilogue(
    consumer_lane: ExecutionLane,
    is_qualified: bool,
) -> EpilogueSelectionRule {
    match consumer_lane {
        ExecutionLane::CoreMlAne if is_qualified => EpilogueSelectionRule::AneWhenQualified,
        ExecutionLane::MlxGpu => EpilogueSelectionRule::AlwaysMetalNative,
        _ => EpilogueSelectionRule::CanonicalWhenUncertain,
    }
}

/// Apply a selection rule to a concrete [`MetalCodegenContext`] and produce
/// the resolved [`MetalKernelEpilogue`].
///
/// The `AneHandoff` and `Canonical` variants clone the context's `output_abi`
/// to capture the exact layout contract that the consumer expects.
pub fn choose_epilogue(
    context: &MetalCodegenContext,
    rule: &EpilogueSelectionRule,
) -> MetalKernelEpilogue {
    match rule {
        EpilogueSelectionRule::AlwaysMetalNative => MetalKernelEpilogue::MetalNative,
        EpilogueSelectionRule::AneWhenQualified => {
            MetalKernelEpilogue::AneHandoff(context.output_abi.clone())
        }
        EpilogueSelectionRule::CanonicalWhenUncertain => {
            MetalKernelEpilogue::Canonical(context.output_abi.clone())
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::backend::placement::ExecutionLane;
    use crate::compilation::activation_abi::{ActivationAbi, SlotLeaseId};
    use crate::compute_image::metal_epilogue::{
        choose_epilogue, select_epilogue, EpilogueSelectionRule, MetalCodegenContext,
        MetalKernelEpilogue,
    };

    /// A stub `ActivationAbi` value used throughout — the actual variant
    /// does not affect the selection logic, so any valid variant suffices.
    fn stub_abi() -> ActivationAbi {
        use crate::compilation::activation_abi::MetalOnlyParams;
        use crate::compilation::phase_ir::TensorDtype;
        ActivationAbi::MetalOnly(MetalOnlyParams {
            name: "test".into(),
            dtype: TensorDtype::Float16,
            byte_count: 4096,
        })
    }

    /// A minimal `MetalCodegenContext` with caller-supplied lanes.
    fn context_with(
        producer_lane: ExecutionLane,
        consumer_lane: ExecutionLane,
    ) -> MetalCodegenContext {
        MetalCodegenContext {
            chosen_epilogue: MetalKernelEpilogue::MetalNative,
            input_slots: vec![],
            output_slot: SlotLeaseId(1),
            output_abi: stub_abi(),
            producer_lane,
            consumer_lane,
        }
    }

    // ── select_epilogue ──────────────────────────────────────────────

    #[test]
    fn test_select_ane_epilogue_for_qualified_ane_consumer() {
        let rule = select_epilogue(ExecutionLane::CoreMlAne, true);
        assert_eq!(rule, EpilogueSelectionRule::AneWhenQualified);
    }

    #[test]
    fn test_select_metal_epilogue_for_metal_consumer() {
        let rule = select_epilogue(ExecutionLane::MlxGpu, false);
        assert_eq!(rule, EpilogueSelectionRule::AlwaysMetalNative);

        // Qualification must not change the outcome for a metal consumer.
        let rule_qualified = select_epilogue(ExecutionLane::MlxGpu, true);
        assert_eq!(rule_qualified, EpilogueSelectionRule::AlwaysMetalNative);
    }

    #[test]
    fn test_select_canonical_when_uncertain() {
        // Accelerate (CPU) → canonical, regardless of qualification
        let rule = select_epilogue(ExecutionLane::AccelerateCpu, false);
        assert_eq!(rule, EpilogueSelectionRule::CanonicalWhenUncertain);

        let rule_q = select_epilogue(ExecutionLane::AccelerateCpu, true);
        assert_eq!(rule_q, EpilogueSelectionRule::CanonicalWhenUncertain);

        // ANE but *not* qualified → canonical
        let rule_unqualified_ane = select_epilogue(ExecutionLane::CoreMlAne, false);
        assert_eq!(
            rule_unqualified_ane,
            EpilogueSelectionRule::CanonicalWhenUncertain
        );

        // Unknown/unexpected lanes → canonical
        let rule_tensix = select_epilogue(ExecutionLane::Tensix, false);
        assert_eq!(rule_tensix, EpilogueSelectionRule::CanonicalWhenUncertain);
    }

    // ── choose_epilogue ──────────────────────────────────────────────

    #[test]
    fn test_choose_epilogue_produces_correct_variant() {
        let ctx = context_with(ExecutionLane::MlxGpu, ExecutionLane::CoreMlAne);

        // AlwaysMetalNative → MetalNative
        let ep = choose_epilogue(&ctx, &EpilogueSelectionRule::AlwaysMetalNative);
        assert_eq!(ep, MetalKernelEpilogue::MetalNative);

        // AneWhenQualified → AneHandoff with the context's output_abi
        let ep = choose_epilogue(&ctx, &EpilogueSelectionRule::AneWhenQualified);
        assert_eq!(ep, MetalKernelEpilogue::AneHandoff(stub_abi()));

        // CanonicalWhenUncertain → Canonical with the context's output_abi
        let ep = choose_epilogue(&ctx, &EpilogueSelectionRule::CanonicalWhenUncertain);
        assert_eq!(ep, MetalKernelEpilogue::Canonical(stub_abi()));
    }
}
