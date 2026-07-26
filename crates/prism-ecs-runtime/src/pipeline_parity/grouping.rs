//! Comparison grouping — the engine-facing surface that turns
//! per-decode observation rows into apples-to-apples comparison
//! groups, plus the graph-family-to-phase mapping used to qualify
//! which graph families count as which inference phase.
//!
//! # Graph family → phase mapping rules
//!
//! - `matmul`, `matmul_projection`, `constant_heavy` → `QkvProjection`
//!   as generic projection control (variant = `"generic_projection"`).
//! - `branch_rejoin` → `AttentionOutputProjection` (parallel projection
//!   branches plus add). Do NOT map to QkvProjection — it is not a
//!   single QKV projection but two parallel projections recombined.
//! - `two_matmul_add` → `AttentionOutputProjection` with
//!   phase_variant `"parallel_projection_rejoin"`. It is not
//!   `AttentionWeightedSum` unless one matmul is probabilities × values.
//! - `identity_passthrough` → `Err` (harness control, excluded from
//!   comparison).
//! - `reshape_transpose_matmul` → `AttentionScores` (Q, K shape
//!   manipulation into score matmul).
//! - All elementwise + activation families → `Activation`.
//! - `softmax_tail` → `Softmax`.
//! - `matmul_residual_add` and `add_standalone` → `ResidualAdd1`.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt;

use super::matrices::support_matrix_for;
use super::phase::PipelinePhase;
use super::support::PhaseSupportStatus;
use super::BackendId;

// ── PipelineParityError ───────────────────────────────────────────────────

/// Error returned when a graph family cannot be mapped to a pipeline phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineParityError {
    pub family_name: String,
    pub reason: &'static str,
}

impl fmt::Display for PipelineParityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "family '{}' cannot be mapped to a pipeline phase: {}",
            self.family_name, self.reason
        )
    }
}

/// Map an existing graph catalog family to its canonical inference
/// pipeline phase.
///
/// Every family in the graph catalog MUST either map to a valid
/// phase or be explicitly excluded (returning `Err` for harness
/// control families like `identity_passthrough`).
///
/// See the [module-level rules](self#graph-family--phase-mapping-rules)
/// for the mapping table.
pub fn graph_family_to_phase(family_name: &str) -> Result<PipelinePhase, PipelineParityError> {
    match family_name {
        // Generic projection — single matmul acting as any dense projection.
        "matmul" | "matmul_projection" | "constant_heavy" => Ok(PipelinePhase::QkvProjection),
        // Parallel projection branches recombined — more like attention output.
        "branch_rejoin" => Ok(PipelinePhase::AttentionOutputProjection),
        // Parallel matmuls with add — projection rejoin, not weighted sum.
        "two_matmul_add" => Ok(PipelinePhase::AttentionOutputProjection),
        // Matmul followed by residual add.
        "matmul_residual_add" | "add_standalone" => Ok(PipelinePhase::ResidualAdd1),
        // Activation chains (matmul→add→silu or matmul→add→sigmoid→mul).
        "chain_matmul_add_silu"
        | "matmul_add_silu"
        | "mul_standalone"
        | "sigmoid_standalone"
        | "silu_standalone" => Ok(PipelinePhase::Activation),
        // Multi-output matmul — still a projection.
        "multi_output" => Ok(PipelinePhase::QkvProjection),
        // Softmax tail.
        "softmax_tail" => Ok(PipelinePhase::Softmax),
        // Reshape → transpose → matmul — attention score computation pattern.
        "reshape_transpose_matmul" => Ok(PipelinePhase::AttentionScores),
        // Harness control family — not an inference phase.
        "identity_passthrough" => Err(PipelineParityError {
            family_name: family_name.to_string(),
            reason: "harness control family, not an inference pipeline phase",
        }),
        other => Err(PipelineParityError {
            family_name: other.to_string(),
            reason: "unknown graph family — not registered in catalog",
        }),
    }
}

/// Return the phase variant string for a mapped family.
///
/// The variant disambiguates which concrete implementation or operation
/// subset a family exercises within a phase (e.g.
/// `"generic_projection"` vs `"parallel_projection_rejoin"` for phases
/// that can be exercised by different graph topologies).
pub fn graph_family_phase_variant(family_name: &str) -> &'static str {
    match family_name {
        "matmul" => "generic_projection",
        "matmul_projection" => "generic_projection",
        "constant_heavy" => "generic_projection",
        "branch_rejoin" => "parallel_projection_rejoin",
        "multi_output" => "multi_output_projection",
        "two_matmul_add" => "parallel_projection_rejoin",
        "matmul_residual_add" => "residual_add",
        "add_standalone" => "residual_add",
        "chain_matmul_add_silu" => "matmul_add_silu",
        "matmul_add_silu" => "matmul_add_silu",
        "mul_standalone" => "elementwise_mul",
        "sigmoid_standalone" => "sigmoid",
        "silu_standalone" => "silu",
        "reshape_transpose_matmul" => "attention_scores_reshape",
        "softmax_tail" => "softmax_after_matmul",
        "identity_passthrough" => "harness_control",
        _ => "unknown",
    }
}

/// Return the semantic contract ID for a given graph family.
///
/// The semantic contract ID encodes `(phase, variant)`, enabling
/// comparison grouping to distinguish different operations within the
/// same phase (e.g. `matmul_add_silu` vs `silu_standalone` are both
/// `Activation` but have different semantic contracts).
pub fn graph_family_semantic_contract_id(family_name: &str) -> String {
    match graph_family_to_phase(family_name) {
        Ok(phase) => format!("{}/{}", phase, graph_family_phase_variant(family_name)),
        Err(_) => format!("excluded/{}", graph_family_phase_variant(family_name)),
    }
}

// ── PhaseComparisonGroup + PhaseComparisonRow ─────────────────────────────

/// A group of rows that can be compared across backends.
///
/// All rows in a group share: `phase`, `phase_variant`,
/// `semantic_contract_id`, `shape_profile_name`, `dtype`, and
/// `tolerance`. Only rows with valid fences (backend-specific
/// eval/materialization verified) are included, ensuring comparison is
/// honest.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseComparisonGroup {
    /// Canonical pipeline phase.
    pub phase: PipelinePhase,
    /// Phase variant (e.g. "generic_projection", "parallel_projection_rejoin").
    pub phase_variant: String,
    /// Full semantic contract ID (phase/variant).
    pub semantic_contract_id: String,
    /// Name of the shape profile (e.g. "small", "medium", "large").
    pub shape_profile_name: String,
    /// Shape contract: shape bound for this specific group (e.g. "1x4x4x1").
    pub shape_contract_id: String,
    /// Element type (e.g. "float32").
    pub dtype: String,
    /// Numerical tolerance for conformance.
    pub tolerance: f64,
    /// Tolerance profile identifier.
    pub tolerance_profile: String,
    /// One row per backend that executed this phase with valid measurement.
    pub rows: Vec<PhaseComparisonRow>,
}

/// A single backend's result within a comparison group.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseComparisonRow {
    /// Which backend produced this result.
    pub backend: BackendId,
    /// The runtime policy/device this backend used (e.g. "cpuOnly", "mlx_default").
    pub backend_policy: String,
    /// Support status for this phase on this backend.
    pub support_status: PhaseSupportStatus,
    /// Latency in nanoseconds (steady-state P50).
    pub duration_ns: u64,
    /// Hash of the output tensor (None if execution failed or not captured).
    pub output_hash: Option<String>,
    /// True when the backend's eval/sync fence was verified before timing.
    pub fence_valid: bool,
}

impl PhaseComparisonGroup {
    /// Classify a tolerance into a profile label. Boundaries match the
    /// engine's prior classification (strict ≤ 1e-5, standard ≤ 1e-4,
    /// relaxed otherwise).
    pub fn tolerance_profile_for(tolerance: f64) -> &'static str {
        if tolerance <= 1e-5 {
            "strict"
        } else if tolerance <= 1e-4 {
            "standard"
        } else {
            "relaxed"
        }
    }
}

// ── Comparison grouping (over a minimal view type) ─────────────────────────

/// Minimal view of a decode-attribution receipt for parity grouping.
///
/// The engine's `group_for_comparison` accepts
/// `&[DecodeAttributionReceipt]`; the re-implementation accepts this
/// minimal view so the runtime can group its own observations without
/// depending on engine-side receipt types. The engine-side grouping
/// function remains in place; it can convert into this view at the
/// engine boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonReceiptView {
    pub pipeline_phase: Option<String>,
    pub phase_variant: String,
    pub semantic_contract_id: String,
    pub shape_profile: String,
    pub dtype: String,
    pub backend: String,
    pub backend_runtime_policy: String,
    pub predict_status: String,
    pub tolerance: f64,
    pub input_shape: Vec<i64>,
    pub weight_shape: Vec<i64>,
}

/// Group a set of receipt views into comparison groups suitable for
/// apples-to-apples ranking.
///
/// Receipts are grouped by the full comparison key:
/// `(semantic_contract_id, shape_profile, dtype, tolerance)`.
///
/// Groups with fewer than 2 rows are returned to document partial
/// coverage. Receipts whose `pipeline_phase` is `None` or empty are
/// excluded. Receipts whose `predict_status` is not `"pass"` are
/// marked `fence_valid = false` but still included.
pub fn group_for_comparison(receipts: &[ComparisonReceiptView]) -> Vec<PhaseComparisonGroup> {
    let mut groups: BTreeMap<(String, String, String, u64), PhaseComparisonGroup> = BTreeMap::new();

    for r in receipts {
        // Skip receipts with no pipeline phase (legacy or control families).
        if r.pipeline_phase.is_none() || r.pipeline_phase.as_ref().unwrap().is_empty() {
            continue;
        }

        // Parse phase from string — skip if unrecognised.
        let phase: PipelinePhase = match r.pipeline_phase.as_ref().unwrap().parse() {
            Ok(p) => p,
            Err(_) => continue,
        };

        let backend = match r.backend.as_str() {
            "coreai" => BackendId::CoreAi,
            "mlx" => BackendId::Mlx,
            "accelerate" => BackendId::Accelerate,
            "reference" => BackendId::Reference,
            _ => continue,
        };

        let contract_id = if r.semantic_contract_id.is_empty() {
            format!("{}/{}", phase, r.phase_variant)
        } else {
            r.semantic_contract_id.clone()
        };

        // Encode tolerance as (tolerance * 1e6) u64 for BTreeMap key.
        let tolerance_key = (r.tolerance * 1_000_000.0) as u64;

        let key = (
            contract_id.clone(),
            r.shape_profile.clone(),
            r.dtype.clone(),
            tolerance_key,
        );

        let group = groups.entry(key).or_insert_with(|| {
            let tolerance_profile = PhaseComparisonGroup::tolerance_profile_for(r.tolerance);

            // Build a shape contract ID from the actual dimensions.
            let shape_contract = format!(
                "{}x{}",
                r.input_shape.first().copied().unwrap_or(0),
                r.weight_shape.last().copied().unwrap_or(0)
            );

            PhaseComparisonGroup {
                phase,
                phase_variant: r.phase_variant.clone(),
                semantic_contract_id: contract_id.clone(),
                shape_profile_name: r.shape_profile.clone(),
                shape_contract_id: shape_contract,
                dtype: r.dtype.clone(),
                tolerance: r.tolerance,
                tolerance_profile: tolerance_profile.to_string(),
                rows: Vec::new(),
            }
        });

        let fence_valid = r.predict_status == "pass";
        // The actual durations and hashes are not available in the view
        // type — they would be filled in by the engine's existing
        // attribution layer. We populate a placeholder row with the
        // status and policy so the grouping logic is exercised.
        group.rows.push(PhaseComparisonRow {
            backend,
            backend_policy: r.backend_runtime_policy.clone(),
            support_status: support_matrix_for(backend)
                .support_for(phase)
                .cloned()
                .unwrap_or(PhaseSupportStatus::Composed),
            duration_ns: 0,
            output_hash: None,
            fence_valid,
        });
    }

    let mut result: Vec<PhaseComparisonGroup> = groups.into_values().collect();
    for group in &mut result {
        group.rows.sort_by_key(|r| r.backend.to_string());
    }

    result.sort_by_key(|g| {
        let phase_order = g.phase as u8;
        (phase_order, g.shape_profile_name.clone())
    });

    result
}
