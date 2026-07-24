//! Fused kernel operation legality checking for SpatialIR.
//!
//! Determines which sequences of operations can be legally fused into a single
//! Metal kernel. The evolution search uses this module to enumerate valid fusion
//! schedules and benchmark them against unfused baselines.
//!
//! # Fusion rules
//!
//! - A fused kernel must start with a matrix-vector multiply (GEMV).
//! - At most one matmul per fusion — two GEMVs cannot share a kernel.
//! - All post-matmul ops must be element-wise (activations, normalization, etc.).
//! - Maximum 4 operations per fused kernel (practical Metal threadgroup limit).
//! - Threadgroup geometry must be valid for Metal (width ≤ 256, height ≤ 64,
//!   total threads ≤ 1024).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// FusableOp
// ---------------------------------------------------------------------------

/// Operations that can participate in fusion.
///
/// Each variant represents a compute operation that can be composed into a
/// single Metal kernel, provided it passes [`check_fusion_legality`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FusableOp {
    /// Ternary GEMV (the base operation for ternary-fused kernels).
    TernaryGemv,
    /// Standard FP16 GEMV.
    FpGemv,
    /// SiLU activation: `x * sigmoid(x)`.
    Silu,
    /// RMS normalization.
    RmsNorm,
    /// Rotary position embedding.
    Rope,
    /// Element-wise addition (residual connection).
    ElementWiseAdd,
}

impl FusableOp {
    /// Returns `true` if this operation is a matrix-vector multiply (GEMV).
    pub fn is_matmul(self) -> bool {
        matches!(self, FusableOp::TernaryGemv | FusableOp::FpGemv)
    }

    /// Returns `true` if this operation is element-wise (can follow a matmul
    /// in a fused kernel without additional threadgroup barriers for data
    /// redistribution).
    pub fn is_element_wise(self) -> bool {
        matches!(
            self,
            FusableOp::Silu | FusableOp::RmsNorm | FusableOp::Rope | FusableOp::ElementWiseAdd,
        )
    }

    /// Human-readable label for debug / diagnostic output.
    pub fn label(self) -> &'static str {
        match self {
            FusableOp::TernaryGemv => "ternary_gemv",
            FusableOp::FpGemv => "fp_gemv",
            FusableOp::Silu => "silu",
            FusableOp::RmsNorm => "rms_norm",
            FusableOp::Rope => "rope",
            FusableOp::ElementWiseAdd => "elem_add",
        }
    }
}

// ---------------------------------------------------------------------------
// FusedPermutation
// ---------------------------------------------------------------------------

/// A legal fused kernel permutation: a sequence of operations that can be
/// executed as a single Metal kernel.
///
/// Each `FusedPermutation` is produced by [`check_fusion_legality`] and carries
/// the threadgroup geometry and a content digest used for cache invalidation
/// and kernel identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FusedPermutation {
    /// Ops in execution order.
    pub ops: Vec<FusableOp>,
    /// Threadgroup width for the fused kernel.
    pub threadgroup_width: u32,
    /// Threadgroup height for the fused kernel.
    pub threadgroup_height: u32,
    /// Expected kernel source digest (SHA-256), for cache invalidation.
    ///
    /// The digest is computed from the canonical serialisation of the op
    /// sequence and threadgroup geometry. Any change to the permutation or
    /// tile layout produces a new digest, invalidating any cached kernel
    /// binary.
    pub kernel_digest: [u8; 32],
}

/// Execution layouts evaluated for a fusible region.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FusionStrategy {
    /// Execute the complete legal permutation in one kernel.
    StandardFused,
    /// Split the permutation at materialization boundaries while retaining
    /// fusion inside each resulting stage.
    InterleavedFused { stages: Vec<Vec<FusableOp>> },
    /// Materialize every operation boundary and dispatch each operation
    /// independently.
    PerOperation,
    /// Evolutionary-search candidate that keeps a compiled execution graph
    /// resident and replays it across invocations.
    PersistentMegakernel { search_generation: u32 },
}

impl FusionStrategy {
    /// Stable artifact/runtime namespace independent of strategy parameters.
    /// Search generation remains provenance on the persistent variant rather
    /// than changing the executable policy key.
    pub fn stable_id(&self) -> &'static str {
        match self {
            Self::StandardFused => "standard_fused",
            Self::InterleavedFused { .. } => "interleaved_fused",
            Self::PerOperation => "per_operation",
            Self::PersistentMegakernel { .. } => "persistent_megakernel",
        }
    }
}

/// A comparable candidate produced by fusion strategy evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FusionStrategyCandidate {
    pub strategy: FusionStrategy,
    pub kernel_count: usize,
    pub estimated_latency_ns: u64,
    pub estimated_materialized_bytes: u64,
    pub score: f64,
    pub measured: bool,
}

/// Runtime evidence for one candidate in a strategy evaluation. The index is
/// stable because candidates are emitted in standard, interleaved, then
/// per-operation order.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct FusionMeasurement {
    pub candidate_index: usize,
    pub latency_ns: u64,
    pub materialized_bytes: u64,
}

/// Evaluation result for standard, interleaved, and per-operation layouts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FusionStrategyEvaluation {
    pub candidates: Vec<FusionStrategyCandidate>,
    pub selected: usize,
}

/// A workload shape for which strategy choice may differ.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct WorkloadScenario {
    pub realtime: bool,
    pub batch_size: u32,
    pub sequence_length: u32,
}

impl WorkloadScenario {
    pub fn validate(&self) -> Result<(), String> {
        if self.batch_size == 0 {
            return Err("workload batch size must be nonzero".into());
        }
        if self.sequence_length == 0 {
            return Err("workload sequence length must be nonzero".into());
        }
        if self.realtime && self.batch_size != 1 {
            return Err("realtime workload batch size must be one".into());
        }
        Ok(())
    }

    /// Return the number of logical elements in this workload, rejecting
    /// overflow instead of allowing a saturated size to influence compilation
    /// or strategy selection.
    pub fn scale_elements(self, per_sample_elements: u64) -> Result<u64, String> {
        self.validate()?;
        per_sample_elements
            .checked_mul(self.batch_size as u64)
            .and_then(|elements| elements.checked_mul(self.sequence_length as u64))
            .ok_or_else(|| "workload element count overflows u64".to_string())
    }
}

/// Strategy evidence and selection for one concrete workload scenario.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkloadStrategyEvaluation {
    pub scenario: WorkloadScenario,
    pub evaluation: FusionStrategyEvaluation,
}

/// Evaluate all execution layouts for one legal fusion permutation.
///
/// This is deliberately deterministic and backend-neutral. Runtime
/// calibration can replace the estimates later, while the compiler already
/// preserves all three alternatives for comparative evidence.
pub fn evaluate_fusion_strategies(
    permutation: &FusedPermutation,
    element_count: u64,
) -> FusionStrategyEvaluation {
    evaluate_fusion_strategies_with_generation(permutation, element_count, 0)
}

/// Evaluate strategies while preserving the evolutionary-search generation
/// that produced the persistent megakernel candidate.
pub fn evaluate_fusion_strategies_with_generation(
    permutation: &FusedPermutation,
    element_count: u64,
    search_generation: u32,
) -> FusionStrategyEvaluation {
    evaluate_fusion_strategies_with_generation_and_measurements(
        permutation,
        element_count,
        search_generation,
        &[],
    )
}

/// Evaluate all layouts, replacing model estimates with measured runtime
/// evidence wherever it is available.
pub fn evaluate_fusion_strategies_with_measurements(
    permutation: &FusedPermutation,
    element_count: u64,
    measurements: &[FusionMeasurement],
) -> FusionStrategyEvaluation {
    evaluate_fusion_strategies_with_generation_and_measurements(
        permutation,
        element_count,
        0,
        measurements,
    )
}

/// Evaluate a fusion permutation for one concrete workload shape.
///
/// `element_count` is the per-token, per-sample element count. Workload
/// evaluation scales it by batch and sequence length so the fallback model
/// remains meaningful when runtime measurements are unavailable.
pub fn evaluate_fusion_strategies_for_workload(
    permutation: &FusedPermutation,
    element_count: u64,
    scenario: WorkloadScenario,
) -> Result<FusionStrategyEvaluation, String> {
    scenario.validate()?;
    let workload_elements = scenario.scale_elements(element_count)?;
    Ok(evaluate_fusion_strategies(permutation, workload_elements))
}

pub fn evaluate_fusion_strategies_with_generation_and_measurements(
    permutation: &FusedPermutation,
    element_count: u64,
    search_generation: u32,
    measurements: &[FusionMeasurement],
) -> FusionStrategyEvaluation {
    let op_count = permutation.ops.len().max(1);
    let bytes = element_count.saturating_mul(4);
    let stage_count = if op_count <= 2 { 1 } else { 2 };
    let split = op_count.div_ceil(stage_count);
    let stages = permutation
        .ops
        .chunks(split)
        .map(|ops| ops.to_vec())
        .collect::<Vec<_>>();

    let estimates = [
        (FusionStrategy::StandardFused, 1usize, 0u64),
        (
            FusionStrategy::InterleavedFused { stages },
            stage_count,
            bytes,
        ),
        (
            FusionStrategy::PerOperation,
            op_count,
            bytes.saturating_mul(op_count.saturating_sub(1) as u64),
        ),
        (
            FusionStrategy::PersistentMegakernel { search_generation },
            1,
            0,
        ),
    ];
    let candidates = estimates
        .into_iter()
        .enumerate()
        .map(|(index, (strategy, kernel_count, materialized))| {
            let modeled_latency_ns = (kernel_count as u64)
                .saturating_mul(1_000)
                .saturating_add(element_count.saturating_mul(op_count as u64));
            let modeled_latency_ns =
                if matches!(strategy, FusionStrategy::PersistentMegakernel { .. }) {
                    modeled_latency_ns.saturating_sub(500)
                } else {
                    modeled_latency_ns
                };
            let measurement = measurements
                .iter()
                .find(|measurement| measurement.candidate_index == index);
            let estimated_latency_ns = measurement
                .map(|measurement| measurement.latency_ns)
                .unwrap_or(modeled_latency_ns);
            let estimated_materialized_bytes = measurement
                .map(|measurement| measurement.materialized_bytes)
                .unwrap_or(materialized);
            let score = estimated_latency_ns as f64 + estimated_materialized_bytes as f64 * 0.01;
            FusionStrategyCandidate {
                strategy,
                kernel_count,
                estimated_latency_ns,
                estimated_materialized_bytes,
                score,
                measured: measurement.is_some(),
            }
        })
        .collect::<Vec<_>>();
    let selected = candidates
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| left.score.total_cmp(&right.score))
        .map(|(index, _)| index)
        .unwrap_or(0);
    FusionStrategyEvaluation {
        candidates,
        selected,
    }
}

/// Benchmark all strategies independently for every workload scenario.
/// The callback is responsible for dispatch, synchronization, and timing on
/// the selected backend. A distinct winner is retained for each scenario.
pub fn benchmark_workload_scenarios<F>(
    permutation: &FusedPermutation,
    element_count: u64,
    scenarios: &[WorkloadScenario],
    mut run: F,
) -> Result<Vec<WorkloadStrategyEvaluation>, String>
where
    F: FnMut(WorkloadScenario, usize, &FusionStrategy) -> Result<(u64, u64), String>,
{
    let mut results = Vec::with_capacity(scenarios.len());
    for &scenario in scenarios {
        scenario.validate()?;
        let modeled =
            evaluate_fusion_strategies_for_workload(permutation, element_count, scenario)?;
        let mut measurements = Vec::with_capacity(modeled.candidates.len());
        for (candidate_index, candidate) in modeled.candidates.iter().enumerate() {
            let (latency_ns, materialized_bytes) =
                run(scenario, candidate_index, &candidate.strategy)?;
            if latency_ns == 0 {
                return Err(format!(
                    "fusion benchmark returned zero latency for {:?} candidate {}",
                    scenario, candidate_index
                ));
            }
            measurements.push(FusionMeasurement {
                candidate_index,
                latency_ns,
                materialized_bytes,
            });
        }
        results.push(WorkloadStrategyEvaluation {
            scenario,
            evaluation: evaluate_fusion_strategies_with_measurements(
                permutation,
                scenario.scale_elements(element_count)?,
                &measurements,
            ),
        });
    }
    Ok(results)
}

/// Benchmark every strategy through a backend/runtime supplied runner.
///
/// The runner receives the stable candidate index and the concrete strategy
/// layout. This keeps timing and device synchronization outside SpatialIR
/// while making measured selection a first-class compiler operation.
pub fn benchmark_fusion_strategies<F>(
    permutation: &FusedPermutation,
    element_count: u64,
    mut run: F,
) -> Result<FusionStrategyEvaluation, String>
where
    F: FnMut(usize, &FusionStrategy) -> Result<(u64, u64), String>,
{
    let modeled = evaluate_fusion_strategies(permutation, element_count);
    let mut measurements = Vec::with_capacity(modeled.candidates.len());
    for (candidate_index, candidate) in modeled.candidates.iter().enumerate() {
        let (latency_ns, materialized_bytes) = run(candidate_index, &candidate.strategy)?;
        if latency_ns == 0 {
            return Err(format!(
                "fusion benchmark returned zero latency for candidate {candidate_index}"
            ));
        }
        measurements.push(FusionMeasurement {
            candidate_index,
            latency_ns,
            materialized_bytes,
        });
    }
    Ok(evaluate_fusion_strategies_with_measurements(
        permutation,
        element_count,
        &measurements,
    ))
}

impl FusedPermutation {
    /// Compute the kernel digest for this permutation.
    ///
    /// The digest covers every element that affects the generated Metal
    /// kernel source: the op labels in order, and the threadgroup dimensions.
    fn compute_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for op in &self.ops {
            hasher.update(op.label().as_bytes());
            hasher.update(b"\0");
        }
        hasher.update(self.threadgroup_width.to_le_bytes());
        hasher.update(self.threadgroup_height.to_le_bytes());
        hasher.finalize().into()
    }

    /// Returns the total number of threads in the threadgroup (`width * height`).
    pub fn total_threads(&self) -> u32 {
        self.threadgroup_width * self.threadgroup_height
    }
}

// ---------------------------------------------------------------------------
// Default threadgroup geometry for fused kernels
// ---------------------------------------------------------------------------

/// Default threadgroup width for fused GEMV + element-wise kernels.
///
/// Derived from Metal's typical warp size (32) × 2, giving a balanced
/// 64-wide dispatch for GEMV and enough parallelism for element-wise tails.
pub const DEFAULT_FUSED_TG_WIDTH: u32 = 64;

/// Default threadgroup height for fused kernels.
///
/// Pinned to 1 for GEMV-oriented fused kernels because the reduction
/// dimension is spread across threadgroup width.
pub const DEFAULT_FUSED_TG_HEIGHT: u32 = 1;

// ---------------------------------------------------------------------------
// Metal threadgroup compatibility
// ---------------------------------------------------------------------------

/// Returns an error string if the threadgroup dimensions exceed Metal limits.
///
/// Metal constraints:
/// - `width ≤ 256`, `height ≤ 64`
/// - `width * height ≤ 1024`
fn validate_metal_threadgroup(width: u32, height: u32) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err(format!(
            "threadgroup dimensions must be positive, got ({width}, {height})"
        ));
    }
    if width > 256 {
        return Err(format!(
            "threadgroup width {width} exceeds Metal limit of 256"
        ));
    }
    if height > 64 {
        return Err(format!(
            "threadgroup height {height} exceeds Metal limit of 64"
        ));
    }
    let total = width * height;
    if total > 1024 {
        return Err(format!(
            "threadgroup total threads {total} exceeds Metal limit of 1024 (width={width}, height={height})"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// check_fusion_legality
// ---------------------------------------------------------------------------

/// Check if a sequence of ops can be legally fused into a single Metal kernel.
///
/// # Rules
///
/// 1. **Non-empty**: An empty sequence is illegal — there is nothing to fuse.
/// 2. **First op is a matmul**: The kernel must start with either
///    [`FusableOp::TernaryGemv`] or [`FusableOp::FpGemv`]. A fusion that begins
///    with an element-wise op is invalid because there is no base computation.
/// 3. **At most one matmul**: Two GEMVs cannot share a single kernel — each
///    matmul has its own reduction dimension and threadgroup layout.
/// 4. **Max 4 ops**: A practical limit on Metal kernel complexity. Beyond four
///    ops, register pressure and threadgroup memory contention dominate any
///    fusion benefit.
/// 5. **Metal threadgroup compatibility**: All ops must be compatible with
///    Metal threadgroup memory. The check is structural — all defined variants
///    are compatible when the sequence passes rules 1–4.
///
/// On success, returns a [`FusedPermutation`] with default threadgroup geometry
/// and a content-derived kernel digest.
///
/// # Examples
///
/// ```
/// use prism_spatial_ir::fused_ops::{check_fusion_legality, FusableOp};
///
/// // Legal: GEMV + SiLU
/// let result = check_fusion_legality(&[FusableOp::TernaryGemv, FusableOp::Silu]);
/// assert!(result.is_ok());
///
/// // Illegal: two matmuls
/// let result = check_fusion_legality(&[FusableOp::FpGemv, FusableOp::TernaryGemv]);
/// assert!(result.is_err());
/// ```
pub fn check_fusion_legality(ops: &[FusableOp]) -> Result<FusedPermutation, String> {
    // Rule 1: non-empty
    if ops.is_empty() {
        return Err("empty fusion sequence is illegal".to_string());
    }

    // Rule 2: first op must be a matmul
    if !ops[0].is_matmul() {
        return Err(format!(
            "first op in a fusion must be a matmul (TernaryGemv or FpGemv), got {:?}",
            ops[0]
        ));
    }

    // Rule 3: at most one matmul
    let matmul_count = ops.iter().filter(|op| op.is_matmul()).count();
    if matmul_count > 1 {
        return Err("cannot fuse two matmuls in a single kernel".to_string());
    }

    // Rule 4: max 4 ops
    if ops.len() > 4 {
        return Err(format!("max 4 ops per fused kernel, got {}", ops.len()));
    }

    // Rule 5: Metal threadgroup memory compatibility.
    //
    // Structural check: every op type defined in FusableOp is compatible
    // with Metal threadgroup memory when preceded by a matmul.  Element-wise
    // ops (SiLU, RMSNorm, RoPE, ElementWiseAdd) operate on the matmul's
    // output one element at a time, with no cross-lane communication beyond
    // what threadgroup memory supports.
    //
    // This rule is a forward-looking guard: if a future variant is added
    // that requires cross-threadgroup synchronization or a different memory
    // scope, the match below will fail to compile — forcing the author to
    // make an explicit decision.
    for &op in ops {
        match op {
            FusableOp::TernaryGemv | FusableOp::FpGemv => {
                // Matmul ops are the fusion entry — their threadgroup usage
                // is vetted by the Metal backend legalizer.
            }
            FusableOp::Silu | FusableOp::RmsNorm | FusableOp::Rope | FusableOp::ElementWiseAdd => {
                // Element-wise: compatible with threadgroup memory.  Each
                // thread reads one or a small contiguous block of elements
                // from shared memory, applies the operation, and writes
                // back.  No cross-threadgroup synchronization required.
            }
        }
    }

    // Validate default threadgroup geometry against Metal limits.
    let threadgroup_width = DEFAULT_FUSED_TG_WIDTH;
    let threadgroup_height = DEFAULT_FUSED_TG_HEIGHT;
    validate_metal_threadgroup(threadgroup_width, threadgroup_height)?;

    let mut perm = FusedPermutation {
        ops: ops.to_vec(),
        threadgroup_width,
        threadgroup_height,
        kernel_digest: [0u8; 32],
    };
    perm.kernel_digest = perm.compute_digest();

    Ok(perm)
}

// ---------------------------------------------------------------------------
// Fusion candidate enumeration helpers
// ---------------------------------------------------------------------------

/// Enumerate the canonical legal fusion permutations for a given matmul op.
///
/// Returns every legal fusion sequence of length 2 to 4 that starts with
/// `matmul_op` and uses only the designated element-wise ops. This is a
/// building block for [`SpatialGraph::available_fusions`] — it produces the
/// candidate set before filtering against graph topology.
///
/// # Panics
///
/// Panics if `matmul_op` is not a matmul variant.
pub fn enumerate_fusion_candidates(matmul_op: FusableOp) -> Vec<FusedPermutation> {
    assert!(
        matmul_op.is_matmul(),
        "expected a matmul op, got {matmul_op:?}"
    );

    let element_wise_ops = [
        FusableOp::Silu,
        FusableOp::RmsNorm,
        FusableOp::Rope,
        FusableOp::ElementWiseAdd,
    ];

    let mut candidates: Vec<FusedPermutation> = Vec::new();

    // Single-element sequences (just the matmul) — a degenerate fused kernel
    // that is always legal but never provides a fusion benefit.  We include
    // it so the search can make an explicit choice.
    if let Ok(perm) = check_fusion_legality(&[matmul_op]) {
        candidates.push(perm);
    }

    // Length-2 pairs: matmul + one element-wise
    for &ew in &element_wise_ops {
        if let Ok(perm) = check_fusion_legality(&[matmul_op, ew]) {
            candidates.push(perm);
        }
    }

    // Length-3 triples: matmul + two element-wise (with repetition allowed)
    for &ew1 in &element_wise_ops {
        for &ew2 in &element_wise_ops {
            if let Ok(perm) = check_fusion_legality(&[matmul_op, ew1, ew2]) {
                candidates.push(perm);
            }
        }
    }

    // Length-4 quadruples: matmul + three element-wise
    for &ew1 in &element_wise_ops {
        for &ew2 in &element_wise_ops {
            for &ew3 in &element_wise_ops {
                if let Ok(perm) = check_fusion_legality(&[matmul_op, ew1, ew2, ew3]) {
                    candidates.push(perm);
                }
            }
        }
    }

    candidates
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Legal sequences ---------------------------------------------------

    #[test]
    fn gemv_silu_legal() {
        let result = check_fusion_legality(&[FusableOp::TernaryGemv, FusableOp::Silu]);
        assert!(result.is_ok(), "TernaryGemv + SiLU should be legal");
        let perm = result.unwrap();
        assert_eq!(perm.ops.len(), 2);
        assert_eq!(perm.ops[0], FusableOp::TernaryGemv);
        assert_eq!(perm.ops[1], FusableOp::Silu);
        assert!(perm.total_threads() <= 1024);
    }

    #[test]
    fn gemv_rmsnorm_legal() {
        let result = check_fusion_legality(&[FusableOp::FpGemv, FusableOp::RmsNorm]);
        assert!(result.is_ok(), "FpGemv + RMSNorm should be legal");
        let perm = result.unwrap();
        assert_eq!(perm.ops.len(), 2);
    }

    #[test]
    fn gemv_silu_rmsnorm_legal() {
        let result =
            check_fusion_legality(&[FusableOp::TernaryGemv, FusableOp::Silu, FusableOp::RmsNorm]);
        assert!(
            result.is_ok(),
            "TernaryGemv + SiLU + RMSNorm should be legal"
        );
    }

    #[test]
    fn gemv_rope_add_legal() {
        let result = check_fusion_legality(&[
            FusableOp::FpGemv,
            FusableOp::Rope,
            FusableOp::ElementWiseAdd,
        ]);
        assert!(result.is_ok(), "FpGemv + RoPE + Add should be legal");
    }

    #[test]
    fn gemv_silu_rmsnorm_rope_legal() {
        let result = check_fusion_legality(&[
            FusableOp::TernaryGemv,
            FusableOp::Silu,
            FusableOp::RmsNorm,
            FusableOp::Rope,
        ]);
        assert!(result.is_ok(), "4-op fusion should be legal");
    }

    #[test]
    fn fusion_digest_deterministic() {
        let a = check_fusion_legality(&[FusableOp::FpGemv, FusableOp::Silu]).unwrap();
        let b = check_fusion_legality(&[FusableOp::FpGemv, FusableOp::Silu]).unwrap();
        assert_eq!(
            a.kernel_digest, b.kernel_digest,
            "digest must be deterministic"
        );
    }

    #[test]
    fn fusion_digest_differs_for_different_ops() {
        let a = check_fusion_legality(&[FusableOp::FpGemv, FusableOp::Silu]).unwrap();
        let b = check_fusion_legality(&[FusableOp::FpGemv, FusableOp::RmsNorm]).unwrap();
        assert_ne!(
            a.kernel_digest, b.kernel_digest,
            "different ops must produce different digests"
        );
    }

    #[test]
    fn single_matmul_degenerate_legal() {
        // A single matmul is a degenerate fused kernel — always legal.
        let result = check_fusion_legality(&[FusableOp::TernaryGemv]);
        assert!(result.is_ok());
        let result = check_fusion_legality(&[FusableOp::FpGemv]);
        assert!(result.is_ok());
    }

    // -- Illegal sequences -------------------------------------------------

    #[test]
    fn empty_sequence_illegal() {
        let result = check_fusion_legality(&[]);
        assert!(result.is_err(), "empty sequence must be illegal");
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn silu_gemv_illegal() {
        let result = check_fusion_legality(&[FusableOp::Silu, FusableOp::TernaryGemv]);
        assert!(
            result.is_err(),
            "SiLU + GEMV must be illegal (first op not matmul)"
        );
    }

    #[test]
    fn gemv_gemv_illegal() {
        let result = check_fusion_legality(&[FusableOp::FpGemv, FusableOp::TernaryGemv]);
        assert!(result.is_err(), "GEMV + GEMV must be illegal (two matmuls)");
    }

    #[test]
    fn two_matmuls_illegal() {
        let result =
            check_fusion_legality(&[FusableOp::TernaryGemv, FusableOp::Silu, FusableOp::FpGemv]);
        assert!(result.is_err(), "two matmuls in a fusion must be illegal");
    }

    #[test]
    fn too_many_ops_illegal() {
        let result = check_fusion_legality(&[
            FusableOp::FpGemv,
            FusableOp::Silu,
            FusableOp::RmsNorm,
            FusableOp::Rope,
            FusableOp::ElementWiseAdd,
        ]);
        assert!(result.is_err(), "5 ops must exceed max fusion length");
    }

    // -- FusableOp helpers -------------------------------------------------

    #[test]
    fn is_matmul_ternary() {
        assert!(FusableOp::TernaryGemv.is_matmul());
    }

    #[test]
    fn is_matmul_fp() {
        assert!(FusableOp::FpGemv.is_matmul());
    }

    #[test]
    fn is_matmul_false_for_element_wise() {
        assert!(!FusableOp::Silu.is_matmul());
        assert!(!FusableOp::RmsNorm.is_matmul());
        assert!(!FusableOp::Rope.is_matmul());
        assert!(!FusableOp::ElementWiseAdd.is_matmul());
    }

    #[test]
    fn is_element_wise_true() {
        assert!(FusableOp::Silu.is_element_wise());
        assert!(FusableOp::RmsNorm.is_element_wise());
        assert!(FusableOp::Rope.is_element_wise());
        assert!(FusableOp::ElementWiseAdd.is_element_wise());
    }

    #[test]
    fn is_element_wise_false_for_matmul() {
        assert!(!FusableOp::TernaryGemv.is_element_wise());
        assert!(!FusableOp::FpGemv.is_element_wise());
    }

    #[test]
    fn label_matches_variant() {
        assert_eq!(FusableOp::TernaryGemv.label(), "ternary_gemv");
        assert_eq!(FusableOp::FpGemv.label(), "fp_gemv");
        assert_eq!(FusableOp::Silu.label(), "silu");
        assert_eq!(FusableOp::RmsNorm.label(), "rms_norm");
        assert_eq!(FusableOp::Rope.label(), "rope");
        assert_eq!(FusableOp::ElementWiseAdd.label(), "elem_add");
    }

    #[test]
    fn strategy_ids_are_stable_across_parameters() {
        assert_eq!(FusionStrategy::StandardFused.stable_id(), "standard_fused");
        assert_eq!(
            FusionStrategy::InterleavedFused { stages: vec![] }.stable_id(),
            "interleaved_fused"
        );
        assert_eq!(FusionStrategy::PerOperation.stable_id(), "per_operation");
        assert_eq!(
            FusionStrategy::PersistentMegakernel {
                search_generation: 9
            }
            .stable_id(),
            "persistent_megakernel"
        );
    }

    // -- Threadgroup validation -------------------------------------------

    #[test]
    fn validate_metal_threadgroup_valid() {
        assert!(validate_metal_threadgroup(64, 1).is_ok());
        assert!(validate_metal_threadgroup(256, 4).is_ok());
        assert!(validate_metal_threadgroup(32, 32).is_ok());
    }

    #[test]
    fn validate_metal_threadgroup_zero() {
        assert!(validate_metal_threadgroup(0, 1).is_err());
        assert!(validate_metal_threadgroup(1, 0).is_err());
        assert!(validate_metal_threadgroup(0, 0).is_err());
    }

    #[test]
    fn validate_metal_threadgroup_width_too_large() {
        let err = validate_metal_threadgroup(300, 1).unwrap_err();
        assert!(err.contains("width"));
        assert!(err.contains("300"));
    }

    #[test]
    fn validate_metal_threadgroup_height_too_large() {
        let err = validate_metal_threadgroup(64, 128).unwrap_err();
        assert!(err.contains("height"));
        assert!(err.contains("128"));
    }

    #[test]
    fn validate_metal_threadgroup_total_overflow() {
        let err = validate_metal_threadgroup(128, 32).unwrap_err();
        assert!(err.contains("total threads"));
        assert!(err.contains("4096"));
    }

    // -- Enumerate fusion candidates --------------------------------------

    #[test]
    fn enumerate_ternary_candidates() {
        let candidates = enumerate_fusion_candidates(FusableOp::TernaryGemv);
        // 1 degenerate + 4 length-2 + 16 length-3 + 64 length-4 = 85 total
        assert_eq!(
            candidates.len(),
            85,
            "expected 85 candidates for TernaryGemv"
        );
        // All must start with TernaryGemv
        for c in &candidates {
            assert_eq!(c.ops[0], FusableOp::TernaryGemv);
        }
    }

    #[test]
    fn enumerate_fp_candidates() {
        let candidates = enumerate_fusion_candidates(FusableOp::FpGemv);
        assert_eq!(candidates.len(), 85, "expected 85 candidates for FpGemv");
    }

    #[test]
    fn enumerate_candidates_unique_digests() {
        let candidates = enumerate_fusion_candidates(FusableOp::TernaryGemv);
        let mut digests = std::collections::HashSet::new();
        for c in &candidates {
            assert!(
                digests.insert(c.kernel_digest),
                "duplicate digest found: {:?} in {:?}",
                c.kernel_digest,
                c.ops.iter().map(|o| o.label()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn enumerate_candidates_all_legal() {
        let candidates = enumerate_fusion_candidates(FusableOp::TernaryGemv);
        for c in &candidates {
            let recheck = check_fusion_legality(&c.ops);
            assert!(
                recheck.is_ok(),
                "candidate {:?} failed re-legalization: {}",
                c.ops.iter().map(|o| o.label()).collect::<Vec<_>>(),
                recheck.unwrap_err()
            );
        }
    }

    // -- Default geometry constants ---------------------------------------

    const _: () = {
        assert!(DEFAULT_FUSED_TG_WIDTH <= 256);
        assert!(DEFAULT_FUSED_TG_HEIGHT <= 64);
        assert!(DEFAULT_FUSED_TG_WIDTH * DEFAULT_FUSED_TG_HEIGHT <= 1024);
    };

    #[test]
    fn default_geometry_passes_validation() {
        assert!(
            validate_metal_threadgroup(DEFAULT_FUSED_TG_WIDTH, DEFAULT_FUSED_TG_HEIGHT).is_ok()
        );
    }

    // -- Serialization round-trip ------------------------------------------

    #[test]
    fn fused_permutation_json_roundtrip() {
        let perm = check_fusion_legality(&[FusableOp::TernaryGemv, FusableOp::Silu]).unwrap();
        let json = serde_json::to_string(&perm).unwrap();
        let restored: FusedPermutation = serde_json::from_str(&json).unwrap();
        assert_eq!(perm, restored);
    }

    #[test]
    fn fusable_op_json_roundtrip() {
        for op in &[
            FusableOp::TernaryGemv,
            FusableOp::FpGemv,
            FusableOp::Silu,
            FusableOp::RmsNorm,
            FusableOp::Rope,
            FusableOp::ElementWiseAdd,
        ] {
            let json = serde_json::to_string(op).unwrap();
            let restored: FusableOp = serde_json::from_str(&json).unwrap();
            assert_eq!(*op, restored);
        }
    }

    #[test]
    fn fusion_strategy_evaluation_compares_all_layouts() {
        let permutation =
            check_fusion_legality(&[FusableOp::FpGemv, FusableOp::Silu, FusableOp::RmsNorm])
                .unwrap();
        let evaluation = evaluate_fusion_strategies(&permutation, 4096);
        assert_eq!(evaluation.candidates.len(), 4);
        assert!(evaluation
            .candidates
            .iter()
            .any(|candidate| matches!(candidate.strategy, FusionStrategy::StandardFused)));
        assert!(evaluation.candidates.iter().any(|candidate| matches!(
            candidate.strategy,
            FusionStrategy::InterleavedFused { .. }
        )));
        assert!(evaluation
            .candidates
            .iter()
            .any(|candidate| matches!(candidate.strategy, FusionStrategy::PerOperation)));
        assert!(evaluation.candidates.iter().any(|candidate| matches!(
            candidate.strategy,
            FusionStrategy::PersistentMegakernel { .. }
        )));
        assert!(evaluation.selected < evaluation.candidates.len());
    }

    #[test]
    fn measured_fusion_evidence_can_change_selected_strategy() {
        let permutation =
            check_fusion_legality(&[FusableOp::FpGemv, FusableOp::Silu, FusableOp::RmsNorm])
                .unwrap();
        let evaluation = evaluate_fusion_strategies_with_measurements(
            &permutation,
            4096,
            &[
                FusionMeasurement {
                    candidate_index: 0,
                    latency_ns: 9_000,
                    materialized_bytes: 0,
                },
                FusionMeasurement {
                    candidate_index: 1,
                    latency_ns: 1_500,
                    materialized_bytes: 16_384,
                },
                FusionMeasurement {
                    candidate_index: 2,
                    latency_ns: 8_000,
                    materialized_bytes: 32_768,
                },
                FusionMeasurement {
                    candidate_index: 3,
                    latency_ns: 7_000,
                    materialized_bytes: 0,
                },
            ],
        );
        assert_eq!(evaluation.selected, 1);
        assert!(evaluation
            .candidates
            .iter()
            .all(|candidate| candidate.measured));
    }

    #[test]
    fn workload_benchmark_selects_different_strategies_per_scenario() {
        let permutation = check_fusion_legality(&[FusableOp::FpGemv, FusableOp::Silu]).unwrap();
        let scenarios = [
            WorkloadScenario {
                realtime: true,
                batch_size: 1,
                sequence_length: 1,
            },
            WorkloadScenario {
                realtime: false,
                batch_size: 32,
                sequence_length: 128,
            },
        ];
        let results = benchmark_workload_scenarios(
            &permutation,
            4096,
            &scenarios,
            |scenario, candidate_index, _| {
                let latency = if scenario.realtime {
                    if candidate_index == 3 {
                        100
                    } else {
                        10_000
                    }
                } else if candidate_index == 0 {
                    100
                } else {
                    10_000
                };
                Ok((latency, 0))
            },
        )
        .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].evaluation.selected, 3);
        assert_eq!(results[1].evaluation.selected, 0);
    }

    #[test]
    fn workload_model_scales_element_count_by_batch_and_sequence() {
        let permutation = check_fusion_legality(&[FusableOp::FpGemv, FusableOp::Silu]).unwrap();
        let realtime = evaluate_fusion_strategies_for_workload(
            &permutation,
            128,
            WorkloadScenario {
                realtime: true,
                batch_size: 1,
                sequence_length: 1,
            },
        )
        .unwrap();
        let batch = evaluate_fusion_strategies_for_workload(
            &permutation,
            128,
            WorkloadScenario {
                realtime: false,
                batch_size: 8,
                sequence_length: 16,
            },
        )
        .unwrap();
        assert!(
            batch.candidates[0].estimated_latency_ns > realtime.candidates[0].estimated_latency_ns
        );
    }

    #[test]
    fn workload_element_scaling_rejects_overflow() {
        let scenario = WorkloadScenario {
            realtime: false,
            batch_size: u32::MAX,
            sequence_length: u32::MAX,
        };
        assert!(scenario.scale_elements(u64::MAX).is_err());
    }

    #[test]
    fn benchmark_runner_executes_every_strategy_and_uses_measurements() {
        let permutation = check_fusion_legality(&[FusableOp::FpGemv, FusableOp::Silu]).unwrap();
        let mut seen = Vec::new();
        let evaluation = benchmark_fusion_strategies(&permutation, 1024, |index, strategy| {
            seen.push((index, strategy.clone()));
            let latency = if index == 2 { 25 } else { 5_000 };
            Ok((latency, index as u64))
        })
        .unwrap();
        assert_eq!(seen.len(), 4);
        assert_eq!(
            seen.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(evaluation.selected, 2);
        assert!(evaluation
            .candidates
            .iter()
            .all(|candidate| candidate.measured));
    }

    #[test]
    fn benchmark_runner_propagates_backend_errors() {
        let permutation = check_fusion_legality(&[FusableOp::FpGemv]).unwrap();
        let error = benchmark_fusion_strategies(&permutation, 1, |index, _| {
            if index == 1 {
                Err("backend timing failed".into())
            } else {
                Ok((1, 0))
            }
        })
        .unwrap_err();
        assert_eq!(error, "backend timing failed");
    }

    #[test]
    fn benchmark_runner_rejects_zero_latency_samples() {
        let permutation = check_fusion_legality(&[FusableOp::FpGemv]).unwrap();
        let error = benchmark_fusion_strategies(&permutation, 1, |_, _| Ok((0, 0))).unwrap_err();
        assert!(error.contains("zero latency"));
    }

    #[test]
    fn workload_scenario_validation_rejects_impossible_shapes() {
        assert!(WorkloadScenario {
            realtime: false,
            batch_size: 0,
            sequence_length: 1,
        }
        .validate()
        .is_err());
        assert!(WorkloadScenario {
            realtime: true,
            batch_size: 2,
            sequence_length: 1,
        }
        .validate()
        .is_err());
        assert!(WorkloadScenario {
            realtime: false,
            batch_size: 8,
            sequence_length: 128,
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn persistent_candidate_preserves_search_generation() {
        let permutation = check_fusion_legality(&[FusableOp::FpGemv]).unwrap();
        let evaluation = evaluate_fusion_strategies_with_generation(&permutation, 1024, 37);
        assert!(evaluation.candidates.iter().any(|candidate| matches!(
            candidate.strategy,
            FusionStrategy::PersistentMegakernel {
                search_generation: 37
            }
        )));
    }
}
