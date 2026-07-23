//! Cost model types for spatial graph estimation.
//!
//! The cost model assigns a multi-dimensional cost to every legalized graph,
//! including latency, memory, energy, and synchronization overhead. These
//! estimates drive the evolutionary search towards better schedules.

use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::calibration_report::M1CalibrationReport;
use crate::graph::{ComputeKind, MemoryRegion, ShapeContract, SpatialGraph, SpatialNode};
use crate::tiling::{validate_joint_tiling_geometry, TilingValidationError};

// ---------------------------------------------------------------------------
// CostEstimate
// ---------------------------------------------------------------------------

/// Multi-dimensional cost estimate for a legalized spatial graph.
///
/// Estimates are produced by the [`CostModel`] trait and consumed by the
/// evolutionary search framework for fitness assessment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CostEstimate {
    /// Estimated total latency (p50).
    pub latency: Duration,
    /// Estimated peak memory usage in bytes.
    pub peak_memory: u64,
    /// Total bytes materialized across domain boundaries.
    pub materialized_bytes: u64,
    /// Number of synchronization boundaries in the schedule.
    pub sync_count: u32,
    /// Estimated energy consumption in joules.
    pub energy: f64,
    /// Error / confidence bound (0.0 = certain, 1.0 = unreliable).
    pub error: f64,
}

impl CostEstimate {
    /// Create a new cost estimate.
    pub fn new(
        latency: Duration,
        peak_memory: u64,
        materialized_bytes: u64,
        sync_count: u32,
        energy: f64,
        error: f64,
    ) -> Self {
        Self {
            latency,
            peak_memory,
            materialized_bytes,
            sync_count,
            energy,
            error,
        }
    }

    /// Returns a zero-cost estimate for unestimated graphs (error is set to 1.0 uncertainty).
    pub fn zero() -> Self {
        Self {
            latency: Duration::ZERO,
            peak_memory: 0,
            materialized_bytes: 0,
            sync_count: 0,
            energy: 0.0,
            error: 1.0, // maximum uncertainty
        }
    }

    /// Computes a scalar fitness score from the multi-dimensional estimate.
    ///
    /// Lower is better. The formula weights latency, memory, and energy
    /// with a penalty for uncertainty.
    pub fn fitness_score(&self) -> f64 {
        let latency_s = self.latency.as_secs_f64();
        let mem_gb = self.peak_memory as f64 / (1024.0 * 1024.0 * 1024.0);
        let energy_j = self.energy;
        let sync_penalty = self.sync_count as f64 * 0.01;
        let error_penalty = self.error * 10.0;

        latency_s + mem_gb * 0.1 + energy_j * 0.001 + sync_penalty + error_penalty
    }
}

/// Cost of one shared tile geometry evaluated independently on ANE and Metal.
///
/// `joint` is the conservative execution envelope: it uses the slower
/// backend's latency and the larger backend resource requirements. The
/// [`fitness_score`](Self::fitness_score) method adds a balance penalty so a
/// candidate that is excellent on one backend but poor on the other does not
/// win the joint search accidentally.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JointTilingCostEstimate {
    pub geometry: crate::graph::TileGeometry,
    pub ane: CostEstimate,
    pub metal: CostEstimate,
    pub joint: CostEstimate,
    pub ane_tile_count: u64,
    pub metal_tile_count: u64,
    pub balance_penalty: f64,
}

impl JointTilingCostEstimate {
    /// Scalar score for comparing shared ANE/Metal candidates. Lower is
    /// better, matching [`CostEstimate::fitness_score`].
    pub fn fitness_score(&self) -> f64 {
        self.joint.fitness_score() + self.balance_penalty
    }
}

/// Error returned when no candidate can be selected for joint execution.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TilingSelectionError {
    #[error("no tiling configurations were provided")]
    NoCandidates,
    #[error("no tiling configuration is legal for both ANE and Metal: {rejected:?}")]
    NoLegalCandidates {
        rejected: Vec<TilingValidationError>,
    },
}

// ---------------------------------------------------------------------------
// CodecVariant
// ---------------------------------------------------------------------------

/// Quantization codec variant for weight representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodecVariant {
    /// 4-bit NormalFloat (QLoRA-style).
    Nf4,
    /// 8-bit integer.
    Int8,
    /// 16-bit float (IEEE half-precision).
    Fp16,
    /// 32-bit float (full precision).
    Fp32,
    /// 4-bit signed integer (symmetric).
    SymInt4,
    /// Ternary encoding: {-1, 0, +1} in 2 bits.
    Ternary,
    /// Ternary 1.58-bit encoding.
    Ternary1_58,
    /// 8-bit quantization (Q8_0 format).
    Q8_0,
    /// 16-bit brain float (bfloat16).
    Bf16,
    /// 2-bit quantization (Q2_0 format).
    Q2_0,
    /// 4-bit quantization (Q4_0 format).
    Q4_0,
}

impl CodecVariant {
    /// Returns the bits-per-element for this codec.
    pub fn bits_per_element(&self) -> usize {
        match self {
            Self::Nf4 | Self::SymInt4 | Self::Q4_0 => 4,
            Self::Int8 | Self::Q8_0 => 8,
            Self::Fp16 | Self::Bf16 => 16,
            Self::Fp32 => 32,
            Self::Ternary | Self::Ternary1_58 | Self::Q2_0 => 2,
        }
    }

    /// Returns `true` if this codec variant is supported by the default backend.
    ///
    /// Supported codecs: Fp32, Fp16, Bf16, Int8, SymInt4, Nf4, Q2_0, Q4_0.
    /// Unsupported codecs include Ternary, Ternary1_58, and Q8_0.
    pub fn is_supported_codec(&self) -> bool {
        matches!(
            self,
            Self::Fp32
                | Self::Fp16
                | Self::Bf16
                | Self::Int8
                | Self::SymInt4
                | Self::Nf4
                | Self::Q2_0
                | Self::Q4_0
        )
    }
}

// ---------------------------------------------------------------------------
// CostModel trait
// ---------------------------------------------------------------------------

/// Trait for cost models that estimate the execution cost of a spatial graph.
///
/// The evolutionary search calls `estimate` thousands of times per run, so
/// implementations must be fast — typically table lookups with interpolation
/// rather than simulation.
pub trait CostModel {
    /// Estimate the cost of a given spatial graph.
    fn estimate(&self, graph: &SpatialGraph) -> CostEstimate;

    /// Evaluate one shared tile geometry on both ANE and Metal.
    ///
    /// This default implementation deliberately sits on the common cost
    /// model interface so evolutionary callers can evaluate candidates
    /// without knowing which concrete calibration model is active.
    fn estimate_joint_tiling(
        &self,
        graph: &SpatialGraph,
        geometry: crate::graph::TileGeometry,
    ) -> Result<JointTilingCostEstimate, TilingValidationError> {
        estimate_joint_tiling(self, graph, geometry)
    }

    /// Returns a human-readable name for this cost model.
    fn name(&self) -> &str;
}

/// Evaluate one shared geometry using a model's graph-level baseline cost.
pub fn estimate_joint_tiling<C: CostModel + ?Sized>(
    model: &C,
    graph: &SpatialGraph,
    geometry: crate::graph::TileGeometry,
) -> Result<JointTilingCostEstimate, TilingValidationError> {
    let area = validate_joint_tiling_geometry(geometry)? as u64;
    let baseline = model.estimate(graph);
    let work_units = graph_tiling_work_units(graph);
    let ane_tile_count = work_units.div_ceil(area.max(1));
    let metal_tile_count = ane_tile_count;

    // ANE and Metal have different useful tile-area sweet spots. These are
    // intentionally heuristic defaults, not claims about a particular SoC;
    // calibrated graph latency remains the source of absolute cost. The
    // shared candidate is scored against both shapes and the slower side.
    let ane = scale_for_tiling(&baseline, geometry, ane_tile_count, 1024, 2_000);
    let metal = scale_for_tiling(&baseline, geometry, metal_tile_count, 256, 1_000);
    let balance_penalty = relative_latency_delta(&ane.latency, &metal.latency);
    let joint = CostEstimate {
        latency: ane.latency.max(metal.latency),
        peak_memory: ane.peak_memory.max(metal.peak_memory),
        materialized_bytes: ane.materialized_bytes.max(metal.materialized_bytes),
        sync_count: ane.sync_count.max(metal.sync_count),
        energy: (ane.energy + metal.energy) * 0.5,
        error: (ane.error.max(metal.error) + balance_penalty * 0.1).min(1.0),
    };

    Ok(JointTilingCostEstimate {
        geometry,
        ane,
        metal,
        joint,
        ane_tile_count,
        metal_tile_count,
        balance_penalty,
    })
}

/// Evaluate every candidate independently. Invalid candidates remain as
/// explicit errors so callers can distinguish rejection from poor cost.
pub fn evaluate_joint_tiling_configurations<C: CostModel + ?Sized>(
    model: &C,
    graph: &SpatialGraph,
    candidates: &[crate::graph::TileGeometry],
) -> Vec<Result<JointTilingCostEstimate, TilingValidationError>> {
    candidates
        .iter()
        .copied()
        .map(|geometry| model.estimate_joint_tiling(graph, geometry))
        .collect()
}

/// Select the lowest-scoring legal shared geometry from a candidate set.
pub fn select_best_joint_tiling<C: CostModel + ?Sized>(
    model: &C,
    graph: &SpatialGraph,
    candidates: &[crate::graph::TileGeometry],
) -> Result<JointTilingCostEstimate, TilingSelectionError> {
    if candidates.is_empty() {
        return Err(TilingSelectionError::NoCandidates);
    }

    let mut best: Option<JointTilingCostEstimate> = None;
    let mut rejected = Vec::new();
    for result in evaluate_joint_tiling_configurations(model, graph, candidates) {
        match result {
            Ok(candidate) => {
                if best
                    .as_ref()
                    .map(|current| candidate.fitness_score() < current.fitness_score())
                    .unwrap_or(true)
                {
                    best = Some(candidate);
                }
            }
            Err(error) => rejected.push(error),
        }
    }

    best.ok_or(TilingSelectionError::NoLegalCandidates { rejected })
}

fn graph_tiling_work_units(graph: &SpatialGraph) -> u64 {
    let mut work = 0u64;
    for node in graph.nodes() {
        if let SpatialNode::Compute { shape, .. } = node {
            for tensor in shape.in_shapes.iter().chain(shape.out_shapes.iter()) {
                let elements = tensor.dims.iter().fold(1u64, |product, dimension| {
                    product.saturating_mul(*dimension as u64)
                });
                work = work.saturating_add(elements);
            }
        }
    }
    work.max(graph.node_count().max(1) as u64)
}

fn scale_for_tiling(
    baseline: &CostEstimate,
    geometry: crate::graph::TileGeometry,
    tile_count: u64,
    target_area: u64,
    per_tile_overhead_ns: u64,
) -> CostEstimate {
    let area = (geometry.width as u64).saturating_mul(geometry.height as u64);
    let utilization = (area as f64 / target_area as f64).clamp(0.125, 1.0);
    let baseline_ns = baseline.latency.as_nanos().min(u64::MAX as u128) as f64;
    let latency_ns = (baseline_ns / utilization + tile_count as f64 * per_tile_overhead_ns as f64)
        .min(u64::MAX as f64) as u64;
    let tile_materialization = tile_count.saturating_sub(1).saturating_mul(area);
    let extra_syncs = tile_count.saturating_sub(1);
    CostEstimate {
        latency: std::time::Duration::from_nanos(latency_ns),
        peak_memory: baseline.peak_memory.saturating_add(area.saturating_mul(2)),
        materialized_bytes: baseline
            .materialized_bytes
            .saturating_add(tile_materialization),
        sync_count: baseline
            .sync_count
            .saturating_add(extra_syncs.min(u32::MAX as u64) as u32),
        energy: baseline.energy * (1.0 / utilization),
        error: (baseline.error + (1.0 - utilization) * 0.05).min(1.0),
    }
}

fn relative_latency_delta(a: &std::time::Duration, b: &std::time::Duration) -> f64 {
    let a = a.as_secs_f64();
    let b = b.as_secs_f64();
    let denominator = a.max(b).max(f64::EPSILON);
    (a - b).abs() / denominator
}

/// Rough per-operation latency in microseconds for a stock M1 NPU/GPU.
/// These weights are relative to one another; the calibration report's
/// absolute `latency_p50_us` provides the scaling anchor.
fn base_op_latency_us(kind: &ComputeKind) -> f64 {
    match kind {
        ComputeKind::MatMul => 100.0,
        ComputeKind::Convolution => 200.0,
        ComputeKind::Elementwise => 10.0,
        ComputeKind::Normalization => 15.0,
        ComputeKind::Softmax => 20.0,
        ComputeKind::Attention => 500.0,
        ComputeKind::RoPE => 30.0,
        ComputeKind::SSM => 150.0,
        ComputeKind::Reshape => 5.0,
        ComputeKind::Gather => 8.0,
        ComputeKind::Custom(_) => 50.0,
    }
}

/// Total number of elements across all input and output shapes in a
/// [`ShapeContract`].
fn shape_contract_elements(contract: &ShapeContract) -> usize {
    let in_elems: usize = contract
        .in_shapes
        .iter()
        .map(|s| s.dims.iter().product::<usize>())
        .sum();
    let out_elems: usize = contract
        .out_shapes
        .iter()
        .map(|s| s.dims.iter().product::<usize>())
        .sum();
    in_elems + out_elems
}

/// Total number of elements in a [`MemoryRegion`].
fn memory_region_elements(region: &MemoryRegion) -> usize {
    region.shape.dims.iter().product()
}

// ---------------------------------------------------------------------------
// CalibratedCostModel
// ---------------------------------------------------------------------------

/// A cost model calibrated against a measured [`M1CalibrationReport`].
///
/// Produces multi-dimensional cost estimates that reflect real hardware
/// characteristics: per-operation latency from calibration, domain-transition
/// costs from the report, and memory pressure computed from tensor shapes and
/// a default 2-byte (fp16) element size.
///
/// # Latency
///
/// `latency = Σ(op_weight) × (cal_p50_us / reference_total_us) + edge_count × materialization_cost_us`
///
/// where `reference_total_us` is the sum of base-op weights for the set of
/// operations measured during calibration (defaults to a typical LLM decoder
/// layer: ~1000 µs).
pub struct CalibratedCostModel<'a> {
    /// Calibration report providing absolute latency anchors and transition costs.
    report: &'a M1CalibrationReport,
}

impl<'a> CalibratedCostModel<'a> {
    /// Create a new calibrated cost model backed by the given report.
    pub fn new(report: &'a M1CalibrationReport) -> Self {
        Self { report }
    }
}

impl CostModel for CalibratedCostModel<'_> {
    fn estimate(&self, graph: &SpatialGraph) -> CostEstimate {
        let _node_count = graph.node_count() as f64;
        let edge_count = graph.edge_count() as f64;
        let report = self.report;

        // ── Compute per-op weighted latency ────────────────────────────
        let reference_total_us = 1000.0;
        let cal_anchor_us = report.latency_p50_us / reference_total_us;

        let mut weighted_op_sum: f64 = 0.0;
        let mut total_elements: usize = 0;

        for node in graph.nodes() {
            match node {
                SpatialNode::Compute { kind, shape, .. } => {
                    weighted_op_sum += base_op_latency_us(kind);
                    total_elements += shape_contract_elements(shape);
                }
                SpatialNode::Memory { region, .. } => {
                    let elems = memory_region_elements(region);
                    total_elements += elems;
                    weighted_op_sum += 2.0;
                }
                SpatialNode::Stream { total_bytes, .. } => {
                    let bytes_gb = *total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                    weighted_op_sum += bytes_gb * 20.0;
                    total_elements += (*total_bytes as usize) / 2;
                }
                SpatialNode::Barrier { .. } => {
                    weighted_op_sum += 1.0;
                }
                SpatialNode::RepeatedDecoder { body, count, .. } => {
                    weighted_op_sum += body.len() as f64 * *count as f64 * 2.0;
                }
            }
        }

        let compute_latency_us = weighted_op_sum * cal_anchor_us;
        let transition_latency_us = edge_count * report.materialization_cost_us;
        let total_latency_us = compute_latency_us + transition_latency_us;
        let latency = Duration::from_secs_f64(total_latency_us / 1_000_000.0);

        let codec_bytes_per_elem = 2.0;
        let peak_memory = (total_elements as f64 * codec_bytes_per_elem) as u64;

        let materialized_bytes = (edge_count as u64) * 1024 * 1024;

        let barrier_count = graph
            .nodes()
            .iter()
            .filter(|n| matches!(n, SpatialNode::Barrier { .. }))
            .count() as u32;
        let sync_count = edge_count as u32 + barrier_count;

        let power_active_w = 15.0;
        let power_idle_w = 3.0;
        let compute_fraction = if total_latency_us > 0.0 {
            compute_latency_us / total_latency_us
        } else {
            0.0
        };
        let avg_power_w =
            compute_fraction * power_active_w + (1.0 - compute_fraction) * power_idle_w;
        let energy = latency.as_secs_f64() * avg_power_w;

        let error = (1.0 - report.confidence) + report.contention * 0.5;
        let error = error.clamp(0.0, 1.0);

        CostEstimate {
            latency,
            peak_memory,
            materialized_bytes,
            sync_count,
            energy,
            error,
        }
    }

    fn name(&self) -> &str {
        "calibrated_m1"
    }
}

// ---------------------------------------------------------------------------
// SimpleCostModel
// ---------------------------------------------------------------------------

/// MI300X-oriented analytical model used while exploring candidate schedules.
///
/// This keeps the evolutionary loop in Prism, but makes its fitness function
/// reflect CDNA3 execution: matrix-heavy work benefits from the matrix cores,
/// while streams and materialization are bounded by HBM bandwidth. A measured
/// calibration multiplier can be applied after a ROCm benchmark pass.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Mi300xCostModel {
    pub matrix_tflops: f64,
    pub hbm_bandwidth_gbytes: f64,
    pub latency_multiplier: f64,
}

impl Default for Mi300xCostModel {
    fn default() -> Self {
        Self {
            matrix_tflops: 1_300.0,
            hbm_bandwidth_gbytes: 5_300.0,
            latency_multiplier: 1.0,
        }
    }
}

impl Mi300xCostModel {
    pub fn with_latency_multiplier(mut self, multiplier: f64) -> Self {
        self.latency_multiplier = multiplier.max(0.01);
        self
    }
}

impl CostModel for Mi300xCostModel {
    fn estimate(&self, graph: &SpatialGraph) -> CostEstimate {
        let mut flops = 0.0_f64;
        let mut bytes = 0_u64;
        let mut barriers = 0_u32;
        for node in graph.nodes() {
            match node {
                SpatialNode::Compute { kind, shape, .. } => {
                    let elements = shape_contract_elements(shape) as f64;
                    bytes = bytes.saturating_add((elements * 2.0) as u64);
                    flops += match kind {
                        ComputeKind::MatMul | ComputeKind::Attention => elements * 2.0,
                        _ => elements,
                    };
                }
                SpatialNode::Memory { region, .. } => {
                    bytes = bytes.saturating_add((memory_region_elements(region) * 2) as u64);
                }
                SpatialNode::Stream { total_bytes, .. } => {
                    bytes = bytes.saturating_add(*total_bytes)
                }
                SpatialNode::Barrier { .. } => barriers += 1,
                SpatialNode::RepeatedDecoder { body, count, .. } => {
                    bytes = bytes.saturating_add((body.len() as u64) * (*count as u64) * 4096);
                }
            }
        }
        let compute_s = flops / (self.matrix_tflops * 1.0e12);
        let transfer_s = bytes as f64 / (self.hbm_bandwidth_gbytes * 1.0e9);
        let sync_s = barriers as f64 * 2.0e-6;
        let latency =
            Duration::from_secs_f64((compute_s + transfer_s + sync_s) * self.latency_multiplier);
        CostEstimate {
            latency,
            peak_memory: bytes,
            materialized_bytes: graph.edge_count() as u64 * 1024 * 1024,
            sync_count: graph.edge_count() as u32 + barriers,
            energy: latency.as_secs_f64() * 600.0,
            error: 0.25,
        }
    }

    fn name(&self) -> &str {
        "mi300x_gfx942"
    }
}

/// A simple heuristic cost model for testing and early-stage use.
///
/// Assigns costs based on node count, compute kind, and edge count, with no
/// calibration data. Useful for development but should be replaced with a
/// calibrated table-based model for production use.
pub struct SimpleCostModel;

impl CostModel for SimpleCostModel {
    fn estimate(&self, graph: &crate::graph::SpatialGraph) -> CostEstimate {
        let node_count = graph.node_count() as f64;
        let edge_count = graph.edge_count() as f64;

        // Heuristic: each compute node costs ~1ms, each edge costs ~10µs
        let latency_s = node_count * 0.001 + edge_count * 0.00001;

        // Memory: assume 64MB per compute node for weights + activations
        let peak_memory = (node_count as u64) * 64 * 1024 * 1024;

        // Materialized bytes: each edge transfers ~1MB
        let materialized_bytes = (edge_count as u64) * 1024 * 1024;

        CostEstimate {
            latency: Duration::from_secs_f64(latency_s),
            peak_memory,
            materialized_bytes,
            sync_count: edge_count as u32,
            energy: node_count * 0.5, // ~0.5J per node
            error: 0.5,               // moderate uncertainty
        }
    }

    fn name(&self) -> &str {
        "simple_heuristic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration_report::M1CalibrationReport;
    use crate::graph::SpatialEdge;
    use crate::graph::SpatialGraph;
    use crate::graph::{
        ComputeIntensity, EdgeDirection, MemoryKind, SpatialEdgeId, SpatialNodeId, TileGeometry,
    };
    use prism_ecs_ir::cimage_types::TensorShape;

    #[test]
    fn zero_cost_estimate() {
        let cost = CostEstimate::zero();
        assert_eq!(cost.latency, Duration::ZERO);
        assert_eq!(cost.peak_memory, 0);
        assert_eq!(cost.error, 1.0);
    }

    #[test]
    fn fitness_score_basic() {
        let cost = CostEstimate::new(
            Duration::from_millis(100),
            1024 * 1024 * 1024, // 1 GiB
            512 * 1024 * 1024,  // 512 MiB
            5,
            10.0,
            0.1,
        );
        let score = cost.fitness_score();
        assert!(score > 0.0);
        // Lower latency should give a lower (better) score
        let cost_fast = CostEstimate::new(
            Duration::from_millis(10),
            1024 * 1024 * 1024,
            512 * 1024 * 1024,
            5,
            10.0,
            0.1,
        );
        assert!(cost_fast.fitness_score() < score);
    }

    #[test]
    fn codec_bits() {
        assert_eq!(CodecVariant::Nf4.bits_per_element(), 4);
        assert_eq!(CodecVariant::Fp16.bits_per_element(), 16);
        assert_eq!(CodecVariant::Bf16.bits_per_element(), 16);
        assert_eq!(CodecVariant::Ternary.bits_per_element(), 2);
        assert_eq!(CodecVariant::Ternary1_58.bits_per_element(), 2);
        assert_eq!(CodecVariant::Q2_0.bits_per_element(), 2);
        assert_eq!(CodecVariant::Q4_0.bits_per_element(), 4);
    }

    #[test]
    fn supported_codec_variants() {
        assert!(CodecVariant::Fp32.is_supported_codec());
        assert!(CodecVariant::Fp16.is_supported_codec());
        assert!(CodecVariant::Bf16.is_supported_codec());
        assert!(CodecVariant::Int8.is_supported_codec());
        assert!(CodecVariant::SymInt4.is_supported_codec());
        assert!(CodecVariant::Nf4.is_supported_codec());
        assert!(CodecVariant::Q2_0.is_supported_codec());
        assert!(CodecVariant::Q4_0.is_supported_codec());
        assert!(!CodecVariant::Ternary.is_supported_codec());
        assert!(!CodecVariant::Ternary1_58.is_supported_codec());
        assert!(!CodecVariant::Q8_0.is_supported_codec());
    }

    #[test]
    fn simple_cost_model() {
        let mut g = SpatialGraph::new();
        g.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });

        let model = SimpleCostModel;
        let estimate = model.estimate(&g);
        assert!(estimate.latency.as_secs_f64() > 0.0);
        assert!(estimate.peak_memory > 0);
        assert_eq!(model.name(), "simple_heuristic");
    }

    #[test]
    fn joint_tiling_cost_evaluates_ane_and_metal_separately() {
        let mut graph = SpatialGraph::new();
        graph.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![128, 128],
                }],
                vec![TensorShape {
                    dims: vec![128, 128],
                }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });

        let model = SimpleCostModel;
        let estimates = evaluate_joint_tiling_configurations(
            &model,
            &graph,
            &[
                TileGeometry {
                    width: 8,
                    height: 8,
                },
                TileGeometry {
                    width: 32,
                    height: 8,
                },
            ],
        );

        let first = estimates[0].as_ref().unwrap();
        let second = estimates[1].as_ref().unwrap();
        assert_ne!(first.ane.latency, second.ane.latency);
        assert_ne!(first.metal.latency, second.metal.latency);
        assert!(first.ane_tile_count > second.ane_tile_count);
        assert!(first.fitness_score().is_finite());
    }

    #[test]
    fn joint_tiling_selection_preserves_invalid_candidate_rejections() {
        let graph = SpatialGraph::new();
        let model = SimpleCostModel;
        let candidates = [
            TileGeometry {
                width: 0,
                height: 8,
            },
            TileGeometry {
                width: 32,
                height: 8,
            },
            TileGeometry {
                width: 256,
                height: 5,
            },
        ];

        let evaluated = evaluate_joint_tiling_configurations(&model, &graph, &candidates);
        assert!(evaluated[0].is_err());
        assert!(evaluated[1].is_ok());
        assert!(evaluated[2].is_err());

        let selected = select_best_joint_tiling(&model, &graph, &candidates).unwrap();
        assert_eq!(selected.geometry, candidates[1]);

        let all_invalid = select_best_joint_tiling(
            &model,
            &graph,
            &[TileGeometry {
                width: 256,
                height: 5,
            }],
        );
        assert!(matches!(
            all_invalid,
            Err(TilingSelectionError::NoLegalCandidates { .. })
        ));
    }

    #[test]
    fn calibrated_cost_model_estimate() {
        let report = M1CalibrationReport::plausible_defaults();
        let model = CalibratedCostModel::new(&report);

        let mut g = SpatialGraph::new();
        g.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![
                    TensorShape {
                        dims: vec![64, 128],
                    },
                    TensorShape {
                        dims: vec![128, 64],
                    },
                ],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.add_node(SpatialNode::Compute {
            id: SpatialNodeId(2),
            kind: ComputeKind::Normalization,
            shape: ShapeContract::new(
                vec![TensorShape { dims: vec![64, 64] }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::MemoryBound,
        });
        g.add_edge(SpatialEdge {
            id: SpatialEdgeId(1),
            source: SpatialNodeId(1),
            sink: SpatialNodeId(2),
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: None,
        });

        let estimate = model.estimate(&g);

        assert!(estimate.latency.as_secs_f64() > 0.0);
        assert!(estimate.latency.as_micros() > 0);
        assert!(estimate.peak_memory > 0);
        assert!(estimate.energy > 0.0);
        assert!(estimate.sync_count >= 1);
        assert!(estimate.error >= 0.0);
        assert!(estimate.error <= 1.0);
        assert_eq!(model.name(), "calibrated_m1");

        // Adding a barrier should increase sync count
        g.add_node(SpatialNode::Barrier {
            id: SpatialNodeId(3),
            dependencies: vec![SpatialNodeId(1), SpatialNodeId(2)],
        });
        let estimate2 = model.estimate(&g);
        assert!(estimate2.sync_count > estimate.sync_count);
    }

    #[test]
    fn calibrated_cost_model_monotonic_larger_graph() {
        let report = M1CalibrationReport::plausible_defaults();
        let model = CalibratedCostModel::new(&report);

        let mut small = SpatialGraph::new();
        small.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::Elementwise,
            shape: ShapeContract::new(
                vec![TensorShape { dims: vec![32, 32] }],
                vec![TensorShape { dims: vec![32, 32] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });

        let mut large = SpatialGraph::new();
        large.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::Elementwise,
            shape: ShapeContract::new(
                vec![TensorShape { dims: vec![32, 32] }],
                vec![TensorShape { dims: vec![32, 32] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        large.add_node(SpatialNode::Compute {
            id: SpatialNodeId(2),
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![
                    TensorShape { dims: vec![32, 32] },
                    TensorShape { dims: vec![32, 32] },
                ],
                vec![TensorShape { dims: vec![32, 32] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        large.add_edge(SpatialEdge {
            id: SpatialEdgeId(1),
            source: SpatialNodeId(1),
            sink: SpatialNodeId(2),
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: None,
        });

        let e_small = model.estimate(&small);
        let e_large = model.estimate(&large);

        assert!(e_large.latency >= e_small.latency);
        assert!(e_large.peak_memory >= e_small.peak_memory);
        assert!(e_large.energy >= e_small.energy);
        assert!(e_large.sync_count >= e_small.sync_count);
    }

    #[test]
    fn calibrated_cost_model_memory_node() {
        let report = M1CalibrationReport::plausible_defaults();
        let model = CalibratedCostModel::new(&report);

        let mut g = SpatialGraph::new();
        g.add_node(SpatialNode::Memory {
            id: SpatialNodeId(1),
            kind: MemoryKind::WeightStorage,
            region: crate::graph::MemoryRegion {
                shape: TensorShape {
                    dims: vec![4096, 4096],
                },
                element_size: 2,
                strides: vec![],
            },
        });

        let estimate = model.estimate(&g);
        // 4096 * 4096 = 16,777,216 elements × 2 bytes = 33,554,432 bytes
        assert!(estimate.peak_memory >= 33_554_432);
        assert!(estimate.latency.as_secs_f64() > 0.0);
        assert!(estimate.peak_memory > 33_000_000);
        assert!(estimate.sync_count == 0);
    }
}
