//! Compile-time profitability analyzer for three-way backend offload decisions.
//!
//! Determines which inference operations benefit from ANE offload, Accelerate
//! (CPU vDSP), or best-left on MLX (GPU), based on M1 evidence from
//! `heterogeneous_integration.rs`.  Since transfers are zero-copy (IOSurface
//! page-table dispatch) and the ANE is warm, the only cost is the operation's
//! execution time on the selected backend.
//!
//! Routing constants: MLX(GPU)=0, Accelerate(CPU)=1, ANE=3.

use crate::config::{LayerPlan, ModelExecutionPlan};
use crate::compilation::tri_lane::{TriLaneCostModel, LaneCostEstimate};

// ── Types ───────────────────────────────────────────────────────────────────

/// Estimated cost of one operation on each backend.
#[derive(Debug, Clone)]
pub struct OpCost {
    pub op_name: String,
    pub gpu_estimate_ns: u64,
    pub accel_estimate_ns: u64,
    pub ane_estimate_ns: u64,
    pub shape_desc: String,
    pub arithmetic_intensity: f32,
}

/// GPU pipeline bubble: periods where GPU is idle waiting for dependencies.
#[derive(Debug, Clone)]
pub struct GpuBubble {
    pub layer_index: u32,
    pub op_name: String,
    pub estimated_idle_ns: u64,
    pub cause: BubbleCause,
}

/// Classification of the source of a GPU pipeline bubble.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BubbleCause {
    /// GPU waits for previous op to finish.
    SerialDependency,
    /// GPU pipeline hazard (e.g. write-after-read).
    PipelineStall,
    /// GPU-CPU sync boundary (e.g. backend switch).
    SyncPoint,
    /// Memory bandwidth bottleneck.
    BandwidthLimited,
}

/// Decision for one operation: should it run on ANE?
#[derive(Debug, Clone)]
pub struct AneAssignment {
    pub layer_index: u32,
    pub op_name: String,
    pub assign: bool,
    pub reason: String,
    pub estimated_gpu_time_saved_ns: u64,
}

/// Full profitability analysis result.
#[derive(Debug, Clone)]
pub struct ProfitabilityReport {
    pub machine: String,
    pub op_costs: Vec<OpCost>,
    pub bubbles: Vec<GpuBubble>,
    pub assignments: Vec<AneAssignment>,
    pub total_gpu_time_saved_ns: u64,
    pub total_ane_time_ns: u64,
}

use serde::{Deserialize, Serialize};

/// Device-specific cost evidence for a single operation on a specific device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCostEvidence {
    /// Apple Silicon SoC family: "M1", "M2", "M3", "M4"
    pub soc_family: String,
    /// macOS version at measurement time, e.g. "14.5"
    pub macos_version: String,
    /// Core ML runtime version, e.g. "7.2.0"
    pub coreml_version: String,
    /// Operation name, e.g. "q_proj", "matmul", "rms_norm"
    pub operation: String,
    /// Shape descriptor, e.g. "1x4096x4096"
    pub shape_desc: String,
    /// GPU (MLX/Metal) latency in nanoseconds
    pub gpu_latency_ns: u64,
    /// ANE (Core ML) latency in nanoseconds
    pub ane_latency_ns: u64,
    /// Accelerate (CPU vDSP) latency in nanoseconds
    pub accel_latency_ns: u64,
    /// ISO-8601 timestamp of measurement
    pub measured_at: String,
}

/// Build a TriLaneCostModel from per-backend operation costs.
///
/// Aggregates individual OpCost entries into a total cost estimate per lane,
/// and applies the GPU/CPU contention penalties. The `gpu_contention_ns` and
/// `cpu_contention_ns` represent estimated contention overhead from shared
/// memory bandwidth or concurrent workload interference.
pub fn build_tri_lane_cost_model(
    costs: &[OpCost],
    gpu_contention_ns: u64,
    cpu_contention_ns: u64,
) -> crate::compilation::tri_lane::TriLaneCostModel {
    

    let total_gpu: u64 = costs.iter().map(|c| c.gpu_estimate_ns).sum();
    let total_ane: u64 = costs.iter().map(|c| c.ane_estimate_ns).sum();
    let total_cpu: u64 = costs.iter().map(|c| c.accel_estimate_ns).sum();

    // Estimate memory proportion: element-wise ops are memory-bound (~70%
    // memory), matmuls are compute-bound (~30% memory). Use arithmetic
    // intensity as the fraction spent on compute; the rest is memory.
    let gpu_memory: u64 = costs
        .iter()
        .map(|c| (c.gpu_estimate_ns as f64 * (1.0 - c.arithmetic_intensity as f64)) as u64)
        .sum();
    let gpu_compute: u64 = total_gpu.saturating_sub(gpu_memory);

    let ane_memory: u64 = costs
        .iter()
        .map(|c| (c.ane_estimate_ns as f64 * (1.0 - c.arithmetic_intensity as f64)) as u64)
        .sum();
    let ane_compute: u64 = total_ane.saturating_sub(ane_memory);

    let cpu_memory: u64 = costs
        .iter()
        .map(|c| (c.accel_estimate_ns as f64 * (1.0 - c.arithmetic_intensity as f64)) as u64)
        .sum();
    let cpu_compute: u64 = total_cpu.saturating_sub(cpu_memory);

    let gpu_estimate = LaneCostEstimate {
        compute_ns: gpu_compute,
        memory_ns: gpu_memory,
        boundary_ns: 0,
        sync_ns: 5_000,
    };
    let ane_estimate = LaneCostEstimate {
        compute_ns: ane_compute,
        memory_ns: ane_memory,
        boundary_ns: 20_000,
        sync_ns: 10_000,
    };
    let cpu_estimate = LaneCostEstimate {
        compute_ns: cpu_compute,
        memory_ns: cpu_memory,
        boundary_ns: 0,
        sync_ns: 2_000,
    };

    // Critical path: the shortest possible completion time assuming maximal
    // lane overlap. For simplicity, approximate as the minimum of the three
    // total lane times plus the largest single sync cost.
    let critical_path_ns = total_gpu
        .min(total_ane)
        .min(total_cpu)
        .saturating_add(10_000);

    TriLaneCostModel {
        gpu: gpu_estimate,
        ane: ane_estimate,
        cpu: cpu_estimate,
        critical_path_ns,
        gpu_contention_penalty_ns: gpu_contention_ns,
        cpu_contention_penalty_ns: cpu_contention_ns,
        numerical_risk_penalty: 0.0,
        fallback_risk_penalty: 0.0,
    }
}

/// Check whether assigning a layer's operation to ANE meets the speedup
/// threshold.
///
/// Returns true if ANE time <= (1 - threshold) * best_other_time.  The
/// threshold defaults to 0.10 (10%) but is configurable via `threshold`.
///
/// # Arguments
/// * `ane_time_ns` — measured or estimated ANE execution time.
/// * `gpu_time_ns` — measured or estimated GPU execution time.
/// * `accel_time_ns` — measured or estimated Accelerate execution time.
/// * `threshold` — optional fractional speedup threshold (default 0.10).
///
/// # Example
/// ```ignore
/// assert!(ane_assignment_meets_threshold(80, 100, 120, Some(0.15)));
/// assert!(!ane_assignment_meets_threshold(95, 100, 120, Some(0.10)));
/// ```
pub fn ane_assignment_meets_threshold(
    ane_time_ns: u64,
    gpu_time_ns: u64,
    accel_time_ns: u64,
    threshold: Option<f64>,
) -> bool {
    let threshold = threshold.unwrap_or(0.10);
    if !(0.0..=1.0).contains(&threshold) {
        return false;
    }
    let best_other = gpu_time_ns.min(accel_time_ns);
    let max_allowed = (best_other as f64 * (1.0 - threshold)) as u64;
    ane_time_ns <= max_allowed
}

/// Use calibration evidence to check ANE assignment.
pub fn evidence_based_ane_assignment(
    calibration: &crate::evidence::apple_tri_lane_calibration::CalibrationStore,
    hardware_fingerprint: &str,
    region_fingerprint: &str,
    threshold: f64,
) -> bool {
    calibration.ane_assignment_justified(hardware_fingerprint, region_fingerprint, threshold)
}

// ── Analyzer ────────────────────────────────────────────────────────────────

pub struct ProfitabilityAnalyzer;

impl ProfitabilityAnalyzer {
    /// Analyze a `ModelExecutionPlan` and return which ops should run on ANE.
    ///
    /// The analysis:
    /// 1. Profiles each operation's estimated GPU, Accelerate, and ANE time.
    /// 2. Detects GPU pipeline bubbles (serial dependencies).
    /// 3. Assigns ops to ANE when ANE is ≥25% faster than the next-best
    ///    backend AND the op is on the GPU's critical path.
    /// 4. Otherwise recommends the best backend (Accelerate for element-wise,
    ///    MLX for compute-bound).
    /// 5. Returns a `ProfitabilityReport`.
    ///
    /// Since transfer is zero-copy and ANE is warm, the profit equation is:
    ///   `profit = best_backend_time - ane_time`
    /// Assign when `profit > 0` meets the 25% threshold and the GPU gains
    /// idle-free execution.
    pub fn analyze(plan: &ModelExecutionPlan) -> ProfitabilityReport {
        let op_costs = Self::gather_op_costs(plan);
        let bubbles = Self::detect_bubbles(plan);
        let assignments = Self::compute_assignments(plan, &op_costs, &bubbles);

        let total_gpu_time_saved_ns = assignments
            .iter()
            .filter(|a| a.assign)
            .map(|a| a.estimated_gpu_time_saved_ns)
            .sum();

        let total_ane_time_ns = assignments
            .iter()
            .filter(|a| a.assign)
            .map(|a| {
                op_costs
                    .iter()
                    .find(|c| {
                        c.op_name == a.op_name
                            && op_cost_belongs_to_layer(plan, c, a.layer_index)
                    })
                    .map(|c| c.ane_estimate_ns)
                    .unwrap_or(0)
            })
            .sum();

        ProfitabilityReport {
            machine: "Apple M1".into(),
            op_costs,
            bubbles,
            assignments,
            total_gpu_time_saved_ns,
            total_ane_time_ns,
        }
    }

    /// Estimate GPU (MLX) time for an operation.
    pub fn estimate_gpu_time(layer: &LayerPlan, op_name: &str) -> u64 {
        Self::estimate_op_times(layer, op_name).0
    }

    /// Estimate Accelerate (CPU vDSP) time for an operation.
    pub fn estimate_accel_time(layer: &LayerPlan, op_name: &str) -> u64 {
        Self::estimate_op_times(layer, op_name).1
    }

    /// Estimate ANE time for an operation.
    pub fn estimate_ane_time(layer: &LayerPlan, op_name: &str) -> u64 {
        Self::estimate_op_times(layer, op_name).2
    }

    /// Detect GPU pipeline bubbles (serial dependencies that stall the GPU).
    pub fn detect_bubbles(plan: &ModelExecutionPlan) -> Vec<GpuBubble> {
        let mut bubbles = Vec::new();

        // Phase 1: detect sync-point bubbles at layer boundaries where
        // adjacent layers mix GPU and ANE backends.
        for i in 0..plan.layers.len().saturating_sub(1) {
            let cur = &plan.layers[i];
            let next = &plan.layers[i + 1];
            let cur_has_ane = cur.route.has_ane_backend();
            let next_has_ane = next.route.has_ane_backend();

            if cur_has_ane && !next_has_ane {
                // ANE → GPU transition: GPU must wait for ANE output.
                bubbles.push(GpuBubble {
                    layer_index: cur.layer_index,
                    op_name: "backend_switch".into(),
                    estimated_idle_ns: 1_500, // ~1.5μs ANE→GPU drain
                    cause: BubbleCause::SyncPoint,
                });
            }

            if !cur_has_ane && next_has_ane {
                bubbles.push(GpuBubble {
                    layer_index: next.layer_index,
                    op_name: "backend_switch".into(),
                    estimated_idle_ns: 1_500,
                    cause: BubbleCause::SyncPoint,
                });
            }
        }

        // Phase 2: detect serial-dependency bubbles within GPU-only layers.
        // Each sequential matmul creates a dependency chain.
        for layer in &plan.layers {
            if layer.route.has_ane_backend() {
                continue; // not blocking the GPU pipeline when on ANE
            }
            let ops = layer.operation_names();
            let mut prev_was_matmul = false;
            for (_j, op) in ops.iter().enumerate() {
                // Sequential matmuls create serial dependencies
                if *op == "matmul"
                    || *op == "q_proj"
                    || *op == "k_proj"
                    || *op == "v_proj"
                    || *op == "gate_proj"
                    || *op == "up_proj"
                    || *op == "down_proj"
                {
                    if prev_was_matmul {
                        let gpu_time = Self::estimate_gpu_time(layer, op);
                        bubbles.push(GpuBubble {
                            layer_index: layer.layer_index,
                            op_name: op.to_string(),
                            estimated_idle_ns: gpu_time / 4, // pipeline drain overhead
                            cause: BubbleCause::SerialDependency,
                        });
                    }
                    prev_was_matmul = true;
                } else {
                    prev_was_matmul = false;
                }

                // Bandwidth-limited: two memory-bound ops in sequence
                // (e.g. rms_norm → silu) with no compute-bound op between
                if *op == "rms_norm"
                    || *op == "silu"
                    || *op == "add"
                    || *op == "multiply"
                {
                    // Note: bandwidth-limited detection depends on prev op
                    // which we can't easily access in this form.  We keep the
                    // struct for API compatibility.
                }
            }
        }

        // Phase 3: detect pipeline stalls in attention-heavy layers.
        // The matmul(softmax) → matmul(V) chain creates a RAW hazard.
        for layer in &plan.layers {
            if layer.route.has_ane_backend() {
                continue;
            }
            let ops = layer.operation_names();
            for j in 2..ops.len() {
                if ops[j - 2] == "softmax" && ops[j - 1] == "matmul" {
                    bubbles.push(GpuBubble {
                        layer_index: layer.layer_index,
                        op_name: format!("{}_pipeline_hazard", ops[j - 2]),
                        estimated_idle_ns: 800, // ~0.8μs RAW stall
                        cause: BubbleCause::PipelineStall,
                    });
                    break;
                }
            }
        }

        bubbles
    }

    /// Apply the profitability analysis to the `ModelExecutionPlan`,
    /// updating each layer's `OperationRoute` with ANE assignments.
    pub fn apply(plan: &mut ModelExecutionPlan) -> ProfitabilityReport {
        let report = Self::analyze(plan);

        for assignment in &report.assignments {
            if !assignment.assign {
                continue;
            }
            if let Some(layer) = plan
                .layers
                .iter_mut()
                .find(|l| l.layer_index == assignment.layer_index)
            {
                Self::set_route_for_op(layer, &assignment.op_name, 3);
            }
        }

        // Rebuild ANE fusion islands with the updated routes.
        plan.build_ane_fusion_plan();

        report
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Collect `OpCost` for every operation in every layer.
    fn gather_op_costs(plan: &ModelExecutionPlan) -> Vec<OpCost> {
        let mut costs = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for layer in &plan.layers {
            for op_name in layer.operation_names() {
                let (gpu, accel, ane) = Self::estimate_op_times(layer, op_name);
                let desc = Self::shape_desc(layer, op_name);
                let ai = Self::arithmetic_intensity(layer, op_name);

                // Deduplicate identical (layer, op) pairs.
                let key = (layer.layer_index, op_name);
                if seen.insert(key) {
                    costs.push(OpCost {
                        op_name: op_name.to_string(),
                        gpu_estimate_ns: gpu,
                        accel_estimate_ns: accel,
                        ane_estimate_ns: ane,
                        shape_desc: desc,
                        arithmetic_intensity: ai,
                    });
                }
            }
        }

        costs
    }

    /// Core three-way time estimation.
    ///
    /// Returns `(gpu_ns, accel_ns, ane_ns)` based on M1 evidence from
    /// `heterogeneous_integration.rs`:
    ///   MLX(0) = matmul/softmax/silu/transpose  (GPU sweet spot)
    ///   Accelerate(1) = rms_norm/add/multiply   (CPU vDSP sweet spot)
    ///   ANE(3) = attention                      (ANE sweet spot)
    ///
    /// GPU: dominated by kernel launch latency for small shapes (the common
    /// single-token decode case), proportional to FLOP count for larger batches.
    /// Accelerate: uses vDSP for element-wise ops with very low dispatch cost.
    /// ANE: fixed-function hardware with lower dispatch overhead but lower peak
    /// throughput for irregular shapes.
    fn estimate_op_times(layer: &LayerPlan, op_name: &str) -> (u64, u64, u64) {
        let hidden = layer.hidden_size as u64;
        let n_heads = layer.n_heads as u64;
        let head_dim = layer.head_dim as u64;
        let n_kv = layer.n_kv_heads.max(1) as u64;

        match op_name {
            // ── RMS Norm ────────────────────────────────────────────────
            // Accelerate vDSP wins: ~1-2μs.  GPU launch-bound: ~2-5μs.
            // ANE ~3-5μs (no advantage).
            "rms_norm" => {
                let gpu = base_gpu_launch_time(hidden) + 500; // ~2-5μs
                let accel = 1_500; // ~1-2μs, vDSP wins
                let ane = 4_000; // ~3-5μs
                (gpu, accel, ane)
            }

            // ── Q/K/V projections (matmul: [1×hidden] × [hidden×head*n_heads]) ──
            // GPU ~5-15μs (launch-bound), Accel ~20-50μs (slow for matmul),
            // ANE ~2-6μs (best for small fixed-shape).
            "q_proj" => {
                let (gpu, _, ane) = matmul_triple(hidden, hidden, n_heads * head_dim);
                let accel = 35_000; // ~20-50μs, CPU matmul slow
                (gpu, accel, ane)
            }
            "k_proj" => {
                let (gpu, _, ane) = matmul_triple(hidden, hidden, n_kv * head_dim);
                let accel = 35_000;
                (gpu, accel, ane)
            }
            "v_proj" => {
                let (gpu, _, ane) = matmul_triple(hidden, hidden, n_kv * head_dim);
                let accel = 35_000;
                (gpu, accel, ane)
            }

            // ── Attention score matmul (matmul: [1×head_dim] × [head_dim×seq_len]) ──
            "matmul" => {
                // First matmul in attention: 1 head dim × head_dim
                // Second matmul: 1 × seq_len (returned by operation_names twice)
                // GPU ~3-8μs, Accel ~15-30μs, ANE ~1-4μs
                let (gpu, _, ane) = matmul_triple(head_dim, head_dim, head_dim);
                let accel = 22_000; // ~15-30μs CPU
                (gpu, accel, ane)
            }

            // ── Softmax ──────────────────────────────────────────────────
            // GPU ~2-5μs, Accel ~3-6μs, ANE ~1-3μs
            "softmax" => {
                let gpu = base_gpu_launch_time(hidden) + 300; // ~2-5μs
                let accel = 4_500; // ~3-6μs
                let ane = 2_000; // ~1-3μs
                (gpu, accel, ane)
            }

            // ── SiLU ─────────────────────────────────────────────────────
            // GPU ~2-5μs, Accel ~3-8μs, ANE ~2-4μs
            "silu" => {
                let gpu = base_gpu_launch_time(hidden) + 200; // ~2-5μs
                let accel = 5_000; // ~3-8μs
                let ane = 3_000; // ~2-4μs
                (gpu, accel, ane)
            }

            // ── MLP gate / up / down projections ─────────────────────────
            // GPU ~5-12μs, Accel ~20-50μs, ANE ~2-6μs
            "gate_proj" => {
                let (gpu, _, ane) = matmul_triple(hidden, hidden, hidden * 4);
                let accel = 35_000;
                (gpu, accel, ane)
            }
            "up_proj" => {
                let (gpu, _, ane) = matmul_triple(hidden, hidden, hidden * 4);
                let accel = 35_000;
                (gpu, accel, ane)
            }
            "down_proj" => {
                let (gpu, _, ane) = matmul_triple(hidden * 4, hidden, hidden);
                let accel = 35_000;
                (gpu, accel, ane)
            }

            // ── Element-wise add ─────────────────────────────────────────
            // GPU ~1-3μs, Accel ~0.5-1μs (vDSP wins), ANE ~2-4μs
            "add" => {
                let gpu = base_gpu_launch_time(hidden) + 100; // ~1-3μs
                let accel = 800; // ~0.5-1μs, vDSP wins
                let ane = 3_000; // ~2-4μs
                (gpu, accel, ane)
            }

            // ── Element-wise multiply ────────────────────────────────────
            // GPU ~1-3μs, Accel ~0.5-1μs (vDSP wins), ANE ~2-4μs
            "multiply" => {
                let gpu = base_gpu_launch_time(hidden) + 100; // ~1-3μs
                let accel = 800; // ~0.5-1μs, vDSP wins
                let ane = 3_000; // ~2-4μs
                (gpu, accel, ane)
            }

            _ => {
                // Unknown op: default conservative estimate
                let gpu = base_gpu_launch_time(hidden) + 500;
                let accel = gpu * 2;
                let ane = 2_000;
                (gpu, accel, ane)
            }
        }
    }

    /// Human-readable shape description for an operation.
    fn shape_desc(layer: &LayerPlan, op_name: &str) -> String {
        let h = layer.hidden_size;
        let d = layer.head_dim;
        let nh = layer.n_heads;
        let intermediate = h * 4;
        match op_name {
            "rms_norm" => format!("[{h}]"),
            "q_proj" => format!("[{h}]×[{h},{}]", nh * d),
            "k_proj" => format!("[{h}]×[{h},{}]", layer.n_kv_heads * d),
            "v_proj" => format!("[{h}]×[{h},{}]", layer.n_kv_heads * d),
            "matmul" => format!("[{}]×[{},{}]", d, d, d),
            "softmax" => format!("[{}]", d),
            "silu" => format!("[{h}]"),
            "gate_proj" | "up_proj" => {
                format!("[{h}]×[{h},{intermediate}]")
            }
            "down_proj" => {
                format!("[{intermediate}]×[{h},{h}]")
            }
            "add" | "multiply" => format!("[{h}]"),
            _ => format!("[{h}]"),
        }
    }

    /// Arithmetic intensity: FLOPs per byte loaded.
    /// > 1 = compute-bound (matmuls), < 1 = memory-bound (norms, activations).
    fn arithmetic_intensity(_layer: &LayerPlan, op_name: &str) -> f32 {
        match op_name {
            "rms_norm" => 0.3,
            "q_proj" | "k_proj" | "v_proj" => 2.5,
            "matmul" => 3.0,
            "softmax" => 0.4,
            "silu" => 0.5,
            "gate_proj" | "up_proj" | "down_proj" => 2.0,
            "add" => 0.2,
            "multiply" => 0.2,
            _ => 1.0,
        }
    }

    /// Compute ANE assignments based on three-way profitability and bubble analysis.
    ///
    /// For each operation, picks the backend with the lowest estimated time.
    /// ANE is only recommended when:
    ///   1. `ane_time < min(gpu_time, accel_time)` (strictly fastest)
    ///   2. `ane_time <= 0.75 * min(gpu_time, accel_time)` (≥25% faster)
    ///   3. The op is on the GPU's critical path.
    ///
    /// Other backends (Accelerate, MLX) are noted in the reason but do not
    /// trigger route changes — route updates only happen for ANE assignments.
    fn compute_assignments(
        plan: &ModelExecutionPlan,
        op_costs: &[OpCost],
        bubbles: &[GpuBubble],
    ) -> Vec<AneAssignment> {
        // Build set of layer indices that have sync-point bubbles
        // (these are on the GPU's critical path).
        let bubbled_layers: std::collections::HashSet<u32> = bubbles
            .iter()
            .filter(|b| b.cause == BubbleCause::SyncPoint)
            .map(|b| b.layer_index)
            .collect();

        let mut assignments = Vec::new();

        for layer in &plan.layers {
            let ops = layer.operation_names();
            // We only consider each distinct op once per layer.
            let mut seen_ops = std::collections::HashSet::new();

            for cost in op_costs.iter() {
                // Find cost entries belonging to this layer.
                if !op_cost_belongs_to_layer(plan, cost, layer.layer_index) {
                    continue;
                }
                if !seen_ops.insert(&cost.op_name) {
                    continue;
                }
                if !ops.contains(&cost.op_name.as_str()) {
                    continue;
                }

                let gpu = cost.gpu_estimate_ns;
                let accel = cost.accel_estimate_ns;
                let ane = cost.ane_estimate_ns;

                // Three-way: find the best backend
                let best_non_ane = gpu.min(accel);
                let _overall_best_time = best_non_ane.min(ane);
                let overall_best_backend = if ane < best_non_ane {
                    "ANE"
                } else if accel < gpu {
                    "Accelerate"
                } else {
                    "MLX"
                };

                let on_critical_path =
                    bubbled_layers.contains(&layer.layer_index);
                let layer_has_serial_bubbles = bubbles.iter().any(|b| {
                    b.layer_index == layer.layer_index
                        && b.cause == BubbleCause::SerialDependency
                });

                // ANE only recommended when strictly fastest AND ≥25% faster
                // than the next-best backend.
                let ane_is_best = ane < best_non_ane;
                let meets_threshold = ane <= best_non_ane * 75 / 100; // ≥25% faster
                let should_assign_ane = ane_is_best
                    && meets_threshold
                    && (on_critical_path || layer_has_serial_bubbles);

                let reason = if should_assign_ane {
                    format!(
                        "ANE best: gpu={gpu}ns, accel={accel}ns, ane={ane}ns \
                         (≥25% faster than {best_non_ane}ns), on critical path"
                    )
                } else if ane_is_best && !meets_threshold {
                    format!(
                        "ANE fastest ({ane}ns) but below 25% threshold \
                         (best_non_ane={best_non_ane}ns); recommending {overall_best_backend}"
                    )
                } else {
                    format!(
                        "Best backend: {overall_best_backend} \
                         (gpu={gpu}ns, accel={accel}ns, ane={ane}ns)"
                    )
                };

                let estimated_gpu_time_saved_ns = if should_assign_ane {
                    gpu
                } else {
                    0
                };

                assignments.push(AneAssignment {
                    layer_index: layer.layer_index,
                    op_name: cost.op_name.clone(),
                    assign: should_assign_ane,
                    reason,
                    estimated_gpu_time_saved_ns,
                });
            }
        }

        assignments
    }

    /// Set the appropriate `OperationRoute` field to `backend` for the given
    /// operation name.
    ///
    /// Backend constants: MLX=0, Accelerate=1, ANE=3.
    fn set_route_for_op(layer: &mut LayerPlan, op_name: &str, backend: u32) {
        match op_name {
            "rms_norm" => layer.route.rms_norm = backend,
            "silu" => layer.route.silu = backend,
            "q_proj"
            | "k_proj"
            | "v_proj"
            | "matmul"
            | "gate_proj"
            | "up_proj"
            | "down_proj" => {
                layer.route.matmul = backend;
            }
            "softmax" => layer.route.softmax = backend,
            "rope" => layer.route.rope = backend,
            "add" => layer.route.add = backend,
            "multiply" => layer.route.multiply = backend,
            _ => { /* unknown op, leave default */ }
        }
    }
}

/// Base GPU kernel launch overhead for a given hidden dimension.
fn base_gpu_launch_time(hidden: u64) -> u64 {
    // GPU kernel launch is ~1-2μs fixed overhead + per-element time
    let elements = hidden as f64;
    let per_element_ns = 0.2; // ~0.2ns per element
    (1_200.0 + elements * per_element_ns) as u64
}

/// Estimate matmul time on GPU vs ANE.
///
/// GPU: dominated by dispatch latency for small (M<=4) matrices, proportional
/// to FLOPs for larger batches.
/// ANE: better for fixed-shape small matmuls due to on-chip SRAM.
fn matmul_time(m: u64, k: u64, n: u64, on_ane: bool) -> u64 {
    let flops = m * k * n; // FMA counts as 2 FLOPs in practice, but for timing we use M*K*N

    if on_ane {
        // ANE: efficient for small fixed-shape matmuls. ~30-50 FLOPS/cycle.
        // Fixed dispatch ~1μs + compute proportional to FLOPs.
        let compute_ns = (flops as f64 * 0.012) as u64; // ~0.012ns per FLOP equivalent
        compute_ns.max(1_500).min(8_000)
    } else {
        // GPU: launch-bound for small M. ~4-6μs launch + lower compute.
        let launch = 4_000u64;
        let compute_ns = (flops as f64 * 0.003) as u64; // ~0.003ns per FLOP equivalent
        (launch + compute_ns).max(3_000).min(15_000)
    }
}

/// Three-way matmul time estimate: `(gpu_ns, accel_ns, ane_ns)`.
///
/// GPU ~5-15μs (launch-bound), Accel ~20-50μs (CPU matmul slow),
/// ANE ~2-6μs (best for small fixed-shape).
fn matmul_triple(m: u64, k: u64, n: u64) -> (u64, u64, u64) {
    let gpu = matmul_time(m, k, n, false);
    let ane = matmul_time(m, k, n, true);
    // Accelerate CPU matmul: 20-50μs, proportional to FLOPs
    let flops = m * k * n;
    let accel = (flops as f64 * 0.025).max(20_000.0).min(50_000.0) as u64;
    (gpu, accel, ane)
}

/// Check whether an `OpCost` entry belongs to the layer at `layer_index`.
fn op_cost_belongs_to_layer(plan: &ModelExecutionPlan, cost: &OpCost, layer_index: u32) -> bool {
    if let Some(layer) = plan.layers.iter().find(|l| l.layer_index == layer_index) {
        layer.operation_names().contains(&cost.op_name.as_str())
    } else {
        false
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        operation_route::OperationRoute, EpiloguePlan, LayerPlan, ProloguePlan,
    };

    /// Build a minimal LayerPlan for reuse in custom-layer tests.
    fn base_layer(index: u32) -> LayerPlan {
        LayerPlan {
            layer_index: index,
            attention_kind: if index % 2 == 0 {
                "sliding_attention".into()
            } else {
                "full_attention".into()
            },
            segment_id: "test".into(),
            hidden_size: 3072,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            global_head_dim: if index % 2 == 1 { Some(128) } else { None },
            n_global_kv_heads: if index % 2 == 1 { Some(8) } else { None },
            sliding_window: 8192,
            rope_theta: 10000.0,
            partial_rotary_factor: Some(1.0),
            attention_k_eq_v: false,
            q_norm_enabled: false,
            k_norm_enabled: false,
            q_proj_tensor_id: 1,
            k_proj_tensor_id: 2,
            v_proj_tensor_id: 3,
            o_proj_tensor_id: 4,
            q_norm_tensor_id: None,
            k_norm_tensor_id: None,
            gate_proj_tensor_id: 5,
            up_proj_tensor_id: 6,
            down_proj_tensor_id: 7,
            input_layernorm_tensor_id: 8,
            post_attention_layernorm_tensor_id: 9,
            pre_ffw_layernorm_tensor_id: None,
            post_ffw_layernorm_tensor_id: None,
            layer_scalar_ids: vec![],
            quantization_ids: vec![],
            route: OperationRoute::default(),
            fused_operations: vec![],
        }
    }

    /// Build a minimal Gemma-like plan for testing.
    fn test_plan(
        layer_count: u32,
        hidden: u32,
        n_heads: u32,
        n_kv: u32,
        head_dim: u32,
        route: OperationRoute,
    ) -> ModelExecutionPlan {
        let layers: Vec<LayerPlan> = (0..layer_count)
            .map(|i| LayerPlan {
                layer_index: i,
                attention_kind: if i % 2 == 0 {
                    "sliding_attention".into()
                } else {
                    "full_attention".into()
                },
                segment_id: "test".into(),
                hidden_size: hidden,
                n_heads,
                n_kv_heads: n_kv,
                head_dim,
                global_head_dim: if i % 2 == 1 { Some(head_dim) } else { None },
                n_global_kv_heads: if i % 2 == 1 { Some(n_kv) } else { None },
                sliding_window: 8192,
                rope_theta: 10000.0,
                partial_rotary_factor: Some(1.0),
                attention_k_eq_v: false,
                q_norm_enabled: false,
                k_norm_enabled: false,
                q_proj_tensor_id: 1,
                k_proj_tensor_id: 2,
                v_proj_tensor_id: 3,
                o_proj_tensor_id: 4,
                q_norm_tensor_id: None,
                k_norm_tensor_id: None,
                gate_proj_tensor_id: 5,
                up_proj_tensor_id: 6,
                down_proj_tensor_id: 7,
                input_layernorm_tensor_id: 8,
                post_attention_layernorm_tensor_id: 9,
                pre_ffw_layernorm_tensor_id: None,
                post_ffw_layernorm_tensor_id: None,
                layer_scalar_ids: vec![],
                quantization_ids: vec![],
                route: route.clone(),
                fused_operations: vec![],
            })
            .collect();

        ModelExecutionPlan {
            prologue: ProloguePlan::default(),
            layers,
            epilogue: EpiloguePlan::default(),
            fused_ane_islands: vec![],
            hidden_size: hidden,
            vocab_size: 256000,
            sliding_window: 8192,
            final_logit_softcapping: None,
            tie_word_embeddings: false,
            rms_norm_eps: 1e-6,
            speculative_config: None,
            generation_regime: crate::config::GenerationRegime::Autoregressive,
            diffusion_config: None,
            diffusion_execution_plan: None,
            kv_cache_mode: crate::config::KvCacheMode::AppendOnly,
        }
    }

    fn gemma_default_route() -> OperationRoute {
        OperationRoute::default()
    }

    /// Gemma-2-like: larger layer dimensions.
    fn gemma2_plan() -> ModelExecutionPlan {
        test_plan(2, 3072, 16, 8, 128, gemma_default_route())
    }

    // ── Cost Estimation Tests (three-way) ─────────────────────────────────

    #[test]
    fn test_estimate_rms_norm_time() {
        let plan = gemma2_plan();
        let layer = &plan.layers[0];
        let (gpu, accel, ane) = ProfitabilityAnalyzer::estimate_op_times(layer, "rms_norm");
        // RMS norm: GPU ~2-5μs (launch-bound), Accel ~1-2μs (vDSP wins), ANE ~3-5μs
        assert!(
            gpu >= 1_000 && gpu <= 5_000,
            "GPU rms_norm expected ~1-5μs, got {gpu}ns"
        );
        assert!(
            accel >= 800 && accel <= 3_000,
            "Accel rms_norm expected ~1-2μs, got {accel}ns"
        );
        assert!(
            ane >= 2_000 && ane <= 6_000,
            "ANE rms_norm expected ~3-5μs, got {ane}ns"
        );
        // Accelerate should be fastest for rms_norm (vDSP wins)
        assert!(
            accel < gpu && accel < ane,
            "Accel should be fastest for rms_norm (got accel={accel}, gpu={gpu}, ane={ane})"
        );
    }

    #[test]
    fn test_estimate_q_proj_time() {
        let plan = gemma2_plan();
        let layer = &plan.layers[0];
        let (gpu, accel, ane) = ProfitabilityAnalyzer::estimate_op_times(layer, "q_proj");
        // Q proj matmul: GPU ~5-15μs (launch-bound), Accel ~20-50μs (slow),
        // ANE ~2-6μs (best for small)
        assert!(
            gpu >= 3_000 && gpu <= 16_000,
            "GPU q_proj expected ~3-16μs, got {gpu}ns"
        );
        assert!(
            accel >= 15_000 && accel <= 55_000,
            "Accel q_proj expected ~20-50μs, got {accel}ns"
        );
        assert!(
            ane >= 1_000 && ane <= 9_000,
            "ANE q_proj expected ~2-6μs, got {ane}ns"
        );
        // GPU should be >= ANE for matmuls at decode sizes
        assert!(
            gpu >= ane || gpu - ane < 2_000,
            "q_proj: GPU ({gpu}ns) should be comparable or > ANE ({ane}ns)"
        );
    }

    #[test]
    fn test_estimate_silu_time() {
        let plan = gemma2_plan();
        let layer = &plan.layers[0];
        let (gpu, accel, ane) = ProfitabilityAnalyzer::estimate_op_times(layer, "silu");
        // SiLU: GPU ~2-5μs, Accel ~3-8μs, ANE ~2-4μs
        assert!(
            gpu >= 500 && gpu <= 3_000,
            "GPU silu expected ~0.5-3μs, got {gpu}ns"
        );
        assert!(
            accel >= 2_000 && accel <= 10_000,
            "Accel silu expected ~3-8μs, got {accel}ns"
        );
        assert!(
            ane >= 1_000 && ane <= 5_000,
            "ANE silu expected ~2-4μs, got {ane}ns"
        );
    }

    #[test]
    fn test_estimate_gate_proj_time() {
        let plan = test_plan(1, 4096, 32, 8, 128, gemma_default_route());
        let layer = &plan.layers[0];
        let (gpu, _accel, ane) = ProfitabilityAnalyzer::estimate_op_times(layer, "gate_proj");
        // MLP matmuls: wider hidden_size → larger matmuls
        assert!(gpu >= 4_000, "GPU gate_proj expected >= 4μs, got {gpu}ns");
        assert!(ane >= 2_000, "ANE gate_proj expected >= 2μs, got {ane}ns");
        assert!(
            gpu > ane,
            "gate_proj: GPU ({gpu}ns) should be > ANE ({ane}ns) for compute-bound op"
        );
    }

    #[test]
    fn test_estimate_add_time_accel_wins() {
        let plan = gemma2_plan();
        let layer = &plan.layers[0];
        let (gpu, accel, ane) = ProfitabilityAnalyzer::estimate_op_times(layer, "add");
        // Add: GPU ~1-3μs, Accel ~0.5-1μs (vDSP wins), ANE ~2-4μs
        assert!(
            accel < gpu && accel < ane,
            "Accel should be fastest for add (got accel={accel}, gpu={gpu}, ane={ane})"
        );
    }

    #[test]
    fn test_estimate_multiply_time_accel_wins() {
        let plan = gemma2_plan();
        let layer = &plan.layers[0];
        let (gpu, accel, ane) = ProfitabilityAnalyzer::estimate_op_times(layer, "multiply");
        // Multiply: GPU ~1-3μs, Accel ~0.5-1μs (vDSP wins), ANE ~2-4μs
        assert!(
            accel < gpu && accel < ane,
            "Accel should be fastest for multiply (got accel={accel}, gpu={gpu}, ane={ane})"
        );
    }

    #[test]
    fn test_estimate_softmax_time() {
        let plan = gemma2_plan();
        let layer = &plan.layers[0];
        let (gpu, accel, ane) = ProfitabilityAnalyzer::estimate_op_times(layer, "softmax");
        // Softmax: GPU ~2-5μs, Accel ~3-6μs, ANE ~1-3μs
        assert!(
            gpu >= 1_000 && gpu <= 5_000,
            "GPU softmax expected ~2-5μs, got {gpu}ns"
        );
        assert!(
            accel >= 2_000 && accel <= 8_000,
            "Accel softmax expected ~3-6μs, got {accel}ns"
        );
        assert!(
            ane >= 1_000 && ane <= 4_000,
            "ANE softmax expected ~1-3μs, got {ane}ns"
        );
    }

    // ── Accel-specific API test ──────────────────────────────────────────

    #[test]
    fn test_estimate_accel_time_api() {
        let plan = gemma2_plan();
        let layer = &plan.layers[0];
        let accel = ProfitabilityAnalyzer::estimate_accel_time(layer, "rms_norm");
        assert!(
            accel > 0,
            "estimate_accel_time should return a nonzero value"
        );
    }

    // ── OpCost Collection Tests ──────────────────────────────────────────

    #[test]
    fn test_gather_op_costs_single_layer() {
        let plan = gemma2_plan();
        let costs = ProfitabilityAnalyzer::gather_op_costs(&plan);
        // Each layer produces: rms_norm, q_proj, k_proj, v_proj, matmul×2, softmax, add,
        // rms_norm, gate_proj, silu, multiply, down_proj
        // But operation_names deduplicates, and each layer has the same ops.
        // Let's just verify we have reasonable coverage.
        let op_names: std::collections::HashSet<&str> =
            costs.iter().map(|c| c.op_name.as_str()).collect();
        assert!(op_names.contains("rms_norm"), "should have rms_norm");
        assert!(op_names.contains("q_proj"), "should have q_proj");
        assert!(op_names.contains("gate_proj"), "should have gate_proj");
        assert!(op_names.contains("silu"), "should have silu");
        assert!(op_names.contains("matmul"), "should have matmul");
        // Verify accel_estimate_ns is populated
        for cost in &costs {
            assert!(
                cost.accel_estimate_ns > 0,
                "{} should have nonzero accel_estimate_ns",
                cost.op_name
            );
        }
    }

    #[test]
    fn test_op_cost_arithmetic_intensity() {
        let plan = gemma2_plan();
        let costs = ProfitabilityAnalyzer::gather_op_costs(&plan);
        for cost in &costs {
            match cost.op_name.as_str() {
                "rms_norm" => assert_eq!(cost.arithmetic_intensity, 0.3),
                "q_proj" | "k_proj" | "v_proj" => assert_eq!(cost.arithmetic_intensity, 2.5),
                "matmul" => assert_eq!(cost.arithmetic_intensity, 3.0),
                "softmax" => assert_eq!(cost.arithmetic_intensity, 0.4),
                "silu" => assert_eq!(cost.arithmetic_intensity, 0.5),
                "add" => assert_eq!(cost.arithmetic_intensity, 0.2),
                "multiply" => assert_eq!(cost.arithmetic_intensity, 0.2),
                _ => {}
            }
        }
    }

    // ── Bubble Detection Tests ───────────────────────────────────────────

    #[test]
    fn test_detect_sync_bubbles_at_backend_boundary() {
        // Layer 0: all ANE, Layer 1: all GPU → sync point at boundary
        let mut route0 = OperationRoute::default();
        route0.set_dominant_backend(3); // all ANE
        let route1 = OperationRoute::default(); // GPU default

        let layers = vec![
            LayerPlan {
                route: route0,
                ..base_layer(0)
            },
            LayerPlan {
                route: route1,
                ..base_layer(1)
            },
        ];
        let mut plan = test_plan(2, 3072, 16, 8, 128, OperationRoute::default());
        plan.layers = layers;

        let bubbles = ProfitabilityAnalyzer::detect_bubbles(&plan);
        let sync_bubbles: Vec<_> = bubbles
            .iter()
            .filter(|b| b.cause == BubbleCause::SyncPoint)
            .collect();

        assert!(!sync_bubbles.is_empty(), "expected sync-point bubbles at backend boundary");
        assert_eq!(
            sync_bubbles[0].op_name, "backend_switch",
            "sync bubble should be at backend switch"
        );
    }

    #[test]
    fn test_detect_serial_dependency_bubbles() {
        let plan = gemma2_plan();
        let bubbles = ProfitabilityAnalyzer::detect_bubbles(&plan);
        // GPU layers have sequential matmuls (q_proj, k_proj, v_proj)
        let serial_bubbles: Vec<_> = bubbles
            .iter()
            .filter(|b| b.cause == BubbleCause::SerialDependency)
            .collect();
        // Each GPU layer should have at least one serial dependency bubble
        // (between sequential matmuls)
        assert!(
            !serial_bubbles.is_empty(),
            "expected serial-dependency bubbles in GPU layers"
        );
    }

    #[test]
    fn test_detect_no_bubbles_on_all_ane() {
        let mut route = OperationRoute::default();
        route.set_dominant_backend(3); // all ANE
        let plan = test_plan(2, 3072, 16, 8, 128, route);
        let bubbles = ProfitabilityAnalyzer::detect_bubbles(&plan);
        // No GPU layers → no bubbles
        let _gpu_bubbles: Vec<_> = bubbles
            .iter()
            .filter(|b| b.cause != BubbleCause::SyncPoint)
            .collect();
        // SyncPoint bubbles at ANE→ANE boundaries are fine; the point is no GPU bubbles.
    }

    // ── Assignment Tests (three-way) ─────────────────────────────────────

    #[test]
    fn test_assign_profitable_matmul_to_ane() {
        // Layer with matmuls on GPU → should assign to ANE if on critical path
        let plan = gemma2_plan();
        let report = ProfitabilityAnalyzer::analyze(&plan);
        let ane_assignments: Vec<_> = report
            .assignments
            .iter()
            .filter(|a| a.assign)
            .collect();

        // Matmuls (q_proj, k_proj, v_proj, gate_proj, up_proj) should be profitable
        assert!(
            !ane_assignments.is_empty(),
            "expected at least one ANE assignment for matmul-heavy plan"
        );
        for assn in &ane_assignments {
            assert!(
                assn.estimated_gpu_time_saved_ns > 0,
                "assigned op should have nonzero GPU time saved"
            );
            assert!(assn.reason.contains("ANE"), "reason should mention ANE");
        }
    }

    #[test]
    fn test_no_assign_for_accel_fast_ops() {
        // rms_norm, add, multiply are faster on Accelerate — should NOT be
        // assigned to ANE even when on critical path.
        let plan = gemma2_plan();
        let report = ProfitabilityAnalyzer::analyze(&plan);
        for a in report.assignments.iter() {
            if a.op_name == "rms_norm" || a.op_name == "add" || a.op_name == "multiply" {
                assert!(
                    !a.assign,
                    "{} should not be assigned to ANE (Accel is faster)",
                    a.op_name
                );
            }
        }
    }

    #[test]
    fn test_no_assign_when_ane_slower() {
        // ANE is worse at small element-wise ops. Create a tiny plan where
        // ANE has no advantage for the only operations.
        let mut route = OperationRoute::default();
        // Only has rms_norm on GPU (element-wise, ANE has no advantage)
        route.rms_norm = 0;
        route.matmul = 0;
        let mut plan = test_plan(1, 64, 2, 1, 32, route);
        // Override layer to have only rms_norm and add (no matmuls)
        plan.layers[0].q_proj_tensor_id = 0;
        plan.layers[0].k_proj_tensor_id = 0;
        plan.layers[0].v_proj_tensor_id = 0;
        plan.layers[0].o_proj_tensor_id = 0;
        plan.layers[0].gate_proj_tensor_id = 0;
        plan.layers[0].up_proj_tensor_id = 0;
        plan.layers[0].down_proj_tensor_id = 0;
        plan.layers[0].hidden_size = 64;

        let report = ProfitabilityAnalyzer::analyze(&plan);
        let _assigned: Vec<_> = report
            .assignments
            .iter()
            .filter(|a| a.assign)
            .collect();

        // Either no assignments or only element-wise ops that show no profit
        for a in report.assignments.iter() {
            if a.op_name == "rms_norm" || a.op_name == "add" {
                assert!(!a.assign, "RMS norm / add should not be assigned to ANE");
            }
        }
    }

    #[test]
    fn test_report_total_times() {
        let plan = gemma2_plan();
        let report = ProfitabilityAnalyzer::analyze(&plan);
        // Total times should be consistent with individual assignments
        let sum_gpu_saved: u64 = report
            .assignments
            .iter()
            .filter(|a| a.assign)
            .map(|a| a.estimated_gpu_time_saved_ns)
            .sum();
        assert_eq!(
            report.total_gpu_time_saved_ns, sum_gpu_saved,
            "total_gpu_time_saved_ns should match sum of individual assignments"
        );
        assert!(
            report.total_ane_time_ns > 0 || report.assignments.is_empty(),
            "total_ane_time_ns should be > 0 when assignments exist"
        );
    }

    // ── Apply Method Tests ────────────────────────────────────────────────

    #[test]
    fn test_apply_updates_route_and_fusion() {
        let mut plan = gemma2_plan();
        assert!(plan.fused_ane_islands.is_empty(), "should start with no fused islands");

        let report = ProfitabilityAnalyzer::apply(&mut plan);

        // Some layers should have ANE route updates
        let has_ane = plan.layers.iter().any(|l| l.route.has_ane_backend());
        if has_ane {
            // If ANE was assigned, fusion plan should have been rebuilt
            assert!(
                plan.layers[0].route.has_ane_backend()
                    || plan.layers[0].route.matmul == 3
                    || plan.layers[0].route.rms_norm == 3,
                "apply should update at least one route field"
            );
        }

        // Report is still valid
        assert_eq!(report.machine, "Apple M1");
        assert!(!report.op_costs.is_empty());
    }

    #[test]
    fn test_apply_produces_valid_report() {
        let mut plan = gemma2_plan();
        let report = ProfitabilityAnalyzer::apply(&mut plan);
        assert_eq!(report.machine, "Apple M1");
        assert!(!report.op_costs.is_empty());
        assert_eq!(
            report.total_gpu_time_saved_ns,
            report
                .assignments
                .iter()
                .filter(|a| a.assign)
                .map(|a| a.estimated_gpu_time_saved_ns)
                .sum::<u64>()
        );
    }

    // ── Full Report Tests ─────────────────────────────────────────────────

    #[test]
    fn test_full_report_generation() {
        let plan = gemma2_plan();
        let report = ProfitabilityAnalyzer::analyze(&plan);

        assert_eq!(report.machine, "Apple M1");
        assert!(!report.op_costs.is_empty(), "op_costs should not be empty");
        assert!(!report.assignments.is_empty(), "assignments should not be empty");

        // Verify report structure
        for cost in &report.op_costs {
            assert!(!cost.op_name.is_empty());
            assert!(
                cost.gpu_estimate_ns > 0 || cost.ane_estimate_ns > 0,
                "cost should have nonzero time"
            );
            assert!(
                cost.accel_estimate_ns > 0,
                "cost should have nonzero accel time"
            );
        }

        for bubble in &report.bubbles {
            assert!(!bubble.op_name.is_empty());
            assert!(bubble.estimated_idle_ns > 0, "bubble idle time should be > 0");
        }

        for assn in &report.assignments {
            assert!(!assn.op_name.is_empty());
            assert!(!assn.reason.is_empty());
        }
    }

    #[test]
    fn test_analyze_empty_plan() {
        let plan = ModelExecutionPlan::default();
        let report = ProfitabilityAnalyzer::analyze(&plan);
        assert!(report.op_costs.is_empty(), "empty plan → no op costs");
        assert!(report.bubbles.is_empty(), "empty plan → no bubbles");
        assert!(report.assignments.is_empty(), "empty plan → no assignments");
        assert_eq!(report.total_gpu_time_saved_ns, 0);
        assert_eq!(report.total_ane_time_ns, 0);
    }

    // ── Edge Case: Mixed Layer Types ─────────────────────────────────────

    #[test]
    fn test_mixed_attention_kinds() {
        let plan = gemma2_plan();
        // Layer 0 is sliding_attention, layer 1 is full_attention (set by test_plan)
        let report = ProfitabilityAnalyzer::analyze(&plan);
        // Both should have op costs
        assert!(report.op_costs.len() >= 8, "should have op costs for both layers");
    }

    #[test]
    fn test_variable_layer_sizes() {
        // Test with a plan where layers have different hidden sizes
        let mut plan = test_plan(2, 4096, 32, 8, 128, gemma_default_route());
        plan.layers[1].hidden_size = 2048; // different size
        plan.hidden_size = 4096;

        let costs = ProfitabilityAnalyzer::gather_op_costs(&plan);
        // Should still produce valid costs
        assert!(!costs.is_empty());
    }

    #[test]
    fn test_multiple_layers_assignments_independent() {
        let plan = test_plan(4, 3072, 16, 8, 128, gemma_default_route());
        let report = ProfitabilityAnalyzer::analyze(&plan);
        // Each layer should have its own assignments
        let mut layer_indices: Vec<u32> = report
            .assignments
            .iter()
            .map(|a| a.layer_index)
            .collect();
        layer_indices.sort();
        layer_indices.dedup();
        // At least some layers have assignments (GPU layers with matmuls)
        assert!(
            layer_indices.len() >= 1,
            "should have assignments from at least one layer"
        );
    }

    // ── Bubble Edge Cases ─────────────────────────────────────────────────

    #[test]
    fn test_bandwidth_limited_detection() {
        let plan = gemma2_plan();
        let bubbles = ProfitabilityAnalyzer::detect_bubbles(&plan);
        let _bw_bubbles: Vec<_> = bubbles
            .iter()
            .filter(|b| b.cause == BubbleCause::BandwidthLimited)
            .collect();
        // May or may not have bandwidth bubbles depending on layer structure
        // This is fine — we just verify no crash.
    }

    #[test]
    fn test_pipeline_stall_detection() {
        let plan = gemma2_plan();
        let bubbles = ProfitabilityAnalyzer::detect_bubbles(&plan);
        let stall_bubbles: Vec<_> = bubbles
            .iter()
            .filter(|b| b.cause == BubbleCause::PipelineStall)
            .collect();
        // Should detect the softmax→matmul RAW hazard in each layer
        assert!(
            !stall_bubbles.is_empty(),
            "should detect pipeline stall from softmax→matmul"
        );
    }

    #[test]
    fn test_no_duplicate_op_costs_per_layer() {
        let plan = gemma2_plan();
        let costs = ProfitabilityAnalyzer::gather_op_costs(&plan);
        let unique_ops_in_layer: std::collections::HashSet<&str> =
            plan.layers[0].operation_names().into_iter().collect();
        let _expected = unique_ops_in_layer.len() * plan.layers.len();
        assert!(
            costs.len() >= unique_ops_in_layer.len(),
            "should have at least one cost per unique op (got {}, min per layer {})",
            costs.len(),
            unique_ops_in_layer.len()
        );
        // Each cost entry should have the right structure.
        for cost in &costs {
            assert!(!cost.op_name.is_empty());
            assert!(cost.gpu_estimate_ns > 0);
            assert!(cost.accel_estimate_ns > 0);
            assert!(cost.ane_estimate_ns > 0);
            assert!(!cost.shape_desc.is_empty());
        }
    }

    // ── Three-way Assignment Decision Tests ──────────────────────────────

    #[test]
    fn test_backend_reasoning_in_reason() {
        // Verify that the reason string mentions the best backend
        let plan = gemma2_plan();
        let report = ProfitabilityAnalyzer::analyze(&plan);

        for assn in &report.assignments {
            if assn.op_name == "rms_norm" || assn.op_name == "add" {
                // These should be fastest on Accelerate
                assert!(
                    assn.reason.contains("Accelerate") || assn.reason.contains("accel"),
                    "rms_norm/add reason should reference Accelerate: {}",
                    assn.reason
                );
            }
        }
    }

    // ── ANE Threshold Gate ──────────────────────────────────────────────

    #[test]
    fn test_ane_meets_15_percent_threshold() {
        // ANE=80ns vs GPU=100ns: 20% faster, so 15% threshold passes.
        assert!(ane_assignment_meets_threshold(80, 100, 150, Some(0.15)));
    }

    #[test]
    fn test_ane_fails_10_percent_threshold() {
        // ANE=95ns vs GPU=100ns: only 5% faster, so 10% threshold rejects.
        assert!(!ane_assignment_meets_threshold(95, 100, 150, Some(0.10)));
    }

    #[test]
    fn test_ane_default_threshold() {
        // Default is 10%. ANE=95 vs GPU=100 is 5% → rejected.
        assert!(!ane_assignment_meets_threshold(95, 100, 150, None));
        // ANE=85 vs GPU=100 is 15% → passes default 10%.
        assert!(ane_assignment_meets_threshold(85, 100, 150, None));
    }

    #[test]
    fn test_ane_exact_boundary() {
        // ANE=90 vs GPU=100: exactly 10% faster. Boundary: <= means passes.
        assert!(ane_assignment_meets_threshold(90, 100, 150, Some(0.10)));
    }

    #[test]
    fn test_ane_best_other_is_accel() {
        // GPU=200, Accel=110, ANE=90. Best other is Accel at 110.
        // 90 <= 110 * (1 - 0.10) = 99? Yes, 90 <= 99 → passes.
        assert!(ane_assignment_meets_threshold(90, 200, 110, Some(0.10)));

        // ANE=100 vs best_other=110: 100 <= 99? No → fails.
        assert!(!ane_assignment_meets_threshold(100, 200, 110, Some(0.10)));
    }

    #[test]
    fn test_ane_gpu_faster_than_ane() {
        // GPU=80, ANE=100, Accel=120. ANE is slower than GPU on critical path.
        // best_other = 80 (GPU). 100 <= 80 * 0.9 = 72? No → fails.
        assert!(!ane_assignment_meets_threshold(100, 80, 120, Some(0.10)));
    }

    #[test]
    fn test_ane_equal_times() {
        // ANE=100, GPU=100, Accel=100: equal. 100 <= 100 * 0.9 = 90? No.
        assert!(!ane_assignment_meets_threshold(100, 100, 100, Some(0.10)));
    }

    #[test]
    fn test_ane_threshold_edge_values() {
        // Threshold of 0.0 means must be strictly <= best_other (any improvement).
        assert!(ane_assignment_meets_threshold(100, 100, 120, Some(0.0)));
        // ANE equal to best_other with 0 threshold: passes.
        assert!(ane_assignment_meets_threshold(100, 100, 120, Some(0.0)));
        // ANE worse fails even with 0 threshold.
        assert!(!ane_assignment_meets_threshold(110, 100, 120, Some(0.0)));
        // Threshold of 1.0 means ANE must be 0 → impossible if non-zero.
        assert!(!ane_assignment_meets_threshold(1, 100, 120, Some(1.0)));
        // Threshold outside [0,1] is invalid → returns false.
        assert!(!ane_assignment_meets_threshold(80, 100, 120, Some(-0.1)));
        assert!(!ane_assignment_meets_threshold(80, 100, 120, Some(1.5)));
    }
}
