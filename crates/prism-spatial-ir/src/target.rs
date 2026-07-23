//! The [`SpatialTarget`] trait — the interface every hardware target implements.
//!
//! Every physical target (Apple Silicon M1, AMD Strix Halo, SambaNova, etc.)
//! provides an implementation of this trait, making the search framework and
//! legalization passes reusable across backends.

use crate::calibration_report::M1CalibrationReport;
use crate::cost::{CalibratedCostModel, CostEstimate, CostModel};
use crate::execution_plan::{lower_to_manifest, ExecutionPlan};
use crate::graph::{SpatialGraph, TileGeometry};
use crate::legalize::{LegalizationError, LegalizedGraph};
use crate::mutation::MutationOp;
use prism_ecs_ir::evolution::compile_plan::FormatPlan;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// TargetCapabilities
// ---------------------------------------------------------------------------

/// Declares the scheduling and concurrency capabilities of a hardware target.
///
/// The legalizer and mutation engine use these flags to determine which graph
/// transformations are valid for a given target.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TargetCapabilities {
    /// If true, the target requires all operations to execute sequentially —
    /// no pipelining or concurrency between nodes.
    pub sequential_schedules: bool,
    /// If true, the target supports concurrent execution across different
    /// compute domains (e.g., GPU + CPU simultaneously).
    pub cross_domain_concurrency: bool,
    /// If true, the target allows GPU and ANE execution to overlap.
    pub gpu_ane_overlap: bool,
    /// If true, the target supports pipeline overlap (e.g., compute while
    /// transfer).
    pub pipeline_overlap: bool,
    /// Maximum number of concurrent compute regions the target supports.
    pub max_concurrent_regions: usize,
    /// Maximum memory in bytes available for model weights.
    pub max_weight_memory_bytes: u64,
    /// Maximum memory in bytes available for activations / scratch.
    pub max_scratch_memory_bytes: u64,
    /// Whether the target supports KV cache in a compressed representation.
    pub supports_compressed_kv_cache: bool,
    /// Whether the target supports multiple GPU devices with tensor sharding.
    pub supports_multi_gpu: bool,
}

impl TargetCapabilities {
    /// Returns the default capabilities for a sequential-only target
    /// (e.g., Apple Silicon M1 single-accelerator constraint).
    pub fn sequential_only() -> Self {
        Self {
            sequential_schedules: true,
            cross_domain_concurrency: false,
            gpu_ane_overlap: false,
            pipeline_overlap: false,
            max_concurrent_regions: 1,
            max_weight_memory_bytes: 6 * 1024 * 1024 * 1024, // 6 GiB
            max_scratch_memory_bytes: 2 * 1024 * 1024 * 1024, // 2 GiB
            supports_compressed_kv_cache: true,
            supports_multi_gpu: false,
        }
    }

    /// Returns capabilities for a target that supports full GPU+CPU concurrency.
    pub fn concurrent_gpu_cpu() -> Self {
        Self {
            sequential_schedules: false,
            cross_domain_concurrency: true,
            gpu_ane_overlap: false,
            pipeline_overlap: true,
            max_concurrent_regions: 4,
            max_weight_memory_bytes: 16 * 1024 * 1024 * 1024, // 16 GiB
            max_scratch_memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GiB
            supports_compressed_kv_cache: true,
            supports_multi_gpu: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Calibration ID
// ---------------------------------------------------------------------------

/// Identifier for a specific calibration report.
///
/// Calibration measurements are machine-specific and versioned. A
/// [`SpatialCompilationPlan`] carries the calibration ID that produced its
/// cost estimates.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CalibrationId(pub String);

// ---------------------------------------------------------------------------
// SpatialTarget trait
// ---------------------------------------------------------------------------

/// Trait implemented by every hardware target in the SpatialIR framework.
///
/// # Type parameters
///
/// * `Calibration` — the calibration report type produced by the target.
/// * `Artifact` — the final compiled artifact type (Metal pipeline state,
///   Core ML model, etc.).
///
/// # Required methods
///
/// * `legalize` — three-tier legality check (semantic, backend, operational).
/// * `estimate` — fast cost estimate for a legalized graph.
/// * `lower` — lower a legalized graph to a target-native artifact.
pub trait SpatialTarget {
    /// Calibration report type for this target.
    type Calibration;
    /// Target-native artifact type after lowering.
    type Artifact;

    /// Three-tier legality check.
    ///
    /// Returns a [`LegalizedGraph`] on success, or a vector of
    /// [`LegalizationError`]s describing every violation found.
    fn legalize(&self, graph: &SpatialGraph) -> Result<LegalizedGraph, Vec<LegalizationError>>;

    /// Fast cost estimate for a legalized graph.
    ///
    /// Called thousands of times per evolutionary search — must be cheap.
    fn estimate(&self, graph: &LegalizedGraph) -> CostEstimate;

    /// Lower a legalized graph to a target-native artifact.
    fn lower(&self, graph: &LegalizedGraph) -> Result<Self::Artifact, LoweringError>;

    /// Lower a legalized graph to a [`KernelManifest`] with batch and realtime
    /// execution plans.
    ///
    /// The default implementation calls [`lower_to_manifest`] on the inner
    /// spatial graph. Targets with specialized lowering (e.g. custom
    /// dispatch optimizations per mode) override this method.
    ///
    /// Returns `LoweringError::BackendError` when the graph contains a cycle
    /// and cannot be topologically sorted.
    fn lower_to_manifest(
        &self,
        graph: &LegalizedGraph,
        cost: CostEstimate,
        format_plan: Option<&FormatPlan>,
    ) -> Result<KernelManifest, LoweringError> {
        let inner = graph.graph();
        lower_to_manifest(inner, cost, format_plan).ok_or_else(|| {
            LoweringError::BackendError(
                "graph contains a cycle: cannot produce execution plans".into(),
            )
        })
    }

    /// Returns the set of mutations available for this target given the
    /// graph's current state.
    fn available_mutations(&self, _graph: &LegalizedGraph) -> Vec<MutationOp> {
        // Default: return all mutation types. Targets override to restrict.
        vec![
            MutationOp::ChangeCodec {
                node_id: crate::graph::SpatialNodeId(0),
                new_codec: crate::cost::CodecVariant::Fp16,
            },
            MutationOp::ChangePlacement {
                node_id: crate::graph::SpatialNodeId(0),
                new_unit: crate::hardware::VirtualComputeUnit::GpuComputeRegion,
            },
            MutationOp::FuseNodes {
                first: crate::graph::SpatialNodeId(0),
                second: crate::graph::SpatialNodeId(0),
            },
            MutationOp::SplitNode {
                node_id: crate::graph::SpatialNodeId(0),
                split_point: 0,
            },
            MutationOp::ChangeTileGeometry {
                node_id: crate::graph::SpatialNodeId(0),
                new_tile_x: 1,
                new_tile_y: 1,
            },
            MutationOp::ChangeMemoryPolicy {
                node_id: crate::graph::SpatialNodeId(0),
                new_region: crate::hardware::VirtualMemoryRegion::UnifiedMemory,
            },
            MutationOp::ChangeKVCachePolicy {
                node_id: crate::graph::SpatialNodeId(0),
                compressed: false,
                bit_width: 16,
            },
        ]
    }

    /// Returns this target's capabilities declaration.
    fn capabilities(&self) -> TargetCapabilities;
}

// ---------------------------------------------------------------------------
// LoweringError
// ---------------------------------------------------------------------------

/// Error returned when lowering a legalized graph to a target-native artifact
/// fails.
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
pub enum LoweringError {
    /// No backend implementation available for the given compute kind.
    #[error("no backend for compute kind: {0}")]
    NoBackendForComputeKind(String),
    /// A required shape contract could not be satisfied by the target.
    #[error("shape contract not satisfiable: {0}")]
    ShapeContractNotSatisfiable(String),
    /// Memory budget exceeded for this target.
    #[error("memory budget exceeded: {needed} > {available}")]
    MemoryBudgetExceeded {
        /// Memory required by the graph.
        needed: u64,
        /// Memory available on the target.
        available: u64,
    },
    /// Backend-specific lowering failure.
    #[error("backend error: {0}")]
    BackendError(String),
}

// ---------------------------------------------------------------------------
// KernelDescriptor / KernelManifest
// ---------------------------------------------------------------------------

/// Describes a single kernel invocation produced by lowering.
///
/// Every compute node in the spatial graph produces one kernel descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelDescriptor {
    /// The spatial node ID this kernel was lowered from.
    pub node_id: crate::graph::SpatialNodeId,
    /// Assigned codec variant (inherited from annotations).
    pub codec: Option<crate::cost::CodecVariant>,
    /// Tile geometry for dispatch.
    pub tile_geometry: Option<TileGeometry>,
    /// Threadgroup size (total threads).
    pub threadgroup_size: u32,
    /// Execution order index (topological position).
    pub schedule_index: usize,
}

/// The lowered artifact of a legalized spatial graph.
///
/// Contains kernel descriptors for every compute node plus separate
/// [`ExecutionPlan`]s for batch and realtime execution modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelManifest {
    /// Per-node kernel descriptors in execution order.
    pub kernels: Vec<KernelDescriptor>,
    /// Total estimated compute cost.
    pub total_cost: CostEstimate,
    /// Batch execution plan (GEMM-oriented, multi-token).
    pub batch_plan: Option<ExecutionPlan>,
    /// Realtime execution plan (GEMV-oriented, single-token).
    pub realtime_plan: Option<ExecutionPlan>,
}

// ---------------------------------------------------------------------------
// AppleSiliconTarget
// ---------------------------------------------------------------------------

/// A concrete [`SpatialTarget`] implementation for Apple Silicon (M-series) hardware.
///
/// Wraps an [`M1CalibrationReport`] for calibrated cost estimation, validates
/// graphs against Metal-specific constraints via [`metal_specific_checks`], and
/// lowers legalized graphs toward Metal pipeline state artifacts.
///
/// # Type parameters
///
/// * `'a` — lifetime of the borrowed calibration report.
pub struct AppleSiliconTarget<'a> {
    /// Calibration report for this specific Apple Silicon machine.
    pub calibration: &'a M1CalibrationReport,
    /// Hardware model identifier, e.g. "Mac14,2".
    pub target_model: &'a str,
}

impl SpatialTarget for AppleSiliconTarget<'_> {
    type Calibration = M1CalibrationReport;
    type Artifact = Vec<u8>;

    fn legalize(&self, graph: &SpatialGraph) -> Result<LegalizedGraph, Vec<LegalizationError>> {
        let graph_for_closure = graph.clone();
        // Clone the input graph so the backend-check closure can reference a
        // copy while the original is consumed by legalize().
        crate::legalize::legalize(graph.clone(), |node| {
            crate::legalize::metal_specific_checks(node, &graph_for_closure)
        })
    }

    fn estimate(&self, graph: &LegalizedGraph) -> CostEstimate {
        let model = CalibratedCostModel::new(self.calibration);
        model.estimate(graph.graph())
    }

    fn lower(&self, _graph: &LegalizedGraph) -> Result<Self::Artifact, LoweringError> {
        // Lower the legalized graph to a KernelManifest via the trait's
        // lower_to_manifest() default implementation, then serialize.
        let manifest = self.lower_to_manifest(_graph, self.estimate(_graph), None)?;
        bincode::serialize(&manifest)
            .map_err(|e| LoweringError::BackendError(format!("bincode serialization: {e}")))
    }

    fn capabilities(&self) -> TargetCapabilities {
        // Apple Silicon M1 is a single-accelerator design (sequential
        // schedules, unified memory, limited concurrency).
        TargetCapabilities::sequential_only()
    }
}

impl<'a> AppleSiliconTarget<'a> {
    /// Create a new Apple Silicon target from a calibration report and model
    /// identifier.
    pub fn new(calibration: &'a M1CalibrationReport, target_model: &'a str) -> Self {
        Self {
            calibration,
            target_model,
        }
    }

    /// Returns the calibration ID derived from this target's report digest.
    pub fn calibration_id(&self) -> CalibrationId {
        CalibrationId(self.calibration.digest())
    }
}

// ---------------------------------------------------------------------------
// Hardware probing
// ---------------------------------------------------------------------------

/// Probe the host machine for Apple Silicon identity.
///
/// Returns `(hardware_model, metal_device_name)`:
///
/// * `hardware_model` — from `sysctl hw.model` (e.g. `"Mac14,2"`).
/// * `metal_device_name` — parsed from `system_profiler SPHardwareDataType`
///   (e.g. `"Apple M1 Pro"`), falling back to `"Apple Silicon"` when the
///   profiler is unavailable.
///
/// This function is intended for use at tool / daemon startup to produce a
/// default [`M1CalibrationReport`].  It makes no Metal or IOKit calls, so it
/// is safe to call from this crate.
pub fn probe_apple_silicon() -> (String, String) {
    let hardware_model = std::process::Command::new("sysctl")
        .arg("-n")
        .arg("hw.model")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "Unknown".to_string());

    let metal_device_name = std::process::Command::new("system_profiler")
        .arg("SPHardwareDataType")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                let stdout = String::from_utf8(o.stdout).ok()?;
                for line in stdout.lines() {
                    let trimmed = line.trim();
                    if let Some(chip) = trimmed.strip_prefix("Chip: ") {
                        return Some(chip.to_string());
                    }
                }
            }
            None
        })
        .unwrap_or_else(|| "Apple Silicon".to_string());

    (hardware_model, metal_device_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration_report::M1CalibrationReport;

    #[test]
    fn test_apple_silicon_target_basic() {
        let report = M1CalibrationReport::plausible_defaults();
        let target = AppleSiliconTarget::new(&report, "Mac14,2");

        assert_eq!(target.target_model, "Mac14,2");
        assert_eq!(target.calibration.hardware_model, "Mac14,2");
        assert_eq!(target.calibration.metal_device, "Apple M1");

        let cid = target.calibration_id();
        assert!(!cid.0.is_empty());
    }

    #[test]
    fn test_apple_silicon_capabilities() {
        let report = M1CalibrationReport::plausible_defaults();
        let target = AppleSiliconTarget::new(&report, "Mac14,2");
        let caps = target.capabilities();

        assert!(caps.sequential_schedules);
        assert!(!caps.cross_domain_concurrency);
        assert!(!caps.gpu_ane_overlap);
        assert!(caps.supports_compressed_kv_cache);
    }

    #[test]
    fn test_probe_apple_silicon_runs() {
        // This test exercises the probe function.  It will return whatever
        // the host machine provides; we simply verify it returns non-empty
        // strings without panicking.
        let (model, device) = probe_apple_silicon();
        assert!(!model.is_empty(), "hardware model should not be empty");
        assert!(!device.is_empty(), "device name should not be empty");
    }
}
