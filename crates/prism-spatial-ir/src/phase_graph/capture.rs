//! This module owns the canonical authority for the capture-plan surface:
//! the `LoweredKernel`, `CapturePlan`, `TinyJitCache`, and `CaptureExecutor`
//! types, plus the `CapturePlan` impl that validates a plan, computes its
//! digest, and dispatches a plan to a `CaptureExecutor`.
//! It does not own graph mutation, kernel rendering, or kernel-group
//! enumeration.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::phase_graph::graph::{GraphError, TinyGraph};
use crate::phase_graph::kernel_group::KernelGroup;
use crate::phase_graph::kernel_op::LoweringTarget;
use crate::phase_graph::plan::{ExecutionReceipt, MemoryPlan, ReplayPlan};
use crate::phase_graph::render::hex_digest;
use crate::phase_graph::uop::UOpId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoweredKernel {
    pub group: KernelGroup,
    pub source: String,
    /// Exact output element count retained for generic elementwise ABI
    /// construction. Older serialized captures may omit it.
    #[serde(default)]
    pub output_elements: Option<usize>,
    /// Digest of deterministic target source, suitable for compiler provenance.
    pub source_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapturePlan {
    pub target: LoweringTarget,
    pub kernels: Vec<LoweredKernel>,
    pub graph_op_count: usize,
    pub replay: ReplayPlan,
    pub graph: TinyGraph,
    pub memory_plan: MemoryPlan,
}

/// TinyJIT-style capture cache. The first invocation lowers and validates a
/// graph; subsequent invocations with the same graph digest and target reuse
/// the immutable command sequence and kernel payloads.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct TinyJitCache {
    captures: BTreeMap<String, CapturePlan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TinyJitArchive {
    version: u32,
    captures: BTreeMap<String, TinyJitArchiveEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TinyJitArchiveEntry {
    capture: CapturePlan,
    identity_digest: String,
}

impl TinyJitCache {
    fn key(
        graph: &TinyGraph,
        target: LoweringTarget,
        strategy: Option<&crate::fused_ops::FusionStrategy>,
    ) -> Result<String, GraphError> {
        let optimized = graph.optimize()?;
        let bytes = serde_json::to_vec(&(target, strategy, optimized))
            .map_err(|error| GraphError::Serialization(error.to_string()))?;
        let mut digest = Sha256::new();
        digest.update(bytes);
        Ok(hex_digest(digest.finalize()))
    }

    pub fn capture(
        &mut self,
        graph: &TinyGraph,
        target: LoweringTarget,
    ) -> Result<(String, bool), GraphError> {
        self.capture_with_strategy(
            graph,
            target,
            &crate::fused_ops::FusionStrategy::StandardFused,
        )
    }

    /// Capture and cache a strategy-specific executable layout. Strategy is
    /// part of the identity so a per-operation or interleaved capture can
    /// never reuse the standard fused command sequence.
    pub fn capture_with_strategy(
        &mut self,
        graph: &TinyGraph,
        target: LoweringTarget,
        strategy: &crate::fused_ops::FusionStrategy,
    ) -> Result<(String, bool), GraphError> {
        let key = Self::key(graph, target, Some(strategy))?;
        if !self.captures.contains_key(&key) {
            self.captures.insert(
                key.clone(),
                graph.lower_with_fusion_strategy(target, strategy)?,
            );
            Ok((key, true))
        } else {
            Ok((key, false))
        }
    }

    /// Materialize a set of strategy alternatives in the same cache. The
    /// returned entries preserve caller order and include the cache-hit bit,
    /// allowing a workload calibrator to compile each executable alternative
    /// once and then benchmark or replay it without rebuilding the graph.
    pub fn capture_strategies(
        &mut self,
        graph: &TinyGraph,
        target: LoweringTarget,
        strategies: &[crate::fused_ops::FusionStrategy],
    ) -> Result<Vec<(crate::fused_ops::FusionStrategy, String, bool)>, GraphError> {
        for (index, strategy) in strategies.iter().enumerate() {
            if strategies[..index].contains(strategy) {
                return Err(GraphError::Serialization(format!(
                    "duplicate fusion strategy at index {index}: {strategy:?}"
                )));
            }
        }
        let mut captures = Vec::with_capacity(strategies.len());
        for strategy in strategies {
            let (key, inserted) = self.capture_with_strategy(graph, target, strategy)?;
            captures.push((strategy.clone(), key, inserted));
        }
        Ok(captures)
    }

    /// Serialize the immutable TinyJIT command cache for durable artifact
    /// storage. Captures are validated before encoding so malformed state
    /// cannot be published as a reusable executable cache.
    pub fn export_bytes(&self) -> Result<Vec<u8>, GraphError> {
        let mut captures = BTreeMap::new();
        for (key, capture) in &self.captures {
            capture
                .validate()
                .map_err(|error| GraphError::Serialization(format!("capture '{key}': {error}")))?;
            let bytes = serde_json::to_vec(&(key, capture))
                .map_err(|error| GraphError::Serialization(error.to_string()))?;
            let mut digest = Sha256::new();
            digest.update(bytes);
            captures.insert(
                key.clone(),
                TinyJitArchiveEntry {
                    capture: capture.clone(),
                    identity_digest: hex_digest(digest.finalize()),
                },
            );
        }
        serde_json::to_vec(&TinyJitArchive {
            version: 1,
            captures,
        })
        .map_err(|error| GraphError::Serialization(error.to_string()))
    }

    /// Import a persisted TinyJIT cache and re-run capture validation before
    /// making any entry available for replay.
    pub fn import_bytes(bytes: &[u8]) -> Result<Self, GraphError> {
        let archive: TinyJitArchive = serde_json::from_slice(bytes)
            .map_err(|error| GraphError::Serialization(error.to_string()))?;
        if archive.version != 1 {
            return Err(GraphError::Serialization(format!(
                "unsupported TinyJIT archive version {}",
                archive.version
            )));
        }
        let mut captures = BTreeMap::new();
        for (key, entry) in archive.captures {
            let bytes = serde_json::to_vec(&(key.clone(), &entry.capture))
                .map_err(|error| GraphError::Serialization(error.to_string()))?;
            let mut digest = Sha256::new();
            digest.update(bytes);
            if hex_digest(digest.finalize()) != entry.identity_digest {
                return Err(GraphError::Serialization(format!(
                    "TinyJIT capture '{key}' identity digest mismatch"
                )));
            }
            let capture = entry.capture;
            capture
                .validate()
                .map_err(|error| GraphError::Serialization(format!("capture '{key}': {error}")))?;
            captures.insert(key, capture);
        }
        Ok(Self { captures })
    }

    pub fn get(&self, key: &str) -> Option<&CapturePlan> {
        self.captures.get(key)
    }

    /// Evict one captured command sequence so a changed lowering policy or
    /// target capability cannot reuse stale TinyJIT state.
    pub fn invalidate(&mut self, key: &str) -> bool {
        self.captures.remove(key).is_some()
    }

    /// Evict every captured command sequence while preserving the cache
    /// object for reuse by a long-lived compiler session.
    pub fn clear(&mut self) {
        self.captures.clear();
    }

    pub fn replay<E: CaptureExecutor>(
        &self,
        key: &str,
        executor: &mut E,
    ) -> Result<ExecutionReceipt, String> {
        self.captures
            .get(key)
            .ok_or_else(|| format!("TinyJIT capture '{key}' is not cached"))?
            .replay(executor)
    }

    pub fn len(&self) -> usize {
        self.captures.len()
    }

    pub fn is_empty(&self) -> bool {
        self.captures.is_empty()
    }
}

pub trait CaptureExecutor {
    fn dispatch(&mut self, command_id: u32, kernel: &LoweredKernel) -> Result<(), String>;
    fn synchronize(&mut self, command_id: u32) -> Result<(), String>;

    fn dispatch_persistent(
        &mut self,
        command_ids: &[u32],
        kernels: &[LoweredKernel],
    ) -> Result<(), String> {
        for (command_id, kernel) in command_ids.iter().zip(kernels) {
            self.dispatch(*command_id, kernel)?;
        }
        Ok(())
    }

    /// Submit a persistent command sequence together with its barrier points.
    /// Device executors that can encode barriers inside a command buffer may
    /// override this hook; the compatibility implementation only supports a
    /// final barrier and rejects interior points before submission.
    fn dispatch_persistent_with_sync_points(
        &mut self,
        command_ids: &[u32],
        kernels: &[LoweredKernel],
        synchronization_points: &[u32],
    ) -> Result<(), String> {
        if let Some(last_command) = command_ids.last() {
            if synchronization_points
                .iter()
                .any(|point| point != last_command)
            {
                return Err(
                    "persistent executor does not support interior synchronization points".into(),
                );
            }
        }
        self.dispatch_persistent(command_ids, kernels)?;
        if let Some(last_command) = command_ids.last() {
            if synchronization_points.contains(last_command) {
                self.synchronize(*last_command)?;
            }
        }
        Ok(())
    }
}

impl CapturePlan {
    pub fn validate(&self) -> Result<(), String> {
        self.graph
            .validate()
            .map_err(|error| format!("capture graph is invalid: {error}"))?;
        if self.graph_op_count != self.graph.ops.len() {
            return Err(format!(
                "capture graph operation count mismatch: recorded {}, embedded {}",
                self.graph_op_count,
                self.graph.ops.len()
            ));
        }
        if self.replay.command_ids.len() != self.kernels.len() {
            return Err("capture command count does not match kernel count".into());
        }
        let expected_command_ids: Vec<u32> = (0..self.kernels.len() as u32).collect();
        if self.replay.command_ids != expected_command_ids {
            return Err("capture command IDs are not canonical".into());
        }
        let graph_ids: std::collections::BTreeSet<UOpId> =
            self.graph.ops.iter().map(|op| op.id).collect();
        for kernel in &self.kernels {
            if kernel.group.ops.is_empty() {
                return Err("capture contains an empty kernel group".into());
            }
            if kernel.source.is_empty() {
                return Err(format!(
                    "capture kernel {:?} has empty rendered source",
                    kernel.group.op_ids()
                ));
            }
            if kernel
                .group
                .op_ids()
                .iter()
                .any(|id| !graph_ids.contains(id))
            {
                return Err("capture kernel references a missing graph UOp".into());
            }
            if let Some(recorded_elements) = kernel.output_elements {
                let output_id = kernel
                    .group
                    .ops
                    .last()
                    .map(KernelOp::id)
                    .ok_or_else(|| "capture kernel has no terminal UOp".to_string())?;
                let expected_elements = self
                    .graph
                    .ops
                    .iter()
                    .find(|op| op.id == output_id)
                    .and_then(|op| {
                        op.shape.iter().try_fold(1usize, |count, dimension| {
                            count.checked_mul(*dimension as usize)
                        })
                    })
                    .ok_or_else(|| "capture terminal UOp has invalid shape".to_string())?;
                if recorded_elements != expected_elements {
                    return Err(format!(
                        "capture kernel output geometry mismatch: recorded {recorded_elements}, expected {expected_elements}"
                    ));
                }
            }
            let mut digest = Sha256::new();
            digest.update(kernel.source.as_bytes());
            if kernel.source_digest != hex_digest(digest.finalize()) {
                return Err(format!(
                    "kernel {:?} source digest does not match source",
                    kernel.group.op_ids()
                ));
            }
        }
        if self
            .replay
            .command_ids
            .windows(2)
            .any(|ids| ids[0] >= ids[1])
        {
            return Err("capture command ids must be strictly increasing".into());
        }
        if self.memory_plan.slot_count
            < self
                .memory_plan
                .allocations
                .iter()
                .map(|allocation| allocation.slot)
                .max()
                .map_or(0, |slot| slot + 1)
        {
            return Err("capture memory plan references an unavailable slot".into());
        }
        if self
            .memory_plan
            .allocations
            .iter()
            .any(|allocation| allocation.first_command > allocation.last_command)
        {
            return Err("capture memory allocation has an invalid lifetime".into());
        }
        let mut allocation_ids = std::collections::BTreeSet::new();
        for allocation in &self.memory_plan.allocations {
            if !allocation_ids.insert(allocation.value) {
                return Err(format!(
                    "capture memory plan allocates UOp {:?} more than once",
                    allocation.value
                ));
            }
            let value = self
                .graph
                .ops
                .iter()
                .find(|op| op.id == allocation.value)
                .ok_or_else(|| {
                    format!(
                        "capture memory plan references missing UOp {:?}",
                        allocation.value
                    )
                })?;
            let expected_elements = value
                .shape
                .iter()
                .try_fold(1usize, |count, dimension| {
                    count.checked_mul(*dimension as usize)
                })
                .ok_or_else(|| "capture memory value shape overflows element count".to_string())?;
            if allocation.elements != expected_elements {
                return Err(format!(
                    "capture memory allocation for {:?} records {} elements; expected {}",
                    allocation.value, allocation.elements, expected_elements
                ));
            }
        }
        if self
            .replay
            .synchronization_points
            .iter()
            .any(|point| !self.replay.command_ids.contains(point))
        {
            return Err("capture synchronization point references an unknown command".into());
        }
        if self
            .replay
            .synchronization_points
            .windows(2)
            .any(|points| points[0] >= points[1])
        {
            return Err("capture synchronization points are not canonical".into());
        }
        Ok(())
    }

    pub fn digest(&self) -> String {
        let bytes = serde_json::to_vec(self)
            // WAIVER: `CapturePlan` is composed entirely of `Serialize`-derived
            // types; the only way serialization can fail is a programmer
            // error introducing a non-serializable field, which is a
            // permanent invariant violation rather than a runtime condition.
            .expect("CapturePlan must be serializable");
        let mut digest = Sha256::new();
        digest.update(bytes);
        hex_digest(digest.finalize())
    }

    /// Verify that an execution receipt still describes this exact capture.
    /// Receipts are transportable evidence, so callers must validate them
    /// before using them for promotion, calibration, or artifact provenance.
    pub fn validate_receipt(&self, receipt: &ExecutionReceipt) -> Result<(), String> {
        self.validate()
            .map_err(|error| format!("capture: {error}"))?;
        if !receipt.replayed {
            return Err("execution receipt is not marked as replayed".into());
        }
        if receipt.target != self.target {
            return Err("execution receipt target does not match capture".into());
        }
        if receipt.capture_digest != self.digest() {
            return Err("execution receipt capture digest does not match capture".into());
        }
        if receipt.command_ids != self.replay.command_ids {
            return Err("execution receipt command sequence does not match capture".into());
        }
        if receipt.persistent != self.replay.persistent {
            return Err("execution receipt persistence mode does not match capture".into());
        }
        let expected_kernel_digests = self
            .kernels
            .iter()
            .map(|kernel| kernel.source_digest.clone())
            .collect::<Vec<_>>();
        if receipt.kernel_digests != expected_kernel_digests {
            return Err("execution receipt kernel digests do not match capture".into());
        }
        Ok(())
    }

    pub fn replay<E: CaptureExecutor>(&self, executor: &mut E) -> Result<ExecutionReceipt, String> {
        self.validate()?;
        if self.replay.persistent {
            executor.dispatch_persistent_with_sync_points(
                &self.replay.command_ids,
                &self.kernels,
                &self.replay.synchronization_points,
            )?;
        } else {
            for (command_id, kernel) in self.replay.command_ids.iter().zip(&self.kernels) {
                executor.dispatch(*command_id, kernel)?;
                if self.replay.synchronization_points.contains(command_id) {
                    executor.synchronize(*command_id)?;
                }
            }
        }
        Ok(ExecutionReceipt {
            target: self.target,
            capture_digest: self.digest(),
            command_ids: self.replay.command_ids.clone(),
            kernel_digests: self
                .kernels
                .iter()
                .map(|kernel| kernel.source_digest.clone())
                .collect(),
            persistent: self.replay.persistent,
            replayed: true,
        })
    }
}

// `KernelOp::id` is a `pub(crate)` helper; this `use` brings it into scope
// for the `validate` walk above without exposing the helper to consumers.
use crate::phase_graph::kernel_op::KernelOp;
