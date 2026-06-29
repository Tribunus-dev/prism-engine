//! Routing policy and strategy — evaluation policies, execution boundary
//! plans, plan validation, research routing policies, and the deterministic
//! router trait.

use sha2::{Digest, Sha256};

use super::*;
use super::lanes::TensorTransferPlan;

// ── Evaluation policy ────────────────────────────────────────────────────

/// Cardinality of evaluation groups in a plan.
#[derive(Debug, Clone)]
pub enum EvaluationGroupCardinality {
    /// Exact number of groups known at compile time.
    Fixed(u32),
    /// One group per materialized operation (determined at plan generation).
    PerOperation,
}

/// Who controls when tensors are materialised and at what granularity.
#[derive(Debug, Clone)]
pub enum EvaluationPolicy {
    /// Preserve the backend's normal lazy behaviour.  MLX builds the full
    /// layer graph; materialisation happens at the backend's discretion.
    BackendLazy,

    /// Tribunus defines one or more explicit fusion regions.  MLX may
    /// still fuse operations inside each region, but must materialise
    /// every `materialized_output` at the region boundary.
    ExplicitRegion,

    /// Insert a materialization request after each operation.  Synchronization
    /// is controlled by the boundary's SynchronizationPolicy, not the policy itself.
    ExplicitOperation,

    /// Require completion before the next operation begins, prohibit
    /// deferred dependencies crossing the boundary, and enforce
    /// deterministic lifetime release.  Synchronization level is derived
    /// from the boundary's SynchronizationPolicy.
    Eager {
        release_inputs_after_use: bool,
        prohibit_deferred_nodes: bool,
    },
}

/// Whether a backend natively supports a given evaluation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationPolicySupport {
    Native,
    Emulated,
    Unsupported,
}

/// Qualifies which backends support which policies.
pub fn policy_support(backend: BackendId, policy: &EvaluationPolicy) -> EvaluationPolicySupport {
    match backend.0 {
        0 => match policy {
            // MLX: all lazy variants native
            EvaluationPolicy::BackendLazy => EvaluationPolicySupport::Native,
            EvaluationPolicy::ExplicitRegion => EvaluationPolicySupport::Native,
            EvaluationPolicy::ExplicitOperation => EvaluationPolicySupport::Native,
            EvaluationPolicy::Eager { .. } => EvaluationPolicySupport::Emulated,
        },
        1 => match policy {
            // Accelerate: naturally eager, lazy is unsupported
            EvaluationPolicy::BackendLazy => EvaluationPolicySupport::Unsupported,
            EvaluationPolicy::ExplicitRegion => EvaluationPolicySupport::Emulated,
            EvaluationPolicy::ExplicitOperation => EvaluationPolicySupport::Native,
            EvaluationPolicy::Eager { .. } => EvaluationPolicySupport::Native,
        },
        2 => match policy {
            // Core ML: region execution native, per-operation unsupported
            EvaluationPolicy::BackendLazy => EvaluationPolicySupport::Unsupported,
            EvaluationPolicy::ExplicitRegion => EvaluationPolicySupport::Native,
            EvaluationPolicy::ExplicitOperation => EvaluationPolicySupport::Unsupported,
            EvaluationPolicy::Eager { .. } => EvaluationPolicySupport::Unsupported,
        },
        3 => match policy {
            // Orion/ANE: compiled programs execute as fused regions
            EvaluationPolicy::BackendLazy => EvaluationPolicySupport::Unsupported,
            EvaluationPolicy::ExplicitRegion => EvaluationPolicySupport::Native,
            EvaluationPolicy::ExplicitOperation => EvaluationPolicySupport::Unsupported,
            EvaluationPolicy::Eager { .. } => EvaluationPolicySupport::Native,
        },
        _ => EvaluationPolicySupport::Unsupported,
    }
}

/// Synchronization requirement for a boundary.
#[derive(Debug, Clone, PartialEq)]
pub enum SynchronizationPolicy {
    None,
    Barrier,
    Stream,
    Device,
}

impl SynchronizationPolicy {
    pub fn is_synchronized(&self) -> bool {
        !matches!(self, SynchronizationPolicy::None)
    }
}

/// One authoritative execution boundary — the single source of truth
/// for evaluation groups, superseding both the older SynchronizationGroup
/// and EvaluationGroupPlan types.
///
/// The compiler guarantees every operation is assigned exactly once,
/// operations are topologically ordered, all materialized tensors are
/// outputs of operations within or before the group, no consumer executes
/// before its producer, and backends transitions have explicit transfer
/// plans.
#[derive(Debug, Clone)]
pub struct ExecutionBoundaryPlan {
    pub group_id: EvaluationGroupId,
    pub backend_id: BackendId,
    pub operations: Vec<OperationId>,
    pub materialized_outputs: Vec<TensorId>,
    pub policy: EvaluationPolicy,
    pub synchronization: SynchronizationPolicy,
    pub release_after: Vec<TensorId>,
    /// Canonical content digest — proves which boundary plan was executed.
    pub content_digest: Option<EvidenceDigest>,
}

/// Sealed plan — digest is mandatory.
#[derive(Debug, Clone)]
pub struct SealedExecutionBoundaryPlan {
    pub plan: ExecutionBoundaryPlan,
    pub content_digest: EvidenceDigest,
}

impl SealedExecutionBoundaryPlan {
    pub fn seal(plan: ExecutionBoundaryPlan) -> Self {
        let digest = compute_boundary_digest(&plan);
        Self {
            plan,
            content_digest: digest,
        }
    }
    pub fn verify(&self) -> bool {
        self.content_digest == compute_boundary_digest(&self.plan)
    }
}

/// Directed edge in the operation dependency graph.
#[derive(Debug, Clone)]
pub struct DependencyEdge {
    pub from: OperationId,
    pub to: OperationId,
    pub via_tensor: TensorId,
}

/// Complete context for boundary-plan validation.
#[derive(Debug, Clone)]
pub struct BoundaryValidationContext<'a> {
    pub expected_operations: &'a [OperationId],
    pub dependency_edges: &'a [DependencyEdge],
    pub transfer_plans: &'a [TensorTransferPlan],
}

// ── Plan validation ───────────────────────────────────────────────────────

/// Errors detected during boundary-plan validation.
#[derive(Debug, Clone)]
pub enum PlanValidationError {
    DuplicateOperation(OperationId),
    MissingOperation(OperationId),
    TopologicalViolation {
        before: OperationId,
        after: OperationId,
    },
    UnreferencedMaterializedOutput(TensorId),
    ConsumerBeforeProducer {
        tensor: TensorId,
        consumer: OperationId,
    },
    BackendTransitionWithoutTransfer {
        from: BackendId,
        to: BackendId,
    },
    EagerWithDeferredDependency {
        op: OperationId,
        via: TensorId,
        crosses_boundary_to: EvaluationGroupId,
    },
    UnsupportedPolicy {
        backend: BackendId,
        policy: EvaluationPolicy,
    },
    EmptyBoundary(EvaluationGroupId),
}

/// Validate boundary plans against the full operation graph, dependency
/// edges, and transfer plans.
pub fn validate_boundary_plans(
    plans: &[ExecutionBoundaryPlan],
    ctx: &BoundaryValidationContext,
) -> Result<(), Vec<PlanValidationError>> {
    let mut errors = Vec::new();
    let mut op_to_boundary = std::collections::HashMap::new();

    // Build operation→boundary index and check duplicates
    for plan in plans {
        for &op in &plan.operations {
            if let Some(&prev_group) = op_to_boundary.get(&op) {
                errors.push(PlanValidationError::DuplicateOperation(op));
                let _ = prev_group;
            }
            op_to_boundary.insert(op, plan.group_id);
        }
    }

    // Check every expected operation is covered
    for &op in ctx.expected_operations {
        if !op_to_boundary.contains_key(&op) {
            errors.push(PlanValidationError::MissingOperation(op));
        }
    }

    // Backend transitions: every crossing dependency whose producer and
    // consumer occupy different backends must have a matching transfer
    // plan for the exact edge.via_tensor.
    {
        let mut backend_of: std::collections::HashMap<EvaluationGroupId, BackendId> =
            std::collections::HashMap::new();
        for plan in plans {
            backend_of.insert(plan.group_id, plan.backend_id);
        }
        for edge in ctx.dependency_edges {
            let from_gid = op_to_boundary.get(&edge.from);
            let to_gid = op_to_boundary.get(&edge.to);
            let (Some(&fg), Some(&tg)) = (from_gid, to_gid) else {
                continue;
            };
            let (Some(&fb), Some(&tb)) = (backend_of.get(&fg), backend_of.get(&tg)) else {
                continue;
            };
            if fb != tb {
                let has_transfer = ctx.transfer_plans.iter().any(|tp| {
                    tp.tensor_id == edge.via_tensor
                        && tp.source_backend == fb
                        && tp.destination_backend == tb
                });
                if !has_transfer {
                    errors.push(PlanValidationError::BackendTransitionWithoutTransfer {
                        from: fb,
                        to: tb,
                    });
                }
            }
        }
    }

    // Build a set of materialized-and-synchronized outputs per boundary
    let mut materialized_outputs: std::collections::HashMap<EvaluationGroupId, Vec<TensorId>> =
        std::collections::HashMap::new();
    for plan in plans {
        materialized_outputs.insert(plan.group_id, plan.materialized_outputs.clone());
    }

    for plan in plans {
        if plan.operations.is_empty() {
            errors.push(PlanValidationError::EmptyBoundary(plan.group_id));
        }

        match policy_support(plan.backend_id, &plan.policy) {
            EvaluationPolicySupport::Unsupported => {
                errors.push(PlanValidationError::UnsupportedPolicy {
                    backend: plan.backend_id,
                    policy: plan.policy.clone(),
                });
            }
            _ => {}
        }

        // ── Per-edge validation (runs for every crossing dependency) ──
        for edge in ctx.dependency_edges {
            let from_boundary = op_to_boundary.get(&edge.from);
            let to_boundary = op_to_boundary.get(&edge.to);
            if from_boundary != Some(&plan.group_id)
                || to_boundary.is_none()
                || to_boundary == from_boundary
            {
                continue;
            }

            // Release-liveness: a tensor consumed downstream must not be
            // released at the producing boundary (graph invariant, all policies).
            if plan.release_after.contains(&edge.via_tensor) {
                errors.push(PlanValidationError::EagerWithDeferredDependency {
                    op: edge.from,
                    via: edge.via_tensor,
                    crosses_boundary_to: *to_boundary.unwrap(),
                });
            }

            // Eager-specific: unevaluated deferred dependency crossing check
            if let EvaluationPolicy::Eager {
                prohibit_deferred_nodes: true,
                ..
            } = &plan.policy
            {
                let output_is_materialized = materialized_outputs
                    .get(&plan.group_id)
                    .map(|outputs| outputs.contains(&edge.via_tensor))
                    .unwrap_or(false);
                if !(output_is_materialized && plan.synchronization.is_synchronized()) {
                    errors.push(PlanValidationError::EagerWithDeferredDependency {
                        op: edge.from,
                        via: edge.via_tensor,
                        crosses_boundary_to: *to_boundary.unwrap(),
                    });
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Canonical SHA-256 content digest for an execution boundary plan.
/// Serialization is versioned, length-delimited, and field-tagged so
/// materially different plans produce different digests.
pub fn compute_boundary_digest(plan: &ExecutionBoundaryPlan) -> EvidenceDigest {
    let mut buf: Vec<u8> = Vec::new();

    // Schema version byte
    buf.push(1u8);

    // group_id (8 bytes LE)
    buf.extend_from_slice(&plan.group_id.0.to_le_bytes());
    // backend_id (4 bytes LE)
    buf.extend_from_slice(&plan.backend_id.0.to_le_bytes());

    // operation count + each ID
    buf.extend_from_slice(&(plan.operations.len() as u32).to_le_bytes());
    for op in &plan.operations {
        buf.extend_from_slice(&op.0.to_le_bytes());
    }

    // materialized output count + each ID
    buf.extend_from_slice(&(plan.materialized_outputs.len() as u32).to_le_bytes());
    for t in &plan.materialized_outputs {
        buf.extend_from_slice(&t.0.to_le_bytes());
    }

    // Policy discriminant (4 bits) + variant fields for Eager
    match &plan.policy {
        EvaluationPolicy::BackendLazy => buf.push(0u8),
        EvaluationPolicy::ExplicitRegion => buf.push(1u8),
        EvaluationPolicy::ExplicitOperation => buf.push(2u8),
        EvaluationPolicy::Eager {
            release_inputs_after_use,
            prohibit_deferred_nodes,
        } => {
            buf.push(3u8);
            buf.push(*release_inputs_after_use as u8);
            buf.push(*prohibit_deferred_nodes as u8);
        }
    }

    // synchronization discriminant
    let sync_disc: u8 = match &plan.synchronization {
        SynchronizationPolicy::None => 0,
        SynchronizationPolicy::Barrier => 1,
        SynchronizationPolicy::Stream => 2,
        SynchronizationPolicy::Device => 3,
    };
    buf.push(sync_disc);

    // release_after count + each ID
    buf.extend_from_slice(&(plan.release_after.len() as u32).to_le_bytes());
    for t in &plan.release_after {
        buf.extend_from_slice(&t.0.to_le_bytes());
    }

    let hash = Sha256::digest(&buf);
    EvidenceDigest(format!("{:x}", hash))
}

// ── Boundary executor ─────────────────────────────────────────────────────

/// Executor that consumes a sealed ExecutionBoundaryPlan and enforces
/// evaluation boundaries at runtime.
pub trait BoundaryExecutor {
    /// Execute all boundaries in a plan, emitting observed receipts.
    fn execute_boundaries(
        &mut self,
        plans: &[ExecutionBoundaryPlan],
    ) -> Result<Vec<BoundaryExecutionReceipt>, String>;
}

/// Observed receipt from executing one evaluation boundary.
#[derive(Debug, Clone)]
pub struct BoundaryExecutionReceipt {
    pub group_id: EvaluationGroupId,
    pub planned_policy: EvaluationPolicy,
    pub backend: BackendId,
    pub operation_count: usize,
    pub planned_materialized_outputs: usize,
    pub actual_eval_calls: usize,
    pub actual_sync_count: usize,
    pub graph_build_ns: u64,
    pub submit_ns: u64,
    pub execution_ns: u64,
    pub wait_ns: u64,
    pub temporary_bytes: u64,
    pub released_tensor_count: usize,
    pub unaccounted_ns: u64,
    pub policy_support: EvaluationPolicySupport,
}

// ── Research routing policy ───────────────────────────────────────────────

/// Static research policy — no learned heuristic.
#[derive(Debug, Clone)]
pub enum ResearchRoutingPolicy {
    MlxControl,
    AccelerateCandidate,
    CoreMlCandidate,
    Shadow {
        authority: BackendId,
        candidate: BackendId,
    },
}

/// Evidence-derived route policy entry.
#[derive(Debug, Clone)]
pub struct RoutePolicyEntry {
    pub predicate: RoutePredicate,
    pub selected_backend: BackendId,
    pub expected_latency_ns: Box<(u64, u64)>, // (median, p99)
    pub expected_memory_bytes: Box<(u64, u64)>,
    pub confidence: f64,
    pub evidence_digest: EvidenceDigest,
    pub fallback_backend: BackendId,
}

/// Condition that must be satisfied for a policy to apply.
#[derive(Debug, Clone)]
pub struct RoutePredicate {
    pub family: OperationFamily,
    pub m_min: Option<u32>,
    pub m_max: Option<u32>,
    pub k_min: Option<u32>,
    pub k_max: Option<u32>,
    pub n_min: Option<u32>,
    pub n_max: Option<u32>,
    pub phase: Option<Phase>,
    pub cold_state: Option<bool>,
    pub integrated: Option<bool>,
}

// ── Deterministic router ──────────────────────────────────────────────────

/// Lookup-only router — does not make decisions, only resolves profiles.
pub trait DeterministicRouter {
    fn route(
        &self,
        profile: &ComputeRouteProfile,
        operation_id: OperationId,
    ) -> Result<RoutedOperation, String>;
}
