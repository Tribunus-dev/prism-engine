//! Compiler module systems — Compilation and FusionDispatch phases.
//!
//! Ported from: compiler/{compile_schedule, backend_assessment, graph_optimizer}
//! and compilation/graph_equalization.rs.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use prism_ecs_kernel::backend::routing::{BackendId, EvidenceDigest, OperationFamily, OperationId};
use crate::ecs::component::compilation::{GraphNode, GraphNodeKind, NodeId};
use prism_ecs_constitutional::config::ModelExecutionPlan;
use prism_ecs_compile::pipeline::backend_assessment::{
    BackendAssessmentPass, GraphOperation, ModelOperationGraph,
};
use prism_ecs_compile::pipeline::compile_schedule::compile_model_to_scheduled_module;
use prism_ecs_compile::pipeline::graph_optimizer::optimize;
use prism_ecs_compile::pipeline::pass::TransformPass;
use prism_ecs_compile::pipeline::scheduled::ScheduledModule;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

// ---------------------------------------------------------------------------
// CompileScheduleSystem
// ---------------------------------------------------------------------------

/// Translates a model manifest to a ScheduledModule with populated regions,
/// memory plan, transfer plan, and evaluation boundaries.
pub struct CompileScheduleSystem;
impl CompilerSystem for CompileScheduleSystem {
    fn name(&self) -> &str {
        "CompileScheduleSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Compilation
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let model_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Model);

        for entity in &model_entities {
            let Some(plan) = world.get_component::<ModelExecutionPlan>(*entity) else {
                continue;
            };

            let first = plan.layers.first();
            let arch = prism_ecs_constitutional::config::TextArchitecture {
                hidden_size: plan.hidden_size as u32,
                intermediate_size: (plan.hidden_size * 4) as u32,
                vocab_size: plan.vocab_size as u32,
                max_position_embeddings: 4096,
                num_hidden_layers: plan.layers.len() as u32,
                num_attention_heads: first.map(|l| l.n_heads).unwrap_or(32),
                num_key_value_heads: first.map(|l| l.n_kv_heads).unwrap_or(8),
                ..Default::default()
            };

            let digest = EvidenceDigest("model-source-v1".into());
            let _module: ScheduledModule = compile_model_to_scheduled_module(plan, &arch, digest);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BackendAssessmentSystem
// ---------------------------------------------------------------------------

/// Scores each operation against all available backends and produces sealed
/// ExecutionBoundaryPlans with cross-backend transfer plans.
pub struct BackendAssessmentSystem;
impl CompilerSystem for BackendAssessmentSystem {
    fn name(&self) -> &str {
        "BackendAssessmentSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Compilation
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let backends = vec![
            BackendId(0), // Metal
            BackendId(1), // Accelerate
            BackendId(2), // ANE
            BackendId(3), // MLX
        ];
        let pass = BackendAssessmentPass::new(backends);

        let model_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Model);
        for entity in &model_entities {
            let Some(plan) = world.get_component::<ModelExecutionPlan>(*entity) else {
                continue;
            };

            let mut ops = Vec::new();
            for (i, _layer) in plan.layers.iter().enumerate() {
                ops.push(GraphOperation {
                    id: OperationId(i as u64),
                    family: OperationFamily::Matmul,
                    m: Some(plan.hidden_size),
                    n: Some(plan.hidden_size),
                    k: Some(plan.hidden_size),
                    quantized: false,
                });
            }
            let graph = ModelOperationGraph {
                operations: ops,
                operand_shapes: HashMap::new(),
            };

            let _ = pass.apply(&graph, EvidenceDigest("input".into()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GraphOptimizerSystem
// ---------------------------------------------------------------------------

/// Runs three-pass graph optimization — constant folding, shape propagation,
/// and dead code elimination — on each model's execution plan.
pub struct GraphOptimizerSystem;
impl CompilerSystem for GraphOptimizerSystem {
    fn name(&self) -> &str {
        "GraphOptimizerSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::FusionDispatch
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let model_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Model);

        // Stage every per-model `GraphNode` insert on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.add_component` calls outside the WorldTxn seam are
        // forbidden.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in &model_entities {
            let Some(plan) = world.get_component::<ModelExecutionPlan>(*entity) else {
                continue;
            };

            let mut owned = plan.clone();
            optimize(&mut owned);
            // GraphNode marking the optimisation
            if let Err(e) = txn.stage_insert(
                *entity,
                GraphNode {
                    id: NodeId::from("graph-optimised"),
                    deps: Vec::new(),
                    kind: GraphNodeKind::Matmul,
                },
            ) {
                tracing::warn!(entity = ?entity, error = %e, "graph_optimizer: stage_insert GraphNode");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "graph_optimizer: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("graph_optimizer: ConstitutionalWorldTxn commit failed: {e}")
        })?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GraphEqualizationSystem
// ---------------------------------------------------------------------------

/// Whether a PhaseIR boundary is legal for scale migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryLegality {
    /// Legal: can safely absorb scales across this boundary.
    Legal,
    /// Illegal: non-linear operation blocks migration.
    Illegal { reason: &'static str },
    /// Conditional: legal only with manifest recording.
    Conditional { requires_inverse: bool },
}

/// A recorded scale migration operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleMigrationRecord {
    /// Source tensor name (the weights being transformed).
    pub source_tensor: String,
    /// Target tensor name (the adjacent weights receiving the inverse).
    pub target_tensor: Option<String>,
    /// The diagonal scale vector D used for migration.
    pub scale_diagonal: Vec<f32>,
    /// Inverse scale vector D^{-1} (element-wise reciprocal).
    pub inverse_diagonal: Vec<f32>,
    /// Whether the migration was actually applied.
    pub applied: bool,
}

/// Check if a pair of adjacent phase types can safely absorb scale migration.
pub fn is_legal_boundary(producer_phase_type: &str, consumer_phase_type: &str) -> BoundaryLegality {
    // Linear operations: safe to absorb scales
    let linear = [
        "QkvProjection",
        "OutputProjection",
        "FfnGate",
        "FfnUp",
        "FfnDown",
        "LoadTeacherRegion",
        "LoadStudentCandidate",
    ];

    // Non-linear operations: unsafe
    let nonlinear = [
        "AttentionSoftmax",
        "RmsNorm",
        "RoPE",
        "SiLU",
        "GELU",
        "ResidualAdd",
        "CausalConvolution",
        "SpatialPatchEmbedding",
        "GridAttention2D",
        "AdaptiveLayerNorm",
        "TimeStepEmbedding",
    ];

    let prod_is_linear = linear.iter().any(|l| producer_phase_type.contains(l));
    let cons_is_nonlinear = nonlinear.iter().any(|n| consumer_phase_type.contains(n));

    if prod_is_linear && !cons_is_nonlinear {
        BoundaryLegality::Conditional {
            requires_inverse: true,
        }
    } else if cons_is_nonlinear {
        BoundaryLegality::Illegal {
            reason: "consumer is non-linear",
        }
    } else {
        BoundaryLegality::Illegal {
            reason: "unsupported phase type pair",
        }
    }
}

/// Checks phase-type boundaries for legal NF4 scale migration.
pub struct GraphEqualizationSystem;
impl CompilerSystem for GraphEqualizationSystem {
    fn name(&self) -> &str {
        "GraphEqualizationSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::FusionDispatch
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let tensor_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Tensor);
        let phase_types = [
            "rms_norm",
            "q_projection",
            "k_projection",
            "v_projection",
            "rotary_embedding",
            "attention_score",
            "softmax",
            "attention_value_aggregation",
            "output_projection",
            "residual_add",
            "gate_projection",
            "silu_activation",
            "up_projection",
            "down_projection",
            "final_norm",
            "logits_projection",
        ];

        // Stage every per-tensor `GraphNode` insert on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.add_component` calls outside the WorldTxn seam are
        // forbidden.
        let mut txn = ConstitutionalWorldTxn::new();
        for window in phase_types.windows(2) {
            let producer = window[0];
            let consumer = window[1];
            match is_legal_boundary(producer, consumer) {
                BoundaryLegality::Illegal { reason } => {
                    for entity in &tensor_entities {
                        if let Err(e) = txn.stage_insert(
                            *entity,
                            GraphNode {
                                id: NodeId::from(format!("ill-boundary-{producer}-{consumer}")),
                                deps: vec![],
                                kind: GraphNodeKind::Unknown(reason.to_string()),
                            },
                        ) {
                            tracing::warn!(entity = ?entity, error = %e, "graph_equalization: stage_insert ill-boundary GraphNode");
                        }
                    }
                }
                BoundaryLegality::Conditional {
                    requires_inverse: _,
                } => {
                    for entity in &tensor_entities {
                        if let Err(e) = txn.stage_insert(
                            *entity,
                            GraphNode {
                                id: NodeId::from(format!("cond-boundary-{producer}-{consumer}")),
                                deps: vec![],
                                kind: GraphNodeKind::Unknown("conditional_scale_boundary".into()),
                            },
                        ) {
                            tracing::warn!(entity = ?entity, error = %e, "graph_equalization: stage_insert cond-boundary GraphNode");
                        }
                    }
                }
                BoundaryLegality::Legal => {}
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "graph_equalization: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("graph_equalization: ConstitutionalWorldTxn commit failed: {e}")
        })?;
        Ok(())
    }
}
