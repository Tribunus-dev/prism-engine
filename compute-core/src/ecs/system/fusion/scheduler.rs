//! ECS-native fusion scheduling systems.
//!
//! Three systems that replace the standalone `fusion_scheduler.rs` functions:
//!   - `SchedulerEvaluationSystem` – evaluates fusion groups against backends
//!   - `GroupGrowthSystem` – greedy group growth for unformed clusters
//!   - `CostEvaluationSystem` – cost estimation + best-candidate selection

use crate::ecs::component::fusion::{
    DataflowGraphHandle, FusionEvaluationData, FusionGroup, FusionScheduleData, LoweringCost,
};
use crate::ecs::execution_profile::PhysicalTileLayout;
use crate::ecs::plan::backend_capability::{
    default_registry, BackendCapabilityRegistry, BackendRole,
};
use crate::ecs::plan::fusion::{
    DataflowNode, DataflowOp, DataflowOpKind, FusedGroup, MatMulContract,
};
use crate::ecs::plan::fusion_scheduler_types::{
    FusionCandidate, FusionPolicy, FusionRejection, FusionSelectionPolicy, FusionSupportLevel,
    LoweringCost as SchedLoweringCost,
};
use crate::ecs::plan::CodecFamily;
#[allow(unused_imports)]
use crate::ecs::Entity;
use crate::ecs::{CompWorld, CompilerSystem, EntityKind, SchedulePhase};

// ── Helpers ────────────────────────────────────────────────────────────────

/// Parse an op kind string into a `DataflowOpKind` discriminant.
fn parse_op_kind(s: &str) -> Option<DataflowOpKind> {
    match s.to_lowercase().as_str() {
        "load_weight" | "loadweight" => Some(DataflowOpKind::LoadWeight),
        "load_activation" | "loadactivation" => Some(DataflowOpKind::LoadActivation),
        "dequantize" => Some(DataflowOpKind::Dequantize),
        "matmul" | "mat_mul" => Some(DataflowOpKind::MatMul),
        "rms_norm" | "rmsnorm" => Some(DataflowOpKind::RmsNorm),
        "silu" => Some(DataflowOpKind::SiLU),
        "gelu" => Some(DataflowOpKind::Gelu),
        "mul" | "multiply" => Some(DataflowOpKind::Mul),
        "add" => Some(DataflowOpKind::Add),
        "residual_add" | "residualadd" => Some(DataflowOpKind::ResidualAdd),
        "store_activation" | "storeactivation" => Some(DataflowOpKind::StoreActivation),
        "kv_read" | "kvread" => Some(DataflowOpKind::KvRead),
        "kv_write" | "kvwrite" => Some(DataflowOpKind::KvWrite),
        "engram_lookup" | "engramlookup" | "engram" => Some(DataflowOpKind::EngramLookup),
        _ => None,
    }
}

/// Build a minimal `DataflowNode` from an op kind string and a node id.
fn node_from_kind(kind: &str, id: usize) -> Option<DataflowNode> {
    let op_kind = parse_op_kind(kind)?;
    let buf_in = format!("buf_node{id}_in");
    let buf_out = format!("buf_node{id}_out");

    let op = match op_kind {
        DataflowOpKind::LoadWeight => DataflowOp::LoadWeight {
            tensor: format!("tensor_{id}"),
            codec: CodecFamily::Fp16,
            layout: PhysicalTileLayout::default(),
        },
        DataflowOpKind::LoadActivation => DataflowOp::LoadWeight {
            tensor: format!("act_tensor_{id}"),
            codec: CodecFamily::Fp16,
            layout: PhysicalTileLayout::default(),
        },
        DataflowOpKind::Dequantize => DataflowOp::Dequantize {
            input: buf_in.clone(),
            output_dtype: crate::ecs::plan::DType::F32,
        },
        DataflowOpKind::MatMul => DataflowOp::MatMul {
            lhs: buf_in.clone(),
            rhs: format!("buf_node{id}_weight"),
            output: buf_out.clone(),
            contract: MatMulContract {
                m: 4096,
                n: 4096,
                k: 4096,
                lhs_transposed: false,
                rhs_transposed: false,
            },
        },
        DataflowOpKind::RmsNorm => DataflowOp::RmsNorm {
            input: buf_in.clone(),
            weight: format!("rms_weight_{id}"),
            output: buf_out.clone(),
            epsilon: 1e-6,
        },
        DataflowOpKind::SiLU => DataflowOp::SiLU {
            input: buf_in.clone(),
            output: buf_out.clone(),
        },
        DataflowOpKind::Gelu => DataflowOp::Gelu {
            input: buf_in.clone(),
            output: buf_out.clone(),
        },
        DataflowOpKind::Mul => DataflowOp::Mul {
            lhs: buf_in.clone(),
            rhs: format!("buf_node{id}_rhs"),
            output: buf_out.clone(),
        },
        DataflowOpKind::Add => DataflowOp::Add {
            lhs: buf_in.clone(),
            rhs: format!("buf_node{id}_rhs"),
            output: buf_out.clone(),
        },
        DataflowOpKind::ResidualAdd => DataflowOp::ResidualAdd {
            residual: buf_in.clone(),
            update: format!("buf_node{id}_update"),
            output: buf_out.clone(),
        },
        DataflowOpKind::StoreActivation => DataflowOp::StoreActivation {
            slot: format!("act_slot_{id}"),
            input: buf_in.clone(),
        },
        DataflowOpKind::KvRead => DataflowOp::KvRead {
            slot: format!("kv_slot_{id}"),
            output: buf_out.clone(),
        },
        DataflowOpKind::KvWrite => DataflowOp::KvWrite {
            slot: format!("kv_slot_{id}"),
            input: buf_in.clone(),
        },
        DataflowOpKind::EngramLookup => DataflowOp::EngramLookup {
            engram_id: format!("engram_slot_{id}"),
            lookup_params: crate::ecs::training_target::spec::EngramLookupParams {
                engram_id: format!("engram_slot_{id}"),
                lookup_policy: crate::ecs::training_target::spec::EngramLookupPolicy::AlwaysApply,
                retrieval_threshold: None,
            },
            weights: buf_in.clone(),
            output: buf_out.clone(),
        },
        DataflowOpKind::AneMatMul => DataflowOp::AneMatMul {
            lhs: buf_in.clone(),
            rhs: format!("buf_node{id}_weight"),
            output: buf_out.clone(),
            contract: MatMulContract {
                m: 4096,
                n: 4096,
                k: 4096,
                lhs_transposed: false,
                rhs_transposed: false,
            },
            sram_budget: 32768,
        },
        DataflowOpKind::AneConv1x1 => DataflowOp::AneConv1x1 {
            input: buf_in.clone(),
            weight: format!("conv_weight_{id}"),
            output: buf_out.clone(),
            sram_budget: 32768,
        },
        DataflowOpKind::AneLoadWeight => DataflowOp::AneLoadWeight {
            tensor: format!("ane_weight_{id}"),
            codec: crate::ecs::plan::CodecFamily::Fp16,
            layout: crate::ecs::execution_profile::PhysicalTileLayout::default(),
            target_sram_region: 0,
        },
        DataflowOpKind::AneStoreOutput => DataflowOp::AneStoreOutput {
            input: buf_in.clone(),
            offset: 0,
        },
    };

    Some(DataflowNode {
        id,
        op,
        inputs: vec![buf_in],
        outputs: vec![buf_out],
    })
}

/// Build a `FusedGroup` from a list of op kind strings.
fn build_synthetic_group(op_kinds: &[String]) -> Option<FusedGroup> {
    let body: Vec<DataflowNode> = op_kinds
        .iter()
        .enumerate()
        .filter_map(|(i, k)| node_from_kind(k, i))
        .collect();
    if body.is_empty() {
        return None;
    }
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut internal_values = Vec::new();

    for node in &body {
        for buf in &node.inputs {
            let produced_by_body = body.iter().any(|n| n.outputs.contains(buf));
            if !produced_by_body && !inputs.contains(buf) {
                inputs.push(buf.clone());
            }
        }
        for buf in &node.outputs {
            let consumed_by_body = body.iter().any(|n| n.inputs.contains(buf));
            if consumed_by_body && !internal_values.contains(buf) {
                internal_values.push(buf.clone());
            } else if !consumed_by_body && !outputs.contains(buf) {
                outputs.push(buf.clone());
            }
        }
    }

    Some(FusedGroup {
        id: format!("g{}", op_kinds.first().map(|s| s.as_str()).unwrap_or("")),
        body,
        inputs,
        outputs,
        internal_values,
        codec_family: CodecFamily::Fp16,
        precision_plan: None,
    })
}

/// Chain all op kinds (root + fused) into one Vec.
fn all_op_kinds(group: &FusionGroup) -> Vec<String> {
    std::iter::once(&group.root_op_kind)
        .chain(group.fused_op_kinds.iter())
        .cloned()
        .collect()
}

// ── SchedulerEvaluationSystem ──────────────────────────────────────────────

/// Iterates Dispatch entities with FusionGroup components, builds synthetic
/// FusedGroup instances from the op kind strings, evaluates each group against
/// all registered backends, and attaches FusionScheduleData + FusionEvaluationData
/// components with the results.
pub struct SchedulerEvaluationSystem {
    /// Registry of backend capabilities.
    pub registry: BackendCapabilityRegistry,
    /// Policy governing fusion acceptance thresholds.
    pub policy: FusionPolicy,
    /// Selection policy for scoring candidates.
    pub selection_policy: FusionSelectionPolicy,
}

impl Default for SchedulerEvaluationSystem {
    fn default() -> Self {
        Self {
            registry: default_registry(),
            policy: FusionPolicy::default(),
            selection_policy: FusionSelectionPolicy::default(),
        }
    }
}

impl CompilerSystem for SchedulerEvaluationSystem {
    fn name(&self) -> &str {
        "SchedulerEvaluationSystem"
    }

    fn phase(&self) -> SchedulePhase {
        SchedulePhase::FusionDispatch
    }

    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let dispatch_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Dispatch);

        for entity in dispatch_entities {
            let group = match world.get_component::<FusionGroup>(entity) {
                Some(g) => g.clone(),
                None => continue,
            };

            let kinds = all_op_kinds(&group);
            let fused = match build_synthetic_group(&kinds) {
                Some(f) => f,
                None => continue,
            };

            // Evaluate against all registered backends (port of evaluate_group).
            let targets = self.registry.all_targets();
            let role = BackendRole::ProductionHotPath;
            let mut candidates = Vec::new();
            let mut rejected = Vec::new();
            let source_nodes: Vec<usize> = fused.body.iter().map(|n| n.id).collect();

            for &target in &targets {
                let support = self.registry.evaluate(target, &fused, role);
                if !support.supported {
                    rejected.push(FusionRejection {
                        group_id: fused.id.clone(),
                        target,
                        reason: support.reason.map(|r| format!("{r:?}")).unwrap_or_default(),
                    });
                    continue;
                }

                let support_level = if fused.body.len() <= 1 || support.supported {
                    FusionSupportLevel::Full
                } else {
                    FusionSupportLevel::Partial
                };

                // Compute cost inline (port of compute_cost).
                let op_count = fused.body.len() as f64;
                let estimated_us = support.estimated_latency_us.unwrap_or_else(|| {
                    let base = op_count * 5.0;
                    base * if op_count > 1.0 { 0.6 } else { 1.0 }
                });
                let bytes_read = support
                    .estimated_memory_bytes
                    .unwrap_or((op_count as u64) * 8192);
                let bytes_written = bytes_read;
                let scratch_bytes = support.estimated_scratch_bytes.unwrap_or(1024 * 1024);
                let materialization_cost = if support.requires_materialization {
                    10.0
                } else {
                    0.0
                };

                candidates.push(FusionCandidate {
                    group: fused.clone(),
                    target,
                    support: support_level,
                    lowering_cost: SchedLoweringCost {
                        estimated_us,
                        bytes_read,
                        bytes_written,
                        scratch_bytes,
                        thread_count: 256,
                        materialization_cost,
                    },
                });
            }

            // Select best using score_select logic (production path).
            let selected = if self.policy.allow_research_fusions {
                score_select(&candidates, &self.selection_policy)
            } else {
                prod_select(&candidates)
            };

            world.add_component(
                entity,
                FusionScheduleData {
                    candidates: candidates.clone(),
                    selected: selected.clone(),
                },
            );

            world.add_component(
                entity,
                FusionEvaluationData {
                    source_nodes,
                    rejected,
                },
            );

            // If a candidate was selected, also attach LoweringCost component.
            if let Some(ref best) = selected {
                world.add_component(entity, LoweringCost(best.lowering_cost.clone()));
            }
        }

        Ok(())
    }
}

// ── GroupGrowthSystem ──────────────────────────────────────────────────────

/// Greedy group growth: for Dispatch entities whose fusion group has no fused
/// ops (singletons), attempt to merge with compatible sibling dispatches in the
/// same layer (same DataflowGraphHandle).
///
/// Growth follows consumer-edge semantics: a singleton looks for sibling
/// dispatches whose root_op_kind forms a supported fusion pattern when chained
/// after this entity's ops.
pub struct GroupGrowthSystem {
    /// Registry of backend capabilities for tentative support checks.
    pub registry: BackendCapabilityRegistry,
    /// Policy limits (max_group_size).
    pub policy: FusionPolicy,
}

impl Default for GroupGrowthSystem {
    fn default() -> Self {
        Self {
            registry: default_registry(),
            policy: FusionPolicy::default(),
        }
    }
}

impl CompilerSystem for GroupGrowthSystem {
    fn name(&self) -> &str {
        "GroupGrowthSystem"
    }

    fn phase(&self) -> SchedulePhase {
        SchedulePhase::FusionDispatch
    }

    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        // Collect dispatch entities with FusionGroup components.
        let dispatch_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Dispatch);

        // Group by DataflowGraphHandle for layer-aware growth.
        let mut by_handle: Vec<(String, Vec<Entity>)> = Vec::new();
        let mut seen_handles: Vec<String> = Vec::new();

        for &entity in &dispatch_entities {
            let handle = match world
                .get_component::<DataflowGraphHandle>(entity)
                .map(|h| h.0.clone())
            {
                Some(h) => h,
                None => continue,
            };
            // Check if this entity is a singleton (no fused ops).
            let is_singleton = matches!(world.get_component::<FusionGroup>(entity), Some(g) if g.fused_op_kinds.is_empty());
            if !is_singleton {
                continue;
            }
            // Find or create the group bucket.
            let pos = seen_handles.iter().position(|h| *h == handle);
            match pos {
                Some(idx) => by_handle[idx].1.push(entity),
                None => {
                    seen_handles.push(handle.clone());
                    by_handle.push((handle.clone(), vec![entity]));
                }
            }
        }

        let role = BackendRole::ProductionHotPath;

        for (_handle, entities) in &by_handle {
            if entities.len() < 2 {
                continue;
            }

            // Try to grow each singleton into a multi-op group.
            for &seed_entity in entities {
                let seed_group = match world.get_component::<FusionGroup>(seed_entity) {
                    Some(g) => g.clone(),
                    None => continue,
                };

                // Don't grow an already-accepted group.
                if seed_group.accepted {
                    continue;
                }

                let mut grown_kinds: Vec<String> = all_op_kinds(&seed_group);

                // Try merging with sibling dispatches until policy limits.
                for &sibling in entities {
                    if sibling == seed_entity {
                        continue;
                    }
                    if grown_kinds.len() >= self.policy.max_group_size {
                        break;
                    }

                    let sib_group = match world.get_component::<FusionGroup>(sibling) {
                        Some(g) => g.clone(),
                        None => continue,
                    };
                    let sib_kinds = all_op_kinds(&sib_group);

                    // Tentative check: would the extended group be supported?
                    let mut tentative = grown_kinds.clone();
                    tentative.extend(sib_kinds);

                    if tentative.len() > self.policy.max_group_size {
                        continue;
                    }

                    if tentative_is_supported(&self.registry, &tentative, role) {
                        grown_kinds = tentative;
                    }
                }

                // If we grew beyond the seed, update the FusionGroup component.
                if grown_kinds.len() > 1 {
                    if let Some(comp) = world.get_component_mut::<FusionGroup>(seed_entity) {
                        comp.root_op_kind = grown_kinds[0].clone();
                        comp.fused_op_kinds = grown_kinds[1..].to_vec();
                    }
                }
            }
        }

        Ok(())
    }
}

/// Check whether at least one registered backend supports a tentative
/// sequence of op kind strings.
fn tentative_is_supported(
    registry: &BackendCapabilityRegistry,
    op_kinds: &[String],
    role: BackendRole,
) -> bool {
    if op_kinds.len() <= 1 {
        return true;
    }
    let parsed: Vec<DataflowOpKind> = op_kinds.iter().filter_map(|s| parse_op_kind(s)).collect();
    if parsed.len() != op_kinds.len() {
        return false;
    }
    registry
        .all_targets()
        .iter()
        .any(|t| registry.supports_sequence(*t, &parsed, role).0)
}

// ── CostEvaluationSystem ───────────────────────────────────────────────────

/// Evaluates cost and selects the best candidate for each dispatch that has
/// FusionScheduleData but no LoweringCost component yet.
///
/// Ports the `score_select()` / `prod_select()` logic from fusion_scheduler.rs.
pub struct CostEvaluationSystem {
    pub selection_policy: FusionSelectionPolicy,
}

impl Default for CostEvaluationSystem {
    fn default() -> Self {
        Self {
            selection_policy: FusionSelectionPolicy::default(),
        }
    }
}

impl CompilerSystem for CostEvaluationSystem {
    fn name(&self) -> &str {
        "CostEvaluationSystem"
    }

    fn phase(&self) -> SchedulePhase {
        SchedulePhase::FusionDispatch
    }

    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let dispatch_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Dispatch);

        for entity in dispatch_entities {
            // Skip if already has a cost.
            if world.get_component::<LoweringCost>(entity).is_some() {
                continue;
            }

            let schedule = match world.get_component::<FusionScheduleData>(entity) {
                Some(s) => s.clone(),
                None => continue,
            };

            // If a selected candidate already exists, attach its cost.
            if let Some(ref selected) = schedule.selected {
                world.add_component(entity, LoweringCost(selected.lowering_cost.clone()));
                continue;
            }

            // Otherwise run selection policy on the candidates.
            let selected = if !schedule.candidates.is_empty() {
                score_select(&schedule.candidates, &self.selection_policy)
                    .or_else(|| prod_select(&schedule.candidates))
            } else {
                None
            };

            if let Some(ref best) = selected {
                world.add_component(entity, LoweringCost(best.lowering_cost.clone()));
            }
        }

        Ok(())
    }
}

// ── Score selection (ported from fusion_scheduler.rs) ──────────────────────

/// Score-based selection using the policy.
fn score_select(
    candidates: &[FusionCandidate],
    policy: &FusionSelectionPolicy,
) -> Option<FusionCandidate> {
    if candidates.is_empty() {
        return None;
    }
    let mut best: Option<&FusionCandidate> = None;
    let mut best_score = f64::NEG_INFINITY;

    for c in candidates {
        let mut score = 0.0;

        match c.support {
            FusionSupportLevel::Full => score += 100.0,
            FusionSupportLevel::Partial => score += 10.0,
            FusionSupportLevel::Unsupported => score -= 1000.0,
        }

        if policy.prefer_lower_latency {
            score -= (c.lowering_cost.estimated_us + c.lowering_cost.materialization_cost) / 1000.0;
        }
        if policy.prefer_memory_efficient {
            score -= ((c.lowering_cost.bytes_read + c.lowering_cost.bytes_written) as f64)
                / (1024.0 * 1024.0);
        }
        if policy.avoid_materialization && c.lowering_cost.materialization_cost > 0.0 {
            score -= c.lowering_cost.materialization_cost;
        }

        if score > best_score {
            best_score = score;
            best = Some(c);
        }
    }

    best.cloned()
}

/// Production selection — prefer Full support, then first Partial.
fn prod_select(candidates: &[FusionCandidate]) -> Option<FusionCandidate> {
    candidates
        .iter()
        .find(|c| c.support == FusionSupportLevel::Full)
        .or_else(|| {
            candidates
                .iter()
                .find(|c| c.support == FusionSupportLevel::Partial)
        })
        .cloned()
}
