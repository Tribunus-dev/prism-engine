use crate::ecs::component::fusion::FusionGroup;
use crate::ecs::plan::backend_capability::{
    default_registry, BackendCapabilityRegistry, BackendRole, UnsupportedFusionReason,
};
use crate::ecs::plan::fusion::DataflowOpKind;
use crate::ecs::plan::fusion_scheduler_types::FusionPolicy;

use crate::ecs::{CompEntity, CompilerSystem, SchedulePhase, World};

// ── Op kind string parser ──────────────────────────────────────────────

/// Maps a string op kind to its `DataflowOpKind` discriminant.
fn parse_op_kind(s: &str) -> Option<DataflowOpKind> {
    match s.to_lowercase().as_str() {
        "load_weight" | "loadweight" => Some(DataflowOpKind::LoadWeight),
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
        _ => None,
    }
}

// ── System ─────────────────────────────────────────────────────────────

/// ECS system that evaluates each candidate fusion group by querying
/// backend capabilities. Each group is checked against every registered
/// backend target. If ANY backend supports the op sequence under the
/// production role, the group is accepted.
///
/// Accepted groups have `FusionGroup.accepted = true`; rejected groups
/// carry the reason from the first rejecting backend.
pub struct FusionHeuristicSystem {
    /// Registry of backend capabilities used for fusion decisions.
    pub registry: BackendCapabilityRegistry,
    /// Policy governing fusion acceptance thresholds.
    pub policy: FusionPolicy,
}

impl Default for FusionHeuristicSystem {
    fn default() -> Self {
        Self {
            registry: default_registry(),
            policy: FusionPolicy::default(),
        }
    }
}

impl CompilerSystem for FusionHeuristicSystem {
    fn name(&self) -> &str {
        "FusionHeuristicSystem"
    }

    fn phase(&self) -> SchedulePhase {
        SchedulePhase::FusionDispatch
    }

    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        // Collect entity ids that carry a FusionGroup component.
        let group_entities: Vec<CompEntity> = (1..=world.entity_count() as u64)
            .map(CompEntity)
            .filter(|e| world.get_component::<FusionGroup>(*e).is_some())
            .collect();

        for entity in group_entities {
            // Clone before mutating so we can read and write the world.
            let group = match world.get_component::<FusionGroup>(entity) {
                Some(g) => g.clone(),
                None => continue,
            };

            // ── 1. Parse the root op kind ────────────────────────────────
            let root_kind = match parse_op_kind(&group.root_op_kind) {
                Some(k) => k,
                None => {
                    if let Some(comp) = world.get_component_mut::<FusionGroup>(entity) {
                        comp.accepted = false;
                        comp.reject_reason =
                            Some(format!("unknown root op kind: {}", group.root_op_kind));
                    }
                    continue;
                }
            };

            // ── 2. Parse each fused op kind string ───────────────────────
            let mut all_kinds: Vec<DataflowOpKind> = vec![root_kind];
            let mut parse_failed = false;
            for op_str in &group.fused_op_kinds {
                match parse_op_kind(op_str) {
                    Some(k) => all_kinds.push(k),
                    None => {
                        if let Some(comp) = world.get_component_mut::<FusionGroup>(entity) {
                            comp.accepted = false;
                            comp.reject_reason = Some(format!("unknown fused op kind: {op_str}"));
                        }
                        parse_failed = true;
                        break;
                    }
                }
            }
            if parse_failed {
                continue;
            }

            // ── 3. Check against policy limits ───────────────────────────
            if all_kinds.len() > self.policy.max_group_size {
                if let Some(comp) = world.get_component_mut::<FusionGroup>(entity) {
                    comp.accepted = false;
                    comp.reject_reason = Some(format!(
                        "op count {} exceeds policy max_group_size {}",
                        all_kinds.len(),
                        self.policy.max_group_size,
                    ));
                }
                continue;
            }

            // ── 4. Backend capability check ──────────────────────────────
            // Query each registered backend in the registry. Accept the
            // group if ANY backend supports the op sequence under the
            // production role.
            let targets = self.registry.all_targets();
            let mut any_supported = false;
            let mut first_rejection: Option<String> = None;

            let role = BackendRole::ProductionHotPath;

            for target in targets {
                let (supported, reason) = self.registry.supports_sequence(target, &all_kinds, role);
                if supported {
                    any_supported = true;
                    break;
                }
                if first_rejection.is_none() {
                    first_rejection = reason.map(|r| {
                        if let UnsupportedFusionReason::NoRuleMatched = r {
                            format!("{target:?}: no matching fusion rule for sequence")
                        } else {
                            format!("{target:?}: {r:?}")
                        }
                    });
                }
            }

            // ── 5. Set accepted / rejected ──────────────────────────────
            if let Some(comp) = world.get_component_mut::<FusionGroup>(entity) {
                if any_supported {
                    comp.accepted = true;
                    comp.reject_reason = None;
                } else {
                    comp.accepted = false;
                    comp.reject_reason = first_rejection
                        .or_else(|| Some("no registered backend supports this op sequence".into()));
                }
            }
        }

        Ok(())
    }
}
