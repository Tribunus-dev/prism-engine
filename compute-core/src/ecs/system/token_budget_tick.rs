#[allow(unused_imports)]
use crate::ecs::Entity;
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, Component, EntityKind, SchedulePhase};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// TokenBudgetComponent
// ---------------------------------------------------------------------------

/// Token budget for admission control — inspired by vLLM V1 token-budget
/// scheduling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TokenBudgetComponent {
    pub tokens_remaining: u64,
    pub max_tokens: u64,
    pub refill_rate: f64,
}
impl Component for TokenBudgetComponent {}

/// Ticks the token budget refill — replenishes tokens toward the
/// configured maximum at the refill rate.
///
/// Designed to run on every tick of the `Execution` phase so that
/// token budgets gradually recover after heavy work bursts.
pub struct TokenBudgetTickSystem;
impl CompilerSystem for TokenBudgetTickSystem {
    fn name(&self) -> &str {
        "TokenBudgetTickSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Tensor);

        for entity in &entities {
            let Some(budget) = world.get_component_mut::<TokenBudgetComponent>(*entity) else {
                continue;
            };

            if budget.tokens_remaining < budget.max_tokens {
                let refill = budget.refill_rate.max(1.0) as u64;
                budget.tokens_remaining = (budget.tokens_remaining + refill).min(budget.max_tokens);
            }
        }

        Ok(())
    }
}
