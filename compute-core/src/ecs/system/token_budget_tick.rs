use crate::ecs::Entity;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;
use crate::ecs::{CompilerSystem, Component, EntityKind, SchedulePhase, World};
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
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::Tensor);

        // Stage every per-entity `TokenBudgetComponent` mutation on a
        // single `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.get_component_mut` calls outside the WorldTxn seam
        // are forbidden.
        //
        // Strategy: snapshot the pre-mutation value via
        // `get_component` (immutable read), compute the post-mutation
        // value locally, and stage the new value as an insert. This
        // is the extract-mutate-insert pattern documented in the
        // Phase 2.5 changelog.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in &entities {
            let Some(budget) = world.get_component::<TokenBudgetComponent>(*entity).cloned()
            else {
                continue;
            };

            let mut updated = budget;
            if updated.tokens_remaining < updated.max_tokens {
                let refill = updated.refill_rate.max(1.0) as u64;
                updated.tokens_remaining = (updated.tokens_remaining + refill).min(updated.max_tokens);
            }
            if let Err(e) = txn.stage_insert(*entity, updated) {
                tracing::warn!(entity = ?entity, error = %e, "token_budget_tick: stage_insert TokenBudgetComponent");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "token_budget_tick: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("token_budget_tick: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
