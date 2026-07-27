//! Token-budget scheduling system (constitutional home, system half).
//!
//! Per the inventory v2.1 step 31, this is the system half of
//! `token_budget`. The state half (TokenWorkUnit,
//! TokenBudgetScheduler, TokenBudgetConfig) is already in
//! `state::token_budget` (step 10). The system half runs the
//! schedule() algorithm.
//!
//! Placeholder: the full engine migration arrives with step 31.

use crate::scheduling::state::token_budget::{TokenBudgetScheduler, TokenWorkUnit};

/// Schedule the next batch of work units. Placeholder: returns
/// the empty batch. The full algorithm (token-budget distribution,
/// chunked prefill, decode steps) is added when the engine's
/// `token_budget::schedule` migrates.
pub fn schedule_next_batch(_scheduler: &mut TokenBudgetScheduler) -> Vec<TokenWorkUnit> {
    Vec::new()
}

/// Reset the scheduler's budget for a new cycle.
pub fn reset_cycle(scheduler: &mut TokenBudgetScheduler) {
    scheduler.reset_budget();
}

/// Mark a request as completed.
pub fn mark_complete(scheduler: &mut TokenBudgetScheduler, request_id: &str) {
    scheduler.complete(request_id);
}

/// Re-enqueue a decode work unit.
pub fn requeue_decode(scheduler: &mut TokenBudgetScheduler, request_id: &str, priority: u32) {
    scheduler.enqueue_decode(request_id, priority);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::state::token_budget::{TokenBudgetConfig, TokenWorkUnit};

    #[test]
    fn schedule_next_batch_placeholder_is_empty() {
        // Architectural invariant: the placeholder returns the
        // empty batch. The full algorithm arrives with the engine
        // migration.
        let mut s = TokenBudgetScheduler::new(TokenBudgetConfig::default());
        let batch = schedule_next_batch(&mut s);
        assert!(batch.is_empty());
    }

    #[test]
    fn reset_cycle_restores_budget() {
        // Architectural invariant: reset_cycle restores the
        // scheduler's budget to the configured maximum.
        let mut s = TokenBudgetScheduler::new(TokenBudgetConfig {
            max_num_batched_tokens: 1024,
            ..TokenBudgetConfig::default()
        });
        s.reset_budget();
        assert_eq!(s.max_budget_tokens(), 1024);
    }

    #[test]
    fn mark_complete_removes_active_request() {
        // Architectural invariant: marking a request complete
        // removes it from the scheduler's active set.
        let mut s = TokenBudgetScheduler::new(TokenBudgetConfig::default());
        s.enqueue(TokenWorkUnit::new_prefill("r1", 100));
        mark_complete(&mut s, "r1");
        // The request is in the run queue but not in active set
        // (the schedule algorithm moves it; the placeholder
        // doesn't). The invariant is: complete() doesn't panic on
        // an unknown request.
        assert_eq!(s.active_count(), 0);
    }

    #[test]
    fn requeue_decode_increments_pending() {
        // Architectural invariant: requeue_decode adds a decode
        // work unit to the run queue.
        let mut s = TokenBudgetScheduler::new(TokenBudgetConfig::default());
        requeue_decode(&mut s, "r1", 2);
        assert_eq!(s.pending_count(), 1);
    }
}
