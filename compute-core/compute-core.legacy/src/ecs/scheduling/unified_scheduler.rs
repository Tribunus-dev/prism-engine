//! Unified token-budget scheduler — ECS-style data model for the vLLM v1
//! scheduling pattern.
//!
//! Replaces the explicit prefill/decode phase split with a token-budget model
//! where each request tracks its own progress and the scheduler distributes
//! a shared token pool proportionally by deficit.
//!
//! These types define the data model for a future `ScheduleSystem`. They
//! coexist with the existing struct-based continuous-batching scheduler.

use super::SchedulerConfig;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// ECS-style state for the unified token-budget scheduler.
///
/// Tracks three request pools — running, waiting, preempted — and the shared
/// token budget that limits how many tokens can be scheduled per step.
#[derive(Debug, Clone)]
pub struct SchedulerState {
    /// Max tokens to schedule per step (shared budget)
    pub max_num_scheduled_tokens: usize,
    /// Max running requests
    pub max_num_running_reqs: usize,
    /// Currently running request IDs
    pub running: Vec<String>,
    /// Waiting request IDs
    pub waiting: Vec<String>,
    /// Preempted request IDs
    pub preempted: Vec<String>,
    /// Request data keyed by ID
    pub requests: HashMap<String, UnifiedRequestData>,
}

/// Per-request state for the unified scheduler.
#[derive(Debug, Clone)]
pub struct UnifiedRequestData {
    /// Total prompt + output + draft tokens in the spec
    pub num_tokens_with_spec: usize,
    /// How many tokens have been computed so far
    pub num_computed_tokens: usize,
    /// Status: waiting, running, preempted
    pub status: UnifiedRequestStatus,
    /// Priority (lower = higher priority)
    pub priority: usize,
    /// Number of times preempted
    pub preemption_count: usize,
}

/// Lifecycle status of a request in the unified scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnifiedRequestStatus {
    /// Awaiting admission to the running pool
    Waiting,
    /// Actively being scheduled
    Running,
    /// Evicted from the running pool (may resume later)
    Preempted,
}

/// Result of one scheduling step — which requests get how many tokens.
#[derive(Debug, Clone)]
pub struct ScheduleOutput {
    /// Tokens to assign per request
    pub assignments: Vec<ScheduleAssignment>,
    /// Total tokens scheduled this step
    pub total_tokens_scheduled: usize,
}

/// Per-request token assignment from a scheduling step.
#[derive(Debug, Clone)]
pub struct ScheduleAssignment {
    pub request_id: String,
    /// New tokens to compute (chunked prefill tokens for waiting, 1 for decode)
    pub num_new_tokens: usize,
    /// KV block IDs for this batch (from KV cache manager)
    pub block_ids: Vec<u64>,
    /// Speculative tokens for this request (0 if none)
    pub spec_tokens: usize,
}

// ---------------------------------------------------------------------------
// SchedulerState methods
// ---------------------------------------------------------------------------

impl SchedulerState {
    /// Create a new scheduler state with the given budget and concurrency limits.
    pub fn new(max_num_scheduled_tokens: usize, max_num_running_reqs: usize) -> Self {
        Self {
            max_num_scheduled_tokens,
            max_num_running_reqs,
            running: Vec::new(),
            waiting: Vec::new(),
            preempted: Vec::new(),
            requests: HashMap::new(),
        }
    }

    /// Create a new scheduler state from a [`SchedulerConfig`].
    ///
    /// Extracts `max_num_scheduled_tokens` and `max_batch_size` from the
    /// config and delegates to [`new`](Self::new).
    pub fn new_with_config(config: &SchedulerConfig) -> Self {
        Self::new(config.max_num_scheduled_tokens, config.max_batch_size)
    }

    /// Add a request to the waiting pool.
    pub fn add_request(&mut self, id: &str, num_tokens_with_spec: usize, priority: usize) {
        self.requests.insert(
            id.to_string(),
            UnifiedRequestData {
                num_tokens_with_spec,
                num_computed_tokens: 0,
                status: UnifiedRequestStatus::Waiting,
                priority,
                preemption_count: 0,
            },
        );
        self.waiting.push(id.to_string());
    }

    /// Remove a request from all pools and state.
    pub fn remove_request(&mut self, id: &str) {
        self.running.retain(|r| r != id);
        self.waiting.retain(|r| r != id);
        self.preempted.retain(|r| r != id);
        self.requests.remove(id);
    }

    /// Run one scheduling step.
    ///
    /// Algorithm:
    /// 1. Move requests from `waiting` → `running` if capacity allows.
    /// 2. Compute token deficit for each running request.
    /// 3. Distribute the shared token budget (`max_num_scheduled_tokens`)
    ///    across running requests proportional to their deficit.
    /// 4. Cap each allocation at the request's deficit (no overshoot).
    /// 5. New requests (not yet prefilled) get chunked prefill tokens capped
    ///    at `chunk_cap = max(max_num_scheduled_tokens / 4, 1)`.
    /// 6. Each running request with a positive deficit gets at least 1 token
    ///    (decode step).
    pub fn schedule_once(&mut self) -> ScheduleOutput {
        // ---- Phase 1: admit waiting requests into the running pool ----
        while self.running.len() < self.max_num_running_reqs {
            match self.waiting.pop() {
                Some(id) => {
                    if let Some(req) = self.requests.get_mut(&id) {
                        req.status = UnifiedRequestStatus::Running;
                    }
                    self.running.push(id);
                }
                None => break,
            }
        }

        // ---- Phase 2: compute deficits ----
        let deficits: Vec<(String, usize)> = self
            .running
            .iter()
            .filter_map(|id| {
                self.requests.get(id).map(|req| {
                    let deficit = req
                        .num_tokens_with_spec
                        .saturating_sub(req.num_computed_tokens);
                    (id.clone(), deficit)
                })
            })
            .collect();

        let total_deficit: usize = deficits.iter().map(|(_, d)| *d).sum();
        if total_deficit == 0 {
            return ScheduleOutput {
                assignments: Vec::new(),
                total_tokens_scheduled: 0,
            };
        }

        // ---- Phase 3: proportional distribution ----
        let budget = self.max_num_scheduled_tokens;
        let chunk_cap = (budget / 4).max(1);
        let mut allocated_total = 0usize;
        let mut assignments = Vec::new();

        for (id, deficit) in &deficits {
            if allocated_total >= budget {
                break;
            }
            let budget_remaining = budget - allocated_total;

            // Proportional share of the total budget
            let mut share = (deficit * budget) / total_deficit;
            // Cap at the request's deficit (never overshoot)
            share = share.min(*deficit);
            // Cap at remaining budget
            share = share.min(budget_remaining);

            // Chunked prefill cap for newly admitted requests
            let is_new = self
                .requests
                .get(id)
                .map(|r| r.num_computed_tokens == 0)
                .unwrap_or(false);
            if is_new {
                share = share.min(chunk_cap);
            }

            // Every running request with a positive deficit gets at least 1 token
            // for decode
            if *deficit > 0 && share == 0 && budget_remaining > 0 {
                share = 1;
            }

            if share > 0 {
                allocated_total += share;
                if let Some(req) = self.requests.get_mut(id) {
                    req.num_computed_tokens += share;
                }
                assignments.push(ScheduleAssignment {
                    request_id: id.clone(),
                    num_new_tokens: share,
                    block_ids: Vec::new(),
                    spec_tokens: 0,
                });
            }
        }

        ScheduleOutput {
            total_tokens_scheduled: allocated_total,
            assignments,
        }
    }

    /// Preempt the lowest-priority running request.
    ///
    /// The request is moved from `running` to `preempted`, its status is
    /// updated, and the preemption counter is incremented.
    ///
    /// Returns the ID of the preempted request, or `None` if no request was
    /// running.
    pub fn preempt_lowest(&mut self) -> Option<String> {
        // Find the running request with the largest priority value (lowest
        // priority).
        let lowest_id = self
            .running
            .iter()
            .filter_map(|id| self.requests.get(id).map(|req| (id.clone(), req.priority)))
            .max_by_key(|(_, priority)| *priority)
            .map(|(id, _)| id);

        if let Some(ref id) = lowest_id {
            self.running.retain(|r| r != id);
            if let Some(req) = self.requests.get_mut(id) {
                req.status = UnifiedRequestStatus::Preempted;
                req.preemption_count += 1;
            }
            self.preempted.push(id.clone());
        }

        lowest_id
    }
}

// ---------------------------------------------------------------------------
// SchedulerRunner — runtime integration wrapper
// ---------------------------------------------------------------------------

/// Runtime wrapper that integrates [`SchedulerState`] with the execution
/// pipeline.
///
/// Provides a lifecycle-oriented API:
/// - [`step`](Self::step) — one scheduling cycle
/// - [`submit_request`](Self::submit_request) — enqueue a new request
/// - [`complete_request`](Self::complete_request) — finalise and remove a
///   finished request
///
/// The existing [`crate::ecs::scheduling::Scheduler`] (in `scheduler.rs`)
/// remains for backwards compatibility with the MLX continuous-batching path.
/// This runner is the authoritative scheduling interface for the unified
/// token-budget path.
#[derive(Debug, Clone)]
pub struct SchedulerRunner {
    /// The underlying scheduler state.
    pub state: SchedulerState,
}

impl SchedulerRunner {
    /// Create a new runner from a [`SchedulerConfig`].
    pub fn new(config: &SchedulerConfig) -> Self {
        Self {
            state: SchedulerState::new_with_config(config),
        }
    }

    /// One scheduling step: run [`schedule_once`] and return the assignments
    /// for the next batch.
    pub fn step(&mut self) -> Result<ScheduleOutput, String> {
        Ok(self.state.schedule_once())
    }

    /// Add a new request to the waiting pool.
    pub fn submit_request(&mut self, id: &str, tokens: usize, priority: usize) {
        self.state.add_request(id, tokens, priority);
    }

    /// Mark a request as complete and remove it from all scheduler state.
    pub fn complete_request(&mut self, id: &str) {
        self.state.remove_request(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_state_is_empty() {
        let state = SchedulerState::new(128, 8);
        assert_eq!(state.max_num_scheduled_tokens, 128);
        assert_eq!(state.max_num_running_reqs, 8);
        assert!(state.running.is_empty());
        assert!(state.waiting.is_empty());
        assert!(state.preempted.is_empty());
        assert!(state.requests.is_empty());
    }

    #[test]
    fn test_add_and_remove_request() {
        let mut state = SchedulerState::new(128, 8);
        state.add_request("req-1", 100, 0);
        assert_eq!(state.waiting.len(), 1);
        assert!(state.requests.contains_key("req-1"));

        state.remove_request("req-1");
        assert!(state.waiting.is_empty());
        assert!(!state.requests.contains_key("req-1"));
    }

    #[test]
    fn test_schedule_once_admits_waiting_requests() {
        let mut state = SchedulerState::new(128, 4);
        state.add_request("a", 50, 0);
        state.add_request("b", 50, 0);
        state.add_request("c", 50, 0);

        let output = state.schedule_once();
        assert_eq!(state.running.len(), 3);
        assert!(state.waiting.is_empty());
        // All three got tokens
        assert_eq!(output.assignments.len(), 3);
        assert!(output.total_tokens_scheduled > 0);
    }

    #[test]
    fn test_schedule_once_respects_max_running_cap() {
        let mut state = SchedulerState::new(128, 2);
        state.add_request("a", 50, 0);
        state.add_request("b", 50, 0);
        state.add_request("c", 50, 0);

        let _ = state.schedule_once();
        assert_eq!(state.running.len(), 2);
        assert_eq!(state.waiting.len(), 1);
    }

    #[test]
    fn test_schedule_once_proportional_distribution() {
        let mut state = SchedulerState::new(100, 8);
        // Add requests as already-running so we skip admission
        state.requests.insert(
            "a".to_string(),
            UnifiedRequestData {
                num_tokens_with_spec: 100,
                num_computed_tokens: 0,
                status: UnifiedRequestStatus::Running,
                priority: 0,
                preemption_count: 0,
            },
        );
        state.running.push("a".to_string());
        state.requests.insert(
            "b".to_string(),
            UnifiedRequestData {
                num_tokens_with_spec: 300,
                num_computed_tokens: 0,
                status: UnifiedRequestStatus::Running,
                priority: 0,
                preemption_count: 0,
            },
        );
        state.running.push("b".to_string());

        let output = state.schedule_once();

        // a has deficit 100, b has deficit 300. Total deficit = 400.
        // Budget = 100. a gets 100 * 100 / 400 = 25. b gets 300 * 100 / 400 = 75.
        // But both are new requests so chunked prefill cap applies:
        // chunk_cap = 100/4 = 25
        // a: min(25, 100, 25) = 25
        // b: min(75, 300, 25) = 25
        // They might also get additional tokens due to round-robin redistribution
        assert_eq!(output.assignments.len(), 2);
        assert_eq!(output.total_tokens_scheduled, 50);
    }

    #[test]
    fn test_schedule_once_no_overshoot() {
        let mut state = SchedulerState::new(1000, 8);
        state.requests.insert(
            "a".to_string(),
            UnifiedRequestData {
                num_tokens_with_spec: 10,
                num_computed_tokens: 5,
                status: UnifiedRequestStatus::Running,
                priority: 0,
                preemption_count: 0,
            },
        );
        state.running.push("a".to_string());

        let output = state.schedule_once();
        // a's deficit is 5, so it should get at most 5 tokens
        assert_eq!(output.assignments.len(), 1);
        assert!(output.assignments[0].num_new_tokens <= 5);
        assert_eq!(output.total_tokens_scheduled, 5);
    }

    #[test]
    fn test_preempt_lowest_priority() {
        let mut state = SchedulerState::new(128, 8);
        state.requests.insert(
            "high".to_string(),
            UnifiedRequestData {
                num_tokens_with_spec: 100,
                num_computed_tokens: 0,
                status: UnifiedRequestStatus::Running,
                priority: 0,
                preemption_count: 0,
            },
        );
        state.running.push("high".to_string());
        state.requests.insert(
            "low".to_string(),
            UnifiedRequestData {
                num_tokens_with_spec: 100,
                num_computed_tokens: 0,
                status: UnifiedRequestStatus::Running,
                priority: 10,
                preemption_count: 0,
            },
        );
        state.running.push("low".to_string());

        let preempted = state.preempt_lowest();
        assert_eq!(preempted.as_deref(), Some("low"));
        assert_eq!(state.running.len(), 1);
        assert_eq!(state.preempted.len(), 1);
        assert_eq!(
            state.requests.get("low").unwrap().status,
            UnifiedRequestStatus::Preempted
        );
        assert_eq!(state.requests.get("low").unwrap().preemption_count, 1);
    }

    #[test]
    fn test_preempt_lowest_empty() {
        let mut state = SchedulerState::new(128, 8);
        assert!(state.preempt_lowest().is_none());
    }

    #[test]
    fn test_remove_request_cleans_all_pools() {
        let mut state = SchedulerState::new(128, 8);
        state.add_request("r", 100, 0);
        state.remove_request("r");
        assert!(!state.requests.contains_key("r"));
        assert!(!state.waiting.contains(&"r".to_string()));

        // Also check running and preempted pools
        state.requests.insert(
            "r".to_string(),
            UnifiedRequestData {
                num_tokens_with_spec: 100,
                num_computed_tokens: 50,
                status: UnifiedRequestStatus::Running,
                priority: 0,
                preemption_count: 0,
            },
        );
        state.running.push("r".to_string());
        state.remove_request("r");
        assert!(!state.running.contains(&"r".to_string()));

        state.preempted.push("r".to_string());
        state.requests.insert(
            "r".to_string(),
            UnifiedRequestData {
                num_tokens_with_spec: 100,
                num_computed_tokens: 50,
                status: UnifiedRequestStatus::Preempted,
                priority: 0,
                preemption_count: 1,
            },
        );
        state.remove_request("r");
        assert!(!state.preempted.contains(&"r".to_string()));
    }

    #[test]
    fn test_empty_schedule_returns_zero() {
        let mut state = SchedulerState::new(128, 8);
        let output = state.schedule_once();
        assert_eq!(output.total_tokens_scheduled, 0);
        assert!(output.assignments.is_empty());
    }

    #[test]
    fn test_chunked_prefill_cap_for_new_requests() {
        let mut state = SchedulerState::new(128, 8);
        // A request with a large deficit but not yet prefilled
        state.add_request("new-req", 1000, 0);

        let output = state.schedule_once();
        assert_eq!(output.assignments.len(), 1);
        // New request should be capped at max_num_scheduled_tokens / 4 = 32
        assert!(output.assignments[0].num_new_tokens <= 32);
    }
}
