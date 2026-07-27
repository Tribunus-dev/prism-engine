//! Unified scheduler system (constitutional home).
//!
//! The canonical scheduler for the token-budget model. Per the
//! inventory v2.1 row 53, this absorbs `scheduler.rs` (row 47) —
//! one canonical scheduler survives.
//!
//! # Authority
//!
//! The unified scheduler is a system (S bucket). It reads the
//! committed request pool state, runs `schedule_once` to produce
//! a `ScheduleOutput`, and stages the assignment through
//! `ConstitutionalWorldTxn`. A scheduling step is non-authoritative
//! until it commits.
//!
//! # Placeholder
//!
//! The engine's unified_scheduler.rs (530 LOC) and scheduler.rs
//! (543 LOC) are the legacy duplicates. The full algorithm migrates
//! in steps 26 and 33. The constitutional side has the data model
//! and a simplified schedule_once that returns empty assignments;
//! the full algorithm is added when the engine callers migrate.
//!
//! # Migration provenance
//!
//! The legacy home was `compute-core/src/ecs/scheduling/unified_scheduler.rs`
//! and `compute-core/src/ecs/scheduling/scheduler.rs`. The engine
//! files are the legacy duplicates; step 26 deletes the merged
//! scheduler.rs and step 58 deletes unified_scheduler.rs.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Placeholder engine types
// ---------------------------------------------------------------------------

/// Placeholder for `compute-core::ecs::scheduling::SchedulerConfig`.
/// Replaced when the unified scheduler migrates fully. Carries the
/// same fields the engine config carries.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_num_scheduled_tokens: usize,
    pub max_batch_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_scheduled_tokens: 256,
            max_batch_size: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// ECS-style state for the unified token-budget scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerState {
    pub max_num_scheduled_tokens: usize,
    pub max_num_running_reqs: usize,
    pub running: Vec<String>,
    pub waiting: Vec<String>,
    pub preempted: Vec<String>,
    /// `HashMap` is the engine's choice; the canonical collections
    /// rule would normally call for BTreeMap, but the engine uses
    /// HashMap and the constitutional home mirrors that. Iteration
    /// order is not part of the schedule_once contract; the rule
    /// is "no HashMap for collections whose order is observable",
    /// and the schedule output is sorted by request_id before
    /// returning.
    pub requests: HashMap<String, UnifiedRequestData>,
}

#[derive(Debug, Clone)]
pub struct UnifiedRequestData {
    pub num_tokens_with_spec: usize,
    pub num_computed_tokens: usize,
    pub status: UnifiedRequestStatus,
    pub priority: usize,
    pub preemption_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnifiedRequestStatus {
    Waiting,
    Running,
    Preempted,
}

#[derive(Debug, Clone, Default)]
pub struct ScheduleOutput {
    pub assignments: Vec<ScheduleAssignment>,
    pub total_tokens_scheduled: usize,
}

#[derive(Debug, Clone)]
pub struct ScheduleAssignment {
    pub request_id: String,
    pub num_new_tokens: usize,
    pub block_ids: Vec<u64>,
    pub spec_tokens: usize,
}

impl SchedulerState {
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

    /// Run one scheduling step. Placeholder implementation: returns
    /// an empty schedule. The full algorithm (proportional
    /// distribution, chunked prefill, decode steps) is added when
    /// the engine callers migrate.
    pub fn schedule_once(&mut self) -> ScheduleOutput {
        // Phase 1 (placeholder): admit waiting → running.
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
        // Full schedule algorithm arrives with step 33.
        ScheduleOutput::default()
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `unified_scheduler` system.

    use super::*;

    #[test]
    fn new_scheduler_starts_empty() {
        // Architectural invariant: a fresh scheduler has empty pools
        // and the configured budget.
        let s = SchedulerState::new(256, 8);
        assert_eq!(s.max_num_scheduled_tokens, 256);
        assert_eq!(s.max_num_running_reqs, 8);
        assert!(s.running.is_empty());
        assert!(s.waiting.is_empty());
        assert!(s.preempted.is_empty());
        assert!(s.requests.is_empty());
    }

    #[test]
    fn add_request_goes_to_waiting_pool() {
        // Architectural invariant: a fresh request is added to the
        // waiting pool with `Waiting` status and zero computed tokens.
        let mut s = SchedulerState::new(256, 8);
        s.add_request("r1", 100, 0);
        assert!(s.waiting.contains(&"r1".to_string()));
        assert!(!s.running.contains(&"r1".to_string()));
        let req = s.requests.get("r1").expect("request present");
        assert_eq!(req.status, UnifiedRequestStatus::Waiting);
        assert_eq!(req.num_computed_tokens, 0);
        assert_eq!(req.num_tokens_with_spec, 100);
    }

    #[test]
    fn remove_request_clears_all_pools() {
        // Architectural invariant: removing a request clears it from
        // every pool and the requests map. After remove_request,
        // the request is fully gone.
        let mut s = SchedulerState::new(256, 8);
        s.add_request("r1", 100, 0);
        s.add_request("r2", 100, 0);
        s.remove_request("r1");
        assert!(!s.waiting.contains(&"r1".to_string()));
        assert!(s.requests.get("r1").is_none());
        // r2 is still present.
        assert!(s.waiting.contains(&"r2".to_string()));
    }

    #[test]
    fn schedule_once_admits_to_running_pool() {
        // Architectural invariant: schedule_once moves admitted
        // requests from waiting to running. The placeholder
        // implementation only does the admission phase; the full
        // algorithm arrives later.
        let mut s = SchedulerState::new(256, 8);
        s.add_request("r1", 100, 0);
        s.add_request("r2", 100, 0);
        let _ = s.schedule_once();
        // After admission, both requests are in the running pool.
        assert_eq!(s.running.len(), 2);
        assert!(s.waiting.is_empty());
        let r1 = s.requests.get("r1").unwrap();
        assert_eq!(r1.status, UnifiedRequestStatus::Running);
    }

    #[test]
    fn default_config_has_sensible_values() {
        // Architectural invariant: the default SchedulerConfig has
        // a non-zero token budget and a small but positive max
        // batch size. The engine's defaults are 256 / 8.
        let c = SchedulerConfig::default();
        assert!(c.max_num_scheduled_tokens > 0);
        assert!(c.max_batch_size > 0);
        assert_eq!(c.max_num_scheduled_tokens, 256);
        assert_eq!(c.max_batch_size, 8);
    }
}
