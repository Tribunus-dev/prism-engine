use crate::trcs::revision::RevisionFrontierId;
use crate::trcs::fact::{RelationId};
use std::collections::HashMap;

pub type CompilationId = u64;
pub type AnalysisId = u64;
pub type BackendId = u64;
pub type RuleId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalStatus {
    Success,
    CapacityEvent,
    PrecisionWidened,
    Fallback,
    Cancelled,
    TimedOut,
    Quarantined,
    Failed,
}

#[derive(Debug, Clone)]
pub struct TraceRunSummary {
    pub total_runs: u32,
    pub active_runs: u32,
    pub base_rows: u64,
    pub recent_rows: u64,
}

#[derive(Debug, Clone)]
pub struct CardinalityStats {
    pub distinct_keys: u64,
    pub total_rows: u64,
    pub max_rows_per_key: u64,
}

// Stubs for receipts that will be expanded in later phases
pub type PrecisionReceiptId = u64;
pub type CapacityEventId = u64;
pub type FallbackEventId = u64;
pub type SummaryUsageId = u64;
pub type AssertionUsageId = u64;
pub type SpeculationTicketId = u64;

#[derive(Debug, Clone)]
pub struct StratificationBarrierReceipt {
    pub lower_stratum_id: u32,
    pub upper_stratum_id: u32,
    pub negative_dependency_relation: RelationId,
    pub sealed_frontier: RevisionFrontierId,
    pub lower_stratum_converged: bool,
    pub consolidation_hash: u64,
    pub reopened_due_to_retraction: bool,
}

#[derive(Debug, Clone)]
pub struct BackendFaultReceipt {
    pub backend_id: BackendId,
    pub error_code: u32,
    pub description: String,
}

/// Comprehensive receipt containing exact semantic, allocation, and diagnostic data.
#[derive(Debug, Clone)]
pub struct AnalysisExecutionReceipt {
    pub compilation_id: CompilationId,
    pub revision_frontier_id: RevisionFrontierId,
    pub analysis_id: AnalysisId,
    pub backend_id: BackendId,
    pub phaseir_hash: u64,
    pub semantic_program_hash: u64,
    pub relation_schema_hash: u64,

    pub final_status: FinalStatus,
    pub converged: bool,
    pub epochs: u32,
    pub bulk_load_used: bool,

    pub trace_run_summary: TraceRunSummary,
    pub rule_execution_counts: HashMap<RuleId, u64>,
    pub relation_cardinality_summary: HashMap<RelationId, CardinalityStats>,

    pub max_delta_cardinality: u64,
    pub max_intermediate_cardinality: u64,
    pub host_memory_peak: u64,
    pub device_memory_peak: u64,
    pub provenance_spill_bytes: u64,

    pub explicit_materialization_count: u64,
    /// Must always be zero.
    pub hidden_copy_count: u64,

    pub widening_events: Vec<PrecisionReceiptId>,
    pub capacity_events: Vec<CapacityEventId>,
    pub fallback_events: Vec<FallbackEventId>,

    pub imported_summary_usage: Vec<SummaryUsageId>,
    pub assertion_usage: Vec<AssertionUsageId>,
    pub speculation_usage: Vec<SpeculationTicketId>,

    pub stratification_barriers: Vec<StratificationBarrierReceipt>,
    pub backend_fault: Option<BackendFaultReceipt>,
    pub determinism_hash: u64,
}

// Phase 2 Concrete Receipts
#[derive(Debug, Clone)]
pub struct BulkLoadReceipt {
    pub relation_id: RelationId,
    pub input_rows: u64,
    pub consolidated_rows: u64,
    pub visible_rows: u64,
    pub base_run_id: crate::trcs::relation::RunId,
    pub bulk_load_used: bool,
    pub determinism_hash: u64,
}

#[derive(Debug, Clone)]
pub struct DeltaApplyReceipt {
    pub relation_id: RelationId,
    pub frontier: RevisionFrontierId,
    pub input_rows: u64,
    pub consolidated_rows: u64,
    pub visible_insertions: u64,
    pub visible_retractions: u64,
    pub support_only_updates: u64,
    pub recent_run_id: crate::trcs::relation::RunId,
    pub determinism_hash: u64,
}

#[derive(Debug, Clone)]
pub struct CompactionReceipt {
    pub relation_id: RelationId,
    pub input_run_ids: Vec<crate::trcs::relation::RunId>,
    pub output_run_id: crate::trcs::relation::RunId,
    pub rows_before: u64,
    pub rows_after: u64,
    pub dead_rows_removed: u64,
    pub published_generation: u64,
    pub determinism_hash_before: u64,
    pub determinism_hash_after: u64,
}
