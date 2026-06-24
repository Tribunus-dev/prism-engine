use crate::trcs::fact::{RelationId};
use crate::trcs::revision::RevisionFrontierId;

pub type RunId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyOrder {
    Ascending,
    Descending,
    Unsorted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    Base,
    Recent,
    Compacting,
    Retired,
}

pub type Key = u64;

/// Represents a contiguous sequence of consolidated rows in a trace.
#[derive(Debug, Clone)]
pub struct ArrangementRun {
    pub run_id: RunId,
    pub relation_id: RelationId,
    pub key_order: KeyOrder,
    pub frontier_min: RevisionFrontierId,
    pub frontier_max: RevisionFrontierId,
    pub row_count: u64,
    pub positive_rows: u64,
    pub negative_rows: u64,
    pub key_min: Key,
    pub key_max: Key,
    pub dead_diff_density: f32,
    pub storage_class: StorageClass,
}

pub type Cost = u64;

/// Determines if a full relation evaluation should bypass differential updates
/// and perform a dense bulk-load operation.
#[derive(Debug, Clone, Copy)]
pub struct BulkLoadEligibility {
    pub full_relation_empty: bool,
    pub delta_to_full_ratio: f64,
    pub input_fact_count: u64,
    pub estimated_incremental_overhead: Cost,
}
