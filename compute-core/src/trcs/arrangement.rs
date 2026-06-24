use std::sync::Arc;
use std::collections::HashMap;
use crate::trcs::fact::CompactTuple;
use crate::trcs::revision::RevisionFrontierId;
use crate::trcs::relation::{RunId, KeyOrder, StorageClass};
use crate::trcs::fact::RelationId;

/// The runtime-arity version of Phase 1's const-generic physical delta row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalDeltaRowDyn {
    pub columns: Box<[u32]>,
    pub diff: i32,
    pub revision_frontier_id: RevisionFrontierId,
    pub provenance_token: u64,
}

pub type RangeIndex = (); // Placeholder for key-range index

/// Replaces ArrangementRun with concrete storage
#[derive(Debug, Clone)]
pub struct RelationRun {
    pub run_id: RunId,
    pub relation_id: RelationId,
    pub key_order: KeyOrder,
    pub frontier_min: RevisionFrontierId,
    pub frontier_max: RevisionFrontierId,
    pub rows: Arc<[PhysicalDeltaRowDyn]>,
    pub key_index: Option<RangeIndex>,
    pub storage_class: StorageClass,
    pub checksum: u64,
}

pub type LogicalFactKey = CompactTuple;

/// The authoritative CPU-side support-count structure
#[derive(Debug, Clone)]
pub struct SupportTable {
    pub entries: HashMap<LogicalFactKey, i64>,
}

impl SupportTable {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
}
