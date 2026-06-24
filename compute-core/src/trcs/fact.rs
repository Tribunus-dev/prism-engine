use crate::trcs::revision::RevisionFrontierId;
use std::hash::Hash;

pub type FactId = u64;
pub type RelationId = u32;

/// A compact representation of a tuple's columns (e.g., EntityIds or literal values).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompactTuple {
    pub columns: Vec<u32>,
}

/// A logical update represented in the TRCS incremental EDB protocol.
/// A fact is visible when its accumulated weight is positive.
/// A fact is retracted when its accumulated weight reaches zero.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WeightedFact {
    pub fact_id: FactId,
    pub relation_id: RelationId,
    pub tuple: CompactTuple,
    pub revision_frontier_id: RevisionFrontierId,
    pub diff: i32,
}
