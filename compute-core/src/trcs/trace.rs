use std::sync::Arc;
use crate::trcs::fact::{RelationId, CompactTuple};
use crate::trcs::arrangement::{RelationRun, SupportTable, PhysicalDeltaRowDyn};

pub type KeySpec = Vec<usize>; // Column indices defining the primary key

#[derive(Debug, Clone)]
pub struct RelationTrace {
    pub relation_id: RelationId,
    pub key_spec: KeySpec,
    pub base: Option<Arc<RelationRun>>,
    pub recent: Vec<Arc<RelationRun>>,
    pub visible_supports: SupportTable,
    pub generation: u64,
}

impl RelationTrace {
    pub fn new(relation_id: RelationId, key_spec: KeySpec) -> Self {
        Self {
            relation_id,
            key_spec,
            base: None,
            recent: Vec::new(),
            visible_supports: SupportTable::new(),
            generation: 0,
        }
    }

    /// K-way merge iterator returning current visible facts.
    /// This is an O(N) linear scan over the active support table for now,
    /// simulating a true merge over base + recent runs, returning canonical ordered output.
    pub fn visible_facts(&self) -> Vec<CompactTuple> {
        let mut facts: Vec<CompactTuple> = self.visible_supports.entries
            .iter()
            .filter(|(_, &support)| support > 0)
            .map(|(tuple, _)| tuple.clone())
            .collect();

        // Canonical deterministic ordering
        facts.sort_by(|a, b| a.columns.cmp(&b.columns));
        facts
    }
}
