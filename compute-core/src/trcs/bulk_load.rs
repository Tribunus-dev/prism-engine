use crate::trcs::relation::BulkLoadEligibility;
use crate::trcs::fact::WeightedFact;
use crate::trcs::errors::TrcsError;
use crate::trcs::arrangement::{RelationRun, SupportTable, PhysicalDeltaRowDyn};
use crate::trcs::consolidate::consolidate_updates;
use crate::trcs::relation::{KeyOrder, StorageClass};
use std::sync::Arc;

pub struct BulkLoadPolicy {
    pub min_rows: u64,
    pub min_authoritative_ratio: f64,
    pub max_incremental_bootstrap_cost: u64,
}

impl Default for BulkLoadPolicy {
    fn default() -> Self {
        Self {
            min_rows: 1024,
            min_authoritative_ratio: 0.95,
            max_incremental_bootstrap_cost: 500, // caller-calibrated baseline
        }
    }
}

impl BulkLoadEligibility {
    pub fn should_bulk_load(&self, policy: &BulkLoadPolicy) -> bool {
        self.full_relation_empty
            && self.input_fact_count >= policy.min_rows
            && self.delta_to_full_ratio >= policy.min_authoritative_ratio
            && self.estimated_incremental_overhead >= policy.max_incremental_bootstrap_cost
    }
}

pub fn execute_bulk_load(
    relation_id: u32,
    snapshot: Vec<WeightedFact>,
    max_arity: usize,
) -> Result<(Arc<RelationRun>, SupportTable, u64, u64), TrcsError> {
    let mut support = SupportTable::new();
    let initial_count = snapshot.len() as u64;

    // Validate schema, canonicalize, sort, consolidate duplicate signed rows
    let (physical_rows, visible_insertions, _) = consolidate_updates(snapshot, &mut support, max_arity)?;

    let base_run = Arc::new(RelationRun {
        run_id: 1, // First run
        relation_id,
        key_order: KeyOrder::Ascending,
        frontier_min: 0,
        frontier_max: 0,
        rows: physical_rows.into_boxed_slice().into(),
        key_index: None,
        storage_class: StorageClass::Base,
        checksum: 0, // In practice, hash physical rows
    });

    Ok((base_run, support, initial_count, visible_insertions))
}
