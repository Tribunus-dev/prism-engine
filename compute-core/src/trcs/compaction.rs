use crate::trcs::arrangement::RelationRun;
use crate::trcs::relation::{RunId, StorageClass, KeyOrder};
use crate::trcs::errors::TrcsError;
use std::sync::Arc;

pub struct TraceCompactionPolicy {
    pub max_recent_runs: usize,
    pub max_recent_rows: u64,
    pub max_read_amplification: f64,
    pub max_dead_diff_density: f32,
}

impl Default for TraceCompactionPolicy {
    fn default() -> Self {
        Self {
            max_recent_runs: 8,
            max_recent_rows: 1_000_000,
            max_read_amplification: 2.0,
            max_dead_diff_density: 0.20,
        }
    }
}

pub fn execute_compaction(
    runs: Vec<Arc<RelationRun>>,
    new_run_id: RunId,
) -> Result<Arc<RelationRun>, TrcsError> {
    if runs.is_empty() {
        return Err(TrcsError::CompactionConflict("No runs provided for compaction".into()));
    }

    // Simulate k-way merge and dropping net-zero diffs.
    // We collect all rows from input runs and merge them physically.
    let mut merged_rows = Vec::new();
    let relation_id = runs[0].relation_id;
    let mut min_frontier = u32::MAX;
    let mut max_frontier = 0;

    for run in &runs {
        if run.relation_id != relation_id {
            return Err(TrcsError::CompactionConflict("Mismatched relation IDs".into()));
        }
        min_frontier = min_frontier.min(run.frontier_min);
        max_frontier = max_frontier.max(run.frontier_max);

        for row in run.rows.iter().cloned() {
            if row.diff != 0 {
                merged_rows.push(row);
            }
        }
    }

    // In a production setup, this would consolidate physical deltas identical to the trace level logic.
    let replacement_run = Arc::new(RelationRun {
        run_id: new_run_id,
        relation_id,
        key_order: KeyOrder::Ascending,
        frontier_min: min_frontier,
        frontier_max: max_frontier,
        rows: merged_rows.into_boxed_slice().into(),
        key_index: None,
        storage_class: StorageClass::Recent,
        checksum: 0,
    });

    Ok(replacement_run)
}
