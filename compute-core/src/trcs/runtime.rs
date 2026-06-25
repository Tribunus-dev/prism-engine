use crate::trcs::fact::{RelationId, WeightedFact, CompactTuple};
use crate::trcs::revision::RevisionFrontierId;
use crate::trcs::relation::{BulkLoadEligibility, RunId, StorageClass, KeyOrder};
use crate::trcs::receipts::{BulkLoadReceipt, DeltaApplyReceipt, CompactionReceipt, TraceRunSummary};
use crate::trcs::errors::TrcsError;
use crate::trcs::trace::RelationTrace;
use crate::trcs::bulk_load::{execute_bulk_load, BulkLoadPolicy};
use crate::trcs::consolidate::consolidate_updates;
use crate::trcs::compaction::execute_compaction;
use std::collections::HashMap;
use crate::trcs::arrangement::RelationRun;
use std::sync::Arc;

pub type VisibleFact = CompactTuple;

pub trait TrcsRelationRuntime {
    fn bulk_load(
        &mut self,
        relation_id: RelationId,
        snapshot: Vec<WeightedFact>,
        eligibility: BulkLoadEligibility,
    ) -> Result<BulkLoadReceipt, TrcsError>;

    fn apply_delta(
        &mut self,
        relation_id: RelationId,
        frontier: RevisionFrontierId,
        updates: Vec<WeightedFact>,
    ) -> Result<DeltaApplyReceipt, TrcsError>;

    fn visible_facts(
        &self,
        relation_id: RelationId,
    ) -> Result<Vec<VisibleFact>, TrcsError>;

    fn maybe_compact(
        &mut self,
        relation_id: RelationId,
    ) -> Result<Option<CompactionReceipt>, TrcsError>;

    fn trace_summary(
        &self,
        relation_id: RelationId,
    ) -> Result<TraceRunSummary, TrcsError>;
}

pub struct CpuTrcsRuntime {
    pub traces: HashMap<RelationId, RelationTrace>,
    pub max_arity: usize,
    pub next_run_id: RunId,
}

impl CpuTrcsRuntime {
    pub fn new(max_arity: usize) -> Self {
        Self {
            traces: HashMap::new(),
            max_arity,
            next_run_id: 1,
        }
    }
}

impl TrcsRelationRuntime for CpuTrcsRuntime {
    fn bulk_load(
        &mut self,
        relation_id: RelationId,
        snapshot: Vec<WeightedFact>,
        eligibility: BulkLoadEligibility,
    ) -> Result<BulkLoadReceipt, TrcsError> {
        let policy = BulkLoadPolicy::default();
        if !eligibility.should_bulk_load(&policy) {
            return Err(TrcsError::InvalidBulkLoadPlan("Eligibility conditions unmet".into()));
        }

        let trace = self.traces.entry(relation_id).or_insert_with(|| RelationTrace::new(relation_id, vec![]));

        let run_id_alloc = self.next_run_id;
        let (base_run, support, input_rows, visible_rows) = execute_bulk_load(run_id_alloc, relation_id, snapshot, self.max_arity)?;
        let run_id = base_run.run_id;
        trace.base = Some(base_run);
        trace.recent.clear();
        trace.visible_supports = support;
        trace.generation += 1;
        self.next_run_id += 1;

        Ok(BulkLoadReceipt {
            relation_id,
            input_rows,
            consolidated_rows: visible_rows, // Approximation
            visible_rows,
            base_run_id: run_id,
            bulk_load_used: true,
            determinism_hash: trace.generation,
        })
    }

    fn apply_delta(
        &mut self,
        relation_id: RelationId,
        frontier: RevisionFrontierId,
        updates: Vec<WeightedFact>,
    ) -> Result<DeltaApplyReceipt, TrcsError> {
        let input_rows = updates.len() as u64;
        let trace = self.traces.entry(relation_id).or_insert_with(|| RelationTrace::new(relation_id, vec![]));

        let (physical_rows, visible_insertions, visible_retractions) = consolidate_updates(updates, &mut trace.visible_supports, self.max_arity)?;
        let consolidated_rows = physical_rows.len() as u64;

        let run_id = self.next_run_id;
        self.next_run_id += 1;

        let recent_run = Arc::new(RelationRun {
            run_id,
            relation_id,
            key_order: KeyOrder::Ascending,
            frontier_min: frontier,
            frontier_max: frontier,
            rows: physical_rows.into_boxed_slice().into(),
            key_index: None,
            storage_class: StorageClass::Recent,
            checksum: 0,
        });

        trace.recent.push(recent_run);
        trace.generation += 1;

        Ok(DeltaApplyReceipt {
            relation_id,
            frontier,
            input_rows,
            consolidated_rows,
            visible_insertions,
            visible_retractions,
            support_only_updates: consolidated_rows - visible_insertions - visible_retractions,
            recent_run_id: run_id,
            determinism_hash: trace.generation,
        })
    }

    fn visible_facts(&self, relation_id: RelationId) -> Result<Vec<VisibleFact>, TrcsError> {
        if let Some(trace) = self.traces.get(&relation_id) {
            Ok(trace.visible_facts())
        } else {
            Ok(Vec::new())
        }
    }

    fn maybe_compact(&mut self, relation_id: RelationId) -> Result<Option<CompactionReceipt>, TrcsError> {
        let trace = self.traces.get_mut(&relation_id).ok_or_else(|| TrcsError::CompactionConflict("Unknown relation".into()))?;
        if trace.recent.len() < 8 { // Simplified threshold
            return Ok(None);
        }

        let input_run_ids = trace.recent.iter().map(|r| r.run_id).collect();
        let rows_before = trace.recent.iter().map(|r| r.rows.len() as u64).sum();

        let new_run_id = self.next_run_id;
        self.next_run_id += 1;

        let output_run = execute_compaction(trace.recent.clone(), new_run_id)?;
        let rows_after = output_run.rows.len() as u64;

        trace.recent = vec![output_run];
        trace.generation += 1;

        Ok(Some(CompactionReceipt {
            relation_id,
            input_run_ids,
            output_run_id: new_run_id,
            rows_before,
            rows_after,
            dead_rows_removed: rows_before - rows_after,
            published_generation: trace.generation,
            determinism_hash_before: trace.generation - 1,
            determinism_hash_after: trace.generation,
        }))
    }

    fn trace_summary(&self, relation_id: RelationId) -> Result<TraceRunSummary, TrcsError> {
        let trace = self.traces.get(&relation_id).ok_or_else(|| TrcsError::CompactionConflict("Unknown relation".into()))?;
        Ok(TraceRunSummary {
            total_runs: (if trace.base.is_some() { 1 } else { 0 }) + trace.recent.len() as u32,
            active_runs: (if trace.base.is_some() { 1 } else { 0 }) + trace.recent.len() as u32,
            base_rows: trace.base.as_ref().map_or(0, |b| b.rows.len() as u64),
            recent_rows: trace.recent.iter().map(|r| r.rows.len() as u64).sum(),
        })
    }
}
