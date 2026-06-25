use crate::trcs::fact::{CompactTuple, WeightedFact};
use crate::trcs::runtime::{CpuTrcsRuntime, TrcsRelationRuntime};

#[test]
fn test_consolidation_correctness() {
    let mut runtime = CpuTrcsRuntime::new(16);
    let facts = vec![
        WeightedFact { fact_id: 1, relation_id: 1, tuple: CompactTuple { columns: vec![1, 2] }, revision_frontier_id: 1, diff: 1 },
        WeightedFact { fact_id: 2, relation_id: 1, tuple: CompactTuple { columns: vec![1, 2] }, revision_frontier_id: 1, diff: 1 },
        WeightedFact { fact_id: 3, relation_id: 1, tuple: CompactTuple { columns: vec![1, 2] }, revision_frontier_id: 1, diff: -1 },
    ];

    let receipt = runtime.apply_delta(1, 1, facts).unwrap();
    // One net visible fact because support 2 - 1 = 1 > 0
    assert_eq!(receipt.visible_insertions, 1);
    assert_eq!(receipt.visible_retractions, 0);

    let final_facts = runtime.visible_facts(1).unwrap();
    assert_eq!(final_facts.len(), 1);
}
#[test]
fn trcs_consolidation_preserves_frontier() {
    let mut runtime = CpuTrcsRuntime::new(16);
    let facts = vec![
        WeightedFact { fact_id: 1, relation_id: 1, tuple: CompactTuple { columns: vec![1, 2] }, revision_frontier_id: 5, diff: 1 },
    ];
    let receipt = runtime.apply_delta(1, 5, facts).unwrap();
    assert_eq!(receipt.visible_insertions, 1);

    let trace = runtime.traces.get(&1).unwrap();
    let recent_run = trace.recent.last().unwrap();
    assert_eq!(recent_run.rows[0].revision_frontier_id, 5);
}

#[test]
fn trcs_consolidation_is_transactional_on_negative_support() {
    let mut runtime = CpuTrcsRuntime::new(16);
    let facts = vec![
        WeightedFact { fact_id: 1, relation_id: 1, tuple: CompactTuple { columns: vec![1, 2] }, revision_frontier_id: 1, diff: -1 },
    ];
    // Must error and abort
    let res = runtime.apply_delta(1, 1, facts);
    assert!(res.is_err());

    // Trace should be completely empty and unmodified
    let visible = runtime.visible_facts(1).unwrap();
    assert_eq!(visible.len(), 0);
}

#[test]
fn consolidation_sorts_relation_tuple_frontier_deterministically() {
    let mut runtime = CpuTrcsRuntime::new(16);
    let facts = vec![
        WeightedFact { fact_id: 1, relation_id: 1, tuple: CompactTuple { columns: vec![2, 2] }, revision_frontier_id: 2, diff: 1 },
        WeightedFact { fact_id: 2, relation_id: 1, tuple: CompactTuple { columns: vec![1, 1] }, revision_frontier_id: 1, diff: 1 },
        WeightedFact { fact_id: 3, relation_id: 1, tuple: CompactTuple { columns: vec![1, 1] }, revision_frontier_id: 2, diff: 1 },
    ];
    let receipt = runtime.apply_delta(1, 1, facts).unwrap();
    let trace = runtime.traces.get(&1).unwrap();
    let recent_run = trace.recent.last().unwrap();

    assert_eq!(recent_run.rows[0].columns.as_ref(), &[1, 1]);
    assert_eq!(recent_run.rows[0].revision_frontier_id, 1); // 1 comes before 2
    assert_eq!(recent_run.rows[1].columns.as_ref(), &[1, 1]);
    assert_eq!(recent_run.rows[1].revision_frontier_id, 2);
    assert_eq!(recent_run.rows[2].columns.as_ref(), &[2, 2]);
}

#[test]
fn consolidation_rejects_mixed_relation_batches() {
    let mut runtime = CpuTrcsRuntime::new(16);
    let facts = vec![
        WeightedFact { fact_id: 1, relation_id: 1, tuple: CompactTuple { columns: vec![1] }, revision_frontier_id: 1, diff: 1 },
        WeightedFact { fact_id: 2, relation_id: 2, tuple: CompactTuple { columns: vec![2] }, revision_frontier_id: 1, diff: 1 },
    ];
    let result = runtime.apply_delta(1, 1, facts);
    assert!(result.is_err());
}
