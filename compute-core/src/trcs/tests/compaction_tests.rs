use crate::trcs::fact::{CompactTuple, WeightedFact};
use crate::trcs::runtime::{CpuTrcsRuntime, TrcsRelationRuntime};

#[test]
fn test_compaction_preserves_facts() {
    let mut runtime = CpuTrcsRuntime::new(16);

    for i in 0..10 {
        let facts = vec![
            WeightedFact { fact_id: i, relation_id: 1, tuple: CompactTuple { columns: vec![i as u32, i as u32] }, revision_frontier_id: i as u32, diff: 1 },
        ];
        runtime.apply_delta(1, i as u32, facts).unwrap();
    }

    let summary_before = runtime.trace_summary(1).unwrap();
    assert_eq!(summary_before.recent_rows, 10);
    assert_eq!(summary_before.active_runs, 10);

    let receipt = runtime.maybe_compact(1).unwrap().unwrap();
    assert_eq!(receipt.rows_after, 10); // 10 unique rows

    let summary_after = runtime.trace_summary(1).unwrap();
    assert_eq!(summary_after.active_runs, 1);
    assert_eq!(summary_after.recent_rows, 10);

    let visible = runtime.visible_facts(1).unwrap();
    assert_eq!(visible.len(), 10);
}
