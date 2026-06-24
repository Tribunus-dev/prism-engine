use crate::trcs::fact::{CompactTuple, WeightedFact};
use crate::trcs::runtime::{CpuTrcsRuntime, TrcsRelationRuntime};

#[test]
fn test_trace_union() {
    let mut runtime = CpuTrcsRuntime::new(16);
    let facts = vec![
        WeightedFact { fact_id: 1, relation_id: 1, tuple: CompactTuple { columns: vec![1, 1] }, revision_frontier_id: 1, diff: 1 },
    ];
    runtime.apply_delta(1, 1, facts).unwrap();

    let facts2 = vec![
        WeightedFact { fact_id: 2, relation_id: 1, tuple: CompactTuple { columns: vec![2, 2] }, revision_frontier_id: 2, diff: 1 },
    ];
    runtime.apply_delta(1, 2, facts2).unwrap();

    let visible = runtime.visible_facts(1).unwrap();
    assert_eq!(visible.len(), 2); // Union of both recent runs
    assert_eq!(visible[0].columns, vec![1, 1]); // sorted order
    assert_eq!(visible[1].columns, vec![2, 2]);
}
