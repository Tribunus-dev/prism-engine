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
