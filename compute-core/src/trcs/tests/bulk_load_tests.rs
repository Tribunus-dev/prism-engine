use crate::trcs::fact::{CompactTuple, WeightedFact};
use crate::trcs::relation::BulkLoadEligibility;
use crate::trcs::runtime::{CpuTrcsRuntime, TrcsRelationRuntime};

#[test]
fn test_bulk_load_equivalence() {
    let mut runtime = CpuTrcsRuntime::new(16);
    let mut facts = Vec::new();
    for i in 0..2000 {
        facts.push(WeightedFact {
            fact_id: i,
            relation_id: 1,
            tuple: CompactTuple { columns: vec![i as u32, i as u32 + 1] },
            revision_frontier_id: 1,
            diff: 1,
        });
    }

    let eligibility = BulkLoadEligibility {
        full_relation_empty: true,
        delta_to_full_ratio: 1.0,
        input_fact_count: 2000,
        estimated_incremental_overhead: 1000,
    };

    let receipt = runtime.bulk_load(1, facts, eligibility).unwrap();
    assert_eq!(receipt.visible_rows, 2000);
    assert_eq!(runtime.visible_facts(1).unwrap().len(), 2000);
}
#[test]
fn trcs_bulk_load_rejects_mixed_relation_input() {
    let mut runtime = CpuTrcsRuntime::new(16);
    let facts = vec![
        WeightedFact { fact_id: 1, relation_id: 1, tuple: CompactTuple { columns: vec![1] }, revision_frontier_id: 1, diff: 1 },
        WeightedFact { fact_id: 2, relation_id: 2, tuple: CompactTuple { columns: vec![2] }, revision_frontier_id: 1, diff: 1 },
    ];
    let eligibility = BulkLoadEligibility { full_relation_empty: true, delta_to_full_ratio: 1.0, input_fact_count: 1024, estimated_incremental_overhead: 1000 };
    let res = runtime.bulk_load(1, facts, eligibility);
    assert!(res.is_err());
}
