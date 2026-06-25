use crate::trcs::fact::WeightedFact;
use crate::trcs::arrangement::{PhysicalDeltaRowDyn, SupportTable};
use crate::trcs::errors::TrcsError;
use std::collections::HashMap;

use crate::trcs::fact::{WeightedFact, RelationId, CompactTuple};
use crate::trcs::revision::RevisionFrontierId;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConsolidationKey {
    pub relation_id: RelationId,
    pub tuple: CompactTuple,
    pub frontier: RevisionFrontierId,
}

pub fn consolidate_updates(
    updates: Vec<WeightedFact>,
    support_table: &mut SupportTable,
    max_arity: usize,
) -> Result<(Vec<PhysicalDeltaRowDyn>, u64, u64), TrcsError> {
    let mut groups: HashMap<ConsolidationKey, i64> = HashMap::new();

    // Ensure all updates belong to the same relation or handle partitioning safely
    let rel_id = updates.first().map(|u| u.relation_id).unwrap_or(0);

    for update in updates {
        if update.relation_id != rel_id {
            return Err(TrcsError::InvalidSupport("Mixed-relation batches are not permitted in single-relation API".into()));
        }
        if update.tuple.columns.len() > max_arity {
            return Err(TrcsError::UnsupportedArity(update.tuple.columns.len()));
        }
        let key = ConsolidationKey {
            relation_id: update.relation_id,
            tuple: update.tuple,
            frontier: update.revision_frontier_id,
        };
        *groups.entry(key).or_insert(0) += update.diff as i64;
    }

    let mut staged_support = support_table.entries.clone();
    let mut physical_rows = Vec::new();
    let mut visible_insertions = 0;
    let mut visible_retractions = 0;

    let mut sorted_keys: Vec<_> = groups.keys().cloned().collect();
    // Deterministic order: tuple columns lexicographically, then frontier
    sorted_keys.sort_by(|a, b| {
        let cmp = a.tuple.columns.cmp(&b.tuple.columns);
        if cmp == std::cmp::Ordering::Equal {
            a.frontier.cmp(&b.frontier)
        } else {
            cmp
        }
    });

    for key in sorted_keys {
        let diff = groups[&key];
        if diff == 0 {
            continue;
        }

        if diff < i32::MIN as i64 || diff > i32::MAX as i64 {
            return Err(TrcsError::InvalidSupport("i32 narrowing overflow detected during physical delta conversion".into()));
        }

        // Support logic
        let logical_key = key.tuple.clone();
        let old_support = *staged_support.get(&logical_key).unwrap_or(&0);
        let new_support = old_support.checked_add(diff).ok_or_else(|| TrcsError::InvalidSupport("ArithmeticOverflow".into()))?;

        if new_support < 0 {
            return Err(TrcsError::InvalidSupport(format!(
                "Negative support calculated for tuple: {:?}",
                logical_key
            )));
        }

        if old_support == 0 && new_support > 0 {
            visible_insertions += 1;
        } else if old_support > 0 && new_support == 0 {
            visible_retractions += 1;
        }

        staged_support.insert(logical_key.clone(), new_support);
        if new_support == 0 {
            staged_support.remove(&logical_key);
        }

        physical_rows.push(PhysicalDeltaRowDyn {
            columns: key.tuple.columns.into_boxed_slice(),
            diff: diff as i32,
            revision_frontier_id: key.frontier,
            provenance_token: 0,
        });
    }

    support_table.entries = staged_support;

    Ok((physical_rows, visible_insertions, visible_retractions))
}
