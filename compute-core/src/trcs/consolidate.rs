use crate::trcs::fact::WeightedFact;
use crate::trcs::arrangement::{PhysicalDeltaRowDyn, SupportTable};
use crate::trcs::errors::TrcsError;
use std::collections::HashMap;

pub fn consolidate_updates(
    updates: Vec<WeightedFact>,
    support_table: &mut SupportTable,
    max_arity: usize,
) -> Result<(Vec<PhysicalDeltaRowDyn>, u64, u64), TrcsError> {
    // Group by key + frontier
    let mut groups: HashMap<(crate::trcs::fact::CompactTuple, crate::trcs::revision::RevisionFrontierId), i64> = HashMap::new();

    for update in updates {
        if update.tuple.columns.len() > max_arity {
            return Err(TrcsError::UnsupportedArity(update.tuple.columns.len()));
        }
        *groups.entry((update.tuple, update.revision_frontier_id)).or_insert(0) += update.diff as i64;
    }

    let mut staged_support = support_table.entries.clone();
    let mut physical_rows = Vec::new();
    let mut visible_insertions = 0;
    let mut visible_retractions = 0;

    let mut sorted_keys: Vec<_> = groups.keys().cloned().collect();
    sorted_keys.sort_by(|a, b| a.0.columns.cmp(&b.0.columns));

    for (key, frontier) in sorted_keys {
        let diff = groups[&(key.clone(), frontier)];
        if diff == 0 {
            continue;
        }

        let old_support = *staged_support.get(&key).unwrap_or(&0);
        let new_support = old_support + diff;

        if new_support < 0 {
            return Err(TrcsError::InvalidSupport(format!(
                "Negative support calculated for tuple: {:?}",
                key
            )));
        }

        if old_support == 0 && new_support > 0 {
            visible_insertions += 1;
        } else if old_support > 0 && new_support == 0 {
            visible_retractions += 1;
        }

        staged_support.insert(key.clone(), new_support);

        if new_support == 0 {
            staged_support.remove(&key);
        }

        physical_rows.push(PhysicalDeltaRowDyn {
            columns: key.columns.into_boxed_slice(),
            diff: diff as i32,
            revision_frontier_id: frontier,
            provenance_token: 0,
        });
    }

    // Commit support table since transaction bounds held
    support_table.entries = staged_support;

    Ok((physical_rows, visible_insertions, visible_retractions))
}
