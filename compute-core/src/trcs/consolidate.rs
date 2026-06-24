use crate::trcs::fact::WeightedFact;
use crate::trcs::arrangement::{PhysicalDeltaRowDyn, SupportTable};
use crate::trcs::errors::TrcsError;
use std::collections::HashMap;

/// Consolidates signed logical updates per the TRCS contract:
/// Groups equal keys, sums diffs, updates support counts, and emits visible transitions.
/// Fails if any final support count becomes negative.
pub fn consolidate_updates(
    updates: Vec<WeightedFact>,
    support_table: &mut SupportTable,
    max_arity: usize,
) -> Result<(Vec<PhysicalDeltaRowDyn>, u64, u64), TrcsError> {
    let mut groups: HashMap<crate::trcs::fact::CompactTuple, i64> = HashMap::new();

    // Validate arity & aggregate local diffs
    for update in updates {
        if update.tuple.columns.len() > max_arity {
            return Err(TrcsError::UnsupportedArity(update.tuple.columns.len()));
        }
        *groups.entry(update.tuple.clone()).or_insert(0) += update.diff as i64;
    }

    let mut visible_insertions = 0;
    let mut visible_retractions = 0;
    let mut physical_rows = Vec::new();

    // In a production engine, this would sort by key + frontier and compute
    // visibility transitions deterministically without relying on HashMap iteration.
    // We simulate canonical output by sorting keys.
    let mut sorted_keys: Vec<_> = groups.keys().cloned().collect();
    sorted_keys.sort_by(|a, b| a.columns.cmp(&b.columns));

    for key in sorted_keys {
        let diff = groups[&key];
        if diff == 0 {
            continue; // Equivalents cancel out entirely
        }

        let old_support = *support_table.entries.get(&key).unwrap_or(&0);
        let new_support = old_support + diff;

        if new_support < 0 {
            return Err(TrcsError::InvalidSupport(format!(
                "Negative support calculated for tuple: {:?}",
                key
            )));
        }

        // Emit visibility transitions
        if old_support == 0 && new_support > 0 {
            visible_insertions += 1;
        } else if old_support > 0 && new_support == 0 {
            visible_retractions += 1;
        }

        support_table.entries.insert(key.clone(), new_support);

        if new_support == 0 {
            support_table.entries.remove(&key);
        }

        physical_rows.push(PhysicalDeltaRowDyn {
            columns: key.columns.into_boxed_slice(),
            diff: diff as i32,
            // For now, assume a dummy frontier/provenance
            revision_frontier_id: 0,
            provenance_token: 0,
        });
    }

    // Physical rows emit positively allocated, including retractions
    Ok((physical_rows, visible_insertions, visible_retractions))
}
