//! Scheduler-facing coalescing for semantic-region physical realizations.

use crate::semantic_region::{PhysicalRegionPlan, PhysicalRegionRealization};
use serde::{Deserialize, Serialize};
use std::ops::Range;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoalescedRegionView {
    pub region_ids: Vec<String>,
    pub packed_buffer: String,
    pub byte_range: Range<u64>,
    pub execution_lane: String,
    pub residency_class: String,
    pub conversion_ops: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionScheduleProjection {
    pub semantic_plan_digest: String,
    pub views: Vec<CoalescedRegionView>,
    pub raw_region_count: usize,
    pub scheduled_view_count: usize,
    pub cross_lane_boundaries: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RegionScheduleError {
    #[error("physical region has no byte range")]
    MissingRange,
    #[error("physical region contains fragmented ranges that require explicit materialization")]
    FragmentedRegion,
    #[error("scheduler view budget exceeded")]
    ViewBudgetExceeded,
}

pub fn project_region_schedule(
    plan: &PhysicalRegionPlan,
    max_views: usize,
) -> Result<RegionScheduleProjection, RegionScheduleError> {
    let mut ordered = plan.realizations.clone();
    ordered.sort_by_key(|realization| {
        realization.byte_ranges.first().map(|range| range.start).unwrap_or(u64::MAX)
    });
    let mut views: Vec<CoalescedRegionView> = Vec::new();
    for realization in &ordered {
        let range = single_range(realization)?;
        if let Some(last) = views.last_mut() {
            let compatible = last.packed_buffer == realization.packed_buffer
                && last.execution_lane == realization.execution_lane
                && last.residency_class == realization.residency_class
                && last.conversion_ops == realization.conversion_ops
                && last.byte_range.end == range.start;
            if compatible {
                last.region_ids.push(realization.semantic_region.0.clone());
                last.byte_range.end = range.end;
                continue;
            }
        }
        views.push(CoalescedRegionView {
            region_ids: vec![realization.semantic_region.0.clone()],
            packed_buffer: realization.packed_buffer.clone(),
            byte_range: range,
            execution_lane: realization.execution_lane.clone(),
            residency_class: realization.residency_class.clone(),
            conversion_ops: realization.conversion_ops.clone(),
        });
    }
    if views.len() > max_views {
        return Err(RegionScheduleError::ViewBudgetExceeded);
    }
    let cross_lane_boundaries = views
        .windows(2)
        .filter(|pair| pair[0].execution_lane != pair[1].execution_lane)
        .count();
    Ok(RegionScheduleProjection {
        semantic_plan_digest: plan.semantic_plan_digest.clone(),
        raw_region_count: plan.realizations.len(),
        scheduled_view_count: views.len(),
        cross_lane_boundaries,
        views,
    })
}

fn single_range(realization: &PhysicalRegionRealization) -> Result<Range<u64>, RegionScheduleError> {
    match realization.byte_ranges.as_slice() {
        [] => Err(RegionScheduleError::MissingRange),
        [range] => Ok(range.clone()),
        _ => Err(RegionScheduleError::FragmentedRegion),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_region::PhysicalRegionRealization;
    use prism_ecs_ir::semantic_region::SemanticRegionId;

    fn realization(id: &str, start: u64, end: u64, lane: &str) -> PhysicalRegionRealization {
        PhysicalRegionRealization {
            semantic_region: SemanticRegionId(id.into()),
            logical_selector_digest: id.into(),
            packed_buffer: "weights".into(),
            byte_ranges: vec![start..end],
            tile_ids: vec![],
            execution_lane: lane.into(),
            residency_class: "resident".into(),
            materialized_bytes: 0,
            conversion_ops: vec![],
            realization_digest: String::new(),
        }
        .seal()
        .unwrap()
    }

    #[test]
    fn compatible_adjacent_regions_coalesce() {
        let plan = PhysicalRegionPlan {
            semantic_plan_digest: "plan".into(),
            realizations: vec![realization("q", 0, 8, "metal"), realization("k", 8, 12, "metal")],
            total_materialized_bytes: 0,
            total_conversion_bytes: 0,
            digest: String::new(),
        };
        let projection = project_region_schedule(&plan, 4).unwrap();
        assert_eq!(projection.raw_region_count, 2);
        assert_eq!(projection.scheduled_view_count, 1);
    }

    #[test]
    fn lane_change_remains_explicit() {
        let plan = PhysicalRegionPlan {
            semantic_plan_digest: "plan".into(),
            realizations: vec![realization("q", 0, 8, "metal"), realization("k", 8, 12, "cpu")],
            total_materialized_bytes: 0,
            total_conversion_bytes: 0,
            digest: String::new(),
        };
        let projection = project_region_schedule(&plan, 4).unwrap();
        assert_eq!(projection.scheduled_view_count, 2);
        assert_eq!(projection.cross_lane_boundaries, 1);
    }
}
