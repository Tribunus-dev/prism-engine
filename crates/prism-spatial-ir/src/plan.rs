//! [`SpatialCompilationPlan`] — a complete, legalized, cost-estimated schedule.
//!
//! The compilation plan is the output of the evolutionary search: a spatial
//! graph that has been legalized, cost-estimated, and assigned to virtual
//! hardware. It carries the calibration identity that produced the estimates.

use crate::cost::CostEstimate;
use crate::graph::SpatialGraph;
use crate::legalize::LegalizedGraph;
use crate::target::CalibrationId;
use serde::{Deserialize, Serialize};

/// A complete spatial compilation plan.
///
/// A plan is produced by the evolutionary search loop and contains the
/// chosen spatial graph, its cost estimate, the hardware binding, and the
/// calibration identity that validated the estimates.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpatialCompilationPlan {
    /// The legalized spatial graph.
    pub graph: LegalizedGraph,
    /// Cost estimate for this plan.
    pub cost: CostEstimate,
    /// Identifier of the hardware binding (target unit assignments).
    pub hardware_binding: String,
    /// Calibration ID that produced the cost estimates.
    pub calibration_id: CalibrationId,
    /// Number of mutations applied to reach this plan.
    pub mutation_count: usize,
}

impl SpatialCompilationPlan {
    /// Create a new compilation plan.
    pub fn new(
        graph: LegalizedGraph,
        cost: CostEstimate,
        hardware_binding: String,
        calibration_id: CalibrationId,
    ) -> Self {
        Self {
            graph,
            cost,
            hardware_binding,
            calibration_id,
            mutation_count: 0,
        }
    }

    /// Returns the number of nodes in the plan's graph.
    pub fn node_count(&self) -> usize {
        self.graph.graph().node_count()
    }

    /// Returns the number of edges in the plan's graph.
    pub fn edge_count(&self) -> usize {
        self.graph.graph().edge_count()
    }

    /// Returns a reference to the inner spatial graph.
    pub fn spatial_graph(&self) -> &SpatialGraph {
        self.graph.graph()
    }

    /// Serializes the plan to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserializes a plan from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::CostModel;
    use crate::cost::SimpleCostModel;
    use crate::graph::{ComputeIntensity, ComputeKind, ShapeContract, SpatialNode, SpatialNodeId};
    use crate::legalize::LegalizedGraph;
    use prism_ecs_ir::cimage_types::TensorShape;

    #[test]
    fn plan_roundtrips_serde() {
        let mut graph = crate::graph::SpatialGraph::new();
        graph.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        let legalized = LegalizedGraph::new(graph, vec![]);
        let model = SimpleCostModel;
        let cost = model.estimate(legalized.graph());

        let plan = SpatialCompilationPlan::new(
            legalized,
            cost,
            "apple_m1_metal".to_string(),
            CalibrationId("cal_001".to_string()),
        );

        assert_eq!(plan.node_count(), 1);
        assert_eq!(plan.edge_count(), 0);

        let json = plan.to_json().expect("serialize");
        let restored = SpatialCompilationPlan::from_json(&json).expect("deserialize");
        assert_eq!(plan, restored);
    }
}
