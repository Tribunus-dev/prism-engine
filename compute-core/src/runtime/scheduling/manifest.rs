//! Schedule manifest — a hashable, versioned, inspectable schedule artifact.
//!
//! The manifest records every scheduling decision: stage ordering, explicit
//! edges, serialization resolutions, hazards, and warnings.  It is produced
//! by `Schedule::compile` and can be serialized for correlation with
//! inference receipts.

use crate::runtime::scheduling::graph::EdgeKind;
use crate::runtime::scheduling::metadata::SystemId;

/// Schema version of the manifest format.
///
/// Increment when the manifest structure changes in a breaking way.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// A single edge recorded in the manifest.
#[derive(Debug, Clone)]
pub struct ManifestEdge {
    /// Origin system ID.
    pub from: SystemId,
    /// Destination system ID.
    pub to: SystemId,
    /// Reason for the edge.
    pub kind: ManifestEdgeKind,
}

/// Serializable classification of an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestEdgeKind {
    /// Stage barrier.
    StageBarrier,
    /// Explicit `after` or `before` declaration.
    Explicit,
    /// Determined by serialization policy.
    Serialization,
}

/// A single hazard record in the manifest.
#[derive(Debug, Clone)]
pub struct HazardRecord {
    /// First system.
    pub system_a: SystemId,
    /// Second system.
    pub system_b: SystemId,
    /// Description of the overlap type.
    pub description: &'static str,
    /// Whether the hazard was resolved.
    pub resolved: bool,
    /// How it was resolved (empty if unresolved).
    pub resolution: &'static str,
}

/// A warning emitted during schedule compilation.
#[derive(Debug, Clone)]
pub struct ManifestWarning {
    /// Category of warning.
    pub kind: ManifestWarningKind,
    /// Human-readable message.
    pub message: String,
}

/// Warning categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestWarningKind {
    /// A commutative hazard was detected and allowed.
    CommutativeHazard,
    /// A stage barrier was implicitly inserted.
    ImplicitStageBarrier,
}

// ---------------------------------------------------------------------------
// ScheduleManifest
// ---------------------------------------------------------------------------

/// Immutable, hashable record of a compiled schedule.
#[derive(Debug, Clone)]
pub struct ScheduleManifest {
    /// Schema version.
    pub schema_version: u32,
    /// Total number of systems in the schedule.
    pub system_count: usize,
    /// Final execution order (SystemIds).
    pub execution_order: Vec<SystemId>,
    /// All dependency edges in the graph.
    pub edges: Vec<ManifestEdge>,
    /// All hazard records.
    pub hazards: Vec<HazardRecord>,
    /// Warnings emitted during compilation.
    pub warnings: Vec<ManifestWarning>,
    /// SHA-256 digest of the canonical manifest bytes.
    pub digest: [u8; 32],
}

impl ScheduleManifest {
    /// Compute the canonical SHA-256 digest.
    ///
    /// The digest is computed over a deterministic byte representation
    /// of the manifest fields: schema_version, system_count, execution_order,
    /// edges (sorted), hazards (sorted), warnings (sorted).
    pub fn compute_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.to_le_bytes());
        hasher.update((self.system_count as u64).to_le_bytes());

        for id in &self.execution_order {
            hasher.update(id.0.to_le_bytes());
        }

        // Sort edges deterministically before hashing.
        let mut sorted_edges = self.edges.clone();
        sorted_edges.sort_by(|a, b| {
            a.from
                .0
                .cmp(&b.from.0)
                .then_with(|| a.to.0.cmp(&b.to.0))
                .then_with(|| format!("{:?}", a.kind).cmp(&format!("{:?}", b.kind)))
        });
        for edge in &sorted_edges {
            hasher.update(edge.from.0.to_le_bytes());
            hasher.update(edge.to.0.to_le_bytes());
            hasher.update(match edge.kind {
                ManifestEdgeKind::StageBarrier => 0u8,
                ManifestEdgeKind::Explicit => 1u8,
                ManifestEdgeKind::Serialization => 2u8,
            }
            .to_le_bytes());
        }

        hasher.finalize().into()
    }
}

// ---------------------------------------------------------------------------
// ManifestBuilder
// ---------------------------------------------------------------------------

pub(crate) struct ManifestBuilder {
    edges: Vec<ManifestEdge>,
    hazards: Vec<HazardRecord>,
    warnings: Vec<ManifestWarning>,
}

impl ManifestBuilder {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            hazards: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn record_edge(&mut self, from: SystemId, to: SystemId, kind: EdgeKind) {
        let manifest_kind = match kind {
            EdgeKind::StageBarrier => ManifestEdgeKind::StageBarrier,
            EdgeKind::ExplicitAfter | EdgeKind::ExplicitBefore => ManifestEdgeKind::Explicit,
            EdgeKind::Data | EdgeKind::Serialization => ManifestEdgeKind::Serialization,
        };
        self.edges.push(ManifestEdge {
            from,
            to,
            kind: manifest_kind,
        });
    }

    pub fn record_hazard(
        &mut self,
        a: SystemId,
        b: SystemId,
        description: &'static str,
        resolved: bool,
        resolution: &'static str,
    ) {
        self.hazards.push(HazardRecord {
            system_a: a,
            system_b: b,
            description,
            resolved,
            resolution,
        });
    }

    pub fn record_warning(&mut self, kind: ManifestWarningKind, message: String) {
        self.warnings.push(ManifestWarning { kind, message });
    }

    pub fn build(
        self,
        execution_order: Vec<SystemId>,
    ) -> ScheduleManifest {
        let mut manifest = ScheduleManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            system_count: execution_order.len(),
            execution_order,
            edges: self.edges,
            hazards: self.hazards,
            warnings: self.warnings,
            digest: [0u8; 32],
        };
        manifest.digest = manifest.compute_digest();
        manifest
    }
}

impl Default for ManifestBuilder {
    fn default() -> Self {
        Self::new()
    }
}
