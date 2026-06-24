use crate::trcs::revision::RevisionFrontierId;

/// Represents a row in a physical delta batch.
/// The physical arena never allocates negative memory; retractions are written
/// as physical rows with diff = -1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalDeltaRow<const ARITY: usize> {
    pub tuple_columns: [u32; ARITY],
    pub diff: i32,
    pub revision_frontier_id: RevisionFrontierId,
    pub provenance_token: u64,
}

/// Tracks the number of physical rows emitted by a processing tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmissionCount {
    pub tile_id: u32,
    pub emitted_rows: u32,
}
