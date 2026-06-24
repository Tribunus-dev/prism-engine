use std::hash::Hash;

/// The compact, device-visible runtime identifier.
/// The GPU sees only slot values in hot relation columns.
/// Generation validation occurs at host/persistence boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EntityId {
    pub slot: u32,
    pub generation: u16,
    pub kind: u8,
    pub reserved: u8,
}

impl EntityId {
    pub fn new(slot: u32, generation: u16, kind: u8) -> Self {
        Self {
            slot,
            generation,
            kind,
            reserved: 0,
        }
    }
}

/// Identifies the kind of entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityKind {
    Operation,
    Block,
    Value,
    Callsite,
    Allocation,
    Object,
    Region,
    Function,
    Module,
    Unknown,
}

impl Default for EntityKind {
    fn default() -> Self {
        Self::Unknown
    }
}

/// A canonical path within the semantic representation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CanonicalPath {
    pub segments: Vec<String>,
}

/// A source anchor mapping an entity back to original source code if available.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceAnchor {
    pub file_hash: u64,
    pub line: u32,
    pub column: u32,
}

/// A portable federated identity claim.
/// A federated key is not itself a runtime entity ID. It is a claim that allows
/// the central workspace to resolve a remote entity into its own local generational identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CanonicalIdentityKey {
    pub workspace_namespace: u64,
    pub module_content_root: u64,
    pub semantic_path: CanonicalPath,
    pub entity_kind: EntityKind,
    pub normalized_structure_hash: u64,
    pub binder_or_scope_hash: u64,
    pub source_anchor: Option<SourceAnchor>,
}

/// The result of attempting to resolve a federated identity to a local identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolutionStatus {
    Matched,
    Created,
    Conflict,
    RequiresRecanonicalization,
}

/// Emitted by the central resolver when processing a remote delta.
#[derive(Debug, Clone)]
pub struct FederatedIdentityResolution {
    pub producer_session_id: u64,
    pub exported_key_to_central_id: std::collections::HashMap<CanonicalIdentityKey, EntityId>,
    pub status: ResolutionStatus,
}
