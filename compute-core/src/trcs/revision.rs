use std::collections::HashSet;

pub type RevisionId = u64;
pub type RevisionFrontierId = u32;
pub type ModuleId = u64;
pub type AuthorId = u64;
pub type WorkItemId = u64;
pub type LeaseId = u64;
pub type Timestamp = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuthorKind {
    Human,
    Agent,
    Merge,
    System,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleSet {
    pub modules: HashSet<ModuleId>,
}

impl ModuleSet {
    pub fn new() -> Self {
        Self {
            modules: HashSet::new(),
        }
    }
}

impl Default for ModuleSet {
    fn default() -> Self {
        Self::new()
    }
}

/// A node in the revision merge DAG.
#[derive(Debug, Clone)]
pub struct WorkspaceRevision {
    pub revision_id: RevisionId,
    pub parent_frontier: RevisionFrontierId,
    pub parent_revisions: Vec<RevisionId>,
    pub changed_modules: ModuleSet,
    pub semantic_delta_hash: u64,
    pub author_kind: AuthorKind,
    pub author_id: AuthorId,
    pub work_item_id: Option<WorkItemId>,
    pub ownership_lease: Option<LeaseId>,
    pub created_at: Timestamp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisjointSemanticMerge {
    StrictDisjoint,
    AllowSummaryOverlap,
}

/// A synthetic merge frontier sharing subagent revisions.
#[derive(Debug, Clone)]
pub struct MergeFrontier {
    pub parent_frontiers: HashSet<RevisionFrontierId>,
    pub member_revisions: HashSet<RevisionId>,
    pub changed_modules: ModuleSet,
    pub merge_policy: DisjointSemanticMerge,
}
