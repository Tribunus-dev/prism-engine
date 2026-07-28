//! Content aliasing and deduplication — pure data types.

use serde::{Deserialize, Serialize};

/// A single content alias entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentAliasEntry {
    /// Alias name.
    pub alias: String,
    /// Canonical object id this alias points to.
    pub canonical_object_id: String,
}

/// A list of alias entries plus the canonical id of the alias
/// source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AliasEntry {
    /// Alias entries.
    pub entries: Vec<ContentAliasEntry>,
}
