//! Compiler provenance — who compiled the image and when.

use serde::{Deserialize, Serialize};

/// Provenance record for a compiled executable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerProvenance {
    /// Compiler name.
    pub compiler_name: String,
    /// Compiler version.
    pub compiler_version: String,
    /// Compilation timestamp (RFC3339 string).
    pub compilation_timestamp: String,
    /// Content hash of the source model.
    pub source_model_hash: String,
    /// Target profile identifiers.
    pub target_profile_ids: Vec<String>,
}
