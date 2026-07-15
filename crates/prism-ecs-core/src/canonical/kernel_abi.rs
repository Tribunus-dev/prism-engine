//! Kernel ABI types — stable identifiers for kernel semantics and implementations.

use serde::{Deserialize, Serialize};

/// Semantic identifier for a kernel purpose (e.g. "prism.linear.nf4.v1").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct KernelSemanticId(pub String);

impl From<&str> for KernelSemanticId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl From<String> for KernelSemanticId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl std::fmt::Display for KernelSemanticId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
