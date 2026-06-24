use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrcsError {
    InvalidSupport(String),
    MalformedOrdering(String),
    CompactionConflict(String),
    UnsupportedArity(usize),
    InvalidBulkLoadPlan(String),
    LifecycleViolation(String),
}

impl fmt::Display for TrcsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrcsError::InvalidSupport(msg) => write!(f, "Invalid support: {}", msg),
            TrcsError::MalformedOrdering(msg) => write!(f, "Malformed ordering: {}", msg),
            TrcsError::CompactionConflict(msg) => write!(f, "Compaction conflict: {}", msg),
            TrcsError::UnsupportedArity(arity) => write!(f, "Unsupported arity: {}", arity),
            TrcsError::InvalidBulkLoadPlan(msg) => write!(f, "Invalid bulk load plan: {}", msg),
            TrcsError::LifecycleViolation(msg) => write!(f, "Lifecycle violation: {}", msg),
        }
    }
}

impl std::error::Error for TrcsError {}
