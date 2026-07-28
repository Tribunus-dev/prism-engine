//! Canonical authority for backend compilation system types that drive executable emission and executable caching.

pub struct BackendCompilationSystem;

impl Default for BackendCompilationSystem {
    fn default() -> Self { Self }
}

pub struct ExecutableCachingSystem;
