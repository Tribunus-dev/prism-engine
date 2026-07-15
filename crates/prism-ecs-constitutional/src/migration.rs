// ── Migration Pattern ────────────────────────────────────────────────────
//
// System migration follows this pattern:
//
// 1. Accept a Command (requested intent)
// 2. Issue an EffectRequest (external work, e.g. file load)
// 3. Receive an EffectOutcome (untrusted result)
// 4. Validate the outcome
// 5. Build a WorldTxn with validated state
// 6. Commit via world.transit(txn)
// 7. Emit DomainEvent on success
//
// The existing system keeps its current implementation as a compat path.
// New functionality uses the constitutional path.

/// Migration-specific errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MigrationError {
    #[error("invalid command: {0}")]
    InvalidCommand(String),
    #[error("validation failed: {0}")]
    ValidationFailed(String),
}

/// A migration-compat system that wraps the old pattern.
/// Instantiate to run an existing system side-by-side with its constitutional replacement.
pub struct CompatBridge {
    pub name: String,
    pub use_constitutional: bool,
}

impl CompatBridge {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            use_constitutional: false,
        }
    }
}
