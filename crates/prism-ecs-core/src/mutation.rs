/// Mutation access policy for a World.
///
/// - `TransactionalOnly`: All mutations through WorldTxn.
/// - `ControlledDirect`: Direct mutations allowed but controlled.
/// - `Bootstrap`: Full direct mutation access for initial construction.
/// - `TestHarness`: Explicit opt-in for test fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationPolicy {
    TransactionalOnly,
    ControlledDirect,
    Bootstrap,
    TestHarness,
}

impl MutationPolicy {
    /// Returns true if this policy permits direct mutations.
    pub fn direct_mutations_allowed(&self) -> bool {
        matches!(
            self,
            MutationPolicy::ControlledDirect
                | MutationPolicy::Bootstrap
                | MutationPolicy::TestHarness
        )
    }

    /// Returns true if this policy requires all mutations through WorldTxn.
    pub fn requires_transactional(&self) -> bool {
        matches!(self, MutationPolicy::TransactionalOnly)
    }
}

impl Default for MutationPolicy {
    fn default() -> Self {
        MutationPolicy::Bootstrap
    }
}
