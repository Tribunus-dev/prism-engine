//! Mutation access policy for [World](crate::ecs::World).
//!
//! Defines the four-tier mutation access model:
//! - [`TransactionalOnly`](MutationPolicy::TransactionalOnly): All mutations must go through
//!   `WorldTxn`. Used by production compiler, runtime, server, and evaluator worlds.
//! - [`ControlledDirect`](MutationPolicy::ControlledDirect): Direct mutations are allowed but
//!   subject to runtime controls.
//! - [`Bootstrap`](MutationPolicy::Bootstrap): Full direct mutation access for initial world
//!   construction before publication. This is the default.
//! - [`TestHarness`](MutationPolicy::TestHarness): Explicit opt-in for test fixtures. Not
//!   available in production paths.
//!
//! The default is `Bootstrap` because the common pattern is to construct a world, then
//! transition it to `TransactionalOnly` before exposing it to concurrent consumers.

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
