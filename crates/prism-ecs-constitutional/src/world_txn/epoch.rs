//! Authority: this module owns the post-commit epoch token
//! [`CommittedEpoch`]. The token is the canonical receipt for a
//! successful [`WorldTxn`] commit; downstream replay and projection
//! rebuilds key off this value. It is distinct from the
//! pre-commit expected epoch recorded on the transaction (which lives in
//! [`crate::world_txn::txn`]).

use crate::types::WorldEpoch;
use serde::{Deserialize, Serialize};

/// The epoch assigned after a successful [`crate::world_txn::txn::WorldTxn`]
/// commit.
///
/// The token is the canonical receipt for the commit; consumers that
/// observe the world's current epoch transitioning to this value know
/// that the commit's mutations are visible. The token is `Copy` to make
/// it cheap to forward through downstream pipelines (replay, projection
/// rebuild, durable event fan-out).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedEpoch(pub WorldEpoch);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::WorldEpoch;

    /// The `CommittedEpoch` token is `Copy + Eq` so it can travel
    /// through fan-out channels (event bus, durable store, projection
    /// rebuild) without lifetime entanglement. Equality is structural
    /// over the wrapped `WorldEpoch`.
    #[test]
    fn committed_epoch_is_copy_and_structurally_equal() {
        let a = CommittedEpoch(WorldEpoch(7));
        let b = a; // Copy semantics — `a` is still usable after.
        assert_eq!(a, b);
        let c = CommittedEpoch(WorldEpoch(8));
        assert_ne!(a, c);
        // `Serialize` round-trip — downstream consumers (replay log,
        // event store) need to persist this token verbatim.
        let json = serde_json::to_string(&a).expect("serialize");
        let back: CommittedEpoch = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(a, back);
    }
}
