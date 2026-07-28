//! Engine-independent pump state constants and `MultiplexerState` wrapper.
//!
//! This module owns the canonical authority for the four pump state
//! constants (`STATE_IDLE`, `STATE_PREFETCHING`, `STATE_READY`,
//! `STATE_EXECUTING`) used by the P-core ANE multiplexer and the
//! `MultiplexerState` newtype that wraps them. The engine-coupled
//! `AgentSlot` (which holds an `AtomicU8` pump state) lives in the
//! engine's `legacy_runtime::agent_slot` and uses these constants
//! from the constitutional surface.
//!
//! Migration map: `compute-core/src/ecs/runtime/agent_slot::*` (engine
//! legacy) → `prism_ecs_runtime::runtime::pump_states::*` (constitutional)
//! for the state constants and `MultiplexerState` newtype.

/// The slot is idle; no prefetch in flight.
pub const STATE_IDLE: u8 = 0;
/// The E-core prefetcher is loading weights for the slot.
pub const STATE_PREFETCHING: u8 = 1;
/// Weights are resident and the slot is ready to dispatch.
pub const STATE_READY: u8 = 2;
/// The P-core multiplexer is dispatching the slot.
pub const STATE_EXECUTING: u8 = 3;

/// Type-safe wrapper around a pump state constant.
///
/// This is the constitutional `MultiplexerState` newtype; it prevents
/// engine callers from passing an arbitrary `u8` where a pump state
/// is expected, and provides `Display` for log output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MultiplexerState {
    /// Slot is idle.
    Idle,
    /// E-core prefetcher is loading weights.
    Prefetching,
    /// Weights are resident; ready to dispatch.
    Ready,
    /// P-core multiplexer is dispatching.
    Executing,
}

impl MultiplexerState {
    /// Convert a `u8` state to a `MultiplexerState`.
    ///
    /// Returns `None` for any value outside the four canonical states
    /// (the engine's `AgentSlot` only ever stores one of the four
    /// canonical states; this `try_from_u8` is the safe decoder).
    pub fn try_from_u8(value: u8) -> Option<Self> {
        match value {
            STATE_IDLE => Some(Self::Idle),
            STATE_PREFETCHING => Some(Self::Prefetching),
            STATE_READY => Some(Self::Ready),
            STATE_EXECUTING => Some(Self::Executing),
            _ => None,
        }
    }

    /// Encode this state as the canonical `u8` for the engine's
    /// `AgentSlot::state: AtomicU8`.
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Idle => STATE_IDLE,
            Self::Prefetching => STATE_PREFETCHING,
            Self::Ready => STATE_READY,
            Self::Executing => STATE_EXECUTING,
        }
    }
}

impl std::fmt::Display for MultiplexerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Prefetching => write!(f, "Prefetching"),
            Self::Ready => write!(f, "Ready"),
            Self::Executing => write!(f, "Executing"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_constants_match_engine() {
        // The engine's AgentSlot uses these exact constants; if any
        // value changes, the engine-side bit pattern diverges and
        // existing live agents will mis-decode.
        assert_eq!(STATE_IDLE, 0);
        assert_eq!(STATE_PREFETCHING, 1);
        assert_eq!(STATE_READY, 2);
        assert_eq!(STATE_EXECUTING, 3);
    }

    #[test]
    fn multiplexer_state_round_trip() {
        for (raw, expected) in [
            (STATE_IDLE, MultiplexerState::Idle),
            (STATE_PREFETCHING, MultiplexerState::Prefetching),
            (STATE_READY, MultiplexerState::Ready),
            (STATE_EXECUTING, MultiplexerState::Executing),
        ] {
            assert_eq!(MultiplexerState::try_from_u8(raw), Some(expected));
            assert_eq!(expected.as_u8(), raw);
        }
    }

    #[test]
    fn try_from_u8_rejects_unknown() {
        assert_eq!(MultiplexerState::try_from_u8(4), None);
        assert_eq!(MultiplexerState::try_from_u8(255), None);
    }

    #[test]
    fn display_matches_state_name() {
        assert_eq!(MultiplexerState::Idle.to_string(), "Idle");
        assert_eq!(MultiplexerState::Prefetching.to_string(), "Prefetching");
        assert_eq!(MultiplexerState::Ready.to_string(), "Ready");
        assert_eq!(MultiplexerState::Executing.to_string(), "Executing");
    }
}
