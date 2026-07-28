//! Per-crate error type for the ANE compile surface.
//!
//! Authority: the constitutional ANE error categorisation.
//!
//! Categorised per the project-wide pattern (Rejected for preflight,
//! Failed for effect, Stale for fencing mismatch). Each backend and
//! each public entry point should return [`AneError`] so the calling
//! code can branch on the failure mode without parsing string
//! messages.

use thiserror::Error;

/// ANE compile-time error categorised by the constitutional pattern.
#[derive(Debug, Error)]
pub enum AneError {
    /// Preflight failure — caught before the ANE effect runs.
    ///
    /// The caller asked to use the surface in a way the engine
    /// cannot satisfy (e.g. an empty input sequence, a token count
    /// that exceeds the configured limit, or an oversized input
    /// buffer).
    #[error("ane preflight rejected: {reason}")]
    PreflightRejected {
        /// Static reason for the rejection.
        reason: &'static str,
    },

    /// Effect failure — the ANE backend reported an error during
    /// execution.
    #[error("ane effect failed: {0}")]
    EffectFailed(String),
}

impl AneError {
    /// Construct a [`Self::EffectFailed`] from a string.
    pub fn effect(msg: impl Into<String>) -> Self {
        Self::EffectFailed(msg.into())
    }

    /// Construct a [`Self::PreflightRejected`] with a static reason.
    pub const fn preflight(reason: &'static str) -> Self {
        Self::PreflightRejected { reason }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effect_helper() {
        let err = AneError::effect("oom");
        assert!(matches!(err, AneError::EffectFailed(ref s) if s == "oom"));
    }

    #[test]
    fn preflight_helper() {
        let err = AneError::preflight("empty");
        assert!(matches!(err, AneError::PreflightRejected { reason: "empty" }));
    }
}
