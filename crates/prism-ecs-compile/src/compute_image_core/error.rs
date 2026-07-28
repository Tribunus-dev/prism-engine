//! Typed errors for the `compute_image_core` constitutional surface.
//!
//! This module owns the canonical error enum for the
//! `compute_image_core` data types. Errors are categorized as
//! `Rejected` (preflight), `Failed` (effect), or `Stale` (fencing
//! mismatch) per the constitutional rules. Variants carry the
//! authority context (entity id, generation, epoch) where applicable.

use thiserror::Error;

/// Canonical error type for `compute_image_core` operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Preflight rejected the operation (validation failure, missing
    /// identity, unsupported capability, etc.). The caller should
    /// repair the request rather than retry.
    #[error("rejected: {0}")]
    Rejected(String),

    /// An effect or transform failed (deserialization, hash mismatch,
    /// materialization error, etc.).
    #[error("failed: {0}")]
    Failed(String),

    /// A stale-outcome rejection: the world advanced past the
    /// operation's expected epoch or generation. The caller should
    /// re-fetch authoritative state and resubmit.
    #[error("stale: {0}")]
    Stale(String),
}

impl Error {
    /// Construct a `Rejected` variant.
    pub fn rejected(reason: impl Into<String>) -> Self {
        Self::Rejected(reason.into())
    }

    /// Construct a `Failed` variant.
    pub fn failed(reason: impl Into<String>) -> Self {
        Self::Failed(reason.into())
    }

    /// Construct a `Stale` variant.
    pub fn stale(reason: impl Into<String>) -> Self {
        Self::Stale(reason.into())
    }

    /// Construct a `Failed` variant from a free-form reason. This
    /// matches the engine's `Error::from_reason` constructor and
    /// exists for source-compatible migration.
    pub fn from_reason(reason: impl Into<String>) -> Self {
        Self::Failed(reason.into())
    }
}

/// Result alias for the `compute_image_core` surface.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Current timestamp as ISO 8601 UTC string. Source-compatible
/// shim for the engine's `crate::now_iso8601` used by manifest
/// builders and fusion sealing timestamps.
pub fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = (secs / 86400) as i64;
    let time_secs = secs % 86400;
    let (year, month, day) = civil_from_days(days);
    let hour = time_secs / 3600;
    let min = (time_secs % 3600) / 60;
    let sec = time_secs % 60;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, min, sec
    )
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    // Howard Hinnant's date algorithm: convert days since 1970-01-01 to
    // (year, month, day) in the proleptic Gregorian calendar.
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as u32, d as u32)
}

/// Hostname or "unknown" if unavailable. Source-compatible shim for
/// the engine's `crate::hostname_or_default` used by build
/// attestation records.
pub fn hostname_or_default() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}
