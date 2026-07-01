//! MonotonicClockResource — world-level monotonic clock for elapsed time.
//!
//! Captures `Instant::now()` at world creation and provides `elapsed()` and
//! `now()` accessors so all systems within a world share a common reference
//! point.  The `Instant` is set once at construction; `elapsed()` always
//! counts from that fixed origin.

use std::time::{Duration, Instant};

/// A fixed-reference monotonic clock resource.
///
/// The `start` instant is captured at construction (typically when the World
/// is initialised), giving all systems a consistent timebase without
/// requiring repeated `Instant::now()` calls.
#[derive(Debug, Clone)]
pub struct MonotonicClockResource {
    start: Instant,
}

impl MonotonicClockResource {
    /// Create a new clock resource, capturing `Instant::now()` as the
    /// reference point.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Return the duration elapsed since this resource was created.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Return the saved start instant.
    pub fn now(&self) -> Instant {
        self.start
    }
}

impl Default for MonotonicClockResource {
    fn default() -> Self {
        Self::new()
    }
}
