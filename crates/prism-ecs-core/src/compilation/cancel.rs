use std::time::{Duration, Instant};

/// A cancellation token that can be checked at deterministic batch boundaries.
///
/// Created with an optional timeout. Calling [`heartbeat`] at regular intervals
/// allows cooperative cancellation of long-running operations.
#[derive(Clone)]
pub struct CancelToken {
    deadline: Option<Instant>,
}

impl CancelToken {
    /// Create a new cancel token.
    ///
    /// Pass `Some(duration)` to set a deadline or `None` for no deadline.
    pub fn new(timeout: Option<Duration>) -> Self {
        Self {
            deadline: timeout.map(|t| Instant::now() + t),
        }
    }

    /// Check whether the deadline has expired.
    ///
    /// Returns `Ok(())` if no deadline is set or the deadline hasn't been
    /// reached. Returns `Err(())` if the deadline has passed.
    pub fn heartbeat(&self) -> Result<(), ()> {
        if let Some(deadline) = self.deadline {
            if Instant::now() >= deadline {
                return Err(());
            }
        }
        Ok(())
    }
}
