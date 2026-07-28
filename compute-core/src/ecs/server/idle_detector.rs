//! Idle detection for background calibration tasks.
//!
//! Tracks time since the last incoming request and coordinates cancellation
//! of background work when a new request arrives during an idle pass.

use prism_ecs_compile::compilation::cancel::{AbortReason, AbortToken, CancelToken};
use std::time::{Duration, Instant};

/// Tracks server idle state and manages cancellation of idle-time background work.
///
/// The background monitor loop calls [`is_idle()`](Self::is_idle) every 10 seconds;
/// when the server has been idle beyond `idle_threshold`, it calls
/// [`enter_idle()`](Self::enter_idle) to obtain a [`CancelToken`], runs the
/// calibration pass, and resets when a new request arrives (which triggers
/// [`on_new_request()`](Self::on_new_request) via [`note_request()`](Self::note_request)).
pub struct IdleDetector {
    last_request: Instant,
    idle_threshold: Duration,
    cancel_token: CancelToken,
    is_idle: bool,
    /// Internal abort handle created by [`enter_idle()`](Self::enter_idle).
    abort_token: Option<AbortToken>,
}

impl IdleDetector {
    /// Create a new detector.
    ///
    /// `idle_threshold` is the duration of inactivity before the server is
    /// considered idle (default 30 seconds).
    pub fn new(idle_threshold: Duration) -> Self {
        Self {
            last_request: Instant::now(),
            idle_threshold,
            cancel_token: CancelToken::default(),
            is_idle: false,
            abort_token: None,
        }
    }

    /// Call on every incoming request — resets the idle timer.
    ///
    /// If an idle pass was in progress, triggers cancellation via
    /// [`on_new_request()`](Self::on_new_request).
    pub fn note_request(&mut self) {
        self.last_request = Instant::now();
        if self.is_idle {
            self.on_new_request();
        }
    }

    /// Returns true if `idle_threshold` has elapsed since the last request.
    pub fn is_idle(&self) -> bool {
        self.last_request.elapsed() >= self.idle_threshold
    }

    /// Called by the background task when entering an idle pass.
    ///
    /// Sets `is_idle` and returns a [`CancelToken`] that the background work
    /// can pass to [`run_background_calibration`] for cooperative cancellation.
    pub fn enter_idle(&mut self) -> CancelToken {
        self.is_idle = true;
        // Create a fresh CancelToken with an emplaceable abort flag.
        let mut token = CancelToken::new(None);
        self.abort_token = Some(token.emplace_abort_token());
        self.cancel_token = token.clone();
        token
    }

    /// Called when a new request arrives during idle — triggers the stored
    /// [`AbortToken`] and resets the idle state.
    pub fn on_new_request(&mut self) {
        if let Some(abort) = &self.abort_token {
            abort.abort(AbortReason::User);
        }
        self.is_idle = false;
        self.cancel_token = CancelToken::default();
        self.abort_token = None;
    }

    /// True when the idle loop is currently running (an idle pass is in progress).
    pub fn is_running(&self) -> bool {
        self.is_idle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_idle_detector_0ms_threshold() {
        let det = IdleDetector::new(Duration::from_millis(0));
        // With a 0ms threshold, is_idle() should return true immediately.
        assert!(
            det.is_idle(),
            "0ms threshold should report idle immediately"
        );
    }

    #[test]
    fn test_note_request_resets_timer() {
        let mut det = IdleDetector::new(Duration::from_millis(50));
        // Wait just past the threshold.
        std::thread::sleep(Duration::from_millis(60));
        assert!(det.is_idle(), "should be idle after sleep");

        // note_request resets the timer.
        det.note_request();
        assert!(!det.is_idle(), "note_request should reset idle timer");
    }

    #[test]
    fn test_enter_idle_cancels_on_new_request() {
        let mut det = IdleDetector::new(Duration::from_secs(3600));
        assert!(!det.is_idle(), "should not be idle with long threshold");

        // Simulate entering an idle pass.
        let token = det.enter_idle();
        assert!(det.is_running(), "should report running after enter_idle");
        assert!(!token.aborted(), "token should not be aborted initially");

        // Simulate a new request arriving.
        det.on_new_request();
        assert!(
            !det.is_running(),
            "should not be running after on_new_request"
        );
        assert!(
            token.aborted(),
            "token should be aborted after on_new_request"
        );
    }
}
