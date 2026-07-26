//! Cooperative cancellation for compiler hot loops.
//!
//! Ported from the OMP pi-shell cancel.rs pattern:
//! - CancelToken combines optional deadline + optional abort flag
//! - heartbeat() returns Err when cancelled or timed out
//! - AbortToken via Weak ref for external cancellation

use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc, Weak,
};
use std::time::{Duration, Instant};
use tokio::sync::Notify;

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum AbortReason {
    Unknown = 1,
    Timeout = 2,
    Signal = 3,
    User = 4,
}

impl TryFrom<u8> for AbortReason {
    type Error = ();
    fn try_from(value: u8) -> std::result::Result<Self, ()> {
        match value {
            0 => Err(()),
            2 => Ok(Self::Timeout),
            3 => Ok(Self::Signal),
            4 => Ok(Self::User),
            _ => Ok(Self::Unknown),
        }
    }
}

struct Flag {
    reason: AtomicU8,
    notifier: Notify,
}

impl Default for Flag {
    fn default() -> Self {
        Self {
            reason: AtomicU8::new(0),
            notifier: Notify::new(),
        }
    }
}

impl Flag {
    fn cause(&self) -> Option<AbortReason> {
        self.reason.load(Ordering::Relaxed).try_into().ok()
    }
    fn abort(&self, reason: AbortReason) {
        let old = self.reason.swap(reason as u8, Ordering::SeqCst);
        if old == 0 {
            self.notifier.notify_waiters();
        }
    }
}

/// Token for cooperative cancellation of blocking work.
#[derive(Clone, Default)]
pub struct CancelToken {
    deadline: Option<Instant>,
    flag: Option<Arc<Flag>>,
}

impl CancelToken {
    /// Create a new token with optional timeout in milliseconds.
    pub fn new(timeout_ms: Option<u64>) -> Self {
        Self {
            deadline: timeout_ms.map(|ms| Instant::now() + Duration::from_millis(ms)),
            flag: None,
        }
    }

    /// Check for cancellation. Returns Ok(()) to continue, or an error if cancelled.
    /// Call this periodically in long-running loops.
    pub fn heartbeat(&self) -> Result<(), String> {
        if let Some(flag) = &self.flag {
            if let Some(reason) = flag.cause() {
                return Err(format!("Aborted: {:?}", reason));
            }
        }
        if let Some(deadline) = self.deadline {
            if deadline < Instant::now() {
                return Err("Aborted: Timeout".into());
            }
        }
        Ok(())
    }

    /// Non-blocking check.
    pub fn aborted(&self) -> bool {
        if let Some(flag) = &self.flag {
            if flag.cause().is_some() {
                return true;
            }
        }
        if let Some(deadline) = self.deadline {
            if deadline < Instant::now() {
                return true;
            }
        }
        false
    }

    /// Get an AbortToken for external cancellation.
    pub fn abort_token(&self) -> AbortToken {
        AbortToken(self.flag.as_ref().map(Arc::downgrade))
    }

    /// Lazily create the abort flag. Returns an AbortToken.
    pub fn emplace_abort_token(&mut self) -> AbortToken {
        let flag = self.flag.get_or_insert_with(Default::default);
        AbortToken(Some(Arc::downgrade(flag)))
    }
}

/// Token for requesting cancellation from outside the loop.
#[derive(Clone, Default)]
pub struct AbortToken(Option<Weak<Flag>>);

impl AbortToken {
    pub fn abort(&self, reason: AbortReason) {
        if let Some(flag) = &self.0 {
            if let Some(flag) = flag.upgrade() {
                flag.abort(reason);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heartbeat_passes_without_timeout() {
        let ct = CancelToken::new(None);
        assert!(ct.heartbeat().is_ok());
    }

    #[test]
    fn test_heartbeat_fails_on_external_abort() {
        let mut ct = CancelToken::new(None);
        let at = ct.emplace_abort_token();
        at.abort(AbortReason::User);
        assert!(ct.heartbeat().is_err());
    }

    #[test]
    fn test_aborted_returns_true_after_abort() {
        let mut ct = CancelToken::new(None);
        let at = ct.emplace_abort_token();
        assert!(!ct.aborted());
        at.abort(AbortReason::Timeout);
        assert!(ct.aborted());
    }

    #[test]
    fn test_abort_token_weak_does_not_leak() {
        let ct = CancelToken::new(None);
        let at = ct.abort_token();
        // Without emplace_abort_token, the weak flag should not exist
        // (abort on an empty token is a no-op)
        at.abort(AbortReason::User);
    }
}
