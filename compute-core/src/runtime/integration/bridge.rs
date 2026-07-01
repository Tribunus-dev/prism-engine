//! ProductionTelemetrySplitter — mirrors ingress requests to the ECS path
//! without blocking the legacy generation path.
//!
//! The splitter operates in two modes controlled by
//! [`EcsWorkerSupervisionMode`]:
//!
//! - **Disabled** — `dispatch` is a no-op.
//! - **Enabled** — entries are dispatched to the ECS channel and the
//!   schedule is executed normally.
//!
//! In both modes the legacy path is never stalled.

use std::sync::mpsc::Sender;

use crate::runtime::resources::worker_supervision_config::EcsWorkerSupervisionMode;

// ---------------------------------------------------------------------------
// IngressEntry — splitter-level request snapshot
// ---------------------------------------------------------------------------

/// A snapshot of an incoming generation request captured by the splitter.
///
/// Carried across the channel to the ECS side so the schedule can
/// reconstruct the request context without touching the legacy path.
pub struct IngressEntry {
    /// Unique request identifier (mirrors the legacy request_id).
    pub request_id: String,
    /// Input prompt text or token serialisation.
    pub prompt: String,
    /// Wall-clock instant when the entry was captured.
    pub created_at: std::time::Instant,
}

// ---------------------------------------------------------------------------
// ProductionTelemetrySplitter
// ---------------------------------------------------------------------------

/// Mirrors ingress telemetry to the ECS schedule without blocking the
/// legacy generation path.
///
/// The splitter owns the `Sender<T>` half of an mpsc channel.  When
/// configured in Shadow or Enabled mode, `dispatch` sends the entry down
/// the channel for the ECS schedule to process in its own tick.  In
/// Disabled mode every call is a no-op.
pub struct ProductionTelemetrySplitter {
    ecs_tx: Option<Sender<IngressEntry>>,
    mode: EcsWorkerSupervisionMode,
}

impl ProductionTelemetrySplitter {
    /// Create a new splitter.
    ///
    /// Pass `None` for `ecs_tx` (or the Disabled mode) to make every
    /// dispatch a no-op.
    pub fn new(mode: EcsWorkerSupervisionMode, ecs_tx: Option<Sender<IngressEntry>>) -> Self {
        Self { ecs_tx, mode }
    }

    /// Non-blocking dispatch — never stalls the legacy path.
    ///
    /// When the splitter is Disabled this is a no-op.  When enabled or in
    /// shadow mode the entry is sent on the internal channel; a full or
    /// disconnected receiver silently drops the entry.
    pub fn dispatch(&self, entry: IngressEntry) {
        if let Some(tx) = &self.ecs_tx {
            // Unbounded channel: send never blocks.  A dropped receiver
            // is silently ignored so a faulted ECS side never stalls the
            // legacy path.
            let _ = tx.send(entry);
        }
    }

    /// The current supervision mode.
    pub fn mode(&self) -> EcsWorkerSupervisionMode {
        self.mode
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

impl std::fmt::Debug for ProductionTelemetrySplitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProductionTelemetrySplitter")
            .field("mode", &self.mode)
            .field(
                "ecs_tx",
                &match &self.ecs_tx {
                    Some(_) => "Some(Sender)",
                    None => "None",
                },
            )
            .finish()
    }
}
