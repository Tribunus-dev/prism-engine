//! Engine-independent `RuntimeSignal` enum and signal bus channel types.
//!
//! This module owns the canonical authority for the runtime signal vocabulary
//! and the channel types that the P-core multiplexer's injection window reads
//! from. The engine-coupled interceptor threads (which fire these signals
//! from background scans) live in the engine's `legacy_runtime::interceptors`
//! and depend on this module.
//!
//! Migration map: `compute-core/src/ecs/runtime/signal_bus.rs` (engine
//! legacy, deleted in the engine-deletion migration) →
//! `prism_ecs_runtime::runtime::signal_bus::*` (constitutional).

use std::sync::mpsc;

/// Signals fired by background interceptors into the P-core multiplexer's
/// injection window.
///
/// The multiplexer drains these between ANE dispatches without blocking;
/// each variant carries enough payload for the multiplexer to decide
/// whether to substitute corrected agent output before the next dispatch.
#[derive(Clone, Debug)]
pub enum RuntimeSignal {
    /// An agent's output code failed validation.
    SyntaxError {
        /// Identifier of the agent whose output failed validation.
        agent_id: u32,
        /// Human-readable description of the validation failure.
        error_text: String,
    },
    /// A watched file changed on disk (config, prompt, context).
    FileChanged {
        /// Absolute path to the file that changed.
        path: String,
    },
    /// Request to interrupt an agent's current execution context.
    ContextInterrupt {
        /// Identifier of the agent whose context should be interrupted.
        agent_id: u32,
        /// Reason for the interrupt.
        reason: String,
    },
}

/// The shared signal bus. The P-core multiplexer drains this via `try_recv()`
/// in its injection window (between bind and dispatch).
pub type SignalBus = mpsc::Sender<RuntimeSignal>;

/// Receiver side of the signal bus. The P-core multiplexer holds the
/// receiver and drains it between dispatches.
pub type SignalReceiver = mpsc::Receiver<RuntimeSignal>;

/// Create a new signal bus with a synchronous channel of the given capacity.
///
/// The capacity is currently advisory: `mpsc::channel()` is unbounded in
/// practice, but the call site documents the intended buffer size so a
/// future implementation can swap in a bounded variant without changing
/// the call site.
pub fn create_signal_bus(_capacity: usize) -> (SignalBus, SignalReceiver) {
    mpsc::channel()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_signal_bus_yields_send_and_receive() {
        let (tx, rx) = create_signal_bus(16);
        tx.send(RuntimeSignal::FileChanged {
            path: "/tmp/x".to_string(),
        })
        .expect("send should succeed");
        let received = rx.recv().expect("recv should succeed");
        match received {
            RuntimeSignal::FileChanged { path } => assert_eq!(path, "/tmp/x"),
            _ => panic!("expected FileChanged"),
        }
    }

    #[test]
    fn signal_variants_carry_payload() {
        let syntax = RuntimeSignal::SyntaxError {
            agent_id: 7,
            error_text: "unbalanced paren".to_string(),
        };
        match syntax {
            RuntimeSignal::SyntaxError { agent_id, .. } => assert_eq!(agent_id, 7),
            _ => panic!("expected SyntaxError"),
        }

        let interrupt = RuntimeSignal::ContextInterrupt {
            agent_id: 9,
            reason: "tool loop".to_string(),
        };
        match interrupt {
            RuntimeSignal::ContextInterrupt { reason, .. } => assert_eq!(reason, "tool loop"),
            _ => panic!("expected ContextInterrupt"),
        }
    }
}
