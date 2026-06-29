use std::sync::mpsc;

/// Signals fired by background interceptors into the P-core multiplexer's
/// injection window.  The multiplexer drains these between ANE dispatches
/// without blocking.
#[derive(Clone, Debug)]
pub enum RuntimeSignal {
    /// An agent's output code failed validation.
    SyntaxError { agent_id: u32, error_text: String },
    /// A watched file changed on disk (config, prompt, context).
    FileChanged { path: String },
    /// Request to interrupt an agent's current execution context.
    ContextInterrupt { agent_id: u32, reason: String },
}

/// The shared signal bus.  The P-core multiplexer drains this via
/// `try_recv()` in its injection window (between bind and dispatch).
/// Background interceptor tasks fire signals via `send()`.
pub type SignalBus = mpsc::Sender<RuntimeSignal>;
pub type SignalReceiver = mpsc::Receiver<RuntimeSignal>;

/// Create a new signal bus with a synchronous channel of the given capacity.
pub fn create_signal_bus(capacity: usize) -> (SignalBus, SignalReceiver) {
    mpsc::channel()
}
