//! Backend-neutral inference contracts.
//!
//! The server owns admission, residency, cancellation, and observability;
//! concrete runtimes only implement a generation session.  This keeps native
//! Prism execution, subprocess engines, and remote OpenAI-compatible engines
//! interchangeable at the control-plane boundary.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use super::server_types::SamplingConfig;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct InferenceTelemetrySnapshot {
    pub requests_started: u64,
    pub requests_completed: u64,
    pub requests_cancelled: u64,
    pub requests_failed: u64,
    pub tokens_generated: u64,
    pub prefill_latency_ms: Vec<f64>,
    pub decode_latency_ms: Vec<f64>,
    pub fallback_reasons: std::collections::BTreeMap<String, u64>,
}

#[derive(Default)]
pub struct InferenceTelemetry {
    requests_started: std::sync::atomic::AtomicU64,
    requests_completed: std::sync::atomic::AtomicU64,
    requests_cancelled: std::sync::atomic::AtomicU64,
    requests_failed: std::sync::atomic::AtomicU64,
    tokens_generated: std::sync::atomic::AtomicU64,
    prefill_latency_ms: parking_lot::Mutex<Vec<f64>>,
    decode_latency_ms: parking_lot::Mutex<Vec<f64>>,
    fallback_reasons: parking_lot::Mutex<std::collections::BTreeMap<String, u64>>,
}

impl InferenceTelemetry {
    pub fn started(&self) {
        self.requests_started
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    pub fn completed(&self, tokens: u64) {
        self.requests_completed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.tokens_generated
            .fetch_add(tokens, std::sync::atomic::Ordering::Relaxed);
    }
    pub fn cancelled(&self) {
        self.requests_cancelled
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    pub fn failed(&self) {
        self.requests_failed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    pub fn prefill_latency(&self, ms: f64) {
        self.prefill_latency_ms.lock().push(ms);
    }
    pub fn decode_latency(&self, ms: f64) {
        self.decode_latency_ms.lock().push(ms);
    }
    pub fn fallback(&self, reason: impl Into<String>) {
        let mut reasons = self.fallback_reasons.lock();
        *reasons.entry(reason.into()).or_default() += 1;
    }
    pub fn snapshot(&self) -> InferenceTelemetrySnapshot {
        InferenceTelemetrySnapshot {
            requests_started: self
                .requests_started
                .load(std::sync::atomic::Ordering::Relaxed),
            requests_completed: self
                .requests_completed
                .load(std::sync::atomic::Ordering::Relaxed),
            requests_cancelled: self
                .requests_cancelled
                .load(std::sync::atomic::Ordering::Relaxed),
            requests_failed: self
                .requests_failed
                .load(std::sync::atomic::Ordering::Relaxed),
            tokens_generated: self
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            prefill_latency_ms: self.prefill_latency_ms.lock().clone(),
            decode_latency_ms: self.decode_latency_ms.lock().clone(),
            fallback_reasons: self.fallback_reasons.lock().clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BackendKind {
    Native,
    Subprocess,
    Remote,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionRecipe {
    pub backend: BackendKind,
    pub runtime: String,
    pub device_order: Vec<String>,
    pub required_capabilities: Vec<String>,
    pub options: std::collections::BTreeMap<String, String>,
}

impl Default for ExecutionRecipe {
    fn default() -> Self {
        Self {
            backend: BackendKind::Native,
            runtime: "prism".into(),
            device_order: vec!["metal".into(), "cpu".into()],
            required_capabilities: Vec::new(),
            options: std::collections::BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BackendCapabilities {
    pub backend: BackendKind,
    pub name: String,
    pub devices: Vec<String>,
    pub modalities: Vec<String>,
    pub max_context_tokens: Option<u32>,
    pub supports_streaming: bool,
    pub supports_cancellation: bool,
    pub supports_tool_calling: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum GenerationEvent {
    Started {
        request_id: String,
        backend: String,
    },
    Token {
        text: String,
        token_id: Option<u32>,
    },
    Finished {
        token_count: u32,
    },
    Cancelled {
        token_count: u32,
    },
    Failed {
        message: String,
        fallback: Option<String>,
    },
}

#[derive(Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

pub trait GenerationSession: Send {
    fn backend(&self) -> &str;
    fn capabilities(&self) -> BackendCapabilities;
    fn prefill(&mut self, prompt: &str) -> Result<(), String>;
    fn decode(&mut self, sampling: &SamplingConfig) -> Result<Option<GenerationEvent>, String>;
    fn cancel(&mut self);
}

pub trait InferenceBackend: Send + Sync {
    fn recipe(&self) -> &ExecutionRecipe;
    fn capabilities(&self) -> BackendCapabilities;
    fn open(
        &self,
        model_path: &std::path::Path,
        cancellation: CancellationToken,
    ) -> Result<Box<dyn GenerationSession>, String>;
}

/// Configuration for an engine hosted outside the Prism process.
/// The adapter is deliberately declarative; transport implementations can be
/// added without changing model admission or the HTTP API.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExternalBackendSpec {
    pub endpoint: String,
    pub protocol: String,
    pub model: Option<String>,
    pub headers: std::collections::BTreeMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_is_shared_and_idempotent() {
        let token = CancellationToken::new();
        let copy = token.clone();
        assert!(!token.is_cancelled());
        copy.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn native_recipe_is_safe_default() {
        let recipe = ExecutionRecipe::default();
        assert_eq!(recipe.backend, BackendKind::Native);
        assert_eq!(recipe.device_order, vec!["metal", "cpu"]);
    }
}
