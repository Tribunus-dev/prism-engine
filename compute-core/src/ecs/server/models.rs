//! Model registry types used by the server routes.
//!
//! Provides [`ModelRegistry`] and [`ModelEntry`] types, stubbed for now —
//! future versions will manage model lifecycle (loading, eviction, metadata).
//!
//! TODO: populate with real model discovery and lifecycle management.

use serde::{Deserialize, Serialize};

/// A single model in the registry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Unique model identifier (e.g. "gpt-4", "qwen2.5-7b-instruct").
    pub id: String,
    /// Human-readable model name.
    pub name: String,
    /// Model publisher / organization.
    pub owned_by: String,
    /// Number of parameters (e.g. "7B", "70B").
    pub parameter_size: String,
    /// Quantization format (e.g. "fp16", "q4_0").
    pub quantization: String,
    /// Whether the model is currently loaded in memory.
    pub is_loaded: bool,
}

/// A registry of known models, exposed via the OpenAI-compatible `/v1/models`
/// endpoint and the internal model listing API.
#[derive(Clone, Debug, Default)]
pub struct ModelRegistry {
    entries: Vec<ModelEntry>,
}

impl ModelRegistry {
    /// Create an empty model registry.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Return all registered models.
    pub fn list(&self) -> &[ModelEntry] {
        &self.entries
    }

    /// Register a model.
    pub fn register(&mut self, entry: ModelEntry) {
        self.entries.push(entry);
    }
}
