//! Inference model lifecycle management.
//!
//! Wraps the inference engine for the server, providing
//! a thread-safe model registry and active session tracking.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::engine::model::Model;
use crate::runtime::wire_runtime::WirePrefillDecodeRuntime;
use prism_ecs_ir::model_graph::{ModelGraph, UnifiedConfig};

/// Loaded model instance wrapping the runtime engine.
pub struct ModelInstance {
    pub name: String,
    pub model_path: PathBuf,
    /// Shared production runtime used by dashboard/API generation.
    pub runtime: Arc<WirePrefillDecodeRuntime>,
    /// Declarative execution metadata surfaced to clients and diagnostics.
    pub execution_recipe: crate::runtime::ExecutionRecipe,
    pub backend_capabilities: crate::runtime::BackendCapabilities,
}

impl ModelInstance {
    /// Load a model from the given file path.
    ///
    /// The model name is derived from the file stem.
    pub fn load(path: &Path) -> Result<Self, String> {
        let name = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let model = Model::load(path)?;
        let config_path = path
            .parent()
            .ok_or_else(|| "model has no parent directory".to_string())?
            .join("config.json");
        let config = UnifiedConfig::from_file(&config_path)
            .map_err(|e| format!("load model config {}: {e}", config_path.display()))?;
        let graph = ModelGraph::build(&config);
        let mut model = model;
        let tokenizer_path = path
            .parent()
            .map(|dir| dir.join("tokenizer.json"))
            .filter(|p| p.exists())
            .ok_or_else(|| format!("tokenizer.json not found beside {}", path.display()))?;
        if let Some(metadata) = model.metadata.as_object_mut() {
            metadata.insert(
                "tokenizer_path".to_string(),
                serde_json::Value::String(tokenizer_path.to_string_lossy().into_owned()),
            );
        }
        let eos_id = model.metadata["eos_token_id"].as_u64().unwrap_or(0) as u32;
        let runtime = Arc::new(WirePrefillDecodeRuntime::new(model, graph, eos_id));
        Ok(Self {
            name,
            model_path: path.to_path_buf(),
            runtime,
            execution_recipe: crate::runtime::ExecutionRecipe::default(),
            backend_capabilities: crate::runtime::BackendCapabilities {
                backend: crate::runtime::BackendKind::Native,
                name: "prism".into(),
                devices: vec!["metal".into(), "cpu".into()],
                modalities: vec!["text".into()],
                max_context_tokens: None,
                supports_streaming: true,
                supports_cancellation: true,
                supports_tool_calling: false,
            },
        })
    }
}

/// Active inference session with KV cache and generation state.
pub struct InferenceSession {
    pub session_id: String,
    pub model: Arc<ModelInstance>,
    pub kv_cache_handle: Option<String>,
    pub created_at: std::time::Instant,
}

/// Thread-safe registry of loaded models.
pub struct ModelRegistry {
    state: Arc<RegistryState>,
}

struct RegistryState {
    models: std::sync::Mutex<HashMap<String, Arc<ModelInstance>>>,
    usage: std::sync::Mutex<HashMap<String, ModelUsage>>,
    max_loaded: usize,
}

struct ModelUsage {
    active: u32,
    last_used: Instant,
}

/// A lease held for the lifetime of one inference request. Its Drop
/// implementation makes eviction safe even when generation fails or the
/// client disconnects.
pub struct ModelLease {
    pub model: Arc<ModelInstance>,
    state: Arc<RegistryState>,
}

impl Drop for ModelLease {
    fn drop(&mut self) {
        if let Ok(mut usage) = self.state.usage.lock() {
            if let Some(entry) = usage.get_mut(&self.model.name) {
                entry.active = entry.active.saturating_sub(1);
                entry.last_used = Instant::now();
            }
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RegistryState {
                models: std::sync::Mutex::new(HashMap::new()),
                usage: std::sync::Mutex::new(HashMap::new()),
                max_loaded: 4,
            }),
        }
    }

    /// Load a model from disk and register it under its file-stem name.
    pub fn load_model(&self, path: &Path) -> Result<Arc<ModelInstance>, String> {
        let instance = ModelInstance::load(path)?;
        let name = instance.name.clone();
        let instance = Arc::new(instance);
        let mut models = self.state.models.lock().map_err(|e| e.to_string())?;
        if models.len() >= self.state.max_loaded && !models.contains_key(&name) {
            let mut usage = self.state.usage.lock().map_err(|e| e.to_string())?;
            let victim = usage
                .iter()
                .filter(|(_, u)| u.active == 0)
                .min_by_key(|(_, u)| u.last_used)
                .map(|(name, _)| name.clone())
                .ok_or_else(|| {
                    "model capacity reached; all loaded models are active".to_string()
                })?;
            models.remove(&victim);
            usage.remove(&victim);
        }
        models.insert(name.clone(), instance);
        self.state.usage.lock().map_err(|e| e.to_string())?.insert(
            name.clone(),
            ModelUsage {
                active: 0,
                last_used: Instant::now(),
            },
        );
        models
            .get(&name)
            .cloned()
            .ok_or_else(|| "model not found after insert".to_string())
    }

    /// Retrieve a loaded model by name.
    pub fn get_model(&self, name: &str) -> Option<Arc<ModelInstance>> {
        self.state.models.lock().ok()?.get(name).cloned()
    }

    /// List all registered model names.
    pub fn list_models(&self) -> Vec<String> {
        self.state
            .models
            .lock()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    }

    pub fn acquire(&self, name: &str) -> Result<ModelLease, String> {
        let model = self
            .get_model(name)
            .ok_or_else(|| format!("model '{name}' not loaded"))?;
        let mut usage = self.state.usage.lock().map_err(|e| e.to_string())?;
        let entry = usage.entry(name.to_string()).or_insert(ModelUsage {
            active: 0,
            last_used: Instant::now(),
        });
        entry.active += 1;
        entry.last_used = Instant::now();
        Ok(ModelLease {
            model,
            state: Arc::clone(&self.state),
        })
    }

    pub fn residency_snapshot(&self) -> Vec<serde_json::Value> {
        let usage = self.state.usage.lock().ok();
        self.list_models().into_iter().map(|name| {
            let item = usage.as_ref().and_then(|u| u.get(&name));
            serde_json::json!({"model": name, "active_leases": item.map(|u| u.active).unwrap_or(0), "resident": true})
        }).collect()
    }
}
