//! Inference model lifecycle management.
//!
//! Wraps the inference engine for the server, providing
//! a thread-safe model registry and active session tracking.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Loaded model instance wrapping the runtime engine.
pub struct ModelInstance {
    pub name: String,
    pub model_path: PathBuf,
    // runtime: Arc<prism_ecs_server::engine::InferenceEngine>,
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
        Ok(Self {
            name,
            model_path: path.to_path_buf(),
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
    models: std::sync::Mutex<HashMap<String, Arc<ModelInstance>>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Load a model from disk and register it under its file-stem name.
    pub fn load_model(&self, path: &Path) -> Result<Arc<ModelInstance>, String> {
        let instance = ModelInstance::load(path)?;
        let name = instance.name.clone();
        let instance = Arc::new(instance);
        let mut models = self.models.lock().map_err(|e| e.to_string())?;
        models.insert(name.clone(), instance);
        models
            .get(&name)
            .cloned()
            .ok_or_else(|| "model not found after insert".to_string())
    }

    /// Retrieve a loaded model by name.
    pub fn get_model(&self, name: &str) -> Option<Arc<ModelInstance>> {
        self.models.lock().ok()?.get(name).cloned()
    }

    /// List all registered model names.
    pub fn list_models(&self) -> Vec<String> {
        self.models
            .lock()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    }
}
