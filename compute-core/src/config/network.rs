//! Server configuration: networking, model loading, caching, speculation, cluster.
//!
//! ServerConfig and its section types loaded from config.toml, environment
//! variables, and CLI arguments (in ascending priority order).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::hardware::{FusedOperation, ModelExecutionPlan};

/// Unified server configuration loaded from config.toml, environment
/// variables, and CLI arguments (in ascending priority order).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub server: ServerConfigSection,
    pub model: ModelConfigSection,
    pub cache: CacheConfigSection,
    pub speculation: SpecConfigSection,
    pub cluster: ClusterConfigSection,
}

/// Server networking and runtime settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfigSection {
    pub port: u16,
    pub host: String,
    pub max_concurrent: u32,
    pub rate_limit_per_min: u32,
    pub rate_limit_tokens_per_sec: f64,
    pub rate_limit_burst: u64,
    pub log_level: String,
    pub runtime_mode: String,
}

/// Model loading and download policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfigSection {
    pub model_path: Option<String>,
    pub auto_download: bool,
    pub max_model_cache_gb: f64,
}

/// KV cache topology and compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheConfigSection {
    pub kv_cache_tiers: u32,
    pub compression_ratio: f64,
    pub evolkv_enabled: bool,
}

/// Speculative decoding parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpecConfigSection {
    pub draft_count: u32,
    pub draft_length: u32,
    pub spechub_enabled: bool,
}

/// EXO cluster membership and autoscaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ClusterConfigSection {
    pub exo_enabled: bool,
    pub exo_port: u16,
    pub autoscale_min: u32,
    pub autoscale_max: u32,
}

impl Default for ServerConfigSection {
    fn default() -> Self {
        Self {
            port: 11434,
            host: "0.0.0.0".into(),
            max_concurrent: 64,
            rate_limit_per_min: 60,
            rate_limit_tokens_per_sec: 100.0,
            rate_limit_burst: 1000,
            log_level: "info".into(),
            runtime_mode: "safe".into(),
        }
    }
}

impl Default for ModelConfigSection {
    fn default() -> Self {
        Self {
            model_path: None,
            auto_download: false,
            max_model_cache_gb: 16.0,
        }
    }
}

impl Default for CacheConfigSection {
    fn default() -> Self {
        Self {
            kv_cache_tiers: 3,
            compression_ratio: 0.5,
            evolkv_enabled: true,
        }
    }
}

impl Default for SpecConfigSection {
    fn default() -> Self {
        Self {
            draft_count: 4,
            draft_length: 16,
            spechub_enabled: true,
        }
    }
}

impl Default for ClusterConfigSection {
    fn default() -> Self {
        Self {
            exo_enabled: false,
            exo_port: 52415,
            autoscale_min: 1,
            autoscale_max: 8,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server: ServerConfigSection::default(),
            model: ModelConfigSection::default(),
            cache: CacheConfigSection::default(),
            speculation: SpecConfigSection::default(),
            cluster: ClusterConfigSection::default(),
        }
    }
}

impl ServerConfig {
    /// Load from config file, then environment variables.
    /// Config file path: $HOME/.tribunus/config.toml
    /// (override with TRIBUNUS_CONFIG_PATH env var).
    pub fn load() -> Self {
        let mut config = Self::default();
        let config_path = std::env::var("TRIBUNUS_CONFIG_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            format!("{}/.tribunus/config.toml", home)
        });
        if let Ok(file_config) = Self::load_config_toml(&config_path) {
            config.merge(file_config);
        }
        config.load_env_overrides();
        config
    }

    /// Parse a TOML config file into a ServerConfig.
    pub fn load_config_toml(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Cannot read config file '{}': {}", path, e))?;
        toml::from_str(&content).map_err(|e| format!("Invalid config file '{}': {}", path, e))
    }

    /// Override fields from environment variables.
    pub fn load_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("TRIBUNUS_PORT") {
            if let Ok(n) = v.parse::<u16>() {
                self.server.port = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_HOST") {
            self.server.host = v;
        }
        if let Ok(v) = std::env::var("TRIBUNUS_MAX_CONCURRENT") {
            if let Ok(n) = v.parse::<u32>() {
                self.server.max_concurrent = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_RATE_LIMIT") {
            if let Ok(n) = v.parse::<u32>() {
                self.server.rate_limit_per_min = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_RATE_LIMIT_TOKENS_PER_SEC") {
            if let Ok(f) = v.parse::<f64>() {
                self.server.rate_limit_tokens_per_sec = f;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_RATE_LIMIT_BURST") {
            if let Ok(n) = v.parse::<u64>() {
                self.server.rate_limit_burst = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_LOG_LEVEL") {
            self.server.log_level = v;
        }
        if let Ok(v) = std::env::var("TRIBUNUS_RUNTIME_MODE") {
            self.server.runtime_mode = v.to_lowercase();
        }
        if let Ok(v) = std::env::var("TRIBUNUS_MODEL_PATH") {
            self.model.model_path = Some(v);
        }
        if let Ok(v) = std::env::var("TRIBUNUS_AUTO_DOWNLOAD") {
            self.model.auto_download = v.eq_ignore_ascii_case("true") || v == "1";
        }
        if let Ok(v) = std::env::var("TRIBUNUS_MAX_MODEL_CACHE_GB") {
            if let Ok(f) = v.parse::<f64>() {
                self.model.max_model_cache_gb = f;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_KV_CACHE_TIERS") {
            if let Ok(n) = v.parse::<u32>() {
                if n >= 2 && n <= 4 {
                    self.cache.kv_cache_tiers = n;
                }
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_COMPRESSION_RATIO") {
            if let Ok(f) = v.parse::<f64>() {
                self.cache.compression_ratio = f;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_EVOLKV_ENABLED") {
            self.cache.evolkv_enabled = v.eq_ignore_ascii_case("true") || v == "1";
        }
        if let Ok(v) = std::env::var("TRIBUNUS_DRAFT_COUNT") {
            if let Ok(n) = v.parse::<u32>() {
                self.speculation.draft_count = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_DRAFT_LENGTH") {
            if let Ok(n) = v.parse::<u32>() {
                self.speculation.draft_length = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_SPECHUB_ENABLED") {
            self.speculation.spechub_enabled = v.eq_ignore_ascii_case("true") || v == "1";
        }
        if let Ok(v) = std::env::var("TRIBUNUS_EXO_ENABLED") {
            self.cluster.exo_enabled = v.eq_ignore_ascii_case("true") || v == "1";
        }
        if let Ok(v) = std::env::var("TRIBUNUS_EXO_PORT") {
            if let Ok(n) = v.parse::<u16>() {
                self.cluster.exo_port = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_AUTOSCALE_MIN") {
            if let Ok(n) = v.parse::<u32>() {
                self.cluster.autoscale_min = n;
            }
        }
        if let Ok(v) = std::env::var("TRIBUNUS_AUTOSCALE_MAX") {
            if let Ok(n) = v.parse::<u32>() {
                self.cluster.autoscale_max = n;
            }
        }
    }

    /// Override fields from CLI arguments.
    /// Must be called after load() so CLI args take highest priority.
    pub fn apply_cli_args(&mut self, args: &[String]) {
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--port" => {
                    i += 1;
                    if i < args.len() {
                        if let Ok(n) = args[i].parse::<u16>() {
                            self.server.port = n;
                        }
                    }
                }
                "--host" => {
                    i += 1;
                    if i < args.len() {
                        self.server.host = args[i].clone();
                    }
                }
                "--model" | "--model-path" => {
                    i += 1;
                    if i < args.len() {
                        self.model.model_path = Some(args[i].clone());
                    }
                }
                "--exo" => {
                    self.cluster.exo_enabled = true;
                }
                "--exo-port" => {
                    i += 1;
                    if i < args.len() {
                        if let Ok(n) = args[i].parse::<u16>() {
                            self.cluster.exo_port = n;
                        }
                    }
                }
                "--runtime-mode" => {
                    i += 1;
                    if i < args.len() {
                        self.server.runtime_mode = args[i].to_lowercase();
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }

    /// Merge another config's non-default fields into self.
    fn merge(&mut self, other: ServerConfig) {
        self.server.port = other.server.port;
        self.server.host = other.server.host;
        self.server.max_concurrent = other.server.max_concurrent;
        self.server.rate_limit_per_min = other.server.rate_limit_per_min;
        self.server.log_level = other.server.log_level;
        self.server.runtime_mode = other.server.runtime_mode;

        if other.model.model_path.is_some() {
            self.model.model_path = other.model.model_path;
        }
        self.model.auto_download = other.model.auto_download;
        self.model.max_model_cache_gb = other.model.max_model_cache_gb;

        self.cache.kv_cache_tiers = other.cache.kv_cache_tiers;
        self.cache.compression_ratio = other.cache.compression_ratio;
        self.cache.evolkv_enabled = other.cache.evolkv_enabled;

        self.speculation.draft_count = other.speculation.draft_count;
        self.speculation.draft_length = other.speculation.draft_length;
        self.speculation.spechub_enabled = other.speculation.spechub_enabled;

        self.cluster.exo_enabled = other.cluster.exo_enabled;
        self.cluster.exo_port = other.cluster.exo_port;
        self.cluster.autoscale_min = other.cluster.autoscale_min;
        self.cluster.autoscale_max = other.cluster.autoscale_max;
    }
}

/// Generate per-backend fusion plans.
pub fn generate_backend_plans(
    plan: &ModelExecutionPlan,
    backends: &[&str],
) -> HashMap<String, std::collections::HashMap<String, Vec<FusedOperation>>> {
    let mut result = std::collections::HashMap::new();
    for backend in backends {
        let layer_ops: std::collections::HashMap<String, Vec<FusedOperation>> = plan
            .layers
            .iter()
            .map(|layer| {
                (
                    layer.layer_index.to_string(),
                    layer.fused_operations.clone(),
                )
            })
            .collect();
        result.insert(backend.to_string(), layer_ops);
    }
    result
}
