//! MLX model adapter — detects MLX format directories and parses MLX config.json
//! into a canonical [`UnifiedConfig`] for `ModelGraph::build`.
//!
//! MLX models use safetensors for weights with a HuggingFace-style config.json.
//! The adapter detects MLX by checking for safetensors files + config.json with
//! `model_type`, then parses the config and converts it to the canonical IR config.

use std::path::Path;

use prism_ecs_ir::UnifiedConfig;

// ── Detection ─────────────────────────────────────────────────────────────

/// Detect whether a directory contains an MLX model.
///
/// Returns `true` if the directory contains at least one `.safetensors` file
/// AND a parseable `config.json` with a non-empty `model_type` field.
pub fn detect_mlx_format(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }

    // Must have safetensors files
    let has_safetensors = {
        let Ok(mut entries) = std::fs::read_dir(dir) else {
            return false;
        };
        entries.any(|e| {
            e.as_ref()
                .ok()
                .is_some_and(|e| e.path().extension() == Some(std::ffi::OsStr::new("safetensors")))
        })
    };

    if !has_safetensors {
        return false;
    }

    // Must have a config.json with model_type
    let config_path = dir.join("config.json");
    let Ok(json_str) = std::fs::read_to_string(&config_path) else {
        return false;
    };
    let Ok(raw): Result<serde_json::Value, _> = serde_json::from_str(&json_str) else {
        return false;
    };

    // Check for model_type field — the canonical MLX/HuggingFace indicator
    raw.get("model_type")
        .and_then(|v| v.as_str())
        .map(|s| !s.is_empty())
        .unwrap_or(false)
}

// ── Model descriptor ───────────────────────────────────────────────────────

/// MLX model descriptor parsed from `config.json`.
///
/// Provides the same fields a HuggingFace config.json exposes, converted to
/// their canonical representation. Most of the heavy lifting is delegated to
/// [`UnifiedConfig::from_file`]; this struct adds MLX-specific metadata
/// (like `model_type`) and a lighter-weight typed surface.
#[derive(Debug, Clone)]
pub struct MlxModelDescriptor {
    /// HuggingFace model type string (e.g. "llama", "mistral").
    pub model_type: String,
    /// Hidden dimension size.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA/MQA).
    pub num_kv_heads: usize,
    /// Intermediate (FFN) dimension.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum position embeddings (context length).
    pub max_position_embeddings: usize,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE theta base frequency.
    pub rope_theta: f64,
    /// Head dimension (inferred from hidden_size / num_attention_heads if absent).
    pub head_dim: Option<usize>,
    /// Underlying canonical config for graph building.
    unified: UnifiedConfig,
}

impl MlxModelDescriptor {
    /// Parse MLX `config.json` from a model directory.
    ///
    /// Reads `<path>/config.json`, validates that required fields are present,
    /// and returns a descriptor with both MLX-specific metadata and a canonical
    /// [`UnifiedConfig`] for graph building.
    pub fn from_dir(path: &Path) -> Result<Self, String> {
        let config_path = path.join("config.json");
        if !config_path.exists() {
            return Err(format!("MLX config.json not found in {}", path.display()));
        }

        let unified = UnifiedConfig::from_file(&config_path)?;

        // Read raw JSON for MLX-specific fields not in UnifiedConfig
        let json_str =
            std::fs::read_to_string(&config_path).map_err(|e| format!("read config.json: {e}"))?;
        let raw: serde_json::Value =
            serde_json::from_str(&json_str).map_err(|e| format!("parse config.json: {e}"))?;

        let model_type = raw
            .get("model_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let hidden_size = resolved_field(&raw, "hidden_size").unwrap_or(4096);
        let num_layers = resolved_field(&raw, "num_hidden_layers").unwrap_or(32);
        let num_attention_heads = resolved_field(&raw, "num_attention_heads").unwrap_or(32);
        let num_kv_heads =
            resolved_field(&raw, "num_key_value_heads").unwrap_or(num_attention_heads);
        let intermediate_size = resolved_field(&raw, "intermediate_size").unwrap_or(11008);
        let vocab_size = resolved_field(&raw, "vocab_size").unwrap_or(151936);
        let max_position_embeddings =
            resolved_field(&raw, "max_position_embeddings").unwrap_or(2048);
        let rms_norm_eps = resolved_field_f64(&raw, "rms_norm_eps").unwrap_or(1e-5);
        let rope_theta = resolved_field_f64(&raw, "rope_theta").unwrap_or(10_000.0);

        let head_dim = if unified.head_dim > 0 {
            Some(unified.head_dim as usize)
        } else if hidden_size > 0 && num_attention_heads > 0 {
            Some(hidden_size / num_attention_heads)
        } else {
            None
        };

        Ok(MlxModelDescriptor {
            model_type,
            hidden_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            intermediate_size,
            vocab_size,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            head_dim,
            unified,
        })
    }

    /// Convert to [`UnifiedConfig`] for `ModelGraph::build`.
    pub fn to_unified_config(&self) -> UnifiedConfig {
        self.unified.clone()
    }
}

/// Resolve a u32 field from config.json, checking cascading config paths.
fn resolved_field(raw: &serde_json::Value, field: &str) -> Option<usize> {
    // Check text_config, language_config, then root
    for scope in ["text_config", "language_config"] {
        if let Some(child) = raw.get(scope) {
            if let Some(v) = child.get(field).and_then(|v| v.as_u64()) {
                return Some(v as usize);
            }
        }
    }
    raw.get(field).and_then(|v| v.as_u64()).map(|v| v as usize)
}

/// Resolve an f64 field from config.json, checking cascading config paths.
fn resolved_field_f64(raw: &serde_json::Value, field: &str) -> Option<f64> {
    for scope in ["text_config", "language_config"] {
        if let Some(child) = raw.get(scope) {
            if let Some(v) = child.get(field).and_then(|v| v.as_f64()) {
                return Some(v);
            }
        }
    }
    raw.get(field).and_then(|v| v.as_f64())
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Generate unique directory name per test to avoid parallel-test collisions.
    fn test_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mlx_test_{name}_{}", std::process::id()))
    }

    /// Write config.json into a directory.
    fn write_config(dir: &std::path::Path, config_json: &str) {
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut f = std::fs::File::create(dir.join("config.json")).unwrap();
        f.write_all(config_json.as_bytes()).unwrap();
    }

    /// Write a minimal placeholder safetensors file (valid header, no tensors).
    fn write_placeholder_safetensors(dir: &std::path::Path) {
        // Valid safetensors: header_len=4 (u64 LE), then 4 bytes of `"{}"`
        let safetensors_data: Vec<u8> = vec![4u8, 0, 0, 0, 0, 0, 0, 0, b'{', b'}', 0, 0];
        let mut sf = std::fs::File::create(dir.join("model.safetensors")).unwrap();
        sf.write_all(&safetensors_data).unwrap();
    }

    /// Helper: full test directory with config + safetensors.
    fn setup_test_dir(name: &str, config_json: &str) -> std::path::PathBuf {
        let dir = test_dir(name);
        write_config(&dir, config_json);
        write_placeholder_safetensors(&dir);
        dir
    }

    #[test]
    fn test_detect_mlx_format() {
        let config = r#"{
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32
        }"#;
        let dir = setup_test_dir("detect_mlx_format", config);

        assert!(detect_mlx_format(&dir), "should detect MLX format");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_detect_mlx_format_no_safetensors() {
        let dir = test_dir("detect_no_safetensors");
        write_config(&dir, r#"{"model_type":"llama"}"#);

        assert!(
            !detect_mlx_format(&dir),
            "should reject without safetensors"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_detect_mlx_format_no_config() {
        let dir = test_dir("detect_no_config");
        std::fs::create_dir_all(&dir).unwrap();
        write_placeholder_safetensors(&dir);

        assert!(
            !detect_mlx_format(&dir),
            "should reject without config.json"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mlx_config_parse() {
        let config = r#"{
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "head_dim": 128,
            "architectures": ["MistralForCausalLM"]
        }"#;
        let dir = setup_test_dir("config_parse", config);

        let desc = MlxModelDescriptor::from_dir(&dir).expect("should parse MLX config");
        assert_eq!(desc.model_type, "mistral");
        assert_eq!(desc.hidden_size, 4096);
        assert_eq!(desc.num_layers, 32);
        assert_eq!(desc.num_attention_heads, 32);
        assert_eq!(desc.num_kv_heads, 8);
        assert_eq!(desc.intermediate_size, 14336);
        assert_eq!(desc.vocab_size, 32000);
        assert_eq!(desc.max_position_embeddings, 4096);
        assert!((desc.rms_norm_eps - 1e-5).abs() < 1e-10);
        assert!((desc.rope_theta - 10000.0).abs() < 1.0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mlx_config_parse_llama() {
        // Minimal Llama-style config (no kv heads = use num_attention_heads)
        let config = r#"{
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "intermediate_size": 8192,
            "vocab_size": 32000,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "architectures": ["LlamaForCausalLM"]
        }"#;
        let dir = setup_test_dir("config_parse_llama", config);

        let desc = MlxModelDescriptor::from_dir(&dir).expect("should parse Llama config");
        assert_eq!(desc.model_type, "llama");
        assert_eq!(desc.hidden_size, 2048);
        assert_eq!(desc.num_layers, 16);
        assert_eq!(desc.num_attention_heads, 16);
        // No num_kv_heads in config → should default to num_attention_heads
        assert_eq!(desc.num_kv_heads, 16);
        assert_eq!(desc.intermediate_size, 8192);
        assert_eq!(desc.vocab_size, 32000);
        assert_eq!(desc.head_dim, Some(128)); // 2048 / 16 = 128

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mlx_to_unified_config() {
        let config = r#"{
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "architectures": ["MistralForCausalLM"]
        }"#;
        let dir = setup_test_dir("to_unified_config", config);

        let desc = MlxModelDescriptor::from_dir(&dir).expect("should parse MLX config");
        let unified = desc.to_unified_config();

        // Verify dimensions map correctly through the canonical config
        assert_eq!(unified.hidden_size, 4096);
        assert_eq!(unified.num_layers, 32);
        assert_eq!(unified.num_heads, 32);
        assert_eq!(unified.num_kv_heads, 8);
        assert_eq!(unified.intermediate_size, 14336);
        assert_eq!(unified.vocab_size, 32000);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mlx_parse_missing_config() {
        let dir = test_dir("parse_missing");
        std::fs::create_dir_all(&dir).unwrap();

        let result = MlxModelDescriptor::from_dir(&dir);
        assert!(result.is_err(), "should fail on missing config.json");
        assert!(
            result.unwrap_err().contains("config.json"),
            "error should mention config.json"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
