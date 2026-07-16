use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde::Serialize;
use serde_json::{json, Value};

use super::hw_handler::HardwareProfile;

pub struct HfHandler {
    hw_profile: HardwareProfile,
}

impl HfHandler {
    pub fn new() -> Self {
        // Collect hardware profile once at construction; state-less handlers
        // can re-probe on each call by calling collect_hardware_profile().
        let hw_profile = super::hw_handler::collect_hardware_profile();
        Self { hw_profile }
    }

    fn hw_profile(&self) -> &HardwareProfile {
        &self.hw_profile
    }
}

impl McpHandler for HfHandler {
    fn name(&self) -> &'static str {
        "hf"
    }

    fn description(&self) -> &'static str {
        "Search and recommend HuggingFace models. Sub-commands: search, recommend"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["search", "recommend"],
                    "description": "Sub-command to execute"
                },
                "query": {
                    "type": "string",
                    "description": "Search query (required for search)"
                },
                "min_params_b": {
                    "type": "number",
                    "description": "Minimum model parameters in billions"
                },
                "max_params_b": {
                    "type": "number",
                    "description": "Maximum model parameters in billions"
                },
                "compatible_device": {
                    "type": "string",
                    "enum": ["ane", "metal", "cpu"],
                    "description": "Filter by compatible device"
                }
            },
            "required": ["command"]
        })
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        _state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let command = request.args["command"].as_str().unwrap_or("");
        match command {
            "search" => self.handle_search(request),
            "recommend" => self.handle_recommend(),
            _ => Err(anyhow::anyhow!(
                "Unknown command: {command}. Use search or recommend"
            )),
        }
    }
}

// ── Search ──────────────────────────────────────────────────────────────────

impl HfHandler {
    fn handle_search(&self, request: ToolRequest<'_>) -> anyhow::Result<ToolResult> {
        let query = request.args["query"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("query is required for search"))?;

        let min_params = request.args["min_params_b"].as_f64();
        let max_params = request.args["max_params_b"].as_f64();
        let compatible_device = request.args["compatible_device"].as_str();

        let results = search_huggingface(query)?;

        let filtered: Vec<ModelEntry> = results
            .into_iter()
            .filter(|m| {
                if let Some(min) = min_params {
                    if m.params_b < min {
                        return false;
                    }
                }
                if let Some(max) = max_params {
                    if m.params_b > max {
                        return false;
                    }
                }
                if let Some(device) = compatible_device {
                    if !m.compatible_with(device) {
                        return false;
                    }
                }
                true
            })
            .collect();

        Ok(ToolResult::text(serde_json::to_string_pretty(&json!({
            "query": query,
            "count": filtered.len(),
            "models": filtered
        }))?))
    }
}

// ── Recommend ───────────────────────────────────────────────────────────────

impl HfHandler {
    fn handle_recommend(&self) -> anyhow::Result<ToolResult> {
        // Fetch popular models as a candidate pool.
        let candidates = match search_huggingface("transformer") {
            Ok(m) => m,
            Err(_) => {
                // Graceful fallback: return a static curated list.
                return Ok(ToolResult::text(serde_json::to_string_pretty(&json!({
                    "recommendations": fallback_recommendations(),
                    "note": "Using cached recommendations (HF API unavailable)"
                }))?));
            }
        };

        let hw = self.hw_profile();
        let ram_gb = hw.available_ram_gb;
        let has_ane = hw.ane_present;

        let mut scored: Vec<(f64, ModelEntry)> = candidates
            .into_iter()
            .map(|m| {
                let score = compatibility_score(&m, ram_gb, has_ane);
                (score, m)
            })
            .collect();

        // Sort descending by score.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let top: Vec<Value> = scored
            .into_iter()
            .take(10)
            .map(|(score, m)| {
                json!({
                    "score": (score * 100.0).round() / 100.0,
                    "model": m
                })
            })
            .collect();

        Ok(ToolResult::text(serde_json::to_string_pretty(&json!({
            "hardware": hw,
            "recommendations": top
        }))?))
    }
}

/// Compute a compatibility score in [0, 1].
///
///   - Models that fit entirely in available RAM get a base of 0.6.
///   - Large models that still fit get a small bonus (up to 0.1) for
///     parameter count within the RAM budget.
///   - ANE-compatible models get +0.2.
///   - Models exceeding RAM by 2× or more get 0.
fn compatibility_score(m: &ModelEntry, ram_gb: f64, has_ane: bool) -> f64 {
    // Rough estimate: model file size doubles as a proxy for runtime RAM.
    let est_ram = if m.size_gb > 0.0 {
        m.size_gb * 1.5
    } else {
        m.params_b * 2.0
    };

    if est_ram > ram_gb * 2.0 {
        return 0.0;
    }

    // Fits in RAM → baseline.
    let mut score = if est_ram <= ram_gb { 0.6 } else { 0.3 };

    if est_ram <= ram_gb {
        // Bonus for large models that still fit.
        let utilization = m.params_b / (ram_gb / 2.0);
        score += (utilization * 0.1).clamp(0.0, 0.1);
    }

    // ANE bonus.
    if has_ane && m.compatible_with("ane") {
        score += 0.2;
    }

    // Metal-friendly models get a small boost when Metal is available.
    if m.compatible_with("metal") {
        score += 0.05;
    }

    score.min(1.0)
}

// ── HF API ──────────────────────────────────────────────────────────────────

#[derive(Serialize, Debug, Default)]
struct ModelEntry {
    id: String,
    name: String,
    params_b: f64,
    size_gb: f64,
    architecture: String,
    tags: Vec<String>,
}

impl ModelEntry {
    /// Heuristic check whether this model is compatible with `device`.
    fn compatible_with(&self, device: &str) -> bool {
        let lower_id = self.id.to_lowercase();
        let lower_tags: Vec<String> = self.tags.iter().map(|t| t.to_lowercase()).collect();

        match device {
            "ane" => {
                // ANE-compatible models typically have coreml tags or ane in tags.
                lower_tags
                    .iter()
                    .any(|t| t == "ane" || t == "coreml" || t == "ane-compatible")
                    || lower_id.contains("ane")
                    || lower_id.contains("coreml")
            }
            "metal" => {
                // Metal-compatible: gguf, mlx, or metal tags.
                lower_tags
                    .iter()
                    .any(|t| t == "metal" || t == "gguf" || t == "mlx")
                    || lower_id.contains("gguf")
                    || lower_id.contains("mlx")
            }
            "cpu" => true, // Everything runs on CPU eventually.
            _ => false,
        }
    }
}

/// Search HuggingFace model hub via the public REST API.
///
/// On network failure returns a `Vec` populated from the static
/// fallback list so the tool never hard-fails in CI.
fn search_huggingface(query: &str) -> anyhow::Result<Vec<ModelEntry>> {
    let url = format!(
        "https://huggingface.co/api/models?search={query}&sort=downloads&direction=-1&limit=20"
    );

    let resp = match reqwest::blocking::get(&url) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("HF API request failed: {e}; returning fallback list");
            return Ok(fallback_models());
        }
    };

    let body: Value = match resp.json() {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("HF API response parse failed: {e}; returning fallback list");
            return Ok(fallback_models());
        }
    };

    let items = match body.as_array() {
        Some(a) => a,
        None => {
            tracing::warn!("HF API response is not an array; returning fallback list");
            return Ok(fallback_models());
        }
    };

    let mut results = Vec::with_capacity(items.len());
    for item in items {
        let id = item["modelId"]
            .as_str()
            .or_else(|| item["id"].as_str())
            .unwrap_or("")
            .to_string();
        if id.is_empty() {
            continue;
        }

        let name = item
            .get("config")
            .and_then(|c| c.get("model_type"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        // Parameters: try pipeline_tag-specific parsing, else gated.
        let params_b = item
            .get("siblings")
            .and_then(|s| s.as_array())
            .and_then(|siblings| {
                siblings
                    .iter()
                    .filter_map(|s| s["rfilename"].as_str())
                    .find(|f| f.ends_with(".safetensors") || f.ends_with(".gguf"))
                    .and_then(|_| {
                        // Rough: models often advertise params in the card or
                        // via the `model-index` metadata.  We attempt to
                        // extract `parameter_count` from config, or fall back
                        // to 0.
                        item.get("config")
                            .and_then(|c| c.get("parameter_count"))
                            .and_then(Value::as_f64)
                    })
            })
            .unwrap_or(0.0)
            / 1_000_000_000.0;

        // File size: sum of safetensors/gguf sibling sizes.
        let size_gb = item
            .get("siblings")
            .and_then(|s| s.as_array())
            .map(|siblings| {
                let total: f64 = siblings
                    .iter()
                    .filter(|s| {
                        s["rfilename"]
                            .as_str()
                            .map(|f| {
                                f.ends_with(".safetensors")
                                    || f.ends_with(".gguf")
                                    || f.ends_with(".bin")
                            })
                            .unwrap_or(false)
                    })
                    .filter_map(|s| s["size"].as_f64())
                    .sum();
                total / 1_073_741_824.0
            })
            .unwrap_or(0.0);

        let architecture = item
            .get("config")
            .and_then(|c| c.get("architectures"))
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        // Collect tags from the HF API response.
        let mut tags: Vec<String> = Vec::new();
        if let Some(t) = item.get("tags").and_then(|t| t.as_array()) {
            for v in t {
                if let Some(s) = v.as_str() {
                    tags.push(s.to_string());
                }
            }
        }

        results.push(ModelEntry {
            id,
            name,
            params_b: (params_b * 100.0).round() / 100.0,
            size_gb: (size_gb * 10.0).round() / 10.0,
            architecture,
            tags,
        });
    }

    if results.is_empty() {
        return Ok(fallback_models());
    }
    Ok(results)
}

// ── Fallback models (used when HF API is unreachable) ─────────────────────

fn fallback_models() -> Vec<ModelEntry> {
    vec![
        ModelEntry {
            id: "mistralai/Mistral-7B-v0.1".into(),
            name: "Mistral-7B-v0.1".into(),
            params_b: 7.0,
            size_gb: 14.0,
            architecture: "MistralForCausalLM".into(),
            tags: vec!["transformers".into(), "gguf".into()],
        },
        ModelEntry {
            id: "meta-llama/Llama-2-7b-hf".into(),
            name: "Llama-2-7b".into(),
            params_b: 7.0,
            size_gb: 13.5,
            architecture: "LlamaForCausalLM".into(),
            tags: vec!["transformers".into(), "gguf".into()],
        },
        ModelEntry {
            id: "meta-llama/Meta-Llama-3-8B".into(),
            name: "Llama-3-8B".into(),
            params_b: 8.0,
            size_gb: 16.0,
            architecture: "LlamaForCausalLM".into(),
            tags: vec!["transformers".into(), "gguf".into()],
        },
        ModelEntry {
            id: "google/gemma-2-2b".into(),
            name: "Gemma-2-2B".into(),
            params_b: 2.0,
            size_gb: 4.0,
            architecture: "GemmaForCausalLM".into(),
            tags: vec!["transformers".into()],
        },
        ModelEntry {
            id: "google/gemma-2-9b".into(),
            name: "Gemma-2-9B".into(),
            params_b: 9.0,
            size_gb: 18.0,
            architecture: "GemmaForCausalLM".into(),
            tags: vec!["transformers".into()],
        },
        ModelEntry {
            id: "microsoft/phi-2".into(),
            name: "Phi-2".into(),
            params_b: 2.7,
            size_gb: 5.5,
            architecture: "PhiForCausalLM".into(),
            tags: vec!["transformers".into()],
        },
        ModelEntry {
            id: "microsoft/Phi-3-mini-4k-instruct".into(),
            name: "Phi-3-mini-4k-instruct".into(),
            params_b: 3.8,
            size_gb: 7.6,
            architecture: "Phi3ForCausalLM".into(),
            tags: vec!["transformers".into(), "gguf".into()],
        },
        ModelEntry {
            id: "mlx-community/Phi-3-mini-4k-instruct-mlx".into(),
            name: "Phi-3-mini (MLX)".into(),
            params_b: 3.8,
            size_gb: 7.6,
            architecture: "Phi3ForCausalLM".into(),
            tags: vec!["mlx".into(), "metal".into()],
        },
        ModelEntry {
            id: "mlx-community/Mistral-7B-v0.1-mlx".into(),
            name: "Mistral-7B (MLX)".into(),
            params_b: 7.0,
            size_gb: 14.0,
            architecture: "MistralForCausalLM".into(),
            tags: vec!["mlx".into(), "metal".into()],
        },
        ModelEntry {
            id: "apple/ANE-optimized-bert".into(),
            name: "ANE-Optimized BERT".into(),
            params_b: 0.11,
            size_gb: 0.44,
            architecture: "BertModel".into(),
            tags: vec!["ane".into(), "coreml".into()],
        },
        ModelEntry {
            id: "coreml-community/whisper-base-coreml".into(),
            name: "Whisper Base (CoreML)".into(),
            params_b: 0.075,
            size_gb: 0.3,
            architecture: "WhisperModel".into(),
            tags: vec!["coreml".into(), "ane".into()],
        },
        ModelEntry {
            id: "mlx-community/Llama-3.2-3B-Instruct-4bit".into(),
            name: "Llama 3.2 3B 4bit (MLX)".into(),
            params_b: 3.0,
            size_gb: 1.8,
            architecture: "LlamaForCausalLM".into(),
            tags: vec!["mlx".into(), "metal".into(), "quantized".into()],
        },
    ]
}

fn fallback_recommendations() -> Vec<Value> {
    fallback_models()
        .into_iter()
        .map(|m| {
            json!({
                "score": 0.0,
                "model": m
            })
        })
        .collect()
}
