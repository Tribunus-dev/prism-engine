//! Stage graph — declarative model decomposition into independently-compilable
//! stages, inspired by the vllm-omni stage graph pattern.
//!
//! Each stage maps to one model component (decoder, vision encoder, projection,
//! TTS codec, etc.) and runs the full ECS compilation pipeline independently
//! with its own quantization gates, backend target, and resource budget.
//! Stages are connected by connectors that define the data plane between them.

use std::collections::HashMap;

use crate::quantization::contract::{BackendKind, RuntimeRepresentationClass};

/// A decomposition of a model into independently-compilable stages.
///
/// Mirrors the vllm-omni `StageGraph` concept: a directed graph where each
/// node is a model component with its own compilation config, and each edge
/// is a connector that routes data between components at runtime.
///
/// # Example (Gemma 4 12B Unified)
///
/// ```ignore
/// Stage 0: TextEmbedding ───connector──→ Stage 1: Decoder(48 layers)
///                                                       ↓
///                                           Stage 2: LmHead
///
/// Stage 3: VisionEncoder ───connector──→ Stage 1 (multimodal)
/// Stage 4: AudioEncoder  ───connector──→ Stage 1 (multimodal)
/// ```
#[derive(Debug, Clone)]
pub struct StageGraph {
    pub stages: Vec<StageConfig>,
    pub connectors: Vec<ConnectorEdge>,
}

/// Configuration for one compilation stage.
#[derive(Debug, Clone)]
pub struct StageConfig {
    /// Unique stage identifier within this graph.
    pub stage_id: u32,
    /// Which model component this stage represents.
    pub component: ComponentType,
    /// Safetensor key patterns for tensors that belong to this stage.
    /// Supports glob-like matching (e.g. "model.layers.*.self_attn.*").
    pub tensor_key_patterns: Vec<String>,
    /// Quantization gates for this stage (per-format thresholds).
    pub quantization: StageQuantizationConfig,
    /// Target backend for this stage's compiled weights.
    pub backend: BackendKind,
    /// Fraction of GPU memory budget for this stage (0.0–1.0).
    pub gpu_memory_utilization: f32,
    /// Tensor parallelism degree (1 = no TP).
    pub tensor_parallel_size: u32,
}

/// Which model component a stage represents.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ComponentType {
    TextEmbedding,
    DecoderLayer,
    LmHead,
    Norm,
    VisionEncoder,
    AudioEncoder,
    Projection,
    MtpDraft,
    /// Custom component not covered by the standard variants.
    Custom(String),
}

impl ComponentType {
    /// Human-readable label for diagnostics.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TextEmbedding => "text_embedding",
            Self::DecoderLayer => "decoder_layer",
            Self::LmHead => "lm_head",
            Self::Norm => "norm",
            Self::VisionEncoder => "vision_encoder",
            Self::AudioEncoder => "audio_encoder",
            Self::Projection => "projection",
            Self::MtpDraft => "mtp_draft",
            Self::Custom(_) => "custom",
        }
    }
}

/// Per-stage quantization gates.
///
/// Each stage declares which formats are eligible and what NRMSE / zero-collapse
/// thresholds to apply during admission.  This fixes the ternary rejection bug
/// where decoder layers used NF4's tight zero-collapse gate instead of
/// ternary's wider gate.
#[derive(Debug, Clone)]
pub struct StageQuantizationConfig {
    /// Formats this stage is permitted to use (in priority order).
    pub permitted_formats: Vec<RuntimeRepresentationClass>,
    /// Per-format weight-space NRMSE threshold overrides.
    /// Stage-level thresholds take precedence over the global defaults
    /// in `weight_screening_threshold()`.
    pub weight_nrmse_thresholds: HashMap<RuntimeRepresentationClass, f64>,
    /// Per-format zero-collapse ratio threshold.
    /// A format is rejected if `zero_collapse_ratio > threshold`.
    /// Ternary needs ~0.85; NF4 needs ~0.0007.
    pub zero_collapse_thresholds: HashMap<RuntimeRepresentationClass, f64>,
}

impl StageQuantizationConfig {
    /// Default config for decoder layers:
    /// tries Ternary first, falls back to NF4, then INT8.
    pub fn decoder_default() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert(RuntimeRepresentationClass::TernaryTile640Base, 0.02);
        thresholds.insert(RuntimeRepresentationClass::Nf4Tile640Base, 0.01);
        thresholds.insert(RuntimeRepresentationClass::Int8Tile640Base, 0.005);

        let mut zero_collapse = HashMap::new();
        zero_collapse.insert(RuntimeRepresentationClass::TernaryTile640Base, 0.85);
        zero_collapse.insert(RuntimeRepresentationClass::Nf4Tile640Base, 0.0007);

        Self {
            permitted_formats: vec![
                RuntimeRepresentationClass::Nf4Tile640Base,
                RuntimeRepresentationClass::TernaryTile640Base,
                RuntimeRepresentationClass::Int8Tile640Base,
                RuntimeRepresentationClass::RawF32,
            ],
            weight_nrmse_thresholds: thresholds,
            zero_collapse_thresholds: zero_collapse,
        }
    }

    /// Default config for vision/audio encoders (NF4-heavy).
    pub fn encoder_default() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert(RuntimeRepresentationClass::Nf4Tile640Base, 0.01);
        thresholds.insert(RuntimeRepresentationClass::Int8Tile640Base, 0.005);

        let mut zero_collapse = HashMap::new();
        zero_collapse.insert(RuntimeRepresentationClass::Nf4Tile640Base, 0.0007);

        Self {
            permitted_formats: vec![
                RuntimeRepresentationClass::Nf4Tile640Base,
                RuntimeRepresentationClass::Int8Tile640Base,
                RuntimeRepresentationClass::RawF32,
            ],
            weight_nrmse_thresholds: thresholds,
            zero_collapse_thresholds: zero_collapse,
        }
    }

    /// Default config for projection matrices (INT8-preferring).
    pub fn projection_default() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert(RuntimeRepresentationClass::Int8Tile640Base, 0.005);
        thresholds.insert(RuntimeRepresentationClass::Nf4Tile640Base, 0.01);

        Self {
            permitted_formats: vec![
                RuntimeRepresentationClass::Int8Tile640Base,
                RuntimeRepresentationClass::Nf4Tile640Base,
                RuntimeRepresentationClass::RawF32,
            ],
            weight_nrmse_thresholds: thresholds,
            zero_collapse_thresholds: HashMap::new(),
        }
    }
}

/// A connector edge between two stages.
///
/// At runtime, data flows from `from_stage` to `to_stage` through the
/// specified connector backend.  Multiple edges to the same `to_stage`
/// are merged at the receiver (e.g., multimodal inputs).
#[derive(Debug, Clone)]
pub struct ConnectorEdge {
    pub from_stage: u32,
    pub to_stage: u32,
    pub connector_kind: ConnectorKind,
}

/// Supported connector transport backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConnectorKind {
    /// Intra-process shared memory buffer (fastest).
    SharedMemory,
    /// Same-process buffer copy (simple, no shared memory).
    LocalBuffer,
    /// Remote TCP connection (for future multi-node exo cluster).
    RemoteTcp,
}

impl ConnectorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SharedMemory => "shared_memory",
            Self::LocalBuffer => "local_buffer",
            Self::RemoteTcp => "remote_tcp",
        }
    }
}

impl StageGraph {
    /// Build a stage graph for a standard decoder-only LLM.
    pub fn decoder_only(_vocab_size: u32, _num_layers: u32) -> Self {
        Self {
            stages: vec![
                StageConfig {
                    stage_id: 0,
                    component: ComponentType::TextEmbedding,
                    tensor_key_patterns: vec!["model.embed_tokens.weight".into()],
                    quantization: StageQuantizationConfig::projection_default(),
                    backend: BackendKind::Metal,
                    gpu_memory_utilization: 0.3,
                    tensor_parallel_size: 1,
                },
                StageConfig {
                    stage_id: 1,
                    component: ComponentType::DecoderLayer,
                    tensor_key_patterns: vec!["model.layers.*.weight".into()],
                    quantization: StageQuantizationConfig::decoder_default(),
                    backend: BackendKind::Metal,
                    gpu_memory_utilization: 0.6,
                    tensor_parallel_size: 1,
                },
                StageConfig {
                    stage_id: 2,
                    component: ComponentType::LmHead,
                    tensor_key_patterns: vec!["lm_head.weight".into()],
                    quantization: StageQuantizationConfig::projection_default(),
                    backend: BackendKind::Metal,
                    gpu_memory_utilization: 0.1,
                    tensor_parallel_size: 1,
                },
            ],
            connectors: vec![
                ConnectorEdge {
                    from_stage: 0,
                    to_stage: 1,
                    connector_kind: ConnectorKind::LocalBuffer,
                },
                ConnectorEdge {
                    from_stage: 1,
                    to_stage: 2,
                    connector_kind: ConnectorKind::LocalBuffer,
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_only_builds() {
        let graph = StageGraph::decoder_only(32000, 48);
        assert_eq!(graph.stages.len(), 3);
        assert_eq!(graph.connectors.len(), 2);
        assert_eq!(graph.stages[0].component, ComponentType::TextEmbedding);
        assert_eq!(graph.stages[1].component, ComponentType::DecoderLayer);
        assert_eq!(graph.stages[2].component, ComponentType::LmHead);
    }

    #[test]
    fn decoder_default_has_ternary() {
        let config = StageQuantizationConfig::decoder_default();
        assert!(config.permitted_formats.contains(
            &RuntimeRepresentationClass::TernaryTile640Base));
        assert!(config.zero_collapse_thresholds.contains_key(
            &RuntimeRepresentationClass::TernaryTile640Base));
        // Ternary zero-collapse gate should be ~0.85, not the NF4 0.0007
        let ternary_zc = config.zero_collapse_thresholds
            [&RuntimeRepresentationClass::TernaryTile640Base];
        assert!(ternary_zc > 0.5, "ternary zero-collapse gate should be wide");
    }

    #[test]
    fn component_labels() {
        assert_eq!(ComponentType::TextEmbedding.as_str(), "text_embedding");
        assert_eq!(ComponentType::MtpDraft.as_str(), "mtp_draft");
        assert_eq!(ComponentType::Custom("x".into()).as_str(), "custom");
    }
}
