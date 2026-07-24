//! Canonical compiler configuration and receipt types.
//!
//! Shared by CLI, standalone API, and DaemonCompilerDispatcher. Defines the
//! configuration surface for Prism's unified compilation pipeline.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use prism_ecs_ir::ArchitectureFamily;

/// Canonical compilation request/configuration shared by CLI, standalone API,
/// and DaemonCompilerDispatcher.
pub struct CanonicalCompileConfig {
    /// Path to the source GGUF file.
    pub source_path: PathBuf,
    /// Path for the output .cimage file.
    pub output_path: PathBuf,
    /// Target hardware profile identifier (e.g. "apple-m1", "apple-m2-pro").
    pub target_hardware: String,
    /// Evolution configuration.
    pub evolution: EvolutionConfig,
    /// Explicit random seed for deterministic search. None = random.
    pub seed: Option<u64>,
    /// Population size for evolutionary search.
    pub population_size: usize,
    /// Maximum generations.
    pub generation_limit: usize,
    /// Stall limit (generations with no improvement before early stop).
    pub stall_limit: usize,
    /// Candidate evaluation budget (max candidates to evaluate).
    pub candidate_budget: Option<usize>,
    /// Wall-clock time budget in seconds.
    pub time_budget_secs: Option<u64>,
    /// Which calibration to use. "auto" = generate on current machine.
    pub calibration: CalibrationPolicy,
    /// Validation policy for candidates.
    pub validation: ValidationPolicy,
    /// Progress callback (phase, current, total, elapsed_secs, estimated_total_secs).
    pub progress: Option<Box<dyn Fn(&str, u32, u32, f64, f64) + Send>>,
    /// Cancellation token.
    pub cancel: Option<Arc<AtomicBool>>,
    /// Enable evolution search (default true).
    pub evolution_enabled: bool,
    /// When true, enforce production-level compilation:
    /// - Real evaluator required when evolution is enabled
    /// - SpatialIR legalization failure is a hard error (no silent fallback)
    /// - All compilation paths enforce strict validation
    pub production_mode: bool,
}

impl Default for CanonicalCompileConfig {
    fn default() -> Self {
        Self {
            source_path: PathBuf::new(),
            output_path: PathBuf::new(),
            target_hardware: String::new(),
            evolution: EvolutionConfig::default(),
            seed: None,
            population_size: 50,
            generation_limit: 100,
            stall_limit: 10,
            candidate_budget: None,
            time_budget_secs: None,
            calibration: CalibrationPolicy::Auto,
            validation: ValidationPolicy::Strict,
            progress: None,
            cancel: None,
            evolution_enabled: true,
            production_mode: false,
        }
    }
}

/// Configuration for a canonical model source — format-agnostic model descriptor.
#[derive(Debug, Clone)]
pub struct CanonicalModelConfig {
    /// Path to the source model file.
    pub source_path: PathBuf,
    /// Path for the output artifact.
    pub output_path: PathBuf,
    /// Target hardware profile identifier.
    pub target_hardware: String,
}

/// Determines how calibration data is sourced during compilation.
pub enum CalibrationPolicy {
    /// Auto-generate calibration on the current machine.
    Auto,
    /// Load from a specific calibration report path.
    Load(PathBuf),
    /// Use the given calibration ID (must already be cached).
    UseCached(String),
}

/// Determines how candidates are validated during compilation.
pub enum ValidationPolicy {
    /// Validate all candidates (slow, thorough).
    Strict,
    /// Validate only finalists.
    FinalistsOnly,
    /// Skip validation (fast, unsafe).
    Skip,
}

/// Configuration for the evolutionary search engine.
pub struct EvolutionConfig {
    /// Crossover rate (0.0–1.0).
    pub crossover_rate: f64,
    /// Mutation rate (0.0–1.0).
    pub mutation_rate: f64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            crossover_rate: 0.8,
            mutation_rate: 0.1,
        }
    }
}

/// Receipt produced by a successful canonical compilation.
///
/// Records the source identity, architecture, search parameters, and output
/// digests for auditability and reproducibility.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct CanonicalCompileReceipt {
    /// SHA-256 digest of the source GGUF file.
    pub source_gguf_digest: String,
    /// Detected architecture family from the GGUF header.
    pub detected_architecture: ArchitectureFamily,
    /// Identity string for the compiler binary that produced this receipt.
    pub compiler_identity: String,
    /// Digest of the calibration data used.
    pub calibration_digest: String,
    /// Number of candidates evaluated during search.
    pub candidates_evaluated: usize,
    /// Number of generations completed.
    pub generations_completed: usize,
    /// Human-readable reason search stopped (e.g. "generation_limit", "stall_limit", "time_budget").
    pub stopping_reason: String,
    /// SHA-256 digest of the output .cimage file.
    pub cimage_digest: String,
    /// Unix timestamp (seconds since epoch) when compilation started.
    pub started_at: u64,
    /// Unix timestamp (seconds since epoch) when compilation finished.
    pub finished_at: u64,
    /// Source file format (e.g. "gguf", "safetensors").
    pub source_format: String,
    /// Digest of the tensor manifest extracted from the source.
    pub tensor_manifest_digest: String,
    /// Digest of the input fed to the evolution search.
    pub search_input_digest: String,
    /// Digest of the selected FormatPlan from evolution search.
    pub format_plan_digest: String,
    /// Digest of the kernel manifest / contract.
    pub kernel_manifest_digest: String,
    /// Digest of the genome that produced the winning candidate.
    pub genome_digest: String,
    /// ISO-8601 timestamp when compilation started.
    pub timestamp: String,
    /// Duration of compilation in milliseconds.
    pub duration_ms: u64,
    /// Mode of evaluation used during search.
    /// "measured" = real hardware evaluation, "synthetic" = no hardware, "none" = no search.
    pub evaluation_mode: String,
    /// How SpatialIR legalization was handled during compilation.
    /// "strict" = production mode with successful SpatialIR,
    /// "none" = no palettized tensors to legalize,
    /// "additive" = non-production mode (silent fallback on failure),
    /// "failed" = production mode failure (should not appear in receipt).
    pub legalization_mode: String,
    /// Ordered forensic trace of the compilation pipeline.
    pub forensic_log: Vec<CompilationEvent>,
}

/// One durable event in the canonical compilation trace.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct CompilationEvent {
    pub sequence: u64,
    pub phase: String,
    pub event: String,
    pub elapsed_ms: u64,
    pub duration_ms: u64,
    pub details: BTreeMap<String, String>,
}

impl CanonicalCompileReceipt {
    /// Create a new receipt with default values for provenance fields.
    pub fn new(source_path: &str, cimage_path: &str) -> Self {
        Self {
            source_gguf_digest: source_path.to_string(),
            detected_architecture: ArchitectureFamily::Llama,
            compiler_identity: String::new(),
            calibration_digest: String::new(),
            candidates_evaluated: 0,
            generations_completed: 0,
            stopping_reason: String::new(),
            cimage_digest: cimage_path.to_string(),
            started_at: 0,
            finished_at: 0,
            source_format: String::new(),
            tensor_manifest_digest: String::new(),
            search_input_digest: String::new(),
            format_plan_digest: String::new(),
            kernel_manifest_digest: String::new(),
            genome_digest: String::new(),
            timestamp: String::new(),
            duration_ms: 0,
            evaluation_mode: "synthetic".to_string(),
            legalization_mode: "none".to_string(),
            forensic_log: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_compile_config_default() {
        let config = CanonicalCompileConfig::default();
        assert_eq!(config.source_path, PathBuf::new());
        assert_eq!(config.output_path, PathBuf::new());
        assert_eq!(config.target_hardware, "");
        assert_eq!(config.evolution.crossover_rate, 0.8);
        assert_eq!(config.evolution.mutation_rate, 0.1);
        assert_eq!(config.seed, None);
        assert_eq!(config.population_size, 50);
        assert_eq!(config.generation_limit, 100);
        assert_eq!(config.stall_limit, 10);
        assert_eq!(config.candidate_budget, None);
        assert_eq!(config.time_budget_secs, None);
        assert!(matches!(config.calibration, CalibrationPolicy::Auto));
        assert!(matches!(config.validation, ValidationPolicy::Strict));
        assert!(config.progress.is_none());
        assert!(config.cancel.is_none());
        assert!(config.evolution_enabled);
    }

    #[test]
    fn evolution_config_default() {
        let config = EvolutionConfig::default();
        assert_eq!(config.crossover_rate, 0.8);
        assert_eq!(config.mutation_rate, 0.1);
    }

    #[test]
    fn receipt_serde_roundtrip() {
        let receipt = CanonicalCompileReceipt {
            source_gguf_digest: "abc123".to_string(),
            detected_architecture: ArchitectureFamily::Llama,
            compiler_identity: "prism-compiler/0.1.0".to_string(),
            calibration_digest: "def456".to_string(),
            candidates_evaluated: 42,
            generations_completed: 10,
            stopping_reason: "stall_limit".to_string(),
            cimage_digest: "fff789".to_string(),
            started_at: 1000,
            finished_at: 2000,
            source_format: String::new(),
            tensor_manifest_digest: String::new(),
            search_input_digest: String::new(),
            format_plan_digest: String::new(),
            kernel_manifest_digest: String::new(),
            genome_digest: String::new(),
            timestamp: String::new(),
            duration_ms: 0,
            evaluation_mode: "synthetic".to_string(),
            legalization_mode: "none".to_string(),
            forensic_log: Vec::new(),
        };
        let json = serde_json::to_string_pretty(&receipt).expect("serialize");
        let restored: CanonicalCompileReceipt = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.source_gguf_digest, "abc123");
        assert_eq!(restored.detected_architecture, ArchitectureFamily::Llama);
        assert_eq!(restored.compiler_identity, "prism-compiler/0.1.0");
        assert_eq!(restored.calibration_digest, "def456");
        assert_eq!(restored.candidates_evaluated, 42);
        assert_eq!(restored.generations_completed, 10);
        assert_eq!(restored.stopping_reason, "stall_limit");
        assert_eq!(restored.cimage_digest, "fff789");
        assert_eq!(restored.started_at, 1000);
        assert_eq!(restored.finished_at, 2000);
    }

    #[test]
    fn tensor_type_variants_exist() {
        use crate::cimage::TensorType;
        assert_eq!(
            serde_json::to_value(TensorType::Bf16).unwrap(),
            serde_json::json!("Bf16")
        );
        assert_eq!(
            serde_json::to_value(TensorType::Int8).unwrap(),
            serde_json::json!("Int8")
        );
        assert_eq!(
            serde_json::to_value(TensorType::Nf8).unwrap(),
            serde_json::json!("Nf8")
        );
        assert_eq!(
            serde_json::to_value(TensorType::TernaryTile640).unwrap(),
            serde_json::json!("TernaryTile640")
        );
    }
}
