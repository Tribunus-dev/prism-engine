use crate::ecs::Component;
use std::path::PathBuf;

// ── Download system component ──────────────────────────────────────────

/// HF model download source information.
#[derive(Debug, Clone)]
pub struct HfDownloadComp {
    pub hub_id: String,
    pub revision: String,
    pub dest_dir: PathBuf,
}
impl Component for HfDownloadComp {}

/// Downloaded model source path (result of a successful download).
#[derive(Debug, Clone)]
pub struct DownloadedSourceComp(pub PathBuf);
impl Component for DownloadedSourceComp {}

// ── ANE archive system component ───────────────────────────────────────

/// ANE archive source-to-destination mapping.
#[derive(Debug, Clone)]
pub struct AneArchiveComp {
    pub src_path: PathBuf,
    pub dst_path: PathBuf,
}
impl Component for AneArchiveComp {}

/// Result of an ANE archive operation.
#[derive(Debug, Clone)]
pub struct AneArchiveResultComp {
    pub paths: Vec<PathBuf>,
}
impl Component for AneArchiveResultComp {}

// ── Draft model system component ───────────────────────────────────────

/// Fused draft weights buffer (output of draft_loader).
#[derive(Debug, Clone)]
pub struct DraftWeightsComp(pub Vec<u8>);
impl Component for DraftWeightsComp {}

// ── TTS compilation system component ───────────────────────────────────

/// TTS weight triplet paths.
#[derive(Debug, Clone)]
pub struct TtsWeightsComp {
    pub weight_path: PathBuf,
    pub scale_path: PathBuf,
    pub bias_path: PathBuf,
}
impl Component for TtsWeightsComp {}

// ── INT4 pack system component ─────────────────────────────────────────

/// Ternary pack result — packed blocks + count.
#[derive(Debug, Clone)]
pub struct TernaryPackResult {
    pub packed_blocks: Vec<u8>,
    pub block_count: u32,
}
impl Component for TernaryPackResult {}

// ── Execution graph system component ───────────────────────────────────

/// Serialised execution graph descriptor bytes.
#[derive(Debug, Clone)]
pub struct ExecutionGraphComp(pub Vec<u8>);
impl Component for ExecutionGraphComp {}

// ── Capability registry system component ───────────────────────────────

/// Placeholder capability key stored per dispatch entity.
#[derive(Debug, Clone)]
pub struct CapabilityKeyComp(pub String);
impl Component for CapabilityKeyComp {}

// ── Portfolio system component ─────────────────────────────────────────

/// Portfolio artifact path stored on a model entity.
#[derive(Debug, Clone)]
pub struct PortfolioArtifactsComp {
    pub artifact_paths: Vec<PathBuf>,
}
impl Component for PortfolioArtifactsComp {}

// ── Tertiary pipeline system component ─────────────────────────────────

/// Sealed cimage binary payload.
#[derive(Debug, Clone)]
pub struct CimageBinaryComp(pub Vec<u8>);
impl Component for CimageBinaryComp {}

// ── Validation matrix system component ─────────────────────────────────

/// Summary of a single validation test result.
#[derive(Debug, Clone)]
pub struct ValidationResultSummary {
    pub test_name: String,
    pub passed: bool,
    pub max_error: f64,
    pub details: String,
}

/// Validation report for a single kernel.
#[derive(Debug, Clone)]
pub struct ValidationReportComp {
    pub kernel_name: String,
    pub results: Vec<ValidationResultSummary>,
    pub overall_pass: bool,
}
impl Component for ValidationReportComp {}
