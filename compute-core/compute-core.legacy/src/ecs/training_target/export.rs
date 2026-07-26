//! JSON export helpers — deterministic serialization and digest computation.
//!
//! Provides functions to write a [`TrainingTargetSpec`] to a JSON file with
//! canonical field ordering, and to compute deterministic digests.

use std::path::Path;

use super::feedback::TrainingFeedbackReport;
use super::spec::TrainingTargetSpec;

/// Serialise a `TrainingTargetSpec` to a JSON file at `path`.
///
/// The output is canonical (sorted keys) so the same spec always produces
/// identical bytes.
pub fn export_spec(spec: &TrainingTargetSpec, path: &Path) -> Result<(), String> {
    let bytes = serde_json::to_vec(spec).map_err(|e| format!("serialization error: {e}"))?;
    std::fs::write(path, &bytes).map_err(|e| format!("write error: {e}"))?;
    Ok(())
}

/// Compute a BLAKE3 hex digest from raw JSON bytes.
pub fn spec_digest_from_bytes(bytes: &[u8]) -> String {
    let hash = blake3::hash(bytes);
    hash.to_hex().to_string()
}

/// Serialise a `TrainingFeedbackReport` to a JSON file at `path`.
pub fn export_feedback(report: &TrainingFeedbackReport, path: &Path) -> Result<(), String> {
    let bytes = serde_json::to_vec(report).map_err(|e| format!("serialization error: {e}"))?;
    std::fs::write(path, &bytes).map_err(|e| format!("write error: {e}"))?;
    Ok(())
}

/// Serialise a spec to a JSON byte vector and compute its digest.
pub fn spec_bytes(spec: &TrainingTargetSpec) -> Vec<u8> {
    serde_json::to_vec(spec).expect("TrainingTargetSpec serialization")
}
