//! CLI subcommands for the Tribunus Compute Kernel.
//!
//! # Training target subcommands
//!
//! These are available as the `training-target` command group:
//! - `export <spec> --policy <path> --out <path>`  — generate TrainingTargetSpec JSON
//! - `check <spec>`                                  — validate spec consistency
//! - `feedback --target <spec> --evidence <ledger>   — generate feedback report
//!           --checkpoint-digest <hash> --out <path>`

use std::path::Path;

use crate::ecs::training_target::export::{export_feedback, export_spec};
use crate::ecs::training_target::feedback::{EvidenceEntry, GateThresholds, TargetWithGates, TrainingFeedbackBuilder};
use crate::ecs::training_target::resolve::{TrainingTargetResolveOptions, TrainingTargetResolver};
use crate::ecs::training_target::spec::TrainingTargetSpec;

/// Export a TrainingTargetSpec from a JSON policy.
pub fn training_target_export(policy_path: &Path, output_path: &Path) {
    match do_training_target_export(policy_path, output_path) {
        Ok(digest) => println!("[cli] training-target export OK — digest: {}", digest),
        Err(e) => eprintln!("[cli] training-target export FAILED: {}", e),
    }
}

fn do_training_target_export(policy_path: &Path, output_path: &Path) -> Result<String, String> {
    let policy_bytes = std::fs::read(policy_path)
        .map_err(|e| format!("read policy: {}", e))?;
    let policy: serde_json::Value = serde_json::from_slice(&policy_bytes)
        .map_err(|e| format!("parse policy: {}", e))?;

    let resolver = TrainingTargetResolver;
    let options = TrainingTargetResolveOptions::default();
    let specs = resolver
        .resolve(&policy, &options)
        .map_err(|e| format!("resolve: {}", e))?;
    let spec = specs.into_iter().next().ok_or_else(|| "no targets resolved from policy".to_string())?;

    // Validate consistency before exporting.
    spec.check_consistency().map_err(|e| format!("consistency check: {}", e))?;

    let digest = spec.digest();
    export_spec(&spec, output_path)
        .map_err(|e| format!("export: {}", e))?;

    Ok(digest)
}

/// Check a TrainingTargetSpec JSON for internal consistency.
pub fn training_target_check(spec_path: &Path) {
    match do_training_target_check(spec_path) {
        Ok(_) => println!("[cli] training-target check OK — spec is consistent"),
        Err(e) => eprintln!("[cli] training-target check FAILED: {}", e),
    }
}

fn do_training_target_check(spec_path: &Path) -> Result<(), String> {
    let bytes = std::fs::read(spec_path).map_err(|e| format!("read spec: {}", e))?;
    let spec: TrainingTargetSpec =
        serde_json::from_slice(&bytes).map_err(|e| format!("parse spec: {}", e))?;
    spec.check_consistency()
}

/// Generate a feedback report by comparing a spec against evidence.
pub fn training_target_feedback(
    target_spec_path: &Path,
    evidence_path: &Path,
    checkpoint_digest: &str,
    output_path: &Path,
) {
    match do_training_target_feedback(target_spec_path, evidence_path, checkpoint_digest, output_path)
    {
        Ok(status) => println!(
            "[cli] training-target feedback OK — status: {:?}",
            status
        ),
        Err(e) => eprintln!("[cli] training-target feedback FAILED: {}", e),
    }
}

fn do_training_target_feedback(
    target_spec_path: &Path,
    evidence_path: &Path,
    checkpoint_digest: &str,
    output_path: &Path,
) -> Result<String, String> {
    let spec_bytes =
        std::fs::read(target_spec_path).map_err(|e| format!("read spec: {}", e))?;
    let spec: TrainingTargetSpec =
        serde_json::from_slice(&spec_bytes).map_err(|e| format!("parse spec: {}", e))?;

    let evidence_bytes =
        std::fs::read(evidence_path).map_err(|e| format!("read evidence: {}", e))?;
    let evidence: Vec<EvidenceEntry> = serde_json::from_slice(&evidence_bytes)
        .map_err(|e| format!("parse evidence: {}", e))?;

    let spec_digest = spec.digest();
    let evidence_ledger_digest =
        crate::training_target::export::spec_digest_from_bytes(&evidence_bytes);

    // Convert spec weight targets into TargetWithGates.
    let targets: Vec<TargetWithGates> = spec
        .weight_targets
        .iter()
        .map(|wt| TargetWithGates {
            target_id: wt.target_id.clone(),
            tensor_key_match: wt.tensor_key_match.clone(),
            tensor_class: wt.tensor_class.clone(),
            gates: GateThresholds {
                max_weight_nrmse: wt.gates.max_weight_nrmse,
                max_zero_collapse_ratio: wt.gates.max_zero_collapse_ratio,
                max_operator_nrmse: wt.gates.max_operator_nrmse,
                min_operator_cosine: wt.gates.min_operator_cosine,
                max_operator_abs_error: wt.gates.max_operator_abs_error,
                min_byte_savings_ratio: wt.gates.min_byte_savings_ratio,
            },
        })
        .collect();

    // Build evidence map: tensor_key → entries.
    let mut evidence_map: std::collections::HashMap<String, Vec<EvidenceEntry>> =
        std::collections::HashMap::new();
    for entry in &evidence {
        evidence_map
            .entry(entry.tensor_key.clone())
            .or_default()
            .push(entry.clone());
    }

    let report = TrainingFeedbackBuilder::build(
        &targets,
        &evidence_map,
        &spec_digest,
        checkpoint_digest,
        &evidence_ledger_digest,
    );

    let status = format!("{:?}", report.status);
    export_feedback(&report, output_path)
        .map_err(|e| format!("export feedback: {}", e))?;

    Ok(status)
}
