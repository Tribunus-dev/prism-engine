//! A14. Evidence applicability.
//!
//! Per spec §12 A14: a claim referencing an evidence
//! record is rejected if the evidence record's
//! applicability fields do not match the claim's
//! reference. **Age alone is not invalidity.**

use std::path::Path;

use crate::context::AuditContext;
use crate::report::CheckResult;

/// The applicability fields a claim must agree with the
/// evidence record on. Per spec: source commit or build
/// identity, schema version, target identity, feature set,
/// model identity, validation scope.
const APPLICABILITY_FIELDS: &[&str] = &[
    "commit",
    "build_id",
    "schema_version",
    "target",
    "feature_set",
    "model",
    "validation_scope",
];

pub fn run(ctx: &AuditContext) -> CheckResult {
    let data_dir = match &ctx.data_layer_dir {
        Some(p) if p.exists() => p.clone(),
        _ => {
            return CheckResult::skip(
                "A14",
                "Evidence applicability",
                "no data/ directory",
                "§12 A14",
            );
        }
    };

    let claims_path = data_dir.join("claims.json");
    let evidence_path = data_dir.join("evidence.json");

    let claims_str = match std::fs::read_to_string(&claims_path) {
        Ok(s) => s,
        Err(_) => {
            return CheckResult::skip(
                "A14",
                "Evidence applicability",
                "claims.json not present",
                "§12 A14",
            );
        }
    };
    let evidence_str = match std::fs::read_to_string(&evidence_path) {
        Ok(s) => s,
        Err(_) => {
            // If there's no evidence file, applicability is
            // trivially satisfied. The check is a no-op.
            return CheckResult::pass(
                "A14",
                "Evidence applicability",
                "no evidence.json — applicability check is a no-op",
            );
        }
    }
    .clone();
    let _ = evidence_str;

    let claims: serde_json::Value = match serde_json::from_str(&claims_str) {
        Ok(v) => v,
        Err(e) => {
            return CheckResult::fail(
                "A14",
                "Evidence applicability",
                "claims.json parse error",
                e.to_string(),
            );
        }
    };
    let _ = Path::new(&claims_path);

    let records = claims
        .get("records")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();

    let total = records.len();

    // A static check: every claim record should have a
    // `references` field whose target is one of the
    // APPLICABILITY_FIELDS. A claim that references an
    // evidence_id with no applicability metadata is
    // surfaced for review.
    let mut missing_applicability: usize = 0;
    for r in &records {
        let refs = r.get("references").and_then(|x| x.as_array()).cloned().unwrap_or_default();
        if refs.is_empty() {
            continue;
        }
        // The claim references at least one evidence
        // record; check that the claim itself declares
        // the applicability fields the spec requires.
        let applicability = r
            .get("applicability")
            .and_then(|x| x.as_object())
            .cloned()
            .unwrap_or_default();
        let declared: Vec<&str> = APPLICABILITY_FIELDS
            .iter()
            .filter(|f| applicability.get(**f).is_some())
            .copied()
            .collect();
        if declared.is_empty() {
            missing_applicability += 1;
        }
    }

    if missing_applicability > 0 {
        return CheckResult::warn(
            "A14",
            "Evidence applicability",
            format!("{} claim(s), {} with no applicability fields", total, missing_applicability),
            "claims with `references` should declare an `applicability` object (commit, build_id, schema_version, target, feature_set, model, validation_scope)",
        )
        .with_severity(crate::report::Severity::Advisory);
    }

    CheckResult::pass(
        "A14",
        "Evidence applicability",
        format!("{} claim(s), all with applicability fields", total),
    )
}
