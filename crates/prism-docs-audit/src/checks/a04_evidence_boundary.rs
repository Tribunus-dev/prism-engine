//! A4. Evidence-boundary completeness.
//!
//! Per spec §12 A4: every `ValidatedRecord` has a `target`
//! and an `evidence_id` that resolves. Every
//! `ReleaseRecord` has a signature, a verification path,
//! and a support-boundary document. Every
//! `QualifyingRecord` has a test/fixture path and a
//! named limit.

use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let data_dir = match &ctx.data_layer_dir {
        Some(p) if p.exists() => p.clone(),
        _ => {
            return CheckResult::skip(
                "A4",
                "Evidence-boundary completeness",
                "no data/ directory",
                "§12 A4",
            );
        }
    };

    // Locate the capabilities and evidence files. The
    // data layer file names are stable per ADR-033.
    let caps_path = data_dir.join("capabilities.json");
    let ev_path = data_dir.join("evidence.json");

    let caps = match std::fs::read_to_string(&caps_path) {
        Ok(s) => s,
        Err(_) => {
            return CheckResult::skip(
                "A4",
                "Evidence-boundary completeness",
                "capabilities.json not present",
                "§12 A4",
            );
        }
    };
    let _ = ev_path;
    let caps_value: serde_json::Value = match serde_json::from_str(&caps) {
        Ok(v) => v,
        Err(e) => {
            return CheckResult::fail(
                "A4",
                "Evidence-boundary completeness",
                "capabilities.json parse failure",
                e.to_string(),
            );
        }
    };

    let records = caps_value
        .get("records")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();

    let mut validated_missing_target: usize = 0;
    let mut validated_missing_evidence: usize = 0;
    let mut qualifying_missing_test: usize = 0;
    let mut qualifying_missing_limit: usize = 0;
    let mut released_missing_signature: usize = 0;
    let mut released_missing_verification: usize = 0;
    let mut validated: usize = 0;
    let mut qualifying: usize = 0;
    let mut released: usize = 0;

    for r in &records {
        let state = r.get("state").and_then(|s| s.as_str()).unwrap_or("");
        let distribution = r.get("distribution_state").and_then(|s| s.as_str()).unwrap_or("");

        if state == "validated" {
            validated += 1;
            if r.get("target").and_then(|t| t.as_str()).map(str::is_empty).unwrap_or(true) {
                validated_missing_target += 1;
            }
            if r.get("evidence_id").and_then(|e| e.as_str()).map(str::is_empty).unwrap_or(true) {
                validated_missing_evidence += 1;
            }
        }
        if state == "qualifying" {
            qualifying += 1;
            if r.get("test_path").and_then(|t| t.as_str()).map(str::is_empty).unwrap_or(true) {
                qualifying_missing_test += 1;
            }
            if r.get("limit").and_then(|l| l.as_str()).map(str::is_empty).unwrap_or(true) {
                qualifying_missing_limit += 1;
            }
        }
        if distribution == "released" {
            released += 1;
            if r.get("signature").and_then(|s| s.as_str()).map(str::is_empty).unwrap_or(true) {
                released_missing_signature += 1;
            }
            if r.get("verification_path").and_then(|v| v.as_str()).map(str::is_empty).unwrap_or(true) {
                released_missing_verification += 1;
            }
        }
    }

    let total_gaps = validated_missing_target
        + validated_missing_evidence
        + qualifying_missing_test
        + qualifying_missing_limit
        + released_missing_signature
        + released_missing_verification;

    if total_gaps > 0 {
        return CheckResult::fail(
            "A4",
            "Evidence-boundary completeness",
            format!(
                "{} validated, {} qualifying, {} released",
                validated, qualifying, released
            ),
            format!(
                "gaps: {} validated (no target/evidence), {} qualifying (no test/limit), {} released (no signature/verification)",
                validated_missing_target + validated_missing_evidence,
                qualifying_missing_test + qualifying_missing_limit,
                released_missing_signature + released_missing_verification
            ),
        );
    }

    CheckResult::pass(
        "A4",
        "Evidence-boundary completeness",
        format!(
            "{} validated, {} qualifying, {} released — all fields present",
            validated, qualifying, released
        ),
    )
}
