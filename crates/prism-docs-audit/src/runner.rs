//! The runner. Orchestrates the 22 axiom checks.

use crate::checks;
use crate::context::{AuditContext, SiteSource};
use crate::report::{CheckResult, Report, Severity, Verdict};
use crate::AuditError;

/// Build the audit context from a local directory and run
/// every check. Returns a `Report` that the CLI prints.
pub fn run_audit(source: &SiteSource) -> Result<Report, AuditError> {
    let ctx = match source {
        SiteSource::LocalDir(p) => AuditContext::from_local_dir(p)?,
        SiteSource::Url(_) => {
            return Err(AuditError::InvalidSource(
                "URL sources are not yet supported; pass a local directory".to_string(),
            ));
        }
    };

    let mut report = Report::new(source.describe());
    let registry = build_registry(&ctx);

    for (id, check) in registry.iter() {
        let result = check(&ctx);
        // Annotate the spec_ref if the check didn't set it.
        let mut result = result;
        if result.spec_ref.is_empty() {
            result.spec_ref = spec_ref_for(id);
        }
        report.push(result);
    }

    Ok(report)
}

type CheckFn = fn(&AuditContext) -> CheckResult;

/// The registry of all 22 axiom checks. The order is the
/// A-list order from the spec.
fn build_registry(ctx: &AuditContext) -> Vec<(&'static str, CheckFn)> {
    let v: Vec<(&'static str, CheckFn)> = vec![
        ("A1",  checks::a01_route_integrity::run as CheckFn),
        ("A2",  checks::a02_status_vocabulary::run as CheckFn),
        ("A3",  checks::a03_data_layer_validation::run as CheckFn),
        ("A4",  checks::a04_evidence_boundary::run as CheckFn),
        ("A5",  checks::a05_chapter_locality::run as CheckFn),
        ("A6",  checks::a06_component_registration::run as CheckFn),
        ("A7",  checks::a07_manuscript_match::run as CheckFn),
        ("A8",  checks::a08_diagram_caption::run as CheckFn),
        ("A9",  checks::a09_reduced_motion::run as CheckFn),
        ("A10", checks::a10_keyboard_parity::run as CheckFn),
        ("A11", checks::a11_screen_reader::run as CheckFn),
        ("A12", checks::a12_no_js_rendering::run as CheckFn),
        ("A13", checks::a13_schema_integrity::run as CheckFn),
        ("A14", checks::a14_evidence_applicability::run as CheckFn),
        ("A15", checks::a15_canonical_urls::run as CheckFn),
        ("A16", checks::a16_build_identity::run as CheckFn),
        ("A17", checks::a17_status_not_color::run as CheckFn),
        ("A18", checks::a18_performance_budget::run as CheckFn),
        ("A19", checks::a19_security_privacy::run as CheckFn),
        ("A20", checks::a20_accessibility_extras::run as CheckFn),
        ("A21", checks::a21_allowlist::run as CheckFn),
        ("A22", checks::a22_deployment_smoke::run as CheckFn),
    ];
    // Provide a default for any check that was not
    // overridden in `ctx` (none currently).
    let _ = ctx;
    v
}

fn spec_ref_for(id: &str) -> String {
    format!("§12.{}", id.trim_start_matches('A'))
}

/// Helper for checks that want to mark a sub-result inside
/// a parent verdict. The aggregate verdict follows the
/// worst child.
pub fn aggregate(results: &[CheckResult]) -> Verdict {
    if results.iter().any(|r| r.verdict == Verdict::Fail) {
        Verdict::Fail
    } else if results.iter().any(|r| r.verdict == Verdict::Warn) {
        Verdict::Warn
    } else if results.iter().all(|r| r.verdict == Verdict::Skip) {
        Verdict::Skip
    } else {
        Verdict::Pass
    }
}

/// Helper to flip a result to advisory severity (does not
/// block CI).
pub fn advisory(result: CheckResult) -> CheckResult {
    result.with_severity(Severity::Advisory)
}
