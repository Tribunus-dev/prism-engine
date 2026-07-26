//! The audit report.
//!
//! Each check produces a `CheckResult`. The runner
//! aggregates the results into a `Report`. The CLI prints
//! the report as a 22-row table and (optionally) writes
//! the same data as JSON for machine consumption.

use std::fmt;

use serde::{Deserialize, Serialize};

/// The verdict of a single axiom check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Verdict {
    /// The axiom holds. Evidence included.
    Pass,
    /// The axiom does not hold. Failures block CI.
    Fail,
    /// The check could not run automatically (e.g. needs a
    /// browser, or needs the architect's eye). The runner
    /// reserves a row and prints the manual step.
    Skip,
    /// The check ran and produced a non-blocking note.
    /// For A2 (status-vocabulary linter) the spec says
    /// "flagged lines are returned for human review (H1),
    /// not auto-rejected" — that is a `Warn`.
    Warn,
}

impl Verdict {
    pub fn symbol(self) -> &'static str {
        match self {
            Verdict::Pass => "✓",
            Verdict::Fail => "✗",
            Verdict::Skip => "○",
            Verdict::Warn => "!",
        }
    }
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Skip => "SKIP",
            Verdict::Warn => "WARN",
        })
    }
}

/// The severity of a check. Most are `Blocking` — failures
/// stop the build. Some are `Advisory` — failures are
/// noted but do not stop CI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Severity {
    /// A failure here is a build error. CI exits non-zero.
    Blocking,
    /// A failure here is recorded but does not stop CI.
    /// The architect reviews and decides.
    Advisory,
}

/// The result of a single axiom check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// The axiom identifier (e.g. "A1").
    pub id: String,
    /// The axiom name (e.g. "Route integrity").
    pub name: String,
    /// The verdict.
    pub verdict: Verdict,
    /// The severity.
    pub severity: Severity,
    /// A short evidence line — what the check observed.
    pub evidence: String,
    /// If the verdict is `Fail`, the reason.
    /// Empty for `Pass` and `Skip`.
    pub detail: String,
    /// The spec section this axiom comes from.
    pub spec_ref: String,
}

impl CheckResult {
    pub fn pass(id: &str, name: &str, evidence: impl Into<String>) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            verdict: Verdict::Pass,
            severity: Severity::Blocking,
            evidence: evidence.into(),
            detail: String::new(),
            spec_ref: String::new(),
        }
    }

    pub fn fail(id: &str, name: &str, evidence: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            verdict: Verdict::Fail,
            severity: Severity::Blocking,
            evidence: evidence.into(),
            detail: detail.into(),
            spec_ref: String::new(),
        }
    }

    pub fn skip(id: &str, name: &str, evidence: impl Into<String>, spec_ref: impl Into<String>) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            verdict: Verdict::Skip,
            severity: Severity::Advisory,
            evidence: evidence.into(),
            detail: String::new(),
            spec_ref: spec_ref.into(),
        }
    }

    pub fn warn(id: &str, name: &str, evidence: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            verdict: Verdict::Warn,
            severity: Severity::Advisory,
            evidence: evidence.into(),
            detail: detail.into(),
            spec_ref: String::new(),
        }
    }

    pub fn with_spec_ref(mut self, spec_ref: impl Into<String>) -> Self {
        self.spec_ref = spec_ref.into();
        self
    }

    pub fn with_severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }
}

/// The aggregated report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// The site source that was audited.
    pub source: String,
    /// The time the report was generated (ISO 8601).
    pub generated_at: String,
    /// The total number of axioms.
    pub total: usize,
    /// The number of axioms that passed.
    pub passed: usize,
    /// The number of axioms that failed (blocking).
    pub failed: usize,
    /// The number of axioms that were skipped (need
    /// manual review or a browser).
    pub skipped: usize,
    /// The number of axioms that produced warnings.
    pub warned: usize,
    /// The per-axiom results.
    pub results: Vec<CheckResult>,
}

impl Report {
    pub fn new(source: String) -> Self {
        Self {
            source,
            generated_at: chrono_like_now(),
            total: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
            warned: 0,
            results: Vec::new(),
        }
    }

    pub fn push(&mut self, result: CheckResult) {
        self.total += 1;
        match result.verdict {
            Verdict::Pass => self.passed += 1,
            Verdict::Fail => self.failed += 1,
            Verdict::Skip => self.skipped += 1,
            Verdict::Warn => self.warned += 1,
        }
        self.results.push(result);
    }

    /// The exit code for the binary: 0 if all blocking
    /// checks pass, 1 otherwise.
    pub fn exit_code(&self) -> i32 {
        if self.results.iter().any(|r| {
            r.verdict == Verdict::Fail && r.severity == Severity::Blocking
        }) {
            1
        } else {
            0
        }
    }

    /// Render the report as a markdown table.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# Prism Observatory v1 — A-list audit\n\n");
        out.push_str(&format!("- **Source:** `{}`\n", self.source));
        out.push_str(&format!("- **Generated:** {}\n", self.generated_at));
        out.push_str(&format!("- **Total:** {}\n", self.total));
        out.push_str(&format!(
            "- **Pass:** {}  **Fail:** {}  **Skip:** {}  **Warn:** {}\n\n",
            self.passed, self.failed, self.skipped, self.warned
        ));
        out.push_str("| # | Status | Axiom | Evidence | Detail |\n");
        out.push_str("|---|--------|-------|----------|--------|\n");
        for (i, r) in self.results.iter().enumerate() {
            let detail = if r.detail.is_empty() {
                String::new()
            } else {
                r.detail.replace('\n', " ").replace('|', "\\|")
            };
            let evidence = r.evidence.replace('|', "\\|");
            out.push_str(&format!(
                "| {} | {} {} | **{}** {} | {} | {} |\n",
                i + 1,
                r.verdict.symbol(),
                r.verdict,
                r.id,
                r.name,
                evidence,
                detail,
            ));
        }
        out
    }
}

fn chrono_like_now() -> String {
    // A tiny ISO-8601 now() without depending on chrono.
    // We use std::time::SystemTime; the format is
    // YYYY-MM-DDTHH:MM:SSZ in UTC.
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Convert to date components using a small algorithm.
    // This is a deliberately minimal dependency-free
    // formatter. It is correct for any time after 1970.
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let days = secs / 86400;
    // Gregorian date from days since 1970-01-01.
    let (y, mo, d) = days_to_ymd(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, mo, d, h, m, s
    )
}

fn days_to_ymd(days_since_epoch: u64) -> (i32, u32, u32) {
    // Algorithm from Howard Hinnant's date library.
    // Shifts the epoch from 1970-01-01 to 0000-03-01.
    let z = days_since_epoch as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y } as i32;
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_aggregates_verdicts() {
        let mut r = Report::new("test".to_string());
        r.push(CheckResult::pass("A1", "Route", "ok"));
        r.push(CheckResult::fail("A2", "Vocab", "found", "details"));
        r.push(CheckResult::skip("A3", "Motion", "needs browser", "§A3"));
        r.push(CheckResult::warn("A4", "Evidence", "warned", "details"));
        assert_eq!(r.total, 4);
        assert_eq!(r.passed, 1);
        assert_eq!(r.failed, 1);
        assert_eq!(r.skipped, 1);
        assert_eq!(r.warned, 1);
        assert_eq!(r.exit_code(), 1);
    }

    #[test]
    fn report_exit_code_zero_when_no_blocking_failures() {
        let mut r = Report::new("test".to_string());
        r.push(CheckResult::skip("A1", "Motion", "needs browser", "§A1"));
        r.push(CheckResult::warn("A2", "Evidence", "warned", "details"));
        assert_eq!(r.exit_code(), 0);
    }

    #[test]
    fn markdown_table_renders_all_rows() {
        let mut r = Report::new("test".to_string());
        r.push(CheckResult::pass("A1", "Route", "ok"));
        r.push(CheckResult::fail("A2", "Vocab", "found", "details"));
        let md = r.to_markdown();
        assert!(md.contains("A1"));
        assert!(md.contains("A2"));
        assert!(md.contains("Route"));
        assert!(md.contains("Vocab"));
        assert!(md.contains("Pass"));
    }

    #[test]
    fn days_to_ymd_handles_epoch() {
        // 1970-01-01 is day 0.
        let (y, m, d) = days_to_ymd(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn days_to_ymd_handles_known_date() {
        // 2026-07-25 is a known reference point.
        // Compute days from 1970-01-01 to 2026-07-25.
        // 56 years, including 14 leap years (72, 76, 80, 84, 88, 92, 96, 2000, 04, 08, 12, 16, 20, 24).
        // 56 * 365 = 20440 days; + 14 = 20454 days to 2026-01-01.
        // + 31+28+31+30+31+30+24 = 205 days into 2026 (Jan=31, Feb=28, Mar=31, Apr=30, May=31, Jun=30, +24 days of July).
        // Total: 20454 + 205 = 20659. (Note: 2024 is a leap year but its Feb 29 is before July, so +1 already.)
        // 2026 is not a leap year, so the math is right.
        let (y, m, d) = days_to_ymd(20659);
        assert_eq!((y, m, d), (2026, 7, 25));
    }
}
