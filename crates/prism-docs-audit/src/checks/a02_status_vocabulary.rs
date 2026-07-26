//! A2. Status-vocabulary purity (linter).
//!
//! Per spec §12 A2: status-bearing language is emitted
//! only through `Claim`, `StatusTable`, and `Release`
//! components. The structured validator handles the
//! structured surfaces. The linter is a text-level sweep
//! for the five forbidden status words (per §3's
//! closed vocabulary).
//!
//! Per spec: "flagged lines are returned for human review
//! (H1), not auto-rejected." A linter hit is a `Warn`,
//! not a `Fail`. The architect reviews and decides.

use crate::checks::FORBIDDEN_STATUS_WORDS;
use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let mut hits: Vec<(String, String, String)> = Vec::new(); // (route, word, snippet)

    for (route, html) in &ctx.html_files {
        let lower = html.to_lowercase();
        for &word in FORBIDDEN_STATUS_WORDS {
            // Find every occurrence.
            let mut start = 0;
            while let Some(pos) = lower[start..].find(word) {
                let abs = start + pos;
                // Take a 60-char window around the match.
                let lo = abs.saturating_sub(30);
                let hi = (abs + word.len() + 30).min(lower.len());
                let snippet = html[lo..hi].replace('\n', " ");
                hits.push((route.clone(), word.to_string(), snippet));
                start = abs + word.len();
                if hits.len() > 20 {
                    break;
                }
            }
        }
        if hits.len() > 20 {
            break;
        }
    }

    if hits.is_empty() {
        return CheckResult::pass(
            "A2",
            "Status-vocabulary purity (linter)",
            format!(
                "no forbidden status words in any rendered page ({} pages scanned)",
                ctx.html_files.len()
            ),
        );
    }

    let mut detail = String::new();
    for (route, word, snippet) in hits.iter().take(8) {
        detail.push_str(&format!(
            "\n  • {}: '{}' in \"{}…\"",
            route,
            word,
            &snippet[..snippet.len().min(80)]
        ));
    }
    if hits.len() > 8 {
        detail.push_str(&format!("\n  • …and {} more", hits.len() - 8));
    }
    CheckResult::warn(
        "A2",
        "Status-vocabulary purity (linter)",
        format!("{} potential match(es)", hits.len()),
        format!("H1 review queue:{}", detail),
    )
    .with_severity(crate::report::Severity::Advisory)
}
