//! A12. No-JS rendering.
//!
//! Per spec §12 A12: every canonical route, including
//! `/observatory/life/`, renders meaningfully without
//! JavaScript. The full sequence at the Observatory is
//! present in HTML.
//!
//! Static check: the rendered HTML carries the page's
//! primary content without relying on scripts. We strip
//! the JS, count visible text-bearing elements, and
//! verify the page has a non-trivial body.

use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let mut min_chars: usize = usize::MAX;
    let mut min_route: &str = "";
    let mut max_chars: usize = 0;
    let mut max_route: &str = "";
    let mut below_threshold: Vec<String> = Vec::new();

    for (route, html) in &ctx.html_files {
        // Strip <script> and <style> blocks; we want to
        // know what the page shows when JS is disabled
        // but CSS is on. (CSS-only is the no-JS state
        // per spec; the no-CSS state is a separate
        // concern handled by A11.)
        let stripped = strip_scripts_and_styles(html);
        let text_chars = stripped
            .chars()
            .filter(|c| !c.is_whitespace())
            .count();
        if text_chars < min_chars {
            min_chars = text_chars;
            min_route = route;
        }
        if text_chars > max_chars {
            max_chars = text_chars;
            max_route = route;
        }
        // A meaningful no-JS page should have at least
        // 1500 visible characters. The Observatory
        // route must show all 12 stages without JS.
        if text_chars < 1500 {
            below_threshold.push(format!("{} ({} chars)", route, text_chars));
        }
    }

    if !below_threshold.is_empty() {
        return CheckResult::fail(
            "A12",
            "No-JS rendering",
            format!(
                "min {} chars at {}, max {} chars at {}",
                min_chars, min_route, max_chars, max_route
            ),
            format!("below 1500 chars: {}", below_threshold.join(", ")),
        );
    }

    // The Observatory must show all 12 stages in HTML.
    if let Some(obs) = ctx.html("/observatory/life/") {
        let stage_count = obs.matches("observatory-stage").count();
        if stage_count < 12 {
            return CheckResult::fail(
                "A12",
                "No-JS rendering",
                "Observatory stages",
                format!(
                    "/observatory/life/ has {} `.observatory-stage` elements, expected 12",
                    stage_count
                ),
            );
        }
    }

    CheckResult::pass(
        "A12",
        "No-JS rendering",
        format!(
            "{} pages, min {} chars, Observatory has 12 stages in HTML",
            ctx.html_files.len(),
            min_chars
        ),
    )
}

fn strip_scripts_and_styles(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut cursor = 0;
    let lower = html.to_lowercase();
    while cursor < html.len() {
        // Find the next <script or <style.
        let script_pos = lower[cursor..].find("<script");
        let style_pos = lower[cursor..].find("<style");
        let next = match (script_pos, style_pos) {
            (Some(s), Some(t)) => Some(s.min(t)),
            (Some(s), None) => Some(s),
            (None, Some(t)) => Some(t),
            (None, None) => None,
        };
        match next {
            Some(n) => {
                let abs = cursor + n;
                out.push_str(&html[cursor..abs]);
                // Find the matching </script> or </style>.
                let end_tag = if lower[abs..].starts_with("<script") {
                    "</script>"
                } else {
                    "</style>"
                };
                let end_pos = lower[abs..]
                    .find(end_tag)
                    .map(|p| abs + p + end_tag.len())
                    .unwrap_or(html.len());
                cursor = end_pos;
            }
            None => {
                out.push_str(&html[cursor..]);
                break;
            }
        }
    }
    out
}
