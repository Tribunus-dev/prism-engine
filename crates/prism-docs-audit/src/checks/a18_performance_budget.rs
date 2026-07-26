//! A18. Performance budget.
//!
//! Per spec §12 A18: HTML per route ≤ 60 KB gzipped.
//! Critical CSS per route ≤ 18 KB gzipped. JavaScript per
//! route ≤ 80 KB gzipped, with the Observatory permitted
//! up to 120 KB. LCP ≤ 2.5 s, CLS ≤ 0.1, INP ≤ 200 ms.
//!
//! Static checks: file-size budgets. LCP/CLS/INP need a
//! browser.
//!
//! The "Critical CSS" budget is the *inlined* critical
//! CSS in each page's `<head>`. The full `site.css` bundle
//! is loaded via `<link rel="stylesheet">` and is
//! cached across pages; it does not count against the
//! 18KB per-route budget. The inlined block is what
//! paints the first viewport; the spec is honest about
//! that distinction.

use std::io::Read;

use flate2::read::GzEncoder;
use flate2::Compression;

use crate::context::AuditContext;
use crate::report::{CheckResult, Severity};

const HTML_BUDGET_BYTES: u64 = 60 * 1024;
const CSS_CRITICAL_BUDGET_BYTES: u64 = 18 * 1024;
const JS_BUDGET_BYTES: u64 = 80 * 1024;
const JS_BUDGET_OBSERVATORY_BYTES: u64 = 120 * 1024;

pub fn run(ctx: &AuditContext) -> CheckResult {
    // HTML: total size per page, gzipped.
    let mut html_overs: Vec<String> = Vec::new();
    let mut max_html: u64 = 0;
    let mut max_html_route: &str = "";
    for (route, html) in &ctx.html_files {
        let gz = gz_size(html.as_bytes());
        if gz > max_html {
            max_html = gz;
            max_html_route = route;
        }
        if gz > HTML_BUDGET_BYTES {
            html_overs.push(format!("{} ({} B)", route, gz));
        }
    }

    // Critical CSS: the inlined `<style data-prism-critical>`
    // block in each page's <head>. The full bundle (in
    // ctx.css) is loaded via <link> and is cached, so it
    // is not counted against the 18KB per-route budget.
    let mut critical_overs: Vec<String> = Vec::new();
    let mut max_critical: u64 = 0;
    let mut max_critical_route: &str = "";
    for (route, html) in &ctx.html_files {
        if let Some(critical) = extract_critical_css(html) {
            let gz = gz_size(critical.as_bytes());
            if gz > max_critical {
                max_critical = gz;
                max_critical_route = route;
            }
            if gz > CSS_CRITICAL_BUDGET_BYTES {
                critical_overs.push(format!("{} ({} B)", route, gz));
            }
        } else {
            // No inlined critical CSS. The page is
            // relying on the full bundle, which violates
            // the per-route critical-CSS budget per §12 A18.
            critical_overs.push(format!("{} (no inlined critical CSS)", route));
        }
    }

    // The full bundle is reported for context. The
    // spec's per-route budget is the inlined critical
    // CSS, not the bundle.
    let bundle_gz = ctx.css.as_deref().map(|c| gz_size(c.as_bytes())).unwrap_or(0);

    // JS: per route, sum the gzipped sizes of the
    // referenced script files.
    let mut js_overs: Vec<String> = Vec::new();
    let mut max_js_route: &str = "";
    let mut max_js: u64 = 0;
    let mut js_observatory_over: Option<String> = None;
    for (route, html) in &ctx.html_files {
        let mut total: u64 = 0;
        let lower = html.to_lowercase();
        let mut cursor = 0;
        while let Some(start) = lower[cursor..].find("src=\"/") {
            let abs = cursor + start + "src=\"".len();
            let close = lower[abs..].find('"').unwrap_or(0);
            let path = &lower[abs..abs + close];
            if let Some(js) = ctx.js(path) {
                total += gz_size(js.as_bytes());
            }
            cursor = abs + close;
        }
        let limit = if route.contains("observatory") {
            JS_BUDGET_OBSERVATORY_BYTES
        } else {
            JS_BUDGET_BYTES
        };
        if total > limit {
            let msg = format!("{} ({} B > {} B)", route, total, limit);
            if route.contains("observatory") {
                js_observatory_over = Some(msg);
            } else {
                js_overs.push(msg);
            }
        }
        if total > max_js {
            max_js = total;
            max_js_route = route;
        }
    }

    let mut failures: Vec<String> = Vec::new();
    if !html_overs.is_empty() {
        failures.push(format!(
            "{} HTML over 60KB: {}",
            html_overs.len(),
            html_overs.join(", ")
        ));
    }
    if !critical_overs.is_empty() {
        failures.push(format!(
            "{} route(s) over 18KB critical CSS or missing inline: {}",
            critical_overs.len(),
            critical_overs.join(", ")
        ));
    }
    if !js_overs.is_empty() {
        failures.push(format!(
            "{} non-Observatory route(s) over JS budget: {}",
            js_overs.len(),
            js_overs.join(", ")
        ));
    }
    if let Some(obs) = js_observatory_over {
        failures.push(format!("Observatory over 120KB: {}", obs));
    }

    if !failures.is_empty() {
        return CheckResult::fail(
            "A18",
            "Performance budget (file sizes)",
            format!(
                "max HTML {} B at {}, max critical CSS {} B at {}, max JS {} B at {}, bundle {} B",
                max_html, max_html_route, max_critical, max_critical_route, max_js, max_js_route, bundle_gz
            ),
            failures.join("; "),
        );
    }

    let note = format!(
        "max HTML {} B at {}, max critical CSS {} B at {}, max JS {} B at {}, bundle {} B; LCP/CLS/INP need a browser",
        max_html, max_html_route, max_critical, max_critical_route, max_js, max_js_route, bundle_gz
    );
    let _ = Severity::Blocking;
    CheckResult::skip(
        "A18",
        "Performance budget",
        note,
        "§12 A18 — run at 1366×768, cold cache, 5 runs, median reported; LCP ≤ 2.5s, CLS ≤ 0.1, INP ≤ 200ms",
    )
}

/// Extract the inlined critical CSS from a page's `<head>`.
/// The renderer emits `<style data-prism-critical="..."> ... </style>`;
/// we return the contents of the first such block.
fn extract_critical_css(html: &str) -> Option<String> {
    let needle = "<style data-prism-critical";
    let start = html.find(needle)?;
    // Find the closing > of the opening tag.
    let open_end = html[start..].find('>')? + start + 1;
    let close = html[open_end..].find("</style>")? + open_end;
    Some(html[open_end..close].to_string())
}

fn gz_size(bytes: &[u8]) -> u64 {
    let mut encoder = GzEncoder::new(bytes, Compression::default());
    let mut out = Vec::new();
    encoder.read_to_end(&mut out).unwrap_or(0);
    out.len() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_critical_finds_inline_block() {
        let html = r#"<head><style data-prism-critical="/">body{color:red}</style></head>"#;
        let c = extract_critical_css(html).expect("found");
        assert!(c.contains("body{color:red}"));
    }

    #[test]
    fn extract_critical_returns_none_when_absent() {
        let html = "<head><link rel=\"stylesheet\" href=\"/site.css\"></head>";
        assert!(extract_critical_css(html).is_none());
    }

    #[test]
    fn gz_size_smaller_than_raw_for_repetitive_content() {
        // 1000 identical bytes compress to a tiny gz stream.
        let raw = "a".repeat(1000);
        let gz = gz_size(raw.as_bytes());
        assert!(gz < 100);
    }
}
