//! A18. Performance budget.
//!
//! Per spec §12 A18: HTML per route ≤ 60 KB gzipped.
//! Critical CSS per route ≤ 18 KB gzipped. JavaScript per
//! route ≤ 80 KB gzipped, with the Observatory permitted
//! up to 120 KB. LCP ≤ 2.5 s, CLS ≤ 0.1, INP ≤ 200 ms.
//!
//! Static checks: file-size budgets. LCP/CLS/INP need a
//! browser.

use std::io::Read;

use flate2::read::GzEncoder;
use flate2::Compression;

use crate::context::AuditContext;
use crate::report::{CheckResult, Severity};

const HTML_BUDGET_BYTES: u64 = 60 * 1024;
const CSS_BUDGET_BYTES: u64 = 18 * 1024;
const JS_BUDGET_BYTES: u64 = 80 * 1024;
const JS_BUDGET_OBSERVATORY_BYTES: u64 = 120 * 1024;

pub fn run(ctx: &AuditContext) -> CheckResult {
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

    // CSS: one bundle shared across pages.
    let css_gz = ctx.css.as_deref().map(|c| gz_size(c.as_bytes())).unwrap_or(0);
    let css_over = css_gz > CSS_BUDGET_BYTES;

    // JS: each script is counted per route. We count
    // the script srcs referenced from each HTML file
    // and sum the gzipped sizes of the JS files
    // referenced.
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
    if css_over {
        failures.push(format!(
            "CSS bundle {} B > 18KB",
            css_gz
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
                "max HTML {} B at {}, max JS {} B at {}, CSS {} B",
                max_html, max_html_route, max_js, max_js_route, css_gz
            ),
            failures.join("; "),
        );
    }

    let mut note = format!(
        "max HTML {} B at {}, max JS {} B at {}, CSS {} B; LCP/CLS/INP need a browser",
        max_html, max_html_route, max_js, max_js_route, css_gz
    );
    let _ = Severity::Blocking;
    let _ = &mut note;
    CheckResult::skip(
        "A18",
        "Performance budget",
        note,
        "§12 A18 — run at 1366×768, cold cache, 5 runs, median reported; LCP ≤ 2.5s, CLS ≤ 0.1, INP ≤ 200ms",
    )
}

fn gz_size(bytes: &[u8]) -> u64 {
    let mut encoder = GzEncoder::new(bytes, Compression::default());
    let mut out = Vec::new();
    encoder.read_to_end(&mut out).unwrap_or(0);
    out.len() as u64
}
