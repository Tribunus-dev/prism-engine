//! A10. Keyboard parity.
//!
//! Per spec §12 A10: every interactive element is
//! reachable by tab. The focus order matches the visual
//! order. The focus ring is visible. The skip link is
//! functional.
//!
//! Browser-required. Static proxies: the page declares a
//! skip link, all interactive elements have an
//! appropriate role or tag, and the focus styles exist.

use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let css = ctx.css.as_deref().unwrap_or("");
    let has_focus_visible = css.contains(":focus-visible");
    let has_skip_link = ctx
        .html_files
        .iter()
        .any(|(_, h)| h.contains("skip-link") || h.contains("Skip to"));
    let has_button_role = ctx
        .html_files
        .iter()
        .any(|(_, h)| h.contains("type=\"button\"") || h.contains("role=\"button\""));

    let mut missing: Vec<&str> = Vec::new();
    if !has_focus_visible {
        missing.push("no :focus-visible rule in site.css");
    }
    if !has_skip_link {
        missing.push("no skip link in any rendered page");
    }
    if !has_button_role {
        missing.push("no <button> elements (toggles, controls)");
    }
    if !missing.is_empty() {
        return CheckResult::fail(
            "A10",
            "Keyboard parity (static proxy)",
            format!("{} static checks passed", 3 - missing.len()),
            missing.join("; "),
        );
    }
    CheckResult::skip(
        "A10",
        "Keyboard parity",
        "static proxies pass (:focus-visible, skip link, <button>); full tab-order traversal needs a browser",
        "§12 A10 — Playwright traversal at canonical viewport, verify focus order matches visual order",
    )
}
