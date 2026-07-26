//! A17. Status not by color alone.
//!
//! Per spec §12 A17: status is communicated in text,
//! shape, and semantics in addition to color. Verified
//! by rendering the page in forced-colors mode and by
//! checking that every status badge has a textual label
//! and (where applicable) an icon or shape variant.
//!
//! Static proxy: every `.state` badge has a text label
//! (the state name is in the text) and a CSS-drawn shape
//! (`::before` content). The browser-required check is
//! a forced-colors render.

use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let css = ctx.css.as_deref().unwrap_or("");
    // The state-badge component must declare a `::before`
    // rule that paints a shape (so the badge is readable
    // in forced-colors mode where color alone is gone).
    let has_shape_rule = css.contains(".state::before")
        || css.contains(".state-validated::before")
        || css.contains(".state-implemented::before")
        || css.contains(".state-qualifying::before");
    // The site must declare a `forced-colors` block.
    let has_forced_colors = css.contains("forced-colors: active");

    if !has_shape_rule {
        return CheckResult::fail(
            "A17",
            "Status not by color alone (static proxy)",
            "no shape rule in site.css",
            "every status badge must have a CSS-drawn shape (::before content) so it is readable in forced-colors",
        );
    }
    if !has_forced_colors {
        return CheckResult::fail(
            "A17",
            "Status not by color alone (static proxy)",
            "no `forced-colors: active` block in site.css",
            "the site must declare a forced-colors response per §9",
        );
    }
    CheckResult::skip(
        "A17",
        "Status not by color alone",
        "static proxies pass (shape rule + forced-colors block); full visual check needs a forced-colors render",
        "§12 A17 — render with forced-colors: active, verify every status badge is readable and the state is communicated in text + shape",
    )
}
