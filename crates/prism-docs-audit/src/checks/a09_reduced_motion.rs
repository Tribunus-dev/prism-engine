//! A9. Reduced motion compliance.
//!
//! Per spec §12 A9: with `prefers-reduced-motion: reduce`,
//! the prism effect is absent, ambient motion is absent,
//! and selection state is preserved through layout.
//!
//! This is a browser-required check. The static CSS check
//! is a partial proxy: the site must declare a reduced-
//! motion media query that zeroes the motion tokens.

use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let css = match &ctx.css {
        Some(c) => c,
        None => {
            return CheckResult::skip(
                "A9",
                "Reduced motion compliance",
                "no site.css in source",
                "§12 A9",
            );
        }
    };
    // Static proxy: the CSS must include a
    // `prefers-reduced-motion: reduce` block that zeroes
    // the motion tokens. The full check (visual rendering
    // + state preservation) needs a browser.
    let has_reduced_motion = css.contains("prefers-reduced-motion: reduce");
    if !has_reduced_motion {
        return CheckResult::fail(
            "A9",
            "Reduced motion compliance",
            "no `prefers-reduced-motion: reduce` block in site.css",
            "the static CSS must declare a reduced-motion block per §9",
        );
    }
    CheckResult::skip(
        "A9",
        "Reduced motion compliance",
        "static CSS has prefers-reduced-motion block; full visual check needs a browser",
        "§12 A9 — render with (prefers-reduced-motion: reduce) forced, verify prism effect and ambient motion are absent, selection state preserved",
    )
}
