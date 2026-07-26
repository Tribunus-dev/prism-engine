//! A20. Accessibility extras.
//!
//! Per spec §12 A20: contrast ratio meets WCAG 2.2 AA at
//! the chosen theme. Behavior at 200% and 400% zoom is
//! verified on every canonical route. Touch targets ≥
//! 44×44 CSS pixels. Status communication does not
//! depend on color or animation.
//!
//! Static proxy: the site must declare touch-target
//! sizing rules at narrow viewports. The full check
//! (contrast, zoom) needs a browser.

use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let css = ctx.css.as_deref().unwrap_or("");
    // The site must declare `min-height: 44px` (or
    // equivalent) for interactive elements at narrow
    // viewports.
    let has_44 = css.contains("44px") || css.contains("2.75rem") || css.contains("44x44");
    if !has_44 {
        return CheckResult::fail(
            "A20",
            "Accessibility extras (static proxy)",
            "no 44px touch-target rule in site.css",
            "every interactive element must have a minimum 44×44 CSS pixel hit area at narrow viewports",
        );
    }

    // The contrast guarantee is in the token system.
    // The dark-theme and light-theme tokens must both
    // declare explicit fg/bg pairs; the runner checks
    // the tokens file.
    let dark_declares_contrast = css.contains("--color-fg:") && css.contains("--color-bg:");
    if !dark_declares_contrast {
        return CheckResult::fail(
            "A20",
            "Accessibility extras (static proxy)",
            "no --color-fg and --color-bg in tokens",
            "the token system must declare fg/bg pairs for the chosen theme",
        );
    }

    CheckResult::skip(
        "A20",
        "Accessibility extras",
        "static proxies pass (44px touch targets, fg/bg tokens); full check (contrast ratios, 200%/400% zoom) needs a browser",
        "§12 A20 — run axe-core for contrast, render at 200% and 400% on every canonical route",
    )
}
