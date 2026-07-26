//! A15. Canonical URLs, sitemap, robots.
//!
//! Per spec §12 A15: every page sets its canonical URL in
//! the head. A sitemap is generated and submitted. A
//! robots policy is set. OG and Twitter card metadata are
//! present.

use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let mut pages_with_canonical: usize = 0;
    let mut pages_with_og: usize = 0;
    let mut pages_with_twitter: usize = 0;
    let total = ctx
        .html_files
        .iter()
        .filter(|(r, _)| !r.contains("404"))
        .count();

    for (route, html) in &ctx.html_files {
        if route.contains("404") {
            continue;
        }
        if html.contains("rel=\"canonical\"") {
            pages_with_canonical += 1;
        }
        if html.contains("og:") || html.contains("property=\"og:") {
            pages_with_og += 1;
        }
        if html.contains("twitter:") || html.contains("name=\"twitter:") {
            pages_with_twitter += 1;
        }
    }

    let mut missing: Vec<&str> = Vec::new();
    if pages_with_canonical < total {
        missing.push("canonical");
    }
    if pages_with_og < total {
        missing.push("og:");
    }
    if pages_with_twitter < total {
        missing.push("twitter:");
    }
    if !missing.is_empty() {
        return CheckResult::fail(
            "A15",
            "Canonical URLs, sitemap, robots",
            format!(
                "{} pages; canonical {}/{}, og {}/{}, twitter {}/{}",
                total, pages_with_canonical, total, pages_with_og, total, pages_with_twitter, total
            ),
            format!("missing: {}", missing.join(", ")),
        );
    }

    // Sitemap and robots are at the site root. The
    // current SSG does not emit them. Per ADR-032 v2
    // §15 the sitemap is a follow-on; we surface a
    // warning if absent.
    let has_sitemap = ctx.html_files.iter().any(|(r, _)| r == "/sitemap.xml");
    let has_robots = ctx.html_files.iter().any(|(r, _)| r == "/robots.txt");

    if !has_sitemap || !has_robots {
        let mut detail = String::new();
        if !has_sitemap {
            detail.push_str("no /sitemap.xml; ");
        }
        if !has_robots {
            detail.push_str("no /robots.txt; ");
        }
        return CheckResult::warn(
            "A15",
            "Canonical URLs, sitemap, robots",
            format!(
                "canonical {}/{}, og {}/{}, twitter {}/{}",
                pages_with_canonical, total, pages_with_og, total, pages_with_twitter, total
            ),
            detail.trim_end_matches("; "),
        );
    }

    CheckResult::pass(
        "A15",
        "Canonical URLs, sitemap, robots",
        format!(
            "canonical, og, twitter on all {} pages; sitemap and robots present",
            total
        ),
    )
}
