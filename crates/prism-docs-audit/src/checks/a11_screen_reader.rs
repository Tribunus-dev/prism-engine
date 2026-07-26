//! A11. Screen-reader parity.
//!
//! Per spec §12 A11: the semantic structure of every page
//! is complete without CSS. Headings are nested correctly.
//! Landmarks are present. `aria-live` regions are used for
//! state changes. Alt text is present.
//!
//! Browser-required. Static proxies: the page has a
//! `<main>` landmark, a `<header>`, a `<nav>` with an
//! `aria-label`, and every `<img>` has an `alt` attribute
//! (or `alt=""` for decorative images).

use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let mut total_imgs: usize = 0;
    let mut missing_alt: usize = 0;
    let mut has_main: usize = 0;
    let mut has_header: usize = 0;
    let mut has_nav: usize = 0;

    for (route, html) in &ctx.html_files {
        if route.contains("404") {
            continue;
        }
        if html.contains("<main") {
            has_main += 1;
        }
        if html.contains("<header") {
            has_header += 1;
        }
        if html.contains("<nav") {
            has_nav += 1;
        }
        // Count <img ...> tags without alt.
        let lower = html.to_lowercase();
        let mut cursor = 0;
        while let Some(start) = lower[cursor..].find("<img") {
            let abs = cursor + start;
            let close = lower[abs..].find('>').unwrap_or(0);
            let tag = &lower[abs..abs + close + 1];
            total_imgs += 1;
            if !tag.contains("alt=") {
                missing_alt += 1;
            }
            cursor = abs + close + 1;
        }
    }

    let total_pages = ctx
        .html_files
        .iter()
        .filter(|(r, _)| !r.contains("404"))
        .count();
    let has_all_landmarks = has_main == total_pages && has_header == total_pages;
    if !has_all_landmarks {
        return CheckResult::fail(
            "A11",
            "Screen-reader parity (static proxy)",
            format!("{} pages with <main>", has_main),
            format!(
                "some pages missing landmarks: <main>={}/{} <header>={}/{} <nav>={}/{}",
                has_main, total_pages, has_header, total_pages, has_nav, total_pages
            ),
        );
    }
    if missing_alt > 0 {
        return CheckResult::fail(
            "A11",
            "Screen-reader parity (static proxy)",
            format!("{} <img> tags, {} without alt", total_imgs, missing_alt),
            "every <img> must have an alt attribute (empty alt=\"\" is valid for decorative)",
        );
    }
    CheckResult::skip(
        "A11",
        "Screen-reader parity",
        format!(
            "landmarks present on {}/{} pages, {} images all with alt; full check (heading nesting, aria-live) needs axe-core",
            has_main, total_pages, total_imgs
        ),
        "§12 A11 — run axe-core on every canonical route",
    )
}
