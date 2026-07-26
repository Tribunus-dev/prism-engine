//! A1. Route integrity.
//!
//! Per spec §12 A1: every canonical path serves its
//! canonical content; no path serves a 404; no legacy
//! URL surface; no `_redirects` or `_headers` files in
//! v1. The 404 page is reachable for any non-canonical
//! path.

use std::path::Path;

use crate::checks::CANONICAL_ROUTES;
use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    // The canonical routes must all be present.
    let mut missing: Vec<&str> = Vec::new();
    for r in CANONICAL_ROUTES {
        if ctx.html(r).is_none() {
            missing.push(r);
        }
    }
    if !missing.is_empty() {
        return CheckResult::fail(
            "A1",
            "Route integrity",
            format!("{} canonical route(s) present", CANONICAL_ROUTES.len() - missing.len()),
            format!("missing routes: {}", missing.join(", ")),
        );
    }

    // No `_redirects` or `_headers` files in v1.
    let source_dir = match &ctx.source {
        crate::context::SiteSource::LocalDir(p) => p.clone(),
        crate::context::SiteSource::Url(_) => {
            return CheckResult::skip(
                "A1",
                "Route integrity",
                "URL source not supported",
                "§12 A1",
            );
        }
    };
    let mut forbidden_files: Vec<String> = Vec::new();
    for entry in walkdir::WalkDir::new(&source_dir)
        .max_depth(2)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let name = entry.file_name().to_string_lossy().to_string();
        if name == "_redirects" || name == "_headers" {
            forbidden_files.push(name);
        }
    }
    if !forbidden_files.is_empty() {
        return CheckResult::fail(
            "A1",
            "Route integrity",
            format!("forbidden surface present: {}", forbidden_files.join(", ")),
            "ADR-032 v2: no _redirects or _headers in v1",
        );
    }

    // The 404 page must be present.
    let has_404 = source_dir.join("404.html").exists()
        || source_dir.join("404/index.html").exists();
    if !has_404 {
        let _ = Path::new("dummy");
        return CheckResult::fail(
            "A1",
            "Route integrity",
            "no 404.html or 404/index.html",
            "the 404 page is authored (§7.4, §15.7) and must be served",
        );
    }

    CheckResult::pass(
        "A1",
        "Route integrity",
        format!(
            "{} canonical routes, 404 present, no legacy surface",
            CANONICAL_ROUTES.len()
        ),
    )
}
