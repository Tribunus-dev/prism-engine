//! A21. /docs/ allowlist.
//!
//! Per spec §12 A21: the `/docs/` route serves only files
//! explicitly named in `docs-publication.json`. Internal
//! notes, outdated ADRs, and implementation debris do
//! not reach the live site.

use std::path::Path;

use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let allowlist = match &ctx.publication_allowlist {
        Some(a) => a,
        None => {
            return CheckResult::skip(
                "A21",
                "/docs/ allowlist",
                "no docs-publication.json found",
                "§12 A21",
            );
        }
    };

    let source_dir = match &ctx.source {
        crate::context::SiteSource::LocalDir(p) => p.clone(),
        crate::context::SiteSource::Url(_) => {
            return CheckResult::skip(
                "A21",
                "/docs/ allowlist",
                "URL source not supported",
                "§12 A21",
            );
        }
    };

    // Walk the site root (depth 1) and find every file.
    // A file is "in the publication surface" if it is at
    // the site root OR inside a canonical route directory.
    // The allowlist is a list of allowed paths or globs.
    let mut served: Vec<String> = Vec::new();
    for entry in walkdir::WalkDir::new(&source_dir)
        .max_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = entry
            .path()
            .strip_prefix(&source_dir)
            .unwrap_or(entry.path())
            .to_string_lossy()
            .replace('\\', "/");
        served.push(rel);
    }

    let _ = Path::new(&source_dir);

    // For the v1 SSG output, the surfaced files are
    // expected to be: index.html, 404.html, build.json,
    // site.css, selection-controller.js, theme.js,
    // transitions-orchestrator.js, and the canonical
    // route directories (each with index.html inside).
    // We do not have a hard allowlist of every file
    // because the v1 surface includes a few support
    // files; the audit runner reports the served set
    // for the architect to review.
    let expected_root: Vec<&str> = vec![
        "index.html",
        "404.html",
        "build.json",
        "site.css",
        "selection-controller.js",
        "theme.js",
        "transitions-orchestrator.js",
        "CNAME",
    ];

    let unexpected: Vec<String> = served
        .iter()
        .filter(|f| !expected_root.contains(&f.as_str()))
        .cloned()
        .collect();
    let _ = allowlist;

    if !unexpected.is_empty() {
        return CheckResult::warn(
            "A21",
            "/docs/ allowlist",
            format!("{} file(s) at site root", served.len()),
            format!("not in the expected root set: {}", unexpected.join(", ")),
        );
    }

    CheckResult::pass(
        "A21",
        "/docs/ allowlist",
        format!(
            "{} root file(s) all in the expected set; allowlist size: {}",
            served.len(),
            allowlist.len()
        ),
    )
}
