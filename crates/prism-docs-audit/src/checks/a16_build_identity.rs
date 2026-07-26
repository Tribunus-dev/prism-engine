//! A16. Build identity and source commit.
//!
//! Per spec §12 A16: every rendered page carries a build
//! identity and a source commit, visible in the page
//! source and in the meta. The site knows what version of
//! itself it is.

use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let mut pages_with_build_id: usize = 0;
    let total = ctx
        .html_files
        .iter()
        .filter(|(r, _)| !r.contains("404"))
        .count();
    for (route, html) in &ctx.html_files {
        if route.contains("404") {
            continue;
        }
        if html.contains("name=\"build-id\"") {
            pages_with_build_id += 1;
        }
    }

    // The build.json must be present and parseable.
    let build = match &ctx.build_json {
        Some(v) => v,
        None => {
            return CheckResult::fail(
                "A16",
                "Build identity and source commit",
                format!("build_id meta on {}/{} pages", pages_with_build_id, total),
                "no /build.json at the site root",
            );
        }
    };
    let has_build_id = build.get("build_id").and_then(|v| v.as_str()).is_some();
    let has_commit = build.get("commit").and_then(|v| v.as_str()).is_some();
    if !has_build_id || !has_commit {
        return CheckResult::fail(
            "A16",
            "Build identity and source commit",
            "build.json present",
            format!(
                "missing: {}{}",
                if !has_build_id { "build_id " } else { "" },
                if !has_commit { "commit" } else { "" }
            ),
        );
    }

    if pages_with_build_id < total {
        return CheckResult::fail(
            "A16",
            "Build identity and source commit",
            format!("{}/{} pages have build-id meta", pages_with_build_id, total),
            "every rendered page must carry meta name=\"build-id\"",
        );
    }

    CheckResult::pass(
        "A16",
        "Build identity and source commit",
        format!(
            "build_id on all {} pages, build.json has build_id and commit",
            total
        ),
    )
}
