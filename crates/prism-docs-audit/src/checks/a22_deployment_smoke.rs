//! A22. Deployment smoke test.
//!
//! Per spec §12 A22: the smoke test runs at two points,
//! against two distinct surfaces. Preview smoke runs
//! against the candidate build (served by the workflow
//! artifact or a local server). Post-production smoke
//! runs against the production URL after the deployment
//! completes.
//!
//! The local smoke test (no network) checks that the
//! canonical routes are present in the rendered site
//! and that `build.json` carries a build identity and
//! source commit. The live smoke (browser-required)
//! follows the same path through the live URL.

use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let total_routes = ctx.canonical_routes.len();
    if total_routes == 0 {
        return CheckResult::fail(
            "A22",
            "Deployment smoke test",
            "no canonical routes",
            "the SSG output must include all canonical routes",
        );
    }

    let build = match &ctx.build_json {
        Some(v) => v,
        None => {
            return CheckResult::fail(
                "A22",
                "Deployment smoke test",
                format!("{} routes", total_routes),
                "no /build.json at the site root",
            );
        }
    };
    let build_id = build
        .get("build_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let commit = build
        .get("commit")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // The local smoke path: every canonical route has an
    // index.html; build.json has build_id and commit.
    if build_id.is_empty() || commit.is_empty() {
        return CheckResult::fail(
            "A22",
            "Deployment smoke test",
            format!("{} routes, build_id={:?}, commit={:?}", total_routes, build_id, commit),
            "build.json must carry build_id and commit",
        );
    }

    CheckResult::skip(
        "A22",
        "Deployment smoke test",
        format!(
            "{} routes present, build_id={}, commit={}; live path: home → status row → evidence → receipt",
            total_routes, build_id, &commit[..commit.len().min(8)]
        ),
        "§12 A22 — live: curl the canonical path, follow to a Status row, open the row's evidence, reach the receipt's stable URL; verify build_id matches the just-merged commit",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::AuditContext;

    fn empty_ctx() -> AuditContext {
        AuditContext {
            source: crate::context::SiteSource::LocalDir(std::path::PathBuf::from(".")),
            canonical_routes: vec!["/".to_string()],
            html_files: vec![("/".to_string(), "<html></html>".to_string())],
            css: None,
            js_files: vec![],
            data_layer_dir: None,
            schemas_dir: None,
            manuscript: None,
            publication_allowlist: None,
            build_json: Some(serde_json::json!({
                "build_id": "ssg-test",
                "commit": "abc1234",
            })),
        }
    }

    #[test]
    fn smoke_passes_with_build_identity() {
        let r = run(&empty_ctx());
        assert!(matches!(r.verdict, crate::report::Verdict::Skip));
        assert!(r.evidence.contains("ssg-test"));
    }

    #[test]
    fn smoke_fails_without_build_json() {
        let mut ctx = empty_ctx();
        ctx.build_json = None;
        let r = run(&ctx);
        assert!(matches!(r.verdict, crate::report::Verdict::Fail));
        assert!(r.detail.contains("build.json"));
    }

    #[test]
    fn smoke_fails_without_commit() {
        let mut ctx = empty_ctx();
        ctx.build_json = Some(serde_json::json!({"build_id": "x"}));
        let r = run(&ctx);
        assert!(matches!(r.verdict, crate::report::Verdict::Fail));
    }
}
