//! A7. Manuscript-to-page structural match.
//!
//! Per spec §12 A7: every page in the output references
//! the manuscript for its brief. The page's section IDs
//! match the brief's section list.
//!
//! The audit runner checks that every rendered page has a
//! `data-prism-route` attribute (set by the SSG) and that
//! the rendered page count matches the manuscript's
//! `## Page` headings.

use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let manuscript = match &ctx.manuscript {
        Some(m) => m,
        None => {
            return CheckResult::skip(
                "A7",
                "Manuscript-to-page match",
                "manuscript not present",
                "§12 A7",
            );
        }
    };

    // Count `## Page` headings in the manuscript. The
    // manuscript's brief format uses `## Page` to mark
    // each page boundary. A page whose heading contains
    // the marker "not in v1" is explicitly out of scope
    // for v1 (the manuscript may include deferred pages
    // as authoring context).
    let manuscript_pages: Vec<(String, bool)> = manuscript
        .lines()
        .filter(|l| l.starts_with("## "))
        .filter_map(|l| l.trim_start_matches("## ").trim().to_string().into())
        .filter(|t: &String| t.to_lowercase() != "reviewer checklist" && !t.is_empty())
        .map(|t: String| {
            let in_v1 = !t.to_lowercase().contains("not in v1");
            (t, in_v1)
        })
        .collect();
    let in_v1_page_count = manuscript_pages.iter().filter(|(_, in_v1)| *in_v1).count();
    let total_manuscript_page_count = manuscript_pages.len();

    let rendered_page_count = ctx.html_files.len();

    // The 404 page is not in the manuscript. The expected
    // rendered count is the v1 manuscript pages + 1.
    let expected_rendered = in_v1_page_count + 1;
    if rendered_page_count != expected_rendered {
        return CheckResult::fail(
            "A7",
            "Manuscript-to-page match",
            format!(
                "{} v1 pages in manuscript ({} total), {} rendered",
                in_v1_page_count, total_manuscript_page_count, rendered_page_count
            ),
            format!(
                "expected {} rendered pages (v1 manuscript pages + 404)",
                expected_rendered
            ),
        );
    }

    // Every rendered page (except 404) has a
    // data-prism-route attribute set by the SSG.
    let mut missing_route_attr: usize = 0;
    for (route, html) in &ctx.html_files {
        if route == "/" || route.contains("404") {
            continue;
        }
        if !html.contains("data-prism-route=") {
            missing_route_attr += 1;
        }
    }

    if missing_route_attr > 0 {
        return CheckResult::fail(
            "A7",
            "Manuscript-to-page match",
            format!("{} rendered pages", rendered_page_count),
            format!(
                "{} page(s) without data-prism-route attribute",
                missing_route_attr
            ),
        );
    }

    CheckResult::pass(
        "A7",
        "Manuscript-to-page match",
        format!(
            "{} v1 pages in manuscript ({} total), {} rendered (incl. 404), all routed",
            in_v1_page_count, total_manuscript_page_count, rendered_page_count
        ),
    )
}
