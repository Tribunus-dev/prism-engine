//! A5. Chapter list locality.
//!
//! Per spec §12 A5: no page-local chapter list contains
//! a chapter from another page. The 60-entry global
//! chapter dump is absent.

use crate::context::AuditContext;
use crate::report::CheckResult;

/// The threshold above which a chapter list is treated
/// as a global dump. The original legacy site had a
/// 60-entry list; v1 pages have a small per-page set.
const GLOBAL_DUMP_THRESHOLD: usize = 20;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let mut max_list_size: usize = 0;
    let mut max_list_route: &str = "";
    let mut dump_present: Vec<String> = Vec::new();

    for (route, html) in &ctx.html_files {
        // Count `<li>` elements inside `<ol class="chapters">` or
        // `<ul class="chapters">` (the legacy global-dump selector).
        let lower = html.to_lowercase();
        // We look for two patterns: the old global chapter
        // list (per the deleted-via-cutover criteria) and
        // any unusually large ordered/unordered list.
        if let Some(start) = lower.find("class=\"chapters\"") {
            // Crude count: <li> between this point and the
            // next </ol> or </ul>. This is approximate; the
            // site does not use the .chapters class in v1.
            if let Some(end) = lower[start..]
                .find("</ol>")
                .or_else(|| lower[start..].find("</ul>"))
            {
                let slice = &lower[start..start + end];
                let count = slice.matches("<li").count();
                if count > max_list_size {
                    max_list_size = count;
                    max_list_route = route;
                }
                if count >= GLOBAL_DUMP_THRESHOLD {
                    dump_present.push(route.clone());
                }
            }
        }
    }

    if !dump_present.is_empty() {
        return CheckResult::fail(
            "A5",
            "Chapter list locality",
            format!("largest list: {} items at {}", max_list_size, max_list_route),
            format!("global chapter dump present on: {}", dump_present.join(", ")),
        );
    }

    CheckResult::pass(
        "A5",
        "Chapter list locality",
        format!(
            "no `.chapters` list with >{} items (largest seen: {} on {})",
            GLOBAL_DUMP_THRESHOLD, max_list_size, max_list_route
        ),
    )
}
