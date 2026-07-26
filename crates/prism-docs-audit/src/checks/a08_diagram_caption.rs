//! A8. Diagram caption and description.
//!
//! Per spec §12 A8: every `<figure>` has a `<figcaption>`
//! and an `aria-describedby` reference to a textual
//! equivalent.

use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let mut total_figures: usize = 0;
    let mut missing_figcaption: Vec<String> = Vec::new();
    let mut missing_describedby: Vec<String> = Vec::new();

    for (route, html) in &ctx.html_files {
        // Find every <figure> ... </figure> block.
        let mut cursor = 0;
        while let Some(start) = html[cursor..].find("<figure") {
            let abs = cursor + start;
            let after = &html[abs..];
            let close = after.find("</figure>").unwrap_or(0);
            let block = &after[..close + "</figure>".len()];
            total_figures += 1;
            if !block.contains("<figcaption") {
                missing_figcaption.push(route.clone());
            }
            if !block.contains("aria-describedby") {
                missing_describedby.push(route.clone());
            }
            cursor = abs + close + 1;
        }
    }

    if total_figures == 0 {
        return CheckResult::pass(
            "A8",
            "Diagram caption and description",
            "no <figure> elements in the rendered site",
        );
    }

    if !missing_figcaption.is_empty() || !missing_describedby.is_empty() {
        let mut detail = String::new();
        if !missing_figcaption.is_empty() {
            detail.push_str(&format!(
                "{} figure(s) without <figcaption>",
                missing_figcaption.len()
            ));
        }
        if !missing_describedby.is_empty() {
            if !detail.is_empty() {
                detail.push_str("; ");
            }
            detail.push_str(&format!(
                "{} figure(s) without aria-describedby",
                missing_describedby.len()
            ));
        }
        return CheckResult::fail(
            "A8",
            "Diagram caption and description",
            format!("{} figure(s) total", total_figures),
            detail,
        );
    }

    CheckResult::pass(
        "A8",
        "Diagram caption and description",
        format!(
            "{} figure(s), each with <figcaption> and aria-describedby",
            total_figures
        ),
    )
}
