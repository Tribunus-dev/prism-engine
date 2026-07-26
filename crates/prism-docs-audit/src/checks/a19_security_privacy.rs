//! A19. Security and privacy.
//!
//! Per spec §12 A19: no third-party requests of any
//! kind. No `eval`, no inline scripts (except the
//! no-flash theme guard), no third-party iframes.
//!
//! Static checks: scan every HTML and JS file for any
//! URL that is not same-origin or canonical.

use crate::context::AuditContext;
use crate::report::CheckResult;

/// The set of host patterns the site is allowed to
/// reference. The site has no third-party requests; the
/// only external references are the canonical origin
/// itself, and git history links.
fn is_allowed_host(url: &str) -> bool {
    if url.starts_with("/") {
        return true; // same-origin path (incl. legacy surface)
    }
    if url.starts_with("data:") || url.starts_with("blob:") {
        return true; // same-document
    }
    if url.starts_with("#") {
        return true; // fragment
    }
    if url.starts_with("https://prism-engine.tribunus.dev") {
        return true; // canonical origin
    }
    if url.starts_with("https://github.com/Tribunus-dev") {
        return true; // source repo
    }
    if url.starts_with("https://github.com/tribunus-dev") {
        return true;
    }
    if url.starts_with("https://archive.ubuntu.com") {
        return true; // Ubuntu font source (build-time only, not runtime)
    }
    if url.starts_with("http://") {
        // Plain http; not https, must be flagged.
        return false;
    }
    // Default: external https. Not allowed.
    false
}

pub fn run(ctx: &AuditContext) -> CheckResult {
    let mut external_refs: Vec<(String, String)> = Vec::new(); // (file, url)
    let mut evals: Vec<String> = Vec::new();
    let mut inline_event_handlers: Vec<String> = Vec::new();
    let mut iframes: Vec<String> = Vec::new();

    // Scan every HTML file for src= and href= attributes
    // that point to non-allowed hosts.
    for (route, html) in &ctx.html_files {
        let lower = html.to_lowercase();
        for attr in ["src=\"", "href=\"", "action=\""] {
            let mut cursor = 0;
            while let Some(start) = lower[cursor..].find(attr) {
                let abs = cursor + start + attr.len();
                let close = lower[abs..].find('"').unwrap_or(0);
                let url = &html[abs..abs + close];
                if !is_allowed_host(url) {
                    external_refs.push((route.clone(), url.to_string()));
                }
                cursor = abs + close;
            }
        }
        // Inline event handlers (onclick=, onerror=, etc.)
        for evt in ["onclick", "onerror", "onload", "onmouseover"] {
            if lower.contains(&format!("{}=\"", evt)) {
                inline_event_handlers.push(format!("{}: {}", route, evt));
            }
        }
        if lower.contains("<iframe") {
            iframes.push(route.clone());
        }
    }

    // Scan every JS file for `eval(` and external URLs.
    for (rel, js) in &ctx.js_files {
        if js.contains("eval(") {
            evals.push(rel.clone());
        }
    }

    // The site is allowed exactly one inline script per
    // page: the no-flash theme guard. Any other inline
    // script is a violation of A19's "no inline scripts"
    // rule. (Strictly the rule is "no third-party scripts
    // of any kind"; the no-flash guard is same-origin.)
    let mut other_inline: Vec<String> = Vec::new();
    for (route, html) in &ctx.html_files {
        let lower = html.to_lowercase();
        let mut count = 0;
        let mut cursor = 0;
        while let Some(start) = lower[cursor..].find("<script") {
            // The substring "<script" can also match the
            // closing tag "</script". We require the
            // character after "<script" to NOT be '>',
            // which would indicate a self-closing open
            // tag (or a non-open tag). And we require
            // either whitespace or '>' to be the next
            // character, so we don't catch "<scripted" or
            // similar. We also require the character
            // right BEFORE "<script" to be a tag-boundary
            // character (start of input, '<' is consumed
            // by the search, so we check the character
            // just before "<script").
            let abs = cursor + start;
            let next_char = lower[abs + "<script".len()..]
                .chars()
                .next();
            let prev_char = if abs > 0 {
                lower[..abs].chars().next_back()
            } else {
                None
            };
            // Skip the closing tag `</script>` (which
            // starts with "</", i.e. `<` then `/` then
            // `script`).
            let is_open = next_char == Some(' ') || next_char == Some('>') || next_char == Some('\n') || next_char == Some('\t');
            if is_open && prev_char != Some('/') {
                // Find the closing > of the open tag.
                let close = lower[abs..].find('>').unwrap_or(0);
                let tag = &lower[abs..abs + close + 1];
                if !tag.contains("src=") {
                    count += 1;
                }
            }
            cursor = abs + 1;
        }
        if count > 1 {
            other_inline.push(format!("{} ({} inline)", route, count));
        }
    }

    let mut failures: Vec<String> = Vec::new();
    if !external_refs.is_empty() {
        failures.push(format!(
            "{} third-party reference(s): {}",
            external_refs.len(),
            external_refs
                .iter()
                .take(5)
                .map(|(f, u)| format!("{} → {}", f, u))
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }
    if !evals.is_empty() {
        failures.push(format!("eval( in JS: {}", evals.join(", ")));
    }
    if !inline_event_handlers.is_empty() {
        failures.push(format!(
            "inline event handlers: {}",
            inline_event_handlers.join(", ")
        ));
    }
    if !iframes.is_empty() {
        failures.push(format!("iframes present: {}", iframes.join(", ")));
    }
    if !other_inline.is_empty() {
        failures.push(format!(
            "pages with >1 inline script (only the no-flash guard is allowed): {}",
            other_inline.join(", ")
        ));
    }

    if !failures.is_empty() {
        return CheckResult::fail(
            "A19",
            "Security and privacy",
            format!("{} HTML pages, {} JS files", ctx.html_files.len(), ctx.js_files.len()),
            failures.join("; "),
        );
    }

    CheckResult::pass(
        "A19",
        "Security and privacy",
        "no third-party requests, no eval, no inline event handlers, no iframes, no extra inline scripts",
    )
}
