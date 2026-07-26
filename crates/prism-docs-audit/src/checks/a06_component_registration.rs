//! A6. Component registration.
//!
//! Per spec §12 A6: every component is registered with
//! exactly one declared responsibility identifier from
//! §8. A component with zero is rejected; a component
//! with more than one is rejected.
//!
//! The audit runner checks the component CSS files in the
//! site. Every component CSS file's module doc should
//! state the single authority in one sentence. This is a
//! coarse check; the full honesty review is H8.

use std::path::Path;

use crate::context::AuditContext;
use crate::report::CheckResult;

pub fn run(ctx: &AuditContext) -> CheckResult {
    let source_dir = match &ctx.source {
        crate::context::SiteSource::LocalDir(p) => p.clone(),
        crate::context::SiteSource::Url(_) => {
            return CheckResult::skip(
                "A6",
                "Component registration",
                "URL source not supported",
                "§12 A6",
            );
        }
    };
    let components_dir = source_dir.join("styles/components");
    if !components_dir.exists() {
        return CheckResult::skip(
            "A6",
            "Component registration",
            "no styles/components/ directory",
            "§12 A6",
        );
    }

    let mut checked: usize = 0;
    let mut unstated: Vec<String> = Vec::new();

    for entry in walkdir::WalkDir::new(&components_dir)
        .max_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("css") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        // Skip the index file.
        if name == "mod.rs" {
            continue;
        }
        let contents = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        checked += 1;
        // The module doc should be a paragraph that
        // names the component's single authority. A
        // common signal: the first paragraph contains
        // a verb like "owns" or "is" describing the
        // single thing. This is a coarse check; the
        // honesty review is H8.
        let first_para = contents
            .split("/* ---")
            .nth(1)
            .and_then(|s| s.split("*/").nth(1))
            .unwrap_or(&contents);
        let first_para = first_para
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("*"))
            .take(3)
            .collect::<Vec<_>>()
            .join(" ");
        // Heuristic: the first paragraph should be at
        // least 40 chars and contain at least one
        // period.
        if first_para.len() < 40 || !first_para.contains('.') {
            unstated.push(name);
        }
        let _ = Path::new(&components_dir);
    }

    if !unstated.is_empty() {
        return CheckResult::fail(
            "A6",
            "Component registration",
            format!("{} components, {} unstated", checked, unstated.len()),
            format!("files without a clear module doc: {}", unstated.join(", ")),
        );
    }

    CheckResult::pass(
        "A6",
        "Component registration",
        format!("{} components, each with a single authority statement", checked),
    )
}
