use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde_json::json;
use std::path::{Path, PathBuf};

pub struct ResolvePathHandler;

impl McpHandler for ResolvePathHandler {
    fn name(&self) -> &'static str {
        "resolve_path"
    }
    fn description(&self) -> &'static str {
        "Resolve a file or directory name within a workspace and return nearest matches and package manifests."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({"type":"object","properties":{"path":{"type":"string"},"query":{"type":"string"},"kind":{"type":"string","enum":["any","file","directory"]},"limit":{"type":"integer","minimum":1,"maximum":100}},"additionalProperties":false})
    }
    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        _state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let root = PathBuf::from(
            request
                .args
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("."),
        );
        let query = request
            .args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let kind = request
            .args
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("any");
        let limit = request
            .args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(25)
            .clamp(1, 100) as usize;
        let requested = root.join(query);
        let direct_match = requested.exists().then(|| display_path(&root, &requested));
        let needle = Path::new(query)
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or(query)
            .to_lowercase();
        let mut matches = Vec::new();
        let mut manifests = Vec::new();
        walk(
            &root,
            &root,
            query.to_lowercase(),
            &needle,
            kind,
            &mut matches,
            &mut manifests,
        )?;
        matches.sort_by_key(|item: &Match| (item.score, item.path.len()));
        matches.truncate(limit);
        manifests.truncate(50);
        Ok(ToolResult::text(serde_json::to_string_pretty(
            &json!({"status":"ok","query":query,"directMatch":direct_match,"matches":matches,"packageManifests":manifests,"searchedRoot":root}),
        )?))
    }
}

#[derive(serde::Serialize)]
struct Match {
    path: String,
    score: u8,
}

fn walk(
    root: &Path,
    dir: &Path,
    query: String,
    needle: &str,
    kind: &str,
    matches: &mut Vec<Match>,
    manifests: &mut Vec<String>,
) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if entry.file_type()?.is_dir() {
            if [".git", "target", "node_modules"].contains(&name.as_str()) {
                continue;
            }
            walk(root, &path, query.clone(), needle, kind, matches, manifests)?;
            continue;
        }
        if (name == "Cargo.toml" || name == "Package.swift") && manifests.len() < 50 {
            manifests.push(display_path(root, &path));
        }
        let acceptable = kind == "any" || kind == "file";
        if !acceptable {
            continue;
        }
        let lower = name.to_lowercase();
        let relative = display_path(root, &path);
        let lower_relative = relative.to_lowercase();
        let score = if lower == needle {
            Some(0)
        } else if lower.contains(needle) {
            Some(1)
        } else if lower_relative.contains(&query) {
            Some(2)
        } else {
            None
        };
        if let Some(score) = score {
            matches.push(Match {
                path: relative,
                score,
            });
        }
    }
    Ok(())
}

fn display_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
        .if_empty_then(".")
}

trait EmptyPath {
    fn if_empty_then(self, fallback: &str) -> String;
}
impl EmptyPath for String {
    fn if_empty_then(self, fallback: &str) -> String {
        if self.is_empty() {
            fallback.into()
        } else {
            self
        }
    }
}
