// prism_mcp_build — MCP tool handlers for build & test operations
//
// Handlers:
//   plan_build              — Plan a build for a component
//   build_component         — Build a workspace component via cargo build -p
//   check_component         — Run cargo check on a component
//   test_scope              — Find and run tests for changed code
//   compare_builds          — Compare two build receipts (text diff)
//   changed_build_surface   — List files changed between git revisions

use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use prism_mcp_core::subprocess::run_with_timeout;
use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde_json::json;
use tracing::info;

// ── Helpers ──────────────────────────────────────────────────────────────

fn json_input_schema(props: serde_json::Value, required: &[&str]) -> serde_json::Value {
    json!({
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": false
    })
}

fn extract_string(args: &serde_json::Value, key: &str) -> Result<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("missing or invalid string field '{}'", key))
}

fn extract_optional_string(args: &serde_json::Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn extract_string_or_default(args: &serde_json::Value, key: &str, default: &str) -> String {
    args.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or(default)
        .to_string()
}

fn extract_optional_u64(args: &serde_json::Value, key: &str) -> Option<u64> {
    args.get(key).and_then(|v| v.as_u64())
}

/// Produce a line-by-line text diff between two strings (like `diff` output).
fn line_diff(a: &str, b: &str) -> String {
    let lines_a: Vec<&str> = a.lines().collect();
    let lines_b: Vec<&str> = b.lines().collect();
    let mut out = String::new();
    let max = lines_a.len().max(lines_b.len());
    for i in 0..max {
        match (lines_a.get(i), lines_b.get(i)) {
            (Some(la), Some(lb)) if la == lb => {
                out.push_str(&format!("  {}\n", la));
            }
            (Some(la), Some(lb)) => {
                out.push_str(&format!("- {}\n+ {}\n", la, lb));
            }
            (Some(la), None) => {
                out.push_str(&format!("- {}\n", la));
            }
            (None, Some(lb)) => {
                out.push_str(&format!("+ {}\n", lb));
            }
            (None, None) => {}
        }
    }
    out
}

/// Resolve the prism-engine workspace root.
///
/// Strategy:
///   1. `CARGO_MANIFEST_DIR` env var (set during `cargo build`) → pop parent.
///   2. Fall back to current working directory.
fn resolve_workspace_root() -> String {
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let mut path = std::path::PathBuf::from(manifest_dir);
        path.pop();
        return path.to_string_lossy().to_string();
    }
    std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string())
}

// ── PlanBuildHandler ──────────────────────────────────────────────────────
/// Plan a Cargo build for a component by describing the steps that would be taken.

pub struct PlanBuildHandler;

impl McpHandler for PlanBuildHandler {
    fn name(&self) -> &'static str {
        "plan_build"
    }

    fn description(&self) -> &'static str {
        "Plan a Cargo build for a component"
    }

    fn input_schema(&self) -> serde_json::Value {
        json_input_schema(
            json!({
                "component": {
                    "type": "string",
                    "description": "Name of the workspace component to build"
                },
                "profile": {
                    "type": "string",
                    "description": "Build profile (debug, release)",
                    "default": "debug"
                },
                "features": {
                    "type": "string",
                    "description": "Optional feature flags, space-separated"
                }
            }),
            &["component"],
        )
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let component = extract_string(request.args, "component")?;
        let profile = extract_string_or_default(request.args, "profile", "debug");
        let features = extract_optional_string(request.args, "features");

        let feature_desc = match &features {
            Some(f) => format!(" --features {}", f),
            None => String::new(),
        };

        let plan = json!({
            "component": component,
            "profile": profile,
            "features": features,
            "steps": [
                format!("1. Run `cargo build -p {} --profile {}{}`", component, profile, feature_desc),
                format!("2. Collect build artifacts for `{}` from target/{}", component, profile),
                format!("3. Verify `{}` was produced successfully", component),
            ],
            "estimated_commands": [
                format!("cargo build -p {} --profile {}", component, profile)
            ]
        });

        Ok(ToolResult::text(serde_json::to_string_pretty(&plan)?))
    }
}

// ── BuildComponentHandler ──────────────────────────────────────────────────
/// Build a workspace component via `cargo build -p`.

pub struct BuildComponentHandler;

impl McpHandler for BuildComponentHandler {
    fn name(&self) -> &'static str {
        "build_component"
    }

    fn description(&self) -> &'static str {
        "Build a workspace component via cargo build -p"
    }

    fn input_schema(&self) -> serde_json::Value {
        json_input_schema(
            json!({
                "component": {
                    "type": "string",
                    "description": "Name of the workspace component to build"
                },
                "profile": {
                    "type": "string",
                    "description": "Build profile (debug, release)",
                    "default": "debug"
                },
                "features": {
                    "type": "string",
                    "description": "Optional feature flags, space-separated"
                },
                "target": {
                    "type": "string",
                    "description": "Optional build target (e.g. aarch64-apple-darwin)"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "default": 300
                }
            }),
            &["component"],
        )
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let component = extract_string(request.args, "component")?;
        let profile = extract_string_or_default(request.args, "profile", "debug");
        let features = extract_optional_string(request.args, "features");
        let target = extract_optional_string(request.args, "target");
        let timeout_secs = extract_optional_u64(request.args, "timeout_secs").unwrap_or(300);

        let workspace_root = resolve_workspace_root();

        let mut owned_args: Vec<String> = vec![
            "build".into(),
            "-p".into(),
            component.clone(),
            "--profile".into(),
            profile.clone(),
        ];

        if let Some(feat) = &features {
            owned_args.push("--features".into());
            owned_args.push(feat.clone());
        }

        if let Some(tgt) = &target {
            owned_args.push("--target".into());
            owned_args.push(tgt.clone());
        }

        let all_strs: Vec<&str> = owned_args.iter().map(|s| s.as_str()).collect();

        info!(
            "building component {} with profile {} (timeout={}s)",
            component, profile, timeout_secs
        );

        let output = run_with_timeout(
            "cargo",
            &all_strs,
            Some(workspace_root.as_str()),
            Duration::from_secs(timeout_secs),
        )
        .with_context(|| format!("cargo build failed for component '{}'", component))?;

        let result = json!({
            "component": component,
            "profile": profile,
            "features": features,
            "target": target,
            "output": output,
            "exit_code": 0
        });

        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

// ── CheckComponentHandler ──────────────────────────────────────────────────
/// Run cargo check on a component.

pub struct CheckComponentHandler;

impl McpHandler for CheckComponentHandler {
    fn name(&self) -> &'static str {
        "check_component"
    }

    fn description(&self) -> &'static str {
        "Run cargo check on a component"
    }

    fn input_schema(&self) -> serde_json::Value {
        json_input_schema(
            json!({
                "component": {
                    "type": "string",
                    "description": "Name of the workspace component to check"
                },
                "features": {
                    "type": "string",
                    "description": "Optional feature flags, space-separated"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "default": 300
                }
            }),
            &["component"],
        )
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let component = extract_string(request.args, "component")?;
        let features = extract_optional_string(request.args, "features");
        let timeout_secs = extract_optional_u64(request.args, "timeout_secs").unwrap_or(300);

        let workspace_root = resolve_workspace_root();

        let mut owned_args: Vec<String> = vec!["check".into(), "-p".into(), component.clone()];

        if let Some(feat) = &features {
            owned_args.push("--features".into());
            owned_args.push(feat.clone());
        }

        let all_strs: Vec<&str> = owned_args.iter().map(|s| s.as_str()).collect();

        info!(
            "checking component {} (timeout={}s)",
            component, timeout_secs
        );

        let output = run_with_timeout(
            "cargo",
            &all_strs,
            Some(workspace_root.as_str()),
            Duration::from_secs(timeout_secs),
        )
        .with_context(|| format!("cargo check failed for component '{}'", component))?;

        let result = json!({
            "component": component,
            "features": features,
            "output": output,
            "exit_code": 0
        });

        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

// ── TestScopeHandler ──────────────────────────────────────────────────────
/// Find and run tests for changed code.

pub struct TestScopeHandler;

impl McpHandler for TestScopeHandler {
    fn name(&self) -> &'static str {
        "test_scope"
    }

    fn description(&self) -> &'static str {
        "Find and run tests for changed code"
    }

    fn input_schema(&self) -> serde_json::Value {
        json_input_schema(
            json!({
                "component": {
                    "type": "string",
                    "description": "Optional component to scope tests to"
                },
                "scope": {
                    "type": "string",
                    "description": "Scope: \"auto\" runs cargo test on the component; \"all\" lists workspace crates",
                    "default": "auto"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "default": 600
                }
            }),
            &[],
        )
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let component = extract_optional_string(request.args, "component");
        let scope = extract_string_or_default(request.args, "scope", "auto");
        let timeout_secs = extract_optional_u64(request.args, "timeout_secs").unwrap_or(600);

        let workspace_root = resolve_workspace_root();

        match scope.as_str() {
            "all" => {
                let output = run_with_timeout(
                    "cargo",
                    &["metadata", "--format-version=1", "--no-deps"],
                    Some(workspace_root.as_str()),
                    Duration::from_secs(60),
                )
                .context("cargo metadata failed")?;

                let metadata: serde_json::Value =
                    serde_json::from_str(&output).context("failed to parse cargo metadata")?;

                let packages = metadata["packages"]
                    .as_array()
                    .map(|pkgs| {
                        pkgs.iter()
                            .map(|p| {
                                json!({
                                    "name": p["name"],
                                    "version": p["version"],
                                    "manifest_path": p["manifest_path"]
                                })
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

                let result = json!({
                    "scope": "all",
                    "packages": packages,
                    "package_count": packages.len()
                });

                Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
            }
            _ => {
                let mut owned_args: Vec<String> = vec!["test".into()];

                if let Some(comp) = &component {
                    owned_args.push("-p".into());
                    owned_args.push(comp.clone());
                }

                let all_strs: Vec<&str> = owned_args.iter().map(|s| s.as_str()).collect();

                info!(
                    "running tests: component={:?} scope={} timeout={}s",
                    component, scope, timeout_secs
                );

                let output = run_with_timeout(
                    "cargo",
                    &all_strs,
                    Some(workspace_root.as_str()),
                    Duration::from_secs(timeout_secs),
                )
                .context("cargo test failed")?;

                let result = json!({
                    "component": component,
                    "scope": "auto",
                    "output": output,
                    "exit_code": 0
                });

                Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
            }
        }
    }
}

// ── CompareBuildsHandler ──────────────────────────────────────────────────
/// Compare two build receipts and produce a line-by-line text diff.

pub struct CompareBuildsHandler;

impl McpHandler for CompareBuildsHandler {
    fn name(&self) -> &'static str {
        "compare_builds"
    }

    fn description(&self) -> &'static str {
        "Compare two build receipts to identify differences"
    }

    fn input_schema(&self) -> serde_json::Value {
        json_input_schema(
            json!({
                "receipt_a": {
                    "type": "string",
                    "description": "First build receipt"
                },
                "receipt_b": {
                    "type": "string",
                    "description": "Second build receipt"
                }
            }),
            &["receipt_a", "receipt_b"],
        )
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let receipt_a = extract_string(request.args, "receipt_a")?;
        let receipt_b = extract_string(request.args, "receipt_b")?;

        let diff = line_diff(&receipt_a, &receipt_b);

        let result = json!({
            "diff": diff,
            "identical": diff.is_empty()
        });

        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

// ── ChangedBuildSurfaceHandler ────────────────────────────────────────────
/// List files changed between git revisions by running `git diff --name-only`.

pub struct ChangedBuildSurfaceHandler;

impl McpHandler for ChangedBuildSurfaceHandler {
    fn name(&self) -> &'static str {
        "changed_build_surface"
    }

    fn description(&self) -> &'static str {
        "List files changed between git revisions"
    }

    fn input_schema(&self) -> serde_json::Value {
        json_input_schema(
            json!({
                "from_revision": {
                    "type": "string",
                    "description": "Base git revision",
                    "default": "HEAD~1"
                },
                "to_revision": {
                    "type": "string",
                    "description": "Head git revision",
                    "default": "HEAD"
                }
            }),
            &[],
        )
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let from_revision = extract_string_or_default(request.args, "from_revision", "HEAD~1");
        let to_revision = extract_string_or_default(request.args, "to_revision", "HEAD");

        let workspace_root = resolve_workspace_root();

        let output = run_with_timeout(
            "git",
            &["diff", "--name-only", &from_revision, &to_revision],
            Some(workspace_root.as_str()),
            Duration::from_secs(30),
        )
        .context("git diff failed — are you in a git repository?")?;

        let files: Vec<&str> = output.lines().collect();

        let result = json!({
            "from_revision": from_revision,
            "to_revision": to_revision,
            "changed_files": files,
            "count": files.len()
        });

        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}
