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

fn json_input_schema(mut props: serde_json::Value, required: &[&str]) -> serde_json::Value {
    if let Some(properties) = props.as_object_mut() {
        properties.insert(
            "workspace_root".into(),
            json!({
                "type": "string",
                "description": "Absolute workspace root. Never inferred from the persistent daemon process CWD."
            }),
        );
    }
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

/// Resolve a workspace independently of the persistent daemon's launchd CWD.
fn resolve_workspace_root(args: &serde_json::Value) -> Result<String> {
    let requested = args
        .get("workspace_root")
        .and_then(serde_json::Value::as_str)
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("PRISM_MCPD_WORKSPACE_ROOT").map(std::path::PathBuf::from))
        .unwrap_or_else(|| {
            let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            path.pop();
            path
        });
    let root = requested
        .canonicalize()
        .with_context(|| format!("workspace does not exist: {}", requested.display()))?;
    if !root.join("Cargo.toml").is_file() {
        anyhow::bail!(
            "workspace_root does not contain Cargo.toml: {}",
            root.display()
        );
    }
    Ok(root.to_string_lossy().into_owned())
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

        let workspace_root = resolve_workspace_root(request.args)?;

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

        let workspace_root = resolve_workspace_root(request.args)?;

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
                    "description": "Scope: \"auto\" runs the component library tests without compiling CLI binaries; \"all\" lists workspace crates",
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

        let workspace_root = resolve_workspace_root(request.args)?;

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
                    // Component-scoped tests default to the library target. A
                    // Cargo test-name filter does not constrain target
                    // compilation, so omitting this would compile every CLI.
                    owned_args.push("--lib".into());
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
                    "target_kind": if component.is_some() { "lib" } else { "workspace-default" },
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

        let workspace_root = resolve_workspace_root(request.args)?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_resolution_uses_explicit_request_not_process_cwd() {
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(workspace.path().join("Cargo.toml"), "[workspace]\n")
            .expect("workspace manifest");
        let args = json!({"workspace_root": workspace.path()});
        assert_eq!(
            std::path::PathBuf::from(resolve_workspace_root(&args).unwrap()),
            workspace.path().canonicalize().unwrap()
        );
    }

    #[test]
    fn build_tool_schema_exposes_workspace_root() {
        let schema = CheckComponentHandler.input_schema();
        assert_eq!(schema["properties"]["workspace_root"]["type"], "string");
    }
}
