use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};

pub struct RepoHandler;

impl RepoHandler {
    pub fn new() -> Self {
        Self
    }
}

impl McpHandler for RepoHandler {
    fn name(&self) -> &'static str {
        "workspace_summary"
    }

    fn description(&self) -> &'static str {
        "Summarize a Cargo workspace structure: crates, bins, dependencies."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to workspace (default: current dir)" }
            }
        })
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        _state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let path = request.args["path"].as_str().unwrap_or(".");
        let metadata = get_cargo_metadata(path)?;

        let mut output = format!(
            "Workspace: {} ({} members)\n\n",
            metadata.workspace_root,
            metadata.members.len()
        );

        for member in &metadata.members {
            output.push_str(&format!("  {} ({})\n", member.name, member.manifest_path));
        }

        output.push_str("\nDependencies:\n");
        for dep in &metadata.dependencies {
            output.push_str(&format!(
                "  {}: {} {}\n",
                dep.package, dep.name, dep.version
            ));
        }

        Ok(ToolResult::text(output))
    }
}

struct CargoMetadata {
    workspace_root: String,
    members: Vec<MemberInfo>,
    dependencies: Vec<DepInfo>,
}

struct MemberInfo {
    name: String,
    manifest_path: String,
}

struct DepInfo {
    package: String,
    name: String,
    version: String,
}

fn get_cargo_metadata(path: &str) -> anyhow::Result<CargoMetadata> {
    let output = std::process::Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(path)
        .output()
        .map_err(|e| anyhow::anyhow!("cargo metadata failed: {}", e))?;

    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "cargo metadata exited with {}",
            output.status
        ));
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let root = json["workspace_root"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();

    let mut members = Vec::new();
    let mut dependencies = Vec::new();

    if let Some(packages) = json["packages"].as_array() {
        for pkg in packages {
            let name = pkg["name"].as_str().unwrap_or("").to_string();
            let manifest = pkg["manifest_path"].as_str().unwrap_or("").to_string();
            members.push(MemberInfo {
                name: name.clone(),
                manifest_path: manifest,
            });

            if let Some(deps) = pkg["dependencies"].as_array() {
                for dep in deps {
                    let dep_name = dep["name"].as_str().unwrap_or("").to_string();
                    let dep_req = dep["req"].as_str().unwrap_or("*").to_string();
                    dependencies.push(DepInfo {
                        package: name.clone(),
                        name: dep_name,
                        version: dep_req,
                    });
                }
            }
        }
    }

    Ok(CargoMetadata {
        workspace_root: root,
        members,
        dependencies,
    })
}
