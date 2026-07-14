// prism_mcp_kernel handlers — 9 kernel management MCP tools
//
// Simplified pattern: no ToolDependencies struct, handlers use DaemonState
// from the trait call signature. Pre-detects system state at construction
// where appropriate (list_kernel_backends).

use anyhow::{Context, Result};
use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde_json::{json, Value};

// ── Helpers ────────────────────────────────────────────────────────────────

fn arg_str<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args[key]
        .as_str()
        .with_context(|| format!("missing required argument \"{}\"", key))
}

fn json_schema(props: Value, required: &[&str]) -> Value {
    let req: Vec<Value> = required.iter().map(|s| json!(s)).collect();
    json!({"type":"object","properties":props,"required":req})
}

// ── 1. list_kernel_backends ─────────────────────────────────────────────────
//
// Pre-detects Metal/LLVM compilers on PATH at construction time.

struct BackendInfo {
    name: String,
    description: String,
    available: bool,
    path: String,
}

fn detect_backends() -> Vec<BackendInfo> {
    let mut b = Vec::new();

    match std::process::Command::new("xcrun")
        .args(["--sdk", "macosx", "-f", "metal"])
        .output()
    {
        Ok(o) => {
            let path = String::from_utf8_lossy(&o.stdout).trim().to_string();
            b.push(BackendInfo {
                name: "metal".into(),
                description: "Apple Metal GPU shading language compiler".into(),
                available: o.status.success() && !path.is_empty(),
                path,
            });
        }
        Err(e) => b.push(BackendInfo {
            name: "metal".into(),
            description: "Apple Metal GPU shading language compiler".into(),
            available: false,
            path: format!("xcrun not found: {}", e),
        }),
    }

    match std::process::Command::new("which").arg("clang").output() {
        Ok(o) => {
            let path = String::from_utf8_lossy(&o.stdout).trim().to_string();
            b.push(BackendInfo {
                name: "llvm".into(),
                description: "LLVM/clang general-purpose compiler".into(),
                available: o.status.success() && !path.is_empty(),
                path,
            });
        }
        Err(e) => b.push(BackendInfo {
            name: "llvm".into(),
            description: "LLVM/clang general-purpose compiler".into(),
            available: false,
            path: format!("which clang failed: {}", e),
        }),
    }

    b
}

pub struct ListKernelBackends {
    backends: Vec<BackendInfo>,
}

impl ListKernelBackends {
    pub fn new() -> Self {
        Self {
            backends: detect_backends(),
        }
    }
}

impl McpHandler for ListKernelBackends {
    fn name(&self) -> &'static str {
        "list_kernel_backends"
    }
    fn description(&self) -> &'static str {
        "List available kernel compilation backends detected on system PATH"
    }
    fn input_schema(&self) -> Value {
        json_schema(json!({}), &[])
    }
    fn call(
        &self,
        _request: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let list: Vec<Value> = self
            .backends
            .iter()
            .map(|b| {
                json!({
                    "name": b.name,
                    "description": b.description,
                    "available": b.available,
                    "path": b.path,
                })
            })
            .collect();
        Ok(ToolResult::text(serde_json::to_string_pretty(&list)?))
    }
}

// ── 2. compile_kernel_recipe ────────────────────────────────────────────────
//
// Compile a Metal kernel source file via xcrun metal subprocess.

pub struct CompileKernelRecipe;

impl McpHandler for CompileKernelRecipe {
    fn name(&self) -> &'static str {
        "compile_kernel_recipe"
    }
    fn description(&self) -> &'static str {
        "Compile a Metal kernel source file via xcrun metal subprocess"
    }
    fn input_schema(&self) -> Value {
        json_schema(
            json!({
                "source_path": {
                    "type": "string",
                    "description": "Path to the .metal kernel source file"
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional output path (default: source_path with .air extension)"
                },
                "backend": {
                    "type": "string",
                    "description": "Target compiler backend",
                    "enum": ["metal", "llvm"],
                    "default": "metal"
                }
            }),
            &["source_path"],
        )
    }
    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let source_path = arg_str(request.args, "source_path")?;
        let backend = request.args["backend"].as_str().unwrap_or("metal");
        let explicit_output = request.args["output_path"]
            .as_str()
            .filter(|s| !s.is_empty());

        let output_path = match explicit_output {
            Some(p) => p.to_string(),
            None => {
                let p = std::path::Path::new(source_path);
                let stem = p
                    .file_stem()
                    .map(|s| s.to_string_lossy())
                    .unwrap_or(std::borrow::Cow::Borrowed("kernel"));
                format!("{}.air", p.with_file_name(stem.as_ref()).display())
            }
        };

        let result = match backend {
            "metal" => {
                let child = std::process::Command::new("xcrun")
                    .args(["-sdk", "macosx", "metal", "-o", &output_path, source_path])
                    .output()
                    .context("failed to spawn xcrun metal")?;

                let stdout = String::from_utf8_lossy(&child.stdout).to_string();
                let stderr = String::from_utf8_lossy(&child.stderr).to_string();

                json!({
                    "backend": "metal",
                    "source_path": source_path,
                    "output_path": output_path,
                    "success": child.status.success(),
                    "exit_code": child.status.code(),
                    "stdout": stdout,
                    "stderr": stderr,
                })
            }
            "llvm" => {
                let child = std::process::Command::new("clang")
                    .args(["-c", "-O2", "-o", &output_path, source_path])
                    .output()
                    .context("failed to spawn clang")?;

                let stdout = String::from_utf8_lossy(&child.stdout).to_string();
                let stderr = String::from_utf8_lossy(&child.stderr).to_string();

                json!({
                    "backend": "llvm",
                    "source_path": source_path,
                    "output_path": output_path,
                    "success": child.status.success(),
                    "exit_code": child.status.code(),
                    "stdout": stdout,
                    "stderr": stderr,
                })
            }
            _ => anyhow::bail!("unsupported backend: {}", backend),
        };

        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

// ── 3. compile_kernel_candidates ────────────────────────────────────────────
//
// Enumerate candidate kernel variant configurations for admission testing.
// Stub — generates variant metadata without actual compilation.

pub struct CompileKernelCandidates;

impl McpHandler for CompileKernelCandidates {
    fn name(&self) -> &'static str {
        "compile_kernel_candidates"
    }
    fn description(&self) -> &'static str {
        "Enumerate candidate kernel variant configurations for admission testing"
    }
    fn input_schema(&self) -> Value {
        json_schema(
            json!({
                "recipe_name": {
                    "type": "string",
                    "description": "Base kernel recipe name"
                },
                "variants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optimization variants: fast, precise, memory, debug"
                },
                "backends": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Target backends: metal, llvm"
                }
            }),
            &["recipe_name"],
        )
    }
    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let recipe_name = arg_str(request.args, "recipe_name")?;
        let variants: Vec<&str> = request.args["variants"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_else(|| vec!["fast", "precise", "memory"]);
        let backends: Vec<&str> = request.args["backends"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_else(|| vec!["metal"]);

        let candidates: Vec<Value> = backends
            .iter()
            .flat_map(|b| {
                variants.iter().map(move |v| {
                    let flags = match *v {
                        "fast" => json!(["-O3"]),
                        "precise" => json!(["-O0"]),
                        "memory" => json!(["-O1"]),
                        "debug" => json!(["-g", "-O0"]),
                        _ => json!(["-O2"]),
                    };
                    json!({
                        "recipe_name": recipe_name,
                        "backend": b,
                        "variant": v,
                        "compiler_flags": flags,
                    })
                })
            })
            .collect();

        let result = json!({
            "recipe_name": recipe_name,
            "candidate_count": candidates.len(),
            "candidates": candidates,
        });
        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

// ── 4. inspect_compiled_kernel ──────────────────────────────────────────────
//
// Run otool -h -l on a compiled kernel binary for Mach-O header metadata.

pub struct InspectCompiledKernel;

impl McpHandler for InspectCompiledKernel {
    fn name(&self) -> &'static str {
        "inspect_compiled_kernel"
    }
    fn description(&self) -> &'static str {
        "Run otool -h -l on a compiled kernel binary for Mach-O metadata"
    }
    fn input_schema(&self) -> Value {
        json_schema(
            json!({
                "binary_path": {
                    "type": "string",
                    "description": "Path to the compiled kernel binary"
                }
            }),
            &["binary_path"],
        )
    }
    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let binary_path = arg_str(request.args, "binary_path")?;

        let otool = std::process::Command::new("otool")
            .args(["-h", "-l", binary_path])
            .output()
            .context("failed to spawn otool")?;

        let stdout = String::from_utf8_lossy(&otool.stdout).to_string();
        let stderr = String::from_utf8_lossy(&otool.stderr).to_string();
        let file_size = std::fs::metadata(binary_path).map(|m| m.len()).unwrap_or(0);

        let result = json!({
            "binary_path": binary_path,
            "file_size_bytes": file_size,
            "success": otool.status.success(),
            "exit_code": otool.status.code(),
            "raw_header": stdout,
            "stderr": stderr,
        });
        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

// ── 5. disassemble_kernel ───────────────────────────────────────────────────
//
// Disassemble a compiled kernel binary using otool -V.

pub struct DisassembleKernel;

impl McpHandler for DisassembleKernel {
    fn name(&self) -> &'static str {
        "disassemble_kernel"
    }
    fn description(&self) -> &'static str {
        "Disassemble a compiled kernel binary using otool -V"
    }
    fn input_schema(&self) -> Value {
        json_schema(
            json!({
                "binary_path": {
                    "type": "string",
                    "description": "Path to the compiled kernel binary"
                },
                "arch": {
                    "type": "string",
                    "description": "Target architecture (e.g. arm64e, x86_64)",
                    "default": "arm64"
                }
            }),
            &["binary_path"],
        )
    }
    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let binary_path = arg_str(request.args, "binary_path")?;
        let arch = request.args["arch"].as_str().unwrap_or("arm64");

        let output = std::process::Command::new("otool")
            .args(["-V", "-arch", arch, binary_path])
            .output()
            .context("failed to spawn otool -V")?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let instruction_count = stdout.lines().filter(|l| l.contains('\t')).count();

        let result = json!({
            "binary_path": binary_path,
            "arch": arch,
            "instruction_count": instruction_count,
            "disassembly": stdout,
            "stderr": stderr,
            "success": output.status.success(),
            "exit_code": output.status.code(),
        });
        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

// ── 6. analyze_kernel_resources ─────────────────────────────────────────────
//
// Parse otool -V output for register usage and instruction profile.

pub struct AnalyzeKernelResources;

impl McpHandler for AnalyzeKernelResources {
    fn name(&self) -> &'static str {
        "analyze_kernel_resources"
    }
    fn description(&self) -> &'static str {
        "Parse otool -V disassembly output for register usage and instruction profile"
    }
    fn input_schema(&self) -> Value {
        json_schema(
            json!({
                "binary_path": {
                    "type": "string",
                    "description": "Path to the compiled kernel binary"
                }
            }),
            &["binary_path"],
        )
    }
    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let binary_path = arg_str(request.args, "binary_path")?;

        // Collect disassembly for register and instruction analysis
        let dis_output = std::process::Command::new("otool")
            .args(["-V", binary_path])
            .output()
            .context("failed to spawn otool -V")?;

        let dis_text = String::from_utf8_lossy(&dis_output.stdout).to_string();

        // Register name tables for ARM64 analysis
        let gp_regs: [&str; 29] = [
            "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13",
            "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25",
            "x26", "x27", "x28",
        ];
        let w_regs: [&str; 31] = [
            "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11", "w12", "w13",
            "w14", "w15", "w16", "w17", "w18", "w19", "w20", "w21", "w22", "w23", "w24", "w25",
            "w26", "w27", "w28", "w29", "w30",
        ];
        let vec_regs: [&str; 32] = [
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
            "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
            "v26", "v27", "v28", "v29", "v30", "v31",
        ];
        let simd_regs: [&str; 32] = [
            "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13",
            "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25",
            "d26", "d27", "d28", "d29", "d30", "d31",
        ];
        let special_regs: [&str; 4] = ["sp", "fp", "lr", "xzr"];

        // Count total references for a set of register names
        let count_refs =
            |regs: &[&str]| -> usize { regs.iter().map(|r| dis_text.matches(r).count()).sum() };

        // Count distinct registers used
        let count_distinct =
            |regs: &[&str]| -> usize { regs.iter().filter(|r| dis_text.contains(*r)).count() };

        let gp_refs = count_refs(&gp_regs);
        let w_refs = count_refs(&w_regs);
        let vec_refs = count_refs(&vec_regs);
        let simd_refs = count_refs(&simd_regs);
        let special_refs = count_refs(&special_regs);

        let distinct_used = count_distinct(&gp_regs)
            + count_distinct(&w_regs)
            + count_distinct(&vec_regs)
            + count_distinct(&simd_regs)
            + count_distinct(&special_regs);

        // Count barrier instructions (dmb, dsb, isb, barrier, sync, fence)
        let barriers = ["dmb", "dsb", "isb", "barrier", "sync", "fence"];
        let barrier_count: usize = barriers
            .iter()
            .map(|i| dis_text.lines().filter(|l| l.contains(i)).count())
            .sum();

        // Count atomic instructions (ldaxr, stlxr, ldxr, stxr, swp, cas, atomic)
        let atomics = ["ldaxr", "stlxr", "ldxr", "stxr", "swp", "cas", "atomic"];
        let atomic_count: usize = atomics
            .iter()
            .map(|i| dis_text.lines().filter(|l| l.contains(i)).count())
            .sum();

        // Count total instructions
        let total_instructions = dis_text.lines().filter(|l| l.contains('\t')).count();

        // Detect SIMD/vector usage
        let uses_vector = vec_regs
            .iter()
            .chain(simd_regs.iter())
            .any(|r| dis_text.contains(*r));

        let file_size = std::fs::metadata(binary_path).map(|m| m.len()).unwrap_or(0);

        let simd_estimate = if uses_vector {
            if vec_refs > 0 {
                "SIMD (vectors)"
            } else {
                "SIMD (scalar-paired)"
            }
        } else {
            "scalar"
        };

        let result = json!({
            "binary_path": binary_path,
            "file_size_bytes": file_size,
            "total_instructions": total_instructions,
            "register_references": {
                "general_purpose_x": gp_refs,
                "w_registers": w_refs,
                "vector_v": vec_refs,
                "simd_d": simd_refs,
                "special_sp_fp_lr": special_refs,
            },
            "distinct_registers_used": distinct_used,
            "barrier_instructions": barrier_count,
            "atomic_instructions": atomic_count,
            "simd_usage": simd_estimate,
            "otool_success": dis_output.status.success(),
        });
        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

// ── 7. validate_kernel_abi ──────────────────────────────────────────────────
//
// Compare recipe bindings against binary entry points.

pub struct ValidateKernelAbi;

impl McpHandler for ValidateKernelAbi {
    fn name(&self) -> &'static str {
        "validate_kernel_abi"
    }
    fn description(&self) -> &'static str {
        "Compare kernel recipe bindings against binary entry points"
    }
    fn input_schema(&self) -> Value {
        json_schema(
            json!({
                "binary_path": {
                    "type": "string",
                    "description": "Path to compiled kernel binary"
                },
                "bindings": {
                    "type": "object",
                    "description": "Expected entry points and argument signatures",
                    "properties": {
                        "entry_points": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Expected function/entry point names"
                        },
                        "buffer_count": {
                            "type": "integer",
                            "description": "Expected number of buffer arguments"
                        }
                    }
                }
            }),
            &["binary_path"],
        )
    }
    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let binary_path = arg_str(request.args, "binary_path")?;
        let expected_entries: Vec<String> = request.args["bindings"]["entry_points"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let expected_buffers = request.args["bindings"]["buffer_count"]
            .as_i64()
            .unwrap_or(0);

        let metadata = std::fs::metadata(binary_path)
            .map_err(|e| anyhow::anyhow!("cannot inspect binary: {e}"))?;
        let symbols = std::process::Command::new("otool")
            .args(["-Iv", binary_path])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        let missing: Vec<String> = expected_entries
            .iter()
            .filter(|entry| !symbols.contains(entry.as_str()))
            .cloned()
            .collect();
        let result = json!({
            "binary_path": binary_path,
            "file_size_bytes": metadata.len(),
            "expected_entry_points": expected_entries,
            "expected_buffer_count": expected_buffers,
            "abi_compatible": missing.is_empty(),
            "abi_version": 1,
            "missing_entry_points": missing,
            "errors": [],
        });
        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

// ── 8. compare_kernels ──────────────────────────────────────────────────────
//
// Compare two kernel binaries by collecting metadata from both and diffing.

pub struct CompareKernels;

impl McpHandler for CompareKernels {
    fn name(&self) -> &'static str {
        "compare_kernels"
    }
    fn description(&self) -> &'static str {
        "Compare two kernel binaries by collecting metadata from both and diffing"
    }
    fn input_schema(&self) -> Value {
        json_schema(
            json!({
                "binary_path_a": {
                    "type": "string",
                    "description": "First kernel binary path"
                },
                "binary_path_b": {
                    "type": "string",
                    "description": "Second kernel binary path"
                }
            }),
            &["binary_path_a", "binary_path_b"],
        )
    }
    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let path_a = arg_str(request.args, "binary_path_a")?;
        let path_b = arg_str(request.args, "binary_path_b")?;

        let collect_meta = |path: &str| -> Result<Value> {
            let meta = std::fs::metadata(path);
            let file_size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
            let header = std::process::Command::new("otool")
                .args(["-h", path])
                .output()
                .ok()
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_default();
            Ok(json!({
                "path": path,
                "exists": meta.is_ok(),
                "file_size_bytes": file_size,
                "header": header,
            }))
        };

        let report_a = collect_meta(path_a)?;
        let report_b = collect_meta(path_b)?;

        let size_a = report_a["file_size_bytes"].as_u64().unwrap_or(0);
        let size_b = report_b["file_size_bytes"].as_u64().unwrap_or(0);
        let size_diff = if size_a > size_b {
            size_a - size_b
        } else {
            size_b - size_a
        };

        let comparison: String = if report_a["header"] == report_b["header"] {
            "identical headers".to_string()
        } else if size_a == size_b {
            "same size, different headers".to_string()
        } else {
            format!("differ: A={}B, B={}B", size_a, size_b)
        };

        let result = json!({
            "binary_a": report_a,
            "binary_b": report_b,
            "file_size_difference_bytes": size_diff,
            "same_size": size_a == size_b,
            "comparison": comparison,
        });
        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

// ── 9. register_kernel ──────────────────────────────────────────────────────
//
// Register a kernel binary by name in the daemon's persistent registry.

pub struct RegisterKernel;

impl McpHandler for RegisterKernel {
    fn name(&self) -> &'static str {
        "register_kernel"
    }
    fn description(&self) -> &'static str {
        "Register a kernel binary path by name in the persistent kernel registry"
    }
    fn input_schema(&self) -> Value {
        json_schema(
            json!({
                "name": {
                    "type": "string",
                    "description": "Logical name for the kernel"
                },
                "binary_path": {
                    "type": "string",
                    "description": "Path to the kernel binary"
                },
                "backend": {
                    "type": "string",
                    "description": "Source backend",
                    "default": "metal"
                }
            }),
            &["name", "binary_path"],
        )
    }
    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let name = arg_str(request.args, "name")?;
        let binary_path = arg_str(request.args, "binary_path")?;
        let backend = request.args["backend"].as_str().unwrap_or("metal");

        let meta = std::fs::metadata(binary_path)
            .map_err(|e| anyhow::anyhow!("cannot register kernel: {e}"))?;
        let file_size = meta.len();
        let bytes = std::fs::read(binary_path)?;
        let digest = blake3::hash(&bytes).to_hex().to_string();
        _state
            .projection_store
            .record_kernel(name, backend, &digest, file_size, binary_path)?;

        let result = json!({
            "name": name,
            "backend": backend,
            "binary_path": binary_path,
            "registered": true,
            "file_size_bytes": file_size,
            "artifact_hash": digest,
            "persistent": true,
        });
        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}
