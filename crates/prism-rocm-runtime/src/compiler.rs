//! AMDGCN-to-HSACO compilation via the ROCm `amdllvm` assembler.
//!
//! `amdllvm` ships with the ROCm LLVM fork and assembles AMDGCN (AMD Graphics
//! Core Next) assembly into HSACO (HSA Code Object) blobs loadable by the HSA
//! runtime. Falls back to `hipcc` when `amdllvm` is unavailable.

use crate::AmdBinary;

#[cfg(feature = "rocm-runtime")]
use std::process::Command;

/// Default GPU architecture target passed to `amdllvm`.
///
/// `gfx1100` (RDNA 3 / RX 7000 series) is a modern widely-supported baseline.
/// Users targeting older or newer hardware can pass a different architecture
/// via the `PRISM_ROCM_GPU` environment variable.
#[cfg(feature = "rocm-runtime")]
const DEFAULT_GPU: &str = "gfx1100";

/// Default kernel entry point name.
#[cfg(feature = "rocm-runtime")]
const DEFAULT_ENTRY: &str = "matmul_kernel";

/// Detect whether `amdllvm` (or `hipcc`) is available on `$PATH` or in a
/// standard ROCm installation.
pub fn rocm_available() -> bool {
    #[cfg(feature = "rocm-runtime")]
    {
        locate_assembler().is_some()
    }
    #[cfg(not(feature = "rocm-runtime"))]
    {
        false
    }
}

/// The GPU architecture string to pass to `amdllvm -mcpu`.
#[cfg(feature = "rocm-runtime")]
fn target_gpu() -> String {
    std::env::var("PRISM_ROCM_GPU").unwrap_or_else(|_| DEFAULT_GPU.to_owned())
}

/// Compile an AMDGCN assembly source string into an `AmdBinary`.
///
/// Writes the source to a temporary file, invokes `amdllvm` (or `hipcc`) to
/// assemble it into an HSACO, then reads the code object bytes back.
///
/// # Errors
///
/// - `"ROCm not installed"` — the assembler is not found on `$PATH` or in
///   `/opt/rocm`.
/// - `"ROCm assembly failed: ..."` — the assembler returned a non-zero exit
///   code (e.g., invalid AMDGCN, unsupported architecture).
/// - I/O errors from writing or reading the temporary files.
pub fn compile_amdgcn(source: &str) -> Result<AmdBinary, String> {
    #[cfg(feature = "rocm-runtime")]
    {
        compile_amdgcn_inner(source)
    }

    #[cfg(not(feature = "rocm-runtime"))]
    {
        let _ = source;
        Err("ROCm not installed".into())
    }
}

/// Inner compilation — only compiled on Linux.
#[cfg(feature = "rocm-runtime")]
fn compile_amdgcn_inner(source: &str) -> Result<AmdBinary, String> {
    // ── 1. Locate the ROCm assembler ──────────────────────────────────
    let assembler = locate_assembler()
        .ok_or_else(|| "ROCm not installed: no amdllvm or hipcc found".to_string())?;

    // ── 2. Write source to a temporary file ───────────────────────────
    let dir = tempfile::tempdir().map_err(|e| format!("failed to create temp dir: {e}"))?;
    let src_path = dir.path().join("kernel.s");
    std::fs::write(&src_path, source).map_err(|e| format!("failed to write source: {e}"))?;

    // ── 3. Determine target architecture ──────────────────────────────
    let gpu = target_gpu();
    let out_path = dir.path().join("kernel.co");

    // ── 4. Invoke the assembler ───────────────────────────────────────
    // amdllvm invocation: amdllvm -triple=amdgcn-amd-amdhsa -mcpu=<gpu>
    //                     -filetype=obj -o <output> <input>
    //
    // hipcc invocation (fallback): hipcc --amdgpu-target=<gpu> -c <input>
    let assembler_name = assembler.file_name().unwrap_or_default().to_string_lossy();
    let output = if assembler_name.contains("amdllvm") {
        Command::new(&assembler)
            .args([
                "-triple=amdgcn-amd-amdhsa",
                "-mcpu=",
                &gpu,
                "-filetype=obj",
                "-o",
            ])
            .arg(&out_path)
            .arg(&src_path)
            .output()
            .map_err(|e| format!("failed to invoke amdllvm '{assembler}': {e}"))?
    } else {
        // hipcc fallback
        Command::new(&assembler)
            .args(["--amdgpu-target=", &gpu, "-c", "-o"])
            .arg(&out_path)
            .arg(&src_path)
            .output()
            .map_err(|e| format!("failed to invoke hipcc '{assembler}': {e}"))?
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "ROCm assembly failed (status {}):\nstdout:\n{}\nstderr:\n{}",
            output.status, stdout, stderr
        ));
    }

    // ── 5. Read the resulting code object ─────────────────────────────
    let code_object =
        std::fs::read(&out_path).map_err(|e| format!("failed to read code object: {e}"))?;

    if code_object.is_empty() {
        return Err("ROCm assembler produced an empty code object".into());
    }

    Ok(AmdBinary {
        code_object,
        entry_point: extract_entry_point(source)
            .unwrap_or(DEFAULT_ENTRY)
            .to_owned(),
        grid_dims: (1, 1, 1),
        block_dims: (256, 1, 1),
    })
}

/// Locate an AMDGCN assembler in the ROCm toolchain.
///
/// Checks common install paths and `$PATH`.
#[cfg(feature = "rocm-runtime")]
fn locate_assembler() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;

    // 1. Check $PATH first via `which amdllvm` / `which hipcc`.
    for name in &["amdllvm", "hipcc"] {
        if Command::new("which")
            .arg(name)
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout)
                        .ok()
                        .map(|s| PathBuf::from(s.trim()))
                } else {
                    None
                }
            })
            .is_some()
        {
            let path = PathBuf::from(name);
            return Some(path);
        }
    }

    // 2. Check standard ROCm paths.
    let rocm_root = locate_rocm_root()?;
    let bin = rocm_root.join("bin");
    for name in &["amdllvm", "hipcc"] {
        let candidate = bin.join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

/// Locate the ROCm installation root.
#[cfg(feature = "rocm-runtime")]
fn locate_rocm_root() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;

    // Check environment variable first.
    if let Ok(path) = std::env::var("ROCM_PATH") {
        let p = PathBuf::from(&path);
        if p.join("bin").exists() {
            return Some(p);
        }
    }
    // Default locations.
    for candidate in &["/opt/rocm", "/usr/lib/rocm"] {
        let p = PathBuf::from(candidate);
        if p.join("bin").exists() {
            return Some(p);
        }
    }
    None
}

/// Extract the entry-point function name from AMDGCN source.
#[allow(dead_code)]
fn extract_entry_point(source: &str) -> Option<&str> {
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(".entry ") {
            let name = trimmed.strip_prefix(".entry ")?.trim();
            return name.split_whitespace().next();
        }
    }
    None
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Without ROCm installed, compile returns the graceful error.
    #[test]
    fn compile_without_rocm_returns_error() {
        // A minimal AMDGCN source string — syntactically valid but never
        // reaches an assembler on a macOS development machine.
        let source = ".text\n.rodata\n.entry matmul_kernel()\n.end\n";
        let result = compile_amdgcn(source);
        assert!(
            result.is_err(),
            "expected error without ROCm, got: {result:?}"
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("ROCm not installed"),
            "expected 'ROCm not installed' error, got: {err}"
        );
    }

    /// extract_entry_point finds the `.entry` directive.
    #[test]
    fn extract_entry_point_works() {
        let source = ".text\n.rodata\n.entry matmul_kernel()\n.end\n";
        assert_eq!(extract_entry_point(source), Some("matmul_kernel()"));

        let source_no_entry = ".text\n.end\n";
        assert_eq!(extract_entry_point(source_no_entry), None);
    }

    /// rocm_available returns false without the feature or on non-Linux.
    #[test]
    fn rocm_available_on_macos() {
        assert!(!rocm_available());
    }
}
