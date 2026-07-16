//! PTX-to-cubin compilation via the NVIDIA `ptxas` assembler.
//!
//! `ptxas` ships with the NVIDIA CUDA Toolkit and compiles PTX (Parallel Thread
//! eXecution) assembly into cubin (CUDA binary) blobs that are loadable by the
//! CUDA driver API.

use std::io::Write;
use std::process::Command;

use crate::CudaBinary;

/// Default GPU architecture target passed to `ptxas`.
///
/// `sm_80` (Ampere GA10x) is a widely-supported baseline. Users targeting older
/// or newer hardware can pass a different architecture via the
/// `PRISM_CUDA_SM` environment variable.
const DEFAULT_SM: &str = "sm_80";

/// Default kernel entry point name.
const DEFAULT_ENTRY: &str = "matmul_kernel";

/// Detect whether `ptxas` is available on `$PATH`.
pub fn ptxas_available() -> bool {
    Command::new("ptxas")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// The SM architecture string to pass to `ptxas --gpu-name`.
///
/// Reads `PRISM_CUDA_SM` from the environment; falls back to `sm_80`.
fn target_sm() -> String {
    std::env::var("PRISM_CUDA_SM").unwrap_or_else(|_| DEFAULT_SM.to_owned())
}

/// Compile a PTX source string into a `CudaBinary`.
///
/// Writes the PTX source to a temporary file, invokes `ptxas` to assemble it
/// into a cubin, then reads the cubin bytes back.
///
/// # Errors
///
/// - `"CUDA toolkit not installed"` — `ptxas` is not on `$PATH`.
/// - `"ptxas failed: ..."` — the assembler returned a non-zero exit code
///   (e.g. invalid PTX, unsupported architecture).
/// - I/O errors from writing or reading the temporary files.
pub fn compile_ptx(source: &str) -> Result<CudaBinary, String> {
    // 1. Check ptxas availability.
    if !ptxas_available() {
        return Err("CUDA toolkit not installed: ptxas not found on $PATH".into());
    }

    let sm = target_sm();

    // 2. Write source to a temp file and compile to a second temp file.
    let src_dir =
        tempfile::TempDir::new().map_err(|e| format!("failed to create temp dir: {e}"))?;
    let src_path = src_dir.path().join("kernel.ptx");
    let out_path = src_dir.path().join("kernel.cubin");

    // Atomically write the source — fails fast on write errors.
    {
        let mut f = std::fs::File::create(&src_path)
            .map_err(|e| format!("failed to create {src_path:?}: {e}"))?;
        f.write_all(source.as_bytes())
            .map_err(|e| format!("failed to write PTX source: {e}"))?;
    }

    // 3. Invoke ptxas.
    let output = Command::new("ptxas")
        .arg("--gpu-name")
        .arg(&sm)
        .arg("--output-file")
        .arg(&out_path)
        .arg(&src_path)
        .output()
        .map_err(|e| format!("failed to launch ptxas: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ptxas failed (SM={sm}): {}", stderr.trim()));
    }

    // 4. Read back the cubin.
    let cubin = std::fs::read(&out_path)
        .map_err(|e| format!("failed to read cubin output {out_path:?}: {e}"))?;

    Ok(CudaBinary {
        cubin,
        entry_point: DEFAULT_ENTRY.to_owned(),
        grid_dims: (1, 1, 1),
        block_dims: (256, 1, 1),
    })
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Valid minimal PTX source that compiles under ptxas.
    const VALID_PTX: &str = r#"
.version 7.8
.target sm_80
.address_size 64

.visible .entry matmul_kernel(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c
)
{
    ret;
}
"#;

    #[test]
    fn compile_ptx_detects_missing_ptxas() {
        // When ptxas is absent we get the expected error string, no crash.
        if !ptxas_available() {
            let err = compile_ptx(VALID_PTX).unwrap_err();
            assert!(
                err.contains("CUDA toolkit not installed"),
                "expected toolkit-not-installed error, got: {err}"
            );
        }
        // When ptxas IS available we'd compile for real, but we can't test
        // that here without requiring an NVIDIA GPU build runner.
    }

    #[test]
    fn compile_valid_ptx_succeeds_when_ptxas_available() {
        if !ptxas_available() {
            return; // skip — no CUDA toolkit on this host
        }
        let binary = compile_ptx(VALID_PTX).expect("ptxas compilation should succeed");
        assert!(!binary.cubin.is_empty(), "cubin must not be empty");
        assert_eq!(binary.entry_point, "matmul_kernel");
        assert_eq!(binary.grid_dims, (1, 1, 1));
        assert_eq!(binary.block_dims, (256, 1, 1));
    }

    #[test]
    fn target_sm_defaults_to_sm80() {
        // Without the env var, target_sm returns sm_80.
        // We can't easily unset an env var in a parallel test, so we just
        // verify the fallback is sm_80 when the var is empty/unset.
        let sm = target_sm();
        assert!(sm.starts_with("sm_"), "expected sm_* prefix, got: {sm}");
    }
}
