//! Host environment capture for decode attribution benchmarks.
//!
//! Captures host chip, macOS version, and toolchain availability
//! (xcode build + coremlcompiler) at benchmark startup. All probes
//! are best-effort: a missing coremlcompiler is reported as
//! `coreai_compiler_available = false` rather than a hard error.

use std::process::Command;

/// Host identity recorded at benchmark startup.
#[derive(Debug, Clone)]
pub struct HostEnvironment {
    /// CPU brand string, e.g. "Apple M1" or "Intel(R) Core(TM) i9-..."
    pub host_chip: String,
    /// macOS version, e.g. "15.5"
    pub macos_version: String,
    /// Xcode build version, e.g. "17C100"
    pub xcode_build_version: String,
    /// coremlcompiler version string
    pub coremlcompiler_version: String,
    /// Whether coremlcompiler is reachable via xcrun and returned a valid version.
    pub coreai_compiler_available: bool,
}

/// Capture the current machine environment by probing sysctl, sw_vers,
/// xcode-select, and xcrun. Records `coreai_compiler_available = false`
/// when coremlcompiler is unreachable, rather than returning a hard error.
pub fn capture_host_environment() -> Result<HostEnvironment, String> {
    // ── Host chip ─────────────────────────────────────────────────────────────
    // On arm64, `machdep.cpu.brand_string` is empty, so fall back to `hw.machine`
    // (returns "arm64"). For Intel the brand string has the full processor name.
    let host_chip = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if !s.is_empty() {
                    return Some(s);
                }
            }
            None
        })
        .unwrap_or_else(|| {
            // arm64 fallback — "machdep.cpu.brand_string" is empty on Apple Silicon
            Command::new("sysctl")
                .args(["-n", "hw.machine"])
                .output()
                .ok()
                .and_then(|o| {
                    if o.status.success() {
                        let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                        if !s.is_empty() {
                            return Some(s);
                        }
                    }
                    None
                })
                .unwrap_or_else(|| "unknown".into())
        });

    // ── macOS version ─────────────────────────────────────────────────────────
    let macos_version = Command::new("sw_vers")
        .args(["-productVersion"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into());

    // ── Toolchain probes (xcode build + coremlcompiler) ──────────────────────
    // Best-effort: a failure here is reported as unavailable rather than
    // a hard error so the benchmark harness can proceed.
    let (xcode_build_version, coremlcompiler_version, coreai_compiler_available) =
        match probe_toolchain() {
            Ok((xcode, compiler)) => (xcode, compiler, true),
            Err(diag) => (
                "unknown".into(),
                format!("unavailable: {diag}"),
                false,
            ),
        };

    Ok(HostEnvironment {
        host_chip,
        macos_version,
        xcode_build_version,
        coremlcompiler_version,
        coreai_compiler_available,
    })
}

/// Probe xcode-select for the Xcode build version and xcrun for
/// coremlcompiler's version. Returns a diagnostic on failure.
fn probe_toolchain() -> Result<(String, String), String> {
    let xcode_build = Command::new("xcode-select")
        .arg("-p")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".into());

    let coremlcompiler = Command::new("xcrun")
        .args(["coremlcompiler", "--version"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .ok_or_else(|| "xcrun coremlcompiler unreachable".to_string())?;

    Ok((xcode_build, coremlcompiler))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_host_environment_does_not_panic() {
        // capture_host_environment probes the host; on a Linux dev box
        // it returns a populated struct with `coreai_compiler_available = false`.
        let env = capture_host_environment();
        assert!(env.is_ok());
        let env = env.expect("ok");
        assert!(!env.host_chip.is_empty());
        assert!(!env.macos_version.is_empty());
    }
}
