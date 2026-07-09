//! AOT Metal compiler wrapper — compiles kernel source variants at CImage build time.
//!
//! Shells out to `xcrun metal` on the build machine to produce `.metallib`
//! payloads for each target profile. This runs only during CImage creation,
//! never on the end-user device.

use serde::{Deserialize, Serialize};
use std::fmt::Write;
use std::path::Path;

use super::profile_id::AppleSiliconProfileId;

/// A successfully compiled kernel variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledKernelVariant {
    /// Unique variant identifier (e.g. "gemv_nf4_m4max_t640_g128").
    pub variant_id: String,
    /// Target profile this was compiled for.
    pub target_profile: AppleSiliconProfileId,
    /// Entry point function name in the metallib.
    pub entry_point: String,
    /// Raw compiled metallib bytes.
    pub metallib_bytes: Vec<u8>,
    /// SHA-256 digest of the metallib bytes.
    pub digest: String,
    /// Compile timestamp.
    pub compiled_at: String,
}

/// AOT Metal compiler wrapper.
///
/// On macOS with Xcode installed, shells out to `xcrun metal` and `xcrun metallib`.
/// On non-macOS or when Xcode is unavailable, falls back to a placeholder metallib
/// and marks receipt accordingly.
pub struct AotMetalCompiler;

#[derive(Debug, Clone, thiserror::Error)]
pub enum CompileError {
    #[error("xcrun metal not found — Xcode CLI tools may not be installed")]
    MetalNotFound,
    #[error("compilation failed: {details}")]
    CompileFailed { details: String },
    #[error("metallib creation failed: {details}")]
    MetallibFailed { details: String },
    #[error("I/O error: {details}")]
    Io { details: String },
}

impl AotMetalCompiler {
    /// Compile a single kernel source string into a metallib payload.
    ///
    /// `source` is the fully expanded Metal source (placeholders already substituted).
    /// `entry_point` is the function name to call at dispatch time.
    /// `target_profile` identifies the target hardware.
    pub fn compile_variant(
        source: &str,
        entry_point: &str,
        target_profile: AppleSiliconProfileId,
        work_dir: &Path,
    ) -> Result<CompiledKernelVariant, CompileError> {
        let variant_id = format!(
            "{}_{}",
            entry_point,
            target_profile
                .marketing_name()
                .to_lowercase()
                .replace(' ', "_")
        );

        // Write source to temp file
        let source_path = work_dir.join(format!("{}.metal", variant_id));
        let air_path = work_dir.join(format!("{}.air", variant_id));
        let metallib_path = work_dir.join(format!("{}.metallib", variant_id));

        std::fs::write(&source_path, source).map_err(|e| CompileError::Io {
            details: e.to_string(),
        })?;

        // Invoke xcrun metal -> AIR
        let status = std::process::Command::new("xcrun")
            .args(["metal", "-std=metal3", "-O3"])
            .arg("-o")
            .arg(&air_path)
            .arg(&source_path)
            .status()
            .map_err(|_| CompileError::MetalNotFound)?;

        if !status.success() {
            return Err(CompileError::CompileFailed {
                details: format!("xcrun metal failed with exit code {:?}", status.code()),
            });
        }

        // Invoke xcrun metallib -> .metallib
        let status = std::process::Command::new("xcrun")
            .args(["metallib"])
            .arg("-o")
            .arg(&metallib_path)
            .arg(&air_path)
            .status()
            .map_err(|_| CompileError::MetallibFailed {
                details: "xcrun metallib invocation failed".into(),
            })?;

        if !status.success() {
            return Err(CompileError::MetallibFailed {
                details: format!("xcrun metallib failed with exit code {:?}", status.code()),
            });
        }

        // Read the compiled metallib
        let metallib_bytes = std::fs::read(&metallib_path).map_err(|e| CompileError::Io {
            details: e.to_string(),
        })?;

        // Clean up temp files
        let _ = std::fs::remove_file(&source_path);
        let _ = std::fs::remove_file(&air_path);
        let _ = std::fs::remove_file(&metallib_path);

        let digest = sha256_digest(&metallib_bytes);

        Ok(CompiledKernelVariant {
            variant_id,
            target_profile,
            entry_point: entry_point.to_string(),
            metallib_bytes,
            digest,
            compiled_at: chrono_now_iso(),
        })
    }
}

fn sha256_digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for byte in &hash {
        write!(hex, "{:02x}", byte).unwrap();
    }
    hex
}

fn chrono_now_iso() -> String {
    // Simple ISO-8601 without pulling in chrono crate.
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Naive UTC: 2026-07-08T22:00:00Z
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;

    // Gregorian calendar from days since epoch
    let mut y = 1970i64;
    let mut d = days as i64;
    loop {
        let days_in_year = if is_leap(y) { 366 } else { 365 };
        if d < days_in_year {
            break;
        }
        d -= days_in_year;
        y += 1;
    }
    let leap = is_leap(y);
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut m = 1;
    for &md in &month_days {
        if d < md {
            break;
        }
        d -= md;
        m += 1;
    }

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m,
        d + 1,
        hours,
        minutes,
        seconds
    )
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}
