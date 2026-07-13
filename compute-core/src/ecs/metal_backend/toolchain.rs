//! MetalToolchain — wraps `xcrun metal` and `xcrun metallib`.
//!
//! Provides a unified interface for compiling Metal source to .metallib,
//! shared across all Metal compilation paths (megakernel, per-layer,
//! fused, primitive, runtime-compiled, and AOT).

use sha2::{Digest, Sha256};

/// Result of compiling a Metal source file.
#[derive(Debug, Clone)]
pub struct MetalCompileOutput {
    pub metallib_bytes: Vec<u8>,
    pub sha256: String,
    pub byte_length: u64,
}

/// Unified Metal toolchain — wraps xcrun metal + metallib.
#[derive(Debug, Clone)]
pub struct MetalToolchain {
    pub sdk: String,
    pub metal_std: String,
    pub optimization: String,
}

impl Default for MetalToolchain {
    fn default() -> Self {
        Self {
            sdk: "macosx".into(),
            metal_std: "metal4.0".into(),
            optimization: "-O3".into(),
        }
    }
}

impl MetalToolchain {
    /// Create a new toolchain with the given parameters.
    pub fn new(sdk: &str, metal_std: &str, optimization: &str) -> Self {
        Self {
            sdk: sdk.into(),
            metal_std: metal_std.into(),
            optimization: optimization.into(),
        }
    }

    /// Check whether xcrun is available on this machine.
    pub fn is_available(&self) -> bool {
        std::process::Command::new("xcrun")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Compile a Metal source string to .metallib bytes.
    ///
    /// Writes source to a temp file, runs xcrun metal to produce AIR,
    /// then xcrun metallib to produce the .metallib. Returns the
    /// compiled bytes and their SHA-256.
    pub fn compile_source(&self, name: &str, source: &str) -> Result<MetalCompileOutput, String> {
        if !self.is_available() {
            return Err(format!("xcrun not found — cannot compile '{name}'"));
        }

        let tmp = std::env::temp_dir().join(format!("tribunus-metal-{}", name));
        let _ = std::fs::create_dir_all(&tmp);

        let src_path = tmp.join("kernel.metal");
        let air_path = tmp.join("kernel.air");
        let metallib_path = tmp.join("kernel.metallib");

        // Write source
        std::fs::write(&src_path, source).map_err(|e| format!("write source: {e}"))?;

        // Compile to AIR
        let status = std::process::Command::new("xcrun")
            .args([
                "-sdk",
                &self.sdk,
                "metal",
                &format!("-std={}", self.metal_std),
                &self.optimization,
                "-c",
            ])
            .arg(src_path.to_str().unwrap())
            .arg("-o")
            .arg(air_path.to_str().unwrap())
            .status()
            .map_err(|e| format!("xcrun metal: {e}"))?;

        if !status.success() {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(format!("metal compile failed for '{name}'"));
        }

        // Link to metallib
        let status = std::process::Command::new("xcrun")
            .args(["-sdk", &self.sdk, "metallib"])
            .arg(air_path.to_str().unwrap())
            .arg("-o")
            .arg(metallib_path.to_str().unwrap())
            .status()
            .map_err(|e| format!("xcrun metallib: {e}"))?;

        if !status.success() {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(format!("metallib link failed for '{name}'"));
        }

        // Read result
        let bytes = std::fs::read(&metallib_path).map_err(|e| format!("read metallib: {e}"))?;

        // Validate MTLB magic
        if bytes.len() < 4 || &bytes[0..4] != b"MTLB" {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(format!("'{name}' .metallib missing MTLB magic"));
        }

        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let sha256 = format!("{:x}", hasher.finalize());
        let byte_length = bytes.len() as u64;

        let _ = std::fs::remove_dir_all(&tmp);

        Ok(MetalCompileOutput {
            metallib_bytes: bytes,
            sha256,
            byte_length,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toolchain_default_values() {
        let tc = MetalToolchain::default();
        assert_eq!(tc.sdk, "macosx");
        assert_eq!(tc.metal_std, "metal4.0");
    }

    #[test]
    fn test_toolchain_is_available_or_false() {
        let tc = MetalToolchain::default();
        // xcrun should be available on macOS with Xcode
        let available = tc.is_available();
        // Just verify it doesn't panic
        let _ = available;
    }

    #[test]
    fn test_compile_invalid_source_returns_error() {
        let tc = MetalToolchain::default();
        if tc.is_available() {
            let result = tc.compile_source("test_bad", "not valid metal source");
            assert!(result.is_err(), "invalid source should fail");
        }
    }
}
