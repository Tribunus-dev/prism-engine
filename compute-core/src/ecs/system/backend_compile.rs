use std::path::PathBuf;

use crate::ecs::component::backend::{
    BackendTarget, BinaryFormat, CompileConfig, CompileMode, CompiledBinary, KernelSource,
};

#[derive(Debug, Clone, thiserror::Error)]
pub enum CompileError {
    #[error("xcrun metal not found \u{2014} Xcode CLI tools may not be installed")]
    MetalNotFound,
    #[error("compilation failed: {details}")]
    CompileFailed { details: String },
    #[error("metallib creation failed: {details}")]
    MetallibFailed { details: String },
    #[error("I/O error: {details}")]
    Io { details: String },
}

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};
use sha2::{Digest, Sha256};

/// Compiles kernel source to backend-specific binaries inline.
///
/// Metal targets are compiled via `xcrun metal` / `xcrun metallib`.
/// ROCm targets receive a placeholder (toolchain not available on macOS).
pub struct BackendCompilationSystem {
    metal_compiler: MetalCompiler,
}

impl Default for BackendCompilationSystem {
    fn default() -> Self {
        Self {
            metal_compiler: MetalCompiler,
        }
    }
}

impl CompilerSystem for BackendCompilationSystem {
    fn name(&self) -> &str {
        "BackendCompilationSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Compilation
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let kernels: Vec<Entity> = world.entities_of_kind(EntityKind::Kernel);
        for entity in kernels {
            let Some(source) = world.get_component::<KernelSource>(entity).cloned() else {
                continue;
            };
            let Some(target) = world.get_component::<BackendTarget>(entity).cloned() else {
                continue;
            };

            let compiled = match target {
                BackendTarget::Metal => {
                    let binary = self
                        .metal_compiler
                        .compile(&source.source, &source.entry_point)?;
                    binary
                }
                BackendTarget::ROCm => compile_rocm(&source)?,
                _ => continue,
            };

            world.add_component(entity, compiled);
        }

        Ok(())
    }
}

// ── MetalCompiler ──────────────────────────────────────────────────────────

/// Owned Metal compilation helper — shells out to `xcrun metal` / `xcrun metallib`.
///
/// This struct has no public entry points outside BackendCompilationSystem;
/// the system owns it as a field.
struct MetalCompiler;

impl MetalCompiler {
    /// Compile Metal source into a `.metallib` binary.
    ///
    /// Uses `-std=metal3 -O3` and captures stderr for diagnostics on failure.
    fn compile(&self, source: &str, entry_point: &str) -> Result<CompiledBinary, CompileError> {
        let tmp = std::env::temp_dir();
        let id = uuid::Uuid::new_v4();
        let source_path = tmp.join(format!("{entry_point}_{id}.metal"));
        let air_path = tmp.join(format!("{entry_point}_{id}.air"));
        let lib_path = tmp.join(format!("{entry_point}_{id}.metallib"));

        // Write source to temp file.
        std::fs::write(&source_path, source).map_err(|e| CompileError::Io {
            details: e.to_string(),
        })?;

        // Clean up temp files on scope exit (best-effort).
        let _cleanup = TempFiles {
            files: vec![source_path.clone(), air_path.clone(), lib_path.clone()],
        };

        // Step 1: .metal -> .air
        let output = std::process::Command::new("xcrun")
            .args(["metal", "-std=metal3", "-O3"])
            .arg("-o")
            .arg(&air_path)
            .arg(&source_path)
            .output()
            .map_err(|_| CompileError::MetalNotFound)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CompileError::CompileFailed {
                details: format!("xcrun metal failed for {entry_point}: {stderr}",),
            });
        }

        // Step 2: .air -> .metallib
        let output = std::process::Command::new("xcrun")
            .args(["metallib"])
            .arg("-o")
            .arg(&lib_path)
            .arg(&air_path)
            .output()
            .map_err(|_| CompileError::MetallibFailed {
                details: "xcrun metallib invocation failed".into(),
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CompileError::MetallibFailed {
                details: format!("xcrun metallib failed for {entry_point}: {stderr}",),
            });
        }

        // Step 3: Read the resulting .metallib
        let data = std::fs::read(&lib_path).map_err(|e| CompileError::Io {
            details: e.to_string(),
        })?;

        let fingerprint = sha256_hex(&data);

        Ok(CompiledBinary {
            format: BinaryFormat::Metallib,
            data,
            fingerprint,
        })
    }
}

// ── ROCm compilation ──────────────────────────────────────────────────────

fn compile_rocm(_source: &KernelSource) -> anyhow::Result<CompiledBinary> {
    Ok(make_placeholder(BinaryFormat::HSACO))
}

// ── Shared helpers ────────────────────────────────────────────────────────

fn make_placeholder(format: BinaryFormat) -> CompiledBinary {
    let data = vec![0u8; 64];
    let fingerprint = sha256_hex(&data);
    CompiledBinary {
        format,
        data,
        fingerprint,
    }
}

/// Best-effort cleanup of temp files when dropped.
struct TempFiles {
    files: Vec<PathBuf>,
}

impl Drop for TempFiles {
    fn drop(&mut self) {
        for f in &self.files {
            let _ = std::fs::remove_file(f);
        }
    }
}

/// Cache-aware caching system for compiled binaries.
///
/// Compares each compiled binary's actual SHA-256 against its stored
/// fingerprint and sets a `CompileConfig` component indicating whether
/// the binary was already consistent (cache hit) or freshly produced
/// (cache miss).
pub struct ExecutableCachingSystem;

impl CompilerSystem for ExecutableCachingSystem {
    fn name(&self) -> &str {
        "ExecutableCachingSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Compilation
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let kernels: Vec<Entity> = world.entities_of_kind(EntityKind::Kernel);
        for entity in kernels {
            let Some(binary) = world.get_component::<CompiledBinary>(entity) else {
                continue;
            };

            let actual = sha256_hex(&binary.data);
            let hit = actual == binary.fingerprint;

            let config = CompileConfig {
                mode: if hit {
                    CompileMode::Optimized
                } else {
                    CompileMode::Debug
                },
                features: vec![if hit {
                    "cache_hit".to_string()
                } else {
                    "cache_miss".to_string()
                }],
            };
            world.add_component(entity, config);
        }

        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn sha256_hex(bytes: &[u8]) -> String {
    let hash = Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for byte in &hash {
        use std::fmt::Write;
        write!(hex, "{:02x}", byte).unwrap();
    }
    hex
}
