//! TT-Metalium compiler bridge.
//!
//! Calls the TT-Metalium C++ compiler (`tt-metalium`) to translate codegen
//! source (TT-Metalium C++ kernel code) into RISCV-32 ELF binaries.
//!
//! TT-Metalium produces RISC-V ELF files for each kernel; the compiler driver
//! lives at `$TT_METALIUM_ROOT/tt_metal/tools/compile_kernel.py` or is
//! invoked via the `metal` Python package.
//!
//! On systems without TT-Metalium installed, `compile` returns a graceful error
//! rather than panicking.

use std::path::PathBuf;
use std::process::Command;

use crate::TtBinary;

/// Default path where TT-Metalium is expected (Metalium source tree).
const DEFAULT_TT_METALIUM_ROOT: &str = "/opt/tt-metalium";

/// Compile a TT-Metalium kernel source string into a RISCV-32 ELF binary.
///
/// The source is written to a temporary file and compiled through the
/// TT-Metalium build toolchain.
///
/// # Errors
///
/// Returns `Err(String)` if:
/// - TT-Metalium is not installed or configured (`$TT_METALIUM_ROOT` unset
///   and the default path does not exist).
/// - The compiler invocation fails (syntax errors, linking failures).
pub fn compile(source: &str, kernel_name: &str) -> Result<TtBinary, String> {
    let metalium_root = resolve_metalium_root()?;

    // Write the kernel source to a temp directory for compilation.
    let temp_dir = std::env::temp_dir().join("prism-tt-compile");
    std::fs::create_dir_all(&temp_dir)
        .map_err(|e| format!("failed to create compile temp dir: {e}"))?;

    // TT-Metalium kernels are C++ source files with a `.cpp` extension.
    // The kernel name maps to the source file name.
    let source_path = temp_dir.join(format!("{kernel_name}.cpp"));
    std::fs::write(&source_path, source)
        .map_err(|e| format!("failed to write kernel source to {source_path:?}: {e}"))?;

    // Invoke the TT-Metalium kernel compiler.
    // Real command: `$TT_METALIUM_ROOT/tt_metal/tools/compile_kernel.py`
    //             or `python -m tt_metal.tools.compile_kernel`
    //
    // Arguments:
    //   --source   <kernel_name>.cpp     the C++ kernel source
    //   --outdir   <temp_dir>            where to place the ELF
    //   --arch     wormhole_b0           target architecture
    //   --name     <kernel_name>         kernel function name
    let compile_script = metalium_root
        .join("tt_metal")
        .join("tools")
        .join("compile_kernel.py");
    let output = Command::new("python3")
        .arg(&compile_script)
        .arg("--source")
        .arg(&source_path)
        .arg("--outdir")
        .arg(&temp_dir)
        .arg("--arch")
        .arg("wormhole_b0")
        .arg("--name")
        .arg(kernel_name)
        .output()
        .map_err(|e| format!("failed to execute TT-Metalium compiler: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "TT-Metalium compilation failed for kernel '{kernel_name}': {stderr}"
        ));
    }

    // The compiled ELF lands at `<temp_dir>/<kernel_name>.elf`
    let elf_path = temp_dir.join(format!("{kernel_name}.elf"));
    let binary_bytes = std::fs::read(&elf_path)
        .map_err(|e| format!("compiled ELF not found at {elf_path:?}: {e}"))?;

    Ok(TtBinary {
        kernel_name: kernel_name.to_string(),
        data: binary_bytes,
        entry_point: kernel_name.to_string(),
        architecture: "wormhole_b0".to_string(),
    })
}

/// Check whether TT-Metalium is installed on the system.
///
/// Returns `Ok(())` if the compiler script is reachable, or a descriptive
/// `Err(String)` explaining what is missing.
pub fn check_installation() -> Result<(), String> {
    resolve_metalium_root()?;
    // Verify the Python package is importable.
    let check = Command::new("python3")
        .arg("-c")
        .arg("import tt_metal; print(tt_metal.__file__)")
        .output()
        .map_err(|e| format!("failed to invoke python3: {e}"))?;

    if !check.status.success() {
        let stderr = String::from_utf8_lossy(&check.stderr);
        return Err(format!(
            "TT-Metalium Python package not found: {stderr}\n\
             Install via: pip install tt-metalium\n\
             Or set TT_METALIUM_ROOT to the TT-Metalium source tree."
        ));
    }
    Ok(())
}

/// Resolve the TT-Metalium root directory, checking env var and default path.
fn resolve_metalium_root() -> Result<PathBuf, String> {
    // Check environment variable first.
    if let Ok(root) = std::env::var("TT_METALIUM_ROOT") {
        let path = PathBuf::from(&root);
        if path
            .join("tt_metal")
            .join("tools")
            .join("compile_kernel.py")
            .exists()
        {
            return Ok(path);
        }
        return Err(format!(
            "$TT_METALIUM_ROOT={root} does not contain a valid TT-Metalium installation \
             (expected {}/tt_metal/tools/compile_kernel.py)",
            path.display()
        ));
    }

    // Fall back to the default path.
    let default = PathBuf::from(DEFAULT_TT_METALIUM_ROOT);
    if default
        .join("tt_metal")
        .join("tools")
        .join("compile_kernel.py")
        .exists()
    {
        return Ok(default);
    }

    Err(format!(
        "TT-Metalium not found.\n\
         Set TT_METALIUM_ROOT to the installation path, or install from:\n\
         https://github.com/tenstorrent/tt-metalium\n\
         Looked at: $TT_METALIUM_ROOT and {DEFAULT_TT_METALIUM_ROOT}/tt_metal/tools/compile_kernel.py"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installation_check_graceful_missing() {
        // On systems without TT-Metalium, this should return Err with a helpful message,
        // not panic.
        match check_installation() {
            Ok(()) => {
                // If it IS installed, that's fine — no assertion.
            }
            Err(msg) => {
                assert!(
                    msg.contains("TT-Metalium"),
                    "expected helpful error message about missing TT-Metalium, got: {msg}"
                );
                assert!(
                    msg.contains("install") || msg.contains("not found"),
                    "expected install guidance in error: {msg}"
                );
            }
        }
    }

    #[test]
    fn resolve_root_graceful_missing() {
        // Clear any env override for the test.
        let original = std::env::var("TT_METALIUM_ROOT").ok();
        // Unset for the test scope (this only affects the current process — fine).
        std::env::remove_var("TT_METALIUM_ROOT");

        // If TT-Metalium isn't actually installed, we get an error.
        match resolve_metalium_root() {
            Ok(path) => {
                assert!(path.exists(), "resolved path {path:?} should exist");
            }
            Err(msg) => {
                assert!(
                    msg.contains("TT-Metalium not found"),
                    "expected 'TT-Metalium not found' error, got: {msg}"
                );
            }
        }

        // Restore the env var if it was set.
        if let Some(val) = original {
            std::env::set_var("TT_METALIUM_ROOT", val);
        }
    }
}
