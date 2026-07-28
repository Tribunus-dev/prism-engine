//! Breadcrumb writer for Core ML predict crash localization.
//!
//! Writes append-only, fsynced breadcrumbs to a file path specified by the
//! `CML_BREADCRUMB_PATH` environment variable. If the env var is not set,
//! writes are silently skipped (breadcrumbs are only needed in the child
//! subprocess where crashes can occur).
//!
//! The last completed breadcrumb survives a child crash because each write
//! is flushed and synced. The parent process reads the breadcrumb file
//! after the child exits to determine the terminal phase.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Process-wide breadcrumb path override. When set, `write_breadcrumb`
/// uses this path instead of reading the `CML_BREADCRUMB_PATH` env var,
/// avoiding races between parallel tests on the process-level env var.
static BREADCRUMB_PATH_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();

/// Write a breadcrumb to the breadcrumb file.
///
/// The file path is resolved in this order:
/// 1. The process-wide override set by [`set_breadcrumb_path`].
/// 2. The `CML_BREADCRUMB_PATH` environment variable.
///
/// Each breadcrumb is a line in the file: `{breadcrumb_name}\n`.
/// The file is flushed and fsynced after each write so the last
/// completed breadcrumb survives a child crash. If no path is
/// configured, the call is silently skipped.
pub fn write_breadcrumb(name: &str) {
    let Some(path) = BREADCRUMB_PATH_OVERRIDE
        .get()
        .cloned()
        .or_else(|| std::env::var("CML_BREADCRUMB_PATH").map(PathBuf::from).ok())
    else {
        return;
    };
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{name}");
        let _ = f.flush();
        // fsync on macOS: use File::sync_all
        let _ = f.sync_all();
    }
}

/// Set the breadcrumb file path for the current process.
///
/// Idempotent: the first call wins. Subsequent calls are ignored so
/// that a misconfigured child process cannot clobber a parent's
/// already-set path. Tests that need to reset the path should fork
/// a new process; in-process tests can use the `CML_BREADCRUMB_PATH`
/// env var or fork via `cargo test --test-threads=1`.
pub fn set_breadcrumb_path(path: &Path) {
    let _ = BREADCRUMB_PATH_OVERRIDE.set(path.to_path_buf());
}

/// Read all breadcrumbs from a breadcrumb file.
/// Returns an empty vec if the file doesn't exist or can't be read.
pub fn read_breadcrumbs(path: &Path) -> Vec<String> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    content.lines().map(|l| l.to_string()).collect()
}

/// Get the last completed breadcrumb name, or None.
pub fn last_breadcrumb(path: &Path) -> Option<String> {
    let crumbs = read_breadcrumbs(path);
    crumbs.last().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breadcrumb_write_and_read() {
        // Use a process-wide unique temp path so we never collide with
        // sibling tests; the breadcrumb writer either reads the override
        // or falls back to CML_BREADCRUMB_PATH, both of which point here.
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("breadcrumb_test_{pid}_{nanos}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("crumbs.txt");

        // The breadcrumb writer reads CML_BREADCRUMB_PATH as a
        // process-level env var. Direct it to a unique test path.
        // In Rust 2021 this call is safe; in 2024 it became unsafe,
        // but the workspace pins to 2021.
        std::env::set_var("CML_BREADCRUMB_PATH", &path);
        write_breadcrumb("phase_1");
        write_breadcrumb("phase_2");

        let crumbs = read_breadcrumbs(&path);
        assert_eq!(crumbs, vec!["phase_1", "phase_2"]);

        let last = last_breadcrumb(&path);
        assert_eq!(last, Some("phase_2".into()));

        std::env::remove_var("CML_BREADCRUMB_PATH");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_env_var_skips_write() {
        // If CML_BREADCRUMB_PATH is not set, write_breadcrumb should not panic.
        std::env::remove_var("CML_BREADCRUMB_PATH");
        write_breadcrumb("should_not_crash");
    }

    #[test]
    fn read_nonexistent_returns_empty() {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("breadcrumb_test_{pid}_{nanos}_read"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let crumbs = read_breadcrumbs(&dir.join("nope.txt"));
        assert!(crumbs.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }
}
