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
use std::path::Path;

#[cfg(test)]
use parking_lot::Mutex as ParkingMutex;
/// Test-only global breadcrumb path override.
/// When set, write_breadcrumb uses this path instead of reading CML_BREADCRUMB_PATH,
/// avoiding races between parallel tests on the process-level env var.
#[cfg(test)]
use std::sync::LazyLock;

#[cfg(test)]
static TEST_BREADCRUMB_PATH: LazyLock<ParkingMutex<Option<std::path::PathBuf>>> =
    LazyLock::new(|| ParkingMutex::new(None));

/// Write a breadcrumb to the breadcrumb file.
///
/// The file path is read from `CML_BREADCRUMB_PATH` env var.
/// Each breadcrumb is a line in the file: `{breadcrumb_name}\n`.
/// The file is flushed and fsynced after each write so the last
/// completed breadcrumb survives a child crash.
pub fn write_breadcrumb(name: &str) {
    #[cfg(test)]
    let path = {
        let guard = TEST_BREADCRUMB_PATH.lock();
        guard.clone()
    };
    #[cfg(not(test))]
    let path = None;
    let path = path.or_else(|| {
        std::env::var("CML_BREADCRUMB_PATH")
            .map(std::path::PathBuf::from)
            .ok()
    });
    let path = match path {
        Some(p) => p,
        None => return,
    };
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{name}");
        let _ = f.flush();
        // fsync on macOS: use File::sync_all
        let _ = f.sync_all();
    }
}

/// Set the breadcrumb file path for the current process.
pub fn set_breadcrumb_path(path: &Path) {
    #[cfg(test)]
    {
        *TEST_BREADCRUMB_PATH.lock() = Some(path.to_path_buf());
    }
    #[cfg(not(test))]
    {
        let _ = path;
        unsafe {
            std::env::set_var("CML_BREADCRUMB_PATH", path);
        }
    }
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

    use std::sync::atomic::{AtomicU32, Ordering};
    static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

    /// Return a unique temp directory for this test invocation.
    /// Avoids filesystem collisions when tests run in parallel via `--lib`.
    fn unique_test_dir() -> std::path::PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("breadcrumb_test_{id}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn breadcrumb_write_and_read() {
        let dir = unique_test_dir();
        let path = dir.join("crumbs.txt");

        set_breadcrumb_path(&path);
        write_breadcrumb("phase_1");
        write_breadcrumb("phase_2");

        let crumbs = read_breadcrumbs(&path);
        assert_eq!(crumbs, vec!["phase_1", "phase_2"]);

        let last = last_breadcrumb(&path);
        assert_eq!(last, Some("phase_2".into()));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_env_var_skips_write() {
        // If CML_BREADCRUMB_PATH is not set, write_breadcrumb should not panic.
        // Clear the test-global path first so we don't race with
        // breadcrumb_write_and_read (which sets TEST_BREADCRUMB_PATH).
        *TEST_BREADCRUMB_PATH.lock() = None;
        unsafe {
            std::env::remove_var("CML_BREADCRUMB_PATH");
        }
        write_breadcrumb("should_not_crash");
    }

    #[test]
    fn read_nonexistent_returns_empty() {
        let dir = unique_test_dir();
        let crumbs = read_breadcrumbs(&dir.join("nope.txt"));
        assert!(crumbs.is_empty());
    }
}
