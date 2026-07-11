use anyhow::{Context, Result};
use nix::fcntl::{flock, FlockArg};
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

/// An exclusive file lock backed by `flock(LOCK_EX)`.
/// Dropping the guard releases the lock.
pub struct FileLockGuard {
    _file: File,
}

impl Drop for FileLockGuard {
    fn drop(&mut self) {
        let _ = flock(self._file.as_raw_fd(), FlockArg::Unlock);
    }
}

/// Cross-process advisory file lock coordinator.
pub struct FileLock {
    path: PathBuf,
}

impl FileLock {
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_owned(),
        }
    }

    pub fn lock(&self) -> Result<FileLockGuard> {
        std::fs::create_dir_all(self.path.parent().unwrap())?;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&self.path)
            .with_context(|| format!("opening lock file {}", self.path.display()))?;
        flock(file.as_raw_fd(), FlockArg::LockExclusive)?;
        Ok(FileLockGuard { _file: file })
    }

    pub fn try_lock(&self) -> Result<Option<FileLockGuard>> {
        std::fs::create_dir_all(self.path.parent().unwrap())?;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&self.path)
            .with_context(|| format!("opening lock file {}", self.path.display()))?;
        match flock(file.as_raw_fd(), FlockArg::LockExclusiveNonblock) {
            Ok(()) => Ok(Some(FileLockGuard { _file: file })),
            Err(nix::errno::Errno::EAGAIN) => Ok(None),
            Err(e) => Err(e).context("flock(LOCK_EX | LOCK_NB)"),
        }
    }
}
