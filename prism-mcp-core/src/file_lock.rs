use anyhow::{Context, Result};
use nix::fcntl::{Flock, FlockArg};
use std::path::{Path, PathBuf};

/// An exclusive file lock backed by `Flock`.
/// Dropping the guard releases the lock.
pub struct FileLockGuard {
    _lock: Flock<std::fs::File>,
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
        let lock = Flock::lock(file, FlockArg::LockExclusive)
            .map_err(|(_file, e)| anyhow::Error::from(e))?;
        Ok(FileLockGuard { _lock: lock })
    }

    pub fn try_lock(&self) -> Result<Option<FileLockGuard>> {
        std::fs::create_dir_all(self.path.parent().unwrap())?;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&self.path)
            .with_context(|| format!("opening lock file {}", self.path.display()))?;
        match Flock::lock(file, FlockArg::LockExclusiveNonblock) {
            Ok(lock) => Ok(Some(FileLockGuard { _lock: lock })),
            Err((_file, nix::errno::Errno::EAGAIN)) => Ok(None),
            Err((_file, e)) => Err(e).context("flock(LOCK_EX | LOCK_NB)"),
        }
    }
}
