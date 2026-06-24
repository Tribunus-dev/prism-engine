//! Release engineering types for the Prism Engine.
//!
//! Provides channel classification, signing status, platform targeting,
//! checksum tracking, full release manifests with compatibility digests,
//! and versioned install-directory management.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

// ── Release channel ───────────────────────────────────────────────────

/// Which update track a release belongs to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReleaseChannel {
    /// Cutting-edge builds pushed on every commit.
    Nightly,
    /// Feature previews that have passed smoke tests but not full
    /// qualification.
    Experimental,
    /// Fully qualified, signed production releases.
    Stable,
}

// ── Signing / integrity status ───────────────────────────────────────

/// How the release's integrity was verified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SigningStatus {
    /// Local development build; no signing performed.
    UnsignedDevelopmentBuild,
    /// Checksums computed and verified against the build manifest but
    /// not cryptographically signed.
    ChecksumVerifiedBuild,
    /// Full cryptographic signature from a trusted release key.
    SignedReleaseBuild,
}

// ── Platform targeting ───────────────────────────────────────────────

/// Describes a single OS+architecture tuple a release supports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformTarget {
    pub os: String,
    pub arch: String,
    /// Minimum macOS version (e.g. "14.0") — `None` on non-macOS
    /// platforms.
    pub min_macos_version: Option<String>,
}

// ── Checksums ────────────────────────────────────────────────────────

/// A single file-level checksum entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseChecksum {
    /// Relative path within the release archive.
    pub path: String,
    /// Hash algorithm (e.g. "sha256", "blake3").
    pub algorithm: String,
    /// Hex-encoded digest.
    pub hex_digest: String,
}

// ── Artifact digest (simple newtype) ──────────────────────────────────

/// Opaque hex digest identifying a build artifact or manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactDigest(pub String);

// ── Release manifest ─────────────────────────────────────────────────

/// Full metadata for a single Prism Engine release.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismReleaseManifest {
    /// Semver of the release itself (e.g. "0.3.0").
    pub release_version: String,
    /// Which channel this build belongs to.
    pub channel: ReleaseChannel,
    /// Version of the `prism` CLI binary.
    pub prism_version: String,
    /// Version of the `tribunus-compute-core` library.
    pub compute_core_version: String,
    /// Git commit SHA the release was built from.
    pub build_commit: String,
    /// ISO-8601 timestamp of the build.
    pub build_timestamp: String,
    /// Platform targets this manifest covers.
    pub supported_platforms: Vec<PlatformTarget>,
    /// Digest of the full compatibility manifest, if one was generated.
    pub compatibility_manifest_digest: Option<ArtifactDigest>,
    /// Schema versions understood by this release's artifacts.
    pub artifact_schema_versions: Vec<u32>,
    /// Per-file checksums for integrity verification.
    pub checksums: Vec<ReleaseChecksum>,
    /// How this build was signed / verified.
    pub signing_status: SigningStatus,
}

// ── Versioned install directory helper ───────────────────────────────

/// Helpers for managing versioned release directories and the `current` /
/// `previous` symlinks under `~/.local/share/prism/`.
pub struct VersionedInstallDir;

impl VersionedInstallDir {
    /// Base directory for all release artifacts.
    ///
    /// `~/.local/share/prism/releases/`
    pub fn releases_dir() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("prism")
            .join("releases")
    }

    /// Symlink pointing at the *currently active* release directory.
    ///
    /// `~/.local/share/prism/current`
    pub fn current_link() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(home).join(".local").join("share").join("prism").join("current")
    }

    /// Symlink pointing at the *previously active* release directory.
    ///
    /// `~/.local/share/prism/previous`
    pub fn previous_link() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(home).join(".local").join("share").join("prism").join("previous")
    }

    /// Create a versioned release directory at
    /// `releases_dir()/<version>/` (including parents).
    ///
    /// Returns the created path, or an error string if the filesystem
    /// operation failed.
    pub fn stage_release(version: &str) -> Result<PathBuf, String> {
        let dir = Self::releases_dir().join(version);
        fs::create_dir_all(&dir).map_err(|e| format!("failed to create {dir:?}: {e}"))?;
        Ok(dir)
    }

    /// Atomically swap the `current` symlink to point at
    /// `releases_dir()/<version>/`.
    ///
    /// The previous `current` target becomes the `previous` symlink, so
    /// `rollback()` can restore it.
    pub fn activate(version: &str) -> Result<(), String> {
        let target = Self::releases_dir().join(version);
        if !target.is_dir() {
            return Err(format!(
                "release directory does not exist: {:?}",
                target
            ));
        }

        let current = Self::current_link();
        let previous = Self::previous_link();

        // If current exists, demote it to previous.
        if current.is_symlink() || current.exists() {
            let _ = fs::remove_file(&previous);
            let _ = fs::rename(&current, &previous);
        }

        // Create / update current -> version.
        let _ = fs::remove_file(&current);
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&target, &current)
                .map_err(|e| format!("failed to symlink {:?} -> {:?}: {e}", current, target))?;
        }
        #[cfg(not(unix))]
        {
            // Fallback: copy so we can still read it later.
            if let Err(e) = fs::copy(&target.join("release.json"), &current.join("release.json"))
            {
                return Err(format!("failed to stage release reference: {e}"));
            }
        }

        Ok(())
    }

    /// Swap the `current` and `previous` symlinks so the previous
    /// release is re-activated.
    pub fn rollback() -> Result<(), String> {
        let current = Self::current_link();
        let previous = Self::previous_link();

        if !previous.is_symlink() && !previous.exists() {
            return Err("no previous release to roll back to".into());
        }

        let tmp = Self::releases_dir().join(".rollback-tmp");
        let _ = fs::remove_file(&tmp);

        // current -> tmp
        fs::rename(&current, &tmp)
            .map_err(|e| format!("failed to swap current aside: {e}"))?;

        // previous -> current
        fs::rename(&previous, &current)
            .map_err(|e| format!("failed to promote previous to current: {e}"))?;

        // tmp -> previous
        fs::rename(&tmp, &previous)
            .map_err(|e| format!("failed to demote former current to previous: {e}"))?;

        Ok(())
    }
}
