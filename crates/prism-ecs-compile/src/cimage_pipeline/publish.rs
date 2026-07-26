//! CImage publishing — copy a staging CImage directory to a canonical
//! destination.
//!
//! This module owns the canonical authority for the publishing step:
//! the `publish_image` function that copies a staging CImage directory
//! to a canonical destination. The publish step is *not* a world
//! mutation; it is an effect on the artifact store that the runtime
//! observes when it reads the canonical destination.
//!
//! The function is the *boundary* between the compile pipeline and
//! the canonical artifact store. A failed publish leaves the staging
//! directory in place and surfaces the error; a successful publish
//! removes the staging directory and surfaces the destination path.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::CImagePipelineError;
use super::CImagePipelineResult;

/// Errors raised by the publish step.
#[derive(Debug, Error)]
pub enum PublishError {
    /// Staging directory does not exist.
    #[error("staging directory does not exist: {0}")]
    StagingMissing(String),
    /// Manifest missing from the staging directory.
    #[error("manifest.json is missing from the staging directory")]
    ManifestMissing,
    /// Destination is not a directory.
    #[error("destination is not a directory: {0}")]
    DestinationInvalid(String),
    /// I/O error during copy.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Publish policy — controls whether the staging directory is
/// removed after a successful copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PublishPolicy {
    /// Copy then remove the staging directory.
    Move,
    /// Copy and leave the staging directory in place.
    Copy,
}

impl Default for PublishPolicy {
    fn default() -> Self {
        Self::Move
    }
}

/// Publish a staging CImage directory to a canonical destination.
pub fn publish_image(staging: &Path, destination: &Path) -> CImagePipelineResult<()> {
    publish_image_with_policy(staging, destination, PublishPolicy::default())
}

/// Publish a staging CImage directory to a canonical destination with
/// a specific policy.
pub fn publish_image_with_policy(
    staging: &Path,
    destination: &Path,
    policy: PublishPolicy,
) -> CImagePipelineResult<()> {
    if !staging.exists() {
        return Err(PublishError::StagingMissing(staging.display().to_string()).into());
    }
    let manifest = staging.join("manifest.json");
    if !manifest.exists() {
        return Err(PublishError::ManifestMissing.into());
    }
    if destination.exists() && !destination.is_dir() {
        return Err(PublishError::DestinationInvalid(destination.display().to_string()).into());
    }
    fs::create_dir_all(destination).map_err(|e| {
        CImagePipelineError::failed(format!("create destination: {e}"))
    })?;
    copy_dir_recursive(staging, destination)?;
    if policy == PublishPolicy::Move {
        fs::remove_dir_all(staging).map_err(|e| {
            CImagePipelineError::failed(format!("remove staging: {e}"))
        })?;
    }
    Ok(())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> CImagePipelineResult<()> {
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            fs::create_dir_all(&dest_path)?;
            copy_dir_recursive(&entry.path(), &dest_path)?;
        } else {
            fs::copy(entry.path(), &dest_path)?;
        }
    }
    Ok(())
}
