use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub mod text_architecture_extract;
#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("artifact is not sealed")]
    Unsealed,
    #[error("digest mismatch")]
    DigestMismatch,
    #[error("serialization: {0}")]
    Serialization(String),
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactManifest {
    pub name: String,
    pub version: String,
    pub payload_digest: String,
    pub sealed: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Artifact {
    pub manifest: ArtifactManifest,
    pub payload: Vec<u8>,
}
impl Artifact {
    pub fn new(name: impl Into<String>, version: impl Into<String>, payload: Vec<u8>) -> Self {
        let d = hex::encode(Sha256::digest(&payload));
        Self {
            manifest: ArtifactManifest {
                name: name.into(),
                version: version.into(),
                payload_digest: d,
                sealed: false,
            },
            payload,
        }
    }
    pub fn seal(&mut self) {
        self.manifest.sealed = true
    }
    pub fn validate(&self) -> Result<(), ArtifactError> {
        if !self.manifest.sealed {
            return Err(ArtifactError::Unsealed);
        }
        if hex::encode(Sha256::digest(&self.payload)) != self.manifest.payload_digest {
            return Err(ArtifactError::DigestMismatch);
        }
        Ok(())
    }
}
