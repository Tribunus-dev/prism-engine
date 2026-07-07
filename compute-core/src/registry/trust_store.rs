use crate::registry::types::*;
use p256::ecdsa::{Signature, VerifyingKey};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub struct TrustStore {
    pub trusted_providers: HashMap<ProviderIdentity, VerifyingKey>,
}

impl TrustStore {
    pub fn new() -> Self {
        Self {
            trusted_providers: HashMap::new(),
        }
    }

    pub fn register_provider(&mut self, provider: ProviderIdentity, key: VerifyingKey) {
        self.trusted_providers.insert(provider, key);
    }

    pub fn verify_manifest(
        &self,
        manifest: &ComputeImageManifest,
        _raw_manifest_bytes: &[u8],
    ) -> Result<(), DeploymentError> {
        // 1. Look up provider in trusted store
        let provider_key = self
            .trusted_providers
            .get(&manifest.provider)
            .ok_or(DeploymentError::UntrustedProvider)?;

        // 2. Compute domain-separated manifest signature target
        let canonical_manifest = manifest
            .to_canonical_bytes()
            .map_err(|e| DeploymentError::SerializationError(e.to_string()))?;
        let mut hasher = Sha256::new();
        hasher.update(b"prism.manifest-signature.v1\0");
        hasher.update(manifest.artifact_digest.as_bytes());
        hasher.update(&canonical_manifest);
        let digest = hasher.finalize();

        // 3. Parse signature and verify using DigetVerifier
        let sig = Signature::from_der(&manifest.artifact_signature.bytes)
            .map_err(|_| DeploymentError::InvalidManifestSignature)?;
        use p256::ecdsa::signature::DigestVerifier;
        // Use digest verifier since we already have the SHA-256 digest
        provider_key
            .verify_digest(Sha256::new_with_prefix(&digest), &sig)
            .map_err(|_| DeploymentError::InvalidManifestSignature)
    }
}

impl Default for TrustStore {
    fn default() -> Self {
        Self::new()
    }
}
