//! Executable seal — Merkle integrity verification.

use serde::{Deserialize, Serialize};

use crate::compute_image_runtime::ContentHash;

/// Merkle-based seal for a sealed executable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableSeal {
    /// Root hash of the Merkle tree.
    pub root_hash: ContentHash,
    /// Hash of the manifest segment.
    pub manifest_hash: ContentHash,
    /// Hashes of the per-profile segments.
    pub profile_hashes: Vec<ContentHash>,
    /// Hash of the receipt bundle.
    pub receipt_bundle_hash: ContentHash,
    /// Optional cryptographic signature.
    pub signature: Option<ExecutableSignature>,
}

/// Cryptographic signature over the seal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableSignature {
    /// Raw signature bytes.
    pub signature_bytes: Vec<u8>,
    /// Identity of the signer.
    pub signer_identity: String,
    /// Signature scheme (e.g., "ed25519").
    pub signature_scheme: String,
}
