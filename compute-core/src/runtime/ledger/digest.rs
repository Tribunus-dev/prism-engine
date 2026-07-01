use serde::{Deserialize, Serialize};

use crate::runtime::ledger::canonical::{CanonicalReceiptEncoder, JcsReceiptEncoder};
use crate::runtime::ledger::entry::DeterministicReceiptPayload;
use crate::runtime::ledger::error::ReceiptDigestError;

pub const TRANSITION_RECEIPT_DIGEST_ALGORITHM: &str = "blake3-256-jcs-rfc8785";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptDigest {
    pub algorithm: String,
    pub hex: String,
}

pub struct ReceiptHasher<E = JcsReceiptEncoder> {
    encoder: E,
}

impl ReceiptHasher<JcsReceiptEncoder> {
    pub fn production() -> Self {
        Self {
            encoder: JcsReceiptEncoder,
        }
    }
}

impl<E: CanonicalReceiptEncoder> ReceiptHasher<E> {
    pub fn compute(
        &self,
        payload: &DeterministicReceiptPayload,
    ) -> Result<ReceiptDigest, ReceiptDigestError> {
        let canonical_bytes = self.encoder.encode(payload)?;
        let hash = blake3::hash(&canonical_bytes);
        Ok(ReceiptDigest {
            algorithm: TRANSITION_RECEIPT_DIGEST_ALGORITHM.to_string(),
            hex: hash.to_hex().to_string(),
        })
    }

    pub fn verify(
        &self,
        payload: &DeterministicReceiptPayload,
        expected: &ReceiptDigest,
    ) -> Result<bool, ReceiptDigestError> {
        if expected.algorithm != TRANSITION_RECEIPT_DIGEST_ALGORITHM {
            return Err(ReceiptDigestError::UnsupportedAlgorithm {
                algorithm: expected.algorithm.clone(),
            });
        }
        let actual = self.compute(payload)?;
        Ok(actual.hex == expected.hex)
    }
}
