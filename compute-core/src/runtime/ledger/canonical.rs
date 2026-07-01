use crate::runtime::ledger::entry::DeterministicReceiptPayload;
use crate::runtime::ledger::error::CanonicalEncodingError;

pub trait CanonicalReceiptEncoder {
    fn encode(
        &self,
        payload: &DeterministicReceiptPayload,
    ) -> Result<Vec<u8>, CanonicalEncodingError>;
}

pub struct JcsReceiptEncoder;

impl CanonicalReceiptEncoder for JcsReceiptEncoder {
    fn encode(
        &self,
        payload: &DeterministicReceiptPayload,
    ) -> Result<Vec<u8>, CanonicalEncodingError> {
        serde_json_canonicalizer::to_vec(payload)
            .map_err(|e| CanonicalEncodingError::Serialization(e.to_string()))
    }
}
