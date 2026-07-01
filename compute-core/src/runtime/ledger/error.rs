use thiserror::Error;

#[derive(Debug, Error)]
pub enum LedgerExportError {
    #[error("failed to serialize transition receipt")]
    Serialization(#[source] serde_json::Error),
}

#[derive(Debug, Error)]
pub enum CanonicalEncodingError {
    #[error("receipt payload violates canonical JSON constraints")]
    InvalidPayload,
    #[error("JCS serialization failed: {0}")]
    Serialization(String),
}

#[derive(Debug, Error)]
pub enum ReceiptDigestError {
    #[error(transparent)]
    CanonicalEncoding(#[from] CanonicalEncodingError),
    #[error("unsupported receipt digest algorithm")]
    UnsupportedAlgorithm { algorithm: String },
}

#[derive(Debug, Error)]
pub enum LedgerProjectionError {
    #[error("command emitted by {system_id:?} has no semantic receipt projection")]
    MissingSemanticProjection {
        system_id: crate::runtime::scheduling::metadata::SystemId,
    },
    #[error("receipt command payload violated semantic receipt invariants")]
    InvalidSemanticPayload,
}
