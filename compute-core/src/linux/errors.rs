use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    AllocationFailed(String),
    TransferFailed(String),
    SubmissionFailed(String),
    DeviceLost(String),
    UnsupportedOperation(String),
    NotReady,
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::AllocationFailed(msg) => write!(f, "Allocation Failed: {}", msg),
            BackendError::TransferFailed(msg) => write!(f, "Transfer Failed: {}", msg),
            BackendError::SubmissionFailed(msg) => write!(f, "Submission Failed: {}", msg),
            BackendError::DeviceLost(msg) => write!(f, "Device Lost: {}", msg),
            BackendError::UnsupportedOperation(msg) => write!(f, "Unsupported Operation: {}", msg),
            BackendError::NotReady => write!(f, "Backend Not Ready"),
        }
    }
}

impl std::error::Error for BackendError {}
