use crate::linux::backend::BackendKind;
use crate::linux::errors::BackendError;

pub struct EventHandle {
    pub backend: BackendKind,
    pub opaque_id: u64,
}

pub enum EventStatus {
    Pending,
    Complete,
    Failed(BackendError),
}
