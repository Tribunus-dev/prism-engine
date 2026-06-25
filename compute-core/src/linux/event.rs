use crate::linux::backend::BackendKind;
use crate::linux::errors::BackendError;

use crate::linux::backend::RuntimeResourceId;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EventHandle {
    pub id: RuntimeResourceId,
}

pub enum EventStatus {
    Pending,
    Complete,
    Failed(BackendError),
}
