use crate::linux::backend::{BackendKind, DeviceId};
use crate::linux::capability::DeviceCapabilities;
use crate::linux::errors::BackendError;
use crate::linux::memory::{AllocationRequest, DeviceBuffer};
use crate::linux::queue::{QueueClass, QueueHandle};
use crate::linux::submission::{Submission, SubmissionHandle, SubmissionStatus};

#[derive(Debug, Clone)]
pub struct DeviceDescriptor {
    pub id: DeviceId,
    pub capabilities: DeviceCapabilities,
}

#[derive(Debug, Clone)]
pub struct BackendUnavailableReceipt {
    pub backend: BackendKind,
    pub reason: String,
}

pub struct LinuxDeviceInventory {
    pub generated_at_unix_ms: u64,
    pub devices: Vec<DeviceDescriptor>,
    pub unavailable_backends: Vec<BackendUnavailableReceipt>,
}

pub trait LinuxDeviceBackend: Send + Sync {
    fn backend_kind(&self) -> BackendKind;

    fn enumerate_devices(&self) -> Result<Vec<DeviceDescriptor>, BackendError>;

    fn probe_capabilities(
        &self,
        device: &DeviceId,
    ) -> Result<DeviceCapabilities, BackendError>;

    fn create_queue(
        &self,
        device: &DeviceId,
        class: QueueClass,
    ) -> Result<QueueHandle, BackendError>;

    fn allocate(
        &self,
        device: &DeviceId,
        request: AllocationRequest,
    ) -> Result<DeviceBuffer, BackendError>;

    fn release(
        &self,
        buffer: crate::linux::memory::BufferHandle,
    ) -> Result<(), BackendError>;

    fn submit(
        &self,
        queue: &QueueHandle,
        submission: Submission,
    ) -> Result<SubmissionHandle, BackendError>;

    fn poll(
        &self,
        submission: &SubmissionHandle,
    ) -> Result<SubmissionStatus, BackendError>;

    fn synchronize(
        &self,
        submission: &SubmissionHandle,
    ) -> Result<(), BackendError>;

    fn create_event(&self, device: &DeviceId) -> Result<crate::linux::event::EventHandle, BackendError>;
    fn record_event(&self, queue: &QueueHandle, event: &crate::linux::event::EventHandle) -> Result<(), BackendError>;
    fn wait_event(&self, queue: &QueueHandle, event: &crate::linux::event::EventHandle) -> Result<(), BackendError>;
}
