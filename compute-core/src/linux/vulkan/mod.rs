use crate::linux::backend::{BackendKind, DeviceId};
use crate::linux::capability::{BackendAvailability, DeviceCapabilities};
use crate::linux::device::{DeviceDescriptor, LinuxDeviceBackend};
use crate::linux::errors::BackendError;
use crate::linux::memory::{AllocationRequest, DeviceBuffer};
use crate::linux::queue::{QueueClass, QueueHandle};
use crate::linux::submission::{Submission, SubmissionHandle, SubmissionStatus};

pub struct VulkanComputeBackend;

impl LinuxDeviceBackend for VulkanComputeBackend {
    fn backend_kind(&self) -> BackendKind { BackendKind::Vulkan }
    fn enumerate_devices(&self) -> Result<Vec<DeviceDescriptor>, BackendError> { Ok(vec![]) }
    fn probe_capabilities(&self, _device: &DeviceId) -> Result<DeviceCapabilities, BackendError> { Ok(DeviceCapabilities { backend: BackendKind::Vulkan, vendor: crate::linux::backend::VendorKind::Unknown, device_name: "Stub".into(), driver_version: None, architecture: None, device_memory_bytes: 0, host_visible_memory: false, unified_addressing: false, managed_memory: false, peer_access: false, async_copy: false, events: false, command_graphs: false, cooperative_launch: false, fp16: false, bf16: false, int8: false, int4: false, subgroup_widths: vec![], max_workgroup_size: 0, max_shared_memory_bytes: 0, max_allocation_bytes: 0, supports_timestamps: false, supports_profiling: false, supports_external_memory: false, availability: BackendAvailability::RuntimeLibraryMissing }) }
    fn create_queue(&self, _device: &DeviceId, _class: QueueClass) -> Result<QueueHandle, BackendError> { Err(BackendError::UnsupportedOperation("Vulkan stub".into())) }
    fn allocate(&self, _device: &DeviceId, _request: AllocationRequest) -> Result<DeviceBuffer, BackendError> { Err(BackendError::UnsupportedOperation("Vulkan stub".into())) }
    fn submit(&self, _queue: &QueueHandle, _submission: Submission) -> Result<SubmissionHandle, BackendError> { Err(BackendError::UnsupportedOperation("Vulkan stub".into())) }
    fn poll(&self, _submission: &SubmissionHandle) -> Result<SubmissionStatus, BackendError> { Ok(SubmissionStatus::Failed) }
    fn synchronize(&self, _submission: &SubmissionHandle) -> Result<(), BackendError> { Err(BackendError::UnsupportedOperation("stub".into())) }
    fn release(&self, _buffer: crate::linux::memory::BufferHandle) -> Result<(), BackendError> { Err(BackendError::UnsupportedOperation("stub".into())) }
    fn create_event(&self, _device: &DeviceId) -> Result<crate::linux::event::EventHandle, BackendError> { Err(BackendError::UnsupportedOperation("stub".into())) }
    fn record_event(&self, _queue: &QueueHandle, _event: &crate::linux::event::EventHandle) -> Result<(), BackendError> { Err(BackendError::UnsupportedOperation("stub".into())) }
    fn wait_event(&self, _queue: &QueueHandle, _event: &crate::linux::event::EventHandle) -> Result<(), BackendError> { Err(BackendError::UnsupportedOperation("stub".into())) }
}
