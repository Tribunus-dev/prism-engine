use crate::linux::backend::{BackendKind, DeviceId, DeviceStableKey, VendorKind};
use crate::linux::capability::{BackendAvailability, DeviceCapabilities};
use crate::linux::device::{DeviceDescriptor, LinuxDeviceBackend};
use crate::linux::errors::BackendError;
use crate::linux::memory::{AllocationRequest, BufferOwnership, DeviceBuffer, MemoryKind};
use crate::linux::queue::{QueueClass, QueueHandle};
use crate::linux::submission::{Submission, SubmissionHandle, SubmissionStatus};

pub struct CpuBackend;

impl LinuxDeviceBackend for CpuBackend {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn enumerate_devices(&self) -> Result<Vec<DeviceDescriptor>, BackendError> {
        let stable_key = DeviceStableKey {
            vendor: VendorKind::Cpu,
            pci_domain: None,
            pci_bus: None,
            pci_device: None,
            uuid_or_luid: None,
            fallback_fingerprint: 0,
        };

        let device_id = DeviceId {
            backend: BackendKind::Cpu,
            ordinal: 0,
            stable_key,
        };

        let capabilities = DeviceCapabilities {
            backend: BackendKind::Cpu,
            vendor: VendorKind::Cpu,
            device_name: "Generic CPU".to_string(),
            driver_version: None,
            architecture: None,
            device_memory_bytes: 0, // OS managed
            host_visible_memory: true,
            unified_addressing: true,
            managed_memory: false,
            peer_access: false,
            async_copy: false,
            events: false,
            command_graphs: false,
            cooperative_launch: false,
            fp16: false,
            bf16: false,
            int8: false,
            int4: false,
            subgroup_widths: vec![],
            max_workgroup_size: 1,
            max_shared_memory_bytes: 0,
            max_allocation_bytes: u64::MAX,
            supports_timestamps: true,
            supports_profiling: true,
            supports_external_memory: false,
            availability: BackendAvailability::Available,
        };

        Ok(vec![DeviceDescriptor {
            id: device_id,
            capabilities,
        }])
    }

    fn probe_capabilities(
        &self,
        device: &DeviceId,
    ) -> Result<DeviceCapabilities, BackendError> {
        let devices = self.enumerate_devices()?;
        devices
            .into_iter()
            .find(|d| d.id == *device)
            .map(|d| d.capabilities)
            .ok_or(BackendError::DeviceLost("CPU not found".into()))
    }

    fn create_queue(
        &self,
        _device: &DeviceId,
        _class: QueueClass,
    ) -> Result<QueueHandle, BackendError> {
        Ok(QueueHandle { opaque_id: 0 })
    }

    fn allocate(
        &self,
        device: &DeviceId,
        request: AllocationRequest,
    ) -> Result<DeviceBuffer, BackendError> {
        // Mock CPU allocation
        Ok(DeviceBuffer {
            buffer_id: 0,
            backend: BackendKind::Cpu,
            device_id: device.clone(),
            size_bytes: request.size_bytes,
            alignment_bytes: request.alignment_bytes,
            memory_kind: MemoryKind::HostPageable,
            ownership: BufferOwnership::HostOwned,
            generation: 1,
        })
    }

    fn submit(
        &self,
        _queue: &QueueHandle,
        _submission: Submission,
    ) -> Result<SubmissionHandle, BackendError> {
        Ok(SubmissionHandle { opaque_id: 0 })
    }

    fn poll(&self, _submission: &SubmissionHandle) -> Result<SubmissionStatus, BackendError> {
        Ok(SubmissionStatus::Complete)
    }

    fn synchronize(&self, _submission: &SubmissionHandle) -> Result<(), BackendError> {
        Ok(())
    }
}
