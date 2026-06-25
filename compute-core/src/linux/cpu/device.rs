use std::sync::Mutex;
use std::collections::HashMap;

use crate::linux::backend::{BackendKind, DeviceId, DeviceStableKey, VendorKind, RuntimeResourceId};
use crate::linux::capability::{BackendAvailability, DeviceCapabilities};
use crate::linux::device::{DeviceDescriptor, LinuxDeviceBackend};
use crate::linux::errors::BackendError;
use crate::linux::memory::{AllocationRequest, BufferOwnership, DeviceBuffer, MemoryKind, BufferHandle};
use crate::linux::queue::{QueueClass, QueueHandle};
use crate::linux::submission::{Submission, SubmissionHandle, SubmissionStatus};
use crate::linux::cpu::memory::CpuBuffer;

pub struct CpuBackend {
    buffers: Mutex<HashMap<u64, CpuBuffer>>,
    next_id: Mutex<u64>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(HashMap::new()),
            next_id: Mutex::new(1),
        }
    }

    fn generate_resource_id(&self, device: &DeviceId) -> RuntimeResourceId {
        let mut id_guard = self.next_id.lock().unwrap();
        let id = *id_guard;
        *id_guard += 1;
        RuntimeResourceId {
            backend: BackendKind::Cpu,
            device: device.clone(),
            generation: 1,
            opaque_id: id,
        }
    }
}

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
        device: &DeviceId,
        class: QueueClass,
    ) -> Result<QueueHandle, BackendError> {
        Ok(QueueHandle {
            id: self.generate_resource_id(device),
            class,
        })
    }

    fn allocate(
        &self,
        device: &DeviceId,
        request: AllocationRequest,
    ) -> Result<DeviceBuffer, BackendError> {
        if request.size_bytes == 0 {
            return Err(BackendError::AllocationFailed("Zero-length allocation".into()));
        }

        let resource_id = self.generate_resource_id(device);
        let handle = BufferHandle { id: resource_id.clone() };

        let buf = CpuBuffer {
            handle: handle.clone(),
            bytes: vec![0; request.size_bytes as usize],
            alignment_bytes: request.alignment_bytes,
            ownership: BufferOwnership::HostOwned,
            usage: request.usage,
        };

        self.buffers.lock().unwrap().insert(resource_id.opaque_id, buf);

        Ok(DeviceBuffer {
            buffer_id: handle,
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
        queue: &QueueHandle,
        submission: Submission,
    ) -> Result<SubmissionHandle, BackendError> {
        let mut buffers = self.buffers.lock().unwrap();

        match submission {
            Submission::Fill { destination, value, element_count } => {
                let dest = buffers.get_mut(&destination.id.opaque_id)
                    .ok_or_else(|| BackendError::SubmissionFailed("Invalid destination buffer".into()))?;

                if dest.ownership == BufferOwnership::Released {
                    return Err(BackendError::SubmissionFailed("Buffer released".into()));
                }

                // Treat bytes as u32 array and fill
                let dest_ptr = dest.bytes.as_mut_ptr() as *mut u32;
    let dest_slice: &mut [u32] = unsafe { std::slice::from_raw_parts_mut(dest_ptr, dest.bytes.len() / 4) };
                if element_count as usize > dest_slice.len() {
                    return Err(BackendError::SubmissionFailed("Out of bounds".into()));
                }

                for i in 0..element_count as usize {
                    dest_slice[i] = value;
                }
            },
            Submission::Copy { source, destination, size_bytes } => {
                let (src_bytes, src_own) = {
                    let src = buffers.get(&source.id.opaque_id)
                        .ok_or_else(|| BackendError::SubmissionFailed("Invalid source buffer".into()))?;
                    if src.ownership == BufferOwnership::Released {
                        return Err(BackendError::SubmissionFailed("Source buffer released".into()));
                    }
                    if size_bytes as usize > src.bytes.len() {
                        return Err(BackendError::SubmissionFailed("Out of bounds read".into()));
                    }
                    (src.bytes[..size_bytes as usize].to_vec(), src.ownership)
                };

                let dest = buffers.get_mut(&destination.id.opaque_id)
                    .ok_or_else(|| BackendError::SubmissionFailed("Invalid destination buffer".into()))?;

                if dest.ownership == BufferOwnership::Released {
                    return Err(BackendError::SubmissionFailed("Dest buffer released".into()));
                }
                if size_bytes as usize > dest.bytes.len() {
                    return Err(BackendError::SubmissionFailed("Out of bounds write".into()));
                }

                dest.bytes[..size_bytes as usize].copy_from_slice(&src_bytes);
            },
            Submission::Reduction { source, destination, element_count, operation } => {
                // Simplistic sum mock to satisfy the contract structurally
                let (src_bytes, _) = {
                    let src = buffers.get(&source.id.opaque_id).ok_or(BackendError::SubmissionFailed("Invalid src".into()))?;
                    (src.bytes.clone(), src.ownership)
                };

                let dest = buffers.get_mut(&destination.id.opaque_id).ok_or(BackendError::SubmissionFailed("Invalid dest".into()))?;

                match operation {
                    crate::linux::submission::ReductionOperation::SumU32 => {
                        let src_ptr = src_bytes.as_ptr() as *const u32;
    let src_slice: &[u32] = unsafe { std::slice::from_raw_parts(src_ptr, src_bytes.len() / 4) };
                        let sum: u32 = src_slice[..element_count as usize].iter().sum();
                        let dest_ptr = dest.bytes.as_mut_ptr() as *mut u32;
    let dest_slice: &mut [u32] = unsafe { std::slice::from_raw_parts_mut(dest_ptr, dest.bytes.len() / 4) };
                        dest_slice[0] = sum;
                    }
                    _ => return Err(BackendError::UnsupportedOperation("Unimplemented reduction".into())),
                }
            },
            Submission::DeterministicHash { .. } => {
                return Err(BackendError::UnsupportedOperation("Stub Hash".into()));
            },
            Submission::ScanPreparation { .. } => {
                return Err(BackendError::UnsupportedOperation("Stub Scan".into()));
            },
        }

        let sub_id = RuntimeResourceId {
            backend: BackendKind::Cpu,
            device: queue.id.device.clone(),
            generation: 1,
            opaque_id: 100, // mock ID
        };

        Ok(SubmissionHandle {
            id: sub_id,
            queue_id: queue.id.clone(),
        })
    }

    fn poll(&self, _submission: &SubmissionHandle) -> Result<SubmissionStatus, BackendError> {
        Ok(SubmissionStatus::Complete)
    }

    fn synchronize(&self, _submission: &SubmissionHandle) -> Result<(), BackendError> {
        Ok(())
    }
}
