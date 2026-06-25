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

use crate::linux::receipt::BackendErrorReceipt;
use crate::linux::submission::SubmissionKind;

pub struct SubmissionRecord {
    pub handle: SubmissionHandle,
    pub queue: QueueHandle,
    pub operation: SubmissionKind,
    pub referenced_buffers: Vec<BufferHandle>,
    pub status: SubmissionStatus,
    pub submitted_at_unix_ms: u64,
    pub started_at_unix_ms: Option<u64>,
    pub completed_at_unix_ms: Option<u64>,
    pub error: Option<BackendErrorReceipt>,
    pub output_hash: Option<u64>,
}

pub struct CpuBackend {
    buffers: Mutex<HashMap<RuntimeResourceId, CpuBuffer>>,
    queues: Mutex<HashMap<RuntimeResourceId, QueueHandle>>,
    events: Mutex<HashMap<RuntimeResourceId, ()>>,
    submissions: Mutex<HashMap<RuntimeResourceId, SubmissionRecord>>,
    next_id: Mutex<u64>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(HashMap::new()),
            queues: Mutex::new(HashMap::new()),
            events: Mutex::new(HashMap::new()),
            submissions: Mutex::new(HashMap::new()),
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
            storage: match request.layout {
            crate::linux::memory::ElementLayout::Bytes => crate::linux::cpu::memory::CpuBufferStorage::Bytes(vec![0; request.size_bytes as usize]),
            crate::linux::memory::ElementLayout::U32 => crate::linux::cpu::memory::CpuBufferStorage::U32(vec![0; (request.size_bytes / 4) as usize]),
            crate::linux::memory::ElementLayout::U64 => crate::linux::cpu::memory::CpuBufferStorage::U64(vec![0; (request.size_bytes / 8) as usize]),
        },
            alignment_bytes: request.alignment_bytes,
            ownership: BufferOwnership::HostOwned,
            usage: request.usage,
        };

        self.buffers.lock().unwrap().insert(resource_id.clone(), buf);

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
                let dest = buffers.get_mut(&destination.id)
                    .ok_or_else(|| BackendError::SubmissionFailed("Invalid destination buffer".into()))?;

                if dest.ownership == BufferOwnership::Released {
                    return Err(BackendError::SubmissionFailed("Buffer released".into()));
                }

                match &mut dest.storage {
                    crate::linux::cpu::memory::CpuBufferStorage::U32(vec) => {
                        if element_count as usize > vec.len() {
                            return Err(BackendError::SubmissionFailed("Out of bounds".into()));
                        }
                        for i in 0..element_count as usize {
                            vec[i] = value;
                        }
                    },
                    _ => return Err(BackendError::SubmissionFailed("Invalid layout for Fill U32".into())),
                }
            },
            Submission::Copy { source, destination, size_bytes } => {
                let src_bytes = {
                    let src = buffers.get(&source.id)
                        .ok_or_else(|| BackendError::SubmissionFailed("Invalid src".into()))?;
                    if src.ownership == BufferOwnership::Released {
                        return Err(BackendError::SubmissionFailed("Src released".into()));
                    }
                    match &src.storage {
                        crate::linux::cpu::memory::CpuBufferStorage::Bytes(v) => {
                            if size_bytes as usize > v.len() { return Err(BackendError::SubmissionFailed("Out of bounds".into())); }
                            v[..size_bytes as usize].to_vec()
                        },
                        _ => return Err(BackendError::SubmissionFailed("Invalid layout for Copy".into())),
                    }
                };

                let dest = buffers.get_mut(&destination.id)
                    .ok_or_else(|| BackendError::SubmissionFailed("Invalid dest".into()))?;

                match &mut dest.storage {
                    crate::linux::cpu::memory::CpuBufferStorage::Bytes(v) => {
                        if size_bytes as usize > v.len() { return Err(BackendError::SubmissionFailed("Out of bounds".into())); }
                        v[..size_bytes as usize].copy_from_slice(&src_bytes);
                    },
                    _ => return Err(BackendError::SubmissionFailed("Invalid layout for Copy".into())),
                }
            },
            Submission::Reduction { source, destination, element_count, operation } => {
                let src = buffers.get(&source.id).ok_or(BackendError::SubmissionFailed("Invalid src".into()))?;
                if src.ownership == BufferOwnership::Released { return Err(BackendError::SubmissionFailed("Src released".into())); }

                // For demonstration, only implementing SumU32 thoroughly as requested in spec snippet
                match operation {
                    crate::linux::submission::ReductionOperation::SumU32 => {
                        let sum = match &src.storage {
                            crate::linux::cpu::memory::CpuBufferStorage::U32(v) => {
                                if element_count as usize > v.len() { return Err(BackendError::SubmissionFailed("OOB".into())); }
                                let mut s: u32 = 0;
                                for &x in &v[..element_count as usize] {
                                    s = s.checked_add(x).ok_or_else(|| BackendError::SubmissionFailed("ArithmeticOverflow".into()))?;
                                }
                                s
                            },
                            _ => return Err(BackendError::SubmissionFailed("Invalid layout".into())),
                        };

                        let dest = buffers.get_mut(&destination.id).ok_or(BackendError::SubmissionFailed("Invalid dest".into()))?;
                        match &mut dest.storage {
                            crate::linux::cpu::memory::CpuBufferStorage::U32(v) => {
                                if v.is_empty() { return Err(BackendError::SubmissionFailed("Dest too small".into())); }
                                v[0] = sum;
                            },
                            _ => return Err(BackendError::SubmissionFailed("Invalid dest layout".into())),
                        }
                    },
                    _ => return Err(BackendError::UnsupportedOperation("Unimplemented reduction".into())),
                }
            },
            Submission::DeterministicHash { .. } => return Err(BackendError::UnsupportedOperation("Stub Hash".into())),
            Submission::ScanPreparation { .. } => return Err(BackendError::UnsupportedOperation("Stub Scan".into())),
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

    fn poll(&self, submission: &SubmissionHandle) -> Result<SubmissionStatus, BackendError> {
        let subs = self.submissions.lock().unwrap();
        if let Some(record) = subs.get(&submission.id) {
            Ok(record.status)
        } else {
            Err(BackendError::SubmissionFailed("Submission not found".into()))
        }
    }

    fn synchronize(&self, submission: &SubmissionHandle) -> Result<(), BackendError> {
        let subs = self.submissions.lock().unwrap();
        if let Some(record) = subs.get(&submission.id) {
            if record.status == SubmissionStatus::Failed {
                return Err(BackendError::SubmissionFailed("Submission previously failed".into()));
            }
            Ok(())
        } else {
            Err(BackendError::SubmissionFailed("Submission not found".into()))
        }
    }

    fn release(&self, buffer: BufferHandle) -> Result<(), BackendError> {
        let mut buffers = self.buffers.lock().unwrap();
        if let Some(buf) = buffers.get_mut(&buffer.id) {
            buf.ownership = BufferOwnership::Released;
            // Immediate drop in CPU backend
            buffers.remove(&buffer.id);
            Ok(())
        } else {
            Err(BackendError::SubmissionFailed("Buffer not found or already released".into()))
        }
    }

    fn create_event(&self, device: &DeviceId) -> Result<crate::linux::event::EventHandle, BackendError> {
        let id = self.generate_resource_id(device);
        self.events.lock().unwrap().insert(id.clone(), ());
        Ok(crate::linux::event::EventHandle { id })
    }

    fn record_event(&self, _queue: &QueueHandle, _event: &crate::linux::event::EventHandle) -> Result<(), BackendError> {
        Ok(())
    }

    fn wait_event(&self, _queue: &QueueHandle, _event: &crate::linux::event::EventHandle) -> Result<(), BackendError> {
        Ok(())
    }
}
