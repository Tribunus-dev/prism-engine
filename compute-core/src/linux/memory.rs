use crate::linux::backend::{BackendKind, DeviceId};

use crate::linux::backend::RuntimeResourceId;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferHandle {
    pub id: RuntimeResourceId,
}

pub type BufferId = BufferHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferOwnership {
    HostOwned,
    UploadPending,
    DeviceOwned,
    ReadbackPending,
    HostReadable,
    Released,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    HostPageable,
    HostPinned,
    DeviceLocal,
    HostVisibleDevice,
    Unified,
}

#[derive(Debug, Clone)]
pub struct DeviceBuffer {
    pub buffer_id: BufferId,
    pub backend: BackendKind,
    pub device_id: DeviceId,
    pub size_bytes: u64,
    pub alignment_bytes: u64,
    pub memory_kind: MemoryKind,
    pub ownership: BufferOwnership,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPreference {
    PreferDeviceLocal,
    PreferHostVisible,
    RequireHostVisible,
    PreferUnified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsage {
    TransferSource,
    TransferDestination,
    KernelReadOnly,
    KernelReadWrite,
    Readback,
    Scratch,
}

#[derive(Debug, Clone)]
pub struct AllocationRequest {
    pub size_bytes: u64,
    pub alignment_bytes: u64,
    pub memory_preference: MemoryPreference,
    pub usage: BufferUsage,
    pub zero_initialize: bool,
    pub layout: ElementLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementLayout {
    Bytes,
    U32,
    U64,
}
