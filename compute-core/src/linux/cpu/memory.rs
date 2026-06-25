use crate::linux::memory::{BufferHandle, BufferOwnership, BufferUsage};

pub enum CpuBufferStorage {
    Bytes(Vec<u8>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

pub struct CpuBuffer {
    pub storage: CpuBufferStorage,
    pub handle: BufferHandle,

    pub alignment_bytes: u64,
    pub ownership: BufferOwnership,
    pub usage: BufferUsage,
}
