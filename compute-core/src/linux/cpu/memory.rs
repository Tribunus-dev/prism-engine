use crate::linux::memory::{BufferHandle, BufferOwnership, BufferUsage};

pub struct CpuBuffer {
    pub handle: BufferHandle,
    pub bytes: Vec<u8>,
    pub alignment_bytes: u64,
    pub ownership: BufferOwnership,
    pub usage: BufferUsage,
}
