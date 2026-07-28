//! Megakernel KV cache — pure data types and pure constants.
//!
//! The actual pack/unpack logic runs inside the Metal shader on the
//! engine side; this file declares the public constants and the data
//! types that flow into / out of the KV slot.

/// Ternary KV block dimension.
pub const KV_BLOCK: u32 = 256;
/// Number of u32 words in a packed ternary KV block.
pub const KV_NIBBLES_U32: u32 = 13;
/// Byte size of a packed ternary KV block.
pub const KV_BLOCK_BYTES: u64 = (KV_NIBBLES_U32 as u64) * 4 + 2;

/// A single KV cache slot in the megakernel.
#[derive(Debug, Clone)]
pub struct KvCacheSlot {
    /// Slot index.
    pub slot_index: u32,
    /// Sequence length cached in this slot.
    pub sequence_length: u32,
    /// Byte offset in the KV buffer.
    pub byte_offset: u64,
    /// State of the slot.
    pub state: KvSlotState,
}

/// State of a KV cache slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvSlotState {
    /// Slot is empty.
    Empty,
    /// Slot is being filled.
    Filling,
    /// Slot is full and valid.
    Full,
    /// Slot is being evicted.
    Evicting,
}
