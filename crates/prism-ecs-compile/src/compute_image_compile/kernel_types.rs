//! Shared `#[repr(C)]` kernel struct types for the distill compiler.
//!
//! All structs are `#[repr(C)]` to match the Metal Shading Language layout
//! exactly. Field order, type size, and alignment must match the
//! corresponding MSL `struct` definitions in the `.metal` template files.
//!
//! Authority: per-page kernel receipt and projection metadata. Pure
//! data — no engine-coupled dependencies.

#![allow(clippy::module_name_repetitions)]

/// A single packed ternary page (640 weights) in the deployable `.cimage` format.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PackedTernaryPage640 {
    /// Packed 2-bit trits: 640 weights × 2 bits = 1280 bits = 160 bytes.
    /// For the tile640 format using base-3 encoding: 20 trits × log₂(3) bits
    /// per word, with 32 words per page (32 × 20 trits = 640).
    pub payload: [u32; 40],
    /// Page header: scale index, sidecar offset range, valid tail length, flags.
    pub header: PageHeader,
}

/// Compact page header.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PageHeader {
    /// Block-scale index in the outer scale table.
    pub scale_index: u16,
    /// Starting sidecar offset for this page (0 = none).
    pub sidecar_offset: u32,
    /// Length of the sidecar payload in bytes.
    pub sidecar_length: u16,
    /// Number of valid tail weights (for non-multiple-of-640 pages).
    pub valid_tail_len: u16,
    /// Page-level flags (bit 0 = outlier page, bit 1 = MoE, bit 2 = reserved, etc.).
    pub flags: u16,
    /// Reserved padding to align the header.
    pub _pad: [u16; 2],
}

/// Sidecar header (FP16 scales attached to a page).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PageSidecarHeader {
    /// Magic bytes: `"PSIDE"`.
    pub magic: [u8; 5],
    /// Format version.
    pub version: u8,
    /// Number of FP16 entries in the sidecar.
    pub count: u16,
    /// Reserved padding to align the header.
    pub _pad: [u8; 3],
}

/// Projection parameters for an attention head.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ProjectionParams {
    /// Input feature count.
    pub in_features: u32,
    /// Output feature count.
    pub out_features: u32,
    /// Head dimension (per head).
    pub head_dim: u16,
    /// Number of heads.
    pub num_heads: u16,
    /// Tile size (typically 640 for tile640).
    pub tile_size: u16,
    /// Reserved padding to align the struct.
    pub _pad: [u16; 3],
}

/// Per-kernel execution receipt.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct KernelReceipt {
    /// Stable kernel identifier (e.g. SHA-256 prefix).
    pub kernel_id: [u8; 16],
    /// Wall-clock execution time in microseconds.
    pub elapsed_us: u64,
    /// Number of tiles processed.
    pub tiles_processed: u32,
    /// Number of bytes read from the input buffer.
    pub bytes_read: u64,
    /// Number of bytes written to the output buffer.
    pub bytes_written: u64,
    /// Reservation depth (number of in-flight operations at peak).
    pub peak_reservation_depth: u16,
    /// Whether the kernel produced numerically-stable output.
    pub numerically_stable: u8,
    /// Reserved padding.
    pub _pad: [u8; 5],
}

/// Activations view — a typed slice of an external activation buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ActivationView {
    /// Pointer to the start of the activation data.
    pub ptr: *const f32,
    /// Number of elements in the activation buffer.
    pub len: usize,
    /// Stride between rows in elements (0 = tightly packed).
    pub row_stride: usize,
}

// SAFETY: `ActivationView` is a borrowed view; the caller is responsible
// for ensuring the underlying buffer outlives the view. Sending the
// raw pointer across threads is the caller's responsibility too.
unsafe impl Send for ActivationView {}
unsafe impl Sync for ActivationView {}

/// Per-tile error partial — partial reduction of `|pred − target|²` for one tile.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ErrorPartial {
    /// Sum of squared errors for this tile.
    pub sum_sq: f64,
    /// Number of elements in this partial.
    pub count: u64,
}

/// Per-page score (e.g. perplexity, gating score).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PageScore {
    /// Page index.
    pub page_index: u32,
    /// Page-level score (higher is better).
    pub score: f32,
}

/// Per-head attention probe.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AttentionProbe {
    /// Head index.
    pub head_index: u16,
    /// Number of tokens attended to.
    pub attended_tokens: u32,
    /// Per-head mean attention weight.
    pub mean_weight: f32,
}

/// Buffer-slot helper types for the page allocator.
pub mod buffer_slot {
    use super::PageHeader;

    /// Per-buffer-slot metadata.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct SlotMeta {
        /// Offset of the slot in the underlying allocation.
        pub offset: u64,
        /// Length of the slot in bytes.
        pub length: u64,
        /// Last-touch epoch for LRU eviction.
        pub last_touch_epoch: u32,
        /// Whether the slot is currently pinned.
        pub pinned: u8,
        /// Reserved padding to align the struct.
        pub _pad: [u8; 3],
    }

    /// Page header for a slot, with an extra offset field.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct SlotPageHeader {
        /// Underlying page header.
        pub header: PageHeader,
        /// Slot index in the parent buffer.
        pub slot_index: u32,
    }
}
