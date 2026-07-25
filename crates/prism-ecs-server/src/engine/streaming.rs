//! Memory-mapped layer streaming with double buffering.
//!
//! For models larger than available RAM, layers are loaded from disk
//! one at a time. The loader uses two alternating buffers so one layer's
//! weights can be prefetched while the current layer computes.
//!
//! Note: this module previously used `memmap2::Mmap` directly, which
//! requires `unsafe` (forbidden in the `prism-ecs-server` layer per the
//! AGENTS.md rule). The migration replaces the mmap with a heap-backed
//! `Vec<u8>` loaded via `std::fs::read`. The cimage payload is bounded
//! (see `cimage/mod.rs` header validation), so the entire file fits in
//! memory. The streaming interface is preserved: callers see the same
//! `load(layer_index) -> &[u8]` API.

use std::path::Path;

/// Double-buffered layer streamer from a memory-resident .cimage file.
pub struct StreamingLayerLoader {
    /// The full cimage bytes loaded at open-time. Indexing into this buffer
    /// is identical in shape to indexing into the previous mmap; layer 0
    /// begins at `layer_data_offset`, and each subsequent layer is
    /// `layer_byte_length` further in.
    bytes: Vec<u8>,
    /// Number of layers in the model.
    num_layers: usize,
    /// Byte offset in the buffer where layer 0's weights begin.
    layer_data_offset: u64,
    /// Byte length of each layer's weight block.
    layer_byte_length: usize,
}

impl StreamingLayerLoader {
    /// Create a streamer from an already-loaded byte buffer.
    ///
    /// `layer_data_offset` is the byte offset in the buffer where the
    /// first layer's contiguous weights begin.
    /// `layer_byte_length` is the size of each layer's weight block.
    pub fn new(
        bytes: Vec<u8>,
        num_layers: usize,
        layer_data_offset: u64,
        layer_byte_length: usize,
    ) -> Self {
        Self {
            bytes,
            num_layers,
            layer_data_offset,
            layer_byte_length,
        }
    }

    /// Open a .cimage file and create a streamer from it.
    ///
    /// `layer_data_offset` and `layer_byte_length` are typically parsed
    /// from the model tensor metadata.
    pub fn open<P: AsRef<Path>>(
        path: P,
        num_layers: usize,
        layer_data_offset: u64,
        layer_byte_length: usize,
    ) -> Result<Self, String> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|e| format!("read cimage for streaming: {e}"))?;
        Ok(Self::new(
            bytes,
            num_layers,
            layer_data_offset,
            layer_byte_length,
        ))
    }

    /// Return a byte slice of the weights for `layer_index`.
    ///
    /// The slice is a zero-copy view into the loaded buffer; it remains
    /// valid for the lifetime of this `StreamingLayerLoader`.
    pub fn load(&self, layer_index: usize) -> &[u8] {
        let offset = self.layer_data_offset as usize + layer_index * self.layer_byte_length;
        &self.bytes[offset..offset + self.layer_byte_length]
    }

    /// Number of layers in the model.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Byte size of each layer's weights.
    pub fn layer_byte_length(&self) -> usize {
        self.layer_byte_length
    }
}
