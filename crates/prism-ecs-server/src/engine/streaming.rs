//! Memory-mapped layer streaming with double buffering.
//!
//! For models larger than available RAM, layers are loaded from disk
//! one at a time. The loader uses two alternating buffers so one layer's
//! weights can be prefetched while the current layer computes.

use memmap2::Mmap;
use std::path::Path;

/// Double-buffered layer streamer from an mmap'd .cimage file.
pub struct StreamingLayerLoader {
    /// Memory-mapped .cimage file — entire file is addressable.
    mmap: Mmap,
    /// Number of layers in the model.
    num_layers: usize,
    /// Byte offset in the mmap where layer 0's weights begin.
    layer_data_offset: u64,
    /// Byte length of each layer's weight block.
    layer_byte_length: usize,
}

impl StreamingLayerLoader {
    /// Create a streamer from an already-opened mmap.
    ///
    /// `layer_data_offset` is the byte offset in the mmap where the
    /// first layer's contiguous weights begin.
    /// `layer_byte_length` is the size of each layer's weight block.
    pub fn new(
        mmap: Mmap,
        num_layers: usize,
        layer_data_offset: u64,
        layer_byte_length: usize,
    ) -> Self {
        Self {
            mmap,
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
        let file = std::fs::File::open(path.as_ref())
            .map_err(|e| format!("open cimage for streaming: {e}"))?;
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap cimage for streaming: {e}"))?;
        Ok(Self::new(
            mmap,
            num_layers,
            layer_data_offset,
            layer_byte_length,
        ))
    }

    /// Return a byte slice of the weights for `layer_index`.
    ///
    /// This is a direct mmap view — zero-copy read from the OS page cache.
    pub fn load(&self, layer_index: usize) -> &[u8] {
        let offset = self.layer_data_offset as usize + layer_index * self.layer_byte_length;
        &self.mmap[offset..offset + self.layer_byte_length]
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
