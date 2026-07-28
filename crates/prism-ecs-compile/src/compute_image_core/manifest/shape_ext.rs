//! CImage v2 — extended shape descriptors for multimodal tensors.
//!
//! Supports 2D (weight matrices) through 5D shapes for U-Net, Vision
//! Encoder, DiT, and other multimodal model weight layouts.

use serde::{Deserialize, Serialize};

/// Shape descriptor for multimodal tensors (up to 5 dimensions).
/// CImage v2 extension: supports U-Net, Vision Encoder, and DiT weight shapes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedShapeDescriptor {
    pub tensor_name: String,
    pub shape: Vec<usize>,
    pub memory_layout: MemoryLayout,
    pub packing_format: String,
    pub swizzle_stride: u32,
    /// ANE-preferred channel ordering (Channels-Last).
    pub nhwc_preferred: bool,
}

/// Memory layout strategy for a tensor's in-memory ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryLayout {
    DenseRowMajor,
    DenseColMajor,
    MortonZ,
    Planar2D,
    Blocked { block_size: usize },
    ChannelLastNhwc,
}

impl ExtendedShapeDescriptor {
    /// Create a 2D (rows × cols) descriptor for a classic weight matrix.
    /// Defaults to DenseRowMajor, no NHWC preference.
    pub fn new_2d(name: &str, rows: usize, cols: usize) -> Self {
        Self {
            tensor_name: name.to_owned(),
            shape: vec![rows, cols],
            memory_layout: MemoryLayout::DenseRowMajor,
            packing_format: String::new(),
            swizzle_stride: 0,
            nhwc_preferred: false,
        }
    }

    /// Create a 4D (N × C × H × W) descriptor, e.g. for convolution weights.
    /// Defaults to DenseRowMajor; `nhwc_preferred` is false (NCHW by convention).
    pub fn new_4d(name: &str, n: usize, c: usize, h: usize, w: usize) -> Self {
        Self {
            tensor_name: name.to_owned(),
            shape: vec![n, c, h, w],
            memory_layout: MemoryLayout::DenseRowMajor,
            packing_format: String::new(),
            swizzle_stride: 0,
            nhwc_preferred: false,
        }
    }

    /// Total number of scalar elements in the tensor.
    pub fn flat_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extended_shape_descriptor_flat_count() {
        let desc = ExtendedShapeDescriptor::new_4d("conv_weights", 1, 64, 32, 32);
        assert_eq!(desc.flat_elements(), 1 * 64 * 32 * 32);
        assert_eq!(desc.shape.len(), 4);
        assert!(!desc.nhwc_preferred);
    }

    #[test]
    fn test_extended_shape_2d_roundtrip() {
        let desc = ExtendedShapeDescriptor::new_2d("q_proj", 4096, 3840);
        assert_eq!(desc.flat_elements(), 4096 * 3840);
        assert_eq!(desc.shape.len(), 2);
    }
}
