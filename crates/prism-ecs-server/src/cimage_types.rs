//! CImage binary format types — ported from compute-core.
//!
//! Self-contained `repr(C)` binary format types with no external
//! dependencies beyond `std`.

use std::fmt;
use std::mem;

// =============================================================================
// Error type
// =============================================================================

/// Errors that can occur when parsing a CimageHeader from bytes.
#[derive(Debug)]
pub enum CimageParseError {
    /// Ran out of data while reading a fixed-size field.
    UnexpectedEof { expected: usize, offset: usize },
    /// Header is smaller than the minimum wire size.
    HeaderTooSmall { actual: usize, expected: usize },
    /// Magic bytes don't match PRISM magic.
    BadMagic { actual: [u8; 8] },
}

impl fmt::Display for CimageParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CimageParseError::UnexpectedEof { expected, offset } => {
                write!(
                    f,
                    "unexpected EOF: needed {expected} bytes at offset {offset}"
                )
            }
            CimageParseError::HeaderTooSmall { actual, expected } => {
                write!(f, "cimage header too small: {actual} < {expected}")
            }
            CimageParseError::BadMagic { actual } => {
                write!(f, "bad magic: {actual:02x?}")
            }
        }
    }
}

// =============================================================================
// Constants
// =============================================================================

/// Prism cimage magic bytes.
pub const PRISM_MAGIC: [u8; 8] = *b"PRISM\0\0\0";

/// Maximum number of segment directory entries.
pub const CIMAGE_SEGMENT_CAPACITY: usize = 32;

/// Wire size of the canonical little-endian CimageHeader: 728 bytes.
pub const CIMAGE_HEADER_WIRE_SIZE: usize = 8       // magic
    + 4   // version
    + 4   // segment_count
    + 32  // payload_hash
    + 4   // num_layers
    + 4   // num_heads
    + 4   // head_dim
    + 4   // hidden_dim
    + 4   // intermediate_dim
    + 4   // vocab_size
    + 4   // quantization_schema
    + 4   // draft_num_layers
    + CIMAGE_SEGMENT_CAPACITY * (4 + 8 + 8)  // segments
    + 8; // _pad

// =============================================================================
// Segment types
// =============================================================================

/// Encodes the type of content in a cimage segment.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    MetalLib = 0,
    TernaryWeights = 1,
    BlockScales = 2,
    LayoutMeta = 3,
    Vocabulary = 4,
    AneArchive = 5,
    TopologyTable = 6,
    CudaLib = 7,
    RocmLib = 8,
    LevelZeroLib = 9,
    VulkanLib = 10,
    IntelNpuBlob = 11,
    AmdNpuBlob = 12,
    QualcommNpuBlob = 13,
    GoogleTpuBlob = 14,
    WebGpuLib = 15,
    HuaweiAscendBlob = 16,
    HailoBlob = 17,
    LayerDirectory = 18,
    MultimodalProjectionWeights = 19,
    MultimodalProjectionScales = 20,
    MultimodalInputDescriptor = 21,
    MultimodalPositionEmbeddings = 22,
    MultimodalAuxiliaryWeights = 23,
    ExecutionGraph = 24,
    ModelArtifacts = 25,
    Nf4Tile640Weights = 26,
    BlockBiases = 27,
    MultimodalProjectionBiases = 28,
    HeterogeneousImage = 29,
    TtsTalkerWeight = 30,
    TtsTalkerScale = 31,
    TtsTalkerBias = 32,
    TtsCodePredictorWeight = 33,
    TtsCodePredictorScale = 34,
    TtsCodePredictorBias = 35,
    TtsCodecWeight = 36,
    TtsCodebook = 37,
    RawFp16Weights = 38,
    SparseMeta = 39,
    SparseIndices = 40,
    MatrixContract = 41,
    PagedAttentionBlockTable = 42,
    QuantizationScaleMeta = 43,
    MemoryPlan = 44,
}

/// A single segment directory entry.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SegmentEntry {
    pub kind: u32,
    pub offset: u64,
    pub length: u64,
}

impl SegmentEntry {
    pub fn new(kind: SegmentKind, offset: u64, length: u64) -> Self {
        Self {
            kind: kind as u32,
            offset,
            length,
        }
    }
}

// =============================================================================
// CimageHeader
// =============================================================================

/// Canonical on-disk header for Prism cimage files.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CimageHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub segment_count: u32,
    pub payload_hash: [u8; 32],
    pub num_layers: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub vocab_size: u32,
    pub quantization_schema: u32,
    pub draft_num_layers: u32,
    pub segments: [SegmentEntry; CIMAGE_SEGMENT_CAPACITY],
    pub _pad: [u8; 8],
}

impl CimageHeader {
    pub fn segment(&self, kind: SegmentKind) -> Option<SegmentEntry> {
        let kind_u32 = kind as u32;
        self.segments.iter().find(|s| s.kind == kind_u32).copied()
    }
    pub fn is_nf4_tile640(&self) -> bool {
        self.quantization_schema == 1
    }
}

// =============================================================================
// CimageLayoutMeta
// =============================================================================

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct CimageLayoutMeta {
    pub embed_clustered: TensorRecord,
    pub centroid_table: TensorRecord,
    pub cluster_map: TensorRecord,
    pub ternary_weights: TensorRecord,
    pub block_scales: TensorRecord,
    pub aux: TensorRecord,
    pub _pad: [u8; 32],
}

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct TensorRecord {
    pub offset: u64,
    pub length: u64,
    pub rows: u32,
    pub cols: u32,
    pub _pad: [u8; 16],
}

// =============================================================================
// Binary I/O
// =============================================================================

/// Read a fixed-size array from the data buffer, advancing the offset.
fn read_array<const N: usize>(data: &[u8], off: &mut usize) -> Result<[u8; N], CimageParseError> {
    let end = off.checked_add(N).ok_or(CimageParseError::UnexpectedEof {
        expected: N,
        offset: *off,
    })?;
    if end > data.len() {
        return Err(CimageParseError::UnexpectedEof {
            expected: N,
            offset: *off,
        });
    }
    let mut arr = [0u8; N];
    arr.copy_from_slice(&data[*off..end]);
    *off = end;
    Ok(arr)
}

pub fn read_cimage_header_le(data: &[u8]) -> Result<CimageHeader, CimageParseError> {
    if data.len() < CIMAGE_HEADER_WIRE_SIZE {
        return Err(CimageParseError::HeaderTooSmall {
            actual: data.len(),
            expected: CIMAGE_HEADER_WIRE_SIZE,
        });
    }
    let mut off = 0usize;

    let magic: [u8; 8] = read_array(data, &mut off)?;
    if magic != PRISM_MAGIC {
        return Err(CimageParseError::BadMagic { actual: magic });
    }
    let header = CimageHeader {
        magic,
        version: u32::from_le_bytes(read_array(data, &mut off)?),
        segment_count: u32::from_le_bytes(read_array(data, &mut off)?),
        payload_hash: read_array(data, &mut off)?,
        num_layers: u32::from_le_bytes(read_array(data, &mut off)?),
        num_heads: u32::from_le_bytes(read_array(data, &mut off)?),
        head_dim: u32::from_le_bytes(read_array(data, &mut off)?),
        hidden_dim: u32::from_le_bytes(read_array(data, &mut off)?),
        intermediate_dim: u32::from_le_bytes(read_array(data, &mut off)?),
        vocab_size: u32::from_le_bytes(read_array(data, &mut off)?),
        quantization_schema: u32::from_le_bytes(read_array(data, &mut off)?),
        draft_num_layers: u32::from_le_bytes(read_array(data, &mut off)?),
        segments: {
            let mut arr = [SegmentEntry {
                kind: 0,
                offset: 0,
                length: 0,
            }; CIMAGE_SEGMENT_CAPACITY];
            for entry in arr.iter_mut() {
                entry.kind = u32::from_le_bytes(read_array(data, &mut off)?);
                entry.offset = u64::from_le_bytes(read_array(data, &mut off)?);
                entry.length = u64::from_le_bytes(read_array(data, &mut off)?);
            }
            arr
        },
        _pad: [0u8; 8],
    };
    Ok(header)
}

pub fn verify_cimage(bytes: &[u8]) -> Result<(CimageHeader, CimageLayoutMeta), CimageParseError> {
    let header = read_cimage_header_le(bytes)?;
    let layout = header
        .segment(SegmentKind::LayoutMeta)
        .and_then(|entry| {
            let end = (entry.offset as usize).checked_add(entry.length as usize)?;
            if end > bytes.len() || entry.length as usize != mem::size_of::<CimageLayoutMeta>() {
                return None;
            }
            Some(unsafe {
                std::ptr::read_unaligned(
                    bytes.as_ptr().add(entry.offset as usize) as *const CimageLayoutMeta
                )
            })
        })
        .unwrap_or_default();
    Ok((header, layout))
}
