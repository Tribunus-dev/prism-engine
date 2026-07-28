//! CImage binary format — header, segment directory, segment kinds, ANE descriptor.
//!
//! This module owns the on-disk wire format for `.cimage` binaries:
//! the magic bytes, header layout, segment directory, and the canonical
//! segment-kind taxonomy. It also owns the segment directory entry
//! type, the layer-directory entry, the ANE model descriptor, and the
//! helper functions for serialising and parsing the header in canonical
//! little-endian order.
//!
//! Authority: CImage wire format — header bytes, segment directory, and
//! segment-kind taxonomy. Pure data; no engine-coupled dependencies.

#![allow(clippy::module_name_repetitions)]

/// Magic bytes for v2 (Prism) CImage binaries: `"PRISM\0\0\0"`.
pub const PRISM_MAGIC: [u8; 8] = *b"PRISM\0\0\0";
/// CImage segment alignment boundary (16 KiB).
pub const CIMAGE_PAGE_SIZE: u64 = 16384;
/// Ternary tile640 quantization schema discriminant.
pub const QUANT_SCHEMA_TERNARY_TILE640: u32 = 0;
/// NF4 tile640 quantization schema discriminant.
pub const QUANT_SCHEMA_NF4_TILE640: u32 = 1;
/// Deprecated alias; use [`CIMAGE_PAGE_SIZE`].
pub const PRISM_PAGE_SIZE: u64 = CIMAGE_PAGE_SIZE;
/// Number of slots in the segment directory.
pub const CIMAGE_SEGMENT_CAPACITY: usize = 32;
/// Maximum number of layers in the MTP draft decoder.
pub const MAX_DRAFT_LAYERS: u8 = 4;

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

/// Type tags for ModelArtifacts segment entries.
pub mod model_artifact_tag {
    /// SentencePiece .model proto.
    pub const TOKENIZER: u32 = 0x01;
    /// Multimodal special token map (JSON).
    pub const TOKEN_MAP: u32 = 0x04;
    /// Chat prompt template string.
    pub const CHAT_TEMPLATE: u32 = 0x05;
    /// Sampling params (JSON).
    pub const GENERATION_CONFIG: u32 = 0x06;
    /// Ternary-packed embedding table (nibbles reordered by cluster).
    pub const EMBED_NIBBLES: u32 = 0x10;
    /// FP16 block scales for the embedding table.
    pub const EMBED_SCALES: u32 = 0x11;
    /// Ternary-packed centroid table (256 centroids × hidden_dim).
    pub const CENTROID_NIBBLES: u32 = 0x12;
    /// FP16 block scales for centroids.
    pub const CENTROID_SCALES: u32 = 0x13;
    /// u32 cluster assignments (vocab_size entries).
    pub const CLUSTER_MAP: u32 = 0x14;
    /// FP16 per-layer RMSNorm weights.
    pub const AUX_NORMS: u32 = 0x15;
}

/// Encodes the type of content in a CImage segment.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    /// Compiled Metal kernel library (.metallib).
    MetalLib = 0,
    /// Ternary tile640 packed weights.
    TernaryWeights = 1,
    /// FP16 block scales for ternary weights.
    BlockScales = 2,
    /// CimageLayoutMeta — tensor records, per-layer metadata.
    LayoutMeta = 3,
    /// Vocabulary / embedding table.
    Vocabulary = 4,
    /// Apple ANE .mlmodelc or .ane.tar archive.
    AneArchive = 5,
    /// Stride/prefetch topology table.
    TopologyTable = 6,
    /// Compiled CUDA kernel blob (.cubin, .fatbin).
    CudaLib = 7,
    /// Compiled ROCm kernel blob (.co, .hsaco).
    RocmLib = 8,
    /// Compiled Level Zero / SPIR-V kernel (.spv).
    LevelZeroLib = 9,
    /// Compiled Vulkan shader (.spv).
    VulkanLib = 10,
    /// Intel NPU compiled model blob.
    IntelNpuBlob = 11,
    /// AMD NPU (XDNA) compiled model blob.
    AmdNpuBlob = 12,
    /// Qualcomm Hexagon / HTP compiled model.
    QualcommNpuBlob = 13,
    /// Google TPU compiled model.
    GoogleTpuBlob = 14,
    /// Compiled WebGPU / WGSL shader.
    WebGpuLib = 15,
    /// Huawei Ascend NPU (DaVinci) compiled model (.om).
    HuaweiAscendBlob = 16,
    /// Hailo NPU compiled executable (.hef).
    HailoBlob = 17,
    /// Per-layer weight offset table (array of `LayerDirectoryEntry`).
    LayerDirectory = 18,
    /// Packed ternary projection weight matrices for multimodal input adapters.
    MultimodalProjectionWeights = 19,
    /// FP16 block scales corresponding to `MultimodalProjectionWeights`.
    MultimodalProjectionScales = 20,
    /// Versioned binary descriptor describing modality support, tensor layout.
    MultimodalInputDescriptor = 21,
    /// Learned two-dimensional position embeddings for image patches.
    MultimodalPositionEmbeddings = 22,
    /// Biases, layer norms, pooling kernels, and small affine parameters.
    MultimodalAuxiliaryWeights = 23,
    /// Binary execution graph descriptor (per-layer DAG, device routing, etc.).
    ExecutionGraph = 24,
    /// Tokenizer, multimodal special token map, audio codebook, chat template.
    ModelArtifacts = 25,
    /// Canonical NF4Tile640 packed weights (raw U8 resident bytes).
    Nf4Tile640Weights = 26,
    /// FP32 bias metadata corresponding to packed quantized weights.
    BlockBiases = 27,
    /// Multimodal projection biases (byte-parallel to `MultimodalProjectionScales`).
    MultimodalProjectionBiases = 28,
    /// JSON-serialized HeterogeneousExecutionImage for tri-lane execution.
    HeterogeneousImage = 29,
    /// TTS Talker (28-layer decoder) nf4tile640 packed weights.
    TtsTalkerWeight = 30,
    /// TTS Talker nf4tile640 scales.
    TtsTalkerScale = 31,
    /// TTS Talker nf4tile640 biases.
    TtsTalkerBias = 32,
    /// TTS Code Predictor weights.
    TtsCodePredictorWeight = 33,
    /// TTS Code Predictor scales.
    TtsCodePredictorScale = 34,
    /// TTS Code Predictor biases.
    TtsCodePredictorBias = 35,
    /// TTS Mimi Codec weights.
    TtsCodecWeight = 36,
    /// TTS codebook embeddings (16 codebooks × 2048 entries × 128 dim).
    TtsCodebook = 37,
    /// Raw FP16 weights (matrices that can't be nf4tile640-quantized).
    RawF16Weights = 38,
    /// Int8Tile640 packed weights (640-byte code stride per tile).
    Int8Tile640Weights = 39,
    /// Quantization sidecars (reduction-axis FP16 scale vectors, etc.).
    QuantizationSidecars = 40,
    /// Array of MatrixWeightBinding records (per-tensor format contract).
    MatrixContract = 41,
    /// Per-expert weight offset directory for MoE models.
    ExpertDirectory = 42,
}

/// One entry in the cimage segment directory.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SegmentEntry {
    /// SegmentKind discriminant.
    pub kind: u32,
    /// Byte offset from start of cimage.
    pub offset: u64,
    /// Byte length of this segment.
    pub length: u64,
}

impl SegmentEntry {
    /// Construct a new segment entry from a `SegmentKind` and offsets.
    pub fn new(kind: SegmentKind, offset: u64, length: u64) -> Self {
        Self {
            kind: kind as u32,
            offset,
            length,
        }
    }
}

/// One entry in the ModelArtifacts segment. Flat binary: tag + length + data.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ModelArtifactEntry;

impl ModelArtifactEntry {
    /// Header size in bytes: u32 tag + u32 length.
    pub const HEADER_SIZE: usize = 8;

    /// Encode a single `(tag, data)` entry, appending it to `out`.
    pub fn encode(tag: u32, data: &[u8], out: &mut Vec<u8>) {
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
    }

    /// Iterate over `(tag, payload)` pairs in a ModelArtifacts blob.
    pub fn iter_entries(blob: &[u8]) -> ModelArtifactIter<'_> {
        ModelArtifactIter { blob, pos: 0 }
    }
}

/// Iterator over ModelArtifacts segment entries.
pub struct ModelArtifactIter<'a> {
    blob: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for ModelArtifactIter<'a> {
    type Item = (u32, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + 8 > self.blob.len() {
            return None;
        }
        let tag = u32::from_le_bytes([
            self.blob[self.pos],
            self.blob[self.pos + 1],
            self.blob[self.pos + 2],
            self.blob[self.pos + 3],
        ]);
        let len = u32::from_le_bytes([
            self.blob[self.pos + 4],
            self.blob[self.pos + 5],
            self.blob[self.pos + 6],
            self.blob[self.pos + 7],
        ]) as usize;
        self.pos += 8;
        if self.pos + len > self.blob.len() {
            return None;
        }
        let data = &self.blob[self.pos..self.pos + len];
        self.pos += len;
        Some((tag, data))
    }
}

/// One entry in the layer directory — exact byte range of a single
/// transformer layer's packed weights, block scales, layer kind, and flags.
///
/// Total size: 6 × u64 = 48 bytes, alignment-friendly.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct LayerDirectoryEntry {
    /// Byte offset from start of TernaryWeights segment data.
    pub weights_offset: u64,
    /// Byte length of this layer's packed u32 weights.
    pub weights_length: u64,
    /// Byte offset from start of BlockScales segment data.
    pub scales_offset: u64,
    /// Byte length of this layer's FP16 block scales.
    pub scales_length: u64,
    /// Kind identifier for this layer (e.g. attention vs. feed-forward).
    pub layer_kind: u64,
    /// Flags for this layer entry.
    pub flags: u64,
}

impl LayerDirectoryEntry {
    /// Construct a layer directory entry from individual fields.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weights_offset: u64,
        weights_length: u64,
        scales_offset: u64,
        scales_length: u64,
        layer_kind: u64,
        flags: u64,
    ) -> Self {
        Self {
            weights_offset,
            weights_length,
            scales_offset,
            scales_length,
            layer_kind,
            flags,
        }
    }
}

/// Role of an embedded ANE/NPU model within the cimage.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AneModelRole {
    /// ANE model runs prompt prefill.
    Prefill = 0,
    /// ANE model runs MTP decode.
    MtpDecode = 1,
    /// ANE model is a vision encoder.
    VisionEncoder = 2,
    /// Role is unknown / not set.
    #[default]
    Unknown = 0xFF,
}

/// Describes the I/O contract of an embedded ANE/NPU model segment.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AneModelDescriptor {
    /// [`AneModelRole`] discriminant.
    pub role: u32,
    /// SHA-256 digest of the input schema.
    pub input_schema_digest: [u8; 32],
    /// SHA-256 digest of the output schema.
    pub output_schema_digest: [u8; 32],
    /// 1 if the model supports stateful decode, 0 otherwise.
    pub supports_stateful_decode: u8,
    /// Maximum supported sequence length.
    pub max_sequence_length: u32,
    /// Offset of the token input name within the descriptor's name table.
    pub token_input_name_offset: u32,
    /// Offset of the logits output name within the descriptor's name table.
    pub logits_output_name_offset: u32,
    /// Reserved padding (alignment to 8 bytes).
    pub _pad: [u8; 9],
}

/// Serialize a `CimageHeader` to a writer in canonical little-endian format.
pub fn write_cimage_header_le<W: std::io::Write>(
    writer: &mut W,
    header: &CimageHeader,
) -> std::io::Result<()> {
    writer.write_all(&header.magic)?;
    writer.write_all(&header.version.to_le_bytes())?;
    writer.write_all(&header.segment_count.to_le_bytes())?;
    writer.write_all(&header.payload_hash)?;
    writer.write_all(&header.num_layers.to_le_bytes())?;
    writer.write_all(&header.num_heads.to_le_bytes())?;
    writer.write_all(&header.head_dim.to_le_bytes())?;
    writer.write_all(&header.hidden_dim.to_le_bytes())?;
    writer.write_all(&header.intermediate_dim.to_le_bytes())?;
    writer.write_all(&header.vocab_size.to_le_bytes())?;
    writer.write_all(&header.quantization_schema.to_le_bytes())?;
    writer.write_all(&header.draft_num_layers.to_le_bytes())?;
    for seg in &header.segments {
        writer.write_all(&seg.kind.to_le_bytes())?;
        writer.write_all(&seg.offset.to_le_bytes())?;
        writer.write_all(&seg.length.to_le_bytes())?;
    }
    writer.write_all(&[0u8; 8])?; // _pad
    Ok(())
}

/// Parse a `CimageHeader` from a byte slice (canonical little-endian format).
pub fn read_cimage_header_le(data: &[u8]) -> Result<CimageHeader, String> {
    if data.len() < CIMAGE_HEADER_WIRE_SIZE {
        return Err(format!(
            "cimage header too small: {} < {}",
            data.len(),
            CIMAGE_HEADER_WIRE_SIZE
        ));
    }
    let mut off = 0usize;
    let mut read = |n: usize| -> Result<&[u8], String> {
        if off + n > data.len() {
            return Err(format!(
                "cimage header short read at offset {off}, need {n} bytes"
            ));
        }
        let slice = &data[off..off + n];
        off += n;
        Ok(slice)
    };
    let magic: [u8; 8] = read(8)?
        .try_into()
        .map_err(|_| "magic slice conversion".to_string())?;
    if &magic != &PRISM_MAGIC {
        return Err(format!("bad magic: {:?}", &magic));
    }
    let version_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "version conversion".to_string())?;
    let segment_count_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "segment_count conversion".to_string())?;
    let payload_hash: [u8; 32] = read(32)?
        .try_into()
        .map_err(|_| "payload_hash conversion".to_string())?;
    let num_layers_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "num_layers conversion".to_string())?;
    let num_heads_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "num_heads conversion".to_string())?;
    let head_dim_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "head_dim conversion".to_string())?;
    let hidden_dim_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "hidden_dim conversion".to_string())?;
    let intermediate_dim_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "intermediate_dim conversion".to_string())?;
    let vocab_size_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "vocab_size conversion".to_string())?;
    let quantization_schema_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "quantization_schema conversion".to_string())?;
    let draft_num_layers_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "draft_num_layers conversion".to_string())?;
    let header = CimageHeader {
        magic,
        version: u32::from_le_bytes(version_bytes),
        segment_count: u32::from_le_bytes(segment_count_bytes),
        payload_hash,
        num_layers: u32::from_le_bytes(num_layers_bytes),
        num_heads: u32::from_le_bytes(num_heads_bytes),
        head_dim: u32::from_le_bytes(head_dim_bytes),
        hidden_dim: u32::from_le_bytes(hidden_dim_bytes),
        intermediate_dim: u32::from_le_bytes(intermediate_dim_bytes),
        vocab_size: u32::from_le_bytes(vocab_size_bytes),
        quantization_schema: u32::from_le_bytes(quantization_schema_bytes),
        draft_num_layers: u32::from_le_bytes(draft_num_layers_bytes),
        segments: {
            let mut arr = [SegmentEntry {
                kind: 0,
                offset: 0,
                length: 0,
            }; CIMAGE_SEGMENT_CAPACITY];
            for entry in arr.iter_mut() {
                let kind_bytes: [u8; 4] = read(4)?
                    .try_into()
                    .map_err(|_| "segment kind conversion".to_string())?;
                let offset_bytes: [u8; 8] = read(8)?
                    .try_into()
                    .map_err(|_| "segment offset conversion".to_string())?;
                let length_bytes: [u8; 8] = read(8)?
                    .try_into()
                    .map_err(|_| "segment length conversion".to_string())?;
                entry.kind = u32::from_le_bytes(kind_bytes);
                entry.offset = u64::from_le_bytes(offset_bytes);
                entry.length = u64::from_le_bytes(length_bytes);
            }
            arr
        },
        _pad: [0u8; 8],
    };
    Ok(header)
}

/// Top-level cimage header — magic, version, architecture dims, segment directory.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CimageHeader {
    /// Always [`PRISM_MAGIC`].
    pub magic: [u8; 8],
    /// Wire-format version.
    pub version: u32,
    /// Number of populated entries in `segments`.
    pub segment_count: u32,
    /// SHA-256 of the concatenated payload (after the header).
    pub payload_hash: [u8; 32],
    /// Number of decoder layers.
    pub num_layers: u32,
    /// Number of attention heads.
    pub num_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Hidden dimension.
    pub hidden_dim: u32,
    /// FFN intermediate dimension.
    pub intermediate_dim: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// [`QUANT_SCHEMA_TERNARY_TILE640`] or [`QUANT_SCHEMA_NF4_TILE640`].
    pub quantization_schema: u32,
    /// Number of layers in the MTP draft decoder (0 = no draft model).
    pub draft_num_layers: u32,
    /// Segment directory.
    pub segments: [SegmentEntry; CIMAGE_SEGMENT_CAPACITY],
    /// Trailing padding to align the header to its declared wire size.
    pub _pad: [u8; 8],
}

impl CimageHeader {
    /// Look up a segment by kind. Returns the entry if found, `None` otherwise.
    pub fn segment(&self, kind: SegmentKind) -> Option<SegmentEntry> {
        let kind_u32 = kind as u32;
        self.segments.iter().find(|s| s.kind == kind_u32).copied()
    }

    /// Whether the cimage uses the NF4 tile640 quantization schema.
    pub fn is_nf4_tile640(&self) -> bool {
        self.quantization_schema == QUANT_SCHEMA_NF4_TILE640
    }

    /// Which segment kind holds the primary weight payload, based on the
    /// quantization schema.
    pub fn primary_weight_segment_kind(&self) -> SegmentKind {
        if self.is_nf4_tile640() {
            SegmentKind::Nf4Tile640Weights
        } else {
            SegmentKind::TernaryWeights
        }
    }

    /// Convenience: the primary weight segment entry, if any.
    pub fn primary_weight_segment(&self) -> Option<SegmentEntry> {
        self.segment(self.primary_weight_segment_kind())
    }
}

/// Backward-compatible type alias for the v1 cimage header.
pub type PrismCimageHeader = CimageHeader;
