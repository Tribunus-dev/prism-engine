//! V1/V2 cimage binary format, cimage compiler, and ANE swizzled payload.

pub const STATE_IDLE: u8 = 0;
pub const STATE_PREFETCHING: u8 = 1;
pub const STATE_READY: u8 = 2;
pub const STATE_EXECUTING: u8 = 3;

// ── V2 (Prism) types ──────────────────────────────────────────────

pub const PRISM_MAGIC: [u8; 8] = *b"PRISM\0\0\0";
pub const CIMAGE_PAGE_SIZE: u64 = 16384;
pub const QUANT_SCHEMA_TERNARY_TILE640: u32 = 0;
pub const QUANT_SCHEMA_NF4_TILE640: u32 = 1;
/// Deprecated alias; use CIMAGE_PAGE_SIZE
pub const PRISM_PAGE_SIZE: u64 = CIMAGE_PAGE_SIZE;
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

/// Serialize a CimageHeader to a writer in canonical little-endian format.
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

/// Parse a CimageHeader from a byte slice (canonical little-endian format).
pub fn read_cimage_header_le(data: &[u8]) -> Result<CimageHeader, String> {
    if data.len() < CIMAGE_HEADER_WIRE_SIZE {
        return Err(format!(
            "cimage header too small: {} < {}",
            data.len(),
            CIMAGE_HEADER_WIRE_SIZE
        ));
    }
    let mut off = 0usize;
    let mut read = |n: usize| -> &[u8] {
        let slice = &data[off..off + n];
        off += n;
        slice
    };
    let magic: [u8; 8] = read(8).try_into().unwrap();
    if &magic != &PRISM_MAGIC {
        return Err(format!("bad magic: {:?}", &magic));
    }
    let header = CimageHeader {
        magic,
        version: u32::from_le_bytes(read(4).try_into().unwrap()),
        segment_count: u32::from_le_bytes(read(4).try_into().unwrap()),
        payload_hash: read(32).try_into().unwrap(),
        num_layers: u32::from_le_bytes(read(4).try_into().unwrap()),
        num_heads: u32::from_le_bytes(read(4).try_into().unwrap()),
        head_dim: u32::from_le_bytes(read(4).try_into().unwrap()),
        hidden_dim: u32::from_le_bytes(read(4).try_into().unwrap()),
        intermediate_dim: u32::from_le_bytes(read(4).try_into().unwrap()),
        vocab_size: u32::from_le_bytes(read(4).try_into().unwrap()),
        quantization_schema: u32::from_le_bytes(read(4).try_into().unwrap()),
        draft_num_layers: u32::from_le_bytes(read(4).try_into().unwrap()),
        segments: {
            let mut arr = [SegmentEntry {
                kind: 0,
                offset: 0,
                length: 0,
            }; CIMAGE_SEGMENT_CAPACITY];
            for entry in arr.iter_mut() {
                entry.kind = u32::from_le_bytes(read(4).try_into().unwrap());
                entry.offset = u64::from_le_bytes(read(8).try_into().unwrap());
                entry.length = u64::from_le_bytes(read(8).try_into().unwrap());
            }
            arr
        },
        _pad: [0u8; 8],
    };
    Ok(header)
}

/// Encodes the type of content in a cimage segment.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    /// Compiled Metal kernel library (.metallib)
    MetalLib = 0,
    /// Ternary tile640 packed weights
    TernaryWeights = 1,
    /// FP16 block scales for ternary weights
    BlockScales = 2,
    /// CimageLayoutMeta — tensor records, per-layer metadata
    LayoutMeta = 3,
    /// Vocabulary / embedding table
    Vocabulary = 4,
    /// Apple ANE .mlmodelc or .ane.tar archive
    AneArchive = 5,
    /// Stride/prefetch topology table
    TopologyTable = 6,
    /// Compiled CUDA kernel blob (.cubin, .fatbin)
    CudaLib = 7,
    /// Compiled ROCm kernel blob (.co, .hsaco)
    RocmLib = 8,
    /// Compiled Level Zero / SPIR-V kernel (.spv)
    LevelZeroLib = 9,
    /// Compiled Vulkan shader (.spv)
    VulkanLib = 10,
    /// Intel NPU compiled model blob
    IntelNpuBlob = 11,
    /// AMD NPU (XDNA) compiled model blob
    AmdNpuBlob = 12,
    /// Qualcomm Hexagon / HTP compiled model
    QualcommNpuBlob = 13,
    /// Google TPU compiled model
    GoogleTpuBlob = 14,
    /// Compiled WebGPU / WGSL shader
    WebGpuLib = 15,
    /// Huawei Ascend NPU (DaVinci) compiled model (.om)
    HuaweiAscendBlob = 16,
    /// Hailo NPU compiled executable (.hef)
    HailoBlob = 17,
    /// Per-layer weight offset table (array of LayerDirectoryEntry).
    /// Enables layer-granular scheduling and ANE/GPU interleaving.
    LayerDirectory = 18,
    /// Packed ternary projection weight matrices for multimodal input adapters.
    /// Supports multiple logical tensors indexed via MultimodalInputDescriptor.
    MultimodalProjectionWeights = 19,
    /// FP16 block scales corresponding to MultimodalProjectionWeights.
    MultimodalProjectionScales = 20,
    /// Versioned binary descriptor (MultimodalInputDescriptorV1) describing
    /// modality support, tensor layout, and processor contract.
    MultimodalInputDescriptor = 21,
    /// Learned two-dimensional position embeddings for image patches.
    MultimodalPositionEmbeddings = 22,
    /// Biases, layer norms, pooling kernels, and small affine parameters
    /// that do not fit the primary projection matrix category.
    MultimodalAuxiliaryWeights = 23,
    /// Binary execution graph descriptor encoding per-layer DAG,
    /// device routing capabilities, KV compaction epochs, and MTP
    /// draft sub-graph references. Self-describing for the runtime.
    ExecutionGraph = 24,
    /// Tokenizer, multimodal special token map, audio codebook,
    /// and chat template. Type-tagged entries — see ModelArtifactEntry.
    ModelArtifacts = 25,
    /// Canonical NF4Tile640 packed weights (raw U8 resident bytes).
    Nf4Tile640Weights = 26,
    /// FP32 bias metadata corresponding to packed quantized weights.
    BlockBiases = 27,
    /// Multimodal projection biases, laid out **byte-parallel to
    /// MultimodalProjectionScales** ([tiles × 5] f32 per row): a projection
    /// record needs no independent bias offsets — its `scale_offset` /
    /// `scale_length` address this segment too. Gated per record by
    /// `ProjectionTensorRecord::FLAG_HAS_BIAS`; v1 artifacts (flags == 0)
    /// never take the bias path (kernels/MULTIMODAL_NF4_BIAS_ABI.md).
    MultimodalProjectionBiases = 28,
    /// JSON-serialized HeterogeneousExecutionImage for tri-lane execution.
    HeterogeneousImage = 29,
    /// TTS Talker (28-layer decoder) nf4tile640 packed weights
    TtsTalkerWeight = 30,
    /// TTS Talker nf4tile640 scales
    TtsTalkerScale = 31,
    /// TTS Talker nf4tile640 biases
    TtsTalkerBias = 32,
    /// TTS Code Predictor weights
    TtsCodePredictorWeight = 33,
    /// TTS Code Predictor scales
    TtsCodePredictorScale = 34,
    /// TTS Code Predictor biases
    TtsCodePredictorBias = 35,
    /// TTS Mimi Codec weights
    TtsCodecWeight = 36,
    /// TTS codebook embeddings (16 codebooks × 2048 entries × 128 dim)
    TtsCodebook = 37,
    /// Raw FP16 weights (for matrices that can't be nf4tile640-quantized).
    /// Loaded as-is, no dequantization needed.
    RawF16Weights = 38,
    /// Int8Tile640 packed weights (640-byte code stride per tile).
    Int8Tile640Weights = 39,
    /// Quantization sidecars — reduction-axis FP16 scale vectors and future
    /// residual metadata. Each entry maps to the sidecar_segment field of a
    /// MatrixWeightBinding.
    QuantizationSidecars = 40,
    /// Array of MatrixWeightBinding records — the per-tensor format contract
    /// between the compiler's packing pass and every runtime dispatch path.
    /// Binary: count (u32) followed by `count` MatrixWeightBinding structs.
    MatrixContract = 41,
}

/// One entry in the cimage segment directory.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SegmentEntry {
    pub kind: u32,
    pub offset: u64,
    pub length: u64,
}
/// Type tags for ModelArtifacts segment entries.
pub mod model_artifact_tag {
    pub const TOKENIZER: u32 = 0x01; // SentencePiece .model proto
    pub const TOKEN_MAP: u32 = 0x04; // Multimodal special token map (JSON)
    pub const CHAT_TEMPLATE: u32 = 0x05; // Chat prompt template string
    pub const GENERATION_CONFIG: u32 = 0x06; // Sampling params (JSON)
    /// Ternary-packed embedding table (nibbles reordered by cluster)
    pub const EMBED_NIBBLES: u32 = 0x10;
    /// FP16 block scales for the embedding table
    pub const EMBED_SCALES: u32 = 0x11;
    /// Ternary-packed centroid table (256 centroids × hidden_dim)
    pub const CENTROID_NIBBLES: u32 = 0x12;
    /// FP16 block scales for centroids
    pub const CENTROID_SCALES: u32 = 0x13;
    /// u32 cluster assignments (vocab_size entries)
    pub const CLUSTER_MAP: u32 = 0x14;
    /// FP16 per-layer RMSNorm weights (all layers × [input, post_attn] × hidden_dim)
    pub const AUX_NORMS: u32 = 0x15;
}

/// One entry in the ModelArtifacts segment. Flat binary: tag + length + data.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ModelArtifactEntry;

impl ModelArtifactEntry {
    pub const HEADER_SIZE: usize = 8;

    pub fn encode(tag: u32, data: &[u8], out: &mut Vec<u8>) {
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
    }

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

impl SegmentEntry {
    pub fn new(kind: SegmentKind, offset: u64, length: u64) -> Self {
        Self {
            kind: kind as u32,
            offset,
            length,
        }
    }
}

/// One entry in the layer directory — exact byte range of a single
/// transformer layer's packed weights, block scales, layer kind, and flags.
///
/// Enables the orchestrator to schedule per-layer Metal barriers and
/// run the ANE MTP model concurrently with GPU weight streaming.
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
    Prefill = 0,
    MtpDecode = 1,
    VisionEncoder = 2,
    #[default]
    Unknown = 0xFF,
}

/// Describes the I/O contract of an embedded ANE/NPU model segment.
/// Stored alongside AneArchive segments so the orchestrator can dispatch
/// without introspecting Core ML metadata at runtime.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AneModelDescriptor {
    pub role: u32,
    pub input_schema_digest: [u8; 32],
    pub output_schema_digest: [u8; 32],
    pub supports_stateful_decode: u8,
    pub max_sequence_length: u32,
    pub token_input_name_offset: u32,
    pub logits_output_name_offset: u32,
    pub _pad: [u8; 9],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CimageHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub segment_count: u32,
    pub payload_hash: [u8; 32],
    // Architecture
    pub num_layers: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub vocab_size: u32,
    pub quantization_schema: u32,
    /// Number of layers in the MTP draft decoder (0 = no draft model).
    pub draft_num_layers: u32,
    // Segment directory
    pub segments: [SegmentEntry; CIMAGE_SEGMENT_CAPACITY],
    pub _pad: [u8; 8],
}

impl CimageHeader {
    /// Look up a segment by kind. Returns the entry if found, None otherwise.
    pub fn segment(&self, kind: SegmentKind) -> Option<SegmentEntry> {
        let kind_u32 = kind as u32;
        self.segments.iter().find(|s| s.kind == kind_u32).copied()
    }

    pub fn is_nf4_tile640(&self) -> bool {
        self.quantization_schema == QUANT_SCHEMA_NF4_TILE640
    }

    pub fn primary_weight_segment_kind(&self) -> SegmentKind {
        if self.is_nf4_tile640() {
            SegmentKind::Nf4Tile640Weights
        } else {
            SegmentKind::TernaryWeights
        }
    }

    pub fn primary_weight_segment(&self) -> Option<SegmentEntry> {
        self.segment(self.primary_weight_segment_kind())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
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
}

impl TensorRecord {
    pub fn new(offset: u64, length: u64) -> Self {
        Self { offset, length }
    }
}

/// Backward-compatible type aliases
pub type PrismCimageHeader = CimageHeader;
pub type PrismCimageLayoutMeta = CimageLayoutMeta;
/// Backward-compatible function alias
/// V1 MatrixWeightBinding with canonical LE serialization.
pub const MATRIX_WEIGHT_BINDING_V1_BYTE_LENGTH: usize = 2 + 4 + 16 + 1 + 2 + 32
    + 4 + 4 + 2 + 4 + 2 + 1 + 1 + 1 + 8 + 8 + 4 + 1 + 8 + 8 + 2
    + 1 + 8 + 8 + 1 + 1 + 4 + 1 + 8 + 8 + 4;
// Verify: 2+4+16+1+2+32+4+4+2+4+2+1+1+1+8+8+4+1+8+8+2+1+8+8+1+1+4+1+8+8+4

#[repr(C)]
pub struct MatrixWeightBindingV1 {
    pub binding_wire_version: u16,
    pub matrix_id: u32,
    pub tensor_id: [u8; 16],
    pub representation: u8,
    pub representation_version: u16,
    pub kernel_abi_digest: [u8; 32],
    pub in_features: u32,
    pub out_features: u32,
    pub reduction_tile_size: u16,
    pub tiles_per_output_channel: u32,
    pub tail_reduction_count: u16,
    pub macro_layout: u8,
    pub tail_handling: u8,
    pub code_segment: u8,
    code_offset: u64,
    code_length: u64,
    code_tile_stride_bytes: u32,
    pub metadata_segment: u8,
    metadata_offset: u64,
    metadata_length: u64,
    metadata_tile_stride_bytes: u16,
    pub sidecar_segment: u8,
    sidecar_offset: u64,
    sidecar_length: u64,
    pub sidecar_kind: u8,
    pub sidecar_element_format: u8,
    pub sidecar_count: u32,
    pub residual_segment: u8,
    residual_offset: u64,
    residual_length: u64,
    pub required_alignment_bytes: u32,
}

/// Serialize a MatrixWeightBindingV1 in canonical little-endian format.
pub fn write_matrix_weight_binding_v1_le<W: std::io::Write>(
    w: &mut W,
    b: &MatrixWeightBindingV1,
) -> std::io::Result<()> {
    w.write_all(&b.binding_wire_version.to_le_bytes())?;
    w.write_all(&b.matrix_id.to_le_bytes())?;
    w.write_all(&b.tensor_id)?;
    w.write_all(&[b.representation])?;
    w.write_all(&b.representation_version.to_le_bytes())?;
    w.write_all(&b.kernel_abi_digest)?;
    w.write_all(&b.in_features.to_le_bytes())?;
    w.write_all(&b.out_features.to_le_bytes())?;
    w.write_all(&b.reduction_tile_size.to_le_bytes())?;
    w.write_all(&b.tiles_per_output_channel.to_le_bytes())?;
    w.write_all(&b.tail_reduction_count.to_le_bytes())?;
    w.write_all(&[b.macro_layout])?;
    w.write_all(&[b.tail_handling])?;
    w.write_all(&[b.code_segment])?;
    w.write_all(&b.code_offset.to_le_bytes())?;
    w.write_all(&b.code_length.to_le_bytes())?;
    w.write_all(&b.code_tile_stride_bytes.to_le_bytes())?;
    w.write_all(&[b.metadata_segment])?;
    w.write_all(&b.metadata_offset.to_le_bytes())?;
    w.write_all(&b.metadata_length.to_le_bytes())?;
    w.write_all(&b.metadata_tile_stride_bytes.to_le_bytes())?;
    w.write_all(&[b.sidecar_segment])?;
    w.write_all(&b.sidecar_offset.to_le_bytes())?;
    w.write_all(&b.sidecar_length.to_le_bytes())?;
    w.write_all(&[b.sidecar_kind])?;
    w.write_all(&[b.sidecar_element_format])?;
    w.write_all(&b.sidecar_count.to_le_bytes())?;
    w.write_all(&[b.residual_segment])?;
    w.write_all(&b.residual_offset.to_le_bytes())?;
    w.write_all(&b.residual_length.to_le_bytes())?;
    w.write_all(&b.required_alignment_bytes.to_le_bytes())?;
    Ok(())
}

/// Parse a MatrixWeightBindingV1 from a byte slice (canonical LE).
pub fn read_matrix_weight_binding_v1_le(data: &[u8]) -> Result<MatrixWeightBindingV1, String> {
    if data.len() < MATRIX_WEIGHT_BINDING_V1_BYTE_LENGTH {
        return Err(format!(
            "MatrixWeightBindingV1 too small: {} < {}",
            data.len(), MATRIX_WEIGHT_BINDING_V1_BYTE_LENGTH
        ));
    }
    let mut off = 0usize;
    let mut read = |n: usize| -> &[u8] { let s = &data[off..off+n]; off += n; s };
    let bv: u16 = u16::from_le_bytes(read(2).try_into().unwrap());
    if bv != 1 {
        return Err(format!("unknown MatrixWeightBindingV1 wire version: {}", bv));
    }
    let rep: u8 = read(1)[0];
    if rep > 3 {
        return Err(format!("unknown representation discriminant: {}", rep));
    }
    let rt: u16 = u16::from_le_bytes(read(2).try_into().unwrap());
    if rep <= 2 && rt != 640 {
        return Err(format!(
            "quantized format requires reduction_tile_size=640, got {}", rt
        ));
    }
    let ifeat: u32 = u32::from_le_bytes(read(4).try_into().unwrap());
    let ofeat: u32 = u32::from_le_bytes(read(4).try_into().unwrap());
    drop(read(4)); // tiles_per_output_channel (derivable)
    let _ = read(4); // tiles_per_output_channel (derivable)
    let trc: u16 = u16::from_le_bytes(read(2).try_into().unwrap());
    if rep <= 2 && trc != (ifeat % 640) as u16 {
        return Err("tail_reduction_count mismatch".into());
    }
    Ok(MatrixWeightBindingV1 {
        binding_wire_version: bv,
        matrix_id: u32::from_le_bytes(read(4).try_into().unwrap()),
        tensor_id: read(16).try_into().unwrap(),
        representation: rep,
        representation_version: u16::from_le_bytes(read(2).try_into().unwrap()),
        kernel_abi_digest: read(32).try_into().unwrap(),
        in_features: ifeat,
        out_features: ofeat,
        reduction_tile_size: rt,
        tiles_per_output_channel: u32::from_le_bytes(read(4).try_into().unwrap()),
        tail_reduction_count: trc,
        macro_layout: read(1)[0],
        tail_handling: read(1)[0],
        code_segment: read(1)[0],
        code_offset: u64::from_le_bytes(read(8).try_into().unwrap()),
        code_length: u64::from_le_bytes(read(8).try_into().unwrap()),
        code_tile_stride_bytes: u32::from_le_bytes(read(4).try_into().unwrap()),
        metadata_segment: read(1)[0],
        metadata_offset: u64::from_le_bytes(read(8).try_into().unwrap()),
        metadata_length: u64::from_le_bytes(read(8).try_into().unwrap()),
        metadata_tile_stride_bytes: u16::from_le_bytes(read(2).try_into().unwrap()),
        sidecar_segment: read(1)[0],
        sidecar_offset: u64::from_le_bytes(read(8).try_into().unwrap()),
        sidecar_length: u64::from_le_bytes(read(8).try_into().unwrap()),
        sidecar_kind: read(1)[0],
        sidecar_element_format: read(1)[0],
        sidecar_count: u32::from_le_bytes(read(4).try_into().unwrap()),
        residual_segment: read(1)[0],
        residual_offset: u64::from_le_bytes(read(8).try_into().unwrap()),
        residual_length: u64::from_le_bytes(read(8).try_into().unwrap()),
        required_alignment_bytes: u32::from_le_bytes(read(4).try_into().unwrap()),
    })
}

pub fn verify_prism_cimage(
    bytes: &[u8],
) -> Result<(PrismCimageHeader, PrismCimageLayoutMeta), String> {
    verify_cimage(bytes)
}

pub fn verify_cimage(bytes: &[u8]) -> Result<(CimageHeader, CimageLayoutMeta), String> {
    let header = read_cimage_header_le(bytes)?;
    // Try to find LayoutMeta segment in directory; return default if absent
    let layout = header
        .segment(SegmentKind::LayoutMeta)
        .and_then(|entry| {
            let end = (entry.offset as usize).checked_add(entry.length as usize)?;
            if end > bytes.len()
                || entry.length as usize != core::mem::size_of::<CimageLayoutMeta>()
            {
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

// ── Swizzled ternary re-pack for ANE Planar Engine gather ────────

/// Map linear (row, col) → (byte_offset, shift_within_byte).
#[inline(always)]
pub fn swizzled_byte_offset(row: usize, col: usize, width: usize) -> (usize, usize) {
    let bpr = width / 16;
    let br = row / 16;
    let bc = col / 16;
    let bi = br * bpr + bc;
    let ir = row % 16;
    let ic = col % 16;
    let ii = ir * 16 + ic;
    (bi * 64 + ii / 4, ii % 4)
}

/// Size of swizzled u8 buffer for tensor shape.
pub fn swizzled_buffer_size(rows: usize, cols: usize) -> usize {
    ((rows + 15) / 16) * ((cols + 15) / 16) * 64
}

/// Decode a u32 base-3 pack into an array of 20 ternary digits [0..2].
#[allow(dead_code)]
#[inline(always)]
fn decode_ternary_u32(packed: u32, digits: &mut [u8; 20]) {
    let mut rem = packed;
    for d in digits.iter_mut() {
        *d = (rem % 3) as u8;
        rem /= 3;
    }
}

/// Re-pack ternary u32 packs from DRAM into 16×16 swizzled u8 in SLC.
///
/// The ternary data uses the tile64 format: u32s at
///   offset = (row × num_tiles × 32 + tile × 32 + lane) × 4
/// Each u32 encodes 20 ternary values in base-3: digit 0→0, 1→+1, 2→-1.
///
/// The ANE reads the swizzled u8 from SLC and expands each quartet to
/// 4 INT8 values via the `gather` LUT (shape [81, 4]).  The scale
/// multiply also happens at gather time.
pub fn repack_ternary_to_swizzled_u8(
    ternary_bytes: &[u8],
    rows: usize,
    cols: usize,
    slc_buf: &mut [u8],
    slc_width: usize,
) {
    let expected = swizzled_buffer_size(rows, cols);
    if slc_buf.len() < expected {
        return;
    }
    slc_buf[..expected].fill(0);

    let ts = 640usize;
    let nt = (cols + ts - 1) / ts;

    // Accumulate quartets per SLC byte, then encode once all 4 slots fill
    let mut temp: Vec<[u8; 4]> = vec![[0u8; 4]; expected];
    let mut count: Vec<u8> = vec![0u8; expected];

    for row in 0..rows {
        for t in 0..nt {
            for lane in 0..32 {
                let po = row * nt * 32 * 4 + t * 32 * 4 + lane * 4;
                if po + 4 > ternary_bytes.len() {
                    break;
                }

                let packed = u32::from_le_bytes([
                    ternary_bytes[po],
                    ternary_bytes[po + 1],
                    ternary_bytes[po + 2],
                    ternary_bytes[po + 3],
                ]);

                let mut rem = packed;
                for vi in 0..20 {
                    let col = t * ts + lane * 20 + vi;
                    if col >= cols {
                        break;
                    }

                    let digit = (rem % 3) as u8;
                    rem /= 3;

                    let (byte_off, shift) = swizzled_byte_offset(row, col, slc_width);
                    if byte_off >= expected {
                        continue;
                    }

                    temp[byte_off][shift as usize] = digit;
                    count[byte_off] += 1;
                }
            }
        }
    }

    // Encode fully-filled quartets into base-3 state bytes
    for bi in 0..expected {
        if count[bi] == 4 {
            let q = &temp[bi];
            slc_buf[bi] = q[0] + q[1] * 3 + q[2] * 9 + q[3] * 27;
        } else if count[bi] > 0 {
            // Partial quartet at tensor edge — encode what's filled
            let mut state: u8 = 0;
            for s in (0..4).rev() {
                state = state * 3
                    + if s < count[bi] {
                        temp[bi][s as usize]
                    } else {
                        0
                    };
            }
            slc_buf[bi] = state;
        }
    }
}

// ── Ternary block quantizer ──────────────────────────────────────

fn f32_to_fp16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let s = ((bits >> 16) & 0x8000) as u16;
    let e = (bits >> 23) & 0xFF;
    let m = bits & 0x7FFFFF;
    if e == 0 {
        return s;
    }
    if e == 0xFF {
        return if m == 0 {
            if s != 0 {
                0xFC00
            } else {
                0x7C00
            }
        } else {
            0x7E00
        };
    }
    let ef = e as i32 - 127 + 15;
    if ef >= 0x1F {
        return if s != 0 { 0xFC00 } else { 0x7C00 };
    }
    if ef <= 0 {
        return s;
    }
    s | ((ef as u16) << 10) | ((m >> 13) as u16)
}

pub fn fp16_to_f32(b: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(b);
    let s = (((bits >> 15) & 1) as f32) * -2.0 + 1.0;
    let e = (bits >> 10) & 0x1F;
    let m = (bits & 0x03FF) as f32;
    if e == 0 {
        return if m == 0.0 {
            0.0
        } else {
            s * (m / 1024.0) * 2.0_f32.powi(-14)
        };
    }
    if e == 0x1F {
        return if m == 0.0 {
            if s > 0.0 {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            }
        } else {
            f32::NAN
        };
    }
    s * (1.0 + m / 1024.0) * 2.0_f32.powi(e as i32 - 15)
}

pub fn ternary_quantize_block(block: &[f32; 256]) -> ([u8; 2], [u8; 64]) {
    let max_mag = block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let scale = if max_mag > 1e-12 { max_mag } else { 1.0f32 };
    let su = f32_to_fp16_bits(scale);
    let mut nib = [0u8; 64];
    for (i, chk) in block.chunks_exact(4).enumerate() {
        let mut b: u8 = 0;
        for (j, &v) in chk.iter().enumerate() {
            let sn = (v / scale).round().clamp(-1.0, 1.0) as i8;
            b |= (match sn {
                1 => 0b01,
                -1 => 0b10,
                _ => 0b00,
            }) << (j * 2);
        }
        nib[i] = b;
    }
    (su.to_le_bytes(), nib)
}

pub fn generate_ane_swizzled_weights(raw_bf16: &[u8], out_dim: u32, in_dim: u32) -> Vec<u8> {
    let rows = out_dim as usize;
    let cols = in_dim as usize;
    let total = swizzled_buffer_size(rows, cols);
    if total == 0 {
        return Vec::new();
    }
    let mut swz = vec![0u8; total];
    let mut temp = vec![[0u8; 4]; total];
    let mut cnt = vec![0u8; total];

    let tv = rows * cols;
    let nb = (tv + 255) / 256;
    for bi in 0..nb {
        let st = bi * 256;
        let n = (tv - st).min(256);
        let mut blk = [0.0f32; 256];
        for j in 0..n {
            let bo = (st + j) * 2;
            if bo + 1 < raw_bf16.len() {
                blk[j] = f32::from_bits(
                    (u16::from_le_bytes([raw_bf16[bo], raw_bf16[bo + 1]]) as u32) << 16,
                );
            }
        }
        let (_sc, nib) = ternary_quantize_block(&blk);
        for j in 0..n {
            let d = match (nib[j / 4] >> ((j % 4) * 2)) & 0x03 {
                0b01 => 1,
                0b10 => 2,
                _ => 0,
            };
            let vi = st + j;
            let (bi2, sh) = swizzled_byte_offset(vi / cols, vi % cols, cols);
            temp[bi2][sh as usize] = d;
            cnt[bi2] += 1;
        }
    }
    for b in 0..total {
        if cnt[b] == 0 {
            continue;
        }
        let q = &temp[b];
        let mut s: u8 = 0;
        for sh in (0..4).rev() {
            s = s * 3 + if sh < cnt[b] { q[sh as usize] } else { 0 };
        }
        swz[b] = s;
    }
    swz
}

/// Requantize FP16 KV cache → swizzled u8 ternary format.
///
/// Reads FP16 KV values from the ANE's output surface (DRAM), quantizes
/// in 256-element blocks, packs 4 ternary digits per u8, and writes in
/// 16×16 block-swizzled order so the ANE Planar Engine `gather` LUT can
/// read it back.  The KV stays in DRAM as ternary packs until the next
/// ANE invocation needs it, at which point the E-core pumps it to SLC.
///
/// `fp16_kv`: raw FP16 bytes from KV cache (`seq_len * kv_dim * 2` bytes).
/// `seq_len`/`kv_dim`: shape of the KV cache slice being requantized.
/// `slc_buf`: pre-allocated output buffer (size = swizzled_buffer_size).
pub fn requantize_kv_to_swizzled_u8(
    fp16_kv: &[u8],
    seq_len: usize,
    kv_dim: usize,
    slc_buf: &mut [u8],
) {
    let total = seq_len * kv_dim;
    let nb = (total + 255) / 256;
    let expected = swizzled_buffer_size(seq_len, kv_dim);
    if slc_buf.len() < expected {
        return;
    }
    slc_buf[..expected].fill(0);

    let mut temp = vec![[0u8; 4]; expected];
    let mut cnt = vec![0u8; expected];

    for bi in 0..nb {
        let st = bi * 256;
        let n = (total - st).min(256);
        let mut blk = [0.0f32; 256];
        for j in 0..n {
            let bo = (st + j) * 2;
            if bo + 1 < fp16_kv.len() {
                let bits = u16::from_le_bytes([fp16_kv[bo], fp16_kv[bo + 1]]);
                blk[j] = fp16_to_f32(bits.to_le_bytes());
            }
        }
        let (_sc, nib) = ternary_quantize_block(&blk);
        for j in 0..n {
            let d = match (nib[j / 4] >> ((j % 4) * 2)) & 0x03 {
                0b01 => 1u8,
                0b10 => 2u8,
                _ => 0u8,
            };
            let vi = st + j;
            let (bi2, sh) = swizzled_byte_offset(vi / kv_dim, vi % kv_dim, kv_dim);
            temp[bi2][sh as usize] = d;
            cnt[bi2] += 1;
        }
    }
    for b in 0..expected {
        if cnt[b] == 0 {
            continue;
        }
        let q = &temp[b];
        let mut s: u8 = 0;
        for sh in (0..4).rev() {
            s = s * 3 + if sh < cnt[b] { q[sh as usize] } else { 0 };
        }
        slc_buf[b] = s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swizzle_bijection() {
        let w = 640;
        let h = 240;
        let tb = ((h + 15) / 16) * (w / 16) * 64;
        let mut seen = vec![[false; 4]; tb];
        for r in 0..h {
            for c in 0..w {
                let (b, sh) = swizzled_byte_offset(r, c, w);
                assert!(b < tb);
                assert!(!seen[b][sh]);
                seen[b][sh] = true;
            }
        }
        for slots in &seen {
            for &u in slots {
                assert!(u);
            }
        }
    }

    #[test]
    fn test_repack_roundtrip() {
        let cols = 640;
        let rows = 32;
        let nt = (cols + 639) / 640;
        // Build mock ternary data in GPU format
        let mut ternary = vec![0u8; rows * nt * 32 * 4];
        let mut expected_digits = vec![0u8; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let d = ((r * cols + c) % 3) as u8;
                expected_digits[r * cols + c] = d;
                // Set digit in the u32 pack
                let tile = c / 640;
                let lane = (c % 640) / 20;
                let vi = (c % 640) % 20;
                let po = r * nt * 32 * 4 + tile * 32 * 4 + lane * 4;
                if po + 4 > ternary.len() {
                    continue;
                }
                let mut pk = u32::from_le_bytes([
                    ternary[po],
                    ternary[po + 1],
                    ternary[po + 2],
                    ternary[po + 3],
                ]);
                let mut mul = 1u32;
                for _ in 0..vi {
                    mul *= 3;
                }
                pk = (pk / (mul * 3)) * (mul * 3) + d as u32 * mul + pk % mul;
                ternary[po..po + 4].copy_from_slice(&pk.to_le_bytes());
            }
        }

        let tb = swizzled_buffer_size(rows, cols);
        let mut slc = vec![0u8; tb];
        repack_ternary_to_swizzled_u8(&ternary, rows, cols, &mut slc, cols);

        let mut lut = [[0i8; 4]; 81];
        for s in 0u8..81 {
            let mut x = s;
            for j in 0..4 {
                lut[s as usize][j] = match x % 3 {
                    1 => 1,
                    2 => -1,
                    _ => 0,
                };
                x /= 3;
            }
        }

        for r in 0..rows {
            for c in 0..cols {
                let (b, sh) = swizzled_byte_offset(r, c, cols);
                let decoded = lut[slc[b] as usize][sh];
                let expected = match expected_digits[r * cols + c] {
                    1 => 1,
                    2 => -1,
                    _ => 0,
                };
                assert_eq!(decoded, expected, "Mismatch at ({r},{c})");
            }
        }
    }

    #[test]
    fn test_generate_ane_swizzled_weights() {
        let mut src = [0u8; 640 * 240 * 2];
        for i in 0..640 * 240 {
            let v = ((i as f32 * 1.618) % 6.0) - 3.0;
            let bits = (v.to_bits() >> 16) as u16;
            src[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
        let swz = generate_ane_swizzled_weights(&src, 240, 640);
        assert!(!swz.is_empty());
        let expected = swizzled_buffer_size(240, 640);
        assert_eq!(swz.len(), expected);
    }
    #[test]
    fn test_pump_smoke() {
        let rows = 32;
        let cols = 640;
        let nt = (cols + 639) / 640;
        let mut src = vec![0.0f32; rows * cols];
        for i in 0..rows * cols {
            src[i] = ((i as f32 * 1.618) % 6.0) - 3.0;
        }
        let mut ternary = vec![0u8; rows * nt * 32 * 4];
        let mut scales = Vec::new();
        let nb = (rows * cols + 255) / 256;
        for bi in 0..nb {
            let st = bi * 256;
            let n = (rows * cols - st).min(256);
            let mut blk = [0.0f32; 256];
            for j in 0..n {
                blk[j] = src[st + j];
            }
            let (sc, nib) = ternary_quantize_block(&blk);
            scales.push(sc);
            for j in 0..n {
                let d = match (nib[j / 4] >> ((j % 4) * 2)) & 0x03 {
                    0b01 => 1,
                    0b10 => 2,
                    _ => 0,
                };
                let vi = st + j;
                let po = (vi / cols) * nt * 32 * 4
                    + ((vi % cols) / 640) * 32 * 4
                    + (((vi % cols) % 640) / 20) * 4;
                if po + 4 > ternary.len() {
                    continue;
                }
                let mut pk = u32::from_le_bytes([
                    ternary[po],
                    ternary[po + 1],
                    ternary[po + 2],
                    ternary[po + 3],
                ]);
                let sub = (vi % cols) % 640 % 20;
                let mut mul = 1u32;
                for _ in 0..sub {
                    mul *= 3;
                }
                pk = (pk / (mul * 3)) * (mul * 3) + d as u32 * mul + pk % mul;
                ternary[po..po + 4].copy_from_slice(&pk.to_le_bytes());
            }
        }
        let tb = swizzled_buffer_size(rows, cols);
        let mut slc = vec![0u8; tb];
        repack_ternary_to_swizzled_u8(&ternary, rows, cols, &mut slc, cols);
        let mut lut = [[0i8; 4]; 81];
        for s in 0u8..81 {
            let mut x = s;
            for j in 0..4 {
                lut[s as usize][j] = match x % 3 {
                    1 => 1,
                    2 => -1,
                    _ => 0,
                };
                x /= 3;
            }
        }
        // Build LUT and decode
        let mut lut = [[0i8; 4]; 81];
        for s in 0u8..81 {
            let mut x = s;
            for j in 0..4 {
                lut[s as usize][j] = match x % 3 {
                    1 => 1,
                    2 => -1,
                    _ => 0,
                };
                x /= 3;
            }
        }
        // Build expected digits directly from the quantizer source
        let mut expected_i8 = vec![0i8; rows * cols];
        for vi in 0..rows * cols {
            let bi = vi / 256;
            let st = bi * 256;
            let mut blk = [0.0f32; 256];
            for j in 0..(rows * cols - st).min(256) {
                blk[j] = src[st + j];
            }
            let (sc, nib) = ternary_quantize_block(&blk);
            let _sc_f32 = fp16_to_f32(sc);
            let off = vi - st;
            let nibble = (nib[off / 4] >> ((off % 4) * 2)) & 0x03;
            expected_i8[vi] = match nibble {
                0b01 => 1,
                0b10 => -1,
                _ => 0,
            };
        }
        // Verify LUT-decoded values match expected
        let mut err = 0u32;
        for r in 0..rows {
            for c in 0..cols {
                let (b, sh) = swizzled_byte_offset(r, c, cols);
                let got = lut[slc[b] as usize][sh];
                let exp = expected_i8[r * cols + c];
                if got != exp {
                    err += 1;
                    if err <= 3 {
                        eprintln!("({r},{c}): got {got} exp {exp}");
                    }
                }
            }
        }
        assert_eq!(
            err, 0,
            "{err} mismatches — pure ternary digit mismatch, not FP16 precision"
        );
        eprintln!("[pump smoke] {rows}x{cols}: {} values match", rows * cols);
    }

    #[test]
    fn test_kv_requantizer_roundtrip() {
        let seq_len = 64;
        let kv_dim = 256;
        let mut kv = vec![0u8; seq_len * kv_dim * 2];
        for i in 0..seq_len * kv_dim {
            let v = ((i as f32 * 1.618) % 2.0) - 1.0;
            let bits = f32_to_fp16_bits(v);
            kv[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
        let mut swz = vec![0u8; swizzled_buffer_size(seq_len, kv_dim)];
        requantize_kv_to_swizzled_u8(&kv, seq_len, kv_dim, &mut swz);
        let mut lut = [[0i8; 4]; 81];
        for s in 0u8..81 {
            let mut x = s;
            for j in 0..4 {
                lut[s as usize][j] = match x % 3 {
                    1 => 1,
                    2 => -1,
                    _ => 0,
                };
                x /= 3;
            }
        }
        for i in 0..(seq_len * kv_dim).min(500) {
            let (b, sh) = swizzled_byte_offset(i / kv_dim, i % kv_dim, kv_dim);
            let got = lut[swz[b] as usize][sh];
            let bits = u16::from_le_bytes([kv[i * 2], kv[i * 2 + 1]]);
            let v = fp16_to_f32(bits.to_le_bytes());
            let bi = i / 256;
            let st = bi * 256;
            let mut max_v = 0.0f32;
            for j in st..(st + 256).min(seq_len * kv_dim) {
                let bj = u16::from_le_bytes([kv[j * 2], kv[j * 2 + 1]]);
                max_v = max_v.max(fp16_to_f32(bj.to_le_bytes()).abs());
            }
            let mag = if max_v > 1e-6 { max_v } else { 1.0 };
            let snapped = if v.abs() > mag * 0.5 {
                if v > 0.0 {
                    mag
                } else {
                    -mag
                }
            } else {
                0.0
            };
            let exp = (snapped / mag).round() as i8;
            assert!(
                (got - exp).abs() <= 1,
                "Mismatch at {i}: got {got} exp {exp} v={v}"
            );
        }
    }

    #[test]
    fn test_embed_ternary_roundtrip() {
        // Embed table: [vocab_size, hidden_dim] = [32000, 3840] typical
        let vocab = 512; // small for test speed
        let hd = 128;
        let mut embed = vec![0u8; (vocab * hd) as usize * 2];
        for i in 0..(vocab * hd) as usize {
            let v = ((i as f32 * 1.618) % 2.0) - 1.0;
            let bits = f32_to_fp16_bits(v);
            embed[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
        let swz = generate_ane_swizzled_weights(&embed, vocab, hd);
        assert_eq!(swz.len(), swizzled_buffer_size(vocab as usize, hd as usize));
        // Decode one row via LUT and verify it matches the original ternary snap
        let mut lut = [[0i8; 4]; 81];
        for s in 0u8..81 {
            let mut x = s;
            for j in 0..4 {
                lut[s as usize][j] = match x % 3 {
                    1 => 1,
                    2 => -1,
                    _ => 0,
                };
                x /= 3;
            }
        }
        // Pick token 42, decode its embedding row
        let row = 42;
        for c in 0..hd.min(32) {
            let col = c as usize;
            let (b, sh) = swizzled_byte_offset(row, col, hd as usize);
            let decoded = lut[swz[b] as usize][sh];
            let i = row * hd as usize + col;
            let bits = u16::from_le_bytes([embed[i * 2], embed[i * 2 + 1]]);
            let v = fp16_to_f32(bits.to_le_bytes());
            let bi = i / 256;
            let st = bi * 256;
            let end = (st + 256).min((vocab * hd) as usize);
            let mut max_v = 0.0f32;
            for j in st..end {
                let bj = u16::from_le_bytes([embed[j * 2], embed[j * 2 + 1]]);
                max_v = max_v.max(fp16_to_f32(bj.to_le_bytes()).abs());
            }
            let mag = if max_v > 1e-6 { max_v } else { 1.0 };
            let snapped = if v.abs() > mag * 0.5 {
                if v > 0.0 {
                    mag
                } else {
                    -mag
                }
            } else {
                0.0
            };
            let exp = (snapped / mag).round() as i8;
            assert!(
                (decoded - exp).abs() <= 1,
                "Embed mismatch at [{row},{col}]: got {decoded} exp {exp} v={v}"
            );
        }
    }

    #[test]
    fn layer_directory_round_trip() {
        let entry = LayerDirectoryEntry {
            weights_offset: 0x1000,
            weights_length: 0x28A_0000, // ~40.7 MB per layer (48 layers of 12B)
            scales_offset: 0x200,
            scales_length: 318_048, // ~318K scales per layer
            layer_kind: 0,          // decoder block
            flags: 0,
        };

        // Serialize to raw bytes
        let bytes: [u8; 48] = unsafe { std::mem::transmute(entry) };

        // Deserialize
        let round_tripped: LayerDirectoryEntry = unsafe { std::mem::transmute(bytes) };

        assert_eq!(round_tripped.weights_offset, entry.weights_offset);
        assert_eq!(round_tripped.weights_length, entry.weights_length);
        assert_eq!(round_tripped.scales_offset, entry.scales_offset);
        assert_eq!(round_tripped.scales_length, entry.scales_length);
        assert_eq!(round_tripped.layer_kind, entry.layer_kind);
        assert_eq!(round_tripped.flags, entry.flags);
    }

    #[test]
    fn layer_directory_entry_size() {
        assert_eq!(std::mem::size_of::<LayerDirectoryEntry>(), 48);
    }

    #[test]
    fn layer_directory_entry_default_zeroed() {
        let e = LayerDirectoryEntry::default();
        assert_eq!(e.weights_offset, 0);
        assert_eq!(e.weights_length, 0);
        assert_eq!(e.scales_offset, 0);
        assert_eq!(e.scales_length, 0);
        assert_eq!(e.layer_kind, 0);
        assert_eq!(e.flags, 0);
    }
}
