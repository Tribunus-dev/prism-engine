//! Execution graph descriptor — per-layer DAG encoded in the cimage.
//!
//! The execution graph is a self-describing binary blob (segment 24)
//! that the runtime reads to understand the model's compute structure
//! without parsing config.json or reverse-engineering tensor names.
//!
//! Authority: binary execution-graph descriptor format. Pure data + pure
//! (de)serialisation. Model-specific constructors (e.g. `gemma4_12b`)
//! live in the engine's `legacy_compute_image_compile/` directory.

/// Magic bytes: `"PRMEXEC1"`.
pub const EXECUTION_GRAPH_MAGIC: [u8; 8] = *b"PRMEXEC1";

/// Declares the per-element encoding of sidecar data payloads.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SidecarElementFormat {
    /// No sidecar data.
    #[default]
    None = 0,
    /// IEEE 754 half-precision floats.
    F16 = 1,
    /// IEEE 754 single-precision floats.
    F32 = 2,
}

/// Declares the logical role of sidecar data.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SidecarKind {
    /// No sidecar data.
    #[default]
    None = 0,
    /// Reduction-axis per-row scales.
    ReductionAxisScale = 1,
}

/// Compute the byte length of a sidecar payload from its declared element format.
pub fn sidecar_byte_len(count: u32, fmt: SidecarElementFormat) -> Option<u64> {
    let element_size = match fmt {
        SidecarElementFormat::None => 0,
        SidecarElementFormat::F16 => 2,
        SidecarElementFormat::F32 => 4,
    };
    (count as u64).checked_mul(element_size)
}

/// Per-matrix weight binding — the full contract between the compiler's
/// packing pass and every runtime dispatch path.
#[derive(Clone, Copy, Debug, Default)]
pub struct MatrixWeightBinding {
    /// Byte offset into `weights_segment` for this matrix's codes.
    pub weights_offset: u64,
    /// Byte length of codes data in `weights_segment`.
    pub weights_bytes: u64,
    /// Byte offset into `tile_metadata_segment` for this matrix's tile metadata.
    pub tile_metadata_offset: u64,
    /// Byte length of tile metadata data.
    pub tile_metadata_bytes: u64,
    /// Byte offset into `sidecar_segment` for reduction scales (0 = no sidecar).
    /// Sentinel: `sidecar_kind == None` means no sidecar (offset may be 0).
    pub sidecar_offset: u64,
    /// Number of sidecar values (element count, not bytes).
    pub sidecar_count: u32,
    /// Index into the bindings array.
    pub matrix_id: u32,
    /// RuntimeRepresentationClass discriminant.
    pub format: u8,
    /// SegmentKind value for the packed codes.
    pub weights_segment: u8,
    /// SegmentKind value for tile metadata (scales + biases).
    pub tile_metadata_segment: u8,
    /// SegmentKind value for the reduction-axis sidecar (0xFF = none).
    pub sidecar_segment: u8,
    /// Logical role of the sidecar data (`SidecarKind` discriminant).
    pub sidecar_kind: u8,
    /// Per-element encoding format (`SidecarElementFormat` discriminant).
    pub sidecar_element_format: u8,
    /// Reserved padding for `repr(C)` alignment.
    pub _pad: [u8; 3],
    /// Logical matrix dimensions.
    pub rows: u32,
    pub cols: u32,
    /// Tile columns per row of this matrix.
    pub tiles_per_row: u32,
}

/// Attention type for a decoder layer.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKind {
    /// Sliding-window attention.
    SlidingWindow = 0,
    /// Full attention (no window).
    FullAttention = 1,
}

/// Which backends a layer can execute on.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceCapability {
    /// No backend can run this node.
    None = 0,
    /// GPU backend.
    Gpu = 1,
    /// ANE / NPU backend.
    Ane = 2,
    /// Either GPU or ANE.
    Both = 3,
}

/// Kind of execution node in the graph.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// Standard decoder transformer layer.
    DecoderLayer = 0,
    /// Vision patch embedding (patch_dense projection).
    VisionPatchEmbed = 1,
    /// Vision final projection into decoder space.
    VisionFinalProjection = 2,
    /// Audio frame embedding.
    AudioFrameEmbed = 3,
    /// Audio projection into decoder space.
    AudioProjection = 4,
    /// Embedding assembly: merge text + vision + audio + position.
    EmbeddingAssembly = 5,
    /// MTP draft decoder layer.
    DraftLayer = 6,
    /// MTP pre-projection: draft hidden to main hidden.
    DraftPreProjection = 7,
    /// MTP post-projection: main hidden to draft hidden.
    DraftPostProjection = 8,
    /// MoE router: hidden → expert routing scores.
    MoERouter = 9,
    /// Single MoE expert MLP layer (gate+up+down proj).
    MoEExpertLayer = 10,
    /// MoE combine: weighted merge of top-K expert outputs.
    MoECombine = 11,
    /// DSpark semi-AR draft layer (confidence-scheduled).
    DSparkDraftLayer = 12,
    /// DSpark semi-AR draft pre-projection.
    DSparkDraftPreProjection = 13,
    /// DSpark semi-AR draft post-projection.
    DSparkDraftPostProjection = 14,
    /// LM head projection: hidden → logits.
    LmHead = 15,
}

/// Per-layer execution node in the decoder DAG.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LayerExecutionNode {
    /// `NodeKind` discriminant.
    pub node_kind: u8,
    /// SWA or Full attention.
    pub attention_kind: u8,
    /// `DeviceCapability`.
    pub device_capability: u8,
    /// Compaction epoch index (0xFF = none).
    pub compaction_epoch: u8,
    /// Layer index (0..num_layers-1 for decoder/draft layers).
    pub layer_index: u32,
    /// Head dimension for this layer.
    pub head_dim: u16,
    /// Number of attention heads.
    pub num_heads: u16,
    /// Hidden dimension for this node.
    pub hidden_dim: u32,
    /// Offset into TernaryWeights segment for this layer's weights.
    pub weight_offset: u64,
    /// Length of ternary weight nibbles for this layer.
    pub weight_length: u64,
    /// Offset into BlockScales segment for this layer's scales.
    pub scale_offset: u64,
    /// Reserved padding to reach 48 bytes.
    pub _reserved: [u8; 8],
}

/// A KV compaction epoch — compresses KV cache at a layer boundary.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CompactionEpoch {
    /// Epoch index (0..num_epochs-1).
    pub epoch_index: u8,
    /// Which layer triggers this compaction (KV written after this layer).
    pub trigger_layer: u8,
    /// Number of compression tiers (1=uniform, 3=progressive).
    pub tier_count: u8,
    /// Reserved padding.
    pub _pad: u8,
    /// Compression ratio per tier as numerator (ratio = ratio/256).
    /// e.g. 85 = ~3:1, 51 = ~5:1, 25 = ~10:1.
    pub compression_ratio: [u8; 4],
    /// Token count boundaries between tiers.
    pub tier_boundaries: [u32; 3],
    /// Access count threshold for adaptive policy (0 = disabled).
    pub access_threshold: u16,
}

/// Reference to the MTP draft decoder sub-graph.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DraftSubGraph {
    /// Number of layers in the draft decoder.
    pub num_layers: u32,
    /// Hidden dimension of the draft decoder.
    pub hidden_dim: u32,
    /// Offset into TernaryWeights segment for draft weights.
    pub weight_offset: u64,
    /// Length of draft ternary weights.
    pub weight_length: u64,
    /// Offset into BlockScales segment for draft scales.
    pub scale_offset: u64,
    /// Length of draft block scales.
    pub scale_length: u64,
    /// Pre-projection weight offset (draft hidden → main hidden, or 0 if none).
    pub pre_proj_offset: u64,
    /// Post-projection weight offset (main hidden → draft hidden, or 0 if none).
    pub post_proj_offset: u64,
}

/// Complete execution graph descriptor.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct ExecutionGraphDescriptor {
    /// Always [`EXECUTION_GRAPH_MAGIC`].
    pub magic: [u8; 8],
    /// Wire-format version.
    pub version: u16,
    /// Number of decoder layers.
    pub num_layers: u16,
    /// Number of MTP draft layers (0..=4).
    pub num_draft_layers: u16,
    /// Number of KV compaction epochs.
    pub num_compaction_epochs: u16,
    /// Total node count (decoder + multimodal + draft + projections + LM head).
    pub node_count: u32,
    /// Reserved padding to 8-byte alignment.
    pub _pad: [u8; 2],
    /// Per-layer execution nodes.
    pub layers: Vec<LayerExecutionNode>,
    /// Compaction epochs.
    pub compaction_epochs: Vec<CompactionEpoch>,
    /// Draft sub-graph reference (`None` if no draft model).
    pub draft_sub_graph: Option<DraftSubGraph>,
}

impl Default for ExecutionGraphDescriptor {
    fn default() -> Self {
        Self {
            magic: EXECUTION_GRAPH_MAGIC,
            version: 1,
            num_layers: 0,
            num_draft_layers: 0,
            num_compaction_epochs: 0,
            node_count: 0,
            _pad: [0u8; 2],
            layers: Vec::new(),
            compaction_epochs: Vec::new(),
            draft_sub_graph: None,
        }
    }
}

impl ExecutionGraphDescriptor {
    /// Serialize to a binary blob for embedding in the cimage.
    /// Format: header (20 bytes) + layer nodes + compaction epochs + draft ref.
    pub fn to_bytes(&self) -> Vec<u8> {
        let header_size = 20;
        let layer_size = self.layers.len() * std::mem::size_of::<LayerExecutionNode>();
        let epoch_size = self.compaction_epochs.len() * std::mem::size_of::<CompactionEpoch>();
        // has_draft flag: 1 byte
        let has_draft: u8 = u8::from(self.draft_sub_graph.is_some());
        // DraftSubGraph requires 8-byte alignment. After the 1-byte flag the
        // offset may be misaligned — reserve padding to restore it.
        let draft_align_pad = if self.draft_sub_graph.is_some() {
            // Compute padding needed after has_draft so the draft struct
            // starts on an 8-byte boundary.
            let base = header_size + layer_size + epoch_size + 1; // after has_draft
            (8 - (base % 8)) % 8
        } else {
            0
        };
        let draft_size = if self.draft_sub_graph.is_some() {
            draft_align_pad + std::mem::size_of::<DraftSubGraph>()
        } else {
            0
        };

        let total = header_size + layer_size + epoch_size + 1 + draft_size;
        let mut buf = vec![0u8; total];
        buf[0..8].copy_from_slice(&self.magic);
        buf[8..10].copy_from_slice(&self.version.to_le_bytes());
        buf[10..12].copy_from_slice(&self.num_layers.to_le_bytes());
        // num_draft_layers packed into u8 (4 layers max)
        buf[12] = self.num_draft_layers as u8;
        buf[13] = self.num_compaction_epochs as u8;
        buf[14..18].copy_from_slice(&self.node_count.to_le_bytes());
        buf[18] = 0; // reserved
        buf[19] = 0; // reserved

        let mut off = 20;
        // Write layers
        for layer in &self.layers {
            // SAFETY: buf is freshly allocated, large enough, and we keep
            // an exclusive borrow of `buf` here.
            unsafe {
                let ptr = buf.as_mut_ptr().add(off) as *mut LayerExecutionNode;
                ptr.write(*layer);
            }
            off += std::mem::size_of::<LayerExecutionNode>();
        }
        // Write compaction epochs
        for epoch in &self.compaction_epochs {
            // SAFETY: buf is large enough for all epochs (see `epoch_size`).
            unsafe {
                let ptr = buf.as_mut_ptr().add(off) as *mut CompactionEpoch;
                ptr.write(*epoch);
            }
            off += std::mem::size_of::<CompactionEpoch>();
        }
        // has_draft flag
        buf[off] = has_draft;
        off += 1;
        // Pad to 8-byte alignment for DraftSubGraph
        off += draft_align_pad;
        // Write draft sub-graph
        if let Some(draft) = &self.draft_sub_graph {
            // SAFETY: `off` is 8-byte aligned and `buf` has `draft_size` bytes
            // remaining at `off` (see `total` computation).
            unsafe {
                let ptr = buf.as_mut_ptr().add(off) as *mut DraftSubGraph;
                ptr.write_unaligned(*draft);
            }
        }
        buf
    }

    /// Deserialize from the binary blob embedded in the cimage.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 20 {
            return Err("execution graph too short".into());
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&data[0..8]);
        if magic != EXECUTION_GRAPH_MAGIC {
            return Err(format!("bad exec graph magic: {:?}", magic));
        }
        let version = u16::from_le_bytes([data[8], data[9]]);
        let num_layers = u16::from_le_bytes([data[10], data[11]]);
        let num_draft_layers = data[12] as u16;
        let num_compaction_epochs = data[13] as u16;
        let node_count = u32::from_le_bytes([data[14], data[15], data[16], data[17]]);

        let ln_size = std::mem::size_of::<LayerExecutionNode>();
        let ep_size = std::mem::size_of::<CompactionEpoch>();
        let mut off = 20;

        let mut layers = Vec::with_capacity(node_count as usize);
        for _ in 0..node_count {
            if off + ln_size > data.len() {
                break;
            }
            // SAFETY: bounds checked above; LayerExecutionNode is POD.
            let node: LayerExecutionNode =
                unsafe { std::ptr::read_unaligned(data.as_ptr().add(off) as *const _) };
            layers.push(node);
            off += ln_size;
        }

        let mut compaction_epochs = Vec::with_capacity(num_compaction_epochs as usize);
        for _ in 0..num_compaction_epochs {
            if off + ep_size > data.len() {
                break;
            }
            // SAFETY: bounds checked above; CompactionEpoch is POD.
            let epoch: CompactionEpoch =
                unsafe { std::ptr::read_unaligned(data.as_ptr().add(off) as *const _) };
            compaction_epochs.push(epoch);
            off += ep_size;
        }

        let has_draft = if off < data.len() { data[off] } else { 0 };
        off += 1;
        // Align to 8
        while off % 8 != 0 {
            off += 1;
        }

        let draft_sub_graph =
            if has_draft != 0 && off + std::mem::size_of::<DraftSubGraph>() <= data.len() {
                // SAFETY: bounds checked above; DraftSubGraph is POD.
                let draft: DraftSubGraph =
                    unsafe { std::ptr::read_unaligned(data.as_ptr().add(off) as *const _) };
                Some(draft)
            } else {
                None
            };

        Ok(Self {
            magic,
            version,
            num_layers,
            num_draft_layers,
            num_compaction_epochs,
            node_count,
            _pad: [0u8; 2],
            layers,
            compaction_epochs,
            draft_sub_graph,
        })
    }
}
