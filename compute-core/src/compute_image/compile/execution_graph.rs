//! Execution graph descriptor — per-layer DAG encoded in the cimage.
//!
//! The execution graph is a self-describing binary blob (segment 24)
//! that the runtime reads to understand the model's compute structure
//! without parsing config.json or reverse-engineering tensor names.

/// Magic bytes: "PRMEXEC1"
pub const EXECUTION_GRAPH_MAGIC: [u8; 8] = *b"PRMEXEC1";

/// Declares the per-element encoding of sidecar data payloads.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SidecarElementFormat {
    #[default]
    None = 0,
    F16 = 1,
    F32 = 2,
}

/// Declares the logical role of sidecar data.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SidecarKind {
    #[default]
    None = 0,
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
///
/// Each matrix gets its own binding after the admission pipeline selects a
/// format. Offsets are independent per segment so changing one matrix's
/// format never shifts another's address.
#[derive(Clone, Copy, Debug, Default)]
pub struct MatrixWeightBinding {
    /// Byte offset into weights_segment for this matrix's codes.
    pub weights_offset: u64,
    /// Byte length of codes data in weights_segment.
    pub weights_bytes: u64,
    /// Byte offset into tile_metadata_segment for this matrix's tile metadata.
    pub tile_metadata_offset: u64,
    /// Byte length of tile metadata data.
    pub tile_metadata_bytes: u64,
    /// Byte offset into sidecar_segment for reduction scales (0 = no sidecar).
    /// Sentinal: sidecar_kind == None means no sidecar (offset may be 0).
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
    /// Logical role of the sidecar data (SidecarKind discriminant).
    pub sidecar_kind: u8,
    /// Per-element encoding format (SidecarElementFormat discriminant).
    pub sidecar_element_format: u8,
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
    SlidingWindow = 0,
    FullAttention = 1,
}

/// Which backends a layer can execute on.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceCapability {
    None = 0,
    Gpu = 1,
    Ane = 2,
    Both = 3,
}
/// Kind of execution node in the graph.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// Standard decoder transformer layer
    DecoderLayer = 0,
    /// Vision patch embedding (patch_dense projection)
    VisionPatchEmbed = 1,
    /// Vision final projection into decoder space (embedding_projection)
    VisionFinalProjection = 2,
    /// Audio frame embedding
    AudioFrameEmbed = 3,
    /// Audio projection into decoder space
    AudioProjection = 4,
    /// Embedding assembly: merge text + vision + audio + position
    EmbeddingAssembly = 5,
    /// MTP draft decoder layer
    DraftLayer = 6,
    /// MTP pre-projection: draft hidden to main hidden
    DraftPreProjection = 7,
    /// MTP post-projection: main hidden to draft hidden
    DraftPostProjection = 8,
}

/// Per-layer execution node in the decoder DAG.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LayerExecutionNode {
    /// NodeKind discriminant.
    pub node_kind: u8,
    /// SWA or Full attention.
    pub attention_kind: u8,
    /// DeviceCapability.
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
    /// Tier 0: 0..tier_boundary[0], Tier 1: tier_boundary[0]..tier_boundary[1],
    /// Tier 2: tier_boundary[1]..
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
    pub magic: [u8; 8],
    pub version: u16,
    pub num_layers: u16,
    pub num_draft_layers: u16,
    pub num_compaction_epochs: u16,
    pub node_count: u32,
    pub _pad: [u8; 2],
    /// Per-layer execution nodes (num_layers entries).
    pub layers: Vec<LayerExecutionNode>,
    /// Compaction epochs (num_compaction_epochs entries).
    pub compaction_epochs: Vec<CompactionEpoch>,
    /// Draft sub-graph reference (None if no draft model).
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
        let has_draft: u8 = if self.draft_sub_graph.is_some() { 1 } else { 0 };
        // DraftSubGraph requires 8-byte alignment.  After the 1-byte flag the
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
        buf[10..12].copy_from_slice(&(self.num_layers as u16).to_le_bytes());
        // num_draft_layers packed into u8 (4 layers max)
        buf[12] = self.num_draft_layers as u8;
        buf[13] = self.num_compaction_epochs as u8;
        buf[14..18].copy_from_slice(&self.node_count.to_le_bytes());
        buf[18] = 0; // reserved
        buf[19] = 0; // reserved

        let mut off = 20;
        // Write layers
        for layer in &self.layers {
            unsafe {
                let ptr = buf.as_mut_ptr().add(off) as *mut LayerExecutionNode;
                ptr.write(*layer);
            }
            off += std::mem::size_of::<LayerExecutionNode>();
        }
        // Write compaction epochs
        for epoch in &self.compaction_epochs {
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
                Some(unsafe { std::ptr::read_unaligned(data.as_ptr().add(off) as *const _) })
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

    /// Build the execution graph for Gemma 4 12B Unified.
    pub fn gemma4_12b() -> Self {
        let swa_layers = [5u8, 11, 17, 23, 29, 35, 41, 47];
        let num_layers = 48u16;
        let mut layers = Vec::with_capacity(59); // 5 multimodal + 48 decoder + 2 projection + 4 draft

        // ── Multimodal preprocessing nodes ────────────────────────
        let vision_patch = LayerExecutionNode {
            node_kind: NodeKind::VisionPatchEmbed as u8,
            attention_kind: 2, // Projection (not attention)
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: 256,
            num_heads: 8,
            hidden_dim: 6912,               // patch_dense output: [3840, 6912]
            weight_offset: 0, // TODO: compute actual weight offsets from cimage layout
            weight_length: 3840 * 6912 / 4, // ternary nibbles
            scale_offset: 0,  // TODO: compute actual scale offsets from cimage layout
            _reserved: [0u8; 8],
        };
        let vision_proj = LayerExecutionNode {
            node_kind: NodeKind::VisionFinalProjection as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: 256,
            num_heads: 8,
            hidden_dim: 3840,
            weight_offset: 0, // TODO: compute actual weight offsets from cimage layout
            weight_length: 3840 * 3840 / 4,
            scale_offset: 0, // TODO: compute actual scale offsets from cimage layout
            _reserved: [0u8; 8],
        };
        let audio_embed = LayerExecutionNode {
            node_kind: NodeKind::AudioFrameEmbed as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: 256,
            num_heads: 8,
            hidden_dim: 2560,
            weight_offset: 0, // TODO: compute actual weight offsets from cimage layout
            weight_length: 128 * 2560 / 4,
            scale_offset: 0, // TODO: compute actual scale offsets from cimage layout
            _reserved: [0u8; 8],
        };
        let audio_proj = LayerExecutionNode {
            node_kind: NodeKind::AudioProjection as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: 256,
            num_heads: 8,
            hidden_dim: 3840,
            weight_offset: 0, // TODO: compute actual weight offsets from cimage layout
            weight_length: 2560 * 3840 / 4,
            scale_offset: 0, // TODO: compute actual scale offsets from cimage layout
            _reserved: [0u8; 8],
        };
        let assembly = LayerExecutionNode {
            node_kind: NodeKind::EmbeddingAssembly as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: 256,
            num_heads: 8,
            hidden_dim: 3840,
            weight_offset: 0, // no weights -- just buffer merge
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0u8; 8],
        };
        layers.push(vision_patch);
        layers.push(vision_proj);
        layers.push(audio_embed);
        layers.push(audio_proj);
        layers.push(assembly);

        // ── Decoder transformer layers ─────────────────────────────

        for i in 0u8..48 {
            let is_swa = swa_layers.contains(&i);
            let compaction_epoch = if is_swa {
                swa_layers.iter().position(|&x| x == i).unwrap() as u8
            } else {
                0xFF
            };
            layers.push(LayerExecutionNode {
                node_kind: NodeKind::DecoderLayer as u8,
                layer_index: i as u32,
                attention_kind: if is_swa {
                    AttentionKind::SlidingWindow as u8
                } else {
                    AttentionKind::FullAttention as u8
                },
                head_dim: if is_swa { 512 } else { 256 },
                num_heads: 8,
                hidden_dim: 3840,
                device_capability: DeviceCapability::Both as u8,
                compaction_epoch,
                weight_offset: 0, // filled by ingest
                scale_offset: 0,  // filled by ingest
                weight_length: 0, // filled by ingest
                _reserved: [0u8; 8],
            });
        }

        // 8 compaction epochs at full-attention layers
        let epochs: Vec<CompactionEpoch> = swa_layers
            .iter()
            .map(|&layer| CompactionEpoch {
                epoch_index: swa_layers.iter().position(|&x| x == layer).unwrap() as u8,
                trigger_layer: layer,
                tier_count: 3,
                _pad: 0,
                compression_ratio: [85, 51, 25, 0], // 3:1, 5:1, 10:1
                tier_boundaries: [4096, 32768, 262144],
                access_threshold: 100,
            })
            .collect();

        // ── MTP projection nodes ───────────────────────────────────
        let pre_proj_node = LayerExecutionNode {
            node_kind: NodeKind::DraftPreProjection as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: 128,
            num_heads: 8,
            hidden_dim: 3840,
            weight_offset: 0, // TODO: compute from cimage layout
            weight_length: 1024 * 3840 / 4,
            scale_offset: 0, // TODO: compute from cimage layout
            _reserved: [0u8; 8],
        };
        let post_proj_node = LayerExecutionNode {
            node_kind: NodeKind::DraftPostProjection as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: 256,
            num_heads: 8,
            hidden_dim: 1024,
            weight_offset: 0, // TODO: compute from cimage layout
            weight_length: 3840 * 1024 / 4,
            scale_offset: 0, // TODO: compute from cimage layout
            _reserved: [0u8; 8],
        };
        layers.push(pre_proj_node);
        layers.push(post_proj_node);

        // ── MTP draft decoder layers ───────────────────────────────
        for d in 0..4 {
            layers.push(LayerExecutionNode {
                node_kind: NodeKind::DraftLayer as u8,
                attention_kind: 1, // Full attention
                device_capability: DeviceCapability::Both as u8,
                compaction_epoch: 0xFF,
                layer_index: d,
                head_dim: 128,
                num_heads: 8,
                hidden_dim: 1024,
                weight_offset: 0, // TODO: compute from cimage layout
                weight_length: 0, // TODO: compute from cimage layout
                scale_offset: 0,  // TODO: compute from cimage layout
                _reserved: [0u8; 8],
            });
        }

        Self {
            magic: EXECUTION_GRAPH_MAGIC,
            version: 1,
            num_layers,
            num_draft_layers: 4,
            num_compaction_epochs: 8,
            _pad: [0u8; 2],
            node_count: 59, // 5 multimodal + 48 decoder + 2 projection + 4 draft
            layers,
            compaction_epochs: epochs,
            draft_sub_graph: Some(DraftSubGraph {
                num_layers: 4,
                hidden_dim: 1024,
                weight_offset: 0, // filled by ingest
                weight_length: 0,
                scale_offset: 0,
                scale_length: 0,
                pre_proj_offset: 0,
                post_proj_offset: 0,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_12b_graph_properties() {
        let graph = ExecutionGraphDescriptor::gemma4_12b();
        assert_eq!(graph.num_layers, 48);
        assert_eq!(graph.num_draft_layers, 4);
        assert_eq!(graph.num_compaction_epochs, 8);
        assert_eq!(graph.node_count, 59);
        assert_eq!(graph.layers.len(), 59); // 5 multimodal + 48 decoder + 2 MTP proj + 4 draft
        assert_eq!(graph.compaction_epochs.len(), 8);
        assert!(graph.draft_sub_graph.is_some());

        // Verify multimodal preprocessing nodes (indices 0..4)
        assert_eq!(graph.layers[0].node_kind, NodeKind::VisionPatchEmbed as u8);
        assert_eq!(graph.layers[0].hidden_dim, 6912);
        assert_eq!(
            graph.layers[1].node_kind,
            NodeKind::VisionFinalProjection as u8
        );
        assert_eq!(graph.layers[1].hidden_dim, 3840);
        assert_eq!(graph.layers[2].node_kind, NodeKind::AudioFrameEmbed as u8);
        assert_eq!(graph.layers[2].hidden_dim, 2560);
        assert_eq!(graph.layers[3].node_kind, NodeKind::AudioProjection as u8);
        assert_eq!(graph.layers[3].hidden_dim, 3840);
        assert_eq!(graph.layers[4].node_kind, NodeKind::EmbeddingAssembly as u8);
        assert_eq!(graph.layers[4].hidden_dim, 3840);
        for mm in 0..5 {
            assert_eq!(graph.layers[mm].attention_kind, 2); // Projection
            assert_eq!(
                graph.layers[mm].device_capability,
                DeviceCapability::Gpu as u8
            );
            assert_eq!(graph.layers[mm].layer_index, 0);
        }

        // Verify decoder SWA layers (indices 5..52)
        let swa_layers = [5u8, 11, 17, 23, 29, 35, 41, 47];
        for i in 0u8..48 {
            let layer = &graph.layers[(i + 5) as usize]; // skip 5 multimodal nodes
            assert_eq!(layer.node_kind, NodeKind::DecoderLayer as u8);
            assert_eq!(layer.layer_index, i as u32);
            let is_swa = swa_layers.contains(&i);
            assert_eq!(
                layer.attention_kind,
                if is_swa {
                    AttentionKind::SlidingWindow as u8
                } else {
                    AttentionKind::FullAttention as u8
                },
                "mismatch at layer {i}"
            );
            assert_eq!(layer.head_dim, if is_swa { 512 } else { 256 });
            assert_eq!(layer.hidden_dim, 3840);
            assert_eq!(layer.num_heads, 8);
            assert_eq!(layer.device_capability, DeviceCapability::Both as u8);
        }

        // Verify compaction epochs
        for (idx, epoch) in graph.compaction_epochs.iter().enumerate() {
            assert_eq!(epoch.epoch_index, idx as u8);
            assert_eq!(epoch.trigger_layer, swa_layers[idx]);
            assert_eq!(epoch.tier_count, 3);
            assert_eq!(epoch.compression_ratio[..3], [85, 51, 25]);
            assert_eq!(epoch.tier_boundaries, [4096, 32768, 262144]);
            assert_eq!(epoch.access_threshold, 100);
        }

        // Verify draft sub-graph
        let draft = graph.draft_sub_graph.as_ref().unwrap();
        assert_eq!(draft.num_layers, 4);
        assert_eq!(draft.hidden_dim, 1024);

        // Verify MTP draft layer nodes (indices 55..58, after decoder + 2 projection)
        for d in 0..4 {
            let node = &graph.layers[55 + d];
            assert_eq!(node.node_kind, NodeKind::DraftLayer as u8);
            assert_eq!(node.layer_index, d as u32);
            assert_eq!(node.head_dim, 128);
            assert_eq!(node.hidden_dim, 1024);
            assert_eq!(node.num_heads, 8);
            assert_eq!(node.attention_kind, 1); // Full attention
            assert_eq!(node.device_capability, DeviceCapability::Both as u8);
            assert_eq!(node.compaction_epoch, 0xFF);
        }

        // Verify pre-projection node (index 53)
        let pre = &graph.layers[53];
        assert_eq!(pre.node_kind, NodeKind::DraftPreProjection as u8);
        assert_eq!(pre.hidden_dim, 3840);
        assert_eq!(pre.attention_kind, 2);

        // Verify post-projection node (index 54)
        let post = &graph.layers[54];
        assert_eq!(post.node_kind, NodeKind::DraftPostProjection as u8);
        assert_eq!(post.hidden_dim, 1024);
        assert_eq!(post.attention_kind, 2);
    }

    #[test]
    fn to_bytes_roundtrip() {
        let graph = ExecutionGraphDescriptor::gemma4_12b();
        let bytes = graph.to_bytes();

        // Check magic
        assert_eq!(&bytes[0..8], EXECUTION_GRAPH_MAGIC);
        // Check version
        assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), 1);
        // Check layer count
        assert_eq!(u16::from_le_bytes([bytes[10], bytes[11]]), 48);
        assert_eq!(bytes[12], 4); // num_draft_layers
        assert_eq!(bytes[13], 8); // num_compaction_epochs
                                  // Check node_count
        assert_eq!(
            u32::from_le_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]),
            59
        );
        assert_eq!(bytes[18], 0); // reserved
        assert_eq!(bytes[19], 0); // reserved

        // Layout: 20 header + 59*48 layers + 8*24 epochs + 1 has_draft + padding + 56 draft
        let expected_min = 20 + 59 * 48 + 8 * 24 + 1;
        // After has_draft at offset = 20+2832+192 = 3044, so off=3045
        // Pad to 8-byte alignment: (8 - 3045%8) % 8 = (8-5)%8 = 3 bytes
        let expected_pad = 3;
        let expected_total = expected_min + expected_pad + 56;
        assert_eq!(bytes.len(), expected_total, "total size mismatch");

        // Verify has_draft flag at correct position
        let base = 20 + 59 * 48 + 8 * 24;
        assert_eq!(bytes[base], 1, "has_draft should be 1");

        // Spot-check multimodal vision_patch node (first node, offset 20)
        let vision_patch_offset = 20;
        assert_eq!(bytes[vision_patch_offset], NodeKind::VisionPatchEmbed as u8); // node_kind
        assert_eq!(bytes[vision_patch_offset + 1], 2); // attention_kind = Projection
        assert_eq!(bytes[vision_patch_offset + 2], DeviceCapability::Gpu as u8);
        assert_eq!(bytes[vision_patch_offset + 3], 0xFF); // compaction_epoch
        assert_eq!(
            u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            0
        ); // layer_index = 0
        assert_eq!(
            u16::from_le_bytes([
                bytes[vision_patch_offset + 8],
                bytes[vision_patch_offset + 9]
            ]),
            256
        ); // head_dim
        assert_eq!(
            u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]),
            6912
        ); // hidden_dim

        // Spot-check first decoder layer (node index 5, offset 20 + 5*48 = 260)
        let decoder0_offset = 20 + 5 * 48;
        assert_eq!(bytes[decoder0_offset], NodeKind::DecoderLayer as u8); // node_kind
        assert_eq!(
            bytes[decoder0_offset + 1],
            AttentionKind::FullAttention as u8
        ); // FullAttention
        assert_eq!(
            u32::from_le_bytes([
                bytes[decoder0_offset + 4],
                bytes[decoder0_offset + 5],
                bytes[decoder0_offset + 6],
                bytes[decoder0_offset + 7]
            ]),
            0
        ); // layer_index = 0
        assert_eq!(
            u16::from_le_bytes([bytes[decoder0_offset + 8], bytes[decoder0_offset + 9]]),
            256
        ); // head_dim
        assert_eq!(
            u32::from_le_bytes([
                bytes[decoder0_offset + 12],
                bytes[decoder0_offset + 13],
                bytes[decoder0_offset + 14],
                bytes[decoder0_offset + 15]
            ]),
            3840
        ); // hidden_dim

        // Spot-check a SWA decoder layer (i=5 -> node index 10, offset 20 + 10*48 = 500)
        let swa_decoder_offset = 20 + 10 * 48;
        assert_eq!(bytes[swa_decoder_offset], NodeKind::DecoderLayer as u8);
        assert_eq!(
            bytes[swa_decoder_offset + 1],
            AttentionKind::SlidingWindow as u8
        );
        assert_eq!(
            u32::from_le_bytes([
                bytes[swa_decoder_offset + 4],
                bytes[swa_decoder_offset + 5],
                bytes[swa_decoder_offset + 6],
                bytes[swa_decoder_offset + 7]
            ]),
            5
        ); // layer_index = 5
        assert_eq!(
            u16::from_le_bytes([bytes[swa_decoder_offset + 8], bytes[swa_decoder_offset + 9]]),
            512
        ); // head_dim

        // Spot-check a compaction epoch
        let epoch0_offset = 20 + 59 * 48;
        assert_eq!(bytes[epoch0_offset], 0); // epoch_index = 0
        assert_eq!(bytes[epoch0_offset + 1], 5); // trigger_layer = 5
        assert_eq!(bytes[epoch0_offset + 2], 3); // tier_count = 3

        // Spot-check a draft layer node (index 55, offset 20 + 55*48 = 2660)
        let draft0_offset = 20 + 55 * 48;
        assert_eq!(bytes[draft0_offset], NodeKind::DraftLayer as u8);
        assert_eq!(bytes[draft0_offset + 1], 1); // Full attention
        assert_eq!(
            u32::from_le_bytes([
                bytes[draft0_offset + 4],
                bytes[draft0_offset + 5],
                bytes[draft0_offset + 6],
                bytes[draft0_offset + 7]
            ]),
            0
        ); // layer_index = 0
        assert_eq!(
            u16::from_le_bytes([bytes[draft0_offset + 8], bytes[draft0_offset + 9]]),
            128
        ); // head_dim
        assert_eq!(
            u32::from_le_bytes([
                bytes[draft0_offset + 12],
                bytes[draft0_offset + 13],
                bytes[draft0_offset + 14],
                bytes[draft0_offset + 15]
            ]),
            1024
        ); // hidden_dim

        // Spot-check draft sub-graph
        let draft_offset = base + 1 + expected_pad;
        assert_eq!(
            u32::from_le_bytes([
                bytes[draft_offset],
                bytes[draft_offset + 1],
                bytes[draft_offset + 2],
                bytes[draft_offset + 3]
            ]),
            4
        ); // num_layers = 4
        assert_eq!(
            u32::from_le_bytes([
                bytes[draft_offset + 4],
                bytes[draft_offset + 5],
                bytes[draft_offset + 6],
                bytes[draft_offset + 7]
            ]),
            1024
        ); // hidden_dim = 1024
    }

    #[test]
    fn to_bytes_default() {
        let graph = ExecutionGraphDescriptor::default();
        let bytes = graph.to_bytes();
        assert_eq!(&bytes[0..8], EXECUTION_GRAPH_MAGIC);
        assert_eq!(bytes.len(), 21); // header 20 + has_draft flag 1
        assert_eq!(bytes[8..10], [1u8, 0]); // version=1
        assert_eq!(bytes[10..12], [0u8, 0]); // num_layers=0
        assert_eq!(bytes[12], 0); // num_draft_layers=0
        assert_eq!(bytes[13], 0); // num_compaction_epochs=0
        assert_eq!(
            u32::from_le_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]),
            0
        ); // node_count
        assert_eq!(bytes[18], 0); // reserved
        assert_eq!(bytes[19], 0); // reserved
    }

    #[test]
    fn default_has_correct_magic() {
        let graph = ExecutionGraphDescriptor::default();
        assert_eq!(graph.magic, EXECUTION_GRAPH_MAGIC);
        assert_eq!(graph.version, 1);
        assert!(graph.layers.is_empty());
        assert!(graph.compaction_epochs.is_empty());
        assert!(graph.draft_sub_graph.is_none());
    }

    #[test]
    fn layer_execution_node_is_repr_c() {
        // Verify expected field offsets via pointer arithmetic.
        // LayerExecutionNode should be 48 bytes.
        assert_eq!(std::mem::size_of::<LayerExecutionNode>(), 48);
        assert_eq!(std::mem::align_of::<LayerExecutionNode>(), 8);
    }

    #[test]
    fn compaction_epoch_is_repr_c() {
        assert_eq!(std::mem::size_of::<CompactionEpoch>(), 24);
        assert_eq!(std::mem::align_of::<CompactionEpoch>(), 4);
    }

    #[test]
    fn draft_sub_graph_is_repr_c() {
        assert_eq!(std::mem::size_of::<DraftSubGraph>(), 56);
        assert_eq!(std::mem::align_of::<DraftSubGraph>(), 8);
    }
}
