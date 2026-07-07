//! Runtime bridge between a compiled .cimage file and Metal GPU buffers.
//!
//! Loads a ternary-quantized .cimage file (produced by [`TernaryCImageCompiler`]),
//! parses the [`CImageHeader`] and [`CImageLayoutMeta`], verifies SHA-256 integrity
//! of the payload, and allocates Metal `MTLStorageModeShared` buffers for the
//! packed ternary weights, FP16 block scales, and optional embedding/norm/scalar
//! tensors (Prism Engine v2 format).
//!
//! Auto-detects the format version by magic bytes: the legacy v1 format uses
//! a 4-byte u32 magic (`CIMAGE_MAGIC` = 0x43494D47) and 64-byte layout metadata;
//! the Prism Engine v2 page-aligned format uses the 8-byte `PRISM_MAGIC`
//! (`*b"CIMAGE4\0"`) and creates Metal buffers for all six tensor sections.
//!
//! [`TernaryCImageCompiler`]: crate::compute_image::ternary_compile::TernaryCImageCompiler
//! [`TernaryCImageCompiler`]: crate::compute_image::compile::ternary::TernaryCImageCompiler

use crate::compute_image::compile::execution_graph::ExecutionGraphDescriptor;
use crate::compute_image::compile::ternary::{
    model_artifact_tag, verify_cimage, LayerDirectoryEntry, ModelArtifactEntry, SegmentEntry,
    SegmentKind, PRISM_MAGIC,
};
use crate::compute_image::megakernel::kernels::HIDDEN_DIM;
use crate::compute_image::multimodal::descriptor::{
    MultimodalArtifactSummary, MultimodalCapabilities, MultimodalInputDescriptorV1,
    ProjectionBackend, ProjectionPrecision, ProjectionTensorRecord,
};
use crate::compute_image::multimodal::SealedMultimodalBindings;
use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io;
use std::path::Path;

// O_ROWS and DOWN_ROWS are defined as private const in kernels.rs;
// declare local copies for use in fused interleave setup.
const O_ROWS: u32 = 4096;
const DOWN_ROWS: u32 = 15360;

// Re-export header types so callers only need `cimage_loader::CImageHeader`.
pub use crate::compute_image::compile::ternary::{PrismCimageHeader, PrismCimageLayoutMeta};

// ── V1 layout metadata (legacy format, kept for backward compat parsing) ─────
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct V1CImageLayoutMeta {
    pub mil_offset: u64,
    pub mil_size: u32,
    _pad0: [u8; 4],
    pub scales_offset: u64,
    pub scales_count: u32,
    _pad1: [u8; 4],
    pub weights_offset: u64,
    pub weights_count: u32,
    _pad2: [u8; 4],
    pub num_layers: u32,
    pub num_weights: u32,
    pub num_blocks: u32,
    _pad3: [u8; 4],
}

fn align64(n: u64) -> u64 {
    (n + 63) & !63
}

/// Load a `.cimage` file via mmap and parse its V3 page-aligned header.
///
/// Returns the mmap handle and the parsed [`CImageHeader`], which contains
/// segment offsets that are 16 KB aligned (guaranteed by
/// [`AlignedMmapBuilder`]). Callers can use these offsets with
/// [`ArenaView::from_mmap_slice`] to create page-aligned views into the
/// mmap'd data, avoiding kernel shadow copies during IOSurface creation.
///
/// The mmap handle keeps the file pages resident. Drop it to release.
///
/// [`ArenaView::from_mmap_slice`]: crate::backend::unified_arena::ArenaView::from_mmap_slice
/// [`AlignedMmapBuilder`]: crate::compute_image::cimage_packer::builder::AlignedMmapBuilder
pub fn load_cimage_mmap(path: &Path) -> io::Result<(Mmap, PrismCimageHeader)> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    if mmap.len() < std::mem::size_of::<PrismCimageHeader>() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            ".cimage file too small",
        ));
    }
    let header: PrismCimageHeader =
        unsafe { std::ptr::read_unaligned(mmap.as_ptr() as *const PrismCimageHeader) };
    if &header.magic != &PRISM_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad .cimage magic",
        ));
    }
    Ok((mmap, header))
}

/// Create a Metal buffer with SLC-bypass (non-temporal) hint for streaming weights.
/// This prevents GPU weight streaming from evicting the ANE's hot SLC lines.
/// On Apple Silicon with unified memory, `StorageModeShared` already uses the
/// write-combining path for GPU reads, and the compiler emits `LDNP`
/// (Load with Non-Temporal hint) instructions when it detects streaming access
/// patterns (e.g. large loops with no reuse).
pub fn new_slc_bypass_buffer(device: &metal::Device, data: &[u8]) -> metal::Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        data.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

///
/// Owns an in-memory copy of the file bytes and Metal shared-memory buffers
/// that the GPU can read directly.  Supports both the legacy v1 format
/// (ternary weights + block scales) and the Prism Engine v2 page-aligned
/// format that also provides embedding tables, norm weights, and per-layer
/// scalars as optional [`metal::Buffer`] handles.
pub struct CimageDeployment {
    /// Parsed binary header (magic, version, payload hash, …).
    /// In v2 mode this holds a best-effort default; the full v2 metadata
    /// lives in the owned [`Self::mmap_data`] bytes.
    pub header: crate::compute_image::manifest::CImageHeader,
    /// Parsed on-disk layout metadata (offsets, counts, dimensions).
    /// In v2 mode this holds a best-effort default; the full v2 layout
    /// lives in the owned [`Self::mmap_data`] bytes.
    pub layout: V1CImageLayoutMeta,
    /// Metal buffer containing the packed ternary weights (2-bit tri-level).
    pub weights_buffer: metal::Buffer,
    /// Metal buffer containing the FP16 block scales.
    pub scales_buffer: metal::Buffer,
    /// Metal buffer containing FP32 block biases for NF4Tile640 images.
    pub biases_buffer: Option<metal::Buffer>,
    /// INT4 block-quantized weights for M5+ Neural Accelerator direct consumption.
    /// Populated at load time by `maybe_expand_to_int4()`. None on M1-M4 or if expansion disabled.
    pub weights_int4_buffer: Option<metal::Buffer>,
    /// Fused interleaved INT4 weights buffer arranged in tile-interleaved order
    /// across all 7 per-layer matrices (Q, K, V, O, Gate, Up, Down).
    /// Populated at load time by `maybe_expand_to_int4()` on M5+.
    /// Improves SLC utilization on M5 Max by laying out tiles for contiguous
    /// GPU streaming access.
    pub fused_int4_buffer: Option<metal::Buffer>,
    /// FP16 embedding table reordered by cluster (vocab_size × hidden_dim), v2 format.
    pub embed_buffer: Option<metal::Buffer>,
    /// FP16 block scales for ternary-quantized embedding table (1 per 256 weights), v2 format.
    /// Split from the unified block_scales section at load time.
    pub embed_scales_buffer: Option<metal::Buffer>,
    /// FP16 block scales for ternary-quantized centroid vectors, v2 format.
    /// Appended after embed_scales in the unified block_scales section.
    pub centroid_scales_buffer: Option<metal::Buffer>,
    /// Ternary-packed centroid vectors (u32), v2 format.
    pub centroid_buffer: Option<metal::Buffer>,
    /// u32 cluster assignments (vocab_size entries), v2 format.
    pub cluster_map_buffer: Option<metal::Buffer>,
    /// FP16 norm weights (input, post_attn, pre_ffn, post_ffn, q_norm,
    /// k_norm × num_layers, plus final norm), present in v2 format (from aux section).
    pub norms_buffer: Option<metal::Buffer>,
    /// FP16 per-layer scalars (num_layers × 2 bytes), present in v2 format (from aux section).
    pub scalars_buffer: Option<metal::Buffer>,
    /// Sealed multimodal projection weights segment, when present.
    pub multimodal_projection_weights_buffer: Option<metal::Buffer>,
    /// Sealed multimodal projection scales segment, when present.
    pub multimodal_projection_scales_buffer: Option<metal::Buffer>,
    /// Resolved `MultimodalProjectionBiases` segment (byte-parallel to the
    /// scales segment). `None` on v1 artifacts and text-only cimages — the
    /// runner then takes the documented zero-bias fallback
    /// (kernels/MULTIMODAL_NF4_BIAS_ABI.md).
    pub multimodal_projection_biases_buffer: Option<metal::Buffer>,
    /// Sealed multimodal position embeddings segment, when present.
    pub multimodal_position_embeddings_buffer: Option<metal::Buffer>,
    /// Sealed multimodal auxiliary weights segment, when present.
    pub multimodal_auxiliary_weights_buffer: Option<metal::Buffer>,
    /// ANE MIL program binary (from aux section tail), v2 format.
    pub mil_buffer: Option<metal::Buffer>,
    /// Pre-compiled Metal kernel library (.metallib) embedded in the aux section
    /// tail, after norms + scalars + MIL.  `None` when no metallib is present
    /// (runtime falls back to JIT compilation from source).
    pub metallib_buffer: Option<metal::Buffer>,
    /// Pre-compiled ANE compaction model bytes (model.mlmodel protobuf from
    /// .mlmodelc bundle), embedded in the aux section tail after the metallib.
    /// Present in .cimage v2 format when compilation included ANE compaction.
    /// At runtime the orchestrator writes these bytes to a temp .mlmodelc
    /// directory and loads via CoreAiModel, avoiding ~3s JIT compilation.
    pub compaction_model_bytes: Option<Vec<u8>>,
    /// Compiled ANE prefill model bytes (model.mlmodel protobuf from
    /// .mlmodelc bundle), embedded in the aux section tail after the
    /// compaction model. Present in .cimage v2 format when compilation
    /// included ANE prefill. At runtime the orchestrator writes these
    /// bytes to a temp .mlmodelc directory and loads via CoreAiModel,
    /// avoiding JIT compilation at startup.
    pub prefill_model_bytes: Option<Vec<u8>>,
    /// Total number of weights (original count before 2-bit packing).
    pub num_weights: u64,
    /// Number of transformer layers.
    pub num_layers: u32,
    /// Owned copy of the full file bytes (keeps backing memory live for the
    /// duration of the deployment).
    pub mmap_data: Vec<u8>,
}

impl CimageDeployment {
    /// Return the sealed cimage header when this deployment uses the v2
    /// Prism format, otherwise `None`.
    pub fn prism_header(&self) -> Option<PrismCimageHeader> {
        if self.mmap_data.len() < std::mem::size_of::<PrismCimageHeader>() {
            return None;
        }
        verify_cimage(&self.mmap_data)
            .ok()
            .map(|(header, _)| header)
    }

    /// True when the deployment carries the explicit NF4Tile640 shared-weight
    /// schema rather than the legacy ternary schema.
    pub fn is_nf4_tile640(&self) -> bool {
        self.prism_header()
            .map(|header| header.is_nf4_tile640())
            .unwrap_or(false)
    }

    /// Return the FP32 bias buffer required by the NF4Tile640 runtime path.
    pub fn require_nf4_biases(&self) -> Result<&metal::Buffer, String> {
        self.biases_buffer
            .as_ref()
            .ok_or_else(|| "NF4Tile640 deployment missing BlockBiases segment".to_string())
    }

    /// Decode the sealed LayerDirectory segment into typed entries.
    pub fn layer_directory_entries(&self) -> Result<Vec<LayerDirectoryEntry>, String> {
        let header = self
            .prism_header()
            .ok_or_else(|| "LayerDirectory unavailable for legacy cimage format".to_string())?;
        let Some(seg) = header
            .segments
            .iter()
            .find(|seg| seg.kind == SegmentKind::LayerDirectory as u32 && seg.length > 0)
        else {
            return Ok(Vec::new());
        };

        let entry_size = std::mem::size_of::<LayerDirectoryEntry>();
        if seg.length as usize % entry_size != 0 {
            return Err(format!(
                "LayerDirectory byte length {} is not a multiple of {}",
                seg.length, entry_size
            ));
        }

        let start = seg.offset as usize;
        let end = start + seg.length as usize;
        if end > self.mmap_data.len() {
            return Err("LayerDirectory extends past end of .cimage mmap".into());
        }

        let mut entries = Vec::with_capacity(seg.length as usize / entry_size);
        let bytes = &self.mmap_data[start..end];
        for chunk in bytes.chunks_exact(entry_size) {
            let entry =
                unsafe { std::ptr::read_unaligned(chunk.as_ptr() as *const LayerDirectoryEntry) };
            entries.push(entry);
        }
        Ok(entries)
    }

    /// Decode the sealed ExecutionGraph segment into a typed descriptor.
    pub fn execution_graph(&self) -> Result<Option<ExecutionGraphDescriptor>, String> {
        let header = self
            .prism_header()
            .ok_or_else(|| "ExecutionGraph unavailable for legacy cimage format".to_string())?;
        let Some(seg) = header
            .segments
            .iter()
            .find(|seg| seg.kind == SegmentKind::ExecutionGraph as u32 && seg.length > 0)
        else {
            return Ok(None);
        };

        let start = seg.offset as usize;
        let end = start + seg.length as usize;
        if end > self.mmap_data.len() {
            return Err("ExecutionGraph extends past end of .cimage mmap".into());
        }

        ExecutionGraphDescriptor::from_bytes(&self.mmap_data[start..end])
            .map(Some)
            .map_err(|e| format!("decode ExecutionGraph: {e}"))
    }

    /// Load a `.cimage` file, auto-detecting the format version, verifying
    /// integrity, and creating Metal shared buffers.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the file is too small, has a bad magic number, fails
    /// SHA-256 verification, or contains an inconsistent layout.
    pub fn load(path: impl AsRef<Path>, device: &metal::Device) -> Result<Self, String> {
        let bytes =
            std::fs::read(path.as_ref()).map_err(|e| format!("failed to read .cimage: {}", e))?;

        // Check for v2 (Prism Engine) format magic
        if bytes.len() >= 8 && &bytes[0..8] == &PRISM_MAGIC {
            return Self::load_v2(bytes, device);
        }

        // Fall through to existing v1 parsing
        Self::load_v1(bytes, device)
    }

    /// Load a legacy v1 `.cimage` file (ternary weights + block scales only).
    fn load_v1(bytes: Vec<u8>, device: &metal::Device) -> Result<Self, String> {
        if bytes.len() < 192 {
            return Err(format!(
                "cimage too small: {} bytes (need >= 192)",
                bytes.len()
            ));
        }

        // ── Parse header (first 128 bytes, #[repr(C, align(64))]) ──────────
        type ManHeader = crate::compute_image::manifest::CImageHeader;
        let header: ManHeader =
            unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const ManHeader) };

        if header.magic != crate::compute_image::manifest::CIMAGE_MAGIC {
            return Err(format!(
                "bad magic: 0x{:08X} (expected 0x{:08X})",
                header.magic,
                crate::compute_image::manifest::CIMAGE_MAGIC
            ));
        }

        // ── Parse layout metadata (bytes 128..192) ────────────────────────
        let layout: V1CImageLayoutMeta = unsafe {
            std::ptr::read_unaligned(bytes.as_ptr().add(128) as *const V1CImageLayoutMeta)
        };

        // ── Verify SHA-256 hash of payload (everything after the header) ──
        let payload = &bytes[128..];
        let computed = Sha256::digest(payload);
        if computed.as_slice() != header.payload_hash {
            return Err("SHA-256 hash mismatch: payload corrupted".into());
        }

        // Expected total file size = header (128) + layout (64) + aligned sections
        let expected_total = 128u64
            + 64u64 // layout meta
            + align64(layout.mil_size as u64)
            + align64(layout.scales_count as u64 * 2)
            + align64(layout.weights_count as u64);
        if (bytes.len() as u64) < expected_total {
            return Err(format!(
                "file truncated: {} bytes, expected >= {}",
                bytes.len(),
                expected_total
            ));
        }

        // ── Extract sub-slices from the backing bytes ────────────────────
        let scales_start = layout.scales_offset as usize;
        let scales_len = (layout.scales_count as usize) * 2; // FP16 → 2 bytes per value
        let scales_end = scales_start + scales_len;
        if scales_end > bytes.len() {
            return Err(format!(
                "scales section out of range: offset={} len={} file_size={}",
                layout.scales_offset,
                scales_len,
                bytes.len()
            ));
        }

        let weights_start = layout.weights_offset as usize;
        let weights_len = layout.weights_count as usize; // packed ternary bytes
        let weights_end = weights_start + weights_len;
        if weights_end > bytes.len() {
            return Err(format!(
                "weights section out of range: offset={} len={} file_size={}",
                layout.weights_offset,
                weights_len,
                bytes.len()
            ));
        }

        // ── Create Metal shared-memory buffers and copy data in ──────────
        let scales_buffer = device.new_buffer_with_data(
            bytes[scales_start..scales_end].as_ptr() as *const std::ffi::c_void,
            scales_len as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let weights_buffer = device.new_buffer_with_data(
            bytes[weights_start..weights_end].as_ptr() as *const std::ffi::c_void,
            weights_len as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let num_weights = layout.num_weights as u64;
        let num_layers = layout.num_layers;

        Ok(Self {
            header,
            layout,
            weights_buffer,
            scales_buffer,
            biases_buffer: None,
            embed_buffer: None,
            centroid_buffer: None,
            centroid_scales_buffer: None,
            cluster_map_buffer: None,
            norms_buffer: None,
            scalars_buffer: None,
            multimodal_projection_weights_buffer: None,
            multimodal_projection_scales_buffer: None,
            multimodal_projection_biases_buffer: None,
            multimodal_position_embeddings_buffer: None,
            multimodal_auxiliary_weights_buffer: None,
            mil_buffer: None,
            metallib_buffer: None,
            compaction_model_bytes: None,
            prefill_model_bytes: None,
            embed_scales_buffer: None,
            weights_int4_buffer: None,
            fused_int4_buffer: None,
            num_weights,
            num_layers,
            mmap_data: bytes,
        })
    }

    /// Load a Prism Engine v2 page-aligned .cimage file.
    ///
    /// Parses [`PrismCimageHeader`] and [`PrismCimageLayoutMeta`], verifies
    /// SHA-256, then creates Metal `StorageModeShared` buffers for all five
    /// tensor sections (embed_ternary, embed_scales, ternary_weights, block_scales, norms,
    /// scalars).
    fn load_v2(bytes: Vec<u8>, device: &metal::Device) -> Result<Self, String> {
        let (header, _layout) = verify_cimage(&bytes)?;

        // ══ Segment-directory based loader (avoids CimageLayoutMeta) ══
        let find_seg = |kind: u32| -> Result<&SegmentEntry, String> {
            header
                .segments
                .iter()
                .find(|s| s.kind == kind && s.length > 0)
                .ok_or_else(|| format!("segment kind {} not found", kind))
        };
        let sg0 = find_seg(SegmentKind::MetalLib as u32)?;
        let weight_kind = if header.quantization_schema
            == crate::compute_image::compile::ternary::QUANT_SCHEMA_NF4_TILE640
        {
            SegmentKind::Nf4Tile640Weights as u32
        } else {
            SegmentKind::TernaryWeights as u32
        };
        let sg1 = find_seg(weight_kind)?;
        let sg2 = find_seg(SegmentKind::BlockScales as u32)?;
        let sg_bias = header
            .segments
            .iter()
            .find(|s| s.kind == SegmentKind::BlockBiases as u32 && s.length > 0);

        let mk_buf = |seg: &SegmentEntry| -> metal::Buffer {
            let off = seg.offset as usize;
            let len = seg.length as usize;
            let src = &bytes[off..off + len];
            device.new_buffer_with_data(
                src.as_ptr() as *const std::ffi::c_void,
                len as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };

        let num_weights = if header.quantization_schema
            == crate::compute_image::compile::ternary::QUANT_SCHEMA_NF4_TILE640
        {
            (sg1.length / 320) * 640
        } else {
            (sg2.length / 2) * 256
        };

        // ── Read auxiliary buffers from ModelArtifacts segment ────────
        let model_artifacts_seg = header
            .segments
            .iter()
            .find(|s| s.kind == SegmentKind::ModelArtifacts as u32 && s.length > 0);
        let (
            embed_buffer,
            embed_scales_buffer,
            centroid_buffer,
            centroid_scales_buffer,
            cluster_map_buffer,
            norms_buffer,
        ) = match model_artifacts_seg {
            Some(seg) => {
                let off = seg.offset as usize;
                let len = seg.length as usize;
                let data = &bytes[off..off + len];
                let mut embed = None;
                let mut escale = None;
                let mut cent = None;
                let mut cscale = None;
                let mut cmap = None;
                let mut norms = None;
                for (tag, payload) in ModelArtifactEntry::iter_entries(data) {
                    match tag {
                        t if t == model_artifact_tag::EMBED_NIBBLES => {
                            embed = Some(device.new_buffer_with_data(
                                payload.as_ptr() as *const _,
                                payload.len() as u64,
                                metal::MTLResourceOptions::StorageModeShared,
                            ))
                        }
                        t if t == model_artifact_tag::EMBED_SCALES => {
                            escale = Some(device.new_buffer_with_data(
                                payload.as_ptr() as *const _,
                                payload.len() as u64,
                                metal::MTLResourceOptions::StorageModeShared,
                            ))
                        }
                        t if t == model_artifact_tag::CENTROID_NIBBLES => {
                            cent = Some(device.new_buffer_with_data(
                                payload.as_ptr() as *const _,
                                payload.len() as u64,
                                metal::MTLResourceOptions::StorageModeShared,
                            ))
                        }
                        t if t == model_artifact_tag::CENTROID_SCALES => {
                            cscale = Some(device.new_buffer_with_data(
                                payload.as_ptr() as *const _,
                                payload.len() as u64,
                                metal::MTLResourceOptions::StorageModeShared,
                            ))
                        }
                        t if t == model_artifact_tag::CLUSTER_MAP => {
                            cmap = Some(device.new_buffer_with_data(
                                payload.as_ptr() as *const _,
                                payload.len() as u64,
                                metal::MTLResourceOptions::StorageModeShared,
                            ))
                        }
                        t if t == model_artifact_tag::AUX_NORMS => {
                            norms = Some(device.new_buffer_with_data(
                                payload.as_ptr() as *const _,
                                payload.len() as u64,
                                metal::MTLResourceOptions::StorageModeShared,
                            ))
                        }
                        _ => {}
                    }
                }
                (embed, escale, cent, cscale, cmap, norms)
            }
            None => (None, None, None, None, None, None),
        };

        // For format v6+, embedding/centroid/norm segments are required
        if header.version >= 6 {
            let required_tags = [
                (model_artifact_tag::EMBED_NIBBLES, "EMBED_NIBBLES"),
                (model_artifact_tag::EMBED_SCALES, "EMBED_SCALES"),
                (model_artifact_tag::CENTROID_NIBBLES, "CENTROID_NIBBLES"),
                (model_artifact_tag::CENTROID_SCALES, "CENTROID_SCALES"),
                (model_artifact_tag::CLUSTER_MAP, "CLUSTER_MAP"),
                (model_artifact_tag::AUX_NORMS, "AUX_NORMS"),
            ];
            match model_artifacts_seg {
                Some(seg) => {
                    let off = seg.offset as usize;
                    let len = seg.length as usize;
                    let data = &bytes[off..off + len];
                    for (tag, name) in &required_tags {
                        let found = ModelArtifactEntry::iter_entries(data).any(|(t, _)| t == *tag);
                        if !found {
                            return Err(format!(
                                "v{} .cimage missing required model artifact tag {} (0x{:02X})",
                                header.version, name, tag
                            ));
                        }
                    }
                }
                None => {
                    return Err(format!(
                        "v{} .cimage missing ModelArtifacts segment (required for v6+)",
                        header.version
                    ))
                }
            }
        }

        Ok(Self {
            header: crate::compute_image::manifest::CImageHeader::default(),
            layout: unsafe { std::mem::zeroed() },
            weights_buffer: mk_buf(sg1),
            scales_buffer: mk_buf(sg2),
            biases_buffer: sg_bias.map(mk_buf),
            weights_int4_buffer: None,
            fused_int4_buffer: None,
            embed_buffer,
            embed_scales_buffer,
            centroid_scales_buffer,
            centroid_buffer,
            cluster_map_buffer,
            norms_buffer,
            scalars_buffer: None,
            multimodal_projection_weights_buffer: header
                .segment(SegmentKind::MultimodalProjectionWeights)
                .map(|seg| mk_buf(&seg)),
            multimodal_projection_scales_buffer: header
                .segment(SegmentKind::MultimodalProjectionScales)
                .map(|seg| mk_buf(&seg)),
            multimodal_projection_biases_buffer: header
                .segment(SegmentKind::MultimodalProjectionBiases)
                .map(|seg| mk_buf(&seg)),
            multimodal_position_embeddings_buffer: header
                .segment(SegmentKind::MultimodalPositionEmbeddings)
                .map(|seg| mk_buf(&seg)),
            multimodal_auxiliary_weights_buffer: header
                .segment(SegmentKind::MultimodalAuxiliaryWeights)
                .map(|seg| mk_buf(&seg)),
            mil_buffer: None,
            metallib_buffer: Some(mk_buf(sg0)),
            compaction_model_bytes: None,
            prefill_model_bytes: None,
            num_weights: num_weights as u64,
            num_layers: header.num_layers,
            mmap_data: bytes,
        })
    }
    pub fn multimodal_capabilities(&self) -> MultimodalCapabilities {
        if let Some(desc) = self.read_multimodal_descriptor() {
            return capabilities_from_descriptor(&desc);
        }

        if let Some(token_map) = self.read_multimodal_token_map() {
            return capabilities_from_token_map(&token_map);
        }

        MultimodalCapabilities::default()
    }

    pub fn multimodal_descriptor(&self) -> Option<MultimodalInputDescriptorV1> {
        self.read_multimodal_descriptor()
    }

    pub fn multimodal_projection_records(&self) -> Vec<ProjectionTensorRecord> {
        let Some(desc) = self.read_multimodal_descriptor() else {
            return Vec::new();
        };
        read_projection_records(&self.mmap_data, &desc).unwrap_or_default()
    }

    pub fn multimodal_artifact_summary(&self) -> MultimodalArtifactSummary {
        let Some(desc) = self.read_multimodal_descriptor() else {
            return MultimodalArtifactSummary::text_only();
        };
        MultimodalArtifactSummary::from_descriptor(
            &desc,
            projection_precision_from_descriptor(&self.mmap_data, &desc),
        )
    }

    pub fn multimodal_bindings(&self) -> Option<SealedMultimodalBindings> {
        SealedMultimodalBindings::from_deployment(self).ok()
    }

    fn read_multimodal_descriptor(&self) -> Option<MultimodalInputDescriptorV1> {
        let (header, _) = verify_cimage(&self.mmap_data).ok()?;
        let entry = header.segment(SegmentKind::MultimodalInputDescriptor)?;
        let start = entry.offset as usize;
        let end = start + entry.length as usize;
        if end > self.mmap_data.len()
            || (entry.length as usize) < std::mem::size_of::<MultimodalInputDescriptorV1>()
        {
            return None;
        }
        let desc = unsafe {
            std::ptr::read_unaligned(
                self.mmap_data[start..end].as_ptr() as *const MultimodalInputDescriptorV1
            )
        };
        desc.validate().ok()?;
        Some(desc)
    }

    fn read_multimodal_token_map(&self) -> Option<serde_json::Value> {
        let (header, _) = verify_cimage(&self.mmap_data).ok()?;
        let entry = header.segment(SegmentKind::ModelArtifacts)?;
        let start = entry.offset as usize;
        let end = start + entry.length as usize;
        if end > self.mmap_data.len() {
            return None;
        }
        let blob = &self.mmap_data[start..end];
        for (tag, payload) in ModelArtifactEntry::iter_entries(blob) {
            if tag == model_artifact_tag::TOKEN_MAP {
                if let Ok(json) = serde_json::from_slice::<serde_json::Value>(payload) {
                    return Some(json);
                }
            }
        }
        None
    }

    /// If running on M5+ (Apple10 GPU family), expand ternary weights to INT4
    /// block-quantized format in a GPU-readable shared buffer.
    /// Called once after load, before any decode.
    pub fn maybe_expand_to_int4(&mut self, device: &metal::Device) -> Result<(), String> {
        if self.multimodal_descriptor().is_some()
            && self.mmap_data.len() >= std::mem::size_of::<PrismCimageHeader>()
        {
            let (header, _) = verify_cimage(&self.mmap_data)?;
            if header.is_nf4_tile640() {
                return Ok(());
            }
        } else if self.mmap_data.len() >= std::mem::size_of::<PrismCimageHeader>() {
            let (header, _) = verify_cimage(&self.mmap_data)?;
            if header.is_nf4_tile640() {
                return Ok(());
            }
        }

        // Check GPU family — activate on M5+ (Apple10).
        // metal-rs 0.29 caps at Apple9; update to Apple10 when the crate adds it.
        if !device.supports_family(metal::MTLGPUFamily::Apple9) {
            return Ok(());
        }

        // If already expanded or no weights loaded, skip
        if self.weights_int4_buffer.is_some() {
            return Ok(());
        }

        let ternary_total = self.weights_buffer.length() as usize;
        // Map CPU-side pointers
        let src = unsafe {
            std::slice::from_raw_parts(
                self.weights_buffer.contents() as *const u32,
                ternary_total / 4,
            )
        };

        // Repack .cimage ternary (20 trits/u32) → TernaryBlock32 (5 trits/byte) format
        let blocks = crate::compute_image::compile::int4_pack::repack_ternary_tensor(src);
        let block_bytes =
            unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * 9) };

        let ternary_buf = device.new_buffer_with_data(
            block_bytes.as_ptr() as *const std::ffi::c_void,
            block_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        self.weights_int4_buffer = Some(ternary_buf);

        // Build fused interleaved ternary buffer from the per-matrix block data
        const Q_WEIGHTS: usize = 3840 * 4096;
        const KV_WEIGHTS: usize = 3840 * 2048;
        const O_WEIGHTS: usize = 4096 * 3840;
        const FFN_WEIGHTS: usize = 3840 * 15360;
        const DOWN_WEIGHTS: usize = 15360 * 3840;

        const Q_BLOCKS: usize = Q_WEIGHTS / 32;
        const KV_BLOCKS: usize = KV_WEIGHTS / 32;
        const O_BLOCKS: usize = O_WEIGHTS / 32;
        const FFN_BLOCKS: usize = FFN_WEIGHTS / 32;
        const DOWN_BLOCKS: usize = DOWN_WEIGHTS / 32;

        const Q_BYTES: usize = Q_BLOCKS * 9;
        const KV_BYTES: usize = KV_BLOCKS * 9;
        const O_BYTES: usize = O_BLOCKS * 9;
        const FFN_BYTES: usize = FFN_BLOCKS * 9;
        const DOWN_BYTES: usize = DOWN_BLOCKS * 9;

        const LAYER_BLOCK_BYTES: usize =
            Q_BYTES + 2 * KV_BYTES + O_BYTES + 2 * FFN_BYTES + DOWN_BYTES;

        let mut fused = Vec::with_capacity(self.num_layers as usize * 120 * 7 * 180);

        for layer in 0..self.num_layers as usize {
            let lbase = layer * LAYER_BLOCK_BYTES;
            let q = &block_bytes[lbase..lbase + Q_BYTES];
            let k = &block_bytes[lbase + Q_BYTES..lbase + Q_BYTES + KV_BYTES];
            let v = &block_bytes[lbase + Q_BYTES + KV_BYTES..lbase + Q_BYTES + 2 * KV_BYTES];
            let o = &block_bytes
                [lbase + Q_BYTES + 2 * KV_BYTES..lbase + Q_BYTES + 2 * KV_BYTES + O_BYTES];
            let gate = &block_bytes[lbase + Q_BYTES + 2 * KV_BYTES + O_BYTES
                ..lbase + Q_BYTES + 2 * KV_BYTES + O_BYTES + FFN_BYTES];
            let up = &block_bytes[lbase + Q_BYTES + 2 * KV_BYTES + O_BYTES + FFN_BYTES
                ..lbase + Q_BYTES + 2 * KV_BYTES + O_BYTES + 2 * FFN_BYTES];
            let down = &block_bytes[lbase + Q_BYTES + 2 * KV_BYTES + O_BYTES + 2 * FFN_BYTES
                ..lbase + LAYER_BLOCK_BYTES];

            let layer_fused =
                crate::compute_image::compile::int4_pack::interleave_fused_ternary_layer(
                    q,
                    k,
                    v,
                    o,
                    gate,
                    up,
                    down,
                    HIDDEN_DIM as usize,
                    HIDDEN_DIM as usize,
                    O_ROWS as usize,
                    HIDDEN_DIM as usize,
                    DOWN_ROWS as usize,
                );
            fused.extend_from_slice(&layer_fused);
        }

        let fused_metal = new_slc_bypass_buffer(device, &fused);
        self.fused_int4_buffer = Some(fused_metal);

        Ok(())
    }

    /// Verify SHA-256 integrity of a `.cimage` file without allocating
    /// Metal buffers.  Useful for preflight checks or offline validation.
    ///
    /// Returns the parsed [`CImageHeader`] and [`CImageLayoutMeta`] on success
    /// for v1 binaries.  For the v2 Prism Engine format, use
    /// [`verify_prism_cimage`] directly instead.
    pub fn verify(
        path: impl AsRef<Path>,
    ) -> Result<
        (
            crate::compute_image::manifest::CImageHeader,
            V1CImageLayoutMeta,
        ),
        String,
    > {
        let bytes =
            std::fs::read(path.as_ref()).map_err(|e| format!("failed to read .cimage: {}", e))?;

        // Reject v2 format — callers should use verify_prism_cimage instead
        if bytes.len() >= 8 && &bytes[0..8] == &PRISM_MAGIC {
            return Err(
                "file uses the Prism Engine v2 format — use verify_prism_cimage instead".into(),
            );
        }

        if bytes.len() < 192 {
            return Err(format!(
                "cimage too small: {} bytes (need >= 192)",
                bytes.len()
            ));
        }

        type VfyHeader = crate::compute_image::manifest::CImageHeader;
        let header: VfyHeader =
            unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const VfyHeader) };

        if header.magic != crate::compute_image::manifest::CIMAGE_MAGIC {
            return Err(format!(
                "bad magic: 0x{:08X} (expected 0x{:08X})",
                header.magic,
                crate::compute_image::manifest::CIMAGE_MAGIC
            ));
        }

        let v1_layout: V1CImageLayoutMeta = unsafe {
            std::ptr::read_unaligned(bytes.as_ptr().add(128) as *const V1CImageLayoutMeta)
        };

        let payload = &bytes[128..];
        let computed = Sha256::digest(payload);
        if computed.as_slice() != header.payload_hash {
            return Err("SHA-256 hash mismatch: payload corrupted".into());
        }

        Ok((header, v1_layout))
    }

    /// Get the number of 640-weight tiles required for the given hidden
    /// dimension.
    ///
    /// Each tile packs 640 tri-level weights into 160 bytes (2-bit nibble
    /// encoding, 4 weights per byte).
    pub fn tiles_for_dim(&self, dim: usize) -> usize {
        dim.div_ceil(640)
    }
}

fn capabilities_from_descriptor(desc: &MultimodalInputDescriptorV1) -> MultimodalCapabilities {
    let image = (desc.modality_mask & 0b0010) != 0;
    let audio = (desc.modality_mask & 0b0100) != 0;
    let projection_backend = if image || audio {
        ProjectionBackend::Metal
    } else {
        ProjectionBackend::None
    };
    MultimodalCapabilities {
        text: true,
        image,
        audio,
        image_projection_backend: if image {
            projection_backend
        } else {
            ProjectionBackend::None
        },
        audio_projection_backend: if audio {
            projection_backend
        } else {
            ProjectionBackend::None
        },
        max_images_per_prompt: if image { 1 } else { 0 },
        max_soft_tokens_per_image: desc.image_max_soft_tokens,
        supports_mixed_embedding_prefill: image || audio,
    }
}

fn capabilities_from_token_map(token_map: &serde_json::Value) -> MultimodalCapabilities {
    let image = token_map.get("image_start_token").is_some()
        || token_map.get("image_end_token").is_some()
        || token_map.get("image_token_count").is_some()
        || token_map.get("vision_patch_size").is_some();
    let audio = token_map.get("audio_start_token").is_some()
        || token_map.get("audio_end_token").is_some()
        || token_map.get("audio_sample_rate").is_some()
        || token_map.get("audio_frame_ms").is_some();

    MultimodalCapabilities {
        text: true,
        image,
        audio,
        image_projection_backend: if image {
            ProjectionBackend::Metal
        } else {
            ProjectionBackend::None
        },
        audio_projection_backend: if audio {
            ProjectionBackend::Metal
        } else {
            ProjectionBackend::None
        },
        max_images_per_prompt: if image { 1 } else { 0 },
        max_soft_tokens_per_image: token_map
            .get("image_token_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
        supports_mixed_embedding_prefill: image || audio,
    }
}

fn read_projection_records(
    mmap_data: &[u8],
    desc: &MultimodalInputDescriptorV1,
) -> Option<Vec<ProjectionTensorRecord>> {
    let start = desc.image_projection_table_offset as usize;
    let total_records = desc.image_projection_count as usize + desc.audio_projection_count as usize;
    let record_size = std::mem::size_of::<ProjectionTensorRecord>();
    let byte_len = total_records.checked_mul(record_size)?;
    let end = start.checked_add(byte_len)?;
    if end > mmap_data.len() {
        return None;
    }

    let mut records = Vec::with_capacity(total_records);
    let mut cursor = start;
    for _ in 0..total_records {
        let record = unsafe {
            std::ptr::read_unaligned(mmap_data[cursor..].as_ptr() as *const ProjectionTensorRecord)
        };
        records.push(record);
        cursor += record_size;
    }
    Some(records)
}

fn projection_precision_from_descriptor(
    mmap_data: &[u8],
    desc: &MultimodalInputDescriptorV1,
) -> ProjectionPrecision {
    let Some((header, _)) = verify_cimage(mmap_data).ok() else {
        return ProjectionPrecision::Unknown;
    };
    let has_weight_segment = header
        .segment(SegmentKind::MultimodalProjectionWeights)
        .is_some();
    let has_scale_segment = header
        .segment(SegmentKind::MultimodalProjectionScales)
        .is_some();
    if !has_weight_segment {
        return ProjectionPrecision::Unknown;
    }

    let records = read_projection_records(mmap_data, desc).unwrap_or_default();
    if records.iter().any(|record| record.is_nf4_tile640()) {
        ProjectionPrecision::Nf4Tile640
    } else if records.iter().any(|record| record.quantization_kind != 0) {
        ProjectionPrecision::Ternary
    } else if has_scale_segment {
        ProjectionPrecision::Hybrid
    } else {
        ProjectionPrecision::Fp16
    }
}

#[cfg(target_os = "macos")]
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub fn load_heterogeneous_executor(
    mmap_data: &[u8],
    header: &PrismCimageHeader,
    cimage_path: Option<&std::path::Path>,
) -> Result<
    Option<(
        crate::backend::heterogeneous_executor::HeterogeneousExecutor,
        crate::compute_image::heterogeneous::types::HeterogeneousExecutionImage,
    )>,
    String,
> {
    use crate::backend::routing::{
        CorrectnessCheckpointPolicy, LogicalShape, OperationDescriptor, OperationId, Phase,
        TensorShape,
    };
    use crate::backend::DType;
    use crate::compute_image::heterogeneous::types::HeterogeneousExecutionImage;

    let Some(segment) = header.segment(SegmentKind::HeterogeneousImage) else {
        return Ok(None);
    };

    let start = segment.offset as usize;
    let end = start
        .checked_add(segment.length as usize)
        .ok_or_else(|| "segment offset/length overflow".to_string())?;
    if end > mmap_data.len() {
        return Err(format!(
            "HeterogeneousImage segment at offset {} length {} exceeds mmap size {}",
            segment.offset,
            segment.length,
            mmap_data.len()
        ));
    }
    let blob = &mmap_data[start..end];

    let image: HeterogeneousExecutionImage = serde_json::from_slice(blob)
        .map_err(|e| format!("failed to deserialize HeterogeneousExecutionImage: {e}"))?;

    let batch_size = 1;
    let int4_mode = false;
    let mut executor = match cimage_path {
        Some(path) => crate::backend::create_inference_executor(path, batch_size, int4_mode)?,
        None => crate::backend::create_heterogeneous_executor()?,
    };

    // Walk compiled phase graph, building operation descriptors from the
    // image's stored operation families (populated at compile time).
    let mut operation_registry = std::collections::HashMap::new();
    for node in &image.phase_graph.nodes {
        let op_id = OperationId(node.phase_id);
        let descriptor = OperationDescriptor {
            operation_id: op_id,
            family: node.operation_family,
            layer_index: Some(node.phase_id as u32),
            phase: Phase::Decode,
            logical_shape: LogicalShape { dims: Vec::new() },
            physical_layout: crate::backend::routing::PhysicalLayout::RowMajor,
            input_dtypes: Vec::new(),
            output_dtype: DType::F32,
            quantization: None,
            expected_output_shape: TensorShape { dims: Vec::new() },
            correctness_checkpoint: CorrectnessCheckpointPolicy::None,
        };
        operation_registry.insert(op_id, descriptor);
    }
    executor.set_operation_registry(operation_registry);
    Ok(Some((executor, image)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::multimodal::descriptor::MULTIMODAL_DESCRIPTOR_MAGIC;

    #[test]
    fn descriptor_capabilities_expose_image_and_audio() {
        let mut desc = MultimodalInputDescriptorV1::default();
        desc.magic = MULTIMODAL_DESCRIPTOR_MAGIC;
        desc.version = 1;
        desc.modality_mask = 0b0110;
        desc.image_max_soft_tokens = 280;

        let caps = capabilities_from_descriptor(&desc);
        assert!(caps.text);
        assert!(caps.image);
        assert!(caps.audio);
        assert_eq!(caps.max_soft_tokens_per_image, 280);
        assert!(caps.supports_mixed_embedding_prefill);
    }

    #[test]
    fn token_map_capabilities_detect_multimodal_support() {
        let token_map = serde_json::json!({
            "image_start_token": "<start_of_image>",
            "audio_start_token": "<start_of_audio>",
            "image_token_count": 256,
            "audio_sample_rate": 16000
        });

        let caps = capabilities_from_token_map(&token_map);
        assert!(caps.image);
        assert!(caps.audio);
        assert_eq!(caps.max_soft_tokens_per_image, 256);
        assert_eq!(caps.image_projection_backend, ProjectionBackend::Metal);
        assert_eq!(caps.audio_projection_backend, ProjectionBackend::Metal);
    }

    #[test]
    fn projection_records_roundtrip_from_descriptor_blob() {
        let mut desc = MultimodalInputDescriptorV1::default();
        desc.magic = MULTIMODAL_DESCRIPTOR_MAGIC;
        desc.version = 1;
        desc.modality_mask = 0b0110;
        desc.image_projection_count = 1;
        desc.audio_projection_count = 1;
        desc.image_projection_table_offset =
            std::mem::size_of::<MultimodalInputDescriptorV1>() as u64;

        let image = ProjectionTensorRecord {
            logical_name_hash: 11,
            role: 2,
            input_width: 3840,
            output_width: 3840,
            ..ProjectionTensorRecord::default()
        };
        let audio = ProjectionTensorRecord {
            logical_name_hash: 22,
            role: 6,
            input_width: 1024,
            output_width: 3840,
            ..ProjectionTensorRecord::default()
        };

        let mut blob = Vec::new();
        blob.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &desc as *const MultimodalInputDescriptorV1 as *const u8,
                std::mem::size_of::<MultimodalInputDescriptorV1>(),
            )
        });
        blob.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &image as *const ProjectionTensorRecord as *const u8,
                std::mem::size_of::<ProjectionTensorRecord>(),
            )
        });
        blob.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &audio as *const ProjectionTensorRecord as *const u8,
                std::mem::size_of::<ProjectionTensorRecord>(),
            )
        });

        let records = read_projection_records(&blob, &desc).expect("records");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].logical_name_hash, 11);
        assert_eq!(records[1].logical_name_hash, 22);
    }
}
