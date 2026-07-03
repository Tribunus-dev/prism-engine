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

use crate::compute_image::compile::ternary::{
    verify_cimage, SegmentEntry, SegmentKind, ModelArtifactEntry, model_artifact_tag, PRISM_MAGIC,
};
use memmap2::Mmap;
use sha2::{Digest, Sha256};
use crate::compute_image::megakernel::kernels::HIDDEN_DIM;
use std::fs::File;
use std::io;
use crate::compute_image::multimodal::descriptor::MultimodalCapabilities;
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

fn align64(n: u64) -> u64 { (n + 63) & !63 }

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
        return Err(io::Error::new(io::ErrorKind::InvalidData, ".cimage file too small"));
    }
    let header: PrismCimageHeader = unsafe {
        std::ptr::read_unaligned(mmap.as_ptr() as *const PrismCimageHeader)
    };
    if &header.magic != &PRISM_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad .cimage magic"));
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
    /// directory and loads via CoreMlModel, avoiding ~3s JIT compilation.
    pub compaction_model_bytes: Option<Vec<u8>>,
    /// Compiled ANE prefill model bytes (model.mlmodel protobuf from
    /// .mlmodelc bundle), embedded in the aux section tail after the
    /// compaction model. Present in .cimage v2 format when compilation
    /// included ANE prefill. At runtime the orchestrator writes these
    /// bytes to a temp .mlmodelc directory and loads via CoreMlModel,
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
        let layout: V1CImageLayoutMeta =
            unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(128) as *const V1CImageLayoutMeta) };

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
            embed_buffer: None,
            centroid_buffer: None,
            centroid_scales_buffer: None,
            cluster_map_buffer: None,
            norms_buffer: None,
            scalars_buffer: None,
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
            header.segments.iter()
                .find(|s| s.kind == kind && s.length > 0)
                .ok_or_else(|| format!("segment kind {} not found", kind))
        };
        let sg0 = find_seg(SegmentKind::MetalLib as u32)?;
        let sg1 = find_seg(SegmentKind::TernaryWeights as u32)?;
        let sg2 = find_seg(SegmentKind::BlockScales as u32)?;

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

        let num_weights = (sg2.length / 2) * 256;

        // ── Read auxiliary buffers from ModelArtifacts segment ────────
        let model_artifacts_seg = header.segments.iter().find(|s| s.kind == SegmentKind::ModelArtifacts as u32 && s.length > 0);
        let (embed_buffer, embed_scales_buffer, centroid_buffer, centroid_scales_buffer,
             cluster_map_buffer, norms_buffer) = match model_artifacts_seg {
            Some(seg) => {
                let off = seg.offset as usize;
                let len = seg.length as usize;
                let data = &bytes[off..off + len];
                let mut embed = None; let mut escale = None;
                let mut cent = None; let mut cscale = None;
                let mut cmap = None; let mut norms = None;
                for (tag, payload) in ModelArtifactEntry::iter_entries(data) {
                    match tag {
                        t if t == model_artifact_tag::EMBED_NIBBLES => embed = Some(device.new_buffer_with_data(payload.as_ptr() as *const _, payload.len() as u64, metal::MTLResourceOptions::StorageModeShared)),
                        t if t == model_artifact_tag::EMBED_SCALES => escale = Some(device.new_buffer_with_data(payload.as_ptr() as *const _, payload.len() as u64, metal::MTLResourceOptions::StorageModeShared)),
                        t if t == model_artifact_tag::CENTROID_NIBBLES => cent = Some(device.new_buffer_with_data(payload.as_ptr() as *const _, payload.len() as u64, metal::MTLResourceOptions::StorageModeShared)),
                        t if t == model_artifact_tag::CENTROID_SCALES => cscale = Some(device.new_buffer_with_data(payload.as_ptr() as *const _, payload.len() as u64, metal::MTLResourceOptions::StorageModeShared)),
                        t if t == model_artifact_tag::CLUSTER_MAP => cmap = Some(device.new_buffer_with_data(payload.as_ptr() as *const _, payload.len() as u64, metal::MTLResourceOptions::StorageModeShared)),
                        t if t == model_artifact_tag::AUX_NORMS => norms = Some(device.new_buffer_with_data(payload.as_ptr() as *const _, payload.len() as u64, metal::MTLResourceOptions::StorageModeShared)),
                        _ => {}
                    }
                }
                (embed, escale, cent, cscale, cmap, norms)
            }
            None => (None, None, None, None, None, None),
        };

        Ok(Self {
            header: crate::compute_image::manifest::CImageHeader::default(),
            layout: unsafe { std::mem::zeroed() },
            weights_buffer: mk_buf(sg1),
            scales_buffer: mk_buf(sg2),
            weights_int4_buffer: None,
            fused_int4_buffer: None,
            embed_buffer,
            embed_scales_buffer,
            centroid_scales_buffer,
            centroid_buffer,
            cluster_map_buffer,
            norms_buffer,
            scalars_buffer: None,
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
        MultimodalCapabilities::default()
    }



    /// If running on M5+ (Apple10 GPU family), expand ternary weights to INT4
    /// block-quantized format in a GPU-readable shared buffer.
    /// Called once after load, before any decode.
    pub fn maybe_expand_to_int4(&mut self, device: &metal::Device) -> Result<(), String> {
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
        let block_bytes = unsafe {
            std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * 9)
        };

        let ternary_buf = device.new_buffer_with_data(
            block_bytes.as_ptr() as *const std::ffi::c_void,
            block_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        self.weights_int4_buffer = Some(ternary_buf);

        // Build fused interleaved ternary buffer from the per-matrix block data
        const Q_WEIGHTS: usize   = 3840 * 4096;
        const KV_WEIGHTS: usize  = 3840 * 2048;
        const O_WEIGHTS: usize   = 4096 * 3840;
        const FFN_WEIGHTS: usize = 3840 * 15360;
        const DOWN_WEIGHTS: usize = 15360 * 3840;

        const Q_BLOCKS: usize    = Q_WEIGHTS / 32;
        const KV_BLOCKS: usize   = KV_WEIGHTS / 32;
        const O_BLOCKS: usize    = O_WEIGHTS / 32;
        const FFN_BLOCKS: usize  = FFN_WEIGHTS / 32;
        const DOWN_BLOCKS: usize = DOWN_WEIGHTS / 32;

        const Q_BYTES: usize    = Q_BLOCKS * 9;
        const KV_BYTES: usize   = KV_BLOCKS * 9;
        const O_BYTES: usize    = O_BLOCKS * 9;
        const FFN_BYTES: usize  = FFN_BLOCKS * 9;
        const DOWN_BYTES: usize = DOWN_BLOCKS * 9;

        const LAYER_BLOCK_BYTES: usize =
            Q_BYTES + 2 * KV_BYTES + O_BYTES + 2 * FFN_BYTES + DOWN_BYTES;

        let mut fused = Vec::with_capacity(self.num_layers as usize * 120 * 7 * 180);

        for layer in 0..self.num_layers as usize {
            let lbase = layer * LAYER_BLOCK_BYTES;
            let q    = &block_bytes[lbase..lbase + Q_BYTES];
            let k    = &block_bytes[lbase + Q_BYTES..lbase + Q_BYTES + KV_BYTES];
            let v    = &block_bytes[lbase + Q_BYTES + KV_BYTES..lbase + Q_BYTES + 2 * KV_BYTES];
            let o    = &block_bytes[lbase + Q_BYTES + 2 * KV_BYTES..lbase + Q_BYTES + 2 * KV_BYTES + O_BYTES];
            let gate = &block_bytes[lbase + Q_BYTES + 2 * KV_BYTES + O_BYTES..
                                    lbase + Q_BYTES + 2 * KV_BYTES + O_BYTES + FFN_BYTES];
            let up   = &block_bytes[lbase + Q_BYTES + 2 * KV_BYTES + O_BYTES + FFN_BYTES..
                                    lbase + Q_BYTES + 2 * KV_BYTES + O_BYTES + 2 * FFN_BYTES];
            let down = &block_bytes[lbase + Q_BYTES + 2 * KV_BYTES + O_BYTES + 2 * FFN_BYTES..
                                    lbase + LAYER_BLOCK_BYTES];

            let layer_fused = crate::compute_image::compile::int4_pack::interleave_fused_ternary_layer(
                q, k, v, o, gate, up, down,
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
    pub fn verify(path: impl AsRef<Path>) -> Result<(crate::compute_image::manifest::CImageHeader, V1CImageLayoutMeta), String> {
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

        let v1_layout: V1CImageLayoutMeta =
            unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(128) as *const V1CImageLayoutMeta) };

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
