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
    verify_prism_cimage, TensorRecord, PRISM_MAGIC, PRISM_PAGE_SIZE,
};
use memmap2::Mmap;
use sha2::{Digest, Sha256};
use crate::compute_image::megakernel::kernels::HIDDEN_DIM;
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
        let min_size = PRISM_PAGE_SIZE as usize;
        if bytes.len() < min_size {
            return Err(format!(
                "prism cimage too small: {} bytes (need >= {})",
                bytes.len(),
                min_size
            ));
        }

        // Verify integrity via verify_prism_cimage (magic + SHA-256).
        // The header and layout are parsed once here; we keep the raw bytes
        // alive in mmap_data.
        let (header, layout) = verify_prism_cimage(&bytes)?;

        // ── Bounds-check each tensor section ─────────────────────────────
        let check_section = |name: &str, rec: &TensorRecord| -> Result<(), String> {
            let end = rec
                .offset
                .checked_add(rec.length)
                .ok_or_else(|| format!("{} section offset+length overflow", name))?;
            if end > bytes.len() as u64 {
                return Err(format!(
                    "{} section out of range: offset={} length={} file_size={}",
                    name,
                    rec.offset,
                    rec.length,
                    bytes.len()
                ));
            }
            Ok(())
        };

        check_section("embed_clustered", &layout.embed_clustered)?;
        check_section("centroid_table", &layout.centroid_table)?;
        check_section("cluster_map", &layout.cluster_map)?;
        check_section("ternary_weights", &layout.ternary_weights)?;
        check_section("block_scales", &layout.block_scales)?;
        check_section("aux", &layout.aux)?;

        // ── Create Metal shared-memory buffers for each section ─────────
        let embed_start = layout.embed_clustered.offset as usize;
        let embed_buffer = {
            let src = &bytes[embed_start..embed_start + layout.embed_clustered.length as usize];
            device.new_buffer_with_data(
                src.as_ptr() as *const std::ffi::c_void,
                layout.embed_clustered.length,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };

        // centroid_table buffer
        let centroid_start = layout.centroid_table.offset as usize;
        let centroid_buffer = {
            let src =
                &bytes[centroid_start..centroid_start + layout.centroid_table.length as usize];
            device.new_buffer_with_data(
                src.as_ptr() as *const std::ffi::c_void,
                layout.centroid_table.length,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };

        // cluster_map buffer
        let cluster_start = layout.cluster_map.offset as usize;
        let cluster_map_buffer = {
            let src = &bytes[cluster_start..cluster_start + layout.cluster_map.length as usize];
            device.new_buffer_with_data(
                src.as_ptr() as *const std::ffi::c_void,
                layout.cluster_map.length,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };

        let weights_start = layout.ternary_weights.offset as usize;
        let weights_buffer = {
            let src = &bytes[weights_start..weights_start + layout.ternary_weights.length as usize];
            device.new_buffer_with_data(
                src.as_ptr() as *const std::ffi::c_void,
                layout.ternary_weights.length,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };

        let scales_start = layout.block_scales.offset as usize;
        // Split block_scales into layer_scales + embed_scales + centroid_scales buffers.
        // The ingest pipeline appends embed_scales then centroid_scales after layer scales.
        let total_scales_length = layout.block_scales.length as usize;
        let embed_blocks = (header.vocab_size as u64 * header.hidden_dim as u64 + 255) / 256;
        let embed_scales_bytes = (embed_blocks * 2) as usize;
        let centroid_blocks = (256u64 * header.hidden_dim as u64 + 255) / 256;
        let centroid_scales_bytes = (centroid_blocks * 2) as usize;
        let layer_scales_bytes = total_scales_length - embed_scales_bytes - centroid_scales_bytes;
        let layer_scales_end = scales_start + layer_scales_bytes;
        let embed_scales_end = layer_scales_end + embed_scales_bytes;

        let scales_buffer = {
            let src = &bytes[scales_start..layer_scales_end];
            device.new_buffer_with_data(
                src.as_ptr() as *const std::ffi::c_void,
                layer_scales_bytes as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };

        let embed_scales_buffer = {
            let src = &bytes[layer_scales_end..embed_scales_end];
            device.new_buffer_with_data(
                src.as_ptr() as *const std::ffi::c_void,
                embed_scales_bytes as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };

        // Centroid scales are the last portion of block_scales.
        let centroid_scales_start = embed_scales_end;
        let centroid_scales_buffer = {
            let src = &bytes[centroid_scales_start..embed_scales_end + centroid_scales_bytes];
            device.new_buffer_with_data(
                src.as_ptr() as *const std::ffi::c_void,
                centroid_scales_bytes as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };

        // ── Split aux section into norms + scalars ───────────────────
        // Norms: 48 layers × (4 × 3840 + 2 × 512) FP16 + final_norm 3840 FP16 = 1,580,544 bytes
        // Scalars: 48 × 2 bytes FP16 = 96 bytes
        // MIL: 256-byte placeholder (ANE Core ML program)
        // Metallib: remaining bytes (pre-compiled Metal kernel library)
        let aux_start = layout.aux.offset as usize;
        let aux_len = layout.aux.length as usize;
        let norms_byte_len = 48 * (4 * 3840 + 2 * 512) * 2 + 3840 * 2; // 1,580,544
        let scalars_byte_len = 48 * 2; // 96 bytes
        let mil_byte_len: usize = 256; // fixed ANE MIL placeholder size
        let (norms_buffer, scalars_buffer, mil_buffer, metallib_buffer, compaction_model_bytes) =
            if aux_len >= norms_byte_len + scalars_byte_len + mil_byte_len {
                let nb = {
                    let src = &bytes[aux_start..aux_start + norms_byte_len];
                    device.new_buffer_with_data(
                        src.as_ptr() as *const std::ffi::c_void,
                        norms_byte_len as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    )
                };
                let sb = {
                    let src = &bytes
                        [aux_start + norms_byte_len..aux_start + norms_byte_len + scalars_byte_len];
                    device.new_buffer_with_data(
                        src.as_ptr() as *const std::ffi::c_void,
                        scalars_byte_len as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    )
                };
                let mil_start = aux_start + norms_byte_len + scalars_byte_len;
                let metallib_start = mil_start + mil_byte_len;
                let mb = {
                    let src = &bytes[mil_start..mil_start + mil_byte_len];
                    Some(device.new_buffer_with_data(
                        src.as_ptr() as *const std::ffi::c_void,
                        mil_byte_len as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    ))
                };
                let tail_remaining = aux_start + aux_len - metallib_start;
                let (metallib_buf, comp_bytes) = if tail_remaining >= 4
                    && bytes[metallib_start..metallib_start + 4] == [0x4D, 0x54, 0x4C, 0x42]
                {
                    // Old format: MTLB magic directly, whole tail is metallib
                    let src = &bytes[metallib_start..aux_start + aux_len];
                    (
                        Some(device.new_buffer_with_data(
                            src.as_ptr() as *const std::ffi::c_void,
                            (aux_start + aux_len - metallib_start) as u64,
                            metal::MTLResourceOptions::StorageModeShared,
                        )),
                        None,
                    )
                } else if tail_remaining >= 8 {
                    // New format: first 4 bytes = metallib length (u32le)
                    let metal_len = u32::from_le_bytes([
                        bytes[metallib_start],
                        bytes[metallib_start + 1],
                        bytes[metallib_start + 2],
                        bytes[metallib_start + 3],
                    ]) as usize;
                    let metal_data_start = metallib_start + 4;
                    let after_metal = metal_data_start + metal_len;
                    let mb = if tail_remaining >= 4 + metal_len {
                        let src = &bytes[metal_data_start..after_metal];
                        Some(device.new_buffer_with_data(
                            src.as_ptr() as *const std::ffi::c_void,
                            metal_len as u64,
                            metal::MTLResourceOptions::StorageModeShared,
                        ))
                    } else {
                        None
                    };
                    // After metallib: comp_len(u32le) + compaction_bytes
                    // After compaction model: prefill_len(u32le) + prefill_model bytes
                    let comp_bytes = if after_metal + 4 <= aux_start + aux_len {
                        let comp_len = u32::from_le_bytes([
                            bytes[after_metal],
                            bytes[after_metal + 1],
                            bytes[after_metal + 2],
                            bytes[after_metal + 3],
                        ]) as usize;
                        let comp_data_start = after_metal + 4;
                        if comp_len > 0 && comp_data_start + comp_len <= aux_start + aux_len {
                            Some(bytes[comp_data_start..comp_data_start + comp_len].to_vec())
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    (mb, comp_bytes)
                } else {
                    (None, None)
                };
                (Some(nb), Some(sb), mb, metallib_buf, comp_bytes)
            } else {
                (None, None, None, None, None)
            };

        // ── Parse prefill model bytes from aux tail ────────────────
        // The aux section tail is structured as:
        //   metallib_len(u32le) + metallib_bytes + comp_len(u32le) + comp_bytes + prefill_len(u32le) + prefill_bytes
        // We skip past norms/scalars/MIL-placeholder/metallib/compaction to reach prefill bytes.
        let prefill_model_bytes: Option<Vec<u8>> = if aux_len
            >= norms_byte_len + scalars_byte_len + mil_byte_len + 8
        {
            let metallib_start = aux_start + norms_byte_len + scalars_byte_len + mil_byte_len;
            let tail_remaining = aux_start + aux_len - metallib_start;
            if tail_remaining < 8 {
                None
            } else if bytes[metallib_start..metallib_start + 4] == [0x4D, 0x54, 0x4C, 0x42] {
                // Old format: no compaction/prefill data
                None
            } else {
                // Skip metallib: first 4 bytes = length
                let metal_len = u32::from_le_bytes([
                    bytes[metallib_start],
                    bytes[metallib_start + 1],
                    bytes[metallib_start + 2],
                    bytes[metallib_start + 3],
                ]) as usize;
                let after_metal = metallib_start + 4 + metal_len;
                // Skip compaction model: at after_metal, 4 bytes = length + bytes
                if after_metal + 8 <= aux_start + aux_len {
                    let comp_len = u32::from_le_bytes([
                        bytes[after_metal],
                        bytes[after_metal + 1],
                        bytes[after_metal + 2],
                        bytes[after_metal + 3],
                    ]) as usize;
                    let after_comp = after_metal + 4 + comp_len;
                    // Read prefill model: at after_comp, 4 bytes = length + bytes
                    if after_comp + 4 <= aux_start + aux_len {
                        let prefill_len = u32::from_le_bytes([
                            bytes[after_comp],
                            bytes[after_comp + 1],
                            bytes[after_comp + 2],
                            bytes[after_comp + 3],
                        ]) as usize;
                        let prefill_start = after_comp + 4;
                        if prefill_len > 0 && prefill_start + prefill_len <= aux_start + aux_len {
                            Some(bytes[prefill_start..prefill_start + prefill_len].to_vec())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        } else {
            None
        };

        // Compute num_weights from scales (each FP16 scale byte-pair covers
        // 256 weights).  The stored length is page-aligned, so the result
        // slightly over-counts, but this is harmless for tile allocation.
        let num_weights = (layer_scales_bytes as u64 / 2) * 256;
        let num_layers = header.num_layers;

        Ok(Self {
            // v1 header/layout are placeholders; all v2 metadata lives
            // in mmap_data for zero-copy reinterpretation by callers.
            header: crate::compute_image::manifest::CImageHeader::default(),
            layout: unsafe {
                let mut v1_layout: V1CImageLayoutMeta = std::mem::zeroed();
                v1_layout.num_layers = num_layers;
                v1_layout.num_weights = num_weights as u32;
                v1_layout.num_blocks = (num_weights / 256) as u32;
                v1_layout
            },
            weights_buffer,
            scales_buffer,
            embed_buffer: Some(embed_buffer),
            embed_scales_buffer: Some(embed_scales_buffer),
            centroid_scales_buffer: Some(centroid_scales_buffer),
            centroid_buffer: Some(centroid_buffer),
            cluster_map_buffer: Some(cluster_map_buffer),
            norms_buffer,
            scalars_buffer,
            mil_buffer,
            metallib_buffer,
            compaction_model_bytes,
            prefill_model_bytes,
            weights_int4_buffer: None,
            fused_int4_buffer: None,
            num_weights,
            num_layers,
            mmap_data: bytes,
        })
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
