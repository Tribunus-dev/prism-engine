//! Top-level orchestration: AOT plan → mmap → segments → header.

use super::archive::archive_mlmodelc_to_mmap;
use super::builder::AlignedMmapBuilder;
use super::layout::{CImageLayoutPlan, CImageTopologyTable, predict_tar_size};
use crate::compute_image::compile::ternary::LayerDirectoryEntry;
use crate::compute_image::compile::ternary::{CimageHeader, SegmentEntry, SegmentKind};
use std::io::{Write, Seek};
use std::fs::File;
use memmap2::MmapMut;
use std::path::Path;
use crate::compute_image::compile::source::LoadedSource;
use crate::config::CompileQuantMode;

/// Compile and pack the unified Gemma4_Unified.cimage.
///
/// 1. predict_tar_size scans the resulting .mlmodelc directories.
/// 2. CImageLayoutPlan::calculate() computes all offsets AOT.
/// 3. File is ftruncate'd + mmap'd at the exact total size.
/// 4. Metal lib + .mlmodelc are copied into the mmap.
/// 5. GPU writes quantized weights directly into the mmap via stream_weights_to_mmap_gpu.
/// 6. CImageHeader is written at offset 0.
///
/// GPU-accelerated TernaryTile640 quantization: streams weight tensors
/// from the loaded source into the cimage mmap, computing per-tensor
/// offsets within the weights segment and passing them to the Metal kernel
/// for direct-to-mmap write via `newBufferWithBytesNoCopy`.
#[cfg(feature = "metal-dispatch")]
pub(crate) fn stream_weights_to_mmap_gpu(
    loaded: &mut LoadedSource,
    plan: &CImageLayoutPlan,
    builder: &mut AlignedMmapBuilder,
    qmode: CompileQuantMode,
) -> crate::Result<()> {
    use crate::compute_image::compile::try_ternary_tile640_pack_gpu;

    if !matches!(qmode, CompileQuantMode::TernaryTile640 { .. }) {
        return Ok(());
    }

    let mmap_base = builder.mmap_base();
    let segment_file_offset = plan.main_weights.offset;

    // Iterate weight bindings in spec order, computing cumulative
    // offsets within the weights segment for each tensor.
    let mut tensor_cursor: u64 = 0;

    // Pre-collect weight binding names so we can freely borrow loaded mutably
    // inside the per-tensor loop.
    let global_weight_names: Vec<String> = loaded.spec.global_tensors.iter()
        .filter(|b| b.name.ends_with(".weight"))
        .map(|b| b.name.clone())
        .collect();

    // --- Global weight tensors ---
    for binding_name in &global_weight_names {
        // Streaming: load one tensor from mmap, extract shape + data, then
        // free the source Vec before GPU dispatch.  Peak heap = ~1 tensor.
        let (out_dim, in_dim) = {
            let mut entry = loaded.source_tensors.get_mut(binding_name).unwrap();
        for mmap in &loaded.mmap_bytes {
                crate::compute_image::compile::source::ensure_tensor_loaded(
                    &mut entry,
                    mmap,
                );
                if !entry.data.is_empty() {
                break;
            }
            }
            if entry.data.len() < 2 || (entry.dtype != "F16" && entry.dtype != "BF16") {
            continue;
            }
            if entry.shape.len() != 2 {
            continue;
            }
            (entry.shape[0], entry.shape[1])
        };
        // Take the Vec out of the SourceTensor, replacing it with empty.
        // The source memory is freed here, before GPU dispatch.
        let data = loaded.source_tensors
            .get_mut(binding_name)
            .map(|t| std::mem::take(&mut t.data))
            .unwrap_or_default();
        if data.is_empty() {
            continue;
        }

        let num_tiles = (in_dim as u64 + 639) / 640;
        let tensor_file_offset = segment_file_offset + tensor_cursor;

        try_ternary_tile640_pack_gpu(
            loaded,
            binding_name,
            &data,
            out_dim,
            in_dim,
            Some((mmap_base, tensor_file_offset)),
        )?;

        // Advance cursor by this tensor's packed size.
        let tensor_bytes = (out_dim as u64) * num_tiles * 32 * 4; // 32 u32 lanes per tile
        tensor_cursor += tensor_bytes;
    }

    // Pre-collect per-layer weight binding names.
    let layer_weight_names: Vec<String> = loaded.spec.layers.iter()
        .flat_map(|layer| layer.tensors.iter())
        .filter(|b| b.name.ends_with(".weight"))
        .map(|b| b.name.clone())
        .collect();

    // --- Per-layer weight tensors ---
    for binding_name in &layer_weight_names {
        let (out_dim, in_dim) = {
            let mut entry = match loaded.source_tensors.get_mut(binding_name) {
                Some(e) => e,
                None => continue,
            };
            for mmap in &loaded.mmap_bytes {
                crate::compute_image::compile::source::ensure_tensor_loaded(&mut entry, mmap);
                if !entry.data.is_empty() {
                    break;
                }
            }
            if entry.data.len() < 2 || (entry.dtype != "F16" && entry.dtype != "BF16") {
                continue;
            }
            if entry.shape.len() != 2 {
                continue;
            }
            (entry.shape[0], entry.shape[1])
        };
        let data = loaded.source_tensors
            .get_mut(binding_name)
            .map(|t| std::mem::take(&mut t.data))
            .unwrap_or_default();
        if data.is_empty() { continue; }
        let num_tiles = (in_dim as u64 + 639) / 640;
        let tensor_file_offset = segment_file_offset + tensor_cursor;

        try_ternary_tile640_pack_gpu(
            loaded,
            binding_name,
            &data,
            out_dim,
            in_dim,
            Some((mmap_base, tensor_file_offset)),
        )?;

            let tensor_bytes = (out_dim as u64) * num_tiles * 32 * 4;
        tensor_cursor += tensor_bytes;
    }
    eprintln!(
        "[cimage] GPU ternary tile640: {} weights streamed into mmap at offset {:#X}, {} bytes total",
        if tensor_cursor > 0 { "all" } else { "no" },
        segment_file_offset,
        tensor_cursor,
    );
    Ok(())
}

pub(crate) fn compile_and_pack_god_binary(
    output_path: &str,
    metallib_path: &Path,
    main_mlmodelc_path: &Path,
    mtp_mlmodelc_path: &Path,
    main_weight_total_elements: u64,
    mtp_weight_total_elements: u64,
    loaded: &mut LoadedSource,
    qmode: CompileQuantMode,
    hidden_size: u32,
    intermediate_size: u32,
    num_layers: u32,
    num_heads: u32,
    head_dim: u32,
) -> std::io::Result<()> {
    let main_graph_len = predict_tar_size(main_mlmodelc_path)?;
    let mtp_graph_len = predict_tar_size(mtp_mlmodelc_path)?;
    let metal_lib_len = std::fs::metadata(metallib_path)?.len();
    let header_size = std::mem::size_of::<CimageHeader>() as u64;

    // ── Locate embed_tokens.weight for the vocabulary segment ──────────────
    // Try common HF key prefixes (Gemma4 / Llama / Qwen families).
    let embed_key = ["language_model.model.embed_tokens.weight",
                     "model.embed_tokens.weight",
                     "embed_tokens.weight"]
        .iter()
        .find(|&&k| loaded.source_tensors.contains_key(k))
        .copied()
        .unwrap_or("");

    // Element count (number of f16/bf16 elements = byte_size / 2).
    let vocab_weight_elements: u64 = if embed_key.is_empty() {
        eprintln!("[cimage] ⚠️  embed_tokens.weight not found — Vocabulary segment will be empty");
        0
    } else {
        let st = &loaded.source_tensors[embed_key];
        if st.shape.len() == 2 {
            st.shape[0] as u64 * st.shape[1] as u64
        } else {
            (st.source_byte_size / 2).max(st.data.len() as u64 / 2)
        }
    };

    let plan = CImageLayoutPlan::calculate(
        header_size, metal_lib_len, main_graph_len,
        main_weight_total_elements, mtp_graph_len, mtp_weight_total_elements,
        vocab_weight_elements,
        num_layers,
        None, None, None,
    );

    let topology_table = CImageTopologyTable::compute(
        hidden_size, intermediate_size,
        num_layers, num_heads, head_dim,
    );

    eprintln!(
        "[cimage] AOT layout: total={} metal_lib={} main_graph={} main_weights={} mtp_graph={} mtp_weights={} vocabulary={}",
        plan.total_file_size,
        plan.metal_lib.length, plan.main_graph.length,
        plan.main_weights.length, plan.mtp_graph.length, plan.mtp_weights.length,
        plan.vocabulary.length,
    );

    let file = std::fs::OpenOptions::new()
        .read(true).write(true).create(true).truncate(true)
        .open(output_path)?;
    file.set_len(plan.total_file_size)?;
    let mut mmap = unsafe { MmapMut::map_mut(&file)? };
    unsafe { std::ptr::write_bytes(mmap.as_mut_ptr(), 0u8, mmap.len()); }
    let mut builder = AlignedMmapBuilder::new(mmap);

    // Segment: Metal megakernel
    let metallib_data = std::fs::read(metallib_path)?;
    builder.align_cursor();
    builder.allocate_slice(metallib_data.len()).copy_from_slice(&metallib_data);

    // Segment: Main .mlmodelc
    builder.align_cursor();
    let main_slice = builder.allocate_slice(plan.main_graph.length as usize);
    let written = archive_mlmodelc_to_mmap(main_mlmodelc_path, main_slice)?;
    eprintln!("[cimage] main .mlmodelc: {} bytes archived", written);

    // Segment: Main weights (GPU writes directly into mmap here)
    builder.align_cursor();
    let _main_weights_ptr = unsafe {
        builder.allocate_hardware_pointer(plan.main_weights.length as usize)
    };

    // GPU-accelerated ternary tile640 quantization streams weights into the mmap.
    #[cfg(feature = "metal-dispatch")]
    {
        stream_weights_to_mmap_gpu(loaded, &plan, &mut builder, qmode)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    }
    #[cfg(not(feature = "metal-dispatch"))]
    let _ = (loaded, qmode); // suppress unused warning when feature disabled

    // Segment: MTP .mlmodelc
    builder.align_cursor();
    let mtp_slice = builder.allocate_slice(plan.mtp_graph.length as usize);
    let written = archive_mlmodelc_to_mmap(mtp_mlmodelc_path, mtp_slice)?;
    eprintln!("[cimage] MTP .mlmodelc: {} bytes archived", written);

    // Segment: MTP weights (GPU writes directly into mmap here)
    builder.align_cursor();
    let _mtp_weights_ptr = unsafe {
        builder.allocate_hardware_pointer(plan.mtp_weights.length as usize)
    };

    // Segment: Topology table
    builder.align_cursor();
    let topology_bytes = unsafe {
        std::slice::from_raw_parts(
            &topology_table as *const CImageTopologyTable as *const u8,
            std::mem::size_of::<CImageTopologyTable>(),
        )
    };
    builder.allocate_slice(topology_bytes.len()).copy_from_slice(topology_bytes);

    // Segment: Vocabulary (embed_tokens.weight in TernaryTile640)
    builder.align_cursor();
    if plan.vocabulary.length > 0 && !embed_key.is_empty() {
        // Collect the raw BF16/FP16 bytes; lazy-load from mmap if needed.
        let raw_bytes: Vec<u8> = {
            let st = loaded.source_tensors.get_mut(embed_key).unwrap();
            for mmap in &loaded.mmap_bytes {
                crate::compute_image::compile::source::ensure_tensor_loaded(st, mmap);
                if !st.data.is_empty() { break; }
            }
            st.data.clone()
        };
        let (vocab_out_dim, vocab_in_dim) = {
            let st = &loaded.source_tensors[embed_key];
            if st.shape.len() == 2 {
                (st.shape[0], st.shape[1])
            } else {
                let elems = raw_bytes.len() as u32 / 2;
                (elems / hidden_size, hidden_size)
            }
        };
        eprintln!("[cimage] Vocabulary: {} ({}×{}) → tile640", embed_key, vocab_out_dim, vocab_in_dim);

        // Reserve the vocabulary slice in the mmap.
        // Capture mmap_base and vocab_file_offset before mutable borrow of builder.
        let mmap_capture = builder.mmap_base();
        let vocab_file_offset = plan.vocabulary.offset;
        let vocab_slice = builder.allocate_slice(plan.vocabulary.length as usize);
        let num_tiles = (vocab_in_dim as u64 + 639) / 640;
        let packed_len = (num_tiles as usize) * 32 * 4 * vocab_out_dim as usize;
        let scales_len = plan.vocabulary.length as usize - packed_len;

        // Try GPU-accelerated path first.
        #[cfg(feature = "metal-dispatch")]
        let gpu_done = {
            let mmap_base = mmap_capture;
            let vocab_file_offset = plan.vocabulary.offset;
            let result = crate::compute_image::compile::try_ternary_tile640_pack_gpu(
                loaded,
                embed_key,
                &raw_bytes,
                vocab_out_dim,
                vocab_in_dim,
                Some((mmap_base, vocab_file_offset)),
            );
            match result {
                Ok(true) => {
                    eprintln!("[cimage] Vocabulary → GPU tile640 done");
                    true
                }
                Ok(false) | Err(_) => false,
            }
        };
        #[cfg(not(feature = "metal-dispatch"))]
        let gpu_done = false;

        // CPU fallback: run apply_ternary_tile640_quantize and copy into mmap.
        if !gpu_done {
            eprintln!("[cimage] Vocabulary → CPU tile640 fallback");
            let f32_vals: Vec<f32> = raw_bytes
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    // FP16 fast path (also handles BF16 with minor precision loss).
                    let st = &loaded.source_tensors[embed_key];
                    if st.dtype == "BF16" {
                        let bf_bits = (bits as u32) << 16;
                        f32::from_bits(bf_bits)
                    } else {
                        crate::compute_image::compile::half_to_f32(bits)
                    }
                })
                .collect();
            let tile_size: usize = 640;
            let padded_cols = ((vocab_in_dim as usize + tile_size - 1) / tile_size) * tile_size;
            let tile_count = padded_cols / tile_size;
            let u32_per_tile = 32usize;
            let total_u32s = vocab_out_dim as usize * tile_count * u32_per_tile;
            let mut packed_u32 = vec![0u32; total_u32s];
            let mut scales_f32 = Vec::with_capacity(vocab_out_dim as usize * tile_count);

            for row in 0..vocab_out_dim as usize {
                let row_offset = row * vocab_in_dim as usize;
                let row_padded: Vec<f32> = {
                    let mut r = f32_vals[row_offset..row_offset + vocab_in_dim as usize].to_vec();
                    r.resize(padded_cols, 0.0);
                    r
                };
                for t in 0..tile_count {
                    let tile = &row_padded[t * tile_size..(t + 1) * tile_size];
                    let absmax = tile.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let scale = if absmax > 1e-12 { absmax } else { 1.0 };
                    scales_f32.push(scale);
                    let inv = 1.0 / scale;
                    for lane in 0..32usize {
                        let mut v: u32 = 0;
                        for w in 0..20usize {
                            let idx = lane * 20 + w;
                            let q = if idx < tile.len() {
                                let x = tile[idx] * inv;
                                if x > 0.5 { 1 } else if x < -0.5 { 2 } else { 0 }
                            } else { 0 };
                            v = v * 3 + q;
                        }
                        packed_u32[row * tile_count * u32_per_tile + t * u32_per_tile + lane] = v;
                    }
                }
            }
            let packed_bytes: Vec<u8> = packed_u32.iter().flat_map(|&w| w.to_le_bytes()).collect();
            let scales_bytes: Vec<u8> = scales_f32.iter().flat_map(|&s| s.to_le_bytes()).collect();
            let copy_len = packed_bytes.len().min(packed_len);
            vocab_slice[..copy_len].copy_from_slice(&packed_bytes[..copy_len]);
            let scale_dst = &mut vocab_slice[packed_len..packed_len + scales_len.min(scales_bytes.len())];
            scale_dst.copy_from_slice(&scales_bytes[..scale_dst.len()]);
        }
    } else if plan.vocabulary.length > 0 {
        // Vocabulary was requested but embed key not found — zero-fill the slot.
        builder.allocate_slice(plan.vocabulary.length as usize);
    }

    // Segment: LayerDirectory (per-layer weight/scale byte offsets)
    builder.align_cursor();
    if num_layers > 0 && plan.layer_directory.length > 0 {
        let layer_weight_bytes = plan.main_weights.length / num_layers as u64;
        let layer_elements = main_weight_total_elements / num_layers as u64;
        let layer_scale_bytes = ((layer_elements + 255) / 256) * 2;

        let layer_dir_slice = builder.allocate_slice(plan.layer_directory.length as usize);
        let num_entries = num_layers as usize;
        let mut entries: Vec<u8> = Vec::with_capacity(num_entries * 48);
        for l in 0..num_layers as u64 {
            let e = LayerDirectoryEntry {
                weights_offset: l * layer_weight_bytes,
                weights_length: layer_weight_bytes,
                scales_offset: l * layer_scale_bytes,
                scales_length: layer_scale_bytes,
                layer_kind: 0,
                flags: 0,
            };
            entries.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &e as *const LayerDirectoryEntry as *const u8,
                    48,
                )
            });
        }
        layer_dir_slice[..entries.len()].copy_from_slice(&entries);
        eprintln!(
            "[cimage] LayerDirectory: {} entries x 48B, {:.1} KB per layer",
            num_layers,
            layer_weight_bytes as f64 / 1024.0,
        );
    }

    // Header at offset 0
    let mut segments = [SegmentEntry { kind: 0, offset: 0, length: 0 }; 9];
    segments[0] = SegmentEntry::new(SegmentKind::MetalLib, plan.metal_lib.offset, plan.metal_lib.length);
    segments[1] = SegmentEntry::new(SegmentKind::TopologyTable, plan.topology_table.offset, plan.topology_table.length);
    // Segment 2: Vocabulary (TernaryTile640-packed embed_tokens.weight)
    if plan.vocabulary.length > 0 {
        segments[2] = SegmentEntry::new(SegmentKind::Vocabulary, plan.vocabulary.offset, plan.vocabulary.length);
    }
    segments[5] = SegmentEntry::new(SegmentKind::AneArchive, plan.main_graph.offset, plan.main_graph.length);
    segments[6] = SegmentEntry::new(SegmentKind::TernaryWeights, plan.main_weights.offset, plan.main_weights.length);
    if plan.mtp_graph.length > 0 {
        // If MTP present, insert as a second AneArchive or LayoutMeta
        segments[3] = SegmentEntry::new(SegmentKind::AneArchive, plan.mtp_graph.offset, plan.mtp_graph.length);
        segments[4] = SegmentEntry::new(SegmentKind::TernaryWeights, plan.mtp_weights.offset, plan.mtp_weights.length);
    }
    // Segment 7: LayerDirectory (per-layer weight/scale offset table)
    if num_layers > 0 {
        segments[7] = SegmentEntry::new(
            SegmentKind::LayerDirectory,
            plan.layer_directory.offset,
            plan.layer_directory.length,
        );
    }
    let vocab_seg = if plan.vocabulary.length > 0 { 1 } else { 0 };
    let mtp_segs = if plan.mtp_graph.length > 0 { 2 } else { 0 };
    let layer_dir_seg = if num_layers > 0 { 1 } else { 0 };
    let seg_count = 5u32 + mtp_segs + vocab_seg + layer_dir_seg;
    let header = CimageHeader {
        magic: *b"PRISM\0\0\0",
        version: 4,
        segment_count: seg_count,
        payload_hash: [0u8; 32],
        num_layers, num_heads: 0, head_dim: 0,
        hidden_dim: 0, intermediate_dim: 0, vocab_size: 0,
        quantization_schema: 0,
            draft_num_layers: 0,
        segments,
        _pad: [0u8; 8],
    };
    let saved = builder.current_offset();
    builder.cursor = 0;
    builder.write_header(&header);
    builder.cursor = saved as usize;

    let mmap = builder.into_mmap();
    mmap.flush()?;
    eprintln!("✅ .cimage: {} bytes, {} segments", plan.total_file_size, header.segment_count);
    Ok(())
}

// ── Sequential fallback (legacy, kept for compatibility) ────────────

const APPLE_PAGE_SIZE: u64 = super::APPLE_PAGE_SIZE as u64;

fn align_to_page<W: std::io::Write + std::io::Seek>(writer: &mut W) -> std::io::Result<u64> {
    let current_pos = writer.stream_position()?;
    let remainder = current_pos % APPLE_PAGE_SIZE;
    if remainder != 0 {
        let padding = vec![0u8; (APPLE_PAGE_SIZE - remainder) as usize];
        writer.write_all(&padding)?;
    }
    writer.stream_position()
}

pub fn pack_unified_cimage(
    output_path: &str,
    metal_lib_bytes: &[u8],
    main_graph_bytes: &[u8],
    main_weights_bytes: &[u8],
    mtp_graph_bytes: &[u8],
    mtp_weights_bytes: &[u8],
) -> std::io::Result<()> {
    let file = File::create(output_path)?;
    let mut writer = std::io::BufWriter::new(file);
    let header_size = std::mem::size_of::<CimageHeader>() as u64;
    // Reserve header space
    writer.write_all(&vec![0u8; header_size as usize])?;

    fn write_segment(writer: &mut std::io::BufWriter<File>, data: &[u8]) -> std::io::Result<(u64, u64)> {
        let offset = align_to_page(writer)?;
        writer.write_all(data)?;
        Ok((offset, data.len() as u64))
    }

    let (metal_lib_offset, metal_lib_len) = write_segment(&mut writer, metal_lib_bytes)?;
    let (main_weights_offset, main_weights_len) = write_segment(&mut writer, main_weights_bytes)?;
    let (mtp_weights_offset, mtp_weights_len) = write_segment(&mut writer, mtp_weights_bytes)?;
    let (main_graph_offset, main_graph_len) = write_segment(&mut writer, main_graph_bytes)?;
    let (mtp_graph_offset, mtp_graph_len) = write_segment(&mut writer, mtp_graph_bytes)?;

    let mut segments = [SegmentEntry { kind: 0, offset: 0, length: 0 }; 9];
    segments[0] = SegmentEntry::new(SegmentKind::MetalLib, metal_lib_offset, metal_lib_len);
    segments[1] = SegmentEntry::new(SegmentKind::TernaryWeights, main_weights_offset, main_weights_len);
    segments[2] = SegmentEntry::new(SegmentKind::TernaryWeights, mtp_weights_offset, mtp_weights_len);
    segments[3] = SegmentEntry::new(SegmentKind::AneArchive, main_graph_offset, main_graph_len);
    segments[4] = SegmentEntry::new(SegmentKind::AneArchive, mtp_graph_offset, mtp_graph_len);

    let header = CimageHeader {
        magic: *b"PRISM\0\0\0",
        version: 4,
        segment_count: if mtp_graph_bytes.is_empty() { 3 } else { 5 },
        payload_hash: [0u8; 32],
        num_layers: 0, num_heads: 0, head_dim: 0, hidden_dim: 0,
        intermediate_dim: 0, vocab_size: 0, quantization_schema: 0,
            draft_num_layers: 0,
        segments,
        _pad: [0u8; 8],
    };

    writer.seek(std::io::SeekFrom::Start(0))?;
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            (&header as *const CimageHeader) as *const u8, header_size as usize,
        )
    };
    writer.write_all(header_bytes)?;
    writer.flush()?;
    Ok(())
}

/// Pack a directory of compiled cimage fragments into a single .cimage binary.
///
/// Reads: model.metallib, segment_*.bin (weights), *.ane.tar (ANE archives)
/// from `input_dir` and produces a V4 unified .cimage at `output_path`.
/// Uses `ftruncate` + `mmap` + `AlignedMmapBuilder` for zero-copy GPU
/// compatibility (16 KB page alignment).
pub fn pack_cimage_from_dir(
    input_dir: &Path,
    output_path: &Path,
) -> std::io::Result<()> {
    // 1. Discover all segments in the directory
    let kernel_patterns: &[(&str, SegmentKind)] = &[
        ("model.metallib", SegmentKind::MetalLib),
        ("model.cubin",    SegmentKind::CudaLib),
        ("model.fatbin",   SegmentKind::CudaLib),
        ("model.co",       SegmentKind::RocmLib),
        ("model.hsaco",    SegmentKind::RocmLib),
        ("model_l0.spv",   SegmentKind::LevelZeroLib),
        ("model_vulkan.spv", SegmentKind::VulkanLib),
        ("model_wgsl.spv", SegmentKind::WebGpuLib),
    ];
    let npu_patterns: &[(&str, SegmentKind)] = &[
        ("npu_intel.bin",    SegmentKind::IntelNpuBlob),
        ("npu_amdxdna.bin", SegmentKind::AmdNpuBlob),
        ("npu_qualcomm.bin", SegmentKind::QualcommNpuBlob),
        ("npu_google.bin",   SegmentKind::GoogleTpuBlob),
        ("npu_ane.tar",      SegmentKind::AneArchive),
        ("npu_huawei.bin",   SegmentKind::HuaweiAscendBlob),
        ("npu_hailo.hef",    SegmentKind::HailoBlob),
    ];

    let mut weight_segments: Vec<Vec<u8>> = Vec::new();
    let mut extra_segments: Vec<(SegmentKind, Vec<u8>)> = Vec::new();
    for entry in std::fs::read_dir(input_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy().to_string();
        if name_str.starts_with("segment_") && name_str.ends_with(".bin") {
            weight_segments.push(std::fs::read(entry.path())?);
            continue;
        }
        let mut matched = false;
        for (pat, kind) in kernel_patterns {
            if name_str == *pat {
                extra_segments.push((*kind, std::fs::read(entry.path())?));
                matched = true; break;
            }
        }
        if matched { continue; }
        for (pat, kind) in npu_patterns {
            if name_str == *pat || (kind == &SegmentKind::AneArchive && name_str.ends_with(".ane.tar")) {
                extra_segments.push((*kind, std::fs::read(entry.path())?));
                matched = true; break;
            }
        }
    }

    // 2. Compute layout
    struct Slot { kind: SegmentKind, offset: u64, length: u64 }
    let mut slots: Vec<Slot> = Vec::new();
    let header_size = std::mem::size_of::<CimageHeader>() as u64;
    let weights_total: u64 = weight_segments.iter().map(|d| d.len() as u64).sum();

    let mut cursor = header_size as u64;
    let mut push_slot = |kind: SegmentKind, len: u64| {
        if len == 0 { return; }
        let r = cursor % APPLE_PAGE_SIZE;
        if r != 0 { cursor += APPLE_PAGE_SIZE - r; }
        slots.push(Slot { kind, offset: cursor, length: len });
        cursor += len;
    };
    for (kind, data) in &extra_segments {
        match kind {
            SegmentKind::MetalLib | SegmentKind::CudaLib
            | SegmentKind::RocmLib | SegmentKind::LevelZeroLib
            | SegmentKind::VulkanLib | SegmentKind::WebGpuLib
                => push_slot(*kind, data.len() as u64),
            _ => {}
        }
    }
    push_slot(SegmentKind::TernaryWeights, weights_total);
    for (kind, data) in &extra_segments {
        match kind {
            SegmentKind::AneArchive | SegmentKind::IntelNpuBlob
            | SegmentKind::AmdNpuBlob | SegmentKind::QualcommNpuBlob
            | SegmentKind::GoogleTpuBlob | SegmentKind::HuaweiAscendBlob
            | SegmentKind::HailoBlob
                => push_slot(*kind, data.len() as u64),
            _ => {}
        }
    }
    let total_file_size = cursor;

    // 3. Allocate and fill via ftruncate + mmap
    use std::fs::OpenOptions;
    let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(output_path)?;
    file.set_len(total_file_size)?;
    let mut mmap = unsafe { MmapMut::map_mut(&file)? };
    unsafe { std::ptr::write_bytes(mmap.as_mut_ptr(), 0u8, mmap.len()); }
    let mut builder = AlignedMmapBuilder::new(mmap);

    // Skip header (written last)
    builder.cursor = header_size as usize;

    // Write kernel segments
    for (kind, bytes) in &extra_segments {
        match kind {
            SegmentKind::MetalLib | SegmentKind::CudaLib
            | SegmentKind::RocmLib | SegmentKind::LevelZeroLib
            | SegmentKind::VulkanLib | SegmentKind::WebGpuLib => {
                if !bytes.is_empty() {
                    builder.align_cursor();
                    builder.allocate_slice(bytes.len()).copy_from_slice(bytes);
                }
            }
            _ => {}
        }
    }

    // Weight segments (concatenated into one TernaryWeights segment)
    if weights_total > 0 {
        builder.align_cursor();
        let mut seg_slice = builder.allocate_slice(weights_total as usize);
        for data in &weight_segments {
            let (head, tail) = seg_slice.split_at_mut(data.len());
            head.copy_from_slice(data);
            seg_slice = tail;
        }
    }

    // NPU model segments
    for (kind, bytes) in &extra_segments {
        match kind {
            SegmentKind::AneArchive | SegmentKind::IntelNpuBlob
            | SegmentKind::AmdNpuBlob | SegmentKind::QualcommNpuBlob
            | SegmentKind::GoogleTpuBlob | SegmentKind::HuaweiAscendBlob
            | SegmentKind::HailoBlob => {
                if !bytes.is_empty() {
                    builder.align_cursor();
                    builder.allocate_slice(bytes.len()).copy_from_slice(bytes);
                }
            }
            _ => {}
        }
    }

    // 4. Build header
    let total_segments = slots.len().min(8);
    let mut segments_dir = [SegmentEntry { kind: 0, offset: 0, length: 0 }; 9];
    for (i, slot) in slots.iter().enumerate().take(8) {
        segments_dir[i] = SegmentEntry::new(slot.kind, slot.offset, slot.length);
    }

    let header = CimageHeader {
        magic: *b"PRISM\0\0\0",
        version: 4,
        segment_count: total_segments as u32,
        payload_hash: [0u8; 32],
        num_layers: 0, num_heads: 0, head_dim: 0,
        hidden_dim: 0, intermediate_dim: 0, vocab_size: 0,
        quantization_schema: 0,
            draft_num_layers: 0,
        segments: segments_dir,
        _pad: [0u8; 8],
    };

    let saved = builder.current_offset();
    builder.cursor = 0;
    builder.write_header(&header);
    builder.cursor = saved as usize;

    let mmap = builder.into_mmap();
    mmap.flush()?;
    eprintln!("[cimage] packed {} bytes, {} segments → {}", total_file_size, total_segments, output_path.display());
    Ok(())
}
