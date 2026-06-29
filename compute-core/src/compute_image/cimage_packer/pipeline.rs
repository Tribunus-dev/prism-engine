//! Top-level orchestration: AOT plan → mmap → segments → header.

use super::archive::archive_mlmodelc_to_mmap;
use super::builder::AlignedMmapBuilder;
use super::layout::{CImageLayoutPlan, CImageTopologyTable, predict_tar_size};
use crate::compute_image::compile::ternary::PrismCimageHeader;
use memmap2::MmapMut;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::mem::size_of;
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
pub fn stream_weights_to_mmap_gpu(
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

pub fn compile_and_pack_god_binary(
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
    let header_size = size_of::<PrismCimageHeader>() as u64;

    let plan = CImageLayoutPlan::calculate(
        header_size, metal_lib_len, main_graph_len,
        main_weight_total_elements, mtp_graph_len, mtp_weight_total_elements,
    );

    let topology_table = CImageTopologyTable::compute(
        hidden_size, intermediate_size,
        num_layers, num_heads, head_dim,
    );

    eprintln!(
        "[cimage] AOT layout: total={} metal_lib={} main_graph={} main_weights={} mtp_graph={} mtp_weights={}",
        plan.total_file_size,
        plan.metal_lib.length, plan.main_graph.length,
        plan.main_weights.length, plan.mtp_graph.length, plan.mtp_weights.length,
    );

    let file = File::create(output_path)?;
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

    // Header at offset 0
    let header = PrismCimageHeader {
        magic: *b"CIMAGE\0\0",
        version: 3, segment_count: 6,
        metal_lib_offset: plan.metal_lib.offset, metal_lib_len: plan.metal_lib.length,
        main_graph_offset: plan.main_graph.offset, main_graph_len: plan.main_graph.length,
        main_weights_offset: plan.main_weights.offset, main_weights_len: plan.main_weights.length,
        mtp_graph_offset: plan.mtp_graph.offset, mtp_graph_len: plan.mtp_graph.length,
        mtp_weights_offset: plan.mtp_weights.offset, mtp_weights_len: plan.mtp_weights.length,
        topology_table_offset: plan.topology_table.offset, topology_table_len: plan.topology_table.length,
        payload_hash: [0u8; 32],
        num_layers: 0,
        num_heads: 0,
        head_dim: 0,
        hidden_dim: 0,
        intermediate_dim: 0,
        vocab_size: 0,
        quantization_schema: 0,
        lane_isolation: 0,
        _pad: [0u8; 111],
    };
    let saved = builder.current_offset();
    builder.cursor = 0;
    builder.write_header(&header);
    builder.cursor = saved as usize;

    let mmap = builder.into_mmap();
    mmap.flush()?;
    eprintln!("✅ Gemma4_Unified.cimage: {} bytes, {} segments", plan.total_file_size, header.segment_count);
    Ok(())
}

// ── Sequential fallback (legacy, kept for compatibility) ────────────

const APPLE_PAGE_SIZE: u64 = super::APPLE_PAGE_SIZE as u64;

fn align_to_page<W: Write + Seek>(writer: &mut W) -> std::io::Result<u64> {
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
    let mut writer = BufWriter::new(file);
    let mut header = PrismCimageHeader {
        magic: *b"CIMAGE\0\0", version: 3, segment_count: 5,
        payload_hash: [0u8; 32],
        num_layers: 0, num_heads: 0, head_dim: 0,
        hidden_dim: 0, intermediate_dim: 0, vocab_size: 0,
        quantization_schema: 0,
        metal_lib_offset: 0, metal_lib_len: 0,
        main_graph_offset: 0, main_graph_len: 0,
        main_weights_offset: 0, main_weights_len: 0,
        mtp_graph_offset: 0, mtp_graph_len: 0,
        mtp_weights_offset: 0, mtp_weights_len: 0,
        topology_table_offset: 0, topology_table_len: 0,
        lane_isolation: 0, _pad: [0u8; 111],
    };
    let header_size = size_of::<PrismCimageHeader>() as u64;
    writer.write_all(&vec![0u8; header_size as usize])?;

    header.metal_lib_offset = align_to_page(&mut writer)?;
    writer.write_all(metal_lib_bytes)?;
    header.metal_lib_len = metal_lib_bytes.len() as u64;

    header.main_graph_offset = align_to_page(&mut writer)?;
    writer.write_all(main_graph_bytes)?;
    header.main_graph_len = main_graph_bytes.len() as u64;

    header.main_weights_offset = align_to_page(&mut writer)?;
    writer.write_all(main_weights_bytes)?;
    header.main_weights_len = main_weights_bytes.len() as u64;

    header.mtp_graph_offset = align_to_page(&mut writer)?;
    writer.write_all(mtp_graph_bytes)?;
    header.mtp_graph_len = mtp_graph_bytes.len() as u64;

    header.mtp_weights_offset = align_to_page(&mut writer)?;
    writer.write_all(mtp_weights_bytes)?;
    header.mtp_weights_len = mtp_weights_bytes.len() as u64;

    writer.seek(SeekFrom::Start(0))?;
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            (&header as *const PrismCimageHeader) as *const u8, header_size as usize,
        )
    };
    writer.write_all(header_bytes)?;
    writer.flush()?;
    Ok(())
}
