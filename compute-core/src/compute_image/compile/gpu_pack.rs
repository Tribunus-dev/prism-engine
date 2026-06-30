// GPU-accelerated TernaryTile640 pack via Metal.
// Included from quantize.rs via `include!()`. Uses the parent module's imports.
#[cfg(feature = "metal-dispatch")]
use metal::*;
use std::sync::LazyLock;

static METAL: LazyLock<Option<(Device, CommandQueue, ComputePipelineState)>> =
    LazyLock::new(|| {
        let device = Device::system_default()?;
        let src = include_str!("../templates/tile640_pack.metal");
        let lib = device
            .new_library_with_source(src, &CompileOptions::new())
            .ok()?;
        let kernel = lib.get_function("tile640_pack", None).ok()?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .ok()?;
        Some((device.clone(), device.new_command_queue(), pipeline))
    });

static Q8_METAL: LazyLock<Option<(Device, CommandQueue, ComputePipelineState)>> =
    LazyLock::new(|| {
        let device = Device::system_default()?;
        let src = include_str!("../templates/tile640_pack.metal");
        let lib = device
            .new_library_with_source(src, &CompileOptions::new())
            .ok()?;
        let kernel = lib.get_function("q8_0_ternary_pack", None).ok()?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .ok()?;
        Some((device.clone(), device.new_command_queue(), pipeline))
    });

/// GPU-accelerated TernaryTile640 pack with optional direct-to-mmap output.
///
/// When `mmap_output` is `Some((ptr, offset))`, the GPU writes packed u32
/// data directly into the pre-allocated .cimage mmap via Metal's
/// `newBufferWithBytesNoCopy` — zero CPU copies of the compressed weights.
/// Scales are always returned to the CPU (they are small — one f32 per tile).
pub(crate) fn try_ternary_tile640_pack_gpu(
    loaded: &mut LoadedSource,
    weight_name: &str,
    raw_bytes: &[u8],
    out_dim: u32,
    in_dim: u32,
    // Optional (mmap_base_ptr, weights_segment_offset_within_mmap).
    // When set, `newBufferWithBytesNoCopy` binds output directly into the
    // file-backed mmap at the pre-computed offset for this tensor.
    mmap_output: Option<(*mut u8, u64)>,
) -> crate::Result<bool> {
    let (ref device, ref queue, ref pipeline) = match METAL.as_ref() {
        Some(m) => m,
        None => return Ok(false),
    };

    let (out_dim_u, in_dim_u) = (out_dim as usize, in_dim as usize);
    let num_tiles = (in_dim_u + 639) / 640;
    let padded_in = num_tiles * 640;
    let total_u32_bytes = (out_dim_u * num_tiles * 32) as u64 * 4;

    // Shared-memory buffers (UMA: CPU and GPU see the same physical RAM).
    let ingest = device.new_buffer(
        (out_dim_u as u64) * (padded_in as u64) * 2,
        MTLResourceOptions::StorageModeShared,
    );

    // Egest buffer: either direct-to-mmap or a regular shared buffer.
    let egest_packed: metal::Buffer = match mmap_output {
        Some((mmap_base, weights_offset)) => {
            // Bind GPU output directly into the .cimage file mmap.
            let out_ptr = unsafe { mmap_base.add(weights_offset as usize) };
            let buf = device.new_buffer_with_bytes_no_copy(
                out_ptr as *mut std::ffi::c_void,
                total_u32_bytes,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            buf
        }
        None => {
            device.new_buffer(total_u32_bytes, MTLResourceOptions::StorageModeShared)
        }
    };

    let egest_scales = device.new_buffer(
        (out_dim_u as u64) * (num_tiles as u64) * 4,
        MTLResourceOptions::StorageModeShared,
    );

    // Copy BF16 data row-by-row into the ingest buffer, zero-padding to 640.
    let ingest_ptr = ingest.contents() as *mut u8;
    for row in 0..out_dim_u {
        let src = row * in_dim_u * 2;
        let dst = row * padded_in * 2;
        unsafe {
            std::ptr::copy_nonoverlapping(raw_bytes.as_ptr().add(src), ingest_ptr.add(dst), in_dim_u * 2);
            std::ptr::write_bytes(ingest_ptr.add(dst + in_dim_u * 2), 0u8, (padded_in - in_dim_u) * 2);
        }
    }

    // Dispatch the GPU kernel.
    let cmd_buf = queue.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&ingest), 0);
    enc.set_buffer(1, Some(&egest_packed), 0);
    enc.set_buffer(2, Some(&egest_scales), 0);

    let k = in_dim;
    let n = out_dim;
    let nt = num_tiles as u32;
    for (i, &val) in [k, n, nt].iter().enumerate() {
        let buf = device.new_buffer_with_data(
            &val as *const u32 as *const std::ffi::c_void,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        enc.set_buffer(3 + i as u64, Some(&buf), 0);
    }

    enc.dispatch_threads(
        MTLSize {
            width: (out_dim_u as u64) * (num_tiles as u64),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // Read back F32 scales (small: one f32 per tile).
    let scales_slice = unsafe {
        std::slice::from_raw_parts(
            egest_scales.contents() as *const f32,
            (out_dim_u * num_tiles) as usize,
        )
    };
    let scales_bytes: Vec<u8> = scales_slice
        .iter()
        .flat_map(|&s| s.to_le_bytes().to_vec())
        .collect();

    let stem = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let scales_name = format!("{}.scales", stem);
    let total_u32_count = (out_dim_u * num_tiles * 32) as u32;
    let packed_shape = crate::config::PackedLinearShapes {
        weight: vec![out_dim, total_u32_count],
        scales: vec![out_dim, num_tiles as u32],
        biases: vec![out_dim, num_tiles as u32],
        bits: 2,
        group_size: 640,
        groups: (out_dim_u * num_tiles) as u32,
    };

    // When mmap_output is set, the packed data was written directly to the
    // file by the GPU — no CPU copy needed. We still update the source tensor
    // metadata so the emission pipeline knows the shape/dtype, but store an
    // empty Vec to avoid a redundant memory allocation.
    if let Some(st) = loaded.source_tensors.get_mut(weight_name) {
        match mmap_output {
            Some(_) => {
                st.data = Vec::new(); // data is already in the mmap
            }
            None => {
                let packed_slice = unsafe {
                    std::slice::from_raw_parts(
                        egest_packed.contents() as *const u32,
                        total_u32_count as usize,
                    )
                };
                st.data = packed_slice.iter().flat_map(|&w| w.to_le_bytes().to_vec()).collect();
            }
        }
        st.dtype = "U32".to_string();
        st.shape = vec![out_dim, total_u32_count];
    }
    loaded.source_tensors.insert(
        scales_name.clone(),
        SourceTensor {
            name: scales_name,
            dtype: "F32".into(),
            shape: vec![out_dim, num_tiles as u32],
            data: scales_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );
    for binding in &mut loaded.spec.global_tensors {
        if binding.name == weight_name && binding.packed_shape.is_none() {
            binding.packed_shape = Some(packed_shape.clone());
        }
    }
    for layer in &mut loaded.spec.layers {
        for binding in &mut layer.tensors {
            if binding.name == weight_name && binding.packed_shape.is_none() {
                binding.packed_shape = Some(packed_shape.clone());
            }
        }
    }

    eprintln!(
        "[quantize:gpu] tile640 packed {}: {}×{} → {} tiles, {} u32 {}",
        weight_name,
        out_dim,
        in_dim,
        num_tiles,
        total_u32_count,
        if mmap_output.is_some() { "→ direct mmap" } else { "" },
    );
    Ok(true)
}

/// GPU-accelerated Q8_0 dequant → transpose → ternary tile640 pack.
///
/// Input: Q8_0 blocks in [K, N] order (GGUF native).
/// The function transposes block indices to [N, K] order on CPU (~0.5ms),
/// then dispatches the Metal `q8_0_ternary_pack` kernel which dequantizes
/// to f32 in threadgroup memory, computes per-tile absmax scale,
/// ternary quantizes, and Base-3 packs — all in one kernel dispatch.
///
/// Returns `(packed_u32_bytes, scales_f32_bytes, num_tiles)`.
/// On failure or missing Metal, returns `None`.
pub(crate) fn try_q8_0_ternary_pack_gpu(
q8_bytes: &[u8],
in_dim: u32,
out_dim: u32,
) -> Option<(Vec<u8>, Vec<u8>, u32)> {
match &Q8_METAL.as_ref() {
        Some((device, queue, pipeline)) => {
            let k = in_dim as usize;
            let n = out_dim as usize;
            let k_blocks = (k + 31) / 32;
            let total_blocks = k_blocks * n;
            let num_tiles = (k + 639) / 640;

            // ── Pre-scan scales for p99 clamp threshold ────────────
            // BF16→f16 overflow in Q8_0 conversion can produce inf/NaN
            // scales. Clamp non-finite to p99 of finite scales.
            let mut fin: Vec<f32> = Vec::with_capacity(total_blocks);
            for b in 0..total_blocks {
                let o = b * 34;
                let b0 = u16::from_le_bytes([q8_bytes[o], q8_bytes[o + 1]]);
                let s = half::f16::from_bits(b0).to_f32();
                if s.is_finite() { fin.push(s.abs()); }
}
            fin.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let clamp = fin.get((fin.len() as f64 * 0.99) as usize)
                .copied().unwrap_or(65504.0);
            let clamped_bits = half::f16::from_f32(clamp).to_bits().to_le_bytes();
            let clamped_count = total_blocks - fin.len()
                + fin.iter().filter(|&&s| s > clamp).count();
            if clamped_count > 0 {
                eprintln!("  [gpu] clamped {} inf/overflow scales to {:.0}", clamped_count, clamp);
}

            // ── CPU: transpose Q8_0 block indices [K,N] → [N,K] ─────
            // Blocks with inf/NaN scales get clamped to p99 threshold.
            let mut transposed = vec![0u8; total_blocks * 34];
            for row_n in 0..n {
                for k_blk in 0..k_blocks {
                    // Source: element (k_blk*32, row_n) in [K,N] layout
                    let src_flat = k_blk * 32 * n + row_n;
                    let src_block = src_flat / 32;
                    let src_off = src_block * 34;
                    // Dest: element (row_n, k_blk*32) in [N,K] layout
                    let dst_block = row_n * k_blocks + k_blk;
                    let dst_off = dst_block * 34;
                    let b0 = u16::from_le_bytes([q8_bytes[src_off], q8_bytes[src_off + 1]]);
                    let s = half::f16::from_bits(b0).to_f32();
                    if !s.is_finite() || s.abs() > clamp {
                        transposed[dst_off..dst_off + 2].copy_from_slice(&clamped_bits);
                        transposed[dst_off + 2..dst_off + 34]
                            .copy_from_slice(&q8_bytes[src_off + 2..src_off + 34]);
                    } else {
                    transposed[dst_off..dst_off + 34]
                        .copy_from_slice(&q8_bytes[src_off..src_off + 34]);
                    }
                }
            }

            // ── Upload transposed Q8_0 blocks to GPU ────────────────
            // ── Upload transposed Q8_0 blocks to GPU ────────────────
            let ingest = device.new_buffer_with_data(
                transposed.as_ptr() as *const std::ffi::c_void,
                transposed.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let total_u32 = n * num_tiles * 32;
            let egest_packed = device.new_buffer(
                (total_u32 as u64) * 4,
                MTLResourceOptions::StorageModeShared,
            );
            let egest_scales = device.new_buffer(
                (n as u64) * (num_tiles as u64) * 4,
                MTLResourceOptions::StorageModeShared,
            );

            // ── Dispatch kernel ─────────────────────────────────────
            let cmd_buf = queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(&ingest), 0);
            enc.set_buffer(1, Some(&egest_packed), 0);
            enc.set_buffer(2, Some(&egest_scales), 0);
            for (i, &val) in [k as u32, n as u32, num_tiles as u32].iter().enumerate() {
                let buf = device.new_buffer_with_data(
                    &val as *const u32 as *const std::ffi::c_void,
                    4,
                    MTLResourceOptions::StorageModeShared,
                );
                enc.set_buffer(3 + i as u64, Some(&buf), 0);
}
            enc.dispatch_threads(
                MTLSize {
                    width: (n as u64) * (num_tiles as u64),
                    height: 1,
                    depth: 1,
                },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            // ── Read back results ───────────────────────────────────
            let packed_slice = unsafe {
                std::slice::from_raw_parts(
                    egest_packed.contents() as *const u32,
                    total_u32,
                )
            };
            let packed_u32: Vec<u8> = packed_slice
                .iter()
                .flat_map(|&w| w.to_le_bytes().to_vec())
                .collect();

            let scales_slice = unsafe {
                std::slice::from_raw_parts(
                    egest_scales.contents() as *const f32,
                    (n * num_tiles) as usize,
                )
            };
            let scales_f32: Vec<u8> = scales_slice
                .iter()
                .flat_map(|&s| s.to_le_bytes().to_vec())
                .collect();

            eprintln!(
                "[quantize:gpu] q8_0→ternary {}×{}(K): {} tiles, {} u32",
                k, n, num_tiles, total_u32,
            );
            Some((packed_u32, scales_f32, num_tiles as u32))
}
        None => None,
}
}
