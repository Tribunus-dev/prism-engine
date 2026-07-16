//! Standalone diagnostic: load patch_dense.weight from HF cache,
//! run nf4tile640 pack→unpack round-trip, profile per-group/tile error.
//!
//! Usage: cargo run --bin diagnose-nf4-roundtrip --features prism-backend

use std::path::PathBuf;

use hf_hub::api::sync::Api;
use memmap2::Mmap;
use safetensors::SafeTensors;

use tribunus_compute_core::nf4tile640::{
    pack_nf4_weights, pack_nf4_weights_awls, unpack_nf4_weights, GROUPS_PER_TILE, GROUP_SIZE,
    TILE_ELEMENTS,
};

fn load_tensor(key: &str, shards: &[(PathBuf, Mmap)]) -> Option<(Vec<f32>, Vec<usize>)> {
    let (_path, mmap) = shards.iter().find(|(_, mmap)| {
        SafeTensors::deserialize(mmap)
            .ok()
            .and_then(|st| st.tensor(key).ok())
            .is_some()
    })?;
    let st = SafeTensors::deserialize(mmap).ok()?;
    let view = st.tensor(key).ok()?;
    let shape = view.shape().to_vec();
    let f32_vals = match view.dtype() {
        safetensors::Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        safetensors::Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|c| {
                let u = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((u as u32) << 16)
            })
            .collect(),
        _ => panic!("unsupported dtype: {:?}", view.dtype()),
    };
    Some((f32_vals, shape))
}

fn profile_packer(data: &[f32], rows: usize, cols: usize, label: &str) {
    // 1. Pack with max-abs
    let (codes, scales, biases, _, _) = pack_nf4_weights(data, rows, cols);

    // 2. Unpack
    let unpacked = unpack_nf4_weights(&codes, &scales, &biases, rows, cols);

    // 3. Global RMSE
    let mut sq_sum = 0.0f64;
    for i in 0..rows * cols {
        let diff = (data[i] - unpacked[i]) as f64;
        sq_sum += diff * diff;
    }
    let rmse = (sq_sum / (rows * cols) as f64).sqrt();
    println!("\n=== {label}: {rows} × {cols} ===");
    println!("Global RMSE: {:.6}", rmse);

    // 4. Per-group dynamic range and RMSE
    let tiles_per_row = cols.div_ceil(TILE_ELEMENTS);
    let mut max_group_range = 0.0f64;
    let mut high_error_groups = 0u64;

    for row in 0..rows {
        for tile_idx in 0..tiles_per_row {
            let col_start = tile_idx * TILE_ELEMENTS;
            for g in 0..GROUPS_PER_TILE {
                let group_start = col_start + g * GROUP_SIZE;
                if group_start >= cols {
                    break;
                }

                let n_in_group = GROUP_SIZE.min(cols - group_start);

                // Per-group RMSE
                let mut grp_sq = 0.0f64;
                let mut max_abs = 0.0f64;
                let mut vals = Vec::new();
                for i in 0..n_in_group {
                    let idx = row * cols + group_start + i;
                    let v = data[idx].abs() as f64;
                    vals.push(v);
                    if v > max_abs {
                        max_abs = v;
                    }
                    let diff = (data[idx] - unpacked[idx]) as f64;
                    grp_sq += diff * diff;
                }
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = vals[vals.len() / 2];
                let range_ratio = max_abs / (median + 1e-10);
                if range_ratio > max_group_range {
                    max_group_range = range_ratio;
                }

                let grp_rmse = (grp_sq / n_in_group as f64).sqrt();
                if grp_rmse > 0.1 {
                    high_error_groups += 1;
                    if high_error_groups <= 10 {
                        println!(
                            "  HIGH ERR [row={row}, tile={tile_idx}, group={g}]: RMSE={grp_rmse:.4}, range={range_ratio:.1} (max_abs={max_abs:.4})"
                        );
                    }
                }
            }
        }
    }
    println!("Max group range ratio: {:.1}", max_group_range);
    println!("High-error groups (>0.1 RMSE): {}", high_error_groups);

    // 5. Per-row RMSE — identifies stride drift
    let step = rows / 5;
    for row in (0..rows).step_by(step).take(5) {
        let mut row_sq = 0.0f64;
        for j in 0..cols {
            let idx = row * cols + j;
            let diff = (data[idx] - unpacked[idx]) as f64;
            row_sq += diff * diff;
        }
        let row_rmse = (row_sq / cols as f64).sqrt();
        println!("  Row {row} RMSE: {row_rmse:.4}");
    }

    // 6. Tile-level analysis (first row and last row)
    println!();
    for tile_idx in 0..tiles_per_row {
        let col_start = tile_idx * TILE_ELEMENTS;
        let n_cols = TILE_ELEMENTS.min(cols - col_start);

        let mut tile_sq_first = 0.0f64;
        for i in 0..n_cols {
            let idx = 0 * cols + col_start + i;
            let diff = (data[idx] - unpacked[idx]) as f64;
            tile_sq_first += diff * diff;
        }
        let tile_rmse_first = (tile_sq_first / n_cols as f64).sqrt();

        let mut tile_sq_last = 0.0f64;
        for i in 0..n_cols {
            let idx = (rows - 1) * cols + col_start + i;
            let diff = (data[idx] - unpacked[idx]) as f64;
            tile_sq_last += diff * diff;
        }
        let tile_rmse_last = (tile_sq_last / n_cols as f64).sqrt();

        println!(
            "  tile[{tile_idx:2}] col={col_start}..{col_end:5}:  row0 RMSE={tile_rmse_first:.4}  row{rows} RMSE={tile_rmse_last:.4}",
            col_end = col_start + n_cols,
        );
    }
}

fn main() {
    let repo_id = "google/gemma-4-12B-it-qat-q4_0-unquantized";
    let api = Api::new().expect("HF API init");
    let repo = api.model(repo_id.to_string());

    println!("Loading model.safetensors...");
    let local_path = repo.get("model.safetensors").unwrap();
    let file = std::fs::File::open(&local_path).unwrap();
    let file_len = file.metadata().unwrap().len();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let shards = vec![(local_path, mmap)];
    println!("  mmap'd {:.1} GB", file_len as f64 / 1e9);

    // Test 1: patch_dense.weight — the vision projection
    let tensor_name = "model.vision_embedder.patch_dense.weight";
    if let Some((data, shape)) = load_tensor(tensor_name, &shards) {
        let rows = shape[0];
        let cols = shape[1];
        assert_eq!(data.len(), rows * cols);
        println!("Loaded {}: {}×{}", tensor_name, rows, cols);

        profile_packer(&data, rows, cols, "patch_dense.weight max-abs");

        // Also try AW-LS
        let channel_sq: Vec<f32> = (0..cols)
            .map(|j| {
                let mut sum = 0.0f64;
                for i in 0..rows {
                    let v = data[i * cols + j] as f64;
                    sum += v * v;
                }
                (sum / rows as f64) as f32
            })
            .collect();

        let (c_awls, s_awls, b_awls, _, _) =
            pack_nf4_weights_awls(&data, rows, cols, Some(&channel_sq), 8);
        let unpacked_awls = unpack_nf4_weights(&c_awls, &s_awls, &b_awls, rows, cols);

        let mut sq_awls = 0.0f64;
        for i in 0..rows * cols {
            let diff = (data[i] - unpacked_awls[i]) as f64;
            sq_awls += diff * diff;
        }
        let rmse_awls = (sq_awls / (rows * cols) as f64).sqrt();
        println!(
            "\n=== patch_dense.weight AW-LS: global RMSE = {:.6} ===",
            rmse_awls
        );
    } else {
        eprintln!("Tensor '{}' not found!", tensor_name);
    }

    // Test 2: reference — a 3840×15360 layer (exact multiple of 640)
    let ref_key = "model.layers.0.mlp.gate_proj.weight";
    if let Some((data, shape)) = load_tensor(ref_key, &shards) {
        let rows = shape[0];
        let cols = shape[1];
        assert_eq!(data.len(), rows * cols);
        println!(
            "\nLoaded {}: {}×{} (cols ÷ 640 = {})",
            ref_key,
            rows,
            cols,
            cols / TILE_ELEMENTS
        );

        profile_packer(&data, rows, cols, "gate_proj max-abs");
    }

    // Test 3: another partial-tile reference — 3840×4096 (q_proj)
    let ref_key2 = "model.layers.0.self_attn.q_proj.weight";
    if let Some((data, shape)) = load_tensor(ref_key2, &shards) {
        let rows = shape[0];
        let cols = shape[1];
        assert_eq!(data.len(), rows * cols);
        println!(
            "\nLoaded {}: {}×{} (cols ÷ 640 = {:.1})",
            ref_key2,
            rows,
            cols,
            cols as f64 / TILE_ELEMENTS as f64
        );

        profile_packer(&data, rows, cols, "q_proj max-abs");
    }
}
