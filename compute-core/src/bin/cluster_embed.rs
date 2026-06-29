//! Cluster the FP16 embedding table from a v2 cimage into 256 semantic clusters,
//! then output reordered centroids, cluster assignments, and the reordered embed.
//!
//! Usage:
//!   cargo run --bin cluster-embed --features prism-backend -- \
//!     --cimage /path/to/model_v2.cimage \
//!     --output-dir /path/to/output/ \
//!     [--k 256] [--iters 20] [--max-rows 262144]

use clap::Parser;
use std::cell::Cell;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "prism-backend")]
use tribunus_compute_core::compute_image::cimage_loader::CimageDeployment;

// ── Half-precision conversion (inline, matches existing codebase pattern) ──

fn f32_from_half(x: u16) -> f32 {
    let bits = x as u32;
    let sign = bits & 0x8000;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    if exp == 0 {
        if mant == 0 {
            return 0.0;
        }
        let norm_exp: i32 = -14;
        let fp32_bits = sign << 16 | ((norm_exp + 127) as u32) << 23 | mant << 13;
        return f32::from_bits(fp32_bits);
    }
    if exp == 0x1F {
        let fp32_bits = sign << 16 | 0x7F800000u32 | mant << 13;
        return f32::from_bits(fp32_bits);
    }
    let fp32_exp = exp.wrapping_add(127 - 15);
    f32::from_bits(fp32_exp << 23 | mant << 13 | sign << 16)
}

fn f32_to_half(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;
    if exp == 0 {
        return sign;
    }
    if exp == 0xFF {
        return if mant == 0 {
            if (bits >> 31) != 0 {
                0xFC00
            } else {
                0x7C00
            }
        } else {
            0x7E00
        };
    }
    let exp_f16: i32 = exp as i32 - 127 + 15;
    if exp_f16 >= 0x1F {
        return if (bits >> 31) != 0 { 0xFC00 } else { 0x7C00 };
    }
    if exp_f16 <= 0 {
        return sign;
    }
    sign | ((exp_f16 as u16) << 10) | ((mant >> 13) as u16)
}

fn half_to_f32_slice(src: &[u16]) -> Vec<f32> {
    src.iter().map(|&h| f32_from_half(h)).collect()
}

fn f32_to_half_slice(src: &[f32]) -> Vec<u16> {
    src.iter().map(|&v| f32_to_half(v)).collect()
}

// ── Constants ──────────────────────────────────────────────────────────

/// Hidden dimension of the embedding table (Gemma 4 27B).
const HIDDEN_DIM: usize = 3840;

// ── Simple LCG RNG (replaces fastrand) ─────────────────────────────────

thread_local! {
    static SEED: Cell<u64> = Cell::new(42);
}

fn rand_f32() -> f32 {
    SEED.with(|seed| {
        let s = seed.get();
        let next = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed.set(next);
        // Use top 24 bits for a float in [0, 1)
        ((next >> 40) as f32) * (1.0 / (1u64 << 24) as f32)
    })
}

fn rand_f64() -> f64 {
    rand_f32() as f64
}

fn rand_range(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    SEED.with(|seed| {
        let s = seed.get();
        let next = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed.set(next);
        (next >> 33) as usize % n
    })
}

// ── CLI ────────────────────────────────────────────────────────────────

#[derive(Parser)]
struct Args {
    /// Path to compiled .cimage v2 file.
    #[arg(long)]
    cimage: PathBuf,

    /// Output directory for centroids, cluster_map, and reordered embed.
    #[arg(long)]
    output_dir: PathBuf,

    /// Number of clusters (k).
    #[arg(long, default_value = "256")]
    k: usize,

    /// Max k-means iterations.
    #[arg(long, default_value = "20")]
    iters: usize,

    /// Max rows to load from the embedding table (subset for testing).
    #[arg(long, default_value = "10000")]
    max_rows: usize,
}

// ── Embedding loading ───────────────────────────────────────────────────

/// Read the FP16 embedding table from CimageDeployment, convert to f32,
/// and return up to `max_rows` rows.
fn read_embed_f32(path: &str, max_rows: usize) -> (Vec<f32>, usize, usize) {
    let device = metal::Device::system_default().expect("Metal device required");
    let deployment = CimageDeployment::load(path, &device).expect("Failed to load cimage");

    let embed_buf = deployment
        .embed_buffer
        .as_ref()
        .expect("v2 cimage must have embed_buffer");
    let ptr = embed_buf.contents() as *const u16;
    let byte_len = embed_buf.length() as usize;
    let n_halves = byte_len / 2;
    let hidden_dim = HIDDEN_DIM;
    let vocab_size = n_halves / HIDDEN_DIM;

    let halves = unsafe { std::slice::from_raw_parts(ptr, n_halves) };
    let take_rows = max_rows.min(vocab_size);
    let take_halves = take_rows * hidden_dim;

    let f32_vec = half_to_f32_slice(&halves[..take_halves]);
    (f32_vec, take_rows, hidden_dim)
}

// ── K-Means++ initialization ───────────────────────────────────────────

/// K-Means++ centroid initialization.
/// Returns k × dim centroids in row-major order.
fn kmeans_plusplus(data: &[f32], k: usize, n_rows: usize, dim: usize) -> Vec<f32> {
    use std::collections::HashSet;

    let mut centroids: Vec<f32> = Vec::with_capacity(k * dim);
    let mut chosen: HashSet<usize> = HashSet::new();

    // 1st centroid: pick a random data point
    let first_idx = rand_range(n_rows);
    chosen.insert(first_idx);
    centroids.extend_from_slice(&data[first_idx * dim..(first_idx + 1) * dim]);

    // Distance cache: for each data point, store min squared distance to any chosen centroid
    let mut min_dist_sq: Vec<f32> = vec![f32::MAX; n_rows];

    // Initialize distances to first centroid
    for i in 0..n_rows {
        let row = &data[i * dim..(i + 1) * dim];
        let dist = row
            .iter()
            .zip(centroids.chunks_exact(dim).last().unwrap())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>();
        min_dist_sq[i] = dist;
    }

    for c in 1..k {
        // Sample proportionally to squared distance
        let total_dist: f64 = min_dist_sq.iter().map(|&d| d as f64).sum();
        if total_dist <= 0.0 {
            // All remaining points are identical — pick randomly
            let idx = loop {
                let idx = rand_range(n_rows);
                if !chosen.contains(&idx) {
                    break idx;
                }
            };
            chosen.insert(idx);
            centroids.extend_from_slice(&data[idx * dim..(idx + 1) * dim]);
            continue;
        }

        let threshold = rand_f64() * total_dist;
        let mut cumulative = 0.0_f64;
        let mut next_idx = 0;
        for i in 0..n_rows {
            cumulative += min_dist_sq[i] as f64;
            if cumulative >= threshold && !chosen.contains(&i) {
                next_idx = i;
                break;
            }
        }

        chosen.insert(next_idx);
        centroids.extend_from_slice(&data[next_idx * dim..(next_idx + 1) * dim]);

        // Update min distances
        let new_centroid = &centroids[c * dim..(c + 1) * dim];
        for i in 0..n_rows {
            if chosen.contains(&i) {
                min_dist_sq[i] = 0.0;
                continue;
            }
            let row = &data[i * dim..(i + 1) * dim];
            let dist = row
                .iter()
                .zip(new_centroid)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>();
            if dist < min_dist_sq[i] {
                min_dist_sq[i] = dist;
            }
        }
    }

    centroids
}

// ── K-Means iteration (assignment + update) ────────────────────────────

/// Run one k-means iteration: assign each point to nearest centroid, then
/// recompute centroids as the mean of assigned points.
///
/// Returns (assignments[n_rows], convergence_delta).
fn kmeans_iterate(
    data: &[f32],
    centroids: &mut [f32],
    n_rows: usize,
    dim: usize,
    k: usize,
) -> (Vec<u32>, f64) {
    // ── Assignment step ──────────────────────────────────────────
    let mut assignments: Vec<u32> = vec![0u32; n_rows];

    // Parallel assignment with rayon
    use rayon::prelude::*;
    let violations: Vec<(usize, u32)> = (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let row = &data[i * dim..(i + 1) * dim];
            let mut best_c = 0u32;
            let mut best_dot = f32::NEG_INFINITY;

            for c in 0..k {
                let centroid = &centroids[c * dim..(c + 1) * dim];
                let dot = row.iter().zip(centroid).map(|(a, b)| a * b).sum::<f32>();
                if dot > best_dot {
                    best_dot = dot;
                    best_c = c as u32;
                }
            }
            (i, best_c)
        })
        .collect();

    for (i, best_c) in violations {
        assignments[i] = best_c;
    }

    // ── Update step ──────────────────────────────────────────────
    let old_centroids: Vec<f32> = centroids.to_vec();

    // Zero centroids
    for c in 0..k {
        let slice = &mut centroids[c * dim..(c + 1) * dim];
        slice.fill(0.0_f32);
    }

    let mut counts: Vec<u64> = vec![0u64; k];
    for i in 0..n_rows {
        let c = assignments[i] as usize;
        counts[c] += 1;
        let row = &data[i * dim..(i + 1) * dim];
        let cent_slice = &mut centroids[c * dim..(c + 1) * dim];
        for j in 0..dim {
            cent_slice[j] += row[j];
        }
    }

    // Divide by counts (handle empty clusters — keep at zero)
    for c in 0..k {
        if counts[c] > 0 {
            let inv = 1.0 / counts[c] as f32;
            let slice = &mut centroids[c * dim..(c + 1) * dim];
            for j in 0..dim {
                slice[j] *= inv;
            }
        }
    }

    // ── Convergence delta: sum of L2 distances between old and new centroids ─
    let mut delta = 0.0_f64;
    for c in 0..k {
        let old = &old_centroids[c * dim..(c + 1) * dim];
        let new = &centroids[c * dim..(c + 1) * dim];
        let l2: f64 = old
            .iter()
            .zip(new)
            .map(|(a, b)| ((a - b) as f64) * ((a - b) as f64))
            .sum();
        delta += l2.sqrt();
    }

    (assignments, delta)
}

// ── Reorder by cluster ─────────────────────────────────────────────────

/// Reorder data rows by cluster assignment.
///
/// Returns (reordered_data, cluster_offsets) where cluster_offsets[c] =
/// (start_row_in_reordered, size_of_cluster_c).
fn reorder_by_cluster(
    data: &[f32],
    assignments: &[u32],
    n_rows: usize,
    dim: usize,
    k: usize,
) -> (Vec<f32>, Vec<(usize, usize)>) {
    // Count cluster sizes
    let mut cluster_sizes: Vec<usize> = vec![0usize; k];
    for &a in assignments {
        cluster_sizes[a as usize] += 1;
    }

    // Compute offsets for each cluster in reordered array
    let mut cluster_offsets: Vec<(usize, usize)> = Vec::with_capacity(k);
    let mut offset = 0usize;
    for c in 0..k {
        let size = cluster_sizes[c];
        cluster_offsets.push((offset, size));
        offset += size * dim;
    }

    let total = offset; // = n_rows * dim
    let mut reordered: Vec<f32> = vec![0.0_f32; total];

    // Temporary position trackers per cluster
    let mut write_pos: Vec<usize> = cluster_offsets.iter().map(|&(off, _)| off).collect();

    for i in 0..n_rows {
        let c = assignments[i] as usize;
        let dst_start = write_pos[c];
        let src_start = i * dim;
        reordered[dst_start..dst_start + dim].copy_from_slice(&data[src_start..src_start + dim]);
        write_pos[c] += dim;
    }

    (reordered, cluster_offsets)
}

// ── FP16 NAN/Inf check ──────────────────────────────────────────────────

fn has_nan_or_inf_f16(data: &[u16]) -> bool {
    for &h in data {
        let exp = (h >> 10) & 0x1F;
        let mant = h & 0x3FF;
        if exp == 0x1F && mant != 0 {
            return true; // NaN
        }
        if exp == 0x1F && mant == 0 {
            return true; // Inf
        }
    }
    false
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    let output_dir = &args.output_dir;
    fs::create_dir_all(output_dir).expect("Failed to create output directory");

    let cimage_str = args
        .cimage
        .to_str()
        .expect("cimage path must be valid UTF-8");
    let k = args.k;
    let iters = args.iters;
    let max_rows = args.max_rows;

    println!("── Embedding K-Means Clustering ──────────────────────");
    println!("  cimage:    {}", cimage_str);
    println!("  output:    {}", output_dir.display());
    println!("  k:         {}", k);
    println!("  max_iters: {}", iters);
    println!("  max_rows:  {}", max_rows);

    // ── Step 1: Load ────────────────────────────────────────────────
    let t0 = Instant::now();
    let (embed_f32, vocab_size, hidden_dim) = read_embed_f32(cimage_str, max_rows);
    let n_rows = embed_f32.len() / hidden_dim;
    println!(
        "  Loaded: {} tokens × {} dim ({:.2}M f32 values) in {:.2}s",
        n_rows,
        hidden_dim,
        embed_f32.len() as f64 / 1_000_000.0,
        t0.elapsed().as_secs_f64()
    );

    // ── Step 2: Normalize rows to unit length ──────────────────────
    // K-means with dot-product argmax effectively does spherical k-means.
    // Normalize so dot product = cosine similarity.
    let t1 = Instant::now();
    let mut embed_unit: Vec<f32> = embed_f32.clone();
    for i in 0..n_rows {
        let row = &mut embed_unit[i * hidden_dim..(i + 1) * hidden_dim];
        let norm_sq: f32 = row.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for x in row.iter_mut() {
                *x *= inv;
            }
        }
    }
    println!(
        "  Normalized {} rows in {:.2}s",
        n_rows,
        t1.elapsed().as_secs_f64()
    );

    // ── Step 3: K-Means++ initialization ───────────────────────────
    let t2 = Instant::now();
    let mut centroids = kmeans_plusplus(&embed_unit, k, n_rows, hidden_dim);
    println!(
        "  K-Means++ initialized {} centroids in {:.2}s",
        k,
        t2.elapsed().as_secs_f64()
    );

    // ── Step 4: Iterate ────────────────────────────────────────────
    let mut assignments: Vec<u32> = Vec::new();
    for i in 0..iters {
        let t3 = Instant::now();
        let (iter_assignments, delta) =
            kmeans_iterate(&embed_unit, &mut centroids, n_rows, hidden_dim, k);
        assignments = iter_assignments;

        // Cluster sizes
        let mut sizes: Vec<usize> = vec![0usize; k];
        for &a in &assignments {
            sizes[a as usize] += 1;
        }
        let min_size = sizes.iter().min().copied().unwrap_or(0);
        let max_size = sizes.iter().max().copied().unwrap_or(0);

        println!(
            "  Iteration {}/{}: delta={:.6}, cluster sizes: {}-{} ({:.2}s)",
            i + 1,
            iters,
            delta,
            min_size,
            max_size,
            t3.elapsed().as_secs_f64()
        );

        if delta < 0.01 {
            println!("  ✓ Converged early at iteration {}", i + 1);
            break;
        }
    }

    // ── Step 5: Reorder ────────────────────────────────────────────
    let t4 = Instant::now();
    let (reordered_f32, cluster_offsets) =
        reorder_by_cluster(&embed_f32, &assignments, n_rows, hidden_dim, k);
    println!(
        "  Reordered {} rows by cluster in {:.2}s",
        n_rows,
        t4.elapsed().as_secs_f64()
    );

    // ── Step 6: Write output files ─────────────────────────────────
    // 6a. Centroids (FP16 raw bytes)
    let cent_f16: Vec<u16> = f32_to_half_slice(&centroids);
    let cent_bytes: Vec<u8> = bytemuck::cast_slice(&cent_f16).to_vec();
    let cent_path = output_dir.join("centroids.bin");
    fs::write(&cent_path, &cent_bytes).expect("Failed to write centroids.bin");
    println!("  Wrote centroids.bin ({} bytes)", cent_bytes.len());

    // 6b. Cluster map (u32 LE)
    let mut cluster_map_bytes: Vec<u8> = Vec::with_capacity(n_rows * 4);
    for &cid in &assignments {
        cluster_map_bytes.extend_from_slice(&cid.to_le_bytes());
    }
    let map_path = output_dir.join("cluster_map.bin");
    fs::write(&map_path, &cluster_map_bytes).expect("Failed to write cluster_map.bin");
    println!(
        "  Wrote cluster_map.bin ({} bytes)",
        cluster_map_bytes.len()
    );

    // 6c. Reordered embed (FP16 raw bytes)
    let reord_f16: Vec<u16> = f32_to_half_slice(&reordered_f32);
    let reord_bytes: Vec<u8> = bytemuck::cast_slice(&reord_f16).to_vec();
    let reord_path = output_dir.join("embed_reordered.bin");
    fs::write(&reord_path, &reord_bytes).expect("Failed to write embed_reordered.bin");
    println!("  Wrote embed_reordered.bin ({} bytes)", reord_bytes.len());

    // 6d. Metadata JSON
    let cluster_sizes: Vec<usize> = (0..k).map(|c| cluster_offsets[c].1).collect();
    let cluster_starts: Vec<usize> = cluster_offsets
        .iter()
        .map(|&(start, _)| start / hidden_dim) // convert row offset to token index
        .collect();
    let meta = serde_json::json!({
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "k": k,
        "n_rows_used": n_rows,
        "cluster_offsets": cluster_starts,
        "cluster_sizes": cluster_sizes,
    });
    let meta_path = output_dir.join("embed_reordered.json");
    let meta_str = serde_json::to_string_pretty(&meta).expect("JSON serialize");
    fs::write(&meta_path, &meta_str).expect("Failed to write embed_reordered.json");
    println!("  Wrote embed_reordered.json");

    // ── Step 7: Verify output files ────────────────────────────────
    println!();
    println!("── Verification ────────────────────────────────────────");

    // 7a. Verify centroids shape
    let expected_cent_len = k * hidden_dim; // FP16 halves
    assert_eq!(
        cent_f16.len(),
        expected_cent_len,
        "centroids length mismatch: expected {} halves, got {}",
        expected_cent_len,
        cent_f16.len()
    );
    // Check no NaN/Inf
    assert!(
        !has_nan_or_inf_f16(&cent_f16),
        "centroids contain NaN or Inf"
    );
    println!("  ✓ centroids shape ({k}, {hidden_dim}) — no NaN/Inf");

    // 7b. Verify cluster map length
    assert_eq!(
        cluster_map_bytes.len(),
        n_rows * 4,
        "cluster_map length mismatch"
    );
    // Verify all cids in [0, k)
    for chunk in cluster_map_bytes.chunks_exact(4) {
        let cid = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        assert!(
            (cid as usize) < k,
            "cluster_map contains out-of-range cid {}",
            cid
        );
    }
    println!(
        "  ✓ cluster_map.bin: {} tokens, all cids in [0, {})",
        n_rows, k
    );

    // 7c. Verify reordered embed size
    let expected_reord_len = n_rows * hidden_dim; // FP16 halves
    assert_eq!(
        reord_f16.len(),
        expected_reord_len,
        "embed_reordered length mismatch: expected {} halves, got {}",
        expected_reord_len,
        reord_f16.len()
    );
    let expected_reord_bytes = n_rows * hidden_dim * 2;
    assert_eq!(
        reord_bytes.len(),
        expected_reord_bytes,
        "embed_reordered byte size mismatch"
    );
    println!(
        "  ✓ embed_reordered.bin: {n_rows} × {hidden_dim} × 2 = {} bytes",
        reord_bytes.len()
    );

    // 7d. Check cluster sizes sum to n_rows
    let total_in_clusters: usize = cluster_sizes.iter().sum();
    assert_eq!(
        total_in_clusters, n_rows,
        "cluster sizes don't sum to n_rows"
    );
    println!("  ✓ Cluster sizes sum to {n_rows}");
    println!();
    println!("── Clustering complete ──────────────────────────────────");
}
