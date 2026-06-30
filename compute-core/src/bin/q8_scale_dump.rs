//! Investigate Q8_0 block scale distribution to find the f16 overflow.
//! Usage: cargo run --features prism-backend --bin q8-scale-dump -- <gguf> [tensor_name_filter]

use std::path::PathBuf;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(&args[1]);
    let filter = args.get(2).map(|s| s.as_str());

    let (metadata, tensors) = tribunus_compute_core::gguf::parse_gguf_header(&path)?;
    let arch = tribunus_compute_core::gguf::extract_architecture(&metadata)?;
    eprintln!("Model: {} ({} layers, hidden={})", arch.model_type, arch.num_hidden_layers, arch.hidden_size);

    // Open mmap for raw block reads
    let f = std::fs::File::open(&path).map_err(|e| format!("open: {e}"))?;
    let mmap = unsafe { memmap2::Mmap::map(&f).map_err(|e| format!("mmap: {e}"))? };

    // Process each 2D weight tensor
    for t in &tensors {
        if t.shape.len() != 2 || !t.name.ends_with(".weight") { continue; }
        if let Some(filt) = filter { if !t.name.contains(filt) { continue; } }

        let rows = t.shape[0] as usize;
        let cols = t.shape[1] as usize;
        let n_blocks = t.byte_size as usize / 34; // Q8_0 block size

        // Scan all Q8_0 blocks, collect scale statistics
        let mut scales: Vec<f32> = Vec::with_capacity(n_blocks);
        let mut inf_count = 0u64;
        let mut nan_count = 0u64;
        let mut overflow_count = 0u64; // scale > 60000
        let mut max_scale = 0.0f32;
        let mut min_scale = f32::MAX;
        let mut sum_scale = 0.0f64;

        let start = t.byte_offset as usize;
        let end = start + t.byte_size as usize;
        let data = &mmap[start..end];

        for b in 0..n_blocks {
            let off = b * 34;
            if off + 34 > data.len() { break; }
            let bits = u16::from_le_bytes([data[off], data[off + 1]]);
            let scale = half::f16::from_bits(bits).to_f32();

            if scale.is_nan() { nan_count += 1; continue; }
            if !scale.is_finite() { inf_count += 1; continue; }

            let abs_s = scale.abs();
            if abs_s > max_scale { max_scale = abs_s; }
            if abs_s < min_scale { min_scale = abs_s; }
            sum_scale += abs_s as f64;
            if abs_s > 60000.0 { overflow_count += 1; }
            scales.push(abs_s);
        }

        if scales.len() == 0 { continue; }
        scales.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = sum_scale / scales.len() as f64;
        let median = scales[scales.len() / 2];
        let p95 = scales[(scales.len() as f64 * 0.95) as usize];
        let p99 = scales[(scales.len() as f64 * 0.99) as usize];
        let p999 = scales[(scales.len() as f64 * 0.999) as usize];
        let max_s = scales.last().unwrap();
        let total = n_blocks as f64;
        let p_inf = inf_count as f64 / total * 100.0;
        let p_overflow = overflow_count as f64 / total * 100.0;

        println!("\n{} [{}×{}]", t.name, rows, cols);
        println!("  blocks: {} | scale: median={:.1} mean={:.1} p95={:.1} p99={:.1} p999={:.1} max={:.1}",
            n_blocks, median, mean, p95, p99, p999, max_s);
        println!("  inf: {:.2}% | >60K: {:.2}% | nan: {}", p_inf, p_overflow, nan_count);
        if *max_s > 60000.0 {
            let exceeded_by = *max_s - 65504.0;
            println!("  ** OVERFLOW: max scale {} exceeds f16 max 65504 by {:.0}", max_s, exceeded_by);
        }
        if *max_s > 65504.0 {
            println!("  ** CONFIRMED: scales exceed f16 max — BF16→f16 overflow during conversion");
        }
    }
    Ok(())
}
