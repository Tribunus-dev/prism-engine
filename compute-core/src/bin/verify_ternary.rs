//! Verify accuracy of Q8_0 → ternary tile640 quantization.
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(&args[1]);
    let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    let (metadata, tensors) = tribunus_compute_core::gguf::parse_gguf_header(&path)?;
    let arch = tribunus_compute_core::gguf::extract_architecture(&metadata)?;
    eprintln!("Model: {} ({} layers, hidden={})", arch.model_type, arch.num_hidden_layers, arch.hidden_size);

    let tens: Vec<_> = tensors.iter()
        .filter(|t| t.shape.len() == 2 && (t.name.contains("attn_") || t.name.contains("ffn_")) && t.name.ends_with(".weight"))
        .take(limit)
        .collect();

    for t in &tens {
        let rows = t.shape[0] as usize;
        let cols = t.shape[1] as usize;
        let n = rows.saturating_mul(cols);
        let orig = tribunus_compute_core::gguf::read_gguf_tensor_f32(&path, t)?;

        // Count finite values
        let finite_n = orig.iter().filter(|v| v.is_finite()).count();
        if finite_n < n / 2 {
            eprintln!("  {} [{}×{}] ** {:.1}% inf/NaN — skip", t.name, rows, cols,
                100.0 * (n - finite_n) as f64 / n as f64);
            continue;
        }

        // Ternary tile640 quantize → dequantize on finite values only
        let num_tiles = (cols + 639) / 640;
        let mut recon = vec![0.0f32; n];
        for tile in 0..num_tiles {
            let ts = tile * 640;
            let te = (ts + 640).min(cols);
            for row in 0..rows {
                let base = row * cols;
                // Only compute scale from finite values in this tile segment
                let mut absmax = 0.0f32;
                for c in ts..te {
                    let v = orig[base + c];
                    if v.is_finite() { let a = v.abs(); if a > absmax { absmax = a; } }
                }
                let scale = if absmax > 1e-12 { absmax } else { 1.0 };
                let inv = 1.0 / scale;
                for c in ts..te {
                    if orig[base + c].is_finite() {
                        let val = orig[base + c] * inv;
                        let d = if val > 0.5 { 1.0 } else if val < -0.5 { -1.0 } else { 0.0 };
                        recon[base + c] = d * scale;
                    }
                }
            }
        }

        // Sample-based accuracy metrics on finite values
        let sample = n.min(200_000);
        let step = if n > sample { n / sample } else { 1 };
        let mut sq = 0.0f64;
        let mut ae = 0.0f64;
        let mut max_ae = 0.0f32;
        let mut max_re = 0.0f32;
        let mut checked = 0usize;
        for i in (0..n).step_by(step) {
            let o = orig[i];
            if !o.is_finite() { continue; }
            let r = recon[i];
            let err = (o as f64) - (r as f64);
            let ab = err.abs() as f32;
            sq += (err * err) as f64;
            ae += ab as f64;
            if ab > max_ae { max_ae = ab; }
            let denom = o.abs().max(1e-10);
            let re = ab / denom;
            if re > max_re { max_re = re; }
            checked += 1;
        }

        let rmse = (sq / checked as f64).sqrt();
        let mae = ae / checked as f64;
        eprintln!("  {} [{}×{}]", t.name, rows, cols);
        eprintln!("    finite: {}/{} ({:.2}%)", finite_n, n, 100.0 * finite_n as f64 / n as f64);
        eprintln!("    RMSE: {:.6e}  MAE: {:.6e}  MaxAbsErr: {:.6e}", rmse, mae, max_ae);
        eprintln!("    MaxRelErr: {:.6}", max_re);
        eprintln!();
    }
    Ok(())
}
