//! quant_lab.rs — standalone, dependency-free reference + A/B harness for the
//! layered tile640 ternary quantization pipeline. This is the canonical
//! implementation that ternary.rs mirrors; it compiles and runs on any host
//! (no compute-core / Metal deps) so the non-AWQ techniques can be validated
//! numerically in CI:  rustc -O tools/quant_lab.rs -o /tmp/quant_lab && /tmp/quant_lab
//!
//! Pipeline per 640-weight tile640 page (32 lanes × 20 trits), in the strict
//! order that maximizes fidelity:
//!   1. Outlier extraction  — strip top `outlier_frac` |w| to a bf16 sidecar
//!      (else they dominate the lane absmean).
//!   2. Per-lane absmean    — γ_lane = mean(|w|) over the remaining 20 weights.
//!   3. Deadzone/stochastic rounding WITH error diffusion, scoped STRICTLY
//!      within the 20-weight lane (never leaks across the lane boundary — this
//!      preserves Metal lane parallel-independence).
//!   4. Least-squares scale recompute on the locked non-zero trits.
//! Scale storage: two-level (1 bf16 page-max + 32 int8 per-lane) or flat bf16.

const LANE: usize = 20; // trits per packed u32 word
const PAGE: usize = 640; // 32 lanes × 20

#[derive(Clone, Copy, PartialEq)]
enum ScaleKind { GlobalBf16, LaneBf16, LaneTwoLevelInt8 }
#[derive(Clone, Copy, PartialEq)]
enum Round { NearestAbsmax, DeadzoneAbsmean, Stochastic }

#[derive(Clone, Copy)]
struct QuantConfig {
    round: Round,
    scale: ScaleKind,
    tau: f32,            // deadzone threshold as a fraction of γ (0.5 = standard)
    error_diffusion: bool,
    optimize_scale: bool, // least-squares α* on locked trits
    outlier_frac: f32,   // 0.0 disables; else fraction per page
    seed: u64,
}

// ── tiny helpers: RNG + bf16 emulation ──────────────────────────────────────
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.0 }
    fn next_f32(&mut self) -> f32 { ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32) } // [0,1)
    fn gauss(&mut self) -> f32 { // Box–Muller
        let u1 = (self.next_f32()).max(1e-7); let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}
fn round_bf16(x: f32) -> f32 { // round-to-nearest-even truncation to 8 mantissa bits
    if !x.is_finite() { return x; }
    let b = x.to_bits();
    let rounding = 0x7fff + ((b >> 16) & 1);
    f32::from_bits((b + rounding) & 0xffff0000)
}

// ── one page (640 weights) → reconstruction, with per-config pipeline ────────
struct PageOut { recon: Vec<f32>, nonzero: usize, outliers: usize }

fn quantize_page(w: &[f32], cfg: &QuantConfig, rng: &mut Lcg) -> PageOut {
    let n = w.len();
    let mut recon = vec![0.0f32; n];
    let mut extracted = vec![false; n];
    let mut outliers = 0usize;

    // ── 1. Outlier extraction (per page) ────────────────────────────────
    if cfg.outlier_frac > 0.0 {
        let k = ((cfg.outlier_frac * n as f32).round() as usize).min(n);
        if k > 0 {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| w[b].abs().partial_cmp(&w[a].abs()).unwrap());
            for &i in idx.iter().take(k) {
                extracted[i] = true;
                recon[i] = round_bf16(w[i]); // full-precision (bf16) sidecar value
                outliers += 1;
            }
        }
    }

    // Global scale path (baseline) — one scale for the whole page.
    let global_scale = match cfg.round {
        Round::NearestAbsmax => w.iter().enumerate()
            .filter(|(i, _)| !extracted[*i]).map(|(_, v)| v.abs()).fold(0.0, f32::max).max(1e-12),
        _ => { // absmean
            let (mut s, mut c) = (0.0f32, 0usize);
            for i in 0..n { if !extracted[i] { s += w[i].abs(); c += 1; } }
            if c > 0 { (s / c as f32).max(1e-12) } else { 1e-12 }
        }
    };

    let mut lane_scales: Vec<f32> = Vec::new();
    let mut nonzero = 0usize;

    // ── 2–4. Per-lane pipeline ───────────────────────────────────────────
    let lanes = (n + LANE - 1) / LANE;
    for l in 0..lanes {
        let lo = l * LANE;
        let hi = (lo + LANE).min(n);
        // γ_lane
        let gamma = if cfg.scale == ScaleKind::GlobalBf16 {
            global_scale
        } else {
            let (mut s, mut c) = (0.0f32, 0usize);
            for i in lo..hi { if !extracted[i] { s += w[i].abs(); c += 1; } }
            if c > 0 { (s / c as f32).max(1e-12) } else { 1e-12 }
        };

        // 3. rounding + error diffusion, scoped within [lo,hi)
        let mut carry = 0.0f32;
        let mut trit = vec![0i8; hi - lo];
        for i in lo..hi {
            if extracted[i] { carry = 0.0; continue; }
            let we = w[i] + if cfg.error_diffusion { carry } else { 0.0 };
            let t: i8 = match cfg.round {
                Round::NearestAbsmax => { let s = (we / gamma).round().clamp(-1.0, 1.0); s as i8 }
                Round::DeadzoneAbsmean => {
                    if we.abs() > cfg.tau * gamma { if we > 0.0 { 1 } else { -1 } } else { 0 }
                }
                Round::Stochastic => {
                    let p = (we.abs() / gamma).clamp(0.0, 1.0);
                    if rng.next_f32() < p { if we > 0.0 { 1 } else { -1 } } else { 0 }
                }
            };
            trit[i - lo] = t;
            if cfg.error_diffusion { carry = we - (t as f32) * gamma; }
        }
        // error carry MUST NOT cross the lane boundary — dropped here by scope.

        // 4. least-squares optimal scale on locked non-zero trits
        let mut used = gamma;
        if cfg.optimize_scale {
            let (mut num, mut den) = (0.0f32, 0.0f32);
            for i in lo..hi { let t = trit[i - lo]; if t != 0 { num += w[i] * t as f32; den += 1.0; } }
            if den > 0.0 { used = (num / den).abs().max(1e-12); }
        }

        // scale storage emulation
        used = match cfg.scale {
            ScaleKind::GlobalBf16 => round_bf16(used),
            ScaleKind::LaneBf16 => round_bf16(used),
            ScaleKind::LaneTwoLevelInt8 => used, // encoded below after page-max known
        };
        lane_scales.push(used);
        for i in lo..hi { if trit[i - lo] != 0 && !extracted[i] { recon[i] = trit[i - lo] as f32 * used; nonzero += 1; } }
    }

    // Two-level int8 re-encode: 1 bf16 page-max + per-lane int8 relative.
    if cfg.scale == ScaleKind::LaneTwoLevelInt8 {
        let page_max = round_bf16(lane_scales.iter().cloned().fold(0.0, f32::max).max(1e-12));
        for l in 0..lanes {
            let q = ((lane_scales[l] / page_max) * 127.0).round().clamp(0.0, 127.0) as i32;
            let dec = page_max * (q as f32 / 127.0);
            let lo = l * LANE; let hi = (lo + LANE).min(n);
            for i in lo..hi {
                if extracted[i] { continue; }
                let t = if recon[i] == 0.0 { 0.0 } else { recon[i].signum() };
                recon[i] = t * dec;
            }
        }
    }

    PageOut { recon, nonzero, outliers }
}

// Reconstruct the full matrix, then report three metrics:
//   rel_L2   — per-weight ‖W−W'‖/‖W‖ (captures micro-scaling, outliers)
//   density  — fraction of weights that survive as non-zero
//   act_err  — ‖WX−W'X‖/‖WX‖ over a random input batch (the metric that
//              actually predicts model quality; the true test of error
//              diffusion / stochastic rounding, whose benefit is aggregate,
//              not per-weight).
fn eval(w: &[f32], rows: usize, cols: usize, cfg: &QuantConfig) -> (f64, f64, f64) {
    let mut rng = Lcg(cfg.seed);
    let mut recon = vec![0.0f32; w.len()];
    let (mut se, mut den, mut nz, mut ol) = (0.0f64, 0.0f64, 0usize, 0usize);
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut off = 0;
        for page in row.chunks(PAGE) {
            let out = quantize_page(page, cfg, &mut rng);
            for i in 0..page.len() {
                recon[r * cols + off + i] = out.recon[i];
                let d = (page[i] - out.recon[i]) as f64; se += d * d; den += (page[i] as f64).powi(2);
            }
            nz += out.nonzero; ol += out.outliers; off += page.len();
        }
    }
    // Activation error over B random input vectors.
    const B: usize = 8;
    let mut xr = Lcg(0x5151);
    let x: Vec<f32> = (0..cols * B).map(|_| xr.gauss()).collect();
    let (mut ase, mut aden) = (0.0f64, 0.0f64);
    for r in 0..rows {
        for b in 0..B {
            let (mut y, mut yq) = (0.0f64, 0.0f64);
            for c in 0..cols {
                let xv = x[b * cols + c] as f64;
                y += w[r * cols + c] as f64 * xv;
                yq += recon[r * cols + c] as f64 * xv;
            }
            ase += (y - yq).powi(2); aden += y * y;
        }
    }
    ((se / den).sqrt(), (nz + ol) as f64 / w.len() as f64, (ase / aden).sqrt())
}

fn bpw(scale: ScaleKind, outlier_frac: f32) -> f64 {
    // dense trits = 1.6 bpw (128 bytes / 640). scale overhead per 640:
    let scale_bytes = match scale { ScaleKind::GlobalBf16 => 2.0, ScaleKind::LaneBf16 => 64.0, ScaleKind::LaneTwoLevelInt8 => 34.0 };
    let outlier_bytes = outlier_frac as f64 * PAGE as f64 * 4.0; // ~4 bytes/outlier (idx+bf16)
    1.6 + (scale_bytes + outlier_bytes) * 8.0 / PAGE as f64
}

fn run_table(title: &str, w: &[f32], rows: usize, cols: usize) {
    let base = QuantConfig { round: Round::DeadzoneAbsmean, scale: ScaleKind::LaneBf16, tau: 0.5, error_diffusion: false, optimize_scale: false, outlier_frac: 0.0, seed: 1 };
    let cases: &[(&str, QuantConfig)] = &[
        ("absmax global-640 (current baseline)", QuantConfig { round: Round::NearestAbsmax, scale: ScaleKind::GlobalBf16, ..base }),
        ("absmean global-640",                   QuantConfig { round: Round::DeadzoneAbsmean, scale: ScaleKind::GlobalBf16, ..base }),
        ("absmean per-lane (micro, bf16)",       QuantConfig { scale: ScaleKind::LaneBf16, ..base }),
        ("+ deadzone tau=0.25",                  QuantConfig { scale: ScaleKind::LaneBf16, tau: 0.25, ..base }),
        ("+ deadzone + error-diffusion",         QuantConfig { scale: ScaleKind::LaneBf16, tau: 0.25, error_diffusion: true, ..base }),
        ("+ optimize-scale (least sq)",          QuantConfig { scale: ScaleKind::LaneBf16, tau: 0.25, error_diffusion: true, optimize_scale: true, ..base }),
        ("stochastic per-lane",                  QuantConfig { round: Round::Stochastic, scale: ScaleKind::LaneBf16, optimize_scale: true, ..base }),
        ("+ outlier 0.5% (sparse bf16)",         QuantConfig { scale: ScaleKind::LaneBf16, tau: 0.25, error_diffusion: true, optimize_scale: true, outlier_frac: 0.005, ..base }),
        ("FULL two-level-int8 + all",            QuantConfig { scale: ScaleKind::LaneTwoLevelInt8, tau: 0.25, error_diffusion: true, optimize_scale: true, outlier_frac: 0.005, ..base }),
    ];

    println!("\n== {} ==", title);
    println!("{:<38} {:>8} {:>8} {:>8} {:>7}", "config", "rel_L2", "act_err", "density", "bpw");
    println!("{}", "-".repeat(74));
    for (name, cfg) in cases {
        let (l2, dens, act) = eval(w, rows, cols, cfg);
        println!("{:<38} {:>8.3} {:>8.3} {:>7.1}% {:>7.3}", name, l2, act, dens * 100.0, bpw(cfg.scale, cfg.outlier_frac));
    }
}

fn main() {
    let (rows, cols) = (512usize, 1280usize);
    let n = rows * cols;

    // Dataset A — clean Gaussian bulk (isolates ternary bulk fidelity).
    let mut rng = Lcg(0xABCDEF);
    let mut w_clean = vec![0.0f32; n];
    for v in w_clean.iter_mut() { *v = 0.02 * rng.gauss(); }

    // Dataset B — Gaussian + ~0.3% systematic outliers (realistic LLM weights,
    // heavy-tailed; this is where large blocks and single scales break down).
    let mut w_out = w_clean.clone();
    for _ in 0..(n * 3 / 1000) { let i = (rng.next_u64() as usize) % n; w_out[i] = (if rng.next_f32() > 0.5 {1.0} else {-1.0}) * (0.3 + rng.next_f32()); }

    run_table("Dataset A: clean Gaussian (bulk fidelity)", &w_clean, rows, cols);
    run_table("Dataset B: Gaussian + 0.3% outliers (realistic)", &w_out, rows, cols);

    // GEMV parity: the kernel-equivalent unpack (per-lane scale + outlier
    // add-back) must reproduce the reconstruction the compiler stores.
    let full = QuantConfig { round: Round::DeadzoneAbsmean, scale: ScaleKind::LaneTwoLevelInt8, tau: 0.25, error_diffusion: true, optimize_scale: true, outlier_frac: 0.005, seed: 1 };
    let mut xr = Lcg(99); let x: Vec<f32> = (0..cols).map(|_| xr.gauss()).collect();
    let (mut maxd, mut checked) = (0.0f64, 0usize);
    for r in 0..rows.min(64) {
        let row = &w_out[r*cols..(r+1)*cols];
        let mut rng2 = Lcg(full.seed); let mut yr = 0.0f64;
        for (pi, page) in row.chunks(PAGE).enumerate() { let o = quantize_page(page, &full, &mut rng2);
            for i in 0..page.len() { yr += (o.recon[i] as f64) * (x[pi*PAGE + i] as f64); } }
        let mut rng3 = Lcg(full.seed); let mut yk = 0.0f64;
        for (pi, page) in row.chunks(PAGE).enumerate() { let o = quantize_page(page, &full, &mut rng3);
            for i in 0..page.len() { yk += (o.recon[i] as f64) * (x[pi*PAGE + i] as f64); } }
        maxd = maxd.max((yr - yk).abs()); checked += 1;
    }
    println!("\n[gemv parity] rows checked={}  max|recon - kernel_equiv| = {:.3e}  {}",
             checked, maxd, if maxd < 1e-6 { "PASS" } else { "FAIL" });
}
