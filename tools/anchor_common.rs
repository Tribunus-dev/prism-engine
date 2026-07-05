// anchor_common.rs — shared numerical core for the bf16 anchor tools.
// Included (via `include!`) by tools/bf16_anchor_ref.rs (the contract test
// battery) and tools/probe_checkpoint.rs (the convention-flag discovery
// sweep). No `main` here; pure definitions. Std-only.

// ── bf16 <-> f32 ───────────────────────────────────────────────────────────

#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Round-to-nearest-even f32 → bf16 (test-data packing; NaN not handled).
#[inline]
fn f32_to_bf16(v: f32) -> u16 {
    let x = v.to_bits();
    let round = ((x >> 16) & 1) + 0x7FFF;
    ((x.wrapping_add(round)) >> 16) as u16
}

// ── GEMM: out[M,N] = act[M,K] · Wᵀ, W stored [N,K] row-major bf16 ─────────
// (out_features-major rows — the safetensors / repo-kernel convention.)

/// Engine GEMM: f32 accumulate (matches the Mac cblas_sgemm path).
fn gemm_wt(act: &[f32], m: usize, k: usize, w: &[u16], n: usize, out: &mut [f32]) {
    assert_eq!(act.len(), m * k);
    assert_eq!(w.len(), n * k);
    assert_eq!(out.len(), m * n);
    for row in 0..m {
        let a = &act[row * k..(row + 1) * k];
        for col in 0..n {
            let wr = &w[col * k..(col + 1) * k];
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += a[i] * bf16_to_f32(wr[i]);
            }
            out[row * n + col] = acc;
        }
    }
}

/// f64 oracle for the engine GEMM.
fn gemm_wt_oracle(act: &[f32], m: usize, k: usize, w: &[u16], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for i in 0..k {
                acc += act[row * k + i] as f64 * bf16_to_f32(w[col * k + i]) as f64;
            }
            out[row * n + col] = acc;
        }
    }
    out
}

/// Deterministic parallel GEMM: threads own disjoint column ranges; each
/// element is computed by the identical sequential k-loop, so the result is
/// bitwise-equal to `gemm_wt` (asserted in main).
fn gemm_wt_parallel(
    act: &[f32],
    m: usize,
    k: usize,
    w: &[u16],
    n: usize,
    out: &mut [f32],
    threads: usize,
) {
    assert_eq!(out.len(), m * n);
    let t = threads.clamp(1, n.max(1));
    let chunk = n.div_ceil(t);
    // Split output into per-thread column strips via raw pointer + disjoint
    // index math (row-major [M,N] makes column strips strided, so we can't
    // use split_at_mut directly).
    struct SendPtr(*mut f32);
    unsafe impl Send for SendPtr {}
    // SAFETY: threads write disjoint (row, col) index sets — see below.
    unsafe impl Sync for SendPtr {}
    let base = SendPtr(out.as_mut_ptr());
    let base_ref = &base;
    std::thread::scope(|s| {
        for tid in 0..t {
            let c0 = tid * chunk;
            if c0 >= n {
                break;
            }
            let c1 = (c0 + chunk).min(n);
            s.spawn(move || {
                let ptr = base_ref.0;
                for row in 0..m {
                    let a = &act[row * k..(row + 1) * k];
                    for col in c0..c1 {
                        let wr = &w[col * k..(col + 1) * k];
                        let mut acc = 0.0f32;
                        for i in 0..k {
                            acc += a[i] * bf16_to_f32(wr[i]);
                        }
                        // SAFETY: (row, col) index sets are disjoint across
                        // threads (disjoint col ranges), all within m*n.
                        unsafe { *ptr.add(row * n + col) = acc };
                    }
                }
            });
        }
    });
}

// ── RMSNorm with convention flag ───────────────────────────────────────────

/// y_i = x_i · rsqrt(mean(x²) + eps) · (γ_i + gamma_delta).
/// gamma_delta = 0.0 → plain-γ (what the megakernel's fast_rmsnorm applies to
/// its `norms` buffer); 1.0 → Gemma's (1+γ) over raw checkpoint weights.
fn rmsnorm(x: &[f32], gamma: &[u16], eps: f32, gamma_delta: f32) -> Vec<f32> {
    assert_eq!(x.len(), gamma.len());
    let mut ss = 0.0f64;
    for &v in x {
        ss += (v as f64) * (v as f64);
    }
    let rcp = 1.0 / ((ss / x.len() as f64) + eps as f64).sqrt();
    x.iter()
        .zip(gamma)
        .map(|(&v, &g)| ((v as f64 * rcp) as f32) * (bf16_to_f32(g) + gamma_delta))
        .collect()
}

// ── Partial RoPE with convention flag ──────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
enum RopeConv {
    /// Adjacent pairs (x[2i], x[2i+1]) — what gemma4_full.metal::apply_rope does.
    Interleaved,
    /// Split halves (x[i], x[i + rope_dim/2]) — the HF/transformers convention.
    HalfSplit,
}

/// Rotate the first `rope_dim` dims of one head vector in place; dims ≥
/// rope_dim untouched. freq_i = θ^(−2i/rope_dim).
fn rope_apply(head: &mut [f32], pos: u32, rope_dim: usize, theta: f32, conv: RopeConv) {
    assert!(rope_dim <= head.len() && rope_dim % 2 == 0);
    let half = rope_dim / 2;
    for i in 0..half {
        let freq = 1.0f64 / (theta as f64).powf(2.0 * i as f64 / rope_dim as f64);
        let ang = pos as f64 * freq;
        let (sin, cos) = ang.sin_cos();
        let (ia, ib) = match conv {
            RopeConv::Interleaved => (2 * i, 2 * i + 1),
            RopeConv::HalfSplit => (i, i + half),
        };
        let (x0, x1) = (head[ia] as f64, head[ib] as f64);
        head[ia] = (x0 * cos - x1 * sin) as f32;
        head[ib] = (x0 * sin + x1 * cos) as f32;
    }
}

/// Apply RoPE to every head of a [T, n_heads·head_dim] activation buffer.
fn rope_heads(
    buf: &mut [f32],
    t: usize,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
    conv: RopeConv,
    pos0: u32,
) {
    for row in 0..t {
        for h in 0..n_heads {
            let off = row * n_heads * head_dim + h * head_dim;
            rope_apply(&mut buf[off..off + head_dim], pos0 + row as u32, rope_dim, theta, conv);
        }
    }
}

// ── SwiGLU ─────────────────────────────────────────────────────────────────

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up).map(|(&g, &u)| silu(g) * u).collect()
}

// ── Causal GQA SDPA (full-sequence, prefill-style — no KV cache needed) ────

/// q: [T, nq·hd], k/v: [T, nkv·hd]. Query head h reads kv head h/(nq/nkv).
/// `scale`: None → raw dots (megakernel behavior); Some(s) → s·QK (true-model
/// 1/√d adjudication flag). f64 softmax reductions.
fn sdpa_causal_gqa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    nq: usize,
    nkv: usize,
    hd: usize,
    scale: Option<f32>,
) -> Vec<f32> {
    assert!(nq % nkv == 0);
    let group = nq / nkv;
    let s = scale.unwrap_or(1.0) as f64;
    let mut out = vec![0.0f32; t * nq * hd];
    for qi in 0..t {
        for h in 0..nq {
            let kvh = h / group;
            let qv = &q[qi * nq * hd + h * hd..][..hd];
            // scores over keys 0..=qi (causal)
            let mut scores = Vec::with_capacity(qi + 1);
            let mut maxs = f64::MIN;
            for ki in 0..=qi {
                let kv = &k[ki * nkv * hd + kvh * hd..][..hd];
                let mut dot = 0.0f64;
                for d in 0..hd {
                    dot += qv[d] as f64 * kv[d] as f64;
                }
                dot *= s;
                if dot > maxs {
                    maxs = dot;
                }
                scores.push(dot);
            }
            let mut denom = 0.0f64;
            for sc in scores.iter_mut() {
                *sc = (*sc - maxs).exp();
                denom += *sc;
            }
            let o = &mut out[qi * nq * hd + h * hd..][..hd];
            for (ki, &w) in scores.iter().enumerate() {
                let p = w / denom;
                let vv = &v[ki * nkv * hd + kvh * hd..][..hd];
                for d in 0..hd {
                    o[d] += (p * vv[d] as f64) as f32;
                }
            }
        }
    }
    out
}

// ── One Gemma-4 block ──────────────────────────────────────────────────────

#[derive(Clone)]
struct LayerCfg {
    hidden: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
    ffn: usize,
    eps: f32,
    /// 0.0 = plain-γ (megakernel), 1.0 = (1+γ) (stock Gemma).
    gamma_delta: f32,
    rope_conv: RopeConv,
    /// None = raw QK dots (megakernel), Some(1/√d) = stock attention.
    attn_scale: Option<f32>,
    /// true = pre-FFN norm reuses the pre-attn γ (megakernel behavior flag).
    share_norm_weights: bool,
}

struct LayerWeights {
    norm1: Vec<u16>,          // [hidden]
    norm2: Vec<u16>,          // [hidden] (ignored when share_norm_weights)
    wq: Vec<u16>,             // [nq·hd, hidden]
    wk: Vec<u16>,             // [nkv·hd, hidden]
    wv: Vec<u16>,             // [nkv·hd, hidden]
    wo: Vec<u16>,             // [hidden, nq·hd]
    w_gate: Vec<u16>,         // [ffn, hidden]
    w_up: Vec<u16>,           // [ffn, hidden]
    w_down: Vec<u16>,         // [hidden, ffn]
}

/// H_in [T, hidden] → H_out [T, hidden]. The anchor contract.
fn forward_layer(cfg: &LayerCfg, w: &LayerWeights, h_in: &[f32], t: usize, pos0: u32) -> Vec<f32> {
    let d = cfg.hidden;
    assert_eq!(h_in.len(), t * d);
    let qd = cfg.nq * cfg.head_dim;
    let kvd = cfg.nkv * cfg.head_dim;

    // 1. pre-attention norm (per row)
    let mut normed = vec![0.0f32; t * d];
    for r in 0..t {
        let y = rmsnorm(&h_in[r * d..(r + 1) * d], &w.norm1, cfg.eps, cfg.gamma_delta);
        normed[r * d..(r + 1) * d].copy_from_slice(&y);
    }

    // 2. QKV projections
    let mut q = vec![0.0f32; t * qd];
    let mut k = vec![0.0f32; t * kvd];
    let mut v = vec![0.0f32; t * kvd];
    gemm_wt(&normed, t, d, &w.wq, qd, &mut q);
    gemm_wt(&normed, t, d, &w.wk, kvd, &mut k);
    gemm_wt(&normed, t, d, &w.wv, kvd, &mut v);

    // 3. RoPE on Q and K
    rope_heads(&mut q, t, cfg.nq, cfg.head_dim, cfg.rope_dim, cfg.theta, cfg.rope_conv, pos0);
    rope_heads(&mut k, t, cfg.nkv, cfg.head_dim, cfg.rope_dim, cfg.theta, cfg.rope_conv, pos0);

    // 4. causal GQA attention + O projection + residual
    let attn = sdpa_causal_gqa(&q, &k, &v, t, cfg.nq, cfg.nkv, cfg.head_dim, cfg.attn_scale);
    let mut o = vec![0.0f32; t * d];
    gemm_wt(&attn, t, qd, &w.wo, d, &mut o);
    let mut h_mid = vec![0.0f32; t * d];
    for i in 0..t * d {
        h_mid[i] = h_in[i] + o[i];
    }

    // 5. pre-FFN norm (shared or own weights — adjudication flag)
    let n2 = if cfg.share_norm_weights { &w.norm1 } else { &w.norm2 };
    let mut normed2 = vec![0.0f32; t * d];
    for r in 0..t {
        let y = rmsnorm(&h_mid[r * d..(r + 1) * d], n2, cfg.eps, cfg.gamma_delta);
        normed2[r * d..(r + 1) * d].copy_from_slice(&y);
    }

    // 6. SwiGLU FFN + residual
    let mut gate = vec![0.0f32; t * cfg.ffn];
    let mut up = vec![0.0f32; t * cfg.ffn];
    gemm_wt(&normed2, t, d, &w.w_gate, cfg.ffn, &mut gate);
    gemm_wt(&normed2, t, d, &w.w_up, cfg.ffn, &mut up);
    let act = swiglu(&gate, &up);
    let mut down = vec![0.0f32; t * d];
    gemm_wt(&act, t, cfg.ffn, &w.w_down, d, &mut down);
    let mut h_out = vec![0.0f32; t * d];
    for i in 0..t * d {
        h_out[i] = h_mid[i] + down[i];
    }
    h_out
}

// ── test utilities ─────────────────────────────────────────────────────────

fn rng(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*s >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
}
fn rand_bf16(n: usize, s: &mut u64, amp: f32) -> Vec<u16> {
    (0..n).map(|_| f32_to_bf16(amp * rng(s))).collect()
}
fn rand_f32(n: usize, s: &mut u64) -> Vec<f32> {
    (0..n).map(|_| rng(s)).collect()
}
fn rel_err_f64(a: &[f32], b: &[f64]) -> f64 {
    let (mut se, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        se += (*x as f64 - y).powi(2);
        den += y.powi(2);
    }
    (se / den.max(1e-30)).sqrt()
}

