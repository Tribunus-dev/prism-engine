//! bf16_anchor_ref.rs — std-only numerical contract for the bf16 anchor forward.
//!
//! The activation-anchored PTQ pipeline (kernels/JOINT_MTP_COMPILE.md) needs a
//! golden reference: run one unquantized Gemma-4 block on the student's
//! incoming activations and yield H_bf16_out. This file IS that contract —
//! every op the anchor runner must implement, in dependency-free Rust that
//! compiles with bare `rustc` and self-verifies on any host (Linux CI
//! included). The Mac production pass binds the same math to the EXISTING
//! Accelerate lane (`backend/accelerate/ops.rs`: cblas_sgemm + vDSP) — this
//! reference is what that binding is parity-tested against.
//!
//! Deliberate design points (each encoded as a test below):
//!   • NO 640-alignment assumption. The model's own projections are not
//!     Tile640 multiples: Q out = 4096 (6.4×640), K/V out = 2048 (3.2×640),
//!     vocab = 262144 (409.6×640). Only hidden (3840) and FFN (15360) are.
//!     GEMM shapes here are arbitrary; per-tile act_err taps are an OUTPUT
//!     slicing concern, not a blocking constraint.
//!   • Convention FLAGS, not hardcoded megakernel quirks. The anchor's job is
//!     to represent the TRUE checkpoint semantics; where the megakernel
//!     disagrees (plain-γ vs (1+γ) RMSNorm, interleaved vs half-split RoPE,
//!     missing 1/√d) the anchor is the instrument that adjudicates. The fold
//!     identity  plain(1+γ) ≡ gemma(γ)  is asserted so the two conventions'
//!     relationship is machine-checked.
//!   • Deterministic parallelism: std::thread::scope over disjoint output
//!     columns, bitwise-equal to the serial path (asserted).
//!   • f32 accumulation in the engine GEMM (matches cblas_sgemm on the Mac
//!     path); f64 accumulation only in oracles and softmax/norm reductions.
//!
//! Build & run:  rustc -O tools/bf16_anchor_ref.rs -o /tmp/bf16ref && /tmp/bf16ref

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

fn main() {
    let mut s = 0xA11C_E5ED_u64 as u64;

    // (1) GEMM vs f64 oracle — shapes deliberately NOT multiples of 640/8,
    // covering the model's own misalignments in miniature (4096→? use 100,
    // 2048→? use 52, vocab→? use 251).
    for &(m, k, n) in &[(3usize, 96usize, 160usize), (2, 64, 100), (4, 37, 52), (1, 128, 251)] {
        let a = rand_f32(m * k, &mut s);
        let w = rand_bf16(n * k, &mut s, 0.05);
        let mut c = vec![0.0f32; m * n];
        gemm_wt(&a, m, k, &w, n, &mut c);
        let oracle = gemm_wt_oracle(&a, m, k, &w, n);
        let e = rel_err_f64(&c, &oracle);
        assert!(e < 1e-5, "gemm rel err {:e} at ({},{},{})", e, m, k, n);
        // deterministic parallel == serial, bitwise
        let mut cp = vec![0.0f32; m * n];
        gemm_wt_parallel(&a, m, k, &w, n, &mut cp, 4);
        assert!(c.iter().zip(&cp).all(|(x, y)| x.to_bits() == y.to_bits()),
            "parallel GEMM not bitwise-equal at ({},{},{})", m, k, n);
    }
    println!("[gemm] engine vs f64 oracle + bitwise parallel parity  PASS");

    // (2) bf16 roundtrip: RNE error bound, exact for 8-bit-mantissa values.
    assert_eq!(bf16_to_f32(f32_to_bf16(1.5)), 1.5);
    assert_eq!(bf16_to_f32(f32_to_bf16(-0.375)), -0.375);
    for _ in 0..1000 {
        let x = 10.0 * rng(&mut s);
        let r = bf16_to_f32(f32_to_bf16(x));
        assert!((r - x).abs() <= x.abs() * 0.00393 + 1e-30, "bf16 rt {} -> {}", x, r);
    }
    println!("[bf16] round-to-nearest-even roundtrip bounds          PASS");

    // (3) Norm-convention fold identity: gemma(γ) == plain(1+γ) — the exact
    // relationship the cimage ingest must implement if the megakernel's
    // plain-γ kernel is to be numerically correct.
    {
        let x = rand_f32(64, &mut s);
        let gamma = rand_bf16(64, &mut s, 0.5);
        let folded: Vec<u16> = gamma.iter().map(|&g| f32_to_bf16(bf16_to_f32(g) + 1.0)).collect();
        let y_gemma = rmsnorm(&x, &gamma, 1e-6, 1.0);
        let y_plain = rmsnorm(&x, &folded, 1e-6, 0.0);
        let mut maxd = 0.0f32;
        for (a, b) in y_gemma.iter().zip(&y_plain) {
            maxd = maxd.max((a - b).abs());
        }
        // folded γ re-quantizes through bf16, so exactness is bf16-bounded
        assert!(maxd < 6e-3, "fold identity max dev {}", maxd);
        // and the two conventions genuinely differ without folding
        let y_wrong = rmsnorm(&x, &gamma, 1e-6, 0.0);
        assert!(y_gemma.iter().zip(&y_wrong).any(|(a, b)| (a - b).abs() > 1e-2));
    }
    println!("[rmsnorm] (1+γ) ≡ plain(fold) identity + divergence     PASS");

    // (4) RoPE: pos-0 identity, norm preservation, tail untouched,
    // conventions differ.
    {
        let head0: Vec<f32> = rand_f32(16, &mut s);
        let mut h = head0.clone();
        rope_apply(&mut h, 0, 8, 1e6, RopeConv::Interleaved);
        assert!(h.iter().zip(&head0).all(|(a, b)| (a - b).abs() < 1e-7), "pos0 not identity");
        let mut h1 = head0.clone();
        rope_apply(&mut h1, 12345, 8, 1e6, RopeConv::Interleaved);
        let n0: f64 = head0[..8].iter().map(|&x| (x as f64).powi(2)).sum();
        let n1: f64 = h1[..8].iter().map(|&x| (x as f64).powi(2)).sum();
        assert!((n0.sqrt() - n1.sqrt()).abs() < 1e-5, "rotation norm drift");
        assert!(h1[8..] == head0[8..], "tail dims must be untouched");
        let mut h2 = head0.clone();
        rope_apply(&mut h2, 12345, 8, 1e6, RopeConv::HalfSplit);
        assert!(h1.iter().zip(&h2).any(|(a, b)| (a - b).abs() > 1e-4),
            "conventions should differ on a generic vector");
    }
    println!("[rope] identity/norm/tail invariants + convention split PASS");

    // (5) SwiGLU hand value: silu(1)·2 ≈ 1.4621172.
    {
        let y = swiglu(&[1.0], &[2.0]);
        assert!((y[0] - 1.462_117_2).abs() < 1e-5, "swiglu {:?}", y);
    }
    println!("[swiglu] hand-computed value                            PASS");

    // (6) SDPA: T=1 output == V row per GQA group; scale flag changes scores.
    {
        let (nq, nkv, hd) = (4usize, 2usize, 8usize);
        let q = rand_f32(nq * hd, &mut s);
        let k = rand_f32(nkv * hd, &mut s);
        let v = rand_f32(nkv * hd, &mut s);
        let o = sdpa_causal_gqa(&q, &k, &v, 1, nq, nkv, hd, None);
        for h in 0..nq {
            let kvh = h / (nq / nkv);
            for d in 0..hd {
                assert!((o[h * hd + d] - v[kvh * hd + d]).abs() < 1e-6,
                    "T=1 softmax must return the group's V row");
            }
        }
        // multi-position: scaled vs unscaled genuinely differ
        let t = 5;
        let q5 = rand_f32(t * nq * hd, &mut s);
        let k5 = rand_f32(t * nkv * hd, &mut s);
        let v5 = rand_f32(t * nkv * hd, &mut s);
        let a = sdpa_causal_gqa(&q5, &k5, &v5, t, nq, nkv, hd, None);
        let b = sdpa_causal_gqa(&q5, &k5, &v5, t, nq, nkv, hd, Some(1.0 / (hd as f32).sqrt()));
        assert!(a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-4));
    }
    println!("[sdpa] GQA group routing + causal + scale flag          PASS");

    // (7) Full layer: zero weights ⇒ identity (residual-only), and bitwise
    // determinism across runs.
    {
        let cfg = LayerCfg {
            hidden: 32, nq: 4, nkv: 2, head_dim: 8, rope_dim: 4, theta: 1e6,
            ffn: 64, eps: 1e-6, gamma_delta: 0.0, rope_conv: RopeConv::Interleaved,
            attn_scale: None, share_norm_weights: true,
        };
        let zw = |r: usize, c: usize| vec![0u16; r * c];
        let w0 = LayerWeights {
            norm1: vec![0u16; 32], norm2: vec![0u16; 32],
            wq: zw(32, 32), wk: zw(16, 32), wv: zw(16, 32), wo: zw(32, 32),
            w_gate: zw(64, 32), w_up: zw(64, 32), w_down: zw(32, 64),
        };
        let t = 3;
        let h_in = rand_f32(t * 32, &mut s);
        let h_out = forward_layer(&cfg, &w0, &h_in, t, 0);
        assert!(h_out.iter().zip(&h_in).all(|(a, b)| a.to_bits() == b.to_bits()),
            "zero-weight layer must be the identity (residual path)");

        // real random weights: deterministic + finite
        let w = LayerWeights {
            norm1: rand_bf16(32, &mut s, 0.5), norm2: rand_bf16(32, &mut s, 0.5),
            wq: rand_bf16(32 * 32, &mut s, 0.1), wk: rand_bf16(16 * 32, &mut s, 0.1),
            wv: rand_bf16(16 * 32, &mut s, 0.1), wo: rand_bf16(32 * 32, &mut s, 0.1),
            w_gate: rand_bf16(64 * 32, &mut s, 0.1), w_up: rand_bf16(64 * 32, &mut s, 0.1),
            w_down: rand_bf16(32 * 64, &mut s, 0.1),
        };
        let cfg2 = LayerCfg { gamma_delta: 1.0, attn_scale: Some(1.0 / (8f32).sqrt()),
            share_norm_weights: false, rope_conv: RopeConv::HalfSplit, ..cfg.clone() };
        let a = forward_layer(&cfg2, &w, &h_in, t, 7);
        let b = forward_layer(&cfg2, &w, &h_in, t, 7);
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "nondeterministic");
        assert!(a.iter().all(|x| x.is_finite()));
        // and the convention flags matter: megakernel-flavored cfg differs
        let c = forward_layer(&cfg, &w, &h_in, t, 7);
        assert!(a.iter().zip(&c).any(|(x, y)| (x - y).abs() > 1e-4),
            "conventions must produce measurably different anchors");
    }
    println!("[layer] zero-weight identity + determinism + flag sweep PASS");

    println!("\nBF16 ANCHOR REFERENCE VERIFIED — numerical contract locked");
}
