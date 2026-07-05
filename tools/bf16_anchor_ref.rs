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

//!
//! Shared math lives in tools/anchor_common.rs (also used by
//! tools/probe_checkpoint.rs); this file is the contract's test battery.

#![allow(dead_code)]

include!("anchor_common.rs");

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
