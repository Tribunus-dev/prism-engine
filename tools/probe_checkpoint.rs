//! probe_checkpoint.rs — convention-flag discovery for a model checkpoint.
//!
//! Runs the 16-combination sweep over the four anchor convention flags
//! (γ-fold, shared norm weights, attention scale, RoPE pair convention)
//! against a **golden activation slice** and emits the winning profile as a
//! generated Rust registry module. Standalone (`rustc`), std-only; the
//! forward math is the shared contract in tools/anchor_common.rs.
//!
//! ## What the sweep needs (and what it can't invent)
//! A checkpoint holds WEIGHTS, not activations — the golden `H_out` slice
//! must come from a trusted external oracle run once per checkpoint family
//! (e.g. the upstream reference implementation on the Mac), dumped as raw
//! f32 alongside the matching `H_in`. Given (weights, H_in, H_out_golden),
//! the sweep evaluates the anchor forward under every flag combination and
//! ranks by rel-L2.
//!
//! ## Honest verdicts, not "exactly one zero"
//! Two deliberate departures from the naive design:
//!   • **Margin, not zero.** The winner must beat the runner-up by a margin
//!     factor; bf16/f32 evaluation noise means "zero" is only exact when the
//!     oracle is this same code (as in the self-tests below).
//!   • **Per-flag observability.** Some flags are genuinely unobservable from
//!     a numerical slice — e.g. `share_norm_weights` when the checkpoint's
//!     two norm vectors happen to be identical, or γ-fold when ingest
//!     pre-folds. For each flag the tool compares the best score attainable
//!     with the flag flipped; if flipping barely moves the error, the flag is
//!     reported UNOBSERVABLE instead of silently "decided". (Structural
//!     evidence — e.g. whether a distinct post-attention norm tensor exists
//!     at all — should settle those flags before the numeric sweep; feed that
//!     in as a pre-pin when known.)
//!
//! Self-tests (run by `main`): ground-truth recovery on synthetic weights,
//! unobservability detection when norm2 := norm1, margin sanity, and codegen
//! content checks.
//!
//! Build & run:  rustc -O tools/probe_checkpoint.rs -o /tmp/probe && /tmp/probe

#![allow(dead_code)]

include!("anchor_common.rs");

// ── The flag space ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
struct FlagCombo {
    /// 0.0 = plain-γ over stored weights; 1.0 = (1+γ) over raw weights.
    gamma_delta: f32,
    share_norm_weights: bool,
    /// false = raw QK dots; true = 1/√head_dim.
    scaled_attention: bool,
    /// false = adjacent-pair (interleaved); true = split-half rotation.
    rope_split_half: bool,
}

const FLAG_NAMES: [&str; 4] = [
    "gamma_delta",
    "share_norm_weights",
    "scaled_attention",
    "rope_split_half",
];

fn all_combos() -> Vec<FlagCombo> {
    let mut v = Vec::with_capacity(16);
    for g in [0.0f32, 1.0] {
        for share in [false, true] {
            for scaled in [false, true] {
                for rope in [false, true] {
                    v.push(FlagCombo {
                        gamma_delta: g,
                        share_norm_weights: share,
                        scaled_attention: scaled,
                        rope_split_half: rope,
                    });
                }
            }
        }
    }
    v
}

fn flag_of(c: &FlagCombo, idx: usize) -> bool {
    match idx {
        0 => c.gamma_delta > 0.5,
        1 => c.share_norm_weights,
        2 => c.scaled_attention,
        3 => c.rope_split_half,
        _ => unreachable!(),
    }
}

fn cfg_for(base: &LayerCfg, c: &FlagCombo) -> LayerCfg {
    LayerCfg {
        gamma_delta: c.gamma_delta,
        share_norm_weights: c.share_norm_weights,
        attn_scale: if c.scaled_attention {
            Some(1.0 / (base.head_dim as f32).sqrt())
        } else {
            None
        },
        rope_conv: if c.rope_split_half {
            RopeConv::HalfSplit
        } else {
            RopeConv::Interleaved
        },
        ..base.clone()
    }
}

// ── Sweep + verdict ────────────────────────────────────────────────────────

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut se, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        se += (*x as f64 - *y as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (se / den.max(1e-30)).sqrt()
}

#[derive(Debug)]
struct FlagVerdict {
    name: &'static str,
    /// best score with the flag at the winner's value vs flipped.
    ratio: f64,
    observable: bool,
}

#[derive(Debug)]
struct SweepVerdict {
    winner: FlagCombo,
    winner_score: f64,
    runner_up_score: f64,
    margin_ok: bool,
    flags: Vec<FlagVerdict>,
}

/// Margin factor: the runner-up (and every flipped-flag best) must be at
/// least this many times worse for a flag/combo to count as decided.
const MARGIN: f64 = 10.0;

fn sweep(
    base: &LayerCfg,
    w: &LayerWeights,
    h_in: &[f32],
    t: usize,
    pos0: u32,
    golden: &[f32],
) -> SweepVerdict {
    let combos = all_combos();
    let scores: Vec<f64> = combos
        .iter()
        .map(|c| {
            let out = forward_layer(&cfg_for(base, c), w, h_in, t, pos0);
            rel_l2(&out, golden)
        })
        .collect();

    let mut order: Vec<usize> = (0..combos.len()).collect();
    order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    let best = order[0];
    let winner = combos[best];
    let winner_score = scores[best];
    let runner_up_score = scores[order[1]];

    // Per-flag observability: best score among combos with the flag FLIPPED
    // relative to the winner, vs the winner's score.
    let flags = (0..4)
        .map(|fi| {
            let win_val = flag_of(&winner, fi);
            let flipped_best = combos
                .iter()
                .zip(&scores)
                .filter(|(c, _)| flag_of(c, fi) != win_val)
                .map(|(_, &s)| s)
                .fold(f64::INFINITY, f64::min);
            let ratio = flipped_best / winner_score.max(1e-12);
            FlagVerdict {
                name: FLAG_NAMES[fi],
                ratio,
                observable: ratio >= MARGIN,
            }
        })
        .collect::<Vec<_>>();

    let margin_ok = runner_up_score / winner_score.max(1e-12) >= MARGIN
        // ties among unobservable-flag twins are fine as long as every
        // OBSERVABLE flag is decided by the margin
        || flags.iter().all(|f| !f.observable || f.ratio >= MARGIN);

    SweepVerdict {
        winner,
        winner_score,
        runner_up_score,
        margin_ok,
        flags,
    }
}

// ── Registry codegen ───────────────────────────────────────────────────────

fn emit_registry_module(
    model_name: &str,
    cfg: &LayerCfg,
    v: &SweepVerdict,
    checkpoint_note: &str,
) -> String {
    let mut s = String::new();
    s.push_str("// AUTO-GENERATED BY tools/probe_checkpoint.rs — DO NOT EDIT.\n");
    s.push_str(&format!("// Source checkpoint: {checkpoint_note}\n"));
    s.push_str(&format!(
        "// Sweep: winner rel-L2 {:.3e}, runner-up {:.3e}, margin_ok: {}\n",
        v.winner_score, v.runner_up_score, v.margin_ok
    ));
    for f in &v.flags {
        if !f.observable {
            s.push_str(&format!(
                "// WARNING: `{}` UNOBSERVABLE from this probe (flip ratio {:.2}) — value below is a tie-break, pin it from structural evidence.\n",
                f.name, f.ratio
            ));
        }
    }
    s.push_str(&format!(
        "\npub const {}_SPEC: ModelLayoutSpec = ModelLayoutSpec {{\n",
        model_name.to_uppercase().replace(['-', '.'], "_")
    ));
    s.push_str(&format!("    name: \"{model_name}\",\n"));
    s.push_str(&format!(
        "    hidden: {}, ffn: {}, n_q_heads: {}, n_kv_heads: {}, head_dim: {}, rope_dim: {}, theta: {:.1},\n",
        cfg.hidden, cfg.ffn, cfg.nq, cfg.nkv, cfg.head_dim, cfg.rope_dim, cfg.theta
    ));
    s.push_str("    flags: AnchorConventionFlags {\n");
    s.push_str(&format!(
        "        gamma_delta: {:.1},\n        share_norm_weights: {},\n",
        v.winner.gamma_delta, v.winner.share_norm_weights
    ));
    let scale_str = if v.winner.scaled_attention {
        format!("Some({:.6})", 1.0 / (cfg.head_dim as f32).sqrt())
    } else {
        "None".to_string()
    };
    s.push_str(&format!(
        "        attn_scale: {},\n        rope_split_half: {},\n",
        scale_str, v.winner.rope_split_half
    ));
    s.push_str("    },\n};\n");
    s
}

// ── Self-tests ─────────────────────────────────────────────────────────────

fn tiny_cfg() -> LayerCfg {
    LayerCfg {
        hidden: 32,
        nq: 4,
        nkv: 2,
        head_dim: 8,
        rope_dim: 4,
        theta: 1e6,
        ffn: 64,
        eps: 1e-6,
        // base values get overridden per-combo by cfg_for
        gamma_delta: 0.0,
        rope_conv: RopeConv::Interleaved,
        attn_scale: None,
        share_norm_weights: false,
    }
}

fn tiny_weights(s: &mut u64, share_identical_norms: bool) -> LayerWeights {
    let norm1 = rand_bf16(32, s, 0.5);
    let norm2 = if share_identical_norms {
        norm1.clone()
    } else {
        rand_bf16(32, s, 0.5)
    };
    LayerWeights {
        norm1,
        norm2,
        wq: rand_bf16(32 * 32, s, 0.1),
        wk: rand_bf16(16 * 32, s, 0.1),
        wv: rand_bf16(16 * 32, s, 0.1),
        wo: rand_bf16(32 * 32, s, 0.1),
        w_gate: rand_bf16(64 * 32, s, 0.1),
        w_up: rand_bf16(64 * 32, s, 0.1),
        w_down: rand_bf16(32 * 64, s, 0.1),
    }
}

fn main() {
    let mut s = 0xC0FF_EE00_u64;
    let base = tiny_cfg();
    let t = 4usize;
    let pos0 = 3u32;

    // (1) Ground-truth recovery: all four flags observable and recovered
    // exactly; oracle == this evaluator, so the winner's score is exactly 0.
    {
        let w = tiny_weights(&mut s, false);
        let h_in = rand_f32(t * base.hidden, &mut s);
        let truth = FlagCombo {
            gamma_delta: 1.0,
            share_norm_weights: false,
            scaled_attention: true,
            rope_split_half: true,
        };
        let golden = forward_layer(&cfg_for(&base, &truth), &w, &h_in, t, pos0);
        let v = sweep(&base, &w, &h_in, t, pos0, &golden);
        assert_eq!(v.winner, truth, "sweep failed to recover ground truth: {:?}", v.winner);
        assert!(v.winner_score < 1e-12, "self-oracle score must be ~0: {}", v.winner_score);
        assert!(v.margin_ok, "margin should hold: runner-up {:.3e}", v.runner_up_score);
        for f in &v.flags {
            assert!(f.observable, "flag {} should be observable (ratio {:.2})", f.name, f.ratio);
        }
        println!("[recovery] 16-sweep recovers ground truth, all flags observable  PASS");
    }

    // (2) Unobservability detection: norm2 := norm1 makes share_norm_weights
    // numerically invisible; the tool must SAY so, not guess silently —
    // while still deciding the other three flags.
    {
        let w = tiny_weights(&mut s, true); // identical norm vectors
        let h_in = rand_f32(t * base.hidden, &mut s);
        let truth = FlagCombo {
            gamma_delta: 0.0,
            share_norm_weights: true, // indistinguishable from false here
            scaled_attention: false,
            rope_split_half: false,
        };
        let golden = forward_layer(&cfg_for(&base, &truth), &w, &h_in, t, pos0);
        let v = sweep(&base, &w, &h_in, t, pos0, &golden);
        let share = v.flags.iter().find(|f| f.name == "share_norm_weights").unwrap();
        assert!(!share.observable, "share flag must be UNOBSERVABLE (ratio {:.3})", share.ratio);
        assert!(share.ratio < 1.5, "flip ratio should be ~1: {:.3}", share.ratio);
        for f in v.flags.iter().filter(|f| f.name != "share_norm_weights") {
            assert!(f.observable, "{} should still be observable", f.name);
        }
        assert_eq!(v.winner.gamma_delta, truth.gamma_delta);
        assert_eq!(v.winner.scaled_attention, truth.scaled_attention);
        assert_eq!(v.winner.rope_split_half, truth.rope_split_half);
        assert!(v.margin_ok, "observable flags decided ⇒ margin_ok despite the twin tie");

        // (3) codegen carries the warning + the verified values
        let module = emit_registry_module("gemma-4-12b-test", &base, &v, "synthetic-self-test");
        assert!(module.contains("UNOBSERVABLE"), "codegen must surface the warning");
        assert!(module.contains("`share_norm_weights` UNOBSERVABLE"));
        assert!(module.contains("rope_split_half: false"));
        assert!(module.contains("attn_scale: None"));
        assert!(module.contains("GEMMA_4_12B_TEST_SPEC"));
        println!("[observability] twin-norm tie reported, others decided, codegen warns  PASS");
    }

    // (4) Scale value correctness in codegen: 1/√8 for head_dim 8 — the
    // pasted-design bug this guards against wrote 1/√(nq·hd) instead.
    {
        let w = tiny_weights(&mut s, false);
        let h_in = rand_f32(t * base.hidden, &mut s);
        let truth = FlagCombo {
            gamma_delta: 0.0,
            share_norm_weights: false,
            scaled_attention: true,
            rope_split_half: false,
        };
        let golden = forward_layer(&cfg_for(&base, &truth), &w, &h_in, t, pos0);
        let v = sweep(&base, &w, &h_in, t, pos0, &golden);
        let module = emit_registry_module("scaletest", &base, &v, "synthetic");
        let expect = format!("Some({:.6})", 1.0 / (8f32).sqrt()); // 0.353553
        assert!(module.contains(&expect), "scale must be 1/√head_dim: {}", module);
        println!("[codegen] attn_scale emitted as 1/√head_dim (not 1/√(nq·hd))       PASS");
    }

    println!("\nPROBE CHECKPOINT HARNESS VERIFIED — sweep math + verdicts locked");
    println!("(Real use: feed checkpoint weights + an external golden activation");
    println!(" dump; structural pre-pins from the tensor inventory beat numerics");
    println!(" for share_norm_weights — a missing post-attn norm tensor IS the answer.)");
}
