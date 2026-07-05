//! KD gate — wires the **real** NF4 teacher into the distillation loop.
//!
//! The producers below load a `.cimage` through [`Gemma4Teacher`] (the
//! megakernel-backed runner), run a teacher-forced pass over a fixed
//! calibration token stream, and hand the per-position logits to
//! `distill_core::{kd_divergence_batch, top1_agreement}`. The resulting
//! [`KdReport`] is stamped into every `BlockReceipt` and enforced as a gate in
//! the job's Verifying phase.
//!
//! ## Scope — honest about granularity
//! This stage scores the **whole student model** against the whole teacher
//! (end-to-end logits KD). It does NOT isolate the KD contribution of a single
//! transformer block: that requires running a hybrid forward with one block
//! swapped (teacher blocks 0..k−1 + student block k + teacher blocks k+1..),
//! which needs the per-op forward (`kernels/PER_OP_FORWARD_PLAN.md`, Stage 7).
//! The [`KdReport::block_kd`] slot exists so block-swap scores can be filled in
//! when that path lands; until then every block receipt carries the model-level
//! numbers, which is the strongest signal the megakernel path can produce.
//!
//! Per-position **windows** give a stability signal the scalar mean hides: a
//! student that matches early positions but diverges as context grows shows up
//! as a rising per-window KD even when the mean looks acceptable.
//!
//! ## Memory
//! Held logits are `positions × vocab × 4` bytes per model — at the default
//! 128 positions × 262144 vocab that is ~134 MB per model (~268 MB while both
//! are alive). The two orchestrators are loaded **sequentially** (teacher is
//! dropped before the student loads), so model weights are never resident
//! twice. Scale `calibration_len` with available memory.

use crate::compilation::distill_core::{kd_divergence_batch, top1_agreement};

/// Gate thresholds. Defaults are **initial** values chosen to be forgiving —
/// calibrate them after the first real teacher-vs-student run on hardware
/// (`prism-bench-ab` reports the same metrics over the same stream).
#[derive(Debug, Clone)]
pub struct KdGateConfig {
    /// KD softmax temperature (matches `prism-bench-ab` default).
    pub temperature: f32,
    /// Gate fails if mean KD (T²·KL) exceeds this.
    pub max_kd: f32,
    /// Gate fails if top-1 agreement falls below this.
    pub min_top1: f32,
    /// Gate fails if the worst per-window KD exceeds this (divergence spikes
    /// that the mean would smooth over).
    pub max_window_kd: f32,
    /// Number of contiguous position windows for the spread signal.
    pub windows: usize,
}

impl Default for KdGateConfig {
    fn default() -> Self {
        KdGateConfig {
            temperature: 2.0,
            max_kd: 0.75,
            min_top1: 0.55,
            max_window_kd: 1.5,
            windows: 8,
        }
    }
}

/// Teacher-forced logits for one model over the calibration stream.
#[derive(Debug, Clone)]
pub struct CalibrationLogits {
    /// Row-major `[positions × vocab]`.
    pub logits: Vec<f32>,
    pub vocab: usize,
    pub positions: usize,
}

/// KD over one contiguous span of calibration positions.
#[derive(Debug, Clone, Copy)]
pub struct KdWindow {
    pub start: usize,
    pub end: usize, // exclusive
    pub kd: f32,
    pub top1: f32,
}

/// Model-level KD comparison of student vs teacher.
#[derive(Debug, Clone)]
pub struct KdReport {
    /// Mean KD (T²·KL) across all calibration positions.
    pub kd: f32,
    /// Top-1 agreement across all positions.
    pub top1: f32,
    /// Worst (max-KD) window — the divergence-spike signal.
    pub worst_window_kd: f32,
    pub worst_window_top1: f32,
    pub windows: Vec<KdWindow>,
    pub positions: usize,
    pub vocab: usize,
    /// Per-block KD contributions from block-swap runs. Empty until the per-op
    /// forward lands (PER_OP_FORWARD_PLAN.md Stage 7); reserved so receipts and
    /// gates don't need an interface change then.
    pub block_kd: Vec<f32>,
}

/// Gate verdict with human-readable reasons for every failed criterion.
#[derive(Debug, Clone)]
pub struct KdGateResult {
    pub passed: bool,
    pub reasons: Vec<String>,
}

/// Score student logits against teacher logits over the same calibration
/// stream. Both must be `[positions × vocab]` over identical tokens.
pub fn score_student_logits(
    teacher: &CalibrationLogits,
    student: &CalibrationLogits,
    cfg: &KdGateConfig,
) -> Result<KdReport, String> {
    if teacher.vocab != student.vocab {
        return Err(format!(
            "vocab mismatch: teacher {} vs student {} — not the same tokenizer/model family",
            teacher.vocab, student.vocab
        ));
    }
    if teacher.positions != student.positions {
        return Err(format!(
            "position count mismatch: teacher {} vs student {} — calibration streams differ",
            teacher.positions, student.positions
        ));
    }
    let vocab = teacher.vocab;
    let positions = teacher.positions;
    if vocab == 0 || positions == 0 {
        return Err("empty calibration logits".into());
    }
    if teacher.logits.len() != positions * vocab || student.logits.len() != positions * vocab {
        return Err("ragged logits: len != positions × vocab".into());
    }

    let kd = kd_divergence_batch(&teacher.logits, &student.logits, vocab, cfg.temperature);
    let top1 = top1_agreement(&teacher.logits, &student.logits, vocab);

    // Contiguous windows over positions (last window absorbs the remainder).
    let n_win = cfg.windows.clamp(1, positions);
    let base = positions / n_win;
    let mut windows = Vec::with_capacity(n_win);
    let mut start = 0usize;
    for w in 0..n_win {
        let end = if w == n_win - 1 { positions } else { start + base };
        let t = &teacher.logits[start * vocab..end * vocab];
        let s = &student.logits[start * vocab..end * vocab];
        windows.push(KdWindow {
            start,
            end,
            kd: kd_divergence_batch(t, s, vocab, cfg.temperature),
            top1: top1_agreement(t, s, vocab),
        });
        start = end;
    }
    let worst = windows
        .iter()
        .fold((0.0f32, 1.0f32), |(wk, wt), w| (wk.max(w.kd), wt.min(w.top1)));

    Ok(KdReport {
        kd,
        top1,
        worst_window_kd: worst.0,
        worst_window_top1: worst.1,
        windows,
        positions,
        vocab,
        block_kd: Vec::new(),
    })
}

/// Apply the gate thresholds to a report.
pub fn kd_gate(report: &KdReport, cfg: &KdGateConfig) -> KdGateResult {
    let mut reasons = Vec::new();
    if report.kd > cfg.max_kd {
        reasons.push(format!(
            "mean KD {:.4} exceeds max_kd {:.4}",
            report.kd, cfg.max_kd
        ));
    }
    if report.top1 < cfg.min_top1 {
        reasons.push(format!(
            "top-1 agreement {:.3} below min_top1 {:.3}",
            report.top1, cfg.min_top1
        ));
    }
    if report.worst_window_kd > cfg.max_window_kd {
        reasons.push(format!(
            "worst window KD {:.4} exceeds max_window_kd {:.4} (divergence spike)",
            report.worst_window_kd, cfg.max_window_kd
        ));
    }
    KdGateResult {
        passed: reasons.is_empty(),
        reasons,
    }
}

/// Deterministic built-in calibration stream (same LCG as `prism-bench-ab`, so
/// bench and gate agree when no real token file is provided). Token IDs are in
/// `[1, vocab_cap]` — keep `vocab_cap` below every model's vocab.
pub fn builtin_calibration_tokens(n: usize, vocab_cap: u32, seed: u64) -> Vec<u32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            1 + ((s >> 40) as u32) % vocab_cap.max(2)
        })
        .collect()
}

/// Load calibration tokens from a comma/whitespace-separated u32 file, or fall
/// back to the deterministic built-in stream.
pub fn load_calibration_tokens(
    path: Option<&std::path::Path>,
    n: usize,
    vocab_cap: u32,
) -> Result<Vec<u32>, String> {
    match path {
        Some(p) => {
            let text = std::fs::read_to_string(p)
                .map_err(|e| format!("read calibration tokens {}: {e}", p.display()))?;
            let toks: Vec<u32> = text
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter(|s| !s.is_empty())
                .map(|s| {
                    s.parse::<u32>()
                        .map_err(|e| format!("bad token id {s:?} in {}: {e}", p.display()))
                })
                .collect::<Result<_, _>>()?;
            if toks.is_empty() {
                return Err(format!("no tokens parsed from {}", p.display()));
            }
            Ok(toks)
        }
        None => Ok(builtin_calibration_tokens(n, vocab_cap, 1)),
    }
}

/// Whether this build can execute cimages for KD scoring (needs the Metal
/// megakernel path). Mirrors the `run_level2_block` availability pattern.
pub const fn kd_available() -> bool {
    cfg!(all(target_os = "macos", feature = "prism-backend"))
}

/// Run a teacher-forced pass over `tokens` on the given `.cimage` and return
/// its per-position logits. Used for BOTH sides of the comparison — the "NF4
/// teacher" and the "ternary student" are each just a cimage run through the
/// megakernel-backed [`Gemma4Teacher`] runner. The orchestrator is dropped on
/// return, so calling this twice never holds two models resident.
#[cfg(all(target_os = "macos", feature = "prism-backend"))]
pub fn compute_calibration_logits(
    cimage: &std::path::Path,
    tokens: &[u32],
) -> Result<CalibrationLogits, String> {
    use crate::compilation::level1::teacher::Gemma4Teacher;

    if tokens.is_empty() {
        return Err("empty calibration token stream".into());
    }
    let mut runner = Gemma4Teacher::load(cimage)
        .map_err(|e| format!("load {}: {e}", cimage.display()))?;
    let per_position = runner
        .teacher_forced(tokens)
        .map_err(|e| format!("teacher_forced on {}: {e}", cimage.display()))?;

    let positions = per_position.len();
    let vocab = per_position.first().map(Vec::len).unwrap_or(0);
    if vocab == 0 {
        return Err(format!("{}: empty logit rows", cimage.display()));
    }
    if per_position.iter().any(|row| row.len() != vocab) {
        return Err(format!("{}: ragged logit rows", cimage.display()));
    }
    let mut logits = Vec::with_capacity(positions * vocab);
    for row in &per_position {
        logits.extend_from_slice(row);
    }
    Ok(CalibrationLogits {
        logits,
        vocab,
        positions,
    })
}

/// Non-Metal builds cannot execute cimages; the distill loop checks
/// [`kd_available`] first and records a skip reason instead of calling this.
#[cfg(not(all(target_os = "macos", feature = "prism-backend")))]
pub fn compute_calibration_logits(
    _cimage: &std::path::Path,
    _tokens: &[u32],
) -> Result<CalibrationLogits, String> {
    Err("KD scoring requires macOS + the prism-backend feature (Metal megakernel)".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(positions: usize, vocab: usize, seed: u64) -> CalibrationLogits {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 40) as f32) / ((1u64 << 24) as f32) * 8.0 - 4.0
        };
        CalibrationLogits {
            logits: (0..positions * vocab).map(|_| next()).collect(),
            vocab,
            positions,
        }
    }

    #[test]
    fn identical_student_scores_zero_and_passes() {
        let t = synth(16, 40, 7);
        let cfg = KdGateConfig::default();
        let r = score_student_logits(&t, &t.clone(), &cfg).unwrap();
        assert!(r.kd.abs() < 1e-5, "kd {} not ~0", r.kd);
        assert_eq!(r.top1, 1.0);
        assert!(r.worst_window_kd.abs() < 1e-5);
        let g = kd_gate(&r, &cfg);
        assert!(g.passed, "reasons: {:?}", g.reasons);
    }

    #[test]
    fn perturbed_student_diverges_and_strict_gate_fails() {
        let t = synth(16, 40, 7);
        let mut s = t.clone();
        // Invert every row's logits — argmax flips, distributions diverge hard.
        for v in s.logits.iter_mut() {
            *v = -*v;
        }
        let cfg = KdGateConfig::default();
        let r = score_student_logits(&t, &s, &cfg).unwrap();
        assert!(r.kd > 0.1, "kd {} unexpectedly small", r.kd);
        assert!(r.top1 < 0.5, "top1 {} unexpectedly high", r.top1);
        let strict = KdGateConfig {
            max_kd: 0.01,
            min_top1: 0.99,
            max_window_kd: 0.01,
            ..KdGateConfig::default()
        };
        let g = kd_gate(&r, &strict);
        assert!(!g.passed);
        assert_eq!(g.reasons.len(), 3, "all three criteria should trip: {:?}", g.reasons);
    }

    #[test]
    fn windows_partition_all_positions() {
        let t = synth(19, 12, 3); // 19 % 8 != 0 → last window absorbs remainder
        let r = score_student_logits(&t, &t.clone(), &KdGateConfig::default()).unwrap();
        assert_eq!(r.windows.first().unwrap().start, 0);
        assert_eq!(r.windows.last().unwrap().end, 19);
        for pair in r.windows.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "windows must be contiguous");
        }
        assert!(r.windows.iter().all(|w| w.end > w.start));
    }

    #[test]
    fn mismatches_are_rejected() {
        let t = synth(8, 40, 1);
        let cfg = KdGateConfig::default();
        let wrong_vocab = synth(8, 41, 2);
        assert!(score_student_logits(&t, &wrong_vocab, &cfg).is_err());
        let wrong_positions = synth(9, 40, 2);
        assert!(score_student_logits(&t, &wrong_positions, &cfg).is_err());
        let mut ragged = synth(8, 40, 2);
        ragged.logits.pop();
        assert!(score_student_logits(&t, &ragged, &cfg).is_err());
    }

    #[test]
    fn builtin_tokens_deterministic_and_bounded() {
        let a = builtin_calibration_tokens(64, 1000, 1);
        let b = builtin_calibration_tokens(64, 1000, 1);
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert!(a.iter().all(|&t| t >= 1 && t <= 1000));
        let c = builtin_calibration_tokens(64, 1000, 2);
        assert_ne!(a, c, "different seeds must differ");
    }
}
