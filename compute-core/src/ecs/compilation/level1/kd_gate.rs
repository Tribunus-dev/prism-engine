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
//! ## Memory — the exact accounting
//! Held logits are ONE flat `positions × vocab × 4`-byte buffer per model —
//! at the default 128 positions × 262144 vocab that is ~134 MB per model,
//! ~268 MB while both sides are held for scoring. The teacher pass streams
//! rows directly into the flat buffer ([`teacher_forced_flat`]), so the
//! transient peak per model is the flat buffer plus a single ~1 MB row — NOT
//! the ~2× peak the earlier nested-then-flatten shape had. The two
//! orchestrators are loaded **sequentially** (teacher is dropped before the
//! student loads), so model weights are never resident twice.
//!
//! `calibration_len` is a **hard cap on both token sources** (built-in
//! generator AND token files — see [`load_calibration_stream`]); predicted
//! validation bytes are therefore bounded by the request before any decode
//! starts, and the worker checks that bound against the declared memory
//! ceiling (Lane B preconditions, PRODUCTION_CONTRACT.md).
//!
//! [`teacher_forced_flat`]: crate::ecs::compilation::level1::teacher::Gemma4Teacher::teacher_forced_flat

use crate::ecs::compilation::distill_core::{kd_divergence_batch, top1_agreement};

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
        let end = if w == n_win - 1 {
            positions
        } else {
            start + base
        };
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
    let worst = windows.iter().fold((0.0f32, 1.0f32), |(wk, wt), w| {
        (wk.max(w.kd), wt.min(w.top1))
    });

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
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            1 + ((s >> 40) as u32) % vocab_cap.max(2)
        })
        .collect()
}

/// A loaded calibration stream with budget accounting.
///
/// `n` is a **hard cap on both sources**: the built-in generator emits exactly
/// `n` tokens, and a file-backed stream is deterministically truncated to its
/// first `n` tokens. A token file can therefore never silently exceed the
/// verification budget the request declared — the pre-hardening behavior
/// (file present ⇒ `n` ignored ⇒ unbounded decode) is gone. The accounting
/// fields exist so receipts can state exactly what happened
/// (PRODUCTION_CONTRACT.md: requested/loaded/used must be auditable).
#[derive(Debug, Clone)]
pub struct CalibrationStream {
    pub tokens: Vec<u32>,
    /// The cap the caller requested (`calibration_len` / `max_parity_tokens`).
    pub requested_tokens: usize,
    /// Tokens present in the source: the file's full parsed count, or
    /// `requested_tokens` for the built-in generator.
    pub loaded_tokens: usize,
    /// Tokens actually used — `min(requested, loaded)`; always `tokens.len()`.
    pub used_tokens: usize,
    /// True iff a file-backed stream was truncated to the cap (first
    /// `requested_tokens` tokens kept — deterministic by construction).
    pub truncated_by_policy: bool,
}

/// Load calibration tokens from a comma/whitespace-separated u32 file, or fall
/// back to the deterministic built-in stream — **always** capped at `n`.
pub fn load_calibration_stream(
    path: Option<&std::path::Path>,
    n: usize,
    vocab_cap: u32,
) -> Result<CalibrationStream, String> {
    if n == 0 {
        // Zero would either mean "no verification" (a policy the caller must
        // express by skipping the stage, not by a degenerate budget) or invite
        // the "0 = unlimited" misreading. Reject it.
        return Err(
            "calibration token budget is zero — declare a positive budget \
                    or skip the verification stage explicitly"
                .into(),
        );
    }
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
            let loaded = toks.len();
            let truncated = loaded > n;
            let tokens = if truncated { toks[..n].to_vec() } else { toks };
            let used = tokens.len();
            Ok(CalibrationStream {
                tokens,
                requested_tokens: n,
                loaded_tokens: loaded,
                used_tokens: used,
                truncated_by_policy: truncated,
            })
        }
        None => {
            let tokens = builtin_calibration_tokens(n, vocab_cap, 1);
            Ok(CalibrationStream {
                requested_tokens: n,
                loaded_tokens: tokens.len(),
                used_tokens: tokens.len(),
                truncated_by_policy: false,
                tokens,
            })
        }
    }
}

/// Compatibility wrapper over [`load_calibration_stream`] returning just the
/// tokens. NOTE: since the budget-hardening pass this **enforces `n` as a hard
/// cap on file-backed streams too** — the old behavior (file present ⇒ `n`
/// ignored) allowed a large token file to silently defeat the verification
/// budget. Prefer the stream API, which also reports the accounting.
pub fn load_calibration_tokens(
    path: Option<&std::path::Path>,
    n: usize,
    vocab_cap: u32,
) -> Result<Vec<u32>, String> {
    Ok(load_calibration_stream(path, n, vocab_cap)?.tokens)
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
    use crate::ecs::compilation::level1::teacher::Gemma4Teacher;

    if tokens.is_empty() {
        return Err("empty calibration token stream".into());
    }
    let mut runner =
        Gemma4Teacher::load(cimage).map_err(|e| format!("load {}: {e}", cimage.display()))?;
    // Streaming: rows land directly in ONE flat buffer (row-length checked as
    // they arrive). No nested Vec<Vec<f32>>, no flatten copy — the resident
    // peak is exactly positions × vocab × 4 bytes plus one transient row.
    let (logits, vocab) = runner
        .teacher_forced_flat(tokens)
        .map_err(|e| format!("teacher_forced_flat on {}: {e}", cimage.display()))?;
    let positions = tokens.len();
    debug_assert_eq!(logits.len(), positions * vocab);
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

// ═══════════════════════════════════════════════════════════════════════════
// Pipelined parity gate — the CPU-side validator math for the Stage 0 taps
// (kernels/STAGE0_TAPS_SPEC.md). Std-only + serde; lands BEFORE the taps so
// the validator's gate math is already tested when the Metal side arrives.
//
// Granularity: one LayerDriftReport per (layer, tap) comparison of a tapped
// activation against its golden source (bf16 anchor / megakernel oracle);
// one ParityManifest per decode token; hard breach ⇒ the pipelined validator
// stops submitting work and dumps the manifest + raw taps (taint preserved).
// All drift accumulation is f64 — the auditor must not add its own error.
// ═══════════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};

/// Which tap point a drift report measures (STAGE0_TAPS_SPEC slot map).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TapKind {
    /// Slot 0 — embedding output, before layer 0.
    PostEmbed,
    /// Slot 2k+1 — layer k after the attention residual.
    PostAttention,
    /// Slot 2k+2 — layer k after the FFN residual (the layer boundary).
    PostLayer,
    /// Last slot — final pre-logits hidden state.
    FinalHidden,
}

/// Drift of one tapped activation against its golden source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDriftReport {
    /// Transformer layer index. Ignored (0) for PostEmbed / FinalHidden.
    pub layer_idx: u32,
    pub tap: TapKind,
    /// ‖actual − golden‖₂ / ‖golden‖₂, f64-accumulated.
    pub rel_l2: f64,
    /// max |actual − golden| (infinity norm), f64.
    pub max_abs_error: f64,
    /// Bitwise equality of the f32 payloads (true ⇒ rel_l2 == 0 exactly).
    pub bitwise_identical: bool,
}

/// Compute the drift report for one tap. `actual` and `golden` must be the
/// same length (a mismatch is a plumbing bug, not a drift — assert, like the
/// distill_core scorers).
pub fn drift_report(
    actual: &[f32],
    golden: &[f32],
    layer_idx: u32,
    tap: TapKind,
) -> LayerDriftReport {
    assert_eq!(
        actual.len(),
        golden.len(),
        "tap length mismatch (layer {layer_idx}, {tap:?}) — wrong slot geometry, not drift"
    );
    let mut se = 0.0f64;
    let mut den = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut bitwise = true;
    for (a, g) in actual.iter().zip(golden) {
        let d = *a as f64 - *g as f64;
        se += d * d;
        den += (*g as f64) * (*g as f64);
        let ad = d.abs();
        if ad > max_abs {
            max_abs = ad;
        }
        if a.to_bits() != g.to_bits() {
            bitwise = false;
        }
    }
    LayerDriftReport {
        layer_idx,
        tap,
        rel_l2: (se / den.max(1e-300)).sqrt(),
        max_abs_error: max_abs,
        bitwise_identical: bitwise,
    }
}

/// Gate thresholds. `warn < hard`; drift ≤ warn passes silently, drift in
/// (warn, hard] logs telemetry (the 48-layer drift-curve signal), drift >
/// hard is a breach that stops the run. Default hard = 0.35 matches the
/// activation-error ceiling in JOINT_MTP_COMPILE.md §4.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ParityThresholds {
    pub hard: f64,
    pub warn: f64,
}

impl Default for ParityThresholds {
    fn default() -> Self {
        ParityThresholds {
            hard: 0.35,
            warn: 0.10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftStatus {
    Pass,
    Warn,
    HardBreach,
}

/// Classify one report against the thresholds.
pub fn classify_drift(report: &LayerDriftReport, t: &ParityThresholds) -> DriftStatus {
    assert!(
        t.warn <= t.hard,
        "ParityThresholds inverted: warn {} > hard {}",
        t.warn,
        t.hard
    );
    if report.rel_l2 > t.hard {
        DriftStatus::HardBreach
    } else if report.rel_l2 > t.warn {
        DriftStatus::Warn
    } else {
        DriftStatus::Pass
    }
}

/// Verdict decomposition over one token's reports. Indices refer into the
/// manifest's `reports` vec (which is in tap-slot order, so
/// `first_hard_breach` is also the EARLIEST point in the forward where parity
/// broke — the root-cause pointer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityVerdict {
    /// No hard breaches (warns allowed — they are telemetry).
    pub passed: bool,
    pub hard_breaches: Vec<usize>,
    pub warns: Vec<usize>,
    pub worst_rel_l2: f64,
    pub worst_index: Option<usize>,
    /// Earliest hard breach in forward order — where to start root-causing.
    pub first_hard_breach: Option<usize>,
}

/// One decode token's auditable parity record — the unit serialized into the
/// `.parity` sidecar (digest recorded in receipts, never baked into a binary).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityManifest {
    pub token_index: u64,
    pub thresholds: ParityThresholds,
    pub reports: Vec<LayerDriftReport>,
    pub verdict: ParityVerdict,
}

impl ParityManifest {
    /// Evaluate reports (in tap-slot / forward order) into a sealed manifest.
    pub fn evaluate(
        token_index: u64,
        thresholds: ParityThresholds,
        reports: Vec<LayerDriftReport>,
    ) -> Self {
        let mut hard_breaches = Vec::new();
        let mut warns = Vec::new();
        let mut worst_rel_l2 = 0.0f64;
        let mut worst_index = None;
        for (i, r) in reports.iter().enumerate() {
            match classify_drift(r, &thresholds) {
                DriftStatus::HardBreach => hard_breaches.push(i),
                DriftStatus::Warn => warns.push(i),
                DriftStatus::Pass => {}
            }
            if r.rel_l2 > worst_rel_l2 {
                worst_rel_l2 = r.rel_l2;
                worst_index = Some(i);
            }
        }
        let verdict = ParityVerdict {
            passed: hard_breaches.is_empty(),
            first_hard_breach: hard_breaches.first().copied(),
            hard_breaches,
            warns,
            worst_rel_l2,
            worst_index,
        };
        ParityManifest {
            token_index,
            thresholds,
            reports,
            verdict,
        }
    }
}

/// Map a STAGE0 tap slot index to its (TapKind, layer_idx) label.
/// Slot map: 0 = post-embed; 2k+1 = layer k post-attention; 2k+2 = layer k
/// post-layer; 2·layers+1 = final pre-logits hidden.
pub fn tap_slot_label(slot: usize, layers: u32) -> (TapKind, u32) {
    let last = 2 * layers as usize + 1;
    if slot == 0 {
        (TapKind::PostEmbed, 0)
    } else if slot == last {
        (TapKind::FinalHidden, 0)
    } else if slot % 2 == 1 {
        (TapKind::PostAttention, ((slot - 1) / 2) as u32)
    } else {
        (TapKind::PostLayer, ((slot - 2) / 2) as u32)
    }
}

/// Validate one token's tap slots against golden slots (both in STAGE0 slot
/// order, `2·layers + 2` entries each) into a sealed [`ParityManifest`].
pub fn validate_token_taps(
    token_index: u64,
    layers: u32,
    actual_slots: &[Vec<f32>],
    golden_slots: &[Vec<f32>],
    thresholds: ParityThresholds,
) -> Result<ParityManifest, String> {
    let expect = 2 * layers as usize + 2;
    if actual_slots.len() != expect || golden_slots.len() != expect {
        return Err(format!(
            "tap slot count mismatch: actual {}, golden {}, expected {expect}",
            actual_slots.len(),
            golden_slots.len()
        ));
    }
    let mut reports = Vec::with_capacity(expect);
    for (slot, (a, g)) in actual_slots.iter().zip(golden_slots).enumerate() {
        if a.len() != g.len() {
            return Err(format!(
                "slot {slot}: actual width {} != golden width {}",
                a.len(),
                g.len()
            ));
        }
        let (tap, layer_idx) = tap_slot_label(slot, layers);
        reports.push(drift_report(a, g, layer_idx, tap));
    }
    Ok(ParityManifest::evaluate(token_index, thresholds, reports))
}

/// Accumulates per-token manifests across an audit run and implements the
/// between-token early-exit policy: on the first hard breach, `push` returns
/// `false` (STOP — submit no further tokens; the caller dumps the taint) and
/// the run records where it stopped. Warn-band manifests continue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityRun {
    pub thresholds: ParityThresholds,
    pub manifests: Vec<ParityManifest>,
    /// Token index of the hard breach that stopped the run, if any.
    pub stopped_at_token: Option<u64>,
    pub worst_rel_l2: f64,
}

impl ParityRun {
    pub fn new(thresholds: ParityThresholds) -> Self {
        ParityRun {
            thresholds,
            manifests: Vec::new(),
            stopped_at_token: None,
            worst_rel_l2: 0.0,
        }
    }

    /// Record one token's manifest. Returns `true` to continue the run,
    /// `false` on a hard breach (between-token early exit).
    pub fn push(&mut self, manifest: ParityManifest) -> bool {
        let cont = manifest.verdict.passed;
        if manifest.verdict.worst_rel_l2 > self.worst_rel_l2 {
            self.worst_rel_l2 = manifest.verdict.worst_rel_l2;
        }
        if !cont && self.stopped_at_token.is_none() {
            self.stopped_at_token = Some(manifest.token_index);
        }
        self.manifests.push(manifest);
        cont
    }

    pub fn all_passed(&self) -> bool {
        self.stopped_at_token.is_none() && self.manifests.iter().all(|m| m.verdict.passed)
    }

    pub fn tokens_validated(&self) -> usize {
        self.manifests.len()
    }

    /// Serialize the whole run — the `.parity` sidecar payload.
    pub fn to_parity_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("serialize .parity: {e}"))
    }

    /// SHA-256 digest of the sidecar payload — the value recorded into
    /// `BlockReceipt`s / manifests so the artifact and its parity evidence
    /// are cryptographically bound (never baked into a binary).
    pub fn parity_digest(&self) -> Result<String, String> {
        use sha2::Digest;
        let json = self.to_parity_json()?;
        let mut h = sha2::Sha256::new();
        h.update(json.as_bytes());
        Ok(format!("{:x}", h.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── calibration budget contract (load_calibration_stream) ──────────────

    fn write_token_file(dir: &std::path::Path, name: &str, toks: &[u32]) -> std::path::PathBuf {
        let p = dir.join(name);
        let text = toks
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        std::fs::write(&p, text).unwrap();
        p
    }

    #[test]
    fn file_longer_than_budget_is_truncated_deterministically() {
        let dir = tempfile::tempdir().unwrap();
        let all: Vec<u32> = (1..=100).collect();
        let p = write_token_file(dir.path(), "toks.txt", &all);
        let s = load_calibration_stream(Some(&p), 16, 1000).unwrap();
        assert_eq!(s.requested_tokens, 16);
        assert_eq!(s.loaded_tokens, 100);
        assert_eq!(s.used_tokens, 16);
        assert!(s.truncated_by_policy);
        assert_eq!(s.tokens, (1..=16).collect::<Vec<u32>>(), "first-n prefix");
        // Determinism: same file, same budget ⇒ same stream.
        let s2 = load_calibration_stream(Some(&p), 16, 1000).unwrap();
        assert_eq!(s.tokens, s2.tokens);
    }

    #[test]
    fn file_shorter_than_budget_is_used_whole_not_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let all: Vec<u32> = (1..=8).collect();
        let p = write_token_file(dir.path(), "toks.txt", &all);
        let s = load_calibration_stream(Some(&p), 64, 1000).unwrap();
        assert_eq!(s.loaded_tokens, 8);
        assert_eq!(s.used_tokens, 8);
        assert!(!s.truncated_by_policy);
        assert_eq!(s.tokens, all);
    }

    #[test]
    fn zero_budget_is_rejected_not_unbounded() {
        let dir = tempfile::tempdir().unwrap();
        let p = write_token_file(dir.path(), "toks.txt", &[1, 2, 3]);
        assert!(load_calibration_stream(Some(&p), 0, 1000).is_err());
        assert!(load_calibration_stream(None, 0, 1000).is_err());
    }

    #[test]
    fn builtin_stream_respects_budget_exactly() {
        let s = load_calibration_stream(None, 32, 500).unwrap();
        assert_eq!(s.used_tokens, 32);
        assert_eq!(s.loaded_tokens, 32);
        assert!(!s.truncated_by_policy);
        assert_eq!(s.tokens, builtin_calibration_tokens(32, 500, 1));
    }

    #[test]
    fn compat_wrapper_enforces_the_cap_too() {
        // The old API must no longer allow a file to defeat the budget.
        let dir = tempfile::tempdir().unwrap();
        let all: Vec<u32> = (1..=50).collect();
        let p = write_token_file(dir.path(), "toks.txt", &all);
        let toks = load_calibration_tokens(Some(&p), 10, 1000).unwrap();
        assert_eq!(toks.len(), 10);
    }

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
        assert_eq!(
            g.reasons.len(),
            3,
            "all three criteria should trip: {:?}",
            g.reasons
        );
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

    // ── pipelined parity gate ──────────────────────────────────────────

    #[test]
    fn identical_taps_report_zero_and_pass() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.37).sin()).collect();
        let r = drift_report(&x, &x.clone(), 7, TapKind::PostLayer);
        assert_eq!(r.rel_l2, 0.0);
        assert_eq!(r.max_abs_error, 0.0);
        assert!(r.bitwise_identical);
        let m = ParityManifest::evaluate(0, ParityThresholds::default(), vec![r]);
        assert!(m.verdict.passed);
        assert!(m.verdict.hard_breaches.is_empty() && m.verdict.warns.is_empty());
    }

    #[test]
    fn drift_matches_hand_math_in_f64() {
        // golden = [3, 4] (‖g‖ = 5); actual adds (+0.3, −0.4) (‖d‖ = 0.5)
        // ⇒ rel_l2 = 0.1 exactly; max_abs = 0.4.
        let golden = vec![3.0f32, 4.0];
        let actual = vec![3.3f32, 3.6];
        let r = drift_report(&actual, &golden, 0, TapKind::PostEmbed);
        // Tolerance is f32-input-quantization bound (3.3f32 != 3.3 exactly);
        // the f64 accumulator adds nothing beyond it.
        assert!((r.rel_l2 - 0.1).abs() < 1e-7, "rel_l2 {}", r.rel_l2);
        assert!((r.max_abs_error - 0.4).abs() < 1e-6);
        assert!(!r.bitwise_identical);
    }

    #[test]
    fn verdict_decomposes_pass_warn_hard_with_earliest_breach() {
        let golden = vec![1.0f32; 64];
        let mk = |scale: f32, layer: u32| {
            let actual: Vec<f32> = golden.iter().map(|g| g + scale).collect();
            drift_report(&actual, &golden, layer, TapKind::PostLayer)
        };
        // rel_l2 == |scale| here (uniform delta over uniform golden).
        let reports = vec![mk(0.01, 0), mk(0.2, 1), mk(0.5, 2), mk(0.9, 3)];
        let m = ParityManifest::evaluate(42, ParityThresholds::default(), reports);
        assert!(!m.verdict.passed);
        assert_eq!(m.verdict.warns, vec![1]); // 0.2 ∈ (0.1, 0.35]
        assert_eq!(m.verdict.hard_breaches, vec![2, 3]);
        assert_eq!(
            m.verdict.first_hard_breach,
            Some(2),
            "earliest breach is the root-cause pointer"
        );
        assert_eq!(m.verdict.worst_index, Some(3));
        assert!((m.verdict.worst_rel_l2 - 0.9).abs() < 1e-6);
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let golden = vec![1.0f32, 2.0, 3.0];
        let actual = vec![1.0f32, 2.5, 3.0];
        let m = ParityManifest::evaluate(
            7,
            ParityThresholds::default(),
            vec![drift_report(&actual, &golden, 12, TapKind::PostAttention)],
        );
        let json = serde_json::to_string(&m).expect("serialize .parity");
        let back: ParityManifest = serde_json::from_str(&json).expect("deserialize .parity");
        assert_eq!(back.token_index, 7);
        assert_eq!(back.reports.len(), 1);
        assert_eq!(back.reports[0].tap, TapKind::PostAttention);
        assert_eq!(back.reports[0].layer_idx, 12);
        assert!((back.reports[0].rel_l2 - m.reports[0].rel_l2).abs() < 1e-15);
        assert_eq!(back.verdict.passed, m.verdict.passed);
    }

    #[test]
    #[should_panic(expected = "ParityThresholds inverted")]
    fn inverted_thresholds_panic() {
        let golden = vec![1.0f32; 8];
        let r = drift_report(&golden, &golden, 0, TapKind::FinalHidden);
        let bad = ParityThresholds {
            hard: 0.1,
            warn: 0.5,
        };
        let _ = classify_drift(&r, &bad);
    }

    // ── pipelined validator run core ───────────────────────────────────

    #[test]
    fn tap_slot_labels_cover_the_map() {
        let layers = 4u32; // 10 slots
        assert_eq!(tap_slot_label(0, layers), (TapKind::PostEmbed, 0));
        assert_eq!(tap_slot_label(1, layers), (TapKind::PostAttention, 0));
        assert_eq!(tap_slot_label(2, layers), (TapKind::PostLayer, 0));
        assert_eq!(tap_slot_label(7, layers), (TapKind::PostAttention, 3));
        assert_eq!(tap_slot_label(8, layers), (TapKind::PostLayer, 3));
        assert_eq!(tap_slot_label(9, layers), (TapKind::FinalHidden, 0));
    }

    #[test]
    fn validate_token_taps_labels_and_gates() {
        let layers = 2u32; // 6 slots
        let golden: Vec<Vec<f32>> = (0..6).map(|s| vec![1.0f32 + s as f32; 8]).collect();
        let mut actual = golden.clone();
        // Corrupt layer 1's post-attention slot (slot 3) hard.
        for v in actual[3].iter_mut() {
            *v += 10.0;
        }
        let m =
            validate_token_taps(5, layers, &actual, &golden, ParityThresholds::default()).unwrap();
        assert!(!m.verdict.passed);
        assert_eq!(m.verdict.first_hard_breach, Some(3));
        assert_eq!(m.reports[3].tap, TapKind::PostAttention);
        assert_eq!(m.reports[3].layer_idx, 1);
        // Slot-count mismatch is a plumbing error, not drift.
        assert!(validate_token_taps(5, layers, &actual[..5], &golden, Default::default()).is_err());
    }

    #[test]
    fn parity_run_early_exits_between_tokens() {
        let layers = 1u32; // 4 slots
        let golden: Vec<Vec<f32>> = (0..4).map(|_| vec![2.0f32; 4]).collect();
        let clean = golden.clone();
        let mut broken = golden.clone();
        for v in broken[2].iter_mut() {
            *v = -2.0; // rel_l2 = 2.0 ≫ hard
        }
        let mut run = ParityRun::new(ParityThresholds::default());
        assert!(run.push(validate_token_taps(0, layers, &clean, &golden, run.thresholds).unwrap()));
        assert!(
            !run.push(validate_token_taps(1, layers, &broken, &golden, run.thresholds).unwrap()),
            "hard breach must signal STOP"
        );
        assert_eq!(run.stopped_at_token, Some(1));
        assert!(!run.all_passed());
        assert_eq!(run.tokens_validated(), 2);
        // Digest is stable for identical runs and changes when content changes.
        let d1 = run.parity_digest().unwrap();
        let d2 = run.parity_digest().unwrap();
        assert_eq!(d1, d2);
        let mut run2 = ParityRun::new(ParityThresholds::default());
        run2.push(validate_token_taps(0, layers, &clean, &golden, run2.thresholds).unwrap());
        assert_ne!(d1, run2.parity_digest().unwrap());
    }
}
