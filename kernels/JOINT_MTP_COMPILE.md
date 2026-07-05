# Joint MTP co-distillation + dual-reference compile — repo mapping & corrections

Response to the joint-distillation / dual-reference / streaming-compiler design
discussion, checked line-by-line against the actual tree. Verdict up front:
the **streaming architecture and bf16-anchor principle are right and largely
already embodied in the v7 pipeline**; two pieces of the proposal contradict
either themselves or the codebase; and the λ_k question has a sharper answer
than a hand-picked schedule.

## 1. What the tree already has (don't rebuild it)

| Proposed | Reality in-tree |
|---|---|
| MTP heads appended to the graph | `NUM_MTP_HEADS = 4`, `MTP_HIDDEN = 2048` (+ own FFN, `MTP_TILES_FFN = 13`) already **execute in the megakernel** — logits land at `slot_logits + (head+1)·VOCAB`, read back via `read_slot_logits(slot, head)`. The execution graph has `DraftLayer` / MTP pre/post-projection node kinds. What's missing is **trained/distilled head weights**, not plumbing. |
| Outliers from bf16, not NF4 | Already the v7 contract: the ternary packer consumes **raw bf16** (`decode_scalar_from_raw`), outlier extraction and per-lane least-squares scales solve against bf16 — NF4 never touches the student's weight path. |
| `min_s ‖w_bf16 − s·t‖₂` per-lane scale solve | Exactly v7 Stage 3 (`s = Σw·t / Σt²`), plus 1D error diffusion with deadzone τ and the ε-flush at the 20-trit lane boundary, plus two-level scales (bf16 page-max + int8 lanes → **2.025 bpw**, matching the proposal's number because it *is* the v7 format). |
| `act_err = ‖W_bf16·x − W_tern·x‖/‖W_bf16·x‖` as gate | `tools/quant_lab.rs` computes it and CI-asserts on it. Promoting it to a **per-tile auto-tuning loop** (spike → tighten τ / raise outlier fraction for that tile) is new and good — it's a search wrapper around existing v7 knobs. |
| mmap-streamed weights | Ingest already mmaps the source safetensors (`loaded.mmap_bytes` + `source_tensor_view`). Streaming is an **iteration-order discipline**, not new infrastructure. One real fix: several paths `.to_vec()` whole raw views — fine per-tensor except **embed/lm_head (≈ 2.0 GB bf16)**, which should stream in row blocks. |
| Joint loss `L_primary + Σ λ_k·L_MTP_k` | Now in `distill_core::joint_kd_divergence` + `geometric_lambdas` (std-only, Linux-tested). Head 0 is hard-wired to weight 1.0; components stay separate for the gates (§4). |

## 2. Corrections to the proposal

### 2.1 "Concurrent distillation is a requirement" — true for training, not for PTQ
The freeze-the-torso argument ("ternary capacity ruthlessly discards MTP
precursor features") applies to **gradient training**. Prism's compile step is
**post-training quantization**: the torso's features are *inherited from the
bf16 QAT weights*, not re-learned — quantization noise degrades them but does
not re-allocate capacity away from MTP-relevant features. So for the current
pipeline, "joint" correctly means:
- quantize the MTP head tensors in the same per-layer pass (same calibration
  activations, same v7 stages), and
- include MTP logit KD in the **acceptance gates** (§4),
not a joint gradient objective. The concurrent-training argument becomes real
the moment block re-distillation does actual weight updates (the
`distill_core` doc reserves MLX/STE for that) — `joint_kd_divergence` is the
loss for that day, and it is deliberately shaped for both uses.

### 2.2 Stage 2 (NF4-guided ternary init) contradicts the anchor principle
Initializing trits from "a global structural threshold derived from the NF4
scales" imports NF4's quantization error into the student's weight path — the
precise thing §"dual-reference" forbids. It also buys nothing: the error-
diffusion loop re-decides every trit against the bf16 target anyway. **Keep
the v7 init (absmean threshold on bf16).** NF4's role stays where it is:
runtime logits reference (KD gates, bench) and structural/layout twin — never
a weight-space source.

### 2.3 Memory table — right conclusion, wrong arithmetic
Per-torso-layer is ≈ 224 M params → **~450 MB bf16** (not 1.1 GB); NF4 copy
~126 MB (not 0.3 GB). The **real streaming peak is embed/lm_head: ~2.0 GB
bf16** (1.007 B params), which the "one layer in RAM" budget must own (row-
block streaming makes it a non-event). Calibration buffers at 32×1024×3840:
~500 MB per f32 stream — and the proposal streams the *wrong pair* (§2.4);
with the right pair in f16 it's ~500 MB total. Realistic peak ≈ 4 GB system
+ ~0.5–1 GB activations + ~2.3 GB worst tensor window ≈ **~7 GB** — the
16 GB M1 fits with 2× the headroom the proposal claimed. Same conclusion
(stream layer-by-layer), friendlier margins.

### 2.4 Buffer the student's own activations, not H_nf4
The proposal keeps dual buffers (H_bf16, H_nf4). For sequential PTQ the pair
that matters is **(H_student_in → H_bf16_out)**: layer L must be optimized on
the inputs it will *actually see at runtime* (the student's own upstream
activations, which carry accumulated quantization error) against the bf16
golden outputs — this is what lets layer L *compensate* upstream error instead
of compounding it. The proposal's blueprint computes student activations from
`h_bf16_input`, which silently drops that compensation. H_nf4 hidden states
buy nothing at compile time; NF4 enters only at the logits-level gates.

### 2.5 "NF4 teacher `forward_layer`" does not exist off-Mac or per-op
The `DistillCompiler` sketch calls `nf4_teacher.forward_layer(...)`. Today the
NF4 teacher executes only as a whole-model megakernel forward — **and the
megakernel does not execute NF4Tile640 natively at all** (KERNEL_AUDIT /
PER_OP_FORWARD_PLAN). Per-layer NF4 teacher activations arrive with the per-op
forward (Stage 7) or megakernel taps (Stage 0). Until then the compile loop's
per-layer references are: bf16 anchor (H via MLX, §3) + logits-level NF4 KD at
the end of the pass. Design the trait now, bind it later:
`trait LayerForward { fn forward_layer(&mut self, l: usize, h_in: &[f32]) -> Vec<f32>; }`.

### 2.6 The missing execution piece: a bf16 anchor forward
Nothing in the tree runs the bf16 QAT model (`calibration/` has no forward;
grep confirms). The anchor stream H_bf16 needs an MLX layer-by-layer runner —
`mlx-rs` is already a dependency and this is its natural job. That runner is
**the** prerequisite for activation-anchored compilation; scope it as its own
deliverable (Mac-only, ~300 LoC against mlx-rs, checkpointed activations to
NVMe exactly as proposed).

## 3. Answer: how to weight λ_k

Four rules, in priority order:

1. **λ never appears in a gate.** Gates read `JointKd`'s components:
   `primary` must pass its own threshold (and `top1_agreement` on the primary
   head must not regress vs the λ=0 baseline) **before** any MTP term is even
   consulted. A joint scalar that can trade primary accuracy for MTP accuracy
   is the failure mode, not the objective — `joint_kd_divergence` returns the
   decomposition precisely so the gate can enforce this.
2. **Geometric decay as the default shape**: `geometric_lambdas(4, 0.3, 0.5)`
   → [0.30, 0.15, 0.075, 0.0375]. Rationale: head k's prediction only becomes
   useful after k−1 earlier acceptances, so its marginal throughput value
   decays roughly like α^k (α = per-token acceptance); its target is also
   intrinsically noisier. Σλ ≈ 0.56 keeps the torso's optimization pressure
   ~64% primary. (Precedent: DeepSeek-V3 ran its single MTP head at λ=0.3
   early → 0.1 late.)
3. **Tune λ against the real objective — projected speculative throughput.**
   `prism-bench-ab` already computes speedup as a function of per-head
   acceptance and measured costs. The right λ is the one that maximizes that
   number, and it's measurable: compile at λ ∈ {0, 0.15, 0.3, 0.6}, run the
   A/B, read the projection. First-order intuition for the trade: **1 pt of
   primary top-1 is worth ~k× more throughput than 1 pt on MTP head k** —
   which is why the schedule decays and the gate is primary-protected.
4. **In the PTQ regime (today), λ mostly weights the auto-tuner**, not
   gradients: when the per-tile τ/outlier search evaluates a candidate config,
   it scores `JointKd::total` over the calibration slice — λ decides how much
   an MTP-head regression can push a tile toward a richer config. Start at the
   geometric default; revisit when real gradient re-distillation lands.

## 4. Gate wiring (extends the existing pieces, no new frameworks)

- **Per-tensor**: `act_err ≤ 0.35` vs bf16 (quant_lab metric, now per-tile in
  the compile loop; spike → auto-tune τ/outlier fraction and retry).
- **Per-layer**: `block_accept(H_student, H_bf16, rel_tol)` — the anchor-space
  activation gate (needs §2.6's MLX runner for H_bf16).
- **Per-model**: `kd_gate` extended with `JointKd` — primary KD + top-1
  thresholds unchanged and independent; per-MTP-head KD thresholds added,
  looser by roughly the λ schedule.
- **Layout**: each MTP head's weight/scale/bias triplet is appended after the
  primary lm_head block and validated through `derive_nf4_tile640_arena_abi`
  like any other triplet (alignment/bounds/overlap come free); the
  `PackedTernaryPage640` student mirrors the same stacked order so the
  megakernel's existing per-head offset math
  (`per_head = MTP_HIDDEN·HID_TILES·LANES + HIDDEN_DIM·MTP_TILES·LANES`)
  keeps working unchanged.

## 5. Status

- **Landed, Linux-verified**: `distill_core::{JointKd, joint_kd_divergence,
  geometric_lambdas}` — 8/8 tests standalone (primary-isolation and λ-scaling
  invariants included).
- **Designed, not built**: streaming per-layer compile driver (iteration-order
  wrapper over existing mmap ingest + v7 stages, with the §2.4 buffer fix),
  MLX bf16 anchor runner (§2.6), per-tile τ/outlier auto-tuner, MTP triplet
  arena append.
- **Deferred until per-op forward lands**: per-layer NF4 teacher activations
  (H_nf4) — not needed for the compile loop per §2.4, useful later for
  layer-wise KD experiments.
