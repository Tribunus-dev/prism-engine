# Speculative decoding in prism-engine — design

**Student drafts, teacher verifies.** The ternary student proposes k tokens;
the NF4 teacher checks them in one (eventually batched) pass; the longest
agreeing prefix commits plus one teacher token. Output is teacher-quality at a
projected 1.5–2× teacher decode speed (`prism-bench-ab`'s spec-decode
projection measures the inputs to that claim on your hardware; this doc designs
the runtime that turns the projection into a measurement).

Three components, per the DSpark discussion mapped to batch-1:
1. **Batched verification** with per-position logits
2. **KV rollback** to the accepted prefix
3. **Confidence-based dynamic draft length**

---

## 1. Ground truth: what the runtime actually has today

Everything below was read out of the current branch, with the surprises called
out — the design stands on these facts.

### 1.1 The decode path is a persistent megakernel fed by a work ring
`MegakernelPipeline::submit_work` (pipeline.rs) writes ring entries of
`{state | kind<<2, token_id, seq_pos, kv_slot_id, reserved}`; the persistent
GPU worker consumes them in order. Two facts matter enormously:

- **`kv_slot_id` is decoupled from the work slot.** A work item can target any
  KV partition — so *several sequential work items can extend the same KV
  context*. This is the hook that makes a verification oracle possible with
  **zero kernel changes** (§3, Phase V0).
- `kind` selects the program: `kind=0` full decode, `kind=3` the in-kernel
  draft model (`submit_draft`, up to N candidate tokens + log-probs). MTP head
  outputs exist per slot (`read_slot_logits(slot, head)`).

### 1.2 Prefill is ANE-batched but emits **no logits**
`prefill_slot` (runner.rs) runs a Core ML model: token_ids `[1, prompt_len]` →
K/V caches for all 48 layers in one shot, copied into Metal scratch and packed
to ternary. There is **no logits head in the prefill model** — so "verification
via the prefill path" cannot mean the ANE path as-built. Verification needs
per-position logits; the prefill model would need re-export with an lm_head
output (and the vocab is 262 144 — a `[k+1, 262144]` f16 output is only ~2.6 MB
for k=4, but ANE round-trip latency for a 5-token window likely eats the win).
**Design decision: verification runs on Metal**, not ANE (§3). The ANE prefill
remains the prompt-ingest path for both models.

### 1.3 Logits are centroid-scout restricted (but masked, not stale)
The megakernel's final projection computes real logits only for the best
centroid cluster `[cstart, cend)` and sweeps the rest of the vocab with a
sentinel mask every step (gemma4_full.metal ~889–925). Consequences:
- Greedy acceptance against these logits is **self-consistent**: it is lossless
  w.r.t. *the same scout-logits teacher users already get* from plain decode.
- The true argmax can live outside the chosen cluster; teacher and student may
  even scout **different clusters** on the same context. For verification this
  is acceptable (we define losslessness w.r.t. the production teacher path),
  but the batched-verify kernel should offer an **exact-logits mode** for
  measurement runs (§3.2) — it amortizes the big lm_head read across k+1
  positions anyway.

### 1.4 KV cache: ternary-packed, slot-partitioned, sinks + FIFO
Per (slot, layer, position, kv-head): base-3 packed nibbles + per-256-block
fp16 scales; `NUM_SINKS=4` StreamingLLM sinks; positions ≥ `MAX_CTX` (2048) map
through a **cyclic FIFO** (`kv_cache_pos`, gemma4_full.metal 467–471). The
Rust side tracks `slot_seq_pos[slot]`. Two rollback-relevant properties:
- Pre-wrap (`pos < MAX_CTX`), the position→address map is the identity: a
  rejected position's KV slot is **exactly** the address the next accepted
  token at that position will overwrite. Rollback is a counter rewind.
- Post-wrap, a speculative write lands in a FIFO slot whose previous occupant
  was *only conditionally* evictable — rolling back would need that occupant
  restored (§4).

### 1.5 Prior art in-tree, honestly assessed
- **`decode_with_mtp` (runner.rs 786–861) is not speculative decoding and is
  unsound as written.** It (a) generates top-K *alternatives for one position*
  from a single distribution — not a draft chain; (b) verifies each candidate
  C with a **fixed-point rule** (accept if `model(context+C)` predicts C
  again — true for stutters like "the the", not for normal text); (c) runs
  candidates against **slots 1+ KV caches**, which hold other sequences'
  context or nothing; and (d) advances `slot_seq_pos[0] += accepted.len()`,
  committing several same-position alternatives as if sequential — corrupting
  the position/KV bookkeeping. This design supersedes it; the function should
  be deprecated or rewritten as the §5 loop.
- **`decode_fused`** dispatches `decode_layer_swa`/`decode_layer_full`, which
  are **identity stubs** (decode_per_layer.metal). Not a usable substrate.
- The **in-kernel draft model** (`kind=3`, DRAFT_* weights) and **MTP heads**
  are real machinery for a *cheaper* draft source than a full second model —
  see §7. The full ternary student remains the primary drafter here because
  the bench projection directly measures its acceptance rate, and distillation
  (the distill loop) is what makes that acceptance high.
- **KV compaction** (entropy-gathered survivors on ANE) runs only inside
  `prefill_slot` — it does not interfere with decode-time speculation, but any
  future decode-time compaction must be fenced out of speculative windows
  (it reorders positions and invalidates rollback assumptions).

---

## 2. The loop (target semantics)

```text
committed context C, last committed token t₀
loop:
  # DRAFT (student, its own KV)
  d ← []
  while |d| < k_max and student_confidence ≥ τ:
      (tok, logits_s) ← student.decode(last)   # advances student KV
      d.push(tok); last ← tok
  # VERIFY (teacher, one pass over k+1 positions)
  L ← teacher.verify([t₀] + d)                 # per-position logits, writes teacher KV
  # ACCEPT: longest prefix where argmax(L[i]) == d[i]   (greedy rule, v1)
  a ← accept_len(L, d)
  commit d[..a] + [argmax(L[a])]               # bonus/correction token
  # ROLLBACK both models to |C| + a + 1
  teacher.truncate_to(base + a + 1)
  student.truncate_to(base + a + 1)            # student replays the correction token
  t₀ ← last committed
```

Properties: output distribution equals the teacher's greedy stream (w.r.t. the
same kernel path used for verification — see §6 on numerical identity);
`k = 0` degenerates to plain teacher decode; the student is *never trusted*,
only used to prefetch agreement.

Sampling-rule acceptance (Leviathan rejection sampling) is a v2 extension: it
needs full per-position distributions from both models plus a committed RNG
stream. Greedy first — it is what the bench measures and what the runtime's
determinism story supports.

---

## 3. Component 1 — verification

Three phases, each independently testable, each subsuming the last.

### Phase V0 — sequential-verify oracle (no kernel changes)
Submit k+1 ordinary `kind=0` work items in ring order, **all with
`kv_slot_id=0`**, positions `base..base+k`, tokens `[t₀, d₁..d_k]`; the
persistent worker consumes them serially, so item i's KV write is visible to
item i+1's attention. Read `read_slot_logits` after each (or poll the last and
read all). Acceptance on CPU.

- Cost: k+1 sequential teacher steps → **no speedup, by design**. This is the
  correctness scaffold: it exercises drafting, acceptance, commit, and rollback
  with the *bitwise-identical* kernel used for plain decode, so the
  losslessness test (§6, T1) is exact.
- Risk: near zero. Pure Rust (`SpecDecoder` in §5), lands first.

### Phase V1 — batched verify mode in the megakernel (`kind=2`)
One work item carrying the whole window: ring entry uses `token_id` as an
index into a small `verify_tokens` buffer and `reserved` as `num_tokens`
(≤ k_max+1). Inside the persistent kernel:

- **Projections (QKV, O, gate/up/down, logits) batch across the M=k+1
  positions**: the tile loop reads each weight tile once and multiply-
  accumulates against M activation vectors — `tile_gemv` → `tile_gemv_m`.
  This is where the memory-bound win lives (weights dominate traffic; the
  extra activation traffic is noise).
  - Threadgroup memory reality: `n_buf` is HIDDEN_DIM halfs = 7.7 KB; M=5
    copies = 38 KB > the 32 KB budget the kernel plans for. **Keep per-position
    activations in device memory** (`[M × 3840]` halfs, ~38 KB device-resident
    — trivial) and stage per-tile slices through threadgroup as needed.
- **Attention stays sequential within the window** (position i must see i−1's
  K/V): per layer, loop i = 0..M−1 { project K/V for i, RoPE, scatter to cache,
  attend over `[0, base+i]` }. The projections inside that loop are still
  batched across i where independence allows (K/V for all M positions are
  independent given the layer input — project all M first, then the serial
  attend loop).
- **Logits**: batched across M with one lm_head weight read. Two modes:
  `scout` (production parity — each position scouts its own cluster) and
  `exact` (full-vocab per position, for measurement + sampling-rule v2).
- Output: `[M × VOCAB]` logits region per slot (extend `LOGITS_PER_SLOT` or
  add a `verify_logits` buffer).

Expected verify cost: ≈ one teacher step + ε (the measured value replaces the
bench's `--spec-verify-factor` assumption — the projection column becomes a
measured column).

### Phase V2 — per-op verification (convergence with PER_OP_FORWARD_PLAN)
The per-op forward's Stage 7 runner generalizes to M>1 naturally (its GEMV
dispatchers already amortize weight reads per dispatch; its SDPA kernel takes
`num_cached` explicitly). When the per-op path lands for layer-wise
distillation, it becomes the second verification substrate — useful for
cross-checking V1 and for NF4 teachers, since **the megakernel does not
execute NF4Tile640 natively** (per-op is the only native NF4 path). Until
then, V0/V1 verify with whatever cimage format the megakernel runs.

---

## 4. Component 2 — KV rollback

### 4.1 Teacher & student, pre-wrap (the common case)
`truncate_to(p)`:
1. Rewind `slot_seq_pos[slot] = p` (Rust).
2. Nothing else. Attention reads are bounded by `num_cached` (derived from the
   work item's `seq_pos`), and the position→address map is deterministic, so
   the rejected positions' packed KV blocks are dead bytes that the next
   accepted tokens at those positions overwrite exactly.

The `entropy_map` accumulates junk contributions from rejected positions —
instrumentation only, but decode-time compaction driven by it must never run
inside a speculative window (today it cannot; keep it that way by fencing
`truncate_to` + verify inside one "speculative section").

### 4.2 Post-wrap (`base + k ≥ MAX_CTX`)
A speculative write at wrapped position p overwrites the FIFO slot of an entry
that was live until the commit decision. Two policies:

- **v1 — don't speculate across the wrap boundary.** When
  `base + k_max ≥ MAX_CTX`, clamp k so the window stays pre-wrap; at the
  boundary itself fall back to plain decode for that step. Costs nothing in
  the common regime (sessions ≤ 2048 positions never hit it) and removes the
  problem.
- **v2 — snapshot/restore.** Before verify, copy the (≤ k) packed KV blocks +
  scales the window will overwrite (~1.7 KB × 48 layers ≈ 82 KB per position;
  ~400 KB for k=4 — trivial on unified memory); restore on rejection. Only
  worth building when long-session speculation matters.

### 4.3 Student rollback
Same mechanism on the student's Orchestrator (own `slot_seq_pos`, own cache).
One asymmetry: after a rejection the student must ingest the teacher's
correction token to stay on the committed trajectory — that is just its next
`decode_token(correction)` call after `truncate_to`.

### 4.4 New Orchestrator API
```rust
/// Rewind a slot to `pos` committed tokens. Pre-wrap this is O(1) counter
/// arithmetic; post-wrap behavior is governed by WrapPolicy.
pub fn truncate_to(&mut self, slot_id: u32, pos: u32) -> Result<(), String>;
pub fn seq_pos(&self, slot_id: u32) -> u32;
```

---

## 5. Component 3 — confidence-based dynamic draft length

DSpark's confidence-scheduled verification, ported to batch-1 (no cross-request
scheduler — that half of DSpark does not apply):

- **Stop rule**: after each student draft step, compute the student's top-1
  probability `c = max softmax(logits_s)`; stop drafting when `c < τ` or
  `k = k_max`. Each avoided student step is a saved Metal dispatch; each
  avoided low-confidence draft is a verify slot not wasted on a likely
  rejection. No trained head needed (DSpark trains one; the student's own
  confidence is the zero-cost proxy — upgrade path in §7).
- **Calibrating τ**: the bench's captured per-position logits already contain
  everything needed — bucket per-position greedy acceptance by student top-1
  probability and pick τ for a target acceptance (e.g. the knee of the curve).
  Extension to `prism-bench-ab`: `--emit-confidence-curve` printing
  `τ → (mean k, acceptance, projected speedup)`. Until measured, default
  τ ≈ 0.5 with k_max from the projection table's best k.
- Cost note: the max-prob scan is over the scout cluster's ~few-hundred live
  logits (rest are masked) — microseconds on CPU; fuse into the kernel later
  only if profiling says so.

```rust
pub struct SpecConfig {
    pub k_max: usize,          // from the bench projection's best k
    pub conf_tau: f32,         // 0.0 = fixed-k
    pub accept: AcceptRule,    // Greedy (v1) | RejectionSample { seed } (v2)
    pub wrap: WrapPolicy,      // ClampAtBoundary (v1) | SnapshotRestore (v2)
    pub verify: VerifyPath,    // SequentialOracle (V0) | BatchedKind2 (V1)
}

pub struct SpecDecoder {
    teacher: Orchestrator,
    student: Orchestrator,
    cfg: SpecConfig,
    stats: SpecStats,          // per-cycle: drafted, accepted, conf_stops, timings
}
impl SpecDecoder {
    pub fn from_cimages(teacher: &Path, student: &Path, cfg: SpecConfig) -> Result<Self, String>;
    pub fn prefill(&mut self, prompt: &[u32]) -> Result<(), String>;   // both models
    pub fn decode_cycle(&mut self, last: u32) -> Result<SpecCycle, String>;
}
```
`SpecStats` feeds the same `bench_metrics` tables — measured acceptance and
tokens/cycle land next to the projection for a direct predicted-vs-actual
readout.

---

## 6. Verification plan (every phase gated, house style)

All on-Mac, env-gated (`TRIBUNUS_TEST_CIMAGE_TEACHER/_STUDENT`); Linux CI keeps
the pure-math pieces (acceptance/rollback bookkeeping unit tests are host-
independent Rust).

| # | Test | Gates |
|---|---|---|
| T0 | **Self-draft identity**: student := teacher cimage. Every draft must accept; committed stream **bitwise equals** teacher-alone greedy over 256 tokens (V0 uses the identical kernel, so exact equality is required, not approximate) | V0 loop, ring ordering, commit bookkeeping |
| T1 | **Losslessness with real student**: committed stream equals teacher-alone greedy over 256 tokens; measured acceptance within ±5 pts of `prism-bench-ab`'s α_greedy on the same stream | acceptance rule, drafting |
| T2 | **Rollback soundness**: run with a deliberately bad student (high rejection); after every rejection continue decoding; final stream still equals teacher-alone. Repeat crossing the `MAX_CTX` boundary → clamp policy engages, stream still exact | truncate_to, wrap policy |
| T3 | **Batched-verify parity** (V1): per-position logits vs V0 oracle — argmax agreement on ≥ 99.9% of positions, rel-L2 within fp16 reduction tolerance; then re-run T1 on V1; then measure tok/s and print predicted-vs-actual vs the projection table (this measurement **replaces** `--spec-verify-factor`) | kind=2 kernel |
| T4 | **Dynamic k**: τ sweep on a fixed stream — mean drafted k decreases monotonically with τ; best-τ speedup ≥ best fixed-k speedup on the same stream | confidence rule |
| T5 | **Determinism**: two identical runs produce identical streams + stats | whole loop |

Numerical-identity caveat, stated once: V1's batched reductions order
differently than single-token decode, so "teacher-alone reference" for T3's
losslessness re-run means *greedy decode through the V1 path itself* (submit
1-token verify windows). Losslessness is always defined w.r.t. the verifying
kernel's own distribution.

---

## 7. Sequencing, effort, and the draft-source ladder

| Stage | Deliverable | LoC (est.) | Risk |
|---|---|---|---|
| S0 | `truncate_to`/`seq_pos` + wrap clamp + unit tests | ~120 | Low |
| S1 | `SpecDecoder` with V0 oracle + T0/T1/T2/T5 | ~450 | Low-Med |
| S2 | Confidence stop rule + τ calibration in bench + T4 | ~200 | Low |
| S3 | `kind=2` batched verify (tile_gemv_m, serial-attend window, batched exact/scout logits) + T3 | ~600 Metal + ~200 Rust | **High** |
| S4 | Sampling acceptance (exact logits + committed RNG) | ~250 | Med |
| S5 | Post-wrap snapshot/restore (only if long sessions demand it) | ~200 | Med |

Draft-source ladder (all verify through the same path — the loop doesn't care
who drafted):
1. **Full ternary student** (this doc's primary): best acceptance, needs both
   models resident (~2.6× compression means teacher+student ≈ 1.6× the teacher
   alone at these bpw — budget it against the machine's 16 GB).
2. **In-kernel draft model** (`kind=3`, already resident): near-zero memory,
   lower acceptance; becomes attractive once the distill loop actually trains
   those DRAFT_* weights.
3. **MTP heads** (DSpark's semi-AR direction): drafts k tokens in ~one pass —
   the long-term ceiling-raiser, blocked on training the heads (DeepSpec-style
   data is exactly what the distillation runtime produces).

The bench projection decides whether S3 is worth building at all: if the
measured α_greedy puts every k below ~1.15× projected speedup, stop after S1
and revisit after the next distillation round improves agreement — a
publishable "not yet" is a valid outcome, and the S1 oracle keeps measuring it
for free.
