# Stage 0 taps — a spec for BOTH execution architectures

Formal spec for extracting per-layer teacher activations (PER_OP_FORWARD_PLAN
Stage 0), responding to the blit-interception / pipelined-parity proposal.

Two architectures are in play and the answer differs on each:
- **Incumbent** — the single-dispatch persistent megakernel
  (`gemma4_full_decode_persistent`): the only path that executes end-to-end
  today. Blit interception is inexpressible here; taps must be in-shader.
- **Target** — fused interleaved per-group kernels (`decode_fused` +
  `fused_swa/full_{pair,triple,quad}`, group size 1–4, decompress interleaved
  with the previous group's matmul): the intended production architecture.
  Today every one of those kernel bodies is an identity stub — but on this
  path the proposal's blit/event pipeline is essentially RIGHT, and mostly
  free.

The proposal's goals stand everywhere — race-free extraction, pipelined
validation, auditable manifests, early exit. Corrections first (they apply to
the incumbent path and to the memory math), then the per-architecture spec,
then the punchline: the two roadmaps meet.

## Corrections (each checked against the tree)

1. **On the incumbent path there is nothing to blit-intercept.** The
   proposal assumes per-layer command encoding ("after encoding
   dispatchThreadgroups for Layer N, insert a DMA copy"). The working runtime
   dispatches the megakernel **once** — pipeline.rs:417: "One-shot dispatch of
   persistent kernel (runs forever)" — and the 48 layers are a `for` loop
   *inside one shader invocation*. No encoder boundary exists between layers;
   `encodeSignalEvent` is command-granular, not loop-iteration-granular. The
   per-group-encoding world the proposal describes is exactly the **target**
   fused-interleaved architecture (`decode_per_layer.metal`) — where it is the
   right design (§Transport B) — but those kernels are identity stubs today.
   Until they are real, per-layer activations from the teacher that actually
   runs require the **shader itself to write the taps** (§Transport A).
2. **"Shader pollution" is not a real cost here — quantitatively or
   procedurally.** Taps are gated by a Metal **function constant**
   (`TAPS_ENABLED`): the taps-off pipeline is specialized at PSO-creation
   time — dead code eliminated, zero branches, zero register pressure — and
   Stage 0's test (c) asserts taps-off output is *bitwise identical* to
   today's kernel. Taps-on cost: `h_buf` (already in threadgroup memory) is
   streamed to a device buffer twice per layer — 2×48×7.5 KB ≈ **735 KB per
   token** against ~2.4 GB of weight traffic per token: ~0.03% bandwidth. And
   the taps-on PSO only runs during audit/compile passes, whose metric is
   parity, not tok/s; production inference never compiles it in.
3. **The ring buffer solves a non-problem.** The proposal sizes taps as
   `batch × seq_len × hidden × f32` (32×1024 tokens → ~503 MB/layer) — that is
   the **prefill/anchor-compile shape**, which lives on the CPU side of the
   pipeline (the bf16 anchor runner and v7 student evaluation are CPU passes;
   their activations are ordinary `Vec<f32>`s, "extraction" is a no-op). The
   megakernel taps serve **decode-granularity** teacher parity: one token's
   full double-tap set is 735 KB. Allocate the whole `[2·48+2]` tap buffer
   (StorageModeShared), no ring, no back-pressure, no risk of the shader
   spin-waiting on a CPU consumer inside a persistent kernel (GPU-side waits
   on CPU flags inside a forever-kernel are watchdog/occupancy hazards — the
   proposal's "release Layer N+1 only after Layer N clears the gate" would
   require exactly that).
4. **Three taps per layer is one too many.** Pre-norm input ≡ previous
   layer's post-residual output (already tapped). Post-projection
   (pre-residual) is recoverable as `post_attn − prev_out` on the CPU when
   needed. Stage 0 keeps: post-embed, per-layer post-attention-residual,
   per-layer post-layer-residual, final pre-logits hidden.
5. **`build.rs` must not bake a PARITY_HASH into the engine binary** — same
   wrong-layer objection as the checkpoint-hash seal (MULTIMODAL/registry
   discussion): the compiler/runtime binary is artifact-agnostic; the seal
   travels **with the artifact** (`.parity` sidecar + digest recorded in the
   cimage manifest / BlockReceipts, where `contract_digest`/`layout_digest`
   infrastructure already exists).
6. **`read_volatile` theatrics are unnecessary.** Reads follow either
   `waitUntilCompleted` (batch mode) or an **Acquire load on a device-scope
   progress atomic** (pipelined mode) — the same shared-buffer atomic
   convention the work ring already uses (`poll_work` + SeqCst fence). Match
   the house convention; don't invent a second one.

## The spec

### Transport A — megakernel (the teacher that exists today)

- **Buffer**: `layer_taps` — `[(2·LAYERS + 2) × HIDDEN]` f16,
  `StorageModeShared`, ~735 KB. Slot 0 = post-embed; `2k+1` = layer k
  post-attention-residual; `2k+2` = layer k post-layer; last = final
  pre-logits hidden.
- **Shader**: behind `TAPS_ENABLED` function constant, after each residual:
  `for (i = tid; i < HIDDEN; i += tg_sz) layer_taps[slot·HIDDEN + i] = h_buf[i];`
  then (pipelined mode only) one thread does an atomic release-store of the
  slot number to `tap_progress`.
- **Progress counter**: `tap_progress` — a device-scope atomic u32 in a shared
  buffer, monotonically set to the last completed slot index for the current
  work item. CPU consumers acquire-load it. This gives the proposal's "event
  monotonicity rule" with the mechanism the persistent kernel can actually
  express (shared events cannot be signaled from inside a dispatch).
- **API**: `Orchestrator::decode_token_logits_with_taps(token) ->
  (logits, TapsView)` — batch mode waits for work-item completion then reads
  the whole buffer (no atomics needed beyond the existing poll); streaming
  mode exposes `tap_progress` for the pipelined validator.
- **Tests** (Stage 0's three, unchanged): tap self-consistency
  (`taps[last]` → final norm → logits reproduces returned logits);
  determinism; **taps-off bitwise identity**.

### Transport B — fused interleaved kernels (the target architecture)

When the `decode_fused` path is real, the proposal's structure applies nearly
verbatim — with one simplification and one policy:

- **Boundary taps are (usually) blit-free.** Between fusion-group dispatches
  the hidden state already lives in a device buffer (`decode_fused` chains
  `current` buffers between groups). If group outputs are freshly allocated
  per group, the tap IS the buffer — record its reference, no copy. Only if
  buffers are ping-ponged/reused does a `MTLBlitCommandEncoder` snapshot into
  a tap slot become necessary (and it is concurrent with compute, as the
  proposal says). `encodeSignalEvent(event, group_idx)` after each group is
  now expressible and correct — the proposal's monotonic-event handshake
  applies as written.
- **Tap granularity = dispatch granularity, by policy.** Fusing 2–4 layers
  exists precisely to eliminate the intermediate global writes — which are
  the states a per-layer tap needs. Don't fight the fusion: **audit/compile
  passes request fusion group size 1** (the analyzer's `group.count == 1`
  path already exists), so every layer boundary is a device buffer and taps
  are free; production inference fuses 2–4 and taps nothing. No function
  constants, no in-shader tap writes, genuinely zero shader deltas on this
  path.

### The punchline — the roadmaps meet

Making `decode_layer_full`/`decode_layer_swa` real (one dispatch = one full
layer) IS the per-op forward plan's destination at layer granularity — the
fused-interleaved migration and PER_OP_FORWARD_PLAN Stages 2–7 are the same
authoring work seen from two angles. And authoring those kernel bodies needs
per-layer ground truth to parity-test against — which only the incumbent
megakernel can provide. So Transport A is not a competing tap design; it is
**the oracle that lets Transport B be built**:

1. Transport A (~150 LoC, function-constant gated) → per-layer truth from the
   teacher that runs today; distillation + gate graduation unblock now.
2. Author the real fused per-layer kernel bodies, parity-gated per layer
   against Transport A taps.
3. Compile/audit passes migrate to Transport B (group size 1, blit-free
   boundary taps, shared-event pipelining); production fuses 2–4.
4. The megakernel remains the cross-check oracle; Transport A stays behind
   its function constant at zero production cost.

### Pipelined parity validation — token-granular, not layer-granular

On Transport A, layer-synchronous gating ("only then release Layer N+1")
cannot be expressed without in-kernel CPU waits. On Transport B it CAN be
expressed (hold group N+1's submission on group N's verdict) — but it should
not be: one token costs ~30–100 ms end to end and the 128-token calibration
pass costs seconds; serializing CPU validation into the GPU pipeline buys
taint-preservation that token-granular gating already provides. The right
pipeline on both transports:

```
GPU: decode token t   → taps written → work item t completes
CPU:                    validate token t−1's taps   (overlapped)
gate: hard breach at token t−1 ⇒ stop submitting t+1, dump taint
```

- CPU validation of token t−1 overlaps GPU decode of token t (submit-ahead
  of exactly 1 token). Zero GPU stalls, zero in-kernel synchronization.
- **Hard breach** (`rel_l2 > hard`): stop submitting work, keep the tap
  buffer + the failing token id, dump all 96 per-layer reports + the raw taps
  to the diagnostic artifact. Taint preserved at full fidelity — better than
  the layer-abort version, which would have destroyed layers > k's states by
  never computing them.
- **Warn band** (`warn < rel_l2 ≤ hard`): telemetry only — the drift-curve
  signal (a spike at layer 24 says exactly where the structural flaw is).
- Golden source: per-layer bf16 anchor activations (Accelerate runner /
  `anchor_common` contract), f64 accumulation in the auditor — both already
  the house rule in every reference tool.

### The auditable manifest

`ParityManifest` (implemented in `level1::kd_gate`, cargo-tested on Linux):
per-(token, layer, tap) `LayerDriftReport { rel_l2, max_abs, bitwise }`,
thresholds echoed, verdict decomposed hard/warn/pass, worst-layer summary.
Serialized as `.parity` beside the cimage; its digest recorded in the block
receipts (`numerical_drift` map) and the cimage manifest — **not** in the
engine binary.

### Where it wires in (the scheduler question, answered)

`Level1Scheduler`'s dispatch loop is **not refactored**. Taps ride the
Orchestrator (megakernel side); the parity gate is consumed by
`distill_worker` — the same seam the model-level KD stage already occupies —
via a per-layer stage that runs when taps + anchor activations are available.
The synthetic `MetalTeacher` keeps exercising scheduling mechanics;
`gates.rs::check_numerical` (today a no-arg stub) grows a real signature fed
by `ParityManifest` when the per-layer path lands. Graduation is additive, not
a scheduler rewrite.
