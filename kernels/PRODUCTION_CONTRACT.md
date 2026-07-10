# Production contract — build surfaces, budgets, lanes, receipts

The narrow contract the production runtime honors, as implemented by the
hardening pass (PR #8). Every claim here is enforced by code or CI, not
convention; file/symbol pointers are given so drift is auditable.

## 1. Three build surfaces

| Surface | Feature set | What it is | Off-repo needs |
|---|---|---|---|
| **Production runtime** | `prism-backend` | Metal dispatch, CoreML, orchestrator, distill worker, taps, fused decode, ingest tools | **None** — Xcode CLT + Rust + this repo |
| **Research** | `mlx-backend` (alias `research` = both) | MLX executor/model stack, profiled sessions, mlx serving routes, GGUF→cimage compile pipeline, encoders | MLX source tree (`TRIBUNUS_MLX_SOURCE_DIR` or sibling `mlx-tribunus/`; actionable panic otherwise — `mlx-rs-fork/mlx-sys/build.rs`) |
| **Tooling** | per-bin `required-features`; std tools ride none | packers, reference harnesses, admission tools | inherits its surface |

Mechanics: `mlx-rs`/`mlx-sys` are **optional dependencies** activated only by
`mlx-backend`; `prism-backend` no longer implies `mlx-backend`
(`compute-core/Cargo.toml`). ~30 modules and ~25 submodules of the legacy MLX
stack are gated behind `mlx-backend` (grep marker: `// research surface`).
Shared modules gate at item level. `prism-server` serves `/v1/chat/completions`
and the dashboard UI on the Orchestrator (megakernel) and is hermetic; the
profiled-MLX serving stack (`server::{engine, routes}`) is research.

**Hermeticity is CI-enforced**, not aspirational: `tools/ci/mac_runtime_gate.sh`
fails if `cargo tree --features prism-backend` contains `mlx-rs|mlx-sys`, then
proves the clean-checkout build; the Linux job asserts the same for
`backend-cpu`.

## 2. Verification budgets are hard contracts

- `calibration_len` caps **both** token sources: the built-in generator emits
  exactly that many, and a token file is **deterministically truncated** to
  its first `calibration_len` tokens
  (`level1::kd_gate::load_calibration_stream`). The pre-hardening behavior —
  file present ⇒ cap silently ignored ⇒ unbounded decode — is gone, including
  through the legacy `load_calibration_tokens` wrapper.
- A **zero budget is rejected**, never read as "unlimited".
- `max_parity_tokens` caps the parity stream independently (default: the KD
  budget).
- Truncation is not silent: `requested_tokens` / `loaded_tokens` /
  `used_tokens` / `truncated_by_policy` ride `CalibrationStream` into the
  operational receipt.

## 3. Honest memory accounting

- The KD stage holds **one flat `positions × vocab × 4` buffer per model**
  (`Gemma4Teacher::teacher_forced_flat` streams rows directly into it). The
  old `Vec<Vec<f32>>`-then-flatten shape — a ~2× transient peak the module
  docs understated — no longer exists.
- Lane B **predicts** its held bytes from the token budgets and the
  megakernel vocabulary *before any model loads*, and refuses to start when
  the prediction exceeds `validation_memory_ceiling_bytes` (default 1 GiB).
  Because the caps are hard, the prediction is a true upper bound.
- The receipt's `accounted_validation_bytes` counts buffers this code
  allocated — a contract check, explicitly not RSS.

## 4. Tap mode is declared, not ambient

`TapMode { Untapped, TappedAudit }` is an **orchestrator construction
parameter** (`Orchestrator::from_cimage_with_mode`), recorded in the
operational receipt:

- The parity stage constructs its teacher explicitly `TappedAudit` — no
  `TRIBUNUS_TAPS` requirement — and refuses an untapped teacher **before any
  decoding begins**.
- An explicitly `Untapped` orchestrator refuses the taps API even when the
  env var is set: the mode wins (test:
  `tap_mode_explicit_construction_beats_env`).
- A precompiled metallib (built without `-DPRISM_TAPS`) is **ignored** in
  tapped-audit builds, which runtime-compile from source — closing the latent
  hole where env-requested taps silently produced an untapped kernel.
- `TapMode::from_env()` remains only as the back-compat default for the plain
  `from_cimage` constructor.
- `teacher_mode: "untapped"` in a request that also asks for parity is
  rejected at Lane B resolution, before any model loads.

## 5. Multimodal NF4 bias residency is real and observable

The `MULTIMODAL_NF4_BIAS_ABI.md` design is **implemented** (format v1-compatible):

- `SegmentKind::MultimodalProjectionBiases = 28`, byte-parallel to the scales
  segment; records gate on `FLAG_HAS_BIAS`; descriptor carries
  `projection_bias_segment_index` in the former `image_reserved` slot
  (stride/size unchanged — asserted by `descriptor_layout_is_stride_stable`).
- The packer fills the bias segment in lockstep with scales from
  `{stem}.biases` sidecars, enforcing all-or-none and offset-parallelism at
  pack time; the loader/binding enforce the parallel view at load time.
- The runner binds resident biases when declared, refuses a flagged record
  whose artifact lacks the segment, and **logs the residency decision for
  every projection** (`RESIDENT` / `ZERO-FALLBACK`).
- Request policy `multimodal_bias_policy`: `auto` (default) /
  `require-resident` (Lane A fails v1 zero-bias artifacts with `abi-mismatch`)
  / `zero-only`. The decision lands in the operational receipt.

## 6. Three validation lanes, ordered and budgeted

| Lane | What | Failure class on violation |
|---|---|---|
| **A — structural** | cimage open/verify, checkpoint-vs-ternary validation, multimodal bias policy vs sealed segments | `abi-mismatch` (artifact) / `operational` (I/O) |
| **B — bounded numerical** | KD over the capped stream; parity over the capped stream; **preconditions**: mode/inputs consistency, tap-mode consistency, predicted-bytes ≤ ceiling — all checked before any model loads | `operational` (budget/config), `gate-rejection` only via verdicts |
| **C — compile/distill** | the block loop + Verifying gates | never starts if A failed or B's preconditions failed |

Requests declare intent: `validation_mode` (`structural` / `kd` /
`kd+parity`) — declaring a mode whose inputs are missing is an operational
error, so ambiguous requests die at resolution, not mid-run.

## 7. Failure taxonomy

`DistillationState::RejectedByGate` (terminal, receipts complete) is distinct
from `Failed` (operational fault). `failure_class` on the job status makes it
machine-readable:

- `gate-rejection` — scientific verdict on complete evidence (KD threshold,
  parity hard breach + taint artifact).
- `abi-mismatch` — the artifact violates a format/contract (checkpoint
  validation, bias-residency policy).
- `operational` — environment/configuration/budget faults (missing
  prerequisites, exceeded ceilings, I/O, task join errors).

## 8. The operational receipt

`compilation::receipt::OperationalReceipt` — one per job, embedded in the job
status **and** written as `<teacher>.ops.json` at every terminal state:
build profile, `mlx_linked`, validation mode, teacher tap mode, bias
policy + residency, requested/loaded/used token counts + truncation flag for
both stages, predicted vs ceiling vs accounted validation bytes, failure
class. This is the field-debugging record: what ran, in which modes, against
which budgets — no prose required.

## 9. What "run as intended" means (the acceptance table)

| Capability | Production requirement | Where enforced |
|---|---|---|
| Compile the runtime | Clean checkout, no out-of-repo MLX | `mac_runtime_gate.sh` §1 |
| Open teacher + student | Sequential residency, never dual unless requested | `kd_gate` module docs + stage code |
| KD validation | Hard token cap + memory ceiling precondition | `load_calibration_stream`, Lane B gate |
| Parity validation | Explicit tapped mode, fail-fast if untapped | `run_parity_stage` |
| Multimodal NF4 projection | Residency stated + logged per projection | `run_nf4_multimodal_projection` |
| Fused decode | Audit lane (group 1, taps) vs production fusion (2–4) stated in receipts | Transport B suite + ops receipt |
| Failures | gate-rejection vs abi-mismatch vs operational | `FailureClass` |

## 10. CI split

- **Linux (hermetic)**: `backend-cpu` build + full std suite + clippy +
  `cargo tree` MLX-absence assert — **no cmake, no MLX checkout** since the
  decouple. The research surface remains Linux-checkable via
  `tools/mlx_harness.sh` (79 known off-target artifacts, 0 real).
- **Mac**: `tools/ci/mac_runtime_gate.sh` — dependency-graph assert →
  clean-checkout production build (lib + `prism-server` with `server-dashboard`, `prism-bench-ab`,
  `tribunus-pack-nf4tile640`) → untapped/tapped smokes when `TRIBUNUS_TEST_CIMAGE` is
  provided.
- Workflow file: `.github/workflows/ci.yml` (integration tokens cannot push
  workflow paths — the PR carries the patch in a comment; apply manually).
