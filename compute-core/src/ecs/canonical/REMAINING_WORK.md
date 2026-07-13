# Remaining work

> Current snapshot: `23e8a5f`. This document
> distinguishes declared contracts, connected call paths, real execution, and
> validation against real inputs or hardware.

## Repository state

The build and CI configuration, Metal/NF4 changes, cimage runtime changes,
quantization changes, runtime boundary changes, and test changes are now split
into reviewable commits. The remaining implementation work is deliberately
kept separate from those commits.

The Apple Silicon build uses optimized incremental local profiles and a
non-incremental GitHub Actions profile with sccache. The macOS CI job targets an
arm64 macOS 26 runner and must still be confirmed by a successful remote run.

## Metal catalogue and execution

The Metal catalogue is authoritative for source lookup and is consumed by
several runtime and compilation paths. The NF4 tile640 ABI now uses a typed,
aligned parameter block, and the real residency test covers M=2, K=4, N=640.

Remaining work is to route every remaining production kernel creation through the
backend compiler contract, remove remaining duplicate shader implementations,
replace zero or symbolic ABI byte sizes with checked contracts, and make
catalogue source identity part of compiled artifact provenance.

## Compiler integration

The canonical compiler routes GGUF and sealed cimage inputs through the real
pipeline under the required feature gates. Compile outcomes are populated from
compiled images and optional request parameters are forwarded.

Remaining work is live event emission from compiler stages and complete source,
policy, artifact, and toolchain identity digests. The event stream currently
contains post-hoc reconstruction with missing identities.

## Shared measured evaluator

Evolutionary search has candidate genomes, mutation, crossover, selection,
budgets, replay, and Pareto structures. The NF4 tile640 Metal evaluator now
compiles catalogue source, dispatches a real fixture, compares against the CPU
oracle, warms the path, and records repeated timing. The legacy `run` path
still uses a hand-written cost function, while `run_measured` now routes
candidates through static, numerical, and performance receipts and rejects
failed candidates.

The evaluator now reuses one compiled Metal pipeline for warm-up and measured
repetitions, and generation promotion accepts numerical/performance evidence
and enforces those gates. The next required implementation is to generalize the
evaluator beyond the NF4 fixture and persist the receipts in the search state.
Device-specific limits are validated for the current fixture but still need
broader target coverage.

## Engram training and runtime

Engram contracts, scheduler operations, payload segments, lookup receipts, CPU
additive and multiplicative application, and generation bindings exist.

The trainer now has a deterministic additive-residual optimizer over the
declared `EngramTrainingDataset`, emits parameter payload bytes, and checks
holdout, validation, and explicit interference loss. Lookup uses cosine
similarity over f32 payload parameters, and CPU application validates payload
width. Metal application is unimplemented, and
low-rank, latent-prefix, and adapter application modes remain placeholders.

The generation API now accepts `TrainedEngram` directly with evaluator evidence,
and the lifecycle test trains a dataset, promotes the exact payload, retrieves
and applies it, then rolls back to the parent generation. Remaining work is
Metal application and wiring this orchestration into the production compiler
service rather than only the typed API.

## Ternarization and assimilation

Ternary candidates, reconstruction gates, residual encoders, assimilation
receipts, and generation-level payload types exist and have unit coverage.

The scale optimizer currently re-evaluates its incumbent scale but does not yet
explore candidate scales. Ternary packaging stores one byte per weight rather
than a native packed representation. Assimilation comparisons can be lossless
because the full dense residual is retained, so strategy fitness does not yet
measure a real storage or execution tradeoff.

Remaining work is a real scale/threshold optimizer, native packing, residual
policy enforcement, executable reconstruction, and replay validation through
the shared evaluator.

## MLIR first-class ECS capability

Prism now has an additive target-independent MLIR execution contract in
`ecs/mlir.rs`. It models dialect requirements, quantization attributes,
transform schedules, lowering targets, deterministic module text, and the
existing NF4 tile640 workload without coupling the ECS crate to MLIR's C++
runtime.

Remaining work is a real lowering adapter, beginning with MLIR-shaped NF4
tile640 output that is compiled and measured through the existing Metal
evaluator. Vendor-specific lowering, HetGPU packaging, and replacing the
handwritten Metal path should follow only after that adapter passes the same
CPU-oracle, numerical, timing, and promotion gates.

## Generations and promotion

Content-addressed payload storage, promotion transactions, current-generation
selection, parent rollback, and digest-checked engram promotion exist.

The typed generation API now provides a single trained-engram promotion entry
point, and the lifecycle test covers training, promotion, retrieval, CPU
application, replay, and rollback. Production compiler-service wiring and
runtime execution on Metal remain separate work.

## Validation gates

The current evidence is strong for contracts and local mechanics: the
prism-backend build checks, NF4 GPU/CPU residency test, engram unit tests,
evolution unit tests, ternarization unit tests, and generation promotion tests
pass. The full library run also exposed and fixed a zero-duration stub warmup
receipt. The remaining tests do not yet prove Metal engram application,
production-service orchestration, or generalized multi-format search.

The completion gate is one end-to-end lifecycle:

```text
base generation
  -> train engram from dataset
  -> search representation and kernel
  -> compile and dispatch candidate
  -> CPU/oracle, holdout, latency, and device gates
  -> store content-addressed payloads
  -> promote generation
  -> replay and rollback
```
