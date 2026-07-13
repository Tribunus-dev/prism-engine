# Remaining work

> Current snapshot: `b2e89ab` plus the uncommitted evaluator work. This document
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

Remaining work is to route every production kernel creation through the
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
budgets, replay, and Pareto structures. The current Metal evaluator can compile
source and validate static dimensions, but its numerical and performance
receipts are nominal. Joint search still uses a hand-written cost function.

The next required implementation is one shared evaluator that accepts an
executable workload fixture, produces CPU-oracle output, compiles and dispatches
the Metal candidate, performs warm-up and repeated timing, validates numerical
error, records device limits and receipts, and only permits promotion after
correctness and performance gates pass. Engram training must use the same
oracle and holdout machinery.

## Engram training and runtime

Engram contracts, scheduler operations, payload segments, lookup receipts, CPU
additive and multiplicative application, and generation bindings exist.

The trainer still serializes calibration metadata instead of optimizing the
declared `EngramTrainingDataset`. Lookup similarity is currently a constant,
latency is fabricated, Metal application is unimplemented, and low-rank,
latent-prefix, and adapter application modes are placeholders.

Remaining work is a real additive-residual or low-rank optimizer, query/payload
similarity, holdout and interference evaluation, executable payload validation,
Metal application, and an end-to-end test that stores the payload and promotes
the resulting generation.

## Ternarization and assimilation

Ternary candidates, reconstruction gates, residual encoders, assimilation
receipts, and generation-level payload types exist and have unit coverage.

The scale optimizer does not currently update its candidate scale during its
iteration loop. Ternary packaging stores one byte per weight rather than a
native packed representation. Assimilation comparisons can be lossless because
the full dense residual is retained, so strategy fitness does not yet measure a
real storage or execution tradeoff.

Remaining work is a real scale/threshold optimizer, native packing, residual
policy enforcement, executable reconstruction, and replay validation through
the shared evaluator.

## Generations and promotion

Content-addressed payload storage, promotion transactions, current-generation
selection, and parent rollback exist.

Generation listing is still a placeholder, rollback does not yet validate the
target in the store, and no complete train/search/validate/promote/replay flow
exists.

## Validation gates

The current evidence is strong for contracts and local mechanics: the
prism-backend build checks, NF4 GPU/CPU residency test, engram unit tests,
evolution unit tests, and ternarization unit tests pass. These tests do not yet
prove measured search, real engram training, or generation promotion.

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
