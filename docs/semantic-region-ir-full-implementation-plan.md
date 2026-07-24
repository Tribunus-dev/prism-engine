# Semantic Region IR — Full Implementation Roadmap

This roadmap carries the initial compile-verified vertical slice through production integration while preserving the evidence boundary.

## Implemented in this branch

- typed nine-axis genome bookkeeping with `GenomeAxis` and `GenomeAxisSet`;
- persistent `SemanticRegionId` attached to `LogicalTensorId`;
- v0 selectors (`WholeTensor`, `AxisSpan`) with `Rect` serialized but fail-closed;
- origin, role, constraints, descriptor, partition, assignment, and plan types;
- canonical digests and fail-closed verification;
- versioned explicit spec loader;
- deterministic discovery and plan receipts;
- mapped SafeTensors tensor/shape verification demo;
- explicit claim classes for compile verification versus unproven quality and unmeasured performance.

## Milestone 1 — Static semantic discovery

Add graph-explicit and architecture-derived discoverers for split/chunk/slice/concat, fused QKV under MHA/GQA, fused gate/up, and verified MoE boundaries. Unsupported tensors fall back to whole-tensor regions. All inferred boundaries require shape validation and provenance.

## Milestone 2 — Region-aware sensitivity

Add mapped region views, selector-aware cache keys, bounded materialization accounting, calibration digests, and `RegionSensitivityReceipt`. Sensitivity evidence must not fabricate semantic roles.

## Milestone 3 — Hierarchical regional search

Keep `CandidateGenome` global. Add `RegionalCandidate`, bounded representation palettes, region-template sharing across repeated layers, hard region-count budgets, and global quality/memory/latency/transfer constraints.

## Milestone 4 — Layout regularization and physical realization

Add adjacent-region coalescing, logical-to-physical axis mapping, homogeneous packed blocks, buffer assignment, explicit conversions/materializations, and reversible semantic-to-byte provenance in `prism-spatial-ir`.

## Milestone 5 — ComputeImage manifest

Add an optional versioned semantic-region manifest with partitions, plans, physical realizations, receipt references, digest sealing, compatibility tests, and safe ignore behavior for older runtimes.

## Milestone 6 — Backend execution

Integrate one measured backend first. Execute coalesced homogeneous region views, include the region plan digest in the execution fingerprint, report copy/conversion bytes, preserve whole-tensor fallback, and require real measurements before latency/throughput claims.

## Milestone 7 — Scheduler and residency

Schedule coalesced physical views rather than individual semantic regions. Add region-aware residency only where it changes a physical placement or transfer boundary. Keep queue growth bounded independently of raw region count.

## Milestone 8 — Evaluation

Compare uniform per-tensor, fixed group, sensitivity-only, graph-semantic, numerical-block, regularized channel, Prism semantic-only, Prism semantic+sensitivity, and Prism semantic+sensitivity+placement baselines. Measure quality, size, latency, throughput, energy, compile/search time, conversions, fragmentation, kernel count, fallback frequency, and receipt completeness.

## Production definition of done

- semantic identity survives every physical repacking;
- discovery provenance classes remain distinct;
- search is hierarchical and bounded;
- every conversion, reorder, copy, and materialization is represented and costed;
- ComputeImage retains semantic-to-physical provenance;
- execution fingerprints include the region-plan digest;
- measured claims require real execution;
- whole-tensor fallback remains admitted and tested;
- the implementation improves a measured quality/latency/memory frontier rather than only producing a visualization.
