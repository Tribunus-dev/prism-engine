# ADR-031: AITER/ATOM-Informed ROCm Provider and Serving Boundaries

## Status

Proposed — research captured; implementation sequence approved for prototyping.

## Context

AMD’s current inference stack separates serving/runtime orchestration from
vendor-optimized kernels. [AITER](https://github.com/ROCm/aiter) provides the
ROCm-oriented tensor engine and optimized kernels, while
[ATOM](https://github.com/ROCm/ATOM) is a lightweight vLLM-like inference and
serving engine built around AITER. ATOM documents support for OpenAI-compatible
serving, piecewise compilation and CUDA graphs, KV-cache policies, speculative
decoding, tensor/data/expert parallelism, and compute/communication overlap.

This separation is compatible with Prism’s intended architecture, but the
ownership boundaries must remain explicit. Prism owns model ingestion,
PhaseIR/Spatial IR semantics, representation search, hardware evidence,
artifact sealing, validation, and portable execution contracts. A ROCm backend
may delegate individual operations to AITER and may interoperate with ATOM,
but neither dependency becomes the authority for Prism model identity,
representation plans, residency, receipts, or lifecycle state.

The relevant AMD stack is therefore:

```text
Prism model/compiler/ECS authority
    -> ROCm provider contract
    -> AITER kernel or ATOM execution adapter
    -> HIP/ROCm runtime
    -> AMD device
```

ATOM’s ecosystem-compatible integration is also instructive: its documented
vLLM plugin path shows that an optimized provider can be adopted without
forcing every serving client to change APIs. Prism should preserve the same
principle through stable artifact and serving adapters.

## Decision

Prism will implement a native Rust ECS ROCm provider with three distinct
interfaces.

| Prism boundary | Responsibility | AMD integration |
|---|---|---|
| `RocmKernelProvider` | Compile, load, bind, and dispatch one admitted kernel or fused operation | HIP, rocBLAS/Composable Kernel, or AITER |
| `RocmExecutionProvider` | Materialize immutable dispatch descriptors and return measured outcomes | HIP streams, events, AITER wrappers, optional ATOM bridge |
| `RocmServingAdapter` | Translate Prism runtime requests to an external OpenAI-compatible serving endpoint when selected | ATOM native server or vLLM-ATOM plugin |

The provider must be capability-driven. AITER support is represented as a
kernel-family capability, not as a universal assumption. The provider reports
which operations, dtypes, layouts, attention variants, MoE paths, communication
primitives, and device architectures it can execute. Search may compare an
AITER-backed candidate against Prism-generated HIP, CPU, or other provider
candidates.

The ECS remains authoritative. The minimum native components are:

| Component | Meaning |
|---|---|
| `RocmDeviceIdentity` | Stable device, GFX architecture, ROCm/runtime, driver, and topology identity |
| `RocmCapabilitySet` | Measured and declared kernel-family, dtype, memory, stream, and communication capabilities |
| `KernelProviderBinding` | Selected provider, AITER operation/family, ABI, source/IR digest, and toolchain identity |
| `RocmDispatchDescriptor` | Immutable buffers, shapes, strides, streams, synchronization, residency, and workload scenario |
| `RocmMeasurementReceipt` | Device timing, feasibility, correctness result, counters, and calibration identity |
| `ParallelExecutionPlan` | Tensor/data/expert parallel routes, collective operations, ranks, and communication evidence |
| `KvTransferPlan` | Prefill/decode ownership, page codec, transport, source/destination identity, and transfer receipt |
| `ServingAdapterBinding` | ATOM/vLLM endpoint, protocol, model identity, request mapping, and compatibility contract |

The compiler may choose AITER-backed operations, but the sealed ComputeImage
must record the exact provider binding and fallback. A runtime must not silently
replace an AITER candidate with another kernel after promotion; replacement is
a new candidate requiring evidence and a new artifact digest.

## What Prism should implement

The first implementation should focus on provider contracts, not a full AITER
rewrite. Prism should add a ROCm provider registry, capability probing,
operation-family dispatch descriptors, AITER-backed candidate bindings for
GEMM/attention/MoE/norm, and measured replay receipts. The initial families
should be selected from the target model and hardware rather than attempting
to expose every AITER operation at once.

Prism should also adopt ATOM’s workload dimensions as explicit search and
runtime scenarios: prefill versus decode, batch size, concurrency, KV-cache
format, speculative-decoding mode, and parallelism topology. These scenarios
must be represented in `WorkloadScenario` and linked to candidate measurements,
not hidden in provider-specific configuration.

For distributed execution, Prism should model tensor, data, and expert
parallelism as execution-plan entities and retain collective communication in
the same evidence chain as compute. ATOM/MORI/RCCL integration belongs behind
the provider boundary. Prism owns the plan, placement, residency, and receipt;
the provider owns the physical collective implementation.

ATOM should initially be integrated in two modes. The native serving adapter
can target ATOM’s OpenAI-compatible server for deployment experiments, while a
provider mode can consume ATOM/vLLM integration points when Prism is acting as
the compiler/artifact authority. Both modes must preserve Prism’s model and
artifact identities and must not make an external server the source of truth.

## What Prism should not implement initially

Prism should not duplicate AITER’s assembly, Composable Kernel, or Triton
kernel portfolio. It should not absorb ATOM’s Python serving runtime, replace
its request scheduler wholesale, or couple `PhaseIR` and `Spatial IR` to ATOM
module internals. It should not mark a candidate measured merely because an
AITER symbol resolved; execution, correctness, and workload evidence are still
required.

## ECS lifecycle

The ROCm path follows the same lifecycle as every Prism backend:

```text
Device discovery
  -> capability calibration
  -> candidate binding
  -> dispatch measurement
  -> behavioral validation
  -> plan admission
  -> CImage sealing
  -> runtime replay/certification
```

Each stage reads the previous stage’s components and publishes its own
validated component. A missing provider, unsupported AITER family, stale
device identity, absent communication receipt, or mismatched artifact digest
fails closed.

## Consequences

### Positive

Prism gains access to AMD’s optimized kernel ecosystem without surrendering
portable compiler semantics or evidence ownership. AITER-backed and generated
HIP candidates can be compared by the same evolutionary machinery. ATOM can be
used as a serving reference or deployment adapter while Prism remains useful
for Metal, CUDA, Level Zero, Core ML, and other providers.

### Negative

The provider boundary must model more than kernel names: ABI, architecture,
layout, stream behavior, communication, KV state, and workload shape all affect
validity. ROCm qualification will require Linux/AMD hardware or an equivalent
remote test environment. External ATOM integration introduces version and
deployment compatibility that must be recorded as provenance.

## Implementation sequence

| Phase | Deliverable | Gate |
|---|---|---|
| 1 | Rust ROCm provider traits, capability schema, and ECS components | Unit tests prove stable identity, capability matching, and fail-closed admission |
| 2 | AITER kernel-family bindings for one GEMM, attention, and MoE path | Real device measurements and behavioral receipts are embedded in a CImage |
| 3 | Per-scenario candidate search across generated HIP and AITER-backed kernels | Search selects by workload scenario and preserves fallback candidates |
| 4 | Tensor/data/expert parallel execution-plan entities and RCCL/MORI adapter boundary | Multi-device replay validates placement and communication receipts |
| 5 | ATOM serving adapter and vLLM-compatible deployment path | OpenAI-compatible requests preserve Prism artifact/model identity |
| 6 | MI300X/MI355X qualification and DeepSeek/MoE recipes | End-to-end prefill/decode, KV, speculative, and batch-size evidence is sealed |

## References

- [ROCm AITER repository](https://github.com/ROCm/aiter)
- [ROCm ATOM repository](https://github.com/ROCm/ATOM)
- [AITER overview](https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html)
- [ATOM documentation](https://rocm.github.io/ATOM/docs/)
- [ATOM distributed inference guide](https://rocm.github.io/ATOM/docs/distributed_guide.html)
- [ATOM serving and benchmark documentation](https://github.com/ROCm/ATOM#profiling--trace-analysis)
