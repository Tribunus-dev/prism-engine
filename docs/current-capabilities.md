# Prism Engine: current capability map

Updated 2026-07-23. This page describes repository surfaces that are implemented or present as explicit integration boundaries. It does not turn compile-verified code into a claim of production hardware support.

## The short version

Prism is more than a compiler that emits a ComputeImage. The repository contains an evidence-preserving deployment workflow: models can be inspected, candidates can be searched and validated, artifacts can be traced, runtime work can be replayed, experiments can be compared, and hardware-specific execution remains behind explicit provider boundaries.

The current product story has three distinct levels:

| Level | Meaning |
| --- | --- |
| Implemented | The repository contains the code path, data model, or tool surface. |
| Qualifying | The path has tests or validation machinery, but hardware, driver, or end-to-end evidence is still being gathered. |
| Planned or developing | The architecture or ADR exists, but the complete capability should not be presented as shipped. |

## Implemented evidence and operations surfaces

The MCP core provides model inspection, tensor listing and classification, memory estimation, asset validation, benchmark planning, benchmark comparison, regression detection, and baseline promotion. The replay surface can capture, run, minimize, compare, import, and export replay bundles. The lab surface can create, run, cancel, compare, promote, and resume experiments.

These tools are backed by repository data structures rather than being website-only concepts. Provenance models connect model, compilation, evidence, and runtime domains. The work journal records operation phases and reconciles unfinished work at startup. Content-addressed artifacts and file locks provide durable boundaries for work that must survive process interruption.

This is the operational value Prism currently under-communicates: a deployment result is intended to remain inspectable and reproducible after the original compiler invocation has finished.

## Hardware and runtime boundaries

Apple has more than one path in the repository. Metal is the primary public validation story, while `prism-ane` and `prism-ane-runtime` contain Core ML and Apple Neural Engine surfaces including MIL construction, model packaging, stateful requests, lowering, arenas, compilation, and dispatch boundaries.

The repository also contains CUDA runtime surfaces for PTX compilation, launch validation, buffer bindings, dispatch, and timing evidence. The AMD XDNA runtime contains lowering, artifact validation, command buffers, topology probing, buffer management, firmware encoding, submission, and preflight checks. These surfaces should be described as implemented integration boundaries until their target-specific evidence is complete.

The compiler and IR also contain backend-specific paths for CPU, Apple, AMD, Intel, and NVIDIA execution. A backend surface, legal plan, or compiled artifact is not by itself a production support claim; the release boundary remains the measured execution and conformance evidence for that target.

## Model classes beyond text-only deployment

The repository includes a multimodal pipeline with image input handling, dynamic tiling, vision encoders, projectors, image-token strategies, and multimodal forward composition. It also includes an audio pipeline with streaming state, resampling, residual vector quantization, temporal attention, codec abstractions, and speech-generation receipts.

The compiler’s representation and validation model is therefore relevant to vision-language and audio workloads, not only decoder-only text models. These paths deserve separate end-to-end qualification before being presented as finished product workflows.

## Search, representation, and model structure

The compiler contains a multi-objective search surface with Pareto frontiers, archives, mutation proposals, progressive stages, sensitivity, deployment objectives, and evolutionary memory. This means the intended optimization problem is not “choose one quantization level.” It is to find legal candidates across quality, memory, latency, and target constraints.

There are also model-specific and kernel-level surfaces for mixture-of-experts workloads, including Qwen 3.6 MoE compilation paths and expert-related kernels. The website should frame this as expert-aware compiler work, while keeping final model and hardware qualification explicit.

## What is not yet safe to claim as shipped

The resumable ternary distill-compiler is specified in [ADR 001](adr-001-resumable-ternary-distill-compiler.md) as accepted but pending implementation from foundation through end-to-end parity. The repository does contain the surrounding compiler, forensic receipt, replay, and validation infrastructure, but the full overnight resumable ternary workflow remains a development target.

Likewise, the Swift swarm layer contains an agent integration surface, while parts of the inference bridge are still stubs. XDNA, CUDA, ANE, multimodal, and audio code should be presented with their evidence boundary rather than as universally qualified hardware support.

## Where to look in the source

The primary implementation surfaces are `prism-mcp-core`, `prism-mcp-build`, `prism-mcp-replay`, `prism-mcp-lab`, `crates/prism-ecs-compile`, `crates/prism-ecs-ir`, `crates/prism-ane`, `crates/prism-ane-runtime`, `crates/prism-cuda-runtime`, `crates/prism-amd-npu-runtime`, `src/multimodal`, and `src/audio`.
