# PRISM-IMAGE-GOLDEN-PATH-0001 Implementation Plan

## Architecture

New files in `prism-engine/src/image/`:
- `types.rs` — all public types (ImageGenerationRequest, ImageGenerationResult, GeneratedImage, ImageGenerationReceipt, all enums, ImageGenerationError)
- `manifest.rs` — CImage modality manifest types
- `provider.rs` — Prism-level ImageGenerationProvider trait + concrete providers
- `admission.rs` — ImageGenerationAdmissionGate
- `router.rs` — deterministic route selector

Files modified:
- `mod.rs` — rewrite as assembly module, re-exporting public API, wiring generate_image()
- `scheduler_registry.rs` — untouched (remains as stubs)
- `diffusion_gemma` inline mod — kept

## Module dependency graph

```
types.rs (no deps)
  └──> manifest.rs (depends on types)
  └──> provider.rs (depends on types)
  └──> admission.rs (depends on types, manifest)
  └──> router.rs (depends on types, manifest, admission)
  └──> mod.rs (depends on all, wires generate_image)
```

## Key design decisions

- `ImageGenerationRequest` replaces old `ImageParams` — richer field set from spec
- `ImageProviderKind` replaces old `ProviderKind` — adds `Unavailable` variant
- `RouteOrigin` is new (ExplicitRequest, AutoSelection, QualifiedFallback, DryRun)
- `ImageGenerationProvider` trait is Prism-owned, NOT a re-export of compute-core's trait
- `ComputeCoreMlxImageProvider` is private to provider.rs — wraps compute-core's TextToImageProvider
- `PrismLutImageProvider` returns typed unavailable
- `FakeImageProvider` is #[cfg(test)] — returns deterministic 2x2 RGBA fixture
- `ImageGenerationAdmissionGate` validates request against CImage manifest before provider dispatch
- `select_provider()` implements the spec's route matrix
- `generate_image()` is the single public entry point — always available, returns MissingFeature error when feature disabled
- `RequestId` = uuid::Uuid, `OutputDigest` = String (blake3 hex), `ArtifactDigest` = String
- `MemoryResidency` = Cpu, UnifiedGpu, DiscreteGpu, Unknown
- `QualificationStatus` = Accepted, Unqualified, Declined(String)
- `GenerationWarning` = String

## Type signatures (from spec, verbatim where possible)

See spec document for exact field signatures. Key types to implement:

```rust
pub struct ImageGenerationRequest { prompt, negative_prompt, width, height, steps, seed, guidance_scale, output_format, device_preference, execution_policy }
pub enum ImageOutputFormat { Rgba8, Png }
pub enum DevicePreference { Auto, ComputeCoreMlx, PrismLut }
pub enum GenerationExecutionPolicy { RequireRequestedProvider, AllowQualifiedFallback, DryRunAdmission }
pub struct ImageGenerationResult { image, receipt }
pub struct GeneratedImage { width, height, format, bytes, digest }
pub struct ImageGenerationReceipt { request_id, model_digest, cimage_digest, requested_provider, selected_provider, route_origin, provider_version, qualification_status, fallback_used, input_tokens, denoising_steps_requested, denoising_steps_completed, width, height, output_format, output_digest, total_latency_ms, provider_latency_ms, materialization, warnings }
pub enum ImageProviderKind { ComputeCoreMlx, PrismLut, Unavailable }
pub enum RouteOrigin { ExplicitRequest, AutoSelection, QualifiedFallback, DryRun }
pub enum ImageGenerationError { FeatureUnavailable, ArtifactNotImageCapable, MissingComponent, ArtifactUnqualified, RequestedProviderUnavailable, ProviderExecutionFailed, InvalidOutput, UnsupportedRequest, AdmissionRefused }
pub struct MaterializationReceipt { provider_output_residency, prism_output_residency, copies_recorded, bytes_materialized, zero_copy_claimed, notes }
pub struct ImageGenerationCapabilityManifest { schema_version, model_family, text_encoder, denoiser, vae_decoder, tokenizer, scheduler, safety_or_policy_components, supported_widths, supported_heights, supported_step_range, provider_artifacts, qualification }
pub enum ComponentAvailability { Absent, DeclaredButUnverified, PresentAndQualified, Unsupported, Refused }
pub struct ImageProviderArtifact { provider, artifact_id, compiler_id, abi_version, required_hardware, tensor_layout, qualification_record }
pub struct ImageGenerationAdmissionDecision { admitted, selected_provider, required_components, missing_components, qualification_status, refusal_reason }
pub trait ImageGenerationProvider: Send + Sync { fn kind(); fn capability_report(); fn generate(); }
pub struct ImageGenerationProviderRequest { installed_image, request, machine, execution_id }
pub struct ImageGenerationProviderResult { rgba_bytes, width, height, provider_latency_ms, provider_metadata, materialization }
```

## Contract

- Every new type MUST be defined exactly once
- No re-exporting compute-core types through Prism public API
- generate_image() is the sole public entry point
- FakeImageProvider is #[cfg(test)] only
- Integration tests are opt-in (ignore by default)
- All new files behind #[cfg(feature = "generation-image")]
