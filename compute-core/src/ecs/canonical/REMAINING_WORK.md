# Remaining migration work

## PR G — Consolidate source implementations (Metal shader level)

Status: Catalogue and backend compiler exist structurally, but are disconnected from real Metal sources. No production path consumes the catalogue.

### Structural (exists, but not consumed)
- `MetalBackendCompiler` with `MetalImplementationCatalogue` exists
- `MetalImplementationCatalogue` registers 12 entries (1 megakernel, 1 per-layer, 9 primitives)
- ABI codegen helpers exist (buffer constants, geometry, validation)
- `MetalBackendCompiler::lower()` and `compile()` exist structurally

### Gaps — Declared vs Actual

| Claimed state | Actual state | Evidence |
|---|---|---|
| "All kernel implementations register through the catalogue" | FALSE. The catalogue is descriptive — 12 registrations exist but NO production path (megakernel, template, runtime, build-script metallib) consumes it. The ~50 `.metal` sources and their host dispatch paths are independently wired. | `catalogue.rs:75-83` (empty ABI buffers/constants), `compiler.rs:113` (empty source), `kernel_dispatch.rs` (references templates directly), `kernel_registry.rs` (build-time metallib env), `region_runner.rs:307-327` (MTLDevice compile) |
| "All ABIs are populated" | STRUCTURAL but empty. The megakernel, per-layer, and all 9 primitive registrations have empty `buffers`, `constants`, and `threadgroup_memory` vectors. | `catalogue.rs:77-79`, `89-91`, `129-131` (empty ABI fields per registration) |
| "No duplicate NF4/ternary math" | FALSE. `cimage_linear_nf4.metal`, `nf4_tile640_gemv.metal`, and the 2972-line megakernel all independently implement NF4 dequantization. | Audit of `.metal` sources — three independent NF4 decode implementations. |

### Remaining
1. **Wire catalogue into production paths** — Replace the three independent NF4 dequantization implementations with a single canonical fragment, then have every consumer include it.
2. **Populate ABIs** — Every catalogue registration must declare its real buffer bindings, function constants, and threadgroup memory allocations matching the actual `.metal` source.
3. **Replace empty source in lower()** — `MetalBackendCompiler::lower()` currently produces `String::new()`. Wire it to a source provider that reads from the catalogue's registered implementation.
4. **Reject empty compilation** — `MetalBackendCompiler::compile()` currently treats empty source as valid, returning a structural artifact. After source providers exist, this must fail.

### Gate
"Every production Metal entrypoint resolves to exactly one catalogue registration; source, entrypoint, ABI, and artifact digest are nonempty and validated."

---

## PR H — Product integration (end-to-end)

Status: `PrismCompiler` API exists structurally, but is disconnected from the real GGUF pipeline. No frontend is registered by default, no production path calls it.

### Structural (exists, but not wired)
- `PrismCompiler` public API with `inspect`, `plan`, `compile` methods — all three
- `ModelFrontend` trait for source ingestion — trait defined, zero implementations
- `CompilePlan`, `CompileOutcome`, `CimageBuildInput`, `CompilerReceipt`, `CompilerStage` types defined
- `compile_gguf_to_canonical()` adapter exists (runs legacy pipeline, wraps result in canonical types)

### Gaps — Declared vs Actual

| Claimed state | Actual state | Evidence |
|---|---|---|
| "One public compile entry point" | FALSE operationally. `prism` binary at lines 759, 775, 793 directly calls `compile_gguf_speculative()`, `compile_gguf_with_authority()`, and `compile_gguf_unchecked()`. `PrismCompiler` is never used. | `prism.rs:754-802` (three direct legacy call sites) |
| "PrismCompiler::compile() exists" | Structurally true, operationally false. `compile()` calls `plan()` then returns a structural `CompileOutcome` with all-zero digest, empty artifacts, no output path. Does not invoke the real GGUF pipeline. | `prism_compiler.rs:143-192` (stub returning empty artifacts) |
| "Default frontend installed" | FALSE. `PrismCompiler::default()` creates empty `frontends: Vec::new()`. No `ModelFrontend` implementation exists anywhere in the workspace. | `prism_compiler.rs:42-47` (empty Vec) |
| "CompileEvent types defined" | FALSE. `CompilerReceipt` and `CompilerStage` exist, but there is no `CompileEvent` type or event stream type. The REMAINING_WORK.md itself correctly calls for "real CompilerEvent stream" in its remaining items. | `compile_plan.rs:77-103` (no CompileEvent type exists anywhere) |
| "Pipeline adapted to emit canonical types" | STRUCTURAL only. `compile_gguf_to_canonical()` builds canonical types AFTER the legacy pipeline runs via an adapter (`pipeline.rs:2252-2278`). Canonical types do not drive compilation; they are a secondary output. | `pipeline.rs:2260-2277` (adapter pattern, canonical types constructed post-hoc) |

### Remaining
1. **Route GGUF-to-cimage through PrismCompiler** — `PrismCompiler::compile()` must detect GGUF source and delegate to `compile_gguf_to_canonical()` or `compile_gguf_with_authority()`. This is the critical path for M1.
2. **Install default GGUF frontend** — `PrismCompiler::default()` must register a GGUF frontend so that `inspect()` and `plan()` work without explicit setup.
3. **Wire authority parameters through compile path** — Extend `CompileRequest` to carry authority, quant mode, hardware target, and optional compiler paths currently passed as separate CLI args.
4. **Route CLI binary through canonical API** — Replace the three direct legacy calls in `prism.rs` with a single `PrismCompiler::compile()` call.
5. **Produce joined receipt graph** — Connect compiler receipts to execution receipts for full provenance. Currently `CompilerReceipt` lacks artifact digest, source digest, policy digest, compiler/toolchain version, kernel implementation identity, parent receipt IDs, timestamps, or failure detail.
6. **Drive server UI from real compiler events** — Define `CompileEvent` type and event stream first, then wire UI consumption.

### Gate
"One GGUF fixture produces a nonempty cimage through PrismCompiler; direct binary calls to unchecked compilation are absent; legacy-vs-canonical manifests match."
