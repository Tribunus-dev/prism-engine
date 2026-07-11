# Remaining migration work

## PR G — Consolidate source implementations (Metal shader level)

Status: Infrastructure exists, shader-level consolidation pending.

### Done
- `MetalBackendCompiler` with `MetalImplementationCatalogue` exists
- All kernel implementations register through the catalogue
- ABI codegen helpers exist (buffer constants, geometry, validation)

### Remaining
1. **Extract shared Metal semantic fragments** — The megakernel (`gemma4_full.metal`, 2972 lines) independently implements NF4 decode, ternary decode, RMSNorm, RoPE, attention, SwiGLU, and KV access. The template shaders in `templates/` implement the same operations independently. Extract these into canonical `#include` fragments and have both the megakernel and template-generated shaders use them.

2. **Remove duplicate NF4/ternary math** — `cimage_linear_nf4.metal`, `nf4_tile640_gemv.metal`, and the megakernel's NF4 decode all implement NF4 dequantization. Consolidate into one canonical implementation.

3. **Replace duplicate template families** — `metal_codegen.rs` generates ad-hoc Rust string templates for QKV proj, attention, etc. Replace with invocations of canonical Metal fragments or a typed Metal AST emitter.

4. **Assemble megakernel from shared fragments** — Once fragments exist, the megakernel source can be assembled from them rather than being one 3000-line monolith.

### Gate
"Handwritten and generated kernels use the same semantic catalogue" — TRUE (all in MetalImplementationCatalogue). "No duplicate NF4/ternary math" — FALSE, still duplicated.

---

## PR H — Product integration (end-to-end)

Status: PrismCompiler API exists, routing and event integration pending.

### Done
- `PrismCompiler` public API with `inspect`, `plan`, `compile` methods
- `ModelFrontend` trait for source ingestion
- `CompileEvent` types defined
- Pipeline adapted to emit canonical types

### Remaining
1. **Route GGUF-to-cimage through PrismCompiler** — The existing `compile_gguf_with_authority()` in `compile/pipeline.rs` should be callable from `PrismCompiler::compile()`.

2. **Wire artifact registration and deployment** — Connect PrismCompiler output to the artifact registry in the constitutional ECS (`artifact.rs`).

3. **Drive server UI from real compiler events** — Replace the current placeholder progress model with the real `CompilerEvent` stream.

4. **Produce joined receipt graph** — Connect compiler receipts to execution receipts for full provenance.

### Gate
"There is one public compile entry point" — TRUE (PrismCompiler). "Compiler events and receipts are emitted from the real pipeline" — STRUCTURAL (types exist, wiring pending).
