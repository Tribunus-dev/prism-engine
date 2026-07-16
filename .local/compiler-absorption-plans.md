# Compiler Repo Absorption Plans

## Group 3: Compiler Infrastructure

---

### 1. llvm-project (MLIR subdirectory)

**Repo:** https://github.com/llvm/llvm-project  
**Language:** C++ (MLIR subproject: C++ with TableGen/ODS declarative definitions)  
**Purpose:** The full LLVM compiler infrastructure. The relevant subdirectory is `mlir/` — MLIR (Multi-Level Intermediate Representation) is a framework for building, transforming, and lowering compiler IRs. It provides a dialect system (modular IR definitions), pass infrastructure, pattern rewriting, type inference, table-driven op definitions (ODS/TableGen), and lowering pipelines from high-level to low-level IRs (e.g., Tensor → Linalg → LLVM).

**Directory layout (key subdirs under `mlir/`):**
- `include/mlir/` — headers for all MLIR APIs: IR, dialects (arith, func, linalg, scf, tensor, vector, gpu, LLVM, etc.), passes, transforms, tablegen
- `lib/` — implementation of IR, transforms, dialects, target lowering, execution engine
- `tools/` — CLI tools: `mlir-opt`, `mlir-translate`, `mlir-tblgen`, `mlir-reduce`, `mlir-query`, `mlir-pdll`
- `python/` — Python bindings via C API
- `test/` — lit-based test suite
- `docs/` — tutorials (Toy tutorial, Transform dialect tutorial), dialect docs, pass docs

**Capabilities Prism should absorb:**

1. **Dialect system architecture** — MLIR's `Dialect`/`Operation`/`Op` hierarchy is the reference pattern. prism-ecs-ir already models this (ArithDialect, FuncDialect, LinalgDialect, ScfDialect structs). Absorb: the registration/lookup pattern, dialect interface mechanism (DialectInterface), and how operations define their semantics declaratively.

2. **Pass pipeline infrastructure** — MLIR's `Pass`/`PassManager`/`OpPassManager` with nested pass pipelines, pass statistics, verifiers, and pass failure reporting. prism-ecs-ir has a `PassManager` already; absorb the nesting pattern (`nest` on specific ops), pass dependency declarations, and the `run` abstraction.

3. **Pattern rewriting framework** — MLIR's `RewritePattern`/`PatternApplicator`/`PatternRewriter` with benefit-based ordering, DAG matching, and folding infrastructure. prism-ecs-ir has a `RewriteDriver`; absorb the pattern matcher API, match/rewrite separation, and fold interface.

4. **TableGen/ODS for op definitions** — MLIR uses TableGen to generate op C++ code from declarative specs (Op Def Spec). prism-ecs-ir could adopt a lightweight Rust equivalent (a proc-macro or build script) for defining ops with their operands, results, attributes, verifiers, and canonicalization patterns — reducing boilerplate.

5. **Multi-level lowering strategy** — MLIR's core contribution: progressive lowering through dialects (e.g., `tensor` → `linalg` → `scf` → `arith` → `LLVM`). prism-ecs-ir should mirror this with explicit conversion passes between its dialect layers (already visible in the codec → quantization → scheduling pipeline plan).

6. **Type inference and verification infrastructure** — MLIR's type inference (OpFoldResult, InferTypeOpInterface) and verifier framework (custom verifiers per op). prism-ecs-ir has `TypeInference` and `SsaVerifier` tasks; absorb the interface pattern.

7. **Memory effects / side-effect modeling** — MLIR's `MemoryEffectOpInterface` and alias analysis. Prism needs this for scheduling and fusion decisions.

**Integration with prism-ecs-ir:**  
Direct architectural alignment — prism-ecs-ir's IR is already MLIR-inspired. The absorption is primarily **design/pattern absorption**, not code import. Read the relevant MLIR source files for each subsystem, then implement the Rust equivalents in prism-ecs-ir using the same abstractions, adapted to Rust's type system. The `tblgen` build script (`tblgen_crate` task) could generate dialect ops from a custom YAML/TOML schema, analogous to MLIR's TableGen/ODS.

**Key files to study:**  
- `mlir/include/mlir/IR/` — Dialect.h, Operation.h, OpDefinition.h, PatternMatch.h, Verifier.h
- `mlir/include/mlir/Pass/` — Pass.h, PassManager.h, OpPassManager.h
- `mlir/include/mlir/Interfaces/` — InferTypeOpInterface, MemoryEffectOpInterface
- `mlir/lib/Transforms/` — canonicalize, CSE, inlining
- `mlir/tools/mlir-tblgen/` — TableGen op generator

---

### 2. Melior

**Repo:** https://github.com/mlir-rs/melior  
**Language:** Rust (wraps LLVM/MLIR C API via `mlir-sys` crate)  
**Purpose:** Safe Rust bindings for the MLIR C API. Provides a Rust-idiomatic API for building MLIR modules, operations, blocks, regions, types, attributes, and running passes — all as bindings to the real MLIR library installed on the system.

**Structure:**
- `melior/` — main crate: Rust wrapper types (`Context`, `Location`, `Module`, `Operation`, `Block`, `Region`, `Type`, `Attribute`, `Value`, dialects, passes)
- `macro/` — proc-macro support for Melior
- Each MLIR dialect gets a submodule (arith, func, scf, linalg, LLVM, etc.)

**Capabilities Prism should absorb:**

1. **Ownership model for MLIR objects** — Melior uses `&T` everywhere (not `&mut T`) to work around MLIR C API's loose ownership, with `RefCell`-style dynamic checks planned. Prism's pure-Rust IR doesn't need this compromise, but the API surface design (how operations are constructed, how blocks are appended) is directly reusable.

2. **Dialect registration pattern** — `register_all_dialects(&registry)` / `context.load_all_available_dialects()` — a clean registry pattern for making dialects available to IR construction. Adapt for prism-ecs-ir's `DialectRegistry`.

3. **Operation builder API** — Each dialect provides typed builder functions (e.g., `arith::addi(context, lhs, rhs, location)`) that construct the operation with the right operands, results, and attributes. This is the pattern prism-ecs-ir should use for its dialect op builders.

4. **Location tracking** — Melior threads `Location` through all IR construction. prism-ecs-ir's `LocationTracking` task should adopt a similar pattern: every operation has a location for debugging and diagnostics.

5. **Verification integration** — Melior exposes `module.as_operation().verify()`. prism-ecs-ir should expose a uniform verify method on modules/regions that runs registered verifiers.

**Integration with prism-ecs-ir:**  
Melior is the **bridge to real MLIR**. If prism-ecs-ir ever needs to lower to LLVM IR or run on real hardware through MLIR's code generation backends, Melior is the Rust path. Two integration modes:

- **Pure Rust mode (primary):** prism-ecs-ir's dialects and passes are pure Rust, no C++ library dependency. The API design — builder functions, dialect registration, Block/Region/Operation construction — follows Melior's patterns but reimplemented in Rust-native types.

- **Interop mode (optional):** Add a `melior` feature to prism-ecs-ir that can emit MLIR C API calls via Melior. This would let Prism use real MLIR passes (canonicalize, CSE, LLVM lowering) as an escape hatch. A `to_melior()` or `lower_via_melior()` conversion function on prism-ecs-ir IR types.

**Key files to study:**
- `melior/src/ir/` — module.rs, operation.rs, block.rs, region.rs, r#type.rs, attribute.rs, location.rs, value.rs
- `melior/src/dialect/` — arith.rs, func.rs, linalg.rs, scf.rs, LLVM.rs
- `melior/src/utility/` — register_all_dialects.rs, register_all_passes.rs
- `melior/src/pass/` — pass_manager.rs
- `melior/Cargo.toml` — mlir-sys version pinning, features (ods-dialects, helpers)

---

### 3. Laufey

**Repo:** https://github.com/littledivy/laufey  
**Language:** Rust + C (CEF backends), C ABI interface  
**Purpose:** Web embedded framework for cross-platform apps using web technologies. Not a compiler. Uses a C ABI to separate browser engine backends (CEF, WebView, Winit) from user application logic. Provides JS↔native bidirectional marshalling.

**Structure:**
- `capi/` — C ABI definitions (`laufey.h`), Rust C API bindings
- `cef/` — Chromium Embedded Framework backend (C++)
- `webview/` — system WebView backend (macOS WKWebView, Windows WebView2)
- `winit/` — windowing-only backend (no web engine)
- `backend-common/` — shared backend support code
- `examples/` — hello world, native e2e, iOS, DDCore

**Capabilities relevant to Prism:**  
**None as a compiler.** Laufey is a web app runtime, unrelated to compiler infrastructure. The only potentially interesting technique is the **C ABI plugin model** (backend/runtime split via `laufey_backend_api_t` interface table) — this is a clean pattern for plugin systems, but Prism already has better patterns (ECS-based systems, MCP daemon handlers).

**Absorption recommendation: Skip.** No compiler capabilities to absorb. Remove from compiler study list.

---

### 4. Sim (simstudioai/sim)

**Repo:** https://github.com/simstudioai/sim  
**Language:** TypeScript/JavaScript (Next.js, Bun, PostgreSQL)  
**Purpose:** AI agent workspace platform — build, deploy, and manage AI agents and workflows. Not a compiler. Provides agent builder UI, integrations (1,000+), knowledge bases, scheduled tasks, monitoring.

**Structure:**
- `apps/sim/` — main Next.js app
- `packages/` — db (Drizzle ORM), ts-sdk, python-sdk, workflow-types, workflow-renderer, security, testing
- `docker/` — Docker deployment files (Dockerfile with vLLM/Ollama support)

**Capabilities relevant to Prism:**  
**None as a compiler.** Sim is a full-stack web application (agent platform), completely different domain from compiler infrastructure. The only marginal overlap is that PrismAgent (the user-facing product) and Sim both build AI agents, but Sim is a SaaS web app while Prism is a Rust-native coding agent. No compiler patterns to absorb.

**Absorption recommendation: Skip.** Remove from compiler study list.

---

## Summary Table

| Repo | Language | Type | Absorption Priority | What to Absorb |
|------|----------|------|---------------------|----------------|
| **llvm-project (MLIR)** | C++/TableGen | Compiler framework | **HIGH** | Dialect system, pass pipeline, pattern rewriting, TableGen/ODS, multi-level lowering, type inference, memory effects |
| **Melior** | Rust | MLIR Rust bindings | **HIGH** | Ownership model, dialect registration, op builder API, location tracking, verification pattern; interop bridge to real MLIR |
| **Laufey** | Rust/C | Web framework | **SKIP** | Not a compiler |
| **Sim** | TS/Next.js | AI agent platform | **SKIP** | Not a compiler |

## Absorption Order

1. **MLIR core patterns** (llvm-project/mlir) — read dialect system, pass manager, pattern rewriting headers — these are the architectural foundations prism-ecs-ir already mirrors. Document key design decisions.

2. **Melior API surface** — the operation builder pattern and dialect submodule organization are the template for prism-ecs-ir's Rust API. Adapt the `arith::addi(context, lhs, rhs, location)` pattern.

3. **TableGen/ODS study** — design a lightweight Rust equivalent for declaring ops with their operands, results, attributes, verifiers, and canonicalization patterns. This is the `tblgen_crate` task.

4. **Interop path** — optionally, design the `to_melior()` conversion that would allow Prism IR to be lowered through real MLIR passes.
