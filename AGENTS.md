# AGENTS.md

Prism Engine is a Rust workspace for a constitutional ECS architecture: native compute-image runtime and governed inference kernel for Apple Silicon. The prime directive is one canonical reality — every state-bearing change is validated, transactional, replayable, attributable, and resistant to stale external outcomes.

This file is the agent-facing project definition. It is short on purpose. The detailed rules live in the **`prism-constitutional-rust-ecs`** skill (see "Skills" below). Load that skill before any non-trivial change. The skill's references (`architecture-map.md`, `module-discipline.md`, `rust-quality.md`, `project-absorption.md`, `implementation-workflow.md`, `review-gates.md`) are required reading at the right phase.

## Setup commands

- Install / build: `cargo build` (workspace)
- Build release: `cargo build --release`
- Test: `cargo test --workspace`
- Lint: `cargo clippy --workspace --all-targets`
- Format: `cargo fmt --all -- --check` and `cargo fmt --all`
- Strict lint (CI gate, currently fails by design — see "Lint baseline" below):
  `cargo clippy --workspace --all-targets -- -D warnings`
- Authority + module-cohesion audit: `bash $SKILL_DIR/scripts/audit_authority.sh .`
  (the script ships with the skill; `--module-cohesion` reports only the file-size violations)

## Project layout

- `crates/prism-ecs-core` — entity, world, storage, identity primitives. Domain-neutral. Lowest layer; `unsafe` allowed.
- `crates/prism-ecs-constitutional` — schemas, typed commands, lifecycle, transactions, durable events, replay semantics, authority-bearing state transitions.
- `crates/prism-ecs-runtime` — provider-neutral runtime kernel: schedule, command handling, admission, dispatch coordination, ports, receipts.
- `crates/prism-ecs-kernel` — compiled-kernel contracts, backend interfaces, target-independent kernel ABI.
- `crates/prism-ecs-{compile,quantization,ir,artifact,spatial-ir,server,protocol,...}` — compiler, quantization, IR, artifact, server, product crates. They must not become alternate runtime authorities.
- `crates/prism-{cuda,rocm,metal,amd-npu,ane,intel-npu,tt,igc}-runtime` — hardware backends. The name is the public target contract.
- `crates/prism-{onnx,pytorch}-ingest`, `crates/prism-gguf` — format adapters. The name is the public format contract.
- `compute-core.legacy/` — archaeology. Treat as legacy unless the canonical path explicitly imports it.
- `CAMPAIGN.md` — subsystem migration state. Read this first; it tells you which subsystems are `Shadow`, `Canonical`, or `LegacyRemoved`.
- `clippy.toml` — strict clippy config (thresholds, disallowed methods). Wired through `[workspace.lints.*]` in root `Cargo.toml` at `warn` level to keep the build passing during migration.
- `PrismAgent/`, `PrismAgentiOS/`, `PrismMenuBar/`, `Sources/`, `deno-dashboard/`, `examples/`, `docs/` — product surfaces and long-form docs. Ingress and projection layers; do not own domain truth.

## Skills

The detailed rules for any state-bearing, ECS-touching, or module/quality-affecting change are in the `prism-constitutional-rust-ecs` skill. It is installed at the user level (alongside `github`, `mini-coder-max`, `ui-ux-*`). Load it before any non-trivial change.

The skill is opinionated and stricter than generic Rust/ECS guidance. Read at minimum:
- `SKILL.md` (prime directive, canonical change flow, non-negotiable invariants, crate boundaries)
- `references/architecture-map.md` (when choosing crate ownership or tracing a request across boundaries)
- `references/module-discipline.md` (when creating or significantly modifying a file)
- `references/rust-quality.md` (when adding a public API or reviewing a change)
- `references/project-absorption.md` (when adding a new file that learns from an external project)
- `references/implementation-workflow.md` (the 8-phase workflow, with the Phase 1.5 module authority inventory)
- `references/review-gates.md` (before claiming a change is complete)

## Architecture in one screen

The canonical change flow: typed command → idempotency and admission → preflight → `WorldTxn` staging → `World::transit` atomic commit → durable domain event → `EventStore` → replay applier → projection rebuild → read path → downstream consumer.

External work is an effect, not a world mutation. Backends execute immutable descriptors and return outcomes; they do not decide canonical lifecycle transitions. The crate dependency direction flows downward: higher-authority crates (`prism-ecs-constitutional`) depend on lower-authority (`prism-ecs-core`); the reverse is forbidden. A product crate must not import a backend crate; a backend crate must not import a product crate.

## The hard rules

A change that violates these fails review. Full reasoning, exceptions, and waiver shape are in the skill.

- **No direct world mutation outside `prism-ecs-core` and `WorldTxn` implementations.** `world.spawn`, `world.add_component`, `world.remove_component`, `get_component_mut`, and direct `component_store` access are forbidden in constitutional, runtime, kernel, compile, server, protocol, and product crates. Use `WorldTxn` and the transit boundary.
- **No new manager, registry, service singleton, global map, or database table** that decides canonical state outside the world. Same for `common.rs`, `utils.rs`, `helpers.rs`, `misc.rs`, `shared.rs`, `manager.rs`, `coordinator.rs`, `controller.rs`, `service.rs`, `facade.rs`, and any `mod.rs` over 200 LOC.
- **No `unsafe` in constitutional, runtime, server, or protocol crates.** `unsafe` only in `prism-ecs-core`, `prism-ecs-kernel`, and hardware crates, with a `// SAFETY: <invariant>` comment naming the type or test that proves the invariant.
- **No `unwrap`, `expect`, `panic!`, `unreachable!`, `todo!`, `unimplemented!` in production paths.** Default error path is `?` or `match`. Waivers are scoped, documented with `// WAIVER: <reason>`, and listed in the change's `Completion report`.
- **No `anyhow::Error` in `prism-ecs-constitutional`, `prism-ecs-runtime`, or `prism-ecs-kernel`.** Per-crate error enums with `thiserror` derives, categorized as `Rejected` (preflight), `Failed` (effect), or `Stale` (fencing mismatch).
- **No `HashMap`/`HashSet` for canonical collections whose order is observable.** Use `BTreeMap`, `IndexMap`, `BTreeSet`.
- **No `String`, `u64`, `Uuid` in constitutional APIs where the value is authority-bearing** (IdempotencyKey, Generation, Epoch, LeaseToken, ArtifactDigest, SchemaKey, CommandId). Newtype them — `if the type doesn't say what it is, the API is wrong.`
- **No file named after an external project** (tinygrad, burn, candle, jax, bonsai, uop, etc.) unless it is a format adapter, hardware backend, or vendored dependency. Absorbed patterns must be re-implemented in Prism's domain and named for what they do here.
- **Every new `.rs` file states a single authority in its module doc, in one sentence.** If the sentence cannot be written, the file owns more than one authority and must be decomposed first.
- **A constitutional change that does not propagate is not a change.** For every state-bearing change, name the propagation chain (durable event → event store → replay applier → projection rebuild → read path → consumer) and include a projection-rebuild test and a replay test.

## Workflow expectations

- **Read `CAMPAIGN.md` before editing** any subsystem. The migration state tells you what is canonical, what is shadow, and what is removed. Respect the cutover protocol.
- **Read the smallest authoritative set before editing.** Authority order: current executable tests → `CAMPAIGN.md` status → accepted ADRs → crate-level docs and manifests → root `README.md` → legacy code.
- **Phase 1.5 of the implementation workflow is the module authority inventory.** Before writing code, list the files you will create or modify, the single authority each new file will own, the proof that each modified file is the right home, and the propagation chain. A change that skips this has not specified the change.
- **The `Completion report` is mandatory.** State the affected subsystem, its `CAMPAIGN.md` status, the canonical authority before and after, every remaining writer, the transaction and effect boundaries, durable and transient schema changes, replay behavior, tests executed, authority-leak audit results, and any legacy path still awaiting purge.
- **The review gates are non-skippable.** Authority, Module cohesion, Rust quality, Project absorption, Transaction, Schema, Effect, Event and evidence, Schedule, Idempotency and concurrency, Replay and recovery, Artifact and backend, Test, Propagation, Review output.

## Lint baseline

`clippy.toml` is installed at the project root. The strict disallowed-methods rule (`Result::unwrap`, `Option::unwrap`, `Option::expect`, `Result::expect`, `OnceLock`) is enforced. Current baseline under default features, all targets:

- ~89 unwrap/expect calls in production paths. Top hot files: `prism-mcp-core/src/protocol.rs` (33), `crates/prism-ecs-core/src/world.rs` (11), `crates/prism-gguf/src/lib.rs` (9), `crates/prism-plugin/src/lib.rs` (8).
- 128 total clippy warnings, 3 errors (pre-existing `not_unsafe_ptr_arg_deref` in `prism-plugin`, not from this config).

The `[workspace.lints.clippy]` keeps `all` / `pedantic` / `nursery` at `warn` to keep the build passing while the migration backlog is cleared. New code must not add to the backlog. Flipping any of the existing warns to deny is a constitutional change and must be reviewed.

## Testing instructions

- Unit + integration tests: `cargo test --workspace` (filter with `-p <crate>` or `test_name` for focused runs)
- Doc tests: `cargo test --doc`
- Adversarial tests are part of the implementation protocol — write them before declaring a change complete, not as a follow-up
- Test names describe the invariant (`stale_fencing_generation_rejected`), not the function (`test_kernel_4`)
- Tests use the same constitutional commands and transactions as production. A test that calls `world.spawn` with `set_direct_mutation_allowed(true)` is a legacy test and must be migrated
- Run the focused crate's tests after each layer; run the affected crate's complete suite before completion

## PR & commit conventions

- Branch from `main`; never push to it directly
- Commit message: conventional commits (`feat:` / `fix:` / `docs:` / `refactor:` / `chore:`). Reference the affected subsystem and its `CAMPAIGN.md` status in the body
- Every PR must include the `Completion report` (see "Workflow expectations")
- Every PR that touches constitutional code must pass the Module cohesion gate, the Rust quality gate, the Project absorption gate, and the Propagation gate
- Every PR must include a propagation test (projection rebuild + replay) for any state-bearing change
- Run `cargo fmt --all` and `cargo clippy --workspace --all-targets` before opening the PR

## Security

- Never commit secrets. `.env` is in `.gitignore`.
- Authority-bearing values (keys, tokens, digests) are typed newtypes, not raw strings. Do not log them in cleartext. The `tracing` crate is the structured logger; use it with redacted fields where appropriate.
- Hardware handles, file descriptors, locks, and process-local channels are execution-plane state. They must not be persisted as durable components or events. They must not appear in error messages that propagate beyond the runtime boundary.
- Backend results are untrusted. Validate identity, fencing generation, deadline, binding ABI, numerical policy, and artifact digest before treating a result as canonical.
