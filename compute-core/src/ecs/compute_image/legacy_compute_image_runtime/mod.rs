//! Legacy engine-internal `compute_image_runtime` surface — the home
//! of the engine-coupled implementations of the runtime + ancillary
//! subsystem that the constitutional crate
//! (`prism_ecs_compile::compute_image_runtime`) cannot depend on.
//!
//! On 2026-07-27, the engine's
//! `compute-core/src/ecs/compute_image/{residency,...,verification}/`
//! directory was renamed to this
//! `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/`
//! directory as part of the constitutional absorption. Data-only
//! types are re-exported from
//! `prism_ecs_compile::compute_image_runtime`. Engine-coupled
//! implementations stay here.

// All 12 subdirs that were originally `pub mod` in
// `compute-core/src/ecs/compute_image/mod.rs` are re-exposed as
// unconditional `pub mod` here so that sibling engine modules
// (e.g. `orchestrator/`) that reference them via
// `crate::ecs::compute_image::multimodal::X` still resolve through
// the legacy path. The engine-internal files retain their own
// feature / OS cfg-gates where required.
pub mod residency;
pub mod heterogeneous;
pub mod megakernel;
pub mod kernel_selection;
pub mod multimodal;
pub mod model_family;
pub mod variants;
pub mod program;
pub mod content_store;
pub mod executable;
pub mod scheduler;
pub mod verification;
