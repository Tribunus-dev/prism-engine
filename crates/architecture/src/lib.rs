//! Workspace-level architecture tests.
//!
//! These tests enforce constitutional rules that span multiple
//! crates. The runtime and kernel crates cannot enforce rules about
//! their own callers; only a workspace-level test can check
//! "no file in the workspace imports the legacy engine scheduling
//! surface", the legacy engine assistant_graph surface, the legacy
//! engine canonical surface, the legacy engine compiler surface,
//! the legacy engine compilation surface, the legacy engine
//! compute_image_compile surface, the legacy engine core surface,
//! the legacy engine evaluator surface, the legacy engine evolution
//! surface, the legacy engine bitnet surface, the legacy engine LUT
//! surface, the legacy engine nf4tile640 surface, the legacy engine
//! kv_arena surface, the legacy engine memory surface, the legacy
//! engine cimage surface, the legacy engine config surface, the legacy
//! engine models surface, the legacy engine runtime surface, the
//! legacy engine system surface, the legacy engine decode_attribution
//! surface, the legacy engine tools surface, the legacy engine
//! compute_image_core surface, the legacy engine compute_image_compile
//! surface, the legacy engine compute_image_runtime surface, or the
//! legacy engine backend surface.

pub mod workspace_legacy_assistant_graph_imports;
pub mod workspace_legacy_ane_imports;
pub mod workspace_legacy_backend_imports;
pub mod workspace_legacy_bitnet_imports;
pub mod workspace_legacy_canonical_imports;
pub mod workspace_legacy_cimage_imports;
pub mod workspace_legacy_compilation_imports;
pub mod workspace_legacy_compute_image_core_imports;
pub mod workspace_legacy_compute_image_runtime_imports;
pub mod workspace_legacy_compiler_imports;
pub mod workspace_legacy_compute_image_compile_imports;
pub mod workspace_legacy_config_imports;
pub mod workspace_legacy_core_imports;
pub mod workspace_legacy_decode_attribution_imports;
pub mod workspace_legacy_evaluator_imports;
pub mod workspace_legacy_evolution_imports;
pub mod workspace_legacy_imports;
pub mod workspace_legacy_kv_arena_imports;
pub mod workspace_legacy_lut_imports;
pub mod workspace_legacy_memory_imports;
pub mod workspace_legacy_models_imports;
pub mod workspace_legacy_nf4tile640_imports;
pub mod workspace_legacy_runtime_imports;
pub mod workspace_legacy_system_imports;
pub mod workspace_legacy_tools_imports;
