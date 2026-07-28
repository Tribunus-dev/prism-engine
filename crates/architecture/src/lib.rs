//! Workspace-level architecture tests.
//!
//! These tests enforce constitutional rules that span multiple
//! crates. The runtime and kernel crates cannot enforce rules about
//! their own callers; only a workspace-level test can check
//! "no file in the workspace imports the legacy engine scheduling
//! surface", the legacy engine assistant_graph surface, the legacy
//! engine evaluator surface, the legacy engine evolution surface,
//! the legacy engine bitnet surface, the legacy engine LUT
//! surface, the legacy engine models surface, the legacy engine
//! system surface, or the legacy engine backend surface.

pub mod workspace_legacy_assistant_graph_imports;
pub mod workspace_legacy_backend_imports;
pub mod workspace_legacy_bitnet_imports;
pub mod workspace_legacy_evaluator_imports;
pub mod workspace_legacy_evolution_imports;
pub mod workspace_legacy_imports;
pub mod workspace_legacy_lut_imports;
pub mod workspace_legacy_models_imports;
pub mod workspace_legacy_system_imports;
