//! Workspace-level architecture tests.
//!
//! These tests enforce constitutional rules that span multiple
//! crates. The runtime and kernel crates cannot enforce rules about
//! their own callers; only a workspace-level test can check
//! "no file in the workspace imports the legacy engine scheduling
//! surface", the legacy engine assistant_graph surface, or the legacy
//! engine evaluator surface.

pub mod workspace_legacy_assistant_graph_imports;
pub mod workspace_legacy_compiler_imports;
pub mod workspace_legacy_evaluator_imports;
pub mod workspace_legacy_imports;
pub mod workspace_legacy_models_imports;
