//! Diagnostic infrastructure for compiler error/warning reporting from ECS passes.
//!
//! Provides [`DiagnosticEngine`] for collecting and printing diagnostics during
//! IR transformation passes, with optional provenance attached via [`Location`]
//! components on the operation entity.

use crate::location::Location;
use prism_ecs_core::{Entity, World};
use serde::{Deserialize, Serialize};

// ── DiagSeverity ─────────────────────────────────────────────────────────────

/// The severity level of a diagnostic message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DiagSeverity {
    /// Informational note (not an error or warning).
    Note,
    /// Non-fatal warning.
    Warning,
    /// Fatal error.
    Error,
}

// ── Diagnostic ───────────────────────────────────────────────────────────────

/// A single diagnostic message carrying severity, text, and optional provenance.
///
/// This is a value type, not an ECS component. Diagnostics are collected by
/// [`DiagnosticEngine`] during pass execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Severity level.
    pub severity: DiagSeverity,
    /// Human-readable diagnostic message.
    pub message: String,
    /// Optional source location of the offending operation.
    pub location: Option<Location>,
    /// Optional name of the operation the diagnostic refers to.
    pub op_name: Option<String>,
}

// ── DiagnosticEngine ─────────────────────────────────────────────────────────

/// A sink for collecting diagnostics across one or more compiler passes.
///
/// All errors and warnings are buffered in memory. Use [`DiagnosticEngine::print`]
/// to flush them to stderr.
pub struct DiagnosticEngine {
    diagnostics: Vec<Diagnostic>,
    max_errors: usize,
}

impl DiagnosticEngine {
    /// Create a new engine that stops collecting after `max_errors` errors
    /// (warnings and notes continue to be recorded).
    ///
    /// Pass `0` to disable error throttling.
    pub fn new(max_errors: usize) -> Self {
        DiagnosticEngine {
            diagnostics: Vec::new(),
            max_errors,
        }
    }

    /// Emit an error-level diagnostic.
    ///
    /// If `op` is `Some(entity)`, the engine looks up the operation's [`Location`]
    /// and name from the ECS world and attaches them to the diagnostic.
    pub fn emit_error(&mut self, msg: &str, op: Option<Entity>, world: &World) {
        if self.max_errors > 0
            && self
                .diagnostics
                .iter()
                .filter(|d| d.severity == DiagSeverity::Error)
                .count()
                >= self.max_errors
        {
            return;
        }
        self.emit(DiagSeverity::Error, msg, op, world);
    }

    /// Emit a warning-level diagnostic.
    pub fn emit_warning(&mut self, msg: &str, op: Option<Entity>, world: &World) {
        self.emit(DiagSeverity::Warning, msg, op, world);
    }

    /// Emit a note-level diagnostic.
    pub fn emit_note(&mut self, msg: &str, op: Option<Entity>, world: &World) {
        self.emit(DiagSeverity::Note, msg, op, world);
    }

    /// Returns `true` when at least one error-level diagnostic has been recorded.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == DiagSeverity::Error)
    }

    /// Prints all collected diagnostics to stderr, prefixed by severity label.
    pub fn print(&self) {
        for diag in &self.diagnostics {
            let severity_label = match diag.severity {
                DiagSeverity::Note => "note",
                DiagSeverity::Warning => "warning",
                DiagSeverity::Error => "error",
            };

            let location_str = diag
                .location
                .as_ref()
                .map(format_location)
                .unwrap_or_default();

            let op_str = diag
                .op_name
                .as_deref()
                .map(|n| format!(" {}", n))
                .unwrap_or_default();

            if location_str.is_empty() {
                eprintln!("{}:{op_str} {}", severity_label, diag.message);
            } else {
                eprintln!(
                    "{}:{op_str} at {}: {}",
                    severity_label, location_str, diag.message
                );
            }
        }
    }

    // ── internal ───────────────────────────────────────────────────────────

    fn emit(&mut self, severity: DiagSeverity, msg: &str, op: Option<Entity>, world: &World) {
        let location = op.and_then(|e| world.get_component::<Location>(e).cloned());
        let op_name = op.and_then(|e| {
            // Try to read the op name from a hypothetical OpName component.
            // We avoid a direct crate dependency on the op module's internals;
            // instead we attempt to find a string-like name via the ECS store.
            world
                .get_component::<crate::op::OpName>(e)
                .map(|n| n.0.clone())
        });

        self.diagnostics.push(Diagnostic {
            severity,
            message: msg.to_string(),
            location,
            op_name,
        });
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Format a [`Location`] into a human-readable string.
fn format_location(loc: &Location) -> String {
    match &loc.0 {
        crate::location::LocKind::Unknown => String::new(),
        crate::location::LocKind::FileLineCol { file, line, col } => {
            format!("{}:{}:{}", file, line, col)
        }
        crate::location::LocKind::Name(name) => name.clone(),
        crate::location::LocKind::CallSite(caller, callee) => {
            format!("call-site({:?}, {:?})", caller, callee)
        }
        crate::location::LocKind::Fused(locs) => {
            let parts: Vec<String> = locs.iter().map(format_location).collect();
            parts.join("; ")
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::EntityKind;

    #[test]
    fn emit_error_with_location() {
        let mut world = World::new();
        let entity: Entity = world
            .spawn(EntityKind::Node, Some("test_op".into()))
            .expect("spawn")
            .into();
        world
            .add_component(entity, Location::file_line_col("foo.mlir", 42, 7))
            .expect("add Location");

        let mut engine = DiagnosticEngine::new(10);
        assert!(!engine.has_errors());

        engine.emit_error("type mismatch", Some(entity), &world);

        assert!(engine.has_errors());
        assert_eq!(engine.diagnostics.len(), 1);

        let diag = &engine.diagnostics[0];
        assert_eq!(diag.severity, DiagSeverity::Error);
        assert_eq!(diag.message, "type mismatch");
        assert!(diag.location.is_some());

        let loc_str = format_location(diag.location.as_ref().unwrap());
        assert!(
            loc_str.contains("foo.mlir"),
            "expected 'foo.mlir' in formatted location, got: {loc_str}"
        );
        assert!(
            loc_str.contains("42"),
            "expected line 42 in formatted location, got: {loc_str}"
        );
        assert!(
            loc_str.contains("7"),
            "expected col 7 in formatted location, got: {loc_str}"
        );
    }

    #[test]
    fn emit_warning() {
        let world = World::new();
        let mut engine = DiagnosticEngine::new(5);
        engine.emit_warning("deprecated pattern", None, &world);
        assert!(!engine.has_errors());
        assert_eq!(engine.diagnostics[0].severity, DiagSeverity::Warning);
    }

    #[test]
    fn max_errors_throttling() {
        let world = World::new();
        let mut engine = DiagnosticEngine::new(2);
        engine.emit_error("e1", None, &world);
        engine.emit_error("e2", None, &world);
        engine.emit_error("e3", None, &world); // throttled
        assert_eq!(engine.has_errors(), true);
        assert_eq!(
            engine
                .diagnostics
                .iter()
                .filter(|d| d.severity == DiagSeverity::Error)
                .count(),
            2
        );
    }

    #[test]
    fn emit_note() {
        let world = World::new();
        let mut engine = DiagnosticEngine::new(10);
        engine.emit_note("candidate fix: add explicit type", None, &world);
        assert!(!engine.has_errors());
        assert_eq!(engine.diagnostics[0].severity, DiagSeverity::Note);
    }

    #[test]
    fn print_does_not_panic() {
        let world = World::new();
        let mut engine = DiagnosticEngine::new(10);
        engine.emit_error("something broke", None, &world);
        engine.emit_warning("something shaky", None, &world);
        // smoke — no crash, no assertion
        engine.print();
    }

    #[test]
    fn emit_error_without_location() {
        let world = World::new();
        let mut engine = DiagnosticEngine::new(10);
        engine.emit_error("no op attached", None, &world);
        let diag = &engine.diagnostics[0];
        assert!(diag.location.is_none());
        assert!(diag.op_name.is_none());
    }

    #[test]
    fn format_location_file_line_col() {
        let loc = Location::file_line_col("bar.mlir", 7, 3);
        let s = format_location(&loc);
        assert_eq!(s, "bar.mlir:7:3");
    }

    #[test]
    fn format_location_unknown() {
        let loc = Location::unknown();
        let s = format_location(&loc);
        assert_eq!(s, "");
    }

    #[test]
    fn format_location_name() {
        let loc = Location::name("my_pass");
        let s = format_location(&loc);
        assert_eq!(s, "my_pass");
    }

    #[test]
    fn zero_max_errors_disables_throttling() {
        let world = World::new();
        let mut engine = DiagnosticEngine::new(0);
        for i in 0..100 {
            engine.emit_error(&format!("e{i}"), None, &world);
        }
        assert_eq!(
            engine
                .diagnostics
                .iter()
                .filter(|d| d.severity == DiagSeverity::Error)
                .count(),
            100
        );
    }
}
