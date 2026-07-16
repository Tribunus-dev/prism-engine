//! Operation location (provenance) tracking.
//!
//! Every operation can carry a [`Location`] component describing where it
//! originated — a source file & line, a symbolic name, a callsite pair, or a
//! fused set of locations. Locations survive serialization round-trips.

use prism_ecs_core::{Component, Entity};
use serde::{Deserialize, Serialize};

// ── LocKind ─────────────────────────────────────────────────────────────────

/// The kind of provenance information carried by a [`Location`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LocKind {
    /// No provenance information.
    Unknown,
    /// A source file, line, and column.
    FileLineCol { file: String, line: u32, col: u32 },
    /// A symbolic name (e.g. a function name, pass label, or debug tag).
    Name(String),
    /// A callsite: the call operation and the callee operation.
    CallSite(Entity, Entity),
    /// A fused set of locations (e.g. from inlining).
    Fused(Vec<Location>),
}

// ── Location component ──────────────────────────────────────────────────────

/// ECS component tracking where an operation originated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Location(pub LocKind);

impl Component for Location {}

impl Location {
    /// Create an Unknown location (the default for new operations).
    pub fn unknown() -> Self {
        Location(LocKind::Unknown)
    }

    /// Create a source-file location.
    pub fn file_line_col(file: &str, line: u32, col: u32) -> Self {
        Location(LocKind::FileLineCol {
            file: file.to_string(),
            line,
            col,
        })
    }

    /// Create a named location.
    pub fn name(name: &str) -> Self {
        Location(LocKind::Name(name.to_string()))
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Retrieve the [`Location`] component from an entity, if present.
pub fn get_location(world: &prism_ecs_core::World, entity: Entity) -> Option<Location> {
    world.get_component::<Location>(entity).cloned()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn location_unknown() {
        let loc = Location::unknown();
        assert_eq!(loc.0, LocKind::Unknown);
    }

    #[test]
    fn location_file_line_col() {
        let loc = Location::file_line_col("foo.mlir", 42, 7);
        assert_eq!(
            loc.0,
            LocKind::FileLineCol {
                file: "foo.mlir".into(),
                line: 42,
                col: 7
            }
        );
    }

    #[test]
    fn location_name() {
        let loc = Location::name("my_pass");
        assert_eq!(loc.0, LocKind::Name("my_pass".into()));
    }

    #[test]
    fn location_serialize_roundtrip() {
        let loc = Location::file_line_col("bar.mlir", 10, 3);
        let json = serde_json::to_string(&loc).unwrap();
        let restored: Location = serde_json::from_str(&json).unwrap();
        assert_eq!(loc, restored);
    }
}
