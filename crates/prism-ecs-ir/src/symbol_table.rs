//! Symbol table — maps symbol names to their defining operations.
//!
//! A resource attached to the ECS World, providing O(1) lookup from
//! symbol name (e.g. "@main") to the operation entity that defines it.

use std::collections::HashMap;

use prism_ecs_core::Entity;
use serde::{Deserialize, Serialize};

// ── SymbolConflict ──────────────────────────────────────────────────────────

/// Error returned when inserting a symbol that already exists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolConflict {
    pub name: String,
}

impl std::fmt::Display for SymbolConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "symbol already exists: {}", self.name)
    }
}

impl std::error::Error for SymbolConflict {}

// ── SymbolTable ─────────────────────────────────────────────────────────────

/// Global symbol table resource.
///
/// Maps symbol names (e.g. "@main", "@helper") to the entity of the
/// operation that defines them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolTable {
    symbols: HashMap<String, Entity>,
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self {
            symbols: HashMap::new(),
        }
    }
}

impl SymbolTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a symbol by name. Returns `None` if not found.
    pub fn lookup(&self, name: &str) -> Option<Entity> {
        self.symbols.get(name).copied()
    }

    /// Insert a new symbol. Fails if the symbol already exists.
    pub fn insert(&mut self, name: String, entity: Entity) -> Result<(), SymbolConflict> {
        if self.symbols.contains_key(&name) {
            return Err(SymbolConflict { name });
        }
        self.symbols.insert(name, entity);
        Ok(())
    }

    /// Remove a symbol. Returns the entity it pointed to.
    pub fn erase(&mut self, name: &str) -> Option<Entity> {
        self.symbols.remove(name)
    }

    /// Check if a symbol exists.
    pub fn contains(&self, name: &str) -> bool {
        self.symbols.contains_key(name)
    }

    /// Number of symbols.
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    /// Iterate over all (name, entity) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, Entity)> {
        self.symbols.iter().map(|(k, v)| (k.as_str(), *v))
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_lookup() {
        let mut st = SymbolTable::new();
        let e = Entity(42, 1);
        assert!(st.insert("@main".into(), e).is_ok());
        assert_eq!(st.lookup("@main"), Some(e));
        assert_eq!(st.len(), 1);
    }

    #[test]
    fn duplicate_insert_fails() {
        let mut st = SymbolTable::new();
        st.insert("@main".into(), Entity(1, 1)).unwrap();
        let err = st.insert("@main".into(), Entity(2, 1)).unwrap_err();
        assert_eq!(err.name, "@main");
    }

    #[test]
    fn erase_symbol() {
        let mut st = SymbolTable::new();
        let e = Entity(42, 1);
        st.insert("@main".into(), e).unwrap();
        assert_eq!(st.erase("@main"), Some(e));
        assert!(st.is_empty());
    }

    #[test]
    fn contains_symbol() {
        let mut st = SymbolTable::new();
        st.insert("@helper".into(), Entity(7, 1)).unwrap();
        assert!(st.contains("@helper"));
        assert!(!st.contains("@missing"));
    }
}
