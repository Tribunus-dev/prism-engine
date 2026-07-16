//! Op interface system — composable interfaces that operations can declare.
//!
//! Interfaces are named trait-like tags that describe what an operation supports
//! (e.g. "InferType", "LoopLike"). Each op entity carries an [`InterfaceRef`]
//! component listing which interfaces it implements. The [`InterfaceRegistry`]
//! provides a central catalog of known interface names and descriptions.

use prism_ecs_core::{Component, Entity, EntityKind, World};

use serde::{Deserialize, Serialize};

// ── InterfaceRegistry ────────────────────────────────────────────────────────

/// A registry of known op interfaces.
///
/// Interfaces are identified by their unique `&'static str` name. The registry
/// stores human-readable descriptions and can be used to validate that an op's
/// declared interfaces are recognized.
#[derive(Debug, Clone)]
pub struct InterfaceRegistry {
    interfaces: std::collections::HashMap<&'static str, InterfaceInfo>,
}

impl InterfaceRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            interfaces: std::collections::HashMap::new(),
        }
    }

    /// Register an interface by name and description.
    pub fn register(&mut self, info: InterfaceInfo) {
        self.interfaces.insert(info.name, info);
    }

    /// Look up an interface by name.
    pub fn get(&self, name: &str) -> Option<&InterfaceInfo> {
        self.interfaces.get(name)
    }

    /// Check whether an interface is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.interfaces.contains_key(name)
    }

    /// Iterate over all registered interfaces.
    pub fn iter(&self) -> impl Iterator<Item = &InterfaceInfo> {
        self.interfaces.values()
    }

    /// Return the number of registered interfaces.
    pub fn len(&self) -> usize {
        self.interfaces.len()
    }

    /// Returns `true` if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.interfaces.is_empty()
    }
}

impl Default for InterfaceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── InterfaceInfo ────────────────────────────────────────────────────────────

/// Metadata about a single op interface.
#[derive(Debug, Clone)]
pub struct InterfaceInfo {
    /// Unique interface name, e.g. `"InferType"`, `"LoopLike"`.
    pub name: &'static str,
    /// Human-readable description of what the interface provides.
    pub description: &'static str,
}

impl InterfaceInfo {
    /// Create a new interface with the given name and description.
    pub const fn new(name: &'static str, description: &'static str) -> Self {
        Self { name, description }
    }
}

// ── InterfaceRef component ──────────────────────────────────────────────────

/// Component that lists which interfaces an operation entity implements.
///
/// Each string is an interface name (e.g. `"InferType"`, `"MemoryEffects"`).
///
/// # Example
/// ```ignore
/// world.add_component(op_entity, InterfaceRef(vec![
///     "InferType".into(),
///     "Callable".into(),
/// ]))?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceRef(pub Vec<String>);

impl Component for InterfaceRef {}

impl InterfaceRef {
    /// Create an empty interface list.
    pub fn empty() -> Self {
        Self(Vec::new())
    }

    /// Create an interface list from string slices.
    pub fn from_strs(names: &[&str]) -> Self {
        Self(names.iter().map(|s| s.to_string()).collect())
    }

    /// Check if this ref includes a specific interface.
    pub fn has(&self, name: &str) -> bool {
        self.0.iter().any(|s| s == name)
    }

    /// Add an interface name.
    pub fn add(&mut self, name: impl Into<String>) {
        self.0.push(name.into());
    }
}

// ── Standard interfaces ─────────────────────────────────────────────────────

/// Names of the standard op interfaces.
pub mod standard {
    /// Op can infer its result types.
    pub const INFER_TYPE: &str = "InferType";
    /// Op reads/writes memory.
    pub const MEMORY_EFFECTS: &str = "MemoryEffects";
    /// Op is a loop (scf.for, scf.while).
    pub const LOOP_LIKE: &str = "LoopLike";
    /// Op uses regions.
    pub const REGION_KIND: &str = "RegionKind";
    /// Op defines a symbol.
    pub const SYMBOL: &str = "Symbol";
    /// Op can be called (func.func).
    pub const CALLABLE: &str = "Callable";

    /// All standard interface names as a slice.
    pub const ALL: &[&str] = &[
        INFER_TYPE,
        MEMORY_EFFECTS,
        LOOP_LIKE,
        REGION_KIND,
        SYMBOL,
        CALLABLE,
    ];
}

/// Register the standard set of op interfaces into the given registry,
/// spawn interface entities in the world, and return the mapping from
/// each entity to its list of trait-like interface names.
///
/// Each standard interface is represented as an entity in the ECS world,
/// enabling ECS-native queries against interfaces.
pub fn register_standard_interfaces(
    world: &mut World,
    registry: &mut InterfaceRegistry,
) -> Vec<(Entity, Vec<&'static str>)> {
    let infos: Vec<(InterfaceInfo, &[&str])> = vec![
        (
            InterfaceInfo::new(
                standard::INFER_TYPE,
                "Op can infer its result types from operands and attributes",
            ),
            &[standard::INFER_TYPE][..],
        ),
        (
            InterfaceInfo::new(
                standard::MEMORY_EFFECTS,
                "Op reads from or writes to memory",
            ),
            &[standard::MEMORY_EFFECTS][..],
        ),
        (
            InterfaceInfo::new(
                standard::LOOP_LIKE,
                "Op is a loop construct (scf.for, scf.while)",
            ),
            &[standard::LOOP_LIKE][..],
        ),
        (
            InterfaceInfo::new(standard::REGION_KIND, "Op uses regions"),
            &[standard::REGION_KIND][..],
        ),
        (
            InterfaceInfo::new(standard::SYMBOL, "Op defines a symbol"),
            &[standard::SYMBOL][..],
        ),
        (
            InterfaceInfo::new(standard::CALLABLE, "Op can be called (func.func)"),
            &[standard::CALLABLE][..],
        ),
    ];

    let mut result = Vec::with_capacity(infos.len());
    for (info, traits) in infos {
        registry.register(info.clone());

        let entity: Entity = world
            .spawn(EntityKind::Node, Some(format!("interface_{}", info.name)))
            .expect("spawn interface entity failed")
            .into();
        result.push((entity, traits.to_vec()));
    }

    result
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::{EntityKind, World};

    #[test]
    fn register_interfaces_and_attach_to_op() {
        let mut world = World::new();
        let mut registry = InterfaceRegistry::new();

        // Register standard interfaces.
        let interface_entities = register_standard_interfaces(&mut world, &mut registry);

        // Verify all standard interfaces are registered.
        assert_eq!(registry.len(), 6);
        for name in standard::ALL {
            assert!(registry.contains(name));
        }

        // Create an op entity and attach an InterfaceRef.
        let op_entity: Entity = world
            .spawn(EntityKind::Node, Some("test_op".into()))
            .expect("spawn failed")
            .into();
        world
            .add_component(
                op_entity,
                InterfaceRef(vec![
                    standard::INFER_TYPE.to_string(),
                    standard::CALLABLE.to_string(),
                    standard::SYMBOL.to_string(),
                ]),
            )
            .expect("add InterfaceRef");

        // Verify lookup — read back the component and check contents.
        let ref_comp = world
            .get_component::<InterfaceRef>(op_entity)
            .expect("InterfaceRef should be present");
        assert!(ref_comp.has(standard::INFER_TYPE));
        assert!(ref_comp.has(standard::CALLABLE));
        assert!(ref_comp.has(standard::SYMBOL));
        assert!(!ref_comp.has(standard::LOOP_LIKE));
        assert!(!ref_comp.has(standard::MEMORY_EFFECTS));

        // Verify the interface entities were created.
        assert_eq!(interface_entities.len(), 6);

        // Check that each interface entity exists in the world.
        for (entity, traits) in &interface_entities {
            assert!(world
                .get_component::<crate::op::OpMarker>(*entity)
                .is_none());
            assert!(traits.len() == 1);
            assert!(registry.contains(traits[0]));
        }

        // Test InterfaceRef helper methods.
        let mut iref = InterfaceRef::empty();
        assert!(!iref.has(standard::INFER_TYPE));
        iref.add(standard::LOOP_LIKE);
        assert!(iref.has(standard::LOOP_LIKE));

        let iref2 = InterfaceRef::from_strs(&[standard::INFER_TYPE, standard::MEMORY_EFFECTS]);
        assert!(iref2.has(standard::INFER_TYPE));
        assert!(iref2.has(standard::MEMORY_EFFECTS));
        assert!(!iref2.has(standard::CALLABLE));
    }
}
