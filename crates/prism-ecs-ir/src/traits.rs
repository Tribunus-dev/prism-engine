//! Op trait system for the ECS-native IR.
//!
//! Traits are bitflags that describe semantic properties of an operation:
//! commutativity, purity, terminator behavior, etc. These are stored as a
//! `Traits` component on the op entity and used during verification and
//! transformation (e.g., CSE can dedup ops with `PURE`, reordering is
//! legal for `COMMUTATIVE`).

use bitflags::bitflags;

use prism_ecs_core::{Component, Entity, World};

bitflags! {
    /// Bitflags describing semantic properties of an operation.
    ///
    /// Each op stores its traits as a `Traits` component in the ECS world.
    /// Traits are combined via bitwise OR at op construction time and serve
    /// as the canonical input to pattern matching, CSE, and reordering passes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct OpTraits: u64 {
        /// Op where swapping operand order produces the same result.
        const COMMUTATIVE = 1 << 0;
        /// Op with no observable side effects (safe to dead-code eliminate).
        const PURE = 1 << 1;
        /// Op whose result type matches its operand types (e.g. arith.addf).
        const SAME_OPERANDS_AND_RESULT_TYPE = 1 << 2;
        /// Op that terminates a block (e.g. func.return, scf.yield).
        const TERMINATOR = 1 << 3;
        /// Op with no side effects (more restrictive than PURE — also no
        /// memory, no I/O, no exceptions).
        const NO_SIDE_EFFECTS = 1 << 4;
        /// Op whose output depends only on constant inputs (foldable).
        const CONSTANT_LIKE = 1 << 5;
        /// Op whose execution is isolated from ops above in the dominance
        /// tree (no influence from dominating ops).
        const ISOLATED_FROM_ABOVE = 1 << 6;
    }
}

/// Human-readable metadata for a registered trait.
#[derive(Debug, Clone)]
pub struct TraitInfo {
    pub name: &'static str,
    pub bit: OpTraits,
    pub description: &'static str,
}

/// Returns all standard traits with their names and descriptions.
pub fn register_standard_traits() -> Vec<TraitInfo> {
    vec![
        TraitInfo {
            name: "COMMUTATIVE",
            bit: OpTraits::COMMUTATIVE,
            description: "swap operands without changing the result",
        },
        TraitInfo {
            name: "PURE",
            bit: OpTraits::PURE,
            description: "no observable side effects; safe to eliminate",
        },
        TraitInfo {
            name: "SAME_OPERANDS_AND_RESULT_TYPE",
            bit: OpTraits::SAME_OPERANDS_AND_RESULT_TYPE,
            description: "result type must match operand types",
        },
        TraitInfo {
            name: "TERMINATOR",
            bit: OpTraits::TERMINATOR,
            description: "terminates a block",
        },
        TraitInfo {
            name: "NO_SIDE_EFFECTS",
            bit: OpTraits::NO_SIDE_EFFECTS,
            description: "no memory writes, I/O, or exceptions",
        },
        TraitInfo {
            name: "CONSTANT_LIKE",
            bit: OpTraits::CONSTANT_LIKE,
            description: "result depends only on constant inputs",
        },
        TraitInfo {
            name: "ISOLATED_FROM_ABOVE",
            bit: OpTraits::ISOLATED_FROM_ABOVE,
            description: "not influenced by dominating ops",
        },
    ]
}

/// ECS component holding an op's trait bitflags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Traits(pub OpTraits);
impl Component for Traits {}

/// Attach a set of traits to an op entity.
pub fn set_traits(world: &mut World, entity: Entity, traits: OpTraits) {
    world
        .add_component(entity, Traits(traits))
        .expect("set_traits");
}

/// Query an op's traits from the world.
pub fn get_traits(world: &World, entity: Entity) -> Option<OpTraits> {
    world.get_component::<Traits>(entity).map(|t| t.0)
}

/// Merge additional traits onto an existing op. No-op if the op has no
/// `Traits` component yet.
pub fn add_traits(world: &mut World, entity: Entity, traits: OpTraits) {
    if let Some(existing) = world.get_component_mut::<Traits>(entity) {
        existing.0 |= traits;
    }
}

/// Check whether an entity is marked with the given trait bit(s).
pub fn has_trait(world: &World, entity: Entity, t: OpTraits) -> bool {
    get_traits(world, entity)
        .map(|bits| bits.contains(t))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::OpMarker;
    use prism_ecs_core::EntityKind;

    #[test]
    fn test_bitflag_constants() {
        let commutative = OpTraits::COMMUTATIVE;
        let pure = OpTraits::PURE;
        let combined = commutative | pure;

        assert!(combined.contains(OpTraits::COMMUTATIVE));
        assert!(combined.contains(OpTraits::PURE));
        assert!(!combined.contains(OpTraits::TERMINATOR));
        assert_eq!(combined.bits(), 0b11);
    }

    #[test]
    fn test_set_and_get_traits() {
        let mut world = World::default();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("test_op".into()))
            .expect("spawn failed")
            .into();
        world.add_component(op, OpMarker).expect("add OpMarker");

        let traits = OpTraits::COMMUTATIVE | OpTraits::PURE;
        set_traits(&mut world, op, traits);

        assert_eq!(get_traits(&world, op), Some(traits));
        assert!(has_trait(&world, op, OpTraits::COMMUTATIVE));
        assert!(has_trait(&world, op, OpTraits::PURE));
        assert!(!has_trait(&world, op, OpTraits::TERMINATOR));
    }

    #[test]
    fn test_add_traits() {
        let mut world = World::default();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("test_op".into()))
            .expect("spawn failed")
            .into();
        world.add_component(op, OpMarker).expect("add OpMarker");

        set_traits(&mut world, op, OpTraits::COMMUTATIVE);
        add_traits(&mut world, op, OpTraits::PURE);

        let bits = get_traits(&world, op).unwrap();
        assert!(bits.contains(OpTraits::COMMUTATIVE));
        assert!(bits.contains(OpTraits::PURE));
    }

    #[test]
    fn test_no_traits_on_untagged_op() {
        let mut world = World::default();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("untagged".into()))
            .expect("spawn failed")
            .into();
        // Entity exists but has no Traits component.
        assert_eq!(get_traits(&world, op), None);
        assert!(!has_trait(&world, op, OpTraits::PURE));
    }

    #[test]
    fn test_register_standard_traits() {
        let infos = register_standard_traits();
        assert_eq!(infos.len(), 7);

        // Every predefined trait must appear in the registry.
        let all_bits: OpTraits = infos.iter().fold(OpTraits::empty(), |acc, ti| acc | ti.bit);
        assert!(all_bits.contains(OpTraits::COMMUTATIVE));
        assert!(all_bits.contains(OpTraits::PURE));
        assert!(all_bits.contains(OpTraits::SAME_OPERANDS_AND_RESULT_TYPE));
        assert!(all_bits.contains(OpTraits::TERMINATOR));
        assert!(all_bits.contains(OpTraits::NO_SIDE_EFFECTS));
        assert!(all_bits.contains(OpTraits::CONSTANT_LIKE));
        assert!(all_bits.contains(OpTraits::ISOLATED_FROM_ABOVE));
    }

    #[test]
    fn test_op_traits_component_on_op() {
        // Create an op with Commutative|Pure traits and verify bitflags match.
        let mut world = World::default();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("test_op".into()))
            .expect("spawn failed")
            .into();
        world.add_component(op, OpMarker).expect("add OpMarker");

        let expected = OpTraits::COMMUTATIVE | OpTraits::PURE;
        set_traits(&mut world, op, expected);

        let stored = world.get_component::<Traits>(op).unwrap().0;
        assert_eq!(stored, expected);
        assert_eq!(stored.bits(), 0b11);
    }

    #[test]
    fn test_traitinfo_name_description() {
        let infos = register_standard_traits();
        let commutative = infos
            .iter()
            .find(|ti| ti.bit == OpTraits::COMMUTATIVE)
            .unwrap();
        assert_eq!(commutative.name, "COMMUTATIVE");
        assert!(!commutative.description.is_empty());
    }
}
