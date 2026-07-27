//! Authority: this module owns the canonical component-classification
//! surface — the [`ComponentClass`], [`DurableClass`], [`TransientClass`]
//! markers, the [`ClassifiedComponent`] association trait, and the
//! [`DurableComponent`] / [`TransientComponent`] constraints. A type
//! implementing [`ClassifiedComponent`] is sealed to be either durable
//! or transient, never both, and the classification decides whether a
//! mutation is journaled.

use crate::types::SchemaKey;
use serde::{de::DeserializeOwned, Serialize};

/// Sealed — only [`DurableClass`] and [`TransientClass`] may implement
/// this. Implemented via a private `Sealed` trait so that no external
/// crate can introduce a third class.
pub trait ComponentClass: private::Sealed {}

/// Marker type for durable (journaled, replayed, snapshotted)
/// components. Mutations against a [`DurableComponent`] are recorded in
/// the mutation journal and re-applied by replay.
pub struct DurableClass;
impl private::Sealed for DurableClass {}
impl ComponentClass for DurableClass {}

/// Marker type for transient (process-local, non-replayed) components.
/// Mutations against a [`TransientComponent`] are not journaled and are
/// not replayed. The component is reconstructed on restart by
/// subsystem startup or reconciliation code.
pub struct TransientClass;
impl private::Sealed for TransientClass {}
impl ComponentClass for TransientClass {}

mod private {
    /// Sealing trait — kept in a private module so the classification
    /// cannot be extended outside this crate.
    pub trait Sealed {}
}

/// A component explicitly classified as durable or transient.
///
/// Each Rust type can implement this only once — it cannot be both. The
/// associated `Class` type is the [`ComponentClass`] marker.
pub trait ClassifiedComponent: prism_ecs_core::Component {
    type Class: ComponentClass;
}

/// A durable component: serializable, journaled, replayable.
///
/// Every durable component must provide a stable [`SchemaKey`] derived
/// from its `SCHEMA_KEY` constant. The key is independent of Rust type
/// names or crate paths, so the same component can be re-implemented
/// under a different Rust type name without breaking the wire format.
pub trait DurableComponent:
    ClassifiedComponent<Class = DurableClass> + Serialize + DeserializeOwned
{
    /// Stable schema key — used by the catalogue to look up the
    /// durable registration and by the journal to record the change.
    const SCHEMA_KEY: SchemaKey;
}

/// A transient component: runtime-only, never journaled or replayed.
///
/// Transient components disappear on restart and must be reconstructed
/// by subsystem startup or reconciliation code. They participate in
/// OCC (their version is bumped) but do not produce a journal entry.
pub trait TransientComponent: ClassifiedComponent<Class = TransientClass> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SchemaKey;
    use prism_ecs_core::Component;

    /// A concrete component can be classified as either durable or
    /// transient — never both. The classification decides whether
    /// mutations against the type are journaled. This test pins the
    /// classification marker pair to the expected associated types.
    #[test]
    fn classification_markers_seal_to_expected_classes() {
        // The sealed module keeps external crates from adding a
        // third class. `ComponentClass` is implemented only for
        // `DurableClass` and `TransientClass`.
        fn assert_class<T: ComponentClass>() {}
        assert_class::<DurableClass>();
        assert_class::<TransientClass>();
    }

    /// `DurableComponent` requires a stable `SCHEMA_KEY` constant.
    /// Two implementations of the same logical component must use
    /// the same schema key — that is the entire point of the key.
    /// The test pins the constant shape so refactors do not
    /// accidentally drop the `const`.
    #[test]
    fn durable_component_schema_key_is_required_const() {
        // A trivial durable component for shape-checking.
        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Position {
            x: f32,
            y: f32,
        }
        impl Component for Position {}
        impl ClassifiedComponent for Position {
            type Class = DurableClass;
        }
        impl DurableComponent for Position {
            const SCHEMA_KEY: SchemaKey = SchemaKey {
                namespace: "geometry",
                id: 1,
                version: 1,
            };
        }
        assert_eq!(Position::SCHEMA_KEY.namespace, "geometry");
        assert_eq!(Position::SCHEMA_KEY.id, 1);
    }
}
