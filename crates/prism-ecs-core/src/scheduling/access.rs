//! Compile-time component and resource access declarations.
//!
//! `ComponentSet` and `ResourceSet` are implemented for tuples of component
//! types and compiled into bitwise masks for O(1) overlap checks.

use crate::scheduling::component_id::{ComponentMask, ResourceMask};
use crate::scheduling::error::MaskError;

/// A set of component types that a system intends to read or write.
///
/// Implemented for `()`, single types, and tuples via macro.
pub trait ComponentSet {
    /// Produce the mask for this set, or fail on an out-of-range ID.
    fn mask() -> Result<ComponentMask, MaskError>;
}

/// A set of resource types that a system intends to read or write.
///
/// Implemented for `()`, single types, and tuples via macro.
pub trait ResourceSet {
    /// Produce the mask for this set, or fail on an out-of-range ID.
    fn mask() -> Result<ResourceMask, MaskError>;
}

// ── Base cases ─────────────────────────────────────────────────────────────

impl ComponentSet for () {
    fn mask() -> Result<ComponentMask, MaskError> {
        Ok(ComponentMask::empty())
    }
}

impl ResourceSet for () {
    fn mask() -> Result<ResourceMask, MaskError> {
        Ok(ResourceMask::empty())
    }
}

// ── Single component ───────────────────────────────────────────────────────

impl<T: crate::scheduling::component_id::SchedulableComponent> ComponentSet for T {
    fn mask() -> Result<ComponentMask, MaskError> {
        let mut mask = ComponentMask::empty();
        mask.insert(T::COMPONENT_ID)?;
        Ok(mask)
    }
}

// ── Single resource ────────────────────────────────────────────────────────

impl<T: crate::scheduling::component_id::SchedulableResource> ResourceSet for T {
    fn mask() -> Result<ResourceMask, MaskError> {
        let mut mask = ResourceMask::empty();
        mask.insert(T::RESOURCE_ID)?;
        Ok(mask)
    }
}

// ── Tuple macros ───────────────────────────────────────────────────────────

macro_rules! impl_component_set_tuple {
    ($($T:ident),+) => {
        impl<$($T: crate::scheduling::component_id::SchedulableComponent),+> ComponentSet for ($($T,)+) {
            #[allow(non_snake_case)]
            fn mask() -> Result<ComponentMask, MaskError> {
                let mut mask = ComponentMask::empty();
                $(
                    mask.insert($T::COMPONENT_ID)?;
                )+
                Ok(mask)
            }
        }
    };
}

macro_rules! impl_resource_set_tuple {
    ($($T:ident),+) => {
        impl<$($T: crate::scheduling::component_id::SchedulableResource),+> ResourceSet for ($($T,)+) {
            #[allow(non_snake_case)]
            fn mask() -> Result<ResourceMask, MaskError> {
                let mut mask = ResourceMask::empty();
                $(
                    mask.insert($T::RESOURCE_ID)?;
                )+
                Ok(mask)
            }
        }
    };
}

impl_component_set_tuple!(A, B);
impl_component_set_tuple!(A, B, C);
impl_component_set_tuple!(A, B, C, D);
impl_component_set_tuple!(A, B, C, D, E);
impl_component_set_tuple!(A, B, C, D, E, F);

impl_resource_set_tuple!(A, B);
impl_resource_set_tuple!(A);
impl_resource_set_tuple!(A, B, C);
impl_resource_set_tuple!(A, B, C, D);
impl_resource_set_tuple!(A, B, C, D, E);
impl_resource_set_tuple!(A, B, C, D, E, F);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::component_id::{ComponentId, SchedulableComponent};

    #[derive(Debug)]
    struct CompA;
    impl crate::Component for CompA {}
    impl SchedulableComponent for CompA {
        const COMPONENT_ID: ComponentId = 5;
        const NAME: &'static str = "comp_a";
    }

    #[derive(Debug)]
    struct CompB;
    impl crate::Component for CompB {}
    impl SchedulableComponent for CompB {
        const COMPONENT_ID: ComponentId = 10;
        const NAME: &'static str = "comp_b";
    }

    #[test]
    fn single_component_set() {
        let mask = <CompA as ComponentSet>::mask().unwrap();
        assert!(mask.contains(5));
        assert!(!mask.contains(10));
    }

    #[test]
    fn tuple_component_set() {
        let mask = <(CompA, CompB) as ComponentSet>::mask().unwrap();
        assert!(mask.contains(5));
        assert!(mask.contains(10));
    }

    #[test]
    fn empty_set_produces_empty_mask() {
        let mask = <() as ComponentSet>::mask().unwrap();
        assert!(!mask.contains(0));
        assert!(!mask.contains(255));
    }
}
