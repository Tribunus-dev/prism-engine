//! Compile-time component and resource access declarations.
//!
//! `ComponentSet` and `ResourceSet` are implemented for tuples of component
//! types and compiled into bitwise masks for O(1) overlap checks.

use crate::runtime::scheduling::component_id::{ComponentMask, ResourceMask};
use crate::runtime::scheduling::error::MaskError;

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

impl<T: crate::runtime::scheduling::component_id::SchedulableComponent> ComponentSet for T {
    fn mask() -> Result<ComponentMask, MaskError> {
        let mut mask = ComponentMask::empty();
        mask.insert(T::COMPONENT_ID)?;
        Ok(mask)
    }
}

// ── Single resource ────────────────────────────────────────────────────────

impl<T: crate::runtime::scheduling::component_id::SchedulableResource> ResourceSet for T {
    fn mask() -> Result<ResourceMask, MaskError> {
        let mut mask = ResourceMask::empty();
        mask.insert(T::RESOURCE_ID)?;
        Ok(mask)
    }
}

// ── Tuple macros ───────────────────────────────────────────────────────────

macro_rules! impl_component_set_tuple {
    ($($ty:ident),+) => {
        impl<$($ty: crate::runtime::scheduling::component_id::SchedulableComponent),+> ComponentSet for ($($ty,)+) {
            fn mask() -> Result<ComponentMask, MaskError> {
                let mut mask = ComponentMask::empty();
                $(mask.insert($ty::COMPONENT_ID)?;)+
                Ok(mask)
            }
        }
    };
}

macro_rules! impl_resource_set_tuple {
    ($($ty:ident),+) => {
        impl<$($ty: crate::runtime::scheduling::component_id::SchedulableResource),+> ResourceSet for ($($ty,)+) {
            fn mask() -> Result<ResourceMask, MaskError> {
                let mut mask = ResourceMask::empty();
                $(mask.insert($ty::RESOURCE_ID)?;)+
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
    use crate::runtime::scheduling::component_id::*;

    struct A;
    impl SchedulableComponent for A {
        const COMPONENT_ID: ComponentId = 0;
        const NAME: &'static str = "A";
    }

    struct B;
    impl SchedulableComponent for B {
        const COMPONENT_ID: ComponentId = 1;
        const NAME: &'static str = "B";
    }

    struct C;
    impl SchedulableComponent for C {
        const COMPONENT_ID: ComponentId = 2;
        const NAME: &'static str = "C";
    }

    #[test]
    fn empty_set() {
        let mask = <() as ComponentSet>::mask().unwrap();
        assert!(mask.is_empty());
    }

    #[test]
    fn single_component_set() {
        let mask = <A as ComponentSet>::mask().unwrap();
        assert!(mask.contains(0));
        assert!(!mask.contains(1));
        assert_eq!(mask.count(), 1);
    }

    #[test]
    fn tuple_set() {
        let mask = <(A, B) as ComponentSet>::mask().unwrap();
        assert!(mask.contains(0));
        assert!(mask.contains(1));
        assert!(!mask.contains(2));
        assert_eq!(mask.count(), 2);
    }

    #[test]
    fn triple_set() {
        let mask = <(A, B, C) as ComponentSet>::mask().unwrap();
        assert_eq!(mask.count(), 3);
    }
}
