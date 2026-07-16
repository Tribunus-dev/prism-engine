//! Stable numeric component and resource IDs for bitwise mask operations.
//!
//! Replaces `TypeId` for the scheduling layer to ensure cross-run determinism,
//! serializable manifests, and O(1) overlap checks.  Capacity is bounded at
//! 256 IDs each — deliberate and checked at registration time.

use crate::scheduling::error::{MaskError, RegistryError};
use crate::Component;

// ---------------------------------------------------------------------------
// ID types
// ---------------------------------------------------------------------------

pub type ComponentId = u16;
pub type ResourceId = u16;

pub const MAX_SCHEDULABLE_COMPONENTS: usize = 256;
pub const MAX_SCHEDULABLE_RESOURCES: usize = 256;

// ---------------------------------------------------------------------------
// ComponentMask
// ---------------------------------------------------------------------------

/// Compact 256-bit mask for O(1) overlap checks.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct ComponentMask([u64; 4]);

impl ComponentMask {
    pub const fn empty() -> Self {
        ComponentMask([0; 4])
    }

    /// Insert a component ID into the mask.
    ///
    /// # Errors
    /// Returns `MaskError::ComponentIdOutOfRange` if `id >= 256`.
    pub fn insert(&mut self, id: ComponentId) -> Result<(), MaskError> {
        if id as usize >= MAX_SCHEDULABLE_COMPONENTS {
            return Err(MaskError::ComponentIdOutOfRange { id });
        }
        let (word, bit) = Self::split(id);
        self.0[word] |= 1u64 << bit;
        Ok(())
    }

    /// Check whether `id` is present in the mask.
    pub fn contains(&self, id: ComponentId) -> bool {
        if id as usize >= MAX_SCHEDULABLE_COMPONENTS {
            return false;
        }
        let (word, bit) = Self::split(id);
        (self.0[word] >> bit) & 1 == 1
    }

    /// True if any bit is set in both masks — O(1).
    pub fn overlaps(&self, other: &ComponentMask) -> bool {
        (self.0[0] & other.0[0]) != 0
            || (self.0[1] & other.0[1]) != 0
            || (self.0[2] & other.0[2]) != 0
            || (self.0[3] & other.0[3]) != 0
    }

    /// True if every bit set in `other` is also set in `self`.
    pub fn is_superset(&self, other: &ComponentMask) -> bool {
        (self.0[0] & other.0[0]) == other.0[0]
            && (self.0[1] & other.0[1]) == other.0[1]
            && (self.0[2] & other.0[2]) == other.0[2]
            && (self.0[3] & other.0[3]) == other.0[3]
    }

    fn split(id: ComponentId) -> (usize, usize) {
        (id as usize / 64, id as usize % 64)
    }
}

impl std::fmt::Debug for ComponentMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ComponentMask(0x{:016x}{:016x}{:016x}{:016x})", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

// ---------------------------------------------------------------------------
// ResourceMask — identical layout
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct ResourceMask([u64; 4]);

impl ResourceMask {
    pub const fn empty() -> Self {
        ResourceMask([0; 4])
    }

    /// Insert a resource ID into the mask.
    ///
    /// # Errors
    /// Returns `MaskError::ResourceIdOutOfRange` if `id >= 256`.
    pub fn insert(&mut self, id: ResourceId) -> Result<(), MaskError> {
        if id as usize >= MAX_SCHEDULABLE_RESOURCES {
            return Err(MaskError::ResourceIdOutOfRange { id });
        }
        let (word, bit) = Self::split(id);
        self.0[word] |= 1u64 << bit;
        Ok(())
    }

    pub fn contains(&self, id: ResourceId) -> bool {
        if id as usize >= MAX_SCHEDULABLE_RESOURCES {
            return false;
        }
        let (word, bit) = Self::split(id);
        (self.0[word] >> bit) & 1 == 1
    }

    /// True if any bit is set in both masks — O(1).
    pub fn overlaps(&self, other: &ResourceMask) -> bool {
        (self.0[0] & other.0[0]) != 0
            || (self.0[1] & other.0[1]) != 0
            || (self.0[2] & other.0[2]) != 0
            || (self.0[3] & other.0[3]) != 0
    }

    /// True if every bit set in `other` is also set in `self`.
    pub fn is_superset(&self, other: &ResourceMask) -> bool {
        (self.0[0] & other.0[0]) == other.0[0]
            && (self.0[1] & other.0[1]) == other.0[1]
            && (self.0[2] & other.0[2]) == other.0[2]
            && (self.0[3] & other.0[3]) == other.0[3]
    }

    fn split(id: ResourceId) -> (usize, usize) {
        (id as usize / 64, id as usize % 64)
    }
}

impl std::fmt::Debug for ResourceMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ResourceMask(0x{:016x}{:016x}{:016x}{:016x})", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

// ---------------------------------------------------------------------------
// SchedulableComponent / SchedulableResource
// ---------------------------------------------------------------------------

/// A component with a stable numeric ID for the scheduling system.
///
/// Implement on any component type that should participate in schedule
/// dependency resolution and receive a slot in the component registry.
pub trait SchedulableComponent: Component {
    const COMPONENT_ID: ComponentId;
    const NAME: &'static str;
}

/// A resource with a stable numeric ID for the scheduling system.
///
/// Implement on any resource type that should appear in system access
/// declarations and schedule dependency resolution.
pub trait SchedulableResource: Send + Sync + 'static {
    const RESOURCE_ID: ResourceId;
    const NAME: &'static str;
}

// ---------------------------------------------------------------------------
// Component registry
// ---------------------------------------------------------------------------

/// Validates component ID uniqueness at registration time.
///
/// Collisions are caught before any schedule is compiled, preventing
/// silent identity confusion in manifests and hazard calculations.
pub struct ComponentRegistry {
    id_to_name: [Option<&'static str>; MAX_SCHEDULABLE_COMPONENTS],
}

impl ComponentRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        ComponentRegistry {
            id_to_name: [None; MAX_SCHEDULABLE_COMPONENTS],
        }
    }

    /// Register a component type `T`, checking for ID collisions.
    ///
    /// Returns `RegistryError::ComponentIdCollision` if another component
    /// already occupies `T::COMPONENT_ID`.
    pub fn register<T: SchedulableComponent>(&mut self) -> Result<(), RegistryError> {
        let id = T::COMPONENT_ID as usize;
        if id >= MAX_SCHEDULABLE_COMPONENTS {
            return Err(RegistryError::ComponentRegistryFull);
        }
        if let Some(existing) = self.id_to_name[id] {
            return Err(RegistryError::ComponentIdCollision {
                id: T::COMPONENT_ID,
                existing,
                incoming: T::NAME,
            });
        }
        self.id_to_name[id] = Some(T::NAME);
        Ok(())
    }

    /// Look up the registered name for a component ID.
    pub fn name_for(&self, id: ComponentId) -> Option<&'static str> {
        self.id_to_name.get(id as usize).copied().flatten()
    }
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Validates resource ID uniqueness at registration time.
pub struct ResourceRegistry {
    id_to_name: [Option<&'static str>; MAX_SCHEDULABLE_RESOURCES],
}

impl ResourceRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        ResourceRegistry {
            id_to_name: [None; MAX_SCHEDULABLE_RESOURCES],
        }
    }

    /// Register a resource type `T`, checking for ID collisions.
    ///
    /// Returns `RegistryError::ResourceIdCollision` if another resource
    /// already occupies `T::RESOURCE_ID`.
    pub fn register<T: SchedulableResource>(&mut self) -> Result<(), RegistryError> {
        let id = T::RESOURCE_ID as usize;
        if id >= MAX_SCHEDULABLE_RESOURCES {
            return Err(RegistryError::ResourceRegistryFull);
        }
        if let Some(existing) = self.id_to_name[id] {
            return Err(RegistryError::ResourceIdCollision {
                id: T::RESOURCE_ID,
                existing,
                incoming: T::NAME,
            });
        }
        self.id_to_name[id] = Some(T::NAME);
        Ok(())
    }

    /// Look up the registered name for a resource ID.
    pub fn name_for(&self, id: ResourceId) -> Option<&'static str> {
        self.id_to_name.get(id as usize).copied().flatten()
    }
}

impl Default for ResourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct DummyComponent;
    impl Component for DummyComponent {}
    impl SchedulableComponent for DummyComponent {
        const COMPONENT_ID: ComponentId = 0;
        const NAME: &'static str = "dummy";
    }

    #[test]
    fn mask_insert_and_check() {
        let mut m = ComponentMask::empty();
        assert!(!m.contains(0));
        m.insert(0).unwrap();
        assert!(m.contains(0));
        assert!(!m.contains(1));
    }

    #[test]
    fn mask_overlap() {
        let mut a = ComponentMask::empty();
        let mut b = ComponentMask::empty();
        a.insert(10).unwrap();
        b.insert(10).unwrap();
        assert!(a.overlaps(&b));

        let mut c = ComponentMask::empty();
        c.insert(20).unwrap();
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn mask_superset() {
        let mut a = ComponentMask::empty();
        let mut b = ComponentMask::empty();
        a.insert(1).unwrap();
        a.insert(2).unwrap();
        b.insert(1).unwrap();
        assert!(a.is_superset(&b));
        assert!(!b.is_superset(&a));
    }

    #[test]
    fn registry_accepts_valid_id() {
        let mut reg = ComponentRegistry::new();
        assert!(reg.register::<DummyComponent>().is_ok());
        assert_eq!(reg.name_for(0), Some("dummy"));
    }

    #[test]
    fn registry_rejects_duplicate_id() {
        let mut reg = ComponentRegistry::new();
        reg.register::<DummyComponent>().unwrap();

        #[derive(Debug)]
        struct OtherComponent;
        impl Component for OtherComponent {}
        impl SchedulableComponent for OtherComponent {
            const COMPONENT_ID: ComponentId = 0;
            const NAME: &'static str = "other";
        }
        let err = reg.register::<OtherComponent>().unwrap_err();
        match err {
            RegistryError::ComponentIdCollision { id: 0, existing: "dummy", incoming: "other" } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn registry_rejects_out_of_range() {
        #[derive(Debug)]
        struct OutOfRange;
        impl Component for OutOfRange {}
        impl SchedulableComponent for OutOfRange {
            const COMPONENT_ID: ComponentId = 999;
            const NAME: &'static str = "out_of_range";
        }
        let mut reg = ComponentRegistry::new();
        let err = reg.register::<OutOfRange>().unwrap_err();
        match err {
            RegistryError::ComponentRegistryFull => {}
            other => panic!("unexpected error: {other}"),
        }
    }
}
