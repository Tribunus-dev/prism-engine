//! Stable numeric component and resource IDs for bitwise mask operations.
//!
//! Replaces `TypeId` for the scheduling layer to ensure cross-run determinism,
//! serializable manifests, and O(1) overlap checks.  Capacity is bounded at
//! 256 IDs each — deliberate and checked at registration time.

use crate::runtime::world::Component;
use crate::runtime::scheduling::error::MaskError;
use crate::runtime::scheduling::error::RegistryError;

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
        Self([0; 4])
    }

    pub fn insert(&mut self, id: ComponentId) -> Result<(), MaskError> {
        let index = (id as usize) / 64;
        if index >= self.0.len() {
            return Err(MaskError::OutOfRange {
                id,
                max: MAX_SCHEDULABLE_COMPONENTS,
            });
        }
        self.0[index] |= 1u64 << (id % 64);
        Ok(())
    }

    /// Returns `true` when any bit is set in the same word position in both masks.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .any(|(a, b)| (a & b) != 0)
    }

    /// Returns `true` when `id`'s bit is set.
    pub fn contains(&self, id: ComponentId) -> bool {
        let index = (id as usize) / 64;
        if index >= self.0.len() {
            return false;
        }
        (self.0[index] & (1u64 << (id % 64))) != 0
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        !self.overlaps(other)
    }

    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }

    /// Number of distinct ID bits set.
    pub fn count(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }
}

impl std::fmt::Debug for ComponentMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Print only the non-zero words for readability
        let words: Vec<String> = self
            .0
            .iter()
            .enumerate()
            .filter(|(_, &w)| w != 0)
            .map(|(i, w)| format!("word{}: {:#x}", i, w))
            .collect();
        if words.is_empty() {
            write!(f, "ComponentMask(empty)")
        } else {
            write!(f, "ComponentMask({})", words.join(", "))
        }
    }
}

// ---------------------------------------------------------------------------
// ResourceMask — identical layout
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct ResourceMask([u64; 4]);

impl ResourceMask {
    pub const fn empty() -> Self {
        Self([0; 4])
    }

    pub fn insert(&mut self, id: ResourceId) -> Result<(), MaskError> {
        let index = (id as usize) / 64;
        if index >= self.0.len() {
            return Err(MaskError::OutOfRange {
                id: id as u16,
                max: MAX_SCHEDULABLE_RESOURCES,
            });
        }
        self.0[index] |= 1u64 << (id % 64);
        Ok(())
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .any(|(a, b)| (a & b) != 0)
    }

    pub fn contains(&self, id: ResourceId) -> bool {
        let index = (id as usize) / 64;
        if index >= self.0.len() {
            return false;
        }
        (self.0[index] & (1u64 << (id % 64))) != 0
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        !self.overlaps(other)
    }

    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }

    pub fn count(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }
}

impl std::fmt::Debug for ResourceMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let words: Vec<String> = self
            .0
            .iter()
            .enumerate()
            .filter(|(_, &w)| w != 0)
            .map(|(i, w)| format!("word{}: {:#x}", i, w))
            .collect();
        if words.is_empty() {
            write!(f, "ResourceMask(empty)")
        } else {
            write!(f, "ResourceMask({})", words.join(", "))
        }
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
    pub fn new() -> Self {
        Self {
            id_to_name: [None; MAX_SCHEDULABLE_COMPONENTS],
        }
    }

    pub fn register<T: SchedulableComponent>(&mut self) -> Result<(), RegistryError> {
        let id = T::COMPONENT_ID as usize;
        if id >= MAX_SCHEDULABLE_COMPONENTS {
            return Err(RegistryError::ComponentIdCollision(
                T::COMPONENT_ID,
                T::NAME,
                "<out of range>",
            ));
        }
        if let Some(existing) = self.id_to_name[id] {
            return Err(RegistryError::ComponentIdCollision(
                T::COMPONENT_ID,
                T::NAME,
                existing,
            ));
        }
        self.id_to_name[id] = Some(T::NAME);
        Ok(())
    }

    pub fn name_for(&self, id: ComponentId) -> Option<&'static str> {
        if (id as usize) < self.id_to_name.len() {
            self.id_to_name[id as usize]
        } else {
            None
        }
    }

    pub fn is_registered(&self, id: ComponentId) -> bool {
        self.name_for(id).is_some()
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
    pub fn new() -> Self {
        Self {
            id_to_name: [None; MAX_SCHEDULABLE_RESOURCES],
        }
    }

    pub fn register<T: SchedulableResource>(&mut self) -> Result<(), RegistryError> {
        let id = T::RESOURCE_ID as usize;
        if id >= MAX_SCHEDULABLE_RESOURCES {
            return Err(RegistryError::ResourceIdCollision(
                T::RESOURCE_ID,
                T::NAME,
                "<out of range>",
            ));
        }
        if let Some(existing) = self.id_to_name[id] {
            return Err(RegistryError::ResourceIdCollision(
                T::RESOURCE_ID,
                T::NAME,
                existing,
            ));
        }
        self.id_to_name[id] = Some(T::NAME);
        Ok(())
    }

    pub fn name_for(&self, id: ResourceId) -> Option<&'static str> {
        if (id as usize) < self.id_to_name.len() {
            self.id_to_name[id as usize]
        } else {
            None
        }
    }

    pub fn is_registered(&self, id: ResourceId) -> bool {
        self.name_for(id).is_some()
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
    use crate::runtime::scheduling::error::{MaskError, RegistryError};

    struct DummyComponent;
    impl SchedulableComponent for DummyComponent {
        const COMPONENT_ID: ComponentId = 0;
        const NAME: &'static str = "dummy";
    }

    #[test]
    fn mask_insert_and_check() {
        let mut m = ComponentMask::empty();
        assert!(m.is_empty());
        assert!(!m.contains(0));
        m.insert(0).unwrap();
        assert!(m.contains(0));
        assert!(!m.contains(1));
        assert!(!m.is_empty());
    }

    #[test]
    fn mask_overlap() {
        let mut a = ComponentMask::empty();
        let mut b = ComponentMask::empty();
        a.insert(0).unwrap();
        b.insert(0).unwrap();
        assert!(a.overlaps(&b));
        assert!(!a.is_disjoint(&b));

        let mut c = ComponentMask::empty();
        c.insert(1).unwrap();
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn mask_out_of_range() {
        let mut m = ComponentMask::empty();
        assert!(matches!(
            m.insert(300),
            Err(MaskError::OutOfRange { .. })
        ));
    }

    #[test]
    fn mask_count() {
        let mut m = ComponentMask::empty();
        assert_eq!(m.count(), 0);
        m.insert(0).unwrap();
        m.insert(7).unwrap();
        m.insert(200).unwrap();
        assert_eq!(m.count(), 3);
    }

    #[test]
    fn registry_rejects_duplicate_id() {
        let mut reg = ComponentRegistry::new();
        reg.register::<DummyComponent>().unwrap();
        let result = reg.register::<DummyComponent>();
        assert!(matches!(result, Err(RegistryError::ComponentIdCollision(0, _, _))));
    }

    #[test]
    fn registry_name_lookup() {
        let mut reg = ComponentRegistry::new();
        reg.register::<DummyComponent>().unwrap();
        assert_eq!(reg.name_for(0), Some("dummy"));
        assert_eq!(reg.name_for(1), None);
    }
}
