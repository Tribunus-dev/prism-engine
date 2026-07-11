use crate::ecs::constitutional::types::DomainId;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ── Backend Capability ─────────────────────────────────────────────────────

/// Backend capability — describes what a device or driver can do.
/// Loose enum that can be extended without breaking existing matches (use `Other` for extensions).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendCapability {
    // Compute primitives
    MatMulF32,
    MatMulF16,
    MatMulInt8,
    // Memory
    UnifiedMemory,
    DedicatedMemory { size_bytes: u64 },
    // Model execution
    MoeDispatch,
    Attention,
    FusedMlp,
    // Host interface
    RequiresHostCopy,
    SupportsPeerAccess,
    // Extension
    Other(String),
}

// ── Driver Info ────────────────────────────────────────────────────────────

/// Human-readable name for a backend/driver family.
pub type BackendName = String;

/// Lightweight info produced by cheap enumeration — no side effects.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DriverInfo {
    pub name: BackendName,
    pub version_major: u32,
    pub version_minor: u32,
    pub available: bool,
    pub description: String,
}

// ── Device Metadata ────────────────────────────────────────────────────────

/// Metadata about a device discovered via try_create.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMetadata {
    pub name: String,
    pub device_id: DomainId,
    pub memory_bytes: u64,
    pub compute_units: u32,
    pub max_alloc_bytes: u64,
}

// ── Driver Create Outcome ──────────────────────────────────────────────────

/// Outcome of DriverFactory::try_create — validated before device entity is committed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DriverCreateOutcome {
    pub handle: String, // Opaque handle; will be typed in later stages
    pub capabilities: Vec<BackendCapability>,
    pub device_metadata: DeviceMetadata,
    pub validation_digest: [u8; 32],
}

// ── Factory Trait ──────────────────────────────────────────────────────────

/// Factory trait — open for registration.
pub trait DriverFactory: Send + Sync {
    /// Cheap enumeration — no side effects, no library loading.
    fn enumerate(&self) -> Vec<DriverInfo>;
    /// Expensive creation — may load libraries, allocate devices.
    fn try_create(&self, info: &DriverInfo) -> Option<DriverCreateOutcome>;
    /// Human-readable name for diagnostics.
    fn name(&self) -> &str;
}

// ── Registry ───────────────────────────────────────────────────────────────

/// Registry of driver factories — an ECS resource.
pub struct DriverRegistry {
    factories: Vec<Arc<dyn DriverFactory>>,
}

impl std::fmt::Debug for DriverRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriverRegistry")
            .field("factory_count", &self.factories.len())
            .finish()
    }
}

impl DriverRegistry {
    pub fn new() -> Self {
        Self {
            factories: Vec::new(),
        }
    }

    pub fn register_factory(&mut self, factory: Arc<dyn DriverFactory>) {
        self.factories.push(factory);
    }

    pub fn enumerate_all(&self) -> Vec<(BackendName, Vec<DriverInfo>)> {
        self.factories
            .iter()
            .map(|f| (f.name().to_string(), f.enumerate()))
            .collect()
    }

    pub fn factory_count(&self) -> usize {
        self.factories.len()
    }

    /// Look up a factory by matching `info` and attempt creation.
    pub fn try_create_from_info(&self, info: &DriverInfo) -> Option<DriverCreateOutcome> {
        for f in &self.factories {
            if f.name() == info.name || f.enumerate().iter().any(|i| i.name == info.name) {
                if let Some(outcome) = f.try_create(info) {
                    return Some(outcome);
                }
            }
        }
        None
    }
}

impl Default for DriverRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Manual Serialize + Deserialize for DriverRegistry — factories are
// runtime objects and cannot cross serialization boundaries. We only
// persist factory_count as a sentinel.
impl Serialize for DriverRegistry {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("DriverRegistry", 1)?;
        s.serialize_field("factory_count", &self.factories.len())?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for DriverRegistry {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::MapAccess;
        use serde::de::Visitor;

        struct RegistryVisitor;

        impl<'de> Visitor<'de> for RegistryVisitor {
            type Value = DriverRegistry;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("DriverRegistry with factory_count")
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<DriverRegistry, V::Error> {
                let mut _count: Option<usize> = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "factory_count" => {
                            _count = Some(map.next_value()?);
                        }
                        _ => {
                            let _: serde::de::IgnoredAny = map.next_value()?;
                        }
                    }
                }
                // Always deserialize to an empty registry — factories must be
                // re-registered at runtime after deserialization.
                Ok(DriverRegistry::new())
            }
        }

        deserializer.deserialize_struct("DriverRegistry", &["factory_count"], RegistryVisitor)
    }
}

// ── Built-in backend name constants ────────────────────────────────────────

/// Well-known backend name constants.
pub mod builtin {
    pub const METAL: &str = "Metal";
    pub const ANE: &str = "ANE";
    pub const ACCELERATE: &str = "Accelerate";
    pub const CUDA: &str = "CUDA";
    pub const ROCM: &str = "ROCm";
    pub const TENSIX: &str = "Tensix";
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A test factory that always reports one available Metal driver.
    struct TestMetalFactory;

    impl DriverFactory for TestMetalFactory {
        fn enumerate(&self) -> Vec<DriverInfo> {
            vec![DriverInfo {
                name: "Metal (test)".into(),
                version_major: 1,
                version_minor: 0,
                available: true,
                description: "Test Metal backend".into(),
            }]
        }

        fn try_create(&self, info: &DriverInfo) -> Option<DriverCreateOutcome> {
            if !info.available {
                return None;
            }
            Some(DriverCreateOutcome {
                handle: "metal-test-0".into(),
                capabilities: vec![
                    BackendCapability::MatMulF32,
                    BackendCapability::MatMulF16,
                    BackendCapability::UnifiedMemory,
                ],
                device_metadata: DeviceMetadata {
                    name: "Test Metal GPU".into(),
                    device_id: DomainId::default(),
                    memory_bytes: 8 * 1024 * 1024 * 1024,
                    compute_units: 16,
                    max_alloc_bytes: 4 * 1024 * 1024 * 1024,
                },
                validation_digest: [0u8; 32],
            })
        }

        fn name(&self) -> &str {
            "test-metal"
        }
    }

    #[test]
    fn registry_new_is_empty() {
        let reg = DriverRegistry::new();
        assert_eq!(reg.factory_count(), 0);
    }

    #[test]
    fn register_and_enumerate() {
        let mut reg = DriverRegistry::new();
        reg.register_factory(Arc::new(TestMetalFactory));
        assert_eq!(reg.factory_count(), 1);

        let all = reg.enumerate_all();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].0, "test-metal");
        assert_eq!(all[0].1.len(), 1);
        assert_eq!(all[0].1[0].name, "Metal (test)");
    }

    #[test]
    fn try_create_from_info_matches_by_name() {
        let mut reg = DriverRegistry::new();
        reg.register_factory(Arc::new(TestMetalFactory));

        let info = DriverInfo {
            name: "Metal (test)".into(),
            version_major: 1,
            version_minor: 0,
            available: true,
            description: "Test Metal backend".into(),
        };

        let outcome = reg.try_create_from_info(&info);
        assert!(outcome.is_some());
        let outcome = outcome.unwrap();
        assert_eq!(outcome.handle, "metal-test-0");
        assert!(outcome.capabilities.contains(&BackendCapability::MatMulF32));
    }

    #[test]
    fn try_create_returns_none_for_unavailable() {
        let mut reg = DriverRegistry::new();
        reg.register_factory(Arc::new(TestMetalFactory));

        let info = DriverInfo {
            name: "Metal (test)".into(),
            version_major: 1,
            version_minor: 0,
            available: false,
            description: "Unavailable".into(),
        };

        assert!(reg.try_create_from_info(&info).is_none());
    }

    #[test]
    fn backend_capability_hash_and_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(BackendCapability::MatMulF32);
        set.insert(BackendCapability::MatMulF16);
        set.insert(BackendCapability::MatMulF32); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn driver_registry_serialize_roundtrip() {
        let mut reg = DriverRegistry::new();
        reg.register_factory(Arc::new(TestMetalFactory));
        assert_eq!(reg.factory_count(), 1);

        // Serialize — factories are skipped, only factory_count remains.
        let json = serde_json::to_string(&reg).unwrap();
        assert!(json.contains("factory_count"));
        assert_eq!(json, r#"{"factory_count":1}"#);

        // Deserialize — factories are lost, registry is empty.
        let reg2: DriverRegistry = serde_json::from_str(&json).unwrap();
        assert_eq!(reg2.factory_count(), 0);
    }

    #[test]
    fn builtin_constants_are_correct() {
        assert_eq!(builtin::METAL, "Metal");
        assert_eq!(builtin::ANE, "ANE");
        assert_eq!(builtin::ACCELERATE, "Accelerate");
        assert_eq!(builtin::CUDA, "CUDA");
        assert_eq!(builtin::ROCM, "ROCm");
        assert_eq!(builtin::TENSIX, "Tensix");
    }
}
