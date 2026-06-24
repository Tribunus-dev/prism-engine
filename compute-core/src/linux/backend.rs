#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Hip,
    LevelZero,
    Vulkan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VendorKind {
    Cpu,
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceStableKey {
    pub vendor: VendorKind,
    pub pci_domain: Option<u16>,
    pub pci_bus: Option<u8>,
    pub pci_device: Option<u8>,
    pub uuid_or_luid: Option<[u8; 16]>,
    pub fallback_fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceId {
    pub backend: BackendKind,
    pub ordinal: u32,
    pub stable_key: DeviceStableKey,
}
