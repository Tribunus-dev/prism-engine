#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MemoryTier {
    UnifiedCpu,
    Device,
    Host,
    Persistent,
}
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResidencyLeaseRecord {
    pub allocation_key: String,
    pub tier: MemoryTier,
    pub bytes: u64,
    pub owner: String,
}
impl ResidencyLeaseRecord {
    pub fn new(allocation_key: String, tier: MemoryTier, bytes: u64, owner: String) -> Self {
        Self {
            allocation_key,
            tier,
            bytes,
            owner,
        }
    }
}
