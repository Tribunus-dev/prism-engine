use std::collections::HashMap;
use std::sync::Mutex;
use prism_ecs_core::memory_tier::MemoryTier;

pub struct TieredMemoryHierarchy { allocations: Mutex<HashMap<(MemoryTier, String), u64>> }
impl TieredMemoryHierarchy {
    pub fn new() -> Self { Self { allocations: Mutex::new(HashMap::new()) } }
    pub fn allocate(&self, tier: MemoryTier, key: &str, bytes: u64) -> Result<(), String> { self.allocations.lock().map_err(|e| e.to_string())?.insert((tier, key.to_string()), bytes); Ok(()) }
    pub fn free(&self, tier: MemoryTier, key: &str) { if let Ok(mut a) = self.allocations.lock() { a.remove(&(tier, key.to_string())); } }
}
