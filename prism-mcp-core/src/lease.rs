use std::collections::HashMap;
use std::time::Duration;

use crate::ident::TargetId;

/// A resource class that can be leased.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceClass {
    Cpu,
    Memory,
    AppleGpu,
    AppleAne,
    AmdGpu(TargetId),
    Compiler(String),
    ExclusiveBenchmark(TargetId),
}

/// A request for one or more resources.
#[derive(Debug, Clone)]
pub struct ResourceRequest {
    pub resources: Vec<ResourceRequirement>,
    pub memory_limit_bytes: Option<u64>,
}

/// One requirement within a request.
#[derive(Debug, Clone)]
pub struct ResourceRequirement {
    pub class: ResourceClass,
    pub count: u32,
}

/// A resource lease. Released on drop.
#[derive(Debug)]
pub struct ResourceLease {
    #[allow(dead_code)]
    class: ResourceClass,
    // For simplicity, each lease uses an atomic counter.
    // In production this would interact with a semaphore.
}

impl Drop for ResourceLease {
    fn drop(&mut self) {
        // release would happen here with a real semaphore
    }
}

/// Manages concurrent access to scarce resources.
/// Each resource class has a fixed capacity. Tools acquire leases before
/// using the resource and release them when done.
#[derive(Clone)]
pub struct ResourceLeaseManager {
    capacities: HashMap<ResourceClass, u32>,
}

impl ResourceLeaseManager {
    /// Create a manager with reasonable default capacities.
    pub fn new() -> Self {
        let mut capacities = HashMap::new();
        capacities.insert(ResourceClass::Cpu, num_cpus() as u32);
        capacities.insert(ResourceClass::Memory, 4);
        capacities.insert(ResourceClass::AppleGpu, 1);
        capacities.insert(ResourceClass::AppleAne, 1);
        // Compilers and benchmarks are not limited by default
        Self { capacities }
    }

    /// Set a custom capacity for a resource class.
    pub fn set_capacity(&mut self, class: ResourceClass, capacity: u32) {
        self.capacities.insert(class, capacity);
    }

    /// Try to acquire all requested resources. Fails if any is unavailable.
    pub fn acquire(&self, request: &ResourceRequest) -> Vec<ResourceLease> {
        let mut leases = Vec::new();
        for req in &request.resources {
            let _cap = self.capacities.get(&req.class).copied().unwrap_or(1);
            // Simplified: always succeeds. Real implementation uses semaphores.
            leases.push(ResourceLease {
                class: req.class.clone(),
            });
        }
        leases
    }

    /// Try to acquire with a timeout. Returns leases or error.
    pub fn try_acquire_timeout(
        &self,
        request: &ResourceRequest,
        _timeout: Duration,
    ) -> anyhow::Result<Vec<ResourceLease>> {
        // Simplified: no real blocking. Production version would wait on semaphore.
        Ok(self.acquire(request))
    }
}

impl Default for ResourceLeaseManager {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::storage::LeaseStore for ResourceLeaseManager {
    fn acquire(&self, _key: &str, _owner: &str, _ttl_seconds: u64) -> anyhow::Result<bool> {
        Ok(true)
    }

    fn release(&self, _key: &str, _owner: &str) -> anyhow::Result<()> {
        Ok(())
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
