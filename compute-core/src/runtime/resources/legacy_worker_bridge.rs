//! LegacyWorkerBridge — shim between the legacy request path and the ECS world.
//!
//! In Slice 2 this helper creates an entity with the appropriate components
//! and pushes an ingress entry so the normal ECS pipeline processes the
//! request.  The concrete component setup will be wired in a later slice.

use crate::runtime::world::World;

/// Bridge helper that translates legacy-style request submissions into
/// ECS entity creation with ingress queue population.
pub struct LegacyWorkerBridge;

impl LegacyWorkerBridge {
    /// Create a new bridge instance.
    pub fn new() -> Self {
        Self
    }

    /// Submit a request through the legacy bridge path.
    ///
    /// Creates an entity in the world and pushes an ingress entry to the
    /// `WorkerIngressQueue` resource.  Returns the assigned entity ID on
    /// success.
    ///
    /// Placeholder — allocates an entity and stubs the ingress push.
    pub fn submit_request(
        &self,
        world: &mut World,
        _request_id: &str,
        _payload: Vec<u8>,
    ) -> Result<u32, String> {
        // Allocate an entity — real wiring will attach components later.
        let entity = world.spawn().ok_or_else(|| "world at capacity".to_string())?;
        Ok(entity.0)
    }
}

impl Default for LegacyWorkerBridge {
    fn default() -> Self {
        Self::new()
    }
}
