//! LegacyWorkerBridge system — thin bridge between the legacy engine-facing
//! request API and the ECS worker supervision pipeline.
//!
//! This system wraps the [`LegacyWorkerBridge`] resource into an
//! [`ErasedSystem`] so it participates in the schedule.  The current
//! implementation is a no-op placeholder; real functionality will be
//! wired when the legacy endpoints are connected.

use crate::ecs::runtime::scheduling::command::CommandWriter;
use crate::ecs::runtime::scheduling::metadata::*;
use crate::ecs::runtime::world::World;
use prism_ecs_runtime::scheduling::systems::agent_bridge::AgentBridge;

// ---------------------------------------------------------------------------
// LegacyBridgeSystem
// ---------------------------------------------------------------------------

/// A no-op system placeholder that bridges the legacy engine-facing API
/// to the ECS worker pipeline.
///
/// Runs during `Stage::Intake` at order `-1` (before the main ingress
/// system) so it can populate the ingress queue before
/// [`WorkerIngressSystem`] drains it.
pub struct LegacyBridgeSystem {
    /// Optional constitutional bridge for agent-run creation before
    /// worker spawns.  Defaults to `None`; set via [`with_agent_bridge`].
    pub agent_bridge: Option<AgentBridge>,
}

impl LegacyBridgeSystem {
    /// Create a new legacy bridge system.
    pub fn new() -> Self {
        Self { agent_bridge: None }
    }

    /// Attach an [`AgentBridge`] for constitutional agent-run creation.
    pub fn with_agent_bridge(mut self, bridge: AgentBridge) -> Self {
        self.agent_bridge = Some(bridge);
        self
    }
}

impl Default for LegacyBridgeSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SystemSpec — compile-time declaration
// ---------------------------------------------------------------------------

impl SystemSpec for LegacyBridgeSystem {
    type Reads = ();
    type Writes = ();
    type ReadResources = ();
    type WriteResources = ();

    const NAME: &'static str = "legacy_bridge";
    const ID: SystemId = SystemId(103);
    const STAGE: Stage = Stage::Intake;
    const ORDER: i32 = -1;
    const EXECUTION_CLASS: ExecutionClass = ExecutionClass::Serial;
    const SERIALIZATION: SerializationPolicy = SerializationPolicy::ExplicitOnly;
}

// ---------------------------------------------------------------------------
// ErasedSystem — object-safe runtime
// ---------------------------------------------------------------------------

impl ErasedSystem for LegacyBridgeSystem {
    fn metadata(&self) -> &SystemMetadata {
        static META: std::sync::LazyLock<SystemMetadata> = std::sync::LazyLock::new(|| {
            <LegacyBridgeSystem as SystemSpec>::metadata()
                .expect("LegacyBridgeSystem metadata construction")
        });
        &META
    }

    fn run(&mut self, _world: &mut World, _commands: &mut CommandWriter) -> SystemResult {
        // Wire create_agent_run before worker spawn: when the
        // bridge is available, iterate queued entities and
        // call bridge.create_agent_run(...) with the session/task/config
        // extracted from the runtime world.
        //
        // TODO: extract session_entity, task, config from runtime components
        // and call bridge.create_agent_run(...) for each candidate.
        if self.agent_bridge.is_some() {
            // Bridge is available — wiring is pending entity mapping.
        }

        SystemResult::ok()
    }
}
