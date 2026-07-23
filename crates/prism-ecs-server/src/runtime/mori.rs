//! Bounded Mori-derived ECS control-plane slice.
//!
//! This module keeps residency and route lifecycle state in the canonical
//! [`prism_ecs_core::World`]. It is deliberately an adapter: it does not
//! replace the existing runtime managers or introduce a second authority.
//!
//! The lifecycle is explicit so a caller can stage metadata before material
//! is visible to an execution lane, acquire session leases only after the
//! residency is live, and drain before eviction. Route descriptors follow the
//! same pattern and are addressed by a normalized capability key.

use prism_ecs_core::{Component, Entity, EntityKind, World, WorldError};
use serde::{Deserialize, Serialize};

use super::backend::ExecutionRecipe;
use super::manifest::{ExecutionLane, InferencePhase, SessionId};
use super::server_types::WeightResidencyKey;

pub const SCHEMA_MORI_RESIDENCY: u64 = 71;
pub const SCHEMA_MORI_ROUTE: u64 = 72;

/// The only legal residency progression for a Mori residency entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MoriResidencyStage {
    Declared,
    Staged,
    Resident,
    Draining,
    Evicted,
}

/// Lifecycle of a capability-keyed route descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MoriRouteStage {
    Staged,
    Active,
    Draining,
    Revoked,
}

/// Canonicalized capability set used as a route lookup key.
///
/// Whitespace is removed, empty capabilities are discarded, and remaining
/// values are sorted and deduplicated. This makes route identity independent
/// of the order in which a provider reports capabilities.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MoriCapabilityKey(Vec<String>);

impl MoriCapabilityKey {
    pub fn new<I, S>(capabilities: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut capabilities: Vec<String> = capabilities
            .into_iter()
            .map(Into::into)
            .map(|capability| capability.trim().to_string())
            .filter(|capability| !capability.is_empty())
            .collect();
        capabilities.sort_unstable();
        capabilities.dedup();
        Self(capabilities)
    }

    pub fn capabilities(&self) -> &[String] {
        &self.0
    }
}

/// ECS component describing one staged or resident weight allocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoriResidency {
    pub key: WeightResidencyKey,
    pub stage: MoriResidencyStage,
    pub byte_length: u64,
    /// Active session leases. A session can hold at most one lease per
    /// residency entity; the list is the canonical lease set.
    pub active_leases: Vec<SessionId>,
}

impl Component for MoriResidency {}

/// ECS component describing one capability-keyed execution route.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoriRouteDescriptor {
    pub capability_key: MoriCapabilityKey,
    pub stage: MoriRouteStage,
    pub lane: ExecutionLane,
    pub phase: InferencePhase,
    pub recipe: ExecutionRecipe,
}

impl Component for MoriRouteDescriptor {}

/// Stateless lifecycle operations over the canonical ECS world.
pub struct MoriEcs;

impl MoriEcs {
    /// Declare a residency allocation before any bytes are staged.
    pub fn declare_residency(
        world: &mut World,
        key: WeightResidencyKey,
        byte_length: u64,
    ) -> Result<Entity, String> {
        let entity = world
            .spawn(EntityKind::Residency, Some("mori-residency".into()))
            .map_err(world_error)?
            .entity;
        world
            .add_component(
                entity,
                MoriResidency {
                    key,
                    stage: MoriResidencyStage::Declared,
                    byte_length,
                    active_leases: Vec::new(),
                },
            )
            .map_err(world_error)?;
        Ok(entity)
    }

    /// Move a declared allocation into the staging window.
    pub fn stage_residency(world: &mut World, entity: Entity) -> Result<(), String> {
        Self::transition_residency(world, entity, MoriResidencyStage::Staged)
    }

    /// Mark staged bytes visible to the runtime lanes.
    pub fn mark_resident(world: &mut World, entity: Entity) -> Result<(), String> {
        Self::transition_residency(world, entity, MoriResidencyStage::Resident)
    }

    /// Acquire a session lease. Acquisition is idempotent for one session.
    pub fn acquire_lease(
        world: &mut World,
        entity: Entity,
        session_id: SessionId,
    ) -> Result<(), String> {
        let residency = world
            .component_mut::<MoriResidency>(entity)
            .map_err(world_error)?;
        if residency.stage != MoriResidencyStage::Resident {
            return Err(format!(
                "cannot acquire residency lease while stage is {:?}",
                residency.stage
            ));
        }
        if !residency.active_leases.contains(&session_id) {
            residency.active_leases.push(session_id);
        }
        Ok(())
    }

    /// Release a session lease and require the caller to own it.
    pub fn release_lease(
        world: &mut World,
        entity: Entity,
        session_id: SessionId,
    ) -> Result<(), String> {
        let residency = world
            .component_mut::<MoriResidency>(entity)
            .map_err(world_error)?;
        let Some(index) = residency
            .active_leases
            .iter()
            .position(|lease| *lease == session_id)
        else {
            return Err("session does not hold a residency lease".into());
        };
        residency.active_leases.swap_remove(index);
        Ok(())
    }

    /// Begin draining a resident allocation. Leases must be released first.
    pub fn begin_residency_drain(world: &mut World, entity: Entity) -> Result<(), String> {
        let residency = world
            .component_mut::<MoriResidency>(entity)
            .map_err(world_error)?;
        if residency.stage != MoriResidencyStage::Resident {
            return Err(format!(
                "cannot drain residency while stage is {:?}",
                residency.stage
            ));
        }
        if !residency.active_leases.is_empty() {
            return Err("cannot drain residency with active session leases".into());
        }
        residency.stage = MoriResidencyStage::Draining;
        Ok(())
    }

    /// Complete eviction after the allocation has entered the drain stage.
    pub fn evict_residency(world: &mut World, entity: Entity) -> Result<(), String> {
        Self::transition_residency(world, entity, MoriResidencyStage::Evicted)
    }

    /// Stage a route descriptor. Capability keys are unique within a world.
    pub fn stage_route(
        world: &mut World,
        capabilities: impl IntoIterator<Item = impl Into<String>>,
        lane: ExecutionLane,
        phase: InferencePhase,
        recipe: ExecutionRecipe,
    ) -> Result<Entity, String> {
        let capability_key = MoriCapabilityKey::new(capabilities);
        if world
            .query::<MoriRouteDescriptor>()
            .any(|(_, route)| route.capability_key == capability_key)
        {
            return Err("route capability key already exists".into());
        }

        let entity = world
            .spawn(EntityKind::Dispatch, Some("mori-route".into()))
            .map_err(world_error)?
            .entity;
        world
            .add_component(
                entity,
                MoriRouteDescriptor {
                    capability_key,
                    stage: MoriRouteStage::Staged,
                    lane,
                    phase,
                    recipe,
                },
            )
            .map_err(world_error)?;
        Ok(entity)
    }

    pub fn activate_route(world: &mut World, entity: Entity) -> Result<(), String> {
        Self::transition_route(world, entity, MoriRouteStage::Active)
    }

    pub fn begin_route_drain(world: &mut World, entity: Entity) -> Result<(), String> {
        Self::transition_route(world, entity, MoriRouteStage::Draining)
    }

    pub fn revoke_route(world: &mut World, entity: Entity) -> Result<(), String> {
        Self::transition_route(world, entity, MoriRouteStage::Revoked)
    }

    /// Resolve only active descriptors; staged and draining routes are not
    /// dispatchable. The returned descriptor remains borrowed from `world`.
    pub fn active_route<'a>(
        world: &'a World,
        capabilities: impl IntoIterator<Item = impl Into<String>>,
    ) -> Option<(Entity, &'a MoriRouteDescriptor)> {
        let capability_key = MoriCapabilityKey::new(capabilities);
        world.query::<MoriRouteDescriptor>().find(|(_, route)| {
            route.stage == MoriRouteStage::Active && route.capability_key == capability_key
        })
    }

    fn transition_residency(
        world: &mut World,
        entity: Entity,
        next: MoriResidencyStage,
    ) -> Result<(), String> {
        let residency = world
            .component_mut::<MoriResidency>(entity)
            .map_err(world_error)?;
        let valid = matches!(
            (residency.stage, next),
            (MoriResidencyStage::Declared, MoriResidencyStage::Staged)
                | (MoriResidencyStage::Staged, MoriResidencyStage::Resident)
                | (MoriResidencyStage::Draining, MoriResidencyStage::Evicted)
        );
        if !valid {
            return Err(format!(
                "invalid residency transition {:?} -> {:?}",
                residency.stage, next
            ));
        }
        if next == MoriResidencyStage::Evicted && !residency.active_leases.is_empty() {
            return Err("cannot evict residency with active session leases".into());
        }
        residency.stage = next;
        Ok(())
    }

    fn transition_route(
        world: &mut World,
        entity: Entity,
        next: MoriRouteStage,
    ) -> Result<(), String> {
        let route = world
            .component_mut::<MoriRouteDescriptor>(entity)
            .map_err(world_error)?;
        let valid = matches!(
            (route.stage, next),
            (MoriRouteStage::Staged, MoriRouteStage::Active)
                | (MoriRouteStage::Active, MoriRouteStage::Draining)
                | (MoriRouteStage::Draining, MoriRouteStage::Revoked)
        );
        if !valid {
            return Err(format!(
                "invalid route transition {:?} -> {:?}",
                route.stage, next
            ));
        }
        route.stage = next;
        Ok(())
    }
}

fn world_error(error: WorldError) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::backend::BackendKind;
    use uuid::Uuid;

    fn key() -> WeightResidencyKey {
        WeightResidencyKey {
            cimage_digest: super::super::server_types::ArtifactDigest("model".into()),
            tensor_manifest_digest: super::super::server_types::ArtifactDigest("weights".into()),
            provider_kind: "runtime:llm".into(),
            dtype_profile: "fp16".into(),
        }
    }

    fn session() -> SessionId {
        SessionId(Uuid::from_u128(7))
    }

    #[test]
    fn residency_requires_staging_and_releases_before_eviction() {
        let mut world = World::new();
        let residency = MoriEcs::declare_residency(&mut world, key(), 4096).unwrap();

        assert_eq!(
            world.component::<MoriResidency>(residency).unwrap().stage,
            MoriResidencyStage::Declared
        );
        assert!(MoriEcs::acquire_lease(&mut world, residency, session()).is_err());

        MoriEcs::stage_residency(&mut world, residency).unwrap();
        MoriEcs::mark_resident(&mut world, residency).unwrap();
        MoriEcs::acquire_lease(&mut world, residency, session()).unwrap();
        MoriEcs::acquire_lease(&mut world, residency, session()).unwrap();
        assert_eq!(
            world
                .component::<MoriResidency>(residency)
                .unwrap()
                .active_leases
                .len(),
            1
        );
        assert!(MoriEcs::begin_residency_drain(&mut world, residency).is_err());

        MoriEcs::release_lease(&mut world, residency, session()).unwrap();
        assert!(MoriEcs::release_lease(&mut world, residency, session()).is_err());
        MoriEcs::begin_residency_drain(&mut world, residency).unwrap();
        MoriEcs::evict_residency(&mut world, residency).unwrap();
        assert_eq!(
            world.component::<MoriResidency>(residency).unwrap().stage,
            MoriResidencyStage::Evicted
        );
    }

    #[test]
    fn routes_are_normalized_unique_and_only_active_routes_resolve() {
        let mut world = World::new();
        let route = MoriEcs::stage_route(
            &mut world,
            [" metal ", "decode", "metal"],
            ExecutionLane::Metal,
            InferencePhase::Decode,
            ExecutionRecipe {
                backend: BackendKind::Native,
                ..ExecutionRecipe::default()
            },
        )
        .unwrap();

        assert!(MoriEcs::active_route(&world, ["decode", "metal"]).is_none());
        MoriEcs::activate_route(&mut world, route).unwrap();
        let (resolved, descriptor) = MoriEcs::active_route(&world, ["metal", "decode"]).unwrap();
        assert_eq!(resolved, route);
        assert_eq!(
            descriptor.capability_key.capabilities(),
            ["decode", "metal"]
        );
        assert!(MoriEcs::stage_route(
            &mut world,
            ["decode", "metal"],
            ExecutionLane::Metal,
            InferencePhase::Decode,
            ExecutionRecipe::default(),
        )
        .is_err());

        MoriEcs::begin_route_drain(&mut world, route).unwrap();
        assert!(MoriEcs::active_route(&world, ["metal", "decode"]).is_none());
        MoriEcs::revoke_route(&mut world, route).unwrap();
    }
}
