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
use uuid::Uuid;

use super::backend::ExecutionRecipe;
use super::manifest::{ExecutionLane, InferencePhase, SessionId};
use super::server_types::WeightResidencyKey;

pub const SCHEMA_MORI_RESIDENCY: u64 = 71;
pub const SCHEMA_MORI_ROUTE: u64 = 72;
pub const SCHEMA_MORI_TRANSFER_SESSION: u64 = 73;
pub const SCHEMA_MORI_TRANSFER_RECEIPT: u64 = 74;

/// The only legal residency progression for a Mori residency entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MoriResidencyStage {
    Pending,
    Reserved,
    Streaming,
    Resident,
    Evicting,
    Evicted,
}

/// Lifecycle state of the transfer session that owns a residency transition.
///
/// The active states intentionally mirror [`MoriResidencyStage`]. Terminal
/// recovery states are retained on the transfer entity so an interrupted
/// copy is observable and retryable instead of silently disappearing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MoriTransferState {
    Pending,
    Reserved,
    Streaming,
    Resident,
    Evicting,
    Evicted,
    Expired,
    RecoveryRequired,
}

/// Stable identifier for a single Mori transfer session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MoriTransferId(pub Uuid);

/// A copy pin held while a transfer may still read or write the allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MoriCopyPin {
    pub transfer_id: MoriTransferId,
    pub expires_at_ms: u64,
}

/// Metadata retained when a transfer expires or cannot be safely completed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoriRecoveryMetadata {
    pub reason: String,
    pub observed_at_ms: u64,
    pub retryable: bool,
    pub bytes_transferred: u64,
}

/// Durable receipt for a transfer or eviction lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoriTransferReceipt {
    pub transfer_id: MoriTransferId,
    pub residency_entity: Entity,
    pub state: MoriTransferState,
    pub bytes_requested: u64,
    pub bytes_transferred: u64,
    pub started_at_ms: u64,
    pub completed_at_ms: u64,
    pub recovery: Option<MoriRecoveryMetadata>,
}

impl Component for MoriTransferReceipt {}

/// ECS component for one transfer session.
///
/// The component is attached to a session entity and linked back from its
/// [`MoriResidency`] component. Keeping both links in the canonical world
/// makes recovery and cleanup queryable without a second manager-owned map.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoriTransferSession {
    pub transfer_id: MoriTransferId,
    pub residency_entity: Entity,
    pub state: MoriTransferState,
    pub bytes_requested: u64,
    pub bytes_transferred: u64,
    pub started_at_ms: u64,
    pub expires_at_ms: u64,
    pub recovery: Option<MoriRecoveryMetadata>,
}

impl Component for MoriTransferSession {}

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
    /// Transfer session linked to the current pending/resident lifecycle.
    pub transfer_session: Option<Entity>,
    /// Copy pins prevent eviction while a transfer still owns the bytes.
    pub copy_pins: Vec<MoriCopyPin>,
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
                    stage: MoriResidencyStage::Pending,
                    byte_length,
                    active_leases: Vec::new(),
                    transfer_session: None,
                    copy_pins: Vec::new(),
                },
            )
            .map_err(world_error)?;
        Ok(entity)
    }

    /// Reserve a pending allocation for one transfer session.
    pub fn reserve_residency(world: &mut World, entity: Entity) -> Result<(), String> {
        if world
            .component::<MoriResidency>(entity)
            .map_err(world_error)?
            .transfer_session
            .is_some()
        {
            return Err("residency is owned by a transfer session".into());
        }
        Self::transition_residency(world, entity, MoriResidencyStage::Reserved)
    }

    /// Compatibility spelling for callers that used the original staging API.
    pub fn stage_residency(world: &mut World, entity: Entity) -> Result<(), String> {
        Self::reserve_residency(world, entity)
    }

    /// Enter the copy window for a reserved allocation.
    pub fn begin_streaming(world: &mut World, entity: Entity) -> Result<(), String> {
        if world
            .component::<MoriResidency>(entity)
            .map_err(world_error)?
            .transfer_session
            .is_some()
        {
            return Err("residency is owned by a transfer session".into());
        }
        Self::transition_residency(world, entity, MoriResidencyStage::Streaming)
    }

    /// Mark copied bytes visible to runtime lanes.
    ///
    /// A direct `Reserved -> Resident` transition is retained for existing
    /// callers that use the high-level residency API. Transfer sessions use
    /// `begin_streaming`, progress, and `complete_transfer` for the strict
    /// path that preserves copy-pin semantics.
    pub fn mark_resident(world: &mut World, entity: Entity) -> Result<(), String> {
        let residency = world
            .component_mut::<MoriResidency>(entity)
            .map_err(world_error)?;
        if residency.transfer_session.is_some() {
            return Err("complete the linked transfer session instead".into());
        }
        if !matches!(
            residency.stage,
            MoriResidencyStage::Reserved | MoriResidencyStage::Streaming
        ) {
            return Err(format!(
                "cannot mark residency resident while stage is {:?}",
                residency.stage
            ));
        }
        residency.stage = MoriResidencyStage::Resident;
        Ok(())
    }

    /// Start a transfer session in the canonical world.
    pub fn begin_transfer(
        world: &mut World,
        residency_entity: Entity,
        now_ms: u64,
        ttl_ms: u64,
    ) -> Result<(Entity, MoriTransferId), String> {
        let transfer_id = MoriTransferId(Uuid::new_v4());
        let transfer_entity =
            Self::begin_transfer_with_id(world, residency_entity, transfer_id, now_ms, ttl_ms)?;
        Ok((transfer_entity, transfer_id))
    }

    /// Deterministic form of [`Self::begin_transfer`] for replay and tests.
    pub fn begin_transfer_with_id(
        world: &mut World,
        residency_entity: Entity,
        transfer_id: MoriTransferId,
        now_ms: u64,
        ttl_ms: u64,
    ) -> Result<Entity, String> {
        let residency = world
            .component::<MoriResidency>(residency_entity)
            .map_err(world_error)?;
        if residency.stage != MoriResidencyStage::Pending || residency.transfer_session.is_some() {
            return Err("residency is not available for a new transfer".into());
        }
        if world
            .query::<MoriTransferSession>()
            .any(|(_, session)| session.transfer_id == transfer_id)
        {
            return Err("transfer id is already present in the world".into());
        }
        let bytes_requested = residency.byte_length;
        let expires_at_ms = now_ms
            .checked_add(ttl_ms)
            .ok_or_else(|| "transfer TTL overflow".to_string())?;
        let transfer_entity = world
            .spawn(EntityKind::Session, Some("mori-transfer-session".into()))
            .map_err(world_error)?
            .entity;
        world
            .add_component(
                transfer_entity,
                MoriTransferSession {
                    transfer_id,
                    residency_entity,
                    state: MoriTransferState::Pending,
                    bytes_requested,
                    bytes_transferred: 0,
                    started_at_ms: now_ms,
                    expires_at_ms,
                    recovery: None,
                },
            )
            .map_err(world_error)?;
        world
            .component_mut::<MoriResidency>(residency_entity)
            .map_err(world_error)?
            .transfer_session = Some(transfer_entity);
        Ok(transfer_entity)
    }

    /// Reserve the allocation and bind the transfer session to it.
    pub fn reserve_transfer(world: &mut World, transfer_entity: Entity) -> Result<(), String> {
        let session = world
            .component::<MoriTransferSession>(transfer_entity)
            .map_err(world_error)?
            .clone();
        if session.state != MoriTransferState::Pending {
            return Err(format!(
                "cannot reserve transfer while state is {:?}",
                session.state
            ));
        }
        let residency = world
            .component_mut::<MoriResidency>(session.residency_entity)
            .map_err(world_error)?;
        if residency.stage != MoriResidencyStage::Pending
            || residency.transfer_session != Some(transfer_entity)
        {
            return Err("transfer is not linked to a pending residency".into());
        }
        residency.stage = MoriResidencyStage::Reserved;
        world
            .component_mut::<MoriTransferSession>(transfer_entity)
            .map_err(world_error)?
            .state = MoriTransferState::Reserved;
        Ok(())
    }

    /// Begin streaming and acquire the transfer's copy pin atomically in ECS.
    pub fn begin_transfer_streaming(
        world: &mut World,
        transfer_entity: Entity,
    ) -> Result<(), String> {
        let session = world
            .component::<MoriTransferSession>(transfer_entity)
            .map_err(world_error)?
            .clone();
        if session.state != MoriTransferState::Reserved {
            return Err(format!(
                "cannot stream transfer while state is {:?}",
                session.state
            ));
        }
        let residency = world
            .component_mut::<MoriResidency>(session.residency_entity)
            .map_err(world_error)?;
        if residency.stage != MoriResidencyStage::Reserved
            || residency.transfer_session != Some(transfer_entity)
        {
            return Err("transfer is not linked to a reserved residency".into());
        }
        residency.stage = MoriResidencyStage::Streaming;
        if !residency
            .copy_pins
            .iter()
            .any(|pin| pin.transfer_id == session.transfer_id)
        {
            residency.copy_pins.push(MoriCopyPin {
                transfer_id: session.transfer_id,
                expires_at_ms: session.expires_at_ms,
            });
        }
        world
            .component_mut::<MoriTransferSession>(transfer_entity)
            .map_err(world_error)?
            .state = MoriTransferState::Streaming;
        Ok(())
    }

    /// Record a monotonic copy-progress update for a streaming session.
    pub fn record_transfer_progress(
        world: &mut World,
        transfer_entity: Entity,
        bytes: u64,
    ) -> Result<(), String> {
        let session = world
            .component_mut::<MoriTransferSession>(transfer_entity)
            .map_err(world_error)?;
        if session.state != MoriTransferState::Streaming {
            return Err(format!(
                "cannot record transfer progress while state is {:?}",
                session.state
            ));
        }
        let next = session
            .bytes_transferred
            .checked_add(bytes)
            .ok_or_else(|| "transfer progress overflow".to_string())?;
        if next > session.bytes_requested {
            return Err(format!(
                "transfer progress {next} exceeds requested bytes {}",
                session.bytes_requested
            ));
        }
        session.bytes_transferred = next;
        Ok(())
    }

    /// Complete a streaming transfer, release its copy pin, and write a
    /// receipt component on the transfer entity.
    pub fn complete_transfer(
        world: &mut World,
        transfer_entity: Entity,
        completed_at_ms: u64,
    ) -> Result<MoriTransferReceipt, String> {
        let session = world
            .component::<MoriTransferSession>(transfer_entity)
            .map_err(world_error)?
            .clone();
        if session.state != MoriTransferState::Streaming {
            return Err(format!(
                "cannot complete transfer while state is {:?}",
                session.state
            ));
        }
        if completed_at_ms > session.expires_at_ms {
            return Err("transfer TTL expired; run cleanup_expired before recovery".into());
        }
        if session.bytes_transferred != session.bytes_requested {
            return Err(format!(
                "transfer is incomplete: {} of {} bytes copied",
                session.bytes_transferred, session.bytes_requested
            ));
        }
        let residency = world
            .component_mut::<MoriResidency>(session.residency_entity)
            .map_err(world_error)?;
        if residency.stage != MoriResidencyStage::Streaming
            || residency.transfer_session != Some(transfer_entity)
        {
            return Err("transfer is not linked to a streaming residency".into());
        }
        residency.stage = MoriResidencyStage::Resident;
        residency
            .copy_pins
            .retain(|pin| pin.transfer_id != session.transfer_id);
        let receipt = MoriTransferReceipt {
            transfer_id: session.transfer_id,
            residency_entity: session.residency_entity,
            state: MoriTransferState::Resident,
            bytes_requested: session.bytes_requested,
            bytes_transferred: session.bytes_transferred,
            started_at_ms: session.started_at_ms,
            completed_at_ms,
            recovery: None,
        };
        let session_component = world
            .component_mut::<MoriTransferSession>(transfer_entity)
            .map_err(world_error)?;
        session_component.state = MoriTransferState::Resident;
        session_component.recovery = None;
        world
            .add_component(transfer_entity, receipt.clone())
            .map_err(world_error)?;
        Ok(receipt)
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

    /// Acquire an explicit copy pin for a transfer already linked to a
    /// residency. Normally [`Self::begin_transfer_streaming`] owns this.
    pub fn acquire_copy_pin(
        world: &mut World,
        entity: Entity,
        transfer_id: MoriTransferId,
        expires_at_ms: u64,
    ) -> Result<(), String> {
        let residency = world
            .component_mut::<MoriResidency>(entity)
            .map_err(world_error)?;
        if !matches!(
            residency.stage,
            MoriResidencyStage::Reserved | MoriResidencyStage::Streaming
        ) {
            return Err("copy pins are only valid before residency completes".into());
        }
        if !residency
            .copy_pins
            .iter()
            .any(|pin| pin.transfer_id == transfer_id)
        {
            residency.copy_pins.push(MoriCopyPin {
                transfer_id,
                expires_at_ms,
            });
        }
        Ok(())
    }

    /// Release a copy pin. Releasing an unknown pin is an error so recovery
    /// cannot accidentally clear another transfer's protection.
    pub fn release_copy_pin(
        world: &mut World,
        entity: Entity,
        transfer_id: MoriTransferId,
    ) -> Result<(), String> {
        let residency = world
            .component_mut::<MoriResidency>(entity)
            .map_err(world_error)?;
        let before = residency.copy_pins.len();
        residency
            .copy_pins
            .retain(|pin| pin.transfer_id != transfer_id);
        if residency.copy_pins.len() == before {
            return Err("transfer does not hold a copy pin".into());
        }
        Ok(())
    }

    /// Begin evicting a resident allocation. Leases and copy pins must be
    /// released first.
    pub fn begin_evicting(world: &mut World, entity: Entity) -> Result<(), String> {
        {
            let residency = world
                .component_mut::<MoriResidency>(entity)
                .map_err(world_error)?;
            if residency.stage != MoriResidencyStage::Resident {
                return Err(format!(
                    "cannot evict residency while stage is {:?}",
                    residency.stage
                ));
            }
            if !residency.active_leases.is_empty() {
                return Err("cannot evict residency with active session leases".into());
            }
            if !residency.copy_pins.is_empty() {
                return Err("cannot evict residency with active copy pins".into());
            }
            residency.stage = MoriResidencyStage::Evicting;
        }
        Ok(())
    }

    /// Compatibility spelling for callers that used the original drain API.
    pub fn begin_residency_drain(world: &mut World, entity: Entity) -> Result<(), String> {
        Self::begin_evicting(world, entity)
    }

    /// Complete eviction after the allocation has entered the evicting stage.
    pub fn complete_eviction(
        world: &mut World,
        entity: Entity,
        completed_at_ms: u64,
    ) -> Result<Option<MoriTransferReceipt>, String> {
        let transfer_entity = {
            let residency = world
                .component_mut::<MoriResidency>(entity)
                .map_err(world_error)?;
            if residency.stage != MoriResidencyStage::Evicting {
                return Err(format!(
                    "cannot complete eviction while stage is {:?}",
                    residency.stage
                ));
            }
            if !residency.active_leases.is_empty() || !residency.copy_pins.is_empty() {
                return Err("cannot complete eviction while residency is protected".into());
            }
            residency.stage = MoriResidencyStage::Evicted;
            residency.transfer_session
        };
        let Some(transfer_entity) = transfer_entity else {
            return Ok(None);
        };
        let session = world
            .component::<MoriTransferSession>(transfer_entity)
            .map_err(world_error)?
            .clone();
        let receipt = MoriTransferReceipt {
            transfer_id: session.transfer_id,
            residency_entity: entity,
            state: MoriTransferState::Evicted,
            bytes_requested: session.bytes_requested,
            bytes_transferred: session.bytes_transferred,
            started_at_ms: session.started_at_ms,
            completed_at_ms,
            recovery: session.recovery,
        };
        world
            .component_mut::<MoriTransferSession>(transfer_entity)
            .map_err(world_error)?
            .state = MoriTransferState::Evicted;
        world
            .add_component(transfer_entity, receipt.clone())
            .map_err(world_error)?;
        Ok(Some(receipt))
    }

    /// Compatibility spelling for callers that used the original eviction API.
    pub fn evict_residency(world: &mut World, entity: Entity) -> Result<(), String> {
        Self::complete_eviction(world, entity, 0).map(|_| ())
    }

    /// Expire in-flight transfer sessions and retain a recovery receipt in
    /// the ECS. Completed resident sessions are intentionally not TTL-cleaned.
    pub fn cleanup_expired(
        world: &mut World,
        now_ms: u64,
    ) -> Result<Vec<MoriTransferReceipt>, String> {
        let expired: Vec<Entity> = world
            .query::<MoriTransferSession>()
            .filter(|(_, session)| {
                matches!(
                    session.state,
                    MoriTransferState::Pending
                        | MoriTransferState::Reserved
                        | MoriTransferState::Streaming
                        | MoriTransferState::Evicting
                ) && session.expires_at_ms <= now_ms
            })
            .map(|(entity, _)| entity)
            .collect();
        let mut receipts = Vec::with_capacity(expired.len());
        for transfer_entity in expired {
            let session = world
                .component::<MoriTransferSession>(transfer_entity)
                .map_err(world_error)?
                .clone();
            let mut recovery = None;
            let final_state = match session.state {
                MoriTransferState::Pending | MoriTransferState::Reserved => {
                    Self::release_expired_reservation(world, &session, transfer_entity)?;
                    MoriTransferState::Expired
                }
                MoriTransferState::Streaming => {
                    Self::release_expired_reservation(world, &session, transfer_entity)?;
                    recovery = Some(MoriRecoveryMetadata {
                        reason: "streaming transfer TTL expired before residency became resident"
                            .into(),
                        observed_at_ms: now_ms,
                        retryable: true,
                        bytes_transferred: session.bytes_transferred,
                    });
                    MoriTransferState::RecoveryRequired
                }
                MoriTransferState::Evicting => {
                    recovery = Some(MoriRecoveryMetadata {
                        reason: "eviction TTL expired; allocation requires reconciliation".into(),
                        observed_at_ms: now_ms,
                        retryable: false,
                        bytes_transferred: session.bytes_transferred,
                    });
                    MoriTransferState::RecoveryRequired
                }
                _ => continue,
            };
            let receipt = MoriTransferReceipt {
                transfer_id: session.transfer_id,
                residency_entity: session.residency_entity,
                state: final_state,
                bytes_requested: session.bytes_requested,
                bytes_transferred: session.bytes_transferred,
                started_at_ms: session.started_at_ms,
                completed_at_ms: now_ms,
                recovery: recovery.clone(),
            };
            let session_component = world
                .component_mut::<MoriTransferSession>(transfer_entity)
                .map_err(world_error)?;
            session_component.state = final_state;
            session_component.recovery = recovery;
            world
                .add_component(transfer_entity, receipt.clone())
                .map_err(world_error)?;
            receipts.push(receipt);
        }
        Ok(receipts)
    }

    fn release_expired_reservation(
        world: &mut World,
        session: &MoriTransferSession,
        transfer_entity: Entity,
    ) -> Result<(), String> {
        let residency = world
            .component_mut::<MoriResidency>(session.residency_entity)
            .map_err(world_error)?;
        if residency.transfer_session == Some(transfer_entity) {
            residency
                .copy_pins
                .retain(|pin| pin.transfer_id != session.transfer_id);
            if matches!(
                residency.stage,
                MoriResidencyStage::Reserved | MoriResidencyStage::Streaming
            ) {
                residency.stage = MoriResidencyStage::Pending;
            }
            residency.transfer_session = None;
        }
        Ok(())
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
            (MoriResidencyStage::Pending, MoriResidencyStage::Reserved)
                | (MoriResidencyStage::Reserved, MoriResidencyStage::Streaming)
                | (MoriResidencyStage::Streaming, MoriResidencyStage::Resident)
                | (MoriResidencyStage::Evicting, MoriResidencyStage::Evicted)
        );
        if !valid {
            return Err(format!(
                "invalid residency transition {:?} -> {:?}",
                residency.stage, next
            ));
        }
        if next == MoriResidencyStage::Evicted
            && (!residency.active_leases.is_empty() || !residency.copy_pins.is_empty())
        {
            return Err("cannot evict residency while it is protected".into());
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
            MoriResidencyStage::Pending
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
    fn transfer_session_tracks_copy_pin_receipt_and_eviction_protection() {
        let mut world = World::new();
        let residency = MoriEcs::declare_residency(&mut world, key(), 4096).unwrap();
        let transfer_id = MoriTransferId(Uuid::from_u128(8));
        let transfer =
            MoriEcs::begin_transfer_with_id(&mut world, residency, transfer_id, 100, 1_000)
                .unwrap();

        MoriEcs::reserve_transfer(&mut world, transfer).unwrap();
        assert_eq!(
            world.component::<MoriResidency>(residency).unwrap().stage,
            MoriResidencyStage::Reserved
        );
        MoriEcs::begin_transfer_streaming(&mut world, transfer).unwrap();
        assert_eq!(
            world
                .component::<MoriResidency>(residency)
                .unwrap()
                .copy_pins
                .len(),
            1
        );
        assert!(MoriEcs::begin_evicting(&mut world, residency).is_err());

        MoriEcs::record_transfer_progress(&mut world, transfer, 4096).unwrap();
        let receipt = MoriEcs::complete_transfer(&mut world, transfer, 200).unwrap();
        assert_eq!(receipt.state, MoriTransferState::Resident);
        assert_eq!(receipt.bytes_transferred, 4096);
        assert!(world
            .component::<MoriTransferReceipt>(transfer)
            .unwrap()
            .recovery
            .is_none());
        assert!(world
            .component::<MoriResidency>(residency)
            .unwrap()
            .copy_pins
            .is_empty());

        MoriEcs::acquire_lease(&mut world, residency, session()).unwrap();
        assert!(MoriEcs::begin_evicting(&mut world, residency).is_err());
        MoriEcs::release_lease(&mut world, residency, session()).unwrap();
        MoriEcs::begin_evicting(&mut world, residency).unwrap();
        let eviction = MoriEcs::complete_eviction(&mut world, residency, 300)
            .unwrap()
            .expect("transfer receipt should be updated on eviction");
        assert_eq!(eviction.state, MoriTransferState::Evicted);
        assert_eq!(
            world
                .component::<MoriTransferSession>(transfer)
                .unwrap()
                .state,
            MoriTransferState::Evicted
        );
    }

    #[test]
    fn ttl_cleanup_releases_reservation_and_retains_streaming_recovery() {
        let mut world = World::new();
        let residency = MoriEcs::declare_residency(&mut world, key(), 1024).unwrap();
        let transfer = MoriEcs::begin_transfer_with_id(
            &mut world,
            residency,
            MoriTransferId(Uuid::from_u128(9)),
            100,
            10,
        )
        .unwrap();
        MoriEcs::reserve_transfer(&mut world, transfer).unwrap();

        let receipts = MoriEcs::cleanup_expired(&mut world, 110).unwrap();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].state, MoriTransferState::Expired);
        assert_eq!(
            world.component::<MoriResidency>(residency).unwrap().stage,
            MoriResidencyStage::Pending
        );
        assert!(world
            .component::<MoriResidency>(residency)
            .unwrap()
            .transfer_session
            .is_none());

        let residency = MoriEcs::declare_residency(&mut world, key(), 1024).unwrap();
        let transfer = MoriEcs::begin_transfer_with_id(
            &mut world,
            residency,
            MoriTransferId(Uuid::from_u128(10)),
            200,
            10,
        )
        .unwrap();
        MoriEcs::reserve_transfer(&mut world, transfer).unwrap();
        MoriEcs::begin_transfer_streaming(&mut world, transfer).unwrap();
        MoriEcs::record_transfer_progress(&mut world, transfer, 256).unwrap();

        let receipts = MoriEcs::cleanup_expired(&mut world, 211).unwrap();
        assert_eq!(receipts[0].state, MoriTransferState::RecoveryRequired);
        assert_eq!(
            receipts[0]
                .recovery
                .as_ref()
                .expect("recovery metadata")
                .bytes_transferred,
            256
        );
        assert!(world
            .component::<MoriResidency>(residency)
            .unwrap()
            .copy_pins
            .is_empty());
        assert_eq!(
            world
                .component::<MoriTransferSession>(transfer)
                .unwrap()
                .state,
            MoriTransferState::RecoveryRequired
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
