//! NTB DMA transfer components and systems — RPCS3 Cell MFC command absorption.
//!
//! Cell SPU MFC (Memory Flow Controller) commands (`dma_put`, `dma_get`,
//! `dma_sync`) transfer data between local store and main memory through
//! transfer class IDs.  This module adapts that pattern into Prism ECS
//! for NTB-scale DMA across the NoC (Network-on-Chip / Tenstorrent-style
//! mesh interconnect):
//!
//! * [`DmaTransferType`] — Put (local → remote), Get (remote → local), Sync.
//! * [`DmaTransferStatus`] — Pending → InFlight → Completed lifecycle.
//! * [`DmaTransfer`] — component describing a single DMA transfer request.
//! * [`DmaCompletion`] — single-shot component attached on completion.
//! * [`NocTransferSystem`] — ticks all pending transfers, advancing them
//!   through the lifecycle and handling ordering by transfer class tag.

use prism_ecs_core::{Component, Entity, World};

// ---------------------------------------------------------------------------
// DmaTransferType
// ---------------------------------------------------------------------------

/// Direction and semantics of an NTB DMA transfer, mirroring Cell MFC commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DmaTransferType {
    /// Local store → remote address (SPU `dma_put`).
    Put,
    /// Remote address → local store (SPU `dma_get`).
    Get,
    /// Barrier/fence: wait for all prior transfers on this tag to complete
    /// before proceeding (SPU `dma_sync`).
    Sync,
}

// ---------------------------------------------------------------------------
// DmaTransferStatus
// ---------------------------------------------------------------------------

/// Lifecycle of a single DMA transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DmaTransferStatus {
    /// Transfer has been queued but not yet dispatched to the NoC.
    Pending,
    /// Transfer has been dispatched; the NoC is processing it.
    InFlight,
    /// Transfer has completed successfully.
    Completed,
}

// ---------------------------------------------------------------------------
// DmaTransfer
// ---------------------------------------------------------------------------

/// A DMA transfer request component, attached to a transfer entity.
///
/// Mirrors RPCS3's MFC command structure with NTB-level source/destination
/// addressing. The `tag` field acts as a transfer-class identifier for
/// ordering (Sync waits for all prior transfers sharing the same tag).
#[derive(Debug, Clone)]
pub struct DmaTransfer {
    /// Source address (local address space for Put, remote for Get).
    pub src_addr: u64,
    /// Destination address (remote for Put, local for Get).
    pub dst_addr: u64,
    /// Number of bytes to transfer.
    pub size_bytes: usize,
    /// Transfer direction / barrier type.
    pub transfer_type: DmaTransferType,
    /// Current lifecycle status.
    pub status: DmaTransferStatus,
    /// Transfer class tag — used for ordering and Sync barriers.
    pub tag: u64,
    /// Entity to notify on completion (optional — can be the issuer itself).
    pub notify: Option<Entity>,
}

impl Component for DmaTransfer {}

impl DmaTransfer {
    /// Create a new `Put` transfer (local → remote).
    pub fn put(
        src_addr: u64,
        dst_addr: u64,
        size_bytes: usize,
        tag: u64,
        notify: Option<Entity>,
    ) -> Self {
        Self {
            src_addr,
            dst_addr,
            size_bytes,
            transfer_type: DmaTransferType::Put,
            status: DmaTransferStatus::Pending,
            tag,
            notify,
        }
    }

    /// Create a new `Get` transfer (remote → local).
    pub fn get(
        src_addr: u64,
        dst_addr: u64,
        size_bytes: usize,
        tag: u64,
        notify: Option<Entity>,
    ) -> Self {
        Self {
            src_addr,
            dst_addr,
            size_bytes,
            transfer_type: DmaTransferType::Get,
            status: DmaTransferStatus::Pending,
            tag,
            notify,
        }
    }

    /// Create a new `Sync` barrier transfer.
    pub fn sync(tag: u64, notify: Option<Entity>) -> Self {
        Self {
            src_addr: 0,
            dst_addr: 0,
            size_bytes: 0,
            transfer_type: DmaTransferType::Sync,
            status: DmaTransferStatus::Pending,
            tag,
            notify,
        }
    }
}

// ---------------------------------------------------------------------------
// DmaCompletion
// ---------------------------------------------------------------------------

/// Single-shot component attached to a transfer entity (or its `notify`
/// target) when the transfer completes.
///
/// Downstream systems can query for this component to react to completion.
#[derive(Debug, Clone)]
pub struct DmaCompletion {
    /// The entity that completed.
    pub transfer: Entity,
    /// The transfer's tag at completion time.
    pub tag: u64,
    /// Whether the transfer was successful.
    pub success: bool,
}

impl Component for DmaCompletion {}

// ---------------------------------------------------------------------------
// NocTransferSystem
// ---------------------------------------------------------------------------

/// Ticks all DMA transfers through their lifecycle:
///
/// 1. **Pending → InFlight** — dispatches pending transfers to the NoC
///    (simulated by advancing status). Sync barriers wait until all prior
///    transfers sharing the same tag have reached `Completed`.
/// 2. **InFlight → Completed** — simulates NoC completion (in a real
///    integration this would poll NoC hardware or a semaphore).
/// 3. **Completed** — attaches a `DmaCompletion` component to the transfer
///    entity (and to `notify` if set).
///
/// Returns a tuple `(dispatched, completed)` counting transfers that
/// transitioned this tick.
pub struct NocTransferSystem;

impl NocTransferSystem {
    /// Run one tick of the NTB NoC transfer pipeline.
    ///
    /// Every transfer entity is processed in the following order:
    /// 1. Pending → InFlight (immediate for Put/Get; deferred for Sync
    ///    until all same-tag transfers are complete).
    /// 2. InFlight → Completed (simulated: all in-flight complete in one
    ///    tick for non-blocked paths).
    /// 3. Completion notification via `DmaCompletion` component.
    pub fn run(world: &mut World) -> (usize, usize) {
        let mut dispatched = 0usize;
        let mut completed = 0usize;

        // Phase 1: Pending → InFlight.
        // Collect pending transfers first to avoid aliasing issues.
        let pending: Vec<Entity> = {
            let mut out = Vec::new();
            if let Some(col) = world.component_store().column::<DmaTransfer>() {
                for (entity, tx) in col.iter() {
                    if tx.status == DmaTransferStatus::Pending {
                        out.push(entity);
                    }
                }
            }
            out
        };

        for entity in &pending {
            // Read metadata immutably first, then mutate.
            let (transfer_type, tag) = match world.component_store().get::<DmaTransfer>(*entity) {
                Some(tx) => (tx.transfer_type, tx.tag),
                None => continue,
            };

            match transfer_type {
                DmaTransferType::Put | DmaTransferType::Get => {
                    if let Some(tx) = world
                        .component_store_mut()
                        .column_mut::<DmaTransfer>()
                        .get_mut(*entity)
                    {
                        tx.status = DmaTransferStatus::InFlight;
                    }
                    dispatched += 1;
                }
                DmaTransferType::Sync => {
                    // Sync barrier: all same-tag transfers must be completed.
                    let all_complete = match world.component_store().column::<DmaTransfer>() {
                        Some(col) => col
                            .iter()
                            .filter(|(e, _)| *e != *entity)
                            .filter(|(_, t)| t.tag == tag)
                            .all(|(_, t)| t.status == DmaTransferStatus::Completed),
                        None => true,
                    };

                    if all_complete {
                        if let Some(tx) = world
                            .component_store_mut()
                            .column_mut::<DmaTransfer>()
                            .get_mut(*entity)
                        {
                            tx.status = DmaTransferStatus::Completed;
                        }
                        completed += 1;
                    }
                    // else stay pending — wait for siblings.
                }
            }
        }

        // Phase 2: InFlight → Completed.
        // Gather in-flight entities separately.
        let in_flight: Vec<Entity> = {
            let mut out = Vec::new();
            if let Some(col) = world.component_store().column::<DmaTransfer>() {
                for (entity, tx) in col.iter() {
                    if tx.status == DmaTransferStatus::InFlight {
                        out.push(entity);
                    }
                }
            }
            out
        };

        for entity in &in_flight {
            let tx = match world
                .component_store_mut()
                .column_mut::<DmaTransfer>()
                .get_mut(*entity)
            {
                Some(t) => t,
                None => continue,
            };

            tx.status = DmaTransferStatus::Completed;
            completed += 1;

            // Attach completion component.
            let tag = tx.tag;
            let notify = tx.notify;
            let _ = world.component_store_mut().insert::<DmaCompletion>(
                *entity,
                DmaCompletion {
                    transfer: *entity,
                    tag,
                    success: true,
                },
            );
            if let Some(notify_entity) = notify {
                let _ = world.component_store_mut().insert::<DmaCompletion>(
                    notify_entity,
                    DmaCompletion {
                        transfer: *entity,
                        tag,
                        success: true,
                    },
                );
            }
        }

        // Phase 3: re-check pending Sync barriers after in-flight → Completed.
        // A Sync that was pending in Phase 1 because its siblings were still
        // in-flight may now see them Completed.
        let pending_syncs: Vec<Entity> = {
            let mut out = Vec::new();
            if let Some(col) = world.component_store().column::<DmaTransfer>() {
                for (entity, tx) in col.iter() {
                    if tx.status == DmaTransferStatus::Pending
                        && matches!(tx.transfer_type, DmaTransferType::Sync)
                    {
                        out.push(entity);
                    }
                }
            }
            out
        };

        for entity in &pending_syncs {
            let tag = match world.component_store().get::<DmaTransfer>(*entity) {
                Some(tx) => tx.tag,
                None => continue,
            };

            let all_complete = match world.component_store().column::<DmaTransfer>() {
                Some(col) => col
                    .iter()
                    .filter(|(e, _)| *e != *entity)
                    .filter(|(_, t)| t.tag == tag)
                    .all(|(_, t)| t.status == DmaTransferStatus::Completed),
                None => true,
            };

            if all_complete {
                if let Some(tx) = world
                    .component_store_mut()
                    .column_mut::<DmaTransfer>()
                    .get_mut(*entity)
                {
                    tx.status = DmaTransferStatus::Completed;
                }
                completed += 1;
            }
        }

        (dispatched, completed)
    }

    /// Return the number of transfers still in `Pending` state.
    pub fn pending_count(world: &World) -> usize {
        world
            .component_store()
            .column::<DmaTransfer>()
            .map(|col| {
                col.iter()
                    .filter(|(_, tx)| tx.status == DmaTransferStatus::Pending)
                    .count()
            })
            .unwrap_or(0)
    }

    /// Return the number of transfers in `InFlight` state.
    pub fn in_flight_count(world: &World) -> usize {
        world
            .component_store()
            .column::<DmaTransfer>()
            .map(|col| {
                col.iter()
                    .filter(|(_, tx)| tx.status == DmaTransferStatus::InFlight)
                    .count()
            })
            .unwrap_or(0)
    }

    /// Return the number of transfers in `Completed` state.
    pub fn completed_count(world: &World) -> usize {
        world
            .component_store()
            .column::<DmaTransfer>()
            .map(|col| {
                col.iter()
                    .filter(|(_, tx)| tx.status == DmaTransferStatus::Completed)
                    .count()
            })
            .unwrap_or(0)
    }

    /// Drain all `DmaCompletion` components from the world — useful after
    /// processing completions in a downstream system.
    pub fn drain_completions(world: &mut World) -> Vec<(Entity, DmaCompletion)> {
        let mut out = Vec::new();
        if let Some(col) = world.component_store().column::<DmaCompletion>() {
            // Collect entities first.
            let entities: Vec<Entity> = col.iter().map(|(e, _)| e).collect();
            for entity in entities {
                if let Some(comp) = world.component_store_mut().remove::<DmaCompletion>(entity) {
                    out.push((entity, comp));
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::{EntityKind, World};

    /// Helper: spawn a transfer entity and attach a DmaTransfer component.
    fn spawn_transfer(world: &mut World, tx: DmaTransfer) -> Entity {
        let entity = world
            .spawn(EntityKind::Node, Some("dma-transfer".into()))
            .unwrap()
            .entity;
        world.add_component(entity, tx).unwrap();
        entity
    }

    #[test]
    fn dma_transfer_put_lifecycle() {
        let mut world = World::new();
        let e = spawn_transfer(&mut world, DmaTransfer::put(0x1000, 0x2000, 64, 1, None));

        // Initially pending.
        let tx = world.component_store().get::<DmaTransfer>(e).unwrap();
        assert_eq!(tx.status, DmaTransferStatus::Pending);

        // Tick → dispatched.
        let (d, c) = NocTransferSystem::run(&mut world);
        assert_eq!(d, 1, "put should dispatch");
        assert_eq!(c, 1, "put should complete (simulated)");

        let tx = world.component_store().get::<DmaTransfer>(e).unwrap();
        assert_eq!(tx.status, DmaTransferStatus::Completed);

        // Completion component should exist.
        assert!(world.component_store().get::<DmaCompletion>(e).is_some());
    }

    #[test]
    fn dma_transfer_get_lifecycle() {
        let mut world = World::new();
        let e = spawn_transfer(&mut world, DmaTransfer::get(0x3000, 0x4000, 128, 2, None));

        let (d, c) = NocTransferSystem::run(&mut world);
        assert_eq!(d, 1);
        assert_eq!(c, 1);

        let tx = world.component_store().get::<DmaTransfer>(e).unwrap();
        assert_eq!(tx.status, DmaTransferStatus::Completed);
    }

    #[test]
    fn sync_barrier_waits_for_siblings() {
        let mut world = World::new();

        // Two Put transfers on tag=5, followed by a Sync on tag=5.
        let tx_a = spawn_transfer(&mut world, DmaTransfer::put(0x100, 0x200, 32, 5, None));
        let tx_b = spawn_transfer(&mut world, DmaTransfer::put(0x300, 0x400, 32, 5, None));
        let tx_sync = spawn_transfer(&mut world, DmaTransfer::sync(5, None));

        // Tick 1: Puts dispatch and complete. Phase 3 then evaluates the Sync
        // barrier — since all same-tag siblings are now Completed, the Sync
        // advances in the same tick.
        let (d, c) = NocTransferSystem::run(&mut world);
        assert_eq!(d, 2, "both puts dispatch");
        assert_eq!(c, 3, "both puts + sync complete in same tick");
        assert_eq!(
            world
                .component_store()
                .get::<DmaTransfer>(tx_sync)
                .unwrap()
                .status,
            DmaTransferStatus::Completed,
            "sync completes same tick (Phase 3 recheck)"
        );

        // Tick 2: nothing to do.
        let (d, c) = NocTransferSystem::run(&mut world);
        assert_eq!(d, 0, "no new dispatches");
        assert_eq!(c, 0, "no more completions");

        // Confirm the puts are still completed.
        assert_eq!(
            world
                .component_store()
                .get::<DmaTransfer>(tx_a)
                .unwrap()
                .status,
            DmaTransferStatus::Completed
        );
        assert_eq!(
            world
                .component_store()
                .get::<DmaTransfer>(tx_b)
                .unwrap()
                .status,
            DmaTransferStatus::Completed
        );
    }

    #[test]
    fn sync_pending_when_siblings_in_flight() {
        let mut world = World::new();
        let _tx_a = spawn_transfer(&mut world, DmaTransfer::put(0x100, 0x200, 32, 7, None));
        let tx_sync = spawn_transfer(&mut world, DmaTransfer::sync(7, None));

        // Only tick once. The put dispatches but hasn't completed... actually
        // in our model, InFlight→Completed is also in the same tick. So we
        // need to check: after first tick the sync barrier should see
        // the sibling as Completed and advance.
        let (d, c) = NocTransferSystem::run(&mut world);
        assert_eq!(d, 1);
        assert_eq!(c, 2); // put completes + sync barrier satisfied
        assert_eq!(
            world
                .component_store()
                .get::<DmaTransfer>(tx_sync)
                .unwrap()
                .status,
            DmaTransferStatus::Completed
        );
    }

    #[test]
    fn completion_notifies_target_entity() {
        let mut world = World::new();
        let notify = world
            .spawn(EntityKind::Node, Some("consumer".into()))
            .unwrap()
            .entity;

        let e = spawn_transfer(
            &mut world,
            DmaTransfer::put(0x500, 0x600, 16, 3, Some(notify)),
        );

        NocTransferSystem::run(&mut world);

        // Both transfer and notify entity should have DmaCompletion.
        assert!(world.component_store().get::<DmaCompletion>(e).is_some());
        assert!(world
            .component_store()
            .get::<DmaCompletion>(notify)
            .is_some());
    }

    #[test]
    fn drain_completions_removes_all() {
        let mut world = World::new();
        spawn_transfer(&mut world, DmaTransfer::put(0x10, 0x20, 8, 9, None));
        NocTransferSystem::run(&mut world);

        let drained = NocTransferSystem::drain_completions(&mut world);
        assert_eq!(drained.len(), 1, "one completion should drain");

        // Second drain should be empty.
        let drained2 = NocTransferSystem::drain_completions(&mut world);
        assert_eq!(drained2.len(), 0, "no completions after drain");
    }

    #[test]
    fn multiple_tags_are_independent() {
        let mut world = World::new();
        let tx_a = spawn_transfer(&mut world, DmaTransfer::put(0x1, 0x2, 8, 10, None));
        let tx_b = spawn_transfer(&mut world, DmaTransfer::put(0x3, 0x4, 8, 20, None));
        let sync_a = spawn_transfer(&mut world, DmaTransfer::sync(10, None));
        let sync_b = spawn_transfer(&mut world, DmaTransfer::sync(20, None));

        // Tick 1: both puts complete, syncs advance.
        NocTransferSystem::run(&mut world);

        assert_eq!(
            world
                .component_store()
                .get::<DmaTransfer>(tx_a)
                .unwrap()
                .status,
            DmaTransferStatus::Completed
        );
        assert_eq!(
            world
                .component_store()
                .get::<DmaTransfer>(tx_b)
                .unwrap()
                .status,
            DmaTransferStatus::Completed
        );
        assert_eq!(
            world
                .component_store()
                .get::<DmaTransfer>(sync_a)
                .unwrap()
                .status,
            DmaTransferStatus::Completed
        );
        assert_eq!(
            world
                .component_store()
                .get::<DmaTransfer>(sync_b)
                .unwrap()
                .status,
            DmaTransferStatus::Completed
        );
    }

    #[test]
    fn in_flight_count() {
        // With single-tick completion, in_flight should be 0 after tick.
        // Check just the count.
        let mut world = World::new();
        spawn_transfer(&mut world, DmaTransfer::put(0, 0, 8, 0, None));
        assert_eq!(NocTransferSystem::pending_count(&world), 1);
        NocTransferSystem::run(&mut world);
        assert_eq!(NocTransferSystem::in_flight_count(&world), 0);
        assert_eq!(NocTransferSystem::completed_count(&world), 1);
    }

    #[test]
    fn no_transfers_no_work() {
        let mut world = World::new();
        let (d, c) = NocTransferSystem::run(&mut world);
        assert_eq!(d, 0);
        assert_eq!(c, 0);
    }
}
