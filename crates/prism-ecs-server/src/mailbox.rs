//! RPCS3-inspired SPU mailbox channels — bounded message queues with
//! jostling protocol for inter-entity communication.
//!
//! Cell SPUs communicate through channels (`spu_channel`) that support
//! blocking push/pop with signal/ack semantics. This module adapts that
//! pattern into Prism ECS components and systems:
//!
//! * [`Mailbox`] — a bounded FIFO attached to a mailbox entity. Messages
//!   are opaque byte payloads (`Vec<u8>`). When the mailbox is configured
//!   as `blocking`, a pop on an empty mailbox suspends the receiver (the
//!   entity is added to `waiters`), and a push wakes them.
//! * [`MailboxSender`] — component marking an entity as a sender, pointing
//!   at the mailbox entity.
//! * [`MailboxReceiver`] — component marking an entity as a receiver,
//!   pointing at the mailbox entity.
//! * [`MailboxMessage`] — an incoming message component attached to the
//!   sender entity, consumed by [`MailboxSendSystem`].
//! * [`MailboxSendSystem`] — reads sender entities carrying `MailboxMessage`,
//!   pushes each message into the target mailbox, signals any blocked waiters.
//! * [`MailboxReceiveSystem`] — reads receiver entities, pops a message
//!   from their mailbox (if non-empty), writes it onto the receiver's
//!   `MailboxMessage` component for downstream systems.
//! * [`MailboxWaiterReady`] — a single-shot component attached to a woken
//!   waiter entity, signalling that a message is now available.

use std::collections::VecDeque;

use prism_ecs_core::{Component, Entity, World};

// ---------------------------------------------------------------------------
// Mailbox
// ---------------------------------------------------------------------------

/// A bounded mailbox channel attached to an ECS entity.
///
/// Stores an ordered queue of byte-vector messages. When `blocking` is `true`,
/// receivers that attempt to pop from an empty mailbox register in `waiters`;
/// the next push wakes them by attaching a `MailboxWaiterReady` component.
#[derive(Debug, Clone)]
pub struct Mailbox {
    /// Message buffer — oldest at front, newest at back.
    pub buffer: VecDeque<Vec<u8>>,
    /// Maximum number of messages before a push is dropped.
    pub capacity: usize,
    /// If `true`, a pop on an empty mailbox blocks (registers a waiter).
    pub blocking: bool,
    /// Entities waiting for a message to arrive.
    pub waiters: Vec<Entity>,
}

impl Component for Mailbox {}

impl Mailbox {
    /// Create a new mailbox with the given capacity and blocking mode.
    pub fn new(capacity: usize, blocking: bool) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            blocking,
            waiters: Vec::new(),
        }
    }

    /// Push a message onto the back of the buffer.
    ///
    /// Returns `true` if the message was accepted, `false` if the mailbox
    /// was at capacity (message is silently dropped — SPU jostling fallback).
    #[must_use]
    pub fn push(&mut self, msg: Vec<u8>) -> bool {
        if self.buffer.len() >= self.capacity {
            return false;
        }
        self.buffer.push_back(msg);
        true
    }

    /// Pop a message from the front of the buffer.
    #[must_use]
    pub fn pop(&mut self) -> Option<Vec<u8>> {
        self.buffer.pop_front()
    }

    /// Register a waiter entity.
    pub fn add_waiter(&mut self, entity: Entity) {
        if !self.waiters.contains(&entity) {
            self.waiters.push(entity);
        }
    }

    /// Drain all registered waiters (returns ownership so the caller can signal them).
    pub fn drain_waiters(&mut self) -> Vec<Entity> {
        std::mem::take(&mut self.waiters)
    }
}

// ---------------------------------------------------------------------------
// MailboxSender / MailboxReceiver
// ---------------------------------------------------------------------------

/// Marks an entity as a sender to a specific mailbox.
#[derive(Debug, Clone)]
pub struct MailboxSender {
    /// The mailbox entity to push messages into.
    pub mailbox: Entity,
}

impl Component for MailboxSender {}

/// Marks an entity as a receiver from a specific mailbox.
#[derive(Debug, Clone)]
pub struct MailboxReceiver {
    /// The mailbox entity to pop messages from.
    pub mailbox: Entity,
}

impl Component for MailboxReceiver {}

// ---------------------------------------------------------------------------
// MailboxMessage
// ---------------------------------------------------------------------------

/// A message payload attached to a sender entity awaiting dispatch, or
/// to a receiver entity that has just popped a message.
#[derive(Debug, Clone)]
pub struct MailboxMessage(pub Vec<u8>);

impl Component for MailboxMessage {}

// ---------------------------------------------------------------------------
// MailboxWaiterReady
// ---------------------------------------------------------------------------

/// Single-shot component attached to a waiter entity when a message becomes
/// available in the mailbox it was waiting on.
///
/// Downstream systems (e.g., the receive system) should check for this
/// component to know the entity can now pop without blocking.
#[derive(Debug, Clone)]
pub struct MailboxWaiterReady;

impl Component for MailboxWaiterReady {}

// ---------------------------------------------------------------------------
// MailboxSendSystem
// ---------------------------------------------------------------------------

/// Reads sender entities that carry a `MailboxMessage`, pushes each message
/// into the sender's target `Mailbox`, then signals any blocked waiters
/// by attaching a `MailboxWaiterReady` component.
///
/// Returns the number of messages dispatched this tick.
pub struct MailboxSendSystem;

impl MailboxSendSystem {
    /// Run one tick of mailbox send processing.
    ///
    /// Iterates every entity with both `MailboxSender` and `MailboxMessage`
    /// components.  For each such entity:
    /// 1. Reads the target mailbox entity from `MailboxSender`.
    /// 2. Pushes the message payload into the mailbox buffer.
    /// 3. Signals every waiter by attaching `MailboxWaiterReady`.
    /// 4. Removes the `MailboxMessage` from the sender.
    ///
    /// Silently drops messages when the target mailbox is at capacity
    /// (matching RPCS3 SPU channel jostling semantics).
    pub fn run(world: &mut World) -> usize {
        // Collect senders with pending messages.
        let senders: Vec<Entity> = {
            let mut out = Vec::new();
            if let Some(col) = world.component_store().column::<MailboxSender>() {
                for (entity, _) in col.iter() {
                    // Check the entity also carries a MailboxMessage.
                    if world.component_store().get::<MailboxMessage>(entity)
                        .is_some()
                    {
                        out.push(entity);
                    }
                }
            }
            out
        };

        if senders.is_empty() {
            return 0;
        }

        let mut dispatched = 0;

        for sender in &senders {
            // Read sender and message components (immutable).
            let (target_mailbox, payload) = {
                let sender_comp = match world.component_store().get::<MailboxSender>(*sender) {
                    Some(s) => s,
                    None => continue,
                };
                let msg = match world.component_store().get::<MailboxMessage>(*sender) {
                    Some(m) => m,
                    None => continue,
                };
                (sender_comp.mailbox, msg.0.clone())
            };

            // Push the message into the target mailbox.
            let waiters = {
                let mailbox = match world.component_store_mut().column_mut::<Mailbox>()
                    .get_mut(target_mailbox)
                {
                    Some(m) => m,
                    None => continue,
                };

                if !mailbox.push(payload) {
                    // Mailbox at capacity — drop the message (SPU jostling).
                    // Still remove the sender's message component below.
                }
                mailbox.drain_waiters()
            };

            // Signal each waiter.
            for waiter in &waiters {
                let _ = world.component_store_mut().insert::<MailboxWaiterReady>(*waiter, MailboxWaiterReady);
            }

            // Remove the consumed message from the sender.
            let _ = world.component_store_mut().remove::<MailboxMessage>(*sender);

            dispatched += 1;
        }

        dispatched
    }

    /// Query: how many senders currently have a pending message?
    pub fn pending_count(world: &World) -> usize {
        let senders = match world.component_store().column::<MailboxSender>() {
            Some(c) => c,
            None => return 0,
        };
        let mut count = 0;
        for (entity, _) in senders.iter() {
            if world.component_store().get::<MailboxMessage>(entity)
                .is_some()
            {
                count += 1;
            }
        }
        count
    }
}

// ---------------------------------------------------------------------------
// MailboxReceiveSystem
// ---------------------------------------------------------------------------

/// Reads receiver entities, pops a message from their target mailbox, and
/// writes the result onto the receiver's `MailboxMessage` component.
///
/// When the mailbox is empty and configured as `blocking`, registers the
/// receiver as a waiter (the entity will be woken by a future send).
///
/// Returns the number of messages received this tick.
pub struct MailboxReceiveSystem;

impl MailboxReceiveSystem {
    /// Run one tick of mailbox receive processing.
    ///
    /// Iterates every entity with a `MailboxReceiver` component. For each:
    /// 1. Reads the target mailbox entity.
    /// 2. If the mailbox has messages, pops one and writes it as a
    ///    `MailboxMessage` component on the receiver.
    /// 3. If the mailbox is empty and `blocking`, registers the receiver
    ///    as a waiter.
    pub fn run(world: &mut World) -> usize {
        let receivers: Vec<Entity> = {
            let mut out = Vec::new();
            if let Some(col) = world.component_store().column::<MailboxReceiver>() {
                for (entity, _) in col.iter() {
                    out.push(entity);
                }
            }
            out
        };

        if receivers.is_empty() {
            return 0;
        }

        let mut received = 0;

        for receiver in &receivers {
            let target_mailbox = match world.component_store().get::<MailboxReceiver>(*receiver) {
                Some(r) => r.mailbox,
                None => continue,
            };

            // Pop from the mailbox (requires &mut access to the column).
            let popped = world.component_store_mut().column_mut::<Mailbox>()
                .get_mut(target_mailbox)
                .and_then(|mailbox| {
                    if let Some(msg) = mailbox.pop() {
                        Some(msg)
                    } else if mailbox.blocking {
                        mailbox.add_waiter(*receiver);
                        None
                    } else {
                        None
                    }
                });

            if let Some(payload) = popped {
                let _ = world.component_store_mut().insert::<MailboxMessage>(*receiver, MailboxMessage(payload));
                // Remove the waiter-ready marker if present.
                let _ = world.component_store_mut().remove::<MailboxWaiterReady>(*receiver);
                received += 1;
            }
        }

        received
    }

    /// Query: how many receivers currently have a message available?
    pub fn receiver_ready_count(world: &World) -> usize {
        let receivers = match world.component_store().column::<MailboxReceiver>() {
            Some(c) => c,
            None => return 0,
        };
        let mut count = 0;
        for (entity, _) in receivers.iter() {
            if world.component_store().get::<MailboxMessage>(entity)
                .is_some()
            {
                count += 1;
            }
        }
        count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::EntityKind;

    /// Helper: create a world with a mailbox entity and a sender + receiver.
    fn setup_world() -> (World, Entity, Entity, Entity) {
        let mut world = World::new();

        // Spawn mailbox entity.
        let mb = world
            .spawn(EntityKind::Node, Some("mailbox".into()))
            .unwrap()
            .entity;
        world.add_component(mb, Mailbox::new(16, true)).unwrap();

        // Spawn sender.
        let sender = world
            .spawn(EntityKind::Node, Some("sender".into()))
            .unwrap()
            .entity;
        world
            .add_component(sender, MailboxSender { mailbox: mb })
            .unwrap();

        // Spawn receiver.
        let receiver = world
            .spawn(EntityKind::Node, Some("receiver".into()))
            .unwrap()
            .entity;
        world
            .add_component(receiver, MailboxReceiver { mailbox: mb })
            .unwrap();

        (world, mb, sender, receiver)
    }

    #[test]
    fn mailbox_push_pop() {
        let mut mb = Mailbox::new(4, false);
        assert!(mb.push(b"hello".to_vec()));
        assert!(mb.push(b"world".to_vec()));
        assert_eq!(mb.pop(), Some(b"hello".to_vec()));
        assert_eq!(mb.pop(), Some(b"world".to_vec()));
        assert_eq!(mb.pop(), None);
    }

    #[test]
    fn mailbox_capacity_drops() {
        let mut mb = Mailbox::new(2, false);
        assert!(mb.push(b"a".to_vec()));
        assert!(mb.push(b"b".to_vec()));
        assert!(!mb.push(b"c".to_vec())); // dropped
        assert_eq!(mb.pop(), Some(b"a".to_vec()));
    }

    #[test]
    fn send_system_dispatches_message() {
        let (mut world, _mb, sender, _receiver) = setup_world();

        // Attach a message to the sender.
        world
            .add_component(sender, MailboxMessage(b"ping".to_vec()))
            .unwrap();

        let count = MailboxSendSystem::run(&mut world);
        assert_eq!(count, 1, "one message should be dispatched");

        // Sender should no longer carry the message.
        assert!(world.component_store().get::<MailboxMessage>(sender)
            .is_none());
    }

    #[test]
    fn receive_system_pops_message() {
        let (mut world, mb, _sender, receiver) = setup_world();

        // Manually push a message into the mailbox.
        let _ = world.component_store_mut().column_mut::<Mailbox>()
            .get_mut(mb)
            .unwrap()
            .push(b"pong".to_vec());

        let count = MailboxReceiveSystem::run(&mut world);
        assert_eq!(count, 1, "one message should be received");

        // Receiver should now carry the message.
        let msg = world.component_store().get::<MailboxMessage>(receiver)
            .expect("receiver should have a message");
        assert_eq!(msg.0, b"pong");
    }

    #[test]
    fn blocking_receiver_registers_waiter() {
        let (mut world, mb, _sender, receiver) = setup_world();

        // Mailbox is empty, receiver calls receive.
        let count = MailboxReceiveSystem::run(&mut world);
        assert_eq!(count, 0, "no message received");

        // Receiver should be registered as a waiter.
        let mailbox = world.component_store().get::<Mailbox>(mb)
            .expect("mailbox should exist");
        assert!(
            mailbox.waiters.contains(&receiver),
            "receiver should be a waiter"
        );
    }

    #[test]
    fn send_wakes_blocked_waiter() {
        let (mut world, _mb, sender, receiver) = setup_world();

        // Receiver blocks on empty mailbox.
        MailboxReceiveSystem::run(&mut world);

        // Sender dispatches a message.
        world
            .add_component(sender, MailboxMessage(b"wake".to_vec()))
            .unwrap();
        MailboxSendSystem::run(&mut world);

        // Waiter should now have MailboxWaiterReady.
        assert!(
            world.component_store().get::<MailboxWaiterReady>(receiver)
                .is_some(),
            "waiter should be woken"
        );

        // Mailbox should have the message for the receiver.
        let count = MailboxReceiveSystem::run(&mut world);
        assert_eq!(count, 1, "receiver should now get the message");
        let msg = world.component_store().get::<MailboxMessage>(receiver)
            .expect("receiver should have message");
        assert_eq!(msg.0, b"wake");
    }

    #[test]
    fn no_message_no_dispatches() {
        let (mut world, _mb, _sender, _receiver) = setup_world();
        let count = MailboxSendSystem::run(&mut world);
        assert_eq!(count, 0);
    }

    #[test]
    fn empty_mailbox_no_receives() {
        let (mut world, _mb, _sender, _receiver) = setup_world();
        let count = MailboxReceiveSystem::run(&mut world);
        assert_eq!(count, 0);
    }

    #[test]
    fn pending_count_tracks_messages() {
        let (mut world, _mb, sender, _receiver) = setup_world();
        assert_eq!(MailboxSendSystem::pending_count(&world), 0);

        world
            .add_component(sender, MailboxMessage(b"x".to_vec()))
            .unwrap();
        assert_eq!(MailboxSendSystem::pending_count(&world), 1);

        // After dispatch, pending count should be 0.
        MailboxSendSystem::run(&mut world);
        assert_eq!(MailboxSendSystem::pending_count(&world), 0);
    }
}
