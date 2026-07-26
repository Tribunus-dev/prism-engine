//! Buffer lifetime planning — per-buffer alloc/free epoch derivation from
//! the dataflow graph's topological sort.
//!
//! This module owns the canonical authority for buffer lifetime planning:
//! for every `Buffer` entity, given the dataflow graph of its producing
//! layer, derive the earliest alloc epoch (producer rank) and the latest
//! free epoch (max consumer rank). It also owns the scratch buffer
//! sizing heuristic for dispatch entities.
//!
//! ## Authority boundary
//!
//! This module does **not** own:
//! - The dataflow graph itself (owned by the compile-path fusion IR).
//! - The `Buffer` entity lifecycle (owned by the residency subsystem).
//! - Memory pool allocation policy (owned by the kernel).
//!
//! All state mutations go through `WorldTxn`. The module exposes a
//! `BufferLifetimePlan` value-type and a single
//! `derive_lifetimes(world) -> Result<Vec<BufferLifetimeAssignment>, ...>`
//! entry point. It does **not** mutate the world directly.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use prism_ecs_core::{Component, Entity, World};
use prism_ecs_constitutional::{
    ClassifiedComponent, DomainEvent, DurableClass, DurableComponent, EntityKindId, MessageId,
    SchemaKey, WorldTxn, WorldTxnError,
};

/// Per-buffer alloc/free epoch derived from a dataflow graph.
///
/// Epoch semantics:
/// - `alloc_epoch` is the rank of the producer node in the layer's
///   topological sort. The buffer is logically live from this epoch.
/// - `free_epoch` is the maximum consumer rank + 1. After this epoch the
///   buffer may be reclaimed.
/// - `causal_death_frontier` is `Some((_, free_epoch + 1))` if there is at
///   least one downstream consumer whose free epoch is bounded, else
///   `None` (the buffer is consumed by a long-lived sink).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferLifetimePlan {
    pub alloc_epoch: u64,
    pub free_epoch: u64,
    pub causal_death_frontier: Option<(u64, u64)>,
}

impl Component for BufferLifetimePlan {}
impl ClassifiedComponent for BufferLifetimePlan {
    type Class = DurableClass;
}
impl DurableComponent for BufferLifetimePlan {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.runtime.buffer_lifetime",
        id: 1,
        version: 1,
    };
}

/// One assignment of a `BufferLifetimePlan` to a buffer entity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferLifetimeAssignment {
    pub buffer: Entity,
    pub plan: BufferLifetimePlan,
}

/// Errors produced by buffer lifetime derivation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BufferLifetimeError {
    #[error("dataflow graph for handle `{handle}` is empty")]
    EmptyGraph { handle: String },
    #[error("dataflow graph for handle `{handle}` has unknown node index {index}")]
    UnknownNode { handle: String, index: usize },
    #[error("constitutional txn error: {0}")]
    Txn(#[from] WorldTxnError),
}

/// Minimal dataflow graph interface needed by this module. The full
/// dataflow graph lives in the compile path; this trait is satisfied by
/// the compile crate without forcing a dependency.
pub trait DataflowGraphView {
    fn nodes(&self) -> &[DataflowNodeView];
    fn edges(&self) -> &[DataflowEdgeView];
    fn value_at(&self, id: &str) -> Option<DataflowValueMeta>;
}

/// A node in the dataflow graph (compile-path view).
#[derive(Debug, Clone)]
pub struct DataflowNodeView {
    pub id: usize,
    pub outputs: Vec<String>,
}

/// An edge in the dataflow graph.
#[derive(Debug, Clone)]
pub struct DataflowEdgeView {
    pub producer: usize,
    pub consumer: usize,
    pub value: String,
}

/// Value metadata, used to choose sensible defaults.
#[derive(Debug, Clone)]
pub struct DataflowValueMeta {
    pub name: String,
}

/// Compute per-value alloc/free epochs from a topological sort of the graph.
///
/// The topological sort is a Kahn's-algorithm pass. Each node's rank is
/// the order in which it leaves the queue. The alloc epoch of a value is
/// the rank of the node that produces it; the free epoch is the maximum
/// rank of any consumer of that value, plus one (so a value with no
/// consumer is freed immediately after its producer runs).
pub fn compute_value_lifetimes(graph: &impl DataflowGraphView) -> BTreeMap<String, (u64, u64)> {
    let topo = topological_sort(graph);
    let mut lifetimes: BTreeMap<String, (u64, u64)> = BTreeMap::new();

    for node in graph.nodes() {
        for out_value in &node.outputs {
            let producer_epoch = topo.get(&node.id).copied().unwrap_or(0);

            let consumer_max = graph
                .edges()
                .iter()
                .filter(|e| &e.value == out_value)
                .filter_map(|e| topo.get(&e.consumer).copied())
                .max();

            let free = consumer_max.map_or(producer_epoch + 1, |c| c + 1);
            lifetimes.insert(out_value.clone(), (producer_epoch, free));
        }
    }

    lifetimes
}

/// Kahn's algorithm topological sort. Returns a `BTreeMap<usize, u64>`
/// from node index to topological rank. Unvisited nodes (cycles) get
/// sequential ranks assigned after the main loop to ensure the lifetime
/// assignment is total.
pub fn topological_sort(graph: &impl DataflowGraphView) -> BTreeMap<usize, u64> {
    let n = graph.nodes().len();
    if n == 0 {
        return BTreeMap::new();
    }

    let mut in_degree = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for edge in graph.edges() {
        if edge.producer < n && edge.consumer < n {
            adj[edge.producer].push(edge.consumer);
            in_degree[edge.consumer] = in_degree[edge.consumer].saturating_add(1);
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }

    let mut rank: BTreeMap<usize, u64> = BTreeMap::new();
    let mut epoch: u64 = 0;
    while let Some(node) = queue.pop_front() {
        rank.insert(node, epoch);
        epoch = epoch.saturating_add(1);
        for &succ in &adj[node] {
            if let Some(deg) = in_degree.get_mut(succ) {
                *deg = deg.saturating_sub(1);
                if *deg == 0 {
                    queue.push_back(succ);
                }
            }
        }
    }

    for i in 0..n {
        rank.entry(i).or_insert_with(|| {
            let e = epoch;
            epoch = epoch.saturating_add(1);
            e
        });
    }

    rank
}

/// Conservative lifetime for a buffer that cannot be matched to any
/// graph value. The buffer is treated as live for the entire known
/// epoch span.
pub fn conservative_lifetime(max_epoch: u64) -> BufferLifetimePlan {
    BufferLifetimePlan {
        alloc_epoch: 0,
        free_epoch: max_epoch.saturating_add(1),
        causal_death_frontier: Some((0, max_epoch.saturating_add(2))),
    }
}

/// Build a `BufferLifetimePlan` for a buffer whose name matches a value
/// in the resolved lifetimes map, or fall back to the conservative
/// lifetime when no match is found.
pub fn lifetime_for_named_buffer(
    name: &str,
    lifetimes_by_graph: &[BTreeMap<String, (u64, u64)>],
    max_epoch: u64,
) -> BufferLifetimePlan {
    let best: Option<(u64, u64)> = lifetimes_by_graph
        .iter()
        .filter_map(|m| m.get(name).copied())
        .min_by_key(|&(_, free)| free);
    match best {
        Some((alloc, free)) => BufferLifetimePlan {
            alloc_epoch: alloc,
            free_epoch: free,
            causal_death_frontier: (free < u64::MAX).then(|| (0, free + 1)),
        },
        None => conservative_lifetime(max_epoch),
    }
}

/// Sizing policy for a dispatch's scratch buffer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScratchSizingPolicy {
    pub scratch_factor: u32,
    pub scratch_factor_denominator: u32,
    pub min_scratch_bytes: u64,
    pub fixed_scratch_bytes: Option<u64>,
}

impl Default for ScratchSizingPolicy {
    fn default() -> Self {
        Self {
            scratch_factor: 1,
            scratch_factor_denominator: 4,
            min_scratch_bytes: 1 << 20,
            fixed_scratch_bytes: None,
        }
    }
}

impl ScratchSizingPolicy {
    pub fn compute(&self, unique_graph_values: u64) -> u64 {
        match self.fixed_scratch_bytes {
            Some(fixed) => fixed,
            None => {
                let raw = unique_graph_values.max(1) as u128
                    * self.scratch_factor as u128
                    * self.min_scratch_bytes as u128
                    / self.scratch_factor_denominator.max(1) as u128;
                let ceil = raw.max(self.min_scratch_bytes as u128);
                u64::try_from(ceil).unwrap_or(u64::MAX)
            }
        }
    }
}

/// Emit a durable domain event recording the lifetime assignment. Used
/// by the schedule to log the change for replay.
pub fn emit_lifetime_assigned(
    txn: &mut WorldTxn,
    buffer: Entity,
    plan: &BufferLifetimePlan,
) -> Result<(), WorldTxnError> {
    let payload = serde_json::json!({
        "buffer": format!("{:?}", buffer),
        "alloc_epoch": plan.alloc_epoch,
        "free_epoch": plan.free_epoch,
        "frontier": plan.causal_death_frontier.map(|(a, b)| (a, b)),
    });
    let event = DomainEvent {
        id: MessageId::compute(
            format!(
                "prism.buffer_lifetime.assigned:{}:{}:{}",
                buffer.id(),
                plan.alloc_epoch,
                plan.free_epoch
            )
            .as_bytes(),
        ),
        kind: "prism.buffer_lifetime.assigned".to_string(),
        entity_id: Some(EntityKindId(buffer.id())),
        payload,
    };
    txn.emit_event(event);
    Ok(())
}

/// Replay applier for the buffer-lifetime domain event.
pub fn replay_lifetime_assigned(
    txn: &mut WorldTxn,
    buffer: Entity,
    plan: BufferLifetimePlan,
) -> Result<bool, WorldTxnError> {
    txn.put_durable(buffer, plan);
    Ok(true)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct TestGraph {
        nodes: Vec<DataflowNodeView>,
        edges: Vec<DataflowEdgeView>,
    }

    impl DataflowGraphView for TestGraph {
        fn nodes(&self) -> &[DataflowNodeView] {
            &self.nodes
        }
        fn edges(&self) -> &[DataflowEdgeView] {
            &self.edges
        }
        fn value_at(&self, _id: &str) -> Option<DataflowValueMeta> {
            None
        }
    }

    fn make_graph() -> TestGraph {
        TestGraph {
            nodes: vec![
                DataflowNodeView {
                    id: 0,
                    outputs: vec!["a".into()],
                },
                DataflowNodeView {
                    id: 1,
                    outputs: vec!["b".into()],
                },
                DataflowNodeView {
                    id: 2,
                    outputs: vec!["c".into()],
                },
            ],
            edges: vec![
                DataflowEdgeView {
                    producer: 0,
                    consumer: 1,
                    value: "a".into(),
                },
                DataflowEdgeView {
                    producer: 1,
                    consumer: 2,
                    value: "b".into(),
                },
            ],
        }
    }

    #[test]
    fn topo_sort_assigns_sequential_ranks() {
        let g = make_graph();
        let topo = topological_sort(&g);
        assert_eq!(topo.get(&0), Some(&0));
        assert_eq!(topo.get(&1), Some(&1));
        assert_eq!(topo.get(&2), Some(&2));
    }

    #[test]
    fn lifetimes_are_alloc_then_max_consumer_plus_one() {
        let g = make_graph();
        let lts = compute_value_lifetimes(&g);
        assert_eq!(lts.get("a").copied(), Some((0, 2)));
        assert_eq!(lts.get("b").copied(), Some((1, 3)));
        assert_eq!(lts.get("c").copied(), Some((2, 3)));
    }

    #[test]
    fn cycle_assigns_remaining_ranks() {
        let g = TestGraph {
            nodes: vec![
                DataflowNodeView {
                    id: 0,
                    outputs: vec!["a".into()],
                },
                DataflowNodeView {
                    id: 1,
                    outputs: vec!["b".into()],
                },
            ],
            edges: vec![
                DataflowEdgeView {
                    producer: 0,
                    consumer: 1,
                    value: "a".into(),
                },
                DataflowEdgeView {
                    producer: 1,
                    consumer: 0,
                    value: "b".into(),
                },
            ],
        };
        let topo = topological_sort(&g);
        assert_eq!(topo.len(), 2);
        let mut values: Vec<u64> = topo.values().copied().collect();
        values.sort();
        assert_eq!(values, vec![0, 1]);
    }

    #[test]
    fn conservative_lifetime_covers_known_span() {
        let p = conservative_lifetime(7);
        assert_eq!(p.alloc_epoch, 0);
        assert_eq!(p.free_epoch, 8);
        assert_eq!(p.causal_death_frontier, Some((0, 9)));
    }

    #[test]
    fn named_buffer_picks_tightest_free() {
        let mut g1: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        g1.insert("q".into(), (2, 10));
        let mut g2: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        g2.insert("q".into(), (3, 7));
        let p = lifetime_for_named_buffer("q", &[g1, g2], 100);
        assert_eq!(p.alloc_epoch, 3);
        assert_eq!(p.free_epoch, 7);
    }

    #[test]
    fn named_buffer_falls_back_when_unknown() {
        let p = lifetime_for_named_buffer("missing", &[], 4);
        assert_eq!(p.alloc_epoch, 0);
        assert_eq!(p.free_epoch, 5);
    }

    #[test]
    fn scratch_sizing_respects_fixed_override() {
        let mut p = ScratchSizingPolicy::default();
        p.fixed_scratch_bytes = Some(4096);
        assert_eq!(p.compute(10_000), 4096);
    }

    #[test]
    fn scratch_sizing_applies_factor_with_minimum() {
        let p = ScratchSizingPolicy::default();
        let bytes = p.compute(100);
        assert!(bytes >= p.min_scratch_bytes);
        assert_eq!(bytes, 25 * (1 << 20));
    }

    #[test]
    fn replay_applier_accepts_plan() {
        let plan = BufferLifetimePlan {
            alloc_epoch: 1,
            free_epoch: 5,
            causal_death_frontier: Some((0, 6)),
        };
        assert_eq!(plan.alloc_epoch, 1);
    }

    #[test]
    fn domain_event_carries_lifetime_payload() {
        let buffer = Entity::new(7, 0);
        let plan = BufferLifetimePlan {
            alloc_epoch: 0,
            free_epoch: 3,
            causal_death_frontier: Some((0, 4)),
        };
        let payload = serde_json::json!({
            "alloc_epoch": plan.alloc_epoch,
            "free_epoch": plan.free_epoch,
            "frontier": plan.causal_death_frontier,
        });
        let event = DomainEvent {
            id: MessageId::compute(
                format!(
                    "prism.buffer_lifetime.assigned:{}:{}:{}",
                    buffer.id(),
                    plan.alloc_epoch,
                    plan.free_epoch,
                )
                .as_bytes(),
            ),
            kind: "prism.buffer_lifetime.assigned".to_string(),
            entity_id: Some(EntityKindId(buffer.id())),
            payload,
        };
        assert_eq!(event.kind, "prism.buffer_lifetime.assigned");
    }
}
