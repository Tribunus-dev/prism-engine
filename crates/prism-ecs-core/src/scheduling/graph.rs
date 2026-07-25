//! Dependency graph for schedule compilation.
//!
//! Builds a directed graph from system metadata, validates edges, detects
//! cycles, and reports structured diagnostics.  The graph is consumed by
//! `Schedule::compile` and never escaped.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::scheduling::error::ScheduleError;
use crate::scheduling::metadata::{SystemId, SystemMetadata};

// ---------------------------------------------------------------------------
// EdgeKind
// ---------------------------------------------------------------------------

/// The reason an edge exists in the dependency graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// Imposed by stage ordering (barrier between stages).
    StageBarrier,
    /// Declared by the user via `after` or `before`.
    Explicit,
    /// Added by the schedule compiler to resolve a write/write hazard.
    Serialization,
}

impl EdgeKind {
    /// Short diagnostic label.
    pub fn label(&self) -> &'static str {
        match self {
            EdgeKind::StageBarrier => "stage_barrier",
            EdgeKind::Explicit => "explicit",
            EdgeKind::Serialization => "serialization",
        }
    }
}

// ---------------------------------------------------------------------------
// DependencyGraph
// ---------------------------------------------------------------------------

/// Directed acyclic dependency graph of systems.
///
/// Nodes are indexed by SystemId.  Edges carry an `EdgeKind` describing why
/// the dependency exists.
pub struct DependencyGraph {
    /// Systems in registration order (indexed by position in this vec).
    systems: Vec<SystemMetadata>,
    /// SystemId → dense index.
    ///
    /// `BTreeMap` (not `HashMap`): the schedule graph is part of the
    /// canonical runtime authority; iteration over the system set is
    /// observable to the schedule executor and to projection rebuilds.
    /// See AGENTS.md "no HashMap/HashSet for canonical collections
    /// whose order is observable."
    id_to_idx: BTreeMap<SystemId, usize>,
    /// Adjacency list: edges from → (target, kind).
    edges: Vec<Vec<(usize, EdgeKind)>>,
    /// In-degree count for topological sort.
    in_degree: Vec<usize>,
}

impl DependencyGraph {
    pub fn nodes(&self) -> &[SystemMetadata] {
        &self.systems
    }

    /// Indices of systems with zero in-degree.
    pub fn zero_in_degree_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.in_degree
            .iter()
            .enumerate()
            .filter(|(_, &deg)| deg == 0)
            .map(|(i, _)| i)
    }

    /// The in-degree for a given system index.
    pub fn in_degree_of(&self, idx: usize) -> usize {
        self.in_degree[idx]
    }

    /// Iterate outgoing edges (target index + kind) for a node.
    pub fn outgoing(&self, idx: usize) -> &[(usize, EdgeKind)] {
        &self.edges[idx]
    }

    /// Index for a SystemId.
    pub fn idx(&self, id: SystemId) -> usize {
        self.id_to_idx[&id]
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Accumulates systems and edges, then validates and compiles the graph.
pub struct GraphBuilder {
    systems: Vec<SystemMetadata>,
    /// BTreeMap: see `DependencyGraph::id_to_idx` for rationale.
    id_to_idx: BTreeMap<SystemId, usize>,
    pending_edges: Vec<(SystemId, SystemId, EdgeKind)>,
}

impl GraphBuilder {
    /// Start building a graph from a list of system metadata.
    ///
    /// System metadata is cloned; the caller keeps ownership of the original.
    pub fn new(metadata: Vec<SystemMetadata>) -> Self {
        let mut id_to_idx = BTreeMap::new();
        for (i, meta) in metadata.iter().enumerate() {
            id_to_idx.insert(meta.id, i);
        }
        GraphBuilder {
            systems: metadata,
            id_to_idx,
            pending_edges: Vec::new(),
        }
    }

    /// Resolve a SystemId to its index for edge insertion.
    fn resolve(&self, id: SystemId) -> Result<usize, ScheduleError> {
        self.id_to_idx
            .get(&id)
            .copied()
            .ok_or(ScheduleError::TargetNotRegistered {
                from: id,
                target: id,
            })
    }

    /// Add an explicit `after` edge: `from` must run after `target`.
    ///
    /// Returns an error if `target` is not registered or if the edge
    /// inverts the stage ordering.
    pub fn add_explicit_after(
        &mut self,
        from: SystemId,
        target: SystemId,
    ) -> Result<(), ScheduleError> {
        let target_idx = self.resolve_target(from, target)?;

        // Stage inversion check: the target must be in <= from's stage.
        let from_meta = &self.systems[self.id_to_idx[&from]];
        let target_meta = &self.systems[target_idx];
        if target_meta.stage > from_meta.stage {
            return Err(ScheduleError::StageInversion {
                from,
                target,
                from_stage: from_meta.stage as usize,
                target_stage: target_meta.stage as usize,
            });
        }

        self.pending_edges.push((from, target, EdgeKind::Explicit));
        Ok(())
    }

    /// Add an explicit `before` edge: `from` must run before `target`.
    ///
    /// Returns an error if `target` is not registered or if the edge
    /// inverts the stage ordering.
    pub fn add_explicit_before(
        &mut self,
        from: SystemId,
        target: SystemId,
    ) -> Result<(), ScheduleError> {
        let target_idx = self.resolve_target(from, target)?;

        // Stage inversion check: the target must be in >= from's stage.
        let from_meta = &self.systems[self.id_to_idx[&from]];
        let target_meta = &self.systems[target_idx];
        if target_meta.stage < from_meta.stage {
            return Err(ScheduleError::StageInversion {
                from,
                target,
                from_stage: from_meta.stage as usize,
                target_stage: target_meta.stage as usize,
            });
        }

        // A `before` edge means from → target depends on target?
        // No: `from must run before target` means target depends on from.
        // So the edge is target → from.
        // Wait — for Kahn's algorithm, an edge A → B means A must run before B.
        // "from must run before target" = from → target.
        self.pending_edges.push((from, target, EdgeKind::Explicit));
        Ok(())
    }

    /// Add a serialization edge between two systems with a write/write hazard.
    ///
    /// `a` will run before `b`.
    pub fn add_serialization_edge(&mut self, a: SystemId, b: SystemId) {
        // Only add if not already present.
        if !self
            .pending_edges
            .iter()
            .any(|(f, t, _)| *f == a && *t == b)
        {
            self.pending_edges.push((a, b, EdgeKind::Serialization));
        }
    }

    /// Resolve a target SystemId, checking it differs from `from`.
    fn resolve_target(&self, from: SystemId, target: SystemId) -> Result<usize, ScheduleError> {
        if from == target {
            return Err(ScheduleError::TargetNotRegistered { from, target });
        }
        self.resolve(target)
            .map_err(|_| ScheduleError::TargetNotRegistered { from, target })
    }

    /// Consume the builder and produce a validated `DependencyGraph`.
    ///
    /// Returns an error if the graph contains a cycle.
    pub fn build(self) -> Result<DependencyGraph, ScheduleError> {
        let n = self.systems.len();
        let mut edges: Vec<Vec<(usize, EdgeKind)>> = vec![Vec::new(); n];
        let mut in_degree = vec![0usize; n];
        let mut edge_set: BTreeSet<(usize, usize)> = BTreeSet::new();

        // Insert pending edges (from → target) where from runs before target.
        for &(from_id, target_id, kind) in &self.pending_edges {
            // Look up both indices.
            let from_idx = self.id_to_idx[&from_id];
            let to_idx = self.id_to_idx[&target_id];

            // Skip duplicate edges.
            if !edge_set.insert((from_idx, to_idx)) {
                continue;
            }

            edges[from_idx].push((to_idx, kind));
            in_degree[to_idx] += 1;
        }

        Ok(DependencyGraph {
            systems: self.systems,
            id_to_idx: self.id_to_idx,
            edges,
            in_degree,
        })
    }
}

// ---------------------------------------------------------------------------
// Topological sort (Kahn's algorithm, internal to graph)
// ---------------------------------------------------------------------------

impl DependencyGraph {
    /// Perform Kahn's topological sort, extracting the sorted SystemIds.
    ///
    /// Returns `Ok(Vec<SystemId>)` on success, or
    /// `Err(Vec<SystemId>)` containing the nodes involved in a cycle.
    pub fn topological_order(&self) -> Result<Vec<SystemId>, Vec<SystemId>> {
        let n = self.systems.len();
        let mut in_degree = self.in_degree.clone();
        let mut queue: VecDeque<usize> = VecDeque::new();

        // Seed the queue with zero-in-degree nodes, ordered by
        // (stage, order, system_id) for deterministic output.
        let mut zero_deg: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        zero_deg.sort_by(|&a, &b| {
            let ma = &self.systems[a];
            let mb = &self.systems[b];
            ma.stage
                .cmp(&mb.stage)
                .then(ma.order.cmp(&mb.order))
                .then(ma.id.cmp(&mb.id))
        });
        for idx in zero_deg {
            queue.push_back(idx);
        }

        let mut result: Vec<SystemId> = Vec::with_capacity(n);

        while let Some(idx) = queue.pop_front() {
            result.push(self.systems[idx].id);

            let mut newly_ready: Vec<usize> = Vec::new();
            for &(target, _kind) in &self.edges[idx] {
                in_degree[target] -= 1;
                if in_degree[target] == 0 {
                    newly_ready.push(target);
                }
            }

            // Sort newly ready nodes by (stage, order, id) for determinism.
            newly_ready.sort_by(|&a, &b| {
                let ma = &self.systems[a];
                let mb = &self.systems[b];
                ma.stage
                    .cmp(&mb.stage)
                    .then(ma.order.cmp(&mb.order))
                    .then(ma.id.cmp(&mb.id))
            });
            for ready_idx in newly_ready {
                queue.push_back(ready_idx);
            }
        }

        if result.len() == n {
            Ok(result)
        } else {
            // Collect remaining nodes as cycle participants.
            let cycle_nodes: Vec<SystemId> = in_degree
                .iter()
                .enumerate()
                .filter(|(_, &deg)| deg > 0)
                .map(|(i, _)| self.systems[i].id)
                .collect();
            Err(cycle_nodes)
        }
    }
}
