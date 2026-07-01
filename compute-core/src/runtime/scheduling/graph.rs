//! Dependency graph for schedule compilation.
//!
//! Builds a directed graph from system metadata, validates edges, detects
//! cycles, and reports structured diagnostics.  The graph is consumed by
//! `Schedule::compile` and never escaped.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::runtime::scheduling::error::ScheduleError;
use crate::runtime::scheduling::metadata::{SystemId, SystemMetadata};

// ---------------------------------------------------------------------------
// EdgeKind
// ---------------------------------------------------------------------------

/// The reason an edge exists in the dependency graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// Imposed by stage ordering (barrier between stages).
    StageBarrier,
    /// Explicit `after` declaration.
    ExplicitAfter,
    /// Explicit `before` declaration.
    ExplicitBefore,
    /// Write-read data dependency from mask overlap (only for
    /// producer-consumer with explicit `after` — never inferred).
    Data,
    /// Deterministic serialization of a `StableOrder` hazard.
    Serialization,
}

impl EdgeKind {
    pub fn is_explicit(&self) -> bool {
        matches!(self, EdgeKind::ExplicitAfter | EdgeKind::ExplicitBefore)
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
    /// SystemId → position in the systems array.
    id_to_idx: HashMap<SystemId, usize>,
    /// Systems indexed by position.
    systems: Vec<SystemMetadata>,
    /// Adjacency list: `edges[from_idx] = Vec<(to_idx, EdgeKind)>`.
    edges: Vec<Vec<(usize, EdgeKind)>>,
    /// Pre-computed in-degree for each node.
    in_degree: Vec<usize>,
}

impl DependencyGraph {
    /// Number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }

    /// Metadata for the system at index.
    pub fn system_at(&self, idx: usize) -> &SystemMetadata {
        &self.systems[idx]
    }

    /// Index of a system by ID.
    pub fn index_of(&self, id: &SystemId) -> Option<usize> {
        self.id_to_idx.get(id).copied()
    }

    /// Edges originating from the node at `from_idx`.
    pub fn outgoing_edges(&self, from_idx: usize) -> &[(usize, EdgeKind)] {
        if from_idx < self.edges.len() {
            &self.edges[from_idx]
        } else {
            &[]
        }
    }

    /// In-degree (number of incoming edges) for the node at `idx`.
    pub fn in_degree_of(&self, idx: usize) -> usize {
        self.in_degree.get(idx).copied().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Accumulates systems and edges, then validates and compiles the graph.
pub struct GraphBuilder {
    systems: Vec<SystemMetadata>,
    id_to_idx: HashMap<SystemId, usize>,
    edges: Vec<Vec<(usize, EdgeKind)>>,
}

impl GraphBuilder {
    /// Start building a graph with the given systems.
    pub fn new(systems: Vec<SystemMetadata>) -> Self {
        let count = systems.len();
        let id_to_idx: HashMap<SystemId, usize> = systems
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id, i))
            .collect();

        Self {
            systems,
            id_to_idx,
            edges: vec![Vec::new(); count],
        }
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
        let from_meta = &self.systems[self.idx(from)];
        let target_meta = &self.systems[target_idx];
        if target_meta.stage > from_meta.stage {
            return Err(ScheduleError::StageInversion {
                system: from,
                target,
            });
        }

        let from_idx = self.idx(from);
        if !self.edges[target_idx].iter().any(|(to, _)| *to == from_idx) {
            self.edges[target_idx].push((from_idx, EdgeKind::ExplicitAfter));
        }
        Ok(())
    }

    /// Add an explicit `before` edge: `from` must run before `target`.
    pub fn add_explicit_before(
        &mut self,
        from: SystemId,
        target: SystemId,
    ) -> Result<(), ScheduleError> {
        // A before edge is the inverse of after: reverse it.
        let target_idx = self.resolve_target(from, target)?;
        let from_meta = &self.systems[self.idx(from)];
        let target_meta = &self.systems[target_idx];
        if from_meta.stage > target_meta.stage {
            return Err(ScheduleError::StageInversion {
                system: from,
                target,
            });
        }

        let from_idx = self.idx(from);
        if !self.edges[from_idx].iter().any(|(to, _)| *to == target_idx) {
            self.edges[from_idx].push((target_idx, EdgeKind::ExplicitBefore));
        }
        Ok(())
    }

    /// Add a serialization edge between two systems for deterministic ordering.
    pub fn add_serialization_edge(&mut self, before: SystemId, after: SystemId) {
        let before_idx = self.idx(before);
        let after_idx = self.idx(after);
        if !self.edges[before_idx]
            .iter()
            .any(|(to, _)| *to == after_idx)
        {
            self.edges[before_idx].push((after_idx, EdgeKind::Serialization));
        }
    }

    /// Build and validate the graph.
    ///
    /// Returns `Err` with a cycle diagnostic if a cycle is detected.
    pub fn build(self) -> Result<DependencyGraph, ScheduleError> {
        let in_degree = self.compute_in_degree();
        let graph = DependencyGraph {
            id_to_idx: self.id_to_idx,
            systems: self.systems,
            edges: self.edges,
            in_degree,
        };

        // Check for cycles via Kahn's algorithm.
        let _ = graph
            .topological_order()
            .map_err(|cycle_nodes| {
                let path = graph.build_cycle_path(&cycle_nodes);
                ScheduleError::CycleDetected(path)
            })?;

        Ok(graph)
    }

    // ── private helpers ────────────────────────────────────────────────

    fn idx(&self, id: SystemId) -> usize {
        self.id_to_idx[&id]
    }

    fn resolve_target(&self, from: SystemId, target: SystemId) -> Result<usize, ScheduleError> {
        self.id_to_idx
            .get(&target)
            .copied()
            .ok_or(ScheduleError::UnknownTarget {
                system: from,
                target,
            })
    }

    fn compute_in_degree(&self) -> Vec<usize> {
        let mut deg = vec![0usize; self.systems.len()];
        for targets in &self.edges {
            for &(to, _) in targets {
                deg[to] += 1;
            }
        }
        deg
    }

}

// ---------------------------------------------------------------------------
// Topological sort (Kahn's algorithm, internal to graph)
// ---------------------------------------------------------------------------

impl DependencyGraph {
    /// Compute a topological order, returning cycle nodes on failure.
    pub(crate) fn topological_order(&self) -> Result<Vec<SystemId>, Vec<SystemId>> {
        let n = self.systems.len();
        let mut in_degree = self.in_degree.clone();
        let mut ready: VecDeque<usize> = (0..n)
            .filter(|&i| in_degree[i] == 0)
            .collect();

        // Sort initial ready set by (stage, order, id) for determinism.
        {
            let mut tmp: Vec<usize> = ready.drain(..).collect();
            tmp.sort_by(|&a, &b| {
                let sa = &self.systems[a];
                let sb = &self.systems[b];
                sa.stage
                    .cmp(&sb.stage)
                    .then_with(|| sa.order.cmp(&sb.order))
                    .then_with(|| sa.id.cmp(&sb.id))
            });
            ready.extend(tmp);
        }

        let mut order = Vec::with_capacity(n);

        while let Some(idx) = ready.pop_front() {
            order.push(self.systems[idx].id);
            for &(next, _) in &self.edges[idx] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    // Maintain deterministic insertion order.
                    let sys = &self.systems[next];
                    let pos = ready
                        .iter()
                        .position(|&r| {
                            let rs = &self.systems[r];
                            sys.stage > rs.stage
                                || (sys.stage == rs.stage && sys.order > rs.order)
                                || (sys.stage == rs.stage
                                    && sys.order == rs.order
                                    && sys.id > rs.id)
                        })
                        .unwrap_or(ready.len());
                    ready.insert(pos, next);
                }
            }
        }

        if order.len() == n {
            Ok(order)
        } else {
            // Find nodes with nonzero in-degree.
            let cycle_nodes: Vec<SystemId> = in_degree
                .iter()
                .enumerate()
                .filter(|(_, &deg)| deg > 0)
                .map(|(i, _)| self.systems[i].id)
                .collect();
            Err(cycle_nodes)
        }
    }

    /// Build an ordered cycle path for diagnostics.
    ///
    /// Performs DFS from each cycle node to find the actual edge sequence.
    fn build_cycle_path(&self, cycle_nodes: &[SystemId]) -> Vec<SystemId> {
        if cycle_nodes.is_empty() {
            return Vec::new();
        }

        let mut visited = HashSet::new();
        let mut path = Vec::new();

        fn dfs(
            current: usize,
            target: usize,
            edges: &[Vec<(usize, EdgeKind)>],
            systems: &[SystemMetadata],
            visited: &mut HashSet<usize>,
            path: &mut Vec<SystemId>,
        ) -> bool {
            if current == target && !path.is_empty() {
                return true;
            }
            if !visited.insert(current) {
                return false;
            }
            for &(next, _) in &edges[current] {
                path.push(systems[next].id);
                if dfs(next, target, edges, systems, visited, path) {
                    return true;
                }
                path.pop();
            }
            false
        }

        // Try each node as start until we find the cycle.
        for &node in cycle_nodes {
            let Some(start_idx) = self.index_of(&node) else {
                continue;
            };
            path.clear();
            visited.clear();
            path.push(node);
            if dfs(
                start_idx,
                start_idx,
                &self.edges,
                &self.systems,
                &mut visited,
                &mut path,
            ) {
                path.push(node); // close the cycle
                return path;
            }
        }

        // Fallback: return the original list.
        cycle_nodes.to_vec()
    }
}
