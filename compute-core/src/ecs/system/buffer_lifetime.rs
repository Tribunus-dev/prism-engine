use crate::ecs::component::fusion::DataflowGraphHandle;
use crate::ecs::component::memory::{BufferLifetime, MemoryPool, PoolPolicy, ScratchConfig};
use crate::ecs::plan::fusion::DataflowGraph;
#[allow(unused_imports)]
use crate::ecs::Entity;
use crate::ecs::{CompWorld, CompilerSystem, EntityKind, SchedulePhase};

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// LifetimeAnalysisSystem
// ---------------------------------------------------------------------------

/// Computes per-buffer lifetimes from the dataflow-graph dependency topology.
///
/// Approach:
/// 1. Collect `DataflowGraphHandle` references from Tensor/Layer entities in
///    the world, and resolve them against an internal graph registry.
/// 2. Topologically sort each graph (Kahn's algorithm) to assign epochs.
/// 3. For each value in the graph compute alloc_epoch (producer rank) and
///    free_epoch (max consumer rank), then assign `BufferLifetime` to every
///    existing `Buffer` entity.
///
/// If no graphs are available from either the registry or the world, a
/// built-in MLP graph is used as fallback so the topology computation always
/// produces results.
///
/// Buffers without a recognised graph value receive a conservative
/// (`0`, `global_max_epoch + 1`) lifetime — safe, not optimal.
///
/// The system can also be seeded with graphs ahead of time via
/// [`register_graph`](LifetimeAnalysisSystem::register_graph).
pub struct LifetimeAnalysisSystem {
    /// Map from handle string → parsed DataflowGraph.
    graphs: HashMap<String, DataflowGraph>,
}

impl LifetimeAnalysisSystem {
    pub fn new() -> Self {
        Self {
            graphs: HashMap::new(),
        }
    }

    /// Register or update a `DataflowGraph` so the system can resolve
    /// handle strings during `run()`.
    pub fn register_graph(&mut self, handle: String, graph: DataflowGraph) {
        self.graphs.insert(handle, graph);
    }

    /// Collect graph lifetimes from all registered graphs that are referenced
    /// by `DataflowGraphHandle` components in the world.
    fn collect_registered_lifetimes(&self, world: &CompWorld) -> Vec<HashMap<String, (u64, u64)>> {
        let tensors = world.entities_of_kind(EntityKind::Tensor);
        let layers = world.entities_of_kind(EntityKind::Layer);
        let all_holders: Vec<Entity> = tensors.into_iter().chain(layers).collect();

        let mut seen = std::collections::HashSet::new();
        let mut all_lifetimes = Vec::new();

        for entity in all_holders {
            let Some(handle) = world.get_component::<DataflowGraphHandle>(entity) else {
                continue;
            };
            if !seen.insert(handle.0.clone()) {
                continue; // already processed this handle
            }
            let Some(graph) = self.graphs.get(&handle.0) else {
                continue;
            };
            all_lifetimes.push(compute_value_lifetimes(graph));
        }

        all_lifetimes
    }
}

impl Default for LifetimeAnalysisSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl CompilerSystem for LifetimeAnalysisSystem {
    fn name(&self) -> &str {
        "LifetimeAnalysisSystem"
    }

    fn phase(&self) -> SchedulePhase {
        SchedulePhase::MemoryPlanning
    }

    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let mut all_lifetimes = self.collect_registered_lifetimes(world);

        if all_lifetimes.is_empty() {
            // Fallback: use a built-in MLP graph so topology logic always runs.
            let fallback = build_fallback_mlp();
            all_lifetimes.push(compute_value_lifetimes(&fallback));
        }

        let max_epoch = all_lifetimes
            .iter()
            .flat_map(|m| m.values().map(|(_, free)| *free))
            .max()
            .unwrap_or(0);

        let buffers = world.entities_of_kind(EntityKind::Buffer);
        for buffer in &buffers {
            let name = world.name(*buffer).unwrap_or("");

            let (alloc_epoch, free_epoch) = if !name.is_empty() {
                // Search all resolved graphs for a matching value name.
                let mut best: Option<(u64, u64)> = None;
                for lifetimes in &all_lifetimes {
                    if let Some(&lt) = lifetimes.get(name) {
                        match best {
                            None => best = Some(lt),
                            Some((_, best_free)) if lt.1 < best_free => {
                                best = Some(lt);
                            }
                            _ => {}
                        }
                    }
                }
                best.unwrap_or((0, max_epoch + 1))
            } else {
                // Unnamed buffer — assign a conservative lifetime covering
                // the entire known epoch span.
                (0, max_epoch + 1)
            };

            let frontier = (free_epoch < u64::MAX).then(|| (0u64, free_epoch + 1));

            world.add_component(
                *buffer,
                BufferLifetime {
                    alloc_epoch,
                    free_epoch,
                    causal_death_frontier: frontier,
                },
            );
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ScratchPlanningSystem
// ---------------------------------------------------------------------------

/// Plans scratch buffer allocations for each dispatch entity.
///
/// Iterates every `Dispatch` entity, computes the required scratch memory
/// size from available graph references (intermediate value count ×
/// scratch_factor), then spawns a `Buffer` entity with `Arena` pool policy
/// and attaches a `ScratchConfig` component to the dispatch.
pub struct ScratchPlanningSystem {
    /// Fraction of total intermediate bytes to allocate as scratch (0.0–1.0).
    pub scratch_factor: f64,
    /// Minimum scratch allocation in bytes.
    pub min_scratch_bytes: u64,
    /// Optional override: if set, every dispatch gets exactly this many
    /// scratch bytes regardless of graph analysis.
    pub fixed_scratch_bytes: Option<u64>,
}

impl ScratchPlanningSystem {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for ScratchPlanningSystem {
    fn default() -> Self {
        Self {
            scratch_factor: 0.25,
            min_scratch_bytes: 1 << 20, // 1 MiB
            fixed_scratch_bytes: None,
        }
    }
}

impl CompilerSystem for ScratchPlanningSystem {
    fn name(&self) -> &str {
        "ScratchPlanningSystem"
    }

    fn phase(&self) -> SchedulePhase {
        SchedulePhase::MemoryPlanning
    }

    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let dispatches: Vec<Entity> = world.entities_of_kind(EntityKind::Dispatch);

        if dispatches.is_empty() {
            return Ok(());
        }

        // Count unique graph handles in the world for scratch sizing.
        let mut seen_handles = std::collections::HashSet::new();
        let mut total_graph_values: u64 = 0;

        for entity in world
            .entities_of_kind(EntityKind::Tensor)
            .iter()
            .chain(world.entities_of_kind(EntityKind::Layer).iter())
        {
            if let Some(h) = world.get_component::<DataflowGraphHandle>(*entity) {
                if seen_handles.insert(h.0.clone()) {
                    total_graph_values += 4; // heuristic: Q/K/V + output scratch
                }
            }
        }

        for dispatch in &dispatches {
            let scratch_bytes = match self.fixed_scratch_bytes {
                Some(fixed) => fixed,
                None => {
                    let raw = total_graph_values.max(1) as f64 * self.scratch_factor;
                    (raw * self.min_scratch_bytes as f64)
                        .ceil()
                        .max(self.min_scratch_bytes as f64) as u64
                }
            };

            // Spawn a scratch Buffer entity with Arena pool policy.
            let scratch_buf = world.spawn(
                EntityKind::Buffer,
                Some(format!("scratch_dispatch_{}", dispatch.0)),
            );
            world.add_component(
                scratch_buf,
                MemoryPool {
                    policy: PoolPolicy::Arena,
                    pool_id: 0,
                    total_bytes: scratch_bytes,
                    used_bytes: 0,
                },
            );

            // Record the scratch config on the dispatch entity.
            world.add_component(
                *dispatch,
                ScratchConfig {
                    per_dispatch_scratch: scratch_bytes,
                    persistent_scratch: 0,
                    arena_policy: PoolPolicy::Arena,
                },
            );
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// For a single `DataflowGraph`, compute per-value alloc/free epochs
/// from the topological sort.
fn compute_value_lifetimes(graph: &DataflowGraph) -> HashMap<String, (u64, u64)> {
    let topo = topo_sort(graph);
    let mut lifetimes = HashMap::new();

    for (buf_id, _value) in &graph.values {
        // Producer = first node that lists this buf_id as output.
        let producer_epoch = graph
            .nodes
            .iter()
            .position(|n| n.outputs.contains(buf_id))
            .and_then(|idx| topo.get(&idx).copied());

        // Consumers = all edges carrying this value.
        let consumer_max = graph
            .edges
            .iter()
            .filter(|e| &e.value == buf_id)
            .filter_map(|e| topo.get(&e.consumer).copied())
            .max();

        let alloc = producer_epoch.unwrap_or(0);
        let free = consumer_max.unwrap_or(alloc + 1);
        lifetimes.insert(buf_id.clone(), (alloc, free));
    }

    lifetimes
}

/// Build a canonical Gemma decoder MLP graph as a fallback when no external
/// graphs are registered.
fn build_fallback_mlp() -> DataflowGraph {
    use crate::ecs::plan::fusion::DataflowGraphBuilder;
    DataflowGraphBuilder::build_mlp()
}

// ---------------------------------------------------------------------------
// Topological sort — Kahn's algorithm
// ---------------------------------------------------------------------------

/// Returns a map from node index → topological rank (smaller = earlier).
fn topo_sort(graph: &DataflowGraph) -> HashMap<usize, u64> {
    let n = graph.nodes.len();
    if n == 0 {
        return HashMap::new();
    }

    let mut in_degree = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for edge in &graph.edges {
        if edge.producer < n && edge.consumer < n {
            adj[edge.producer].push(edge.consumer);
            in_degree[edge.consumer] += 1;
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }

    let mut rank = HashMap::with_capacity(n);
    let mut epoch: u64 = 0;
    while let Some(node) = queue.pop_front() {
        rank.insert(node, epoch);
        epoch += 1;
        for &succ in &adj[node] {
            in_degree[succ] = in_degree[succ].saturating_sub(1);
            if in_degree[succ] == 0 {
                queue.push_back(succ);
            }
        }
    }

    // Assign remaining epoch to any unvisited nodes (safety for cycles).
    for (i, _) in graph.nodes.iter().enumerate() {
        rank.entry(i).or_insert_with(|| {
            let e = epoch;
            epoch += 1;
            e
        });
    }

    rank
}
