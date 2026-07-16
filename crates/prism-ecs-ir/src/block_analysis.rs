//! Block analysis — RPCS3 inspired pattern matching on op sequences.
//!
//! RPCS3's `block_info` analyser detects known code patterns in compiled SPU
//! blocks (e.g. load-hit-store, DMA lists, branch hints) to guide optimisation.
//! This module generalises that concept: a [`BlockAnalyzer`] walks the ordered
//! op entities in a block and matches patterns by op-name heuristics, then
//! produces [`AnalysisResult`] values that downstream fusion passes can use
//! to decide whether to fuse or reorder ops.
//!
//! # Components
//!
//! * [`AnalysisResult`] — component attached to a block entity after analysis.
//!   Contains the discovered pattern kind, the matched ops, and an optional
//!   fusion suggestion.
//!
//! # Systems
//!
//! * [`BlockAnalyzer`] — resource that scans a block's ops and identifies
//!   known patterns.
//!
//! # Integration
//!
//! The [`fusion`](crate::fusion) module's `partition_fusion_groups` function
//! can optionally consult a block's [`AnalysisResult`] component to guide
//! fusion decisions — for example, fusing a matmul + bias-add sequence into
//! one kernel, or keeping a load-store pair separate to avoid register pressure.

use prism_ecs_core::{Component, Entity, World};
use serde::{Deserialize, Serialize};

use crate::block;
use crate::op;

// ---------------------------------------------------------------------------
// PatternKind — the detected code pattern
// ---------------------------------------------------------------------------

/// Known op-sequence patterns that the analyser can recognise.
///
/// Each variant corresponds to a pattern that RPCS3's `block_info` analysis
/// looks for in SPU JIT blocks, generalised to the Prism IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternKind {
    /// MatMul immediately followed by a broadcast/bias add — candidate for
    /// fused matmul + bias in one kernel.
    MatmulBias,
    /// GEMM followed by a ReLU / activation — candidate for fused
    /// matmul + activation.
    GemmActivation,
    /// Load immediately followed by a store with overlapping address range
    /// — candidate for store-forwarding or memory aliasing.
    LoadHitStore,
    /// A chain of element-wise arithmetic ops (add, mul, sub) — candidate
    /// for horizontal fusion into a single kernel.
    ElementWiseChain,
    /// A reduction (sum, max) followed by a broadcast — common in layer-norm.
    ReduceBroadcast,
    /// Gather from a set of indices followed by a scatter — analogue of
    /// SPU's DMA list pattern.
    GatherScatter,
    /// A simple fused multiply-accumulate: `a = b + c * d`.
    Fma,
    /// A recognised activation function block (ReLU, sigmoid, silu, gelu, etc.).
    Activation,
    /// No known pattern was found.
    Unknown,
}

// ---------------------------------------------------------------------------
// FusionSuggestion — hint for downstream fusion passes
// ---------------------------------------------------------------------------

/// A suggested fusion strategy based on the detected pattern.
///
/// This is written into [`AnalysisResult`] so that [`partition_fusion_groups`]
/// in the fusion module can optionally incorporate it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FusionSuggestion {
    /// The matched ops should be fused into one group.
    Fuse,
    /// The matched ops should stay separate (e.g. load-hit-store).
    Separate,
    /// No particular suggestion — use the default fusion policy.
    Default,
}

// ---------------------------------------------------------------------------
// AnalysisResult component
// ---------------------------------------------------------------------------

/// The result of analysing a block's op sequence.
///
/// Attached to a block entity after [`BlockAnalyzer::analyze_block`] runs.
/// Downstream passes (notably fusion) read this component to guide decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Which pattern was detected.
    pub pattern_kind: PatternKind,
    /// The entities of the ops that matched the pattern.
    pub ops_matched: Vec<Entity>,
    /// Whether the analyser suggests fusing or separating these ops.
    pub suggested_fusion: FusionSuggestion,
}

impl Component for AnalysisResult {}

impl AnalysisResult {
    fn new(
        pattern_kind: PatternKind,
        ops_matched: Vec<Entity>,
        suggested_fusion: FusionSuggestion,
    ) -> Self {
        Self {
            pattern_kind,
            ops_matched,
            suggested_fusion,
        }
    }

    /// Convenience: check if the analyser suggested fusing this block's ops.
    pub fn should_fuse(&self) -> bool {
        self.suggested_fusion == FusionSuggestion::Fuse
    }
}

// ---------------------------------------------------------------------------
// BlockAnalyzer
// ---------------------------------------------------------------------------

/// ECS resource that analyses block op sequences for known patterns.
///
/// The analyser walks the ordered op entities in a block (via the
/// [`BlockOps`](crate::block::BlockOps) component) and looks for
/// op-name-based patterns by scanning a sliding window.
///
/// # Usage
///
/// ```ignore
/// let result = analyzer.analyze_block(&world, block_entity);
/// if result.pattern_kind != PatternKind::Unknown {
///     let ops = result.ops_matched;
///     // wire into fusion logic …
/// }
/// ```
#[derive(Debug, Default)]
pub struct BlockAnalyzer;

impl BlockAnalyzer {
    /// Create a new block analyser.
    pub fn new() -> Self {
        Self
    }

    /// Analyse a single block's ops and return an [`AnalysisResult`].
    ///
    /// Walks the block's ops (via `block_ops`) in order and checks every
    /// contiguous subsequence of length 1 to 3 for known patterns.
    /// The first match wins.
    pub fn analyze_block(&self, world: &World, block_entity: Entity) -> AnalysisResult {
        let ops = block::block_ops(world, block_entity);
        if ops.is_empty() {
            return AnalysisResult::new(PatternKind::Unknown, vec![], FusionSuggestion::Default);
        }

        // Single-op patterns
        if let Some(result) = self.match_single_op(world, &ops) {
            return result;
        }
        // Three-op patterns
        if let Some(result) = self.match_three_ops(world, &ops) {
            return result;
        }

        // Two-op patterns
        if let Some(result) = self.match_two_ops(world, &ops) {
            return result;
        }

        // Default: no recognised pattern.
        AnalysisResult::new(PatternKind::Unknown, vec![], FusionSuggestion::Default)
    }

    /// Analyse all blocks in the world that have [`BlockOps`] and write the
    /// [`AnalysisResult`] component onto each one.
    ///
    /// Returns the number of blocks that had a recognised pattern.
    pub fn analyze_all_blocks(&self, world: &mut World) -> usize {
        let blocks: Vec<Entity> = world
            .query::<block::BlockMarker>()
            .map(|(e, _)| e)
            .collect();
        let mut matched = 0usize;

        for block_entity in blocks {
            // Check it has ops (skip blocks without BlockOps component).
            if block::block_ops(world, block_entity).is_empty() {
                continue;
            }

            let result = self.analyze_block(world, block_entity);
            if result.pattern_kind != PatternKind::Unknown {
                matched += 1;
            }

            // Attach the analysis result to the block entity.
            let _ = world.add_component(block_entity, result);
        }

        matched
    }

    /// Classify a single-op pattern based on the op name.
    fn classify_op_name(&self, name: &str) -> Option<PatternKind> {
        match name {
            n if n == "linalg.matmul" || n == "linalg.batch_matmul" => None, // single matmul is not a "pattern"
            n if n.starts_with("arith.")
                && (n.contains("addf")
                    || n.contains("subf")
                    || n.contains("mulf")
                    || n.contains("divf"))
                && !n.contains("reduce")
                && !n.contains("broadcast") =>
            {
                Some(PatternKind::ElementWiseChain)
            }
            n if n.ends_with("relu") || n.ends_with("sigmoid") || n.ends_with("gelu") => {
                Some(PatternKind::Activation)
            }
            n if n.contains("silu") || n.contains("tanh") || n.contains("hardswish") => {
                Some(PatternKind::Activation)
            }
            _ => None,
        }
    }

    /// Check single-op patterns (activation functions).
    fn match_single_op(&self, world: &World, ops: &[Entity]) -> Option<AnalysisResult> {
        if ops.len() != 1 {
            return None;
        }
        let first = ops.first()?;
        let name = op::op_name(world, *first)?;
        self.classify_op_name(&name).map(|kind| {
            let suggestion = if kind == PatternKind::Activation {
                FusionSuggestion::Fuse
            } else {
                FusionSuggestion::Default
            };
            AnalysisResult::new(kind, vec![*first], suggestion)
        })
    }

    /// Check two-op patterns: matmul+bias, load+store.
    fn match_two_ops(&self, world: &World, ops: &[Entity]) -> Option<AnalysisResult> {
        if ops.len() < 2 {
            return None;
        }

        // Sliding window over pairs
        for pair in ops.windows(2) {
            let a = op::op_name(world, pair[0]);
            let b = op::op_name(world, pair[1]);
            let (a, b) = match (a, b) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };

            // MatMul + Bias (arith.addf with broadcast semantics)
            let is_matmul = a.contains("matmul") || a.contains("gemm");
            let is_bias = b == "arith.addf" || b.contains("broadcast_in_dim");
            if is_matmul && is_bias {
                return Some(AnalysisResult::new(
                    PatternKind::MatmulBias,
                    vec![pair[0], pair[1]],
                    FusionSuggestion::Fuse,
                ));
            }

            // GEMM + Activation
            let is_gemm = a.contains("gemm") || a.contains("matmul");
            let is_activation = self.classify_op_name(&b) == Some(PatternKind::Activation);
            if is_gemm && is_activation {
                return Some(AnalysisResult::new(
                    PatternKind::GemmActivation,
                    vec![pair[0], pair[1]],
                    FusionSuggestion::Fuse,
                ));
            }

            // Load + Store (load-hit-store detection).
            let is_load = a.starts_with("memref.load") || a.starts_with("linalg.load");
            let is_store = b.starts_with("memref.store") || b.starts_with("linalg.store");
            if is_load && is_store {
                return Some(AnalysisResult::new(
                    PatternKind::LoadHitStore,
                    vec![pair[0], pair[1]],
                    FusionSuggestion::Separate,
                ));
            }

            // Element-wise chain: two element-wise ops in a row.
            let is_elem_a = self.classify_op_name(&a) == Some(PatternKind::ElementWiseChain);
            let is_elem_b = self.classify_op_name(&b) == Some(PatternKind::ElementWiseChain);
            if is_elem_a && is_elem_b {
                return Some(AnalysisResult::new(
                    PatternKind::ElementWiseChain,
                    vec![pair[0], pair[1]],
                    FusionSuggestion::Fuse,
                ));
            }

            // Reduce + Broadcast
            let is_reduce = a.contains("reduce") || a.contains("sum");
            let is_broadcast = b.contains("broadcast");
            if is_reduce && is_broadcast {
                return Some(AnalysisResult::new(
                    PatternKind::ReduceBroadcast,
                    vec![pair[0], pair[1]],
                    FusionSuggestion::Separate,
                ));
            }
        }

        None
    }

    /// Check three-op patterns: FMA (mul + add), gather + scatter with index.
    fn match_three_ops(&self, world: &World, ops: &[Entity]) -> Option<AnalysisResult> {
        if ops.len() < 3 {
            return None;
        }

        // Sliding window over triples
        for triple in ops.windows(3) {
            let names: Vec<Option<String>> =
                triple.iter().map(|e| op::op_name(world, *e)).collect();
            let names: Vec<&str> = names.iter().filter_map(|n| n.as_deref()).collect();
            if names.len() < 3 {
                continue;
            }

            // FMA: mul -> add (with no load/store between)
            // Pattern: arith.mulf followed by arith.addf where the mul result
            // is one of the add operands.
            let is_mul = names[0].contains("mulf") || names[0] == "arith.muli";
            let is_add = names[1].contains("addf");
            if is_mul && is_add {
                return Some(AnalysisResult::new(
                    PatternKind::Fma,
                    vec![triple[0], triple[1]],
                    FusionSuggestion::Fuse,
                ));
            }

            // Gather + transpose/interleave + scatter (SPU DMA list pattern).
            let is_gather = names[0].contains("gather") || names[0].contains("transfer_read");
            let is_scatter = names[2].contains("scatter") || names[2].contains("transfer_write");
            if is_gather && is_scatter {
                return Some(AnalysisResult::new(
                    PatternKind::GatherScatter,
                    vec![triple[0], triple[1], triple[2]],
                    FusionSuggestion::Separate,
                ));
            }
        }

        None
    }

    /// Run the pattern analyser on every block in the world, writing results
    /// as components, then return the list of detected patterns grouped by
    /// [`PatternKind`].
    ///
    /// A convenience wrapper that calls [`analyze_all_blocks`](Self::analyze_all_blocks)
    /// and additionally returns structured results.
    pub fn analyze_and_classify(&self, world: &mut World) -> Vec<(Entity, AnalysisResult)> {
        let blocks: Vec<Entity> = world
            .query::<block::BlockMarker>()
            .map(|(e, _)| e)
            .collect();
        let mut results = Vec::new();

        for block_entity in blocks {
            let result = self.analyze_block(world, block_entity);
            if result.pattern_kind != PatternKind::Unknown {
                results.push((block_entity, result.clone()));
            }
            // Always attach the component even for unknown — so fusion can check presence.
            if world
                .get_component::<AnalysisResult>(block_entity)
                .is_none()
            {
                let _ = world.add_component(block_entity, result);
            }
        }

        results
    }
}

// ---------------------------------------------------------------------------
// Free functions for direct use in fusion.rs
// ---------------------------------------------------------------------------

/// Convenience: run the block analyser on a single block entity and return
/// the analysis result, or a default unknown result if the block has no ops.
pub fn analyze_block(world: &World, block_entity: Entity) -> AnalysisResult {
    let analyzer = BlockAnalyzer::new();
    analyzer.analyze_block(world, block_entity)
}

/// Convenience: run the block analyser on all blocks and return detected
/// patterns suitable for fusion guidance.
pub fn analyze_all_blocks(world: &mut World) -> Vec<(Entity, AnalysisResult)> {
    let analyzer = BlockAnalyzer::new();
    analyzer.analyze_and_classify(world)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockMarker, BlockOps};
    use crate::op::{OpMarker, OpName};
    use prism_ecs_core::{EntityKind, World};

    fn spawn_op(world: &mut World, name: &str) -> Entity {
        let e: Entity = world.spawn(EntityKind::Node, None).unwrap().into();
        world.add_component(e, OpMarker).unwrap();
        world.add_component(e, OpName(name.into())).unwrap();
        e
    }

    fn spawn_block(world: &mut World, ops: &[Entity]) -> Entity {
        let b: Entity = world.spawn(EntityKind::Node, None).unwrap().into();
        world.add_component(b, BlockMarker).unwrap();
        world.add_component(b, BlockOps(ops.to_vec())).unwrap();
        b
    }

    #[test]
    fn empty_block_returns_unknown() {
        let world = World::new();
        // We need a block entity in the world, but it won't have BlockOps.
        // analyze_block returns unknown for empty ops.
        let result = analyze_block(&world, Entity(0, 0));
        assert_eq!(result.pattern_kind, PatternKind::Unknown);
    }

    #[test]
    fn detect_matmul_bias() {
        let mut world = World::new();
        let matmul = spawn_op(&mut world, "linalg.matmul");
        let bias = spawn_op(&mut world, "arith.addf");
        let block = spawn_block(&mut world, &[matmul, bias]);
        let result = analyze_block(&world, block);
        assert_eq!(result.pattern_kind, PatternKind::MatmulBias);
        assert_eq!(result.ops_matched.len(), 2);
        assert_eq!(result.suggested_fusion, FusionSuggestion::Fuse);
    }

    #[test]
    fn detect_gemm_activation() {
        let mut world = World::new();
        let gemm = spawn_op(&mut world, "linalg.matmul");
        let relu = spawn_op(&mut world, "arith.relu");
        let block = spawn_block(&mut world, &[gemm, relu]);
        let result = analyze_block(&world, block);
        assert_eq!(result.pattern_kind, PatternKind::GemmActivation);
        assert!(result.should_fuse());
    }

    #[test]
    fn detect_load_hit_store() {
        let mut world = World::new();
        let load = spawn_op(&mut world, "memref.load");
        let store = spawn_op(&mut world, "memref.store");
        let block = spawn_block(&mut world, &[load, store]);
        let result = analyze_block(&world, block);
        assert_eq!(result.pattern_kind, PatternKind::LoadHitStore);
        assert_eq!(result.suggested_fusion, FusionSuggestion::Separate);
    }

    #[test]
    fn detect_element_wise_chain() {
        let mut world = World::new();
        let add = spawn_op(&mut world, "arith.addf");
        let mul = spawn_op(&mut world, "arith.mulf");
        let block = spawn_block(&mut world, &[add, mul]);
        let result = analyze_block(&world, block);
        assert_eq!(result.pattern_kind, PatternKind::ElementWiseChain);
        assert!(result.should_fuse());
    }

    #[test]
    fn detect_reduce_broadcast() {
        let mut world = World::new();
        let reduce = spawn_op(&mut world, "arith.reduce_sum");
        let bcast = spawn_op(&mut world, "arith.broadcast");
        let block = spawn_block(&mut world, &[reduce, bcast]);
        let result = analyze_block(&world, block);
        assert_eq!(result.pattern_kind, PatternKind::ReduceBroadcast);
        assert_eq!(result.suggested_fusion, FusionSuggestion::Separate);
    }

    #[test]
    fn detect_fma() {
        let mut world = World::new();
        let mul = spawn_op(&mut world, "arith.mulf");
        let add = spawn_op(&mut world, "arith.addf");
        let store = spawn_op(&mut world, "memref.store");
        let block = spawn_block(&mut world, &[mul, add, store]);
        let result = analyze_block(&world, block);
        assert_eq!(result.pattern_kind, PatternKind::Fma);
        assert!(result.should_fuse());
    }

    #[test]
    fn detect_gather_scatter() {
        let mut world = World::new();
        let gather = spawn_op(&mut world, "memref.gather");
        let interleave = spawn_op(&mut world, "arith.addf");
        let scatter = spawn_op(&mut world, "memref.scatter");
        let block = spawn_block(&mut world, &[gather, interleave, scatter]);
        let result = analyze_block(&world, block);
        assert_eq!(result.pattern_kind, PatternKind::GatherScatter);
        assert_eq!(result.suggested_fusion, FusionSuggestion::Separate);
    }

    #[test]
    fn detect_activation_single_op() {
        let mut world = World::new();
        let relu = spawn_op(&mut world, "arith.relu");
        let block = spawn_block(&mut world, &[relu]);
        let result = analyze_block(&world, block);
        assert_eq!(result.pattern_kind, PatternKind::Activation);
        assert!(result.should_fuse());
    }

    #[test]
    fn analyze_and_classify_writes_components() {
        let mut world = World::new();
        let matmul = spawn_op(&mut world, "linalg.matmul");
        let bias = spawn_op(&mut world, "arith.addf");
        let block = spawn_block(&mut world, &[matmul, bias]);

        let results = analyze_all_blocks(&mut world);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, block);
        assert_eq!(results[0].1.pattern_kind, PatternKind::MatmulBias);

        // Component should be attached
        let ar = world.get_component::<AnalysisResult>(block);
        assert!(ar.is_some());
        assert_eq!(ar.unwrap().pattern_kind, PatternKind::MatmulBias);
    }
}
