//! Rayon threading strategy and scratch-plan types for the CPU fusion backend.
//!
//! These types govern how a fused program's work is partitioned across
//! threads and how scratch buffers are sized and aliased.

use serde::{Deserialize, Serialize};

// ── RayonStrategy ───────────────────────────────────────────────────────────

/// How a fused program's parallel work is subdivided across Rayon threads.
///
/// The strategy is chosen at lowering time based on op characteristics
/// (arithmetic intensity, tensor sizes) and the available thread count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RayonStrategy {
    /// Let Rayon choose the subdivision automatically (default).
    Auto,

    /// Fixed-size chunks — each thread processes `chunk_size` elements.
    Static {
        /// Number of threads to use (0 = use all available).
        num_threads: usize,
        /// Elements per chunk.
        chunk_size: usize,
    },

    /// Adaptive chunk sizing — starts small, grows as work stealing proceeds.
    Dynamic {
        /// Minimum elements per chunk.
        min_chunk: usize,
        /// Maximum elements per chunk.
        max_chunk: usize,
    },

    /// Pure work-stealing — Rayon's default `par_iter()`.
    WorkStealing,
}

impl Default for RayonStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

// ── CpuThreadingPolicy ──────────────────────────────────────────────────────

/// Whether a fused CPU program should execute sequentially, in parallel,
/// or switch between modes based on a size threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CpuThreadingPolicy {
    /// Execute all ops on a single thread.
    Sequential,

    /// Always use the given parallel strategy.
    Parallel {
        /// The Rayon subdivision strategy.
        strategy: RayonStrategy,
    },

    /// Use sequential execution below `threshold_elements` and parallel
    /// above it. Avoids paying parallelism overhead for tiny tensors.
    Hybrid {
        /// Tensor elements above which parallelism is engaged.
        threshold_elements: usize,
        /// The strategy used above the threshold.
        parallel_strategy: RayonStrategy,
    },
}

impl Default for CpuThreadingPolicy {
    fn default() -> Self {
        Self::Hybrid {
            threshold_elements: 4096,
            parallel_strategy: RayonStrategy::WorkStealing,
        }
    }
}

// ── CpuScratchPlan ──────────────────────────────────────────────────────────

/// Describes the scratch (spill) buffer layout for a CPU fused program.
///
/// Since CPU kernels cannot alias arbitrary input buffers, intermediate
/// values that would live in registers in a Metal kernel must spill to
/// temporary scratch memory. The scratch plan records sizes, alignment,
/// and legal aliasing relationships so the arena planner can budget
/// correctly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuScratchPlan {
    /// Total bytes required for the largest single spill.
    pub spill_buffer_size: usize,

    /// Byte alignment for the spill buffer base address.
    pub spill_buffer_alignment: usize,

    /// Descriptions of alias groups — buffers whose lifetimes do not
    /// overlap and may share the same scratch region.
    pub alias_groups: Vec<AliasGroupSpec>,

    /// Maximum number of concurrent spill allocations the program needs.
    pub max_concurrent_spills: usize,
}

/// One alias group in the scratch plan.
///
/// Members of the same group have non-overlapping lifetimes and may be
/// allocated at the same offset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AliasGroupSpec {
    /// Stable identifier for this alias group.
    pub group_id: String,
    /// Number of spill buffers in the group.
    pub member_count: usize,
}

impl Default for CpuScratchPlan {
    fn default() -> Self {
        Self {
            spill_buffer_size: 0,
            spill_buffer_alignment: 64, // cache-line aligned
            alias_groups: Vec::new(),
            max_concurrent_spills: 1,
        }
    }
}

// ── Defaults for simple programs ────────────────────────────────────────────

impl CpuScratchPlan {
    /// Create a minimal scratch plan for a program that needs a single
    /// spill buffer (no aliasing).
    pub fn single_spill(spill_bytes: usize) -> Self {
        Self {
            spill_buffer_size: spill_bytes,
            spill_buffer_alignment: 64,
            alias_groups: vec![AliasGroupSpec {
                group_id: "spill_0".into(),
                member_count: 1,
            }],
            max_concurrent_spills: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rayon_strategy_defaults() {
        // Default strategy is Auto.
        let s: RayonStrategy = Default::default();
        assert_eq!(
            s,
            RayonStrategy::Auto,
            "default rayon strategy should be Auto"
        );

        // Default threading policy is Hybrid with sensible defaults.
        let p: CpuThreadingPolicy = Default::default();
        match p {
            CpuThreadingPolicy::Hybrid {
                threshold_elements,
                parallel_strategy,
            } => {
                assert_eq!(threshold_elements, 4096, "hybrid threshold should be 4096");
                assert_eq!(
                    parallel_strategy,
                    RayonStrategy::WorkStealing,
                    "hybrid parallel strategy should be WorkStealing"
                );
            }
            _ => panic!("default threading policy should be Hybrid"),
        }

        // Default scratch plan is a single spill with 0 bytes.
        let sp: CpuScratchPlan = Default::default();
        assert_eq!(sp.spill_buffer_size, 0);
        assert_eq!(sp.spill_buffer_alignment, 64);
        assert!(sp.alias_groups.is_empty());
        assert_eq!(sp.max_concurrent_spills, 1);
    }
}
