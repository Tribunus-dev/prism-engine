//! Cross-platform compute-backend abstraction (scaffold).
//!
//! This is the seam the cross-platform roadmap hangs off (see the
//! "Cross-Platform Production Hardening Guide", §2). Today the GPU path is
//! Metal-specific; this trait generalizes *execution* across Apple Metal,
//! NVIDIA/AMD/Intel GPUs, matmul-only NPUs, and a portable CPU fallback.
//!
//! Design rules embodied here:
//!   * **Object-safe** — a heterogeneous machine holds `Vec<Box<dyn ComputeBackend>>`.
//!   * **Capability-driven routing** — the scheduler routes by [`BackendCaps`],
//!     never by `cfg!(target_os = ...)`.
//!   * **The oracle is mandatory** — every backend must reproduce the reference
//!     kernel within tolerance. [`CpuBackend`] *is* that reference, and the
//!     tests below are the admission gate any future backend must also pass.
//!
//! This module is deliberately dependency-free so it builds and is tested on
//! every platform (including Linux CI). Device backends (CUDA/Metal/…) live
//! behind their own feature flags and implement the same trait + oracle test.

/// Where a buffer physically lives. Drives whether `upload` is a real DMA or a
/// zero-copy view (unified-memory systems: Apple Silicon, Strix Halo, Lunar Lake).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemKind {
    /// CPU/GPU/NPU share one physical pool — uploads are zero-copy views.
    Unified,
    /// Discrete device memory (dGPU VRAM) — uploads cross PCIe.
    Discrete,
}

/// Static capabilities a backend advertises so the router can place work.
#[derive(Debug, Clone)]
pub struct BackendCaps {
    /// Human-readable backend name, e.g. "cpu", "metal", "cuda:0".
    pub name: String,
    /// Memory model — gates zero-copy vs. explicit residency.
    pub mem_kind: MemKind,
    /// Has a planar/LUT gather engine (Apple ANE). If false (Intel/AMD NPUs,
    /// most GPUs) the router must use the dequant-then-matmul prefill route.
    pub has_planar_lut: bool,
    /// Native fp16 arithmetic.
    pub supports_fp16: bool,
    /// Largest batch the device handles efficiently (routing hint).
    pub max_batch: u32,
}

/// Reference representation of a block-scaled ternary weight matrix.
///
/// Values are `{-1, 0, +1}` (one `i8` each in this portable reference; the
/// production `.cimage` packs them as base-3 `tile640` `u32`s). Each contiguous
/// run of `block_size` logical elements shares one fp32 scale.
#[derive(Debug, Clone)]
pub struct TernaryWeights {
    pub out_dim: usize,
    pub in_dim: usize,
    pub block_size: usize,
    /// Row-major ternary digits, length `out_dim * in_dim`.
    pub digits: Vec<i8>,
    /// One scale per `block_size` elements, length `ceil(out_dim*in_dim/block_size)`.
    pub scales: Vec<f32>,
}

impl TernaryWeights {
    /// Dequantized weight at (row, col): `digit * block_scale`.
    #[inline]
    pub fn dequant(&self, row: usize, col: usize) -> f32 {
        let flat = row * self.in_dim + col;
        let blk = flat / self.block_size;
        self.digits[flat] as f32 * self.scales[blk]
    }
}

/// The minimal execution surface every backend implements. Kept intentionally
/// small: the fused ternary GEMV is the hot kernel that dominates decode, so it
/// is the first thing a new backend must implement and pass the oracle on.
pub trait ComputeBackend: Send + Sync {
    fn caps(&self) -> BackendCaps;

    /// Fused dequant + GEMV: `y[o] = Σ_i dequant(w)[o,i] * x[i]`.
    ///
    /// Real backends fuse the dequant into the matmul so the expanded weights
    /// never leave on-chip memory (this reference does it inline for clarity).
    fn fused_ternary_gemv(&self, w: &TernaryWeights, x: &[f32], y: &mut [f32]);
}

/// Portable, dependency-free reference backend. This defines *correct* — every
/// accelerated backend is validated against the output of this one.
#[derive(Debug, Default)]
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn caps(&self) -> BackendCaps {
        BackendCaps {
            name: "cpu".to_string(),
            mem_kind: MemKind::Unified, // host memory; uploads are no-ops
            has_planar_lut: false,
            supports_fp16: false,
            max_batch: u32::MAX,
        }
    }

    fn fused_ternary_gemv(&self, w: &TernaryWeights, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), w.in_dim, "input dim mismatch");
        assert_eq!(y.len(), w.out_dim, "output dim mismatch");
        for o in 0..w.out_dim {
            let mut acc = 0.0f32;
            for i in 0..w.in_dim {
                acc += w.dequant(o, i) * x[i];
            }
            y[o] = acc;
        }
    }
}

/// Capability-based routing: pick the first backend satisfying a requirement.
/// The real scheduler (ECS, see guide §4) extends this with load and residency;
/// this shows the seam is cap-driven, not `cfg`-driven.
pub fn route<'a>(
    backends: &'a [Box<dyn ComputeBackend>],
    require_planar: bool,
) -> Option<&'a dyn ComputeBackend> {
    backends
        .iter()
        .map(|b| b.as_ref())
        .find(|b| !require_planar || b.caps().has_planar_lut)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive, independent oracle — a second implementation of the same math so
    /// the test does not merely re-run the code under test.
    fn oracle_gemv(w: &TernaryWeights, x: &[f32]) -> Vec<f32> {
        (0..w.out_dim)
            .map(|o| {
                (0..w.in_dim)
                    .map(|i| {
                        let flat = o * w.in_dim + i;
                        let scale = w.scales[flat / w.block_size];
                        (w.digits[flat] as f32 * scale) * x[i]
                    })
                    .sum()
            })
            .collect()
    }

    fn make_weights(out_dim: usize, in_dim: usize, block_size: usize) -> TernaryWeights {
        let n = out_dim * in_dim;
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 33
        };
        let digits: Vec<i8> = (0..n).map(|_| (next() % 3) as i8 - 1).collect(); // {-1,0,1}
        let n_blocks = n.div_ceil(block_size);
        let scales: Vec<f32> = (0..n_blocks)
            .map(|_| 0.05 + (next() % 100) as f32 / 1000.0)
            .collect();
        TernaryWeights { out_dim, in_dim, block_size, digits, scales }
    }

    #[test]
    fn cpu_backend_matches_oracle() {
        let w = make_weights(64, 320, 256);
        let x: Vec<f32> = (0..w.in_dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut y = vec![0.0f32; w.out_dim];

        CpuBackend.fused_ternary_gemv(&w, &x, &mut y);
        let expected = oracle_gemv(&w, &x);

        for (o, (got, exp)) in y.iter().zip(&expected).enumerate() {
            assert!(
                (got - exp).abs() <= 1e-4 * exp.abs().max(1.0),
                "row {o}: backend {got} vs oracle {exp}"
            );
        }
    }

    #[test]
    fn routing_is_capability_driven() {
        let backends: Vec<Box<dyn ComputeBackend>> = vec![Box::new(CpuBackend)];
        // No planar/LUT device present → planar-required work has nowhere to go.
        assert!(route(&backends, true).is_none());
        // Any backend can take non-planar work.
        assert!(route(&backends, false).is_some());
        assert_eq!(route(&backends, false).unwrap().caps().name, "cpu");
    }
}
