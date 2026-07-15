use crate::ecs::component::backend::BackendTarget;
use crate::ecs::component::memory::{BufferLifetime, MemoryDomain, MemoryPool, PoolPolicy};
use crate::ecs::component::tensor::{CanonicalRoleComp, CodecFamilyComp, Shape};
use crate::ecs::plan::CodecFamily;

use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

// ── Helpers ────────────────────────────────────────────────────────────────

/// Compute the storage allocation size in bytes for a tensor given its shape
/// and codec-family descriptor (codec + group_size).
fn compute_storage_bytes(shape: &Shape, codec: CodecFamilyComp) -> u64 {
    let elem_count: u64 = shape.0.iter().map(|&d| d as u64).product();
    let (family, group_size) = (codec.0, codec.1);
    codec_storage_bytes(elem_count, family, group_size)
}

/// Raw storage bytes needed for `elem_count` elements under a given codec family.
///
/// For block-quantized formats the total includes block-level metadata (scales,
/// mins / maxes), producing realistic allocation sizes for buffer planning.
fn codec_storage_bytes(elem_count: u64, codec: CodecFamily, group_size: u32) -> u64 {
    match codec {
        // ── Unquantized float / integer ──
        CodecFamily::RawF32 | CodecFamily::Mixed => elem_count.saturating_mul(4),
        CodecFamily::Fp16 => elem_count.saturating_mul(2),
        CodecFamily::Int8 => elem_count,

        // ── 4-bit per element (packed) ──
        CodecFamily::Nf4 | CodecFamily::SymInt4 => {
            let bits = elem_count.saturating_mul(4);
            bits.div_ceil(8)
        }

        // ── 2-bit ternary ──
        CodecFamily::Ternary | CodecFamily::Ternary1_58 => {
            let bits = elem_count.saturating_mul(2);
            bits.div_ceil(8)
        }

        // ── Block-quantized: Q8_0 (1 byte data + 2 byte scale per block) ──
        CodecFamily::Q8_0 => {
            let block = group_size.max(1) as u64;
            let blocks = elem_count.div_ceil(block);
            elem_count + blocks.saturating_mul(2) // 2 bytes block scale
        }

        // ── Block-quantized: Q4_K (4-bit data + 16 bytes block metadata) ──
        CodecFamily::Q4_K => {
            let block = group_size.max(1) as u64;
            let blocks = elem_count.div_ceil(block);
            let data_bytes = elem_count.saturating_mul(4).div_ceil(8);
            data_bytes + blocks.saturating_mul(16)
        }

        // ── Block-quantized: Q2_K (2-bit data + 16 bytes block metadata) ──
        CodecFamily::Q2_K => {
            let block = group_size.max(1) as u64;
            let blocks = elem_count.div_ceil(block);
            let data_bytes = elem_count.saturating_mul(2).div_ceil(8);
            data_bytes + blocks.saturating_mul(16)
        }

        // ── Block-quantized: IQ2_XXS (~2.06 bits/element + light metadata) ──
        CodecFamily::IQ2_XXS => {
            let block = group_size.max(1) as u64;
            let blocks = elem_count.div_ceil(block);
            let data_bytes = (elem_count * 206 + 999) / 1000 / 8; // ~2.06 bits/elem
                                                                  // At minimum each element needs ceil(elem/4) bytes raw
            let data_min = elem_count.div_ceil(4);
            data_bytes.max(data_min) + blocks.saturating_mul(8)
        }
    }
}

// ── MemoryDomainAssignmentSystem ───────────────────────────────────────────

/// Assigns a [`MemoryDomain`] to every tensor based on its [`BackendTarget`].
///
/// GPU targets (Metal, ROCm, CUDA, Vulkan) → `DeviceLocal`.
/// CPU target → `HostVisible`.
pub struct MemoryDomainAssignmentSystem;

impl CompilerSystem for MemoryDomainAssignmentSystem {
    fn name(&self) -> &str {
        "MemoryDomainAssignmentSystem"
    }

    fn phase(&self) -> SchedulePhase {
        SchedulePhase::MemoryPlanning
    }

    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let tensors = world.entities_of_kind(EntityKind::Tensor);
        for tensor in tensors {
            let target = world
                .get_component::<BackendTarget>(tensor)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "MemoryDomainAssignment: TensorEntity {:?} has no BackendTarget",
                        tensor
                    )
                })?;

            let domain = match target {
                BackendTarget::Metal
                | BackendTarget::ROCm
                | BackendTarget::CUDA
                | BackendTarget::Vulkan => MemoryDomain::DeviceLocal,
                BackendTarget::CPU => MemoryDomain::HostVisible,
            };

            let _ = world.add_component(tensor, domain);;
        }
        Ok(())
    }
}

// ── BufferAllocationSystem ─────────────────────────────────────────────────

/// Creates a [`BufferEntity`] for every tensor that has assembled its shape,
/// codec, and domain information, assigning [`MemoryPool`] and
/// [`BufferLifetime`] components.
///
/// Pool policy decisions:
/// - **Dedicated** — for tensors carrying a [`CanonicalRoleComp`] (weights).
/// - **Arena** — for scratch / intermediate tensors.
///
/// All arena-scoped buffers share pool id `0`; each dedicated allocation
/// receives its own pool id.
pub struct BufferAllocationSystem;

impl CompilerSystem for BufferAllocationSystem {
    fn name(&self) -> &str {
        "BufferAllocationSystem"
    }

    fn phase(&self) -> SchedulePhase {
        SchedulePhase::MemoryPlanning
    }

    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let tensors = world.entities_of_kind(EntityKind::Tensor);
        let mut next_dedicated_pool: u32 = 1; // 0 is reserved for arena

        for tensor in tensors {
            // Skip tensors that haven't been fully described yet.
            let Some(shape) = world.get_component::<Shape>(tensor) else {
                continue;
            };
            let Some(codec) = world.get_component::<CodecFamilyComp>(tensor) else {
                continue;
            };
            let _domain = match world.get_component::<MemoryDomain>(tensor) {
                Some(d) => *d,
                None => continue,
            };

            let storage_bytes = compute_storage_bytes(shape, *codec);

            // ── Decide pool policy ──────────────────────────────────────
            // Tensors with a canonical role are model weights → dedicated.
            // Everything else is a scratch / activation buffer → arena.
            let is_weight = world.get_component::<CanonicalRoleComp>(tensor).is_some();

            let (policy, pool_id) = if is_weight {
                let pid = next_dedicated_pool;
                next_dedicated_pool += 1;
                (PoolPolicy::Dedicated, pid)
            } else {
                (PoolPolicy::Arena, 0)
            };

            // ── Allocate buffer entity ──────────────────────────────────
            let buffer = world.spawn(EntityKind::Buffer, None)?;

            let _ = world.add_component(buffer,
            MemoryPool {
                policy,
                pool_id,
                total_bytes: storage_bytes,
                used_bytes: 0,
            },);;

            let _ = world.add_component(buffer,
            BufferLifetime {
                alloc_epoch: 0,       // allocated in planning phase
                free_epoch: u64::MAX, // unknown until liveness analysis
                causal_death_frontier: None,
            },);;
        }

        Ok(())
    }
}
