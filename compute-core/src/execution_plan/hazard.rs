//! Hazard checking and arena planning for execution regions.
//!
//! Provides [`HazardChecker`] for analysing buffer-access dependencies within an
//! [`ExecutionRegion`] and [`ArenaPlanner`] for alias-compacting scratch allocations
//! via interval-graph coloring.

use std::collections::HashMap;

use super::*;

// ---------------------------------------------------------------------------
// HazardChecker
// ---------------------------------------------------------------------------

impl HazardChecker {
    /// Validate an execution region's buffer accesses and aliasing plan.
    ///
    /// Returns an error if two distinct ops write to the same buffer (WAW hazard).
    /// Otherwise returns a [`HazardPlan`] with encoder boundaries and memory
    /// barriers for RAW (read-after-write) and WAR (write-after-read) dependencies.
    pub fn validate_region(region: &ExecutionRegion) -> Result<HazardPlan, HazardError> {
        let mut boundaries: Vec<EncoderBoundary> = Vec::new();
        let mut barriers: Vec<MemoryBarrier> = Vec::new();
        let mut hazards_found = false;

        // ── 1. Group accesses by buffer ──────────────────────────────────────
        let mut buf_accesses: HashMap<&str, Vec<(usize, AccessMode, &str)>> = HashMap::new();
        for (idx, op) in region.ops.iter().enumerate() {
            for use_ in &op.buffer_uses {
                buf_accesses
                    .entry(use_.buffer_id.as_str())
                    .or_default()
                    .push((idx, use_.access, op.op_id.as_str()));
            }
        }

        // ── 2. Scan each buffer's access sequence ──────────────────────────
        for (buf_id, accesses) in &buf_accesses {
            // Accesses are in op order because we pushed them in insertion order.
            // For each buffer, track the sequence of accesses and detect hazards.
            let mut last_write_idx: Option<usize> = None;
            let mut last_write_op: Option<&str> = None;
            // op indices that are still "live reads" — a later write would clobber them.
            let mut pending_reads: Vec<usize> = Vec::new();

            for &(idx, mode, op_id) in accesses {
                let is_read = matches!(mode, AccessMode::Read | AccessMode::ReadWrite);
                let is_write = matches!(mode, AccessMode::Write | AccessMode::ReadWrite);

                // ── WAW: any earlier write from a different op ──────────────
                if is_write && last_write_idx.is_some_and(|lw| lw != idx) {
                    return Err(HazardError::OverlappingReadWrite {
                        buffer_id: buf_id.to_string(),
                        op_a: last_write_op.unwrap_or("?").to_string(),
                        op_b: op_id.to_string(),
                    });
                }

                // ── RAW: this read depends on the last writer ───────────────
                if is_read {
                    if let Some(w_idx) = last_write_idx {
                        if w_idx < idx {
                            boundaries.push(EncoderBoundary {
                                after_op_index: w_idx,
                                reason: format!(
                                    "RAW boundary: {} reads {} written by {}",
                                    op_id, buf_id, last_write_op.unwrap_or("?")
                                ),
                            });
                            barriers.push(MemoryBarrier {
                                after_op_index: w_idx,
                                before_op_index: idx,
                                mem_type: format!("buffer:{}", buf_id),
                            });
                            hazards_found = true;
                        }
                    }
                    pending_reads.push(idx);
                }

                // ── WAR: this write clobbers pending live reads ─────────────
                if is_write {
                    for &r_idx in &pending_reads {
                        if r_idx < idx {
                            boundaries.push(EncoderBoundary {
                                after_op_index: r_idx,
                                reason: format!(
                                    "WAR boundary: {} writes {} while reader {} is live",
                                    op_id, buf_id, r_idx
                                ),
                            });
                            barriers.push(MemoryBarrier {
                                after_op_index: r_idx,
                                before_op_index: idx,
                                mem_type: format!("buffer:{}", buf_id),
                            });
                            hazards_found = true;
                        }
                    }
                    pending_reads.clear();
                    last_write_idx = Some(idx);
                    last_write_op = Some(op_id);
                }
            }
        }

        // ── 3. Check cyclic dependencies ────────────────────────────────────
        let op_count = region.ops.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); op_count];
        for b in &boundaries {
            if b.after_op_index + 1 < op_count {
                adj[b.after_op_index].push(b.after_op_index + 1);
            }
        }
        for b in &barriers {
            let f = b.after_op_index;
            let t = b.before_op_index;
            if f < op_count && t < op_count && f != t {
                adj[f].push(t);
            }
        }
        if has_cycle(&adj) {
            let ids: Vec<String> = region.ops.iter().map(|o| o.op_id.clone()).collect();
            return Err(HazardError::CyclicDependency { ops: ids });
        }

        Ok(HazardPlan {
            encoder_boundaries: boundaries,
            required_barriers: barriers,
            aliasing_approved: !hazards_found,
            safe: !hazards_found,
        })
    }
}

/// DFS-based cycle detection on a directed graph represented as adjacency lists.
fn has_cycle(adj: &[Vec<usize>]) -> bool {
    let n = adj.len();
    if n == 0 {
        return false;
    }
    let mut state = vec![0u8; n]; // 0=unvisited, 1=in-stack, 2=done
    fn dfs(v: usize, adj: &[Vec<usize>], state: &mut [u8]) -> bool {
        state[v] = 1;
        for &w in &adj[v] {
            match state[w] {
                1 => return true,
                0 => {
                    if dfs(w, adj, state) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        state[v] = 2;
        false
    }
    (0..n).any(|v| state[v] == 0 && dfs(v, adj, &mut state))
}

// ---------------------------------------------------------------------------
// ArenaPlanner
// ---------------------------------------------------------------------------

impl ArenaPlanner {
    /// Plan scratch buffer allocation with interval-based aliasing.
    ///
    /// Collects [`BufferUse`] records from every op, computes lifetime intervals
    /// per logical buffer, and uses interval-graph colouring to assign offsets
    /// so that buffers with non-overlapping lifetimes share storage.
    ///
    /// Buffers with [`LifetimeClass::PersistentWeight`] or
    /// [`LifetimeClass::PersistentKvCache`] are excluded from arena allocation.
    pub fn plan_arena(ops: &[ScheduledKernelOp], arena_id: &str) -> ActivationArenaPlan {
        // ── Step 1: Collect per-buffer lifetimes ────────────────────────────
        struct BufferSlot {
            buffer_id: String,
            size_bytes: u64,
            alignment_bytes: u32,
            first_op: usize,
            last_op: usize,
            alias_group: Option<String>,
        }

        let mut slots: HashMap<String, BufferSlot> = HashMap::new();

        for (op_idx, op) in ops.iter().enumerate() {
            for use_ in &op.buffer_uses {
                // Skip persistent buffers — they live outside the arena.
                if matches!(
                    use_.lifetime,
                    LifetimeClass::PersistentWeight | LifetimeClass::PersistentKvCache
                ) {
                    continue;
                }

                let size = use_
                    .byte_range
                    .as_ref()
                    .map(|r| r.end - r.start)
                    .unwrap_or(0);
                if size == 0 {
                    continue;
                }

                let entry = slots.entry(use_.buffer_id.clone()).or_insert_with(|| {
                    BufferSlot {
                        buffer_id: use_.buffer_id.clone(),
                        size_bytes: size,
                        alignment_bytes: 256,
                        first_op: op_idx,
                        last_op: op_idx,
                        alias_group: use_.alias_group.clone(),
                    }
                });

                entry.size_bytes = entry.size_bytes.max(size);
                entry.first_op = entry.first_op.min(op_idx);
                entry.last_op = entry.last_op.max(op_idx);
            }
        }

        // ── Step 2: Sort by descending size ─────────────────────────────────
        let mut sorted: Vec<BufferSlot> = slots.into_values().collect();
        sorted.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));

        // ── Step 3: Interval-graph colouring ────────────────────────────────
        let mut placed: Vec<PlacedSlot> = Vec::new();
        let mut allocations: Vec<ArenaAllocation> = Vec::new();

        for slot in &sorted {
            let aligned = find_colored_offset(
                &placed,
                slot.size_bytes,
                slot.alignment_bytes,
                slot.first_op,
                slot.last_op,
            );

            allocations.push(ArenaAllocation {
                logical_buffer_id: slot.buffer_id.clone(),
                offset: aligned,
                size_bytes: slot.size_bytes,
                alignment_bytes: slot.alignment_bytes,
                lifetime_start_op: slot.first_op,
                lifetime_end_op: slot.last_op,
                alias_group: slot.alias_group.clone(),
            });

            placed.push(PlacedSlot {
                offset: aligned,
                size: slot.size_bytes,
                first_op: slot.first_op,
                last_op: slot.last_op,
            });
        }

        // ── Step 4: Compute total bytes and peak live bytes ─────────────────
        let total_bytes = placed
            .iter()
            .map(|p| p.offset + p.size)
            .max()
            .unwrap_or(0);

        let peak_live_bytes = if ops.is_empty() || placed.is_empty() {
            0
        } else {
            let n = ops.len();
            let mut peak = 0u64;
            for op_idx in 0..n {
                let live: u64 = placed
                    .iter()
                    .filter(|p| p.first_op <= op_idx && op_idx <= p.last_op)
                    .map(|p| p.size)
                    .sum();
                peak = peak.max(live);
            }
            peak
        };

        // ── Step 5: Build alias groups ──────────────────────────────────────
        let mut alias_groups: Vec<AliasGroupPlan> = Vec::new();
        {
            let mut by_group: HashMap<&str, Vec<&str>> = HashMap::new();
            for a in &allocations {
                if let Some(ref g) = a.alias_group {
                    by_group.entry(g.as_str()).or_default().push(a.logical_buffer_id.as_str());
                }
            }
            for (gid, members) in &by_group {
                let mut group_total = 0u64;
                for m in members {
                    if let Some(a) = allocations.iter().find(|a| a.logical_buffer_id == *m) {
                        group_total = group_total.max(a.offset + a.size_bytes);
                    }
                }
                alias_groups.push(AliasGroupPlan {
                    group_id: gid.to_string(),
                    members: members.iter().map(|s| s.to_string()).collect(),
                    total_bytes: group_total,
                });
            }
        }

        ActivationArenaPlan {
            arena_id: arena_id.to_string(),
            total_bytes,
            allocations,
            alias_groups,
            peak_live_bytes,
        }
    }
    }

/// A slot that has already been placed in the arena during interval coloring.
struct PlacedSlot {
    offset: u64,
    size: u64,
    first_op: usize,
    last_op: usize,
}

/// Interval-graph coloring: find the smallest aligned offset for a buffer of
/// `size` bytes with lifetime `[first_op, last_op]` that does not conflict with
/// any already-placed buffer.
///
/// Two buffers conflict when their lifetimes overlap *and* their byte ranges
/// overlap. Non-overlapping lifetimes can alias (share the same offset).
fn find_colored_offset(
    placed: &[PlacedSlot],
    size: u64,
    alignment: u32,
    first_op: usize,
    last_op: usize,
) -> u64 {
    let align = alignment as u64;

    // Candidate offsets: 0 and immediately after each placed buffer.
    // We also check if the gap between two non-adjacent placed buffers can fit.
    let mut candidates: Vec<u64> = vec![0];
    for p in placed {
        candidates.push(p.offset + p.size);
    }
    // Also try the start of each placed buffer (in case the new buffer can fit
    // in a gap where the preceding placed buffer doesn't overlap in lifetime).
    for p in placed {
        candidates.push(p.offset);
    }
    candidates.sort();
    candidates.dedup();

    for &raw in &candidates {
        let candidate = align_up(raw, align);
        let new_end = candidate + size;

        let mut conflict = false;
        for p in placed {
            // No lifetime overlap? → no conflict (they can alias).
            let lifetime_overlap = first_op <= p.last_op && last_op >= p.first_op;
            if !lifetime_overlap {
                continue;
            }

            // Byte-range overlap check: [candidate, candidate+size) vs [p.offset, p.offset+p.size)
            let byte_overlap = candidate < p.offset + p.size && p.offset < new_end;
            if byte_overlap {
                conflict = true;
                break;
            }
        }

        if !conflict {
            return candidate;
        }
    }

    // Fallback: place right after the rightmost placed buffer.
    let max_end = placed
        .iter()
        .map(|p| p.offset + p.size)
        .max()
        .unwrap_or(0);
    align_up(max_end, align)
}

fn align_up(addr: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return addr;
    }
    (addr + alignment - 1) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────────────

    fn kernel_specialization_key_fixture() -> KernelSpecializationKey {
        KernelSpecializationKey {
            template_id: KernelTemplateId::Nf4Tile640Gemv,
            execution_phase: ExecutionPhase::Decode,
            codec: CodecFamily::Nf4,
            tile_shape: TileShape::tile640_decode(),
            group_size: 32,
            group_axis: Axis::PackedContiguous,
            affine_mode: AffineMode::ScaleOnly,
            metadata_layout: MetadataLayout::AdjacentTile,
            input_dtype: DType::F32,
            output_dtype: DType::F16,
            hardware_profile: HardwareProfileId::AppleMBaseMemoryBound,
            mode_flags: 0,
        }
    }

    fn make_dispatch() -> DispatchShape {
        DispatchShape {
            grid_x: 1,
            grid_y: 1,
            grid_z: 1,
            threadgroup_m: 32,
            threadgroup_n: 1,
            threadgroup_p: 1,
        }
    }

    fn make_cost() -> EstimatedKernelCost {
        EstimatedKernelCost {
            compute_us: 1.0,
            memory_bytes_read: 0,
            memory_bytes_written: 0,
        }
    }

    fn make_validation() -> KernelValidationRequirements {
        KernelValidationRequirements {
            allows_in_place_input_output: false,
            requires_zeroed_output: false,
            requires_aligned_metadata: false,
            requires_hardware_validation: false,
        }
    }

    fn make_op(
        op_id: &str,
        op_kind: KernelOpKind,
        buffer_uses: Vec<BufferUse>,
    ) -> ScheduledKernelOp {
        ScheduledKernelOp {
            op_id: op_id.into(),
            op_kind,
            tensor_key: None,
            tensor_class: None,
            specialization: kernel_specialization_key_fixture(),
            bindings: vec![],
            dependencies: vec![],
            buffer_uses,
            dispatch_shape: make_dispatch(),
            estimated_cost: make_cost(),
            validation_requirements: make_validation(),
        }
    }

    fn make_use(
        buf_id: &str,
        access: AccessMode,
        lifetime: LifetimeClass,
        alias_group: Option<&str>,
        start: u64,
        size: u64,
    ) -> BufferUse {
        BufferUse {
            buffer_id: buf_id.into(),
            access,
            lifetime,
            alias_group: alias_group.map(String::from),
            byte_range: Some(ByteRange {
                start,
                end: start + size,
            }),
        }
    }

    fn make_default_arena() -> ActivationArenaPlan {
        ActivationArenaPlan {
            arena_id: "test".into(),
            total_bytes: 0,
            allocations: vec![],
            alias_groups: vec![],
            peak_live_bytes: 0,
        }
    }

    // ── Arena planner tests ────────────────────────────────────────────────

    #[test]
    fn test_non_overlapping_scratch_buffers_alias() {
        // Two scratch buffers, each used by a different op (non-overlapping
        // lifetimes). Interval coloring should let them share arena space.
        let ops = vec![
            make_op(
                "op_a",
                KernelOpKind::RmsNorm,
                vec![make_use("buf_a", AccessMode::Write, LifetimeClass::OpScratch, None, 0, 4096)],
            ),
            make_op(
                "op_b",
                KernelOpKind::QkvProjection,
                vec![make_use("buf_b", AccessMode::Write, LifetimeClass::OpScratch, None, 0, 4096)],
            ),
        ];

        let plan = ArenaPlanner::plan_arena(&ops, "arena1");

        // Lifetime [0,0] for buf_a, [1,1] for buf_b → no overlap → alias.
        // Total bytes should be ≤ max of both sizes (≤ 4096), not the sum (8192).
        assert!(
            plan.total_bytes <= 4096,
            "non-overlapping buffers should alias: total_bytes={}",
            plan.total_bytes
        );
        assert_eq!(plan.peak_live_bytes, 4096, "only one live at a time");
    }

    #[test]
    fn test_overlapping_read_write_buffers_do_not_alias() {
        // Two buffers with overlapping lifetimes must not get the same offset.
        let ops = vec![make_op(
            "op1",
            KernelOpKind::RmsNorm,
            vec![
                make_use("input", AccessMode::Read, LifetimeClass::RegionInput, None, 0, 512),
                make_use("scratch", AccessMode::Write, LifetimeClass::OpScratch, None, 0, 4096),
            ],
        )];

        let plan = ArenaPlanner::plan_arena(&ops, "arena1");

        let input_alloc = plan
            .allocations
            .iter()
            .find(|a| a.logical_buffer_id == "input")
            .expect("input allocation should exist");
        let scratch_alloc = plan
            .allocations
            .iter()
            .find(|a| a.logical_buffer_id == "scratch")
            .expect("scratch allocation should exist");

        // Both live at op index 0 → overlapping lifetimes → must not alias.
        let a_end = input_alloc.offset + input_alloc.size_bytes;
        let b_end = scratch_alloc.offset + scratch_alloc.size_bytes;
        let overlap_start = input_alloc.offset.max(scratch_alloc.offset);
        let overlap_end = a_end.min(b_end);
        assert!(
            overlap_start >= overlap_end,
            "overlapping-lifetime buffers must not alias"
        );
        assert!(
            plan.total_bytes >= 4096 + 512,
            "overlapping buffers need separate arena space, total_bytes={}",
            plan.total_bytes
        );
    }

    #[test]
    fn test_persistent_kv_never_aliases() {
        // PersistentKvCache buffers are excluded from arena allocation.
        let ops = vec![
            make_op(
                "op1",
                KernelOpKind::AttentionScore,
                vec![make_use("kv", AccessMode::Read, LifetimeClass::PersistentKvCache, None, 0, 1_000_000)],
            ),
            make_op(
                "op2",
                KernelOpKind::AttentionApply,
                vec![make_use("scratch", AccessMode::Write, LifetimeClass::LayerScratch, None, 0, 8192)],
            ),
        ];

        let plan = ArenaPlanner::plan_arena(&ops, "arena1");

        assert!(
            !plan.allocations.iter().any(|a| a.logical_buffer_id == "kv"),
            "PersistentKvCache must not get arena allocations"
        );
        assert_eq!(plan.allocations.len(), 1);
        assert_eq!(plan.allocations[0].logical_buffer_id, "scratch");
        assert_eq!(plan.allocations[0].size_bytes, 8192);
    }

    #[test]
    fn test_region_peak_scratch_is_less_than_naive_sum() {
        // Three ops with independent scratch buffers → interval coloring aliases
        // them so peak_live_bytes < naive sum.
        let ops = vec![
            make_op("op1", KernelOpKind::RmsNorm, vec![make_use("s1", AccessMode::Write, LifetimeClass::OpScratch, None, 0, 4096)]),
            make_op("op2", KernelOpKind::QkvProjection, vec![make_use("s2", AccessMode::Write, LifetimeClass::OpScratch, None, 0, 8192)]),
            make_op("op3", KernelOpKind::AttentionScore, vec![make_use("s3", AccessMode::Write, LifetimeClass::OpScratch, None, 0, 2048)]),
        ];

        let plan = ArenaPlanner::plan_arena(&ops, "arena1");
        let naive_sum: u64 = ops
            .iter()
            .flat_map(|o| &o.buffer_uses)
            .filter_map(|u| {
                (!matches!(
                    u.lifetime,
                    LifetimeClass::PersistentWeight | LifetimeClass::PersistentKvCache
                ))
                .then(|| u.byte_range.as_ref().map(|r| r.end - r.start))?
            })
            .sum();

        assert_eq!(
            plan.peak_live_bytes, 8192,
            "peak live should be the max single scratch size (non-overlapping lifetimes)"
        );
        assert!(
            plan.peak_live_bytes < naive_sum,
            "peak_live_bytes={} < naive_sum={}",
            plan.peak_live_bytes,
            naive_sum
        );
        // Total arena bytes should be ≤ the largest buffer (they all alias).
        assert!(
            plan.total_bytes <= 8192,
            "total_bytes={} should be <= 8192",
            plan.total_bytes
        );
    }

    // ── Hazard checker tests ───────────────────────────────────────────────

    #[test]
    fn test_hazard_raw_detects_dependency() {
        // Op0 writes buf_a → Op1 reads buf_a → RAW dependency.
        let region = ExecutionRegion {
            region_id: "raw_test".into(),
            region_kind: ExecutionRegionKind::DecoderLayerDecode,
            layer_index: Some(0),
            phase: ExecutionPhase::Decode,
            ops: vec![
                make_op("producer", KernelOpKind::RmsNorm, vec![make_use("buf_a", AccessMode::Write, LifetimeClass::LayerScratch, None, 0, 4096)]),
                make_op("consumer", KernelOpKind::AttentionScore, vec![make_use("buf_a", AccessMode::Read, LifetimeClass::LayerScratch, None, 0, 4096)]),
            ],
            command_buffer_policy: CommandBufferPolicy::decode_default(),
            hazard_policy: HazardPolicy::Conservative,
            arena_plan: make_default_arena(),
            timing_policy: TimingPolicy::Disabled,
        };

        let plan = HazardChecker::validate_region(&region).expect("RAW should not produce error");
        assert!(
            !plan.encoder_boundaries.is_empty(),
            "RAW dependency must produce encoder boundaries"
        );
        assert!(
            !plan.required_barriers.is_empty(),
            "RAW dependency must produce memory barriers"
        );
        assert!(!plan.safe, "RAW hazard means region is not safe");
    }

    #[test]
    fn test_hazard_overlapping_writes_rejected() {
        // Two ops writing the same buffer → Err(OverlappingReadWrite).
        let region = ExecutionRegion {
            region_id: "waw_test".into(),
            region_kind: ExecutionRegionKind::DecoderLayerDecode,
            layer_index: Some(0),
            phase: ExecutionPhase::Decode,
            ops: vec![
                make_op("op1", KernelOpKind::RmsNorm, vec![make_use("shared", AccessMode::Write, LifetimeClass::OpScratch, None, 0, 1024)]),
                make_op("op2", KernelOpKind::QkvProjection, vec![make_use("shared", AccessMode::Write, LifetimeClass::OpScratch, None, 0, 1024)]),
            ],
            command_buffer_policy: CommandBufferPolicy::decode_default(),
            hazard_policy: HazardPolicy::Conservative,
            arena_plan: make_default_arena(),
            timing_policy: TimingPolicy::Disabled,
        };

        let result = HazardChecker::validate_region(&region);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            HazardError::OverlappingReadWrite { buffer_id, .. } => {
                assert_eq!(buffer_id, "shared");
            }
            _ => panic!("expected OverlappingReadWrite error"),
        }
    }
}
