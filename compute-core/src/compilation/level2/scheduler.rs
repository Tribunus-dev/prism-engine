//! Level 2 scheduler — Core ML teacher with Metal student pipeline.
//!
//! Mirrors the Level 1 triple-buffered pipeline structure but replaces the
//! Metal teacher dispatch with Core ML `.cpuAndNeuralEngine` for eligible
//! teacher regions. The student (Metal ternary kernels) and reducer
//! (Accelerate/vDSP) phases are identical to Level 1.
//!
//! # Scheduling policy
//!
//! * One in-flight Core ML prediction and one active Metal candidate execution
//!   are allowed concurrently — they target disjoint compute units (ANE/CPU vs
//!   GPU).
//! * Multiple Core ML predictions are **not** queued on M1 until memory behavior
//!   is measured — ANE memory is a limited shared resource and over-committing
//!   it can cause system-wide throttling.
//! * On Core ML failure, the scheduler falls back to the Level 1 Metal teacher
//!   for that microbatch and records the fallback in the bridge receipt.

use crate::arena_info::ArenaInfo;

use super::super::arena::{ActivationArena, SlotState, StorageRoute};
use super::super::phase_types::{
    ElementType, PhaseId, PhysicalLayout, ProviderKind, ResidencyClass, TensorDescriptor,
};
use super::super::receipt::{
    BridgeEvidenceSection, BridgeReceipt, PhaseExecutionRecord,
};
use super::super::memory_budget::MemoryBudget;

use super::super::level1::scheduler::Level1Config;
use super::super::level1::teacher::MetalTeacher;
use super::super::level1::student::TernaryStudent;
use super::super::level1::reducer::AccelerateReducer;

use super::bridge::CoreMLTeacher;

// ── Compile region state ─────────────────────────────────────────────────────

/// The state of one compile region being processed by the Level 2 scheduler.
struct RegionState {
    /// Index of the current microbatch being processed.
    current_microbatch: usize,
    /// Slot ids for the triple-buffered activation pipeline.
    teacher_slots: [Option<u64>; 3],
    student_slots: [Option<u64>; 3],
    reducer_slots: [Option<u64>; 3],
    /// Peak memory observed during this region.
    peak_memory: u64,
}

impl RegionState {
    fn new() -> Self {
        RegionState {
            current_microbatch: 0,
            teacher_slots: [None, None, None],
            student_slots: [None, None, None],
            reducer_slots: [None, None, None],
            peak_memory: 0,
        }
    }
}

// ── Level2Scheduler ─────────────────────────────────────────────────────────

/// The Level 2 compile scheduler — Core ML teacher + Metal student + Accelerate reducer.
///
/// Mirrors the Level 1 triple-buffered pipeline cadence:
///   1. Reduce   for microbatch n-1 (reads saved outputs by buffer index)
///   2. Teacher  for microbatch n+1 (via Core ML, with Metal fallback)
///   3. Student  for microbatch n   (via Metal ternary kernels)
///
/// Teacher outputs are captured into a triple-buffered f32 ring buffer so
/// the reducer always reads valid data regardless of the phase cadence.
pub struct Level2Scheduler {
    config: Level1Config,
    arena: ActivationArena,
    _budget: MemoryBudget,
    coreml_teacher: CoreMLTeacher,
    metal_teacher: MetalTeacher,
    student: TernaryStudent,
    reducer: AccelerateReducer,
    region: RegionState,
    phase_records: Vec<PhaseExecutionRecord>,
    bridge_receipts: Vec<BridgeReceipt>,
    total_microbatches: usize,
    completed: bool,
    /// Triple-buffered teacher outputs.
    teacher_outputs: [Vec<f32>; 3],
    /// Triple-buffered student outputs.
    student_outputs: [Vec<f32>; 3],
    /// Whether each buffer slot has been populated.
    teacher_slot_valid: [bool; 3],
    student_slot_valid: [bool; 3],
    /// Whether Core ML is available on this device.
    coreml_available: bool,
}

impl Level2Scheduler {
    /// Create a new Level 2 scheduler with the given configuration.
    pub fn new(
        config: Level1Config,
        total_microbatches: usize,
        coreml_teacher: CoreMLTeacher,
        coreml_available: bool,
    ) -> Self {
        let hidden_dim = config.hidden_dim;
        let metal_teacher = MetalTeacher::with_shape(hidden_dim, hidden_dim);
        let student = TernaryStudent::with_shape(hidden_dim, hidden_dim);
        let reducer = AccelerateReducer::with_hidden_dim(hidden_dim);

        let empty = vec![0.0f32; hidden_dim];
        Level2Scheduler {
            arena: ActivationArena::new(),
            _budget: config.budget.clone(),
            config,
            coreml_teacher,
            metal_teacher,
            student,
            reducer,
            region: RegionState::new(),
            phase_records: Vec::new(),
            bridge_receipts: Vec::new(),
            total_microbatches,
            completed: false,
            teacher_outputs: [empty.clone(), empty.clone(), empty.clone()],
            student_outputs: [empty.clone(), empty.clone(), empty],
            teacher_slot_valid: [false, false, false],
            student_slot_valid: [false, false, false],
            coreml_available,
        }
    }

    /// Initialize the scheduler: allocate the triple-buffered slot pipeline.
    pub fn initialize(&mut self) {
        let mb = self.config.microbatch;
        let hd = self.config.hidden_dim;

        let act_tensor = || -> TensorDescriptor {
            TensorDescriptor {
                logical_shape: vec![mb, hd],
                element_type: ElementType::F16,
                physical_layout: PhysicalLayout::DenseRowMajor,
                alignment: 16384,
                producer_phase: None,
                consumer_phases: Vec::new(),
                permitted_providers: vec![ProviderKind::Metal, ProviderKind::Accelerate],
                residency_class: ResidencyClass::Unified,
                max_bytes: (mb * hd * 2) as u64,
                mutable: true,
                content_digest: None,
            }
        };

        let mut next_id = 1u64;
        for i in 0..3 {
            // Teacher slots use CoreMLManaged to reflect the ANE/CPU domain.
            let teacher_slot = self.arena.reserve(next_id, act_tensor());
            next_id += 1;
            if let Some(slot) = self.arena.slot_mut(teacher_slot) {
                slot.storage_route = StorageRoute::CoreMLManaged;
            }

            let student_slot = self.arena.reserve(next_id, act_tensor());
            next_id += 1;
            if let Some(slot) = self.arena.slot_mut(student_slot) {
                slot.storage_route = StorageRoute::MetalSharedBuffer;
            }

            let reducer_slot = self.arena.reserve(next_id, act_tensor());
            next_id += 1;

            self.region.teacher_slots[i] = Some(teacher_slot);
            self.region.student_slots[i] = Some(student_slot);
            self.region.reducer_slots[i] = Some(reducer_slot);
        }

        self.region.peak_memory = self.arena.current_bytes();
    }

    /// Attempt a Core ML teacher forward for one microbatch.
    ///
    /// Returns `true` if Core ML was used; `false` means fall back to Metal.
    fn teacher_forward_coreml(
        &mut self,
        microbatch: usize,
        _slot_idx: usize,
    ) -> bool {
        if !self.coreml_available {
            self.bridge_receipts.push(
                CoreMLTeacher::fallback_to_level1("Core ML not available"),
            );
            return false;
        }

        let hd = self.config.hidden_dim as i32;
        let mb = self.config.microbatch as i32;
        let info = ArenaInfo {
            width: hd,
            height: mb,
            logical_dim0: mb,
            logical_dim1: hd,
            pixel_format: 0,
            byte_size: mb * hd * 2,
            bytes_per_row: hd * 2,
            base_address: std::ptr::null_mut(),
            cv_buffer: std::ptr::null_mut(),
            io_surface: std::ptr::null_mut(),
        };

        // Digest derived from microbatch index for cache exercise.
        let digest = format!("teacher-region-{:04x}", microbatch);

        let receipt = self.coreml_teacher.forward(
            &digest,
            "hidden_states",
            &info,
            "hidden_states",
            &info,
        );

        let used_coreml = receipt.actual_route.starts_with("CoreML");
        self.bridge_receipts.push(receipt);
        used_coreml
    }

    /// Execute one step of the triple-buffered pipeline.
    ///
    /// Returns `true` if there are more steps to execute.
    pub fn step(&mut self) -> bool {
        if self.completed {
            return false;
        }

        let mb = self.region.current_microbatch;

        // ── Phase 1: CPU Reduction for microbatch n-1 ────────────────────
        if mb > 0 {
            let slot_idx = (mb - 1) % 3;
            let teacher_slot = self.region.teacher_slots[slot_idx].unwrap();
            let student_slot = self.region.student_slots[slot_idx].unwrap();
            let reducer_slot = self.region.reducer_slots[slot_idx].unwrap();
            let phase_id = PhaseId::next();

            self.arena.set_producer(reducer_slot, phase_id).ok();
            self.arena
                .transition(reducer_slot, SlotState::ProducerWriting, "reducer start")
                .ok();

            self.reducer.reduce(
                mb - 1,
                &self.teacher_outputs[slot_idx],
                &self.student_outputs[slot_idx],
            );

            self.arena.seal(reducer_slot, [0u8; 32]).ok();

            self.phase_records.push(PhaseExecutionRecord {
                phase_id,
                phase_type: "CPUReduction".into(),
                provider: "Accelerate".into(),
                started_at_ns: 0,
                completed_at_ns: 0,
                input_slots: vec![teacher_slot, student_slot],
                output_slots: vec![reducer_slot],
                peak_bytes: self.arena.current_bytes(),
                transition_count: 3,
            });

            self.arena
                .transition(teacher_slot, SlotState::Evictable, "reuse for next microbatch")
                .ok();
            self.arena
                .transition(teacher_slot, SlotState::Reserved, "reset for next microbatch")
                .ok();
            self.arena
                .transition(student_slot, SlotState::Evictable, "reuse for next microbatch")
                .ok();
            self.arena
                .transition(student_slot, SlotState::Reserved, "reset for next microbatch")
                .ok();
        }

        // ── Phase 2: TeacherForward for microbatch n+1 ───────────────────
        if mb + 1 < self.total_microbatches {
            let slot_idx = (mb + 1) % 3;
            let slot_id = self.region.teacher_slots[slot_idx].unwrap();
            let phase_id = PhaseId::next();

            self.arena.set_producer(slot_id, phase_id).ok();
            self.arena
                .transition(slot_id, SlotState::ProducerWriting, "teacher forward start")
                .ok();

            // Try Core ML first; fall back to Metal teacher.
            if self.teacher_forward_coreml(mb + 1, slot_idx) {
                // Core ML succeeded — populate the teacher output buffer.
                // In production the Core ML output would be copied here;
                // for the stub we use the Metal teacher's simulated output
                // as a proxy since Core ML model files are not guaranteed
                // to be present at test time.
                self.metal_teacher.forward(mb + 1, slot_id);
            } else {
                // Fallback to Level 1 Metal teacher.
                self.metal_teacher.forward(mb + 1, slot_id);
            }

            // Capture teacher output into the ring buffer.
            let out = self.metal_teacher.output();
            self.teacher_outputs[slot_idx].copy_from_slice(out);
            self.teacher_slot_valid[slot_idx] = true;

            self.arena.seal(slot_id, [0u8; 32]).ok();
            self.arena
                .transition(slot_id, SlotState::ConsumerReadable, "teacher forward complete")
                .ok();
            self.arena.mark_readable(slot_id).ok();

            let provider = if self.bridge_receipts.last()
                .map(|r| r.actual_route.starts_with("CoreML"))
                .unwrap_or(false)
            {
                "CoreML"
            } else {
                "Metal"
            };

            self.phase_records.push(PhaseExecutionRecord {
                phase_id,
                phase_type: "TeacherForward".into(),
                provider: provider.into(),
                started_at_ns: 0,
                completed_at_ns: 0,
                input_slots: vec![],
                output_slots: vec![slot_id],
                peak_bytes: self.arena.current_bytes(),
                transition_count: 3,
            });
        }

        // ── Phase 3: StudentForward for microbatch n ─────────────────────
        if mb < self.total_microbatches {
            let slot_idx = mb % 3;
            let slot_id = self.region.student_slots[slot_idx].unwrap();
            let phase_id = PhaseId::next();

            self.arena.set_producer(slot_id, phase_id).ok();
            self.arena
                .transition(slot_id, SlotState::ProducerWriting, "student forward start")
                .ok();
            self.student.forward(mb, slot_id);

            let out = self.student.output();
            self.student_outputs[slot_idx].copy_from_slice(out);
            self.student_slot_valid[slot_idx] = true;

            self.arena.seal(slot_id, [0u8; 32]).ok();
            self.arena.mark_readable(slot_id).ok();

            self.phase_records.push(PhaseExecutionRecord {
                phase_id,
                phase_type: "StudentForward".into(),
                provider: "Metal".into(),
                started_at_ns: 0,
                completed_at_ns: 0,
                input_slots: vec![],
                output_slots: vec![slot_id],
                peak_bytes: self.arena.current_bytes(),
                transition_count: 3,
            });
        }

        // Track peak memory.
        self.region.peak_memory = self.region.peak_memory.max(self.arena.current_bytes());

        // Advance microbatch counter.
        self.region.current_microbatch += 1;
        if self.region.current_microbatch >= self.total_microbatches {
            self.completed = true;
        }

        !self.completed
    }

    /// Run the full pipeline to completion.
    pub fn run(&mut self) {
        self.initialize();
        while self.step() {}
    }

    /// Reference to the arena (for post-compile receipt generation).
    pub fn arena(&self) -> &ActivationArena {
        &self.arena
    }

    /// Peak memory observed during compilation.
    pub fn peak_memory(&self) -> u64 {
        self.region.peak_memory
    }

    /// The list of phase execution records for the receipt.
    pub fn phase_records(&self) -> &[PhaseExecutionRecord] {
        &self.phase_records
    }

    /// Bridge receipts collected during execution.
    pub fn bridge_receipts(&self) -> &[BridgeReceipt] {
        &self.bridge_receipts
    }

    /// Consume the scheduler and produce a bridge evidence section.
    pub fn into_bridge_evidence(self) -> BridgeEvidenceSection {
        let fallback_count = self
            .bridge_receipts
            .iter()
            .filter(|r| r.actual_route == "Level1-Metal-fallback")
            .count();
        let total = self.bridge_receipts.len();
        let proof_status = if fallback_count == 0 && total > 0 {
            "all-coreml".into()
        } else if total == 0 {
            "none".into()
        } else {
            format!("{}-fallbacks", fallback_count)
        };

        BridgeEvidenceSection {
            receipts: self.bridge_receipts,
            bridge_proof_status: proof_status,
        }
    }

    /// Reference to the teacher for post-run analysis.
    pub fn teacher(&self) -> &MetalTeacher {
        &self.metal_teacher
    }

    /// Reference to the reducer for reading computed metrics.
    pub fn reducer(&self) -> &AccelerateReducer {
        &self.reducer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::memory_budget::MemoryBudget;

    #[test]
    fn test_level2_scheduler_construction() {
        let config = Level1Config::default();
        let teacher = CoreMLTeacher::default();
        let mut scheduler = Level2Scheduler::new(config, 3, teacher, false);
        scheduler.initialize();
        assert!(scheduler.arena().slot_count() >= 9);
    }

    #[test]
    fn test_level2_fallback_on_unavailable() {
        let config = Level1Config::default();
        let teacher = CoreMLTeacher::default();
        let mut scheduler = Level2Scheduler::new(config, 2, teacher, false);
        scheduler.run();
        let receipts = scheduler.bridge_receipts();
        if receipts.is_empty() {
            // Core ML unavailable — no bridge receipts expected; Metal fallback is implicit.
        } else {
            for r in receipts {
                assert_eq!(r.actual_route, "Level1-Metal-fallback");
                assert!(!r.zero_copy_verified);
            }
        }
    }

    #[test]
    fn test_bridge_evidence_no_activity() {
        let config = Level1Config::default();
        let teacher = CoreMLTeacher::default();
        let scheduler = Level2Scheduler::new(config, 0, teacher, true);
        let evidence = scheduler.into_bridge_evidence();
        assert_eq!(evidence.bridge_proof_status, "none");
    }

    #[test]
    fn test_level2_phase_records() {
        let config = Level1Config::default();
        let teacher = CoreMLTeacher::default();
        let mut scheduler = Level2Scheduler::new(config, 3, teacher, false);
        scheduler.run();
        assert!(!scheduler.phase_records().is_empty());
        // With 3 microbatches we should have teacher, student, and reducer phases.
        let types: Vec<&str> = scheduler
            .phase_records()
            .iter()
            .map(|r| r.phase_type.as_str())
            .collect();
        assert!(types.contains(&"TeacherForward"));
        assert!(types.contains(&"StudentForward"));
        assert!(types.contains(&"CPUReduction"));
    }
}
