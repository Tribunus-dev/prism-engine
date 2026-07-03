//! Level 1 scheduler — triple-buffered compile phase cadence.
//!
//! Maintains one active teacher microbatch (n+1), one active student candidate
//! microbatch (n), and one reducer task (n-1). Uses triple-buffered logical
//! activation slots and triple-buffered f32 output vectors so the reducer
//! always reads the correct microbatch data regardless of phase cadence.
//!
//! Safe phase cadence (each step):
//!   1. Reduce   for microbatch n-1 (reads saved outputs by buffer index)
//!   2. Teacher  for microbatch n+1 (writes to slot (n+1) % 3)
//!   3. Student  for microbatch n   (writes to slot n % 3)

use super::super::arena::{ActivationArena, SlotState, StorageRoute};
use super::super::memory_budget::MemoryBudget;
use super::super::phase_types::{
    ElementType, PhysicalLayout, ProviderKind, ResidencyClass, TensorDescriptor,
};
use super::super::receipt::PhaseExecutionRecord;
use super::super::phase_types::PhaseId;

use super::teacher::MetalTeacher;
use super::student::TernaryStudent;
use super::reducer::AccelerateReducer;

// ── Scheduler configuration ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Level1Config {
    /// Microbatch size (number of tokens per step).
    pub microbatch: usize,
    /// Hidden dimension of the model.
    pub hidden_dim: usize,
    /// Number of pages per row (in_dim / PAGE).
    pub pages_per_row: usize,
    /// Memory budget.
    pub budget: MemoryBudget,
}

impl Default for Level1Config {
    fn default() -> Self {
        Level1Config {
            microbatch: 4096,
            hidden_dim: 3840,
            pages_per_row: 2, // 1280 / 640
            budget: MemoryBudget::m1_16gb_default(),
        }
    }
}

// ── Compile region state ────────────────────────────────────────────────────

/// The state of one compile region being processed.
pub(crate) struct RegionState {
    /// Index of the current microbatch being processed.
    pub(crate) current_microbatch: usize,
    /// Teacher frontier slot id.
    pub(crate) teacher_frontier_slot: Option<u64>,
    /// Student frontier slot id.
    pub(crate) student_frontier_slot: Option<u64>,
    /// Slot ids for the triple-buffered activation pipeline.
    pub(crate) teacher_slots: [Option<u64>; 3],
    pub(crate) student_slots: [Option<u64>; 3],
    pub(crate) reducer_slots: [Option<u64>; 3],
    /// Peak memory observed during this region.
    pub(crate) peak_memory: u64,
}

impl RegionState {
    fn new() -> Self {
        RegionState {
            current_microbatch: 0,
            teacher_frontier_slot: None,
            student_frontier_slot: None,
            teacher_slots: [None, None, None],
            student_slots: [None, None, None],
            reducer_slots: [None, None, None],
            peak_memory: 0,
        }
    }
}

// ── Level 1 scheduler ───────────────────────────────────────────────────────

#[cfg(feature = "prism-backend")]
/// The Level 1 compile scheduler.
///
/// Implements the triple-buffered phase cadence: teacher forward for microbatch
/// n+1, student candidate forward for microbatch n, and CPU reduction for
/// microbatch n-1.  Each step advances the pipeline by one microbatch.
///
/// Three ring-buffered f32 output vectors (one per slot) for each of teacher
/// and student ensure the reducer always reads valid data for the target
/// microbatch, regardless of the phase ordering within a single step.
pub struct Level1Scheduler {
    pub(crate) config: Level1Config,
    pub(crate) arena: ActivationArena,
    pub(crate) budget: MemoryBudget,
    pub(crate) teacher: MetalTeacher,
    pub(crate) student: TernaryStudent,
    pub(crate) reducer: AccelerateReducer,
    pub(crate) region: RegionState,
    pub(crate) phase_records: Vec<PhaseExecutionRecord>,
    pub(crate) total_microbatches: usize,
    pub(crate) completed: bool,
    /// Triple-buffered teacher outputs: teacher_outputs[slot_idx] holds the
    /// forward result for the microbatch that wrote to that slot.
    pub(crate) teacher_outputs: [Vec<f32>; 3],
    /// Triple-buffered student outputs (same indexing as slots).
    pub(crate) student_outputs: [Vec<f32>; 3],
    /// Whether each buffer slot has been populated by at least one forward.
    pub(crate) teacher_slot_valid: [bool; 3],
    pub(crate) student_slot_valid: [bool; 3],
}

impl Level1Scheduler {
    /// Create a new Level 1 scheduler with the given configuration.
    pub fn new(config: Level1Config, total_microbatches: usize) -> Self {
        let hidden_dim = config.hidden_dim;
        let teacher = MetalTeacher::with_shape(hidden_dim, hidden_dim);
        let student = TernaryStudent::with_shape(hidden_dim, hidden_dim);
        let reducer = AccelerateReducer::with_hidden_dim(hidden_dim);

        let empty = vec![0.0f32; hidden_dim];
        Level1Scheduler {
            arena: ActivationArena::new(),
            budget: config.budget.clone(),
            config,
            teacher,
            student,
            reducer,
            region: RegionState::new(),
            phase_records: Vec::new(),
            total_microbatches,
            completed: false,
            teacher_outputs: [empty.clone(), empty.clone(), empty.clone()],
            student_outputs: [empty.clone(), empty.clone(), empty],
            teacher_slot_valid: [false, false, false],
            student_slot_valid: [false, false, false],
        }
    }

    /// Initialize the scheduler: allocate the triple-buffered slot pipeline.
    pub fn initialize(&mut self) {
        let mb = self.config.microbatch;
        let hd = self.config.hidden_dim;

        // Create a TensorDescriptor for a [microbatch x hidden_dim] activation.
        let act_tensor = |mutable: bool| -> TensorDescriptor {
            TensorDescriptor {
                logical_shape: vec![mb, hd],
                element_type: ElementType::F16,
                physical_layout: PhysicalLayout::DenseRowMajor,
                alignment: 16384,
                producer_phase: None,
                consumer_phases: Vec::new(),
                permitted_providers: vec![ProviderKind::Metal, ProviderKind::Accelerate],
                residency_class: ResidencyClass::Unified,
                max_bytes: (mb * hd * 2) as u64, // F16 = 2 bytes
                mutable,
                content_digest: None,
            }
        };

        // Allocate 3 teacher slots, 3 student slots, 3 reducer slots.
        let mut next_id = 1u64;
        for i in 0..3 {
            let teacher_slot = self.arena.reserve(next_id, act_tensor(true));
            next_id += 1;
            if let Some(slot) = self.arena.slot_mut(teacher_slot) {
                slot.storage_route = StorageRoute::MetalSharedBuffer;
            }
            let student_slot = self.arena.reserve(next_id, act_tensor(true));
            next_id += 1;
            if let Some(slot) = self.arena.slot_mut(student_slot) {
                slot.storage_route = StorageRoute::MetalSharedBuffer;
            }
            let reducer_slot = self.arena.reserve(next_id, act_tensor(true));
            next_id += 1;
            // Reducer stays CpuOwned (default).
            self.region.teacher_slots[i] = Some(teacher_slot);
            self.region.student_slots[i] = Some(student_slot);
            self.region.reducer_slots[i] = Some(reducer_slot);
        }

        // Track peak memory.
        self.region.peak_memory = self.arena.current_bytes();
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
        // Runs FIRST so the reducer reads the outputs that were written
        // to the ring buffer in the *previous* step (before they are
        // overwritten by this step's forward passes).
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

            // Read saved outputs from the triple-buffered vectors.
            // These were written by the forward passes in prior steps.
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

            // Release the used teacher and student slots.
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
            self.teacher.forward(mb + 1, slot_id);

            // Capture the teacher output into the ring buffer so the
            // reducer can read it when it processes this microbatch.
            let out = self.teacher.output();
            self.teacher_outputs[slot_idx].copy_from_slice(out);
            self.teacher_slot_valid[slot_idx] = true;

            self.arena.seal(slot_id, [0u8; 32]).ok();
            self.arena
                .transition(slot_id, SlotState::ConsumerReadable, "teacher forward complete")
                .ok();
            self.arena.mark_readable(slot_id).ok();

            self.phase_records.push(PhaseExecutionRecord {
                phase_id,
                phase_type: "TeacherForward".into(),
                provider: "Metal".into(),
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

            // Capture student output into the ring buffer.
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

    /// Reference to the teacher for post-run analysis.
    pub fn teacher(&self) -> &MetalTeacher {
        &self.teacher
    }

    /// Reference to the student for post-run analysis.
    pub fn student(&self) -> &TernaryStudent {
        &self.student
    }

    /// Reference to the reducer for reading computed metrics.
    pub fn reducer(&self) -> &AccelerateReducer {
        &self.reducer
    }
}
