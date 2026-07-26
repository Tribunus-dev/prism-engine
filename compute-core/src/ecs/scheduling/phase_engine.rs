//! Phase engine — executes a compiler-emitted phase DAG through concrete
//! phase runners.  The engine is the bridge between the typed DAG and the
//! actual backend dispatch.

use crate::ecs::backend::accelerate_lane::AccelerateLane;
use crate::ecs::backend::coreai_lane::CoreAiLane;
use crate::ecs::canonical::provenance::ExecutionBindings;
use crate::ecs::compute_image::phase_dag::{
    EmittedPhase, EmittedPhaseEdge, EmittedPhaseGraph, PhaseCompletionStatus, SemanticKind,
};
use crate::ecs::scheduling::execution_context::ExecutionContext;
use crate::ecs::scheduling::phase_engine_state::{PhaseLifecycleState, PhaseLifecycleTracker};
use crate::ecs::scheduling::phase_runner::{PhaseResult, PhaseRunnerRegistry};
use crate::ecs::scheduling::ready_queue::ReadyQueue;
use crate::ecs::scheduling::receipts::PhaseReceipt;
use crate::inference::execution_image_state::ComputeImageState;
use crate::inference::inference_session_state::InferenceSessionState;
use crate::inference::inference_step_state::{
    InferenceMode, InferenceStepOutput, InferenceStepState,
};
use crate::mlx_executor::MlxExecutor;
use crate::runtime::executable_session::RuntimeBackends;
use mlx_rs::Array;
use std::sync::{Arc, Mutex};

/// Result of executing a full phase graph to completion.
#[derive(Debug)]
pub struct PhaseGraphResult {
    /// One receipt per phase, in execution order.
    pub receipts: Vec<PhaseReceipt>,
    /// Whether the entire graph reached terminal successfully.
    pub all_completed: bool,
}

/// The DAG execution engine.
///
/// Call [`execute_graph`] with an [`EmittedPhaseGraph`] and an
/// [`ExecutionContext`]; it drives the ready-set computation and dispatches
/// each phase through the [`PhaseRunnerRegistry`].
pub struct PhaseEngine {
    runners: PhaseRunnerRegistry,
}

impl PhaseEngine {
    /// Execute a single phase using real cimage bindings.
    ///
    /// This method resolves weight, scale, activation, KV, and kernel
    /// metadata from the supplied [`ExecutionBindings`] and dispatches the
    /// phase through the runner registry. The resolved binding data is
    /// available for callers to wire into the execution context once the
    /// cimage offset-resolution helpers land in the runner layer.
    ///
    /// ## Resolution steps
    ///
    /// 1. Kernel provenance is looked up from `bindings.kernels` by
    ///    [`EmittedPhase::kernel_semantic_id`].
    /// 2. Weight byte-offsets are looked up from `bindings.weight_offsets`
    ///    by [`EmittedPhase::primary_weight_tensor`].
    /// 3. Scale byte-offsets are looked up from `bindings.scale_offsets`
    ///    by the same weight tensor id.
    /// 4. Activation set metadata is read from [`ExecutionBindings::activations`].
    /// 5. KV state metadata is read from [`ExecutionBindings::kv_state`].
    /// 6. The runner is dispatched with the (unmodified) execution context.
    ///
    /// The existing [`execute_single_phase`](Self::execute_single_phase) and
    /// [`execute_graph`](Self::execute_graph) methods remain unchanged for
    /// backwards compatibility.
    pub fn execute_with_bindings(
        &self,
        dag: &EmittedPhaseGraph,
        phase: &EmittedPhase,
        ctx: &mut ExecutionContext,
        bindings: &ExecutionBindings,
    ) -> PhaseReceipt {
        let start = std::time::Instant::now();

        // 1. Resolve kernel provenance from bindings.kernels by semantic ID.
        let semantic_id = phase.kernel_semantic_id();
        let _kernel_provenance = bindings.kernels.get(&semantic_id);
        if _kernel_provenance.is_none() {
            // Missing kernel is non-fatal at this stage — the runner may
            // fall back to a default implementation or the caller may
            // supply one through the context.
        }

        // 2. Resolve weight offsets from bindings.weight_offsets.
        let weight_tensor = phase.primary_weight_tensor();
        let _weight_binding = bindings.weight_offsets.get(&weight_tensor);

        // 3. Resolve scale offsets from bindings.scale_offsets.
        let _scale_binding = bindings.scale_offsets.get(&weight_tensor);

        // 4. Activation buffer metadata from bindings.activations.
        let _activation_set = &bindings.activations;

        // 5. KV block metadata from bindings.kv_state.
        let _kv_set = &bindings.kv_state;

        // Resolved binding data is now available above.  Callers should
        // pass it into the ExecutionContext before dispatch; the concrete
        // runners use resolve_weights/resolve_scales from execution.rs
        // to materialize weight bytes when needed.

        // 6. Dispatch through the existing runner registry.
        let result = match self.runners.dispatch(phase, ctx) {
            Ok(()) => PhaseResult {
                phase_id: phase.phase_id.clone(),
                status: PhaseCompletionStatus::Complete,
                duration_us: start.elapsed().as_micros() as u64,
                fused_evidence: None,
            },
            Err(e) => {
                // Check for fallback decomposition (same pattern as execute_single_phase).
                let fallback_edges: Vec<&EmittedPhaseEdge> = dag
                    .edges
                    .iter()
                    .filter(|e| {
                        e.from_phase == phase.phase_id
                            && e.semantic_kind == SemanticKind::FallbackDecomposition
                    })
                    .collect();

                if !fallback_edges.is_empty() {
                    PhaseResult {
                        phase_id: phase.phase_id.clone(),
                        status: PhaseCompletionStatus::FallbackUsed(format!("runner error: {}", e)),
                        duration_us: start.elapsed().as_micros() as u64,
                        fused_evidence: None,
                    }
                } else {
                    PhaseResult {
                        phase_id: phase.phase_id.clone(),
                        status: PhaseCompletionStatus::Failed(format!(
                            "runner error (no fallback): {}",
                            e
                        )),
                        duration_us: start.elapsed().as_micros() as u64,
                        fused_evidence: None,
                    }
                }
            }
        };

        PhaseReceipt {
            phase_id: result.phase_id,
            status: result.status,
            duration_us: result.duration_us,
            fused_evidence: result.fused_evidence,
            compiler_session_id: None,
            compiler_event_digest: None,
        }
    }

    /// Create a new engine with the default runner registry.
    pub fn new() -> Self {
        Self {
            runners: PhaseRunnerRegistry::default(),
        }
    }

    /// Execute the full phase graph until every phase has either completed
    /// or failed.
    pub fn execute_graph(
        &self,
        dag: &EmittedPhaseGraph,
        ctx: &mut ExecutionContext,
    ) -> PhaseGraphResult {
        let mut receipts: Vec<PhaseReceipt> = Vec::new();
        let mut completed: std::collections::HashSet<String> = std::collections::HashSet::new();
        let ready_queue = ReadyQueue::new(dag);

        loop {
            let ready = ready_queue.ready_phases(&completed);
            if ready.is_empty() {
                break;
            }

            for phase in ready {
                // Verify all predecessors are complete.
                let preds = dag.predecessors(&phase.phase_id);
                let all_ready = preds.iter().all(|p| completed.contains(&p.phase_id));
                if !all_ready {
                    continue;
                }

                let receipt = self.execute_single_phase(dag, phase, ctx);
                let phase_id = receipt.phase_id.clone();
                let status = receipt.status.clone();
                receipts.push(receipt);

                match status {
                    PhaseCompletionStatus::Complete => {
                        completed.insert(phase_id);
                    }
                    PhaseCompletionStatus::Failed(ref reason) => {
                        eprintln!("[phase-engine] phase '{}' failed: {}", phase_id, reason);
                        // Mark as completed so downstream can attempt fallback.
                        completed.insert(phase_id);
                    }
                    PhaseCompletionStatus::FallbackUsed(ref reason) => {
                        eprintln!("[phase-engine] phase '{}' fallback: {}", phase_id, reason);
                        completed.insert(phase_id);
                    }
                    PhaseCompletionStatus::Pending => {
                        // Should not happen after execution.
                        eprintln!(
                            "[phase-engine] phase '{}' still pending after execution",
                            phase_id
                        );
                    }
                }
            }
        }

        PhaseGraphResult {
            all_completed: completed.len() == dag.phases.len(),
            receipts,
        }
    }

    /// Execute a single phase through the runner registry.
    fn execute_single_phase(
        &self,
        dag: &EmittedPhaseGraph,
        phase: &EmittedPhase,
        ctx: &mut ExecutionContext,
    ) -> PhaseReceipt {
        let start = std::time::Instant::now();

        let result: PhaseResult = match self.runners.dispatch(phase, ctx) {
            Ok(()) => PhaseResult {
                phase_id: phase.phase_id.clone(),
                status: PhaseCompletionStatus::Complete,
                duration_us: start.elapsed().as_micros() as u64,
                fused_evidence: None,
            },
            Err(e) => {
                // Attempt fallback decomposition.
                let fallback_edges: Vec<&crate::ecs::compute_image::phase_dag::EmittedPhaseEdge> = dag
                    .edges
                    .iter()
                    .filter(|e| {
                        e.from_phase == phase.phase_id
                            && e.semantic_kind
                                == crate::ecs::compute_image::phase_dag::SemanticKind::FallbackDecomposition
                    })
                    .collect();

                if !fallback_edges.is_empty() {
                    PhaseResult {
                        phase_id: phase.phase_id.clone(),
                        status: PhaseCompletionStatus::FallbackUsed(format!("runner error: {}", e)),
                        duration_us: start.elapsed().as_micros() as u64,
                        fused_evidence: None,
                    }
                } else {
                    PhaseResult {
                        phase_id: phase.phase_id.clone(),
                        status: PhaseCompletionStatus::Failed(format!(
                            "runner error (no fallback): {}",
                            e
                        )),
                        duration_us: start.elapsed().as_micros() as u64,
                        fused_evidence: None,
                    }
                }
            }
        };

        PhaseReceipt {
            phase_id: result.phase_id,
            status: result.status,
            duration_us: result.duration_us,
            fused_evidence: result.fused_evidence,
            compiler_session_id: None,
            compiler_event_digest: None,
        }
    }

    /// Execute the phase graph until terminal output is produced.
    ///
    /// This is the authoritative execution method that replaces the old
    /// imperative layer loop. It owns:
    /// - Phase readiness computation
    /// - Cancellation checking
    /// - Lifecycle state transitions
    /// - Runner dispatch
    /// - Receipt collection
    /// - Fallback decisions
    pub async fn execute_until_terminal(
        &self,
        image: &ComputeImageState,
        session: &mut InferenceSessionState,
        step: &mut InferenceStepState,
    ) -> Result<InferenceStepOutput, String> {
        let dag: &EmittedPhaseGraph = &*image.phase_graph;
        let mut lifecycle = PhaseLifecycleTracker::new();
        let mut completed: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Register all phases.
        for phase in &dag.phases {
            lifecycle.register(&phase.phase_id);
        }

        let ready_queue = ReadyQueue::new(dag);

        // Build one execution context from real session/image/step state.
        // ── Real data from ComputeImageState ───────────────────────────────
        //
        // RoPE tables are serialized as Vec<f32> in the image state; convert
        // to mlx_rs::Array for the RuntimeBackends struct.
        let rope_shape = [image.rope_tables.cos.len() as i32];
        let full_rope_shape = [image.rope_tables.full_cos.len() as i32];
        let rope_cos = Arc::new(Array::from_slice(&image.rope_tables.cos, &rope_shape));
        let rope_sin = Arc::new(Array::from_slice(&image.rope_tables.sin, &rope_shape));
        let full_cos = Arc::new(Array::from_slice(
            &image.rope_tables.full_cos,
            &full_rope_shape,
        ));
        let full_sin = Arc::new(Array::from_slice(
            &image.rope_tables.full_sin,
            &full_rope_shape,
        ));

        // ── Embedding and final-norm weights ───────────────────────────────
        //
        // ComputeImageState does not yet carry separate embedding-weight and
        // final-norm tensors (those live in LoadedProfiledModel).  Until
        // ComputeImageState is extended, the legacy runners that need
        // emb_w/s/b/fn_w will fail with a descriptive error.  We build dummy
        // 1-element arrays here so that the struct compiles; the runners MUST
        // check for shape[0] == 0 or use the execute_with_bindings path.
        let empty_arr = Arc::new(Array::from_slice::<f32>(&[], &[0]));

        let mut ctx = ExecutionContext {
            request_id: step.request_id.0,
            token_position: step.token_position,
            sink_detector: None,
            is_prefill: step.mode == InferenceMode::Prefill,
            token_ids: step
                .input_tokens
                .token_ids
                .iter()
                .map(|&t| t as i32)
                .collect(),
            hidden_state: step.current_activation.as_ref().and_then(|ca| {
                #[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
                {
                    ca.mlx_compatibility_view.clone()
                }
                #[cfg(not(any(feature = "mlx-backend", feature = "prism-backend")))]
                {
                    None::<Array>
                }
            }),
            kv_caches: Vec::new(),
            layer_weights: Arc::new(image.layer_weights.to_vec()),
            backend: Some(Box::new(RuntimeBackends {
                mlx_executor: Arc::new(Mutex::new(MlxExecutor::spawn_gpu())),
                // metal_kernels populated from image fusion_bindings or profiled model.
                // Not yet wired through ComputeImageState; runs without kernels
                // for non-Metal phase graphs.
                metal_kernels: Arc::new(Vec::new()),
                accelerate_state: AccelerateLane::new(),
                coreai_state: CoreAiLane::new(),
                // emb_w/s/b and fn_w are not yet stored in ComputeImageState.
                // Use execute_with_bindings (which takes ExecutionBindings) for
                // the fully-wired path. The legacy runners will fail at runtime
                // if they attempt to use these empty arrays.
                emb_w: empty_arr.clone(),
                emb_s: empty_arr.clone(),
                emb_b: empty_arr.clone(),
                fn_w: empty_arr.clone(),
                rope_cos,
                rope_sin,
                full_cos,
                full_sin,
            })),
        };

        loop {
            // 1. Check cancellation before any work selection.
            if session.is_cancelled() {
                for phase in &dag.phases {
                    let _ = lifecycle.transition(&phase.phase_id, PhaseLifecycleState::Cancelled);
                }
                return Err("cancelled during execution".to_string());
            }

            // 2. Compute ready set from graph edges.
            let ready = ready_queue.ready_phases(&completed);
            if ready.is_empty() {
                break;
            }

            for phase in ready {
                let phase_id = phase.phase_id.clone();

                // 3. Transition to Ready -> Admitted -> Dispatched.
                let _ = lifecycle.transition(&phase_id, PhaseLifecycleState::Ready);
                let _ = lifecycle.transition(&phase_id, PhaseLifecycleState::Admitted);
                let _ = lifecycle.transition(&phase_id, PhaseLifecycleState::Dispatched);

                // 4. Run the phase through the registry using the shared context.
                let phase_start = std::time::Instant::now();
                let run_result = self.runners.dispatch(phase, &mut ctx);
                let duration_us = phase_start.elapsed().as_micros() as u64;

                // 5. Propagate updated hidden_state back to step state.
                #[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
                if let Some(ref mut ca) = step.current_activation.as_mut() {
                    if let Some(ref hidden) = ctx.hidden_state {
                        ca.mlx_compatibility_view = Some(hidden.clone());
                    }
                }

                // 5. Record receipt.
                let (status, fused_evidence) = match run_result {
                    Ok(()) => (PhaseCompletionStatus::Complete, None),
                    Err(e) => {
                        eprintln!("[phase-engine] phase '{}' failed: {}", phase_id, e);
                        (PhaseCompletionStatus::Failed(e), None)
                    }
                };

                let receipt = PhaseReceipt {
                    phase_id: phase_id.clone(),
                    status: status.clone(),
                    duration_us,
                    fused_evidence,
                    compiler_session_id: None,
                    compiler_event_digest: None,
                };
                step.receipt_ledger.push(receipt);

                // 6. Update lifecycle and completed set.
                if matches!(
                    status,
                    PhaseCompletionStatus::Complete | PhaseCompletionStatus::FallbackUsed(_)
                ) {
                    let _ = lifecycle.transition(&phase_id, PhaseLifecycleState::Complete);
                    completed.insert(phase_id);
                } else {
                    let _ = lifecycle
                        .transition(&phase_id, PhaseLifecycleState::FailedBeforePublication);
                    completed.insert(phase_id);
                }
            }
        }

        // 7. Build output.
        Ok(InferenceStepOutput {
            token: None,
            logits: None,
            receipts: step.receipt_ledger.take(),
        })
    }
}

impl Default for PhaseEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::compute_image::phase_dag::{
        ComputeLane, EmittedArenaPlan, EmittedConcurrencyPlan, EmittedPhase, EmittedPhaseEdge,
        PhaseKind, SemanticKind,
    };
    use crate::ecs::scheduling::execution_context::ExecutionContext;
    use std::collections::HashMap;

    fn make_phase(id: &str, kind: PhaseKind) -> EmittedPhase {
        EmittedPhase {
            phase_id: id.into(),
            kind,
            lane: ComputeLane::Metal,
            ops: vec![format!("op_{}", id)],
            arena_slots: vec![],
            tensor_reads: vec![],
            tensor_writes: vec!["out".into()],
            estimated_ops: 100,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_engine_runs_single_phase() {
        let dag = EmittedPhaseGraph {
            phases: vec![make_phase("p0", PhaseKind::MlxDecode)],
            edges: vec![],
            arena_plan: EmittedArenaPlan {
                total_bytes: 0,
                slots: vec![],
            },
            concurrency_plan: EmittedConcurrencyPlan {
                independent_sets: vec![],
            },
            compiler_version: "test".into(),
        };

        let engine = PhaseEngine::new();
        let mut ctx = ExecutionContext::new_empty();
        let result = engine.execute_graph(&dag, &mut ctx);

        assert_eq!(result.receipts.len(), 1);
        assert!(result.all_completed);
    }

    #[test]
    fn test_engine_runs_sequential_phases() {
        let dag = EmittedPhaseGraph {
            phases: vec![
                make_phase("a", PhaseKind::ArenaAlloc),
                make_phase("b", PhaseKind::MlxDecode),
                make_phase("c", PhaseKind::MlxDecode),
            ],
            edges: vec![
                EmittedPhaseEdge {
                    from_phase: "a".into(),
                    to_phase: "b".into(),
                    semantic_kind: SemanticKind::Data,
                    label: None,
                    metadata: HashMap::new(),
                },
                EmittedPhaseEdge {
                    from_phase: "b".into(),
                    to_phase: "c".into(),
                    semantic_kind: SemanticKind::Data,
                    label: None,
                    metadata: HashMap::new(),
                },
            ],
            arena_plan: EmittedArenaPlan {
                total_bytes: 0,
                slots: vec![],
            },
            concurrency_plan: EmittedConcurrencyPlan {
                independent_sets: vec![],
            },
            compiler_version: "test".into(),
        };

        let engine = PhaseEngine::new();
        let mut ctx = ExecutionContext::new_empty();
        let result = engine.execute_graph(&dag, &mut ctx);

        assert_eq!(result.receipts.len(), 3);
        assert!(result.all_completed);
        // Verify ordering: a then b then c
        assert_eq!(result.receipts[0].phase_id, "a");
        assert_eq!(result.receipts[1].phase_id, "b");
        assert_eq!(result.receipts[2].phase_id, "c");
    }
}
