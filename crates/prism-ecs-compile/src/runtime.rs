//! Unified batch and realtime runtime.
//!
//! Loads a `.cimage` compilation artifact (manifest, tensors, kernels) and
//! dispatches execution in either batch (multi-token) or autoregressive
//! (prefill / decode) mode.
//!
//! The runtime owns CImage loading, kernel dispatch contracts, execution-plan
//! validation, and batch/realtime state. Backend-specific execution is
//! selected from the loaded kernel descriptors and AOT plan.
//!
//! # Unified Execution
//!
//! Both batch and realtime modes load the **same** `.cimage` file. The
//! [`CImageManifest`] carries every tensor, kernel, and execution plan
//! needed for either path — no separate compilation target is required.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use prism_amd_npu_runtime::{XdnaArtifact, XdnaExecutionPhase, XdnaRuntime};
use prism_ecs_kernel::{
    BackendKind, CpuBackend, KernelArtifact, KernelBackend, KernelCompileRequest,
    KernelDispatchRequest, KernelManifest, KernelPayload, KernelVariant,
};
use prism_ecs_quantization::ternarization::promotion::NativeTernaryPromotionEvidence;

use crate::cimage::{CImageManifest, CImageReader};
use crate::uop::UOpCompiledProgram;
use prism_spatial_ir::execution::HeterogeneousExecutionReceipt;
use prism_spatial_ir::execution_plan::FusedScheduleStep;
use prism_spatial_ir::execution_plan::{ExecutionPlan, InferencePhase, PlanBackend};
use prism_spatial_ir::target::KernelManifest as SpatialKernelManifest;
use prism_spatial_ir::{
    AotScheduler, BindingResolver, BufferStorage, CapturePlan, HeterogeneousExecutor,
    ResolvedBuffer, RouteDispatch, RoutedExecutor, WorkloadScenario,
};

#[cfg(all(feature = "ane", target_os = "macos"))]
fn copy_int8_to_arena(arena: &prism_ane::Arena, values: &[i8]) -> Result<(), String> {
    if values.len() * std::mem::size_of::<i8>() > arena.info.byte_size as usize {
        return Err("int8 input exceeds IOSurface arena".into());
    }
    arena.lock()?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr() as *const u8,
            arena.info.base_address as *mut u8,
            values.len(),
        );
    }
    arena.unlock()
}

#[cfg(all(feature = "ane", target_os = "macos"))]
fn read_int32_from_arena(arena: &prism_ane::Arena, len: usize) -> Result<Vec<i32>, String> {
    if len * std::mem::size_of::<i32>() > arena.info.byte_size as usize {
        return Err("int32 output exceeds IOSurface arena".into());
    }
    arena.lock()?;
    let values =
        unsafe { std::slice::from_raw_parts(arena.info.base_address as *const i32, len).to_vec() };
    arena.unlock()?;
    Ok(values)
}

// ---------------------------------------------------------------------------
// ExecutionMode
// ---------------------------------------------------------------------------

/// Execution mode for the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Batch mode — process multiple tokens simultaneously (GEMM-heavy).
    Batch,
    /// Autoregressive prefill — process prompt tokens in one forward pass.
    RealtimePrefill,
    /// Autoregressive decode — generate one token at a time with KV cache
    /// (GEMV-heavy).
    RealtimeDecode,
}

// ---------------------------------------------------------------------------
// RuntimeModel
// ---------------------------------------------------------------------------

/// A manifest-loaded model ready for execution.
///
/// Holds the parsed [`CImageManifest`], eagerly-loaded tensor and kernel
/// payloads, and a lazy offset map for on-demand tensor access. All of the
/// payload data is loaded into memory at construction time so dispatch paths
/// never touch the file system during inference.
#[derive(Debug)]
pub struct RuntimeModel {
    /// Path to the `.cimage` file.
    pub cimage_path: PathBuf,
    /// Parsed manifest metadata.
    pub manifest: CImageManifest,
    /// Loaded tensor payloads indexed by tensor name.
    pub tensors: HashMap<String, Vec<u8>>,
    /// Tensor records carrying shape and representation metadata.
    pub tensor_records: HashMap<String, crate::cimage::TensorRecord>,
    /// Per-group scale payloads linked from native ternary tensor records.
    /// These remain packed and are exposed separately so native kernels can
    /// bind codes and scales without reconstructing FP16/FP32 weights.
    pub tensor_scales: HashMap<String, Vec<u8>>,
    /// Loaded kernel payloads indexed by kernel name.
    pub kernels: HashMap<String, Vec<u8>>,
    /// Typed descriptors paired with kernel payloads in the CImage header.
    pub kernel_descriptors: HashMap<String, prism_ecs_kernel::KernelDescriptor>,
    /// Optional tinygrad-inspired UOp capture embedded by the compiler.
    ///
    /// This is validated while loading so callers can safely use it as the
    /// executable graph contract rather than treating the JSON envelope as
    /// untrusted metadata.
    pub uop_capture: Option<CapturePlan>,
    /// Validated executable UOp program retained for production dispatch.
    pub uop_program: Option<UOpCompiledProgram>,
    /// Additional executable UOp programs indexed by published strategy ID.
    pub uop_strategy_programs: HashMap<String, UOpCompiledProgram>,
    /// Sealed workload measurements and validated strategy choices.
    pub uop_workload_evidence: Vec<crate::cimage::UOpWorkloadEvidence>,
    /// Embedded stateless int8 ANE programs and their input/output contract.
    pub ane_programs: HashMap<String, (crate::cimage::AneProgramRecord, Vec<u8>)>,
    /// Validated native XDNA artifacts embedded in the CImage, indexed by
    /// artifact name. These are ready for handoff to the AMD NPU runtime.
    pub xdna_artifacts: HashMap<String, prism_amd_npu_runtime::XdnaArtifact>,
    /// Compiler-selected progressive KV compression policy, retained as its
    /// canonical serialized manifest value for serving/runtime coordination.
    pub kv_compression_policy: Option<String>,
    /// Namespaced specialised-model registry embedded in the CImage header.
    pub model_manifest: Option<crate::model_manifest::MultiModelManifest>,
    /// Backend promotion evidence carried by native ternary CImages.
    pub native_ternary_promotion: Option<NativeTernaryPromotionEvidence>,
    /// Joint ANE/Metal tiling evidence retained by the CImage.
    pub joint_tiling_evidence: Option<crate::search::JointTilingEvidence>,
    /// AOT heterogeneous schedule and residency contract emitted by the
    /// compiler.
    pub execution_plan: Option<ExecutionPlan>,
    /// Realtime schedule paired with the batch schedule in a KernelManifest.
    /// Older artifacts leave this unset and use `execution_plan` for their
    /// single schedule.
    pub realtime_execution_plan: Option<ExecutionPlan>,
    /// Lazy tensor offset map: tensor name → (file offset, byte size).
    /// Populated during parsing so individual tensors can be loaded
    /// on demand without re-reading the manifest region.
    pub tensor_offsets: HashMap<String, (u64, u64)>,
    /// Read-only mapping of the CImage file for zero-copy backend views.
    pub mapped_cimage: Option<memmap2::Mmap>,
}

/// Header-only view used to admit production-scale artifacts without copying
/// tensor payloads into the heap.  This is intentionally separate from
/// [`RuntimeModel`], whose legacy execution fields own decoded payloads.
#[derive(Debug, Clone)]
pub struct CImageInspection {
    pub path: PathBuf,
    pub file_bytes: u64,
    pub tensor_bytes: u64,
    pub kernel_bytes: u64,
    pub ane_program_bytes: u64,
    pub xdna_artifact_bytes: u64,
    pub tensor_count: usize,
    pub kernel_count: usize,
    pub ane_program_count: usize,
    pub xdna_artifact_count: usize,
    pub has_native_xdna: bool,
    pub has_batch_plan: bool,
    pub has_realtime_plan: bool,
    pub model_manifest: Option<crate::model_manifest::MultiModelManifest>,
    /// Promotion receipt proving the native ternary artifact passed its
    /// required backend and replay gates.
    pub native_ternary_promotion: Option<NativeTernaryPromotionEvidence>,
    pub joint_tiling_evidence: Option<crate::search::JointTilingEvidence>,
    pub kv_compression_policy: Option<String>,
}

/// Resolves AOT tensor bindings against the loaded CImage tensor table.
pub struct CImageBindingResolver<'a> {
    pub model: &'a RuntimeModel,
    pub runtime_outputs: HashMap<String, ResolvedBuffer>,
}

/// Concrete ANE entry points used by the runtime route table. The ANE
/// implementation owns Core ML model loading and IOSurface arena binding;
/// the scheduler supplies the already-resolved tensor contract.
pub trait AneRouteBackend {
    fn dispatch_planar(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String>;
    fn dispatch_matrix(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String>;
}

/// ANE route implementation backed by the runtime's embedded stateless Core
/// ML programs. The produced output is retained by the caller's output
/// binding layer; this adapter owns only the device invocation.
pub struct EmbeddedAneRouteBackend<'a> {
    pub runtime: &'a UnifiedRuntime,
    pub outputs: HashMap<String, Vec<i8>>,
}

impl EmbeddedAneRouteBackend<'_> {
    #[cfg(all(feature = "ane", target_os = "macos"))]
    fn dispatch_int8(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        let program = self
            .runtime
            .model
            .ane_program_for_step(step)
            .ok_or_else(|| format!("no embedded ANE program matches step {}", step.step_id))?
            .0;
        let activation = inputs
            .iter()
            .find(|buffer| buffer.name == program.activation_input)
            .ok_or_else(|| "ANE activation binding is unresolved".to_string())?;
        let weights = inputs
            .iter()
            .find(|buffer| buffer.name == program.weights_input)
            .ok_or_else(|| "ANE weight binding is unresolved".to_string())?;
        let activation_owned;
        let activation_bytes = if let Some(payload) = activation.payload.as_deref() {
            payload
        } else {
            activation_owned = self
                .runtime
                .model
                .tensors
                .get(&activation.name)
                .cloned()
                .ok_or_else(|| {
                    format!("ANE activation payload '{}' is missing", activation.name)
                })?;
            &activation_owned
        };
        let weights_owned;
        let weight_bytes = if let Some(payload) = weights.payload.as_deref() {
            payload
        } else {
            weights_owned = self
                .runtime
                .model
                .tensors
                .get(&weights.name)
                .cloned()
                .ok_or_else(|| format!("ANE weight payload '{}' is missing", weights.name))?;
            &weights_owned
        };
        if activation_bytes.len() % std::mem::size_of::<i8>() != 0
            || weight_bytes.len() % std::mem::size_of::<i8>() != 0
        {
            return Err("ANE int8 binding payload is not byte aligned".into());
        }
        let activation_shape = shape_2d(activation)?;
        let weight_shape = shape_2d(weights)?;
        let activation_values = activation_bytes
            .iter()
            .map(|&value| value as i8)
            .collect::<Vec<_>>();
        let weight_values = weight_bytes
            .iter()
            .map(|&value| value as i8)
            .collect::<Vec<_>>();
        let output = self
            .runtime
            .dispatch_ane_int8(
                &program.name,
                &activation_values,
                activation_shape,
                &weight_values,
                weight_shape,
            )
            .map_err(|error| error.to_string())?;
        if let Some(binding) = step.output_tensors.first() {
            if let Some(buffer) = outputs.first_mut() {
                buffer.byte_length = output.len();
                buffer.payload = Some(output.iter().map(|&value| value as u8).collect());
            }
            self.outputs.insert(binding.name.clone(), output);
        }
        Ok(())
    }

    #[cfg(all(feature = "ane", target_os = "macos"))]
    fn dispatch_fp16(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        let program = self
            .runtime
            .model
            .ane_program_for_step(step)
            .ok_or_else(|| {
                format!(
                    "no embedded ANE planar program matches step {}",
                    step.step_id
                )
            })?
            .0;
        let activation = inputs
            .iter()
            .find(|buffer| buffer.name == program.activation_input)
            .ok_or_else(|| "ANE planar activation binding is unresolved".to_string())?;
        let bias = inputs
            .iter()
            .find(|buffer| buffer.name == program.weights_input)
            .ok_or_else(|| "ANE planar bias binding is unresolved".to_string())?;
        let activation_owned;
        let activation_bytes = if let Some(payload) = activation.payload.as_deref() {
            payload
        } else {
            activation_owned = self
                .runtime
                .model
                .tensors
                .get(&activation.name)
                .cloned()
                .ok_or_else(|| format!("ANE planar activation '{}' is missing", activation.name))?;
            &activation_owned
        };
        let bias_owned;
        let bias_bytes = if let Some(payload) = bias.payload.as_deref() {
            payload
        } else {
            bias_owned = self
                .runtime
                .model
                .tensors
                .get(&bias.name)
                .cloned()
                .ok_or_else(|| format!("ANE planar bias '{}' is missing", bias.name))?;
            &bias_owned
        };
        let result = self
            .runtime
            .dispatch_ane_int8_planar(
                &program.name,
                activation_bytes,
                shape_2d(activation)?,
                bias_bytes,
                shape_2d(bias)?,
            )
            .map_err(|error| error.to_string())?;
        if let Some(binding) = outputs.first_mut() {
            binding.byte_length = result.len();
            binding.payload = Some(result.clone());
        }
        if let Some(binding) = step.output_tensors.first() {
            self.outputs.insert(
                binding.name.clone(),
                result.iter().map(|&v| v as i8).collect(),
            );
        }
        Ok(())
    }

    #[cfg(not(all(feature = "ane", target_os = "macos")))]
    fn dispatch_int8(
        &mut self,
        _step: &FusedScheduleStep,
        _inputs: &[ResolvedBuffer],
        _outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Err("ANE route is unavailable on this target or feature set".into())
    }
}

#[cfg(all(feature = "ane", target_os = "macos"))]
fn shape_2d(buffer: &ResolvedBuffer) -> Result<(u32, u32), String> {
    let dims = buffer.shape.as_slice();
    if dims.len() != 2 || dims.iter().any(|&dim| dim == 0 || dim > u32::MAX as u64) {
        return Err(format!(
            "ANE binding '{}' requires a non-zero 2D shape",
            buffer.name
        ));
    }
    Ok((dims[0] as u32, dims[1] as u32))
}

impl AneRouteBackend for EmbeddedAneRouteBackend<'_> {
    fn dispatch_planar(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        _outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        #[cfg(all(feature = "ane", target_os = "macos"))]
        {
            self.dispatch_fp16(step, inputs, _outputs)
        }
        #[cfg(not(all(feature = "ane", target_os = "macos")))]
        {
            self.dispatch_int8(step, inputs, _outputs)
        }
    }

    fn dispatch_matrix(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        _outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        self.dispatch_int8(step, inputs, _outputs)
    }
}

/// Runtime composition of the concrete CPU/Accelerate/Metal kernel backends
/// and the ANE IOSurface backend. This is the production implementation of
/// [`RouteDispatch`] used by [`UnifiedRuntime::replay_aot_routed`].
pub struct KernelRouteDispatcher<'a> {
    pub model: &'a RuntimeModel,
    pub ane: &'a mut dyn AneRouteBackend,
    pub accelerate: &'a dyn KernelBackend,
    pub metal: &'a dyn KernelBackend,
    pub cpu: &'a dyn KernelBackend,
    pub xdna: Option<&'a mut dyn XdnaRouteBackend>,
}

pub trait XdnaRouteBackend {
    fn dispatch_xdna(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String>;
}

impl KernelRouteDispatcher<'_> {
    fn dispatch_kernel(
        &self,
        backend: &dyn KernelBackend,
        expected_backend: BackendKind,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        let names = kernel_names_for_backend(&self.model.kernel_descriptors, expected_backend);
        let name = names
            .get(step.step_id)
            .ok_or_else(|| format!("no compiled kernel for AOT step {}", step.step_id))?;
        let artifact = self
            .model
            .kernel_artifact(name)
            .map_err(|error| error.to_string())?;
        let input_payloads = inputs
            .iter()
            .map(|buffer| {
                buffer
                    .payload
                    .clone()
                    .or_else(|| self.model.tensors.get(&buffer.name).cloned())
                    .ok_or_else(|| format!("no payload for runtime binding '{}'", buffer.name))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bindings = artifact
            .manifest
            .kernels
            .first()
            .map(|descriptor| descriptor.binding_signature.clone())
            .unwrap_or_default();
        let result = backend
            .dispatch(&KernelDispatchRequest {
                artifact,
                inputs: input_payloads,
                bindings,
            })
            .map_err(|error| error.to_string())?;
        if result.outputs.len() != outputs.len() {
            return Err(format!(
                "backend returned {} outputs for {} scheduled bindings",
                result.outputs.len(),
                outputs.len()
            ));
        }
        for (binding, payload) in outputs.iter_mut().zip(result.outputs.iter()) {
            if binding.byte_length != 0 && binding.byte_length != payload.len() {
                return Err(format!(
                    "backend output for '{}' has {} bytes; binding requires {}",
                    binding.name,
                    payload.len(),
                    binding.byte_length
                ));
            }
            binding.byte_length = payload.len();
            binding.payload = Some(payload.clone());
        }
        Ok(())
    }
}

fn kernel_names_for_backend(
    descriptors: &HashMap<String, prism_ecs_kernel::KernelDescriptor>,
    backend: BackendKind,
) -> Vec<&String> {
    let mut names: Vec<&String> = descriptors
        .iter()
        .filter_map(|(name, descriptor)| (descriptor.backend == backend).then_some(name))
        .collect();
    names.sort();
    names
}

impl RouteDispatch for KernelRouteDispatcher<'_> {
    fn ensure_residency(&mut self, window_id: usize) -> Result<(), String> {
        let plan = self
            .model
            .execution_plan
            .as_ref()
            .ok_or_else(|| "cannot ensure residency without an execution plan".to_string())?;
        let window = plan.residency_windows.get(window_id).ok_or_else(|| {
            format!("residency window {window_id} is not present in the AOT plan")
        })?;
        let cimage_bytes = self
            .model
            .mapped_cimage
            .as_ref()
            .map(|mapped| mapped.len() as u64)
            .or_else(|| {
                std::fs::metadata(&self.model.cimage_path)
                    .ok()
                    .map(|m| m.len())
            })
            .ok_or_else(|| "cannot determine CImage size for residency validation".to_string())?;
        if window.model_bytes == 0 || window.model_bytes > cimage_bytes {
            return Err(format!(
                "residency window {window_id} requires {} bytes, CImage has {}",
                window.model_bytes, cimage_bytes
            ));
        }
        Ok(())
    }

    fn dispatch_ane_planar(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        self.ane.dispatch_planar(step, inputs, outputs)
    }

    fn dispatch_ane_matrix(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        self.ane.dispatch_matrix(step, inputs, outputs)
    }

    fn dispatch_accelerate(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        _outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        self.dispatch_kernel(self.accelerate, BackendKind::CPU, step, inputs, _outputs)
    }

    fn dispatch_metal(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        _outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        self.dispatch_kernel(self.metal, BackendKind::Metal, step, inputs, _outputs)
    }

    fn dispatch_cpu(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        _outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        self.dispatch_kernel(self.cpu, BackendKind::CPU, step, inputs, _outputs)
    }

    fn dispatch_xdna(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        self.xdna
            .as_deref_mut()
            .ok_or_else(|| "XDNA route is not configured in KernelRouteDispatcher".to_string())?
            .dispatch_xdna(step, inputs, outputs)
    }

    fn synchronize(&mut self, _step: &FusedScheduleStep) -> Result<(), String> {
        Ok(())
    }
}

/// Native XDNA route for a loaded CImage. This keeps XDNA execution inside
/// the same dependency-aware scheduler as CPU, GPU, and ANE work while
/// leaving device submission to the Prism-owned `XdnaDevice` contract.
pub struct CImageXdnaRouteDispatcher<'a, D> {
    pub model: &'a RuntimeModel,
    pub runtime: XdnaRuntime,
    pub device: D,
    /// Phase used when the scheduler dispatches this XDNA island. Callers
    /// switch this before replaying a prefill or decode schedule.
    pub phase: XdnaExecutionPhase,
}

impl<'a, D> CImageXdnaRouteDispatcher<'a, D> {
    pub fn new(model: &'a RuntimeModel, device: D) -> Result<Self, String> {
        if model.xdna_artifacts.is_empty() {
            return Err("CImage has no native XDNA artifacts".into());
        }
        Ok(Self {
            model,
            runtime: XdnaRuntime::new(),
            device,
            phase: XdnaExecutionPhase::Decode,
        })
    }

    pub fn set_phase(&mut self, phase: XdnaExecutionPhase) {
        self.phase = phase;
    }

    fn unsupported(route: &str) -> Result<(), String> {
        Err(format!("XDNA route cannot dispatch {route} work"))
    }

    fn payloads_for_inputs(&self, inputs: &[ResolvedBuffer]) -> HashMap<String, Vec<u8>> {
        inputs
            .iter()
            .filter_map(|input| {
                input
                    .payload
                    .clone()
                    .or_else(|| self.model.tensors.get(&input.name).cloned())
                    .map(|payload| (input.name.clone(), payload))
            })
            .collect()
    }
}

impl<D: prism_amd_npu_runtime::XdnaCommandSubmitter> RouteDispatch
    for CImageXdnaRouteDispatcher<'_, D>
{
    fn ensure_residency(&mut self, _window_id: usize) -> Result<(), String> {
        Ok(())
    }

    fn dispatch_ane_planar(
        &mut self,
        _: &FusedScheduleStep,
        _: &[ResolvedBuffer],
        _: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Self::unsupported("ANE planar")
    }

    fn dispatch_ane_matrix(
        &mut self,
        _: &FusedScheduleStep,
        _: &[ResolvedBuffer],
        _: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Self::unsupported("ANE matrix")
    }

    fn dispatch_accelerate(
        &mut self,
        _: &FusedScheduleStep,
        _: &[ResolvedBuffer],
        _: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Self::unsupported("Accelerate")
    }

    fn dispatch_metal(
        &mut self,
        _: &FusedScheduleStep,
        _: &[ResolvedBuffer],
        _: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Self::unsupported("Metal")
    }

    fn dispatch_cpu(
        &mut self,
        _: &FusedScheduleStep,
        _: &[ResolvedBuffer],
        _: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Self::unsupported("CPU")
    }

    fn dispatch_xdna(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        let artifact_name = step.model_id.as_deref().unwrap_or("main");
        let artifact: &XdnaArtifact = self
            .model
            .xdna_artifact(artifact_name)
            .or_else(|| self.model.xdna_artifacts.get("main"))
            .ok_or_else(|| format!("no XDNA artifact for route {artifact_name}"))?;
        let mut payloads = self.payloads_for_inputs(inputs);
        // The native matmul lowering uses stable operand buffers A/B/C while
        // scheduler bindings use source-model tensor names. Bind by operand
        // position as the canonical fallback, then retain named bindings for
        // artifacts that expose model-native buffer identifiers.
        for (buffer_id, input) in ["A", "B", "C"].into_iter().zip(inputs.iter()) {
            if let Some(payload) = input.payload.as_ref() {
                payloads
                    .entry(buffer_id.into())
                    .or_insert_with(|| payload.clone());
            }
        }
        let command = artifact
            .command_buffer()
            .map_err(|error| format!("build XDNA command buffer: {error}"))?;
        self.runtime.submit_bound_artifact_phase_with_payloads(
            artifact,
            &command,
            self.phase,
            &payloads,
            &mut self.device,
        )?;
        if let Some(payload) =
            self.runtime
                .download_buffer(&artifact.program, "C", &mut self.device)?
        {
            if let Some(output) = outputs.first_mut() {
                output.byte_length = payload.len();
                output.payload = Some(payload);
                output.storage = BufferStorage::RuntimeOwned;
                output.zero_copy = false;
                output.file_offset = None;
            }
        }
        Ok(())
    }

    fn synchronize(&mut self, _: &FusedScheduleStep) -> Result<(), String> {
        Ok(())
    }
}

impl<D: prism_amd_npu_runtime::XdnaCommandSubmitter> XdnaRouteBackend
    for CImageXdnaRouteDispatcher<'_, D>
{
    fn dispatch_xdna(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        RouteDispatch::dispatch_xdna(self, step, inputs, outputs)
    }
}

impl BindingResolver for CImageBindingResolver<'_> {
    fn resolve_inputs(&mut self, step: &FusedScheduleStep) -> Result<Vec<ResolvedBuffer>, String> {
        step.input_tensors
            .iter()
            .map(|binding| {
                self.runtime_outputs
                    .get(&binding.name)
                    .cloned()
                    .map(Ok)
                    .unwrap_or_else(|| self.resolve(binding, &step.input_region, step.zero_copy))
            })
            .collect()
    }

    fn resolve_outputs(&mut self, step: &FusedScheduleStep) -> Result<Vec<ResolvedBuffer>, String> {
        step.output_tensors
            .iter()
            .map(|binding| {
                if self.model.tensors.contains_key(&binding.name) {
                    self.resolve(binding, &step.output_region, step.zero_copy)
                } else {
                    Ok(ResolvedBuffer {
                        name: binding.name.clone(),
                        element_type: binding.element_type.clone(),
                        region: step.output_region.clone(),
                        byte_length: binding
                            .shape
                            .iter()
                            .copied()
                            .fold(1u64, u64::saturating_mul)
                            .saturating_mul(match binding.element_type.as_str() {
                                "int8" => 1,
                                "int32" => 4,
                                "fp16" => 2,
                                "fp32" => 4,
                                _ => 1,
                            }) as usize,
                        zero_copy: false,
                        file_offset: None,
                        storage: BufferStorage::RuntimeOwned,
                        shape: binding.shape.clone(),
                        payload: Some(vec![
                            0;
                            binding
                                .shape
                                .iter()
                                .copied()
                                .fold(1u64, u64::saturating_mul)
                                .saturating_mul(match binding.element_type.as_str() {
                                    "int8" => 1,
                                    "int32" | "fp32" => 4,
                                    "fp16" => 2,
                                    _ => 1,
                                }) as usize
                        ]),
                    })
                }
            })
            .collect()
    }

    fn commit_outputs(
        &mut self,
        _step: &FusedScheduleStep,
        outputs: &[ResolvedBuffer],
    ) -> Result<(), String> {
        for output in outputs {
            self.runtime_outputs
                .insert(output.name.clone(), output.clone());
        }
        Ok(())
    }
}

impl CImageBindingResolver<'_> {
    fn resolve(
        &self,
        binding: &prism_spatial_ir::execution_plan::TensorBinding,
        region: &str,
        _zero_copy: bool,
    ) -> Result<ResolvedBuffer, String> {
        let payload = self
            .model
            .tensors
            .get(&binding.name)
            .ok_or_else(|| format!("CImage tensor binding '{}' is missing", binding.name))?;
        let record = self
            .model
            .tensor_records
            .get(&binding.name)
            .ok_or_else(|| format!("CImage tensor record '{}' is missing", binding.name))?;
        let expected_bytes = match binding.element_type.as_str() {
            "int8" => (record.dim_m as usize).saturating_mul(record.dim_n as usize),
            "int32" => (record.dim_m as usize)
                .saturating_mul(record.dim_n as usize)
                .saturating_mul(4),
            "fp16" => (record.dim_m as usize)
                .saturating_mul(record.dim_n as usize)
                .saturating_mul(2),
            _ => payload.len(),
        };
        if payload.len() < expected_bytes {
            return Err(format!(
                "CImage tensor '{}' is {} bytes, expected at least {}",
                binding.name,
                payload.len(),
                expected_bytes
            ));
        }
        Ok(ResolvedBuffer {
            name: binding.name.clone(),
            element_type: binding.element_type.clone(),
            region: region.into(),
            byte_length: payload.len(),
            // RuntimeModel currently owns copied Vec<u8> payloads. Do not
            // advertise zero-copy without also exposing the mapped file
            // offset for a backend to bind.
            zero_copy: self.model.mapped_cimage.is_some(),
            file_offset: Some(record.offset),
            storage: BufferStorage::MappedCImage,
            shape: if record.dim_m == 0 || record.dim_n == 0 {
                vec![]
            } else {
                vec![record.dim_m as u64, record.dim_n as u64]
            },
            payload: Some(payload.clone()),
        })
    }
}

impl RuntimeModel {
    /// Inspect only the CImage header and payload extents.
    ///
    /// This path is safe for multi-gigabyte artifacts: it never reads or
    /// allocates tensor, kernel, or Core ML payloads.  Use it for admission,
    /// recovery, inventory, and memory-budget checks before calling `load`.
    pub fn inspect(path: &Path) -> Result<CImageInspection, RuntimeError> {
        Self::inspect_with_promotion(path, true)
    }

    /// Inspect a freshly emitted artifact before backend promotion. This is
    /// intended only for the qualification workflow; production admission
    /// must use [`RuntimeModel::inspect`].
    pub fn inspect_for_validation(path: &Path) -> Result<CImageInspection, RuntimeError> {
        Self::inspect_with_promotion(path, false)
    }

    fn inspect_with_promotion(
        path: &Path,
        require_promotion: bool,
    ) -> Result<CImageInspection, RuntimeError> {
        let reader = CImageReader::open(path).map_err(RuntimeError::InvalidCImage)?;
        let validation = if require_promotion {
            reader.validate_payload_ranges()
        } else {
            reader.validate_payload_ranges_for_validation()
        };
        validation.map_err(RuntimeError::InvalidCImage)?;
        let file_bytes = std::fs::metadata(path)
            .map_err(|e| RuntimeError::FileNotFound(e.to_string()))?
            .len();
        let tensor_bytes = reader.header.tensors.values().map(|r| r.size).sum();
        let kernel_bytes = reader.header.kernels.values().map(|r| r.size).sum();
        let ane_program_bytes = reader.header.ane_programs.values().map(|r| r.size).sum();
        let xdna_artifact_bytes = reader.header.xdna_artifacts.values().map(|r| r.size).sum();
        let (has_batch_plan, has_realtime_plan) = reader
            .header
            .execution_plan
            .as_deref()
            .map(|json| {
                if let Ok(plan) = serde_json::from_str::<ExecutionPlan>(json) {
                    (true, plan.mode.is_realtime())
                } else if let Ok(manifest) = serde_json::from_str::<SpatialKernelManifest>(json) {
                    (
                        manifest.batch_plan.is_some(),
                        manifest.realtime_plan.is_some(),
                    )
                } else {
                    (false, false)
                }
            })
            .unwrap_or((false, false));
        Ok(CImageInspection {
            path: path.to_path_buf(),
            file_bytes,
            tensor_bytes,
            kernel_bytes,
            ane_program_bytes,
            xdna_artifact_bytes,
            tensor_count: reader.header.tensors.len(),
            kernel_count: reader.header.kernels.len(),
            ane_program_count: reader.header.ane_programs.len(),
            xdna_artifact_count: reader.header.xdna_artifacts.len(),
            has_native_xdna: !reader.header.xdna_artifacts.is_empty(),
            has_batch_plan,
            has_realtime_plan,
            model_manifest: reader.header.model_manifest.clone(),
            native_ternary_promotion: reader.header.native_ternary_promotion.clone(),
            joint_tiling_evidence: reader.header.joint_tiling_evidence.clone(),
            kv_compression_policy: reader.header.kv_compression_policy.clone(),
        })
    }

    /// Load a `.cimage` file from disk into memory.
    ///
    /// Parses the binary header, validates the magic and schema version,
    /// deserialises the JSON manifest, then reads every tensor and kernel
    /// payload into the corresponding maps.
    ///
    pub fn load(path: &Path) -> Result<Self, RuntimeError> {
        Self::load_with_promotion(path, true)
    }

    /// Load an emitted CImage for backend qualification before promotion.
    /// This does not weaken the normal production [`RuntimeModel::load`]
    /// admission path.
    pub fn load_for_validation(path: &Path) -> Result<Self, RuntimeError> {
        Self::load_with_promotion(path, false)
    }

    fn load_with_promotion(path: &Path, require_promotion: bool) -> Result<Self, RuntimeError> {
        let reader = CImageReader::open(path).map_err(RuntimeError::InvalidCImage)?;
        let validation = if require_promotion {
            reader.validate_payload_ranges()
        } else {
            reader.validate_payload_ranges_for_validation()
        };
        validation.map_err(RuntimeError::InvalidCImage)?;
        let mut file = File::open(path).map_err(|e| RuntimeError::FileNotFound(e.to_string()))?;
        let mapped_cimage = unsafe { memmap2::MmapOptions::new().map(&file) }
            .map_err(|e| RuntimeError::InvalidCImage(format!("mmap CImage: {e}")))?;
        let cimage_len = file
            .metadata()
            .map_err(|e| RuntimeError::InvalidCImage(format!("stat CImage: {e}")))?
            .len();
        let mut tensors = HashMap::new();
        let mut tensor_records = HashMap::new();
        let mut tensor_offsets = HashMap::new();
        let mut tensor_scales = HashMap::new();
        for (name, record) in &reader.header.tensors {
            let payload = read_region(&mut file, record.offset, record.size)?;
            tensor_offsets.insert(name.clone(), (record.offset, record.size));
            tensor_records.insert(name.clone(), record.clone());
            tensors.insert(name.clone(), payload);
            if let (Some(offset), Some(size)) = (record.scale_offset, record.scale_size) {
                tensor_scales.insert(name.clone(), read_region(&mut file, offset, size)?);
            }
        }
        let mut kernels = HashMap::new();
        let mut kernel_descriptors = HashMap::new();
        for (name, record) in &reader.header.kernels {
            kernels.insert(
                name.clone(),
                read_region(&mut file, record.offset, record.size)?,
            );
            if let Some(descriptor) = &record.descriptor {
                kernel_descriptors.insert(name.clone(), descriptor.clone());
            }
        }
        let mut ane_programs = HashMap::new();
        for (name, record) in &reader.header.ane_programs {
            ane_programs.insert(
                name.clone(),
                (
                    record.clone(),
                    read_region(&mut file, record.offset, record.size)?,
                ),
            );
        }
        let mut xdna_artifacts = HashMap::new();
        for name in reader.header.xdna_artifacts.keys() {
            let payload = reader.xdna_artifact(name).map_err(|error| {
                RuntimeError::InvalidCImage(format!("read XDNA artifact {name}: {error}"))
            })?;
            let artifact =
                prism_amd_npu_runtime::XdnaArtifact::decode(&payload).map_err(|error| {
                    RuntimeError::InvalidCImage(format!("decode XDNA artifact {name}: {error}"))
                })?;
            artifact.validate().map_err(|error| {
                RuntimeError::InvalidCImage(format!("validate XDNA artifact {name}: {error}"))
            })?;
            xdna_artifacts.insert(name.clone(), artifact);
        }
        let (mut execution_plan, mut realtime_execution_plan) = reader
            .header
            .execution_plan
            .as_deref()
            .and_then(|json| {
                if let Ok(plan) = serde_json::from_str::<ExecutionPlan>(json) {
                    return Some((Some(plan), None));
                }
                let manifest = serde_json::from_str::<SpatialKernelManifest>(json).ok()?;
                Some((manifest.batch_plan, manifest.realtime_plan))
            })
            .unwrap_or((None, None));
        let uop_capture = reader
            .header
            .execution_plan
            .as_deref()
            .filter(|plan| {
                serde_json::from_str::<serde_json::Value>(plan)
                    .ok()
                    .and_then(|value| value.get("capture_digest").cloned())
                    .is_some()
            })
            .map(|_| {
                reader.uop_capture().map_err(|error| {
                    RuntimeError::InvalidCImage(format!("invalid UOp capture: {error}"))
                })
            })
            .transpose()?;
        let uop_program = uop_capture
            .as_ref()
            .map(|capture| {
                UOpCompiledProgram::compile(capture.clone()).map_err(|error| {
                    RuntimeError::InvalidCImage(format!("compile embedded UOp capture: {error}"))
                })
            })
            .transpose()?;
        let mut uop_strategy_programs = HashMap::new();
        for strategy in reader.header.uop_captures.keys() {
            let program =
                UOpCompiledProgram::from_cimage_strategy(&reader, strategy).map_err(|error| {
                    RuntimeError::InvalidCImage(format!(
                        "load UOp strategy program {strategy:?}: {error}"
                    ))
                })?;
            uop_strategy_programs.insert(strategy.clone(), program);
        }
        let uop_workload_evidence = reader
            .uop_workload_evidence()
            .map_err(|error| {
                RuntimeError::InvalidCImage(format!("invalid UOp workload evidence: {error}"))
            })?
            .to_vec();
        let normalize_plan = |mut plan: ExecutionPlan| {
            for window in &mut plan.residency_windows {
                // A zero value in compiler output means “the complete
                // model”. Resolve that contract against the actual
                // mapped CImage size at load time so every stream event
                // carries an explicit whole-model residency requirement.
                if window.model_bytes == 0 {
                    window.model_bytes = cimage_len;
                }
            }
            plan
        };
        execution_plan = execution_plan.map(normalize_plan);
        realtime_execution_plan = realtime_execution_plan.map(normalize_plan);
        if let Some(plan) = execution_plan.as_ref().or(realtime_execution_plan.as_ref()) {
            for step in &plan.fused_steps {
                if let Some(model_id) = &step.model_id {
                    let Some(model_manifest) = reader.header.model_manifest.as_ref() else {
                        return Err(RuntimeError::InvalidCImage(format!(
                            "execution plan references namespaced model {model_id:?} but CImage has no model manifest"
                        )));
                    };
                    if model_manifest.get(model_id).is_none() {
                        return Err(RuntimeError::InvalidCImage(format!(
                            "execution plan references unknown model {model_id:?}"
                        )));
                    }
                }
            }
        }
        if let Some(model_manifest) = &reader.header.model_manifest {
            for model in model_manifest.models.values() {
                for projector in &model.projectors {
                    let record = reader
                        .header
                        .tensors
                        .get(&projector.tensor_name)
                        .ok_or_else(|| {
                            RuntimeError::InvalidCImage(format!(
                                "model {:?} projector tensor {:?} is missing",
                                model.id, projector.tensor_name
                            ))
                        })?;
                    if record.dim_m != projector.output_dim || record.dim_n != projector.input_dim {
                        return Err(RuntimeError::InvalidCImage(format!(
                            "model {:?} projector {:?} has shape {}x{}, expected {}x{}",
                            model.id,
                            projector.tensor_name,
                            record.dim_m,
                            record.dim_n,
                            projector.output_dim,
                            projector.input_dim
                        )));
                    }
                }
            }
        }
        Ok(Self {
            cimage_path: path.to_path_buf(),
            manifest: CImageManifest {
                schema_version: "TRB_CIMG/1".into(),
                source_digest: String::new(),
                tensor_count: tensors.len(),
                kernel_count: kernels.len(),
            },
            tensors,
            tensor_records,
            tensor_scales,
            kernels,
            kernel_descriptors,
            uop_capture,
            uop_program,
            uop_strategy_programs,
            uop_workload_evidence,
            ane_programs,
            xdna_artifacts,
            kv_compression_policy: reader.header.kv_compression_policy,
            model_manifest: reader.header.model_manifest,
            native_ternary_promotion: reader.header.native_ternary_promotion,
            joint_tiling_evidence: reader.header.joint_tiling_evidence,
            execution_plan,
            realtime_execution_plan,
            tensor_offsets,
            mapped_cimage: Some(mapped_cimage),
        })
    }

    /// Get a tensor's data by name.
    pub fn get_tensor(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(|v| v.as_slice())
    }

    /// Return sealed strategy evidence for one exact workload shape.
    pub fn uop_workload_evidence_for(
        &self,
        scenario: WorkloadScenario,
    ) -> Option<&crate::cimage::UOpWorkloadEvidence> {
        self.uop_workload_evidence
            .iter()
            .find(|entry| entry.scenario == scenario)
    }

    /// Return the embedded UOp capture after it has passed CImage admission.
    pub fn uop_capture(&self) -> Option<&CapturePlan> {
        self.uop_capture.as_ref()
    }

    /// Get the packed per-group scales for a native ternary tensor.
    pub fn get_tensor_scales(&self, name: &str) -> Option<&[u8]> {
        self.tensor_scales.get(name).map(|v| v.as_slice())
    }

    /// Return the compiler-selected progressive KV policy, if this CImage
    /// contains measured KV compression evidence.
    pub fn kv_compression_policy(&self) -> Option<&str> {
        self.kv_compression_policy.as_deref()
    }

    /// Return the validated MoE placement descriptor for a tensor.
    pub fn moe_descriptor(&self, name: &str) -> Option<&crate::cimage::MoeTensorDescriptor> {
        self.tensor_records
            .get(name)
            .and_then(|record| record.moe.as_ref())
    }

    /// Return the validated multimodal vision descriptor for a tensor.
    pub fn vision_descriptor(&self, name: &str) -> Option<&crate::cimage::VisionTensorDescriptor> {
        self.tensor_records
            .get(name)
            .and_then(|record| record.vision.as_ref())
    }

    /// Get a kernel's compiled binary by name.
    pub fn get_kernel(&self, name: &str) -> Option<&[u8]> {
        self.kernels.get(name).map(|v| v.as_slice())
    }

    /// Return a decoded, validated native XDNA artifact by name.
    pub fn xdna_artifact(&self, name: &str) -> Option<&prism_amd_npu_runtime::XdnaArtifact> {
        self.xdna_artifacts.get(name)
    }

    /// Resolve a specialised model before dispatching any of its programs.
    pub fn select_model(
        &self,
        modality: crate::model_manifest::ModelModality,
    ) -> Result<&crate::model_manifest::ModelManifest, RuntimeError> {
        self.model_manifest
            .as_ref()
            .ok_or_else(|| {
                RuntimeError::UnsupportedMode("CImage has no multi-model manifest".into())
            })?
            .select_modality(modality)
            .map_err(RuntimeError::UnsupportedMode)
    }

    pub fn validate_model_io(
        &self,
        model_id: &str,
        inputs: &[&str],
        outputs: &[&str],
    ) -> Result<(), RuntimeError> {
        self.model_manifest
            .as_ref()
            .ok_or_else(|| {
                RuntimeError::UnsupportedMode("CImage has no multi-model manifest".into())
            })?
            .validate_io(model_id, inputs, outputs)
            .map_err(RuntimeError::UnsupportedMode)
    }

    pub fn validate_model_hardware(
        &self,
        model_id: &str,
        available: crate::model_manifest::HardwareCapabilities,
    ) -> Result<(), RuntimeError> {
        self.model_manifest
            .as_ref()
            .ok_or_else(|| {
                RuntimeError::UnsupportedMode("CImage has no multi-model manifest".into())
            })?
            .validate_hardware(model_id, available)
            .map_err(RuntimeError::UnsupportedMode)
    }

    pub fn model_for_fused_step(
        &self,
        step: &FusedScheduleStep,
    ) -> Result<Option<&crate::model_manifest::ModelManifest>, RuntimeError> {
        let Some(model_id) = step.model_id.as_deref() else {
            return Ok(None);
        };
        let manifest = self.model_manifest.as_ref().ok_or_else(|| {
            RuntimeError::UnsupportedMode("CImage has no multi-model manifest".into())
        })?;
        manifest.get(model_id).map(Some).ok_or_else(|| {
            RuntimeError::InvalidCImage(format!("unknown fused-step model {model_id:?}"))
        })
    }

    /// Resolve a CImage-backed binding to a bounds-checked read-only mapped
    /// slice for zero-copy backend binding.
    pub fn mapped_buffer<'a>(&'a self, buffer: &ResolvedBuffer) -> Result<&'a [u8], RuntimeError> {
        if !buffer.zero_copy {
            return Err(RuntimeError::UnsupportedMode(format!(
                "buffer '{}' is not marked zero-copy",
                buffer.name
            )));
        }
        let offset = buffer.file_offset.ok_or_else(|| {
            RuntimeError::InvalidCImage(format!("buffer '{}' has no file offset", buffer.name))
        })? as usize;
        let end = offset.checked_add(buffer.byte_length).ok_or_else(|| {
            RuntimeError::InvalidCImage(format!("buffer '{}' range overflow", buffer.name))
        })?;
        let mapped = self
            .mapped_cimage
            .as_ref()
            .ok_or_else(|| RuntimeError::InvalidCImage("CImage is not memory-mapped".into()))?;
        mapped.get(offset..end).ok_or_else(|| {
            RuntimeError::InvalidCImage(format!("buffer '{}' exceeds CImage mapping", buffer.name))
        })
    }

    /// Return an embedded ANE program and its declared multi-input contract.
    pub fn get_ane_program(&self, name: &str) -> Option<(&crate::cimage::AneProgramRecord, &[u8])> {
        self.ane_programs
            .get(name)
            .map(|(record, payload)| (record, payload.as_slice()))
    }

    /// Select the embedded ANE program whose declared tensor contract matches
    /// an AOT step. This keeps program selection tied to compiler-emitted
    /// bindings rather than positional or name-prefix guesses.
    pub fn ane_program_for_step(
        &self,
        step: &FusedScheduleStep,
    ) -> Option<(&crate::cimage::AneProgramRecord, &[u8])> {
        let input_names: std::collections::HashSet<&str> = step
            .input_tensors
            .iter()
            .map(|binding| binding.name.as_str())
            .collect();
        let output_names: std::collections::HashSet<&str> = step
            .output_tensors
            .iter()
            .map(|binding| binding.name.as_str())
            .collect();
        self.ane_programs.values().find_map(|(record, payload)| {
            (input_names.contains(record.activation_input.as_str())
                && input_names.contains(record.weights_input.as_str())
                && output_names.contains(record.output.as_str()))
            .then_some((record, payload.as_slice()))
        })
    }

    /// Reconstruct a typed artifact suitable for backend dispatch.
    pub fn kernel_artifact(&self, name: &str) -> Result<KernelArtifact, RuntimeError> {
        let binary = self
            .kernels
            .get(name)
            .ok_or_else(|| RuntimeError::KernelNotFound(name.into()))?;
        let descriptor = self
            .kernel_descriptors
            .get(name)
            .ok_or_else(|| {
                RuntimeError::InvalidCImage(format!("kernel '{name}' has no descriptor"))
            })?
            .clone();
        Ok(KernelArtifact {
            payloads: vec![KernelPayload {
                binary: binary.clone(),
                descriptor: descriptor.clone(),
            }],
            manifest: KernelManifest {
                kernels: vec![descriptor],
                fusion_plan: None,
                manifest_digest: String::new(),
            },
            artifact_digest: String::new(),
        })
    }

    /// Number of layers in the model (inferred from the manifest's tensor
    /// names or an explicit layer count in the execution plan).
    pub fn num_layers(&self) -> usize {
        // Phase 9: parse layer count from execution_plan or deduce from
        //          tensor-name patterns (e.g. "layers.N.attention").
        0
    }
}

fn read_region(file: &mut File, offset: u64, size: u64) -> Result<Vec<u8>, RuntimeError> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| RuntimeError::InvalidCImage(format!("seek payload: {e}")))?;
    let mut payload = vec![0u8; size as usize];
    file.read_exact(&mut payload)
        .map_err(|e| RuntimeError::InvalidCImage(format!("read payload: {e}")))?;
    Ok(payload)
}

// ---------------------------------------------------------------------------
// UnifiedRuntime
// ---------------------------------------------------------------------------

/// Unified runtime that dispatches batch or realtime execution from the same
/// loaded model.
///
/// Owns the [`RuntimeModel`], an optional hardware [`KernelBackend`], an
/// optional KV cache for autoregressive decode, and the current
/// [`ExecutionMode`].
///
/// # State Machine
///
/// ```text
/// new(model) ──► with_backend ──► run_batch / run_prefill ──► run_decode ──► …
///                  │                                                │
///                  └── optional ─────────────────────────────────────┘
///
/// reset_kv_cache resets decode state without reloading the model.
/// ```
pub struct UnifiedRuntime {
    /// Loaded model data.
    model: RuntimeModel,
    /// Optional hardware backend for accelerated dispatch. When `None`,
    /// the runtime falls back to the CPU reference path (Phase 9+).
    backend: Option<Box<dyn KernelBackend>>,
    /// KV cache slots for autoregressive decode (one per layer).
    kv_cache: Option<Vec<Vec<u8>>>,
    /// Current execution mode.
    mode: ExecutionMode,
    /// Runtime measurements can override the static plan for a concrete
    /// workload without mutating the sealed CImage artifact.
    measured_strategy_overrides: HashMap<WorkloadScenario, String>,
    requested_batch_size: Option<u32>,
}

impl UnifiedRuntime {
    /// Create a new unified runtime from a loaded model.
    ///
    /// Defaults to [`ExecutionMode::Batch`] with no backend and no KV cache.
    /// Call [`with_backend`](Self::with_backend) to attach a hardware
    /// accelerator, and [`run_prefill`](Self::run_prefill) to switch to
    /// autoregressive mode.
    pub fn new(model: RuntimeModel) -> Self {
        let measured_strategy_overrides = model
            .uop_workload_evidence
            .iter()
            .map(|evidence| (evidence.scenario, evidence.selected_strategy.clone()))
            .collect();
        Self {
            model,
            backend: None,
            kv_cache: None,
            mode: ExecutionMode::Batch,
            measured_strategy_overrides,
            requested_batch_size: None,
        }
    }

    /// Install a measured strategy choice for one validated workload shape.
    /// The strategy must already exist in the sealed candidate set.
    pub fn install_measured_strategy(
        &mut self,
        scenario: WorkloadScenario,
        strategy_id: impl Into<String>,
    ) -> Result<(), String> {
        scenario.validate()?;
        let strategy_id = strategy_id.into();
        if !self.model.uop_strategy_programs.contains_key(&strategy_id) {
            return Err(format!(
                "UOp strategy '{strategy_id}' is not embedded in the model"
            ));
        }
        self.measured_strategy_overrides
            .insert(scenario, strategy_id);
        Ok(())
    }

    /// Select and install the best measured UOp candidate for one workload.
    /// Selection and installation are a single operation so a caller cannot
    /// accidentally publish a strategy ID that does not correspond to the
    /// measurement vector it just evaluated.
    pub fn install_measured_strategy_choice(
        &mut self,
        scenario: WorkloadScenario,
        strategies: &[prism_spatial_ir::FusionStrategy],
        measurements: &[prism_spatial_ir::FusionMeasurement],
    ) -> Result<String, String> {
        let (strategy_id, _) = crate::select_measured_uop_strategy(strategies, measurements)?;
        self.install_measured_strategy(scenario, strategy_id.clone())?;
        Ok(strategy_id)
    }

    /// Return the strategy currently selected for an exact workload shape.
    /// This exposes the policy decision separately from program dispatch so
    /// receipts and diagnostics can report why a workload took a path.
    pub fn selected_measured_strategy(&self, scenario: WorkloadScenario) -> Option<&str> {
        self.measured_strategy_for_scenario(scenario)
            .map(String::as_str)
    }

    fn measured_strategy_for_scenario(&self, scenario: WorkloadScenario) -> Option<&String> {
        self.measured_strategy_overrides.get(&scenario).or_else(|| {
            self.measured_strategy_overrides
                .iter()
                .filter(|(candidate, _)| {
                    candidate.realtime == scenario.realtime
                        && candidate.batch_size == scenario.batch_size
                })
                .min_by_key(|(candidate, strategy)| {
                    (
                        candidate.sequence_length.abs_diff(scenario.sequence_length),
                        strategy.as_str(),
                    )
                })
                .map(|(_, strategy)| strategy)
        })
    }

    /// Attach a hardware backend.
    ///
    /// When a backend is present, all dispatch calls route through
    /// [`KernelBackend::dispatch`]. Without one, the runtime uses the CPU
    /// reference path (where available).
    pub fn with_backend(mut self, backend: Box<dyn KernelBackend>) -> Self {
        self.backend = Some(backend);
        self
    }

    fn active_execution_plan(&self) -> Option<&ExecutionPlan> {
        match self.mode {
            ExecutionMode::Batch => self.model.execution_plan.as_ref(),
            ExecutionMode::RealtimePrefill | ExecutionMode::RealtimeDecode => self
                .model
                .realtime_execution_plan
                .as_ref()
                .or(self.model.execution_plan.as_ref()),
        }
    }

    /// Validate the AOT heterogeneous schedule before replay. This is the
    /// runtime admission gate for dependency order and streamed-model
    /// workload coverage.
    pub fn validate_aot_schedule(&self) -> Result<(), RuntimeError> {
        let plan = self.active_execution_plan().ok_or_else(|| {
            RuntimeError::UnsupportedMode("CImage has no AOT execution plan".into())
        })?;
        plan.validate().map_err(RuntimeError::InvalidCImage)?;
        for step in &plan.fused_steps {
            if step
                .depends_on
                .iter()
                .any(|dependency| *dependency >= step.step_id)
            {
                return Err(RuntimeError::InvalidCImage(format!(
                    "AOT schedule step {} has a forward dependency",
                    step.step_id
                )));
            }
        }
        if !plan.supports_all_streamed_workloads() {
            return Err(RuntimeError::UnsupportedMode(
                "AOT residency window does not support realtime text, batched text, and batched audio".into(),
            ));
        }
        for step in &plan.fused_steps {
            self.model.model_for_fused_step(step)?;
        }
        Ok(())
    }

    /// Replay the compiler-emitted heterogeneous plan through concrete
    /// backend hooks. The hook implementation owns buffer resolution and
    /// backend-specific dispatch; this method owns plan admission and receipt
    /// production.
    pub fn replay_aot<E: HeterogeneousExecutor>(
        &self,
        executor: &mut E,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        self.validate_aot_schedule()?;
        let plan = self.active_execution_plan().ok_or_else(|| {
            RuntimeError::UnsupportedMode("CImage has no AOT execution plan".into())
        })?;
        let mut resolver = CImageBindingResolver {
            model: &self.model,
            runtime_outputs: HashMap::new(),
        };
        AotScheduler::replay_resolved(plan, &mut resolver, executor)
            .map_err(RuntimeError::ExecutionFailed)
    }

    /// Replay the plan specialized for one realtime or batch workload. The
    /// selected fusion strategy is attached to each dispatch step before the
    /// backend sees it.
    pub fn replay_aot_for_workload<E: HeterogeneousExecutor>(
        &self,
        scenario: WorkloadScenario,
        executor: &mut E,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        self.validate_aot_schedule()?;
        let plan = self
            .active_execution_plan()
            .ok_or_else(|| {
                RuntimeError::UnsupportedMode("CImage has no AOT execution plan".into())
            })?
            .try_specialize_for_workload(scenario)
            .map_err(RuntimeError::UnsupportedMode)?;
        let mut resolver = CImageBindingResolver {
            model: &self.model,
            runtime_outputs: HashMap::new(),
        };
        AotScheduler::replay_resolved(&plan, &mut resolver, executor)
            .map_err(RuntimeError::ExecutionFailed)
    }

    /// Replay a phase-specialized plan using current backend queue telemetry.
    /// The plan migrates only dispatchable XDNA/Metal/CPU islands; fixed
    /// CPU-side attention and ANE routes remain unchanged.
    pub fn replay_aot_for_phase<E: HeterogeneousExecutor>(
        &self,
        phase: InferencePhase,
        queue_depths: &[(PlanBackend, u32)],
        executor: &mut E,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        self.validate_aot_schedule()?;
        let plan = self
            .active_execution_plan()
            .ok_or_else(|| {
                RuntimeError::UnsupportedMode("CImage has no AOT execution plan".into())
            })?
            .specialize_for_phase(phase, queue_depths);
        let mut resolver = CImageBindingResolver {
            model: &self.model,
            runtime_outputs: HashMap::new(),
        };
        AotScheduler::replay_resolved(&plan, &mut resolver, executor)
            .map_err(RuntimeError::ExecutionFailed)
    }

    /// Replay an AOT plan through the explicit ANE/Metal/Accelerate/CPU route
    /// table. This is the preferred integration point for production runtime
    /// backends because route labels cannot be silently collapsed into one
    /// generic dispatch method.
    pub fn replay_aot_routed<R: RouteDispatch>(
        &self,
        routes: R,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        self.validate_aot_schedule()?;
        let plan = self.active_execution_plan().ok_or_else(|| {
            RuntimeError::UnsupportedMode("CImage has no AOT execution plan".into())
        })?;
        let mut resolver = CImageBindingResolver {
            model: &self.model,
            runtime_outputs: HashMap::new(),
        };
        let mut executor = RoutedExecutor { routes };
        AotScheduler::replay_resolved(plan, &mut resolver, &mut executor)
            .map_err(RuntimeError::ExecutionFailed)
    }

    /// Routed counterpart to [`Self::replay_aot_for_workload`].
    pub fn replay_aot_routed_for_workload<R: RouteDispatch>(
        &self,
        scenario: WorkloadScenario,
        routes: R,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        self.validate_aot_schedule()?;
        let plan = self
            .active_execution_plan()
            .ok_or_else(|| {
                RuntimeError::UnsupportedMode("CImage has no AOT execution plan".into())
            })?
            .try_specialize_for_workload(scenario)
            .map_err(RuntimeError::UnsupportedMode)?;
        let mut resolver = CImageBindingResolver {
            model: &self.model,
            runtime_outputs: HashMap::new(),
        };
        let mut executor = RoutedExecutor { routes };
        AotScheduler::replay_resolved(&plan, &mut resolver, &mut executor)
            .map_err(RuntimeError::ExecutionFailed)
    }

    /// Routed phase-aware replay using live queue telemetry.
    pub fn replay_aot_routed_for_phase<R: RouteDispatch>(
        &self,
        phase: InferencePhase,
        queue_depths: &[(PlanBackend, u32)],
        routes: R,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        self.validate_aot_schedule()?;
        let plan = self
            .active_execution_plan()
            .ok_or_else(|| {
                RuntimeError::UnsupportedMode("CImage has no AOT execution plan".into())
            })?
            .specialize_for_phase(phase, queue_depths);
        let mut resolver = CImageBindingResolver {
            model: &self.model,
            runtime_outputs: HashMap::new(),
        };
        let mut executor = RoutedExecutor { routes };
        AotScheduler::replay_resolved(&plan, &mut resolver, &mut executor)
            .map_err(RuntimeError::ExecutionFailed)
    }

    /// Replay the compiler-emitted plan through Prism's assembled Apple
    /// route table. This is the production convenience entry point: ANE
    /// programs use the embedded Core ML/IOSurface adapter, while Metal,
    /// Accelerate, and CPU use the shared kernel backend contract.
    pub fn replay_aot_apple(&self) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        let mut ane = EmbeddedAneRouteBackend {
            runtime: self,
            outputs: HashMap::new(),
        };
        let accelerate = prism_ecs_kernel::AccelerateBackend;
        let metal = prism_ecs_kernel::MetalBackend::new();
        let cpu = prism_ecs_kernel::CpuBackend;
        let routes = KernelRouteDispatcher {
            model: &self.model,
            ane: &mut ane,
            accelerate: &accelerate,
            metal: &metal,
            cpu: &cpu,
            xdna: None,
        };
        self.replay_aot_routed(routes)
    }

    /// Replay the assembled Apple routes using the strategy selected for a
    /// concrete realtime or batch workload scenario.
    pub fn replay_aot_apple_for_workload(
        &self,
        scenario: WorkloadScenario,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        let mut ane = EmbeddedAneRouteBackend {
            runtime: self,
            outputs: HashMap::new(),
        };
        let accelerate = prism_ecs_kernel::AccelerateBackend;
        let metal = prism_ecs_kernel::MetalBackend::new();
        let cpu = prism_ecs_kernel::CpuBackend;
        let routes = KernelRouteDispatcher {
            model: &self.model,
            ane: &mut ane,
            accelerate: &accelerate,
            metal: &metal,
            cpu: &cpu,
            xdna: None,
        };
        self.replay_aot_routed_for_workload(scenario, routes)
    }

    /// Replay the compiler-emitted plan with a native XDNA island included in
    /// the same route table as the Apple/CPU backends. This is the public
    /// heterogeneous entry point for deployments that provide an
    /// `XdnaDevice` implementation.
    pub fn replay_aot_with_xdna<D: prism_amd_npu_runtime::XdnaCommandSubmitter>(
        &self,
        device: D,
    ) -> Result<HeterogeneousExecutionReceipt, RuntimeError> {
        let mut ane = EmbeddedAneRouteBackend {
            runtime: self,
            outputs: HashMap::new(),
        };
        let accelerate = prism_ecs_kernel::AccelerateBackend;
        let metal = prism_ecs_kernel::MetalBackend::new();
        let cpu = prism_ecs_kernel::CpuBackend;
        let mut xdna = CImageXdnaRouteDispatcher::new(&self.model, device)
            .map_err(RuntimeError::InvalidCImage)?;
        if matches!(self.mode, ExecutionMode::RealtimePrefill) {
            xdna.set_phase(XdnaExecutionPhase::Prefill { tokens: 1 });
        }
        let routes = KernelRouteDispatcher {
            model: &self.model,
            ane: &mut ane,
            accelerate: &accelerate,
            metal: &metal,
            cpu: &cpu,
            xdna: Some(&mut xdna),
        };
        self.replay_aot_routed(routes)
    }

    /// Dispatch a registered stateless int8 ANE program. The model payload is
    /// unpacked only at program-load time; all activation, weight, and output
    /// tensors use IOSurface-backed arenas for the actual prediction.
    #[cfg(all(feature = "ane", target_os = "macos"))]
    pub fn dispatch_ane_int8(
        &self,
        program_name: &str,
        activation: &[i8],
        activation_shape: (u32, u32),
        weights: &[i8],
        weight_shape: (u32, u32),
    ) -> Result<Vec<i8>, RuntimeError> {
        self.dispatch_ane_int8_i32(
            program_name,
            activation,
            activation_shape,
            weights,
            weight_shape,
        )
        .map(|output| {
            output
                .into_iter()
                .map(|value| value.clamp(i8::MIN as i32, i8::MAX as i32) as i8)
                .collect()
        })
    }

    #[cfg(all(feature = "ane", target_os = "macos"))]
    pub fn dispatch_ane_int8_i32(
        &self,
        program_name: &str,
        activation: &[i8],
        activation_shape: (u32, u32),
        weights: &[i8],
        weight_shape: (u32, u32),
    ) -> Result<Vec<i32>, RuntimeError> {
        let (record, packed_model) = self.model.get_ane_program(program_name).ok_or_else(|| {
            RuntimeError::ExecutionFailed(format!("ANE program not found: {program_name}"))
        })?;
        if record.input_dtype != "int8" || record.output_dtype != "int8" {
            return Err(RuntimeError::UnsupportedMode(
                "ANE program is not int8".into(),
            ));
        }
        let activation_len = (activation_shape.0 as usize)
            .checked_mul(activation_shape.1 as usize)
            .ok_or_else(|| RuntimeError::ExecutionFailed("activation shape overflows".into()))?;
        let weight_len = (weight_shape.0 as usize)
            .checked_mul(weight_shape.1 as usize)
            .ok_or_else(|| RuntimeError::ExecutionFailed("weight shape overflows".into()))?;
        if activation.len() != activation_len || weights.len() != weight_len {
            return Err(RuntimeError::ExecutionFailed(
                "int8 inputs must exactly match their declared IOSurface shapes".into(),
            ));
        }
        let base = std::env::temp_dir().join(format!("prism-ane-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).map_err(|e| RuntimeError::ExecutionFailed(e.to_string()))?;
        let result = (|| {
            prism_ane::unpack_mlmodelc(packed_model, &base)
                .map_err(RuntimeError::ExecutionFailed)?;
            let model = prism_ane::coreml_bridge::CoreMlModel::load(&base)
                .map_err(RuntimeError::ExecutionFailed)?;
            let activation_arena = prism_ane::Arena::new(
                activation_shape.0,
                activation_shape.1,
                prism_ane::arena::Dtype::Int8,
            )
            .map_err(RuntimeError::ExecutionFailed)?;
            let weights_arena = prism_ane::Arena::new(
                weight_shape.0,
                weight_shape.1,
                prism_ane::arena::Dtype::Int8,
            )
            .map_err(RuntimeError::ExecutionFailed)?;
            let output_shape = (activation_shape.0, weight_shape.1);
            let mut output_arena = prism_ane::Arena::new(
                output_shape.0,
                output_shape.1,
                prism_ane::arena::Dtype::Int32,
            )
            .map_err(RuntimeError::ExecutionFailed)?;
            copy_int8_to_arena(&activation_arena, activation)
                .map_err(RuntimeError::ExecutionFailed)?;
            copy_int8_to_arena(&weights_arena, weights).map_err(RuntimeError::ExecutionFailed)?;
            model
                .predict_two_int8(
                    &record.activation_input,
                    &activation_arena,
                    &record.weights_input,
                    &weights_arena,
                    &record.output,
                    &mut output_arena,
                )
                .map_err(RuntimeError::ExecutionFailed)?;
            read_int32_from_arena(
                &output_arena,
                output_shape.0 as usize * output_shape.1 as usize,
            )
            .map_err(RuntimeError::ExecutionFailed)
        })();
        let _ = std::fs::remove_dir_all(&base);
        result
    }

    /// Dispatch a complete matrix through stateless ANE tiles, preserving
    /// int32 accumulators while K-slices are combined on the host.
    #[cfg(all(feature = "ane", target_os = "macos"))]
    pub fn dispatch_ane_int8_tiled(
        &self,
        program_name: &str,
        plan: &prism_ecs_quantization::ane_orchestration::AneTiledDispatchPlan,
        activation: &[i8],
        weights: &[i8],
    ) -> Result<Vec<i32>, RuntimeError> {
        let first_shape = plan
            .dispatches
            .first()
            .map(|tile| (tile.rows, tile.cols, tile.depth));
        if plan
            .dispatches
            .iter()
            .any(|tile| Some((tile.rows, tile.cols, tile.depth)) != first_shape)
        {
            return Err(RuntimeError::UnsupportedMode(
                "fixed ANE program cannot execute heterogeneous edge-tile shapes; use dispatch_ane_int8_tiled_with_programs".into(),
            ));
        }
        self.dispatch_ane_int8_tiled_with_programs(plan, activation, weights, |_| {
            Ok(program_name.to_string())
        })
    }

    /// Shape-aware variant for plans with edge tiles. The resolver selects a
    /// separately compiled stateless Core ML program for `(rows, cols, depth)`.
    #[cfg(all(feature = "ane", target_os = "macos"))]
    pub fn dispatch_ane_int8_tiled_with_programs<F>(
        &self,
        plan: &prism_ecs_quantization::ane_orchestration::AneTiledDispatchPlan,
        activation: &[i8],
        weights: &[i8],
        mut program_for_shape: F,
    ) -> Result<Vec<i32>, RuntimeError>
    where
        F: FnMut((usize, usize, usize)) -> Result<String, String>,
    {
        prism_ecs_quantization::ane_orchestration::execute_tiled_int8(
            plan,
            activation,
            weights,
            |(rows, cols), tile_activation, tile_weights| {
                let depth = tile_activation.len() / rows;
                let program_name =
                    program_for_shape((rows, cols, depth)).map_err(|error| error.to_string())?;
                self.dispatch_ane_int8_i32(
                    &program_name,
                    tile_activation,
                    (rows as u32, depth as u32),
                    tile_weights,
                    (depth as u32, cols as u32),
                )
                .map_err(|error| error.to_string())
            },
        )
        .map_err(RuntimeError::ExecutionFailed)
    }

    #[cfg(all(feature = "ane", target_os = "macos"))]
    pub fn dispatch_ane_int8_planar(
        &self,
        program_name: &str,
        activation: &[u8],
        activation_shape: (u32, u32),
        bias: &[u8],
        bias_shape: (u32, u32),
    ) -> Result<Vec<u8>, RuntimeError> {
        let (record, packed_model) = self.model.get_ane_program(program_name).ok_or_else(|| {
            RuntimeError::ExecutionFailed(format!("ANE program not found: {program_name}"))
        })?;
        if record.input_dtype != "int8" || record.output_dtype != "int8" {
            return Err(RuntimeError::UnsupportedMode(
                "ANE program is not int8 planar".into(),
            ));
        }
        let activation_len = (activation_shape.0 as usize)
            .checked_mul(activation_shape.1 as usize)
            .ok_or_else(|| {
                RuntimeError::ExecutionFailed("planar activation shape overflows".into())
            })?;
        let bias_len = (bias_shape.0 as usize)
            .checked_mul(bias_shape.1 as usize)
            .ok_or_else(|| RuntimeError::ExecutionFailed("planar bias shape overflows".into()))?;
        if activation.len() != activation_len || bias.len() != bias_len {
            return Err(RuntimeError::ExecutionFailed(
                "planar int8 inputs must exactly match their declared IOSurface shapes".into(),
            ));
        }
        let base = std::env::temp_dir().join(format!("prism-ane-planar-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).map_err(|e| RuntimeError::ExecutionFailed(e.to_string()))?;
        let result = (|| {
            prism_ane::unpack_mlmodelc(packed_model, &base)
                .map_err(RuntimeError::ExecutionFailed)?;
            let model = prism_ane::coreml_bridge::CoreMlModel::load(&base)
                .map_err(RuntimeError::ExecutionFailed)?;
            let activation_arena = prism_ane::Arena::new(
                activation_shape.0,
                activation_shape.1,
                prism_ane::arena::Dtype::Int8,
            )
            .map_err(RuntimeError::ExecutionFailed)?;
            let bias_arena =
                prism_ane::Arena::new(bias_shape.0, bias_shape.1, prism_ane::arena::Dtype::Int8)
                    .map_err(RuntimeError::ExecutionFailed)?;
            let mut output_arena = prism_ane::Arena::new(
                activation_shape.0,
                activation_shape.1,
                prism_ane::arena::Dtype::Int8,
            )
            .map_err(RuntimeError::ExecutionFailed)?;
            for (arena, bytes) in [(&activation_arena, activation), (&bias_arena, bias)] {
                if bytes.len() > arena.info.byte_size as usize {
                    return Err(RuntimeError::ExecutionFailed(
                        "int8 planar input exceeds IOSurface arena".into(),
                    ));
                }
                arena.lock().map_err(RuntimeError::ExecutionFailed)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        arena.info.base_address as *mut u8,
                        bytes.len(),
                    );
                }
                arena.unlock().map_err(RuntimeError::ExecutionFailed)?;
            }
            model
                .predict_two_int8_planar(
                    &record.activation_input,
                    &activation_arena,
                    &record.weights_input,
                    &bias_arena,
                    &record.output,
                    &mut output_arena,
                )
                .map_err(RuntimeError::ExecutionFailed)?;
            output_arena.lock().map_err(RuntimeError::ExecutionFailed)?;
            let output = unsafe {
                std::slice::from_raw_parts(
                    output_arena.info.base_address as *const u8,
                    output_arena.info.byte_size as usize,
                )
                .to_vec()
            };
            output_arena
                .unlock()
                .map_err(RuntimeError::ExecutionFailed)?;
            Ok(output)
        })();
        let _ = std::fs::remove_dir_all(&base);
        result
    }

    /// Run batch inference on the loaded model.
    ///
    /// Processes all input tokens in parallel, producing one logit vector
    /// per token. This is the GEMM-heavy code path used for scoring or
    /// classification.
    ///
    pub fn run_batch(&mut self, input_tokens: &[u32]) -> Result<Vec<f32>, RuntimeError> {
        self.mode = ExecutionMode::Batch;
        self.dispatch_tokens(input_tokens)
    }

    /// Run batch inference with an explicit workload batch shape. The batch
    /// value is policy metadata for strategy selection; callers remain
    /// responsible for packing the corresponding token buffer.
    pub fn run_batch_for_workload(
        &mut self,
        input_tokens: &[u32],
        batch_size: u32,
    ) -> Result<Vec<f32>, RuntimeError> {
        if batch_size == 0 {
            return Err(RuntimeError::ExecutionFailed(
                "batch workload size must be nonzero".into(),
            ));
        }
        if input_tokens.is_empty() || input_tokens.len() % batch_size as usize != 0 {
            return Err(RuntimeError::ExecutionFailed(
                "batch workload tokens must contain a nonempty, whole number of sequences".into(),
            ));
        }
        let previous = self.requested_batch_size.replace(batch_size);
        self.mode = ExecutionMode::Batch;
        let result = self.dispatch_tokens(input_tokens);
        self.requested_batch_size = previous;
        result
    }

    /// Run autoregressive prefill.
    ///
    /// Processes all prompt tokens in a single forward pass, populating the
    /// KV cache and returning the first generated token(s). After prefill
    /// the caller switches to [`run_decode`](Self::run_decode) for each
    /// subsequent token.
    ///
    pub fn run_prefill(&mut self, input_tokens: &[u32]) -> Result<Vec<u32>, RuntimeError> {
        if input_tokens.is_empty() {
            return Err(RuntimeError::ExecutionFailed(
                "prefill requires tokens".into(),
            ));
        }
        let logits = self.dispatch_tokens(input_tokens)?;
        self.kv_cache = Some(vec![input_tokens
            .iter()
            .flat_map(|t| t.to_ne_bytes())
            .collect()]);
        self.mode = ExecutionMode::RealtimePrefill;
        Ok(vec![argmax_token(&logits)])
    }

    /// Run canonical realtime prefill and return its logits. This is the
    /// adapter-facing form used by runtimes that own token sampling and KV
    /// lifecycle themselves.
    pub fn run_prefill_logits(&mut self, input_tokens: &[u32]) -> Result<Vec<f32>, RuntimeError> {
        if input_tokens.is_empty() {
            return Err(RuntimeError::ExecutionFailed(
                "prefill requires tokens".into(),
            ));
        }
        let logits = self.dispatch_tokens(input_tokens)?;
        self.kv_cache = Some(vec![input_tokens
            .iter()
            .flat_map(|token| token.to_ne_bytes())
            .collect()]);
        self.mode = ExecutionMode::RealtimePrefill;
        Ok(logits)
    }

    /// Run a single autoregressive decode step.
    ///
    /// Consumes the last generated token (stored in KV cache state),
    /// runs a single-token forward pass, and returns the next token ID.
    ///
    /// Must be preceded by a call to [`run_prefill`](Self::run_prefill).
    ///
    pub fn run_decode(&mut self) -> Result<u32, RuntimeError> {
        let cache = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| RuntimeError::UnsupportedMode("decode requires prefill".into()))?;
        let last = cache
            .first()
            .and_then(|bytes| bytes.rchunks_exact(4).next())
            .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
            .ok_or_else(|| RuntimeError::ExecutionFailed("decode cache is empty".into()))?;
        let logits = self.dispatch_tokens(&[last])?;
        self.mode = ExecutionMode::RealtimeDecode;
        if let Some(cache) = self.kv_cache.as_mut() {
            cache[0].extend_from_slice(&argmax_token(&logits).to_ne_bytes());
        }
        Ok(argmax_token(&logits))
    }

    /// Run one canonical realtime decode step and return logits to the caller
    /// that owns sampling.
    pub fn run_decode_logits(&mut self) -> Result<Vec<f32>, RuntimeError> {
        let cache = self
            .kv_cache
            .as_ref()
            .ok_or_else(|| RuntimeError::UnsupportedMode("decode requires prefill".into()))?;
        let last = cache
            .first()
            .and_then(|bytes| bytes.rchunks_exact(4).next())
            .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
            .ok_or_else(|| RuntimeError::ExecutionFailed("decode cache is empty".into()))?;
        let logits = self.dispatch_tokens(&[last])?;
        self.mode = ExecutionMode::RealtimeDecode;
        if let Some(cache) = self.kv_cache.as_mut() {
            cache[0].extend_from_slice(&argmax_token(&logits).to_ne_bytes());
        }
        Ok(logits)
    }

    /// Decode a caller-supplied token while retaining the canonical KV state.
    pub fn run_decode_logits_for_token(&mut self, token: u32) -> Result<Vec<f32>, RuntimeError> {
        {
            let cache = self
                .kv_cache
                .as_mut()
                .ok_or_else(|| RuntimeError::UnsupportedMode("decode requires prefill".into()))?;
            let slot = cache
                .first_mut()
                .ok_or_else(|| RuntimeError::ExecutionFailed("decode cache is empty".into()))?;
            slot.extend_from_slice(&token.to_ne_bytes());
        }
        let logits = self.dispatch_tokens(&[token])?;
        self.mode = ExecutionMode::RealtimeDecode;
        if let Some(slot) = self.kv_cache.as_mut().and_then(|cache| cache.first_mut()) {
            slot.extend_from_slice(&argmax_token(&logits).to_ne_bytes());
        }
        Ok(logits)
    }

    fn dispatch_tokens(&self, input_tokens: &[u32]) -> Result<Vec<f32>, RuntimeError> {
        let sequence_length = if self.mode == ExecutionMode::Batch {
            self.requested_batch_size
                .map(|batch_size| (input_tokens.len() / batch_size.max(1) as usize).max(1) as u32)
                .unwrap_or(input_tokens.len().max(1) as u32)
        } else {
            input_tokens.len().max(1) as u32
        };
        if let Some(program) = self.selected_uop_program(sequence_length) {
            if self.uop_program_accepts_tokens(program, input_tokens.len()) {
                return self.dispatch_uop_tokens(program, input_tokens);
            }
        }
        if self.backend.is_none() {
            // Keep the runtime usable on hosts without an attached hardware
            // backend. The same canonical packing and CPU kernel contracts
            // used by certification provide the reference execution path.
            return cpu_reference_inference(&self.model, input_tokens);
        }
        let backend = self.backend.as_ref().expect("backend checked above");
        let name = self
            .model
            .kernel_descriptors
            .keys()
            .next()
            .cloned()
            .ok_or_else(|| RuntimeError::KernelNotFound("no described kernels".into()))?;
        let artifact = self.model.kernel_artifact(&name)?;
        let descriptor = &artifact.payloads[0].descriptor;
        let token_bytes: Vec<u8> = input_tokens
            .iter()
            .flat_map(|t| (*t as f32).to_ne_bytes())
            .collect();
        let mut tensor_names = self.model.tensors.keys().cloned().collect::<Vec<_>>();
        tensor_names.sort();
        let inputs =
            match &descriptor.variant {
                KernelVariant::FP16GEMV => {
                    let weights_name = tensor_names
                        .first()
                        .ok_or_else(|| RuntimeError::TensorNotFound("no weights".into()))?;
                    let weights = self.model.tensors.get(weights_name).unwrap();
                    vec![weights.clone(), token_bytes]
                }
                KernelVariant::QuantizedGEMV => {
                    let weights_name = tensor_names.first().ok_or_else(|| {
                        RuntimeError::TensorNotFound("no quantized GEMV weights".into())
                    })?;
                    let weights = self.model.tensors.get(weights_name).ok_or_else(|| {
                        RuntimeError::TensorNotFound(format!("missing tensor {weights_name:?}"))
                    })?;
                    let record = self.model.tensor_records.get(weights_name).ok_or_else(|| {
                        RuntimeError::InvalidCImage("quantized GEMV weights have no shape".into())
                    })?;
                    let dims = [record.dim_m, record.dim_n];
                    vec![
                        weights.clone(),
                        input_tokens
                            .iter()
                            .flat_map(|token| (*token as f32).to_ne_bytes())
                            .collect(),
                        dims.iter().flat_map(|value| value.to_ne_bytes()).collect(),
                    ]
                }
                KernelVariant::FP16Matmul => {
                    let a_name = tensor_names
                        .first()
                        .ok_or_else(|| RuntimeError::TensorNotFound("no matrix A".into()))?;
                    let b_name = tensor_names
                        .get(1)
                        .ok_or_else(|| RuntimeError::TensorNotFound("no matrix B".into()))?;
                    let a_shape = self.model.tensor_records.get(a_name).ok_or_else(|| {
                        RuntimeError::InvalidCImage("matrix A has no shape".into())
                    })?;
                    let b_shape = self.model.tensor_records.get(b_name).ok_or_else(|| {
                        RuntimeError::InvalidCImage("matrix B has no shape".into())
                    })?;
                    let dims = [a_shape.dim_m, b_shape.dim_n, a_shape.dim_n];
                    vec![
                        self.model.tensors[a_name].clone(),
                        self.model.tensors[b_name].clone(),
                        dims.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                    ]
                }
                KernelVariant::INT8Tile640 => {
                    let weights_name = tensor_names
                        .first()
                        .ok_or_else(|| RuntimeError::TensorNotFound("no INT8 weights".into()))?;
                    let scales_name = tensor_names.get(1).ok_or_else(|| {
                        RuntimeError::TensorNotFound("no INT8 weight scales".into())
                    })?;
                    let input_scale_name = tensor_names.get(2).ok_or_else(|| {
                        RuntimeError::TensorNotFound("no INT8 input scale".into())
                    })?;
                    let record = self.model.tensor_records.get(weights_name).ok_or_else(|| {
                        RuntimeError::InvalidCImage("INT8 weights have no shape".into())
                    })?;
                    let dims = [record.dim_n, record.dim_m];
                    vec![
                        self.model.tensors[weights_name].clone(),
                        input_tokens
                            .iter()
                            .map(|token| *token as i8 as u8)
                            .collect(),
                        self.model.tensors[scales_name].clone(),
                        self.model.tensors[input_scale_name].clone(),
                        dims.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                    ]
                }
                KernelVariant::NF4Tile640 => {
                    let weights_name = tensor_names
                        .first()
                        .ok_or_else(|| RuntimeError::TensorNotFound("no NF4 weights".into()))?;
                    let scales_name = tensor_names.get(1).ok_or_else(|| {
                        RuntimeError::TensorNotFound("no NF4 weight scales".into())
                    })?;
                    let biases_name = tensor_names.get(2).ok_or_else(|| {
                        RuntimeError::TensorNotFound("no NF4 weight biases".into())
                    })?;
                    let record = self.model.tensor_records.get(weights_name).ok_or_else(|| {
                        RuntimeError::InvalidCImage("NF4 weights have no shape".into())
                    })?;
                    let tiles = (record.dim_n as usize).div_ceil(640);
                    let groups = tiles * 5;
                    let expected_codes = record.dim_m as usize * tiles * 320;
                    let expected_metadata = record.dim_m as usize * groups * 4;
                    if self.model.tensors[weights_name].len() != expected_codes
                        || self.model.tensors[scales_name].len() != expected_metadata
                        || self.model.tensors[biases_name].len() != expected_metadata
                    {
                        return Err(RuntimeError::InvalidCImage(
                            "NF4 Tile640 payload or group metadata is truncated".into(),
                        ));
                    }
                    let dims = [record.dim_n, record.dim_m];
                    vec![
                        self.model.tensors[weights_name].clone(),
                        input_tokens
                            .iter()
                            .flat_map(|token| (*token as f32).to_ne_bytes())
                            .collect(),
                        self.model.tensors[scales_name].clone(),
                        self.model.tensors[biases_name].clone(),
                        dims.iter().flat_map(|value| value.to_ne_bytes()).collect(),
                    ]
                }
                KernelVariant::TernaryTile640(_) => {
                    let name = tensor_names
                        .first()
                        .ok_or_else(|| RuntimeError::TensorNotFound("no ternary weights".into()))?;
                    let record = self.model.tensor_records.get(name).ok_or_else(|| {
                        RuntimeError::InvalidCImage("ternary weights have no shape".into())
                    })?;
                    let input_half: Vec<u8> = input_tokens
                        .iter()
                        .flat_map(|t| half::f16::from_f32(*t as f32).to_le_bytes())
                        .collect();
                    let pages = (record.dim_n as usize).div_ceil(640);
                    let packed_len = record.dim_m as usize * pages * 4;
                    let page_len = record.dim_m as usize * pages * 2;
                    let lane_len = record.dim_m as usize * pages;
                    let packed = &self.model.tensors[name];
                    if packed.len() < packed_len + page_len + lane_len {
                        return Err(RuntimeError::InvalidCImage(
                            "ternary payload is truncated".into(),
                        ));
                    }
                    let dims = [record.dim_n, record.dim_m];
                    vec![
                        packed[..packed_len].to_vec(),
                        input_half,
                        packed[packed_len..packed_len + page_len].to_vec(),
                        packed[packed_len + page_len..packed_len + page_len + lane_len].to_vec(),
                        dims.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                    ]
                }
                variant => {
                    return Err(RuntimeError::UnsupportedMode(format!(
                        "runtime input packing for {variant:?} is not implemented"
                    )))
                }
            };
        let output = backend
            .dispatch(&KernelDispatchRequest {
                artifact,
                inputs,
                bindings: vec![],
            })
            .map_err(|e| RuntimeError::BackendError(e.to_string()))?
            .outputs
            .into_iter()
            .next()
            .ok_or_else(|| RuntimeError::ExecutionFailed("backend returned no output".into()))?;
        if output.len() % 4 != 0 {
            return Err(RuntimeError::ExecutionFailed(
                "backend output is not FP32".into(),
            ));
        }
        Ok(output
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect())
    }

    fn selected_uop_program(&self, sequence_length: u32) -> Option<&UOpCompiledProgram> {
        let fallback = self.model.uop_program.as_ref();
        let Some(plan) = self.active_execution_plan() else {
            return fallback;
        };
        let realtime = matches!(
            self.mode,
            ExecutionMode::RealtimePrefill | ExecutionMode::RealtimeDecode
        );
        let batch_size = if realtime {
            1
        } else if let Some(batch_size) = self.requested_batch_size {
            batch_size
        } else {
            plan.batch_size.max(1)
        };
        let scenario = WorkloadScenario {
            realtime,
            batch_size,
            sequence_length: sequence_length.max(1),
        };
        let measured_strategy = self.measured_strategy_for_scenario(scenario);
        if let Some(strategy_id) = measured_strategy {
            return self
                .model
                .uop_strategy_programs
                .get(strategy_id)
                .or(fallback);
        }
        let Some(strategy) = plan.selected_workload_strategy(scenario) else {
            return fallback;
        };
        self.model
            .uop_strategy_programs
            .get(strategy.stable_id())
            .or(fallback)
    }

    fn dispatch_uop_tokens(
        &self,
        program: &UOpCompiledProgram,
        input_tokens: &[u32],
    ) -> Result<Vec<f32>, RuntimeError> {
        let mut inputs = std::collections::BTreeMap::new();
        for op in &program.capture.graph.ops {
            let prism_spatial_ir::UOpKind::Input { name } = &op.kind else {
                continue;
            };
            let values = if let Some(payload) = self.model.tensors.get(name) {
                if payload.len() % std::mem::size_of::<f32>() != 0 {
                    return Err(RuntimeError::InvalidCImage(format!(
                        "UOp tensor input {name:?} is not FP32-aligned"
                    )));
                }
                payload
                    .chunks_exact(std::mem::size_of::<f32>())
                    .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
                    .collect::<Vec<_>>()
            } else {
                input_tokens.iter().map(|token| *token as f32).collect()
            };
            let expected = op
                .shape
                .iter()
                .try_fold(1usize, |size, dimension| {
                    size.checked_mul(*dimension as usize)
                })
                .ok_or_else(|| {
                    RuntimeError::InvalidCImage(format!("UOp input {name:?} shape overflows"))
                })?;
            if values.len() != expected {
                return Err(RuntimeError::ExecutionFailed(format!(
                    "UOp input {name:?} has {} values, expected {expected}",
                    values.len()
                )));
            }
            inputs.insert(name.clone(), values);
        }
        let result = program
            .dispatch(&inputs)
            .map_err(RuntimeError::ExecutionFailed)?;
        result
            .outputs
            .into_values()
            .next()
            .ok_or_else(|| RuntimeError::ExecutionFailed("UOp program produced no outputs".into()))
    }

    fn uop_program_accepts_tokens(&self, program: &UOpCompiledProgram, token_count: usize) -> bool {
        let token_inputs = program.capture.graph.ops.iter().filter(|op| {
            let prism_spatial_ir::UOpKind::Input { name } = &op.kind else {
                return false;
            };
            !self.model.tensors.contains_key(name)
        });
        let mut saw_token_input = false;
        let all_match = token_inputs.clone().all(|op| {
            saw_token_input = true;
            op.shape.iter().try_fold(1usize, |size, dimension| {
                size.checked_mul(*dimension as usize)
            }) == Some(token_count)
        });
        saw_token_input && all_match
    }

    /// Reset the KV cache without reloading the model.
    ///
    /// After calling this, the runtime is back to a fresh prefill-ready
    /// state. The loaded tensors and kernels remain intact.
    pub fn reset_kv_cache(&mut self) {
        self.kv_cache = None;
        self.mode = ExecutionMode::Batch;
    }
}

fn argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index as u32)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// RuntimeError
// ---------------------------------------------------------------------------

/// Errors that can occur during runtime construction and execution.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// The `.cimage` file does not exist at the given path.
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// The file exists but is not a valid CImage (bad magic, corrupt header,
    /// or truncated payload).
    #[error("Invalid CImage: {0}")]
    InvalidCImage(String),

    /// The CImage schema version is not compatible with this runtime.
    #[error("Incompatible schema: {0}")]
    IncompatibleSchema(String),

    /// A required tensor is not present in the loaded model.
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    /// A required kernel is not present in the loaded model.
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),

    /// Execution failed at the runtime or kernel level.
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    /// The kernel backend returned an error.
    #[error("Backend error: {0}")]
    BackendError(String),

    /// The requested execution mode is not supported by this build or
    /// backend configuration.
    #[error("Unsupported execution mode: {0}")]
    UnsupportedMode(String),
}

// ---------------------------------------------------------------------------
// Certification
// ---------------------------------------------------------------------------

/// Outcome of comparing a backend's inference output against the CPU
/// reference implementation.
pub struct CertificationResult {
    /// Whether all output tensors matched within the specified tolerance.
    pub passed: bool,
    /// Maximum absolute error across all compared tensors.
    pub max_error: f32,
    /// Mean absolute error across all compared tensors.
    pub mean_error: f32,
    /// Names of tensors whose error exceeded the tolerance threshold.
    pub failed_tensors: Vec<String>,
}

/// Run CPU reference inference for certification.
///
/// Performs a forward pass using the CPU graph executor (when available)
/// and returns the raw logit vector. This is the correctness oracle that
/// all backend-accelerated paths are measured against.
///
/// The portable reference supports the canonical FP16, INT8, NF4, and
/// ternary Tile640 payload contracts. Unsupported custom variants still fail
/// closed rather than being mislabeled as CPU-compatible.
pub fn cpu_reference_inference(
    model: &RuntimeModel,
    input_tokens: &[u32],
) -> Result<Vec<f32>, RuntimeError> {
    let (descriptor, inputs) = pack_fp16_gemv_inputs(model, input_tokens)?;
    let mut cpu_descriptor = descriptor;
    cpu_descriptor.backend = BackendKind::CPU;
    let cpu = CpuBackend;
    let artifact = cpu
        .compile(&KernelCompileRequest {
            source: b"prism-cpu-reference".to_vec(),
            descriptor: cpu_descriptor,
            source_path: None,
        })
        .map_err(|error| RuntimeError::BackendError(error.to_string()))?;
    let output = cpu
        .dispatch(&KernelDispatchRequest {
            artifact,
            inputs,
            bindings: vec![],
        })
        .map_err(|error| RuntimeError::BackendError(error.to_string()))?;
    decode_f32_output(output.outputs.first())
}

/// Run backend inference and compare with the CPU reference.
///
/// Dispatches inference on the given hardware backend, runs the CPU
/// reference path for the same inputs, compares every output tensor
/// element-wise within `tolerance`, and returns a [`CertificationResult`].
///
/// Certification currently supports the `FP16GEMV` and `FP16Matmul` kernel
/// contracts. It uses
/// identical packed inputs for both paths and reports numerical error rather
/// than treating successful dispatch as proof of correctness.
pub fn certify_inference(
    model: &RuntimeModel,
    input_tokens: &[u32],
    backend: &dyn KernelBackend,
    tolerance: f32,
) -> Result<CertificationResult, RuntimeError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(RuntimeError::ExecutionFailed(
            "certification tolerance must be finite and non-negative".into(),
        ));
    }
    let reference = cpu_reference_inference(model, input_tokens)?;
    let (descriptor, inputs) = pack_fp16_gemv_inputs(model, input_tokens)?;
    let name = descriptor.name.clone();
    let artifact = model.kernel_artifact(&name)?;
    let output = backend
        .dispatch(&KernelDispatchRequest {
            artifact,
            inputs,
            bindings: vec![],
        })
        .map_err(|error| RuntimeError::BackendError(error.to_string()))?;
    let actual = decode_f32_output(output.outputs.first())?;
    if actual.len() != reference.len() {
        return Err(RuntimeError::ExecutionFailed(format!(
            "certification output length mismatch: CPU {}, backend {}",
            reference.len(),
            actual.len()
        )));
    }
    let mut max_error = 0.0f32;
    let mut total_error = 0.0f32;
    let mut failed = Vec::new();
    for (index, (expected, observed)) in reference.iter().zip(actual.iter()).enumerate() {
        let error = (expected - observed).abs();
        max_error = max_error.max(error);
        total_error += error;
        if error > tolerance {
            failed.push(format!("output[{index}]"));
        }
    }
    Ok(CertificationResult {
        passed: failed.is_empty(),
        max_error,
        mean_error: total_error / reference.len().max(1) as f32,
        failed_tensors: failed,
    })
}

fn pack_fp16_gemv_inputs(
    model: &RuntimeModel,
    input_tokens: &[u32],
) -> Result<(prism_ecs_kernel::KernelDescriptor, Vec<Vec<u8>>), RuntimeError> {
    if input_tokens.is_empty() {
        return Err(RuntimeError::ExecutionFailed(
            "inference requires input tokens".into(),
        ));
    }
    let name = model
        .kernel_descriptors
        .keys()
        .next()
        .cloned()
        .ok_or_else(|| RuntimeError::KernelNotFound("no described kernels".into()))?;
    let artifact = model.kernel_artifact(&name)?;
    let descriptor = artifact
        .payloads
        .first()
        .ok_or_else(|| RuntimeError::KernelNotFound(name.clone()))?
        .descriptor
        .clone();
    if !matches!(
        descriptor.variant,
        KernelVariant::FP16GEMV
            | KernelVariant::FP16Matmul
            | KernelVariant::INT8Tile640
            | KernelVariant::NF4Tile640
            | KernelVariant::TernaryTile640(_)
    ) {
        return Err(RuntimeError::UnsupportedMode(format!(
            "CPU certification supports FP16GEMV, FP16Matmul, INT8Tile640, NF4Tile640, and TernaryTile640, got {:?}",
            descriptor.variant
        )));
    }
    let mut tensor_names = model.tensors.keys().cloned().collect::<Vec<_>>();
    tensor_names.sort();
    let first = tensor_names
        .first()
        .ok_or_else(|| RuntimeError::TensorNotFound("no kernel input tensor".into()))?;
    let first_data = model
        .tensors
        .get(first)
        .cloned()
        .ok_or_else(|| RuntimeError::TensorNotFound(first.clone()))?;
    if matches!(descriptor.variant, KernelVariant::FP16GEMV) {
        let token_bytes = input_tokens
            .iter()
            .flat_map(|token| (*token as f32).to_ne_bytes())
            .collect();
        Ok((descriptor, vec![first_data, token_bytes]))
    } else if matches!(descriptor.variant, KernelVariant::INT8Tile640) {
        let scales = tensor_names
            .get(1)
            .ok_or_else(|| RuntimeError::TensorNotFound("no INT8 weight scales tensor".into()))?;
        let input_scale = tensor_names
            .get(2)
            .ok_or_else(|| RuntimeError::TensorNotFound("no INT8 input scale tensor".into()))?;
        let record = model
            .tensor_records
            .get(first)
            .ok_or_else(|| RuntimeError::InvalidCImage("INT8 weights have no shape".into()))?;
        let dims = [record.dim_n, record.dim_m];
        Ok((
            descriptor,
            vec![
                first_data,
                input_tokens
                    .iter()
                    .map(|token| *token as i8 as u8)
                    .collect(),
                model.tensors[scales].clone(),
                model.tensors[input_scale].clone(),
                dims.iter().flat_map(|value| value.to_ne_bytes()).collect(),
            ],
        ))
    } else if matches!(descriptor.variant, KernelVariant::TernaryTile640(_)) {
        let record = model
            .tensor_records
            .get(first)
            .ok_or_else(|| RuntimeError::InvalidCImage("ternary weights have no shape".into()))?;
        let pages = (record.dim_n as usize).div_ceil(640);
        let packed_len = record.dim_m as usize * pages * 4;
        let page_len = record.dim_m as usize * pages * 2;
        let lane_len = record.dim_m as usize * pages;
        if first_data.len() < packed_len + page_len + lane_len {
            return Err(RuntimeError::InvalidCImage(
                "ternary payload is truncated".into(),
            ));
        }
        let dims = [record.dim_n, record.dim_m];
        Ok((
            descriptor,
            vec![
                first_data[..packed_len].to_vec(),
                input_tokens
                    .iter()
                    .flat_map(|token| half::f16::from_f32(*token as f32).to_le_bytes())
                    .collect(),
                first_data[packed_len..packed_len + page_len].to_vec(),
                first_data[packed_len + page_len..packed_len + page_len + lane_len].to_vec(),
                dims.iter().flat_map(|value| value.to_ne_bytes()).collect(),
            ],
        ))
    } else if matches!(descriptor.variant, KernelVariant::NF4Tile640) {
        let scales = tensor_names
            .get(1)
            .ok_or_else(|| RuntimeError::TensorNotFound("no NF4 weight scales tensor".into()))?;
        let biases = tensor_names
            .get(2)
            .ok_or_else(|| RuntimeError::TensorNotFound("no NF4 weight biases tensor".into()))?;
        let record = model
            .tensor_records
            .get(first)
            .ok_or_else(|| RuntimeError::InvalidCImage("NF4 weights have no shape".into()))?;
        let dims = [record.dim_n, record.dim_m];
        Ok((
            descriptor,
            vec![
                first_data,
                input_tokens
                    .iter()
                    .flat_map(|token| (*token as f32).to_ne_bytes())
                    .collect(),
                model.tensors[scales].clone(),
                model.tensors[biases].clone(),
                dims.iter().flat_map(|value| value.to_ne_bytes()).collect(),
            ],
        ))
    } else {
        let second = tensor_names
            .get(1)
            .ok_or_else(|| RuntimeError::TensorNotFound("no matrix B tensor".into()))?;
        let second_data = model
            .tensors
            .get(second)
            .cloned()
            .ok_or_else(|| RuntimeError::TensorNotFound(second.clone()))?;
        let a = model
            .tensor_records
            .get(first)
            .ok_or_else(|| RuntimeError::InvalidCImage("matrix A has no shape".into()))?;
        let b = model
            .tensor_records
            .get(second)
            .ok_or_else(|| RuntimeError::InvalidCImage("matrix B has no shape".into()))?;
        let dims = [a.dim_m, b.dim_n, a.dim_n];
        Ok((
            descriptor,
            vec![
                first_data,
                second_data,
                dims.iter().flat_map(|value| value.to_ne_bytes()).collect(),
            ],
        ))
    }
}

fn decode_f32_output(output: Option<&Vec<u8>>) -> Result<Vec<f32>, RuntimeError> {
    let bytes =
        output.ok_or_else(|| RuntimeError::ExecutionFailed("backend returned no output".into()))?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(RuntimeError::ExecutionFailed(
            "backend output is not f32-aligned".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that every [`RuntimeError`] variant formats correctly and
    /// that the `std::error::Error` trait (from `thiserror`) is satisfied.
    #[test]
    fn test_runtime_error_types() {
        let err = RuntimeError::FileNotFound("missing.cimage".into());
        assert_eq!(format!("{err}"), "File not found: missing.cimage");

        let err = RuntimeError::InvalidCImage("bad magic".into());
        assert_eq!(format!("{err}"), "Invalid CImage: bad magic");

        let err = RuntimeError::IncompatibleSchema("v2 required".into());
        assert_eq!(format!("{err}"), "Incompatible schema: v2 required");

        let err = RuntimeError::TensorNotFound("weights".into());
        assert_eq!(format!("{err}"), "Tensor not found: weights");

        let err = RuntimeError::KernelNotFound("matmul".into());
        assert_eq!(format!("{err}"), "Kernel not found: matmul");

        let err = RuntimeError::ExecutionFailed("OOM".into());
        assert_eq!(format!("{err}"), "Execution failed: OOM");

        let err = RuntimeError::BackendError("GPU hung".into());
        assert_eq!(format!("{err}"), "Backend error: GPU hung");

        let err = RuntimeError::UnsupportedMode("decode".into());
        assert_eq!(format!("{err}"), "Unsupported execution mode: decode");
    }

    /// [`ExecutionMode`] derives `Clone + Copy`, so a copied value must be
    /// equal to the original and independent.
    #[test]
    fn test_execution_mode_copy() {
        let batch = ExecutionMode::Batch;
        let prefill = ExecutionMode::RealtimePrefill;
        let decode = ExecutionMode::RealtimeDecode;

        // Copy semantics — second binding is a bitwise copy.
        let batch2 = batch;
        let prefill2 = prefill;
        let decode2 = decode;

        assert_eq!(batch, batch2);
        assert_eq!(prefill, prefill2);
        assert_eq!(decode, decode2);

        // All variants are distinct.
        assert_ne!(batch, prefill);
        assert_ne!(batch, decode);
        assert_ne!(prefill, decode);
    }

    /// Construct a [`RuntimeModel`] with empty maps and verify the
    /// accessor methods return `None` for unknown names.
    #[test]
    fn test_runtime_model_new() {
        let manifest = CImageManifest::default();
        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            manifest,
            tensors: HashMap::new(),
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: None,
            uop_program: None,
            uop_strategy_programs: HashMap::new(),
            uop_workload_evidence: Vec::new(),
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            execution_plan: None,
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };

        assert_eq!(model.cimage_path.to_str(), Some("test.cimage"));
        assert!(model.get_tensor("nonexistent").is_none());
        assert!(model.get_kernel("nonexistent").is_none());
        assert_eq!(model.num_layers(), 0);
    }

    #[test]
    fn kernel_selection_is_scoped_to_route_backend() {
        let geometry = prism_ecs_kernel::DispatchGeometry {
            threads_per_threadgroup: [1, 1, 1],
            threadgroups_per_grid: [1, 1, 1],
            threads_per_grid: [1, 1, 1],
        };
        let descriptor = |backend| prism_ecs_kernel::KernelDescriptor {
            name: String::new(),
            variant: prism_ecs_kernel::KernelVariant::Custom("test".into()),
            backend,
            source_digest: String::new(),
            binary_digest: String::new(),
            binding_signature: Vec::new(),
            dispatch_geometry: geometry,
        };
        let mut descriptors = HashMap::new();
        descriptors.insert("cpu_step".into(), descriptor(BackendKind::CPU));
        descriptors.insert("metal_step".into(), descriptor(BackendKind::Metal));

        let cpu_names = kernel_names_for_backend(&descriptors, BackendKind::CPU);
        let metal_names = kernel_names_for_backend(&descriptors, BackendKind::Metal);
        assert_eq!(cpu_names.iter().map(|name| name.as_str()).collect::<Vec<_>>(), ["cpu_step"]);
        assert_eq!(
            metal_names.iter().map(|name| name.as_str()).collect::<Vec<_>>(),
            ["metal_step"]
        );
    }

    #[test]
    fn xdna_route_materializes_mapped_tensor_inputs_when_payload_is_absent() {
        let mut tensors = HashMap::new();
        tensors.insert("weights".into(), vec![1, 2, 3, 4]);
        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            manifest: CImageManifest::default(),
            tensors,
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: None,
            uop_program: None,
            uop_strategy_programs: HashMap::new(),
            uop_workload_evidence: Vec::new(),
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            execution_plan: None,
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };
        let dispatcher = CImageXdnaRouteDispatcher {
            model: &model,
            runtime: XdnaRuntime::new(),
            device: (),
            phase: XdnaExecutionPhase::Decode,
        };
        let inputs = vec![ResolvedBuffer {
            name: "weights".into(),
            element_type: "u8".into(),
            region: "unified-memory".into(),
            byte_length: 4,
            zero_copy: true,
            file_offset: Some(128),
            storage: BufferStorage::MappedCImage,
            shape: vec![4],
            payload: None,
        }];
        assert_eq!(
            dispatcher.payloads_for_inputs(&inputs)["weights"],
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn runtime_selects_workload_strategy_program() {
        let mut graph = prism_spatial_ir::TinyGraph::default();
        let input = graph.add(
            prism_spatial_ir::UOpKind::Input { name: "x".into() },
            vec![],
            vec![2],
        );
        let relu = graph.add(prism_spatial_ir::UOpKind::Relu, vec![input], vec![2]);
        let exp = graph.add(prism_spatial_ir::UOpKind::Exp, vec![relu], vec![2]);
        graph.add(
            prism_spatial_ir::UOpKind::Output { name: "y".into() },
            vec![exp],
            vec![2],
        );
        let standard = UOpCompiledProgram::compile(
            graph
                .lower(prism_spatial_ir::LoweringTarget::Portable)
                .unwrap(),
        )
        .unwrap();
        let per_operation = UOpCompiledProgram::compile(
            graph
                .lower_with_fusion_strategy(
                    prism_spatial_ir::LoweringTarget::Portable,
                    &prism_spatial_ir::FusionStrategy::PerOperation,
                )
                .unwrap(),
        )
        .unwrap();
        let scenario = WorkloadScenario {
            realtime: false,
            batch_size: 32,
            sequence_length: 1,
        };
        let evaluation = prism_spatial_ir::FusionStrategyEvaluation {
            candidates: vec![
                prism_spatial_ir::FusionStrategyCandidate {
                    strategy: prism_spatial_ir::FusionStrategy::StandardFused,
                    kernel_count: 1,
                    estimated_latency_ns: 20,
                    estimated_materialized_bytes: 0,
                    score: 20.0,
                    measured: true,
                },
                prism_spatial_ir::FusionStrategyCandidate {
                    strategy: prism_spatial_ir::FusionStrategy::PerOperation,
                    kernel_count: 2,
                    estimated_latency_ns: 10,
                    estimated_materialized_bytes: 0,
                    score: 10.0,
                    measured: true,
                },
            ],
            selected: 1,
        };
        let plan = ExecutionPlan::new(
            prism_spatial_ir::execution_plan::ExecutionMode::Batch,
            vec![],
            32,
            false,
        )
        .with_workload_evaluations(vec![prism_spatial_ir::WorkloadStrategyEvaluation {
            scenario,
            evaluation,
        }]);
        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            manifest: CImageManifest::default(),
            tensors: HashMap::new(),
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: Some(standard.capture.clone()),
            uop_program: Some(standard.clone()),
            uop_strategy_programs: HashMap::from([
                ("standard_fused".into(), standard.clone()),
                ("per_operation".into(), per_operation),
            ]),
            uop_workload_evidence: vec![crate::cimage::UOpWorkloadEvidence {
                scenario,
                strategies: vec!["standard_fused".into(), "per_operation".into()],
                candidate_capture_digests: Vec::new(),
                measurements: vec![
                    prism_spatial_ir::FusionMeasurement {
                        candidate_index: 0,
                        latency_ns: 100,
                        materialized_bytes: 0,
                    },
                    prism_spatial_ir::FusionMeasurement {
                        candidate_index: 1,
                        latency_ns: 1,
                        materialized_bytes: 0,
                    },
                ],
                selected_strategy: "per_operation".into(),
            }],
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            execution_plan: Some(plan),
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };
        let mut runtime = UnifiedRuntime::new(model);
        assert_eq!(
            runtime.selected_measured_strategy(scenario),
            Some("per_operation")
        );
        assert_eq!(
            runtime.selected_uop_program(2).unwrap().capture.digest(),
            runtime.model.uop_strategy_programs["per_operation"]
                .capture
                .digest()
        );
        assert_eq!(
            runtime.selected_measured_strategy(WorkloadScenario {
                realtime: false,
                batch_size: 32,
                sequence_length: 2,
            }),
            Some("per_operation")
        );
        assert!(runtime.run_batch_for_workload(&[1], 0).is_err());
        assert!(runtime.run_batch_for_workload(&[1, 2, 3], 2).is_err());
        let batch_logits = runtime
            .run_batch_for_workload(&[1, 2], 1)
            .expect("valid packed batch should dispatch");
        assert!(!batch_logits.is_empty());
        assert_eq!(
            runtime
                .model
                .uop_workload_evidence_for(scenario)
                .unwrap()
                .selected_strategy,
            "per_operation"
        );
        assert_eq!(
            runtime
                .selected_uop_program(1)
                .unwrap()
                .capture
                .kernels
                .len(),
            2
        );
        let selected = runtime
            .install_measured_strategy_choice(
                scenario,
                &[
                    prism_spatial_ir::FusionStrategy::StandardFused,
                    prism_spatial_ir::FusionStrategy::PerOperation,
                ],
                &[
                    prism_spatial_ir::FusionMeasurement {
                        candidate_index: 0,
                        latency_ns: 1,
                        materialized_bytes: 0,
                    },
                    prism_spatial_ir::FusionMeasurement {
                        candidate_index: 1,
                        latency_ns: 100,
                        materialized_bytes: 0,
                    },
                ],
            )
            .unwrap();
        assert_eq!(selected, "standard_fused");
        assert_eq!(
            runtime
                .selected_uop_program(1)
                .unwrap()
                .capture
                .kernels
                .len(),
            1
        );
    }

    #[test]
    fn validation_load_binds_native_ternary_scales() {
        let path = std::env::temp_dir().join(format!(
            "prism_runtime_native_scales_{}.cimage",
            std::process::id()
        ));
        let mut writer = crate::cimage::CImageWriter::new(&path).expect("create CImage");
        writer
            .append_native_ternary_with_scales(
                "weights",
                &[0, 1, 2, 0],
                &[0, 0, 128, 63],
                1,
                4,
                crate::cimage::TensorType::Ternary158,
                crate::cimage::TernaryDescriptor::legacy_for_type(
                    &crate::cimage::TensorType::Ternary158,
                )
                .unwrap(),
            )
            .expect("append native payload");
        writer.finalize().expect("finalize CImage");

        let model = RuntimeModel::load_for_validation(&path).expect("load validation model");
        assert_eq!(model.get_tensor("weights"), Some(&[0, 1, 2, 0][..]));
        assert_eq!(
            model.get_tensor_scales("weights"),
            Some(&[0, 0, 128, 63][..])
        );
        assert!(model.get_tensor_scales("missing").is_none());
        let _ = std::fs::remove_file(path);
    }

    /// Construct a [`UnifiedRuntime`] from a default model and verify
    /// default execution mode and missing backend.
    #[test]
    fn test_unified_runtime_new() {
        let manifest = CImageManifest::default();
        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            manifest,
            tensors: HashMap::new(),
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: None,
            uop_program: None,
            uop_strategy_programs: HashMap::new(),
            uop_workload_evidence: Vec::new(),
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            execution_plan: None,
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };

        let mut rt = UnifiedRuntime::new(model);

        // Default mode is batch.
        assert_eq!(rt.mode, ExecutionMode::Batch);

        // No backend attached by default.
        assert!(rt.backend.is_none());

        // No KV cache until prefill.
        assert!(rt.kv_cache.is_none());

        // Stub methods should return errors.
        assert!(rt.run_batch(&[0, 1, 2]).is_err());
        assert!(rt.run_prefill(&[0, 1, 2]).is_err());
        assert!(rt.run_decode().is_err());

        // Reset should be a no-op without KV cache.
        rt.reset_kv_cache();
        assert!(rt.kv_cache.is_none());
        assert_eq!(rt.mode, ExecutionMode::Batch);
    }

    /// Verify that [`RuntimeModel::load`] returns an error (stub).
    #[test]
    fn test_runtime_model_load_stub() {
        let result = RuntimeModel::load(Path::new("nonexistent.cimage"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, RuntimeError::InvalidCImage(_)));
    }

    #[test]
    fn replay_aot_covers_streamed_text_and_audio_workloads() {
        use prism_spatial_ir::execution_plan::{
            FusedScheduleStep, PlanBackend, ResidencyWindow, ResidencyWorkload,
        };
        use prism_spatial_ir::{BufferStorage, HeterogeneousExecutor, ResolvedStep};

        struct Executor {
            events: Vec<String>,
        }

        impl HeterogeneousExecutor for Executor {
            fn ensure_residency(&mut self, window_id: usize) -> Result<(), String> {
                self.events.push(format!("resident:{window_id}"));
                Ok(())
            }

            fn dispatch(
                &mut self,
                backend: PlanBackend,
                step: &FusedScheduleStep,
            ) -> Result<(), String> {
                self.events
                    .push(format!("route:{backend:?}:{}", step.step_id));
                Ok(())
            }

            fn dispatch_resolved(
                &mut self,
                backend: PlanBackend,
                resolved: &mut ResolvedStep<'_>,
            ) -> Result<(), String> {
                assert!(resolved
                    .inputs
                    .iter()
                    .chain(resolved.outputs.iter())
                    .all(|buffer| matches!(buffer.storage, BufferStorage::RuntimeOwned)));
                self.events
                    .push(format!("resolved:{backend:?}:{}", resolved.step.step_id));
                Ok(())
            }

            fn synchronize(&mut self, step: &FusedScheduleStep) -> Result<(), String> {
                self.events.push(format!("sync:{}", step.step_id));
                Ok(())
            }
        }

        let plan = prism_spatial_ir::execution_plan::ExecutionPlan {
            mode: prism_spatial_ir::execution_plan::ExecutionMode::Batch,
            schedule: vec![],
            batch_size: 32,
            persistent_cache: false,
            dispatch_policy: Default::default(),
            device_island: Default::default(),
            fused_steps: vec![FusedScheduleStep {
                step_id: 0,
                model_id: None,
                node_ids: vec![],
                backend: PlanBackend::AneMatrix,
                depends_on: vec![],
                input_region: "ane-memory".into(),
                output_region: "ane-memory".into(),
                zero_copy: true,
                estimated_latency_ns: 10,
                input_tensors: vec![],
                output_tensors: vec![],
                dispatch_geometry: [1, 1, 1],
                fusion_strategy: None,
            }],
            residency_windows: vec![ResidencyWindow {
                window_id: 7,
                model_bytes: 4096,
                required_workloads: vec![
                    ResidencyWorkload::RealtimeText,
                    ResidencyWorkload::BatchedText,
                    ResidencyWorkload::BatchedAudio,
                ],
                resident_devices: vec!["ane-memory".into(), "unified-memory".into()],
                prefetch_step: Some(0),
                eviction_step: None,
            }],
            fusion_evaluations: vec![],
            workload_evaluations: vec![],
        };
        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            manifest: CImageManifest::default(),
            tensors: HashMap::new(),
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: None,
            uop_program: None,
            uop_strategy_programs: HashMap::new(),
            uop_workload_evidence: Vec::new(),
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            execution_plan: Some(plan),
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };
        let runtime = UnifiedRuntime::new(model);
        let mut executor = Executor { events: vec![] };
        let receipt = runtime.replay_aot(&mut executor).unwrap();
        assert_eq!(receipt.steps.len(), 1);
        assert_eq!(receipt.model_residency_windows, 1);
        assert_eq!(
            executor.events,
            vec!["resident:7", "resolved:AneMatrix:0", "sync:0"]
        );
    }

    /// Verify that the free-standing stub functions return errors.
    #[test]
    fn test_stub_functions() {
        let manifest = CImageManifest::default();
        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            manifest,
            tensors: HashMap::new(),
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: None,
            uop_program: None,
            uop_strategy_programs: HashMap::new(),
            uop_workload_evidence: Vec::new(),
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            execution_plan: None,
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };

        let ref_result = cpu_reference_inference(&model, &[0, 1]);
        assert!(ref_result.is_err());

        let cert_result = certify_inference(
            &model,
            &[0, 1],
            // Use a concrete empty backend — we stub it here as a
            // reference, but in Phase 9 a real backend will be wired.
            &MockBackend,
            0.01,
        );
        assert!(cert_result.is_err());
    }

    /// Dummy backend for testing — returns errors for every method.
    struct MockBackend;

    impl KernelBackend for MockBackend {
        fn validate(
            &self,
            _descriptor: &prism_ecs_kernel::KernelDescriptor,
        ) -> Result<(), prism_ecs_kernel::KernelError> {
            Err(prism_ecs_kernel::KernelError::UnsupportedBackend(
                "mock".into(),
            ))
        }

        fn compile(
            &self,
            _request: &prism_ecs_kernel::KernelCompileRequest,
        ) -> Result<prism_ecs_kernel::KernelArtifact, prism_ecs_kernel::KernelError> {
            Err(prism_ecs_kernel::KernelError::UnsupportedBackend(
                "mock".into(),
            ))
        }

        fn dispatch(
            &self,
            _request: &prism_ecs_kernel::KernelDispatchRequest,
        ) -> Result<prism_ecs_kernel::KernelOutput, prism_ecs_kernel::KernelError> {
            Err(prism_ecs_kernel::KernelError::UnsupportedBackend(
                "mock".into(),
            ))
        }

        fn measure(
            &self,
            _request: &prism_ecs_kernel::KernelMeasurementRequest,
        ) -> Result<prism_ecs_kernel::KernelMeasurement, prism_ecs_kernel::KernelError> {
            Err(prism_ecs_kernel::KernelError::UnsupportedBackend(
                "mock".into(),
            ))
        }

        fn name(&self) -> &str {
            "mock"
        }
    }
}
