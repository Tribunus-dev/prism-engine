//! Kernel-backed execution resources for the ECS schedule.
//!
//! The runtime owns the lifetime of backend objects and compiled artifacts.
//! Dispatch requests carry only an immutable artifact reference plus concrete
//! inputs; the artifact bytes and backend trait object stay in this registry
//! across schedule ticks.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use prism_ecs_core::Component;
use prism_ecs_kernel::{
    BackendKind, BindingSlot, CpuBackend, KernelArtifact, KernelBackend, KernelDispatchRequest,
    KernelOutput, MetalBackend,
};
use serde::{Deserialize, Serialize};

use crate::ports::{
    DispatchError, DispatchHandle, DispatchRequest, DispatchStatus, RuntimeError, WorkDispatcher,
};

/// The serializable part of a kernel dispatch carried by a work item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelDispatchSpec {
    pub artifact_digest: String,
    pub backend: String,
    pub kernel_name: String,
    pub inputs: Vec<Vec<u8>>,
    pub bindings: Vec<BindingSlot>,
}

/// ECS component that binds a leased work item to a registered compiled
/// artifact. The component is deliberately a reference, not a second copy of
/// the artifact, so the kernel registry remains the only execution authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelArtifactBinding {
    pub artifact_digest: String,
    pub backend: String,
    pub kernel_name: String,
    pub inputs: Vec<Vec<u8>>,
    pub bindings: Vec<BindingSlot>,
}

impl Component for KernelArtifactBinding {}

impl KernelArtifactBinding {
    /// Create a work-item binding from an artifact accepted by the registry.
    pub fn for_artifact(
        artifact: &KernelArtifact,
        inputs: Vec<Vec<u8>>,
        bindings: Vec<BindingSlot>,
    ) -> Result<Self, RuntimeError> {
        let payload = artifact
            .payloads
            .first()
            .ok_or_else(|| RuntimeError::Dispatch("artifact has no payloads".into()))?;
        Ok(Self {
            artifact_digest: artifact.artifact_digest.clone(),
            backend: backend_label(payload.descriptor.backend).into(),
            kernel_name: payload.descriptor.name.clone(),
            inputs,
            bindings,
        })
    }

    pub fn dispatch_spec(&self) -> KernelDispatchSpec {
        KernelDispatchSpec {
            artifact_digest: self.artifact_digest.clone(),
            backend: self.backend.clone(),
            kernel_name: self.kernel_name.clone(),
            inputs: self.inputs.clone(),
            bindings: self.bindings.clone(),
        }
    }

    /// Preserve the existing provider-selection JSON while adding the typed
    /// kernel contract consumed by [`KernelBackendDispatcher`].
    pub fn dispatch_config<T: Serialize>(&self, provider_selection: &T) -> String {
        serde_json::json!({
            "provider_selection": provider_selection,
            "kernel_dispatch": self.dispatch_spec(),
        })
        .to_string()
    }
}

struct RegistryState {
    backends: HashMap<BackendKind, Arc<dyn KernelBackend>>,
    artifacts: HashMap<String, KernelArtifact>,
}

/// Persistent backend and compiled-artifact resources owned by a
/// [`crate::KernelHandle`]. Registering the same backend or artifact twice is
/// idempotent and retains the first resource instance.
#[derive(Clone)]
pub struct BackendExecutionRegistry {
    state: Arc<Mutex<RegistryState>>,
    statuses: Arc<Mutex<HashMap<String, DispatchStatus>>>,
}

impl Default for BackendExecutionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendExecutionRegistry {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(RegistryState {
                backends: HashMap::new(),
                artifacts: HashMap::new(),
            })),
            statuses: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register one long-lived implementation of a kernel backend.
    pub fn register_backend(
        &self,
        kind: BackendKind,
        backend: Arc<dyn KernelBackend>,
    ) -> Result<(), RuntimeError> {
        if backend.name().is_empty() {
            return Err(RuntimeError::Dispatch("backend has no name".into()));
        }
        let mut state = self.state.lock();
        state.backends.entry(kind).or_insert(backend);
        Ok(())
    }

    /// Register a compiled artifact after structural and backend validation.
    /// CPU and portable Metal resources are installed lazily; hardware-specific
    /// backends must be supplied by the composition root through
    /// [`Self::register_backend`].
    pub fn register_artifact(
        &self,
        artifact: KernelArtifact,
    ) -> Result<KernelArtifactBinding, RuntimeError> {
        validate_artifact(&artifact)?;
        let payload = artifact
            .payloads
            .first()
            .ok_or_else(|| RuntimeError::Dispatch("artifact has no payloads".into()))?;
        let kind = payload.descriptor.backend;
        self.install_default_backend(kind)?;
        let binding = KernelArtifactBinding::for_artifact(&artifact, Vec::new(), Vec::new())?;
        let mut state = self.state.lock();
        state
            .artifacts
            .entry(artifact.artifact_digest.clone())
            .or_insert(artifact);
        Ok(binding)
    }

    pub fn artifact_count(&self) -> usize {
        self.state.lock().artifacts.len()
    }

    pub fn backend_count(&self) -> usize {
        self.state.lock().backends.len()
    }

    pub fn validate_dispatch(
        &self,
        request_backend: &str,
        spec: &KernelDispatchSpec,
    ) -> Result<(), RuntimeError> {
        let requested_kind = parse_backend(request_backend)?;
        let spec_kind = parse_backend(&spec.backend)?;
        if requested_kind != spec_kind {
            return Err(RuntimeError::Dispatch(format!(
                "dispatch backend '{}' does not match artifact backend '{}'",
                request_backend, spec.backend
            )));
        }
        let state = self.state.lock();
        let artifact = state.artifacts.get(&spec.artifact_digest).ok_or_else(|| {
            RuntimeError::Dispatch(format!(
                "compiled artifact '{}' is not registered",
                spec.artifact_digest
            ))
        })?;
        let payload = artifact
            .payloads
            .iter()
            .find(|payload| payload.descriptor.name == spec.kernel_name)
            .ok_or_else(|| {
                RuntimeError::Dispatch(format!(
                    "kernel '{}' is absent from artifact '{}'",
                    spec.kernel_name, spec.artifact_digest
                ))
            })?;
        if payload.descriptor.backend != requested_kind {
            return Err(RuntimeError::Dispatch(format!(
                "kernel '{}' targets {:?}, request targets {:?}",
                spec.kernel_name, payload.descriptor.backend, requested_kind
            )));
        }
        if !spec.bindings.is_empty() && payload.descriptor.binding_signature != spec.bindings {
            return Err(RuntimeError::Dispatch(format!(
                "binding signature mismatch for kernel '{}'",
                spec.kernel_name
            )));
        }
        if !state.backends.contains_key(&requested_kind) {
            return Err(RuntimeError::Dispatch(format!(
                "backend resource {:?} is not registered",
                requested_kind
            )));
        }
        Ok(())
    }

    pub fn dispatch(
        &self,
        request_backend: &str,
        spec: &KernelDispatchSpec,
    ) -> Result<KernelOutput, RuntimeError> {
        self.validate_dispatch(request_backend, spec)?;
        let kind = parse_backend(request_backend)?;
        let (backend, artifact) = {
            let state = self.state.lock();
            (
                state.backends.get(&kind).cloned().ok_or_else(|| {
                    RuntimeError::Dispatch(format!("backend resource {:?} is not registered", kind))
                })?,
                state
                    .artifacts
                    .get(&spec.artifact_digest)
                    .cloned()
                    .ok_or_else(|| {
                        RuntimeError::Dispatch(format!(
                            "compiled artifact '{}' is not registered",
                            spec.artifact_digest
                        ))
                    })?,
            )
        };
        let payload = artifact
            .payloads
            .iter()
            .find(|payload| payload.descriptor.name == spec.kernel_name)
            .cloned()
            .ok_or_else(|| RuntimeError::Dispatch("kernel payload disappeared".into()))?;
        let mut selected = artifact;
        selected.payloads = vec![payload.clone()];
        selected.manifest.kernels = vec![payload.descriptor.clone()];
        backend
            .dispatch(&KernelDispatchRequest {
                artifact: selected,
                inputs: spec.inputs.clone(),
                bindings: spec.bindings.clone(),
            })
            .map_err(|error| RuntimeError::Dispatch(error.to_string()))
    }

    fn install_default_backend(&self, kind: BackendKind) -> Result<(), RuntimeError> {
        let backend: Arc<dyn KernelBackend> = match kind {
            BackendKind::CPU => Arc::new(CpuBackend),
            BackendKind::Metal => Arc::new(MetalBackend::new()),
            unsupported => {
                let state = self.state.lock();
                if state.backends.contains_key(&unsupported) {
                    return Ok(());
                }
                return Err(RuntimeError::Dispatch(format!(
                    "backend {:?} requires an explicitly registered resource",
                    unsupported
                )));
            }
        };
        self.register_backend(kind, backend)
    }
}

/// A provider-neutral dispatcher backed by the kernel registry. Start performs
/// validation and execution against the persistent backend resource; poll then
/// exposes the actual backend outcome to the ECS Collect stage.
pub struct KernelBackendDispatcher {
    registry: BackendExecutionRegistry,
}

impl KernelBackendDispatcher {
    pub fn new(registry: BackendExecutionRegistry) -> Self {
        Self { registry }
    }

    pub fn registry(&self) -> BackendExecutionRegistry {
        self.registry.clone()
    }
}

impl WorkDispatcher for KernelBackendDispatcher {
    fn start(&self, request: &DispatchRequest) -> Result<DispatchHandle, DispatchError> {
        let value: serde_json::Value = serde_json::from_str(&request.config).map_err(|error| {
            DispatchError::StartFailed(format!("invalid dispatch config: {error}"))
        })?;
        let spec: KernelDispatchSpec =
            serde_json::from_value(value.get("kernel_dispatch").cloned().ok_or_else(|| {
                DispatchError::StartFailed("kernel dispatch binding is missing".into())
            })?)
            .map_err(|error| {
                DispatchError::StartFailed(format!("invalid kernel dispatch binding: {error}"))
            })?;
        self.registry
            .validate_dispatch(&request.backend, &spec)
            .map_err(|error| DispatchError::StartFailed(error.to_string()))?;
        let id = format!(
            "kernel:{}:{}:{}",
            request.work_entity, request.attempt, spec.artifact_digest
        );
        let status = match self.registry.dispatch(&request.backend, &spec) {
            Ok(output) => {
                DispatchStatus::Completed(output.outputs.into_iter().next().unwrap_or_default())
            }
            Err(error) => DispatchStatus::Failed(error.to_string()),
        };
        self.registry.statuses.lock().insert(id.clone(), status);
        Ok(DispatchHandle {
            id,
            work_entity: request.work_entity,
            attempt: request.attempt,
        })
    }

    fn poll(&self, handle: &DispatchHandle) -> Result<DispatchStatus, DispatchError> {
        self.registry
            .statuses
            .lock()
            .get(&handle.id)
            .cloned()
            .ok_or_else(|| DispatchError::PollFailed(format!("unknown dispatch {}", handle.id)))
    }

    fn cancel(&self, handle: &DispatchHandle) -> Result<(), DispatchError> {
        self.registry.statuses.lock().remove(&handle.id);
        Ok(())
    }
}

fn backend_label(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Metal => "metal",
        BackendKind::CPU => "cpu",
        BackendKind::ANE => "ane",
        BackendKind::CUDA => "cuda",
        BackendKind::Vulkan => "vulkan",
        BackendKind::AmdNpu => "amd-npu",
    }
}

/// Parse a backend name (case-insensitive) into a [`BackendKind`]. Used by
/// the runtime schedule, the kernel, and the FFI bridge. The string `"auto"`
/// resolves to [`BackendKind::CPU`] as the safe default; callers that need
/// a different resolution should pass an explicit backend name.
pub fn parse_backend(name: &str) -> Result<BackendKind, RuntimeError> {
    match name.to_ascii_lowercase().as_str() {
        "metal" => Ok(BackendKind::Metal),
        "cpu" | "cpu-reference" => Ok(BackendKind::CPU),
        "ane" => Ok(BackendKind::ANE),
        "cuda" => Ok(BackendKind::CUDA),
        "vulkan" => Ok(BackendKind::Vulkan),
        "amd-npu" | "amdnpu" => Ok(BackendKind::AmdNpu),
        "auto" => Ok(BackendKind::CPU),
        other => Err(RuntimeError::Dispatch(format!("unknown backend '{other}'"))),
    }
}

fn validate_artifact(artifact: &KernelArtifact) -> Result<(), RuntimeError> {
    if artifact.artifact_digest.is_empty() {
        return Err(RuntimeError::Dispatch(
            "compiled artifact has no digest".into(),
        ));
    }
    if artifact.payloads.is_empty() {
        return Err(RuntimeError::Dispatch(
            "compiled artifact has no payloads".into(),
        ));
    }
    for payload in &artifact.payloads {
        if payload.binary.is_empty() {
            return Err(RuntimeError::Dispatch(format!(
                "kernel '{}' has an empty binary",
                payload.descriptor.name
            )));
        }
        if payload.descriptor.name.is_empty() || payload.descriptor.binary_digest.is_empty() {
            return Err(RuntimeError::Dispatch(
                "kernel payload is missing name or binary digest".into(),
            ));
        }
        if payload.descriptor.backend
            != artifact
                .manifest
                .kernels
                .iter()
                .find(|kernel| kernel.name == payload.descriptor.name)
                .map(|kernel| kernel.backend)
                .unwrap_or(payload.descriptor.backend)
        {
            return Err(RuntimeError::Dispatch(format!(
                "manifest backend disagrees with payload '{}'",
                payload.descriptor.name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_kernel::{
        BindingSlot, BufferRole, CpuBackend, DispatchGeometry, KernelCompileRequest,
        KernelDescriptor, KernelVariant,
    };

    fn cpu_artifact() -> KernelArtifact {
        CpuBackend
            .compile(&KernelCompileRequest {
                source: b"uop-test".to_vec(),
                descriptor: KernelDescriptor {
                    name: "uop_test".into(),
                    variant: KernelVariant::FP16GEMV,
                    backend: BackendKind::CPU,
                    source_digest: String::new(),
                    binary_digest: String::new(),
                    binding_signature: vec![BindingSlot {
                        index: 0,
                        role: BufferRole::Input,
                        data_type: prism_ecs_kernel::BindingDataType::Float16,
                    }],
                    dispatch_geometry: DispatchGeometry {
                        threads_per_threadgroup: [1, 1, 1],
                        threadgroups_per_grid: [1, 1, 1],
                        threads_per_grid: [1, 1, 1],
                    },
                },
                source_path: None,
            })
            .unwrap()
    }

    #[test]
    fn registry_reuses_backend_and_artifact_resources() {
        let registry = BackendExecutionRegistry::new();
        let artifact = cpu_artifact();
        let first = registry.register_artifact(artifact.clone()).unwrap();
        let second = registry.register_artifact(artifact).unwrap();
        assert_eq!(first.artifact_digest, second.artifact_digest);
        assert_eq!(registry.backend_count(), 1);
        assert_eq!(registry.artifact_count(), 1);
    }

    #[test]
    fn dispatch_validation_rejects_wrong_artifact_backend() {
        let registry = BackendExecutionRegistry::new();
        let artifact = cpu_artifact();
        let binding = registry.register_artifact(artifact).unwrap();
        let error = registry
            .validate_dispatch(
                "metal",
                &KernelDispatchSpec {
                    backend: binding.backend,
                    artifact_digest: binding.artifact_digest,
                    kernel_name: binding.kernel_name,
                    inputs: vec![],
                    bindings: vec![],
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("does not match"));
    }

    #[test]
    fn dispatcher_returns_actual_cpu_outcome_across_dispatcher_instances() {
        let registry = BackendExecutionRegistry::new();
        let artifact = cpu_artifact();
        let binding = KernelArtifactBinding::for_artifact(
            &artifact,
            vec![vec![0x00, 0x3c], 2.0f32.to_ne_bytes().to_vec()],
            vec![BindingSlot {
                index: 0,
                role: BufferRole::Input,
                data_type: prism_ecs_kernel::BindingDataType::Float16,
            }],
        )
        .unwrap();
        registry.register_artifact(artifact).unwrap();
        let first_dispatcher = KernelBackendDispatcher::new(registry.clone());
        let handle = first_dispatcher
            .start(&DispatchRequest {
                work_entity: 41,
                attempt: 1,
                plan_generation: 0,
                lease_token: "work-lease:41".into(),
                deadline_ms: u64::MAX,
                backend: "cpu".into(),
                config: binding.dispatch_config(&serde_json::json!({"provider": "cpu"})),
                input_path: String::new(),
                output_path: String::new(),
            })
            .unwrap();
        let second_dispatcher = KernelBackendDispatcher::new(registry);
        let status = second_dispatcher.poll(&handle).unwrap();
        match status {
            DispatchStatus::Completed(output) => {
                assert_eq!(f32::from_ne_bytes(output[..4].try_into().unwrap()), 2.0);
            }
            other => panic!("expected CPU output, got {other:?}"),
        }
    }
}
