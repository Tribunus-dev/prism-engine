Tribunus Compute — Linux Device Island Foundation Specification

1. Phase Identity

Name: LINUX-DEVICE-ISLAND-FOUNDATION-0001

Primary repository: Tribunus-dev/prism-engine

Primary implementation location:

compute-core/src/linux/

Primary validation environment: Ubuntu Linux virtual machine.

Primary objective: establish a backend-neutral Linux accelerator substrate that can discover, qualify, allocate on, submit work to, synchronize with, and recover from CPU, CUDA, HIP, Level Zero, and Vulkan compute devices.

This phase does not yet make GPUs authoritative for TRCS semantic maintenance. The CPU reference runtime remains the semantic truth engine. Linux devices are introduced first as explicitly qualified physical execution islands.

2. Strategic Position

Tribunus must not develop separate CUDA, ROCm, Intel, and Vulkan inference stacks.

Instead, it needs one target-neutral execution contract:

ComputeImage or TRCS operator plan
  -> backend-neutral device-island contract
  -> capability-qualified backend lowering
  -> CUDA / HIP / Level Zero / Vulkan / CPU execution
  -> explicit receipt

The CPU remains responsible for canonical IDs, revision frontiers, support accounting, signed consolidation authority, provenance publication, transaction boundaries, fallback decisions, and evidence receipts.

Accelerators initially own only bounded data-parallel work whose outputs can be verified by the CPU reference backend.

The first eligible operations are device discovery, buffer allocation, host-device transfer, vector fill, vector copy, deterministic reduction, radix histogram preparation, and scan preparation.

No backend may claim semantic correctness merely because a kernel executed successfully.

3. Scope

This phase implements five execution providers.

CpuBackend
CudaBackend
HipBackend
LevelZeroBackend
VulkanComputeBackend

CUDA is the first fully executable external backend.

HIP is second and must support AMD execution through ROCm where a compatible runtime is installed.

Level Zero is the Intel low-level backend.

Vulkan compute is a portable probe and fallback backend. It is not the primary high-performance path for NVIDIA, AMD, or Intel.

The phase also implements a common device-probing layer, memory ownership state machine, queue and event abstraction, submission receipts, backend quarantine behavior, and shared microkernel conformance tests.

4. Non-Goals

This phase does not implement Metal.

This phase does not implement GPU-side differential fixpoint evaluation.

This phase does not implement TRCS sparse merge joins, full compaction offload, radix sorting, or production scan kernels.

This phase does not implement model inference kernels, quantized matmuls, paged attention, or ComputeImage lowering.

This phase does not implement multi-GPU sharding, peer-to-peer transfer, NCCL, RCCL, collective communication, or distributed inference.

This phase does not implement SYCL as the initial Intel path. SYCL may sit above the device-island contract later, but Level Zero is the low-level Intel provider for this phase.

This phase does not require any external accelerator to be present in the Ubuntu VM. CPU-only operation is a valid, tested result.

5. Core Invariants

Every backend is optional at build time and optional at runtime.

A missing CUDA, ROCm, Level Zero, Vulkan, or vendor driver library is not a build failure for CPU-only operation.

A backend feature enabled at compile time but unavailable at runtime must produce a structured “unavailable” capability result, not a panic.

No buffer crosses a backend boundary without an explicit ownership transition.

No device result is trusted until it passes deterministic readback and CPU comparison during conformance mode.

No backend failure may corrupt CPU-authoritative TRCS state.

No backend may write directly into semantic support tables, provenance stores, revision state, or active trace metadata during this phase.

No driver capability is inferred from vendor name alone. Every capability must be probed.

6. Module Layout

compute-core/src/linux/
  mod.rs
  backend.rs
  capability.rs
  device.rs
  topology.rs
  memory.rs
  queue.rs
  event.rs
  submission.rs
  receipt.rs
  probe.rs
  fallback.rs
  errors.rs
  conformance.rs
  cpu/
    mod.rs
    device.rs
    memory.rs
    queue.rs
    kernels.rs
  cuda/
    mod.rs
    ffi.rs
    probe.rs
    memory.rs
    queue.rs
    submission.rs
    kernels.rs
  hip/
    mod.rs
    ffi.rs
    probe.rs
    memory.rs
    queue.rs
    submission.rs
    kernels.rs
  level_zero/
    mod.rs
    ffi.rs
    probe.rs
    memory.rs
    queue.rs
    submission.rs
    kernels.rs
  vulkan/
    mod.rs
    probe.rs
    memory.rs
    queue.rs
    submission.rs
    kernels.rs

The backend-independent modules must not import CUDA, HIP, Level Zero, Vulkan, ROCm, or vendor-specific FFI symbols.

Vendor implementations may depend on backend-neutral contracts, but the contract layer may not depend on a vendor implementation.

7. Cargo Feature Model

The default Linux build must remain CPU-only and dependency-light.

default = ["cpu-backend"]
cpu-backend = []
cuda-backend = ["dep:cudarc-or-equivalent"]
hip-backend = ["dep:hip-runtime-binding"]
level-zero-backend = ["dep:level-zero-binding"]
vulkan-backend = ["dep:ash-or-equivalent"]
linux-device-islands = [
  "cpu-backend",
  "cuda-backend",
  "hip-backend",
  "level-zero-backend",
  "vulkan-backend"
]

Feature enablement means “compile backend support.” It does not mean “require that backend to exist on this host.”

The runtime probe determines whether the backend is available.

8. Backend Kind and Device Identity

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Hip,
    LevelZero,
    Vulkan,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VendorKind {
    Cpu,
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceId {
    pub backend: BackendKind,
    pub ordinal: u32,
    pub stable_key: DeviceStableKey,
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceStableKey {
    pub vendor: VendorKind,
    pub pci_domain: Option<u16>,
    pub pci_bus: Option<u8>,
    pub pci_device: Option<u8>,
    pub uuid_or_luid: Option<[u8; 16]>,
    pub fallback_fingerprint: u64,
}

The stable key is used for device continuity across process restarts and receipt correlation. It is not a security identity.

9. Capability Contract

#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub backend: BackendKind,
    pub vendor: VendorKind,
    pub device_name: String,
    pub driver_version: Option<String>,
    pub architecture: Option<String>,
    pub device_memory_bytes: u64,
    pub host_visible_memory: bool,
    pub unified_addressing: bool,
    pub managed_memory: bool,
    pub peer_access: bool,
    pub async_copy: bool,
    pub events: bool,
    pub command_graphs: bool,
    pub cooperative_launch: bool,
    pub fp16: bool,
    pub bf16: bool,
    pub int8: bool,
    pub int4: bool,
    pub subgroup_widths: Vec<u32>,
    pub max_workgroup_size: u32,
    pub max_shared_memory_bytes: u64,
    pub max_allocation_bytes: u64,
    pub supports_timestamps: bool,
    pub supports_profiling: bool,
    pub supports_external_memory: bool,
    pub availability: BackendAvailability,
}
#[derive(Debug, Clone)]
pub enum BackendAvailability {
    Available,
    DriverMissing,
    RuntimeLibraryMissing,
    UnsupportedHardware,
    PermissionDenied,
    FeatureNotCompiled,
    ProbeFailed {
        reason: String,
    },
}

A capability report must include both positive and negative information. “No BF16” is a capability result, not a probe failure.

10. Device Discovery API

pub trait LinuxDeviceBackend: Send + Sync {
    fn backend_kind(&self) -> BackendKind;
    fn enumerate_devices(
        &self,
    ) -> Result<Vec<DeviceDescriptor>, BackendError>;
    fn probe_capabilities(
        &self,
        device: &DeviceId,
    ) -> Result<DeviceCapabilities, BackendError>;
    fn create_queue(
        &self,
        device: &DeviceId,
        class: QueueClass,
    ) -> Result<QueueHandle, BackendError>;
    fn allocate(
        &self,
        device: &DeviceId,
        request: AllocationRequest,
    ) -> Result<DeviceBuffer, BackendError>;
    fn submit(
        &self,
        queue: &QueueHandle,
        submission: Submission,
    ) -> Result<SubmissionHandle, BackendError>;
    fn poll(
        &self,
        submission: &SubmissionHandle,
    ) -> Result<SubmissionStatus, BackendError>;
    fn synchronize(
        &self,
        submission: &SubmissionHandle,
    ) -> Result<(), BackendError>;
}

Device discovery must be best-effort across all compiled backends.

One failing backend must not suppress valid devices from other backends.

The global device probe result is:

pub struct LinuxDeviceInventory {
    pub generated_at_unix_ms: u64,
    pub devices: Vec<DeviceDescriptor>,
    pub unavailable_backends: Vec<BackendUnavailableReceipt>,
}

11. Queue Model

The initial queue classes are semantic scheduling labels. They must not imply unsupported hardware priority guarantees.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueClass {
    ForegroundCompute,
    BackgroundAnalysis,
    Transfer,
    Conformance,
}

The scheduler may map queue classes to separate CUDA streams, HIP streams, Level Zero command queues, Vulkan queues, or CPU work executors where available.

The runtime must not claim that this creates OS-level preemption.

Queue ordering is guaranteed only according to the actual backend contract.

12. Event Model

pub struct EventHandle {
    pub backend: BackendKind,
    pub opaque_id: u64,
}
pub enum EventStatus {
    Pending,
    Complete,
    Failed(BackendError),
}

Events are used for submission completion, transfer ordering, profiling timestamps where available, and conformance synchronization.

Every backend must support a fallback implementation even if native events are unavailable. The fallback may use queue synchronization but must report reduced capability.

13. Buffer Ownership Model

Every buffer is governed by an explicit ownership state machine.

pub enum BufferOwnership {
    HostOwned,
    UploadPending,
    DeviceOwned,
    ReadbackPending,
    HostReadable,
    Released,
}
pub struct DeviceBuffer {
    pub buffer_id: BufferId,
    pub backend: BackendKind,
    pub device_id: DeviceId,
    pub size_bytes: u64,
    pub alignment_bytes: u64,
    pub memory_kind: MemoryKind,
    pub ownership: BufferOwnership,
    pub generation: u64,
}
pub enum MemoryKind {
    HostPageable,
    HostPinned,
    DeviceLocal,
    HostVisibleDevice,
    Unified,
}

The permitted ownership transitions are:

HostOwned
  -> UploadPending
  -> DeviceOwned
  -> ReadbackPending
  -> HostReadable
  -> Released

A backend may expose unified or host-visible memory, but TRCS must still record an explicit ownership transition.

Unified addressing does not eliminate mutation discipline.

14. Allocation Requests

pub struct AllocationRequest {
    pub size_bytes: u64,
    pub alignment_bytes: u64,
    pub memory_preference: MemoryPreference,
    pub usage: BufferUsage,
    pub zero_initialize: bool,
}
pub enum MemoryPreference {
    PreferDeviceLocal,
    PreferHostVisible,
    RequireHostVisible,
    PreferUnified,
}
pub enum BufferUsage {
    TransferSource,
    TransferDestination,
    KernelReadOnly,
    KernelReadWrite,
    Readback,
    Scratch,
}

The backend must reject impossible allocations with structured errors.

It must not silently downgrade RequireHostVisible into device-local memory.

15. Submission Model

Phase one of Linux device execution supports only explicit conformance workloads.

pub enum Submission {
    Fill {
        destination: BufferId,
        value: u32,
        element_count: u64,
    },
    Copy {
        source: BufferId,
        destination: BufferId,
        size_bytes: u64,
    },
    Reduction {
        source: BufferId,
        destination: BufferId,
        element_count: u64,
        operation: ReductionOperation,
    },
    DeterministicHash {
        source: BufferId,
        destination: BufferId,
        element_count: u64,
        seed: u64,
    },
    ScanPreparation {
        source: BufferId,
        destination: BufferId,
        element_count: u64,
    },
}
pub enum ReductionOperation {
    SumU32,
    XorU64,
    MinU32,
    MaxU32,
}

The initial backend implementation must not accept arbitrary kernel source strings, arbitrary PTX, arbitrary SPIR-V, or unrestricted device-side code generation.

That comes later after the ownership, receipts, and recovery model are proven.

16. Submission Receipts

Every submission produces a receipt.

pub struct DeviceSubmissionReceipt {
    pub submission_id: SubmissionId,
    pub backend: BackendKind,
    pub device_id: DeviceId,
    pub queue_class: QueueClass,
    pub submitted_at_unix_ms: u64,
    pub completed_at_unix_ms: Option<u64>,
    pub operation_kind: SubmissionKind,
    pub input_buffer_ids: Vec<BufferId>,
    pub output_buffer_ids: Vec<BufferId>,
    pub status: SubmissionStatus,
    pub bytes_transferred: u64,
    pub host_wait_ns: Option<u64>,
    pub device_elapsed_ns: Option<u64>,
    pub validation_mode: ValidationMode,
    pub output_hash: Option<u64>,
    pub error: Option<BackendErrorReceipt>,
}
pub enum ValidationMode {
    NotValidated,
    DeviceOnly,
    CpuReferenceCompared,
}

A receipt with DeviceOnly may not unlock semantic TRCS offload.

Only CpuReferenceCompared establishes conformance.

17. CPU Reference Provider

The CPU backend is mandatory and must implement every Phase 1 submission.

The CPU provider is not a fallback stub. It is the conformance oracle.

For every test input accepted by a device backend, the CPU backend must produce the canonical expected output and deterministic output hash.

CPU result
  -> canonical output
  -> deterministic hash
  -> device readback comparison
  -> conformance receipt

The same test vectors must execute on CPU-only Ubuntu installations.

18. CUDA Provider

CUDA is the first complete Linux accelerator provider.

The initial CUDA provider must implement:

runtime library discovery
device enumeration
capability probe
device UUID or stable identification where exposed
stream creation
event creation
device-local allocation
host-pinned allocation where available
host-to-device copy
device-to-host copy
fill
copy
reduction
deterministic hash
synchronization
structured failure receipts

CUDA graph support may be probed and reported, but graph execution is not required in this phase.

Peer access may be probed and reported, but multi-device execution is not required.

19. HIP Provider

The HIP provider must expose the same logical contract as CUDA.

The initial HIP provider must implement:

runtime library discovery
AMD device enumeration
capability probe
stream creation
event creation
device-local allocation
host-to-device copy
device-to-host copy
fill
copy
reduction
deterministic hash
synchronization
structured failure receipts

The HIP implementation may share backend-neutral conformance vectors with CUDA, but it must not directly assume CUDA-specific device attributes or runtime semantics.

The conformance suite must identify the precise backend and device architecture for every failure.

20. Level Zero Provider

The Level Zero provider is the initial Intel execution path.

It must implement:

loader discovery
driver enumeration
device enumeration
device property probe
command queue creation
command list creation
event pool and event creation
device allocation
host allocation
shared allocation where supported
memory copy
fill
reduction or deterministic hash
synchronization
structured failure receipts

The Level Zero provider may initially return UnsupportedOperation for operations that have not yet been lowered, but it must never pretend a device supports them.

Intel GPU availability must be tested independently from the presence of an Intel CPU.

21. Vulkan Provider

The Vulkan compute provider is a portability layer and probe fallback.

It must implement:

instance creation
physical device enumeration
compute queue selection
device capability reporting
buffer allocation
host-visible transfer buffers
command buffer recording
buffer copy
fill
deterministic hash or reduction
fence synchronization
structured failure receipts

Vulkan may be used where CUDA, HIP, and Level Zero are unavailable.

It must not become the preferred backend when a vendor-native provider is both available and qualified for the requested operation.

22. Backend Selection Policy

The backend selector must be deterministic and policy-driven.

pub struct BackendSelectionPolicy {
    pub preferred_backends: Vec<BackendKind>,
    pub require_cpu_validation: bool,
    pub allow_vulkan_fallback: bool,
    pub minimum_device_rows: u64,
    pub maximum_transfer_overhead_ratio: f64,
}

The default preference order is:

CUDA
HIP
LevelZero
Vulkan
CPU

This is a priority policy, not a semantic guarantee.

A backend is selected only when:

the required primitive is supported
allocation succeeds within memory policy
estimated device work exceeds offload threshold
transfer cost is acceptable
the device is not quarantined
the operation has passed conformance for this device class

The CPU is selected whenever any condition fails.

23. Device Quarantine and Recovery

A device backend may fail without invalidating the CPU runtime.

pub enum DeviceHealth {
    Healthy,
    Suspect,
    Quarantined,
    RecoveryProbe,
    Disabled,
}

A backend enters Suspect after a failed submission.

It enters Quarantined after repeated failures, timeout, invalid readback, output mismatch, driver reset, allocation corruption, or device-lost condition.

While quarantined, the scheduler must route work to CPU or another qualified backend.

device submission mismatch
  -> mark receipt failed
  -> discard device output
  -> CPU recompute
  -> increment device fault counter
  -> quarantine after policy threshold

A recovery probe must use only small conformance workloads.

No device returns to normal scheduling until it passes the probe.

24. TRCS Offload Boundary

During this phase, TRCS may use devices only for bounded preparatory operations.

Allowed:

bulk relation buffer upload
physical row copy
radix histogram preparation
scan preparation
checksum/hash verification
compaction candidate sizing
sorted-run validation

Not allowed:

support table mutation
signed consolidation authority
revision frontier ordering authority
visibility transition authority
provenance publication
assertion validation
negation stratification
semantic rule evaluation

The CPU remains authoritative for all semantic transitions.

This boundary is non-negotiable until CPU/device equivalence is proven for each future operator class.

25. ComputeImage Offload Boundary

During this phase, ComputeImage may use the Linux substrate for device discovery and low-level conformance only.

It may not yet route production quantized projections, attention, or fused kernels through the generic device-island interface.

The only allowed ComputeImage use is backend qualification:

probe available accelerators
  -> verify transfer and execution conformance
  -> record capability receipt
  -> expose eligible devices to future lowering planner

26. Linux Capability Evidence

The Ubuntu agent must emit a machine-readable capability report.

{
  "schema_version": 1,
  "host": {
    "os": "ubuntu",
    "kernel": "string",
    "arch": "x86_64"
  },
  "backends": [
    {
      "backend": "cpu",
      "availability": "available",
      "devices": []
    },
    {
      "backend": "cuda",
      "availability": "runtime_library_missing",
      "devices": []
    }
  ],
  "generated_at_unix_ms": 0
}

The report must be safe to attach to CI artifacts.

It must not include secrets, host usernames, full filesystem paths, environment variable values, or unrelated machine inventory.

27. Test Matrix

The Ubuntu agent must validate all CPU-only tests on every run.

cargo fmt --check
cargo check -p compute-core --no-default-features
cargo test -p compute-core --no-default-features
cargo test -p compute-core linux
cargo clippy -p compute-core --no-default-features -- -D warnings

Feature checks must compile independently:

cargo check -p compute-core --features cuda-backend
cargo check -p compute-core --features hip-backend
cargo check -p compute-core --features level-zero-backend
cargo check -p compute-core --features vulkan-backend

Runtime tests must degrade cleanly when drivers are absent.

CPU-only host:
  CPU backend available.
  External providers unavailable with structured receipts.
  No panic.
  No linker failure.
  No device probe timeout.
CUDA host:
  CUDA conformance vector suite passes.
  CPU/device hashes match.
AMD ROCm host:
  HIP conformance vector suite passes.
  CPU/device hashes match.
Intel Level Zero host:
  Level Zero conformance vector suite passes.
  CPU/device hashes match.
Vulkan-only host:
  Vulkan copy/fill/hash suite passes.
  CPU/device hashes match.

28. Required Conformance Vectors

The first deterministic test vectors are:

fill 1,024 u32 elements with fixed value
copy 1 MiB patterned buffer
sum reduction over fixed pseudo-random u32 input
xor reduction over fixed pseudo-random u64 input
deterministic hash over fixed buffer
scan preparation shape validation
zero-length buffer rejection or no-op behavior
unaligned allocation rejection or explicit alignment handling
oversized allocation failure receipt
repeated allocation-release lifecycle
parallel queue submission ordering test
device error propagation test

Every vector must record the canonical CPU hash.

A device provider passes only when readback bytes or canonical output hashes match exactly.

29. Performance Measurements

This phase records performance but does not optimize aggressively.

The required measurements are:

device discovery duration
queue creation duration
allocation latency
host-to-device bandwidth
device-to-host bandwidth
fill throughput
copy throughput
reduction throughput
hash throughput
synchronization latency
CPU comparison cost

The measurements inform later offload thresholds.

They may not be used to claim production inference performance.

30. Delivery Sequence

The first implementation slice introduces backend-neutral contracts, typed errors, CPU provider, capability report serialization, and CPU-only tests.

The second slice adds dynamic backend discovery and unavailable-backend receipts for CUDA, HIP, Level Zero, and Vulkan without requiring any driver in the Ubuntu VM.

The third slice adds CUDA executable conformance on a CUDA-capable Linux runner.

The fourth slice adds HIP executable conformance on an AMD ROCm runner.

The fifth slice adds Level Zero executable conformance on an Intel GPU runner.

The sixth slice adds Vulkan fallback conformance.

The seventh slice integrates backend qualification into TRCS and ComputeImage capability planning without moving semantic authority off CPU.

31. Definition of Done

This phase is complete when Ubuntu can run a CPU-only Tribunus build that discovers all compiled backend providers safely, emits a structured device inventory, and executes the complete CPU conformance suite.

It is complete for each external backend only when that backend can enumerate a real device, allocate memory, transfer deterministic buffers, execute the approved microkernels, synchronize, read back results, and match the CPU reference output exactly.

At completion, Tribunus has one Linux device-island abstraction rather than disconnected vendor experiments. CUDA, HIP, Level Zero, Vulkan, and CPU are interchangeable physical providers beneath one ownership model, one receipt format, one fallback policy, and one conformance protocol.
