Tribunus Compute — Linux Device Island Safety, Determinism, and Conformance Expansion

1. Phase Identity

Name: LINUX-DEVICE-ISLAND-SAFETY-AND-CONFORMANCE-0002

Purpose

This campaign turns the newly functional CPU Device Island backend into a trustworthy execution and conformance authority.

The campaign does not add full CUDA execution yet. It closes the remaining safety, lifecycle, arithmetic, determinism, and observability gaps in the Linux core so that CUDA, HIP, Level Zero, and Vulkan providers can later inherit a stable contract rather than replicate unsafe scaffolding.

The central completion condition is:

Every resource has a globally valid identity.
Every submission has an observable lifecycle.
Every CPU operation is memory-safe and deterministic.
Every invalid operation fails before mutating state.
Every conformance result can be replayed and explained.

Strategic Boundary

The CPU backend remains the reference implementation.

It is not a lightweight mock and it is not permitted to use unsafe shortcuts that would invalidate its role as the conformance oracle.

Future CUDA, HIP, Level Zero, and Vulkan providers must be judged against the CPU backend’s exact operation semantics, output bytes, error behavior, ownership transitions, receipt format, and deterministic hashes.

The CPU backend must therefore be stricter than later providers, not less strict.

Scope

This campaign includes:

resource registry hardening
submission lifecycle registry
buffer release and generation reuse
safe typed buffer access
overflow-safe reduction and hashing
full CPU conformance operation coverage
deterministic queue semantics
receipt persistence and inspection
inventory aggregation
external backend availability classification
TRCS consolidation correctness hardening
TRCS property and replay testing
Linux and macOS CI coverage

This campaign does not include:

real CUDA kernel launch
real HIP kernel launch
real Level Zero command list execution
real Vulkan compute dispatch
multi-GPU scheduling
peer-to-peer memory transfer
GPU-side TRCS semantics
production inference kernels

Workstream One: Resource Registry Completion

The current runtime resource model uses RuntimeResourceId, but the CPU backend still indexes buffers only by opaque_id.

That is insufficient because an opaque ID is not independently valid outside the full { backend, device, generation, opaque_id } identity.

Replace all raw registry keys with full typed resource IDs.

pub struct ResourceRegistryKey {
    pub backend: BackendKind,
    pub device: DeviceId,
    pub generation: u32,
    pub opaque_id: u64,
}

The CPU backend must maintain separate registries for:

buffers
queues
events
submissions
retired resource generations

No registry lookup may use opaque_id alone.

The registry must reject:

forged backend identity
wrong device identity
stale generation
released handle
cross-backend handle
cross-device handle
queue from another backend
buffer submitted through another device domain

Workstream Two: Monotonic IDs and Generation Reuse

Every resource allocation must receive a unique opaque ID within its backend-device domain.

Submission handles must not use hard-coded identifiers.

Queue handles must not use synthetic identifiers after construction.

The resource allocator must support this lifecycle:

allocate resource
  -> generation 1
release resource
  -> mark retired
reuse slot only when safe
  -> generation 2
stale generation 1 handle rejected

The initial implementation may never reuse slots and may keep monotonically increasing IDs. That is acceptable.

Generation reuse becomes mandatory only when memory/resource pressure requires slot recycling.

The immediate requirement is that no two live resources share the same complete runtime identity.

Workstream Three: Safe Typed Buffer Access

The CPU backend must remove all unsafe casts from Vec<u8> to typed slices.

A Vec<u8> allocation has byte alignment, not guaranteed u32 or u64 alignment. Reinterpreting it as &[u32] or &mut [u32] through raw pointers is undefined behavior on platforms that require aligned access.

Replace the current implementation with one of these permitted designs.

Preferred design:

pub enum CpuBufferStorage {
    Bytes(Vec<u8>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

Alternative design:

pub struct AlignedBytes {
    pub bytes: Vec<AlignedWord>,
}

The public AllocationRequest must include an explicit element-layout requirement.

pub enum ElementLayout {
    Bytes,
    U32,
    U64,
}

The CPU backend must reject a submission whose operation layout does not match the allocated buffer layout.

For example:

Fill u32:
  destination must be U32 layout
SumU32:
  source must be U32 layout
  destination must hold at least one U32
XorU64:
  source must be U64 layout
  destination must hold at least one U64

No unsafe pointer conversion is permitted in the CPU reference implementation.

Workstream Four: Allocation and Release API

The device backend trait must gain explicit release behavior.

fn release(
    &self,
    buffer: BufferHandle,
) -> Result<ReleaseReceipt, BackendError>;

The release operation must:

validate full resource identity
fail if submission references remain active
mark the buffer Released
retire storage
record release generation
emit receipt
reject all future access through the old handle

The initial CPU backend may release memory immediately after confirming no active submission references exist.

A buffer cannot be released while it is:

UploadPending
DeviceOwned by an active submission
ReadbackPending
referenced by queued work

The CPU backend may execute synchronously, but its lifecycle model must still enforce this rule.

Workstream Five: Submission Registry and Lifecycle

The backend must store submissions instead of returning a completed status unconditionally.

pub struct SubmissionRecord {
    pub handle: SubmissionHandle,
    pub queue: QueueHandle,
    pub operation: SubmissionKind,
    pub referenced_buffers: Vec<BufferHandle>,
    pub status: SubmissionStatus,
    pub submitted_at_unix_ms: u64,
    pub started_at_unix_ms: Option<u64>,
    pub completed_at_unix_ms: Option<u64>,
    pub error: Option<BackendErrorReceipt>,
    pub output_hash: Option<u64>,
}

The CPU backend may complete synchronously, but it must still transition through:

Pending
  -> Running
  -> Complete
Pending
  -> Running
  -> Failed

poll() must validate the full submission identity and return the persisted status.

synchronize() must validate the full submission identity and return either successful completion or the stored structured failure.

A caller must never be able to poll a submission belonging to another device or backend.

Workstream Six: Complete CPU Primitive Coverage

The CPU backend must implement all operations declared in the common submission contract.

Required operations:

Fill U32
Copy bytes
SumU32 reduction
XorU64 reduction
MinU32 reduction
MaxU32 reduction
DeterministicHash
ScanPreparation

ScanPreparation does not yet need a full parallel scan. It must validate the required source/destination shape, initialize deterministic metadata, and produce a canonical result contract that future GPU implementations can match.

DeterministicHash must use one repository-standard algorithm with stable byte order and documented seed behavior.

The hash must not use Rust’s randomized standard library hasher.

All operations must reject:

released buffers
mismatched element layout
cross-device buffers
cross-backend buffers
out-of-bounds element count
insufficient destination capacity
unsupported operation
invalid queue
integer overflow
invalid ownership transition

Workstream Seven: Arithmetic Safety

All arithmetic that affects semantic truth, output bytes, or receipt hashes must be checked.

CPU reductions must use explicit arithmetic policy.

pub enum ArithmeticPolicy {
    Checked,
    Wrapping,
    Saturating,
}

The default for conformance operations is Checked.

For SumU32, overflow must return an explicit arithmetic error rather than wrapping silently.

For TRCS support counts, arithmetic remains i64 and uses checked addition.

A grouped physical delta may be converted to i32 only after a checked range validation.

i64 accumulated diff
  -> ensure i32::MIN <= diff <= i32::MAX
  -> emit physical i32 row
  -> otherwise fail with ArithmeticOverflow

Saturation is prohibited for TRCS logical support accounting.

Workstream Eight: Deterministic Queue Semantics

The CPU queue model must define ordering explicitly.

For this campaign, each queue is FIFO.

Submissions on the same queue execute in submission order.

Submissions on different queues may execute independently, but the CPU backend may initially serialize them while preserving receipt correctness.

The contract must distinguish:

queue-local order
cross-queue independence
event-based dependency
host synchronize

Add event support to the core contract.

pub trait LinuxDeviceBackend {
    fn create_event(
        &self,
        device: &DeviceId,
    ) -> Result<EventHandle, BackendError>;
    fn record_event(
        &self,
        queue: &QueueHandle,
        event: &EventHandle,
    ) -> Result<(), BackendError>;
    fn wait_event(
        &self,
        queue: &QueueHandle,
        event: &EventHandle,
    ) -> Result<(), BackendError>;
}

The CPU implementation may use immediate completion semantics, but must validate event/backend/device identity and record the dependency in receipts.

Workstream Nine: Receipt Completeness

Every execution path must emit complete receipts.

pub struct DeviceSubmissionReceipt {
    pub submission_id: SubmissionHandle,
    pub backend: BackendKind,
    pub device_id: DeviceId,
    pub queue_id: QueueHandle,
    pub operation_kind: SubmissionKind,
    pub submitted_at_unix_ms: u64,
    pub started_at_unix_ms: Option<u64>,
    pub completed_at_unix_ms: Option<u64>,
    pub input_buffers: Vec<BufferHandle>,
    pub output_buffers: Vec<BufferHandle>,
    pub status: SubmissionStatus,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub output_hash: Option<ContentHash>,
    pub cpu_reference_hash: Option<ContentHash>,
    pub validation_mode: ValidationMode,
    pub error: Option<BackendErrorReceipt>,
}

A complete receipt is required for both success and failure.

Receipt contents must never include raw buffer contents, secrets, environment variables, or unrestricted host paths.

Workstream Ten: CPU Capability Probe

Replace "Generic CPU" placeholder inventory metadata with a bounded real host capability probe.

The CPU capability record should report:

architecture
logical CPU count
physical-core count when available
cache line size when available
supported vector features
host address width
memory total only when permission-safe
endianness

Do not depend on unstable host-specific strings for semantic identity.

The CPU DeviceStableKey must be stable enough for one host session and receipt grouping, but it must not expose a raw machine serial number.

Workstream Eleven: External Provider Availability Model

The external provider stubs must be validated through one uniform inventory path.

Each compiled provider must produce one of:

Available
RuntimeLibraryMissing
DriverMissing
UnsupportedHardware
PermissionDenied
FeatureNotCompiled
ProbeFailed

Operational APIs such as allocate() and submit() must only return NotReady after a provider has already been identified as present but not initialized.

They must not use NotReady as a replacement for capability reporting.

The inventory aggregator must collect all provider results without one provider error preventing CPU discovery.

Workstream Twelve: TRCS Consolidation Hardening

The existing staged support-table approach is correct in direction but incomplete.

The grouping contract must include the full logical relation identity.

pub struct ConsolidationKey {
    pub relation_id: RelationId,
    pub tuple: CompactTuple,
    pub frontier: RevisionFrontierId,
}

The deterministic order must be:

relation_id
  -> tuple columns lexicographically
  -> revision frontier
  -> provenance token

HashMap iteration must never influence emitted physical row order.

The sort comparator must include frontier after tuple ordering.

All support-table arithmetic must be checked.

All physical row differential conversions must be checked.

Mixed-relation batches must either be rejected by one-relation APIs or partitioned by the caller before consolidation.

A transaction failure must leave unchanged:

support table
trace runs
visible insertions
visible retractions
physical row output
receipt state
provenance reservation state

Workstream Thirteen: TRCS Property Testing

Add property-based tests using generated signed update streams.

The test oracle is replay equivalence.

For the same logical update history, varying batch partitioning, insertion order, and compaction timing must produce identical:

visible fact set
support table
canonical physical rows
trace determinism hash
compaction result
receipt summaries

Required property categories:

batch partition invariance
input permutation invariance
zero-diff cancellation
negative-support transaction rollback
frontier ordering preservation
bulk-load versus incremental replay equivalence
compaction versus no-compaction equivalence
mixed relation rejection
i64 to i32 overflow rejection
repeated retraction rejection
stable receipt hash

Workstream Fourteen: macOS Build Recovery

The campaign must repair the existing build-script regression.

On macOS Apple Silicon when the Metal backend is enabled:

xcrun metal failure:
  fail build
xcrun metallib failure:
  fail build
missing metallib output:
  fail build
successful output:
  emit TRIBUNUS_METALLIB

On Linux:

do not invoke xcrun
do not emit fake metallib environment variable
do not suppress unrelated build errors

Workstream Fifteen: CI Matrix

Required Linux checks:

cargo fmt --check
cargo check -p compute-core --no-default-features
cargo check -p compute-core --features linux-device-core,cpu-backend
cargo test -p compute-core --features linux-device-core,cpu-backend
cargo clippy -p compute-core --features linux-device-core,cpu-backend -- -D warnings
cargo test -p compute-core trcs

Required provider compile checks:

cargo check -p compute-core --features linux-device-core,cuda-backend
cargo check -p compute-core --features linux-device-core,hip-backend
cargo check -p compute-core --features linux-device-core,level-zero-backend
cargo check -p compute-core --features linux-device-core,vulkan-backend

Required macOS checks:

cargo check -p compute-core --features mlx-backend
cargo test -p compute-core --features mlx-backend

The CI matrix must include a negative Metal build fixture that confirms shader compilation failure fails the build.

The current PR must be rebased or otherwise reconciled with main before merge evaluation.

Required Tests

The campaign is not complete until these tests exist and pass:

resource_identity_is_unique_across_buffers_queues_events_and_submissions
resource_registry_rejects_opaque_id_only_lookup
stale_generation_is_rejected
cross_backend_handle_is_rejected
cross_device_handle_is_rejected
release_invalidates_buffer_handle
release_fails_with_active_submission_reference
fill_u32_is_memory_safe_and_deterministic
copy_is_memory_safe_and_deterministic
sum_u32_overflow_is_rejected
xor_u64_matches_canonical_output
min_u32_matches_canonical_output
max_u32_matches_canonical_output
deterministic_hash_matches_fixed_vector
scan_preparation_matches_contract
invalid_element_layout_is_rejected
unaligned_typed_access_is_not_constructible
submission_lifecycle_is_persisted
poll_returns_recorded_terminal_status
synchronize_returns_recorded_failure
fifo_queue_order_is_preserved
event_dependency_is_recorded
receipt_contains_complete_resource_identity
cpu_inventory_reports_real_architecture_metadata
unavailable_external_backends_do_not_block_cpu_inventory
consolidation_sorts_relation_tuple_frontier_deterministically
consolidation_rejects_i32_narrowing_overflow
consolidation_rejects_mixed_relation_batches
consolidation_rollback_preserves_original_support_table
trcs_replay_is_batch_partition_invariant
trcs_replay_is_input_order_invariant
compaction_is_semantics_preserving
macos_metal_compile_failure_is_fatal
linux_build_never_invokes_xcrun
