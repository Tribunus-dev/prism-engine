Tribunus Relational Compiler Substrate — Production Specification v1

1. Purpose

Tribunus Relational Compiler Substrate, TRCS, is the live, incremental, explainable analysis system beneath the ComputeImage compiler and collaborative workspace.

It converts mutable PhaseIR into canonical sparse relations, maintains compiler facts across human and agent edits, evaluates recursive analyses through signed differential updates, produces proof DAGs for every material optimization decision, and feeds exact legality, alias, effect, residency, fusion, and placement facts into the ComputeImage scheduler.

The portable core is the canonical semantic program, relation schema, proof model, and execution receipts. MLX, Metal, ROCm, CUDA, SYCL, and Level Zero are lowering and execution providers beneath that core rather than sources of semantic truth.

TRCS is not an embedding-based “AI compiler.” Learned models may rank valid physical plans. They may not establish semantic facts such as NoAlias, MustAlias, CanFuse, non-escape, or backend legality.

mutable source + agent revisions
  -> incremental PhaseIR
  -> canonical signed EDB deltas
  -> differential semantic maintenance
  -> sparse relational execution
  -> proof DAGs and receipts
  -> ComputeImage legality and scheduling

2. Core Principles

Every device-visible semantic object is an interned integer identity.

Every derived update carries a logical signed weight and a revision frontier.

Every physical output row is allocated positively, including retractions.

Every precision loss, fallback, imported-summary use, assertion dependency, capacity event, and GPU fault is recorded.

Every optimization enabled by an assertion or speculative assumption remains conditional until its authority, shape binding, trust class, and artifact dependencies validate.

Every lower stratum referenced negatively must converge before a negation-dependent stratum evaluates.

Every arrangement is log-structured. Large established data is not rewritten for tiny edits.

Every GPU task runs under bounded admission, bounded memory, bounded command-buffer windows, and host-side fault observation.

3. Runtime Components

PhaseIR Canonicalizer
Federated Identity Resolver
Revision DAG Coordinator
Semantic Planner
Negation Stratifier
Relation Arena
LSM Trace Manager
Differential Execution Planner
Precision Governor
Assertion Authority Service
Speculative Artifact Manager
Metal Relational Backend
CPU Reference Backend
ANE Planner Provider
GPU Lane Arbiter
Evidence Recorder
Provenance Spill Manager
Submission Watchdog
Cockpit Proof API

4. Identity Model

TRCS uses two identity layers.

The first is local, compact, and device-visible.

EntityId {
  slot: u32,
  generation: u16,
  kind: u8,
  reserved: u8,
}

The GPU sees only slot values in hot relation columns. Generation validation occurs at host boundaries, persistence boundaries, provenance boundaries, imported-summary boundaries, and federated synchronization boundaries.

The second is a portable federated identity claim.

CanonicalIdentityKey {
  workspace_namespace: Hash,
  module_content_root: Hash,
  semantic_path: CanonicalPath,
  entity_kind: EntityKind,
  normalized_structure_hash: Hash,
  binder_or_scope_hash: Hash,
  source_anchor: Optional<SourceAnchor>,
}

A federated key is not itself a runtime entity ID. It is a claim that allows the central workspace to resolve a remote entity into its own local generational identity.

This avoids the failure mode where two isolated agents use incompatible local counters, while also avoiding the instability and collision hazards of using raw structural hashes as permanent IDs.

5. Revision DAG and Concurrent Agents

Revisions form a merge DAG rather than a linear global counter.

WorkspaceRevision {
  revision_id: RevisionId,
  parent_frontier: RevisionFrontierId,
  parent_revisions: SmallVec<RevisionId>,
  changed_modules: ModuleSet,
  semantic_delta_hash: Hash,
  author_kind: Human | Agent | Merge | System,
  author_id: AuthorId,
  work_item_id: Optional<WorkItemId>,
  ownership_lease: Optional<LeaseId>,
  created_at: Timestamp,
}

The device-visible representation of logical time is a fixed-width RevisionFrontierId. The host maintains the underlying partial-order DAG.

Two subagent revisions may share a synthetic merge frontier only if their changed semantic regions are disjoint. They must not be merged concurrently when they overlap exported summaries, type declarations, module graph edges, generated-code ownership, assertion scopes, or shared interface contracts.

MergeFrontier {
  parent_frontiers: Set<RevisionFrontierId>,
  member_revisions: Set<RevisionId>,
  changed_modules: ModuleSet,
  merge_policy: DisjointSemanticMerge,
}

6. Federated Canonicalization Protocol

Remote agents never transmit raw local relation arenas as authoritative central state.

They transmit portable identity claims, signed fact deltas, assertion updates, provenance digests, and a local-to-exported identity map.

FederatedPhaseIRDelta {
  producer_session_id: SessionId,
  base_workspace_frontier: RevisionFrontierId,
  module_identity: ModuleIdentity,
  entity_claims: Vec<CanonicalIdentityKey>,
  local_to_exported_map: Map<LocalEntityId, CanonicalIdentityKey>,
  edb_retractions: Vec<FederatedFactDelta>,
  edb_insertions: Vec<FederatedFactDelta>,
  assertion_updates: Vec<FederatedAssertionUpdate>,
  provenance_digest: Hash,
}

The central resolver produces:

FederatedIdentityResolution {
  producer_session_id: SessionId,
  exported_key_to_central_id: Map<CanonicalIdentityKey, EntityId>,
  status: Matched | Created | Conflict | RequiresRecanonicalization,
}

Remote recent batches may be reused only after their tuple columns, frontier IDs, provenance references, and assertion references have been remapped into central IDs. They are never directly spliced into a central trace using foreign local IDs.

This is consistent with differential arrangements being maintained indexed update traces rather than universally portable storage blobs. Differential Dataflow models traces as sequences of batches that can be merged or compacted for efficiency.

7. Logical Fact Model

Every logical update is represented as:

WeightedFact {
  fact_id: FactId,
  relation_id: RelationId,
  tuple: CompactTuple,
  revision_frontier_id: RevisionFrontierId,
  diff: i32,
}

A fact is visible when its accumulated weight is positive.

A fact is retracted when its accumulated weight reaches zero.

A negative accumulated support count is an integrity fault.

Logical signed weight is not physical allocation count.

A retraction with diff = -1 still produces one physical row in a delta batch. The physical arena never allocates negative memory.

8. Physical Delta Model

PhysicalDeltaRow {
  tuple_columns: [u32; arity],
  diff: i32,
  revision_frontier_id: u32,
  provenance_token: u64,
}
EmissionCount {
  tile_id: u32,
  emitted_rows: u32,
}

The processing order is fixed:

count physical emitted rows
  -> exclusive scan
  -> write signed rows
  -> arrange or sort
  -> consolidate equal keys
  -> sum signed diffs
  -> update support counts
  -> emit visible insertions and visible retractions

The count stage counts physical emitted rows only. It never sums diff.

The consolidation stage is the only place where signed weights may cancel each other. If equivalent positive and negative updates net to zero, the physical rows are removed only after consolidation.

9. Core Relation Schema

The initial logical schema includes:

Operation(op_id, opcode_id, block_id, result_begin, result_count, flags)
Block(block_id, function_id, loop_depth, dom_in, dom_out, flags)
CFGEdge(src_block_id, dst_block_id)
Value(value_id, type_id, defining_op_id, flags)
Def(value_id, op_id)
Use(value_id, op_id)
Call(call_op_id, callsite_id, callee_id, flags)
Allocation(object_id, allocation_site_id, object_kind, address_space, owner_function_id)
PointerBase(pointer_value_id, object_id)
PointerRegion(pointer_value_id, region_id)
Copy(dst_value_id, src_value_id)
Load(dst_value_id, address_value_id)
Store(address_value_id, src_value_id)
GEP(result_value_id, base_value_id, offset_class_id)
Escape(value_id, escape_kind)
PointsTo(pointer_value_id, object_id)
PointsToRegion(pointer_value_id, region_id)
MemoryContents(region_id, object_id)
MemoryRead(access_op_id, region_id)
MemoryWrite(access_op_id, region_id)
Reachable(block_id)
LiveIn(block_id, value_id)
LiveOut(block_id, value_id)
MayAlias(access_a_id, access_b_id)
MustAlias(access_a_id, access_b_id)
Effect(owner_id, object_or_region_id, effect_flags)
ContextMap(callsite_id, caller_context_id, callee_context_id)
WidenedContext(original_context_id, widened_context_id, reason_code)
PrecisionLoss(entity_id, precision_kind, reason_code)

10. Log-Structured Arrangement Model

Full(R) is a logical relation, not necessarily one contiguous sorted allocation.

Full(R) = BaseArrangement(R) ⊕ RecentArrangement(R)[0..n]

BaseArrangement is large, consolidated, sorted, and rarely rewritten.

Each RecentArrangement is immutable after sealing, internally sorted, locally consolidated, tagged with a frontier range, and small enough for efficient incremental maintenance.

ArrangementRun {
  run_id: RunId,
  relation_id: RelationId,
  key_order: KeyOrder,
  frontier_min: RevisionFrontierId,
  frontier_max: RevisionFrontierId,
  row_count: u64,
  positive_rows: u64,
  negative_rows: u64,
  key_min: Key,
  key_max: Key,
  dead_diff_density: f32,
  storage_class: Base | Recent | Compacting | Retired,
}

A join probes the base plus relevant recent runs using a batch directory and key-range metadata.

Compaction runs asynchronously and copy-on-write. It folds selected recent runs into a new base or intermediate run only when justified by read amplification, recent-run count, dead-diff density, memory pressure, or observed query cost.

TraceCompactionPolicy {
  max_recent_runs: u32,
  max_read_amplification: f32,
  max_dead_diff_density: f32,
  base_rebuild_threshold_rows: u64,
  background_gpu_credit_limit: CreditBudget,
}

This follows the trace-and-batch model used by Differential Dataflow arrangements, which retain a sequence of indexed batches and merge them to control lookup cost.

11. Cold-Start Bulk-Load Fast Path

Differential maintenance is not used blindly when loading an empty workspace.

When a relation has no active base and the initial delta is effectively the whole relation, TRCS enters bulk-load mode.

BulkLoadEligibility {
  full_relation_empty: bool,
  delta_to_full_ratio: f32,
  input_fact_count: u64,
  estimated_incremental_overhead: Cost,
}

The default trigger is:

Full(R) empty
and initial EDB fact set is authoritative
and bulk-load plan cost < differential bootstrap plan cost

The bulk-load pipeline is:

parse workspace
  -> canonicalize full PhaseIR snapshot
  -> intern IDs
  -> emit dense EDB columns directly
  -> CPU or GPU radix sort by required arrangements
  -> deduplicate and consolidate once
  -> build BaseArrangement buffers
  -> seal base
  -> enable differential recent batches

This bypasses per-epoch delta bookkeeping during initial workspace hydration. It does not bypass semantic validation, signed consolidation, index construction, provenance initialization, or receipt generation.

The output is a sealed initial base trace. Only subsequent revisions enter the ordinary signed-delta path.

12. Incremental EDB Protocol

Every source or agent change emits signed updates:

FactDelta {
  relation_id: RelationId,
  tuple: CompactTuple,
  diff: i32,
  revision_frontier_id: RevisionFrontierId,
  source_revision: RevisionId,
  invalidation_scope: Function | Module | Summary | Workspace,
}

The controller computes the smallest semantically closed impact cone.

changed function
  -> local CFG and SSA facts
  -> local liveness
  -> local points-to and region deltas
  -> local effects
  -> changed exported summary, if any
  -> only dependent caller summaries
  -> affected legality and placement facts

TRCS uses three update modes.

Local nonrecursive analysis:
  direct signed propagation.
Bounded recursive SCC:
  differential maintenance.
Large edit, summary fanout, memory pressure, capacity event, or trace degradation:
  recompute affected SCC from sealed EDB.

13. Recursive Differential Evaluation

For each recursive relation:

Full(R): accumulated visible support state
Delta(R): signed visible changes entering the epoch
New(R): signed candidate updates generated during the epoch

The execution protocol is:

select active rules
  -> execute over signed deltas
  -> materialize physical rows
  -> consolidate
  -> update supports
  -> emit visibility transitions
  -> continue until all SCC deltas are empty

A retraction is never processed as an imperative “delete descendants” traversal. It is propagated as a negative differential update, preserving a fact while any remaining derivation supports it.

14. Negation and Strict Stratification

Rules involving absence are separated from rules deriving the negated relation.

CanFuse(a, b) :-
  CandidateFuse(a, b),
  NOT MayAlias(a, b),
  NOT HasBlockingEffect(a, b)

CanFuse may not execute in the same recursive SCC as MayAlias or HasBlockingEffect.

The semantic planner builds a signed dependency graph. Positive edges may remain within recursive SCCs. Negative edges impose stratum boundaries. Negative cycles are rejected.

StratificationBarrierReceipt {
  lower_stratum_id: StratumId,
  upper_stratum_id: StratumId,
  negative_dependency_relation: RelationId,
  sealed_frontier: RevisionFrontierId,
  lower_stratum_converged: bool,
  consolidation_hash: Hash,
  reopened_due_to_retraction: bool,
}

A retraction or insertion in a lower stratum invalidates dependent negation strata from that frontier onward. The upper stratum cannot be patched inside the same ICB iteration window before the lower stratum reconverges.

15. Pointer and Region Semantics

Pointer arithmetic is modeled symbolically:

ExactByteOffset(n)
ConstantRange(lower, upper, stride)
FieldPath(type_id, field_path_id)
ArraySlice(base_region, lower, upper, stride)
AffineIndex(induction_var_id, scale, bias, bounds_id)
UnionRegion(region_a, region_b)
UnknownWithinObject
UnknownExternal

The two-tier memory model is:

PointsTo(pointer, abstract_object)
PointsToRegion(pointer, memory_region)
MemoryContents(memory_region, abstract_object)
SummaryRegion(abstract_object, memory_region)

Unknown arithmetic widens to UnknownWithinObject. Foreign or opaque behavior widens to UnknownExternal.

No analysis may enumerate a large dynamic array element-by-element unless the allocation is statically bounded below a specific small threshold.

16. Alias Contract

NoAlias
MustAlias
MayAlias
UnknownAlias

NoAlias requires proven object and region disjointness.

MustAlias requires canonical pointer identity and compatible region identity.

MayAlias indicates possible overlap.

UnknownAlias indicates opaque behavior, unsupported operations, external memory, or policy-forced widening.

UnknownAlias is at least as conservative as MayAlias.

17. Context Sensitivity

Context sensitivity is bounded and receipt-producing.

ContextBudget {
  max_contexts_per_function: u32,
  max_contexts_per_scc: u32,
  max_total_contexts: u32,
  max_expansion_per_epoch: u32,
  widening_policy: DeterministicPolicyId,
}

Supported initial modes are context-insensitive, call-site sensitive depth one or two, receiver-sensitive depth one, recursion summaries, and widened summaries.

The GPU receives compact ContextId values only.

18. Semantic Assertions and Agent Authority

Assertions are privileged EDB facts, not direct final compiler conclusions.

An agent cannot directly inject NoAlias, MustAlias, CanFuse, or “this transformation is legal.”

An agent may submit a scoped semantic precondition, such as a disjoint-region claim, non-escape claim, ownership claim, or external-effect claim.

SemanticAssertion {
  assertion_id: AssertionId,
  assertion_kind: AssertionKind,
  subject_a: EntityId,
  subject_b_or_scope: EntityId,
  asserted_property: Property,
  owner_agent_id: AgentId,
  lease_id: LeaseId,
  issued_frontier: RevisionFrontierId,
  expires_frontier: Optional<RevisionFrontierId>,
  validation_mode: Verified | Checked | Assumed,
  trust_class: TrustClass,
}
AssertionBinding {
  assertion_id: AssertionId,
  phaseir_shape_hash: Hash,
  module_id: ModuleId,
  function_id: FunctionId,
  operation_scope: OperationScope,
  region_scope: RegionScope,
}

The Precision Governor applies five activation gates.

Authority:
  agent holds a valid ownership lease for the full semantic scope.
Shape binding:
  referenced PhaseIR, values, objects, regions, and call edges still match.
Scope minimization:
  claim is no broader than necessary.
Validation class:
  Verified, Checked, or Assumed.
Conflict handling:
  no incompatible active assertion governs overlapping semantic scope.

Only verified assertions may support unconditional production legality.

Checked assertions may enable guarded or runtime-validated specialization.

Assumed assertions require explicit human approval or produce only experimental artifacts.

Lease expiry, ownership transfer, shape mismatch, failed validation, summary invalidation, conflict, or human revocation emits a normal signed assertion retraction.

assertion revocation
  -> -1 ActiveAssertion
  -> retract derived precision gain
  -> recompute dependent aliases and legality
  -> invalidate dependent artifacts
  -> update proof DAG

19. Speculative Artifact Pipeline

TRCS never injects hypothetical assertions into the authoritative semantic lattice.

Instead, when a high-value optimization is blocked by unknown information, it may compile a dormant physical candidate.

SpeculationTicket {
  ticket_id: TicketId,
  candidate_optimization: OptimizationId,
  required_assertion_shape: AssertionShape,
  required_trust_class: Verified | Checked,
  phaseir_hash: Hash,
  semantic_frontier: RevisionFrontierId,
  artifact_dependencies: DependencySet,
  expiration_policy: ExpirationPolicy,
}
Authoritative path:
  UnknownExternal
  -> UnknownAlias
  -> no fusion
  -> safe active artifact
Speculative path:
  hypothetical assertion predicate
  -> provisional CanFuse
  -> dormant fused artifact
  -> cannot publish until ticket validates

A speculative artifact is promoted only when the issued assertion is lease-valid, shape-valid, accepted by the governor, and has exact dependency-hash compatibility with its speculation ticket.

If any dependency changes, the artifact is discarded.

20. Imported Summary Facts

External calls are classified as:

InlineAvailable
SummaryAvailable
OpaqueConservative
UnsafeOrUnknown

Imported summaries seed EDB only after compatibility checks.

ImportedSummaryManifest {
  producer_module_hash: Hash,
  producer_phaseir_version: Version,
  analysis_program_hash: Hash,
  target_abi_hash: Hash,
  summary_schema_hash: Hash,
  precision_contract: SummaryPrecisionContract,
  provenance_root: Hash,
  trust_policy: TrustPolicy,
}

The imported fact family includes:

ImportedEffect(function_id, object_or_region_id, effect_flags)
ImportedPointsToRegion(parameter_index, region_id)
ImportedEscape(parameter_index, escape_kind)
ImportedReturnAlias(return_index, parameter_index_or_object_id)
ImportedAllocationEffect(function_id, allocation_class)
ImportedCallSummary(call_target_id, summary_id)

A producer change retracts or replaces imported summary facts through ordinary signed EDB maintenance.

21. Provenance Model

Every derived fact receives a stable FactId.

Fact {
  fact_id: FactId,
  relation_id: RelationId,
  tuple_hash: u64,
  logical_weight: i32,
  first_frontier_id: RevisionFrontierId,
  last_frontier_id: RevisionFrontierId,
}
DerivationEdge {
  derived_fact_id: FactId,
  rule_id: RuleId,
  parent_fact_ids: CompactFactIdList,
  created_frontier_id: RevisionFrontierId,
  retracted_frontier_id: Optional<RevisionFrontierId>,
  retention_class: Active | Historical | Debug | Ephemeral,
  derivation_kind: Direct | Join | Widening | ImportedSummary | Assertion | Fallback,
}

Modes are:

None:
  no proof graph.
Witness:
  one deterministic proof per visible fact.
Complete:
  all proof edges for bounded diagnostics only.

Witness selection is deterministic:

lowest rule_id
  -> earliest frontier
  -> earliest epoch
  -> lexicographically smallest parent FactId set

22. Provenance Spill and Evidence Archiving

Active proof metadata remains in memory only while needed for current explanations, active compiler decisions, or a configured recent-retention window.

Detailed derivation edges are appended to an immutable, segmented spill store backed by local storage.

ProvenanceSegment {
  segment_id: SegmentId,
  frontier_min: RevisionFrontierId,
  frontier_max: RevisionFrontierId,
  fact_id_min: FactId,
  fact_id_max: FactId,
  edge_count: u64,
  compression: CompressionKind,
  checksum: Hash,
  storage_state: Resident | Mapped | Spilled | Archived,
}

The spill store uses append-only segments, block indexes keyed by derived FactId, parent FactId, source revision, and assertion ID, plus checksums and atomic manifests.

The Proof API loads only the minimal reachable witness slice for the requested fact. It does not rehydrate an entire epoch or global proof graph.

ProvenanceRetentionPolicy {
  resident_witness_budget_bytes: u64,
  spill_segment_target_bytes: u64,
  active_window_revisions: u32,
  historical_window_revisions: u32,
  max_complete_proof_bytes: u64,
  witness_retention: Keep | Summarize | Drop,
  compaction_mode: Translate | Archive | Discard,
}

When a fact retracts, active edges become tombstoned. After the retention frontier passes the final visible frontier, detailed edges may be reclaimed or archived, while retaining:

HistoricalProofSummary {
  fact_id: FactId,
  relation_id: RelationId,
  first_visible_frontier: RevisionFrontierId,
  last_visible_frontier: RevisionFrontierId,
  terminal_reason: ReasonCode,
  source_revision_set: RevisionSet,
  proof_digest: Hash,
}

The spill system must reserve disk quotas, monitor write latency, checksum every sealed segment, and degrade from Complete to Witness mode before allowing host-memory exhaustion.

23. ID Reclamation and Session Compaction

ID slots progress through:

Live
Tombstoned
Reclaimable
Reused

A slot may be reused only after all relevant GPU epochs, retained proofs, imported summaries, and revision readers have advanced beyond the retirement frontier.

GcEpoch {
  active_revision_floor: RevisionFrontierId,
  oldest_inflight_gpu_epoch: EpochId,
  oldest_provenance_retention_frontier: RevisionFrontierId,
  oldest_imported_summary_frontier: RevisionFrontierId,
}

Minor compaction removes dead rows and stale temporary buffers.

Major compaction performs copy-on-write remapping:

CompactSession {
  old_session_id: SessionId,
  new_session_id: SessionId,
  old_to_new_id_map: Relation<EntityId, EntityId>,
  retained_fact_map: Relation<FactId, FactId>,
  retained_summary_map: Relation<SummaryId, SummaryId>,
  provenance_remap_policy: Translate | Archive | Discard,
}

24. Metal Backend

The Metal backend executes sparse relational primitives:

RadixHistogram
RadixPartition
RadixSort
SegmentedScan
MergeJoin
RangeJoin
BitmapJoin
Deduplicate
Compact
Filter
UnionMerge
DifferenceConsolidate
Histogram

It uses SIMD-group execution, threadgroup staging, bounded atomics, explicit scans, columnar memory access, command-buffer windows, and ICB templates.

It does not use unrestricted dynamic device allocation.

It does not use a generic global barrier inside ordinary compute dispatches.

It does not depend on persistent polling kernels as the baseline convergence mechanism.

25. ICB Window and Predicated Fault Handling

ICBs are used to amortize command encoding and reuse SCC iteration templates. They do not permit unrestricted autonomous recursive command submission.

SccIterationTemplate {
  scheduler
  -> count
  -> scan
  -> write
  -> dedup
  -> consolidate
  -> merge
  -> convergence update
}

The parent command buffer executes bounded iteration windows.

scheduler
  -> execute template
  -> barrier
  -> scheduler
  -> execute template
  -> barrier
  -> final control checkpoint

The control block is:

DeviceControlBlock {
  current_epoch: AtomicU32,
  active_delta_mask: AtomicU64,
  overflow_mask: AtomicU64,
  is_faulted: AtomicU32,
  fault_kind: AtomicU32,
  fault_rule_id: AtomicU32,
  fault_relation_id: AtomicU32,
  predicted_rows: AtomicU64,
  available_rows: AtomicU64,
  cancellation_flag: AtomicU32,
  completed_rule_mask: AtomicU64,
}

When count detects overflow, it records the event and sets is_faulted.

Every later kernel begins with:

if is_faulted != 0 || cancellation_flag != 0 {
  return;
}

The remainder of the command-buffer window becomes no-op work. It may not mutate relations, support counts, provenance, or scheduler state.

26. GPU Lane Arbiter

Metal does not provide the general compute-priority and arbitrary preemption mechanism needed to guarantee that live TRCS analysis never contends with final lowering. The practical policy is Tribunus-level admission control at command-buffer boundaries.

GpuLanePolicy {
  foreground_lowering_reservation_pct: u8,
  background_trcs_max_inflight_windows: u32,
  trcs_max_gpu_time_per_epoch: Duration,
  trcs_max_memory_bytes: u64,
  trcs_preemption_boundary: CommandBufferWindow,
  dense_compile_exclusion_mode: Strict | Soft,
}

Foreground ComputeImage lowering may reserve memory and suspend submission of new TRCS windows.

TRCS retains its differential state and resumes after the next permitted boundary.

A submitted Metal command buffer is not assumed to be safely preemptible by the application. The design therefore keeps windows deliberately bounded. Apple documents command buffers as encoded chunks of GPU work, while command-buffer timeout is an OS-terminated error condition, not a normal user-controlled cancel primitive.

27. ANE Planner Provider

The ANE is not used for sparse joins, scans, pointer analysis, structural hashing, canonicalization, or semantic fixpoint evaluation.

It may run optional dense learned ranking models through Core ML.

ANE inputs:
  relation cardinalities
  distinct-key estimates
  skew histograms
  SCC depth
  previous kernel timings
  memory pressure
  read amplification
  recent-run count
  artifact value estimate
ANE outputs:
  join-order ranking
  partition-count ranking
  skew-risk score
  compaction priority
  batch-size recommendation
  speculative-artifact value score

The Core ML configuration uses cpuAndNeuralEngine when supported, which permits CPU and Neural Engine execution while excluding GPU use.

The ANE model is advisory. It cannot override semantic, capacity, safety, or backend capability constraints.

28. Submission Watchdog and Recovery

The runtime monitors every backend submission.

SubmissionWatch {
  submission_id: SubmissionId,
  backend_id: BackendId,
  submitted_at: Timestamp,
  deadline: Timestamp,
  command_window_id: WindowId,
  scc_id: SccId,
  phaseir_hash: Hash,
  artifact_or_rule_id: Optional<Id>,
  recovery_policy: RecoveryPolicy,
}

The watchdog uses completion callbacks, elapsed-time monitoring, control-buffer progress markers, and backend error status.

The application must not promise an emergency user-space GPU cancellation mechanism that Metal does not expose for arbitrary already-running compute work. Instead, recovery has three layers.

Before submission:
  bound window duration, validate control buffers, reserve budgets.
During normal cooperative execution:
  use cancellation_flag at safe kernel boundaries.
After a timeout or command-buffer error:
  stop future submissions to the affected lane,
  mark the rule or artifact quarantined,
  invalidate its output,
  switch affected analysis to CPU,
  collect diagnostics,
  require explicit backend recovery before re-enabling GPU work.

Apple documents MTLCommandBufferError.Code.timeout as indicating that the system interrupted and terminated the command buffer before completion. It may occur because work exceeded the system allowance or because the buffer waited too long for an event.

The watchdog states are:

Healthy
Suspect
TimedOut
BackendFaulted
Quarantined
CpuFallback
RecoveryProbe
Restored

A timeout does not imply that partial mutable GPU state is trustworthy. The affected epoch’s GPU outputs are discarded unless the backend can prove completion and receipt consistency.

TimeoutRecovery {
  1. Mark submission failed.
  2. Stop admitting new windows on affected backend lane.
  3. Discard unsealed GPU delta outputs.
  4. Roll back to last sealed relation frontier.
  5. Re-run affected SCC on CPU reference backend.
  6. Quarantine offending rule, kernel configuration, or artifact hash.
  7. Emit fault receipt and proof-accessible diagnostic.
  8. Re-enable GPU only through controlled recovery probe.
}

Metal errors also include conditions such as invalid resources, out-of-memory, page faults, timeout, and access revocation; TRCS records the specific error code in the execution receipt.

29. CPU Reference Backend

The CPU backend implements identical semantic rules, signed consolidation, assertion authority behavior, imported-summary validation, stratification, proof witness selection, and retention policies.

It is mandatory for cold-start fallback, tiny workloads, severe skew, unsupported operators, timeout recovery, debug replay, and CPU/GPU equivalence testing.

A GPU result is accepted only when it matches CPU output under the same canonical order and precision policy.

30. Developer and Agent Proof Interface

The primary user surface is a compact decision card.

Fusion blocked: possible write/read alias
Affected region: projection epilogue
Status: sound, precision widened
Revision: merge frontier 1842

The next level is a plain-language witness path.

The store at cache.rs:214 may modify data read by projection.rs:88.
Reason:
  `kv_out` may point to `session_cache`
  `weights_view` may also point to `session_cache`
  the second path originates in an opaque external call
  no compatible imported effect summary exists

Each node links to its source span, PhaseIR operation, source revision, agent work item, assertion dependency, summary artifact, precision receipt, or backend fault record.

The raw graph is shown only on demand.

MayAlias
  -> witness
      -> PointsTo(ptr_a, object_42)
      -> PointsTo(ptr_b, object_42)
  -> alternate derivations collapsed

Agents receive structured output:

OptimizationExplanation {
  decision: Allowed | Blocked | Degraded,
  optimization_id: OptimizationId,
  primary_reason: ReasonCode,
  witness_path: ProofSlice,
  source_locations: Vec<SourceLocation>,
  introduced_by_revisions: Vec<RevisionId>,
  precision_losses: Vec<PrecisionReceiptId>,
  assertion_dependencies: Vec<AssertionId>,
  imported_summaries: Vec<SummaryUsageId>,
  candidate_repairs: Vec<RepairClass>,
}

Repair recommendations must distinguish genuine illegality from missing information.

UnknownExternal:
  import or derive external effect summary.
UnknownWithinObject:
  refine symbolic region.
Context widening:
  raise local context budget.
Assertion lease expired:
  reacquire ownership or seek human approval.
Real shared allocation:
  transformation is genuinely illegal.
GPU timeout:
  inspect quarantined rule or artifact; CPU fallback remains authoritative.

31. Execution Receipt

AnalysisExecutionReceipt {
  compilation_id: CompilationId,
  revision_frontier_id: RevisionFrontierId,
  analysis_id: AnalysisId,
  backend_id: BackendId,
  phaseir_hash: Hash,
  semantic_program_hash: Hash,
  relation_schema_hash: Hash,
  final_status: Success | CapacityEvent | PrecisionWidened |
                Fallback | Cancelled | TimedOut | Quarantined | Failed,
  converged: bool,
  epochs: u32,
  bulk_load_used: bool,
  trace_run_summary: TraceRunSummary,
  rule_execution_counts: Map<RuleId, u64>,
  relation_cardinality_summary: Map<RelationId, CardinalityStats>,
  max_delta_cardinality: u64,
  max_intermediate_cardinality: u64,
  host_memory_peak: u64,
  device_memory_peak: u64,
  provenance_spill_bytes: u64,
  explicit_materialization_count: u64,
  hidden_copy_count: u64,
  widening_events: Vec<PrecisionReceiptId>,
  capacity_events: Vec<CapacityEventId>,
  fallback_events: Vec<FallbackEventId>,
  imported_summary_usage: Vec<SummaryUsageId>,
  assertion_usage: Vec<AssertionUsageId>,
  speculation_usage: Vec<SpeculationTicketId>,
  stratification_barriers: Vec<StratificationBarrierReceipt>,
  backend_fault: Optional<BackendFaultReceipt>,
  determinism_hash: Hash,
}

hidden_copy_count must always be zero.

32. Required Verification Gates

Bulk-load equivalence:
  bulk-loaded base equals incremental bootstrap result.
Cold-start performance:
  bulk mode avoids per-delta bootstrap overhead at workspace scale.
LSM trace equivalence:
  base plus recent runs equals monolithic canonical relation.
Trace compaction equivalence:
  compacted trace equals uncompacted trace.
Federated remap equivalence:
  remote delta after identity resolution equals central canonicalization.
Disjoint merge-frontier equivalence:
  concurrent disjoint revision merge equals deterministic sequential merge.
Signed physical emission correctness:
  negative updates allocate and propagate physical rows correctly.
Consolidation correctness:
  equivalent signed rows cancel only after proper support accounting.
Negation stratification:
  no upper stratum reads an unconverged negatively referenced relation.
Assertion revocation:
  lease expiry or shape mismatch retracts all dependent conclusions.
Speculation isolation:
  dormant artifact never becomes active without validated ticket dependencies.
Provenance spill integrity:
  paged proof slices match resident proof DAGs.
Provenance memory bound:
  sustained agent churn cannot exceed resident evidence budget.
ICB fault predication:
  post-fault commands produce no mutable writes.
GPU lane isolation:
  foreground lowering blocks new background windows at command-buffer boundaries.
Watchdog recovery:
  timeout discards unsealed output, quarantines offender, and restores analysis through CPU fallback.
CPU/GPU equivalence:
  semantic facts, precision receipts, and witness provenance match.

33. Delivery Sequence

Phase one implements canonical IDs, revision DAGs, bulk-load construction, CPU relations, signed deltas, and basic receipts.

Phase two implements log-structured base and recent arrangements, batch directories, trace compaction, and incremental local analysis.

Phase three implements provenance witnesses, spill segments, proof queries, retention GC, and source-linked decision cards.

Phase four implements allocation-site points-to, symbolic regions, aliasing, effects, context budgets, and imported summaries.

Phase five implements assertion authority, lease-bound semantic pragmas, conflict resolution, revocation propagation, and speculative artifact tickets.

Phase six implements Metal radix primitives, count-scan-write, signed consolidation, ICB windows, predicated fault handling, and GPU lane arbitration.

Phase seven implements ANE advisory plan ranking, runtime telemetry, watchdog quarantine, CPU fallback, and backend recovery probes.

Phase eight validates federated agent synchronization, large workspace cold starts, multi-agent churn, proof spill pressure, and recovery from command-buffer timeout faults.

34. Final Position

TRCS is a live compiler truth engine built around exact sparse semantics, incremental signed updates, log-structured arrangements, bounded accelerator execution, auditable agent authority, and proof-carrying optimization decisions.

Its production resilience comes from avoiding three common failures: it does not rewrite giant sorted relations for tiny edits; it does not hold every historical proof edge in RAM; and it does not assume an application can safely cancel an already-running GPU workload.

The resulting system remains responsive under cold starts, distributed agent work, live edits, assertion revocations, opaque dependencies, provenance-heavy debugging, sparse GPU execution, foreground lowering pressure, and backend faults.
