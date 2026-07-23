//! Persistent evolutionary runtime.
//!
//! A compile is a disposable session. The runtime owns knowledge that should
//! survive sessions: archives, receipts, lineages, operator statistics,
//! descriptors, surrogate metadata, and hardware profiles.

use crate::evolution::emitters::{
    Emitter, LocalEmitter, MemoryEmitter, RandomEmitter, StrategyEmitter,
};
use crate::evolution::emitters::{EmitterKind, EmitterPolicy};
use crate::evolution::foundation::{CandidateGenome, MetalGeometryAxis};
use crate::evolution::memory::{
    EvolutionContextKey, EvolutionReceipt, EvolutionaryMemory, ReceiptSurrogate,
};
use crate::evolution::objectives::{ArchiveEntry, BehaviorDescriptor, QualityDiversityArchive};
use crate::evolution::variation::{AdaptiveVariationController, VariationOperator};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};

const RUNTIME_SNAPSHOT_SCHEMA_VERSION: u32 = 1;

fn default_snapshot_schema_version() -> u32 {
    RUNTIME_SNAPSHOT_SCHEMA_VERSION
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct HardwareProfileKey {
    pub backend: String,
    pub device: String,
    pub driver: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub key: HardwareProfileKey,
    pub measurements: u64,
    pub last_seen_unix_ms: u64,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageRecord {
    pub session_id: String,
    pub parent_digest: Option<String>,
    pub child_digest: String,
    pub operator: VariationOperator,
    pub generation: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptorRecord {
    pub candidate_digest: String,
    pub descriptor: BehaviorDescriptor,
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrogateRecord {
    pub name: String,
    pub version: String,
    pub observations: u64,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Default)]
pub struct ArchiveDb {
    pub execution: QualityDiversityArchive,
    pub hardware: HashMap<HardwareProfileKey, ArchiveEntry>,
    pub tensor: HashMap<String, QualityDiversityArchive>,
    pub operator: HashMap<VariationOperator, AdaptiveVariationController>,
}

#[derive(Debug, Default)]
pub struct EvolutionKnowledge {
    pub archives: ArchiveDb,
    pub receipts: EvolutionaryMemory,
    pub lineages: Vec<LineageRecord>,
    pub descriptors: Vec<DescriptorRecord>,
    pub surrogates: Vec<SurrogateRecord>,
    pub hardware: HashMap<HardwareProfileKey, HardwareProfile>,
    pub emitters: EmitterPolicy,
    pub contextual_emitters: HashMap<EvolutionContextKey, EmitterPolicy>,
    pub contextual_operators: HashMap<EvolutionContextKey, AdaptiveVariationController>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionSession {
    pub session_id: String,
    pub model_context: String,
    pub hardware_context: String,
    pub started_unix_ms: u64,
}

#[derive(Clone, Default)]
pub struct EvolutionRuntime {
    knowledge: Arc<Mutex<EvolutionKnowledge>>,
    persistence_path: Arc<Mutex<Option<std::path::PathBuf>>>,
    merged_snapshot_digests: Arc<Mutex<HashSet<String>>>,
}

impl std::fmt::Debug for EvolutionRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EvolutionRuntime")
            .finish_non_exhaustive()
    }
}

impl EvolutionRuntime {
    pub fn global() -> Self {
        static GLOBAL: OnceLock<EvolutionRuntime> = OnceLock::new();
        GLOBAL
            .get_or_init(|| {
                std::env::var_os("PRISM_EVOLUTION_RUNTIME_PATH")
                    .map(std::path::PathBuf::from)
                    .and_then(|path| EvolutionRuntime::new_persistent(path).ok())
                    .unwrap_or_else(EvolutionRuntime::new)
            })
            .clone()
    }

    pub fn emit_candidates(
        &self,
        seed: &CandidateGenome,
        context: &[u8],
    ) -> (EmitterKind, Vec<CandidateGenome>) {
        self.emit_candidates_for_context(
            seed,
            context,
            &EvolutionContextKey {
                hardware: "*".into(),
                model_family: "*".into(),
                tensor_family: "*".into(),
            },
        )
    }

    pub fn emit_candidates_for_context(
        &self,
        seed: &CandidateGenome,
        context: &[u8],
        context_key: &EvolutionContextKey,
    ) -> (EmitterKind, Vec<CandidateGenome>) {
        let mut rng = rand::thread_rng();
        let (policy, memory) = self
            .knowledge
            .lock()
            .map(|knowledge| {
                let policy = resolve_contextual_policy(&knowledge, context_key);
                (policy, knowledge.receipts.clone())
            })
            .unwrap_or_default();
        let kind = policy.choose(&mut rng);
        let candidates = match kind {
            EmitterKind::Local => LocalEmitter.emit(seed, context, &memory, &mut rng),
            EmitterKind::Memory => MemoryEmitter.emit(seed, context, &memory, &mut rng),
            EmitterKind::Random => RandomEmitter.emit(seed, context, &memory, &mut rng),
            other => StrategyEmitter(other).emit(seed, context, &memory, &mut rng),
        };
        (kind, candidates)
    }

    pub fn record_emitter_feedback(&self, emitter: EmitterKind, reward: f64) {
        self.record_emitter_reward(emitter, reward);
    }
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_persistent(path: impl Into<std::path::PathBuf>) -> Result<Self, String> {
        let runtime = Self::new();
        let path = path.into();
        if path.exists() {
            runtime.load_snapshot(&path)?;
        }
        runtime.set_persistence_path(path);
        Ok(runtime)
    }

    pub fn set_persistence_path(&self, path: impl Into<std::path::PathBuf>) {
        if let Ok(mut configured) = self.persistence_path.lock() {
            *configured = Some(path.into());
        }
    }

    pub fn persist(&self) -> Result<(), String> {
        let path = self
            .persistence_path
            .lock()
            .map_err(|error| error.to_string())?
            .clone()
            .ok_or_else(|| "evolution runtime has no persistence path".to_string())?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
        }
        self.save_snapshot(path)
    }

    pub fn persist_if_configured(&self) -> Result<(), String> {
        let configured = self
            .persistence_path
            .lock()
            .map_err(|error| error.to_string())?
            .is_some();
        if configured {
            self.persist()
        } else {
            Ok(())
        }
    }

    pub fn begin_session(
        &self,
        model_context: impl Into<String>,
        hardware_context: impl Into<String>,
    ) -> EvolutionSession {
        EvolutionSession {
            session_id: uuid::Uuid::new_v4().to_string(),
            model_context: model_context.into(),
            hardware_context: hardware_context.into(),
            started_unix_ms: now_unix_ms(),
        }
    }

    pub fn record_receipt(&self, receipt: EvolutionReceipt) {
        if let Ok(mut knowledge) = self.knowledge.lock() {
            let duplicate = knowledge.receipts.receipts.iter().any(|existing| {
                existing.parent_digest == receipt.parent_digest
                    && existing.child_digest == receipt.child_digest
                    && existing.measurement_receipt_digest == receipt.measurement_receipt_digest
            });
            if !duplicate {
                knowledge.receipts.record(receipt);
            }
        }
    }

    pub fn record_lineage(&self, lineage: LineageRecord) {
        if let Ok(mut knowledge) = self.knowledge.lock() {
            let duplicate = knowledge.lineages.iter().any(|existing| {
                existing.session_id == lineage.session_id
                    && existing.parent_digest == lineage.parent_digest
                    && existing.child_digest == lineage.child_digest
                    && existing.operator == lineage.operator
                    && existing.generation == lineage.generation
            });
            if !duplicate {
                knowledge.lineages.push(lineage);
            }
        }
    }

    pub fn record_descriptor(&self, descriptor: DescriptorRecord) {
        if let Ok(mut knowledge) = self.knowledge.lock() {
            let duplicate = knowledge.descriptors.iter().any(|existing| {
                existing.candidate_digest == descriptor.candidate_digest
                    && existing.session_id == descriptor.session_id
            });
            if !duplicate {
                knowledge.descriptors.push(descriptor);
            }
        }
    }

    pub fn record_hardware_profile(&self, profile: HardwareProfile) {
        if let Ok(mut knowledge) = self.knowledge.lock() {
            if let Some(existing) = knowledge.hardware.get_mut(&profile.key) {
                if profile.last_seen_unix_ms > existing.last_seen_unix_ms {
                    existing.measurements =
                        existing.measurements.saturating_add(profile.measurements);
                    existing.last_seen_unix_ms = profile.last_seen_unix_ms;
                    existing.metadata = profile.metadata;
                } else if profile.last_seen_unix_ms == existing.last_seen_unix_ms {
                    existing.metadata = profile.metadata;
                }
            } else {
                knowledge.hardware.insert(profile.key.clone(), profile);
            }
        }
    }

    pub fn insert_execution_elite(&self, entry: ArchiveEntry) {
        if let Ok(mut knowledge) = self.knowledge.lock() {
            knowledge.archives.execution.insert(entry);
        }
    }

    pub fn insert_tensor_elite(&self, tensor_family: impl Into<String>, entry: ArchiveEntry) {
        if let Ok(mut knowledge) = self.knowledge.lock() {
            knowledge
                .archives
                .tensor
                .entry(tensor_family.into())
                .or_default()
                .insert(entry);
        }
    }

    pub fn insert_hardware_elite(&self, profile: HardwareProfileKey, entry: ArchiveEntry) {
        if let Ok(mut knowledge) = self.knowledge.lock() {
            let replace = knowledge
                .archives
                .hardware
                .get(&profile)
                .map(|existing| entry.objectives.dominates(&existing.objectives))
                .unwrap_or(true);
            if replace {
                knowledge.archives.hardware.insert(profile, entry);
            }
        }
    }

    pub fn record_surrogate(&self, record: SurrogateRecord) {
        if let Ok(mut knowledge) = self.knowledge.lock() {
            if let Some(existing) = knowledge
                .surrogates
                .iter_mut()
                .find(|existing| existing.name == record.name && existing.version == record.version)
            {
                existing.observations = existing.observations.max(record.observations);
                existing.metadata = record.metadata;
            } else {
                knowledge.surrogates.push(record);
            }
        }
    }

    pub fn record_operator_reward(&self, operator: VariationOperator, reward: f64) {
        if let Ok(mut knowledge) = self.knowledge.lock() {
            knowledge
                .archives
                .operator
                .entry(operator)
                .or_default()
                .record(operator, reward);
        }
    }

    pub fn record_contextual_operator_reward(
        &self,
        context: EvolutionContextKey,
        operator: VariationOperator,
        reward: f64,
    ) {
        if let Ok(mut knowledge) = self.knowledge.lock() {
            knowledge
                .contextual_operators
                .entry(context)
                .or_default()
                .record(operator, reward);
        }
    }

    pub fn variation_controller(
        &self,
        operator: VariationOperator,
    ) -> Option<AdaptiveVariationController> {
        self.knowledge
            .lock()
            .ok()
            .and_then(|knowledge| knowledge.archives.operator.get(&operator).cloned())
    }

    pub fn contextual_variation_controller(
        &self,
        context: &EvolutionContextKey,
    ) -> Option<AdaptiveVariationController> {
        self.knowledge
            .lock()
            .ok()
            .and_then(|knowledge| resolve_contextual_operator_controller(&knowledge, context))
    }

    pub fn record_geometry_observation(
        &self,
        geometry: &MetalGeometryAxis,
        shared_memory_bytes: u32,
        score: f64,
    ) {
        if let Ok(mut knowledge) = self.knowledge.lock() {
            knowledge
                .archives
                .operator
                .entry(VariationOperator::Geometry)
                .or_default()
                .geometry_covariance
                .observe(geometry, shared_memory_bytes, score);
        }
    }

    pub fn record_emitter_reward(&self, emitter: EmitterKind, reward: f64) {
        if let Ok(mut knowledge) = self.knowledge.lock() {
            knowledge.emitters.record(emitter, reward);
        }
    }

    pub fn record_contextual_emitter_reward(
        &self,
        context: EvolutionContextKey,
        emitter: EmitterKind,
        reward: f64,
    ) {
        if let Ok(mut knowledge) = self.knowledge.lock() {
            knowledge
                .contextual_emitters
                .entry(context)
                .or_default()
                .record(emitter, reward);
        }
    }

    pub fn emitter_policy(&self) -> EmitterPolicy {
        self.knowledge
            .lock()
            .map(|knowledge| knowledge.emitters.clone())
            .unwrap_or_default()
    }

    pub fn emitter_policy_for_context(&self, context: &EvolutionContextKey) -> EmitterPolicy {
        self.knowledge
            .lock()
            .map(|knowledge| resolve_contextual_policy(&knowledge, context))
            .unwrap_or_default()
    }

    pub fn replay_candidates(
        &self,
        genome: &CandidateGenome,
        context: &crate::evolution::memory::EvolutionContextKey,
        minimum_improvement: f64,
    ) -> Vec<CandidateGenome> {
        self.knowledge
            .lock()
            .map(|knowledge| {
                knowledge
                    .receipts
                    .replay_candidates(genome, context, minimum_improvement)
            })
            .unwrap_or_default()
    }

    pub fn successful_receipts(
        &self,
        context: &EvolutionContextKey,
        minimum_improvement: f64,
    ) -> Vec<EvolutionReceipt> {
        self.knowledge
            .lock()
            .map(|knowledge| {
                knowledge
                    .receipts
                    .successful_mutations(context, minimum_improvement)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Reconstruct the receipt-backed surrogate for one search context from
    /// persistent evidence. The returned model is deliberately local and
    /// mutable: new measured observations can be fed into it during a session
    /// without mutating the runtime until the session records its metadata.
    pub fn receipt_surrogate(
        &self,
        context: &EvolutionContextKey,
        minimum_improvement: f64,
        max_observations: usize,
    ) -> ReceiptSurrogate {
        let receipts = self.successful_receipts(context, minimum_improvement);
        ReceiptSurrogate::from_receipts(&receipts, max_observations)
    }

    pub fn execution_elites(&self, limit: usize) -> Vec<CandidateGenome> {
        self.knowledge
            .lock()
            .map(|knowledge| {
                knowledge
                    .archives
                    .execution
                    .ranked_elites()
                    .into_iter()
                    .take(limit)
                    .map(|entry| entry.genome.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Return persisted execution-archive entries, including their measured
    /// objectives and behavioral descriptors. Callers can hydrate a local
    /// archive with these entries so a new session continues the previous
    /// quality-diversity search instead of merely reusing the genomes.
    pub fn execution_archive_entries(&self, limit: usize) -> Vec<ArchiveEntry> {
        self.knowledge
            .lock()
            .map(|knowledge| {
                knowledge
                    .archives
                    .execution
                    .ranked_elites()
                    .into_iter()
                    .take(limit)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Return the persisted entries for one tensor-family archive.
    pub fn tensor_archive_entries(&self, tensor_family: &str, limit: usize) -> Vec<ArchiveEntry> {
        self.knowledge
            .lock()
            .map(|knowledge| {
                knowledge
                    .archives
                    .tensor
                    .get(tensor_family)
                    .map(|archive| {
                        archive
                            .ranked_elites()
                            .into_iter()
                            .take(limit)
                            .cloned()
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .unwrap_or_default()
    }

    /// Return the persisted hardware-specific elite, if one exists.
    pub fn hardware_archive_entries(&self, profile: &HardwareProfileKey) -> Vec<ArchiveEntry> {
        self.knowledge
            .lock()
            .map(|knowledge| {
                knowledge
                    .archives
                    .hardware
                    .get(profile)
                    .cloned()
                    .into_iter()
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn hardware_elites(
        &self,
        profile: &HardwareProfileKey,
        limit: usize,
    ) -> Vec<CandidateGenome> {
        self.knowledge
            .lock()
            .ok()
            .and_then(|knowledge| {
                knowledge
                    .archives
                    .hardware
                    .get(profile)
                    .map(|entry| vec![entry.genome.clone()].into_iter().take(limit).collect())
            })
            .unwrap_or_default()
    }

    pub fn tensor_elites(&self, tensor_family: &str, limit: usize) -> Vec<CandidateGenome> {
        self.knowledge
            .lock()
            .ok()
            .and_then(|knowledge| {
                knowledge.archives.tensor.get(tensor_family).map(|archive| {
                    archive
                        .ranked_elites()
                        .into_iter()
                        .take(limit)
                        .map(|entry| entry.genome.clone())
                        .collect()
                })
            })
            .unwrap_or_default()
    }

    pub fn snapshot_counts(&self) -> RuntimeCounts {
        self.knowledge
            .lock()
            .map(|knowledge| RuntimeCounts {
                execution_cells: knowledge.archives.execution.cells.len(),
                tensor_cells: knowledge
                    .archives
                    .tensor
                    .values()
                    .map(|archive| archive.cells.len())
                    .sum(),
                hardware_elite_count: knowledge.archives.hardware.len(),
                operator_count: knowledge.archives.operator.len(),
                receipt_count: knowledge.receipts.receipts.len(),
                lineage_count: knowledge.lineages.len(),
                descriptor_count: knowledge.descriptors.len(),
                hardware_profile_count: knowledge.hardware.len(),
                surrogate_count: knowledge.surrogates.len(),
                emitter_attempts: knowledge
                    .emitters
                    .stats
                    .values()
                    .map(|stats| stats.attempts)
                    .sum(),
                contextual_emitter_count: knowledge.contextual_emitters.len(),
                contextual_operator_count: knowledge.contextual_operators.len(),
            })
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeCounts {
    pub execution_cells: usize,
    pub tensor_cells: usize,
    pub hardware_elite_count: usize,
    pub operator_count: usize,
    pub receipt_count: usize,
    pub lineage_count: usize,
    pub descriptor_count: usize,
    pub hardware_profile_count: usize,
    pub surrogate_count: usize,
    pub emitter_attempts: u64,
    pub contextual_emitter_count: usize,
    pub contextual_operator_count: usize,
}

/// Portable knowledge payload for continual-learning checkpoints and
/// cross-machine archive exchange. Runtime locks never cross this boundary.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeSnapshot {
    #[serde(default = "default_snapshot_schema_version")]
    pub schema_version: u32,
    pub execution_elites: Vec<ArchiveEntry>,
    pub tensor_elites: Vec<(String, Vec<ArchiveEntry>)>,
    pub hardware_elites: Vec<(HardwareProfileKey, ArchiveEntry)>,
    pub operator_stats: Vec<(VariationOperator, AdaptiveVariationController)>,
    pub contextual_emitters: Vec<(EvolutionContextKey, EmitterPolicy)>,
    #[serde(default)]
    pub contextual_operators: Vec<(EvolutionContextKey, AdaptiveVariationController)>,
    pub global_emitters: EmitterPolicy,
    pub receipts: Vec<EvolutionReceipt>,
    pub lineages: Vec<LineageRecord>,
    pub descriptors: Vec<DescriptorRecord>,
    pub hardware: Vec<HardwareProfile>,
    pub surrogates: Vec<SurrogateRecord>,
}

impl EvolutionRuntime {
    /// Serialize the portable knowledge snapshot for transport to another
    /// evolution worker or persistence service.
    pub fn snapshot_bytes(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(&self.snapshot()).map_err(|error| error.to_string())
    }

    /// Merge a transported snapshot payload using the same schema checks and
    /// duplicate protection as `load_snapshot`.
    pub fn merge_snapshot_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        let snapshot: RuntimeSnapshot =
            serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        if snapshot.schema_version > RUNTIME_SNAPSHOT_SCHEMA_VERSION {
            return Err(format!(
                "unsupported evolution snapshot schema version {}",
                snapshot.schema_version
            ));
        }
        // `merge_snapshot` performs the canonical decoded-payload digest and
        // owns the actual merge, so disk and transport callers share exactly
        // one duplicate-protection path.
        self.merge_snapshot(snapshot);
        Ok(())
    }

    pub fn save_snapshot(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        let path = path.as_ref();
        let bytes =
            serde_json::to_vec_pretty(&self.snapshot()).map_err(|error| error.to_string())?;
        let temporary = path.with_extension(format!(
            "json.tmp-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        if let Err(error) = std::fs::write(&temporary, bytes) {
            let _ = std::fs::remove_file(&temporary);
            return Err(error.to_string());
        }
        std::fs::rename(&temporary, path).map_err(|error| {
            let _ = std::fs::remove_file(&temporary);
            error.to_string()
        })
    }

    pub fn load_snapshot(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
        self.merge_snapshot_bytes(&bytes)
    }

    pub fn snapshot(&self) -> RuntimeSnapshot {
        self.knowledge
            .lock()
            .map(|knowledge| RuntimeSnapshot {
                schema_version: RUNTIME_SNAPSHOT_SCHEMA_VERSION,
                execution_elites: knowledge
                    .archives
                    .execution
                    .cells
                    .values()
                    .cloned()
                    .collect(),
                tensor_elites: knowledge
                    .archives
                    .tensor
                    .iter()
                    .map(|(family, archive)| {
                        (family.clone(), archive.cells.values().cloned().collect())
                    })
                    .collect(),
                hardware_elites: knowledge
                    .archives
                    .hardware
                    .iter()
                    .map(|(profile, entry)| (profile.clone(), entry.clone()))
                    .collect(),
                operator_stats: knowledge
                    .archives
                    .operator
                    .iter()
                    .map(|(operator, controller)| (*operator, controller.clone()))
                    .collect(),
                contextual_emitters: knowledge
                    .contextual_emitters
                    .iter()
                    .map(|(context, policy)| (context.clone(), policy.clone()))
                    .collect(),
                contextual_operators: knowledge
                    .contextual_operators
                    .iter()
                    .map(|(context, controller)| (context.clone(), controller.clone()))
                    .collect(),
                global_emitters: knowledge.emitters.clone(),
                receipts: knowledge.receipts.receipts.clone(),
                lineages: knowledge.lineages.clone(),
                descriptors: knowledge.descriptors.clone(),
                hardware: knowledge.hardware.values().cloned().collect(),
                surrogates: knowledge.surrogates.clone(),
            })
            .unwrap_or_default()
    }

    /// Merge remote knowledge without allowing it to bypass local archive
    /// dominance or evidence bookkeeping. This is the federation boundary.
    pub fn merge_snapshot(&self, snapshot: RuntimeSnapshot) {
        if snapshot.schema_version > RUNTIME_SNAPSHOT_SCHEMA_VERSION {
            return;
        }
        let snapshot_digest = canonical_snapshot_digest(&snapshot).ok();
        if let Some(digest) = snapshot_digest {
            if let Ok(mut merged) = self.merged_snapshot_digests.lock() {
                if !merged.insert(digest) {
                    return;
                }
            }
        }
        if let Ok(mut knowledge) = self.knowledge.lock() {
            for entry in snapshot.execution_elites {
                knowledge.archives.execution.insert(entry);
            }
            for (family, entries) in snapshot.tensor_elites {
                let archive = knowledge.archives.tensor.entry(family).or_default();
                for entry in entries {
                    archive.insert(entry);
                }
            }
            for (profile, entry) in snapshot.hardware_elites {
                let replace = knowledge
                    .archives
                    .hardware
                    .get(&profile)
                    .map(|existing| entry.objectives.dominates(&existing.objectives))
                    .unwrap_or(true);
                if replace {
                    knowledge.archives.hardware.insert(profile, entry);
                }
            }
            for (operator, controller) in snapshot.operator_stats {
                knowledge
                    .archives
                    .operator
                    .entry(operator)
                    .or_default()
                    .merge(&controller);
            }
            for (context, policy) in snapshot.contextual_emitters {
                let local = knowledge.contextual_emitters.entry(context).or_default();
                merge_emitter_policy(local, &policy);
            }
            for (context, controller) in snapshot.contextual_operators {
                knowledge
                    .contextual_operators
                    .entry(context)
                    .or_default()
                    .merge(&controller);
            }
            merge_emitter_policy(&mut knowledge.emitters, &snapshot.global_emitters);
            for receipt in snapshot.receipts {
                let duplicate = knowledge.receipts.receipts.iter().any(|existing| {
                    existing.parent_digest == receipt.parent_digest
                        && existing.child_digest == receipt.child_digest
                        && existing.measurement_receipt_digest == receipt.measurement_receipt_digest
                });
                if !duplicate {
                    knowledge.receipts.record(receipt);
                }
            }
            for lineage in snapshot.lineages {
                if !knowledge.lineages.iter().any(|existing| {
                    existing.child_digest == lineage.child_digest
                        && existing.session_id == lineage.session_id
                }) {
                    knowledge.lineages.push(lineage);
                }
            }
            for descriptor in snapshot.descriptors {
                if !knowledge.descriptors.iter().any(|existing| {
                    existing.candidate_digest == descriptor.candidate_digest
                        && existing.session_id == descriptor.session_id
                }) {
                    knowledge.descriptors.push(descriptor);
                }
            }
            for profile in snapshot.hardware {
                if let Some(existing) = knowledge.hardware.get_mut(&profile.key) {
                    if profile.last_seen_unix_ms > existing.last_seen_unix_ms {
                        existing.measurements =
                            existing.measurements.saturating_add(profile.measurements);
                        existing.last_seen_unix_ms = profile.last_seen_unix_ms;
                        existing.metadata = profile.metadata;
                    } else if profile.last_seen_unix_ms == existing.last_seen_unix_ms {
                        existing.metadata = profile.metadata;
                    }
                } else {
                    knowledge.hardware.insert(profile.key.clone(), profile);
                }
            }
            for surrogate in snapshot.surrogates {
                if let Some(existing) = knowledge.surrogates.iter_mut().find(|existing| {
                    existing.name == surrogate.name && existing.version == surrogate.version
                }) {
                    let is_newer = surrogate.observations >= existing.observations;
                    existing.observations = existing.observations.max(surrogate.observations);
                    if is_newer {
                        existing.metadata = surrogate.metadata;
                    }
                } else {
                    knowledge.surrogates.push(surrogate);
                }
            }
        }
    }
}

fn sha256_digest_bytes(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(bytes);
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn merge_emitter_policy(local: &mut EmitterPolicy, remote: &EmitterPolicy) {
    for (kind, stats) in &remote.stats {
        let current = local.stats.entry(*kind).or_default();
        current.attempts = current.attempts.saturating_add(stats.attempts);
        current.successes = current.successes.saturating_add(stats.successes);
        current.reward_sum += stats.reward_sum;
    }
    local.exploration = (local.exploration + remote.exploration) / 2.0;
}

fn resolve_contextual_operator_controller(
    knowledge: &EvolutionKnowledge,
    context: &EvolutionContextKey,
) -> Option<AdaptiveVariationController> {
    knowledge
        .contextual_operators
        .iter()
        .filter(|(candidate, _)| candidate.matches(context))
        .max_by_key(|(candidate, _)| {
            (candidate.hardware != "*") as u8
                + (candidate.model_family != "*") as u8
                + (candidate.tensor_family != "*") as u8
        })
        .map(|(_, controller)| controller.clone())
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn context_specificity(context: &EvolutionContextKey) -> usize {
    [
        &context.hardware,
        &context.model_family,
        &context.tensor_family,
    ]
    .into_iter()
    .filter(|value| value.as_str() != "*")
    .count()
}

fn canonical_snapshot_digest(snapshot: &RuntimeSnapshot) -> Result<String, serde_json::Error> {
    fn canonicalize(value: serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(object) => {
                let mut entries: Vec<_> = object.into_iter().collect();
                entries.sort_by(|left, right| left.0.cmp(&right.0));
                serde_json::Value::Object(
                    entries
                        .into_iter()
                        .map(|(key, value)| (key, canonicalize(value)))
                        .collect(),
                )
            }
            serde_json::Value::Array(values) => {
                serde_json::Value::Array(values.into_iter().map(canonicalize).collect())
            }
            scalar => scalar,
        }
    }

    serde_json::to_value(snapshot)
        .map(canonicalize)
        .and_then(|value| serde_json::to_vec(&value))
        .map(|bytes| sha256_digest_bytes(&bytes))
}

fn resolve_contextual_policy(
    knowledge: &EvolutionKnowledge,
    context: &EvolutionContextKey,
) -> EmitterPolicy {
    knowledge
        .contextual_emitters
        .iter()
        .filter(|(stored, _)| stored.matches(context))
        .max_by_key(|(stored, _)| context_specificity(stored))
        .map(|(_, policy)| policy.clone())
        .unwrap_or_else(|| knowledge.emitters.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sessions_are_disposable_but_knowledge_survives() {
        let runtime = EvolutionRuntime::new();
        let first = runtime.begin_session("moe", "apple-m1");
        runtime.record_descriptor(DescriptorRecord {
            candidate_digest: "candidate-1".into(),
            descriptor: BehaviorDescriptor::from_genome(&CandidateGenome::new()),
            session_id: first.session_id.clone(),
        });
        let second = runtime.begin_session("moe", "apple-m1");
        assert_ne!(first.session_id, second.session_id);
        assert_eq!(runtime.snapshot_counts().descriptor_count, 1);
    }

    #[test]
    fn hardware_profiles_are_keyed_and_replaced() {
        let runtime = EvolutionRuntime::new();
        let key = HardwareProfileKey {
            backend: "metal".into(),
            device: "m1".into(),
            driver: "1".into(),
        };
        runtime.record_hardware_profile(HardwareProfile {
            key: key.clone(),
            measurements: 1,
            last_seen_unix_ms: 1,
            metadata: serde_json::json!({}),
        });
        runtime.record_hardware_profile(HardwareProfile {
            key: key.clone(),
            measurements: 2,
            last_seen_unix_ms: 2,
            metadata: serde_json::json!({}),
        });
        assert_eq!(runtime.snapshot_counts().hardware_profile_count, 1);
        assert_eq!(runtime.snapshot().hardware[0].measurements, 3);
        let genome = CandidateGenome::new();
        runtime.insert_hardware_elite(
            key.clone(),
            ArchiveEntry {
                genome: genome.clone(),
                descriptor: BehaviorDescriptor::from_genome(&genome),
                objectives: crate::evolution::objectives::ObjectiveVector::new(vec![
                    crate::evolution::objectives::ObjectiveValue::maximize("fitness", 0.8),
                ]),
                generation: 1,
                novelty: 1.0,
            },
        );
        assert_eq!(runtime.snapshot_counts().hardware_elite_count, 1);
        assert_eq!(runtime.hardware_elites(&key, 1).len(), 1);
    }

    #[test]
    fn repeated_hardware_snapshot_merge_is_idempotent() {
        let source = EvolutionRuntime::new();
        let target = EvolutionRuntime::new();
        source.record_hardware_profile(HardwareProfile {
            key: HardwareProfileKey {
                backend: "metal".into(),
                device: "m1".into(),
                driver: "1".into(),
            },
            measurements: 5,
            last_seen_unix_ms: 10,
            metadata: serde_json::json!({"run": 1}),
        });
        let snapshot = source.snapshot();
        target.merge_snapshot(snapshot.clone());
        target.merge_snapshot(snapshot);
        assert_eq!(target.snapshot().hardware[0].measurements, 5);
    }

    #[test]
    fn snapshots_merge_without_duplicate_receipts_or_elites() {
        let first = EvolutionRuntime::new();
        let second = EvolutionRuntime::new();
        second.merge_snapshot(first.snapshot());
        assert_eq!(second.snapshot_counts().receipt_count, 0);
    }

    #[test]
    fn snapshot_merge_accumulates_distributed_emitter_evidence() {
        let first = EvolutionRuntime::new();
        let second = EvolutionRuntime::new();
        first.record_emitter_reward(EmitterKind::Local, 0.5);
        second.record_emitter_reward(EmitterKind::Local, 0.25);
        let snapshot = first.snapshot();
        second.merge_snapshot(snapshot.clone());
        second.merge_snapshot(snapshot);
        assert_eq!(
            second.emitter_policy().stats[&EmitterKind::Local].attempts,
            2
        );
        assert!(
            (second.emitter_policy().stats[&EmitterKind::Local].reward_sum - 0.75).abs() < 1e-9
        );
    }

    #[test]
    fn snapshot_merge_keeps_newest_surrogate_observation() {
        let first = EvolutionRuntime::new();
        let second = EvolutionRuntime::new();
        first.record_surrogate(SurrogateRecord {
            name: "receipt-surrogate".into(),
            version: "online".into(),
            observations: 12,
            metadata: serde_json::json!({"source": "first"}),
        });
        second.record_surrogate(SurrogateRecord {
            name: "receipt-surrogate".into(),
            version: "online".into(),
            observations: 4,
            metadata: serde_json::json!({"source": "second"}),
        });
        second.merge_snapshot(first.snapshot());
        let surrogate = second
            .snapshot()
            .surrogates
            .into_iter()
            .find(|record| record.name == "receipt-surrogate")
            .unwrap();
        assert_eq!(surrogate.observations, 12);
        assert_eq!(surrogate.metadata["source"], "first");
    }

    #[test]
    fn live_receipt_recording_is_idempotent() {
        let runtime = EvolutionRuntime::new();
        let descriptor = BehaviorDescriptor::from_genome(&CandidateGenome::new());
        runtime.record_lineage(LineageRecord {
            session_id: "session".into(),
            parent_digest: Some("parent".into()),
            child_digest: "child".into(),
            operator: VariationOperator::Geometry,
            generation: 1,
        });
        runtime.record_lineage(LineageRecord {
            session_id: "session".into(),
            parent_digest: Some("parent".into()),
            child_digest: "child".into(),
            operator: VariationOperator::Geometry,
            generation: 1,
        });
        runtime.record_descriptor(DescriptorRecord {
            candidate_digest: "child".into(),
            descriptor,
            session_id: "session".into(),
        });
        runtime.record_descriptor(DescriptorRecord {
            candidate_digest: "child".into(),
            descriptor,
            session_id: "session".into(),
        });
        let receipt = EvolutionReceipt {
            parent_digest: "parent".into(),
            child_digest: "child".into(),
            operator: VariationOperator::Geometry,
            context: EvolutionContextKey {
                hardware: "m1".into(),
                model_family: "moe".into(),
                tensor_family: "attention".into(),
            },
            descriptor: BehaviorDescriptor::from_genome(&CandidateGenome::new()),
            objectives: crate::evolution::objectives::ObjectiveVector::new(vec![
                crate::evolution::objectives::ObjectiveValue::maximize("fitness", 0.8),
            ]),
            improvement: 0.1,
            measurement_receipt_digest: "measurement".into(),
        };
        runtime.record_receipt(receipt.clone());
        runtime.record_receipt(receipt);
        let counts = runtime.snapshot_counts();
        assert_eq!(counts.receipt_count, 1);
        assert_eq!(counts.lineage_count, 1);
        assert_eq!(counts.descriptor_count, 1);
    }

    #[test]
    fn contextual_policy_resolution_prefers_specific_context() {
        let runtime = EvolutionRuntime::new();
        let wildcard = EvolutionContextKey {
            hardware: "*".into(),
            model_family: "moe".into(),
            tensor_family: "*".into(),
        };
        let exact = EvolutionContextKey {
            hardware: "m1".into(),
            model_family: "moe".into(),
            tensor_family: "attention".into(),
        };
        runtime.record_contextual_emitter_reward(wildcard, EmitterKind::Memory, 1.0);
        runtime.record_contextual_emitter_reward(exact.clone(), EmitterKind::Failure, 1.0);
        runtime.record_contextual_operator_reward(exact.clone(), VariationOperator::Geometry, 0.4);
        let policy = runtime.emitter_policy_for_context(&EvolutionContextKey {
            hardware: "m1".into(),
            model_family: "moe".into(),
            tensor_family: "attention".into(),
        });
        assert_eq!(policy.stats[&EmitterKind::Failure].attempts, 1);
        assert_eq!(policy.stats[&EmitterKind::Memory].attempts, 0);
        assert_eq!(runtime.snapshot().contextual_operators.len(), 1);
    }

    #[test]
    fn archived_execution_elites_are_reusable() {
        let runtime = EvolutionRuntime::new();
        let genome = CandidateGenome::new();
        runtime.insert_execution_elite(ArchiveEntry {
            descriptor: BehaviorDescriptor::from_genome(&genome),
            genome: genome.clone(),
            objectives: crate::evolution::objectives::ObjectiveVector::new(vec![
                crate::evolution::objectives::ObjectiveValue::maximize("fitness", 0.9),
            ]),
            generation: 1,
            novelty: 1.0,
        });
        let elites = runtime.execution_elites(1);
        assert_eq!(elites.len(), 1);
        assert_eq!(
            crate::evolution::memory::genome_digest(&elites[0]),
            crate::evolution::memory::genome_digest(&genome)
        );
        runtime.insert_tensor_elite(
            "attention",
            ArchiveEntry {
                descriptor: BehaviorDescriptor::from_genome(&genome),
                genome: genome.clone(),
                objectives: crate::evolution::objectives::ObjectiveVector::new(vec![
                    crate::evolution::objectives::ObjectiveValue::maximize("fitness", 0.9),
                ]),
                generation: 1,
                novelty: 1.0,
            },
        );
        assert_eq!(runtime.tensor_elites("attention", 1).len(), 1);
    }

    #[test]
    fn receipt_surrogate_is_reconstructed_from_contextual_memory() {
        let runtime = EvolutionRuntime::new();
        let genome = CandidateGenome::new();
        let context = EvolutionContextKey {
            hardware: "m1".into(),
            model_family: "moe".into(),
            tensor_family: "attention".into(),
        };
        runtime.record_receipt(EvolutionReceipt {
            parent_digest: "parent".into(),
            child_digest: serde_json::to_string(&genome).unwrap(),
            operator: VariationOperator::Geometry,
            context: context.clone(),
            descriptor: BehaviorDescriptor::from_genome(&genome),
            objectives: crate::evolution::objectives::ObjectiveVector::new(vec![
                crate::evolution::objectives::ObjectiveValue::maximize("fitness", 0.9),
            ]),
            improvement: 0.2,
            measurement_receipt_digest: "measurement".into(),
        });

        let surrogate = runtime.receipt_surrogate(&context, 0.0, 8);
        assert_eq!(surrogate.observations.len(), 1);
        assert_eq!(
            surrogate
                .predict_with_uncertainty(&genome)
                .unwrap()
                .neighbors,
            1
        );
    }

    #[test]
    fn persistent_runtime_round_trips_knowledge() {
        let path = std::env::temp_dir().join(format!(
            "prism-evolution-runtime-{}-{}.json",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let first = EvolutionRuntime::new_persistent(&path).unwrap();
        first.record_emitter_reward(EmitterKind::Local, 0.75);
        let transport_payload = first.snapshot_bytes().unwrap();
        first.persist().unwrap();
        let second = EvolutionRuntime::new_persistent(&path).unwrap();
        assert_eq!(
            second.emitter_policy().stats[&EmitterKind::Local].attempts,
            1
        );
        second.load_snapshot(&path).unwrap();
        assert_eq!(
            second.emitter_policy().stats[&EmitterKind::Local].attempts,
            1
        );
        second.merge_snapshot_bytes(&transport_payload).unwrap();
        assert_eq!(
            second.emitter_policy().stats[&EmitterKind::Local].attempts,
            1
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn persistent_runtime_round_trips_archive_evidence() {
        let path = std::env::temp_dir().join(format!(
            "prism-evolution-archive-{}-{}.json",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let first = EvolutionRuntime::new_persistent(&path).unwrap();
        let genome = CandidateGenome::new();
        let descriptor = BehaviorDescriptor::from_genome(&genome);
        let objectives = crate::evolution::objectives::ObjectiveVector::new(vec![
            crate::evolution::objectives::ObjectiveValue::maximize("fidelity", 0.97),
            crate::evolution::objectives::ObjectiveValue::minimize("latency_ms", 2.5),
        ]);
        first.insert_execution_elite(ArchiveEntry {
            genome: genome.clone(),
            descriptor,
            objectives,
            generation: 7,
            novelty: 0.8,
        });
        first.persist().unwrap();

        let second = EvolutionRuntime::new_persistent(&path).unwrap();
        let entries = second.execution_archive_entries(1);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].generation, 7);
        assert_eq!(entries[0].descriptor, descriptor);
        assert_eq!(entries[0].objectives.values.len(), 2);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn federated_snapshot_bytes_merge_idempotently() {
        let source = EvolutionRuntime::new();
        source.record_emitter_reward(EmitterKind::Semantic, 0.6);
        let payload = source.snapshot_bytes().unwrap();

        let target = EvolutionRuntime::new();
        target.merge_snapshot_bytes(&payload).unwrap();
        target.merge_snapshot_bytes(&payload).unwrap();
        assert_eq!(
            target.emitter_policy().stats[&EmitterKind::Semantic].attempts,
            1
        );
    }
}
