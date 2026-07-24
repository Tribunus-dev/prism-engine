//! Deployment-level multi-objective optimization contracts.
//!
//! The older evolution modules score genomes while they are being searched.
//! This module is the durable boundary for candidates that have actually been
//! compiled, measured, and admitted for a deployment target.

use crate::evolution::{CandidateGenome, ObjectiveDirection, ObjectiveValue, ObjectiveVector};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeploymentIdentity {
    pub model_digest: String,
    pub tokenizer_digest: String,
    pub engram_artifact: Option<String>,
    pub target: String,
    pub workload_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GateStatus {
    Passed,
    Failed,
    Unmeasured,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HardGate {
    pub name: String,
    pub status: GateStatus,
    pub observed: Option<f64>,
    pub limit: Option<f64>,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeploymentGatePolicy {
    pub min_quality: Option<f64>,
    pub max_p99_latency_ms: Option<f64>,
    pub max_peak_memory_bytes: Option<u64>,
    pub require_measurements: bool,
}

impl DeploymentGatePolicy {
    pub fn evaluate(&self, measurements: &DeploymentMeasurements) -> Vec<HardGate> {
        let mut gates = Vec::new();
        let mut gate =
            |name: &str, observed: Option<f64>, limit: Option<f64>, passed: bool, detail: &str| {
                gates.push(HardGate {
                    name: name.into(),
                    status: if passed {
                        GateStatus::Passed
                    } else {
                        GateStatus::Failed
                    },
                    observed,
                    limit,
                    detail: detail.into(),
                });
            };
        if self.require_measurements {
            let required_present = self.min_quality.is_none_or(|min_quality| {
                measurements
                    .quality
                    .is_some_and(|value| value >= min_quality)
            }) && self.max_p99_latency_ms.is_none_or(|max_latency| {
                measurements
                    .p99_latency_ms
                    .is_some_and(|value| value <= max_latency)
            }) && self.max_peak_memory_bytes.is_none_or(|max_memory| {
                measurements
                    .peak_memory_bytes
                    .is_some_and(|value| value <= max_memory)
            });
            gate(
                "measurement_presence",
                measurements.quality,
                Some(1.0),
                required_present,
                "required measurements must be present and satisfy configured policy limits",
            );
        }
        if let Some(limit) = self.min_quality {
            gate(
                "quality",
                measurements.quality,
                Some(limit),
                measurements.quality.is_some_and(|value| value >= limit),
                "quality floor",
            );
        }
        if let Some(limit) = self.max_p99_latency_ms {
            gate(
                "p99_latency",
                measurements.p99_latency_ms,
                Some(limit),
                measurements
                    .p99_latency_ms
                    .is_some_and(|value| value <= limit),
                "latency ceiling",
            );
        }
        if let Some(limit) = self.max_peak_memory_bytes {
            gate(
                "peak_memory",
                measurements.peak_memory_bytes.map(|value| value as f64),
                Some(limit as f64),
                measurements
                    .peak_memory_bytes
                    .is_some_and(|value| value <= limit),
                "memory ceiling",
            );
        }
        gates
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeploymentMeasurements {
    pub quality: Option<f64>,
    pub p50_latency_ms: Option<f64>,
    pub p99_latency_ms: Option<f64>,
    pub throughput_tokens_per_second: Option<f64>,
    pub peak_memory_bytes: Option<u64>,
    pub kv_memory_bytes: Option<u64>,
    pub power_watts: Option<f64>,
    pub transfer_bytes: Option<u64>,
    pub engram_residency_bytes: Option<u64>,
    pub engram_lookup_latency_ms: Option<f64>,
    pub engram_hit_rate: Option<f64>,
}

impl DeploymentMeasurements {
    pub fn objectives(&self) -> ObjectiveVector {
        let mut values = Vec::new();
        let add = |values: &mut Vec<ObjectiveValue>, name: &str, value: Option<f64>, direction| {
            if let Some(value) = value.filter(|value| value.is_finite()) {
                values.push(ObjectiveValue {
                    name: name.into(),
                    value,
                    direction,
                });
            }
        };
        add(
            &mut values,
            "quality",
            self.quality,
            ObjectiveDirection::Maximize,
        );
        add(
            &mut values,
            "p99_latency_ms",
            self.p99_latency_ms,
            ObjectiveDirection::Minimize,
        );
        add(
            &mut values,
            "peak_memory_bytes",
            self.peak_memory_bytes.map(|v| v as f64),
            ObjectiveDirection::Minimize,
        );
        add(
            &mut values,
            "throughput_tokens_per_second",
            self.throughput_tokens_per_second,
            ObjectiveDirection::Maximize,
        );
        add(
            &mut values,
            "engram_lookup_latency_ms",
            self.engram_lookup_latency_ms,
            ObjectiveDirection::Minimize,
        );
        add(
            &mut values,
            "engram_hit_rate",
            self.engram_hit_rate,
            ObjectiveDirection::Maximize,
        );
        ObjectiveVector::new(values)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeploymentEvidence {
    pub candidate_digest: String,
    pub cimage_digest: Option<String>,
    pub compiler_version: String,
    pub backend_version: String,
    pub measurements: DeploymentMeasurements,
    pub gates: Vec<HardGate>,
    pub receipt_ids: Vec<String>,
}

impl DeploymentEvidence {
    pub fn admitted(&self) -> bool {
        !self.gates.is_empty() && self.gates.iter().all(|g| g.status == GateStatus::Passed)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentCandidate {
    pub candidate_digest: String,
    pub identity: DeploymentIdentity,
    pub genome: CandidateGenome,
    pub generation: u64,
    pub evidence: DeploymentEvidence,
}

impl DeploymentCandidate {
    pub fn new(identity: DeploymentIdentity, genome: CandidateGenome, generation: u64) -> Self {
        let bytes = serde_json::to_vec(&(identity.clone(), &genome, generation))
            .expect("candidate is serializable");
        let digest = format!("sha256:{}", hex_digest(&bytes));
        Self {
            candidate_digest: digest.clone(),
            identity,
            genome,
            generation,
            evidence: DeploymentEvidence {
                candidate_digest: digest,
                cimage_digest: None,
                compiler_version: String::new(),
                backend_version: String::new(),
                measurements: DeploymentMeasurements {
                    quality: None,
                    p50_latency_ms: None,
                    p99_latency_ms: None,
                    throughput_tokens_per_second: None,
                    peak_memory_bytes: None,
                    kv_memory_bytes: None,
                    power_watts: None,
                    transfer_bytes: None,
                    engram_residency_bytes: None,
                    engram_lookup_latency_ms: None,
                    engram_hit_rate: None,
                },
                gates: Vec::new(),
                receipt_ids: Vec::new(),
            },
        }
    }

    pub fn objectives(&self) -> ObjectiveVector {
        self.evidence.measurements.objectives()
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParetoArchive {
    pub candidates: BTreeMap<String, DeploymentCandidate>,
}

impl ParetoArchive {
    /// Serialize the archive into a stable payload suitable for a
    /// content-addressed artifact store.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("ParetoArchive is serializable")
    }

    pub fn content_digest(&self) -> String {
        let digest = Sha256::digest(self.canonical_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    pub fn insert(&mut self, candidate: DeploymentCandidate) -> bool {
        if !candidate.evidence.admitted()
            || self
                .candidates
                .values()
                .any(|c| c.objectives().dominates(&candidate.objectives()))
        {
            return false;
        }
        let dominated: Vec<String> = self
            .candidates
            .iter()
            .filter_map(|(id, c)| {
                if candidate.objectives().dominates(&c.objectives()) {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();
        for id in dominated {
            self.candidates.remove(&id);
        }
        self.candidates
            .insert(candidate.candidate_digest.clone(), candidate);
        true
    }

    pub fn select(&self, policy: &DeploymentPolicy) -> Option<&DeploymentCandidate> {
        self.candidates
            .values()
            .filter(|c| policy.accepts(&c.evidence.measurements))
            .max_by(|a, b| {
                policy
                    .score(&a.evidence.measurements)
                    .total_cmp(&policy.score(&b.evidence.measurements))
            })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeploymentPolicy {
    pub max_p99_latency_ms: Option<f64>,
    pub max_peak_memory_bytes: Option<u64>,
    pub min_quality: Option<f64>,
    pub min_throughput: Option<f64>,
    pub priorities: Vec<(String, ObjectiveDirection, f64)>,
}

impl DeploymentPolicy {
    pub fn quality_first() -> Self {
        Self {
            priorities: vec![
                ("quality".into(), ObjectiveDirection::Maximize, 1.0),
                ("p99_latency_ms".into(), ObjectiveDirection::Minimize, 0.01),
            ],
            ..Default::default()
        }
    }

    pub fn accepts(&self, m: &DeploymentMeasurements) -> bool {
        self.max_p99_latency_ms
            .is_none_or(|v| m.p99_latency_ms.is_some_and(|x| x <= v))
            && self
                .max_peak_memory_bytes
                .is_none_or(|v| m.peak_memory_bytes.is_some_and(|x| x <= v))
            && self
                .min_quality
                .is_none_or(|v| m.quality.is_some_and(|x| x >= v))
            && self
                .min_throughput
                .is_none_or(|v| m.throughput_tokens_per_second.is_some_and(|x| x >= v))
    }

    fn score(&self, m: &DeploymentMeasurements) -> f64 {
        self.priorities
            .iter()
            .map(|(name, direction, weight)| {
                let v = match name.as_str() {
                    "quality" => m.quality,
                    "p99_latency_ms" => m.p99_latency_ms,
                    "peak_memory_bytes" => m.peak_memory_bytes.map(|x| x as f64),
                    "throughput_tokens_per_second" => m.throughput_tokens_per_second,
                    _ => None,
                }
                .unwrap_or(f64::NEG_INFINITY);
                weight
                    * if *direction == ObjectiveDirection::Maximize {
                        v
                    } else {
                        -v
                    }
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::foundation::RepresentationAxis;
    fn candidate(q: f64, latency: f64) -> DeploymentCandidate {
        let mut c = DeploymentCandidate::new(
            DeploymentIdentity {
                model_digest: "m".into(),
                tokenizer_digest: "t".into(),
                engram_artifact: None,
                target: "cpu".into(),
                workload_digest: "w".into(),
            },
            CandidateGenome::new(),
            0,
        );
        c.evidence.measurements.quality = Some(q);
        c.evidence.measurements.p99_latency_ms = Some(latency);
        c.evidence.gates.push(HardGate {
            name: "quality".into(),
            status: GateStatus::Passed,
            observed: Some(q),
            limit: None,
            detail: String::new(),
        });
        c
    }
    #[test]
    fn archive_preserves_tradeoffs() {
        let mut a = ParetoArchive::default();
        assert!(a.insert(candidate(0.9, 10.)));
        let mut faster = candidate(0.8, 5.);
        faster.genome.representation = RepresentationAxis::Int8;
        faster.candidate_digest = DeploymentCandidate::new(
            faster.identity.clone(),
            faster.genome.clone(),
            faster.generation,
        )
        .candidate_digest;
        faster.evidence.candidate_digest = faster.candidate_digest.clone();
        assert!(a.insert(faster));
        assert_eq!(a.candidates.len(), 2);
    }
    #[test]
    fn policy_selects_valid_candidate() {
        let mut a = ParetoArchive::default();
        a.insert(candidate(0.9, 10.));
        a.insert(candidate(0.8, 5.));
        let p = DeploymentPolicy {
            max_p99_latency_ms: Some(6.),
            priorities: vec![("quality".into(), ObjectiveDirection::Maximize, 1.)],
            ..Default::default()
        };
        assert_eq!(
            a.select(&p).unwrap().evidence.measurements.quality,
            Some(0.8)
        );
    }
    #[test]
    fn archive_round_trips_with_stable_digest() {
        let mut a = ParetoArchive::default();
        a.insert(candidate(0.9, 10.));
        let bytes = a.canonical_bytes();
        let restored = ParetoArchive::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(a.content_digest(), restored.content_digest());
    }
    #[test]
    fn gate_policy_rejects_quality_latency_and_memory_violations() {
        let policy = DeploymentGatePolicy {
            min_quality: Some(0.9),
            max_p99_latency_ms: Some(5.0),
            max_peak_memory_bytes: Some(100),
            require_measurements: true,
        };
        let measurements = DeploymentMeasurements {
            quality: Some(0.8),
            p50_latency_ms: Some(8.0),
            p99_latency_ms: Some(8.0),
            throughput_tokens_per_second: None,
            peak_memory_bytes: Some(200),
            kv_memory_bytes: None,
            power_watts: None,
            transfer_bytes: None,
            engram_residency_bytes: None,
            engram_lookup_latency_ms: None,
            engram_hit_rate: None,
        };
        let gates = policy.evaluate(&measurements);
        assert_eq!(gates.len(), 4);
        assert!(gates.iter().all(|gate| gate.status == GateStatus::Failed));
    }
}
