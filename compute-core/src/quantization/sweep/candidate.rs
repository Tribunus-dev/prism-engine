//! Sweep receipt types — per-candidate results, per-class policy selection,
//! and scoring defaults.

use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::quantization::contract::{TensorClass, WeightValidationReport};
use crate::quantization::sweep::spec::{OverlayMode, QuantPolicy, RescueGranularity, RescueSchedule, RescueSelector, SweepScoringConfig};
use crate::quantization::sweep::SweepCandidateStatus;

// ── QuantSweepReceipt ───────────────────────────────────────────────────────────

/// Immutable record of one candidate run within a sweep.
#[derive(Debug, Clone)]
pub struct QuantSweepReceipt {
    pub receipt_version: u16,
    pub run_id: String,
    pub tensor_key: String,
    pub tensor_class: TensorClass,
    pub source_shape: Vec<usize>,
    pub family: String,
    pub parameters: serde_json::Value,
    pub code_bytes: u64,
    pub metadata_bytes: u64,
    pub total_bytes: u64,
    pub compression_ratio_vs_f32: f64,
    pub weight: WeightValidationReport,
    pub status: SweepCandidateStatus,
    pub score: f64,
    pub wall_ms: u64,
}

// ── Manual serde for QuantSweepReceipt ──────────────────────────────────────────
// TensorClass and WeightValidationReport lack serde derives in contract.rs,
// so we implement serde manually for the receipt and per-class policy.

impl Serialize for QuantSweepReceipt {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("QuantSweepReceipt", 15)?;
        s.serialize_field("receipt_version", &self.receipt_version)?;
        s.serialize_field("run_id", &self.run_id)?;
        s.serialize_field("tensor_key", &self.tensor_key)?;
        s.serialize_field("tensor_class", &tensor_class_name(&self.tensor_class))?;
        s.serialize_field("source_shape", &self.source_shape)?;
        s.serialize_field("family", &self.family)?;
        s.serialize_field("parameters", &self.parameters)?;
        s.serialize_field("code_bytes", &self.code_bytes)?;
        s.serialize_field("metadata_bytes", &self.metadata_bytes)?;
        s.serialize_field("total_bytes", &self.total_bytes)?;
        s.serialize_field("compression_ratio_vs_f32", &self.compression_ratio_vs_f32)?;
        s.serialize_field("weight", &WeightReportHelper::from(&self.weight))?;
        s.serialize_field("status", &self.status)?;
        s.serialize_field("score", &self.score)?;
        s.serialize_field("wall_ms", &self.wall_ms)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for QuantSweepReceipt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            ReceiptVersion,
            RunId,
            TensorKey,
            TensorClass,
            SourceShape,
            Family,
            Parameters,
            CodeBytes,
            MetadataBytes,
            TotalBytes,
            CompressionRatioVsF32,
            Weight,
            Status,
            Score,
            WallMs,
        }

        use serde::de::{self, MapAccess, Visitor};

        struct ReceiptVisitor;

        impl<'de> Visitor<'de> for ReceiptVisitor {
            type Value = QuantSweepReceipt;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("struct QuantSweepReceipt")
            }

            fn visit_map<V>(self, mut map: V) -> Result<QuantSweepReceipt, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut receipt_version = None::<u16>;
                let mut run_id = None::<String>;
                let mut tensor_key = None::<String>;
                let mut tensor_class = None::<TensorClass>;
                let mut source_shape = None::<Vec<usize>>;
                let mut family = None::<String>;
                let mut parameters = None::<serde_json::Value>;
                let mut code_bytes = None::<u64>;
                let mut metadata_bytes = None::<u64>;
                let mut total_bytes = None::<u64>;
                let mut compression_ratio_vs_f32 = None::<f64>;
                let mut weight = None::<WeightValidationReport>;
                let mut status = None::<SweepCandidateStatus>;
                let mut score = None::<f64>;
                let mut wall_ms = None::<u64>;

                while let Some(key) = map.next_key::<Field>()? {
                    match key {
                        Field::ReceiptVersion => {
                            receipt_version = Some(map.next_value()?);
                        }
                        Field::RunId => {
                            run_id = Some(map.next_value()?);
                        }
                        Field::TensorKey => {
                            tensor_key = Some(map.next_value()?);
                        }
                        Field::TensorClass => {
                            let name: String = map.next_value()?;
                            tensor_class =
                                Some(parse_tensor_class(&name).map_err(de::Error::custom)?);
                        }
                        Field::SourceShape => {
                            source_shape = Some(map.next_value()?);
                        }
                        Field::Family => {
                            family = Some(map.next_value()?);
                        }
                        Field::Parameters => {
                            parameters = Some(map.next_value()?);
                        }
                        Field::CodeBytes => {
                            code_bytes = Some(map.next_value()?);
                        }
                        Field::MetadataBytes => {
                            metadata_bytes = Some(map.next_value()?);
                        }
                        Field::TotalBytes => {
                            total_bytes = Some(map.next_value()?);
                        }
                        Field::CompressionRatioVsF32 => {
                            compression_ratio_vs_f32 = Some(map.next_value()?);
                        }
                        Field::Weight => {
                            let helper: WeightReportHelper = map.next_value()?;
                            weight = Some(helper.into_report());
                        }
                        Field::Status => {
                            status = Some(map.next_value()?);
                        }
                        Field::Score => {
                            score = Some(map.next_value()?);
                        }
                        Field::WallMs => {
                            wall_ms = Some(map.next_value()?);
                        }
                    }
                }

                let receipt_version =
                    receipt_version.ok_or_else(|| de::Error::missing_field("receipt_version"))?;
                let run_id = run_id.ok_or_else(|| de::Error::missing_field("run_id"))?;
                let tensor_key =
                    tensor_key.ok_or_else(|| de::Error::missing_field("tensor_key"))?;
                let tensor_class =
                    tensor_class.ok_or_else(|| de::Error::missing_field("tensor_class"))?;
                let source_shape =
                    source_shape.ok_or_else(|| de::Error::missing_field("source_shape"))?;
                let family = family.ok_or_else(|| de::Error::missing_field("family"))?;
                let parameters =
                    parameters.ok_or_else(|| de::Error::missing_field("parameters"))?;
                let code_bytes =
                    code_bytes.ok_or_else(|| de::Error::missing_field("code_bytes"))?;
                let metadata_bytes =
                    metadata_bytes.ok_or_else(|| de::Error::missing_field("metadata_bytes"))?;
                let total_bytes =
                    total_bytes.ok_or_else(|| de::Error::missing_field("total_bytes"))?;
                let compression_ratio_vs_f32 = compression_ratio_vs_f32
                    .ok_or_else(|| de::Error::missing_field("compression_ratio_vs_f32"))?;
                let weight = weight.ok_or_else(|| de::Error::missing_field("weight"))?;
                let status = status.ok_or_else(|| de::Error::missing_field("status"))?;
                let score = score.ok_or_else(|| de::Error::missing_field("score"))?;
                let wall_ms = wall_ms.ok_or_else(|| de::Error::missing_field("wall_ms"))?;

                Ok(QuantSweepReceipt {
                    receipt_version,
                    run_id,
                    tensor_key,
                    tensor_class,
                    source_shape,
                    family,
                    parameters,
                    code_bytes,
                    metadata_bytes,
                    total_bytes,
                    compression_ratio_vs_f32,
                    weight,
                    status,
                    score,
                    wall_ms,
                })
            }
        }

        deserializer.deserialize_struct(
            "QuantSweepReceipt",
            &[
                "receipt_version",
                "run_id",
                "tensor_key",
                "tensor_class",
                "source_shape",
                "family",
                "parameters",
                "code_bytes",
                "metadata_bytes",
                "total_bytes",
                "compression_ratio_vs_f32",
                "weight",
                "status",
                "score",
                "wall_ms",
            ],
            ReceiptVisitor,
        )
    }
}

// ── PerClassPolicy ──────────────────────────────────────────────────────────────

/// Stratified policy for one tensor class, derived from the best candidates.
#[derive(Debug, Clone)]
pub struct PerClassPolicy {
    pub tensor_class: TensorClass,
    pub preferred: Vec<FamilyPolicyEntry>,
    pub fallback: String,
}

impl Serialize for PerClassPolicy {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("PerClassPolicy", 3)?;
        s.serialize_field("tensor_class", &tensor_class_name(&self.tensor_class))?;
        s.serialize_field("preferred", &self.preferred)?;
        s.serialize_field("fallback", &self.fallback)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for PerClassPolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            TensorClass,
            Preferred,
            Fallback,
        }

        use serde::de::{self, MapAccess, Visitor};

        struct PolicyVisitor;

        impl<'de> Visitor<'de> for PolicyVisitor {
            type Value = PerClassPolicy;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("struct PerClassPolicy")
            }

            fn visit_map<V>(self, mut map: V) -> Result<PerClassPolicy, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut tensor_class = None::<TensorClass>;
                let mut preferred = None::<Vec<FamilyPolicyEntry>>;
                let mut fallback = None::<String>;

                while let Some(key) = map.next_key::<Field>()? {
                    match key {
                        Field::TensorClass => {
                            let name: String = map.next_value()?;
                            tensor_class =
                                Some(parse_tensor_class(&name).map_err(de::Error::custom)?);
                        }
                        Field::Preferred => {
                            preferred = Some(map.next_value()?);
                        }
                        Field::Fallback => {
                            fallback = Some(map.next_value()?);
                        }
                    }
                }

                let tensor_class =
                    tensor_class.ok_or_else(|| de::Error::missing_field("tensor_class"))?;
                let preferred =
                    preferred.ok_or_else(|| de::Error::missing_field("preferred"))?;
                let fallback =
                    fallback.ok_or_else(|| de::Error::missing_field("fallback"))?;

                Ok(PerClassPolicy {
                    tensor_class,
                    preferred,
                    fallback,
                })
            }
        }

        deserializer.deserialize_struct(
            "PerClassPolicy",
            &["tensor_class", "preferred", "fallback"],
            PolicyVisitor,
        )
    }
}

// ── FamilyPolicyEntry ───────────────────────────────────────────────────────────

/// One candidate entry in a per-class policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FamilyPolicyEntry {
    pub family: String,
    pub parameters: serde_json::Value,
    pub weight_nrmse: f64,
    pub score: f64,
    pub total_bytes: u64,
}

// ── Mixed-tile trajectory receipts ─────────────────────────────────────────

/// Receipt for one round of a mixed-tile iterative rescue trajectory.
#[derive(Debug, Clone)]
pub struct MixedTileRoundReceipt {
    pub round_index: u32,
    pub rescued_units_this_round: u32,
    pub rescued_units_total: u32,
    pub rescued_fraction_total: f64,
    pub base_code_bytes: u64,
    pub rescue_code_bytes: u64,
    pub metadata_bytes: u64,
    pub total_bytes: u64,
    pub weight_nrmse: f64,
    pub max_abs_error: f64,
    pub zero_collapse_ratio: f64,
    pub marginal_quality_gain: f64,
    pub marginal_bytes_added: u64,
    pub gain_per_kib: f64,
}

/// Full trajectory receipt for a mixed-tile iterative rescue run.
#[derive(Debug, Clone)]
pub struct MixedTileTrajectoryReceipt {
    pub tensor_key: String,
    pub tensor_class: TensorClass,
    pub base_policy: QuantPolicy,
    pub rescue_policy: String,
    pub rescue_granularity: RescueGranularity,
    pub selector: RescueSelector,
    pub schedule: RescueSchedule,
    pub overlay_mode: OverlayMode,
    pub rounds: Vec<MixedTileRoundReceipt>,
    pub final_score: f64,
    pub bytes_saved_vs_int8: i64,
    pub relative_to_int8_weight_nrmse: f64,
}

// ── SweepScoringConfig defaults ─────────────────────────────────────────────────

impl Default for SweepScoringConfig {
    fn default() -> Self {
        let mut max_weight_nrmse_by_family = HashMap::new();
        max_weight_nrmse_by_family.insert("Nf4Tile640".to_string(), 0.15);
        max_weight_nrmse_by_family.insert("SymInt4Tile640".to_string(), 0.15);
        max_weight_nrmse_by_family.insert("Int8Tile640".to_string(), 0.02);
        max_weight_nrmse_by_family.insert("TernaryTile640".to_string(), 0.90);
        max_weight_nrmse_by_family.insert("MixedTile".to_string(), 0.10);
        SweepScoringConfig {
            max_weight_nrmse_by_family,
            max_zero_collapse: 0.01,
            byte_weight: 0.3,
        }
    }
}

// ── Scoring function ────────────────────────────────────────────────────────────

/// Compute a scalar score for a candidate receipt.
///
/// Lower is better. The score blends weight-space quality against byte cost:
///
/// score = (1 - min(1, nrmse / max_nrmse_for_family))
///        - byte_weight * min(1, bytes_per_elem / 4.0)
pub fn score_receipt(receipt: &QuantSweepReceipt, config: &SweepScoringConfig) -> f64 {
    let max_nrmse = config
        .max_weight_nrmse_by_family
        .get(&receipt.family)
        .copied()
        .unwrap_or(1.0);
    let quality_score = 1.0 - (receipt.weight.nrmse / max_nrmse).min(1.0);

    let total_elements_f64: f64 = receipt.source_shape.iter().product::<usize>() as f64;
    let bytes_per_elem = if total_elements_f64 > 0.0 {
        receipt.total_bytes as f64 / total_elements_f64
    } else {
        4.0
    };
    let size_penalty = config.byte_weight * (bytes_per_elem / 4.0).min(1.0);

    quality_score - size_penalty
}

// ── Serde helpers ───────────────────────────────────────────────────────────────

/// Convert a `TensorClass` variant to its string name for serialization.
fn tensor_class_name(tc: &TensorClass) -> &'static str {
    match tc {
        TensorClass::DecoderAttentionProjection => "DecoderAttentionProjection",
        TensorClass::DecoderMlpProjection => "DecoderMlpProjection",
        TensorClass::TokenEmbedding => "TokenEmbedding",
        TensorClass::VisionPatchProjection => "VisionPatchProjection",
        TensorClass::CrossModalBridge => "CrossModalBridge",
        TensorClass::OutputHead => "OutputHead",
        TensorClass::Unknown => "Unknown",
    }
}

/// Parse a `TensorClass` from its string name.
fn parse_tensor_class(s: &str) -> Result<TensorClass, String> {
    match s {
        "DecoderAttentionProjection" => Ok(TensorClass::DecoderAttentionProjection),
        "DecoderMlpProjection" => Ok(TensorClass::DecoderMlpProjection),
        "TokenEmbedding" => Ok(TensorClass::TokenEmbedding),
        "VisionPatchProjection" => Ok(TensorClass::VisionPatchProjection),
        "CrossModalBridge" => Ok(TensorClass::CrossModalBridge),
        "OutputHead" => Ok(TensorClass::OutputHead),
        "Unknown" => Ok(TensorClass::Unknown),
        other => Err(format!("unknown TensorClass variant: {other}")),
    }
}

/// Serializable bridge for `WeightValidationReport`.
#[derive(Serialize, Deserialize)]
struct WeightReportHelper {
    rmse: f64,
    nrmse: f64,
    max_abs_error: f64,
    zero_collapse_ratio: f64,
}

impl WeightReportHelper {
    fn from(report: &WeightValidationReport) -> Self {
        Self {
            rmse: report.rmse,
            nrmse: report.nrmse,
            max_abs_error: report.max_abs_error,
            zero_collapse_ratio: report.zero_collapse_ratio,
        }
    }

    fn into_report(self) -> WeightValidationReport {
        WeightValidationReport {
            rmse: self.rmse,
            nrmse: self.nrmse,
            max_abs_error: self.max_abs_error,
            zero_collapse_ratio: self.zero_collapse_ratio,
        }
    }
}
