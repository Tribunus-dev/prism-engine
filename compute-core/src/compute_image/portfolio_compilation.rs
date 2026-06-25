//! Core ML portfolio compilation — compiles a set of Core ML islands across
//! shape buckets into a portfolio of deployable .mlpackage artifacts.
//!
//! PRISM-COREML-PORTFOLIO-COMPILATION-0001

use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use crate::compilation::activation_abi::{ActivationAbi, ActivationContract, PhysicalLayout};
use crate::compilation::ane_eligibility::{ShapeBucket, ShapeBucketFamily};
use crate::compilation::region_planner::CoreMlIsland;

// ── Public types ──────────────────────────────────────────────────────────

/// Kind of neural-network packet being compiled.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PacketKind {
    MlpGateUp,
    MlpDown,
    ResidualNorm,
    MultimodalProjector,
    VisionEncoderBlock,
}

/// Encoding scheme for model weights.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightEncoding {
    Float16,
    Palette4Bit,
    Palette6Bit,
    Palette8Bit,
}

/// Minimum OS and Core ML runtime version for a deployment target.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeploymentTarget {
    pub minimum_os: String,
    pub coreml_version: String,
}

/// Uniquely identifies a single compiled Core ML packet within a portfolio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreMlArtifactKey {
    pub model_identity: String,
    pub packet_kind: PacketKind,
    pub layer_start: u32,
    pub layer_end: u32,
    pub function_name: String,
    pub shape_bucket: ShapeBucket,
    pub input_abi: ActivationAbi,
    pub output_abi: ActivationAbi,
    pub weight_encoding: WeightEncoding,
    pub source_package_digest: String,
}

/// A single compiled .mlpackage artifact in the portfolio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreMlPacketArtifact {
    pub packet_key: CoreMlArtifactKey,
    pub mlpackage_path: PathBuf,
    pub compiled_modelc_path: Option<PathBuf>,
    pub package_digest: String,
    pub weight_count: u32,
    pub parameter_count: u64,
    pub byte_size: u64,
    pub compile_latency_ms: u64,
    pub shape_bucket_count: u32,
    pub input_contract: ActivationContract,
    pub output_contract: ActivationContract,
    pub function_name: String,
    pub minimum_deployment_target: DeploymentTarget,
}

/// Request to compile one Core ML packet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlPacketCompilationRequest {
    pub island: CoreMlIsland,
    pub shape_buckets: Vec<ShapeBucket>,
    pub input_abi: ActivationAbi,
    pub output_abi: ActivationAbi,
    pub weight_encoding: WeightEncoding,
    pub model_identity: String,
    pub deployment_target: DeploymentTarget,
    pub max_package_bytes: u64,
}

/// Key used to qualify an ANE deployment against available hardware.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AneQualificationKey {
    pub artifact_key: CoreMlArtifactKey,
    pub hardware_identifier: String,
    pub os_build: String,
    pub coreml_runtime: String,
}

/// Inclusive start / end layer range.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerGroup {
    pub start: u32,
    pub end: u32,
}

// ── Public functions ──────────────────────────────────────────────────────

/// Build a deterministic function name for a packet.
///
/// The returned name is purely a function of its arguments: same inputs always
/// produce the same string.
pub fn build_function_name(
    packet_kind: &PacketKind,
    layer_start: u32,
    layer_end: u32,
    _abi: &ActivationAbi,
    seq_bucket: u32,
    precision: &str,
) -> String {
    let kind_str = match packet_kind {
        PacketKind::MlpGateUp => "mlp_gate_up",
        PacketKind::MlpDown => "mlp_down",
        PacketKind::ResidualNorm => "residual_norm",
        PacketKind::MultimodalProjector => "multimodal_projector",
        PacketKind::VisionEncoderBlock => "vision_encoder",
    };
    format!("{kind_str}_l{layer_start}-{layer_end}_seq{seq_bucket}_{precision}")
}

/// Stub: "compile" a Core ML packet, producing a synthetic artifact.
///
/// Real implementations will invoke the Core ML compiler toolchain; this stub
/// fills every field from the request and generates a placeholder digest.
pub fn compile_packet(
    request: CoreMlPacketCompilationRequest,
    output_dir: &std::path::Path,
) -> Result<CoreMlPacketArtifact, String> {
    // Determine layer bounds from the island ops (min/max layer indices).
    let layer_start = request
        .island
        .ops
        .iter()
        .map(|op| op.layer)
        .min()
        .unwrap_or(0);
    let layer_end = request
        .island
        .ops
        .iter()
        .map(|op| op.layer)
        .max()
        .unwrap_or(0);

    // Use the first shape bucket from the request.
    let bucket = request
        .shape_buckets
        .first()
        .cloned()
        .unwrap_or(ShapeBucket {
            batch: 1,
            sequence: 1,
            hidden: 1,
            rank: 1,
            family: ShapeBucketFamily::Decode,
        });

    let packet_kind = infer_packet_kind(&request.island);
    let seq_bucket = bucket.sequence;
    let precision = match &request.weight_encoding {
        WeightEncoding::Float16 => "fp16",
        WeightEncoding::Palette4Bit => "p4",
        WeightEncoding::Palette6Bit => "p6",
        WeightEncoding::Palette8Bit => "p8",
    };
    let function_name =
        build_function_name(&packet_kind, layer_start, layer_end, &request.input_abi, seq_bucket, precision);
    let mlpackage_path = output_dir.join(format!("{}.mlpackage", &function_name));

    // Build placeholder contracts from the ABIs.
    let input_contract = ActivationContract {
        abi: request.input_abi.clone(),
        element_count: 0,
        byte_count: 0,
        shape: vec![],
        stride: vec![],
        physical_layout: PhysicalLayout::ContiguousRowMajor,
        alignment: 64,
    };
    let output_contract = ActivationContract {
        abi: request.output_abi.clone(),
        element_count: 0,
        byte_count: 0,
        shape: vec![],
        stride: vec![],
        physical_layout: PhysicalLayout::ContiguousRowMajor,
        alignment: 64,
    };

    let package_digest = format!("placeholder-digest-{function_name}");

    Ok(CoreMlPacketArtifact {
        packet_key: CoreMlArtifactKey {
            model_identity: request.model_identity,
            packet_kind,
            layer_start,
            layer_end,
            function_name: function_name.clone(),
            shape_bucket: bucket,
            input_abi: request.input_abi,
            output_abi: request.output_abi,
            weight_encoding: request.weight_encoding,
            source_package_digest: package_digest.clone(),
        },
        mlpackage_path,
        compiled_modelc_path: None,
        package_digest,
        weight_count: 0,
        parameter_count: 0,
        byte_size: request.max_package_bytes,
        compile_latency_ms: 0,
        shape_bucket_count: request.shape_buckets.len() as u32,
        input_contract,
        output_contract,
        function_name,
        minimum_deployment_target: request.deployment_target,
    })
}

/// Compile a portfolio of Core ML artifacts across all islands × shape buckets.
///
/// Each (island, bucket) pair produces one `CoreMlPacketArtifact`. Failed
/// compilations are logged and skipped, so a partial portfolio is still
/// returned.
pub fn build_portfolio(
    islands: &[CoreMlIsland],
    buckets: &[ShapeBucket],
    abi: &ActivationAbi,
    output_dir: &std::path::Path,
) -> Vec<CoreMlPacketArtifact> {
    let mut artifacts = Vec::with_capacity(islands.len() * buckets.len());
    for island in islands {
        for bucket in buckets {
            let request = CoreMlPacketCompilationRequest {
                island: island.clone(),
                shape_buckets: vec![bucket.clone()],
                input_abi: abi.clone(),
                output_abi: abi.clone(),
                weight_encoding: WeightEncoding::Float16,
                model_identity: "default".to_string(),
                deployment_target: DeploymentTarget {
                    minimum_os: "14.0".to_string(),
                    coreml_version: "7".to_string(),
                },
                max_package_bytes: 1_073_741_824, // 1 GiB
            };
            match compile_packet(request, output_dir) {
                Ok(artifact) => artifacts.push(artifact),
                Err(e) => {
                    eprintln!("[portfolio_compilation] skipped island {}: {e}", island.island_id);
                }
            }
        }
    }
    artifacts
}

// ── Private helpers ───────────────────────────────────────────────────────

/// Infer the most likely `PacketKind` from the roles of ops in an island.
fn infer_packet_kind(island: &CoreMlIsland) -> PacketKind {
    // Check `role` first — it carries the semantically richest label.
    for op in &island.ops {
        match op.role.as_str() {
            "gate_proj" | "up_proj" => return PacketKind::MlpGateUp,
            "down_proj" => return PacketKind::MlpDown,
            "input_layernorm" | "post_attention_layernorm" | "rms_norm" => {
                return PacketKind::ResidualNorm;
            }
            "encoder_block" | "vision_encoder" => return PacketKind::VisionEncoderBlock,
            _ => {}
        }
    }
    // Fall back on the broader operator family.
    for op in &island.ops {
        match op.operator_family.as_str() {
            "attention" | "self_attn" => return PacketKind::ResidualNorm,
            "mlp" => return PacketKind::MlpGateUp,
            "projector" | "projection" => return PacketKind::MultimodalProjector,
            _ => {}
        }
    }
    PacketKind::ResidualNorm
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compilation::activation_abi::{
        ActivationAbi, ActivationContract, DecodeActivationV1Params, PhysicalLayout,
    };
    use crate::compilation::phase_ir::TensorDtype;
    use crate::compilation::ane_eligibility::ShapeBucketFamily;
    use crate::compilation::region_catalogue::RegionAdmission;
    use crate::compilation::region_planner::{CoreMlIsland, ScheduledOp};

    fn sample_shape_bucket() -> ShapeBucket {
        ShapeBucket {
            batch: 1,
            sequence: 128,
            hidden: 4096,
            rank: 1,
            family: ShapeBucketFamily::Decode,
        }
    }

    fn sample_abi() -> ActivationAbi {
        ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
            dtype: TensorDtype::Float16,
            seq_bucket: 128,
            hidden_dim: 4096,
            physical_layout: PhysicalLayout::ContiguousRowMajor,
            alignment: 64,
            stride_constraint: None,
        })
    }

    fn sample_island() -> CoreMlIsland {
        CoreMlIsland {
            island_id: 1,
            ops: vec![ScheduledOp {
                op_index: 0,
                layer: 0,
                operator_family: "mlp".into(),
                role: "gate_proj".into(),
                admission: RegionAdmission::CoreMlProduction,
                island_id: Some(1),
            }],
            input_slots: vec![0, 1],
            output_slots: vec![2],
        }
    }

    fn sample_island_vision() -> CoreMlIsland {
        CoreMlIsland {
            island_id: 2,
            ops: vec![ScheduledOp {
                op_index: 0,
                layer: 10,
                operator_family: "encoder".into(),
                role: "vision_encoder".into(),
                admission: RegionAdmission::CoreMlProduction,
                island_id: Some(2),
            }],
            input_slots: vec![0],
            output_slots: vec![1],
        }
    }

    // ── test_function_name_deterministic ───────────────────────────────────

    #[test]
    fn test_function_name_deterministic() {
        let kind = PacketKind::MlpGateUp;
        let abi = sample_abi();

        let a = build_function_name(&kind, 0, 3, &abi, 128, "fp16");
        let b = build_function_name(&kind, 0, 3, &abi, 128, "fp16");

        assert_eq!(a, b, "same inputs must produce identical names");
    }

    // ── test_function_name_uniqueness ──────────────────────────────────────

    #[test]
    fn test_function_name_uniqueness() {
        let abi = sample_abi();

        // Different precision
        let fp16 = build_function_name(&PacketKind::MlpGateUp, 0, 3, &abi, 128, "fp16");
        let p4 = build_function_name(&PacketKind::MlpGateUp, 0, 3, &abi, 128, "p4");
        assert_ne!(fp16, p4, "different precision must yield different names");

        // Different layer range
        let early = build_function_name(&PacketKind::MlpGateUp, 0, 3, &abi, 128, "fp16");
        let late = build_function_name(&PacketKind::MlpGateUp, 4, 7, &abi, 128, "fp16");
        assert_ne!(early, late, "different layer range must yield different names");

        // Different packet kind
        let gate_up = build_function_name(&PacketKind::MlpGateUp, 0, 3, &abi, 128, "fp16");
        let down = build_function_name(&PacketKind::MlpDown, 0, 3, &abi, 128, "fp16");
        assert_ne!(gate_up, down, "different packet kind must yield different names");
    }

    // ── test_compile_packet_produces_artifact ──────────────────────────────

    #[test]
    fn test_compile_packet_produces_artifact() {
        let island = sample_island();
        let bucket = sample_shape_bucket();
        let abi = sample_abi();

        let request = CoreMlPacketCompilationRequest {
            island,
            shape_buckets: vec![bucket],
            input_abi: abi.clone(),
            output_abi: abi.clone(),
            weight_encoding: WeightEncoding::Float16,
            model_identity: "test-model".into(),
            deployment_target: DeploymentTarget {
                minimum_os: "14.0".into(),
                coreml_version: "7".into(),
            },
            max_package_bytes: 512_000_000,
        };

        let tmp = std::env::temp_dir();
        let result = compile_packet(request, &tmp);

        assert!(result.is_ok(), "compile_packet should succeed");
        let artifact = result.unwrap();

        // Key fields should be populated from the request.
        assert_eq!(artifact.packet_key.model_identity, "test-model");
        assert_eq!(artifact.packet_key.packet_kind, PacketKind::MlpGateUp);
        assert_eq!(artifact.packet_key.layer_start, 0);
        assert_eq!(artifact.packet_key.layer_end, 0);
        assert_eq!(artifact.packet_key.weight_encoding, WeightEncoding::Float16);
        assert_eq!(artifact.minimum_deployment_target.minimum_os, "14.0");
        assert!(artifact.package_digest.starts_with("placeholder-digest-"));
        assert_eq!(artifact.byte_size, 512_000_000);
        assert_eq!(artifact.shape_bucket_count, 1);
        assert!(artifact.compiled_modelc_path.is_none());

        // The function_name should be deterministic.
        let expected_name = "mlp_gate_up_l0-0_seq128_fp16";
        assert_eq!(artifact.function_name, expected_name);
        assert!(artifact.mlpackage_path.to_string_lossy().ends_with("mlp_gate_up_l0-0_seq128_fp16.mlpackage"));
    }

    // ── test_build_portfolio_multiple_buckets ──────────────────────────────

    #[test]
    fn test_build_portfolio_multiple_buckets() {
        let islands = vec![sample_island(), sample_island_vision()];
        let buckets = vec![
            ShapeBucket {
                batch: 1,
                sequence: 64,
                hidden: 4096,
                rank: 1,
                family: ShapeBucketFamily::Decode,
            },
            ShapeBucket {
                batch: 1,
                sequence: 128,
                hidden: 4096,
                rank: 1,
                family: ShapeBucketFamily::Decode,
            },
        ];
        let abi = sample_abi();
        let tmp = std::env::temp_dir();

        let portfolio = build_portfolio(&islands, &buckets, &abi, &tmp);

        // 2 islands × 2 buckets = 4 artifacts
        assert_eq!(portfolio.len(), 4, "should produce one artifact per island×bucket");

        // Every artifact should have a valid digest and non-zero size.
        for artifact in &portfolio {
            assert!(
                artifact.package_digest.starts_with("placeholder-digest-"),
                "all artifacts need a digest"
            );
            assert_eq!(artifact.byte_size, 1_073_741_824, "default max_package_bytes");
            assert!(artifact.compiled_modelc_path.is_none());
        }

        // The MlpGateUp (island 1) vs VisionEncoderBlock (island 2) should
        // produce different function names for the same bucket.
        let names: Vec<&str> = portfolio.iter().map(|a| a.function_name.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("mlp_gate_up")));
        assert!(names.iter().any(|n| n.contains("vision_encoder")));
    }

    // ── test_serde_roundtrip ───────────────────────────────────────────────

    #[test]
    fn test_serde_roundtrip() {
        let key = CoreMlArtifactKey {
            model_identity: "test".into(),
            packet_kind: PacketKind::ResidualNorm,
            layer_start: 4,
            layer_end: 7,
            function_name: "residual_norm_l4-7_seq128_fp16".into(),
            shape_bucket: sample_shape_bucket(),
            input_abi: sample_abi(),
            output_abi: sample_abi(),
            weight_encoding: WeightEncoding::Palette4Bit,
            source_package_digest: "abc123".into(),
        };

        let json = serde_json::to_string(&key).expect("serialize");
        let deserialized: CoreMlArtifactKey = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(key, deserialized, "roundtrip must preserve value");

        // Also round-trip a full artifact through JSON.
        let artifact = CoreMlPacketArtifact {
            packet_key: key,
            mlpackage_path: PathBuf::from("/tmp/test.mlpackage"),
            compiled_modelc_path: None,
            package_digest: "digest-xyz".into(),
            weight_count: 42,
            parameter_count: 1_000_000,
            byte_size: 256_000_000,
            compile_latency_ms: 3_500,
            shape_bucket_count: 1,
            input_contract: ActivationContract {
                abi: sample_abi(),
                element_count: 1,
                byte_count: 8,
                shape: vec![1, 128, 4096],
                stride: vec![],
                physical_layout: PhysicalLayout::ContiguousRowMajor,
                alignment: 64,
            },
            output_contract: ActivationContract {
                abi: sample_abi(),
                element_count: 1,
                byte_count: 8,
                shape: vec![1, 128, 4096],
                stride: vec![],
                physical_layout: PhysicalLayout::ContiguousRowMajor,
                alignment: 64,
            },
            function_name: "residual_norm_l4-7_seq128_fp16".into(),
            minimum_deployment_target: DeploymentTarget {
                minimum_os: "15.0".into(),
                coreml_version: "8".into(),
            },
        };

        let json2 = serde_json::to_string(&artifact).expect("serialize artifact");
        let deserialized2: CoreMlPacketArtifact = serde_json::from_str(&json2).expect("deserialize artifact");
        assert_eq!(artifact, deserialized2, "full artifact roundtrip must preserve value");
    }
}
