use crate::ecs::aot::parameters::KernelParameters;
use crate::ecs::aot::template::{MetalKernelTemplate, TemplateError};
use crate::ecs::component::backend::{BackendTarget, GPUArch, KernelSource};
use crate::ecs::component::fusion::FusionGroup;
use crate::ecs::component::tensor::{CodecFamilyComp, Shape};
use crate::ecs::plan::{CodecFamily, KernelTemplateId};

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};
use std::collections::{HashMap, HashSet};

/// Selects a kernel template for each dispatch based on its root op and codec.
///
/// Creates a `KernelEntity` per `DispatchEntity` that has a `FusionGroup`,
/// assigns the matching `KernelTemplateId`, and stores a stub `KernelSource`
/// component referencing the template name.
pub struct TemplateSelectionSystem;
impl CompilerSystem for TemplateSelectionSystem {
    fn name(&self) -> &str {
        "TemplateSelectionSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::KernelGeneration
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        // Collect Dispatch entities with FusionGroup data up front.
        let dispatch_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Dispatch);
        let payload: Vec<Entity> = dispatch_entities
            .into_iter()
            .filter(|e| world.get_component::<FusionGroup>(*e).is_some())
            .collect();

        if payload.is_empty() {
            return Ok(());
        }

        // Collect per-entity data while world is immutably borrowed.
        struct DispatchInfo {
            entity: Entity,
            root_op: String,
            codec: CodecFamily,
        }
        let infos: Vec<DispatchInfo> = payload
            .iter()
            .map(|e| {
                let group = world.get_component::<FusionGroup>(*e).unwrap();
                let codec = world
                    .get_component::<CodecFamilyComp>(*e)
                    .map(|c| c.0)
                    .unwrap_or(CodecFamily::RawF32);
                DispatchInfo {
                    entity: *e,
                    root_op: group.root_op_kind.clone(),
                    codec,
                }
            })
            .collect();

        // Spawn kernels and add components (mutable borrow).
        for info in &infos {
            let template_id = select_template(&info.root_op, info.codec);
            let kernel = world.spawn(EntityKind::Kernel, Some(template_id.name().into()))?;
            // Attach the template id via a string component for downstream lookups.
            let _ = world.add_component(kernel,
            KernelSource {
                language: crate::ecs::component::backend::ShaderLanguage::MSL,
                source: template_id.name().to_string(),
                entry_point: template_id.default_entry_point().to_string(),
            },);;
            // Propagate backend arch and GPU info from the dispatch.
            if let Some(arch) = world.get_component::<GPUArch>(info.entity) {
                let _ = world.add_component(kernel, arch.clone());;
            }
            if let Some(target) = world.get_component::<BackendTarget>(info.entity) {
                let _ = world.add_component(kernel, *target);;
            }
            if let Some(shape) = world.get_component::<Shape>(info.entity) {
                let _ = world.add_component(kernel, shape.clone());;
            }
        }

        Ok(())
    }
}

/// Resolves `KernelParameters` from the dispatch's `FusionGroup` and tensor
/// shapes, then attaches them to each `KernelEntity`.
pub struct ParameterResolutionSystem;
impl CompilerSystem for ParameterResolutionSystem {
    fn name(&self) -> &str {
        "ParameterResolutionSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::KernelGeneration
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        use crate::ecs::aot::parameters::KernelParameters;

        let kernel_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Kernel);
        if kernel_entities.is_empty() {
            return Ok(());
        }

        // Collect dispatch-level data for each kernel (via the kernel source entry point).
        struct KernelResolve {
            entity: Entity,
            shape: Option<crate::ecs::component::tensor::Shape>,
            source_name: String,
        }
        let resolves: Vec<KernelResolve> = kernel_entities
            .iter()
            .filter_map(|e| {
                let source = world.get_component::<KernelSource>(*e)?;
                let shape = world.get_component::<crate::ecs::component::tensor::Shape>(*e);
                Some(KernelResolve {
                    entity: *e,
                    shape: shape.cloned(),
                    source_name: source.source.clone(),
                })
            })
            .collect();

        for r in &resolves {
            let family = template_name_to_family(&r.source_name);
            // Determine tile width from shape or use a default (640 for tile-like kernels).
            let tile_width = r
                .shape
                .as_ref()
                .and_then(|s| s.0.first().copied())
                .unwrap_or(640);

            let params = KernelParameters {
                kernel_family: family,
                codec_family: CodecFamily::Nf4,
                tile_width,
                group_size: 32,
                threadgroup_size: 256,
                simdgroup_width: 32,
                groups_per_tile: tile_width / 32,
                lane_values: 4,
                unroll_factor: 2,
                use_threadgroup_memory: true,
                prefetch_distance: 2,
                accumulation_dtype: crate::ecs::aot::parameters::DType::Fp32,
                output_dtype: crate::ecs::aot::parameters::DType::Fp16,
            };
            let _ = world.add_component(r.entity, params);;
        }

        Ok(())
    }
}

/// Strict template expander. Rejects unknown placeholders and unexpanded
/// `{{...}}` patterns in the result.
#[derive(Debug)]
pub(crate) struct TemplateExpander;

impl TemplateExpander {
    pub fn expand(
        &self,
        template: &MetalKernelTemplate,
        params: &KernelParameters,
    ) -> Result<String, TemplateError> {
        let entries = params.to_placeholder_map();
        let mut map: HashMap<&str, &str> = HashMap::with_capacity(entries.len());
        for (k, v) in &entries {
            map.insert(k, v.as_str());
        }
        let known_keys: HashSet<&str> = map.keys().copied().collect();

        let mut result = String::with_capacity(template.source.len());
        let mut chars = template.source.chars().peekable();

        loop {
            match chars.next() {
                None => break,
                Some('{') => {
                    if chars.peek() == Some(&'{') {
                        chars.next(); // consume second '{'
                        let mut ph = String::new();
                        loop {
                            match chars.next() {
                                None => {
                                    return Err(TemplateError::MissingValue {
                                        template: template.template_id.clone(),
                                        placeholder: ph,
                                    });
                                }
                                Some('}') if chars.peek() == Some(&'}') => {
                                    chars.next(); // consume second '}'
                                    break;
                                }
                                Some(c) if c == '}' => {
                                    ph.push('}');
                                }
                                Some(c) => ph.push(c),
                            }
                        }

                        if let Some(value) = map.get(ph.as_str()) {
                            result.push_str(value);
                        } else if !known_keys.contains(ph.as_str()) {
                            return Err(TemplateError::UnknownPlaceholder {
                                template: template.template_id.clone(),
                                placeholder: ph,
                            });
                        } else {
                            // Known key but no value: shouldn't happen after validate_params
                            result.push_str(&ph);
                        }
                    } else {
                        result.push('{');
                    }
                }
                Some(c) => result.push(c),
            }
        }

        // Post-expansion validation: check for unexpanded {{...}} patterns.
        Self::check_unexpanded(&template.template_id, &result)?;

        Ok(result)
    }

    fn check_unexpanded(template_id: &str, source: &str) -> Result<(), TemplateError> {
        let mut chars = source.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '{' && chars.peek() == Some(&'{') {
                chars.next();
                let mut ph = String::new();
                loop {
                    match chars.next() {
                        None => break,
                        Some('}') if chars.peek() == Some(&'}') => {
                            chars.next();
                            if !ph.is_empty() {
                                return Err(TemplateError::UnexpandedPlaceholder {
                                    template: template_id.to_string(),
                                    remnant: ph,
                                });
                            }
                            break;
                        }
                        Some(c) => ph.push(c),
                    }
                }
            }
        }
        Ok(())
    }
}

/// Expands the selected template using resolved `KernelParameters` and writes
/// the expanded source back to the `KernelSource` component on each kernel.
pub struct TemplateExpansionSystem {
    pub(crate) expander: TemplateExpander,
}

impl TemplateExpansionSystem {
    pub fn new() -> Self {
        Self {
            expander: TemplateExpander,
        }
    }
}

impl CompilerSystem for TemplateExpansionSystem {
    fn name(&self) -> &str {
        "TemplateExpansionSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::KernelGeneration
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let kernel_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Kernel);
        if kernel_entities.is_empty() {
            return Ok(());
        }

        // Collect data per kernel (immutable borrow).
        struct ExpandWork {
            entity: Entity,
            source: KernelSource,
            params: KernelParameters,
        }
        let work: Vec<ExpandWork> = kernel_entities
            .iter()
            .filter_map(|e| {
                let source = world.get_component::<KernelSource>(*e).cloned()?;
                let params = world.get_component::<KernelParameters>(*e).cloned()?;
                Some(ExpandWork {
                    entity: *e,
                    source,
                    params,
                })
            })
            .collect();

        for w in &work {
            // Build template from the kernel source text and validate.
            let template =
                MetalKernelTemplate::from_source(&w.source.entry_point, &w.source.source);
            template.validate_params(&w.params).map_err(|e| {
                anyhow::anyhow!("template {} validation failed: {}", w.source.entry_point, e)
            })?;

            let expanded = self.expander.expand(&template, &w.params)?;

            // Update the KernelSource with expanded source.
            let _ = world.add_component(w.entity,
            KernelSource {
                language: w.source.language.clone(),
                source: expanded,
                entry_point: w.source.entry_point.clone(),
            },);;
        }

        Ok(())
    }
}

// ── Helpers ────────────────────────────────────────────────────────

impl KernelTemplateId {
    fn name(&self) -> &'static str {
        match self {
            Self::Nf4Tile640Gemv => "nf4_tile640_gemv",
            Self::Int8Tile640Gemv => "int8_tile640_gemv",
            Self::FusedGateUp => "fused_gate_up",
            Self::FusedGateUpActivation => "fused_gate_up_activation",
            Self::FusedDownProjResidual => "fused_down_proj_residual",
            Self::FusedOProjResidual => "fused_o_proj_residual",
            Self::FusedRmsNormQkv => "fused_rms_norm_qkv",
            Self::FusedAttentionScoreProbe => "fused_attention_score_probe",
            Self::Gemma4FullInt4 => "gemma4_full_int4",
            Self::RawF32Matmul => "raw_f32_matmul",
            Self::Fp16Matmul => "fp16_matmul",
        }
    }
    fn default_entry_point(&self) -> &'static str {
        match self {
            Self::Nf4Tile640Gemv => "gemv_nf4_tile640",
            Self::Int8Tile640Gemv => "gemv_int8_tile640",
            Self::FusedGateUp => "fused_gate_up",
            Self::FusedGateUpActivation => "fused_gate_up_activation",
            Self::FusedDownProjResidual => "fused_down_proj_residual",
            Self::FusedOProjResidual => "fused_o_proj_residual",
            Self::FusedRmsNormQkv => "fused_rms_norm_qkv",
            Self::FusedAttentionScoreProbe => "fused_attention_score_probe",
            Self::Gemma4FullInt4 => "gemma4_full_int4",
            Self::RawF32Matmul => "raw_f32_matmul",
            Self::Fp16Matmul => "fp16_matmul",
        }
    }
}

/// Map an op-kind string + codec to the best `KernelTemplateId`.
fn select_template(root_op: &str, codec: CodecFamily) -> KernelTemplateId {
    match (root_op, codec) {
        ("mlp_gate_up", CodecFamily::Nf4) => KernelTemplateId::Nf4Tile640Gemv,
        ("mlp_gate_up", CodecFamily::Int8) => KernelTemplateId::Int8Tile640Gemv,
        ("mlp_down", CodecFamily::Nf4) => KernelTemplateId::FusedDownProjResidual,
        ("mlp_down", CodecFamily::Int8) => KernelTemplateId::Int8Tile640Gemv,
        ("fused_gate_up", _) => KernelTemplateId::FusedGateUp,
        ("fused_down_proj", _) => KernelTemplateId::FusedDownProjResidual,
        ("fused_o_proj", _) => KernelTemplateId::FusedOProjResidual,
        ("fused_rms_norm_qkv", _) => KernelTemplateId::FusedRmsNormQkv,
        ("attention_score", _) => KernelTemplateId::FusedAttentionScoreProbe,
        ("gemma4_full", _) => KernelTemplateId::Gemma4FullInt4,
        _ if codec == CodecFamily::RawF32 => KernelTemplateId::RawF32Matmul,
        _ if codec == CodecFamily::Fp16 => KernelTemplateId::Fp16Matmul,
        _ => KernelTemplateId::Nf4Tile640Gemv,
    }
}

/// Map a template name string back to a `KernelFamily` for parameter resolution.
fn template_name_to_family(name: &str) -> crate::ecs::aot::parameters::KernelFamily {
    match name {
        "nf4_tile640_gemv" => crate::ecs::aot::parameters::KernelFamily::GemvNf4Tile,
        "int8_tile640_gemv" => crate::ecs::aot::parameters::KernelFamily::GemvInt8Tile,
        "fused_gate_up" | "fused_gate_up_activation" => {
            crate::ecs::aot::parameters::KernelFamily::MlpFused
        }
        "fused_down_proj_residual" => crate::ecs::aot::parameters::KernelFamily::MlpFused,
        "fused_o_proj_residual" => crate::ecs::aot::parameters::KernelFamily::MlpFused,
        "fused_rms_norm_qkv" => crate::ecs::aot::parameters::KernelFamily::MlpFused,
        "fused_attention_score_probe" => crate::ecs::aot::parameters::KernelFamily::AttentionScores,
        "gemma4_full_int4" => crate::ecs::aot::parameters::KernelFamily::DecoderLayerStaged,
        "raw_f32_matmul" => crate::ecs::aot::parameters::KernelFamily::GemvInt8Tile,
        "fp16_matmul" => crate::ecs::aot::parameters::KernelFamily::GemvNf4Tile,
        _ => crate::ecs::aot::parameters::KernelFamily::GemvNf4Tile,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::aot::parameters::{DType, KernelFamily};

    fn sample_params() -> KernelParameters {
        KernelParameters {
            kernel_family: KernelFamily::GemvNf4Tile,
            codec_family: crate::ecs::plan::CodecFamily::Nf4,
            tile_width: 640,
            group_size: 128,
            threadgroup_size: 32,
            simdgroup_width: 32,
            groups_per_tile: 5,
            lane_values: 4,
            unroll_factor: 4,
            use_threadgroup_memory: false,
            prefetch_distance: 2,
            accumulation_dtype: DType::Fp32,
            output_dtype: DType::Fp16,
        }
    }

    #[test]
    fn expander_rejects_unknown_placeholder() {
        let template = MetalKernelTemplate {
            template_id: "test".into(),
            source: "const uint X = {{UNKNOWN_VAR}};".into(),
            required_placeholders: vec![],
        };
        let expander = TemplateExpander;
        let result = expander.expand(&template, &sample_params());
        assert!(result.is_err());
        match result.unwrap_err() {
            TemplateError::UnknownPlaceholder { placeholder, .. } => {
                assert_eq!(placeholder, "UNKNOWN_VAR");
            }
            _ => panic!("expected UnknownPlaceholder"),
        }
    }

    #[test]
    fn expander_produces_expected_constexprs() {
        let template = MetalKernelTemplate {
            template_id: "test".into(),
            source: "const uint TW = {{TILE_WIDTH}};\nconst uint GS = {{GROUP_SIZE}};\nconst uint LV = {{LANE_VALUES}};"
                .into(),
            required_placeholders: vec![
                "TILE_WIDTH".into(),
                "GROUP_SIZE".into(),
                "LANE_VALUES".into(),
            ],
        };
        let expander = TemplateExpander;
        let result = expander.expand(&template, &sample_params()).unwrap();
        assert!(result.contains("TW = 640;"), "result: {}", result);
        assert!(result.contains("GS = 128;"), "result: {}", result);
        assert!(result.contains("LV = 4;"), "result: {}", result);
    }

    #[test]
    fn expander_detects_unexpanded_placeholder() {
        let template = MetalKernelTemplate {
            template_id: "test".into(),
            source: "const uint TW = {{TILE_WIDTH}};\nconst uint BAD = {{NOT_IN_PARAMS}};".into(),
            required_placeholders: vec!["TILE_WIDTH".into()],
        };
        let expander = TemplateExpander;
        let result = expander.expand(&template, &sample_params());
        assert!(result.is_err());
    }

    #[test]
    fn from_source_discovers_placeholders() {
        let source = "{{A}} hello {{B}} world {{C}}".to_string();
        let tpl = MetalKernelTemplate::from_source("test", &source);
        assert_eq!(tpl.required_placeholders, vec!["A", "B", "C"]);
    }

    #[test]
    fn expander_rejects_missing_placeholder() {
        let template = MetalKernelTemplate {
            template_id: "test".into(),
            source: "const uint X = {{MISSING}};".into(),
            required_placeholders: vec!["MISSING".into()],
        };
        let params = sample_params();
        assert!(template.validate_params(&params).is_err());
    }
}
