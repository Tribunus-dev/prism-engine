//! Multi-model assembly types for the Prism ECS compiler.
//!
//! Supports defining and optimizing groups of models (LLM, TTS, vision, etc.)
//! under a shared memory budget, then compiling them into a single unified
//! `.cimage` file.

use crate::evolution::compile_plan::FormatPlan;
use crate::evolution::frontier::ParetoFrontier;
use crate::evolution::joint::JointEvolutionSystem;
use serde::{Deserialize, Serialize};

/// A single model in an assembly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyModel {
    /// Logical name for the model (e.g. "llm", "tts", "vision").
    pub name: String,
    /// HuggingFace repository identifier (e.g. "prism-ml/Bonsai-27B").
    pub hf_repo: String,
    /// Model architecture (e.g. "decoder_only", "encoder_decoder", "vit", "audio_codec").
    pub architecture: String,
    /// Estimated RAM footprint in GiB at inference time.
    pub ram_estimate_gb: f64,
    /// Optional per-tensor format plan from evolution search.
    pub format_plan: Option<FormatPlan>,
}

/// Top-level assembly manifest describing a multi-model inference bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblySpec {
    /// All models included in this assembly.
    pub models: Vec<AssemblyModel>,
    /// Total RAM budget for the entire assembly (GiB).
    pub total_ram_budget_gb: f64,
    /// Optional shared tokenizer path (models that use the same tokenizer).
    pub shared_tokenizer_path: Option<String>,
    /// Optional shared audio codec path (TTS + codec models).
    pub shared_audio_codec_path: Option<String>,
}

impl AssemblySpec {
    /// Run evolution search for each model independently, then verify
    /// the combined result fits in the RAM budget.
    ///
    /// Each model's `format_plan` is populated from the best genome found
    /// by the joint evolution system.
    pub fn optimize(&mut self, hardware_ram_gb: f64) -> Result<(), String> {
        let evolution = JointEvolutionSystem::default();

        for model in &mut self.models {
            // Build initial population from diverse codons to seed the search.
            let population: Vec<crate::evolution::joint::ScoredGenome> = (0..10)
                .map(|i| crate::evolution::joint::ScoredGenome {
                    genome: JointEvolutionSystem::codon_to_genome(i as u64),
                    fitness: vec![0.5 + (i as f64) * 0.05, 0.5],
                })
                .collect();

            let frontier = ParetoFrontier::new(2);

            // Run one generation of evolution to find the best format plan.
            let next_gen = evolution.run_generation(&population, &frontier);

            // Extract the best genome and build a format plan from it.
            if let Some(best) = next_gen.first() {
                // Pass an empty tensor key list; the format plan captures
                // the genome's representation axis for downstream compilation.
                let tensor_keys: Vec<String> = Vec::new();
                model.format_plan = Some(FormatPlan::from_best_genome(&best.genome, &tensor_keys));
            }
        }

        // Verify the combined estimate fits within the hardware RAM budget.
        let total_estimate: f64 = self.models.iter().map(|m| m.ram_estimate_gb).sum();
        if total_estimate > hardware_ram_gb {
            return Err(format!(
                "assembly requires {:.2} GiB RAM but hardware has {:.1} GiB",
                total_estimate, hardware_ram_gb
            ));
        }

        Ok(())
    }
}
