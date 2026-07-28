//! Program serialization — pure data types and pure algorithms for
//! serializing a phase program to bytes.

use serde::{Deserialize, Serialize};

use super::phase_program::{PhaseProgram, SerializedPhaseProgram};

/// Program serialization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProgramFormat {
    /// Bincode binary format.
    Bincode,
    /// JSON text format.
    Json,
    /// Postcard binary format.
    Postcard,
}

/// Program serializer.
#[derive(Debug, Clone, Default)]
pub struct ProgramSerializer;

impl ProgramSerializer {
    /// Create a new serializer.
    pub fn new() -> Self {
        Self
    }

    /// Serialize a phase program to a [`SerializedPhaseProgram`].
    pub fn serialize(
        &self,
        program: &PhaseProgram,
        format: ProgramFormat,
    ) -> Result<SerializedPhaseProgram, String> {
        let bytes = match format {
            ProgramFormat::Bincode => bincode::serialize(program).map_err(|e| e.to_string())?,
            ProgramFormat::Json => serde_json::to_vec(program).map_err(|e| e.to_string())?,
            ProgramFormat::Postcard => {
                return Err("postcard not yet supported".to_string());
            }
        };
        let program_hash = crate::compute_image_runtime::ContentHash::from(
            blake3::hash(&bytes).as_bytes()[..8]
                .iter()
                .fold(0u64, |acc, b| (acc << 8) | u64::from(*b)),
        );
        Ok(SerializedPhaseProgram {
            program_id: program.program_id.clone(),
            bytes,
            program_hash,
            format_version: 1,
            state_domain_id: "default".to_string(),
            receipt_id: "default".to_string(),
        })
    }
}
