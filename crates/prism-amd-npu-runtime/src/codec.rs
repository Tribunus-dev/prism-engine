//! Canonical binary encoding for native XDNA artifacts.

use crate::XdnaArtifact;

impl XdnaArtifact {
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        bincode::serialize(self).map_err(|error| format!("encode XDNA artifact: {error}"))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let artifact: Self = bincode::deserialize(bytes)
            .map_err(|error| format!("decode XDNA artifact: {error}"))?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn decode_hex_envelope(source: &str) -> Result<Self, String> {
        let encoded = source
            .strip_prefix("prism-xdna-bincode-v1:")
            .ok_or_else(|| "unsupported XDNA artifact envelope".to_string())?;
        if encoded.len() % 2 != 0 {
            return Err("odd-length XDNA artifact envelope".into());
        }
        let mut bytes = Vec::with_capacity(encoded.len() / 2);
        let chars = encoded.as_bytes();
        for pair in chars.chunks_exact(2) {
            let high = (pair[0] as char)
                .to_digit(16)
                .ok_or_else(|| "invalid XDNA artifact hex".to_string())?;
            let low = (pair[1] as char)
                .to_digit(16)
                .ok_or_else(|| "invalid XDNA artifact hex".to_string())?;
            bytes.push(((high << 4) | low) as u8);
        }
        Self::decode(&bytes)
    }
}
