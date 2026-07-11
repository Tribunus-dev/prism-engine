use serde::{Deserialize, Serialize};
use std::fmt;

/// A header embedded in serialized schemas for forward/backward compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaHeader {
    pub type_name: String,
    pub version: u32,
}

impl SchemaHeader {
    pub fn new(type_name: &str) -> Self {
        Self {
            type_name: type_name.to_string(),
            version: 1,
        }
    }

    pub fn check(&self, expected_version: u32) -> anyhow::Result<()> {
        if self.version != expected_version {
            anyhow::bail!(
                "schema version mismatch for {}: expected v{}, got v{}",
                self.type_name,
                expected_version,
                self.version
            );
        }
        Ok(())
    }
}

impl fmt::Display for SchemaHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} v{}", self.type_name, self.version)
    }
}
