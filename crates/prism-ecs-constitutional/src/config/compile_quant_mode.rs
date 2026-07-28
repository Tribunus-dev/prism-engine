//! Compile-time quantization mode for the ComputeImage compiler.
//!
//! Authority: the canonical enumeration of the compile-time quantization
//! modes a Constitutional Engine caller may select, plus parse/format
//! helpers. No engine-internal types are referenced; this surface is
//! pure data.

/// Compile-time quantization mode for the ComputeImage compiler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompileQuantMode {
    /// 4-bit NormalFloat (NF4) block quantization.
    Nf4 { group_size: u32 },
    /// 4-bit NormalFloat (NF4) with 640-element tile-transposed storage.
    Nf4Tile640 { group_size: u32 },
    /// 8-bit affine quantization.
    Af8 { group_size: u32 },
    /// Ternary 1.58-bit quantization (2-bit nibble encoding, 4 per byte).
    Ternary { group_size: u32 },
    /// Ternary 1.58-bit quantization with 640-weight SIMD-aligned tiles.
    TernaryTile640 { group_size: u32 },
}

impl CompileQuantMode {
    /// Parse a quant mode name into a CompileQuantMode.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "nf4" => Some(Self::Nf4 { group_size: 64 }),
            "nf4-128" => Some(Self::Nf4 { group_size: 128 }),
            "nf4tile640" | "nf4-tile640" | "nftile640" => {
                Some(Self::Nf4Tile640 { group_size: 128 })
            }
            "8bit" => Some(Self::Af8 { group_size: 64 }),
            "ternary" | "1.58" => Some(Self::Ternary { group_size: 32 }),
            "ternary_tile640" | "tile640" => Some(Self::TernaryTile640 { group_size: 640 }),
            "none" => None,
            _ => None,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Nf4 { group_size: 64 } => "nf4",
            Self::Nf4 { group_size: 128 } => "nf4-128",
            Self::Nf4Tile640 { .. } => "nf4tile640",
            Self::Af8 { .. } => "8bit",
            Self::Ternary { .. } => "ternary",
            Self::TernaryTile640 { .. } => "ternary_tile640",
            _ => "nf4-64",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_named_modes_round_trip() {
        assert_eq!(CompileQuantMode::from_name("nf4"), Some(CompileQuantMode::Nf4 { group_size: 64 }));
        assert_eq!(CompileQuantMode::from_name("nf4-128"), Some(CompileQuantMode::Nf4 { group_size: 128 }));
        assert_eq!(CompileQuantMode::from_name("nf4tile640"), Some(CompileQuantMode::Nf4Tile640 { group_size: 128 }));
        assert_eq!(CompileQuantMode::from_name("8bit"), Some(CompileQuantMode::Af8 { group_size: 64 }));
        assert_eq!(CompileQuantMode::from_name("ternary"), Some(CompileQuantMode::Ternary { group_size: 32 }));
        assert_eq!(
            CompileQuantMode::from_name("ternary_tile640"),
            Some(CompileQuantMode::TernaryTile640 { group_size: 640 })
        );
    }

    #[test]
    fn parse_unknown_returns_none() {
        assert_eq!(CompileQuantMode::from_name("nope"), None);
        assert_eq!(CompileQuantMode::from_name("none"), None);
    }

    #[test]
    fn name_returns_canonical_label() {
        assert_eq!(CompileQuantMode::Nf4 { group_size: 64 }.name(), "nf4");
        assert_eq!(CompileQuantMode::Nf4 { group_size: 128 }.name(), "nf4-128");
        assert_eq!(CompileQuantMode::Ternary { group_size: 32 }.name(), "ternary");
    }
}
