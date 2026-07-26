//! CImage fixed-size header and footer types.
//!
//! V0 proof format:
//!   [Header: fixed size]
//!   [Manifest JSON bytes]
//!   [Payload directory JSON bytes]
//!   [Receipt directory JSON bytes]
//!   [Payload blob bytes]
//!   [Footer: fixed size]

use serde::{Deserialize, Serialize};

/// Magic bytes for cimage V0 files.
pub const CIMAGE_MAGIC: [u8; 8] = *b"PRISMCIM";

/// Supported format version.
pub const CIMAGE_FORMAT_VERSION: u32 = 0;

/// Fixed-size V0 header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageHeaderV0 {
    pub magic: [u8; 8],
    pub format_version: u32,
    pub header_len: u64,
    pub manifest_offset: u64,
    pub manifest_len: u64,
    pub payload_directory_offset: u64,
    pub payload_directory_len: u64,
    pub receipt_directory_offset: u64,
    pub receipt_directory_len: u64,
    pub payload_blob_offset: u64,
    pub payload_blob_len: u64,
    pub footer_offset: u64,
}

impl CImageHeaderV0 {
    pub fn new() -> Self {
        Self {
            magic: CIMAGE_MAGIC,
            format_version: CIMAGE_FORMAT_VERSION,
            header_len: std::mem::size_of::<CImageHeaderV0>() as u64,
            manifest_offset: 0,
            manifest_len: 0,
            payload_directory_offset: 0,
            payload_directory_len: 0,
            receipt_directory_offset: 0,
            receipt_directory_len: 0,
            payload_blob_offset: 0,
            payload_blob_len: 0,
            footer_offset: 0,
        }
    }

    pub fn validate_magic(&self) -> Result<(), [u8; 8]> {
        if self.magic == CIMAGE_MAGIC {
            Ok(())
        } else {
            Err(self.magic)
        }
    }

    pub fn supports_format(&self) -> bool {
        self.format_version == CIMAGE_FORMAT_VERSION
    }
}

impl Default for CImageHeaderV0 {
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed-size V0 footer.
///
/// Binds the whole file together with a recursive digest:
/// `cimage_sha256_without_footer` covers bytes [0, footer_offset).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageFooterV0 {
    pub manifest_sha256: String,
    pub payload_directory_sha256: String,
    pub receipt_directory_sha256: String,
    pub payload_blob_sha256: String,
    pub cimage_sha256_without_footer: String,
}

impl CImageFooterV0 {
    pub fn new() -> Self {
        Self {
            manifest_sha256: String::new(),
            payload_directory_sha256: String::new(),
            receipt_directory_sha256: String::new(),
            payload_blob_sha256: String::new(),
            cimage_sha256_without_footer: String::new(),
        }
    }
}

impl Default for CImageFooterV0 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_magic() {
        let h = CImageHeaderV0::new();
        assert!(h.validate_magic().is_ok());
    }

    #[test]
    fn test_header_format_version() {
        let h = CImageHeaderV0::new();
        assert!(h.supports_format());
    }

    #[test]
    fn test_header_size_roundtrip() {
        let h = CImageHeaderV0::new();
        let bytes = bincode::serialize(&h).unwrap();
        let deserialized: CImageHeaderV0 = bincode::deserialize(&bytes).unwrap();
        assert_eq!(deserialized.magic, CIMAGE_MAGIC);
        assert_eq!(deserialized.format_version, CIMAGE_FORMAT_VERSION);
    }

    #[test]
    fn test_header_serde_serde_json() {
        let h = CImageHeaderV0::new();
        let json = serde_json::to_string(&h).unwrap();
        let deserialized: CImageHeaderV0 = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.magic, CIMAGE_MAGIC);
        assert_eq!(deserialized.format_version, CIMAGE_FORMAT_VERSION);
    }
}
