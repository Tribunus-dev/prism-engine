use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use std::io::Read;

pub struct CImageHandler;

impl CImageHandler {
    pub fn new() -> Self {
        Self
    }
}

impl McpHandler for CImageHandler {
    fn name(&self) -> &'static str {
        "inspect_cimage"
    }

    fn description(&self) -> &'static str {
        "Inspect a cimage binary file: header, sections, layers."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to .cimage file" }
            },
            "required": ["path"]
        })
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        _state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let path = request.args["path"].as_str().unwrap_or("");
        if path.is_empty() {
            return Err(anyhow::anyhow!("path is required"));
        }

        let mut file = std::fs::File::open(path)?;
        let mut header = [0u8; 64];
        file.read_exact(&mut header)?;

        let mut output = format!("CImage: {}\n\n", path);

        // Basic header inspection
        let magic = &header[0..8];
        let magic_str = String::from_utf8_lossy(magic);
        output.push_str(&format!("Magic: {}\n", magic_str));

        if magic_str.contains("CIMAGE") || magic_str.contains("CIMG") {
            // Parse version bytes (little-endian u32 at offset 8)
            let version = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
            output.push_str(&format!("Version: {}\n", version));

            // Section count at offset 12
            let section_count =
                u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
            output.push_str(&format!("Section count: {}\n", section_count));

            // Total size at offset 16
            let total_size = u64::from_le_bytes([
                header[16], header[17], header[18], header[19], header[20], header[21], header[22],
                header[23],
            ]);
            output.push_str(&format!("Total size: {} bytes\n", total_size));

            // Checksum at offset 24
            let checksum = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);
            output.push_str(&format!("Header checksum: 0x{:08x}\n", checksum));

            output.push_str("\nRaw header hex dump:\n");
            for (i, chunk) in header.chunks(16).enumerate() {
                let hex: Vec<String> = chunk.iter().map(|b| format!("{:02x}", b)).collect();
                let ascii: String = chunk
                    .iter()
                    .map(|b| {
                        if b.is_ascii_graphic() || *b == b' ' {
                            *b as char
                        } else {
                            '.'
                        }
                    })
                    .collect();
                output.push_str(&format!("  {:04x}: {}  {}\n", i * 16, hex.join(" "), ascii));
            }
        } else {
            output.push_str("(not a recognized cimage format — showing raw hex)\n");
            for (i, chunk) in header.chunks(16).enumerate() {
                let hex: Vec<String> = chunk.iter().map(|b| format!("{:02x}", b)).collect();
                let ascii: String = chunk
                    .iter()
                    .map(|b| {
                        if b.is_ascii_graphic() || *b == b' ' {
                            *b as char
                        } else {
                            '.'
                        }
                    })
                    .collect();
                output.push_str(&format!("  {:04x}: {}  {}\n", i * 16, hex.join(" "), ascii));
            }
        }

        // File size
        let file_len = std::fs::metadata(path)?.len();
        output.push_str(&format!(
            "\nFile size: {} bytes ({:.2} MB)\n",
            file_len,
            file_len as f64 / 1_048_576.0
        ));

        Ok(ToolResult::text(output))
    }
}
