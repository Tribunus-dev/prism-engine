pub mod diffusion_ane;
pub mod coreml_bridge;
pub mod coreml_state;
#[cfg(feature = "ane")]
pub mod mil_gen_full;
#[cfg(feature = "ane")]
pub mod compile_full_model;
pub mod coreml_audit;
pub mod arena_info;
pub mod arena;
#[cfg(feature = "ane")]
pub mod mil_builder;
#[cfg(feature = "ane")]
pub mod mlpackage;
#[cfg(feature = "ane")]
pub mod mil_helpers;

pub use arena_info::ArenaInfo;
pub use arena::Arena;

/// Pack a .mlmodelc directory into a flat byte buffer for .cimage embedding.
/// Format: [name_len:u32][name_bytes][data_len:u64][data_bytes]+ per file.
pub fn pack_mlmodelc(dir: &std::path::Path) -> Result<Vec<u8>, String> {
    fn collect(base: &std::path::Path, dir: &std::path::Path, buf: &mut Vec<u8>) -> Result<(), String> {
        for entry in std::fs::read_dir(dir).map_err(|e| format!("read dir: {e}"))? {
            let entry = entry.map_err(|e| format!("entry: {e}"))?;
            let path = entry.path();
            let rel = path.strip_prefix(base).map_err(|_| "strip")?;
            let name = rel.to_string_lossy().to_string();
            if path.is_dir() {
                collect(base, &path, buf)?;
            } else {
                let data = std::fs::read(&path).map_err(|e| format!("read: {e}"))?;
                buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
                buf.extend_from_slice(name.as_bytes());
                buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
                buf.extend_from_slice(&data);
            }
        }
        Ok(())
    }
    let mut buf = Vec::new();
    collect(dir, dir, &mut buf)?;
    Ok(buf)
}

/// Unpack a packed .mlmodelc back to a directory on disk.
pub fn unpack_mlmodelc(data: &[u8], dest: &std::path::Path) -> Result<(), String> {
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let name_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + name_len > data.len() { break; }
        let name = std::str::from_utf8(&data[pos..pos + name_len])
            .map_err(|e| format!("utf8: {e}"))?;
        pos += name_len;
        if pos + 8 > data.len() { break; }
        let data_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        if pos + data_len > data.len() { break; }
        let file_data = &data[pos..pos + data_len];
        pos += data_len;
        let dest_path = dest.join(name);
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
        }
        std::fs::write(&dest_path, file_data).map_err(|e| format!("write: {e}"))?;
    }
    Ok(())
}
