//! .mlmodelc directory → uncompressed tar inside a mmap slice.

use std::path::Path;

/// Write a .mlmodelc directory tree into the given byte slice as an
/// uncompressed tar archive.  Returns the number of bytes written.
/// The slice must be large enough (use `predict_tar_size` to pre-compute).
pub fn archive_mlmodelc_to_mmap(src: &Path, dst: &mut [u8]) -> std::io::Result<u64> {
    let mut cursor = std::io::Cursor::new(dst);
    let mut builder = tar::Builder::new(&mut cursor);
    builder.append_dir_all(".", src)?;
    builder.finish()?;
    drop(builder);
    Ok(cursor.position())
}
