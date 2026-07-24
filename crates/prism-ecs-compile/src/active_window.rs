//! Bounded mmap-backed canary window shared by backend evaluators.
use memmap2::{Mmap, MmapOptions};
use std::{
    fs::File,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Debug)]
pub struct ActiveLayerWindow {
    reference_map: Mmap,
    candidate_map: Option<Mmap>,
    reference_range: (usize, usize),
    candidate_range: Option<(usize, usize)>,
    generation: AtomicU64,
}

impl ActiveLayerWindow {
    pub fn open(
        reference: &Path,
        reference_range: (u64, u64),
        candidate: Option<(&Path, (u64, u64))>,
    ) -> Result<Self, String> {
        let reference_map = map_range(reference, reference_range)?;
        let (candidate_map, candidate_range) = match candidate {
            Some((path, range)) => (Some(map_range(path, range)?), Some((0, range.1 as usize))),
            None => (None, None),
        };
        Ok(Self {
            reference_map,
            candidate_map,
            reference_range: (0, reference_range.1 as usize),
            candidate_range,
            generation: AtomicU64::new(1),
        })
    }

    pub fn reference(&self) -> &[u8] {
        &self.reference_map[self.reference_range.0..self.reference_range.1]
    }

    pub fn candidate(&self) -> Option<&[u8]> {
        self.candidate_map
            .as_ref()
            .zip(self.candidate_range)
            .map(|(map, (start, end))| &map[start..end])
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Invalidates the active views before the next layer is mapped. Backend
    /// command buffers must have signaled their corresponding Metal event
    /// before callers recycle this window.
    pub fn recycle(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::AcqRel) + 1
    }
}

fn map_range(path: &Path, range: (u64, u64)) -> Result<Mmap, String> {
    let file =
        File::open(path).map_err(|e| format!("open active tensor {}: {e}", path.display()))?;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    let end = range
        .0
        .checked_add(range.1)
        .ok_or_else(|| "active tensor range overflow".to_string())?;
    if end > metadata.len() || range.1 == 0 {
        return Err(format!(
            "active tensor range {:?} exceeds {}",
            range,
            metadata.len()
        ));
    }
    unsafe {
        MmapOptions::new()
            .offset(range.0)
            .len(range.1 as usize)
            .map(&file)
            .map_err(|e| format!("mmap active tensor: {e}"))
    }
}
