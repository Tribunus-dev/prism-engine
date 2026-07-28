//! Batch scheduler — manages a queue of ready slot ids and selects
//! decode batches sized by the current load level.

use std::collections::VecDeque;

use super::LoadLevel;

/// Default maximum batch size when no caller override is provided.
pub const DEFAULT_MAX_BATCH: usize = 8;

/// Manages a queue of ready slot-ids and selects decode batches sized
/// according to the current system load level.
#[derive(Debug)]
pub struct BatchScheduler {
    queue: VecDeque<u32>,
    load_level: LoadLevel,
    max_batch: usize,
}

impl Default for BatchScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchScheduler {
    /// Create a new scheduler with the default maximum batch size
    /// ([`DEFAULT_MAX_BATCH`]).
    pub fn new() -> Self {
        Self::with_max_batch(DEFAULT_MAX_BATCH)
    }

    /// Create a new scheduler with a custom maximum batch size.
    pub fn with_max_batch(max_batch: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            load_level: LoadLevel::Low,
            max_batch,
        }
    }

    /// Push a slot onto the ready queue.
    pub fn enqueue(&mut self, slot_id: u32) {
        self.queue.push_back(slot_id);
    }

    /// Pop up to `max_batch` slots from the ready queue, respecting the
    /// current load level:
    /// - `Low` → at most 1 slot
    /// - `Medium` → at most 4 slots
    /// - `High` → at most `min(max_batch, queue.len())` slots
    pub fn select_batch(&mut self) -> Vec<u32> {
        let limit = match self.load_level {
            LoadLevel::Low => 1,
            LoadLevel::Medium => 4,
            LoadLevel::High => self.max_batch.min(self.queue.len()),
        };
        self.queue.drain(..limit).collect()
    }

    /// Re-enqueue the given slots after a batch completes.
    pub fn batch_completed(&mut self, slot_ids: &[u32]) {
        for &id in slot_ids {
            self.queue.push_back(id);
        }
    }

    /// Clear the queue and reset the load level to `Low`.
    pub fn clear(&mut self) {
        self.queue.clear();
        self.load_level = LoadLevel::Low;
    }

    /// Update the load level (used by the sibling [`LoadMonitor`]).
    pub fn set_load_level(&mut self, level: LoadLevel) {
        self.load_level = level;
    }

    /// Current load level.
    pub fn load_level(&self) -> LoadLevel {
        self.load_level
    }

    /// Current queue length.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enqueue_and_select_by_load() {
        let mut sched = BatchScheduler::new();
        for i in 0..10 {
            sched.enqueue(i);
        }
        assert_eq!(sched.queue_len(), 10);

        sched.set_load_level(LoadLevel::Low);
        let batch = sched.select_batch();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0], 0);

        sched.enqueue(0);
        assert_eq!(sched.queue_len(), 10);

        sched.set_load_level(LoadLevel::Medium);
        let batch = sched.select_batch();
        assert_eq!(batch.len(), 4);

        for &id in &batch {
            sched.enqueue(id);
        }
        assert_eq!(sched.queue_len(), 10);

        sched.set_load_level(LoadLevel::High);
        let batch = sched.select_batch();
        assert_eq!(batch.len(), 8);
    }

    #[test]
    fn batch_completed_re_enqueues() {
        let mut sched = BatchScheduler::new();
        for i in 0..5 {
            sched.enqueue(i);
        }
        let batch = sched.select_batch();
        assert_eq!(batch.len(), 1);
        sched.batch_completed(&[42, 99]);
        assert_eq!(sched.queue_len(), 6);
    }

    #[test]
    fn clear_resets_state() {
        let mut sched = BatchScheduler::new();
        sched.enqueue(1);
        sched.enqueue(2);
        sched.set_load_level(LoadLevel::High);
        sched.clear();
        assert_eq!(sched.queue_len(), 0);
        sched.enqueue(10);
        let batch = sched.select_batch();
        assert_eq!(batch.len(), 1);
    }
}
