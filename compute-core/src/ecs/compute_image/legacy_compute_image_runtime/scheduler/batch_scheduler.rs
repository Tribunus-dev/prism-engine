use std::collections::VecDeque;

use super::LoadLevel;

/// Manages a queue of ready slot-ids and selects decode batches
/// sized according to the current system load level.
pub struct BatchScheduler {
    queue: VecDeque<u32>,
    load_level: LoadLevel,
    max_batch: usize,
}

impl BatchScheduler {
    /// Create a new scheduler with the given maximum batch size (default 8).
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            load_level: LoadLevel::Low,
            max_batch: 8,
        }
    }

    /// Push a slot onto the ready queue.
    pub fn enqueue(&mut self, slot_id: u32) {
        self.queue.push_back(slot_id);
    }

    /// Pop up to `max_batch` slots from the ready queue, respecting the
    /// current load level:
    /// - `Low`    → at most 1 slot
    /// - `Medium` → at most 4 slots
    /// - `High`   → at most `min(max_batch, queue.len())` slots
    pub fn select_batch(&mut self) -> Vec<u32> {
        let limit = match self.load_level {
            LoadLevel::Low => 1,
            LoadLevel::Medium => 4,
            LoadLevel::High => self.max_batch.min(self.queue.len()),
        };
        self.queue.drain(..limit).collect()
    }

    /// Re-enqueue the given slots after a batch completes (next token).
    pub fn batch_completed(&mut self, slot_ids: &[u32]) {
        for &id in slot_ids {
            self.queue.push_back(id);
        }
    }

    /// Clear the queue and reset load level to `Low`.
    pub fn clear(&mut self) {
        self.queue.clear();
        self.load_level = LoadLevel::Low;
    }

    // -- internal helpers for the sibling load monitor ----------

    #[allow(dead_code)]
    pub(crate) fn set_load_level(&mut self, level: LoadLevel) {
        self.load_level = level;
    }

    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enqueue_and_select_by_load() {
        let mut sched = BatchScheduler::new();
        for i in 0..10 {
            sched.enqueue(i);
        }
        assert_eq!(sched.queue_len(), 10);

        // Low → 1 slot
        sched.set_load_level(LoadLevel::Low);
        let batch = sched.select_batch();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0], 0);

        // Re-enqueue the remaining 9 are still in queue after drain
        sched.enqueue(0);
        assert_eq!(sched.queue_len(), 10);

        // Medium → 4 slots
        sched.set_load_level(LoadLevel::Medium);
        let batch = sched.select_batch();
        assert_eq!(batch.len(), 4);
        // (the exact ids depend on ordering, but we just assert count)

        // Re-enqueue for High test
        for &id in &batch {
            sched.enqueue(id);
        }
        assert_eq!(sched.queue_len(), 10);

        // High → min(8, 10) = 8
        sched.set_load_level(LoadLevel::High);
        let batch = sched.select_batch();
        assert_eq!(batch.len(), 8);
    }

    #[test]
    fn test_batch_completed_re_enqueues() {
        let mut sched = BatchScheduler::new();
        for i in 0..5 {
            sched.enqueue(i);
        }
        let batch = sched.select_batch();
        // default Low → 1 drained, 4 remaining
        assert_eq!(batch.len(), 1);
        sched.batch_completed(&[42, 99]);
        // 4 remaining + 2 re-enqueued = 6
        assert_eq!(sched.queue_len(), 6);
    }

    #[test]
    fn test_clear_resets_state() {
        let mut sched = BatchScheduler::new();
        sched.enqueue(1);
        sched.enqueue(2);
        sched.set_load_level(LoadLevel::High);
        sched.clear();
        assert_eq!(sched.queue_len(), 0);
        // After clear, load level resets to Low
        sched.enqueue(10);
        let batch = sched.select_batch();
        assert_eq!(batch.len(), 1);
    }
}
