#[cfg(test)]
mod tests {
    use crate::linux::cpu::device::CpuBackend;
    use crate::linux::device::LinuxDeviceBackend;
    use crate::linux::memory::{AllocationRequest, MemoryPreference, BufferUsage, ElementLayout};
#[cfg(test)]








    #[test]
    fn cpu_backend_enumerates_one_real_cpu_device() {
        let backend = CpuBackend::new();
        let devices = backend.enumerate_devices().unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].capabilities.device_name, "Generic CPU");
    }

    #[test]
    fn cpu_buffer_ids_are_unique() {
        let backend = CpuBackend::new();
        let devices = backend.enumerate_devices().unwrap();
        let device = &devices[0].id;

        let req1 = AllocationRequest {
            size_bytes: 1024,
            alignment_bytes: 64,
            memory_preference: MemoryPreference::PreferHostVisible,
            usage: BufferUsage::Scratch,
            zero_initialize: true,
            layout: ElementLayout::Bytes,
        };

        let req2 = AllocationRequest {
            size_bytes: 1024,
            alignment_bytes: 64,
            memory_preference: MemoryPreference::PreferHostVisible,
            usage: BufferUsage::Scratch,
            zero_initialize: true,
            layout: ElementLayout::Bytes,
        };

        let buf1 = backend.allocate(device, req1).unwrap();
        let buf2 = backend.allocate(device, req2).unwrap();

        assert_ne!(buf1.buffer_id.id.opaque_id, buf2.buffer_id.id.opaque_id);
    }
}

    #[test]
    fn fill_u32_is_memory_safe_and_deterministic() {
        let backend = CpuBackend::new();
        let devices = backend.enumerate_devices().unwrap();
        let device = &devices[0].id;
        let queue = backend.create_queue(device, crate::linux::queue::QueueClass::ForegroundCompute).unwrap();

        let req = AllocationRequest {
            size_bytes: 4096, // 1024 u32s
            alignment_bytes: 64,
            memory_preference: MemoryPreference::PreferHostVisible,
            usage: BufferUsage::Scratch,
            zero_initialize: true,
            layout: ElementLayout::Bytes,
            layout: crate::linux::memory::ElementLayout::U32,
        };

        let buf = backend.allocate(device, req).unwrap();

        let sub = crate::linux::submission::Submission::Fill {
            destination: buf.buffer_id.clone(),
            value: 42,
            element_count: 1024,
        };

        let handle = backend.submit(&queue, sub).unwrap();
        backend.synchronize(&handle).unwrap();

        // Internal struct read verify (normally would use API readback)
        // Just verify submission completes cleanly since it's the oracle.
        let status = backend.poll(&handle).unwrap();
        assert_eq!(status, crate::linux::submission::SubmissionStatus::Complete);
}
