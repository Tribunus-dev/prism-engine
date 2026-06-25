#[cfg(test)]
mod tests {
    use crate::linux::cpu::device::CpuBackend;
    use crate::linux::device::LinuxDeviceBackend;
    use crate::linux::memory::{AllocationRequest, MemoryPreference, BufferUsage};

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
        };

        let req2 = AllocationRequest {
            size_bytes: 1024,
            alignment_bytes: 64,
            memory_preference: MemoryPreference::PreferHostVisible,
            usage: BufferUsage::Scratch,
            zero_initialize: true,
        };

        let buf1 = backend.allocate(device, req1).unwrap();
        let buf2 = backend.allocate(device, req2).unwrap();

        assert_ne!(buf1.buffer_id.id.opaque_id, buf2.buffer_id.id.opaque_id);
    }
}
