use crate::linux::backend::{BackendKind, DeviceId};
use crate::linux::capability::{BackendAvailability, DeviceCapabilities};
use crate::linux::device::{DeviceDescriptor, LinuxDeviceBackend};
use crate::linux::errors::BackendError;
use crate::linux::memory::{AllocationRequest, DeviceBuffer};
use crate::linux::queue::{QueueClass, QueueHandle};
use crate::linux::submission::{Submission, SubmissionHandle, SubmissionStatus};

pub struct CudaBackend;

impl LinuxDeviceBackend for CudaBackend {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn enumerate_devices(&self) -> Result<Vec<DeviceDescriptor>, BackendError> {
        Ok(vec![])
    }

    fn probe_capabilities(
        &self,
        _device: &DeviceId,
    ) -> Result<DeviceCapabilities, BackendError> {
        Err(BackendError::NotReady)
    }

    fn create_queue(
        &self,
        _device: &DeviceId,
        _class: QueueClass,
    ) -> Result<QueueHandle, BackendError> {
        Err(BackendError::UnsupportedOperation("CUDA FeatureNotCompiled".into()))
    }

    fn allocate(
        &self,
        _device: &DeviceId,
        _request: AllocationRequest,
    ) -> Result<DeviceBuffer, BackendError> {
        Err(BackendError::UnsupportedOperation("CUDA FeatureNotCompiled".into()))
    }

    fn submit(
        &self,
        _queue: &QueueHandle,
        _submission: Submission,
    ) -> Result<SubmissionHandle, BackendError> {
        Err(BackendError::UnsupportedOperation("CUDA FeatureNotCompiled".into()))
    }

    fn poll(&self, _submission: &SubmissionHandle) -> Result<SubmissionStatus, BackendError> {
        Ok(SubmissionStatus::Failed)
    }

    fn synchronize(&self, _submission: &SubmissionHandle) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("CUDA FeatureNotCompiled".into()))
    }
}
