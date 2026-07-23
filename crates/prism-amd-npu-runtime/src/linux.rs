//! Direct Linux amdxdna DRM-accel discovery.
//!
//! This module intentionally uses the kernel UAPI directly. It does not
//! depend on XRT or the amdxdna userspace shim. Submission remains separated
//! from discovery because command-buffer layout is firmware-versioned.

#![cfg(target_os = "linux")]

use crate::command::{DeviceAddressBinding, XdnaCommandBuffer, XdnaFirmwareEncoder};
use prism_spatial_ir::xdna::{TileCoord, XdnaGeneration, XdnaProgram, XdnaTopology};
use prism_spatial_ir::XdnaTarget;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

const DRM_IOCTL_BASE: u8 = b'd';
const DRM_COMMAND_BASE: u8 = 0x40;
const AMDXDNA_GET_INFO: u8 = 7;
const AMDXDNA_CREATE_BO: u8 = 3;
const AMDXDNA_GET_BO_INFO: u8 = 4;
const AMDXDNA_SYNC_BO: u8 = 5;
const AMDXDNA_EXEC_CMD: u8 = 6;
const AMDXDNA_QUERY_AIE_METADATA: u32 = 1;

#[repr(C)]
#[derive(Default)]
struct GetInfo {
    param: u32,
    buffer_size: u32,
    buffer: u64,
}

#[repr(C)]
#[derive(Default)]
struct CreateBo {
    flags: u64,
    vaddr: u64,
    size: u64,
    bo_type: u32,
    handle: u32,
}

#[repr(C)]
struct SyncBo {
    handle: u32,
    direction: u32,
    offset: u64,
    size: u64,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct XdnaBoInfo {
    pub map_offset: u64,
    pub user_address: u64,
    pub device_address: u64,
}

#[repr(C)]
#[derive(Default)]
struct GetBoInfo {
    ext: u64,
    ext_flags: u64,
    handle: u32,
    pad: u32,
    map_offset: u64,
    user_address: u64,
    device_address: u64,
}

#[repr(C)]
struct ExecCmd {
    ext: u64,
    ext_flags: u64,
    hwctx: u32,
    command_type: u32,
    cmd_handles: u64,
    args: u64,
    cmd_count: u32,
    arg_count: u32,
    seq: u64,
}

pub const BO_TYPE_SHARE: u32 = 1;
pub const BO_TYPE_DEV: u32 = 3;
pub const BO_TYPE_CMD: u32 = 4;
pub const SYNC_TO_DEVICE: u32 = 0;
pub const SYNC_FROM_DEVICE: u32 = 1;
pub const CMD_SUBMIT_EXEC_BUF: u32 = 0;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
struct TileMetadata {
    row_count: u16,
    row_start: u16,
    dma_channel_count: u16,
    lock_count: u16,
    event_reg_count: u16,
    pad: [u16; 3],
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
struct AieMetadata {
    col_size: u32,
    cols: u16,
    rows: u16,
    version_major: u32,
    version_minor: u32,
    core: TileMetadata,
    mem: TileMetadata,
    shim: TileMetadata,
}

pub struct LinuxXdnaProbe {
    device: File,
    path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct XdnaSubmissionPreflight {
    pub device_path: PathBuf,
    pub hwctx: u32,
    pub topology: XdnaTopology,
}

impl LinuxXdnaProbe {
    pub fn open() -> Result<Self, String> {
        for candidate in ["/dev/accel/accel0", "/dev/amdxdna"] {
            if Path::new(candidate).exists() {
                let device = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(candidate)
                    .map_err(|error| format!("open {candidate}: {error}"))?;
                return Ok(Self {
                    device,
                    path: candidate.into(),
                });
            }
        }
        Err("no amdxdna acceleration device found".into())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Allocate an XDNA buffer object through the kernel UAPI.
    pub fn create_bo(&self, size: u64, bo_type: u32) -> Result<u32, String> {
        if size == 0 {
            return Err("cannot allocate a zero-sized XDNA buffer object".into());
        }
        if !matches!(bo_type, BO_TYPE_SHARE | BO_TYPE_DEV | BO_TYPE_CMD) {
            return Err(format!("unsupported XDNA buffer type {bo_type}"));
        }
        let mut request = CreateBo {
            size,
            bo_type,
            ..Default::default()
        };
        ioctl_call(self.device.as_raw_fd(), ioctl_create_bo(), &mut request)?;
        if request.handle == 0 {
            return Err("amdxdna returned an invalid zero buffer handle".into());
        }
        Ok(request.handle)
    }

    pub fn sync_bo(&self, handle: u32, direction: u32, size: u64) -> Result<(), String> {
        if handle == 0 || size == 0 {
            return Err("XDNA BO sync requires a valid handle and nonzero size".into());
        }
        if !matches!(direction, SYNC_TO_DEVICE | SYNC_FROM_DEVICE) {
            return Err(format!("unsupported XDNA BO sync direction {direction}"));
        }
        let mut request = SyncBo {
            handle,
            direction,
            offset: 0,
            size,
        };
        ioctl_call(self.device.as_raw_fd(), ioctl_sync_bo(), &mut request)
    }

    /// Map a shareable BO, copy persistent tensor contents, and make the
    /// bytes visible to the device. The mapping is temporary; residency is
    /// retained by the BO handle owned by the caller.
    pub fn upload_payload(&self, handle: u32, payload: &[u8]) -> Result<(), String> {
        if handle == 0 || payload.is_empty() {
            return Err("XDNA payload upload requires a handle and nonempty bytes".into());
        }
        let info = self.bo_info(handle)?;
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                payload.len(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.device.as_raw_fd(),
                info.map_offset as libc::off_t,
            )
        };
        if mapping == libc::MAP_FAILED {
            return Err(format!(
                "map XDNA payload BO: {}",
                std::io::Error::last_os_error()
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(payload.as_ptr(), mapping.cast::<u8>(), payload.len());
        }
        let unmap_result = unsafe { libc::munmap(mapping, payload.len()) };
        if unmap_result != 0 {
            return Err(format!(
                "unmap XDNA payload BO: {}",
                std::io::Error::last_os_error()
            ));
        }
        self.sync_bo(handle, SYNC_TO_DEVICE, payload.len() as u64)
    }

    /// Synchronize a BO from the device and copy its completed contents out
    /// of the temporary shared mapping. This is the native counterpart to
    /// [`Self::upload_payload`] for activation/output handoff.
    pub fn download_payload(&self, handle: u32, size: usize) -> Result<Vec<u8>, String> {
        if handle == 0 || size == 0 {
            return Err("XDNA payload download requires a handle and nonzero size".into());
        }
        let info = self.bo_info(handle)?;
        self.sync_bo(handle, SYNC_FROM_DEVICE, size as u64)?;
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                self.device.as_raw_fd(),
                info.map_offset as libc::off_t,
            )
        };
        if mapping == libc::MAP_FAILED {
            return Err(format!(
                "map XDNA output BO: {}",
                std::io::Error::last_os_error()
            ));
        }
        let mut payload = vec![0u8; size];
        unsafe {
            std::ptr::copy_nonoverlapping(mapping.cast::<u8>(), payload.as_mut_ptr(), size);
        }
        let unmap_result = unsafe { libc::munmap(mapping, size) };
        if unmap_result != 0 {
            return Err(format!(
                "unmap XDNA output BO: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(payload)
    }

    pub fn bo_info(&self, handle: u32) -> Result<XdnaBoInfo, String> {
        if handle == 0 {
            return Err("cannot query an invalid zero XDNA BO handle".into());
        }
        let mut request = GetBoInfo {
            handle,
            ..Default::default()
        };
        ioctl_call(self.device.as_raw_fd(), ioctl_get_bo_info(), &mut request)?;
        if request.device_address == 0 {
            return Err("amdxdna returned an invalid zero device address".into());
        }
        Ok(XdnaBoInfo {
            map_offset: request.map_offset,
            user_address: request.user_address,
            device_address: request.device_address,
        })
    }

    /// Resolve all lowered program resources through the driver and attach
    /// their device addresses to a command buffer.
    pub fn bind_command_buffer(
        &self,
        command: XdnaCommandBuffer,
        bo_handles: &HashMap<String, u32>,
    ) -> Result<XdnaCommandBuffer, String> {
        let mut bindings = Vec::with_capacity(command.program.buffers.len());
        for buffer in &command.program.buffers {
            let handle = bo_handles
                .get(&buffer.id)
                .ok_or_else(|| format!("missing BO handle for XDNA resource {}", buffer.id))?;
            let info = self.bo_info(*handle)?;
            bindings.push(DeviceAddressBinding {
                resource: buffer.id.clone(),
                device_address: info.device_address,
                bytes: buffer.bytes,
            });
        }
        command.with_addresses(bindings)
    }

    /// Submit a kernel command buffer handle to an existing hardware context.
    /// The command buffer itself is produced by a future firmware-specific
    /// lowering pass; Prism does not reinterpret it as an arbitrary bincode
    /// packet.
    pub fn exec_command_buffer(&self, hwctx: u32, command_bo: u32) -> Result<u64, String> {
        if hwctx == 0 || command_bo == 0 {
            return Err("XDNA execution requires valid context and command handles".into());
        }
        let mut command = command_bo;
        let mut request = ExecCmd {
            ext: 0,
            ext_flags: 0,
            hwctx,
            command_type: CMD_SUBMIT_EXEC_BUF,
            cmd_handles: (&mut command as *mut u32) as u64,
            args: 0,
            cmd_count: 1,
            arg_count: 0,
            seq: 0,
        };
        ioctl_call(self.device.as_raw_fd(), ioctl_exec_cmd(), &mut request)?;
        Ok(request.seq)
    }

    /// Reject portable Prism command envelopes at the firmware boundary. Use
    /// [`Self::submit_firmware_command_buffer`] after a firmware-specific
    /// encoder has produced the device packet.
    pub fn submit_command_buffer(
        &self,
        hwctx: u32,
        command: &XdnaCommandBuffer,
    ) -> Result<u64, String> {
        self.preflight_submission(hwctx, &command.program)?;
        // `XdnaCommandBuffer::encode` is Prism's portable inspection and
        // artifact format, not the firmware-versioned packet accepted by
        // AMDXDNA_EXEC_CMD. Never place that envelope in a command BO.
        let _ = command;
        Err("amdxdna firmware command encoding is required before Linux submission; use submit_firmware_command_buffer".into())
    }

    /// Submit bytes produced by a firmware-specific XDNA command encoder.
    /// Prism owns validation, BO allocation, mapping, synchronization, and
    /// ioctl submission; the encoder owns the firmware ABI packet layout.
    pub fn submit_firmware_command_buffer(
        &self,
        hwctx: u32,
        program: &XdnaProgram,
        payload: &[u8],
    ) -> Result<u64, String> {
        self.preflight_submission(hwctx, program)?;
        if payload.is_empty() {
            return Err("XDNA firmware command payload is empty".into());
        }
        let size = u64::try_from(payload.len())
            .map_err(|_| "XDNA command buffer is too large".to_string())?;
        let handle = self.create_bo(size, BO_TYPE_CMD)?;
        let info = self.bo_info(handle)?;
        if info.map_offset == 0 {
            return Err("amdxdna returned no mappable command-BO offset".into());
        }
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                payload.len(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.device.as_raw_fd(),
                info.map_offset as libc::off_t,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(format!(
                "map XDNA command BO: {}",
                std::io::Error::last_os_error()
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(payload.as_ptr(), mapped.cast::<u8>(), payload.len());
        }
        let sync_result = self.sync_bo(handle, SYNC_TO_DEVICE, size);
        unsafe {
            libc::munmap(mapped, payload.len());
        }
        sync_result?;
        self.exec_command_buffer(hwctx, handle)
    }

    /// Encode and submit a validated Prism command through a caller-supplied
    /// firmware adapter. The adapter is deliberately small: it only owns the
    /// versioned packet layout, while this probe owns all Linux resource and
    /// synchronization operations.
    pub fn submit_encoded_command_buffer<E: XdnaFirmwareEncoder>(
        &self,
        hwctx: u32,
        command: &XdnaCommandBuffer,
        encoder: &E,
    ) -> Result<u64, String> {
        self.preflight_submission(hwctx, &command.program)?;
        let payload = encoder
            .encode_firmware(command)
            .map_err(|error| format!("encode XDNA firmware command: {error}"))?;
        self.submit_firmware_command_buffer(hwctx, &command.program, &payload)
    }

    /// Reject a command before allocating/submitting a command BO when its
    /// compiled XDNA topology cannot execute on this adapter.
    pub fn validate_program_topology(&self, program: &XdnaProgram) -> Result<(), String> {
        validate_program_topology(program, &self.topology()?)
    }

    /// Validate all locally observable prerequisites before a firmware packet
    /// is allocated or submitted. Hardware-context creation is intentionally
    /// outside this module because its ABI is firmware/driver-versioned.
    pub fn preflight_submission(
        &self,
        hwctx: u32,
        program: &XdnaProgram,
    ) -> Result<XdnaSubmissionPreflight, String> {
        if hwctx == 0 {
            return Err("XDNA submission preflight requires a nonzero hardware context".into());
        }
        let topology = self.topology()?;
        validate_program_topology(program, &topology)?;
        Ok(XdnaSubmissionPreflight {
            device_path: self.path.clone(),
            hwctx,
            topology,
        })
    }

    pub fn topology(&self) -> Result<XdnaTopology, String> {
        let mut metadata = AieMetadata::default();
        let mut request = GetInfo {
            param: AMDXDNA_QUERY_AIE_METADATA,
            buffer_size: std::mem::size_of::<AieMetadata>() as u32,
            buffer: (&mut metadata as *mut AieMetadata) as u64,
        };
        let result =
            unsafe { libc::ioctl(self.device.as_raw_fd(), ioctl_get_info(), &mut request) };
        if result < 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        if metadata.cols == 0 || metadata.rows == 0 {
            return Err("amdxdna returned an empty AIE topology".into());
        }
        let compute_tiles = (0..metadata.cols)
            .flat_map(|col| {
                (metadata.core.row_start..metadata.core.row_start + metadata.core.row_count)
                    .map(move |row| TileCoord { col, row })
            })
            .collect();
        let memory_tiles = (0..metadata.cols)
            .flat_map(|col| {
                (metadata.mem.row_start..metadata.mem.row_start + metadata.mem.row_count)
                    .map(move |row| TileCoord { col, row })
            })
            .collect();
        let generation = if metadata.version_major >= 2 {
            XdnaGeneration::Aie2p
        } else {
            XdnaGeneration::Aie2
        };
        let topology = XdnaTopology {
            generation,
            columns: metadata.cols,
            rows: metadata.rows,
            compute_tiles,
            memory_tiles,
            shim_dma_channels: metadata.shim.dma_channel_count.max(1),
            // The UAPI reports geometry, not local memory capacity. Keep the
            // conservative target default until firmware exposes this value.
            tile_memory_bytes: if generation == XdnaGeneration::Aie2p {
                32 * 1024
            } else {
                16 * 1024
            },
            l2_memory_bytes: if generation == XdnaGeneration::Aie2p {
                4 * 1024 * 1024
            } else {
                2560 * 1024
            },
            max_fifo_depth: if generation == XdnaGeneration::Aie2p {
                16
            } else {
                8
            },
        };
        topology.validate().map_err(|errors| errors.join("; "))?;
        Ok(topology)
    }

    /// Discover the target profile consumed directly by Prism lowering.
    pub fn target(&self) -> Result<XdnaTarget, String> {
        Ok(XdnaTarget {
            topology: self.topology()?,
            default_element_type: prism_spatial_ir::xdna::XdnaElementType::Int8,
        })
    }
}

fn validate_program_topology(program: &XdnaProgram, device: &XdnaTopology) -> Result<(), String> {
    program.validate().map_err(|errors| errors.join("; "))?;
    if program.topology.generation != device.generation {
        return Err(format!(
            "XDNA generation mismatch: program {:?}, device {:?}",
            program.topology.generation, device.generation
        ));
    }
    if program.topology.columns > device.columns || program.topology.rows > device.rows {
        return Err(format!(
            "XDNA topology {}x{} exceeds device {}x{}",
            program.topology.columns, program.topology.rows, device.columns, device.rows
        ));
    }
    let in_device = |tile: TileCoord| {
        tile.col < device.columns && tile.row < device.rows && device.compute_tiles.contains(&tile)
    };
    for tile in program.workers.iter().map(|worker| worker.tile).chain(
        program
            .fifos
            .iter()
            .flat_map(|fifo| [fifo.producer, fifo.consumer]),
    ) {
        if !in_device(tile) {
            return Err(format!(
                "XDNA program references tile ({}, {}) absent from device topology",
                tile.col, tile.row
            ));
        }
    }
    if program
        .transfers
        .iter()
        .any(|transfer| transfer.channel >= device.shim_dma_channels)
    {
        return Err("XDNA program uses a DMA channel absent from device topology".into());
    }
    Ok(())
}

fn ioctl_get_info() -> libc::c_ulong {
    // Linux _IOWR encoding: direction=read|write, type='d', command nr,
    // payload size. This is the DRM command ioctl for amdxdna GET_INFO.
    const IOC_NRBITS: u32 = 8;
    const IOC_TYPEBITS: u32 = 8;
    const IOC_SIZEBITS: u32 = 14;
    const IOC_DIRBITS: u32 = 2;
    const IOC_NRSHIFT: u32 = 0;
    const IOC_TYPESHIFT: u32 = IOC_NRSHIFT + IOC_NRBITS;
    const IOC_SIZESHIFT: u32 = IOC_TYPESHIFT + IOC_TYPEBITS;
    const IOC_DIRSHIFT: u32 = IOC_SIZESHIFT + IOC_SIZEBITS;
    let dir = (1_u32 << IOC_DIRBITS) - 1;
    ((dir << IOC_DIRSHIFT)
        | ((DRM_IOCTL_BASE as u32) << IOC_TYPESHIFT)
        | (((DRM_COMMAND_BASE + AMDXDNA_GET_INFO) as u32) << IOC_NRSHIFT)
        | ((std::mem::size_of::<GetInfo>() as u32) << IOC_SIZESHIFT)) as libc::c_ulong
}

fn ioctl_create_bo() -> libc::c_ulong {
    drm_iowr(AMDXDNA_CREATE_BO, std::mem::size_of::<CreateBo>())
}
fn ioctl_get_bo_info() -> libc::c_ulong {
    drm_iowr(AMDXDNA_GET_BO_INFO, std::mem::size_of::<GetBoInfo>())
}
fn ioctl_sync_bo() -> libc::c_ulong {
    drm_iowr(AMDXDNA_SYNC_BO, std::mem::size_of::<SyncBo>())
}
fn ioctl_exec_cmd() -> libc::c_ulong {
    drm_iowr(AMDXDNA_EXEC_CMD, std::mem::size_of::<ExecCmd>())
}

fn drm_iowr(command: u8, size: usize) -> libc::c_ulong {
    const IOC_NRBITS: u32 = 8;
    const IOC_TYPEBITS: u32 = 8;
    const IOC_SIZEBITS: u32 = 14;
    const IOC_DIRBITS: u32 = 2;
    let dir = (1_u32 << IOC_DIRBITS) - 1;
    ((dir << (IOC_SIZEBITS + IOC_TYPEBITS + IOC_NRBITS))
        | ((DRM_IOCTL_BASE as u32) << IOC_NRBITS)
        | ((DRM_COMMAND_BASE + command) as u32)
        | ((size as u32) << (IOC_NRBITS + IOC_TYPEBITS))) as libc::c_ulong
}

fn ioctl_call<T>(
    fd: std::os::fd::RawFd,
    request: libc::c_ulong,
    value: &mut T,
) -> Result<(), String> {
    let result = unsafe { libc::ioctl(fd, request, value as *mut T) };
    if result < 0 {
        Err(std::io::Error::last_os_error().to_string())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_info_ioctl_is_nonzero() {
        assert_ne!(ioctl_get_info(), 0);
        assert_ne!(ioctl_create_bo(), 0);
        assert_ne!(ioctl_get_bo_info(), 0);
        assert_ne!(ioctl_sync_bo(), 0);
        assert_ne!(ioctl_exec_cmd(), 0);
    }
}
