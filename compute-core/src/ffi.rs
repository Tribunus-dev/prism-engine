//! C-compatible FFI bridge for PrismEngine Swift menu bar app.
//! Rust-native API (formerly extern "C" FFI). Raw-pointer params require unsafe.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::sync::Arc;

use crate::compute_image::cimage_loader::load_cimage_mmap;
use crate::compute_image::cimage_packer::pipeline::compile_and_pack_god_binary;
use crate::compute_image::compile::source::load_source;
use crate::config::CompileQuantMode;
use crate::runtime::agent_slot::MultiplexerState;

use crate::device::DeviceRegistry;

/// Opaque pointer for Swift to hold the runtime multiplexer state.
pub struct OpaqueMultiplexer {
    pub inner: Arc<MultiplexerState>,
}

/// Compile a .cimage from downloaded safetensors + bundled resources.
/// Returns 0 on success, negative on error.
pub unsafe fn prism_compile_and_pack(
    safetensors_dir: *const c_char,
    output_cimage_path: *const c_char,
    resource_dir: *const c_char,
) -> c_int {
    if safetensors_dir.is_null() || output_cimage_path.is_null() || resource_dir.is_null() {
        return -1;
    }
    let safetensors = match CStr::from_ptr(safetensors_dir).to_str() {
        Ok(s) => Path::new(s),
        Err(_) => return -2,
    };
    let output = match CStr::from_ptr(output_cimage_path).to_str() {
        Ok(s) => Path::new(s),
        Err(_) => return -3,
    };
    let resources = match CStr::from_ptr(resource_dir).to_str() {
        Ok(s) => Path::new(s),
        Err(_) => return -4,
    };

    let metallib = resources.join("default.metallib");
    let main_mlmodelc = resources.join("main_12b.mlmodelc");
    let mtp_mlmodelc = resources.join("mtp_1b.mlmodelc");

    let output_str = output.to_str().unwrap_or("");

    // Load source metadata from the safetensors directory.
    let mut loaded = match load_source(safetensors, true) {
        Ok(ls) => ls,
        Err(_) => return -5,
    };

    // Compute total element counts from the loaded safetensors.
    // Iterates all source tensors, summing BF16 element counts.
    let main_elements: u64 = loaded
        .source_tensors
        .values()
        .filter(|t| t.name.ends_with(".weight"))
        .map(|t| (t.data.len() / 2) as u64)
        .sum();

    // MTP head elements: derived from architecture if available, else auto-detect
    // from tensors matching ".draft" or "mtp" pattern.  Default to 1B if unknown.
    let mtp_elements: u64 = loaded
        .source_tensors
        .values()
        .filter(|t| t.name.contains("mtp") || t.name.contains("draft"))
        .map(|t| (t.data.len() / 2) as u64)
        .sum::<u64>()
        .max(1_000_000_000); // at least ~1B for a reasonable draft head

    // Extract architecture dimensions for the topology table.
    let hs = loaded.arch.hidden_size;
    let interm = loaded.arch.intermediate_size;
    let n_layers = loaded.arch.num_hidden_layers;
    let n_heads = loaded.arch.num_attention_heads;
    let head_dim = loaded.arch.head_dim;

    let qmode = CompileQuantMode::Nf4Tile640 { group_size: 128 };

    match compile_and_pack_god_binary(
        output_str,
        &metallib,
        &main_mlmodelc,
        &mtp_mlmodelc,
        main_elements,
        mtp_elements,
        &mut loaded,
        qmode,
        hs,
        interm,
        n_layers,
        n_heads,
        head_dim,
    ) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!(
                "[ffi] compile_and_pack failed: {} ({} elements)",
                e, main_elements
            );
            -6
        }
    }
}

/// Initialize the runtime multiplexer from a compiled .cimage.
/// Returns a pointer to an OpaqueMultiplexer, or null on failure.
pub unsafe fn prism_runtime_init(cimage_path: *const c_char) -> *mut OpaqueMultiplexer {
    if cimage_path.is_null() {
        return std::ptr::null_mut();
    }
    let path = match CStr::from_ptr(cimage_path).to_str() {
        Ok(s) => Path::new(s),
        Err(_) => return std::ptr::null_mut(),
    };

    match load_cimage_mmap(path) {
        Ok((mmap, header)) => {
            let mmap_arc = Arc::new(mmap);
            let mut state = MultiplexerState::new();
            // Dimensions come from the topology table embedded in the .cimage.
            // For now, use Gemma 4 12B defaults — in production they are
            // parsed from the topology table at init time.
            state.init_from_cimage(mmap_arc, &header, 3840, 18432);
            let opaque = Box::new(OpaqueMultiplexer {
                inner: Arc::new(state),
            });
            Box::into_raw(opaque)
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a previously initialized OpaqueMultiplexer.
pub unsafe fn prism_runtime_free(multiplexer: *mut OpaqueMultiplexer) {
    if !multiplexer.is_null() {
        let _ = Box::from_raw(multiplexer);
    }
}

#[repr(C)]
pub struct MultimodalPayload {
    pub text_prompt: *const c_char,
    pub image_surface_id: u32,
    pub audio_surface_id: u32,
}

pub unsafe fn prism_execute_multimodal(
    multiplexer: *mut OpaqueMultiplexer,
    agent_id: u32,
    payload: MultimodalPayload,
) {
    if multiplexer.is_null() {
        return;
    }
    let _state = &(*multiplexer).inner;
    let prompt = if !payload.text_prompt.is_null() {
        CStr::from_ptr(payload.text_prompt).to_str().unwrap_or("")
    } else {
        ""
    };
    eprintln!(
        "[ffi] multimodal: agent={} prompt_len={} image_surface={} audio_surface={}",
        agent_id,
        prompt.len(),
        payload.image_surface_id,
        payload.audio_surface_id,
    );
}

/// Extended multimodal execution with priority and lane pinning.
/// Stores the lane hint on the agent slot for the tri-lane orchestrator.
pub unsafe fn prism_execute_multimodal_ex(
    multiplexer: *mut OpaqueMultiplexer,
    agent_id: u32,
    payload: MultimodalPayload,
    priority: u32,
    lane_hint: u32,
) {
    if multiplexer.is_null() {
        return;
    }
    let state = &(*multiplexer).inner;
    let prompt = if !payload.text_prompt.is_null() {
        CStr::from_ptr(payload.text_prompt).to_str().unwrap_or("")
    } else {
        ""
    };

    // Validate lane_hint range [0, 3]
    let lane = lane_hint.min(3);

    // Store lane hint + priority directly on the multiplexer state.
    // The tri-lane orchestrator reads these during PhaseVariant selection.
    if let Ok(mut hints) = state.agent_hints.lock() {
        hints.insert(agent_id, (lane, priority));
    }

    eprintln!(
        "[ffi] multimodal_ex: agent={} prompt_len={} priority={} lane_hint={} image_surface={} audio_surface={}",
        agent_id,
        prompt.len(),
        priority,
        lane,
        payload.image_surface_id,
        payload.audio_surface_id,
    );
}

// ── Device discovery FFI ───────────────────────────────────────────────────

/// C-compatible device info struct for host apps.
#[repr(C)]
pub struct PrismDeviceInfo {
    /// Device index.
    pub index: u32,
    /// Device kind: 0=CPU, 1=dGPU, 2=iGPU, 3=Unified GPU, 4=NPU, 5=Accelerator
    pub kind: u32,
    /// Backend kind: 0=Metal, 1=CUDA, 2=ROCm, 3=Level Zero, 4=Core ML, 5=ANE, etc.
    pub backend: u32,
    /// Pointer to a null-terminated UTF-8 device name string.
    /// Must be freed with prism_device_info_free_name().
    pub name: *const c_char,
    /// Pointer to a null-terminated UTF-8 vendor string.
    pub vendor: *const c_char,
    /// Total memory in bytes (0 if unknown).
    pub memory_bytes: u64,
    /// Number of compute units / GPU cores.
    pub compute_units: u32,
    /// Whether memory is unified with CPU.
    pub unified_memory: bool,
    /// Whether FP16 is supported.
    pub supports_f16: bool,
    /// Whether the device is the default inference target.
    pub is_default: bool,
}

/// Return the number of discovered compute devices.
pub fn prism_device_count() -> u32 {
    DeviceRegistry::discover().count() as u32
}

/// Fill a PrismDeviceInfo struct for device at `index`.
/// Returns 0 on success, -1 if index out of range.
pub unsafe fn prism_device_info(index: u32, info: *mut PrismDeviceInfo) -> c_int {
    let registry = DeviceRegistry::discover();
    let devices = registry.enumerate();
    let device = match devices.get(index as usize) {
        Some(d) => d,
        None => return -1,
    };

    let default_id = registry.default().map(|d| d.id.0);

    unsafe {
        *info = PrismDeviceInfo {
            index,
            kind: device.kind as u32,
            backend: device.backend as u32,
            name: std::ffi::CString::new(device.name.clone())
                .unwrap_or_default()
                .into_raw(),
            vendor: std::ffi::CString::new(device.vendor.clone())
                .unwrap_or_default()
                .into_raw(),
            memory_bytes: device.memory.total_bytes,
            compute_units: device.compute_units,
            unified_memory: device.memory.unified_with_cpu,
            supports_f16: device.supports_f16,
            is_default: Some(device.id.0) == default_id,
        };
    }
    0
}

/// Free the name and vendor strings allocated by prism_device_info.
pub unsafe fn prism_device_info_free_name(name: *mut c_char) {
    if !name.is_null() {
        let _ = std::ffi::CString::from_raw(name);
    }
}

/// Free the vendor string allocated by prism_device_info.
pub unsafe fn prism_device_info_free_vendor(vendor: *mut c_char) {
    if !vendor.is_null() {
        let _ = std::ffi::CString::from_raw(vendor);
    }
}

/// Get device info as a JSON string. Caller must free with prism_free_json_string().
pub unsafe fn prism_device_list_json() -> *mut c_char {
    let registry = DeviceRegistry::discover();
    let json = registry.to_json_pretty();
    std::ffi::CString::new(json).unwrap_or_default().into_raw()
}

/// Free a JSON string allocated by prism_device_list_json().
pub unsafe fn prism_free_json_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = std::ffi::CString::from_raw(s);
    }
}

// ── Config FFI ────────────────────────────────────────────────────────────────

/// C-compatible server config struct for host apps.
#[repr(C)]
pub struct PrismServerConfig {
    pub port: u16,
    pub max_concurrent: u32,
    pub rate_limit_per_min: u32,
    pub log_level: [u8; 32],    // fixed-size buffer for C safety
    pub runtime_mode: [u8; 32], // fixed-size buffer for C safety
    pub kv_cache_tiers: u32,
    pub compression_ratio: f64,
    pub evolkv_enabled: bool,
    pub draft_count: u32,
    pub draft_length: u32,
    pub spechub_enabled: bool,
    pub exo_enabled: bool,
    pub exo_port: u16,
    pub model_path: [u8; 1024], // fixed-size buffer, empty if none
    pub auto_download: bool,
}

/// Fill a PrismServerConfig struct with the current configuration.
/// Loads from default config file path + env vars.
pub fn prism_load_config(config: *mut PrismServerConfig) {
    use crate::config::ServerConfig;
    let cfg = ServerConfig::load();

    unsafe {
        let log_level = cfg.server.log_level;
        let log_bytes = log_level.as_bytes();
        let mut log_buf = [0u8; 32];
        let copy_len = log_bytes.len().min(31);
        log_buf[..copy_len].copy_from_slice(&log_bytes[..copy_len]);
        log_buf[copy_len] = 0;

        let runtime_mode = cfg.server.runtime_mode;
        let mode_bytes = runtime_mode.as_bytes();
        let mut mode_buf = [0u8; 32];
        let copy_len = mode_bytes.len().min(31);
        mode_buf[..copy_len].copy_from_slice(&mode_bytes[..copy_len]);
        mode_buf[copy_len] = 0;

        let model_path = cfg.model.model_path.unwrap_or_default();
        let path_bytes = model_path.as_bytes();
        let mut path_buf = [0u8; 1024];
        let copy_len = path_bytes.len().min(1023);
        path_buf[..copy_len].copy_from_slice(&path_bytes[..copy_len]);
        path_buf[copy_len] = 0;

        *config = PrismServerConfig {
            port: cfg.server.port,
            max_concurrent: cfg.server.max_concurrent,
            rate_limit_per_min: cfg.server.rate_limit_per_min,
            log_level: log_buf,
            runtime_mode: mode_buf,
            kv_cache_tiers: cfg.cache.kv_cache_tiers,
            compression_ratio: cfg.cache.compression_ratio,
            evolkv_enabled: cfg.cache.evolkv_enabled,
            draft_count: cfg.speculation.draft_count,
            draft_length: cfg.speculation.draft_length,
            spechub_enabled: cfg.speculation.spechub_enabled,
            exo_enabled: cfg.cluster.exo_enabled,
            exo_port: cfg.cluster.exo_port,
            model_path: path_buf,
            auto_download: cfg.model.auto_download,
        };
    }
}
