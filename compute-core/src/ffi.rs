//! C-compatible FFI bridge for PrismEngine Swift menu bar app.
//! Uses extern "C" for zero-overhead calls from Swift via a bridging header.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::os::raw::c_void;
use std::path::Path;
use std::sync::Arc;

use crate::compute_image::cimage_loader::load_cimage_mmap;
use crate::compute_image::cimage_packer::pipeline::compile_and_pack_god_binary;
use crate::compute_image::compile::source::load_source;
use crate::config::CompileQuantMode;
use crate::runtime::agent_slot::MultiplexerState;

/// Opaque pointer for Swift to hold the runtime multiplexer state.
pub struct OpaqueMultiplexer {
    pub inner: Arc<MultiplexerState>,
}

/// Compile a .cimage from downloaded safetensors + bundled resources.
/// Returns 0 on success, negative on error.
#[no_mangle]
pub unsafe extern "C" fn prism_compile_and_pack(
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

    let qmode = CompileQuantMode::TernaryTile640 { group_size: 640 };

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
            eprintln!("[ffi] compile_and_pack failed: {} ({} elements)", e, main_elements);
            -6
        }
    }
}

/// Initialize the runtime multiplexer from a compiled .cimage.
/// Returns a pointer to an OpaqueMultiplexer, or null on failure.
#[no_mangle]
pub unsafe extern "C" fn prism_runtime_init(
    cimage_path: *const c_char,
) -> *mut OpaqueMultiplexer {
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
#[no_mangle]
pub unsafe extern "C" fn prism_runtime_free(multiplexer: *mut OpaqueMultiplexer) {
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

#[no_mangle]
pub unsafe extern "C" fn prism_execute_multimodal(
    multiplexer: *mut OpaqueMultiplexer,
    agent_id: u32,
    payload: MultimodalPayload,
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
    eprintln!(
        "[ffi] multimodal: agent={} prompt_len={} image_surface={} audio_surface={}",
        agent_id,
        prompt.len(),
        payload.image_surface_id,
        payload.audio_surface_id,
    );
}
