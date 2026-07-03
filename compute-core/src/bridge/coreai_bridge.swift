// Tribunus Core AI Swift bridge — `@C` attribute FFI wrappers.
//
// Replaces coreai_exec.mm once build.rs supports Swift compilation.
// Core AI types (AIModel, InferenceFunction, NDArray) are Swift
// structs — not bridgeable from ObjC++.
//
// Compile: swiftc -c -emit-object \
//   -module-name CoreAiBridge \
//   -o coreai_bridge.o \
//   src/bridge/coreai_bridge.swift

import CoreAI
import Foundation
import os

private let coreai_log = OSLog(
    subsystem: "com.tribunus.compute",
    category: "coreai_bridge"
)

// ── C-compatible struct mirror ──────────────────────────────────────────
//
// Mirrors TribunusArenaInfo from coreai_arena.h so we can read arena
// pointers directly without an ObjC bridging header. Layout must match
// #[repr(C)] order exactly.

private struct ArenaInfo {
    let width: Int32
    let height: Int32
    let logical_dim0: Int32
    let logical_dim1: Int32
    let pixel_format: Int32
    let byte_size: Int32
    let bytes_per_row: UInt32
    let base_address: UnsafeMutableRawPointer?
    let cv_buffer: UnsafeMutableRawPointer?
    let io_surface: UnsafeMutableRawPointer?
}

// ── Model lifecycle ──────────────────────────────────────────────────────

/// Load a compiled .aimodel file and retrieve its main InferenceFunction.
/// Returns 0 on success, negative on error.
@C("tribunus_coreai_load_model")
public func coreai_load_model(
    _ out_model: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
    _ model_path: UnsafePointer<CChar>,
    _ compute_units: Int64
) -> Int32 {
    let path = String(cString: model_path)
    let url = URL(fileURLWithPath: path)

    guard let model = try? AIModel(contentsOf: url) else {
        os_log(.error, log: coreai_log,
               "coreai_load_model: failed to open %{public}s", path)
        return -2
    }

    guard let fn = try? model.loadFunction(named: "main") else {
        os_log(.error, log: coreai_log,
               "coreai_load_model: no 'main' function in %{public}s", path)
        return -3
    }

    let ptr = Unmanaged.passRetained(fn).toOpaque()
    out_model.pointee = ptr
    os_log(.info, log: coreai_log,
           "coreai_load_model: loaded %{public}s", path)
    return 0
}

/// Release a loaded InferenceFunction handle.
@C("tribunus_coreai_free_model")
public func coreai_free_model(_ model_ptr: UnsafeMutableRawPointer?) {
    guard let ptr = model_ptr else { return }
    Unmanaged<InferenceFunction>.fromOpaque(ptr).release()
}

// ── Stateless inference ──────────────────────────────────────────────────

/// Run stateless inference: copy arena data into NDArray, execute, copy back.
/// TODO(#coreai): switch to zero-copy once Core AI NDArray supports
/// wrapping an IOSurface or raw data pointer directly.
@C("tribunus_coreai_predict")
public func coreai_predict(
    _ model_ptr: UnsafeMutableRawPointer?,
    _ input_name: UnsafePointer<CChar>?,
    _ input_arena: UnsafePointer<ArenaInfo>?,
    _ output_name: UnsafePointer<CChar>?,
    _ output_arena: UnsafeMutablePointer<ArenaInfo>?
) -> Int32 {
    guard let model_ptr = model_ptr,
          let input_name = input_name,
          let input_arena = input_arena,
          let output_name = output_name,
          let output_arena = output_arena else {
        return -1
    }

    let fn = Unmanaged<InferenceFunction>
        .fromOpaque(model_ptr)
        .takeUnretainedValue()
    let arena = input_arena.pointee
    let out_info = output_arena.pointee

    let dim0 = Int(arena.logical_dim0)
    let dim1 = Int(arena.logical_dim1)
    let count = dim0 * dim1

    guard let in_data = arena.base_address?
        .assumingMemoryBound(to: Float.self) else { return -2 }
    guard let out_data = out_info.base_address?
        .assumingMemoryBound(to: Float.self) else { return -3 }

    let in_name = String(cString: input_name)
    let out_name = String(cString: output_name)

    // Build NDArrays (copy-based for now)
    var in_nd = NDArray(shape: [dim0, dim1], scalarType: .float32)
    in_nd.mutableView(as: Float.self)
        .withUnsafeMutableBufferPointer { dst in
            _ = dst.initialize(
                from: UnsafeBufferPointer(start: in_data, count: count)
            )
        }

    var out_nd = NDArray(shape: [dim0, dim1], scalarType: .float32)

    let inputs = InferenceFunction.Inputs()
    inputs.set(in_nd, for: in_name)

    var outputs = InferenceFunction.Outputs()
    outputs.set(&out_nd, for: out_name)

    do {
        try fn.run(inputs: inputs, outputs: &outputs)
        // Copy output back to arena
        out_nd.view(as: Float.self)
            .withUnsafeBufferPointer { src in
                UnsafeMutableBufferPointer(
                    start: out_data, count: count
                ).initialize(from: src)
            }
        return 0
    } catch {
        os_log(.error, log: coreai_log,
               "coreai_predict: %{public}s",
               error.localizedDescription)
        return -5
    }
}

// ── Shutdown ──────────────────────────────────────────────────────────────

@C("tribunus_coreai_shutdown")
public func coreai_shutdown() {
    // No global resources to release in current prototype.
}
