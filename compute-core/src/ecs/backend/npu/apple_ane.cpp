// Apple ANE backend — Core ML C FFI wrapper.
// Real implementation using ANE via the existing tribunus_ane_eval C API
// defined in compute-core/src/ane_bridge.rs.
//
// The ANE loads pre-compiled .mlmodelc bundles (produced by coremlcompiler).
// Input/output are IOSurface-backed buffers for zero-copy.

// On macOS, these are declared in ane_bridge.rs and linked as C symbols.
// This file provides the unified NPU dispatch interface over them.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations of the existing ANE FFI functions from ane_bridge.rs
// These are compiled from Rust and linked into the final binary.
extern void* tribunus_ane_load(const char* modelc_path);
extern int tribunus_ane_eval(void* program, void** inputs, int num_inputs,
                              void** outputs, int num_outputs);
extern void tribunus_ane_unload(void* program);

struct AppleAneSession {
    void* model_handle;  // Core ML model handle from tribunus_ane_load
    char input_name[64];
    char output_name[64];
};

// Load a compiled Core ML .mlmodelc bundle for ANE execution.
// modelc_path: path to the .mlmodelc directory.
void* apple_ane_load_graph(const char* modelc_path) {
    auto* session = (AppleAneSession*)calloc(1, sizeof(AppleAneSession));
    if (!session) return nullptr;

    session->model_handle = tribunus_ane_load(modelc_path);
    if (!session->model_handle) {
        free(session);
        return nullptr;
    }

    // Default input/output names for ANE subgraphs
    strncpy(session->input_name, "input", sizeof(session->input_name) - 1);
    strncpy(session->output_name, "output", sizeof(session->output_name) - 1);

    return session;
}

// Submit async ANE execution.
// The ANE fires and returns immediately; completion is polled via
// IOSurface atomic flags (lock-free, matching Metal's pattern).
uint64_t apple_ane_submit(void* session_handle, void* input_buf, void* output_buf,
                           size_t input_bytes, size_t output_bytes) {
    auto* session = (AppleAneSession*)session_handle;
    if (!session) return 0;

    static uint64_t next_id = 1;
    uint64_t submission_id = __sync_fetch_and_add(&next_id, 1);

    void* inputs[] = { input_buf };
    void* outputs[] = { output_buf };

    int rc = tribunus_ane_eval(
        session->model_handle,
        inputs, 1,  // num_inputs
        outputs, 1  // num_outputs
    );

    if (rc != 0) return 0;

    // ANE execution is synchronous in the current tribunus_ane_eval
    // implementation. For true async, the IOSurface pixel buffer path
    // with Core ML's async prediction API would be used.
    // Return the ID for tracking in the ECS ledger.
    return submission_id;
}

// Poll for completion. For the ANE, the current tribunus_ane_eval is
// synchronous, so completion is immediate. With IOSurface-backed async
// prediction, this would check an atomic completion flag on the surface.
int apple_ane_poll(void* session_handle, uint64_t submission_id) {
    (void)session_handle;
    (void)submission_id;
    // Synchronous model: always complete after submit returns.
    return 1;
}

void apple_ane_destroy_session(void* session_handle) {
    auto* session = (AppleAneSession*)session_handle;
    if (!session) return;
    if (session->model_handle) {
        tribunus_ane_unload(session->model_handle);
    }
    free(session);
}

#ifdef __cplusplus
}
#endif
