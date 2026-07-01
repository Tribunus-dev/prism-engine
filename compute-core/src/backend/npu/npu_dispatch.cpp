// Unified NPU dispatch — routes to the correct vendor backend.
// Called by the Rust FFI bindings in backend/npu/ffi.rs

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

// Vendor backend implementations
#ifdef __APPLE__
extern void* apple_ane_load_graph(const char* path);
extern uint64_t apple_ane_submit(void* session, void* in, void* out, size_t in_sz, size_t out_sz);
extern int apple_ane_poll(void* session, uint64_t id);
extern void apple_ane_destroy_session(void* session);
#endif

#ifdef __linux__
extern void* intel_vpu_load_graph(const char* path);
extern uint64_t intel_vpu_submit(void* session, void* in, void* out, size_t in_sz, size_t out_sz);
extern int intel_vpu_poll(void* session, uint64_t id);
extern void intel_vpu_destroy_session(void* session);

extern void* amd_xdna_load_graph(const char* xclbin, const char* kernel);
extern uint64_t amd_xdna_submit(void* session, void* in, void* out, size_t in_sz, size_t out_sz);
extern int amd_xdna_poll(void* session, uint64_t id);
extern void amd_xdna_destroy_session(void* session);
#endif

// Target NPU enum matching backend/npu/ffi.rs
enum TargetNpu {
    AppleAne = 0,
    IntelVpu = 1,
    AmdXdna = 2,
};

#ifdef __cplusplus
extern "C" {
#endif

// ── Graph loading ─────────────────────────────────────────────────────────

void* npu_load_graph(int target, const char* blob_path) {
    switch (target) {
#ifdef __APPLE__
        case AppleAne:
            return apple_ane_load_graph(blob_path);
#endif
#ifdef __linux__
        case IntelVpu:
            return intel_vpu_load_graph(blob_path);
        case AmdXdna:
            // AMD xclbin path: blob_path is "xclbin_path:kernel_name"
            const char* colon = strchr(blob_path, ':');
            if (!colon) return nullptr;
            size_t path_len = colon - blob_path;
            char* xclbin = (char*)malloc(path_len + 1);
            memcpy(xclbin, blob_path, path_len);
            xclbin[path_len] = '\0';
            const char* kernel = colon + 1;
            void* session = amd_xdna_load_graph(xclbin, kernel);
            free(xclbin);
            return session;
#endif
        default:
            return nullptr;
    }
}

// ── Async submission ──────────────────────────────────────────────────────

uint64_t npu_submit_execution(int target, void* session,
                               void* input_buf, void* output_buf,
                               size_t input_bytes, size_t output_bytes) {
    switch (target) {
#ifdef __APPLE__
        case AppleAne:
            return apple_ane_submit(session, input_buf, output_buf, input_bytes, output_bytes);
#endif
#ifdef __linux__
        case IntelVpu:
            return intel_vpu_submit(session, input_buf, output_buf, input_bytes, output_bytes);
        case AmdXdna:
            return amd_xdna_submit(session, input_buf, output_buf, input_bytes, output_bytes);
#endif
        default:
            return 0;
    }
}

// ── Non-blocking poll ─────────────────────────────────────────────────────

int npu_poll_completion(int target, void* session, uint64_t submission_id) {
    switch (target) {
#ifdef __APPLE__
        case AppleAne:
            return apple_ane_poll(session, submission_id);
#endif
#ifdef __linux__
        case IntelVpu:
            return intel_vpu_poll(session, submission_id);
        case AmdXdna:
            return amd_xdna_poll(session, submission_id);
#endif
        default:
            return 0;
    }
}

// ── Session teardown ──────────────────────────────────────────────────────

void npu_destroy_session(int target, void* session) {
    switch (target) {
#ifdef __APPLE__
        case AppleAne:
            apple_ane_destroy_session(session);
            break;
#endif
#ifdef __linux__
        case IntelVpu:
            intel_vpu_destroy_session(session);
            break;
        case AmdXdna:
            amd_xdna_destroy_session(session);
            break;
#endif
        default:
            break;
    }
}

#ifdef __cplusplus
}
#endif
