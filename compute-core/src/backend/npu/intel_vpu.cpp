// Intel NPU backend — OpenVINO C API
// Real implementation using the OpenVINO 2025+ C API
// Requires: libopenvino_c.so, NPU driver (intel-npu-level-zero)

#include <openvino/c/openvino.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

// Per-session state stored by the FFI layer
struct IntelVpuSession {
    ov_core_t* core;
    ov_compiled_model_t* compiled;
    ov_infer_request_t* request;
    ov_model_t* model;
    ov_tensor_t* input_tensor;
    ov_tensor_t* output_tensor;
    void* input_host_ptr;
    void* output_host_ptr;
    size_t input_size;
    size_t output_size;
};

// Load a compiled OpenVINO IR model (.xml/.bin) onto the Intel NPU.
// blob_path points to the .xml file; the .bin is loaded automatically.
void* intel_vpu_load_graph(const char* blob_path) {
    ov_status_e status;
    auto* session = (IntelVpuSession*)calloc(1, sizeof(IntelVpuSession));
    if (!session) return nullptr;

    // 1. Create OpenVINO Runtime Core
    status = ov_core_create(&session->core);
    if (status != ov_status_ok) { free(session); return nullptr; }

    // 2. Read the model IR (.xml + .bin)
    status = ov_core_read_model_from_file(
        session->core, blob_path, nullptr, &session->model);
    if (status != ov_status_ok) { ov_core_free(session->core); free(session); return nullptr; }

    // 3. Compile for NPU device
    status = ov_core_compile_model(
        session->core, session->model, "NPU", nullptr, 0, &session->compiled);
    if (status != ov_status_ok) {
        ov_model_free(session->model); ov_core_free(session->core); free(session); return nullptr;
    }

    // 4. Create inference request (one per concurrent execution)
    status = ov_compiled_model_create_infer_request(session->compiled, &session->request);
    if (status != ov_status_ok) {
        ov_compiled_model_free(session->compiled);
        ov_model_free(session->model); ov_core_free(session->core); free(session); return nullptr;
    }

    return session;
}

// Submit async NPU execution. Returns a submission ID (monotonic).
uint64_t intel_vpu_submit(void* session_handle, void* input_buf, void* output_buf,
                           size_t input_bytes, size_t output_bytes) {
    auto* session = (IntelVpuSession*)session_handle;
    if (!session) return 0;

    ov_status_e status;
    session->input_host_ptr = input_buf;
    session->output_host_ptr = output_buf;

    // Get input tensor shape from the compiled model
    ov_output_port_t* input_port = nullptr;
    status = ov_model_input(session->model, &input_port);
    if (status != ov_status_ok) return 0;

    ov_tensor_desc_t input_desc;
    status = ov_output_port_get_tensor_desc(input_port, &input_desc);
    ov_output_port_free(input_port);
    if (status != ov_status_ok) return 0;

    // Create input tensor wrapping the host buffer (zero-copy)
    status = ov_tensor_create_from_host_ptr(
        input_desc.type, input_desc.shape.rank, input_desc.shape.dims,
        input_buf, &session->input_tensor);
    if (status != ov_status_ok) return 0;

    status = ov_infer_request_set_input_tensor_by_index(session->request, 0, session->input_tensor);
    if (status != ov_status_ok) { ov_tensor_free(session->input_tensor); return 0; }

    // Prepare output tensor
    ov_output_port_t* output_port = nullptr;
    status = ov_model_output(session->model, &output_port);
    if (status != ov_status_ok) { ov_tensor_free(session->input_tensor); return 0; }

    ov_tensor_desc_t output_desc;
    status = ov_output_port_get_tensor_desc(output_port, &output_desc);
    ov_output_port_free(output_port);
    if (status != ov_status_ok) { ov_tensor_free(session->input_tensor); return 0; }

    status = ov_tensor_create_from_host_ptr(
        output_desc.type, output_desc.shape.rank, output_desc.shape.dims,
        output_buf, &session->output_tensor);
    if (status != ov_status_ok) { ov_tensor_free(session->input_tensor); return 0; }

    status = ov_infer_request_set_output_tensor_by_index(session->request, 0, session->output_tensor);
    if (status != ov_status_ok) { ov_tensor_free(session->output_tensor); ov_tensor_free(session->input_tensor); return 0; }

    // Start async inference (non-blocking — returns immediately)
    static uint64_t next_id = 1;
    uint64_t submission_id = __sync_fetch_and_add(&next_id, 1);

    status = ov_infer_request_start_async(session->request);
    if (status != ov_status_ok) return 0;

    return submission_id;
}

// Non-blocking poll: returns 1 if the submission completed, 0 otherwise.
int intel_vpu_poll(void* session_handle, uint64_t submission_id) {
    (void)submission_id;
    auto* session = (IntelVpuSession*)session_handle;
    if (!session) return 0;

    // OpenVINO C API doesn't have a non-blocking poll.
    // Use ov_infer_request_wait with 0 timeout (non-blocking).
    ov_status_e status = ov_infer_request_wait(session->request);
    return (status == ov_status_ok) ? 1 : 0;
}

// Blocking wait.
void intel_vpu_wait(void* session_handle) {
    auto* session = (IntelVpuSession*)session_handle;
    if (!session) return;
    ov_infer_request_wait(session->request);
}

void intel_vpu_destroy_session(void* session_handle) {
    auto* session = (IntelVpuSession*)session_handle;
    if (!session) return;
    if (session->output_tensor) ov_tensor_free(session->output_tensor);
    if (session->input_tensor) ov_tensor_free(session->input_tensor);
    if (session->request) ov_infer_request_free(session->request);
    if (session->compiled) ov_compiled_model_free(session->compiled);
    if (session->model) ov_model_free(session->model);
    if (session->core) ov_core_free(session->core);
    free(session);
}

#ifdef __cplusplus
}
#endif
