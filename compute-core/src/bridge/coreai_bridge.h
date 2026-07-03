// Tribunus Core AI Bridge — consolidated public C API header.
//
// Thread-safety contract:
//   All functions are thread-safe unless noted otherwise.
//   AIModel objects are thread-safe for inference per Apple documentation.
//   InferenceFunction objects are NOT safe for concurrent prediction — callers must
//   serialize access to a single InferenceFunction.  Different InferenceFunctions may be
//   used concurrently.
//   Async completion handlers run on arbitrary dispatch queues — the handler
//   MUST NOT assume it is on any specific queue.
//
// Null-safety contract:
//   All pointer parameters are checked for null before dereference.
//   Functions return -1 or equivalent error code on null input.
//   Passing NULL to destroy/release functions is a no-op.

#pragma once
#include <stdint.h>
#include "coreai_arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Opaque handles ─────────────────────────────────────────────────────────

/// Opaque handle for a Core AI model (AIModel*).
/// Created via tribunus_coreai_load_model, released via tribunus_coreai_free_model.
typedef struct TribunusCoreAiModel TribunusCoreAiModel;

/// Opaque handle for a Core AI InferenceFunction object.
/// Created via tribunus_coreai_state_create, released via tribunus_coreai_state_destroy.
typedef struct TribunusCoreAiState TribunusCoreAiState;

/// Opaque handle for an in-flight stateful inference request.
/// Created via tribunus_coreai_predict_stateful_async, released via
/// tribunus_coreai_stateful_request_destroy (or the refcount-autoreleased callback).
/// Safe to call destroy from any thread once; the request internally uses a
/// refcount to prevent use-after-free between the caller and the completion handler.
typedef struct TribunusCoreAiStatefulRequest TribunusCoreAiStatefulRequest;

// ── Model lifecycle (stateless) ─────────────────────────────────────────────

/// Load a compiled Core AI model (.aimodel file).
/// Thread-safe: may be called from any thread.
/// @param out_model  [out] receives the opaque model handle (NULL-terminated on error).
/// @param model_path path to the .aimodel file.  Must be non-NULL.
/// @return 0 on success, negative on error (error details printed to stderr).
int tribunus_coreai_load_model(
    void** out_model,
    const char* model_path
);

/// Release a loaded model.  Safe to call with NULL.
void tribunus_coreai_free_model(void* model_ptr);

// ── State lifecycle ─────────────────────────────────────────────────────────

/// Create a new InferenceFunction from a loaded model.
/// Thread-safe: AIModel's function creation may be called from any thread.
/// @param out_state [out] receives the opaque state handle (NULL on error).
/// @param model_ptr opaque model handle from tribunus_coreai_load_model.
/// @return 0 on success, negative on error.
int tribunus_coreai_state_create(
    TribunusCoreAiState** out_state,
    void* model_ptr
);

/// Destroy an InferenceFunction object.  Safe to call with NULL.
void tribunus_coreai_state_destroy(TribunusCoreAiState* state);

// ── Stateless inference ──────────────────────────────────────────────────────

/// Run stateless inference: input NDArray -> model -> output NDArray.
/// Thread-safe for concurrent calls on the same model.
/// @return 0 on success, negative on error.
int tribunus_coreai_predict(
    void* model_ptr,
    const char* input_name,
    const TribunusArenaInfo* input_arena_info,
    const char* output_name,
    const TribunusArenaInfo* output_arena_info
);

/// Run stateless inference using IOSurface/CVPixelBuffer input.
/// Thread-safe for concurrent calls on the same model.
/// @return 0 on success, negative on error.
int tribunus_coreai_predict_pixelbuffer(
    void* model_ptr,
    const char* input_name,
    const TribunusArenaInfo* input_arena,
    const char* output_name,
    TribunusArenaInfo* output_arena
);

/// Run stateless inference with multiple named inputs and outputs.
/// All inputs are set up as a feature dictionary, all outputs as backings.
/// Thread-safe for concurrent calls on the same model.
/// @return 0 on success, negative on error.
int tribunus_coreai_predict_multi(
    void* model_ptr,
    const char** input_names,
    const TribunusArenaInfo** input_arenas,
    int num_inputs,
    const char** output_names,
    TribunusArenaInfo** output_arenas,
    int num_outputs
);

// ── Stateful inference (synchronous) ────────────────────────────────────────

/// Run stateful inference: input NDArray -> model + state -> output NDArray.
/// The InferenceFunction state is read and updated atomically by CoreAI.
/// NOT safe for concurrent calls on the same InferenceFunction — serialize access.
/// Thread-safe for concurrent calls on different InferenceFunction objects.
/// @return 0 on success, negative on error.
int tribunus_coreai_predict_stateful(
    void* model_ptr,
    TribunusCoreAiState* state,
    const char* input_name,
    void* input_arena_info,    // const TribunusArenaInfo*
    const char* output_name,
    void* output_arena_info    // TribunusArenaInfo* (output is written here)
);

// ── Stateful inference (asynchronous) ───────────────────────────────────────

/// Start an async stateful inference.  Returns immediately.
/// The completion handler runs on an arbitrary dispatch queue.
/// NOT safe for concurrent calls on the same InferenceFunction — serialize access.
///
/// @param out_request [out] receives an opaque request handle.
///        The caller owns one reference; call tribunus_coreai_stateful_request_destroy
///        to release it (or let the Rust Drop impl handle it).
/// @return 0 on success, negative on error (request not created).
int tribunus_coreai_predict_stateful_async(
    TribunusCoreAiStatefulRequest** out_request,
    void* model_ptr,
    TribunusCoreAiState* state,
    const char* input_name,
    void* input_arena_info,
    const char* output_name,
    void* output_arena_info
);

/// Check if an async request has completed.
/// Thread-safe (atomic read).
/// @return 1 if complete, 0 if still pending, -1 on NULL request.
int tribunus_coreai_stateful_request_is_complete(TribunusCoreAiStatefulRequest* request);

/// Set the Rust waker to wake when the async request completes.
/// Thread-safe.  May be called from any thread, including from the completion
/// handler itself.  If the request has already completed, the waker is called
/// immediately.
void tribunus_coreai_stateful_request_set_waker(
    TribunusCoreAiStatefulRequest* request,
    void* waker
);

/// Block until the async request completes.
/// NOT safe to call from the completion handler's dispatch queue (deadlock risk).
/// @return 0 on success, negative on inference failure, -1 on NULL request.
int tribunus_coreai_stateful_request_wait(TribunusCoreAiStatefulRequest* request);

/// Release one reference on an async request handle.
/// Internally uses a refcount — safe to call once from the creator thread even
/// if the completion handler is still in-flight.  The underlying memory is freed
/// only when both the creator and the completion handler release their references.
/// Thread-safe (atomic refcount).
void tribunus_coreai_stateful_request_destroy(TribunusCoreAiStatefulRequest* request);

// ── Shutdown ────────────────────────────────────────────────────────────────

/// Graceful shutdown: signal all in-flight async requests, prevent new ones,
/// and release global resources.
///
/// Call once at process shutdown.  After this function returns:
///   - No new async requests will be accepted (returns error).
///   - All pending requests have been signalled/completed.
///   - Callers waiting on tribunus_coreai_stateful_request_wait will unblock.
///
/// Thread-safe.  Idempotent — subsequent calls are no-ops.
void tribunus_coreai_shutdown(void);

#ifdef __cplusplus
}
#endif
