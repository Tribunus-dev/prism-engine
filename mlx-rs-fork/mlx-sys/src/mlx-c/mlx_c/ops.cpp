#include "mlx_c/ops.h"
#include <cstdlib>
#include <cstdio>
#include <vector>

#ifdef MLX_C_ENABLE_MLX_BACKEND
#include "mlx/array.h"
#include "mlx/ops.h"
#endif

// Private definition to unwrap array handles
struct mlx_c_array {
#ifdef MLX_C_ENABLE_MLX_BACKEND
    mlx::core::array arr;
    mlx_c_array(mlx::core::array a) : arr(std::move(a)) {}
#else
    int dummy;
#endif
};

namespace {

inline mlx_c_status_t create_null_pointer_error(const char* func, const char* msg) {
    mlx_c_status_t status = {MLX_C_STATUS_NULL_POINTER, func, ""};
    snprintf(status.message, sizeof(status.message), "%s", msg);
    return status;
}

inline mlx_c_status_t create_shape_error(const char* func, const char* msg) {
    mlx_c_status_t status = {MLX_C_STATUS_SHAPE_ERROR, func, ""};
    snprintf(status.message, sizeof(status.message), "%s", msg);
    return status;
}

inline mlx_c_status_t create_dtype_error(const char* func, const char* msg) {
    mlx_c_status_t status = {MLX_C_STATUS_DTYPE_UNSUPPORTED, func, ""};
    snprintf(status.message, sizeof(status.message), "%s", msg);
    return status;
}

inline mlx_c_status_t create_backend_error(const char* func) {
    mlx_c_status_t status = {MLX_C_STATUS_BACKEND_UNAVAILABLE, func, "MLX backend is not enabled"};
    return status;
}

#ifdef MLX_C_ENABLE_MLX_BACKEND

inline mlx_c_status_t wrap_exception(const char* func, const std::exception& e) {
    mlx_c_status_t status = {MLX_C_STATUS_UPSTREAM_EXCEPTION, func, ""};
    snprintf(status.message, sizeof(status.message), "Exception: %s", e.what());
    return status;
}

inline mlx_c_status_t validate_f32_dtype(const char* func, const mlx_c_array_t* arr) {
    if (arr->arr.dtype() != mlx::core::float32) {
        return create_dtype_error(func, "only float32 is supported in v0");
    }
    return mlx_c_status_ok();
}

inline mlx_c_status_t validate_same_shape(const char* func, const mlx_c_array_t* a, const mlx_c_array_t* b) {
    if (a->arr.shape() != b->arr.shape()) {
        return create_shape_error(func, "inputs must have identical shapes for binary ops in v0");
    }
    return mlx_c_status_ok();
}

#endif // MLX_C_ENABLE_MLX_BACKEND

} // namespace

extern "C" mlx_c_status_t mlx_c_array_copy(const mlx_c_array_t* input, mlx_c_array_t** out) {
    if (out) *out = nullptr;
    if (!input || !out) return create_null_pointer_error("mlx_c_array_copy", "input or out cannot be null");

#ifndef MLX_C_ENABLE_MLX_BACKEND
    return create_backend_error("mlx_c_array_copy");
#else
    try {
        mlx_c_status_t stat = validate_f32_dtype("mlx_c_array_copy", input);
        if (!mlx_c_status_is_ok(stat)) return stat;

        // In MLX, array acts as a shared pointer to a computation graph node.
        // Copying the array object copies the handle, making it independently freeable in C.
        // If we want a physical copy, we could use identity(), but a simple copy of the wrapper is fine
        // since the C interface requires explicit freeing anyway.
        *out = new mlx_c_array(mlx::core::array(input->arr));
        return mlx_c_status_ok();
    } catch (const std::exception& e) {
        return wrap_exception("mlx_c_array_copy", e);
    } catch (...) {
        return {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_array_copy", "Unknown exception"};
    }
#endif
}

extern "C" mlx_c_status_t mlx_c_add(const mlx_c_array_t* lhs, const mlx_c_array_t* rhs, mlx_c_array_t** out) {
    if (out) *out = nullptr;
    if (!lhs || !rhs || !out) return create_null_pointer_error("mlx_c_add", "lhs, rhs, or out cannot be null");

#ifndef MLX_C_ENABLE_MLX_BACKEND
    return create_backend_error("mlx_c_add");
#else
    try {
        mlx_c_status_t stat;
        if (!mlx_c_status_is_ok(stat = validate_f32_dtype("mlx_c_add", lhs))) return stat;
        if (!mlx_c_status_is_ok(stat = validate_f32_dtype("mlx_c_add", rhs))) return stat;
        if (!mlx_c_status_is_ok(stat = validate_same_shape("mlx_c_add", lhs, rhs))) return stat;

        *out = new mlx_c_array(mlx::core::add(lhs->arr, rhs->arr));
        return mlx_c_status_ok();
    } catch (const std::exception& e) {
        return wrap_exception("mlx_c_add", e);
    } catch (...) {
        return {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_add", "Unknown exception"};
    }
#endif
}

extern "C" mlx_c_status_t mlx_c_multiply(const mlx_c_array_t* lhs, const mlx_c_array_t* rhs, mlx_c_array_t** out) {
    if (out) *out = nullptr;
    if (!lhs || !rhs || !out) return create_null_pointer_error("mlx_c_multiply", "lhs, rhs, or out cannot be null");

#ifndef MLX_C_ENABLE_MLX_BACKEND
    return create_backend_error("mlx_c_multiply");
#else
    try {
        mlx_c_status_t stat;
        if (!mlx_c_status_is_ok(stat = validate_f32_dtype("mlx_c_multiply", lhs))) return stat;
        if (!mlx_c_status_is_ok(stat = validate_f32_dtype("mlx_c_multiply", rhs))) return stat;
        if (!mlx_c_status_is_ok(stat = validate_same_shape("mlx_c_multiply", lhs, rhs))) return stat;

        *out = new mlx_c_array(mlx::core::multiply(lhs->arr, rhs->arr));
        return mlx_c_status_ok();
    } catch (const std::exception& e) {
        return wrap_exception("mlx_c_multiply", e);
    } catch (...) {
        return {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_multiply", "Unknown exception"};
    }
#endif
}

extern "C" mlx_c_status_t mlx_c_sigmoid(const mlx_c_array_t* input, mlx_c_array_t** out) {
    if (out) *out = nullptr;
    if (!input || !out) return create_null_pointer_error("mlx_c_sigmoid", "input or out cannot be null");

#ifndef MLX_C_ENABLE_MLX_BACKEND
    return create_backend_error("mlx_c_sigmoid");
#else
    try {
        mlx_c_status_t stat;
        if (!mlx_c_status_is_ok(stat = validate_f32_dtype("mlx_c_sigmoid", input))) return stat;

        *out = new mlx_c_array(mlx::core::sigmoid(input->arr));
        return mlx_c_status_ok();
    } catch (const std::exception& e) {
        return wrap_exception("mlx_c_sigmoid", e);
    } catch (...) {
        return {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_sigmoid", "Unknown exception"};
    }
#endif
}

extern "C" mlx_c_status_t mlx_c_silu(const mlx_c_array_t* input, mlx_c_array_t** out) {
    if (out) *out = nullptr;
    if (!input || !out) return create_null_pointer_error("mlx_c_silu", "input or out cannot be null");

#ifndef MLX_C_ENABLE_MLX_BACKEND
    return create_backend_error("mlx_c_silu");
#else
    try {
        mlx_c_status_t stat;
        if (!mlx_c_status_is_ok(stat = validate_f32_dtype("mlx_c_silu", input))) return stat;

        // Composite silu: x * sigmoid(x)
        auto sig = mlx::core::sigmoid(input->arr);
        *out = new mlx_c_array(mlx::core::multiply(input->arr, sig));
        return mlx_c_status_ok();
    } catch (const std::exception& e) {
        return wrap_exception("mlx_c_silu", e);
    } catch (...) {
        return {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_silu", "Unknown exception"};
    }
#endif
}

extern "C" mlx_c_status_t mlx_c_matmul(const mlx_c_array_t* lhs, const mlx_c_array_t* rhs, mlx_c_array_t** out) {
    if (out) *out = nullptr;
    if (!lhs || !rhs || !out) return create_null_pointer_error("mlx_c_matmul", "lhs, rhs, or out cannot be null");

#ifndef MLX_C_ENABLE_MLX_BACKEND
    return create_backend_error("mlx_c_matmul");
#else
    try {
        mlx_c_status_t stat;
        if (!mlx_c_status_is_ok(stat = validate_f32_dtype("mlx_c_matmul", lhs))) return stat;
        if (!mlx_c_status_is_ok(stat = validate_f32_dtype("mlx_c_matmul", rhs))) return stat;

        if (lhs->arr.ndim() != 2 || rhs->arr.ndim() != 2) {
            return create_shape_error("mlx_c_matmul", "only rank-2 matmul is supported in v0");
        }

        if (lhs->arr.shape()[1] != rhs->arr.shape()[0]) {
            return create_shape_error("mlx_c_matmul", "incompatible inner dimensions for matmul");
        }

        *out = new mlx_c_array(mlx::core::matmul(lhs->arr, rhs->arr));
        return mlx_c_status_ok();
    } catch (const std::exception& e) {
        return wrap_exception("mlx_c_matmul", e);
    } catch (...) {
        return {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_matmul", "Unknown exception"};
    }
#endif
}

extern "C" mlx_c_status_t mlx_c_reshape(const mlx_c_array_t* input, const int64_t* new_shape, size_t new_ndim, mlx_c_array_t** out) {
    if (out) *out = nullptr;
    if (!input || !new_shape || !out) return create_null_pointer_error("mlx_c_reshape", "input, new_shape, or out cannot be null");

    if (new_ndim == 0) {
        return create_shape_error("mlx_c_reshape", "zero dimensions are not supported in v0");
    }

    size_t total_elements = 1;
    for (size_t i = 0; i < new_ndim; ++i) {
        if (new_shape[i] <= 0) {
            return create_shape_error("mlx_c_reshape", "dimensions must be strictly positive");
        }
        if (new_shape[i] > 2147483647LL) {
            return create_shape_error("mlx_c_reshape", "dimension exceeds INT_MAX");
        }

        size_t prev = total_elements;
        total_elements *= new_shape[i];
        if (total_elements / new_shape[i] != prev) {
            return create_shape_error("mlx_c_reshape", "shape multiplication overflow");
        }
    }

#ifndef MLX_C_ENABLE_MLX_BACKEND
    return create_backend_error("mlx_c_reshape");
#else
    try {
        if (total_elements != input->arr.size()) {
            return create_shape_error("mlx_c_reshape", "new shape element count does not match input array size");
        }

        std::vector<int> std_shape;
        for (size_t i = 0; i < new_ndim; ++i) {
            std_shape.push_back(static_cast<int>(new_shape[i]));
        }
        mlx::core::Shape mlx_shape(std_shape.begin(), std_shape.end());

        *out = new mlx_c_array(mlx::core::reshape(input->arr, mlx_shape));
        return mlx_c_status_ok();
    } catch (const std::exception& e) {
        return wrap_exception("mlx_c_reshape", e);
    } catch (...) {
        return {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_reshape", "Unknown exception"};
    }
#endif
}

extern "C" mlx_c_status_t mlx_c_transpose(const mlx_c_array_t* input, const int64_t* axes, size_t axes_len, mlx_c_array_t** out) {
    if (out) *out = nullptr;
    if (!input || !axes || !out) return create_null_pointer_error("mlx_c_transpose", "input, axes, or out cannot be null");

#ifndef MLX_C_ENABLE_MLX_BACKEND
    return create_backend_error("mlx_c_transpose");
#else
    try {
        size_t ndim = input->arr.ndim();
        if (axes_len != ndim) {
            return create_shape_error("mlx_c_transpose", "axes_len must match input ndim");
        }

        std::vector<int> std_axes;
        std::vector<bool> seen(ndim, false);

        for (size_t i = 0; i < axes_len; ++i) {
            if (axes[i] < 0 || axes[i] >= (int64_t)ndim) {
                return create_shape_error("mlx_c_transpose", "axis out of range");
            }
            if (seen[axes[i]]) {
                return create_shape_error("mlx_c_transpose", "duplicate axis in transpose");
            }
            seen[axes[i]] = true;
            std_axes.push_back(static_cast<int>(axes[i]));
        }

        *out = new mlx_c_array(mlx::core::transpose(input->arr, std_axes));
        return mlx_c_status_ok();
    } catch (const std::exception& e) {
        return wrap_exception("mlx_c_transpose", e);
    } catch (...) {
        return {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_transpose", "Unknown exception"};
    }
#endif
}

extern "C" mlx_c_status_t mlx_c_softmax(const mlx_c_array_t* input, int64_t axis, mlx_c_array_t** out) {
    if (out) *out = nullptr;
    if (!input || !out) return create_null_pointer_error("mlx_c_softmax", "input or out cannot be null");

#ifndef MLX_C_ENABLE_MLX_BACKEND
    return create_backend_error("mlx_c_softmax");
#else
    try {
        mlx_c_status_t stat;
        if (!mlx_c_status_is_ok(stat = validate_f32_dtype("mlx_c_softmax", input))) return stat;

        size_t ndim = input->arr.ndim();
        if (axis < 0 || axis >= (int64_t)ndim) {
            return create_shape_error("mlx_c_softmax", "axis out of range");
        }

        std::vector<int> axes = {static_cast<int>(axis)};
        *out = new mlx_c_array(mlx::core::softmax(input->arr, axes));
        return mlx_c_status_ok();
    } catch (const std::exception& e) {
        return wrap_exception("mlx_c_softmax", e);
    } catch (...) {
        return {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_softmax", "Unknown exception"};
    }
#endif
}
