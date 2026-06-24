#include "mlx_c/array.h"
#include <cstdlib>
#include <cstdio>
#include <vector>

#ifdef MLX_C_ENABLE_MLX_BACKEND
#include "mlx/array.h"
#include "mlx/ops.h"
#endif

struct mlx_c_array {
#ifdef MLX_C_ENABLE_MLX_BACKEND
    mlx::core::array arr;
    mlx_c_array(mlx::core::array a) : arr(std::move(a)) {}
#else
    int dummy;
#endif
};

extern "C" mlx_c_status_t mlx_c_array_create_from_f32(
    mlx_c_context_t* ctx,
    const float* data,
    const int64_t* shape,
    size_t ndim,
    mlx_c_array_t** out_array) {
    if (!out_array) {
        return {MLX_C_STATUS_NULL_POINTER, "mlx_c_array_create_from_f32", "out_array cannot be null"};
    }
    *out_array = nullptr;

    if (!ctx) {
        return {MLX_C_STATUS_NULL_POINTER, "mlx_c_array_create_from_f32", "ctx cannot be null"};
    }

    if (!data) {
        return {MLX_C_STATUS_NULL_POINTER, "mlx_c_array_create_from_f32", "data cannot be null"};
    }

    if (ndim == 0) {
        return {MLX_C_STATUS_INVALID_ARGUMENT, "mlx_c_array_create_from_f32", "ndim must be > 0 in v0"};
    }

    if (!shape) {
        return {MLX_C_STATUS_NULL_POINTER, "mlx_c_array_create_from_f32", "shape cannot be null"};
    }

    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] <= 0) {
            return {MLX_C_STATUS_SHAPE_ERROR, "mlx_c_array_create_from_f32", "dimensions must be strictly positive"};
        }

        // check that int64_t doesn't exceed INT_MAX before we cast it into std::vector<int>
        if (shape[i] > 2147483647LL) { // INT_MAX
            return {MLX_C_STATUS_SHAPE_ERROR, "mlx_c_array_create_from_f32", "dimension exceeds INT_MAX"};
        }

        // overflow check
        size_t prev = total_elements;
        total_elements *= shape[i];
        if (total_elements / shape[i] != prev) {
            return {MLX_C_STATUS_SHAPE_ERROR, "mlx_c_array_create_from_f32", "shape multiplication overflow"};
        }
    }

#ifndef MLX_C_ENABLE_MLX_BACKEND
    return {MLX_C_STATUS_BACKEND_UNAVAILABLE, "mlx_c_array_create_from_f32", "MLX backend is not enabled"};
#else
    try {
        std::vector<int> std_shape;
        for (size_t i = 0; i < ndim; ++i) {
            std_shape.push_back(static_cast<int>(shape[i]));
        }

        // mlx::core::array constructor copies data when given an iterator (like raw pointer) and shape
        mlx::core::Shape mlx_shape(std_shape.begin(), std_shape.end());
        mlx_c_array_t* new_array = new mlx_c_array(mlx::core::array(data, mlx_shape, mlx::core::float32));
        *out_array = new_array;
        return mlx_c_status_ok();
    } catch (const std::exception& e) {
        mlx_c_status_t status = {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_array_create_from_f32", ""};
        snprintf(status.message, sizeof(status.message), "Exception: %s", e.what());
        return status;
    } catch (...) {
        return {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_array_create_from_f32", "Unknown exception"};
    }
#endif
}

extern "C" void mlx_c_array_free(mlx_c_array_t* array) {
    if (array) {
        delete array;
    }
}

extern "C" mlx_c_status_t mlx_c_array_dtype(const mlx_c_array_t* array, mlx_c_dtype_t* out_dtype) {
    if (!array || !out_dtype) {
        return {MLX_C_STATUS_NULL_POINTER, "mlx_c_array_dtype", "array or out_dtype cannot be null"};
    }

#ifndef MLX_C_ENABLE_MLX_BACKEND
    return {MLX_C_STATUS_BACKEND_UNAVAILABLE, "mlx_c_array_dtype", "MLX backend is not enabled"};
#else
    try {
        if (array->arr.dtype() == mlx::core::float32) {
            *out_dtype = MLX_C_DTYPE_FLOAT32;
        } else {
            *out_dtype = MLX_C_DTYPE_UNKNOWN;
        }
        return mlx_c_status_ok();
    } catch (...) {
        return {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_array_dtype", "Unknown exception"};
    }
#endif
}

extern "C" mlx_c_status_t mlx_c_array_ndim(const mlx_c_array_t* array, size_t* out_ndim) {
    if (!array || !out_ndim) {
        return {MLX_C_STATUS_NULL_POINTER, "mlx_c_array_ndim", "array or out_ndim cannot be null"};
    }

#ifndef MLX_C_ENABLE_MLX_BACKEND
    return {MLX_C_STATUS_BACKEND_UNAVAILABLE, "mlx_c_array_ndim", "MLX backend is not enabled"};
#else
    try {
        *out_ndim = array->arr.ndim();
        return mlx_c_status_ok();
    } catch (...) {
        return {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_array_ndim", "Unknown exception"};
    }
#endif
}

extern "C" mlx_c_status_t mlx_c_array_shape(const mlx_c_array_t* array, int64_t* dims_out, size_t dims_capacity, size_t* ndim_out) {
    if (!array || !ndim_out) {
        return {MLX_C_STATUS_NULL_POINTER, "mlx_c_array_shape", "array or ndim_out cannot be null"};
    }

    if (dims_capacity > 0 && !dims_out) {
        return {MLX_C_STATUS_INVALID_ARGUMENT, "mlx_c_array_shape", "dims_out cannot be null if capacity > 0"};
    }

#ifndef MLX_C_ENABLE_MLX_BACKEND
    return {MLX_C_STATUS_BACKEND_UNAVAILABLE, "mlx_c_array_shape", "MLX backend is not enabled"};
#else
    try {
        const auto& shape = array->arr.shape();
        *ndim_out = shape.size();

        if (dims_out && dims_capacity > 0) {
            if (dims_capacity < shape.size()) {
                return {MLX_C_STATUS_INVALID_ARGUMENT, "mlx_c_array_shape", "dims_capacity is too small"};
            }
            size_t copy_count = shape.size();
            for (size_t i = 0; i < copy_count; ++i) {
                dims_out[i] = static_cast<int64_t>(shape[i]);
            }
        }

        return mlx_c_status_ok();
    } catch (...) {
        return {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_array_shape", "Unknown exception"};
    }
#endif
}

extern "C" mlx_c_status_t mlx_c_array_size(const mlx_c_array_t* array, size_t* out_size) {
    if (!array || !out_size) {
        return {MLX_C_STATUS_NULL_POINTER, "mlx_c_array_size", "array or out_size cannot be null"};
    }

#ifndef MLX_C_ENABLE_MLX_BACKEND
    return {MLX_C_STATUS_BACKEND_UNAVAILABLE, "mlx_c_array_size", "MLX backend is not enabled"};
#else
    try {
        *out_size = array->arr.size();
        return mlx_c_status_ok();
    } catch (...) {
        return {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_array_size", "Unknown exception"};
    }
#endif
}

extern "C" mlx_c_status_t mlx_c_array_copy_to_f32(const mlx_c_array_t* array, float* data_out, size_t capacity) {
    if (!array || !data_out) {
        return {MLX_C_STATUS_NULL_POINTER, "mlx_c_array_copy_to_f32", "array or data_out cannot be null"};
    }

#ifndef MLX_C_ENABLE_MLX_BACKEND
    return {MLX_C_STATUS_BACKEND_UNAVAILABLE, "mlx_c_array_copy_to_f32", "MLX backend is not enabled"};
#else
    try {
        if (array->arr.dtype() != mlx::core::float32) {
            return {MLX_C_STATUS_DTYPE_UNSUPPORTED, "mlx_c_array_copy_to_f32", "array is not float32"};
        }

        size_t arr_size = array->arr.size();
        if (capacity < arr_size) {
            return {MLX_C_STATUS_SHAPE_ERROR, "mlx_c_array_copy_to_f32", "capacity is too small"};
        }

        // This evaluates the array before fetching its data.
        const_cast<mlx::core::array&>(array->arr).eval();

        // If the array has strides/is a view, we need a contiguous array to copy it properly
        mlx::core::array contig_arr = mlx::core::contiguous(array->arr);
        contig_arr.eval();

        const float* arr_data = contig_arr.data<float>();
        for (size_t i = 0; i < arr_size; ++i) {
            data_out[i] = arr_data[i];
        }

        return mlx_c_status_ok();
    } catch (...) {
        return {MLX_C_STATUS_UPSTREAM_EXCEPTION, "mlx_c_array_copy_to_f32", "Unknown exception"};
    }
#endif
}
