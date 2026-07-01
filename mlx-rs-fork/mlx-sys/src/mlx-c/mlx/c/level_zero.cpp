/* Copyright © 2025 Apple Inc. */
/* Level Zero C API */

#include "mlx/c/level_zero.h"
#include "mlx/backend/level_zero/device.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" int mlx_level_zero_is_available(bool* res) {
  try {
    *res = mlx::core::level_zero::is_available();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_level_zero_set_capture_dir(const char* path) {
  try {
    mlx::core::level_zero::Device::set_capture_dir(std::string(path));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
