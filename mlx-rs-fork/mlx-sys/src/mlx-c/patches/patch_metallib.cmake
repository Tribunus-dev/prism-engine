# CMake patch script for MLX Metal shader compatibility on macOS 26+.
# Called from FetchContent_Declare PATCH_COMMAND with MLX_SRC defined.
#
# This script modifies:
#   1. bf16_math.h — wraps math overloads in __has_extension(metal_bfloat) guard
#   2. utils.h — adds using metal::vec; and makes bfloat16_t Limits conditional

# --- Patch bf16_math.h ---
set(MATH_FILE "${MLX_SRC}/mlx/backend/metal/kernels/bf16_math.h")
file(READ "${MATH_FILE}" MATH_CONTENT)

# Check if already patched
string(FIND "${MATH_CONTENT}" "BF16_MATH_GUARD" GUARD_POS)
if(GUARD_POS EQUAL -1)
  set(MATH_GUARD "#if __has_extension(metal_bfloat)\n")
  file(WRITE "${MATH_FILE}" "${MATH_GUARD}${MATH_CONTENT}\n#endif")
  message(STATUS "patched bf16_math.h (conditional)")
endif()

# --- Patch utils.h ---
set(UTILS_FILE "${MLX_SRC}/mlx/backend/metal/kernels/utils.h")
file(READ "${UTILS_FILE}" UTILS_CONTENT)

# Add using metal::vec; after logging.h include (if not already present)
string(FIND "${UTILS_CONTENT}" "using metal::vec" VEC_POS)
if(VEC_POS EQUAL -1)
  string(REPLACE "#include \"mlx/backend/metal/kernels/logging.h\"\n"
                 "#include \"mlx/backend/metal/kernels/logging.h\"\nusing metal::vec;\n"
                 UTILS_CONTENT "${UTILS_CONTENT}")
  file(WRITE "${UTILS_FILE}" "${UTILS_CONTENT}")
  message(STATUS "patched utils.h (using metal::vec)")
endif()

# Make instantiate_float_limit(bfloat16_t) conditional on metal_bfloat
string(FIND "${UTILS_CONTENT}" "instantiate_float_limit(bfloat16_t)" BFLT_POS)
string(FIND "${UTILS_CONTENT}" "BHAS_EXT" BHAS_POS)
if(NOT BFLT_POS EQUAL -1 AND BHAS_POS EQUAL -1)
  string(REPLACE "instantiate_float_limit(bfloat16_t);"
                 "#if __has_extension(metal_bfloat)\n    instantiate_float_limit(bfloat16_t);\n#endif"
                 UTILS_CONTENT "${UTILS_CONTENT}")
  file(WRITE "${UTILS_FILE}" "${UTILS_CONTENT}")
  message(STATUS "patched utils.h (bfloat16_t limit conditional)")
endif()

message(STATUS "MLX Metal shader patches complete")
