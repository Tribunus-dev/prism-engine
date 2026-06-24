#!/bin/bash
# Patch MLX Metal shader files for macOS 26+ compatibility.
set -e
MLX_SRC="$1"

echo "Patching Metal shaders at $MLX_SRC"

# 1. bf16.h: bfloat16_t = half when Metal doesn't provide bfloat
cp "/Users/user/Developer/GitHub/mlx-c-fork/patches/bf16_patched.h" \
   "$MLX_SRC/mlx/backend/metal/kernels/bf16.h"
echo "  patched bf16.h"

# 2. bf16_math.h: wrap the math overloads so they only apply when
#    bfloat16_t is NOT half (i.e., when Metal has native bfloat).
MATH_FILE="$MLX_SRC/mlx/backend/metal/kernels/bf16_math.h"
if ! grep -q 'BF16_MATH_GUARD' "$MATH_FILE" 2>/dev/null; then
    sed -i '' '1s/^/#if __has_extension(metal_bfloat)\n/' "$MATH_FILE"
    echo '#endif' >> "$MATH_FILE"
    echo "  patched bf16_math.h (conditional)"
fi

# 3. utils.h: make instantiate_float_limit(bfloat16_t) conditional
UTILS_FILE="$MLX_SRC/mlx/backend/metal/kernels/utils.h"
if ! grep -q 'using metal::vec' "$UTILS_FILE" 2>/dev/null; then
    # Add using metal::vec
    sed -i '' '/^#include.*logging\.h/ a\
using metal::vec;
' "$UTILS_FILE"
    echo "  patched utils.h (using metal::vec)"
fi
# Make instantiate_float_limit(bfloat16_t) conditional on bfloat support
if grep -q 'instantiate_float_limit(bfloat16_t);' "$UTILS_FILE" && \
   ! grep -q '^#if __has_extension.*bfloat' "$UTILS_FILE"; then
    sed -i '' 's/instantiate_float_limit(bfloat16_t);/#if __has_extension(metal_bfloat)\n    instantiate_float_limit(bfloat16_t);\n#endif/' "$UTILS_FILE"
    echo "  patched utils.h (bfloat16_t limit conditional)"
fi

# 4. struct Limits<complex64_t> needs the same treatment if bfloat isn't available
#    But complex64_t depends on bfloat16_t, so it should be fine.

echo "Done patching Metal shaders"
