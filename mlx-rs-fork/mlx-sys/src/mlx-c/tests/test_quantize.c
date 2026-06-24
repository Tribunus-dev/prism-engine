#include "test_utils.h"
#include "../include/quantize.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void test_q4_0() {
    int64_t n = 32;
    float* src = (float*)malloc(n * sizeof(float));
    float* dst = (float*)malloc(n * sizeof(float));
    char* q = (char*)malloc(n / 16 * sizeof(block_q4_0));

    // Initialize with some data. Range is roughly -0.7 to +0.7
    for (int i = 0; i < n; i++) {
        src[i] = (float)sin(i) * 0.7f;
    }

    quantize_row_q4_0(src, q, n);
    dequantize_row_q4_0(q, dst, n);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabs(src[i] - dst[i]);
        if (err > max_err) {
            max_err = err;
        }
    }

    printf("Q4_0 max error: %f\n", max_err);
    EXPECT_TRUE(max_err < 0.1f);

    // Verify block boundaries
    EXPECT_TRUE(sizeof(block_q4_0) == 10); // 2 bytes scale + 8 bytes data
    free(src);
    free(dst);
    free(q);
}

void test_q4_1() {
    int64_t n = 32;
    float* src = (float*)malloc(n * sizeof(float));
    float* dst = (float*)malloc(n * sizeof(float));
    char* q = (char*)malloc(n / 16 * sizeof(block_q4_1));

    // Initialize with some data. Range is roughly -0.7 to +0.7
    for (int i = 0; i < n; i++) {
        src[i] = (float)cos(i) * 0.7f;
    }

    quantize_row_q4_1(src, q, n);
    dequantize_row_q4_1(q, dst, n);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabs(src[i] - dst[i]);
        if (err > max_err) {
            max_err = err;
        }
    }

    printf("Q4_1 max error: %f\n", max_err);
    EXPECT_TRUE(max_err < 0.1f); 

    // Verify block boundaries
    EXPECT_TRUE(sizeof(block_q4_1) == 12); // 2 bytes scale + 2 bytes min + 8 bytes data
    free(src);
    free(dst);
    free(q);
}

int main() {
    test_q4_0();
    test_q4_1();
    printf("All quantize tests passed!\n");
    return 0;
}
