#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// block_q4_0: 16 x int4 weights (packed) + fp16 scale = 18 bytes per 16 weights
typedef struct {
    uint16_t scale; // fp16
    uint8_t qs[8];  // 16 int4 weights packed into 8 bytes
} block_q4_0;

// block_q4_1: 16 x int4 weights (packed) + fp16 scale + fp16 min = 20 bytes per 16 weights
typedef struct {
    uint16_t scale; // fp16
    uint16_t min;   // fp16
    uint8_t qs[8];  // 16 int4 weights packed into 8 bytes
} block_q4_1;

void quantize_row_q4_0(const float* src, void* dst, int64_t n);
void dequantize_row_q4_0(const void* src, float* dst, int64_t n);

void quantize_row_q4_1(const float* src, void* dst, int64_t n);
void dequantize_row_q4_1(const void* src, float* dst, int64_t n);

void quantize_fp32_to_q4_0(int64_t n, const float* src, char* dst);

// Q8_0 block format: 32 int8 weights + 1 fp16 scale
#define QK8_0 32
#pragma pack(push, 1)
typedef struct {
    uint16_t d;       // fp16 scale
    int8_t qs[QK8_0]; // quants
} block_q8_0;
#pragma pack(pop)

void quantize_row_q8_0(const float* src, void* dst, int64_t n);
void dequantize_row_q8_0(const void* src, float* dst, int64_t n);

// K-quant types
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    uint16_t d;
} block_q6_K;

void quantize_row_q4_K(const float * x, block_q4_K * y, int k);
void quantize_row_q5_K(const float * x, block_q5_K * y, int k);
void quantize_row_q6_K(const float * x, block_q6_K * y, int k);

void dequantize_row_q4_K(const block_q4_K * x, float * y, int k);
void dequantize_row_q5_K(const block_q5_K * x, float * y, int k);
void dequantize_row_q6_K(const block_q6_K * x, float * y, int k);


#ifdef __cplusplus
}
#endif

#endif // QUANTIZE_H
