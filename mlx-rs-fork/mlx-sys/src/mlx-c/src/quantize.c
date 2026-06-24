#include "../include/quantize.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;

    if (exp == 0) {
        if (frac == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            return (sign ? -1.0f : 1.0f) * ldexpf((float)frac, -24);
        }
    } else if (exp == 0x1f) {
        if (frac == 0) {
            return sign ? -INFINITY : INFINITY;
        } else {
            return NAN;
        }
    } else {
        uint32_t fp32_val = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
        float result;
        memcpy(&result, &fp32_val, sizeof(result));
        return result;
    }
}

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t fp32_val;
    memcpy(&fp32_val, &f, sizeof(fp32_val));

    uint32_t sign = (fp32_val >> 16) & 0x8000;
    int32_t exp = ((fp32_val >> 23) & 0xff) - 127;
    uint32_t frac = fp32_val & 0x7fffff;

    if (exp > 15) {
        return sign | 0x7c00;
    } else if (exp < -14) {
        if (exp < -24) {
            return sign;
        }
        uint32_t shifted_frac = (frac | 0x800000) >> (-exp - 14);
        if ((shifted_frac & 0x1fff) > 0x1000 || ((shifted_frac & 0x1fff) == 0x1000 && (shifted_frac & 0x2000))) {
            shifted_frac += 0x2000;
        }
        return sign | (shifted_frac >> 13);
    } else {
        uint32_t shifted_frac = frac;
        if ((shifted_frac & 0x1fff) > 0x1000 || ((shifted_frac & 0x1fff) == 0x1000 && (shifted_frac & 0x2000))) {
            shifted_frac += 0x2000;
            if (shifted_frac & 0x800000) {
                shifted_frac &= 0x7fffff;
                exp++;
                if (exp > 15) {
                    return sign | 0x7c00; 
                }
            }
        }
        return sign | ((exp + 15) << 10) | (shifted_frac >> 13);
    }
}

void quantize_row_q4_0(const float* src, void* dst, int64_t n) {
    block_q4_0* blocks = (block_q4_0*)dst;
    int num_blocks = n / 16;
    
    for (int i = 0; i < num_blocks; i++) {
        const float* block_src = src + i * 16;
        float max_val = 0.0f;
        
        for (int j = 0; j < 16; j++) {
            float abs_val = fabsf(block_src[j]);
            if (abs_val > max_val) {
                max_val = abs_val;
            }
        }
        
        float scale = max_val / -8.0f;
        blocks[i].scale = fp32_to_fp16(scale);
        
        float inv_scale = scale != 0.0f ? 1.0f / scale : 0.0f;
        
        for (int j = 0; j < 8; j++) {
            float v0 = block_src[j] * inv_scale;
            float v1 = block_src[j + 8] * inv_scale;
            
            int8_t i0 = (int8_t)roundf(v0);
            int8_t i1 = (int8_t)roundf(v1);
            
            if (i0 < -8) i0 = -8;
            if (i0 > 7) i0 = 7;
            if (i1 < -8) i1 = -8;
            if (i1 > 7) i1 = 7;
            
            blocks[i].qs[j] = ((i0 + 8) & 0x0f) | (((i1 + 8) & 0x0f) << 4);
        }
    }
}

void dequantize_row_q4_0(const void* src, float* dst, int64_t n) {
    const block_q4_0* blocks = (const block_q4_0*)src;
    int num_blocks = n / 16;
    
    for (int i = 0; i < num_blocks; i++) {
        float scale = fp16_to_fp32(blocks[i].scale);
        float* block_dst = dst + i * 16;
        
        for (int j = 0; j < 8; j++) {
            uint8_t q = blocks[i].qs[j];
            int8_t i0 = (q & 0x0f) - 8;
            int8_t i1 = ((q >> 4) & 0x0f) - 8;
            
            block_dst[j] = i0 * scale;
            block_dst[j + 8] = i1 * scale;
        }
    }
}

void quantize_row_q4_1(const float* src, void* dst, int64_t n) {
    block_q4_1* blocks = (block_q4_1*)dst;
    int num_blocks = n / 16;
    
    for (int i = 0; i < num_blocks; i++) {
        const float* block_src = src + i * 16;
        float min_val = block_src[0];
        float max_val = block_src[0];
        
        for (int j = 1; j < 16; j++) {
            if (block_src[j] < min_val) min_val = block_src[j];
            if (block_src[j] > max_val) max_val = block_src[j];
        }
        
        float scale = (max_val - min_val) / 15.0f;
        blocks[i].scale = fp32_to_fp16(scale);
        blocks[i].min = fp32_to_fp16(min_val);
        
        float inv_scale = scale != 0.0f ? 1.0f / scale : 0.0f;
        
        for (int j = 0; j < 8; j++) {
            float v0 = (block_src[j] - min_val) * inv_scale;
            float v1 = (block_src[j + 8] - min_val) * inv_scale;
            
            int8_t i0 = (int8_t)roundf(v0);
            int8_t i1 = (int8_t)roundf(v1);
            
            if (i0 < 0) i0 = 0;
            if (i0 > 15) i0 = 15;
            if (i1 < 0) i1 = 0;
            if (i1 > 15) i1 = 15;
            
            blocks[i].qs[j] = (i0 & 0x0f) | ((i1 & 0x0f) << 4);
        }
    }
}

void dequantize_row_q4_1(const void* src, float* dst, int64_t n) {
    const block_q4_1* blocks = (const block_q4_1*)src;
    int num_blocks = n / 16;
    
    for (int i = 0; i < num_blocks; i++) {
        float scale = fp16_to_fp32(blocks[i].scale);
        float min_val = fp16_to_fp32(blocks[i].min);
        float* block_dst = dst + i * 16;
        
        for (int j = 0; j < 8; j++) {
            uint8_t q = blocks[i].qs[j];
            int8_t i0 = q & 0x0f;
            int8_t i1 = (q >> 4) & 0x0f;
            
            block_dst[j] = i0 * scale + min_val;
            block_dst[j + 8] = i1 * scale + min_val;
        }
    }
}

void quantize_fp32_to_q4_0(int64_t n, const float* src, char* dst) {
    quantize_row_q4_0(src, dst, n);
}

// ---- Q8_0 (Session 1) ----

static inline uint16_t q8_fp32_to_fp16(float f) {
    union { float f; uint32_t u; } u = {f};
    uint32_t b = u.u + 0x00001000;
    uint32_t e = (b & 0x7F800000) >> 23;
    uint32_t m = b & 0x007FFFFF;
    uint16_t sign = (u.u >> 16) & 0x8000;
    if (e > 142) return sign | 0x7C00;
    if (e < 113) {
        uint32_t shift = 113 - e;
        if (shift > 24) return sign;
        m = (m | 0x00800000) >> shift;
        return sign | (m >> 13);
    }
    return sign | ((e - 112) << 10) | (m >> 13);
}

static inline float q8_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);
    union { float f; uint32_t u; } u;
    if (exp == 0) {
        if (mant == 0) {
            u.u = sign;
            return u.f;
        }
        while ((mant & 0x0400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
        u.u = sign | ((exp + 112) << 23) | (mant << 13);
        return u.f;
    } else if (exp == 0x1F) {
        u.u = sign | 0x7F800000 | (mant << 13);
        return u.f;
    }
    u.u = sign | ((exp + 112) << 23) | (mant << 13);
    return u.f;
}

void quantize_row_q8_0(const float* src, void* dst, int64_t n) {
    int num_blocks = n / QK8_0;
    block_q8_0* blocks = (block_q8_0*)dst;
    
    for (int i = 0; i < num_blocks; ++i) {
        float amax = 0.0f;
        
        for (int j = 0; j < QK8_0; ++j) {
            float v = src[i * QK8_0 + j];
            if (v < 0) v = -v;
            if (v > amax) amax = v;
        }
        
        float d = amax / 127.0f;
        float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        
        blocks[i].d = q8_fp32_to_fp16(d);
        
        for (int j = 0; j < QK8_0; ++j) {
            float v0 = src[i * QK8_0 + j] * id;
            float v = roundf(v0);
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            blocks[i].qs[j] = (int8_t)v;
        }
    }
}

void dequantize_row_q8_0(const void* src, float* dst, int64_t n) {
    int num_blocks = n / QK8_0;
    const block_q8_0* blocks = (const block_q8_0*)src;
    
    for (int i = 0; i < num_blocks; ++i) {
        float d = q8_fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < QK8_0; ++j) {
            dst[i * QK8_0 + j] = blocks[i].qs[j] * d;
        }
    }
}

// ---- K-quant (Session 2) ----

#define MAX_K(a, b) ((a) > (b) ? (a) : (b))
#define MIN_K(a, b) ((a) < (b) ? (a) : (b))

#ifndef GGML_RESTRICT
#define GGML_RESTRICT restrict
#endif

#define K_GROUP_MAX_EPS 1e-15f

static inline uint16_t GGML_FP32_TO_FP16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    if ((x & 0x7fffffff) == 0) return (x >> 16) & 0x8000;
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x007fffff;
    if (exp <= 0) return sign;
    else if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | (mant >> 13);
}

static inline float GGML_FP16_TO_FP32(uint16_t h) {
    if ((h & 0x7fff) == 0) {
        float f = 0.0f;
        uint32_t x; memcpy(&x, &f, sizeof(x));
        x |= (h & 0x8000) << 16;
        memcpy(&f, &x, sizeof(x));
        return f;
    }
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = ((h & 0x7c00) >> 10) - 15 + 127;
    uint32_t mant = (h & 0x03ff) << 13;
    uint32_t x = sign | (exp << 23) | mant;
    float f;
    memcpy(&f, &x, sizeof(x));
    return f;
}

static inline int nearest_int_k(float fval) {
    return (int)roundf(fval);
}

static float make_qx_quants(int n, int nmax, const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT L, int rmse_type,
        const float * GGML_RESTRICT qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < K_GROUP_MAX_EPS) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_k(iscale * x[i]);
            L[i] = nmax + MAX_K(-nmax, MIN_K(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_k(iscale * x[i]);
        l = MAX_K(-nmax, MIN_K(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_k(iscale * x[i]);
            l = MAX_K(-nmax, MIN_K(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int_k(iscale * x[i]);
                L[i] = nmax + MAX_K(-nmax, MIN_K(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

static float make_qkx1_quants(int n, int nmax, const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min,
        int ntry, float alpha) {
    float min = x[0];
    float max = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
    }
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = 0;
        return 0.f;
    }
    if (min > 0) min = 0;
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    for (int itry = 0; itry < ntry; ++itry) {
        float sumlx = 0; int suml2 = 0;
        bool did_change = false;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_k(iscale*(x[i] - min));
            l = MAX_K(0, MIN_K(nmax, l));
            if (l != L[i]) {
                L[i] = l;
                did_change = true;
            }
            sumlx += (x[i] - min)*l;
            suml2 += l*l;
        }
        scale = sumlx/suml2;
        float sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += x[i] - scale*L[i];
        }
        min = alpha*min + (1 - alpha)*sum/n;
        if (min > 0) min = 0;
        iscale = 1/scale;
        if (!did_change) break;
    }
    *the_min = -min;
    return scale;
}

static float make_qkx2_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_k(iscale*(x[i] - min));
        L[i] = MAX_K(0, MIN_K(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_k(iscale*(x[i] - min));
            l = MAX_K(0, MIN_K(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

void quantize_row_q4_K(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int k) {
    // (body elided - full implementation)
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    uint8_t L[QK_K];
    uint8_t Laux[32];
    float   weights[32];
    float mins[QK_K/32];
    float scales[QK_K/32];
    for (int i = 0; i < nb; i++) {
        float max_scale = 0;
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            float scale = scales[j];
            if (scale > max_scale) max_scale = scale;
            float min = mins[j];
            if (min > max_min) max_min = min;
        }
        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int_k(inv_scale*scales[j]);
            uint8_t lm = nearest_int_k(inv_min*mins[j]);
            ls = MIN_K(63, ls);
            lm = MIN_K(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);
        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int_k((x[32*j + ii] + dm)/d);
                l = MAX_K(0, MIN_K(15, l));
                L[32*j + ii] = l;
            }
        }
        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }
        x += QK_K;
    }
}

void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;
        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

void quantize_row_q5_K(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    uint8_t L[QK_K];
    float mins[QK_K/32];
    float scales[QK_K/32];
    float weights[32];
    uint8_t Laux[32];
    for (int i = 0; i < nb; i++) {
        float max_scale = 0;
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 31, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.5f, 0.1f, 15, false);
            float scale = scales[j];
            if (scale > max_scale) max_scale = scale;
            float min = mins[j];
            if (min > max_min) max_min = min;
        }
        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int_k(inv_scale*scales[j]);
            uint8_t lm = nearest_int_k(inv_min*mins[j]);
            ls = MIN_K(63, ls);
            lm = MIN_K(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);
        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int_k((x[32*j + ii] + dm)/d);
                l = MAX_K(0, MIN_K(31, l));
                L[32*j + ii] = l;
            }
        }
        uint8_t * GGML_RESTRICT qh = y[i].qh;
        uint8_t * GGML_RESTRICT ql = y[i].qs;
        memset(qh, 0, QK_K/8);
        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < QK_K; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j];
                if (l1 > 15) { l1 -= 16; qh[j] |= m1; }
                int l2 = L[n + j + 32];
                if (l2 > 15) { l2 -= 16; qh[j] |= m2; }
                ql[j] = l1 | (l2 << 4);
            }
            m1 <<= 2; m2 <<= 2;
            ql += 32;
        }
        x += QK_K;
    }
}

void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t * ql = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

void quantize_row_q6_K(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    int8_t L[QK_K];
    float   scales[QK_K/16];
    for (int i = 0; i < nb; i++) {
        float max_scale = 0;
        float max_abs_scale = 0;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
            scales[ib] = scale;
            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }
        if (max_abs_scale < K_GROUP_MAX_EPS) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = GGML_FP32_TO_FP16(0.f);
            x += QK_K;
            continue;
        }
        float iscale = -128.f/max_scale;
        y[i].d = GGML_FP32_TO_FP16(1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN_K(127, nearest_int_k(iscale*scales[ib]));
        }
        for (int j = 0; j < QK_K/16; ++j) {
            float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            if (!d) continue;
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int_k(x[16*j + ii]/d);
                l = MAX_K(-32, MIN_K(31, l));
                L[16*j + ii] = l + 32;
            }
        }
        uint8_t * GGML_RESTRICT ql = y[i].ql;
        uint8_t * GGML_RESTRICT qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }
        x += QK_K;
    }
}

void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * GGML_RESTRICT ql = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}
