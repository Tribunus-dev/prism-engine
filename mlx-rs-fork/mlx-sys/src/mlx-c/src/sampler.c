#include "sampler.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>

/* Helper structure for sorting logits */
struct token_prob {
    int index;
    float prob;
};

static int compare_token_prob(const void* a, const void* b) {
    const struct token_prob* ta = (const struct token_prob*)a;
    const struct token_prob* tb = (const struct token_prob*)b;
    if (ta->prob < tb->prob) return 1;
    if (ta->prob > tb->prob) return -1;
    return 0;
}

static float random_uniform(unsigned int* seed) {
    /* Simple thread-safe uniform [0, 1) */
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return (float)*seed / (float)0x7fffffff;
}

int sampler_argmax(const float* logits, int n) {
    if (n <= 0) return -1;
    
    int max_idx = 0;
    float max_val = logits[0];
    
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

void sampler_temperature(float* logits, int n, float temp) {
    if (n <= 0) return;
    
    if (temp <= 0.0f) {
        return; // Temperature handled elsewhere (argmax)
    }

    float max_logit = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < n; i++) {
        logits[i] = expf((logits[i] - max_logit) / temp);
        sum_exp += logits[i];
    }

    for (int i = 0; i < n; i++) {
        logits[i] /= sum_exp;
    }
}

void sampler_repetition_penalty(float* logits, int n, float penalty, const int* prev_tokens, int n_prev) {
    if (penalty <= 1.0f || !prev_tokens || n_prev <= 0 || n <= 0) return;

    // We only want to apply the penalty once per unique token in prev_tokens.
    // Instead of allocating a full size 'n' array (which could be very large for vocabulary),
    // we iterate and keep track of tokens we already penalized if n_prev is small,
    // but since we modify logits in place, a simple way is to use a bitset or byte array.
    // Since n can be large (e.g. 32000 or 128000), a dynamically allocated byte array is safe.
    
    unsigned char* penalized = (unsigned char*)calloc(n, sizeof(unsigned char));
    if (!penalized) return;

    for (int i = 0; i < n_prev; i++) {
        int token = prev_tokens[i];
        if (token >= 0 && token < n && !penalized[token]) {
            if (logits[token] > 0) {
                logits[token] /= penalty;
            } else {
                logits[token] *= penalty;
            }
            penalized[token] = 1;
        }
    }
    
    free(penalized);
}

static int sample_distribution(const struct token_prob* probs, int count, unsigned int* seed) {
    float r = random_uniform(seed);
    float cdf = 0.0f;
    
    for (int i = 0; i < count; i++) {
        cdf += probs[i].prob;
        if (r < cdf) {
            return probs[i].index;
        }
    }
    
    // Fallback to last element if rounding errors occur
    return probs[count - 1].index;
}

int sampler_top_k(const float* probs, int n, int k, unsigned int* seed) {
    if (n <= 0) return -1;
    if (k <= 0 || k >= n) {
        // Just sample from the full distribution
        float r = random_uniform(seed);
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probs[i];
            if (r < cdf) return i;
        }
        return n - 1;
    }

    struct token_prob* tp = (struct token_prob*)malloc(n * sizeof(struct token_prob));
    for (int i = 0; i < n; i++) {
        tp[i].index = i;
        tp[i].prob = probs[i];
    }

    // Sort descending
    qsort(tp, n, sizeof(struct token_prob), compare_token_prob);

    // Renormalize top k
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sum += tp[i].prob;
    }

    for (int i = 0; i < k; i++) {
        tp[i].prob /= sum;
    }

    int result = sample_distribution(tp, k, seed);
    free(tp);
    return result;
}

int sampler_top_p(const float* probs, int n, float p, unsigned int* seed) {
    if (n <= 0) return -1;
    if (p <= 0.0f || p >= 1.0f) {
        float r = random_uniform(seed);
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probs[i];
            if (r < cdf) return i;
        }
        return n - 1;
    }

    struct token_prob* tp = (struct token_prob*)malloc(n * sizeof(struct token_prob));
    for (int i = 0; i < n; i++) {
        tp[i].index = i;
        tp[i].prob = probs[i];
    }

    // Sort descending
    qsort(tp, n, sizeof(struct token_prob), compare_token_prob);

    float cum_prob = 0.0f;
    int k = 0;
    for (int i = 0; i < n; i++) {
        cum_prob += tp[i].prob;
        k++;
        if (cum_prob >= p) {
            break;
        }
    }

    // Renormalize
    for (int i = 0; i < k; i++) {
        tp[i].prob /= cum_prob;
    }

    int result = sample_distribution(tp, k, seed);
    free(tp);
    return result;
}

int sampler_min_p(const float* probs, int n, float min_p, unsigned int* seed) {
    if (n <= 0) return -1;
    if (min_p <= 0.0f || min_p >= 1.0f) {
        float r = random_uniform(seed);
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probs[i];
            if (r < cdf) return i;
        }
        return n - 1;
    }

    struct token_prob* tp = (struct token_prob*)malloc(n * sizeof(struct token_prob));
    float max_prob = probs[0];
    for (int i = 0; i < n; i++) {
        tp[i].index = i;
        tp[i].prob = probs[i];
        if (probs[i] > max_prob) {
            max_prob = probs[i];
        }
    }

    float threshold = max_prob * min_p;
    
    // Sort descending
    qsort(tp, n, sizeof(struct token_prob), compare_token_prob);

    int k = 0;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        if (tp[i].prob < threshold && k > 0) {
            break;
        }
        sum += tp[i].prob;
        k++;
    }

    // Renormalize
    for (int i = 0; i < k; i++) {
        tp[i].prob /= sum;
    }

    int result = sample_distribution(tp, k, seed);
    free(tp);
    return result;
}

int sampler_sample(float* logits, int n, struct sampler_params* params) {
    if (n <= 0) return -1;

    // 1. Repetition penalty
    if (params->repetition_penalty > 1.0f && params->prev_tokens && params->n_prev > 0) {
        sampler_repetition_penalty(logits, n, params->repetition_penalty, params->prev_tokens, params->n_prev);
    }

    // 2. Temperature and convert to probabilities
    if (params->temp <= 0.0f) {
        return sampler_argmax(logits, n);
    }
    
    sampler_temperature(logits, n, params->temp);

    // 3. Filtering (Top-K, Top-P, Min-P combined)
    // To properly chain them, we need to track probabilities and apply masks.
    struct token_prob* tp = (struct token_prob*)malloc(n * sizeof(struct token_prob));
    float max_prob = logits[0];
    for (int i = 0; i < n; i++) {
        tp[i].index = i;
        tp[i].prob = logits[i];
        if (logits[i] > max_prob) {
            max_prob = logits[i];
        }
    }

    // Sort descending
    qsort(tp, n, sizeof(struct token_prob), compare_token_prob);

    int active_k = n;

    // Apply top_k
    if (params->top_k > 0 && params->top_k < active_k) {
        active_k = params->top_k;
    }

    // Apply top_p
    if (params->top_p > 0.0f && params->top_p < 1.0f) {
        float cum_prob = 0.0f;
        int p_k = 0;
        for (int i = 0; i < active_k; i++) {
            cum_prob += tp[i].prob;
            p_k++;
            if (cum_prob >= params->top_p) {
                break;
            }
        }
        if (p_k < active_k) {
            active_k = p_k;
        }
    }

    // Apply min_p
    if (params->min_p > 0.0f && params->min_p < 1.0f) {
        float threshold = max_prob * params->min_p;
        int min_p_k = 0;
        for (int i = 0; i < active_k; i++) {
            if (tp[i].prob < threshold && min_p_k > 0) {
                break;
            }
            min_p_k++;
        }
        if (min_p_k < active_k) {
            active_k = min_p_k;
        }
    }

    // Renormalize what's left in active_k
    float sum = 0.0f;
    for (int i = 0; i < active_k; i++) {
        sum += tp[i].prob;
    }

    for (int i = 0; i < active_k; i++) {
        tp[i].prob /= sum;
    }

    int result = sample_distribution(tp, active_k, &params->seed);
    free(tp);
    return result;
}
