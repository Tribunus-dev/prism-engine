#ifndef SAMPLER_H
#define SAMPLER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

/*
 * sampler_params
 * Parameters used for the sampler_sample function chain.
 * Note: Not all parameters need to be active. Setting top_k <= 0 disables it.
 * Setting top_p <= 0.0 or >= 1.0 disables it.
 * Setting min_p <= 0.0 disables it.
 * Setting temp <= 0.0 does argmax instead.
 * Setting repetition_penalty <= 1.0 disables it.
 */
struct sampler_params {
    float temp;
    int top_k;
    float top_p;
    float min_p;
    float repetition_penalty;
    const int* prev_tokens;
    int n_prev;
    unsigned int seed; /* For rng */
};

/*
 * Returns the token index with the highest probability/logit.
 */
int sampler_argmax(const float* logits, int n);

/*
 * Filters to the `k` highest probabilities, renormalizes, and samples.
 * If k <= 0 or k >= n, this does not filter the distribution.
 * Returns the sampled token index.
 */
int sampler_top_k(const float* probs, int n, int k, unsigned int* seed);

/*
 * Nucleus sampling. Filters to the smallest set of tokens whose cumulative
 * probability exceeds `p`.
 * Renormalizes and samples.
 * If p <= 0.0 or p >= 1.0, this does not filter the distribution.
 * Returns the sampled token index.
 */
int sampler_top_p(const float* probs, int n, float p, unsigned int* seed);

/*
 * Applies softmax with temperature to the logits.
 * Overwrites the original logits array with the probabilities.
 */
void sampler_temperature(float* logits, int n, float temp);

/*
 * Applies repetition penalty to the logits based on previously seen tokens.
 * Penalty > 1.0 reduces the probability of already generated tokens.
 * Modifies the logits in-place.
 */
void sampler_repetition_penalty(float* logits, int n, float penalty, const int* prev_tokens, int n_prev);

/*
 * Filters out tokens with probabilities less than `min_p` times the probability
 * of the most likely token.
 * Renormalizes and samples.
 * Returns the sampled token index.
 */
int sampler_min_p(const float* probs, int n, float min_p, unsigned int* seed);

/*
 * Runs the configured sampling chain.
 * The typical chain is:
 * 1. Repetition penalty
 * 2. Temperature (if temp <= 0, just return argmax)
 * 3. Top-K, Top-P, or Min-P
 * Returns the sampled token index.
 * Note: this function takes a non-const logits array because it modifies it
 * during the sampling process (e.g. applying temperature, penalties).
 */
int sampler_sample(float* logits, int n, struct sampler_params* params);

#ifdef __cplusplus
}
#endif

#endif // SAMPLER_H
