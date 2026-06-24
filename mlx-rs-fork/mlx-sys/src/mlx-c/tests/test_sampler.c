#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "sampler.h"

#define EXPECT_EQ(a, b) do { \
    if ((a) != (b)) { \
        fprintf(stderr, "%s:%d: Expected " #a " == " #b ", got %d vs %d\n", __FILE__, __LINE__, (int)(a), (int)(b)); \
        return 1; \
    } \
} while(0)

#define EXPECT_FLOAT_NEAR(a, b, tol) do { \
    if (fabs((a) - (b)) > (tol)) { \
        fprintf(stderr, "%s:%d: Expected " #a " ≈ " #b " within %f, got %f vs %f\n", __FILE__, __LINE__, (float)(tol), (float)(a), (float)(b)); \
        return 1; \
    } \
} while(0)

int test_argmax() {
    float logits[] = {1.0f, 5.0f, 2.0f, 0.5f};
    int res = sampler_argmax(logits, 4);
    EXPECT_EQ(res, 1);
    return 0;
}

int test_temperature() {
    float logits[] = {1.0f, 2.0f};
    sampler_temperature(logits, 2, 1.0f);
    
    // softmax of [1.0, 2.0]: exp(1)/ (exp(1)+exp(2)) vs exp(2) / (exp(1)+exp(2))
    // we use max subtraction internally, so:
    // exp(1-2) / (exp(1-2)+exp(0)) vs exp(0) / (exp(-1)+exp(0))
    // e^-1 / (1 + e^-1) = 0.3678 / 1.3678 = 0.2689
    // 1 / 1.3678 = 0.731
    EXPECT_FLOAT_NEAR(logits[0], 0.268941f, 1e-4);
    EXPECT_FLOAT_NEAR(logits[1], 0.731058f, 1e-4);

    float logits2[] = {1.0f, 2.0f};
    sampler_temperature(logits2, 2, 0.5f); // colder, peaks should be sharper
    // exp((1-2)/0.5) = exp(-2) = 0.1353
    // exp((2-2)/0.5) = exp(0) = 1.0
    // sum = 1.1353
    // prob 0 = 0.1353 / 1.1353 = 0.119
    // prob 1 = 1 / 1.1353 = 0.881
    EXPECT_FLOAT_NEAR(logits2[0], 0.119202f, 1e-4);
    EXPECT_FLOAT_NEAR(logits2[1], 0.880797f, 1e-4);
    
    return 0;
}

int test_top_k() {
    float probs[] = {0.1f, 0.6f, 0.2f, 0.05f, 0.05f};
    unsigned int seed = 42;
    // With k=1, should always pick the max
    int res = sampler_top_k(probs, 5, 1, &seed);
    EXPECT_EQ(res, 1);

    // With k=2, should pick between index 1 and 2
    int count_1 = 0, count_2 = 0;
    for (int i = 0; i < 1000; i++) {
        res = sampler_top_k(probs, 5, 2, &seed);
        if (res == 1) count_1++;
        if (res == 2) count_2++;
    }
    // Probs for top 2: 0.6 and 0.2
    // Renormalized: 0.75 and 0.25
    EXPECT_EQ(count_1 + count_2, 1000);
    // Should roughly be 75% 1s, 25% 2s, we can check basic bounds
    if (count_1 < 650 || count_1 > 850) {
        fprintf(stderr, "test_top_k distribution looks wrong: count_1=%d\n", count_1);
        return 1;
    }
    
    return 0;
}

int test_top_p() {
    float probs[] = {0.1f, 0.6f, 0.2f, 0.05f, 0.05f};
    unsigned int seed = 42;
    
    // Top P = 0.5: should only include index 1 (0.6 > 0.5)
    int res = sampler_top_p(probs, 5, 0.5f, &seed);
    EXPECT_EQ(res, 1);
    
    // Top P = 0.80: should include 0.6 and 0.2
    int count_1 = 0, count_2 = 0;
    for (int i = 0; i < 1000; i++) {
        res = sampler_top_p(probs, 5, 0.80f, &seed);
        if (res == 1) count_1++;
        if (res == 2) count_2++;
    }
    EXPECT_EQ(count_1 + count_2, 1000);
    if (count_1 < 650 || count_1 > 850) {
        fprintf(stderr, "test_top_p distribution looks wrong: count_1=%d\n", count_1);
        return 1;
    }

    return 0;
}

int test_min_p() {
    float probs[] = {0.1f, 0.6f, 0.2f, 0.05f, 0.05f};
    unsigned int seed = 42;

    // max is 0.6.
    // min_p = 0.5 -> threshold = 0.3. Only index 1 (0.6) passes.
    int res = sampler_min_p(probs, 5, 0.5f, &seed);
    EXPECT_EQ(res, 1);

    // min_p = 0.25 -> threshold = 0.15. Index 1 (0.6) and Index 2 (0.2) pass.
    int count_1 = 0, count_2 = 0;
    for (int i = 0; i < 1000; i++) {
        res = sampler_min_p(probs, 5, 0.25f, &seed);
        if (res == 1) count_1++;
        if (res == 2) count_2++;
    }
    EXPECT_EQ(count_1 + count_2, 1000);

    return 0;
}

int test_repetition_penalty() {
    float logits[] = {1.0f, 2.0f, -1.0f, 0.5f};
    int prev_tokens[] = {1, 2}; // penalize index 1 and 2
    
    // penalty 2.0
    // index 1: positive logit, 2.0 / 2.0 = 1.0
    // index 2: negative logit, -1.0 * 2.0 = -2.0
    sampler_repetition_penalty(logits, 4, 2.0f, prev_tokens, 2);
    
    EXPECT_FLOAT_NEAR(logits[0], 1.0f, 1e-5);
    EXPECT_FLOAT_NEAR(logits[1], 1.0f, 1e-5);
    EXPECT_FLOAT_NEAR(logits[2], -2.0f, 1e-5);
    EXPECT_FLOAT_NEAR(logits[3], 0.5f, 1e-5);
    
    return 0;
}

int test_sample_chain() {
    float logits[] = {1.0f, 5.0f, 2.0f, 0.5f};
    int prev_tokens[] = {1};
    
    struct sampler_params params = {
        .temp = 0.5f,
        .top_k = 0, // disabled
        .top_p = 0.9f,
        .min_p = 0.0f, // disabled
        .repetition_penalty = 10.0f, // heavily penalize token 1
        .prev_tokens = prev_tokens,
        .n_prev = 1,
        .seed = 12345
    };
    
    // Original max is token 1.
    // Penalty on token 1 reduces its logit to 5.0 / 10.0 = 0.5.
    // New logits: {1.0, 0.5, 2.0, 0.5}
    // Argmax is now token 2.
    // Temperature 0.5 will make 2.0 highly dominant.
    // Top P = 0.9 will likely just pick token 2.
    
    int res = sampler_sample(logits, 4, &params);
    EXPECT_EQ(res, 2);
    
    return 0;
}

int main() {
    int failed = 0;
    
    printf("Running test_argmax...\n");
    failed += test_argmax();
    
    printf("Running test_temperature...\n");
    failed += test_temperature();
    
    printf("Running test_top_k...\n");
    failed += test_top_k();
    
    printf("Running test_top_p...\n");
    failed += test_top_p();
    
    printf("Running test_min_p...\n");
    failed += test_min_p();
    
    printf("Running test_repetition_penalty...\n");
    failed += test_repetition_penalty();
    
    printf("Running test_sample_chain...\n");
    failed += test_sample_chain();
    
    if (failed) {
        printf("%d tests failed!\n", failed);
        return 1;
    }
    
    printf("All tests passed.\n");
    return 0;
}
