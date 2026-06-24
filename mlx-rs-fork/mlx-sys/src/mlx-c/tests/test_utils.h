#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EXPECT_EQ_STATUS(status, expected) \
    do { \
        if ((status) != (expected)) { \
            fprintf(stderr, "%s:%d: Expected status %d, got %d\n", __FILE__, __LINE__, (expected), (status)); \
            exit(1); \
        } \
    } while (0)

#define EXPECT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "%s:%d: Expected condition to be true\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

#define EXPECT_NEAR_FLOAT(val1, val2, tol) \
    do { \
        if (fabs((val1) - (val2)) > (tol)) { \
            fprintf(stderr, "%s:%d: Expected %f near %f within %f\n", __FILE__, __LINE__, (val1), (val2), (tol)); \
            exit(1); \
        } \
    } while (0)

#endif
