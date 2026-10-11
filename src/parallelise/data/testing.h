/* testing.h — a minimal, dependency-free C unit-test harness.
 *
 * Just enough to assert and count, so the kernel under test is unit-tested without
 * pulling in a framework. Include it, write CHECK / CHECK_CLOSE assertions in a
 * main(), and return TEST_SUMMARY() (non-zero exit = some test failed).
 */
#ifndef TESTING_H
#define TESTING_H

#include <math.h>
#include <stdio.h>

static int tests_run = 0;
static int tests_failed = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        tests_run++;                                                          \
        if (!(cond)) {                                                        \
            tests_failed++;                                                   \
            printf("FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);          \
        }                                                                     \
    } while (0)

#define CHECK_CLOSE(got, want, tol, msg)                                      \
    do {                                                                      \
        tests_run++;                                                          \
        if (fabs((double)(got) - (double)(want)) > (double)(tol)) {           \
            tests_failed++;                                                   \
            printf("FAIL: %s: got %.12g want %.12g (tol %g) (%s:%d)\n",       \
                   (msg), (double)(got), (double)(want), (double)(tol),       \
                   __FILE__, __LINE__);                                       \
        }                                                                     \
    } while (0)

#define TEST_SUMMARY()                                                        \
    (printf("%d checks, %d failed\n", tests_run, tests_failed),               \
     tests_failed ? 1 : 0)

#endif /* TESTING_H */
