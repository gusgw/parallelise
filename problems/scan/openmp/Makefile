KERNEL  := scan
CC      ?= gcc
CFLAGS  ?= -O2 -fopenmp -Wall -Wextra -std=c11
LDLIBS  ?= -lm
BENCH_N ?= 134217728

all: test_$(KERNEL) bench

$(KERNEL).o: $(KERNEL).c $(KERNEL).h
	$(CC) $(CFLAGS) -c $(KERNEL).c -o $(KERNEL).o

test_$(KERNEL): test_$(KERNEL).c $(KERNEL).o $(KERNEL).h reference.h testing.h
	$(CC) $(CFLAGS) test_$(KERNEL).c $(KERNEL).o -o test_$(KERNEL) $(LDLIBS)

bench: bench.c $(KERNEL).o $(KERNEL).h
	$(CC) $(CFLAGS) bench.c $(KERNEL).o -o bench $(LDLIBS)

# the full gate: build + unit test at 1 thread AND at many (the answer must not depend
# on the thread count; explicit counts so an inherited OMP_NUM_THREADS cannot skip the
# multithreaded check) + benchmark-stats validation. The scan is memory-bandwidth-
# bound (and the block scan reads the input twice), so its speed-up floor is lower
# than the compute-bound kernels'.
check: test_$(KERNEL) bench
	OMP_NUM_THREADS=1 ./test_$(KERNEL)
	OMP_NUM_THREADS=$$(nproc) ./test_$(KERNEL)
	./bench $(BENCH_N) > bench.csv
	SPEEDUP_FLOOR=1.25 python3 check_bench.py bench.csv

clean:
	rm -f $(KERNEL).o test_$(KERNEL) bench bench.csv

.PHONY: all check clean
