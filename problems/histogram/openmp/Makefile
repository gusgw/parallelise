ROUTINE := histogram
CC      ?= gcc
CFLAGS  ?= -O2 -fopenmp -Wall -Wextra -std=c11
LDLIBS  ?= -lm
BENCH_N ?= 200000000

all: test_$(ROUTINE) bench

$(ROUTINE).o: $(ROUTINE).c $(ROUTINE).h
	$(CC) $(CFLAGS) -c $(ROUTINE).c -o $(ROUTINE).o

test_$(ROUTINE): test_$(ROUTINE).c $(ROUTINE).o $(ROUTINE).h reference.h testing.h
	$(CC) $(CFLAGS) test_$(ROUTINE).c $(ROUTINE).o -o test_$(ROUTINE) $(LDLIBS)

bench: bench.c $(ROUTINE).o $(ROUTINE).h
	$(CC) $(CFLAGS) bench.c $(ROUTINE).o -o bench $(LDLIBS)

# the full check: build + unit test at 1 thread AND at many (the answer must not depend
# on the thread count; explicit counts so an inherited OMP_NUM_THREADS cannot skip the
# multithreaded check) + benchmark-stats validation. Binning streams a 1.6 GB array
# with a couple of operations per element, so it is memory-bandwidth-bound; the
# speed-up floor is lower than the compute-bound routines'.
check: test_$(ROUTINE) bench
	OMP_NUM_THREADS=1 ./test_$(ROUTINE)
	OMP_NUM_THREADS=$$(nproc) ./test_$(ROUTINE)
	./bench $(BENCH_N) > bench.csv
	SPEEDUP_FLOOR=1.5 python3 check_bench.py bench.csv

clean:
	rm -f $(ROUTINE).o test_$(ROUTINE) bench bench.csv

.PHONY: all check clean
