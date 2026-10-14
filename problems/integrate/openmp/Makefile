CC      ?= gcc
CFLAGS  ?= -O2 -fopenmp -Wall -Wextra -std=c11
LDLIBS  ?= -lm

all: test_integrate bench

integrate.o: integrate.c integrate.h
	$(CC) $(CFLAGS) -c integrate.c -o integrate.o

test_integrate: test_integrate.c integrate.o integrate.h reference.h testing.h
	$(CC) $(CFLAGS) test_integrate.c integrate.o -o test_integrate $(LDLIBS)

bench: bench.c integrate.o integrate.h
	$(CC) $(CFLAGS) bench.c integrate.o -o bench $(LDLIBS)

test: test_integrate
	./test_integrate

bench-run: bench
	./bench

# the full check: build (via deps) + unit test at 1 thread AND at many (the answer must
# not depend on the thread count; the many-threads count is explicit so an inherited
# OMP_NUM_THREADS=1 in the environment cannot quietly skip the multithreaded check)
check: test_integrate bench
	OMP_NUM_THREADS=1 ./test_integrate
	OMP_NUM_THREADS=$$(nproc) ./test_integrate
	./bench 50000000 > bench.csv
	python3 check_bench.py bench.csv

clean:
	rm -f integrate.o test_integrate bench bench.csv

.PHONY: all test bench-run check clean
