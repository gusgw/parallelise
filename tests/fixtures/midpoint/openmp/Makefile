KERNEL  := midpoint
CC      ?= gcc
CFLAGS  ?= -O2 -fopenmp -Wall -Wextra -std=c11
LDLIBS  ?= -lm

all: test_$(KERNEL) bench

$(KERNEL).o: $(KERNEL).c $(KERNEL).h
	$(CC) $(CFLAGS) -c $(KERNEL).c -o $(KERNEL).o

test_$(KERNEL): test_$(KERNEL).c $(KERNEL).o $(KERNEL).h reference.h testing.h
	$(CC) $(CFLAGS) test_$(KERNEL).c $(KERNEL).o -o test_$(KERNEL) $(LDLIBS)

bench: bench.c $(KERNEL).o $(KERNEL).h
	$(CC) $(CFLAGS) bench.c $(KERNEL).o -o bench $(LDLIBS)

# The full gate: the unit test at one thread and at many, then the benchmark check.
check: test_$(KERNEL) bench
	OMP_NUM_THREADS=1 ./test_$(KERNEL)
	OMP_NUM_THREADS=$$(nproc) ./test_$(KERNEL)
	./bench 20000000 > bench.csv
	SPEEDUP_FLOOR=1.3 python3 check_bench.py bench.csv

clean:
	rm -f $(KERNEL).o test_$(KERNEL) bench bench.csv

.PHONY: all check clean
