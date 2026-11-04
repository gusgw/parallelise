ROUTINE := midpoint
CC      ?= gcc
CFLAGS  ?= -O2 -fopenmp -Wall -Wextra -std=c11
LDLIBS  ?= -lm

all: test_$(ROUTINE) bench

$(ROUTINE).o: $(ROUTINE).c $(ROUTINE).h
	$(CC) $(CFLAGS) -c $(ROUTINE).c -o $(ROUTINE).o

test_$(ROUTINE): test_$(ROUTINE).c $(ROUTINE).o $(ROUTINE).h reference.h testing.h
	$(CC) $(CFLAGS) test_$(ROUTINE).c $(ROUTINE).o -o test_$(ROUTINE) $(LDLIBS)

bench: bench.c $(ROUTINE).o $(ROUTINE).h
	$(CC) $(CFLAGS) bench.c $(ROUTINE).o -o bench $(LDLIBS)

# The full check: the unit test at one thread and at many, then the benchmark check.
check: test_$(ROUTINE) bench
	OMP_NUM_THREADS=1 ./test_$(ROUTINE)
	OMP_NUM_THREADS=$$(nproc) ./test_$(ROUTINE)
	./bench 20000000 > bench.csv
	SPEEDUP_FLOOR=1.3 python3 check_bench.py bench.csv

clean:
	rm -f $(ROUTINE).o test_$(ROUTINE) bench bench.csv

.PHONY: all check clean
