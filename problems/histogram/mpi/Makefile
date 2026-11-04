ROUTINE := histogram
MPICC   ?= mpicc
CFLAGS  ?= -O2 -Wall -Wextra -Werror -std=c11
LDLIBS  ?= -lm
NP      ?= 4
BENCH_N ?= 200000000

all: test_$(ROUTINE) bench

$(ROUTINE).o: $(ROUTINE).c $(ROUTINE).h
	$(MPICC) $(CFLAGS) -c $(ROUTINE).c -o $(ROUTINE).o

test_$(ROUTINE): test_$(ROUTINE).c $(ROUTINE).o $(ROUTINE).h reference.h testing.h
	$(MPICC) $(CFLAGS) test_$(ROUTINE).c $(ROUTINE).o -o test_$(ROUTINE) $(LDLIBS)

bench: bench.c $(ROUTINE).o $(ROUTINE).h
	$(MPICC) $(CFLAGS) bench.c $(ROUTINE).o -o bench $(LDLIBS)

# full check: build + unit test at 1 process AND at NP (the answer must not depend on
# the process count) + real-speedup benchmark validation. Binning streams the input
# array with a couple of operations per element, so it is memory-bandwidth-bound; the
# speed-up floor is lower than the compute-bound routines'.
check: test_$(ROUTINE) bench
	mpirun --oversubscribe -np 1 ./test_$(ROUTINE)
	mpirun --oversubscribe -np $(NP) ./test_$(ROUTINE)
	bash bench_sweep.sh $(BENCH_N) > bench.csv
	SPEEDUP_FLOOR=1.5 python3 check_bench.py bench.csv

clean:
	rm -f $(ROUTINE).o test_$(ROUTINE) bench bench.csv

.PHONY: all check clean
