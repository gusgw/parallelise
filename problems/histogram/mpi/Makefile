KERNEL  := histogram
MPICC   ?= mpicc
CFLAGS  ?= -O2 -Wall -Wextra -Werror -std=c11
LDLIBS  ?= -lm
NP      ?= 4
BENCH_N ?= 200000000

all: test_$(KERNEL) bench

$(KERNEL).o: $(KERNEL).c $(KERNEL).h
	$(MPICC) $(CFLAGS) -c $(KERNEL).c -o $(KERNEL).o

test_$(KERNEL): test_$(KERNEL).c $(KERNEL).o $(KERNEL).h reference.h testing.h
	$(MPICC) $(CFLAGS) test_$(KERNEL).c $(KERNEL).o -o test_$(KERNEL) $(LDLIBS)

bench: bench.c $(KERNEL).o $(KERNEL).h
	$(MPICC) $(CFLAGS) bench.c $(KERNEL).o -o bench $(LDLIBS)

# full check: build + unit test at 1 process AND at NP (the answer must not depend on
# the process count) + real-speedup benchmark validation. Binning streams the input
# array with a couple of operations per element, so it is memory-bandwidth-bound; the
# speed-up floor is lower than the compute-bound kernels'.
check: test_$(KERNEL) bench
	mpirun --oversubscribe -np 1 ./test_$(KERNEL)
	mpirun --oversubscribe -np $(NP) ./test_$(KERNEL)
	bash bench_sweep.sh $(BENCH_N) > bench.csv
	SPEEDUP_FLOOR=1.5 python3 check_bench.py bench.csv

clean:
	rm -f $(KERNEL).o test_$(KERNEL) bench bench.csv

.PHONY: all check clean
