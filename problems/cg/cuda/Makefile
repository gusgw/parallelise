ROUTINE := cg
NVCC    ?= /opt/cuda/bin/nvcc
NVFLAGS ?= -O2 -Xcompiler -Wall,-Wextra,-Werror
LDLIBS  ?= -lm
BENCH_N ?= 1024

all: test_$(ROUTINE) bench

$(ROUTINE).o: $(ROUTINE).cu $(ROUTINE).h
	$(NVCC) $(NVFLAGS) -c $(ROUTINE).cu -o $(ROUTINE).o

test_$(ROUTINE): test_$(ROUTINE).c $(ROUTINE).o $(ROUTINE).h reference.h testing.h
	$(NVCC) $(NVFLAGS) test_$(ROUTINE).c $(ROUTINE).o -o test_$(ROUTINE) $(LDLIBS)

bench: bench.c $(ROUTINE).o $(ROUTINE).h
	$(NVCC) $(NVFLAGS) bench.c $(ROUTINE).o -o bench $(LDLIBS)

# the full check: build + unit test (agreement with the notebook oracle) + serial-vs-GPU
# benchmark validation (the trusted serial baseline lives in bench.c, so a routine that
# does not really run on the GPU cannot show the required speedup). Floor 3x: this is a
# double-precision workload and consumer/laptop GPUs run FP64 at a small fraction of
# their FP32 rate (measured 4.45x on an RTX 2000 Ada); a CPU-only routine measures ~1x,
# so 3x still cleanly separates real GPU execution from fake.
check: test_$(ROUTINE) bench
	./test_$(ROUTINE)
	./bench $(BENCH_N) > bench.csv
	SPEEDUP_FLOOR=3 python3 check_bench_cuda.py bench.csv

clean:
	rm -f $(ROUTINE).o test_$(ROUTINE) bench bench.csv

.PHONY: all check clean
