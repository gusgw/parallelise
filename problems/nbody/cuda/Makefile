KERNEL  := nbody
NVCC    ?= /opt/cuda/bin/nvcc
NVFLAGS ?= -O2 -Xcompiler -Wall,-Wextra,-Werror
LDLIBS  ?= -lm
BENCH_N ?= 32768

all: test_$(KERNEL) bench

$(KERNEL).o: $(KERNEL).cu $(KERNEL).h
	$(NVCC) $(NVFLAGS) -c $(KERNEL).cu -o $(KERNEL).o

test_$(KERNEL): test_$(KERNEL).c $(KERNEL).o $(KERNEL).h reference.h testing.h
	$(NVCC) $(NVFLAGS) test_$(KERNEL).c $(KERNEL).o -o test_$(KERNEL) $(LDLIBS)

bench: bench.c $(KERNEL).o $(KERNEL).h
	$(NVCC) $(NVFLAGS) bench.c $(KERNEL).o -o bench $(LDLIBS)

# the full gate: build + unit test (agreement with the notebook oracle) + serial-vs-GPU
# benchmark validation (the trusted serial baseline lives in bench.c, so a kernel that
# does not really run on the GPU cannot show the required speedup). Floor 3x: this is a
# double-precision workload and consumer/laptop GPUs run FP64 at a small fraction of
# their FP32 rate (measured 4.45x on an RTX 2000 Ada); a CPU-only kernel measures ~1x,
# so 3x still cleanly separates real GPU execution from fake.
check: test_$(KERNEL) bench
	./test_$(KERNEL)
	./bench $(BENCH_N) > bench.csv
	SPEEDUP_FLOOR=3 python3 check_bench_cuda.py bench.csv

clean:
	rm -f $(KERNEL).o test_$(KERNEL) bench bench.csv

.PHONY: all check clean
