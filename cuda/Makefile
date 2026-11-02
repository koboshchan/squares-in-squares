# Run make -C cuda ARCH=sm_120 with CUDA 12.8+ on Blackwell.
NVCC ?= nvcc
ARCH ?= sm_89
NVCCFLAGS ?= -O3 -std=c++17
PROGRAMS = packing seeded selftest globaltest seeded_test
.PHONY: all check clean
all: $(PROGRAMS)
packing: packing.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
seeded: seeded.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
selftest globaltest: %: tests/%.cu packing.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
seeded_test: tests/seeded_test.cu seeded.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
check: all
	./selftest
	./globaltest
	./seeded_test
clean:
	rm -f $(PROGRAMS)
