# Auto-detects host GPU compute capability (e.g. sm_89 on 4070, sm_120 on RTX PRO 6000).
NVCC ?= nvcc
CXX ?= c++
DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d .)
ARCH ?= $(if $(DETECTED_CC),sm_$(DETECTED_CC),sm_89)
NVCCFLAGS ?= -O3 -std=c++17 -use_fast_math
CXXFLAGS ?= -O3 -std=c++17 -pthread
PROGRAMS = packing seeded langevin selftest globaltest seeded_test certifier families_cli
.PHONY: all check clean
all: $(PROGRAMS)
packing: packing.cu families.hpp
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
seeded: seeded.cu families.hpp
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
langevin: langevin.cu families.hpp
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
selftest globaltest: %: tests/%.cu packing.cu families.hpp
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
seeded_test: tests/seeded_test.cu seeded.cu families.hpp
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
certifier: certifier.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
families_cli: families_cli.cpp families.hpp
	$(CXX) $(CXXFLAGS) $< -o $@
check: all
	./selftest
	./globaltest
	./seeded_test
clean:
	rm -f $(PROGRAMS)
