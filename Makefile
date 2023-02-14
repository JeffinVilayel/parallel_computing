# Location of the CUDA Toolkit
NVCC := /usr/local/cuda/bin/nvcc
V1_CCFLAGS := -O2 -DTYPE=1
V2_CCFLAGS := -O2 -DTYPE=2

build: quamsimV1 quamsimV2

quamsimV1.o:quamsim.cu
	$(NVCC) $(INCLUDES) $(V1_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

quamsimV1: quamsimV1.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

quamsimV2.o:quamsim.cu
	$(NVCC) $(INCLUDES) $(V2_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

quamsimV2: quamsimV2.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./quamsim

clean:
	rm -f quamsimV1 quamsimV2 *.o
