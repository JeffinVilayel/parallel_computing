NVCC := /usr/local/cuda/bin/nvcc
#V1_CCFLAGS := -O2 -DOPT=1
#V2_CCFLAGS := -O2 -DOPT=2
build: quamsimV1 quamsimV2 quamsimV3
quamsimV1.o:quamsim1.cu
	$(NVCC) $(INCLUDES) $(V1_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
quamsimV1: quamsimV1.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
quamsimV2.o:quamsim2.cu
	$(NVCC) $(INCLUDES) $(V2_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
quamsimV2: quamsimV2.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
quamsimV3.o:quamsim3.cu
	$(NVCC) $(INCLUDES) $(V2_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
quamsimV3: quamsimV3.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
run: build
	$(EXEC) ./quamsim
clean:
	rm -f quamsimV1 quamsimV2 quamsimV3 *.o
