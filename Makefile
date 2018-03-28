# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?= g++
CC	:= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# Gencode arguments
# SMS ?= 30 35 37 50 52 60
#  Note:  tesla requires 20 (which is now deprecated by NVIDIA)
#  37  is best fit for K-80 chips
SMS = 37
SMS ?= 30 35 37 50 52 60
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# internal flags
CCFLAGS     := -std=c++11
LDFLAGS     :=

NVCC_FLAGS   := -std=c++11 -m64 $(GENCODE_FLAGS) --generate-line-info -O2 --use_fast_math -default-stream per-thread
NVCC_INCLUDES := -I$(CUDA_PATH)/samples/common/inc
NVCC_LIBRARIES :=

$(foreach lf,$(LDFLAGS),$(eval LDFLAGS4NVCC +=  -Xcompiler \"$(lf)\"))
NVCC_LDFLAGS := $(LDFLAGS4NVCC)

################################################################################

# Target rules
all: build

build: RunDecoder

SOURCES := $(wildcard *.cu)
### OBJECTS := $(patsubst %.cu, %.o, $(SOURCES))
OBJECTS := Tpkt.o loader_pool.o decoder_pool.o fast_loader.o bitEstimates.o   cnpOptimalBlock.o  Decoder.o \
	calcParityBits.o  Encoder.o  RunDecoder.o  ReadAlistFile.o

INCLUDEFILES := Tpkt.h bundleElt.h loader_pool.h decoder_pool.h fast_loader.h
## cnpMinSumBlock.o cnpOptimal.o cnpMinSum.o cnpOptimalNaive.o  copyBitsToCheckMatrix.o \

%.ptx: %.cu $(INCLUDEFILES)
	$(NVCC) $(NVCC_INCLUDES) $(INCLUDES) $(NVCC_FLAGS) -o $@ -ptx -src-in-ptx -c $<

%.o: %.cu $(INCLUDEFILES)
	$(NVCC) $(NVCC_INCLUDES) $(INCLUDES) $(NVCC_FLAGS) -o $@ -c $<

RunDecoder: $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS)  $(NVCC_LDFLAGS) -o $@ $+ $(LIBRARIES) $(NVCC_LIBRARIES)

run: build
	./RunDecoder

TestMaps.o: TestMaps.cu
	$(NVCC) $(NVCC_INCLUDES)  $(INCLUDES) $(NVCC_FLAGS) -o $@ -c $<
TestMaps: TestMaps.o
	$(NVCC) $(NVCC_FLAGS)  $(NVCC_LDFLAGS) -o $@ $+ $(LIBRARIES) $(NVCC_LIBRARIES)

TestEncoder.o: TestEncoder.cu
	$(NVCC) $(NVCC_INCLUDES)  $(INCLUDES) $(NVCC_FLAGS) -o $@ -c $<
TestEncoder: Encoder.o TestEncoder.o
	$(NVCC) $(NVCC_FLAGS)  $(NVCC_LDFLAGS) -o $@ $+ $(LIBRARIES) $(NVCC_LIBRARIES)

concurrentKernels.o: concurrentKernels.cu
	$(NVCC) $(NVCC_INCLUDES)  $(INCLUDES) $(NVCC_FLAGS) -o $@ -c $<
concurrentKernels: concurrentKernels.o
	$(NVCC) $(NVCC_FLAGS)  $(NVCC_LDFLAGS) -o $@ $+ $(LIBRARIES) $(NVCC_LIBRARIES)

clean:
	rm -f RunDecoder $(OBJECTS)
