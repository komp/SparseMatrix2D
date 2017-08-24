# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?= g++
CC	:= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# Gencode arguments
# SMS ?= 20 30 35 37 50 52 60
#  Note:  tesla requires 20 (which is now deprecated by NVIDIA)
#  37  is best fit for K-80 chips
SMS ?= 20 30 35 37 50 52 60
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# internal flags
CCFLAGS     :=
LDFLAGS     :=

NVCC_FLAGS   := -m64 -Wno-deprecated-gpu-targets $(GENCODE_FLAGS) --generate-line-info -O2
NVCC_INCLUDES := -I$(CUDA_PATH)/samples/common/inc
NVCC_LIBRARIES :=

$(foreach lf,$(LDFLAGS),$(eval LDFLAGS4NVCC +=  -Xcompiler \"$(lf)\"))
NVCC_LDFLAGS := $(LDFLAGS4NVCC)

################################################################################

# Target rules
all: build

build: RunDecoder

Decoder.o: Decoder.cu
	$(NVCC) $(NVCC_INCLUDES) $(INCLUDES) $(NVCC_FLAGS) -o $@ -c $<

RunDecoder.o: RunDecoder.cu
	$(NVCC) $(NVCC_INCLUDES)  $(INCLUDES) $(NVCC_FLAGS) -o $@ -c $<

RunDecoder: Decoder.o RunDecoder.o
	$(NVCC) $(NVCC_FLAGS)  $(NVCC_LDFLAGS) -o $@ $+ $(LIBRARIES) $(NVCC_LIBRARIES)

run: build
	./RunDecoder

TestMaps.o: TestMaps.cu
	$(NVCC) $(NVCC_INCLUDES)  $(INCLUDES) $(NVCC_FLAGS) -o $@ -c $<
TestMaps: TestMaps.o
	$(NVCC) $(NVCC_FLAGS)  $(NVCC_LDFLAGS) -o $@ $+ $(LIBRARIES) $(NVCC_LIBRARIES)

clean:
	rm -f RunDecoder RunDecoder.o Decoder.o
