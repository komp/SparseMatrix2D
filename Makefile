# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda-8.0

HOST_COMPILER ?= g++
CC	:= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m64
#  Required to use the c++ Chrono time package (with our release of g++)
CCFLAGS     := -std=gnu++11
LDFLAGS     :=

# required to read matlab-generated input file
# note:  also requires an old version (2016a)
#        for std-c library compatibility.
INCLUDES  := -I/tools/matlab2016a/extern/include/
LIBRARIES := -L/tools/matlab2016a/bin/glnxa64  -lmx -lmex -lmat -Wl,-rpath=/tools/matlab2016a/bin/glnxa64

# Gencode arguments
# SMS ?= 20 30 35 37 50 52 60
#  Note:  tesla requires 20 (which is now deprecated by NVIDIA)
SMS := 20
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

################################################################################

# Target rules
all: build

build: RunDecoder

Decoder.o: Decoder.cpp
	$(CC) $(INCLUDES) $(CCFLAGS) -o $@ -c $<

RunDecoder.o: RunDecoder.cpp
	$(CC) $(INCLUDES) $(CCFLAGS) -o $@ -c $<

RunDecoder: Decoder.o RunDecoder.o
	$(CC) $(ALL_LDFLAGS) $(CCFLAGS) -o $@ $+ $(LIBRARIES)

run: build
	./RunDecoder

clean:
	rm -f RunDecoder RunDecoder.o Decoder.o
