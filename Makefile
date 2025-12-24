# CUDA image convolution build

NVCC ?= nvcc
ARCH ?= sm_75
CXXFLAGS ?= -O3

SRC := src/main.cpp src/kernels.cu
BINARY := bin/convolution
INCLUDE_DIR := include
STB_URL := https://raw.githubusercontent.com/nothings/stb/master
STB_HEADERS := $(INCLUDE_DIR)/stb_image.h $(INCLUDE_DIR)/stb_image_write.h

.PHONY: all run clean fetch_stb

all: $(BINARY)

run: $(BINARY)
	$(BINARY)

$(BINARY): $(SRC) $(STB_HEADERS) | bin
	$(NVCC) $(SRC) -o $(BINARY) -std=c++17 -arch=$(ARCH) -I$(INCLUDE_DIR) $(CXXFLAGS)

bin:
	@mkdir -p bin

$(INCLUDE_DIR)/%.h:
	@mkdir -p $(INCLUDE_DIR)
	@echo "Fetching $(@F)"
	@wget -q -O $@ $(STB_URL)/$(@F)

fetch_stb: $(STB_HEADERS)

clean:
	rm -rf bin $(INCLUDE_DIR)/stb_image.h $(INCLUDE_DIR)/stb_image_write.h
