SHELL := /bin/sh
CC = mpicxx
CU = nvcc
CFLAGS = -O3 -ffast-math -g0
CUFLAGS = -O3 --use_fast_math --extra-device-vectorization -lineinfo
LDLIBS = -lpthread
OBJ_DIR = build
CU_OPT = \
	-gencode=arch=compute_52,code=sm_52 \
	-gencode=arch=compute_62,code=sm_62 \
	-gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_80,code=compute_80
PROGRAM = gpucmt
VERSION = v1


.PHONY: all build clean $(PROGRAM)_$(VERSION)

all: clean $(PROGRAM)_$(VERSION)

$(OBJ_DIR)/host.o: main.cpp data/dataset.cpp click_models/evaluation.cpp parallel_em/communicator.cpp utils/macros.cpp
	@echo "[1/4] Compiling host code..."
	@if [ ! -d "$(OBJ_DIR)" ]; then mkdir $(OBJ_DIR); fi
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDLIBS) -std=c++11 -c $^
	ld -r *.o -o $@
	@rm -f *.o

$(OBJ_DIR)/device.a: utils/utils.cu parallel_em/parallel_em.cu parallel_em/kernel.cu data/search.cu click_models/base.cu click_models/param.cu click_models/common.cu
	@echo "[2/4] Compiling device code..."
	@if [ ! -d "$(OBJ_DIR)" ]; then mkdir $(OBJ_DIR); fi
	$(CU) $(CPPFLAGS) $(CUFLAGS) $(CU_OPT) -std=c++11 -rdc=true -c $^ click_models/*.cu
	@rm -f $@
	ar cr $@ *.o
	@ranlib $@

$(OBJ_DIR)/device_link.o: $(OBJ_DIR)/device.a
	@echo "[3/4] Linking device code..."
	@if [ ! -d "$(OBJ_DIR)" ]; then mkdir $(OBJ_DIR); fi
	$(CU) $(CUFLAGS) $(CU_OPT) -std=c++11 --device-link *.o -o $@ -lcudart

$(PROGRAM)_$(VERSION): $(OBJ_DIR)/host.o $(OBJ_DIR)/device_link.o $(OBJ_DIR)/device.a
	@echo "[4/4] Building executable..."
	$(CC) $(CFLAGS) $(LDLIBS) -std=c++11 $^ -o $@ -lcudart
	@rm -f *.o

clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR) $(PROGRAM)_$(VERSION) mpi_hosts *.o
