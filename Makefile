SHELL := /bin/sh
CC_MPI = mpicxx
CC_CU = nvcc
L_FLAG = -L/cm/shared/apps/cuda91/toolkit/9.1.85/lib64
OBJ_DIR = build
EXEC = $(PRUN_ETC)/prun-openmpi
SCHEDULE = prun
NODES = 2
DEVICES = 2
PROG = gpucmt
VERSION = v0
CU_OPT = \
	-gencode=arch=compute_52,code=sm_52 \
	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
	-gencode=arch=compute_62,code=sm_62 \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_75,code=sm_75 \
	-gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_80,code=compute_80

.PHONY: all build test run clean cake nompi nompi-valgrind nompi-debug-cuda

all: clean build

build: $(PROG)_$(VERSION)

# $(CC_MPI) -O3 $(CU_OPT) -ffast-math -funroll-loops -std=c++11 -c *.cpp utils/*.cpp data/*.cpp click_models/*.cpp
$(OBJ_DIR)/host.o: main.cpp utils/utility_functions.cpp data/dataset.cpp click_models/evaluation.cpp parallel_em/communicator.cpp
	@echo "Compiling host code..."
	@if [ ! -d "$(OBJ_DIR)" ]; then mkdir $(OBJ_DIR); fi
	$(CC_MPI) -pthread -O3 -ffast-math -funroll-loops -std=c++11 -c *.cpp utils/*.cpp parallel_em/*.cpp data/*.cpp click_models/*.cpp
	ld -r *.o -o $(OBJ_DIR)/host.o $(DEF)
	@rm -f *.o

# $(CC_CU) -O3 $(CU_OPT) -std=c++11 -rdc=true -lineinfo -c utils/*.cu parallel_em/*.cu data/*.cu click_models/*.cu
$(OBJ_DIR)/device.a: utils/cuda_utils.cu parallel_em/parallel_em.cu data/search.cu click_models/base.cu click_models/pbm.cu click_models/param.cu
	@echo "Compiling device code..."
	@if [ ! -d "$(OBJ_DIR)" ]; then mkdir $(OBJ_DIR); fi
	$(CC_CU) -O3 $(CU_OPT) -std=c++11 -rdc=true -lineinfo -c utils/*.cu parallel_em/*.cu data/*.cu click_models/*.cu
	@rm -f $(OBJ_DIR)/device.a
	ar cr $(OBJ_DIR)/device.a *.o
	@ranlib $(OBJ_DIR)/device.a

# $(CC_CU) -O3 $(CU_OPT) -std=c++11 --device-link *.o -o $(OBJ_DIR)/device_link.o -lcudart
$(OBJ_DIR)/device_link.o: $(OBJ_DIR)/device.a
	@echo "Linking device code..."
	@if [ ! -d "$(OBJ_DIR)" ]; then mkdir $(OBJ_DIR); fi
	$(CC_CU) -O3 $(CU_OPT) -std=c++11 --device-link *.o -o $(OBJ_DIR)/device_link.o -lcudart

# $(CC_MPI) -O3 -ffast-math -funroll-loops -std=c++11 $(OBJ_DIR)/host.o $(OBJ_DIR)/device_link.o $(OBJ_DIR)/device.a -o $(PROG)_$(VERSION) -lcudart
$(PROG)_$(VERSION): $(OBJ_DIR)/host.o $(OBJ_DIR)/device_link.o
	@echo "Building executable..."
	$(CC_MPI) -pthread -O3 -ffast-math -funroll-loops -std=c++11 $(OBJ_DIR)/host.o $(OBJ_DIR)/device_link.o $(OBJ_DIR)/device.a -o $(PROG)_$(VERSION) -lcudart
	@rm -f *.o

cake:
	@printf '  )  (  )  (\n (^)(^)(^)(^)\n _i__i__i__i_\n(____________)\n|############|\n(____________)\n'

nompi: clean build
	@echo "\nRunning without prun or MPI..."
	./$(PROG)_$(VERSION)

cuda: clean build
	@echo "\nTesting without MPI..."
	cuda-memcheck ./$(PROG)_$(VERSION) | more

valgrind: clean build
	@echo "\nTesting without MPI using valgrind..."
	valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --verbose \
         --log-file=valgrind-out.txt \
	./$(PROG)_$(VERSION)

run:
	@echo "Scheduling job..."
	$(SCHEDULE) -v -np $(NODES) -1 -native '--gres=gpu:$(DEVICES)' -script $(EXEC) ./$(PROG)_$(VERSION)

clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR) $(PROG)_$(VERSION) mpi_hosts *.o
