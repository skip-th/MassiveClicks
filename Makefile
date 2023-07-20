CC      = mpicxx
CU      = nvcc
CFLAGS  = -O3 -ffast-math -g0 -std=c++11
CUFLAGS = -O3 --use_fast_math -std=c++11
LDLIBS  = -pthread -lcudart -L/usr/local/cuda-12.1/lib64
OBJ_DIR = build
CU_OPT  = \
         -gencode=arch=compute_52,code=sm_52 \
         -gencode=arch=compute_62,code=compute_62 \
         -gencode=arch=compute_80,code=sm_80 \
         -gencode=arch=compute_80,code=compute_80
PROGRAM = mclicks
VERSION = 1

SRCS    = main.cpp data/dataset.cpp click_models/evaluation.cpp parallel_em/communicator.cpp utils/macros.cpp utils/timer.cpp
CU_SRCS = utils/utils.cu parallel_em/parallel_em.cu parallel_em/kernel.cu data/search.cu \
	      click_models/base.cu click_models/param.cu click_models/common.cu click_models/factor.cu \
          click_models/pbm.cu click_models/ccm.cu click_models/dbn.cu click_models/ubm.cu

OBJ     = $(addprefix $(OBJ_DIR)/,$(SRCS:.cpp=.o))
CU_OBJ  = $(addprefix $(OBJ_DIR)/,$(CU_SRCS:.cu=.o))

.PHONY: all build clean $(PROGRAM)_$(VERSION)

all: $(PROGRAM)_$(VERSION)

$(OBJ_DIR)/%.o: %.cpp
	@echo -n " $<"
	@mkdir -p $(dir $@)
	@$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cu
	@echo -n " $<"
	@mkdir -p $(dir $@)
	@$(CU) $(CPPFLAGS) $(CUFLAGS) $(CU_OPT) -rdc=true -c $< -o $@ -lcudart -lcuda

$(OBJ_DIR)/device.a: $(CU_OBJ)
	@echo "Building device archive..."
	@ar cr $@ $^
	@ranlib $@

$(OBJ_DIR)/device_link.o: $(OBJ_DIR)/device.a
	@echo "Linking device code..."
	$(CU) $(CUFLAGS) $(CU_OPT) --device-link $^ -o $@ -lcudart -lcuda

$(PROGRAM)_$(VERSION): $(OBJ) $(OBJ_DIR)/device_link.o $(OBJ_DIR)/device.a
	@echo "Building executable..."
	$(CC) $(CFLAGS) $(LDLIBS) $^ -o $@ -lcudart -lcuda

clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR) $(PROGRAM)_$(VERSION)
