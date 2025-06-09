CC = gcc
CXX = g++
CFLAGS = -lm -O3

lab2_pthread: CFLAGS += -pthread
lab2_omp: CFLAGS += -fopenmp
lab2_hybrid: CXX = mpicxx
lab2_hybrid: CFLAGS += -fopenmp

CXXFLAGS = $(CFLAGS)

# 定義物件檔目錄
# OBJ_DIR = obj

# 檔案名稱與物件檔區分，並將 .o 檔案放到 obj/ 資料夾
SRCS := $(wildcard *.cc)
# OBJS_pthread := $(addprefix $(OBJ_DIR)/, $(filter lab2_pthread.o, $(SRCS:.cc=.o)))
# OBJS_omp := $(addprefix $(OBJ_DIR)/, $(filter lab2_omp.o, $(SRCS:.cc=.o)))
# OBJS_hybrid := $(addprefix $(OBJ_DIR)/, $(filter lab2_hybrid.o, $(SRCS:.cc=.o)))
OBJS_pthread := $(filter lab2_pthread.o, $(SRCS:.cc=.o))
OBJS_omp := $(filter lab2_omp.o, $(SRCS:.cc=.o))
OBJS_hybrid := $(filter lab2_hybrid.o, $(SRCS:.cc=.o))
OBJS := $(SRCS:.cc=.o)

TARGETS = lab2_pthread lab2_omp lab2_hybrid

.PHONY: all run clean

# 針對所有目標進行編譯
# all: $(OBJ_DIR) $(TARGETS)
all: $(TARGETS)

# 確保 obj 目錄存在
# $(OBJ_DIR):
# 	mkdir -p $(OBJ_DIR)

# 針對各個目標的編譯
lab2_pthread: $(OBJS_pthread)
	$(CXX) $(CXXFLAGS) $(OBJS_pthread) -o $@

lab2_omp: $(OBJS_omp)
	$(CXX) $(CXXFLAGS) $(OBJS_omp) -o $@

lab2_hybrid: $(OBJS_hybrid)
	$(CXX) $(CXXFLAGS) $(OBJS_hybrid) -o $@

# 編譯規則，將生成的 .o 檔案放進 obj 資料夾
# $(OBJ_DIR)/%.o: %.cc
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

# clean:
# 	rm -f $(TARGETS) $(OBJ_DIR)/*.o
# 	rm -rf $(OBJ_DIR)
clean:
	rm -f $(TARGETS) $(OBJS)

run-pthread:
	srun -c4 -n1 ./lab2_pthread 5 100

run-omp:
	srun -c4 -n1 ./lab2_omp 5 100

run-hybrid:
	srun -c4 -n6 ./lab2_hybrid 5 100

pthread-judge:
	lab2_pthread-judge

omp-judge:
	lab2_omp-judge

hybrid-judge:
	lab2_hybrid-judge


