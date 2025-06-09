CC = gcc
CXX = g++
CFLAGS = -lm -O3 -mavx512f -ffp-contract=off
LDLIBS = -lpng
hw2a: CFLAGS += -pthread
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)

# 定義物件檔目錄
# OBJ_DIR = obj

# 檔案名稱與物件檔區分，並將 .o 檔案放到 obj/ 資料夾
SRCS := $(wildcard *.cc)

OBJS_pthread := $(filter hw2a.o, $(SRCS:.cc=.o))
OBJS_hybrid := $(filter hw2b.o, $(SRCS:.cc=.o))
OBJS := $(SRCS:.cc=.o)

TARGETS = hw2a hw2b

.PHONY: all run clean

# 針對所有目標進行編譯
# all: $(OBJ_DIR) $(TARGETS)
all: $(TARGETS)

# 確保 obj 目錄存在
# $(OBJ_DIR):
# 	mkdir -p $(OBJ_DIR)

# 針對各個目標的編譯
hw2a: $(OBJS_pthread)
	$(CXX) $(CXXFLAGS) $(OBJS_pthread) -o $@ $(LDLIBS)

hw2b: $(OBJS_hybrid)
	$(CXX) $(CXXFLAGS) $(OBJS_hybrid) -o $@ $(LDLIBS)

# 編譯規則，將生成的 .o 檔案放進 obj 資料夾
# $(OBJ_DIR)/%.o: %.cc
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

# clean:
# 	rm -f $(TARGETS) $(OBJ_DIR)/*.o
# 	rm -rf $(OBJ_DIR)
clean:
	rm -f $(TARGETS) $(OBJS)

run-pthread:
	srun -n1 -c1 ./wrapper.sh ./hw2a strict_33_out.png 10000 -0.29899250664589705 -0.2772002993319328 -0.6327591639095336 -0.6433840614725646 7680 4320

run-hybrid:
	srun -n3 -c5 ./wrapper.sh ./hw2b b_out.png 2602 -3 0.2 -3 0.2 979 2355
pthread-judge:
	hw2a-judge

hybrid-judge:
	hw2b-judge



