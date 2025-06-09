# CC = mpicc
CXX = mpicxx
# CXXFLAGS = -std=c++17 -O3
CXXFLAGS = -flto -std=c++17 -fopenmp -lm -Ofast -mtune=native -march=native -funroll-loops -falign-functions=32 -falign-loops=32
# CFLAGS = -O3

SRCS := $(wildcard *.cc)

OBJS := $(SRCS:.cc=.o)

TARGETS = lab1

.PHONY: all clean run
all: $(TARGETS)

$(TARGETS): $(OBJS)
		$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGETS)

.cc.o:
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGETS) $(OBJS)

run:
	srun -N1 -n2 ./lab1 5 21

judge:
	lab1-judge
