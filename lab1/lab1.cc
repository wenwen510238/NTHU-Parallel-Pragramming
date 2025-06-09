#include "mpi.h"
#include <omp.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
	int rank, size;
	// double startTime, endTime;

    MPI_Init(&argc, &argv);
	unsigned long long r = atoll(argv[1]), k = atoll(argv[2]), pixels = 0, answer;
	MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// if(rank == 0)
	// 	startTime = omp_get_wtime();
	// #pragma omp parallel for
	int remain = r%size;
	unsigned long long eachPart = r/size;
	unsigned long long y, sqt, start = rank*eachPart, end = start + eachPart;
	if(remain!=0){
		if(rank < remain){
			start+=rank;
			end = end + rank + 1;
		}else{
			start+=remain;
			end+=remain;
		}
	}
	// for(unsigned long long i=rank; i<r; i+=size){
	for(unsigned long long i=start; i<end; i++){
		if(i == start ){
			y = ceil(sqrtl(r*r - i*i));
			sqt = (y-1) *(y-1);
		}
		else{
			unsigned long long tmp = r*r - i*i;
			while(tmp <= sqt){
				y = y - 1;
				sqt = (y-1) *(y-1);
			}
		}
		pixels += y;
	}
	pixels %= k;
	MPI_Reduce(&pixels, &answer, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	if(rank == 0)	printf("%llu\n",(4*answer)%k);
	// if(rank == 0){
	// 	endTime = omp_get_wtime();
	// 	printf("Elapsed time: %f s\n", (endTime - startTime));
	// }
	MPI_Finalize();
	return 0;
}
