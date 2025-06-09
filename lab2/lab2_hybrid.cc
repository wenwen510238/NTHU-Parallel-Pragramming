#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include <omp.h>

using namespace std;

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	MPI_Init(&argc, &argv);
    int mpi_rank, mpi_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	// printf("mpi_rank %2d/%2d\n", mpi_rank, mpi_size);
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long answer = 0;

	int remain = r % mpi_size;
	unsigned long long eachPart = r / mpi_size;
	unsigned long long start = mpi_rank * eachPart, end = start + eachPart, local_pixels = 0;
	if(remain!=0){
		if(mpi_rank < remain){
			start += mpi_rank;
			end = end + mpi_rank + 1;
		}else{
			start += remain;
			end += remain;
		}
	}
	
#pragma omp parallel reduction(+:local_pixels) 
	{
        int omp_rank = omp_get_thread_num();
		int omp_size = omp_get_num_threads();

		int local_size = end - start;
		int local_remain = local_size % omp_size;
		unsigned long long local_eachPart = local_size / omp_size;
		unsigned long long local_start = start + omp_rank * local_eachPart, local_end = local_start + local_eachPart;
		unsigned long long y, sqt;
		if(local_remain!=0){
			if(omp_rank < local_remain){
				local_start += omp_rank;
				local_end = local_end + omp_rank + 1;
			}else{
				local_start += local_remain;
				local_end += local_remain;
			}
		}

		// unsigned long long local_pixels  = 0;
		// printf("omp_rank %2d/%2d\n", omp_rank, omp_size);
		
		// #pragma omp for schedule(dynamic) 
		for(unsigned long long i=local_start; i<local_end; i++){
			if(i == local_start ){
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
			local_pixels += y;
		}
	}
	// pixels += local_pixels;
	local_pixels %= k;
	MPI_Reduce(&local_pixels, &answer, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if(mpi_rank == 0)	printf("%llu\n", (4 * answer) % k);
	MPI_Finalize();

}