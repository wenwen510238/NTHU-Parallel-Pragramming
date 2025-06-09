#include <pthread.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

struct ThreadArgs{
	unsigned long long r;
	unsigned long long rank;
	unsigned long long ncpus;
	unsigned long long res;
	unsigned long long k;
};

void* computePixel(void* arg){
	ThreadArgs* args = (ThreadArgs*)arg;

	int remain = args->r % args->ncpus;
	unsigned long long eachPart = (args->r)/(args->ncpus), pixel = 0;
	unsigned long long y, sqt, start = (args->rank)*eachPart, end = start + eachPart;
	if(remain!=0){
		if((args->rank) < remain){
			start += (args->rank);
			end = end + (args->rank) + 1;
		}else{
			start += remain;
			end += remain;
		}
	}
	for(unsigned long long i=start; i<end; i++){
		if(i == start ){
			y = ceil(sqrtl((args->r)*(args->r) - i*i));
			sqt = (y-1) *(y-1);
		}
		else{
			unsigned long long tmp = (args->r)*(args->r) - i*i;
			while(tmp <= sqt){
				y = y - 1;
				sqt = (y-1) *(y-1);
			}
		}
		pixel += y;
	}
	pixel %= (args->k);
	args->res = pixel;
    pthread_exit((void*)args);

}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);
	// printf("cpu: %llu\n", ncpus);

	ThreadArgs args[ncpus];

	unsigned long long pixels = 0;
    pthread_t threads[ncpus];

	for (int i=0; i<ncpus; i++) {
		args[i].rank = i;
		args[i].r = atoll(argv[1]);
		args[i].ncpus = ncpus;
		args[i].k = atoll(argv[2]);
        int rc = pthread_create(&threads[i], NULL, computePixel, &args[i]);
		if(rc != 0){
			printf("pthread_create error\n");
			return 1;
		}
	}
	for (int i=0; i<ncpus; i++) {
		ThreadArgs* result;
        if(pthread_join(threads[i], (void**)&result) != 0){
			printf("pthread_join error\n");
			return 1;
		}
		// printf("Thread %d result: %llu\n", threads[i], result->res);
		pixels += result->res;

	}
	printf("%llu\n", (4 * pixels) % args[0].k);
}
