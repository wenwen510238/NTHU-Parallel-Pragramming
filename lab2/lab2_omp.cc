#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <vector>
using namespace std;
int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long answer = 0;

	vector<unsigned long long> pixelArr;
	pixelArr.clear();
	
#pragma omp parallel	
	{
        int rank = omp_get_thread_num();
		int size = omp_get_num_threads();

		unsigned long long pixels = 0;
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
		#pragma omp critical
		{
			pixelArr.push_back(pixels);
		}
	}
	for(int i=0; i<pixelArr.size(); i++){
		answer += pixelArr[i];
	}
	printf("%llu\n", (4 * answer) % k);

}
