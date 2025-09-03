#include <stdio.h>
#include <stdlib.h>
// #include <cuda_runtime.h>
#include <cuda.h>
#include <omp.h>
// #include <device_launch_parameters.h>
// #include <nvToolsExt.h>
#include <iostream>

#define INF ((1 << 30) - 1)
#define DEV_NO 0
cudaDeviceProp prop;

int n, m, v_real;
int* Dist_host;
#define blocksize 64

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    v_real = n;
    if(n % blocksize != 0)  n += (blocksize - n % blocksize);
    Dist_host = (int*)malloc(n * n * sizeof(int));
    cudaHostRegister(Dist_host, n * n * sizeof(int), cudaHostRegisterDefault);
    // cudaHostAlloc((void**)&Dist_host, n * n * sizeof(int), cudaHostAllocDefault);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if(i == j)  Dist_host[i*n + j] = 0;
            else    Dist_host[i*n + j] = INF;
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist_host[pair[0] * n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < v_real; ++i) {
        fwrite(&Dist_host[i * n], sizeof(int), v_real, outfile);
    }
    fclose(outfile);
}

__global__ void phase1(int* Dist, int round, int n) {
    __shared__ int shared_mem[blocksize][blocksize];
    int x = threadIdx.x;
    int y = threadIdx.y;
    int x_global = x + round * blocksize;
    int y_global = y + round * blocksize;
    int ind_global = y_global * n + x_global;

    shared_mem[y][x] = Dist[ind_global];
    shared_mem[y][x + 32] = Dist[ind_global + 32];
    shared_mem[y + 32][x] = Dist[ind_global + n * 32];
    shared_mem[y + 32][x + 32] = Dist[ind_global + n * 32 + 32];
    __syncthreads();

    // #pragma unroll
    for (int k = 0; k < blocksize; ++k) {
        shared_mem[y][x] = min(shared_mem[y][x], shared_mem[y][k] + shared_mem[k][x]);
        shared_mem[y][x + 32] = min(shared_mem[y][x + 32], shared_mem[y][k] + shared_mem[k][x + 32]);
        shared_mem[y + 32][x] = min(shared_mem[y + 32][x], shared_mem[y + 32][k] + shared_mem[k][x]);
        shared_mem[y + 32][x + 32] = min(shared_mem[y + 32][x + 32], shared_mem[y + 32][k] + shared_mem[k][x + 32]);
        __syncthreads();
    }

    Dist[ind_global] = shared_mem[y][x];
    Dist[ind_global + 32] = shared_mem[y][x + 32];
    Dist[ind_global + n * 32] = shared_mem[y + 32][x];
    Dist[ind_global + n * 32 + 32] = shared_mem[y + 32][x + 32];
}

__global__ void phase2(int* Dist, int round, int n){
    if(round == blockIdx.x) return;
    __shared__ int shared_row[blocksize][blocksize];
    __shared__ int shared_col[blocksize][blocksize];
    __shared__ int shared_pivot[blocksize][blocksize];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int x_global = x + round * blocksize;
    int y_global = y + round * blocksize;

    //pivot
    int ind_global = y_global * n + x_global;
    shared_pivot[y][x] = Dist[ind_global];
    shared_pivot[y][x + 32] = Dist[ind_global + 32];
    shared_pivot[y + 32][x] = Dist[ind_global + n * 32];
    shared_pivot[y + 32][x + 32] = Dist[ind_global + n * 32 + 32];
    //shared_row
    int row_x = x + blockIdx.x * blocksize;
    int ind_row_global = y_global * n + row_x;
    shared_row[y][x] = Dist[ind_row_global];
    shared_row[y][x + 32] = Dist[ind_row_global + 32];
    shared_row[y + 32][x] = Dist[ind_row_global + n * 32];
    shared_row[y + 32][x + 32] = Dist[ind_row_global + n * 32 + 32];
    //shared_col
    int col_y = y + blockIdx.x * blocksize;
    int ind_col_global = col_y * n + x_global;
    shared_col[y][x] = Dist[ind_col_global];
    shared_col[y][x + 32] = Dist[ind_col_global + 32];
    shared_col[y + 32][x] = Dist[ind_col_global + n * 32];
    shared_col[y + 32][x + 32] = Dist[ind_col_global + n * 32 + 32];
    __syncthreads();

    // #pragma unroll
    for (int k = 0; k < blocksize; ++k) {
        shared_row[y][x] = min(shared_row[y][x], shared_pivot[y][k] + shared_row[k][x]);
        shared_row[y][x + 32] = min(shared_row[y][x + 32], shared_pivot[y][k] + shared_row[k][x + 32]);
        shared_row[y + 32][x] = min(shared_row[y + 32][x], shared_pivot[y + 32][k] + shared_row[k][x]);
        shared_row[y + 32][x + 32] = min(shared_row[y + 32][x + 32], shared_pivot[y + 32][k] + shared_row[k][x + 32]);

        shared_col[y][x] = min(shared_col[y][x], shared_col[y][k] + shared_pivot[k][x]);
        shared_col[y][x + 32] = min(shared_col[y][x + 32], shared_col[y][k] + shared_pivot[k][x + 32]);
        shared_col[y + 32][x] = min(shared_col[y + 32][x], shared_col[y + 32][k] + shared_pivot[k][x]);
        shared_col[y + 32][x + 32] = min(shared_col[y + 32][x + 32], shared_col[y + 32][k] + shared_pivot[k][x + 32]);
        __syncthreads();
    }

    Dist[ind_row_global] = shared_row[y][x];
    Dist[ind_row_global + 32] = shared_row[y][x + 32];
    Dist[ind_row_global + n * 32] = shared_row[y + 32][x];
    Dist[ind_row_global + n * 32 + 32] = shared_row[y + 32][x + 32];

    Dist[ind_col_global] = shared_col[y][x];
    Dist[ind_col_global + 32] = shared_col[y][x + 32];
    Dist[ind_col_global + n * 32] = shared_col[y + 32][x];
    Dist[ind_col_global + n * 32 + 32] = shared_col[y + 32][x + 32];
}

__global__ void phase3(int* Dist, int round, int n, int start) {
    if(round == blockIdx.x || round == blockIdx.y + start)  return;

    // const int block_y = (blockIdx.y >= round) ? blockIdx.y + 1 : blockIdx.y ;
    // const int block_x = (blockIdx.x >= round) ? blockIdx.x + 1 : blockIdx.x ;

    __shared__ int shared_row[blocksize][blocksize];
    __shared__ int shared_col[blocksize][blocksize];
    __shared__ int shared_mem[blocksize][blocksize];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int x_global = x + round * blocksize;
    int y_global = y + round * blocksize;

    int i = x + blockIdx.x * blocksize;
    int j = y + (blockIdx.y + start) * blocksize;
    int ind_global = j * n + i;
    shared_mem[y][x] = Dist[ind_global];
    shared_mem[y][x + 32] = Dist[ind_global + 32];
    shared_mem[y + 32][x] = Dist[ind_global + n * 32];
    shared_mem[y + 32][x + 32] = Dist[ind_global + n * 32 + 32];
    //shared_row
    int ind_col_global = j * n + x_global;
    shared_col[y][x] = Dist[ind_col_global];
    shared_col[y][x + 32] = Dist[ind_col_global + 32];
    shared_col[y + 32][x] = Dist[ind_col_global + n * 32];
    shared_col[y + 32][x + 32] = Dist[ind_col_global + n * 32 + 32];
    //shared_col
    int ind_row_global = y_global * n + i;
    shared_row[y][x] = Dist[ind_row_global];
    shared_row[y][x + 32] = Dist[ind_row_global + 32];
    shared_row[y + 32][x] = Dist[ind_row_global + n * 32];
    shared_row[y + 32][x + 32] = Dist[ind_row_global + n * 32 + 32];
    __syncthreads();

    // #pragma unroll 8
    for (int k = 0; k < blocksize; ++k) {
        shared_mem[y][x] = min(shared_mem[y][x], shared_col[y][k] + shared_row[k][x]);
        shared_mem[y][x + 32] = min(shared_mem[y][x + 32], shared_col[y][k] + shared_row[k][x + 32]);
        shared_mem[y + 32][x] = min(shared_mem[y + 32][x], shared_col[y + 32][k] + shared_row[k][x]);
        shared_mem[y + 32][x + 32] = min(shared_mem[y + 32][x + 32], shared_col[y + 32][k] + shared_row[k][x + 32]);
    }

    Dist[ind_global] = shared_mem[y][x];
    Dist[ind_global + 32] = shared_mem[y][x + 32];
    Dist[ind_global + n * 32] = shared_mem[y + 32][x];
    Dist[ind_global + n * 32 + 32] = shared_mem[y + 32][x + 32];
}

int main(int argc, char* argv[]) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    input(argv[1]);

    // cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    size_t size = n * n * sizeof(int);
    // Dist_res = (int*)malloc(d_size);
    // cudaHostAlloc((void**)&Dist_res, d_size, cudaHostAllocDefault);
    // cudaHostRegister(Dist_res, d_size, cudaHostRegisterDefault);

    int* device[2];

    int round = n / blocksize;
    dim3 threadsPerBlock(32, 32);
    dim3 grid_phase2(round, 1);

    #pragma omp parallel num_threads(2)
    {
        int d_id = omp_get_thread_num();
        int data_len, start;
        if(d_id == 0){
            data_len = round/2;
            start = 0;
        }
        else{
            data_len = round - round/2;
            start = round/2;
        }
        dim3 grid_phase3(round, data_len);
        cudaSetDevice(d_id);
        cudaMalloc(&device[d_id], size);
        cudaMemcpy(device[d_id], Dist_host, size, cudaMemcpyHostToDevice);

        int tmp = blocksize * n;
        // nvtxRangePushA("compute");
        for (int r = 0; r < round; ++r) {
            if(start <= r && r < start + data_len)  cudaMemcpy(device[!d_id] +  r * tmp, device[d_id] + r * tmp, tmp * sizeof(int), cudaMemcpyDeviceToDevice);

            #pragma omp barrier

            phase1<<<1, threadsPerBlock>>>(device[d_id], r, n);

            phase2<<<grid_phase2, threadsPerBlock>>>(device[d_id], r, n);

            phase3<<<grid_phase3, threadsPerBlock>>>(device[d_id], r, n, start);
        }
        // nvtxRangePop();
        // cudaMemcpy(Dist_host, Dist_dev, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(Dist_host + start * tmp, device[d_id] + start * tmp, data_len * tmp * sizeof(int), cudaMemcpyDeviceToHost);
    }

    output(argv[2]);

    // cudaFree(Dist_dev);
    // cudaFreeHost(Dist_host);
    // cudaFreeHost(Dist_res);
    cudaHostUnregister(Dist_host);
    // cudaHostUnregister(Dist_res);
    free(Dist_host);
    // free(Dist_res);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) +  (end.tv_nsec - start.tv_nsec)/1e9;
    std::cout << "Elapsed time: " << elapsed << " s\n";
    return 0;
}
