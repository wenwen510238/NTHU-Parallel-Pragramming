#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 32
#define MAX_BC 128
#define MAX_D 64

void input(char *input_filename);
void output(char *output_filename);
void flash_attention(float *q_d, float *k_d, float *v_d, float *o_d);

__global__ void init_m_kernel(float *m_d, int N);
__global__ void QKDotAndScalar_CUDA(float *q_d, float *k_d, float *sij_d, int br, int bc, int d, float scalar);
__global__ void RowMax_CUDA(float *mij_d, float *sij_d, int br, int bc);
__global__ void MinusMaxAndExp_CUDA(float *pij_d, float *sij_d, float *mij_d, int br, int bc);
__global__ void RowSum_CUDA(float *lij_d, float *pij_d, int br, int bc);
__global__ void ComputeMiLiNew_CUDA(float *mi_d, float *li_d, float *mij_d, float *lij_d, float *mi_new_d, float *li_new_d, int br);
__global__ void UpdateOi_CUDA(float *oi_d, float *mi_d, float *li_d, float *mi_new_d, float *li_new_d, float *lij_d, float *pij_d, float *vj_d, float *mij_d, int br, int bc, int d);
__global__ void CopyMiLiNew_CUDA(float *mi_d, float *li_d, float *mi_new_d, float *li_new_d, int br);

float _max(float a, float b) { return a > b ? a : b; }
float _min(float a, float b) { return a < b ? a : b; }
double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;
float *Q_d, *K_d, *V_d, *O_d;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    double start, end;
    start = getTimeStamp();

    flash_attention(
        Q_d,
        K_d,
        V_d,
        O_d
    );

    end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);

    return 0;
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    size_t total_size = B * N * d * sizeof(float);

    Q = (float *)malloc(total_size);
    K = (float *)malloc(total_size);
    V = (float *)malloc(total_size);
    O = (float *)malloc(total_size);

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, total_size);

    fclose(file);

    // Allocate device memory
    cudaMalloc((void **)&Q_d, total_size);
    cudaMalloc((void **)&K_d, total_size);
    cudaMalloc((void **)&V_d, total_size);
    cudaMalloc((void **)&O_d, total_size);

    // Copy data from host to device
    cudaMemcpy(Q_d, Q, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(O_d, O, total_size, cudaMemcpyHostToDevice);
}

void output(char *output_filename) {
    size_t total_size = B * N * d * sizeof(float);

    // Copy data from device to host
    cudaMemcpy(O, O_d, total_size, cudaMemcpyDeviceToHost);

    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    // Free device memory
    cudaFree(Q_d);
    cudaFree(K_d);
    cudaFree(V_d);
    cudaFree(O_d);

    fclose(file);
}

void flash_attention(float *q_d, float *k_d, float *v_d, float *o_d) {
    // Allocate temporary arrays on device
    float *l_d, *m_d;
    cudaMalloc((void **)&l_d, N * sizeof(float));
    cudaMalloc((void **)&m_d, N * sizeof(float));

    // Other temporary arrays
    float *kj_d, *vj_d, *qi_d, *oi_d, *li_d, *mi_d;
    float *sij_d, *pij_d, *mij_d, *lij_d;
    float *mi_new_d, *li_new_d;

    // int a = 4*d;
    // int bc = N/a, br = _min(N/a, d);
    // if (N % a != 0) bc++;

    int bc = min(MAX_BC, N), br = N;

    int tr = (N + br - 1) / br, tc = (N + bc - 1) / bc;

    size_t kj_size = bc * d * sizeof(float);
    size_t vj_size = bc * d * sizeof(float);
    size_t qi_size = br * d * sizeof(float);
    size_t oi_size = br * d * sizeof(float);
    size_t li_size = br * sizeof(float);
    size_t mi_size = br * sizeof(float);
    size_t sij_size = br * bc * sizeof(float);
    size_t pij_size = br * bc * sizeof(float);
    size_t mij_size = br * sizeof(float);
    size_t lij_size = br * sizeof(float);

    cudaMalloc((void **)&kj_d, kj_size);
    cudaMalloc((void **)&vj_d, vj_size);
    cudaMalloc((void **)&qi_d, qi_size);
    cudaMalloc((void **)&oi_d, oi_size);
    cudaMalloc((void **)&li_d, li_size);
    cudaMalloc((void **)&mi_d, mi_size);
    cudaMalloc((void **)&sij_d, sij_size);
    cudaMalloc((void **)&pij_d, pij_size);
    cudaMalloc((void **)&mij_d, mij_size);
    cudaMalloc((void **)&lij_d, lij_size);
    cudaMalloc((void **)&mi_new_d, mi_size);
    cudaMalloc((void **)&li_new_d, li_size);

    // printf("br: %d, bc: %d, tr: %d, tc: %d\n", br, bc, tr, tc);

    for (int k = 0; k < B; k++) {

        // Initialize l_d to zeros
        cudaMemset(l_d, 0, N * sizeof(float));
        // Initialize m_d to -FLT_MAX
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        init_m_kernel<<<blocksPerGrid, threadsPerBlock>>>(m_d, N);
        size_t offset = k * N * d;

        for (int j = 0; j < tc; j++) {
            cudaMemcpy(kj_d, k_d + offset + j * bc * d, bc * d * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(vj_d, v_d + offset + j * bc * d, bc * d * sizeof(float), cudaMemcpyDeviceToDevice);

            for (int i = 0; i < tr; i++) {
                cudaMemcpy(qi_d, q_d + offset + i * br * d, br * d * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(oi_d, o_d + offset + i * br * d, br * d * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(li_d, l_d + i * br, br * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(mi_d, m_d + i * br, br * sizeof(float), cudaMemcpyDeviceToDevice);

                dim3 blockDimQK(BLOCK_SIZE, BLOCK_SIZE);
                dim3 gridDimQK((bc + blockDimQK.x - 1) / blockDimQK.x, (br + blockDimQK.y - 1) / blockDimQK.y);
                QKDotAndScalar_CUDA<<<gridDimQK, blockDimQK>>>(qi_d, kj_d, sij_d, br, bc, d, 1.0 / sqrt(d));

                int threadsPerBlockRM = BLOCK_SIZE;
                int blocksPerGridRM = (br + threadsPerBlockRM - 1) / threadsPerBlockRM;
                RowMax_CUDA<<<blocksPerGridRM, threadsPerBlockRM>>>(mij_d, sij_d, br, bc);

                dim3 blockDimMME(BLOCK_SIZE, BLOCK_SIZE);
                dim3 gridDimMME((bc + blockDimMME.x - 1) / blockDimMME.x, (br + blockDimMME.y - 1) / blockDimMME.y);
                MinusMaxAndExp_CUDA<<<gridDimMME, blockDimMME>>>(pij_d, sij_d, mij_d, br, bc);
                // int threadsPerBlockMi = BLOCK_SIZE;
                // int blocksPerGridMi = (br + threadsPerBlockMi - 1) / threadsPerBlockMi;
                // MinusMaxAndExp_CUDA<<<blocksPerGridMi, threadsPerBlockMi>>>(pij_d, sij_d, mij_d, br, bc);

                int threadsPerBlockRS = BLOCK_SIZE;
                int blocksPerGridRS = (br + threadsPerBlockRS - 1) / threadsPerBlockRS;
                RowSum_CUDA<<<blocksPerGridRS, threadsPerBlockRS>>>(lij_d, pij_d, br, bc);

                // int threadsPerBlockCM = 256;
                // int blocksPerGridCM = (br + threadsPerBlockCM - 1) / threadsPerBlockCM;
                // ComputeMiLiNew_CUDA<<<blocksPerGridCM, threadsPerBlockCM>>>(mi_d, li_d, mij_d, lij_d, mi_new_d, li_new_d, br);

                int blocksPerGridOi = (br + BLOCK_SIZE - 1) / BLOCK_SIZE;
                UpdateOi_CUDA<<<blocksPerGridOi, BLOCK_SIZE>>>(oi_d, mi_d, li_d, mi_new_d, li_new_d, lij_d, pij_d, vj_d, mij_d, br, bc, d);

                // CopyMiLiNew_CUDA<<<blocksPerGridCM, threadsPerBlockCM>>>(mi_d, li_d, mi_new_d, li_new_d, br);
                cudaMemcpy(o_d + offset + i * br * d, oi_d, br * d * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(l_d + i * br, li_d, br * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(m_d + i * br, mi_d, br * sizeof(float), cudaMemcpyDeviceToDevice);
            }
        }
    }

    cudaFree(kj_d);
    cudaFree(vj_d);
    cudaFree(qi_d);
    cudaFree(oi_d);
    cudaFree(li_d);
    cudaFree(mi_d);
    cudaFree(sij_d);
    cudaFree(pij_d);
    cudaFree(mij_d);
    cudaFree(lij_d);
    cudaFree(mi_new_d);
    cudaFree(li_new_d);
    cudaFree(l_d);
    cudaFree(m_d);
}

__global__ void init_m_kernel(float *m_d, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        m_d[idx] = -FLT_MAX;
    }
}

// __global__ void QKDotAndScalar_CUDA(float *q_d, float *k_d, float *sij_d, int br, int bc, int d, float scalar) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y; // i
//     int col = blockIdx.x * blockDim.x + threadIdx.x; // j

//     // if (row < br && col < bc) {
//         float sum = 0.0f;
//         for (int t = 0; t < d; t++) {
//             sum += q_d[row * d + t] * k_d[col * d + t];
//         }
//         sij_d[row * bc + col] = sum * scalar;
//     // }
// }

__global__ void QKDotAndScalar_CUDA(float *q_d, float *k_d, float *sij_d, int br, int bc, int d, float scalar) {
    // Calculate row and column indices
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y; // i
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x; // j

    // Initialize the sum for each thread
    float sum = 0.0f;

    // Declare shared memory tiles for q_d and k_d
    __shared__ float q_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float k_shared[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over tiles of dimension 'd'
    for (int m = 0; m < d / BLOCK_SIZE; ++m) {
        q_shared[threadIdx.y][threadIdx.x] = q_d[row * d + m * BLOCK_SIZE + threadIdx.x];

        k_shared[threadIdx.x][threadIdx.y] = k_d[col * d + m * BLOCK_SIZE + threadIdx.y];
        __syncthreads();

        for (int t = 0; t < BLOCK_SIZE; ++t)
            sum += q_shared[threadIdx.y][t] * k_shared[threadIdx.x][t];

        __syncthreads();
    }

    sij_d[row * bc + col] = sum * scalar;
}

__global__ void RowMax_CUDA(float *mij_d, float *sij_d, int br, int bc) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    // if (row < br) {
        float max_val = sij_d[row * bc];
        for (int j = 1; j < bc; j++) {
            float val = sij_d[row * bc + j];
            if (val > max_val) {
                max_val = val;
            }
        }
        mij_d[row] = max_val;
    // }
}

__global__ void MinusMaxAndExp_CUDA(float *pij_d, float *sij_d, float *mij_d, int br, int bc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // if (row < br && col < bc) {
        float val = sij_d[row * bc + col] - mij_d[row];
        pij_d[row * bc + col] = expf(val);
    // }
}

// __global__ void MinusMaxAndExp_CUDA(float *pij_d, float *sij_d, float *mij_d, int br, int bc) {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     // if (row < br && col < bc) {
//     for(int j = 0; j < bc; j++){
//         float val = sij_d[row * bc + j] - mij_d[row];
//         pij_d[row * bc + j] = expf(val);
//     }
//     // }
// }

__global__ void RowSum_CUDA(float *lij_d, float *pij_d, int br, int bc) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    // if (row < br) {
        float sum = 0.0f;
        for (int j = 0; j < bc; j++) {
            sum += pij_d[row * bc + j];
        }
        lij_d[row] = sum;
    // }
}

__global__ void UpdateOi_CUDA(float *oi_d, float *mi_d, float *li_d, float *mi_new_d, float *li_new_d, float *lij_d, float *pij_d, float *vj_d, float *mij_d, int br, int bc, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx < br) {
        // float mi = mi_d[idx];
        // float mij = mij_d[idx];
        // float li = li_d[idx];
        // float lij = lij_d[idx];

        // float mi_new = fmaxf(mi, mij);
        // float li_new = expf(mi - mi_new) * li + expf(mij - mi_new) * lij;

        // mi_new_d[idx] = mi_new;
        // li_new_d[idx] = li_new;
    // }
    __shared__ float shared_pij_d[BLOCK_SIZE * MAX_BC];
    __shared__ float shared_vj_d[MAX_BC * MAX_D];

    for(int i = 0; i < BLOCK_SIZE; i++){
        for (int t = 0; t < bc; t += BLOCK_SIZE) {
            shared_pij_d[i * bc + t + threadIdx.x] = pij_d[(blockIdx.x * blockDim.x + i) * bc + t + threadIdx.x];
        }
    }

    for(int t = 0; t < bc; t++){
        for(int i = 0; i < d; i += BLOCK_SIZE){
            shared_vj_d[t * d + i + threadIdx.x] = vj_d[t * d + i + threadIdx.x];
        }
    }

    __syncthreads();

    mi_new_d[idx] = max(mi_d[idx], mij_d[idx]);
    li_new_d[idx] = exp(mi_d[idx] - mi_new_d[idx]) * li_d[idx] + exp(mij_d[idx] - mi_new_d[idx]) * lij_d[idx];

    // if (row < br && col < d) {
    for (int i = 0; i < d; i++) {
        // Compute pv
        float pv = 0.0f;

        for (int t = 0; t < bc; t++) {
            pv += shared_pij_d[threadIdx.x * bc + t] * shared_vj_d[t * d + i];
        }

        oi_d[idx * d + i] = (li_d[idx] * exp(mi_d[idx] - mi_new_d[idx]) * oi_d[idx * d + i] + exp(mij_d[idx] - mi_new_d[idx]) * pv) / li_new_d[idx];
    }

    mi_d[idx] = mi_new_d[idx];
    li_d[idx] = li_new_d[idx];
    // }
}

