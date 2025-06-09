#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>

#define Z 2
#define Y 5
#define X 5
#define xBound X / 2
#define yBound Y / 2
#define SCALE 8
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

int read_png(const char* filename, unsigned char** image, unsigned* height,
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width,
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    // png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_set_filter(png_ptr, 0, PNG_FILTER_NONE);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);
    // png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__constant__ char mask[Z][Y][X] = { { { -1, -4, -6, -4, -1 },
                                        { -2, -8, -12, -8, -2 },
                                        { 0, 0, 0, 0, 0 },
                                        { 2, 8, 12, 8, 2 },
                                        { 1, 4, 6, 4, 1 } },
                                      { { -1, -2, 0, 2, 1 },
                                        { -4, -8, 0, 8, 4 },
                                        { -6, -12, 0, 12, 6 },
                                        { -4, -8, 0, 8, 4 },
                                        { -1, -2, 0, 2, 1 } } };

inline __device__ int bound_check(int val, int lower, int upper) {
    if (val >= lower && val < upper)
        return 1;
    else
        return 0;
}

__global__ void sobel(unsigned char *s, unsigned char *t, unsigned height, unsigned width, unsigned channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    __shared__ unsigned char shared_s[TILE_HEIGHT + Y -1][TILE_WIDTH + X -1][3];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int shared_x = tx + xBound;
    int shared_y = ty + yBound;

    // Load data into shared memory
    for (int dy = -yBound; dy <= yBound; dy++) {
        for (int dx = -xBound; dx <= xBound; dx++) {
            int global_x = x + dx;
            int global_y = y + dy;
            int shared_x_idx = shared_x + dx;
            int shared_y_idx = shared_y + dy;

            if (shared_x_idx >= 0 && shared_x_idx < TILE_WIDTH + X - 1 &&
                shared_y_idx >= 0 && shared_y_idx < TILE_HEIGHT + Y - 1) {
                if (bound_check(global_x, 0, width) && bound_check(global_y, 0, height)) {
                    int index = channels * (width * global_y + global_x);
                    shared_s[shared_y_idx][shared_x_idx][0] = s[index + 0];
                    shared_s[shared_y_idx][shared_x_idx][1] = s[index + 1];
                    shared_s[shared_y_idx][shared_x_idx][2] = s[index + 2];
                } else {
                    shared_s[shared_y_idx][shared_x_idx][0] = 0;
                    shared_s[shared_y_idx][shared_x_idx][1] = 0;
                    shared_s[shared_y_idx][shared_x_idx][2] = 0;
                }
            }
        }
    }

    __syncthreads();

    float val[Z][3] = {0};

    // Apply the Sobel operator using shared memory
    for (int i = 0; i < Z; ++i) {
        val[i][0] = 0.f;
        val[i][1] = 0.f;
        val[i][2] = 0.f;

        for (int v = -yBound; v <= yBound; ++v) {
            for (int u = -xBound; u <= xBound; ++u) {
                int s_x = shared_x + u;
                int s_y = shared_y + v;
                float mask_value = mask[i][u + xBound][v + yBound];

                val[i][0] += shared_s[s_y][s_x][0] * mask_value;
                val[i][1] += shared_s[s_y][s_x][1] * mask_value;
                val[i][2] += shared_s[s_y][s_x][2] * mask_value;
            }
        }
    }

    float totalB = 0.f, totalG = 0.f, totalR = 0.f;
    for (int i = 0; i < Z; ++i) {
        totalB += val[i][0] * val[i][0];
        totalG += val[i][1] * val[i][1];
        totalR += val[i][2] * val[i][2];
    }
    totalB = sqrtf(totalB) / SCALE;
    totalG = sqrtf(totalG) / SCALE;
    totalR = sqrtf(totalR) / SCALE;

    unsigned char cB = (totalB > 255.f) ? 255 : (unsigned char)totalB;
    unsigned char cG = (totalG > 255.f) ? 255 : (unsigned char)totalG;
    unsigned char cR = (totalR > 255.f) ? 255 : (unsigned char)totalR;

    int out_idx = channels * (y * width + x);
    t[out_idx + 2] = cR;
    t[out_idx + 1] = cG;
    t[out_idx + 0] = cB;
}

int main(int argc, char **argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *src = NULL, *dst;
    unsigned char *dsrc, *ddst;

    /* read the image to src, and get height, width, channels */
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }

    dst = (unsigned char *)malloc(height * width * channels * sizeof(unsigned char));
    cudaHostRegister(src, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);

    // cudaMalloc(...) for device src and device dst
    cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char));

    // cudaMemcpy(...) copy source image to device (mask matrix if necessary)
    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // decide to use how many blocks and threads
    // const int num_threads = 256;
    // const int num_blocks = height / num_threads + 1;

    // // launch cuda kernel
    // sobel<<<num_blocks, num_threads>>>(dsrc, ddst, height, width, channels);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(ceil(width / TILE_WIDTH), ceil(height / TILE_WIDTH));
    sobel<<<grid, threadsPerBlock>>>(dsrc, ddst, height, width, channels);


    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(dst, ddst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    write_png(argv[2], dst, height, width, channels);
    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);
    return 0;
}
