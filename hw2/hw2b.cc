#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include <omp.h>
#include <immintrin.h>

#define TASK_TAG 1
#define DATA_TAG 2
#define TERMINATE_TAG 0

double left, right, lower, upper;
int width, height, iters;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_FILTER_NONE);
    // png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);
    // png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void computeMandelbrot_SIMD(int row, int* row_image) {
    const int simd_width = 8;  // 512 / 64bit
    int simd_end = width - (width % (2 * simd_width));
    __m512d vec_left = _mm512_set1_pd(left);
    __m512d vec_right_left_diff = _mm512_set1_pd((right - left) / width);
    __m512d vec_y0 = _mm512_set1_pd(row * ((upper - lower) / height) + lower);
    __m512d vec_length_squared_limit = _mm512_set1_pd(4.0);
    __m512d vec_two = _mm512_set1_pd(2.0);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < simd_end; i += 2*simd_width) {
        __m512d vec_i1 = _mm512_set_pd(i + 7, i + 6, i + 5, i + 4, i + 3, i + 2, i + 1, i);
        __m512d vec_x0_1 = _mm512_fmadd_pd(vec_i1, vec_right_left_diff, vec_left);
        __m512d vec_x1 = _mm512_setzero_pd();
        __m512d vec_y1 = _mm512_setzero_pd();
        __m512d vec_length_squared1 = _mm512_setzero_pd();
        __m512d vec_repeats1 = _mm512_setzero_pd();

        __m512d vec_i2 = _mm512_set_pd(i + 15, i + 14, i + 13, i + 12, i + 11, i + 10, i + 9, i + 8);
        __m512d vec_x0_2 = _mm512_fmadd_pd(vec_i2, vec_right_left_diff, vec_left);
        __m512d vec_x2 = _mm512_setzero_pd();
        __m512d vec_y2 = _mm512_setzero_pd();
        __m512d vec_length_squared2 = _mm512_setzero_pd();
        __m512d vec_repeats2 = _mm512_setzero_pd();

        for (int rep = 0; rep < iters; ++rep) {
            __mmask8 mask1 = _mm512_cmp_pd_mask(vec_length_squared1, vec_length_squared_limit, _CMP_LT_OQ);
            __mmask8 mask2 = _mm512_cmp_pd_mask(vec_length_squared2, vec_length_squared_limit, _CMP_LT_OQ);

            if (mask1 == 0 && mask2 == 0) break;
            if(mask1 != 0){
                __m512d vec_temp1 = _mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(vec_x1, vec_x1), _mm512_mul_pd(vec_y1, vec_y1)), vec_x0_1);
                vec_y1 = _mm512_fmadd_pd(vec_two, _mm512_mul_pd(vec_x1, vec_y1), vec_y0);
                vec_x1 = vec_temp1;
                vec_length_squared1 = _mm512_add_pd(_mm512_mul_pd(vec_x1, vec_x1), _mm512_mul_pd(vec_y1, vec_y1));
                vec_repeats1 = _mm512_mask_add_pd(vec_repeats1, mask1, vec_repeats1, _mm512_set1_pd(1.0));
            }
            if (mask2 != 0){
                __m512d vec_temp2 = _mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(vec_x2, vec_x2), _mm512_mul_pd(vec_y2, vec_y2)), vec_x0_2);
                vec_y2 = _mm512_fmadd_pd(vec_two, _mm512_mul_pd(vec_x2, vec_y2), vec_y0);
                vec_x2 = vec_temp2;
                vec_length_squared2 = _mm512_add_pd(_mm512_mul_pd(vec_x2, vec_x2), _mm512_mul_pd(vec_y2, vec_y2));
                vec_repeats2 = _mm512_mask_add_pd(vec_repeats2, mask2, vec_repeats2, _mm512_set1_pd(1.0));
            }
        }

        // for (int j = 0; j < simd_width; ++j) {
        //     row_image[i + j] = (int)result[j];
        // }

        double result1[simd_width];
        double result2[simd_width];
        _mm512_storeu_pd(result1, vec_repeats1);
        _mm512_storeu_pd(result2, vec_repeats2);
        for (int j = 0; j < simd_width; ++j) {
            row_image[i + j] = (int)result1[j];
            row_image[i + simd_width + j] = (int)result2[j];
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = simd_end; i < width; ++i) {
        double x0 = i * ((right - left) / width) + left;
        double x = 0, y = 0, length_squared = 0;
        int repeats = 0;
        while (repeats < iters && length_squared < 4) {
            double temp = x * x - y * y + x0;
            y = 2 * x * y + (row * ((upper - lower) / height) + lower);
            x = temp;
            length_squared = x * x + y * y;
            ++repeats;
        }
        row_image[i] = repeats;
    }
}


int main(int argc, char** argv) {
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], NULL, 10);
    left = strtod(argv[3], NULL);
    right = strtod(argv[4], NULL);
    lower = strtod(argv[5], NULL);
    upper = strtod(argv[6], NULL);
    width = strtol(argv[7], NULL, 10);
    height = strtol(argv[8], NULL, 10);

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);

    printf("%d cpus available\n", ncpus);

    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    /* master */
    if (mpi_rank == 0) {
        int* image = (int*)malloc(width * height * sizeof(int));
        assert(image);

        int next_row = 0;
        int num_workers = 0;
        MPI_Status status;

        // #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i <= mpi_size-1; ++i) {
            if (next_row < height) {
                MPI_Send(&next_row, 1, MPI_INT, i, TASK_TAG, MPI_COMM_WORLD);
                ++next_row;
                ++num_workers;
            } else {
                int terminate = -1;
                MPI_Send(&terminate, 1, MPI_INT, i, TERMINATE_TAG, MPI_COMM_WORLD);
                --num_workers;
            }
        }

        while (num_workers > 0) {
            int row;
            MPI_Recv(&row, 1, MPI_INT, MPI_ANY_SOURCE, DATA_TAG, MPI_COMM_WORLD, &status);

            int* row_data = (int*)malloc(width * sizeof(int));
            MPI_Recv(row_data, width, MPI_INT, status.MPI_SOURCE, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            memcpy(&image[row * width], row_data, width * sizeof(int));
            free(row_data);

            if (next_row < height) {
                MPI_Send(&next_row, 1, MPI_INT, status.MPI_SOURCE, TASK_TAG, MPI_COMM_WORLD);
                ++next_row;
            } else {
                int terminate = -1;
                MPI_Send(&terminate, 1, MPI_INT, status.MPI_SOURCE, TERMINATE_TAG, MPI_COMM_WORLD);
                --num_workers;
            }
        }

        write_png(filename, iters, width, height, image);
        free(image);
    }
    else {
        int row;
        MPI_Status status;
        int* row_data = (int*)malloc(width * sizeof(int));
        assert(row_data);

        while (1) {
            MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TERMINATE_TAG) {
                break;
            }

            computeMandelbrot_SIMD(row, row_data);

            MPI_Send(&row, 1, MPI_INT, 0, DATA_TAG, MPI_COMM_WORLD);
            MPI_Send(row_data, width, MPI_INT, 0, DATA_TAG, MPI_COMM_WORLD);

        }
        free(row_data);
    }
    MPI_Finalize();
    return 0;
}